/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sddmm_csr.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <c10/core/DeviceGuard.h>
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "sddmm_csr/op_kernel/arch22/sddmm_csr.h"
#include "sddmm_csr/op_kernel/arch22/sddmm_csr_tiling.h"

namespace opsgnn {

namespace {

constexpr uint32_t kSddmmUbReserved = 2 * 1024;
constexpr uint32_t kSddmmAlignBytes = 32;
constexpr uint32_t kSddmmMaxDotBatch = 255;
constexpr uint32_t kSddmmMaxBatch = 4095;

struct SddmmCsrOpInfo {
    uint32_t code;
    bool useLhs;
    bool useRhs;
};

SddmmCsrOpInfo ResolveOp(const std::string& op)
{
    if (op == "dot") return {SDDMM_CSR_OP_DOT, true, true};
    if (op == "add") return {SDDMM_CSR_OP_ADD, true, true};
    if (op == "sub") return {SDDMM_CSR_OP_SUB, true, true};
    if (op == "mul") return {SDDMM_CSR_OP_MUL, true, true};
    if (op == "div") return {SDDMM_CSR_OP_DIV, true, true};
    if (op == "copy_lhs") return {SDDMM_CSR_OP_COPY_LHS, true, false};
    if (op == "copy_rhs") return {SDDMM_CSR_OP_COPY_RHS, false, true};
    TORCH_CHECK(false, "sddmm_csr: op must be dot, add, sub, mul, div, copy_lhs or copy_rhs");
    return {SDDMM_CSR_OP_DOT, true, true};
}

uint32_t ResolveTarget(int64_t target, const char* name)
{
    TORCH_CHECK(target >= 0 && target <= 2,
                "sddmm_csr: ", name, " must be 0 (src), 1 (edge) or 2 (dst)");
    return static_cast<uint32_t>(target);
}

uint32_t AlignElementCount(int64_t dim, int64_t elementBytes)
{
    const uint64_t safeElementBytes =
        elementBytes > 0 ? static_cast<uint64_t>(elementBytes) : 1;
    const uint64_t bytes = static_cast<uint64_t>(dim) * safeElementBytes;
    const uint64_t aligned = (bytes + kSddmmAlignBytes - 1) / kSddmmAlignBytes * kSddmmAlignBytes;
    return static_cast<uint32_t>(aligned / safeElementBytes);
}

void CheckTensor(const torch::Tensor& tensor, const c10::Device& device, const char* name)
{
    TORCH_CHECK(tensor.defined(), "sddmm_csr: ", name, " must be defined");
    TORCH_CHECK(tensor.device() == device, "sddmm_csr: ", name, " must be on the same NPU");
}

void CheckCsr(const torch::Tensor& indptr, const torch::Tensor& indices,
              int64_t m, int64_t nnz, int64_t dstRows)
{
    TORCH_CHECK(indptr.dim() == 1 && indices.dim() == 1 && indptr.numel() >= 1,
                "sddmm_csr: indptr and indices must be one-dimensional; indptr must be nonempty");
    TORCH_CHECK((indptr.scalar_type() == at::kInt || indptr.scalar_type() == at::kLong) &&
                indices.scalar_type() == indptr.scalar_type(),
                "sddmm_csr: indptr and indices must have the same int32/int64 dtype");

    const bool isInt64 = indptr.scalar_type() == at::kLong;
    torch::Tensor rangeOk;
    if (isInt64) {
        const torch::Tensor maxPtr = indptr.max();
        const torch::Tensor maxIdx = nnz > 0 ? indices.max() : maxPtr;
        rangeOk = at::maximum(maxPtr, maxIdx)
                      .le(std::numeric_limits<int32_t>::max());
    }

    const torch::Tensor ptrSorted =
        m > 0 ? indptr.slice(0, 1).ge(indptr.slice(0, 0, m)).all()
              : torch::ones({}, indptr.options().dtype(at::kBool));
    const bool checkDst = dstRows != std::numeric_limits<int64_t>::max() && nnz > 0;
    torch::Tensor idxOk = indices.ge(0);
    if (checkDst) {
        idxOk = idxOk.logical_and(indices.lt(dstRows));
    }
    torch::Tensor valid =
        (indptr.select(0, 0) == 0)
            .logical_and(indptr.select(0, m) == nnz)
            .logical_and(indptr.ge(0).logical_and(indptr.le(nnz)).all())
            .logical_and(ptrSorted)
            .logical_and(idxOk.all());
    if (isInt64) {
        valid = valid.logical_and(rangeOk);
    }
    if (valid.item<bool>()) {
        return;
    }

    if (isInt64) {
        TORCH_CHECK(rangeOk.item<bool>(),
                    "sddmm_csr: indices exceed the int32 range");
    }
    if (checkDst) {
        TORCH_CHECK(indices.lt(dstRows).all().item<bool>(),
                    "sddmm_csr: CSR indices are out of range for dst features");
    }
    TORCH_CHECK(false, "sddmm_csr: invalid CSR (indptr/indices)");
}

void CheckTargetRange(const torch::Tensor& feature, uint32_t target,
                      int64_t m, int64_t nnz, const char* name)
{
    const int64_t rows = feature.size(0);
    if (target == SDDMM_CSR_TARGET_SRC) {
        TORCH_CHECK(rows >= m,
                    "sddmm_csr: ", name, " has fewer rows than the CSR row count");
    } else if (target == SDDMM_CSR_TARGET_EDGE) {
        TORCH_CHECK(rows >= nnz,
                    "sddmm_csr: ", name, " has fewer rows than the number of edges");
    }
}


uint32_t ComputeBatchSize(uint32_t opCode, int64_t featDim, int64_t elementBytes,
                          uint32_t alignedNative, uint32_t alignedF32,
                          uint32_t ubBytes)
{
    const uint64_t available = ubBytes - kSddmmUbReserved;
    if (opCode == SDDMM_CSR_OP_DOT) {
        uint64_t fixed = 0;
        if (elementBytes == 4) {
            fixed = 4ull * alignedNative * 4;
        } else {
            fixed = 4ull * alignedNative * 2 + 2ull * alignedF32 * 4;
        }
        const uint64_t perBatch = static_cast<uint64_t>(alignedF32) * 4 + 8 +
                                  static_cast<uint64_t>(elementBytes) + 4 +
                                  (elementBytes == 2 ? 4 : 0);
        TORCH_CHECK(fixed + perBatch <= available,
                    "sddmm_csr: feature_dim exceeds UB limit");
        uint64_t batch = (available - fixed) / perBatch;
        batch = std::max<uint64_t>(batch, 1);
        batch = std::min<uint64_t>(batch, kSddmmMaxDotBatch);
        return static_cast<uint32_t>(batch);
    }

    if (opCode == SDDMM_CSR_OP_COPY_LHS || opCode == SDDMM_CSR_OP_COPY_RHS) {
        const uint64_t fixed = 2ull * alignedNative * elementBytes;
        const uint64_t bufBytes = static_cast<uint64_t>(alignedNative) * elementBytes;
        TORCH_CHECK(fixed + 8 <= available,
                    "sddmm_csr: feature_dim exceeds UB limit");
        uint64_t batch = available / bufBytes;
        batch = std::max<uint64_t>(batch, 1);
        batch = std::min<uint64_t>(batch, kSddmmMaxBatch);
        TORCH_CHECK(fixed + 8ull * batch <= available,
                    "sddmm_csr: feature_dim exceeds UB limit");
        return static_cast<uint32_t>(batch);
    }

    const uint64_t fixed = 6ull * alignedNative * elementBytes +
                           (elementBytes == 2 ? 3ull * alignedF32 * 4 : 0);
    const uint64_t bufBytes = elementBytes == 2
                                  ? static_cast<uint64_t>(alignedF32) * 4
                                  : static_cast<uint64_t>(alignedNative) * elementBytes;
    TORCH_CHECK(fixed + 8 <= available,
                "sddmm_csr: feature_dim exceeds UB limit");
    uint64_t batch = available / (bufBytes * 2);
    batch = std::max<uint64_t>(batch, 1);
    batch = std::min<uint64_t>(batch, kSddmmMaxBatch);
    TORCH_CHECK(fixed + 8ull * batch <= available,
                "sddmm_csr: feature_dim exceeds UB limit");
    return static_cast<uint32_t>(batch);
}

struct SddmmCsrFeatures {
    const torch::Tensor* primary;
    int64_t featDim;
    at::ScalarType dtype;
    int64_t elementBytes;
};

std::pair<const torch::Tensor*, const torch::Tensor*> ValidateOperands(
    const SddmmCsrOpInfo& info, const std::string& op,
    const std::optional<torch::Tensor>& lhs,
    const std::optional<torch::Tensor>& rhs,
    const c10::Device& device)
{
    const torch::Tensor* lhsPtr = nullptr;
    const torch::Tensor* rhsPtr = nullptr;
    if (info.useLhs) {
        TORCH_CHECK(lhs.has_value(), "sddmm_csr: lhs is required for op ", op);
        CheckTensor(*lhs, device, "lhs");
        lhsPtr = &*lhs;
    } else {
        TORCH_CHECK(!lhs.has_value(), "sddmm_csr: lhs is not used by op ", op);
    }
    if (info.useRhs) {
        TORCH_CHECK(rhs.has_value(), "sddmm_csr: rhs is required for op ", op);
        CheckTensor(*rhs, device, "rhs");
        rhsPtr = &*rhs;
    } else {
        TORCH_CHECK(!rhs.has_value(), "sddmm_csr: rhs is not used by op ", op);
    }
    return {lhsPtr, rhsPtr};
}

SddmmCsrFeatures ResolveFeatures(const torch::Tensor* lhsPtr,
                                 const torch::Tensor* rhsPtr)
{
    const torch::Tensor* primary = lhsPtr != nullptr ? lhsPtr : rhsPtr;
    TORCH_CHECK(primary != nullptr, "sddmm_csr: at least one feature tensor is required");
    const at::ScalarType dtype = primary->scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
                "sddmm_csr: features must be float16 or float32");
    int64_t featDim = 0;
    for (const torch::Tensor* feature : {lhsPtr, rhsPtr}) {
        if (feature == nullptr) continue;
        TORCH_CHECK(feature->dim() == 2 && feature->size(1) > 0,
                    "sddmm_csr: features must be two-dimensional with a positive feature dim");
        TORCH_CHECK(feature->scalar_type() == dtype,
                    "sddmm_csr: lhs and rhs must have the same dtype");
        TORCH_CHECK(!feature->requires_grad(),
                    "sddmm_csr: autograd is not supported");
        if (featDim == 0) {
            featDim = feature->size(1);
        } else {
            TORCH_CHECK(featDim == feature->size(1),
                        "sddmm_csr: lhs and rhs must have the same feature dim");
        }
    }
    return {primary, featDim, dtype, dtype == at::kFloat ? 4 : 2};
}

void ValidateLimits(int64_t m, int64_t nnz, int64_t featDim,
                    const torch::Tensor* lhsPtr, const torch::Tensor* rhsPtr)
{
    const uint64_t offsetLimit = std::numeric_limits<uint32_t>::max();
    const int64_t dimensionLimit = std::numeric_limits<int32_t>::max();
    TORCH_CHECK(m <= dimensionLimit && nnz <= dimensionLimit &&
                featDim <= dimensionLimit,
                "sddmm_csr: dimensions exceed int32 limits");
    uint64_t maxRows = static_cast<uint64_t>(std::max<int64_t>(m, nnz));
    for (const torch::Tensor* feature : {lhsPtr, rhsPtr}) {
        if (feature != nullptr) {
            maxRows = std::max<uint64_t>(maxRows, static_cast<uint64_t>(feature->size(0)));
        }
    }
    TORCH_CHECK(maxRows * static_cast<uint64_t>(featDim) <= offsetLimit &&
                static_cast<uint64_t>(nnz) * static_cast<uint64_t>(featDim) <= offsetLimit,
                "sddmm_csr: feature or output offsets exceed uint32 limits");
    TORCH_CHECK(static_cast<uint64_t>(m + 1) * sizeof(int32_t) <= offsetLimit &&
                static_cast<uint64_t>(nnz) * sizeof(int32_t) <= offsetLimit,
                "sddmm_csr: CSR offsets exceed uint32 limits");
}

int64_t ResolveDstRows(const torch::Tensor* lhsPtr, const torch::Tensor* rhsPtr,
                       uint32_t lhsTargetCode, uint32_t rhsTargetCode)
{
    int64_t dstRows = std::numeric_limits<int64_t>::max();
    if (lhsPtr != nullptr && lhsTargetCode == SDDMM_CSR_TARGET_DST) {
        dstRows = std::min(dstRows, lhsPtr->size(0));
    }
    if (rhsPtr != nullptr && rhsTargetCode == SDDMM_CSR_TARGET_DST) {
        dstRows = std::min(dstRows, rhsPtr->size(0));
    }
    return dstRows;
}

void ValidateTargetRanges(const torch::Tensor* lhsPtr, const torch::Tensor* rhsPtr,
                          uint32_t lhsTargetCode, uint32_t rhsTargetCode,
                          int64_t m, int64_t nnz)
{
    if (lhsPtr != nullptr) {
        CheckTargetRange(*lhsPtr, lhsTargetCode, m, nnz, "lhs");
    }
    if (rhsPtr != nullptr) {
        CheckTargetRange(*rhsPtr, rhsTargetCode, m, nnz, "rhs");
    }
}

torch::Tensor BuildOutput(const std::vector<int64_t>& outShape,
                          const std::optional<torch::Tensor>& out,
                          const torch::Tensor& indptr, const torch::Tensor& indices,
                          const torch::Tensor* lhsPtr, const torch::Tensor* rhsPtr,
                          const torch::Tensor& primary, at::ScalarType dtype,
                          const c10::Device& device)
{
    if (!out.has_value()) {
        return torch::empty(outShape, primary.options());
    }
    const torch::Tensor& candidate = *out;
    TORCH_CHECK(candidate.defined(), "sddmm_csr: out must be defined");
    TORCH_CHECK(candidate.device() == device && candidate.scalar_type() == dtype,
                "sddmm_csr: out must match the feature device and dtype");
    TORCH_CHECK(candidate.dim() == static_cast<int64_t>(outShape.size()),
                "sddmm_csr: out has the wrong rank");
    for (size_t i = 0; i < outShape.size(); ++i) {
        TORCH_CHECK(candidate.size(static_cast<int64_t>(i)) == outShape[i],
                    "sddmm_csr: out has the wrong shape");
    }
    TORCH_CHECK(candidate.is_contiguous(), "sddmm_csr: out must be contiguous");
    TORCH_CHECK(!candidate.requires_grad(),
                "sddmm_csr: out must not require gradients");
    TORCH_CHECK(!candidate.is_alias_of(indptr) && !candidate.is_alias_of(indices),
                "sddmm_csr: out must not alias any input storage");
    for (const torch::Tensor* feature : {lhsPtr, rhsPtr}) {
        if (feature != nullptr) {
            TORCH_CHECK(!candidate.is_alias_of(*feature),
                        "sddmm_csr: out must not alias any input storage");
        }
    }
    return candidate;
}

struct SddmmCsrLaunchConfig {
    uint32_t blocks;
    uint32_t edgesPerCore;
    uint32_t batchSize;
    uint32_t alignedNative;
    uint32_t alignedF32;
    uint64_t ubBytes;
};

SddmmCsrLaunchConfig ComputeLaunchConfig(const SddmmCsrOpInfo& info,
                                         int64_t featDim, int64_t elementBytes,
                                         int64_t nnz)
{
    const uint64_t offsetLimit = std::numeric_limits<uint32_t>::max();
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "sddmm_csr: platform information unavailable");
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    TORCH_CHECK(ubBytes > kSddmmUbReserved && ubBytes <= offsetLimit,
                "sddmm_csr: invalid platform UB capacity");
    const uint32_t cores = platform->GetCoreNumAiv();
    TORCH_CHECK(cores > 0, "sddmm_csr: no AIV cores available");

    const uint32_t alignedNative = AlignElementCount(featDim, elementBytes);
    const uint32_t alignedF32 = AlignElementCount(featDim, 4);
    const uint32_t batchSize = ComputeBatchSize(
        info.code, featDim, elementBytes, alignedNative, alignedF32,
        static_cast<uint32_t>(ubBytes));

    const uint32_t blocks = static_cast<uint32_t>(std::min<int64_t>(cores, nnz));
    const uint32_t edgesPerCore = static_cast<uint32_t>(
        (nnz + static_cast<int64_t>(blocks) - 1) / static_cast<int64_t>(blocks));
    return {blocks, edgesPerCore, batchSize, alignedNative, alignedF32, ubBytes};
}

void LaunchSddmmCsr(const torch::Tensor& indptr, const torch::Tensor& indices,
                    const torch::Tensor* lhsPtr, const torch::Tensor* rhsPtr,
                    const torch::Tensor& primary, const torch::Tensor& result,
                    const c10::Device& device, const SddmmCsrOpInfo& info,
                    int64_t m, int64_t nnz, int64_t featDim,
                    uint32_t lhsTargetCode, uint32_t rhsTargetCode,
                    at::ScalarType dtype, const SddmmCsrLaunchConfig& config)
{
    torch::Tensor lhsContiguous;
    torch::Tensor rhsContiguous;
    void* lhsData = nullptr;
    void* rhsData = nullptr;
    if (lhsPtr != nullptr) {
        lhsContiguous = lhsPtr->contiguous();
        lhsData = lhsContiguous.data_ptr();
    }
    if (rhsPtr != nullptr) {
        rhsContiguous = rhsPtr->contiguous();
        rhsData = rhsContiguous.data_ptr();
    }

    torch::Tensor ptr = indptr.contiguous().to(at::kInt).contiguous();
    torch::Tensor idx = indices.contiguous().to(at::kInt).contiguous();

    std::vector<int32_t> hostTiling = {
        static_cast<int32_t>(nnz),
        static_cast<int32_t>(m),
        static_cast<int32_t>(featDim),
        static_cast<int32_t>(config.blocks),
        static_cast<int32_t>(config.edgesPerCore),
        static_cast<int32_t>(config.batchSize),
        static_cast<int32_t>(dtype == at::kFloat ? SDDMM_CSR_DTYPE_FP32 : SDDMM_CSR_DTYPE_FP16),
        static_cast<int32_t>(info.code),
        static_cast<int32_t>(lhsTargetCode),
        static_cast<int32_t>(rhsTargetCode),
        static_cast<int32_t>(config.alignedNative),
        static_cast<int32_t>(config.alignedF32),
        static_cast<int32_t>(config.ubBytes)};
    auto tiling = torch::tensor(hostTiling, torch::TensorOptions().dtype(at::kInt)).to(device);

    auto npuStream = c10_npu::getCurrentNPUStream(primary.get_device());
    SddmmCsr(config.blocks, npuStream.stream(), lhsData, rhsData, ptr.data_ptr(),
             idx.data_ptr(), result.data_ptr(), tiling.data_ptr());
}

}  // namespace

torch::Tensor SddmmCsrForward(const torch::Tensor& indptr, const torch::Tensor& indices,
                              const std::optional<torch::Tensor>& lhs,
                              const std::optional<torch::Tensor>& rhs,
                              const std::string& op, int64_t lhsTarget, int64_t rhsTarget,
                              const std::optional<torch::Tensor>& out)
{
    const SddmmCsrOpInfo info = ResolveOp(op);
    const uint32_t lhsTargetCode = ResolveTarget(lhsTarget, "lhs_target");
    const uint32_t rhsTargetCode = ResolveTarget(rhsTarget, "rhs_target");

    TORCH_CHECK(indptr.defined() && indices.defined(),
                "sddmm_csr: indptr and indices must be defined");
    TORCH_CHECK(indptr.device().type() == c10::DeviceType::PrivateUse1 &&
                indices.device().type() == c10::DeviceType::PrivateUse1,
                "sddmm_csr: indptr and indices must be NPU tensors");
    const c10::Device device = indices.device();
    CheckTensor(indptr, device, "indptr");

    const auto [lhsPtr, rhsPtr] = ValidateOperands(info, op, lhs, rhs, device);
    const int64_t m = indptr.numel() - 1;
    const int64_t nnz = indices.numel();
    const SddmmCsrFeatures features = ResolveFeatures(lhsPtr, rhsPtr);
    const int64_t featDim = features.featDim;
    const at::ScalarType dtype = features.dtype;
    ValidateLimits(m, nnz, featDim, lhsPtr, rhsPtr);

    const int64_t dstRows =
        ResolveDstRows(lhsPtr, rhsPtr, lhsTargetCode, rhsTargetCode);
    c10::DeviceGuard guard(device);
    CheckCsr(indptr, indices, m, nnz, dstRows);
    ValidateTargetRanges(lhsPtr, rhsPtr, lhsTargetCode, rhsTargetCode, m, nnz);

    const bool isDot = info.code == SDDMM_CSR_OP_DOT;
    const std::vector<int64_t> outShape =
        isDot ? std::vector<int64_t>{nnz} : std::vector<int64_t>{nnz, featDim};
    torch::Tensor result = BuildOutput(outShape, out, indptr, indices, lhsPtr, rhsPtr,
                                       *features.primary, dtype, device);
    if (nnz == 0) {
        return result;
    }

    const SddmmCsrLaunchConfig config =
        ComputeLaunchConfig(info, featDim, features.elementBytes, nnz);
    LaunchSddmmCsr(indptr, indices, lhsPtr, rhsPtr, *features.primary, result, device,
                   info, m, nnz, featDim, lhsTargetCode, rhsTargetCode, dtype, config);
    return result;
}

}  // namespace opsgnn

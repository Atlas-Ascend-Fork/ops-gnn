/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "unified_spmm.h"
#include "common/spmm/host_proof_cache.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include <c10/core/DeviceGuard.h>
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "unified_spmm/op_kernel/arch22/unified_spmm.h"
#include "unified_spmm/op_kernel/arch22/unified_spmm_tiling.h"

namespace opsgnn {

namespace {

uint32_t ReduceCode(const std::string& reduce)
{
    if (reduce == "sum") return UNIFIED_SPMM_REDUCE_SUM;
    if (reduce == "max") return UNIFIED_SPMM_REDUCE_MAX;
    if (reduce == "min") return UNIFIED_SPMM_REDUCE_MIN;
    TORCH_CHECK(false, "unified_spmm_csr: reduce must be 'sum', 'max' or 'min'");
    return UNIFIED_SPMM_REDUCE_SUM;
}

uint32_t MessageCode(const std::string& op)
{
    TORCH_CHECK(op == "copy_lhs" || op == "copy_rhs",
                "unified_spmm_csr: op must be 'copy_lhs' or 'copy_rhs'");
    return op == "copy_rhs" ? UNIFIED_SPMM_COPY_RHS : UNIFIED_SPMM_COPY_LHS;
}

void CheckMetadata(const torch::Tensor& indptr, const torch::Tensor& indices,
                   const torch::Tensor& x, const std::optional<torch::Tensor>& out,
                   int64_t m, int64_t n)
{
    for (const auto& tensor : {indptr, indices, x}) {
        TORCH_CHECK(tensor.defined() && tensor.device().type() == c10::DeviceType::PrivateUse1,
                    "unified_spmm_csr: inputs must be NPU tensors");
        TORCH_CHECK(tensor.device() == x.device(),
                    "unified_spmm_csr: inputs must be on the same NPU");
    }
    TORCH_CHECK(x.dim() == 2 && (x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat),
                "unified_spmm_csr: x must be a two-dimensional float16/float32 tensor");
    TORCH_CHECK(!x.requires_grad(), "unified_spmm_csr: autograd is not supported");
    TORCH_CHECK(indptr.dim() == 1 && indices.dim() == 1 && indptr.numel() >= 1,
                "unified_spmm_csr: CSR tensors must be one-dimensional; indptr must be nonempty");
    TORCH_CHECK((indptr.scalar_type() == at::kInt || indptr.scalar_type() == at::kLong) &&
                indices.scalar_type() == indptr.scalar_type(),
                "unified_spmm_csr: CSR tensors must have the same int32/int64 dtype");
    if (!out) return;
    TORCH_CHECK(out->defined() && out->device() == x.device() &&
                out->scalar_type() == x.scalar_type() && out->dim() == 2 &&
                out->size(0) == m && out->size(1) == n,
                "unified_spmm_csr: out must match output shape, dtype and device");
    TORCH_CHECK(out->is_contiguous() && !out->requires_grad(),
                "unified_spmm_csr: out must be contiguous and must not require gradients");
    TORCH_CHECK(!out->is_alias_of(x) && !out->is_alias_of(indptr) && !out->is_alias_of(indices),
                "unified_spmm_csr: out must not alias any input storage");
}

void CheckCsr(const torch::Tensor& indptr, const torch::Tensor& indices,
              int64_t m, int64_t featureRows, int64_t nnz, uint32_t message)
{
    ops_gnn::sparse::CheckCachedSpmmCsr(indptr, indices, m, featureRows, nnz,
                                      "unified_spmm_csr", message == UNIFIED_SPMM_COPY_LHS);
    if (message == UNIFIED_SPMM_COPY_RHS) {
        TORCH_CHECK(featureRows == nnz,
                    "unified_spmm_csr: copy_rhs requires x.size(0) == indices.numel()");
    }
}

uint32_t CheckLimitsAndUb(int64_t m, int64_t k, int64_t n, int64_t nnz,
                          int64_t elementBytes, uint32_t& blocks)
{
    const uint64_t limit = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(m < limit && k <= limit && n > 0 && nnz <= limit &&
                static_cast<uint64_t>(n) * elementBytes <= limit - 31,
                "unified_spmm_csr: dimensions exceed uint32 limits or feature_dim is zero");
    TORCH_CHECK(static_cast<uint64_t>(k) * n <= limit &&
                static_cast<uint64_t>(m) * n <= limit &&
                static_cast<uint64_t>(m + 1) * sizeof(uint32_t) <= limit &&
                static_cast<uint64_t>(nnz) * sizeof(uint32_t) <= limit,
                "unified_spmm_csr: element or address offsets exceed uint32 limits");

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "unified_spmm_csr: platform information unavailable");
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    TORCH_CHECK(ubBytes > 2048 && ubBytes <= limit,
                "unified_spmm_csr: invalid platform UB capacity");
    uint64_t rowBytes = static_cast<uint64_t>(n) * elementBytes;
    uint64_t aligned = (rowBytes + 31) / 32 * 32;
    uint64_t fixed = 2048 + 4 * aligned;
    if (elementBytes == 2) fixed += 2 * (aligned / 2 * sizeof(float));
    TORCH_CHECK(fixed <= ubBytes,
                "unified_spmm_csr: feature_dim exceeds UB capacity");
    uint32_t cores = platform->GetCoreNumAiv();
    TORCH_CHECK(cores > 0, "unified_spmm_csr: no AIV cores available");
    blocks = static_cast<uint32_t>(std::min<int64_t>(cores, m));
    return static_cast<uint32_t>(ubBytes);
}

}  // namespace

torch::Tensor UnifiedSpmmCsr(const torch::Tensor& indptr, const torch::Tensor& indices,
                             const torch::Tensor& x, const std::string& op,
                             const std::string& reduce,
                             const std::optional<torch::Tensor>& out)
{
    uint32_t reduceCode = ReduceCode(reduce);
    uint32_t messageCode = MessageCode(op);
    int64_t m = indptr.defined() && indptr.dim() == 1 && indptr.numel() >= 1
                    ? indptr.numel() - 1 : 0;
    int64_t n = x.defined() && x.dim() == 2 ? x.size(1) : 0;
    CheckMetadata(indptr, indices, x, out, m, n);
    TORCH_CHECK(op != "copy_rhs" || x.scalar_type() == at::kFloat,
                "unified_spmm_csr: copy_rhs supports only float32");
    c10::DeviceGuard guard(x.device());
    ops_gnn::sparse::SpmmProofScope proofs(out.has_value());
    int64_t k = x.size(0);
    int64_t nnz = indices.numel();
    CheckCsr(indptr, indices, m, k, nnz, messageCode);
    uint32_t blocks = 0;
    uint32_t ubBytes = CheckLimitsAndUb(m, k, n, nnz, x.element_size(), blocks);
    auto result = out ? *out : torch::empty({m, n}, x.options());
    if (m == 0) return result;

    auto split = torch::arange(static_cast<int64_t>(blocks) + 1,
                               indptr.options().dtype(at::kLong));
    split = at::floor_divide(split * m, blocks).to(at::kInt);
    auto ptr = indptr.to(at::kInt).contiguous();
    auto idx = indices.to(at::kInt).contiguous();
    auto features = x.contiguous();
    uint32_t dtype = x.scalar_type() == at::kFloat ?
                     UNIFIED_SPMM_DTYPE_FP32 : UNIFIED_SPMM_DTYPE_FP16;
    uint32_t hasNan = ops_gnn::sparse::CachedHasNan(x);
    std::vector<int32_t> hostTiling = {
        static_cast<int32_t>(m), static_cast<int32_t>(k), static_cast<int32_t>(n),
        static_cast<int32_t>(nnz), static_cast<int32_t>(ubBytes),
        static_cast<int32_t>(dtype), static_cast<int32_t>(reduceCode),
        static_cast<int32_t>(messageCode), static_cast<int32_t>(hasNan)};
    auto tiling = torch::tensor(hostTiling, torch::TensorOptions().dtype(at::kInt)).to(x.device());
    auto stream = c10_npu::getCurrentNPUStream(x.get_device()).stream();
    UnifiedSpmm(blocks, stream, features.data_ptr(), result.data_ptr(), ptr.data_ptr(),
                      idx.data_ptr(), split.data_ptr(), tiling.data_ptr());
    return result;
}

}  // namespace opsgnn

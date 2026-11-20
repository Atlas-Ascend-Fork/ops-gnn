/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "segment_max_csr.h"
#include "segment_max_csr/op_kernel/arch35/segment_max_csr.h"
#include "tiling/platform/platform_ascendc.h"
#include "segment_max_csr/op_kernel/arch35/segment_max_csr_tiling.h"
#include <acl/acl_base.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

static uint32_t GetCeilInt(uint64_t value1, uint64_t value2)
{
    TORCH_CHECK(value2 > 0, "segment_max_csr: GetCeilInt divisor must be > 0");
    return static_cast<uint32_t>((value1 + value2 - 1) / value2);
}

namespace {

struct DimInfo {
    uint32_t srcDim;
    uint32_t indptrDim;
    uint32_t E_1;
    uint32_t M;
    uint32_t nSegments;
    uint32_t K;
};

void ValidateInputs(const torch::Tensor& src, const torch::Tensor& indptr,
                    const torch::Tensor& optional_out, bool hasOptionalOut)
{
    auto indptrDim = indptr.dim();
    TORCH_CHECK(indptrDim >= 1, "indptr must have at least 1 dim, got ", indptrDim);
    TORCH_CHECK(src.dim() >= indptrDim, "src.dim() (", src.dim(),
                ") must be >= indptr.dim() (", indptrDim, ")");
    TORCH_CHECK(indptr.size(indptrDim - 1) >= 1,
                "indptr last dim must be >= 1, got ", indptr.size(indptrDim - 1));
    TORCH_CHECK(indptr.scalar_type() == at::ScalarType::Int,
                "indptr must be int32, got ", indptr.scalar_type());
    for (uint32_t axis = 0; axis + 1 < indptrDim; ++axis) {
        TORCH_CHECK(indptr.size(axis) == 1 || indptr.size(axis) == src.size(axis),
                    "segment_max_csr: indptr dimension ", axis, " with size ", indptr.size(axis),
                    " cannot broadcast to src size ", src.size(axis));
    }
    if (hasOptionalOut) {
        TORCH_CHECK(optional_out.scalar_type() == src.scalar_type(),
                    "optional_out dtype must match src, got ", optional_out.scalar_type(),
                    " vs ", src.scalar_type());
    }
}

void ValidateIndptrValues(const torch::Tensor& indptr, uint32_t indptrDim, uint32_t M)
{
    const int64_t rowWidth = indptr.size(indptrDim - 1);
    TORCH_CHECK(rowWidth > 0, "indptr last dim must be >= 1, got ", rowWidth);
    auto cpu = indptr.to(at::kCPU).contiguous();
    const auto* values = cpu.data_ptr<int32_t>();
    const int64_t batchCount = cpu.numel() / rowWidth;
    for (int64_t batch = 0; batch < batchCount; ++batch) {
        const auto* row = values + batch * rowWidth;
        for (int64_t i = 0; i < rowWidth; ++i) {
            TORCH_CHECK(row[i] >= 0 && static_cast<uint32_t>(row[i]) <= M,
                        "segment_max_csr: indptr value out of range at batch ", batch,
                        ", index ", i, ": expected [0, ", M, "], got ", row[i]);
        }
        for (int64_t i = 0; i + 1 < rowWidth; ++i) {
            TORCH_CHECK(row[i] <= row[i + 1],
                        "segment_max_csr: indptr must be non-decreasing at batch ", batch,
                        ", indices ", i, " and ", i + 1);
        }
    }
}

DimInfo ComputeDimInfo(const torch::Tensor& src, const torch::Tensor& indptr)
{
    DimInfo info;
    info.srcDim = src.dim();
    info.indptrDim = indptr.dim();

    info.E_1 = 1;
    for (uint32_t d = 0; d < info.indptrDim - 1; ++d) {
        info.E_1 *= src.size(d);
    }
    info.M = src.size(info.indptrDim - 1);
    uint32_t indptrLastDim = indptr.size(info.indptrDim - 1);
    info.nSegments = indptrLastDim - 1;

    info.K = 1;
    for (uint32_t d = info.indptrDim; d < info.srcDim; ++d) {
        info.K *= src.size(d);
    }
    return info;
}

SegmentMaxCsrTilingData ComputeTiling(
    uint32_t srcLength, uint32_t E_1, uint32_t M, uint32_t nSegments,
    uint32_t K, uint32_t indptrPhysicalNum, uint32_t sizeofdatatype,
    uint32_t aivCanUseNum, uint32_t strideIndptr,
    uint32_t hasOptionalOutVal)
{
    TORCH_CHECK(sizeofdatatype > 0, "sizeofdatatype must be > 0");
    uint32_t ALIGN_NUM = 32 / sizeofdatatype;

    uint32_t aivNum = (aivCanUseNum < E_1) ? aivCanUseNum : E_1;
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t rowsPerAiv = GetCeilInt(E_1, aivNum);
    const uint64_t rowsBeforeLast = static_cast<uint64_t>(rowsPerAiv) * (aivNum - 1);
    uint32_t rowsLastAiv = rowsBeforeLast < E_1 ? static_cast<uint32_t>(E_1 - rowsBeforeLast) : 0;

    uint32_t coreDataNum = 2048;

    uint32_t KloopTime = GetCeilInt(K, 2048);

    SegmentMaxCsrTilingData tiling;
    tiling.srcLength = srcLength;
    tiling.E_1 = E_1;
    tiling.M = M;
    tiling.nSegments = nSegments;
    tiling.K = K;
    tiling.indptrPhysicalNum = indptrPhysicalNum;
    tiling.coreDataNum = coreDataNum;
    tiling.ALIGN_NUM = ALIGN_NUM;
    tiling.aivNum = aivNum;
    tiling.rowsPerAiv = rowsPerAiv;
    tiling.rowsLastAiv = rowsLastAiv;
    tiling.KloopTime = KloopTime;
    tiling.hasOptionalOut = hasOptionalOutVal;
    tiling.strideIndptr = strideIndptr;
    return tiling;
}

void LaunchKernel(const torch::Tensor& src, const torch::Tensor& indptr,
                  const torch::Tensor& optional_out, const torch::Tensor& out,
                  bool hasOptionalOut, const SegmentMaxCsrTilingData& tiling,
                  aclrtStream stream)
{
    auto scalar_type = src.scalar_type();
    if (scalar_type == at::ScalarType::Float) {
        SegmentMaxCsr(
            src.data_ptr<float>(),
            indptr.data_ptr<int32_t>(),
            hasOptionalOut ? optional_out.data_ptr<float>() : nullptr,
            out.data_ptr<float>(),
            tiling, stream);
    } else if (scalar_type == at::ScalarType::Half) {
        SegmentMaxCsr(
            reinterpret_cast<uint16_t*>(src.data_ptr<c10::Half>()),
            indptr.data_ptr<int32_t>(),
            hasOptionalOut ? reinterpret_cast<uint16_t*>(optional_out.data_ptr<c10::Half>()) : nullptr,
            reinterpret_cast<uint16_t*>(out.data_ptr<c10::Half>()),
            tiling, stream);
    } else if (scalar_type == at::ScalarType::Int) {
        SegmentMaxCsr(
            src.data_ptr<int32_t>(),
            indptr.data_ptr<int32_t>(),
            hasOptionalOut ? optional_out.data_ptr<int32_t>() : nullptr,
            out.data_ptr<int32_t>(),
            tiling, stream);
    } else if (scalar_type == at::ScalarType::Short) {
        SegmentMaxCsr(
            src.data_ptr<int16_t>(),
            indptr.data_ptr<int32_t>(),
            hasOptionalOut ? optional_out.data_ptr<int16_t>() : nullptr,
            out.data_ptr<int16_t>(),
            tiling, stream);
    } else {
        TORCH_CHECK(false, "segment_max_csr only supports float, half, int32, and int16 types");
    }
}

} // namespace

torch::Tensor segment_max_csr(torch::Tensor src, torch::Tensor indptr, torch::Tensor optional_out)
{
    bool hasOptionalOut = (optional_out.defined() && optional_out.numel() > 0);
    ValidateInputs(src, indptr, optional_out, hasOptionalOut);

    DimInfo dim = ComputeDimInfo(src, indptr);
    ValidateIndptrValues(indptr, dim.indptrDim, dim.M);

    std::vector<int64_t> outShape(src.sizes().begin(), src.sizes().end());
    outShape[dim.indptrDim - 1] = dim.nSegments;

    if (hasOptionalOut) {
        TORCH_CHECK(optional_out.device() == src.device(),
                    "segment_max_csr: optional_out must be on the same device as src");
        TORCH_CHECK(optional_out.sizes() == c10::IntArrayRef(outShape),
                    "segment_max_csr: optional_out shape must match the output shape");
        optional_out = optional_out.contiguous();
    }

    if (dim.K == 0) {
        return torch::empty(outShape, src.options());
    }

    uint32_t indptrPreProduct = 1;
    for (uint32_t d = 0; d < dim.indptrDim - 1; ++d) {
        indptrPreProduct *= indptr.size(d);
    }
    uint32_t indptrLastDim = indptr.size(dim.indptrDim - 1);
    uint32_t strideIndptr;
    if (indptrPreProduct == dim.E_1) {
        strideIndptr = indptrLastDim;
    } else if (indptrPreProduct == 1) {
        strideIndptr = 0;
    } else {
        std::vector<int64_t> expandedShape(indptr.sizes().begin(), indptr.sizes().end());
        for (uint32_t d = 0; d + 1 < dim.indptrDim; ++d) {
            expandedShape[d] = src.size(d);
        }
        indptr = indptr.expand(expandedShape).contiguous();
        strideIndptr = indptrLastDim;
    }

    torch::Tensor out = torch::empty(outShape, src.options());

    uint32_t sizeofdatatype = (src.dtype() == torch::kFloat16 || src.dtype() == torch::kInt16) ? 2 : 4;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivCanUseNum = ascendcPlatform->GetCoreNumAiv();

    SegmentMaxCsrTilingData tiling = ComputeTiling(
        src.numel(), dim.E_1, dim.M, dim.nSegments, dim.K,
        indptr.numel(), sizeofdatatype, aivCanUseNum,
        strideIndptr, hasOptionalOut ? 1 : 0);

    aclrtStream stream = c10_npu::getCurrentNPUStream(src.get_device()).stream();
    LaunchKernel(src, indptr, optional_out, out, hasOptionalOut, tiling, stream);

    return out;
}

}  // namespace opsgnn

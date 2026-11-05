/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "scatter.h"

#include <acl/acl.h>
#include <algorithm>
#include <c10/core/DeviceGuard.h>
#include <cstdint>
#include <vector>

#include "scatter/op_kernel/arch35/scatter.h"
#include "scatter/op_kernel/arch35/scatter_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

namespace {

uint64_t Product(const at::IntArrayRef& sizes, int64_t begin, int64_t end)
{
    uint64_t result = 1;
    for (int64_t axis = begin; axis < end; ++axis) {
        result *= static_cast<uint64_t>(sizes[axis]);
    }
    return result;
}

torch::Tensor EmptyNpuWorkspace(const at::IntArrayRef& sizes,
                                at::ScalarType dtype,
                                const c10::Device& device)
{
    const auto options = torch::TensorOptions()
                             .dtype(dtype)
                             .device(device)
                             .layout(torch::kStrided);
    return torch::empty(sizes, options);
}

torch::Tensor ZeroNpuWorkspace(const at::IntArrayRef& sizes,
                               at::ScalarType dtype,
                               const c10::Device& device)
{
    auto workspace = EmptyNpuWorkspace(sizes, dtype, device);
    workspace.zero_();
    return workspace;
}

ScatterDType ToScatterDType(at::ScalarType dtype)
{
    switch (dtype) {
        case at::ScalarType::Half:
            return SCATTER_FLOAT16;
        case at::ScalarType::BFloat16:
            return SCATTER_BFLOAT16;
        case at::ScalarType::Float:
            return SCATTER_FLOAT32;
        case at::ScalarType::Char:
            return SCATTER_INT8;
        case at::ScalarType::Short:
            return SCATTER_INT16;
        case at::ScalarType::Int:
            return SCATTER_INT32;
        case at::ScalarType::Byte:
            return SCATTER_UINT8;
        default:
            TORCH_CHECK(false, "scatter NPU kernel received an unsupported dtype");
    }
}

bool SupportsAtomicSum(at::ScalarType dtype)
{
    return dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16 ||
           dtype == at::ScalarType::Float || dtype == at::ScalarType::Int;
}

struct ScatterPlan {
    uint64_t before;
    uint64_t dimLength;
    uint64_t after;
    uint64_t outputDim;
    bool compactDimIndex;
    bool atomicPath;
    bool vectorAtomicPath;
    bool f16FloatAtomicPath;
    bool vectorMinMaxPath;
};

void ValidateScatterInputs(const torch::Tensor& src, const torch::Tensor& index,
                           const torch::Tensor& out, int64_t dim,
                           int64_t reduce, int64_t hotTarget)
{
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(index.scalar_type() == at::ScalarType::Long,
                "index must have int64 dtype");
    TORCH_CHECK(dim >= 0 && dim < src.dim(), "dim is out of range");
    TORCH_CHECK(reduce >= SCATTER_SUM && reduce <= SCATTER_MAX,
                "invalid scatter reduction code");
    TORCH_CHECK(hotTarget >= -1, "hot_target must be -1 or non-negative");
}

ScatterPlan BuildScatterPlan(const torch::Tensor& src,
                             const torch::Tensor& index,
                             const torch::Tensor& out, int64_t dim,
                             int64_t reduce, bool hasOut)
{
    const auto sizes = src.sizes();
    const uint64_t before = Product(sizes, 0, dim);
    const uint64_t dimLength = static_cast<uint64_t>(src.size(dim));
    const uint64_t after = Product(sizes, dim + 1, src.dim());
    const bool compact = index.dim() == 1 &&
        (index.numel() == 1 || static_cast<uint64_t>(index.numel()) == dimLength ||
         (index.numel() == 0 && dimLength == 0));
    TORCH_CHECK(compact || index.numel() == src.numel(),
                "index must be a broadcast dim vector or have src.numel elements");

    const bool atomicCandidate = SupportsAtomicSum(src.scalar_type()) &&
        (reduce == SCATTER_SUM || reduce == SCATTER_MEAN) && src.numel() >= 4096;
    const uint64_t rowBytes = after * static_cast<uint64_t>(src.element_size());
    const bool alignedRows = compact && after > 0 && rowBytes % 32 == 0;
    const bool vectorAtomic = atomicCandidate && src.scalar_type() == at::kFloat && alignedRows;
    const bool f16Atomic = atomicCandidate && !hasOut &&
        src.scalar_type() == at::kHalf && alignedRows;
    const bool orderedLowPrecision =
        (src.scalar_type() == at::kHalf && !f16Atomic) ||
        src.scalar_type() == at::kBFloat16;
    const bool atomic = atomicCandidate && !orderedLowPrecision;
    const uint64_t minMaxAlignmentElements =
        src.scalar_type() == at::kHalf ? 16 : 8;
    const bool vectorMinMax = !hasOut && src.numel() >= 4096 &&
        (reduce == SCATTER_MIN || reduce == SCATTER_MAX) &&
        (src.scalar_type() == at::kFloat || src.scalar_type() == at::kHalf) &&
        alignedRows &&
        static_cast<uint64_t>(out.numel()) % minMaxAlignmentElements == 0;
    return {before, dimLength, after, static_cast<uint64_t>(out.size(dim)),
            compact, atomic, vectorAtomic, f16Atomic, vectorMinMax};
}

torch::Tensor AllocateCount(const ScatterPlan& plan, const torch::Tensor& src,
                            const torch::Tensor& out, int64_t reduce)
{
    if (reduce == SCATTER_MEAN) {
        if (plan.vectorAtomicPath || plan.f16FloatAtomicPath) {
            return EmptyNpuWorkspace(
                {static_cast<int64_t>(plan.before * plan.outputDim)},
                torch::kInt32, out.device());
        }
        if (plan.atomicPath) {
            return ZeroNpuWorkspace(
                out.sizes(), out.scalar_type(), out.device());
        }
        return ZeroNpuWorkspace(out.sizes(), torch::kInt32, out.device());
    }
    if (plan.vectorMinMaxPath && src.scalar_type() == at::kHalf) {
        return EmptyNpuWorkspace(out.sizes(), torch::kFloat, out.device());
    }
    return out.flatten().slice(0, 0, 0);
}

torch::Tensor AllocateArgOut(const ScatterPlan& plan, const torch::Tensor& index,
                             const torch::Tensor& out, int64_t reduce)
{
    if (plan.f16FloatAtomicPath) {
        return EmptyNpuWorkspace(out.sizes(), torch::kFloat, out.device());
    }
    if (plan.vectorMinMaxPath) {
        return EmptyNpuWorkspace(out.sizes(), torch::kInt32, out.device());
    }
    if (reduce == SCATTER_MIN || reduce == SCATTER_MAX) {
        return EmptyNpuWorkspace(out.sizes(), index.scalar_type(), out.device());
    }
    return index.flatten().slice(0, 0, 0);
}

uint32_t SelectBlockNum(const ScatterPlan& plan, const torch::Tensor& src)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t available = std::max<uint32_t>(platform->GetCoreNumAiv(), 1);
    const bool vectorPath = plan.vectorAtomicPath || plan.f16FloatAtomicPath ||
                            plan.vectorMinMaxPath;
    const uint64_t workItems = vectorPath ? plan.before * plan.dimLength :
        (plan.atomicPath ? static_cast<uint64_t>(src.numel()) :
                           plan.before * plan.after);
    const uint64_t useful = std::max<uint64_t>((workItems + 255) / 256, 1);
    if (plan.vectorMinMaxPath) {
        available = std::min<uint32_t>(available, 48);
    } else if (plan.vectorAtomicPath || plan.f16FloatAtomicPath) {
        available = std::min<uint32_t>(available, 56);
    }
    uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(available, useful));
    if (plan.vectorMinMaxPath) {
        const uint64_t outNumel = plan.before * plan.outputDim * plan.after;
        const uint64_t alignmentElements =
            src.scalar_type() == at::kHalf ? 16 : 8;
        while (blocks > 1 &&
               outNumel % (static_cast<uint64_t>(blocks) * alignmentElements) != 0) {
            --blocks;
        }
    }
    return blocks;
}

uint32_t SelectTilingKey(const ScatterPlan& plan, int64_t reduce)
{
    if (plan.vectorMinMaxPath) {
        return reduce == SCATTER_MIN ? TILING_VECTOR_MIN : TILING_VECTOR_MAX;
    }
    if (plan.f16FloatAtomicPath) {
        return reduce == SCATTER_MEAN ? TILING_F16_FLOAT_ATOMIC_MEAN :
                                        TILING_F16_FLOAT_ATOMIC_SUM;
    }
    if (plan.vectorAtomicPath) {
        return reduce == SCATTER_MEAN ? TILING_VECTOR_ATOMIC_MEAN :
                                        TILING_VECTOR_ATOMIC_SUM;
    }
    return (plan.atomicPath ? 10U : 0U) + static_cast<uint32_t>(reduce);
}

ScatterTilingData BuildTiling(const ScatterPlan& plan,
                              const torch::Tensor& src,
                              const torch::Tensor& index,
                              const torch::Tensor& out, int64_t reduce,
                              bool hasOut, int64_t hotTarget)
{
    ScatterTilingData tiling{};
    tiling.beforeDim = plan.before;
    tiling.dimLength = plan.dimLength;
    tiling.afterDim = plan.after;
    tiling.outputDim = plan.outputDim;
    tiling.srcNumel = static_cast<uint64_t>(src.numel());
    tiling.outNumel = static_cast<uint64_t>(out.numel());
    tiling.laneCount = plan.before * plan.after;
    tiling.indexDimLength = static_cast<uint64_t>(index.numel());
    tiling.reduce = static_cast<uint32_t>(reduce);
    tiling.path = plan.vectorMinMaxPath ? SCATTER_VECTOR_MINMAX :
        (plan.f16FloatAtomicPath ? SCATTER_F16_FLOAT_ATOMIC :
         (plan.vectorAtomicPath ? SCATTER_VECTOR_ATOMIC :
          (plan.atomicPath ? SCATTER_ATOMIC : SCATTER_LANE_OWNER)));
    tiling.indexMode = plan.compactDimIndex ? SCATTER_INDEX_DIM_VECTOR :
                                              SCATTER_INDEX_FULL;
    tiling.tilingKey = SelectTilingKey(plan, reduce);
    tiling.blockNum = SelectBlockNum(plan, src);
    tiling.hasOut = hasOut ? 1 : 0;
    tiling.hotTarget = hotTarget;
    return tiling;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> scatter_forward(
    torch::Tensor src, torch::Tensor index, int64_t dim, torch::Tensor out,
    int64_t reduce, bool hasOut, int64_t hotTarget)
{
    ValidateScatterInputs(src, index, out, dim, reduce, hotTarget);
    const c10::DeviceGuard deviceGuard(src.device());
    const ScatterPlan plan = BuildScatterPlan(src, index, out, dim, reduce, hasOut);
    torch::Tensor count = AllocateCount(plan, src, out, reduce);
    torch::Tensor argOut = AllocateArgOut(plan, index, out, reduce);
    if (!hasOut && plan.atomicPath && !plan.vectorAtomicPath &&
        !plan.f16FloatAtomicPath) {
        out.zero_();
    }
    if (out.numel() == 0 || (plan.atomicPath && src.numel() == 0)) {
        return std::make_tuple(out, argOut);
    }
    const ScatterTilingData tiling = BuildTiling(
        plan, src, index, out, reduce, hasOut, hotTarget);
    aclrtStream stream = c10_npu::getCurrentNPUStream(src.get_device()).stream(true);
    Scatter(
        src.data_ptr(), index.data_ptr<int64_t>(), out.data_ptr(),
        count.numel() > 0 ? static_cast<int32_t*>(count.data_ptr()) : nullptr,
        argOut.numel() > 0 ? argOut.data_ptr() : nullptr,
        ToScatterDType(src.scalar_type()), tiling, stream);
    return std::make_tuple(out, argOut);
}

}  // namespace opsgnn

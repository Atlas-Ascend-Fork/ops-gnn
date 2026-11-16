/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_softmax/op_host/scatter_softmax.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "c10/core/DeviceGuard.h"
#include "common/aiv_launch_utils.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_launch.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_tiling.h"
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

ScatterSoftmaxDType ToScatterSoftmaxDType(at::ScalarType dtype)
{
    switch (dtype) {
        case at::ScalarType::Half:
            return SCATTER_SOFTMAX_F16;
        case at::ScalarType::BFloat16:
            return SCATTER_SOFTMAX_BF16;
        case at::ScalarType::Float:
            return SCATTER_SOFTMAX_F32;
        default:
            TORCH_CHECK(false, "scatter_softmax: unsupported source dtype");
    }
}

void ValidateInputs(const torch::Tensor& src, const torch::Tensor& index,
                    int64_t dim, int64_t dimSize)
{
    TORCH_CHECK(src.is_contiguous(), "scatter_softmax: src must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "scatter_softmax: index must be contiguous");
    TORCH_CHECK(index.scalar_type() == at::ScalarType::Int,
                "scatter_softmax: index must have int32 dtype");
    TORCH_CHECK(index.device() == src.device(),
                "scatter_softmax: src and index must be on the same device");
    TORCH_CHECK(src.dim() >= 1, "scatter_softmax: src must have at least one dimension");
    TORCH_CHECK(dim >= 0 && dim < src.dim(), "scatter_softmax: dim is out of range");
    TORCH_CHECK(dimSize >= 0, "scatter_softmax: dim_size must be non-negative");
    TORCH_CHECK(dimSize <= std::numeric_limits<int32_t>::max(),
                "scatter_softmax: dim_size exceeds the int32 index range");
    TORCH_CHECK(static_cast<uint64_t>(index.numel()) ==
                    static_cast<uint64_t>(src.size(dim)),
                "scatter_softmax: index length must match src.size(dim)");
    switch (src.scalar_type()) {
        case at::ScalarType::Half:
        case at::ScalarType::BFloat16:
        case at::ScalarType::Float:
            break;
        default:
            TORCH_CHECK(false,
                        "scatter_softmax: src dtype must be float16, bfloat16 or float32");
    }
}

ScatterSoftmaxTilingData BuildTiling(const torch::Tensor& src, int64_t dim,
                                     int64_t dimSize)
{
    const at::IntArrayRef sizes = src.sizes();
    ScatterSoftmaxTilingData tiling{};
    tiling.batch = Product(sizes, 0, dim);
    tiling.dimLength = static_cast<uint64_t>(src.size(dim));
    tiling.groupCount = static_cast<uint64_t>(dimSize);
    tiling.after = Product(sizes, dim + 1, src.dim());
    tiling.afterPadded =
        (tiling.after + SCATTER_SOFTMAX_WORKSPACE_ALIGN - 1) /
        SCATTER_SOFTMAX_WORKSPACE_ALIGN * SCATTER_SOFTMAX_WORKSPACE_ALIGN;
    tiling.rows = tiling.batch * tiling.dimLength;
    tiling.srcNumel = tiling.rows * tiling.after;
    tiling.workspaceElems = tiling.batch * tiling.groupCount * tiling.afterPadded;

    // Narrow rows keep a small UB footprint; wide rows are tiled along the
    // feature axis. `colTile` is also the stride between the per-row slices of
    // a batch, so it stays a multiple of SCATTER_SOFTMAX_ROW_ALIGN: every slice
    // then starts on a 32-byte boundary for both 2-byte and 4-byte elements.
    const uint64_t rowStep =
        (tiling.after + SCATTER_SOFTMAX_ROW_ALIGN - 1) / SCATTER_SOFTMAX_ROW_ALIGN *
        SCATTER_SOFTMAX_ROW_ALIGN;
    tiling.colTile = static_cast<uint32_t>(std::min<uint64_t>(
        SCATTER_SOFTMAX_MAX_COL_TILE,
        std::max<uint64_t>(rowStep, SCATTER_SOFTMAX_ROW_ALIGN)));

    const uint64_t requestedRows = std::max<uint64_t>(tiling.rows, 1);
    tiling.blockNum = ops_gnn::sparse::ResolveAivCoreNum(static_cast<uint32_t>(
        std::min<uint64_t>(requestedRows, std::numeric_limits<uint32_t>::max())));

    // A core owns a contiguous chunk of rows and buffers a batch of them, so
    // the GM round trip of one row overlaps the transfers of the rest.
    tiling.rowChunk = (tiling.rows + tiling.blockNum - 1) / tiling.blockNum;

    // When the whole reduction workspace fits the accumulator budget it stays
    // in UB for the duration of a pass: the reduction passes accumulate in
    // place and flush once per core, and the passes that read it read it once.
    const uint64_t accumBytes = tiling.workspaceElems * sizeof(float);
    tiling.localAccum =
        (tiling.workspaceElems != 0 && accumBytes <= SCATTER_SOFTMAX_ACC_BUDGET_BYTES)
            ? 1U : 0U;

    // The exp pass holds two copies at once - the cached maximum workspace and
    // the sum it accumulates into - so that is what the row budget reserves.
    const uint64_t reserved = (tiling.localAccum != 0) ? 2 * accumBytes : 0;
    const uint64_t rowBudget = (SCATTER_SOFTMAX_ROW_BUDGET_BYTES > reserved)
        ? (SCATTER_SOFTMAX_ROW_BUDGET_BYTES - reserved) : 0;
    // 20 bytes per (row, feature) covers both dtype buffers and three fp32 tiles.
    const uint64_t perRow = static_cast<uint64_t>(tiling.colTile) * 20;
    uint64_t batch = (perRow != 0) ? rowBudget / perRow : 1;
    batch = std::min<uint64_t>(batch, SCATTER_SOFTMAX_MAX_ROW_BATCH);
    batch = std::min<uint64_t>(batch, tiling.rowChunk);
    tiling.rowBatch = static_cast<uint32_t>(std::max<uint64_t>(batch, 1));
    return tiling;
}

void CheckLaunchable(const ScatterSoftmaxTilingData& tiling)
{
    TORCH_CHECK(tiling.workspaceElems <=
                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "scatter_softmax: reduction workspace is too large");
}

}  // namespace

torch::Tensor scatter_softmax_forward(const torch::Tensor& src,
                                     const torch::Tensor& index,
                                     int64_t dim, int64_t dimSize)
{
    ValidateInputs(src, index, dim, dimSize);
    const c10::DeviceGuard deviceGuard(src.device());

    torch::Tensor out = torch::empty(src.sizes(), src.options());
    const ScatterSoftmaxTilingData tiling = BuildTiling(src, dim, dimSize);
    if (tiling.rows == 0 || tiling.after == 0 || tiling.groupCount == 0) {
        return out;
    }
    CheckLaunchable(tiling);

    const torch::TensorOptions options =
        torch::TensorOptions().dtype(at::kFloat).device(src.device());
    const int64_t workspaceElems = static_cast<int64_t>(tiling.workspaceElems);
    // The maximum starts at the neutral value so the first atomic max wins, and
    // the sum starts empty for the atomic add of the second pass.
    torch::Tensor maximum = torch::full(
        {workspaceElems}, SCATTER_SOFTMAX_NEUTRAL_MAXIMUM, options);
    torch::Tensor total = torch::zeros({workspaceElems}, options);

    aclrtStream stream = c10_npu::getCurrentNPUStream(src.get_device());
    ScatterSoftmaxForward(src.data_ptr(), index.data_ptr<int32_t>(),
                          out.data_ptr(), maximum.data_ptr<float>(),
                          total.data_ptr<float>(),
                          ToScatterSoftmaxDType(src.scalar_type()), tiling, stream);
    return out;
}

torch::Tensor scatter_softmax_backward(const torch::Tensor& out,
                                       const torch::Tensor& gradOut,
                                       const torch::Tensor& index,
                                       int64_t dim, int64_t dimSize)
{
    ValidateInputs(out, index, dim, dimSize);
    TORCH_CHECK(gradOut.is_contiguous(),
                "scatter_softmax: grad_out must be contiguous");
    TORCH_CHECK(gradOut.sizes() == out.sizes(),
                "scatter_softmax: grad_out shape must match the forward output");
    TORCH_CHECK(gradOut.scalar_type() == out.scalar_type(),
                "scatter_softmax: grad_out dtype must match the forward output");
    const c10::DeviceGuard deviceGuard(out.device());

    torch::Tensor gradSrc = torch::empty(out.sizes(), out.options());
    const ScatterSoftmaxTilingData tiling = BuildTiling(out, dim, dimSize);
    if (tiling.rows == 0 || tiling.after == 0 || tiling.groupCount == 0) {
        return gradSrc;
    }
    CheckLaunchable(tiling);

    const torch::TensorOptions options =
        torch::TensorOptions().dtype(at::kFloat).device(out.device());
    const int64_t workspaceElems = static_cast<int64_t>(tiling.workspaceElems);
    torch::Tensor total = torch::zeros({workspaceElems}, options);

    aclrtStream stream = c10_npu::getCurrentNPUStream(out.get_device());
    ScatterSoftmaxBackward(out.data_ptr(), gradOut.data_ptr(),
                           index.data_ptr<int32_t>(), gradSrc.data_ptr(),
                           total.data_ptr<float>(),
                           ToScatterSoftmaxDType(out.scalar_type()), tiling,
                           stream);
    return gradSrc;
}

}  // namespace opsgnn

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Shared host-side argument validation and tiling of the grouped-softmax
 * family (scatter_softmax and scatter_log_softmax). Both operators reduce along
 * `dim` over the groups defined by `index` and share the same tiling and the
 * same fp32 reduction workspace, so they differ only in the device arithmetic.
 *
 * The header lives under scatter_softmax/common rather than csrc/npu/common
 * because it is specific to this operator family, not a repository-wide helper.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <torch/extension.h>

#include "acl/acl.h"
#include "c10/core/DeviceGuard.h"
#include "common/aiv_launch_utils.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

namespace grouped_softmax {

// Size of the (possibly empty) slice sizes[begin .. end-1).
inline uint64_t Product(const at::IntArrayRef& sizes, int64_t begin, int64_t end)
{
    uint64_t result = 1;
    for (int64_t axis = begin; axis < end; ++axis) {
        result *= static_cast<uint64_t>(sizes[axis]);
    }
    return result;
}

inline ScatterSoftmaxDType ResolveDType(at::ScalarType dtype, const char* op)
{
    switch (dtype) {
        case at::ScalarType::Half:
            return SCATTER_SOFTMAX_F16;
        case at::ScalarType::BFloat16:
            return SCATTER_SOFTMAX_BF16;
        case at::ScalarType::Float:
            return SCATTER_SOFTMAX_F32;
        default:
            TORCH_CHECK(false, op, ": unsupported source dtype");
    }
}

// Checks shared by the forward pass and its analytic backward: both run on the
// same `src`-shaped tensor and the same group vector.
inline void ValidateInputs(const char* op, const torch::Tensor& src,
                           const torch::Tensor& index, int64_t dim, int64_t dimSize)
{
    TORCH_CHECK(src.is_contiguous(), op, ": src must be contiguous");
    TORCH_CHECK(index.is_contiguous(), op, ": index must be contiguous");
    TORCH_CHECK(index.scalar_type() == at::ScalarType::Int,
                op, ": index must have int32 dtype");
    TORCH_CHECK(index.device() == src.device(),
                op, ": src and index must be on the same device");
    TORCH_CHECK(src.dim() >= 1, op, ": src must have at least one dimension");
    TORCH_CHECK(dim >= 0 && dim < src.dim(), op, ": dim is out of range");
    TORCH_CHECK(dimSize >= 0, op, ": dim_size must be non-negative");
    TORCH_CHECK(dimSize <= std::numeric_limits<int32_t>::max(),
                op, ": dim_size exceeds the int32 index range");
    TORCH_CHECK(static_cast<uint64_t>(index.numel()) ==
                    static_cast<uint64_t>(src.size(dim)),
                op, ": index length must match src.size(dim)");
    switch (src.scalar_type()) {
        case at::ScalarType::Half:
        case at::ScalarType::BFloat16:
        case at::ScalarType::Float:
            break;
        default:
            TORCH_CHECK(false, op, ": src dtype must be float16, bfloat16 or float32");
    }
}

inline ScatterSoftmaxTilingData BuildTiling(const torch::Tensor& src, int64_t dim,
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
    tiling.outRows = tiling.batch * tiling.groupCount;

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
    // The log-domain normalization holds the same two copies (maximum and sum),
    // so one reservation serves both operators.
    const uint64_t reserved = (tiling.localAccum != 0) ? 2 * accumBytes : 0;
    const uint64_t rowBudget = (SCATTER_SOFTMAX_ROW_BUDGET_BYTES > reserved)
        ? (SCATTER_SOFTMAX_ROW_BUDGET_BYTES - reserved) : 0;
    // 20 bytes per (row, feature) covers both dtype buffers and three fp32 tiles.
    const uint64_t perRow = static_cast<uint64_t>(tiling.colTile) * 20;
    uint64_t batch = (perRow != 0) ? rowBudget / perRow : 1;
    batch = std::min<uint64_t>(batch, SCATTER_SOFTMAX_MAX_ROW_BATCH);
    batch = std::min<uint64_t>(batch, tiling.rowChunk);
    tiling.rowBatch = static_cast<uint32_t>(std::max<uint64_t>(batch, 1));

    // The reduced output walks (batch, group) rows instead of source rows, so
    // its own core split is derived from `outRows`; the per-row buffer plan is
    // the same one, capped by the shorter chunk.
    const uint64_t requestedOutRows = std::max<uint64_t>(tiling.outRows, 1);
    tiling.outBlockNum = ops_gnn::sparse::ResolveAivCoreNum(static_cast<uint32_t>(
        std::min<uint64_t>(requestedOutRows, std::numeric_limits<uint32_t>::max())));
    tiling.outRowChunk =
        (tiling.outRows + tiling.outBlockNum - 1) / tiling.outBlockNum;
    uint64_t outBatch = std::min<uint64_t>(tiling.rowBatch, tiling.outRowChunk);
    tiling.outRowBatch = static_cast<uint32_t>(std::max<uint64_t>(outBatch, 1));
    return tiling;
}

inline void CheckLaunchable(const char* op, const ScatterSoftmaxTilingData& tiling)
{
    TORCH_CHECK(tiling.workspaceElems <=
                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                op, ": reduction workspace is too large");
}

// True when the shape has nothing to reduce, so both operators return the
// freshly allocated result without launching a kernel.
inline bool IsEmpty(const ScatterSoftmaxTilingData& tiling)
{
    return tiling.rows == 0 || tiling.after == 0 || tiling.groupCount == 0;
}

// Runs the forward scaffolding every grouped-softmax operator shares: argument
// checks, the empty case, the fp32 reduction workspace and the strided launch.
// `launch` receives the device entry point of one operator, so the per-operator
// host file only names its kernel.
template <typename Launch>
inline torch::Tensor RunForward(const char* op, const torch::Tensor& src,
                                const torch::Tensor& index, int64_t dim,
                                int64_t dimSize, Launch launch)
{
    ValidateInputs(op, src, index, dim, dimSize);
    const c10::DeviceGuard deviceGuard(src.device());

    torch::Tensor out = torch::empty(src.sizes(), src.options());
    const ScatterSoftmaxTilingData tiling = BuildTiling(src, dim, dimSize);
    if (IsEmpty(tiling)) {
        return out;
    }
    CheckLaunchable(op, tiling);

    const torch::TensorOptions options =
        torch::TensorOptions().dtype(at::kFloat).device(src.device());
    const int64_t workspaceElems = static_cast<int64_t>(tiling.workspaceElems);
    // The maximum starts at the neutral value so the first atomic max wins, and
    // the sum starts empty for the atomic add of the second pass. The log
    // normalization reads both again, so the forward allocates both.
    torch::Tensor maximum = torch::full(
        {workspaceElems}, SCATTER_SOFTMAX_NEUTRAL_MAXIMUM, options);
    torch::Tensor total = torch::zeros({workspaceElems}, options);

    launch(src.data_ptr(), index.data_ptr<int32_t>(), out.data_ptr(),
           maximum.data_ptr<float>(), total.data_ptr<float>(),
           ResolveDType(src.scalar_type(), op), tiling,
           c10_npu::getCurrentNPUStream(src.get_device()));
    return out;
}

// Backward scaffolding: the same checks on the saved output and the incoming
// gradient, and one fp32 group-sum workspace.
template <typename Launch>
inline torch::Tensor RunBackward(const char* op, const torch::Tensor& out,
                                 const torch::Tensor& gradOut,
                                 const torch::Tensor& index, int64_t dim,
                                 int64_t dimSize, Launch launch)
{
    ValidateInputs(op, out, index, dim, dimSize);
    TORCH_CHECK(gradOut.is_contiguous(),
                op, ": grad_out must be contiguous");
    TORCH_CHECK(gradOut.sizes() == out.sizes(),
                op, ": grad_out shape must match the forward output");
    TORCH_CHECK(gradOut.scalar_type() == out.scalar_type(),
                op, ": grad_out dtype must match the forward output");
    TORCH_CHECK(gradOut.device() == out.device(),
                op, ": grad_out must be on the same device as the output");
    const c10::DeviceGuard deviceGuard(out.device());

    torch::Tensor gradSrc = torch::empty(out.sizes(), out.options());
    const ScatterSoftmaxTilingData tiling = BuildTiling(out, dim, dimSize);
    if (IsEmpty(tiling)) {
        return gradSrc;
    }
    CheckLaunchable(op, tiling);

    const torch::TensorOptions options =
        torch::TensorOptions().dtype(at::kFloat).device(out.device());
    const int64_t workspaceElems = static_cast<int64_t>(tiling.workspaceElems);
    // Only the group sum of grad_out is accumulated here: the backward kernel
    // recovers the softmax probability from the saved forward output.
    torch::Tensor total = torch::zeros({workspaceElems}, options);

    launch(out.data_ptr(), gradOut.data_ptr(), index.data_ptr<int32_t>(),
           gradSrc.data_ptr(), total.data_ptr<float>(),
           ResolveDType(out.scalar_type(), op), tiling,
           c10_npu::getCurrentNPUStream(out.get_device()));
    return gradSrc;
}

}  // namespace grouped_softmax

}  // namespace opsgnn

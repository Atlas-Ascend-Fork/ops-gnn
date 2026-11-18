/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Host-side scaffolding of the grouped log-sum-exp.
 *
 * The operator reduces `src` along `dim` over the groups defined by `index` and
 * returns one row per (batch, group) pair, so its output has `dim_size` in place
 * of `src.size(dim)`. It shares the argument checks, the tiling and the fp32
 * reduction workspace of the grouped-softmax family; only the reduced output
 * shape and the empty-group policy are its own.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <torch/extension.h>

#include "acl/acl.h"
#include "c10/core/DeviceGuard.h"
#include "scatter_softmax/common/grouped_softmax_shared.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

namespace grouped_logsumexp {

// `src` shape with `dim` replaced by `dimSize`.
inline std::vector<int64_t> ReducedSizes(const torch::Tensor& src, int64_t dim,
                                         int64_t dimSize)
{
    std::vector<int64_t> sizes(src.sizes().begin(), src.sizes().end());
    sizes[dim] = dimSize;
    return sizes;
}

// Forward scaffolding: argument checks, the empty cases, the fp32 reduction
// workspace and the three-pass launch. `launch` receives the device entry point,
// so the per-operator host file only names its kernel.
template <typename Launch>
inline torch::Tensor RunForward(const char* op, const torch::Tensor& src,
                                const torch::Tensor& index, int64_t dim,
                                int64_t dimSize, Launch launch)
{
    grouped_softmax::ValidateInputs(op, src, index, dim, dimSize);
    const c10::DeviceGuard deviceGuard(src.device());

    torch::Tensor out = torch::empty(ReducedSizes(src, dim, dimSize), src.options());
    const ScatterSoftmaxTilingData tiling = grouped_softmax::BuildTiling(src, dim, dimSize);
    if (tiling.rows == 0) {
        // No member at all: every group is empty, and the log-sum-exp of an
        // empty group is `-inf`. The kernel refuses a zero-row grid, so the
        // whole output is written here.
        out.fill_(-std::numeric_limits<float>::infinity());
        return out;
    }
    if (tiling.after == 0 || tiling.groupCount == 0) {
        // The reduced output has no element to write.
        return out;
    }
    grouped_softmax::CheckLaunchable(op, tiling);

    const torch::TensorOptions options =
        torch::TensorOptions().dtype(at::kFloat).device(src.device());
    const int64_t workspaceElems = static_cast<int64_t>(tiling.workspaceElems);
    // The maximum starts at the neutral value so the first atomic max wins and
    // an empty group keeps it, and the sum starts empty for the atomic add of
    // the second pass. The finalize pass reads both.
    torch::Tensor maximum = torch::full(
        {workspaceElems}, SCATTER_SOFTMAX_NEUTRAL_MAXIMUM, options);
    torch::Tensor total = torch::zeros({workspaceElems}, options);

    launch(src.data_ptr(), index.data_ptr<int32_t>(), out.data_ptr(),
           maximum.data_ptr<float>(), total.data_ptr<float>(),
           grouped_softmax::ResolveDType(src.scalar_type(), op), tiling,
           c10_npu::getCurrentNPUStream(src.get_device()));
    return out;
}

// Backward scaffolding. The analytic gradient needs the source itself
// (`grad_out[group] * exp(src - out[group])`), so the saved source is an
// argument rather than being recovered from the output.
template <typename Launch>
inline torch::Tensor RunBackward(const char* op, const torch::Tensor& src,
                                 const torch::Tensor& out, const torch::Tensor& gradOut,
                                 const torch::Tensor& index, int64_t dim,
                                 int64_t dimSize, Launch launch)
{
    grouped_softmax::ValidateInputs(op, src, index, dim, dimSize);
    const std::vector<int64_t> expected = ReducedSizes(src, dim, dimSize);
    TORCH_CHECK(out.is_contiguous(), op, ": out must be contiguous");
    TORCH_CHECK(out.sizes() == at::IntArrayRef(expected),
                op, ": out shape must be src with dim replaced by dim_size");
    TORCH_CHECK(out.scalar_type() == src.scalar_type(),
                op, ": out dtype must match src");
    TORCH_CHECK(out.device() == src.device(),
                op, ": out must be on the same device as src");
    TORCH_CHECK(gradOut.is_contiguous(), op, ": grad_out must be contiguous");
    TORCH_CHECK(gradOut.sizes() == out.sizes(),
                op, ": grad_out shape must match the forward output");
    TORCH_CHECK(gradOut.scalar_type() == out.scalar_type(),
                op, ": grad_out dtype must match the forward output");
    TORCH_CHECK(gradOut.device() == out.device(),
                op, ": grad_out must be on the same device as the output");
    const c10::DeviceGuard deviceGuard(src.device());

    if (dimSize == 0) {
        // No group exists, so the reduced output holds no element and the
        // gradient of the source is identically zero.
        return torch::zeros(src.sizes(), src.options());
    }
    torch::Tensor gradSrc = torch::empty(src.sizes(), src.options());
    const ScatterSoftmaxTilingData tiling = grouped_softmax::BuildTiling(src, dim, dimSize);
    if (tiling.rows == 0 || tiling.after == 0) {
        return gradSrc;
    }

    launch(src.data_ptr(), out.data_ptr(), gradOut.data_ptr(),
           index.data_ptr<int32_t>(), gradSrc.data_ptr(),
           grouped_softmax::ResolveDType(src.scalar_type(), op), tiling,
           c10_npu::getCurrentNPUStream(src.get_device()));
    return gradSrc;
}

}  // namespace grouped_logsumexp

}  // namespace opsgnn

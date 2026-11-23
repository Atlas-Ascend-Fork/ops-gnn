/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Host-side scaffolding of the grouped standard deviation.
 *
 * The operator reduces `src` along `dim` over the groups defined by `index` and
 * returns one row per (batch, group) pair, so its output has `dim_size` in place
 * of `src.size(dim)`. It shares the argument checks, the tiling and the fp32
 * reduction workspace of the grouped-softmax family; what it adds is the member
 * count of every group and the second workspace the centered squared deviations
 * are accumulated into.
 */

#pragma once

// The shared header brings the torch, acl and NPU-stream declarations below,
// and it is the only include the two reduced-output operators list: keeping it
// single keeps their scaffolding from repeating the same block.
#include "scatter_softmax/common/grouped_softmax_shared.h"

namespace opsgnn {

namespace grouped_std {

// Forward scaffolding: argument checks, the two fp32 reduction workspaces, the
// member-count workspace and the three-pass launch. `launch` receives the device
// entry point, so the per-operator host file only names its kernel.
//
// The result is `{out, means, counts}`: the reduced output and the two fp32
// tensors the backward reuses. `means` is the group-sum workspace after the
// finalize pass replaced every sum with the mean it derived from it.
template <typename Launch>
inline std::vector<torch::Tensor> RunForward(const char* op, const torch::Tensor& src,
                                             const torch::Tensor& index, int64_t dim,
                                             int64_t dimSize, bool unbiased, Launch launch)
{
    grouped_softmax::ValidateInputs(op, src, index, dim, dimSize);
    const c10::DeviceGuard deviceGuard(src.device());

    torch::Tensor out = torch::empty(
        grouped_softmax::ReducedSizes(src, dim, dimSize), src.options());
    ScatterSoftmaxTilingData tiling =
        grouped_softmax::BuildTiling(src, dim, dimSize, true);
    tiling.correction = unbiased ? 1.0F : 0.0F;

    const torch::TensorOptions options =
        torch::TensorOptions().dtype(at::kFloat).device(src.device());
    const int64_t workspaceElems = static_cast<int64_t>(tiling.workspaceElems);
    const int64_t countElems = static_cast<int64_t>(tiling.countElems);
    // The group sums and the squared deviations start empty. The finalize pass
    // turns the first into the group means, which is what the backward reads;
    // the counts are accumulated by the sum pass.
    torch::Tensor means = torch::zeros({workspaceElems}, options);
    torch::Tensor deviation = torch::zeros({workspaceElems}, options);
    torch::Tensor counts = torch::zeros({countElems}, options);

    if (tiling.rows == 0) {
        // No member at all: every group is empty, and a standard deviation
        // without members is NaN, like torch.std. The mean workspace follows
        // the same rule, so no group reports a mean it does not have.
        out.fill_(std::numeric_limits<float>::quiet_NaN());
        means.fill_(std::numeric_limits<float>::quiet_NaN());
        return {out, means, counts};
    }
    if (tiling.after == 0 || tiling.groupCount == 0) {
        // The reduced output has no element to write.
        return {out, means, counts};
    }
    grouped_softmax::CheckLaunchable(op, tiling);

    launch(src.data_ptr(), index.data_ptr<int32_t>(), out.data_ptr(),
           means.data_ptr<float>(), deviation.data_ptr<float>(), counts.data_ptr<float>(),
           grouped_softmax::ResolveDType(src.scalar_type(), op), tiling,
           c10_npu::getCurrentNPUStream(src.get_device()));
    return {out, means, counts};
}

// Backward scaffolding. The analytic gradient needs the source, the saved
// output, the group means and the member counts, so all four are arguments
// rather than being recovered from the output.
template <typename Launch>
inline torch::Tensor RunBackward(const char* op, const torch::Tensor& src,
                                 const torch::Tensor& out, const torch::Tensor& gradOut,
                                 const torch::Tensor& mean, const torch::Tensor& count,
                                 const torch::Tensor& index, int64_t dim, int64_t dimSize,
                                 bool unbiased, Launch launch)
{
    ScatterSoftmaxTilingData tiling =
        grouped_softmax::BeginReducedBackward(op, src, out, gradOut, index, dim, dimSize, true);
    const c10::DeviceGuard deviceGuard(src.device());

    // The two workspaces the forward saved: the group means and the counts.
    TORCH_CHECK(mean.is_contiguous(), op, ": mean workspace must be contiguous");
    TORCH_CHECK(mean.scalar_type() == at::ScalarType::Float,
                op, ": mean workspace must have float32 dtype");
    TORCH_CHECK(mean.device() == src.device(),
                op, ": mean workspace must be on the same device as src");
    TORCH_CHECK(count.is_contiguous(), op, ": count workspace must be contiguous");
    TORCH_CHECK(count.scalar_type() == at::ScalarType::Float,
                op, ": count workspace must have float32 dtype");
    TORCH_CHECK(count.device() == src.device(),
                op, ": count workspace must be on the same device as src");

    if (dimSize == 0) {
        // No group exists, so the reduced output holds no element and the
        // gradient of the source is identically zero.
        return torch::zeros(src.sizes(), src.options());
    }
    torch::Tensor gradSrc = torch::empty(src.sizes(), src.options());
    tiling.correction = unbiased ? 1.0F : 0.0F;
    TORCH_CHECK(static_cast<int64_t>(mean.numel()) ==
                    static_cast<int64_t>(tiling.workspaceElems),
                op, ": mean workspace does not match the forward shape");
    TORCH_CHECK(static_cast<int64_t>(count.numel()) ==
                    static_cast<int64_t>(tiling.countElems),
                op, ": count workspace does not match the forward shape");
    if (tiling.rows == 0 || tiling.after == 0) {
        return gradSrc;
    }

    launch(src.data_ptr(), out.data_ptr(), gradOut.data_ptr(), index.data_ptr<int32_t>(),
           gradSrc.data_ptr(), mean.data_ptr<float>(), count.data_ptr<float>(),
           grouped_softmax::ResolveDType(src.scalar_type(), op), tiling,
           c10_npu::getCurrentNPUStream(src.get_device()));
    return gradSrc;
}

}  // namespace grouped_std

}  // namespace opsgnn

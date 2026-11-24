/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Host-side scaffolding of the edge softmax.
 *
 * The operator normalizes every CSR segment of `src` along axis 0 and returns a
 * tensor shaped like `src`: a segment is a contiguous row range, so the grouped
 * softmax of the scatter family applies unchanged once the group labels are read
 * from `indptr` instead of from a per-row vector. What differs on the host is
 * therefore the argument check - a CSR pointer instead of a label vector - while
 * the tiling and the fp32 reduction workspace are the ones of the family.
 */

#pragma once

#include <limits>

#include "scatter_softmax/common/grouped_softmax_shared.h"

namespace opsgnn {

namespace edge_softmax {

// Segments a CSR pointer vector describes: one less than its number of
// boundaries.
inline int64_t SegmentCount(const torch::Tensor& indptr)
{
    return static_cast<int64_t>(indptr.numel()) - 1;
}

// The source of the forward and the saved output of the backward are both
// `src`-shaped tensors reduced along axis 0, so the two calls share one check.
// The pointer values themselves are validated on the host side of the public
// API: the kernel clamps its cursor to the last segment, so it never indexes
// outside `indptr` even when the vector is not a valid CSR pointer.
//
// A pointer of fewer than two boundaries describes no segment at all. Rejecting
// it here matters because the empty cases below return an uninitialised tensor
// of `src`'s shape: with zero segments that tensor would hold no result at all,
// so a direct C++ caller would silently get garbage instead of an error.
inline void ValidateCsr(const char* op, const torch::Tensor& src,
                        const torch::Tensor& indptr)
{
    TORCH_CHECK(src.is_contiguous(), op, ": src must be contiguous");
    TORCH_CHECK(src.dim() >= 1, op, ": src must have at least one dimension");
    TORCH_CHECK(indptr.is_contiguous(), op, ": indptr must be contiguous");
    TORCH_CHECK(indptr.scalar_type() == at::ScalarType::Int,
                op, ": indptr must have int32 dtype");
    TORCH_CHECK(indptr.device() == src.device(),
                op, ": src and indptr must be on the same device");
    TORCH_CHECK(indptr.dim() == 1, op, ": indptr must be 1-D");
    TORCH_CHECK(indptr.numel() >= 2,
                op, ": indptr must contain at least two boundaries (one segment)");
    // The kernel keeps one int32 label per row and one int32 cursor per segment.
    TORCH_CHECK(SegmentCount(indptr) <= std::numeric_limits<int32_t>::max(),
                op, ": indptr describes more segments than the int32 range");
    TORCH_CHECK(src.size(0) <= std::numeric_limits<int32_t>::max(),
                op, ": src.size(0) exceeds the int32 range");
    switch (src.scalar_type()) {
        case at::ScalarType::Half:
        case at::ScalarType::BFloat16:
        case at::ScalarType::Float:
            break;
        default:
            TORCH_CHECK(false, op, ": src dtype must be float16, bfloat16 or float32");
    }
}

// The reduction runs along axis 0 over the CSR segments, so the grouped tiling
// sees a single batch, one group per segment and `src.size(0)` rows.
inline ScatterSoftmaxTilingData BuildCsrTiling(const torch::Tensor& src,
                                               const torch::Tensor& indptr)
{
    return grouped_softmax::BuildTiling(src, 0, SegmentCount(indptr));
}

// Forward scaffolding every CSR softmax shares: the argument checks, the empty
// case, the fp32 reduction workspace and the three-pass launch. `launch`
// receives the device entry point, so the per-operator host file only names its
// kernel.
template <typename Launch>
inline torch::Tensor RunForward(const char* op, const torch::Tensor& src,
                                const torch::Tensor& indptr, Launch launch)
{
    ValidateCsr(op, src, indptr);
    const c10::DeviceGuard deviceGuard(src.device());

    torch::Tensor out = torch::empty(src.sizes(), src.options());
    const ScatterSoftmaxTilingData tiling = BuildCsrTiling(src, indptr);
    if (grouped_softmax::IsEmpty(tiling)) {
        return out;
    }
    grouped_softmax::CheckLaunchable(op, tiling);

    const torch::TensorOptions options =
        torch::TensorOptions().dtype(at::kFloat).device(src.device());
    const int64_t workspaceElems = static_cast<int64_t>(tiling.workspaceElems);
    // The maximum starts at the neutral value so the first atomic max wins, and
    // the sum starts empty for the atomic add of the second pass.
    torch::Tensor maximum = torch::full(
        {workspaceElems}, SCATTER_SOFTMAX_NEUTRAL_MAXIMUM, options);
    torch::Tensor total = torch::zeros({workspaceElems}, options);

    launch(src.data_ptr(), indptr.data_ptr<int32_t>(), out.data_ptr(),
           maximum.data_ptr<float>(), total.data_ptr<float>(),
           grouped_softmax::ResolveDType(src.scalar_type(), op), tiling,
           c10_npu::getCurrentNPUStream(src.get_device()));
    return out;
}

// Backward scaffolding: the same CSR checks on the saved output, the incoming
// gradient check and one fp32 segment-sum workspace.
template <typename Launch>
inline torch::Tensor RunBackward(const char* op, const torch::Tensor& out,
                                 const torch::Tensor& gradOut,
                                 const torch::Tensor& indptr, Launch launch)
{
    ValidateCsr(op, out, indptr);
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
    const ScatterSoftmaxTilingData tiling = BuildCsrTiling(out, indptr);
    if (grouped_softmax::IsEmpty(tiling)) {
        return gradSrc;
    }
    grouped_softmax::CheckLaunchable(op, tiling);

    const torch::TensorOptions options =
        torch::TensorOptions().dtype(at::kFloat).device(out.device());
    // Only the segment sum of out * grad_out is accumulated here: the backward
    // kernel recovers the probability from the saved forward output.
    torch::Tensor total = torch::zeros(
        {static_cast<int64_t>(tiling.workspaceElems)}, options);

    launch(out.data_ptr(), gradOut.data_ptr(), indptr.data_ptr<int32_t>(),
           gradSrc.data_ptr(), total.data_ptr<float>(),
           grouped_softmax::ResolveDType(out.scalar_type(), op), tiling,
           c10_npu::getCurrentNPUStream(out.get_device()));
    return gradSrc;
}

}  // namespace edge_softmax

}  // namespace opsgnn

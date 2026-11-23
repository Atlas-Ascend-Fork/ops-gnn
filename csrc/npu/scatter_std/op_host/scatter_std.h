/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_STD_OP_HOST_H
#define OPS_GNN_SCATTER_STD_OP_HOST_H

#include <torch/extension.h>

namespace opsgnn {

// Grouped standard deviation along `dim`: every group defined by `index` is
// reduced to `sqrt(sum((src - mean)^2) / (n - 1))` (`unbiased`) and the result
// has one row per (batch, group) pair, so the output replaces `src.size(dim)`
// with `dimSize`. `src` must be contiguous, `index` a contiguous int32 vector of
// length src.size(dim), and `dimSize` the number of groups.
//
// The forward also returns the two fp32 tensors the backward reuses: the group
// means, one row per (batch, group) pair, and the member count of every group.
std::vector<torch::Tensor> scatter_std_forward(const torch::Tensor& src,
                                               const torch::Tensor& index,
                                               int64_t dim, int64_t dimSize,
                                               bool unbiased);

// Analytic grouped standard deviation backward:
//
//   grad_src = grad_out[group] * (src - mean[group]) / ((n - correction) * std)
//
// `src` is the forward input, `out` the forward result, `gradOut` the upstream
// gradient of the reduced output, and `mean` / `count` are the two tensors the
// forward returned. The result has the shape and dtype of `src`.
torch::Tensor scatter_std_backward(const torch::Tensor& src,
                                   const torch::Tensor& out,
                                   const torch::Tensor& gradOut,
                                   const torch::Tensor& mean,
                                   const torch::Tensor& count,
                                   const torch::Tensor& index,
                                   int64_t dim, int64_t dimSize,
                                   bool unbiased);

}  // namespace opsgnn

#endif  // OPS_GNN_SCATTER_STD_OP_HOST_H

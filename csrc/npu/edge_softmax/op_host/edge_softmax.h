/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_EDGE_SOFTMAX_OP_HOST_H
#define OPS_GNN_EDGE_SOFTMAX_OP_HOST_H

#include <torch/extension.h>

namespace opsgnn {

// Softmax over the edges of every CSR segment. `src` must be contiguous with
// shape `(E, ...)`, `indptr` a contiguous int32 vector of `segments + 1`
// boundaries with `indptr[0] == 0` and `indptr[segments] == E`. Segment `i`
// normalizes the rows `[indptr[i], indptr[i + 1])` along axis 0, every trailing
// feature independently, and the result has the shape and dtype of `src`.
torch::Tensor edge_softmax_forward(const torch::Tensor& src,
                                   const torch::Tensor& indptr);

// Analytic backward of the edge softmax. `out` is the forward result and
// `gradOut` its upstream gradient, both `src`-shaped; `indptr` is the forward
// pointer vector. The result is
//
//   grad_src[e] = out[e] * (grad_out[e] - sum_{j in segment(e)} out[j] * grad_out[j])
//
// with the shape, dtype and device of `src`.
torch::Tensor edge_softmax_backward(const torch::Tensor& out,
                                    const torch::Tensor& gradOut,
                                    const torch::Tensor& indptr);

}  // namespace opsgnn

#endif  // OPS_GNN_EDGE_SOFTMAX_OP_HOST_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_LOG_SOFTMAX_OP_HOST_H
#define OPS_GNN_SCATTER_LOG_SOFTMAX_OP_HOST_H

#include <torch/extension.h>

namespace opsgnn {

// Grouped log-softmax along `dim`: every group defined by `index` is normalized
// on its own and returned in the log domain. `src` must be contiguous, `index`
// a contiguous int32 vector of length src.size(dim), and `dimSize` the number of
// groups.
torch::Tensor scatter_log_softmax_forward(const torch::Tensor& src,
                                          const torch::Tensor& index,
                                          int64_t dim, int64_t dimSize);

// Analytic grouped log-softmax backward. `out` is the forward result (the log
// probability), `gradOut` the upstream gradient; the returned tensor has the
// shape and dtype of `out`.
torch::Tensor scatter_log_softmax_backward(const torch::Tensor& out,
                                           const torch::Tensor& gradOut,
                                           const torch::Tensor& index,
                                           int64_t dim, int64_t dimSize);

}  // namespace opsgnn

#endif  // OPS_GNN_SCATTER_LOG_SOFTMAX_OP_HOST_H

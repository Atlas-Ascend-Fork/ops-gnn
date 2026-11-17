/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_log_softmax/op_host/scatter_log_softmax.h"

#include "scatter_softmax/common/grouped_softmax_shared.h"
#include "scatter_log_softmax/op_kernel/common/scatter_log_softmax_launch.h"

namespace opsgnn {

namespace {
constexpr const char* kOpName = "scatter_log_softmax";
}  // namespace

torch::Tensor scatter_log_softmax_forward(const torch::Tensor& src,
                                          const torch::Tensor& index,
                                          int64_t dim, int64_t dimSize)
{
    return grouped_softmax::RunForward(kOpName, src, index, dim, dimSize,
                                       ScatterLogSoftmaxForward);
}

torch::Tensor scatter_log_softmax_backward(const torch::Tensor& out,
                                           const torch::Tensor& gradOut,
                                           const torch::Tensor& index,
                                           int64_t dim, int64_t dimSize)
{
    return grouped_softmax::RunBackward(kOpName, out, gradOut, index, dim,
                                        dimSize, ScatterLogSoftmaxBackward);
}

}  // namespace opsgnn

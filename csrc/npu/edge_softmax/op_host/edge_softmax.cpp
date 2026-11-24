/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "edge_softmax/op_host/edge_softmax.h"

#include "edge_softmax/common/edge_softmax_shared.h"
#include "edge_softmax/op_kernel/common/edge_softmax_launch.h"

namespace opsgnn {

namespace {
constexpr const char* kOpName = "edge_softmax";
}  // namespace

torch::Tensor edge_softmax_forward(const torch::Tensor& src,
                                   const torch::Tensor& indptr)
{
    return edge_softmax::RunForward(kOpName, src, indptr, EdgeSoftmaxForward);
}

torch::Tensor edge_softmax_backward(const torch::Tensor& out,
                                    const torch::Tensor& gradOut,
                                    const torch::Tensor& indptr)
{
    return edge_softmax::RunBackward(kOpName, out, gradOut, indptr,
                                     EdgeSoftmaxBackward);
}

}  // namespace opsgnn

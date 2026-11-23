/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_std/op_host/scatter_std.h"

#include "scatter_std/common/grouped_std_shared.h"
#include "scatter_std/op_kernel/common/scatter_std_launch.h"

namespace opsgnn {

namespace {
constexpr const char* kOpName = "scatter_std";
}  // namespace

std::vector<torch::Tensor> scatter_std_forward(const torch::Tensor& src,
                                               const torch::Tensor& index,
                                               int64_t dim, int64_t dimSize,
                                               bool unbiased)
{
    return grouped_std::RunForward(kOpName, src, index, dim, dimSize, unbiased,
                                   ScatterStdForward);
}

torch::Tensor scatter_std_backward(const torch::Tensor& src,
                                   const torch::Tensor& out,
                                   const torch::Tensor& gradOut,
                                   const torch::Tensor& mean,
                                   const torch::Tensor& count,
                                   const torch::Tensor& index,
                                   int64_t dim, int64_t dimSize,
                                   bool unbiased)
{
    return grouped_std::RunBackward(kOpName, src, out, gradOut, mean, count, index,
                                    dim, dimSize, unbiased, ScatterStdBackward);
}

}  // namespace opsgnn

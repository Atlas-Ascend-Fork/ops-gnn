/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_logsumexp/op_host/scatter_logsumexp.h"

#include "scatter_logsumexp/common/grouped_logsumexp_shared.h"
#include "scatter_logsumexp/op_kernel/common/scatter_logsumexp_launch.h"

namespace opsgnn {

namespace {
constexpr const char* kOpName = "scatter_logsumexp";
}  // namespace

torch::Tensor scatter_logsumexp_forward(const torch::Tensor& src,
                                        const torch::Tensor& index,
                                        int64_t dim, int64_t dimSize)
{
    return grouped_logsumexp::RunForward(kOpName, src, index, dim, dimSize,
                                         ScatterLogSumExpForward);
}

torch::Tensor scatter_logsumexp_backward(const torch::Tensor& src,
                                         const torch::Tensor& out,
                                         const torch::Tensor& gradOut,
                                         const torch::Tensor& index,
                                         int64_t dim, int64_t dimSize)
{
    return grouped_logsumexp::RunBackward(kOpName, src, out, gradOut, index, dim,
                                          dimSize, ScatterLogSumExpBackward);
}

}  // namespace opsgnn

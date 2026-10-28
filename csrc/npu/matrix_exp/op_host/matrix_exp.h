/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <acl/acl.h>
#include <torch/extension.h>

namespace opsgnn {

// 计算批量方阵的矩阵指数 exp(A)，接口与语义对齐 torch.linalg.matrix_exp（设计文档 1.4）。
// A: (..., n, n)，dtype ∈ {float32, float16, bfloat16}；
// out: 可选输出张量，传空张量表示未提供。
torch::Tensor matrix_exp(torch::Tensor A, torch::Tensor out = torch::Tensor());

}  // namespace opsgnn

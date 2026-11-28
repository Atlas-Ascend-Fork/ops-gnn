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
#include <ATen/ATen.h>
#include <tuple>

namespace opsgnn {

// Sparse x sparse matrix multiplication on Ascend NPU (950 and A2/A3).
//
// ``indexA``/``indexB`` are int64 COO indices of shape ``[2, nnz]``; the values
// are 1-D and share one floating dtype. The result is the coalesced COO product
// in row-major, column-ascending order.
std::tuple<at::Tensor, at::Tensor> SpspmmForward(at::Tensor indexA, at::Tensor valueA,
                                                 at::Tensor indexB, at::Tensor valueB,
                                                 int64_t m, int64_t k, int64_t n);

}  // namespace opsgnn

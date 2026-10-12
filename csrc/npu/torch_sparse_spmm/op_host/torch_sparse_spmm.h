/**
 * Copyright (c) 2026 Eternal_ZCL.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <tuple>

namespace opsgnn {
std::tuple<at::Tensor, at::Tensor> TorchSparseSpmmForward(at::Tensor ptr, at::Tensor col,
                                                          c10::optional<at::Tensor> value, at::Tensor mat, int reduce,
                                                          int outType);
at::Tensor TorchSparseSpmmValidate(at::Tensor ptr, at::Tensor col, int64_t n);
at::Tensor TorchSparseSpmmValueGrad(at::Tensor ptr, at::Tensor col, at::Tensor mat, at::Tensor grad, int reduce,
                                    int64_t width);
at::Tensor TorchSparseSpmmConvert(at::Tensor src, int64_t output, bool toPtr);
at::Tensor TorchSparseSpmmPermuteValues(at::Tensor value, at::Tensor perm);
c10::optional<int64_t> TorchSparseSpmmTensorVersion(const at::Tensor& tensor);
}  // namespace opsgnn

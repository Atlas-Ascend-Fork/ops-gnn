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

#include <optional>
#include <tuple>

#include <torch/extension.h>

namespace opsgnn {

// Structural transpose of an (N, M) CSR matrix into an (M, N) CSR matrix.
//
// `indptr` holds the N + 1 row boundaries of the input, `indices` its E column
// ids. When E is positive `indptr` must hold at least two boundaries, otherwise
// no edge has a row to report. `numCols` is the number of rows of the result;
// when it is omitted the operator derives it from `indices.max() + 1`, which
// synchronises the stream.
//
// Returns `outIndptr` (M + 1 boundaries) and `outIndices` (E row ids), both
// int64 and on the same NPU as the input. The row ids of every output column are
// ascending.
std::tuple<torch::Tensor, torch::Tensor> CsrTranspose(const torch::Tensor& indptr,
                                                      const torch::Tensor& indices,
                                                      std::optional<int64_t> numCols);

}  // namespace opsgnn

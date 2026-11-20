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

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace opsgnn {

// DGL `DisjointUnionCoo` on Ascend NPU: concatenate COO matrices into one.
//
// Graph i contributes its source/dest ids shifted by the accumulated row/column
// counts of the preceding graphs, and its data array shifted by the accumulated
// nnz. A graph whose `datas[i]` is empty contributes its edge index instead,
// matching the implicit `Range(0, nnz)` DGL materialises for COO matrices
// without an explicit data array.
//
// Returns (source, dest, data, total_rows, total_cols). `data` is an undefined
// tensor when no graph carries a data array.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t> DisjointUnionCoo(
    const std::vector<torch::Tensor>& sources, const std::vector<torch::Tensor>& dests,
    const std::vector<int64_t>& numRows, const std::vector<int64_t>& numCols,
    const std::vector<std::optional<torch::Tensor>>& datas);

}  // namespace opsgnn

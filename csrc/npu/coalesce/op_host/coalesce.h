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
#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <tuple>

namespace opsgnn {

// PyG-compatible COO coalescing on Ascend NPU.
//
// Sorts edges by (row, col) -- or (col, row) when `sort_by_row` is false --
// merges duplicate edges, and reduces edge_attr per group with `reduce`
// (0=sum, 1=mean, 2=min, 3=max). When `edge_attr` is absent the second tensor
// of the returned tuple is undefined (Python None).
std::tuple<torch::Tensor, torch::Tensor> CoalesceForward(
    torch::Tensor edge_index,
    c10::optional<torch::Tensor> edge_attr,
    c10::optional<int64_t> num_nodes,
    int64_t reduce,
    bool is_sorted,
    bool sort_by_row);

} // namespace opsgnn

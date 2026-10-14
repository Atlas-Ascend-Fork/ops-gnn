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

namespace opsgnn {

torch::Tensor torch_sparse_random_walk_npu(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor start,
    int64_t walkLength);

// Exact-verification variant accepting a caller supplied [S, L] float32 rand
// tensor (RNG-injection mode used by the precision tests).
torch::Tensor torch_sparse_random_walk_with_rand_npu(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor start,
    int64_t walkLength,
    torch::Tensor rand);

} // namespace opsgnn

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
#include <cstdint>

namespace opsgnn {

constexpr int64_t KNN_LOCAL_TOPK_LIMIT = 64;
constexpr int64_t KNN_TREE_NODE_FIELDS = 8;

template <typename T, int DType = 0>
void Knn(
    T* x, T* y, int64_t* ptr_x, int64_t* ptr_y,
    int64_t* row, int64_t* col, int64_t k, int64_t n, int64_t m,
    int64_t feature_dim, int64_t batch_size, bool cosine,
    float* workspace, int64_t* tree, float* bounds, aclrtStream stream);

}  // namespace opsgnn

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
#include <torch/extension.h>

namespace opsgnn {

torch::Tensor knn_npu(
    torch::Tensor x,
    torch::Tensor y,
    std::optional<torch::Tensor> ptr_x,
    std::optional<torch::Tensor> ptr_y,
    int64_t k,
    bool cosine,
    int64_t num_workers,
    int64_t stream_handle);

}  // namespace opsgnn

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

std::tuple<torch::Tensor, torch::Tensor> scatter_forward(torch::Tensor src, torch::Tensor index, int64_t dim,
                                                         torch::Tensor out, int64_t reduce, bool hasOut,
                                                         int64_t hotTarget);

#ifdef OPSGNN_DAV_2201
uint32_t ScatterA2DtypeCode(at::ScalarType dtype, const char *errorMessage);
std::tuple<int64_t, int64_t> scatter_index_stats(torch::Tensor index);
std::tuple<torch::Tensor, torch::Tensor> scatter_a2_forward(torch::Tensor src, torch::Tensor index, int64_t dim,
                                                            std::optional<torch::Tensor> out,
                                                            std::optional<int64_t> dimSize, int64_t reduce);
#endif

} // namespace opsgnn

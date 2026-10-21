/*
 * Copyright (c) 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#ifdef OPSGNN_DAV_2201
#include "segment_coo/op_kernel/arch22/segment_coo_tiling.h"
#else
#include "segment_coo/op_kernel/arch35/segment_coo_tiling.h"
#endif
#include <torch/extension.h>
#include <optional>
#include <tuple>
std::tuple<torch::Tensor, torch::Tensor> segment_coo_forward(torch::Tensor src, torch::Tensor index,
                                                             std::optional<torch::Tensor> out,
                                                             std::optional<int64_t> dim_size, int64_t reduce,
                                                             bool return_arg);

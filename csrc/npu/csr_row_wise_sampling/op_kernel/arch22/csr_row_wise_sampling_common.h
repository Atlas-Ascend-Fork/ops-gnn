/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file csr_row_wise_sampling_common.h
 * @brief Shared weighted-sampling helpers for the arch22 CSR kernels.
 */

#pragma once

#include "kernel_operator.h"

namespace {

// Only positive finite weights are eligible. NaN and +inf fail the upper bound,
// -inf/zero/negative fail the lower bound, so none of them can be picked.
__aicore__ inline bool IsUsableWeight(float weight) {
  return weight > 0.0f && weight <= 3.402823466e+38F;
}

__aicore__ inline float SanitizeWeight(float weight) {
  return IsUsableWeight(weight) ? weight : 0.0f;
}

}  // namespace

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

constexpr uint32_t UNIFIED_SPMM_DTYPE_FP32 = 0;
constexpr uint32_t UNIFIED_SPMM_DTYPE_FP16 = 1;
constexpr uint32_t UNIFIED_SPMM_REDUCE_SUM = 0;
constexpr uint32_t UNIFIED_SPMM_REDUCE_MAX = 1;
constexpr uint32_t UNIFIED_SPMM_REDUCE_MIN = 2;
constexpr uint32_t UNIFIED_SPMM_COPY_LHS = 0;
constexpr uint32_t UNIFIED_SPMM_COPY_RHS = 1;

struct UnifiedSpmmTilingData {
    uint32_t numDstRows;
    uint32_t numFeatureRows;
    uint32_t featureDim;
    uint32_t nonZeroCount;
    uint32_t ubBytes;
    uint32_t dtype;
    uint32_t reduce;
    uint32_t message;
    uint32_t hasNan;
    uint32_t batchCount;
};

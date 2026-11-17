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

enum ScatterReduceCode : uint32_t {
    SCATTER_SUM = 0,
    SCATTER_MUL = 1,
    SCATTER_MEAN = 2,
    SCATTER_MIN = 3,
    SCATTER_MAX = 4,
};

enum ScatterKernelPath : uint32_t {
    SCATTER_LANE_OWNER = 0,
    SCATTER_ATOMIC = 1,
    SCATTER_VECTOR_ATOMIC = 2,
    SCATTER_F16_FLOAT_ATOMIC = 3,
    SCATTER_VECTOR_MINMAX = 4,
};

enum ScatterIndexMode : uint32_t {
    SCATTER_INDEX_FULL = 0,
    SCATTER_INDEX_DIM_VECTOR = 1,
};

enum ScatterDType : uint32_t {
    SCATTER_FLOAT16 = 0,
    SCATTER_BFLOAT16 = 1,
    SCATTER_FLOAT32 = 2,
    SCATTER_INT8 = 3,
    SCATTER_INT16 = 4,
    SCATTER_INT32 = 5,
    SCATTER_UINT8 = 6,
};

struct ScatterTilingData {
    uint64_t beforeDim;
    uint64_t dimLength;
    uint64_t afterDim;
    uint64_t outputDim;
    uint64_t srcNumel;
    uint64_t outNumel;
    uint64_t laneCount;
    uint64_t indexDimLength;
    uint32_t reduce;
    uint32_t path;
    uint32_t indexMode;
    uint32_t blockNum;
    uint32_t hasOut;
    int64_t hotTarget;
};

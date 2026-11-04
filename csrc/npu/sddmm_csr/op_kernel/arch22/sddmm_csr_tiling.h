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

constexpr uint32_t SDDMM_CSR_DTYPE_FP32 = 0;
constexpr uint32_t SDDMM_CSR_DTYPE_FP16 = 1;
constexpr uint32_t SDDMM_CSR_BUFFER_NUM = 2;
constexpr uint32_t SDDMM_CSR_MAX_REDUCE_MASK_FLOAT = 64;

constexpr uint32_t SDDMM_CSR_OP_DOT = 0;
constexpr uint32_t SDDMM_CSR_OP_ADD = 1;
constexpr uint32_t SDDMM_CSR_OP_SUB = 2;
constexpr uint32_t SDDMM_CSR_OP_MUL = 3;
constexpr uint32_t SDDMM_CSR_OP_DIV = 4;
constexpr uint32_t SDDMM_CSR_OP_COPY_LHS = 5;
constexpr uint32_t SDDMM_CSR_OP_COPY_RHS = 6;

constexpr uint32_t SDDMM_CSR_TARGET_SRC = 0;
constexpr uint32_t SDDMM_CSR_TARGET_EDGE = 1;
constexpr uint32_t SDDMM_CSR_TARGET_DST = 2;

struct SddmmCsrTilingData {
    uint32_t numEdges;
    uint32_t numRows;
    uint32_t featDim;
    uint32_t blockDim;
    uint32_t edgesPerCore;
    uint32_t batchSize;
    uint32_t dtype;
    uint32_t op;
    uint32_t lhsTarget;
    uint32_t rhsTarget;
    uint32_t featDimAligned;
    uint32_t featDimAlignedF32;
    uint32_t ubBytes;
};

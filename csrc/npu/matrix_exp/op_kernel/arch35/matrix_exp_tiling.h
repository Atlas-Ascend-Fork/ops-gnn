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

#include <stdint.h>

namespace opsgnn {

// matrix_exp 迭代一支持的最大矩阵维度：矩阵可全驻留 UB/L0（设计文档 3.2）。
constexpr uint32_t MATRIX_EXP_N_MAX_UB = 128;

// matrix_exp kernel 的 Tiling 参数（host 侧 ComputeTiling 填充）。
// 注意：本结构体同时被 op_host 与 device kernel 引用，字段布局不可随意变更。
struct MatrixExpTilingData {
    uint64_t batchSize;        // 批量矩阵总数（batch 展平后的矩阵个数）
    uint32_t n;                // 矩阵维度 n（n <= MATRIX_EXP_N_MAX_UB，迭代一约束）
    uint32_t coreNum;          // 参与计算的 block 数（MIX 下等于 AIC 核数）
    uint32_t matricesPerCore;  // 每核基础矩阵数（floor 切分）
    uint32_t coresWithExtra;   // 前 coresWithExtra 个核多处理 1 个矩阵（余数均摊）
    uint32_t dtSize;           // 输入/输出 dtype 位宽：4=fp32，2=fp16/bf16
    uint32_t nSmall;           // 预留字段（迭代一未使用，恒 0）
};

}  // namespace opsgnn

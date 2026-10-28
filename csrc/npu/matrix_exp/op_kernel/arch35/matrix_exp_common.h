/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// matrix_exp kernel 公共定义：include、常量与事件 id（被 aic/aiv/kernel 头共享）。
#pragma once

#include "kernel_operator.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tensor_api/tensor.h"
#include "matrix_exp_tiling.h"
#include "matrix_exp.h"

using namespace AscendC;

namespace opsgnn {

// 缩放平方 + Taylor-Horner 算法参数（设计文档 3.1）。
constexpr float MATRIX_EXP_THETA = 0.5f;            // 保证 ||sigma*A||_F <= theta
constexpr uint32_t MATRIX_EXP_S_MAX = 30;           // 平方次数上限（防御 inf/nan）
constexpr uint32_t MATRIX_EXP_HORNER_ROUNDS = 8;    // K=9 阶，E 初始化免 1 次 Matmul

// CrossCore flag id（MODE4，intra-block；AIC 只与 subBlock0 AIV 交互）。
constexpr uint8_t MATRIX_EXP_AIV_READY_FLAG = 0;    // slot 0/1 轮换
constexpr uint8_t MATRIX_EXP_AIC_DONE_FLAG = 2;     // slot 2/3 轮换

// AIC 核内事件 id（不同 HardEvent 共享物理寄存器，类型间必须独立 id）。
constexpr uint8_t EVT_AIC_MTE2_MTE1 = 0;
constexpr uint8_t EVT_AIC_MTE1_M = 1;
constexpr uint8_t EVT_AIC_M_FIX = 2;
constexpr uint8_t EVT_AIC_FIX_MTE2 = 3;
constexpr uint8_t EVT_AIC_M_MTE1 = 4;
constexpr uint8_t EVT_AIC_FIX_M = 5;
constexpr uint8_t EVT_AIC_MTE1_MTE2 = 6;
constexpr uint8_t EVT_AIC_MTE2_S = 7;
// AIV 核内事件 id。
constexpr uint8_t EVT_AIV_MTE2_V = 0;
constexpr uint8_t EVT_AIV_V_MTE3 = 1;
constexpr uint8_t EVT_AIV_V_S = 2;
constexpr uint8_t EVT_AIV_V_MTE2 = 3;  // V 清零 -> MTE2 搬入同 buffer 排序
constexpr uint8_t EVT_AIV_S_MTE2 = 4;  // S 标量读 -> MTE2 重写同 buffer 排序
constexpr uint8_t EVT_AIV_MTE2_MTE3 = 5;  // MTE2 搬入 -> MTE3 读出同 buffer 排序
constexpr uint8_t EVT_AIV_MTE3_V = 6;     // MTE3 读出 -> V 改写同 buffer 排序

// GM workspace 布局见 matrix_exp.h（MATRIX_EXP_WS_META_FLOATS）。
constexpr uint32_t MATRIX_EXP_SMALL_CHUNK = 8192;   // n==1 路径每块元素数

}  // namespace opsgnn

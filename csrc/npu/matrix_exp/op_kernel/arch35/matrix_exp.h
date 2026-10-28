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
#include "matrix_exp_tiling.h"

namespace opsgnn {

// GM workspace 每核布局（fp32 元素）：[A 区 nPad^2 | E 区 nPad^2 | tmp 区 nPad^2 | meta 区 8]，
// 其中 nPad = ceil(n/16)*16。A/E/tmp 三区均为 nPad x nPad（行距 nPad，尾行/行尾零填充）：
// AIC 侧 CopyGM2L1(ND->NZ) 的 d_value 与 fixpipe L0C->GM 的行宽均要求 >= 32B（fp32 即
// >= 8 元素），紧凑 n 布局在 n < 8 时触发 device error 507015；AIC 无 V pipe 无法自行
// 重排用户 GM 紧凑张量，故 A 经 AIV 中转至对齐 A 区（数学上等价 nPad 阶零填充矩阵）。
// meta 槽起始按 32B 对齐（plain DataCopy 写 GM 要求 32B 对齐地址；AIC 标量读同槽，
// 对齐消除潜在访存约束）。nPad^2 为 256 的倍数，各区起始地址自然 32B 对齐。
// 布局计算用宏实现：该头文件同时被 host（op_host）与 device（bisheng ASC）编译，
// 函数会带 __host__/__aicore__ 归属导致其中一侧不可调用，宏无此问题。
constexpr uint32_t MATRIX_EXP_WS_META_FLOATS = 8;    // 8 fp32 = 32B
constexpr uint32_t MATRIX_EXP_WS_ALIGN_FLOATS = 8;   // 8 fp32 = 32B 对齐粒度

// nPad：矩阵阶数向上取整到 16（L0 分形边长）。
#define MATRIX_EXP_WS_NPAD(n) ((((n) + 15ULL) / 16ULL) * 16ULL)

// 每核 workspace 步长（fp32 元素）：3*nPad^2 + meta 槽（nPad^2 已 32B 对齐）。
#define MATRIX_EXP_WS_STRIDE_FLOATS(n)                                        \
    (3ULL * MATRIX_EXP_WS_NPAD(n) * MATRIX_EXP_WS_NPAD(n) +                   \
     MATRIX_EXP_WS_META_FLOATS)

// A/E/tmp/meta 区相对每核 workspace 起始的偏移（fp32 元素）。
#define MATRIX_EXP_WS_A_OFF_FLOATS(n) 0ULL
#define MATRIX_EXP_WS_E_OFF_FLOATS(n) \
    (MATRIX_EXP_WS_NPAD(n) * MATRIX_EXP_WS_NPAD(n))
#define MATRIX_EXP_WS_TMP_OFF_FLOATS(n) \
    (2ULL * MATRIX_EXP_WS_NPAD(n) * MATRIX_EXP_WS_NPAD(n))
#define MATRIX_EXP_WS_META_OFF_FLOATS(n) \
    (MATRIX_EXP_WS_STRIDE_FLOATS(n) - MATRIX_EXP_WS_META_FLOATS)

// matrix_exp kernel 启动入口（MIX：1 AIC + 2 AIV/block，设计文档 3.4）。
// 参数按 dtypeTag 解释 a/out 的位模式：0=float, 1=half, 2=bfloat16。
// ws：fp32 workspace（每核 [A nPad^2 | E nPad^2 | tmp nPad^2 | meta 8]，host 侧按
// coreNum 分配，布局契约见上方宏）；eye：nPad16 方阵单位矩阵（fp32，全核共享只读）。
void MatrixExp(const uint8_t* a, uint8_t* out, const uint8_t* ws, const uint8_t* eye,
               const MatrixExpTilingData& tiling, uint32_t dtypeTag, aclrtStream stream);

}  // namespace opsgnn

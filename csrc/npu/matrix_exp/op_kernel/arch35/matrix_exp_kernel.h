/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// matrix_exp device kernel（950/DAV_3510，MIX 架构）。
//
// 执行模型（设计文档 3.4，按 dav-3510 AIC 无 V pipe 事实修订）：
//   KERNEL_TYPE_MIX_AIC_1_2：每 block 含 1 AIC + 2 AIV；仅 subBlock 0 的 AIV 参与
//   计算与 flag 交互，subBlock 1 直接空转退出。
//   - AIV（matrix_exp_aiv.h）：GM->UB 搬入、Cast fp32、Frobenius 范数、定 s/sigma、
//     E 初始化与 Horner 标量乘加、n==1 Exp 快速路径、输出 Cast + 搬出。
//   - AIC（matrix_exp_aic.h）：Horner 8 次 Mmad(A@E) + s 次 Mmad(E@E)，全 fp32。
//   - 数据交接：fp32 全链路（UB->L1 直通仅支持 fp16/bf16），E 每轮更新由 AIV 写
//     GM workspace，AIC 以 CopyGM2L1 做 ND->NZ/ZN 重排后供 Mmad；Mmad 结果经
//     fixpipe 写回 GM workspace tmp 区。Cube 链使用 CANN SDK tensor_api
//     （AscendC::Te，CamelCase；Mmad/Fixpipe unitFlag 均为 3=FINAL_ACCUMULATION）。
//     ws 的 A/E/tmp 三区均为 nPad(=ceil(n/16)*16) 方阵布局：AIC 侧 ND2NZ 的
//     d_value 与 fixpipe 行宽要求 >= 32B（fp32 即 8 元素），A 经 AIV 中转至对齐
//     A 区（AIC 无 V pipe，无法自行重排用户 GM 紧凑张量）；Mmad 维度取 nPad
//     （零填充不影响 n x n 有效块结果）。
//   - 同步：CrossCore MODE4 flag（AIV_READY/AIC_DONE 各 2 slot 轮换）严格锁步配对；
//     核内 Set/Wait 完成对，事件类型各占独立 eventID。
#pragma once

#include "matrix_exp_aic.h"
#include "matrix_exp_aiv.h"

namespace opsgnn {

// ===========================================================================
// kernel 入口：MIX（1 AIC + 2 AIV，仅 subBlock 0 的 AIV 参与计算）
// ===========================================================================
template <typename T>
__global__ __aicore__ void matrix_exp_kernel(GM_ADDR a, GM_ADDR out, GM_ADDR ws, GM_ADDR eye,
                                             const MatrixExpTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    if ASCEND_IS_NOT_AIV {
        MatrixExpTilingData td = tiling;
        MatrixExpAic aic;
        aic.Init(ws, td);
        aic.Process();
    } else {
        if (GetSubBlockIdx() != 0) {
            return;  // subBlock 1 空转：不参与任何 flag 交互
        }
        TPipe pipe;
        MatrixExpTilingData td = tiling;
        MatrixExpAiv<T> aiv;
        aiv.Init(a, out, ws, eye, td, &pipe);
        aiv.Process();
    }
}

}  // namespace opsgnn

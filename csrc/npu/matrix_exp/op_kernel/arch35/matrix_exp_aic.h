/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AIC 侧：Cube/Mmad 链（全 fp32，tensor/layout 函数内构造）。
#pragma once

#include "matrix_exp_common.h"

namespace opsgnn {

// ===========================================================================
// AIC 侧：Cube/Mmad 链（全 fp32，tensor/layout 函数内构造）
// ===========================================================================
class MatrixExpAic {
public:
    __aicore__ inline MatrixExpAic() { SetMMLayoutTransform(true); }
    __aicore__ inline ~MatrixExpAic() { SetMMLayoutTransform(false); }

    __aicore__ inline void Init(GM_ADDR ws, const MatrixExpTilingData& tiling)
    {
        n_ = tiling.n;
        const uint32_t blockIdx = GetBlockIdx();
        matrixBeg_ = blockIdx * tiling.matricesPerCore +
                     (blockIdx < tiling.coresWithExtra ? blockIdx : tiling.coresWithExtra);
        matrixEnd_ = matrixBeg_ + tiling.matricesPerCore + (blockIdx < tiling.coresWithExtra ? 1 : 0);
        wsPtr_ = reinterpret_cast<__gm__ uint8_t*>(ws);
        nPad_ = ((n_ + 15U) / 16U) * 16U;
        l1EBOff_ = nPad_ * nPad_ * sizeof(float);   // L1: A NZ 区在 0，E B 侧 NZ 区随后
        wsStride_ = MATRIX_EXP_WS_STRIDE_FLOATS(n_) * sizeof(float);
        wsEOff_ = MATRIX_EXP_WS_E_OFF_FLOATS(n_) * sizeof(float);
        wsTmpOff_ = MATRIX_EXP_WS_TMP_OFF_FLOATS(n_) * sizeof(float);
        wsMetaOff_ = MATRIX_EXP_WS_META_OFF_FLOATS(n_) * sizeof(float);
        // meta 槽 DCache 失效句柄（ProcessOneMatrix 每矩阵读 s 前刷新，见该处注释）。
        metaGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(wsPtr_ + blockIdx * wsStride_ + wsMetaOff_),
            MATRIX_EXP_WS_META_FLOATS);

        // 单缓冲反向依赖预发（首轮消费，之后逐轮交替闭合，矩阵间串行续接）。
        SetFlag<HardEvent::M_MTE1>(EVT_AIC_M_MTE1);
        SetFlag<HardEvent::FIX_M>(EVT_AIC_FIX_M);
        SetFlag<HardEvent::MTE1_MTE2>(EVT_AIC_MTE1_MTE2);
    }

    __aicore__ inline void Process()
    {
        // n==1 或本核无矩阵：直接退出（配对 AIV 同样不做 flag 交互，无死锁）。
        if (n_ <= 1 || matrixEnd_ <= matrixBeg_) {
            return;
        }
        for (uint64_t b = matrixBeg_; b < matrixEnd_; ++b) {
            ProcessOneMatrix();
        }
    }

private:
    __aicore__ inline void SyncMte2ToMte1()
    {
        SetFlag<HardEvent::MTE2_MTE1>(EVT_AIC_MTE2_MTE1);
        WaitFlag<HardEvent::MTE2_MTE1>(EVT_AIC_MTE2_MTE1);
    }
    __aicore__ inline void SyncMte1ToM()
    {
        SetFlag<HardEvent::MTE1_M>(EVT_AIC_MTE1_M);
        WaitFlag<HardEvent::MTE1_M>(EVT_AIC_MTE1_M);
    }
    __aicore__ inline void SyncMte1ToMte2()
    {
        SetFlag<HardEvent::MTE1_MTE2>(EVT_AIC_MTE1_MTE2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_AIC_MTE1_MTE2);
    }
    __aicore__ inline void SyncMToFix()
    {
        SetFlag<HardEvent::M_FIX>(EVT_AIC_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVT_AIC_M_FIX);
    }
    __aicore__ inline void WaitFixMte2() { WaitFlag<HardEvent::FIX_MTE2>(EVT_AIC_FIX_MTE2); }
    __aicore__ inline void SetFixMte2() { SetFlag<HardEvent::FIX_MTE2>(EVT_AIC_FIX_MTE2); }
    __aicore__ inline void WaitMUsedL0() { WaitFlag<HardEvent::M_MTE1>(EVT_AIC_M_MTE1); }
    __aicore__ inline void SetMUsedL0() { SetFlag<HardEvent::M_MTE1>(EVT_AIC_M_MTE1); }
    __aicore__ inline void WaitFixUsedL0c() { WaitFlag<HardEvent::FIX_M>(EVT_AIC_FIX_M); }
    __aicore__ inline void SetFixUsedL0c() { SetFlag<HardEvent::FIX_M>(EVT_AIC_FIX_M); }

    // S pipe 等待 MTE2 通过其队列中已排入的 wait_intra_block（即 READY 已到达）。
    // wait_intra_block 只阻塞所在 pipe（MTE2）的指令队列，S pipe 的标量代码会超前
    // 执行；读 AIV 经 flag 同步才可见的 GM 数据前必须用 MTE2_S 事件显式排序。
    __aicore__ inline void SyncMte2ToS()
    {
        SetFlag<HardEvent::MTE2_S>(EVT_AIC_MTE2_S);
        WaitFlag<HardEvent::MTE2_S>(EVT_AIC_MTE2_S);
    }

    __aicore__ inline void WaitAivReady(uint32_t& cnt)
    {
        CrossCoreWaitFlag<0x4, PIPE_MTE2>(MATRIX_EXP_AIV_READY_FLAG + (cnt & 1U));
        ++cnt;
    }
    __aicore__ inline void SetAicDone(uint32_t& cnt)
    {
        CrossCoreSetFlag<0x4, PIPE_FIX>(MATRIX_EXP_AIC_DONE_FLAG + (cnt & 1U));
        ++cnt;
    }

    // GM nPad 方阵区（A/E/tmp 均按 nPad 布局，见 matrix_exp.h 布局契约）。
    __aicore__ inline auto MakeGmTensor(__gm__ uint8_t* src)
    {
        return Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ float*>(src)),
                              Te::MakeFrameLayout<Te::NDExtLayoutPtn>(nPad_, nPad_));
    }
    // GM(nPad 方阵) -> L1 NZ(l1Off) -> L0A(NZ)。
    __aicore__ inline void LoadL1AndL0a(__gm__ uint8_t* src)
    {
        auto tGm = MakeGmTensor(src);
        auto tL1 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, float>(0),
                                  Te::MakeFrameLayout<Te::NZLayoutPtn, float>(nPad_, nPad_));
        auto tL0 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0A, float>(0),
                                   Te::MakeFrameLayout<Te::NZLayoutPtn, float>(nPad_, nPad_));
        Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tL1, tGm);
        SyncMte2ToMte1();
        SyncMte1ToMte2();
        Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}), tL0, tL1);
        SyncMte1ToM();
    }
    // GM(nPad 方阵) -> L1 NZ(l1EBOff_) -> L0B(ZN)。B 侧 L1 沿用非转置 NZ 模式，
    // L1(NZ)->L0B(ZN) 由 CopyL12L0B 按 pattern 差异自动选 TRANS 搬运（官方样例同法）。
    __aicore__ inline void LoadL1AndL0b(__gm__ uint8_t* src)
    {
        auto tGm = MakeGmTensor(src);
        auto tL1 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, float>(l1EBOff_),
                                  Te::MakeFrameLayout<Te::NZLayoutPtn, float>(nPad_, nPad_));
        auto tL0 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0B, float>(0),
                                   Te::MakeFrameLayout<Te::ZNLayoutPtn, float>(nPad_, nPad_));
        Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}), tL1, tGm);
        SyncMte2ToMte1();
        SyncMte1ToMte2();
        Te::Copy(Te::MakeCopy(Te::CopyL12L0B{}), tL0, tL1);
        SyncMte1ToM();
    }
    // L0C -> GM(nPad 方阵)。前置：SyncMToFix() 已保证 M 完成。
    __aicore__ inline void StoreL0cToGm(__gm__ uint8_t* dst)
    {
        // L0C 必须显式 BLOCK_CUBE=16（fp32 下 LayoutTraitDefault 会算成 C0=8，skill 陷阱表）。
        auto tL0c = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0C, float>(0),
                                   Te::MakeFrameLayout<Te::NZLayoutPtn, Std::Int<16>>(nPad_, nPad_));
        auto tGm = MakeGmTensor(dst);
        // unitFlag=3（FINAL_ACCUMULATION）：单发 Mmad + 单次写回，Mmad/Fixpipe 须同为 3。
        auto copyOp = Te::MakeCopy(Te::CopyL0C2GM{}).with(Te::FixpipeParams{3});
        Te::Copy(copyOp, tGm, tL0c);
        SetFixUsedL0c();
    }
    // L0C = L0A @ L0B（cmatrixInitVal=true：单缓冲每轮重置累加器）。维度取 nPad：
    // 零填充分形不改变 n x n 有效块结果，且规避小维度/行宽 < 32B 的搬运约束。
    __aicore__ inline void Mmad()
    {
        auto tL0a = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0A, float>(0),
                                   Te::MakeFrameLayout<Te::NZLayoutPtn, float>(nPad_, nPad_));
        auto tL0b = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0B, float>(0),
                                   Te::MakeFrameLayout<Te::ZNLayoutPtn, float>(nPad_, nPad_));
        auto tL0c = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0C, float>(0),
                                   Te::MakeFrameLayout<Te::NZLayoutPtn, Std::Int<16>>(nPad_, nPad_));
        auto atom = Te::MakeMmad(Te::MmadOperation{});
        const uint16_t nn = static_cast<uint16_t>(nPad_);
        Te::Mmad(atom.with(Te::MmadParams{nn, nn, nn, 3, true}), tL0c, tL0a, tL0b);
        SetMUsedL0();
    }

    __aicore__ inline void ProcessOneMatrix()
    {
        __gm__ uint8_t* wsBase = wsPtr_ + GetBlockIdx() * wsStride_;
        __gm__ uint8_t* aGm = wsBase;                        // A 区（AIV 中转，nPad 布局）
        __gm__ uint8_t* eGm = wsBase + wsEOff_;              // E 区
        __gm__ uint8_t* tmpGm = wsBase + wsTmpOff_;          // tmp 区

        // 1) 等 AIV 的 A 中转 + E0（含 meta 槽 s）就绪；S pipe 标量读必须排在 READY 之后。
        //    标量单元读 GM 命中核内 DCache，与他核（AIV）MTE3 写入不保持一致：每核首读
        //    miss 取到新值后，后续矩阵永远 hit 陈旧行、沿用核首矩阵的 s，squaring 次数
        //    系统性错位（实测 out = E8(sigma_true)^(2^s_coreFirst)）。读前须
        //    DataCacheCleanAndInvalid 失效 meta 行强制重取 GM；volatile 仅防编译器
        //    CSE、不解决硬件 DCache（实测单独无效），保留作双保险。
        WaitAivReady(readyCnt_);
        SyncMte2ToS();
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            metaGm_);
        const uint32_t s = static_cast<uint32_t>(
            *(reinterpret_cast<volatile __gm__ float*>(wsBase + wsMetaOff_)));

        // 2) A 常驻 L0A（自 ws 对齐 A 区读取，READY 已保证 AIV 写入可见）。
        //    WaitMUsedL0（M_MTE1，Wait 归属 MTE1 pipe）阻塞其后入队的 CopyL12L0A/B：
        //    必须置于装载之前才能防住上一矩阵尾轮 Mmad 对 L0A 的读竞争（首矩阵由
        //    Init 预发的 M_MTE1 flag 满足；后续矩阵消费上一尾轮 Mmad 的残留 Set）。
        WaitMUsedL0();
        LoadL1AndL0a(aGm);

        // 3) Horner 8 轮：tmp = A @ E_j；AIV 侧 E_{j+1} = E_j + (sigma/k_j) * tmp。
        for (uint32_t j = 0; j < MATRIX_EXP_HORNER_ROUNDS; ++j) {
            if (j > 0) {
                WaitMUsedL0();  // 等上一轮 Mmad 读完 L0B（j=0 时 L0A 装载与 L0B 不冲突）
            }
            LoadL1AndL0b(eGm);
            WaitFixUsedL0c();   // 上一轮 FIX 已读完 L0C
            Mmad();
            SyncMToFix();       // M_FIX 事件须在 Mmad 之后入 M 队列（早于 Mmad 会导致 FIX 抢读 L0C）
            StoreL0cToGm(tmpGm);
            SetAicDone(doneCnt_);
            if ((j + 1 < MATRIX_EXP_HORNER_ROUNDS) || (s > 0)) {
                WaitAivReady(readyCnt_);  // 等 AIV 更新出的下一轮 E
            }
        }
        if (s == 0) {
            return;  // AIV 持有最终 E 直接输出，尾轮不发 READY、AIC 不再等待
        }

        // 4) 平方 s 轮：E <- E @ E（纯 AIC）。i==0 读 E 区，之后读 tmp 区。
        __gm__ uint8_t* src = eGm;
        for (uint32_t i = 0; i < s; ++i) {
            if (i > 0) {
                WaitFixMte2();  // 上一轮 FIX 写完 GM tmp，MTE2 才可读
                src = tmpGm;
            }
            // 同上：L0A/B 重装须置于装载之前——Horner 尾轮 Mmad 的 M_MTE1 Set 由
            // 此处消费，否则 CopyL12L0A 与其竞争 L0A（实测大范数场景零星元素错）。
            WaitMUsedL0();
            LoadL1AndL0a(src);  // E 的 A 侧，覆盖已无用的 A
            LoadL1AndL0b(src);
            WaitFixUsedL0c();
            Mmad();
            SyncMToFix();       // M_FIX 事件须在 Mmad 之后入 M 队列
            StoreL0cToGm(tmpGm);
            if (i + 1 < s) {
                SetFixMte2();
            }
        }
        // 5) 通知 AIV 取最终结果。
        SetAicDone(doneCnt_);
    }

    uint32_t n_;
    uint32_t nPad_;
    uint64_t matrixBeg_;
    uint64_t matrixEnd_;
    __gm__ uint8_t* wsPtr_;
    uint64_t wsStride_;
    uint64_t wsEOff_;
    uint64_t wsTmpOff_;
    uint64_t wsMetaOff_;
    uint64_t l1EBOff_;
    GlobalTensor<float> metaGm_;
    uint32_t readyCnt_ = 0;
    uint32_t doneCnt_ = 0;
};

}  // namespace opsgnn

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AIV 侧：Vector 路径（范数/Horner 标量/Cast/Exp），仅 subBlock 0。
#pragma once

#include "matrix_exp_common.h"

namespace opsgnn {

// ===========================================================================
// AIV 侧：Vector 路径（范数/Horner 标量/Cast/Exp），仅 subBlock 0
// ===========================================================================
template <typename T>
class MatrixExpAiv {
public:
    __aicore__ inline MatrixExpAiv() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR out, GM_ADDR ws, GM_ADDR eye,
                                const MatrixExpTilingData& tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        n_ = tiling.n;
        // 3510 MIX：AIV 的 GetBlockIdx() 返回 rawIdx = blockIdx*subblockdim + subblockid，
        // 须除以 GetTaskRation() 换算逻辑 block（SDK 生产 kernel 同法），否则 vec0 会
        // 处理错位 block 的矩阵、奇数 block 无人处理（多核输出缺失/AIC-AIV 错位挂死）。
        int64_t taskRation = GetTaskRation();
        if (taskRation < 1) {
            taskRation = 1;  // 防御兜底：taskRation 恒 > 0，避免除零
        }
        const uint32_t blockIdx =
            static_cast<uint32_t>(static_cast<int64_t>(GetBlockIdx()) / taskRation);
        matrixBeg_ = blockIdx * tiling.matricesPerCore +
                     (blockIdx < tiling.coresWithExtra ? blockIdx : tiling.coresWithExtra);
        matrixEnd_ = matrixBeg_ + tiling.matricesPerCore + (blockIdx < tiling.coresWithExtra ? 1 : 0);
        nPad_ = ((n_ + 15U) / 16U) * 16U;
        bufElems_ = (nPad_ * nPad_ > MATRIX_EXP_SMALL_CHUNK) ? nPad_ * nPad_
                                                             : MATRIX_EXP_SMALL_CHUNK;

        aPtr_ = reinterpret_cast<__gm__ uint8_t*>(a);
        outPtr_ = reinterpret_cast<__gm__ uint8_t*>(out);
        __gm__ uint8_t* wsPtr = reinterpret_cast<__gm__ uint8_t*>(ws);
        const uint64_t wsStrideFloats = MATRIX_EXP_WS_STRIDE_FLOATS(n_);
        __gm__ float* wsBase = reinterpret_cast<__gm__ float*>(wsPtr + blockIdx * wsStrideFloats * 4);
        const uint64_t nPad2 = static_cast<uint64_t>(nPad_) * nPad_;
        wsAGm_.SetGlobalBuffer(wsBase + MATRIX_EXP_WS_A_OFF_FLOATS(n_), nPad2);
        wsEGm_.SetGlobalBuffer(wsBase + MATRIX_EXP_WS_E_OFF_FLOATS(n_), nPad2);
        wsTmpGm_.SetGlobalBuffer(wsBase + MATRIX_EXP_WS_TMP_OFF_FLOATS(n_), nPad2);
        metaGm_.SetGlobalBuffer(wsBase + MATRIX_EXP_WS_META_OFF_FLOATS(n_), MATRIX_EXP_WS_META_FLOATS);
        eyeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eye), nPad_ * nPad_);

        pipe_->InitBuffer(bufA_, bufElems_ * sizeof(float));
        pipe_->InitBuffer(bufE_, bufElems_ * sizeof(float));
        pipe_->InitBuffer(bufTmp_, bufElems_ * sizeof(float));
        pipe_->InitBuffer(bufMeta_, MATRIX_EXP_WS_META_FLOATS * sizeof(float));
        if constexpr (sizeof(T) != sizeof(float)) {
            pipe_->InitBuffer(bufIo_, bufElems_ * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (matrixEnd_ <= matrixBeg_) {
            return;  // 空闲核：不发不收 flag（配对 AIC 同样空转）
        }
        if (n_ == 1) {
            ProcessSmall();
            return;
        }
        for (uint64_t b = matrixBeg_; b < matrixEnd_; ++b) {
            ProcessOneMatrix(b);
        }
    }

private:
    __aicore__ inline void SyncMte2V()
    {
        SetFlag<HardEvent::MTE2_V>(EVT_AIV_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_AIV_MTE2_V);
    }
    __aicore__ inline void SyncVMte3()
    {
        SetFlag<HardEvent::V_MTE3>(EVT_AIV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_AIV_V_MTE3);
    }
    __aicore__ inline void SyncVS()
    {
        SetFlag<HardEvent::V_S>(EVT_AIV_V_S);
        WaitFlag<HardEvent::V_S>(EVT_AIV_V_S);
    }
    __aicore__ inline void SyncVMte2()
    {
        SetFlag<HardEvent::V_MTE2>(EVT_AIV_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(EVT_AIV_V_MTE2);
    }
    __aicore__ inline void SyncMte2Mte3()
    {
        SetFlag<HardEvent::MTE2_MTE3>(EVT_AIV_MTE2_MTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(EVT_AIV_MTE2_MTE3);
    }
    __aicore__ inline void SyncMte3V()
    {
        SetFlag<HardEvent::MTE3_V>(EVT_AIV_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_AIV_MTE3_V);
    }
    __aicore__ inline void WaitAicDone(uint32_t& cnt)
    {
        CrossCoreWaitFlag<0x4, PIPE_MTE2>(MATRIX_EXP_AIC_DONE_FLAG + (cnt & 1U));
        ++cnt;
    }
    __aicore__ inline void SetAivReady(uint32_t& cnt)
    {
        CrossCoreSetFlag<0x4, PIPE_MTE3>(MATRIX_EXP_AIV_READY_FLAG + (cnt & 1U));
        ++cnt;
    }

    // 矩阵行级搬运：GM(n 行紧凑) -> UB(行距 nPad_)。行尾 dummy 由 paddingValue=0 填充
    //（isPad=true 且 left/rightPadding=0 时非对齐尾巴取 paddingValue）。3510 的
    // DataCopyPad rightPadding 实测仅支持 <= 1 个对齐块（32B/dtSize 个元素）的尾部
    // 填充，nPad-n 跨块填充直接触发 device error 507015（n=2..7、n=17 均崩，n=8
    // 恰 1 块、n=12/15 尾块内正常），故 padding 置 0，UB 行距改由 dstStride（UB 侧
    // 单位 32B 块）精确控制：每行占 alignUp(n*dtSize,32) + dstStride*32 == nPad*dtSize。
    __aicore__ inline void CopyGmToUb(LocalTensor<T>& ub, const GlobalTensor<T>& gm)
    {
        const uint32_t rowBytes = n_ * sizeof(T);
        const uint32_t rowAligned = (rowBytes + 31U) / 32U * 32U;
        const int64_t dstStride = static_cast<int64_t>((nPad_ * sizeof(T) - rowAligned) / 32U);
        DataCopyPad(ub, gm, DataCopyExtParams{n_, rowBytes, 0, dstStride, 0},
                    DataCopyPadExtParams<T>{true, 0, 0, 0});
    }
    // UB(行距 nPad_) -> GM(n 行紧凑)。3510 的 UB->GM DataCopyPad 3 参版 srcStride 单位
    // 为 32B 块：实测 srcStride!=0 且 blockLen 非 32B 对齐时直接触发 device error 507015
    // （n=2..7、n=17 崩；n=8 恰 32B 对齐、n=12/15 srcStride 截断为 0 均正常）。改为逐行
    // 搬出：每行 blockCount=1、无 stride，GM 行地址按字节精确步进（n <= 128 行数小）。
    __aicore__ inline void CopyUbToGm(const GlobalTensor<T>& gm, const LocalTensor<T>& ub)
    {
        for (uint32_t r = 0; r < n_; ++r) {
            DataCopyPad(gm[static_cast<uint64_t>(r) * n_], ub[static_cast<uint64_t>(r) * nPad_],
                        DataCopyExtParams{1, n_ * sizeof(T), 0, 0, 0});
        }
    }
    // UB fp32 计算缓冲(nPad 方阵) -> GM fp32 workspace(nPad 方阵)。全程 32B 对齐
    // （区域起始 nPad^2*4 为 1024 倍数、长度 nPad^2*4 为 32 倍数），用 plain DataCopy。
    __aicore__ inline void CopyUbToWs(const GlobalTensor<float>& gm, const LocalTensor<float>& ub)
    {
        DataCopy(gm, ub, static_cast<uint64_t>(nPad_) * nPad_);
    }
    // GM fp32 workspace(nPad 方阵) -> UB fp32 计算缓冲(nPad 方阵)。
    __aicore__ inline void CopyWsToUb(LocalTensor<float>& ub, const GlobalTensor<float>& gm)
    {
        DataCopy(ub, gm, static_cast<uint64_t>(nPad_) * nPad_);
    }

    __aicore__ inline void ProcessOneMatrix(uint64_t b)
    {
        auto aF32 = bufA_.Get<float>();
        auto eF32 = bufE_.Get<float>();
        auto tmpF32 = bufTmp_.Get<float>();
        auto metaF32 = bufMeta_.Get<float>();

        const uint64_t matFloats = n_ * n_;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(aPtr_ + b * matFloats * sizeof(T)),
                             matFloats);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outPtr_ + b * matFloats * sizeof(T)),
                              matFloats);

        StageInputToWs(aF32);
        uint32_t s = 0;
        float sigma = 1.0f;
        ComputeScaling(aF32, tmpF32, metaF32, s, sigma);
        InitE0(aF32, eF32, metaF32, sigma);
        RunHornerRounds(aF32, eF32, tmpF32, s, sigma);
        WriteOutput(eF32, tmpF32, s);
    }

    // 阶段 1：A -> UB(fp32) 并中转 ws A 区。fp16/bf16 经 ioBuf Cast；尾行/行尾补零
    // 保证范数与 Horner 正确。Duplicate 清零（V）与 DataCopyPad 搬入（MTE2）写同一
    // buffer，必须 V->MTE2 排序。随后将 raw A（nPad 方阵零填充）中转至 ws A 区：
    // AIC 侧 ND2NZ 的 d_value 与 fixpipe 行宽要求 >= 32B，紧凑 n 布局 n < 8 时非法
    //（AIC 无 V pipe 无法自行重排）。
    __aicore__ inline void StageInputToWs(LocalTensor<float>& aF32)
    {
        const uint64_t nPad2 = static_cast<uint64_t>(nPad_) * nPad_;
        if constexpr (sizeof(T) == sizeof(float)) {
            Duplicate(aF32, 0.0f, bufElems_);
            SyncVMte2();
            CopyGmToUb(aF32, aGm_);
            SyncMte2V();
            SyncMte2Mte3();  // MTE2 写完 aF32，MTE3 才可读出（MTE2/MTE3 并行）
            DataCopy(wsAGm_, aF32, nPad2);
            SyncMte3V();     // MTE3 读走 aF32，V 才可改写（E0 段 Muls 原地缩放）
        } else {
            auto ioT = bufIo_.Get<T>();
            // 上一矩阵 WriteOutput 经 ioT 搬出（MTE3 读）完成后才可清零复用，须 MTE3->V
            SyncMte3V();
            Duplicate(ioT, static_cast<T>(0), bufElems_);
            SyncVMte2();
            CopyGmToUb(ioT, aGm_);
            SyncMte2V();
            Cast(aF32, ioT, RoundMode::CAST_NONE, bufElems_);
            SyncVMte3();     // V 写完 aF32，MTE3 才可读出
            DataCopy(wsAGm_, aF32, nPad2);
            SyncMte3V();     // MTE3 读走 aF32，V 才可改写（E0 段 Muls 原地缩放）
        }
    }

    // 阶段 2：Frobenius 范数 -> s、sigma（标量，AIC 与本侧由相同输入决定，确定性一致）。
    // 3510 无 (dst, src, ws, count) 形式的 WholeReduceSum（vec 形式已废弃且依赖保留
    // UB 编译选项），改用树形归约：VEC 段按 8 元素(32B)对齐折半（src2 偏移必须
    // 32B 对齐，错误码 340），尾段 <=16 元素转标量累加。
    __aicore__ inline void ComputeScaling(const LocalTensor<float>& aF32, LocalTensor<float>& tmpF32,
                                          LocalTensor<float>& metaF32, uint32_t& s, float& sigma)
    {
        Mul(tmpF32, aF32, aF32, bufElems_);
        uint32_t reduceLen = bufElems_;
        while (reduceLen > 16) {
            const uint32_t half = reduceLen / 16U * 8U;  // 8 元素对齐
            Add(tmpF32, tmpF32, tmpF32[half], reduceLen - half);
            reduceLen -= half;
        }
        SyncVS();
        float normSq = 0.0f;
        for (uint32_t i = 0; i < reduceLen; ++i) {
            normSq += tmpF32.GetValue(i);
        }
        s = 0;
        float bound = MATRIX_EXP_THETA * MATRIX_EXP_THETA;  // theta^2 * 4^0
        while (s < MATRIX_EXP_S_MAX && normSq > bound) {
            bound *= 4.0f;
            ++s;
        }
        sigma = 1.0f;
        for (uint32_t i = 0; i < s; ++i) {
            sigma *= 0.5f;
        }
        // meta 槽写 s（Duplicate 全 8 元素，AIC 只读 [0]；V pipe 产物，随 MTE3 一并发出）。
        Duplicate(metaF32, static_cast<float>(s), MATRIX_EXP_WS_META_FLOATS);
        // S pipe 标量读 tmpF32 已结束；Horner 首轮 MTE2 将重写 tmpF32，须 S->MTE2 排序
        //（S 与 MTE2 并行，否则 copy 可能覆盖正在被读的归约结果）。
        SetFlag<HardEvent::S_MTE2>(EVT_AIV_S_MTE2);
    }

    // 阶段 3：E0 = I + (sigma/9) * A（eye 为 nPad_ 方阵，尾行/行尾零）。
    // 缩放结果写 aF32 原地（tmpF32 不得再被 V 写，S 读依赖见 ComputeScaling）。
    __aicore__ inline void InitE0(LocalTensor<float>& aF32, LocalTensor<float>& eF32,
                                  const LocalTensor<float>& metaF32, float sigma)
    {
        DataCopyPad(eF32, eyeGm_,
                    DataCopyExtParams{nPad_, nPad_ * sizeof(float), 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        SyncMte2V();
        Muls(aF32, aF32, sigma / static_cast<float>(MATRIX_EXP_HORNER_ROUNDS + 1), bufElems_);
        Add(eF32, eF32, aF32, bufElems_);
        SyncVMte3();
        DataCopy(metaGm_, metaF32, MATRIX_EXP_WS_META_FLOATS);
        CopyUbToWs(wsEGm_, eF32);
        SetAivReady(readyCnt_);  // E0 + meta 就绪
    }

    // 阶段 4：Horner 8 轮嵌套递推：E_{j+1} = I + (sigma/k_j) * (A @ E_j)，k_j = 8..1。
    // 每轮从 identity 重新累加（非乘积形式）；I 由首轮从 eyeGm_ 装入 aF32 常驻
    //（A 已用完；装载经 WaitAicDone -> AIC DONE <- READY <- V 完成链自动排序）。
    __aicore__ inline void RunHornerRounds(LocalTensor<float>& aF32, LocalTensor<float>& eF32,
                                           LocalTensor<float>& tmpF32, uint32_t s, float sigma)
    {
        for (uint32_t j = 0; j < MATRIX_EXP_HORNER_ROUNDS; ++j) {
            WaitAicDone(doneCnt_);
            if (j == 0) {
                // MTE2 等 S 读完 tmpF32（见 ComputeScaling SetFlag），再重写 tmpF32 / 覆盖 aF32。
                WaitFlag<HardEvent::S_MTE2>(EVT_AIV_S_MTE2);
                DataCopyPad(aF32, eyeGm_,
                            DataCopyExtParams{nPad_, nPad_ * sizeof(float), 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
            CopyWsToUb(tmpF32, wsTmpGm_);
            SyncMte2V();
            Muls(tmpF32, tmpF32, sigma / static_cast<float>(MATRIX_EXP_HORNER_ROUNDS - j),
                 bufElems_);
            Adds(eF32, aF32, 0.0f, bufElems_);           // E <- I（UB 常驻拷贝）
            Add(eF32, eF32, tmpF32, bufElems_);          // E <- I + (sigma/k) * (A@E_j)
            SyncVMte3();
            CopyUbToWs(wsEGm_, eF32);
            if ((j + 1 < MATRIX_EXP_HORNER_ROUNDS) || (s > 0)) {
                SetAivReady(readyCnt_);
            }
        }
    }

    // 阶段 5：尾轮输出。s==0 用 eF32（AIC 尾轮 DONE 仅被消费不使用）；s>0 取 AIC 最终 tmp。
    __aicore__ inline void WriteOutput(LocalTensor<float>& eF32, LocalTensor<float>& tmpF32, uint32_t s)
    {
        if (s > 0) {
            WaitAicDone(doneCnt_);
            CopyWsToUb(tmpF32, wsTmpGm_);
            SyncMte2V();
        }
        const LocalTensor<float>& src = (s > 0) ? tmpF32 : eF32;
        if constexpr (sizeof(T) == sizeof(float)) {
            SyncVMte3();
            CopyUbToGm(outGm_, src);
        } else {
            auto ioT = bufIo_.Get<T>();
            Cast(ioT, src, RoundMode::CAST_RINT, bufElems_);
            SyncVMte3();
            CopyUbToGm(outGm_, ioT);
        }
    }

    // n==1 快速路径：batch 维逐元素 exp，纯 Vector，分块搬运，与 AIC 无交互。
    __aicore__ inline void ProcessSmall()
    {
        auto aF32 = bufA_.Get<float>();
        const uint32_t alignTo = 32U / sizeof(T);  // fp32:8, fp16/bf16:16
        uint64_t off = matrixBeg_;
        while (off < matrixEnd_) {
            // 上一轮 DataCopyPad(out) 尚在 MTE3 读 UB 缓冲，须等其完成 V 才可清零复用
            SyncMte3V();
            const uint32_t chunk = static_cast<uint32_t>(
                (matrixEnd_ - off < MATRIX_EXP_SMALL_CHUNK) ? (matrixEnd_ - off)
                                                            : MATRIX_EXP_SMALL_CHUNK);
            const uint32_t alignElems = (chunk + alignTo - 1U) / alignTo * alignTo;
            const uint32_t padElems = alignElems - chunk;
            aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(aPtr_ + off * sizeof(T)), chunk);
            outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outPtr_ + off * sizeof(T)), chunk);
            if constexpr (sizeof(T) == sizeof(float)) {
                Duplicate(aF32, 0.0f, alignElems);
                SyncVMte2();
                DataCopyPad(aF32, aGm_, DataCopyExtParams{1, chunk * sizeof(T), 0, 0, 0},
                            DataCopyPadExtParams<T>{true, 0, static_cast<uint8_t>(padElems), 0});
                SyncMte2V();
                Exp(aF32, aF32, alignElems);
                SyncVMte3();
                DataCopyPad(outGm_, aF32, DataCopyExtParams{1, chunk * sizeof(T), 0, 0, 0});
            } else {
                auto ioT = bufIo_.Get<T>();
                Duplicate(ioT, static_cast<T>(0), alignElems);
                SyncVMte2();
                DataCopyPad(ioT, aGm_, DataCopyExtParams{1, chunk * sizeof(T), 0, 0, 0},
                            DataCopyPadExtParams<T>{true, 0, static_cast<uint8_t>(padElems), 0});
                SyncMte2V();
                Cast(aF32, ioT, RoundMode::CAST_NONE, alignElems);
                Exp(aF32, aF32, alignElems);
                Cast(ioT, aF32, RoundMode::CAST_RINT, alignElems);
                SyncVMte3();
                DataCopyPad(outGm_, ioT, DataCopyExtParams{1, chunk * sizeof(T), 0, 0, 0});
            }
            off += chunk;
        }
    }

    TPipe* pipe_;
    TBuf<TPosition::VECCALC> bufA_;
    TBuf<TPosition::VECCALC> bufE_;
    TBuf<TPosition::VECCALC> bufTmp_;
    TBuf<TPosition::VECCALC> bufMeta_;
    TBuf<TPosition::VECCALC> bufIo_;
    GlobalTensor<T> aGm_;
    GlobalTensor<T> outGm_;
    GlobalTensor<float> wsAGm_;
    GlobalTensor<float> wsEGm_;
    GlobalTensor<float> wsTmpGm_;
    GlobalTensor<float> metaGm_;
    GlobalTensor<float> eyeGm_;
    __gm__ uint8_t* aPtr_;
    __gm__ uint8_t* outPtr_;
    uint32_t n_;
    uint32_t nPad_;
    uint32_t bufElems_;
    uint64_t matrixBeg_;
    uint64_t matrixEnd_;
    uint32_t readyCnt_ = 0;
    uint32_t doneCnt_ = 0;
};

}  // namespace opsgnn

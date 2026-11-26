/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Coalesce segment reduction (arch22).
//
// The Host sorts the PyG integer edge keys and gathers edge_attr into sorted
// order, so every duplicate group is a contiguous run. `indptr` delimits those
// runs; each AIV owns a range of segments and folds each run into one output
// row with a vector reduction (sum / mean / min / max). Duplicate-free inputs
// are the degenerate one-row-per-segment case.

#include "coalesce.h"
#include "kernel_operator.h"
#include <type_traits>

namespace {
using namespace AscendC;

template <HardEvent Event> __aicore__ inline void Sync() {
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}

template <typename T, uint32_t Reduce> class CoalesceReduceKernel {
    static constexpr bool Half = std::is_same<T, half>::value;

  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR indptr, GM_ADDR out,
                                const opsgnn::CoalesceTilingData &t, TPipe *pipe) {
        t_ = &t;
        pipe_ = pipe;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), t.srcNumel);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), t.outNumel);
        indptr_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(indptr), t.U + 1);
        const uint32_t width = t.width;
        pipe_->InitBuffer(srcBuf_, t.tileRows * width * sizeof(T));
        pipe_->InitBuffer(accBuf_, width * sizeof(float));
        pipe_->InitBuffer(outBuf_, width * sizeof(T));
        if constexpr (Half)
            pipe_->InitBuffer(workBuf_, t.tileRows * width * sizeof(float));
        if constexpr (Reduce == opsgnn::kCoalesceMean)
            pipe_->InitBuffer(denomBuf_, width * sizeof(float));
    }

    __aicore__ inline void Process() {
        const uint32_t block = GetBlockIdx();
        const uint32_t blocks = GetBlockNum() ? GetBlockNum() : 1;
        const uint64_t gBegin = static_cast<uint64_t>(block) * t_->U / blocks;
        const uint64_t gEnd = static_cast<uint64_t>(block + 1) * t_->U / blocks;
        const uint32_t width = t_->width;
        const uint32_t channels = static_cast<uint32_t>(t_->C);
        auto acc = accBuf_.Get<float>();
        for (uint64_t g = gBegin; g < gEnd; ++g) {
            const uint64_t start = static_cast<uint64_t>(indptr_.GetValue(g));
            const uint64_t end = static_cast<uint64_t>(indptr_.GetValue(g + 1));
            const uint64_t rows = end - start;
            if constexpr (Reduce == opsgnn::kCoalesceMin)
                Duplicate(acc, __builtin_inff(), width);
            else if constexpr (Reduce == opsgnn::kCoalesceMax)
                Duplicate(acc, -__builtin_inff(), width);
            else
                Duplicate(acc, 0.0f, width);
            PipeBarrier<PIPE_V>();
            for (uint64_t rowBase = 0; rowBase < rows; rowBase += t_->tileRows) {
                const uint32_t count = static_cast<uint32_t>(MinU64(t_->tileRows, rows - rowBase));
                LoadRows((start + rowBase) * t_->C, count, channels, width);
                auto work = WorkRows(count, width);
                for (uint32_t r = 0; r < count; ++r) {
                    if constexpr (Reduce == opsgnn::kCoalesceMin)
                        Min(acc, acc, work[r * width], width);
                    else if constexpr (Reduce == opsgnn::kCoalesceMax)
                        Max(acc, acc, work[r * width], width);
                    else
                        Add(acc, acc, work[r * width], width);
                    PipeBarrier<PIPE_V>();
                }
                Sync<HardEvent::V_MTE2>();
            }
            if constexpr (Reduce == opsgnn::kCoalesceMean) {
                auto denominator = denomBuf_.Get<float>();
                // Scalar conversion on AIV accepts signed int32 only; a segment
                // length fits int32 for every addressable edge list.
                Duplicate(denominator, static_cast<float>(static_cast<int32_t>(rows)), width);
                PipeBarrier<PIPE_V>();
                Div(acc, acc, denominator, width);
                PipeBarrier<PIPE_V>();
            }
            StoreRow(g, channels, width);
            Sync<HardEvent::MTE3_V>();
        }
    }

  private:
    __aicore__ static inline uint64_t MinU64(uint64_t a, uint64_t b) { return a < b ? a : b; }

    __aicore__ inline void LoadRows(uint64_t offset, uint32_t rows, uint32_t valid, uint32_t width) {
        auto source = srcBuf_.Get<T>();
        // Destination rows occupy `width` lanes; only `valid` carry data and the
        // tail is zero padded inside the rounded 32-byte block.
        const auto pad = DataCopyPadExtParams<T>(true, 0, static_cast<uint8_t>(width - valid), 0);
        DataCopyPad(source, src_[offset],
                    {static_cast<uint16_t>(rows), valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}, pad);
        Sync<HardEvent::MTE2_V>();
    }

    __aicore__ inline LocalTensor<float> WorkRows(uint32_t rows, uint32_t width) {
        if constexpr (Half) {
            auto work = workBuf_.Get<float>();
            Cast(work, srcBuf_.Get<T>(), RoundMode::CAST_NONE, rows * width);
            PipeBarrier<PIPE_V>();
            return work;
        } else {
            return srcBuf_.Get<float>();
        }
    }

    __aicore__ inline void StoreRow(uint64_t g, uint32_t valid, uint32_t width) {
        auto acc = accBuf_.Get<float>();
        if constexpr (Half) {
            auto output = outBuf_.Get<T>();
            Cast(output, acc, RoundMode::CAST_RINT, width);
            PipeBarrier<PIPE_V>();
            Sync<HardEvent::V_MTE3>();
            DataCopyPad(out_[g * t_->C], output, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        } else {
            Sync<HardEvent::V_MTE3>();
            DataCopyPad(out_[g * t_->C], acc, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        }
    }

    const opsgnn::CoalesceTilingData *t_ = nullptr;
    TPipe *pipe_ = nullptr;
    GlobalTensor<T> src_, out_;
    GlobalTensor<int64_t> indptr_;
    TBuf<TPosition::VECCALC> srcBuf_, workBuf_, accBuf_, outBuf_, denomBuf_;
};

template <typename T, uint32_t Reduce>
__global__ __aicore__ void CoalesceKernel(GM_ADDR src, GM_ADDR indptr, GM_ADDR out,
                                          opsgnn::CoalesceTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CoalesceReduceKernel<T, Reduce> op;
    op.Init(src, indptr, out, t, &pipe);
    op.Process();
}

template <typename T>
void Dispatch(void *src, int64_t *indptr, void *out, const opsgnn::CoalesceTilingData &t, aclrtStream stream) {
#define LAUNCH(R)                                                                                                      \
    CoalesceKernel<T, R>                                                                                               \
        <<<t.usedBlockNum, nullptr, stream>>>(                                                                         \
            reinterpret_cast<uint8_t *>(src), reinterpret_cast<uint8_t *>(indptr),                                     \
            reinterpret_cast<uint8_t *>(out), t)
    switch (t.reduce) {
    case opsgnn::kCoalesceSum:
        LAUNCH(opsgnn::kCoalesceSum);
        break;
    case opsgnn::kCoalesceMean:
        LAUNCH(opsgnn::kCoalesceMean);
        break;
    case opsgnn::kCoalesceMin:
        LAUNCH(opsgnn::kCoalesceMin);
        break;
    case opsgnn::kCoalesceMax:
        LAUNCH(opsgnn::kCoalesceMax);
        break;
    }
#undef LAUNCH
}

} // namespace

namespace opsgnn {

aclError CoalesceSegmentReduce(void *src, int64_t *indptr, void *out, const CoalesceTilingData &t,
                               aclrtStream stream) {
    if (!t.hasAttr || t.C == 0 || t.U == 0)
        return ACL_SUCCESS;
    if (t.dtype == kCoalesceFloat32)
        Dispatch<float>(src, indptr, out, t, stream);
    else if (t.dtype == kCoalesceFloat16)
        Dispatch<half>(src, indptr, out, t, stream);
    else
        return ACL_ERROR_INVALID_PARAM;
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

} // namespace opsgnn

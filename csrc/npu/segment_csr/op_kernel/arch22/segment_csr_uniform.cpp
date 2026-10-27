/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "segment_csr_common.h"

using namespace AscendC;
namespace opsgnn::csr {

using namespace AscendC;

// Only a complete current-index proof permits this path. Each tree handles
// adjacent equal-length segments independently, with native modular int32 adds.
// No integer value is converted to floating point, including signed division.
template <SegmentReduce Reduce>
__aicore__ inline void CombineInt32(LocalTensor<int32_t> dst, LocalTensor<int32_t> left, LocalTensor<int32_t> right,
                                    uint32_t count) {
    if constexpr (Reduce == SegmentMin)
        AscendC::Min(dst, left, right, count);
    else if constexpr (Reduce == SegmentMax)
        AscendC::Max(dst, left, right, count);
    else
        Add(dst, left, right, count);
    PipeBarrier<PIPE_V>();
}

template <SegmentReduce Reduce> class SegmentInt32Batch {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR out, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer((__gm__ int32_t *)src, t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(int32_t))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        out_.SetGlobalBuffer((__gm__ int32_t *)out, t.outNumel);
        pipe_.InitBuffer(input_, t.tileRows * t.tileK * 4);
        pipe_.InitBuffer(scratch_, t.tileRows * t.tileK * 2);
        if constexpr (Reduce == SegmentMean) {
            pipe_.InitBuffer(sign_, t.outputGroup * t.tileK * 4);
            pipe_.InitBuffer(bias_, t.outputGroup * t.tileK * 4);
        }
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        const uint32_t rows = t_->M / t_->nSegments;
        for (uint64_t unit = begin; unit < end; ++unit) {
            const uint64_t first = unit / t_->featureTiles * t_->outputGroup;
            const uint64_t col = unit % t_->featureTiles * t_->tileK;
            const uint32_t groups = Min(t_->outputGroup, t_->totalTasks - first);
            const uint32_t width = Min(t_->tileK, t_->K - col);
            auto work = input_.Get<int32_t>();
            if (width == t_->K)
                DataCopy(work, src_[first * rows * t_->K], groups * rows * width);
            else
                DataCopyPad(
                    work, src_[first * rows * t_->K + col],
                    {static_cast<uint16_t>(groups * rows), width * 4, static_cast<uint32_t>((t_->K - width) * 4), 0, 0},
                    DataCopyPadExtParams<int32_t>(false, 0, 0, 0));
            Sync<HardEvent::MTE2_V>();
            work = ReduceRows(work, rows, groups, width);
            if constexpr (Reduce == SegmentMean) Divide(work, groups * width, rows);
            Sync<HardEvent::V_MTE3>();
            if (width == t_->K)
                DataCopy(out_[first * t_->K], work, groups * width);
            else
                DataCopyPad(
                    out_[first * t_->K + col], work,
                    {static_cast<uint16_t>(groups), width * 4, 0, static_cast<uint32_t>((t_->K - width) * 4), 0});
            Sync<HardEvent::MTE3_V>();
            Sync<HardEvent::V_MTE2>();
        }
    }

  private:
    __aicore__ inline LocalTensor<int32_t> ReduceRows(LocalTensor<int32_t> work, uint32_t rows, uint32_t groups,
                                                      uint32_t width) {
        auto next = scratch_.Get<int32_t>();
        bool inScratch = false;
        for (uint32_t remaining = rows; remaining > 1; remaining /= 2) {
            // Integer reductions are associative in their exact modulo
            // domain. Pack adjacent rows into one full vector before each
            // pair stage, preserving independent segment boundaries.
            uint32_t stripe = 1;
            while (stripe * 2 <= remaining / 2 && stripe * 2 * width <= kVectorLanes) stripe *= 2;
            const uint32_t span = stripe * width;
            const uint32_t pairs = groups * remaining / (2 * stripe);
            const BinaryRepeatParams pair{1,
                                          1,
                                          1,
                                          static_cast<uint8_t>(span / 8),
                                          static_cast<uint8_t>(span / 4),
                                          static_cast<uint8_t>(span / 4)};
            for (uint32_t base = 0; base < pairs; base += kVectorRepeatLimit) {
                for (uint32_t c = 0; c < span; c += kVectorLanes) {
                    auto dst = next[base * span + c];
                    auto left = work[2 * base * span + c];
                    auto right = work[(2 * base + 1) * span + c];
                    const uint64_t active = Min(kVectorLanes, span - c);
                    const uint8_t repeats = static_cast<uint8_t>(Min(kVectorRepeatLimit, pairs - base));
                    if constexpr (Reduce == SegmentMin)
                        AscendC::Min(dst, left, right, active, repeats, pair);
                    else if constexpr (Reduce == SegmentMax)
                        AscendC::Max(dst, left, right, active, repeats, pair);
                    else
                        Add(dst, left, right, active, repeats, pair);
                }
            }
            PipeBarrier<PIPE_V>();
            work = next;
            inScratch = !inScratch;
            next = inScratch ? input_.Get<int32_t>() : scratch_.Get<int32_t>();
        }
        return work;
    }
    __aicore__ static inline uint64_t Min(uint64_t a, uint64_t b) { return a < b ? a : b; }
    __aicore__ inline void Divide(LocalTensor<int32_t> value, uint32_t count, uint32_t rows) {
        if (rows == 1) return;
        uint32_t shift = 0;
        for (uint32_t n = rows; n > 1; n /= 2) ++shift;
        auto sign = sign_.Get<uint32_t>();
        auto bias = bias_.Get<uint32_t>();
        auto bits = value.ReinterpretCast<uint32_t>();
        ShiftRight(sign, bits, uint32_t(31), count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(bias, sign, shift, count);
        PipeBarrier<PIPE_V>();
        Sub(bias.ReinterpretCast<int32_t>(), bias.ReinterpretCast<int32_t>(), sign.ReinterpretCast<int32_t>(), count);
        PipeBarrier<PIPE_V>();
        // A negative numerator receives rows-1 before arithmetic shifting,
        // implementing truncation toward zero after source-width sum overflow.
        Add(value, value, bias.ReinterpretCast<int32_t>(), count);
        PipeBarrier<PIPE_V>();
        ShiftRight(sign, bits, uint32_t(31), count);
        Duplicate(bias.ReinterpretCast<int32_t>(), 0, count);
        PipeBarrier<PIPE_V>();
        Sub(sign.ReinterpretCast<int32_t>(), bias.ReinterpretCast<int32_t>(), sign.ReinterpretCast<int32_t>(), count);
        ShiftRight(bits, bits, shift, count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(sign, sign, uint32_t(32) - shift, count);
        PipeBarrier<PIPE_V>();
        Or(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(), sign.ReinterpretCast<uint16_t>(),
           count * 2);
        PipeBarrier<PIPE_V>();
    }
    const SegmentCsrTilingData *t_;
    GlobalTensor<int32_t> src_, out_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_, scratch_, sign_, bias_;
};

template <SegmentReduce Reduce>
__global__ __aicore__ void SegmentInt32BatchKernel(GM_ADDR src, GM_ADDR out, SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentInt32Batch<Reduce> op;
    op.Init(src, out, t);
    op.Process();
}

// Complete short segments use dense full-row DMA. Queue ownership allows
// the next input transfer and previous output transfer to overlap the current
// native-int tree; the serial fallback below is retained for wider features.
template <SegmentReduce Reduce> class SegmentInt32Pipeline {
  public:
    __aicore__ inline void Init(GM_ADDR source, GM_ADDR destination, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(source), t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(int32_t))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(destination), t.outNumel);
        const uint32_t inputs = t.outputGroup * t.tileRows * t.tileK;
        const uint32_t outputs = t.outputGroup * t.tileK;
        pipe_.InitBuffer(input_, 2, inputs * 4);
        pipe_.InitBuffer(scratch_, inputs * 2);
        pipe_.InitBuffer(output_, 2, outputs * 4);
        if constexpr (Reduce == SegmentMean) {
            pipe_.InitBuffer(sign_, outputs * 4);
            pipe_.InitBuffer(bias_, outputs * 4);
        }
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        if (begin == end) return;
        Load(begin);
        for (uint64_t unit = begin; unit < end; ++unit) {
            auto input = input_.DeQue<int32_t>();
            if (unit + 1 < end) Load(unit + 1);
            const uint64_t first = unit * t_->outputGroup;
            const uint32_t groups = Small(t_->outputGroup, t_->totalTasks - first);
            const uint32_t width = t_->tileK, rows = t_->tileRows;
            auto work = ReduceRows(input, rows, groups, width);
            const uint32_t count = groups * width;
            if constexpr (Reduce == SegmentMean) {
                uint32_t shift = 0;
                for (uint32_t n = rows; n > 1; n /= 2) ++shift;
                auto sign = sign_.Get<int32_t>(), bias = bias_.Get<int32_t>();
                ShiftRight(sign, work, static_cast<int32_t>(31), count);
                PipeBarrier<PIPE_V>();
                Muls(bias, sign, -static_cast<int32_t>(rows - 1), count);
                PipeBarrier<PIPE_V>();
                Add(work, work, bias, count);
                PipeBarrier<PIPE_V>();
                ShiftRight(work, work, static_cast<int32_t>(shift), count);
                PipeBarrier<PIPE_V>();
            }
            auto output = output_.AllocTensor<int32_t>();
            Adds(output, work, static_cast<int32_t>(0), count);
            output_.EnQue(output);
            input_.FreeTensor(input);
            auto ready = output_.DeQue<int32_t>();
            DataCopy(out_[first * width], ready, count);
            output_.FreeTensor(ready);
        }
    }

  private:
    __aicore__ inline LocalTensor<int32_t> ReduceRows(LocalTensor<int32_t> input, uint32_t rows, uint32_t groups,
                                                      uint32_t width) {
        auto work = input, next = scratch_.Get<int32_t>();
        bool inScratch = false;
        for (uint32_t remaining = rows; remaining > 1; remaining /= 2) {
            uint32_t stripe = 1;
            while (stripe * 2 <= remaining / 2 && stripe * 2 * width <= kVectorLanes) stripe *= 2;
            const uint32_t span = stripe * width;
            const uint32_t pairs = groups * remaining / (2 * stripe);
            const BinaryRepeatParams repeatParams{1,
                                                  1,
                                                  1,
                                                  static_cast<uint8_t>(span / 8),
                                                  static_cast<uint8_t>(span / 4),
                                                  static_cast<uint8_t>(span / 4)};
            for (uint32_t base = 0; base < pairs; base += kVectorRepeatLimit) {
                for (uint32_t col = 0; col < span; col += kVectorLanes) {
                    auto dst = next[base * span + col];
                    auto left = work[2 * base * span + col];
                    auto right = work[(2 * base + 1) * span + col];
                    const uint64_t active = Small(kVectorLanes, span - col);
                    const uint8_t repeats = static_cast<uint8_t>(Small(kVectorRepeatLimit, pairs - base));
                    if constexpr (Reduce == SegmentMin)
                        AscendC::Min(dst, left, right, active, repeats, repeatParams);
                    else if constexpr (Reduce == SegmentMax)
                        AscendC::Max(dst, left, right, active, repeats, repeatParams);
                    else
                        Add(dst, left, right, active, repeats, repeatParams);
                }
            }
            PipeBarrier<PIPE_V>();
            work = next;
            inScratch = !inScratch;
            next = inScratch ? input : scratch_.Get<int32_t>();
        }
        return work;
    }
    __aicore__ static inline uint32_t Small(uint64_t a, uint64_t b) { return static_cast<uint32_t>(a < b ? a : b); }
    __aicore__ inline void Load(uint64_t unit) {
        const uint64_t first = unit * t_->outputGroup;
        const uint32_t groups = Small(t_->outputGroup, t_->totalTasks - first);
        auto input = input_.AllocTensor<int32_t>();
        DataCopy(input, src_[first * t_->tileRows * t_->tileK], groups * t_->tileRows * t_->tileK);
        input_.EnQue(input);
    }
    const SegmentCsrTilingData *t_;
    GlobalTensor<int32_t> src_, out_;
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> input_;
    TQue<TPosition::VECOUT, 2> output_;
    TBuf<TPosition::VECCALC> scratch_, sign_, bias_;
};

template <SegmentReduce Reduce>
__global__ __aicore__ void SegmentInt32PipelineKernel(GM_ADDR src, GM_ADDR out, SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentInt32Pipeline<Reduce> op;
    op.Init(src, out, t);
    op.Process();
}

// Long uniform segments keep complete feature rows in dense DMA tiles. Two
// queue slots overlap the next row transfer with the current native-int tree.
// Each in-place stage aliases dst exactly with its first source and reads a
// disjoint second half; no overlapping pair compaction is performed.
template <SegmentReduce Reduce> class SegmentInt32Long {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR out, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer((__gm__ int32_t *)src, t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(int32_t))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        out_.SetGlobalBuffer((__gm__ int32_t *)out, t.outNumel);
        pipe_.InitBuffer(input_, 2, t.tileRows * t.tileK * 4);
        pipe_.InitBuffer(acc_, t.tileK * 4);
        if constexpr (Reduce == SegmentMean) {
            pipe_.InitBuffer(sign_, t.tileK * 4);
            pipe_.InitBuffer(bias_, t.tileK * 4);
        }
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalTasks / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalTasks / t_->usedBlockNum;
        const uint32_t rows = t_->M / t_->nSegments;
        const uint32_t width = t_->tileK;
        for (uint64_t task = begin; task < end; ++task) {
            auto acc = acc_.Get<int32_t>();
            Duplicate(acc, Reduce == SegmentMin ? INT32_MAX : Reduce == SegmentMax ? INT32_MIN : 0, width);
            PipeBarrier<PIPE_V>();
            CopyIn(task, 0, Small(rows, t_->tileRows));
            for (uint32_t row = 0; row < rows; row += t_->tileRows) {
                auto values = input_.DeQue<int32_t>();
                const uint32_t active = Small(t_->tileRows, rows - row);
                const uint32_t next = row + active;
                if (next < rows) CopyIn(task, next, Small(t_->tileRows, rows - next));
                for (uint32_t remaining = active; remaining > 1; remaining /= 2) {
                    const uint32_t count = (remaining / 2) * width;
                    Combine(values, values, values[count], count);
                }
                Combine(acc, acc, values, width);
                input_.FreeTensor(values);
            }
            if constexpr (Reduce == SegmentMean) Divide(acc, width, rows);
            Sync<HardEvent::V_MTE3>();
            DataCopy(out_[task * width], acc, width);
            Sync<HardEvent::MTE3_V>();
        }
    }

  private:
    __aicore__ static inline uint32_t Small(uint32_t a, uint32_t b) { return a < b ? a : b; }
    __aicore__ inline void CopyIn(uint64_t task, uint32_t row, uint32_t rows) {
        auto value = input_.AllocTensor<int32_t>();
        DataCopy(value, src_[(task * (t_->M / t_->nSegments) + row) * t_->K], rows * t_->tileK);
        input_.EnQue(value);
    }
    __aicore__ inline void Combine(LocalTensor<int32_t> dst, LocalTensor<int32_t> left, LocalTensor<int32_t> right,
                                   uint32_t count) {
        CombineInt32<Reduce>(dst, left, right, count);
    }
    __aicore__ inline void Divide(LocalTensor<int32_t> values, uint32_t count, uint32_t rows) {
        uint32_t shift = 0;
        for (uint32_t n = rows; n > 1; n /= 2) ++shift;
        auto sign = sign_.Get<int32_t>(), bias = bias_.Get<int32_t>();
        ShiftRight(sign, values, static_cast<int32_t>(31), count);
        PipeBarrier<PIPE_V>();
        Muls(bias, sign, -static_cast<int32_t>(rows - 1), count);
        PipeBarrier<PIPE_V>();
        Add(values, values, bias, count);
        PipeBarrier<PIPE_V>();
        ShiftRight(values, values, static_cast<int32_t>(shift), count);
        PipeBarrier<PIPE_V>();
    }
    const SegmentCsrTilingData *t_;
    GlobalTensor<int32_t> src_, out_;
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> input_;
    TBuf<TPosition::VECCALC> acc_, sign_, bias_;
};

template <SegmentReduce Reduce>
__global__ __aicore__ void SegmentInt32LongKernel(GM_ADDR src, GM_ADDR out, SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentInt32Long<Reduce> op;
    op.Init(src, out, t);
    op.Process();
}
} // namespace opsgnn::csr

// ============================================================================
// Proven INT64 narrow reductions
// ============================================================================

namespace opsgnn::csr {
using namespace AscendC;
struct CsrPairStage {
    uint32_t span, pairs;
    BinaryRepeatParams repeat;
};
__aicore__ inline CsrPairStage PairStage(uint32_t rows, uint32_t groups, uint32_t width) {
    uint32_t stripe = 1;
    while (stripe * 2 * width <= kVectorLanes && rows % (stripe * 4) == 0) stripe *= 2;
    const uint32_t span = stripe * width;
    return {span,
            groups * rows / (2 * stripe),
            {1, 1, 1, static_cast<uint8_t>(span / 8), static_cast<uint8_t>(span / 4), static_cast<uint8_t>(span / 4)}};
}

template <SegmentReduce Reduce> class SegmentInt64Narrow {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR out, const SegmentCsrTilingData &t) {
        t_ = &t;
        capacity_ = t.tileRows * t.tileK;
        outCapacity_ = t.outputGroup * t.tileK;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(src), t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(int64_t))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(out), t.outNumel);
        pipe_.InitBuffer(input_, capacity_ * 8);
        pipe_.InitBuffer(low_, capacity_ * 4);
        pipe_.InitBuffer(acc_, t.tileK * 4);
        pipe_.InitBuffer(packed_, outCapacity_ * 8);
        pipe_.InitBuffer(offsets_, outCapacity_ * 8);
        maskBytes_ = ((outCapacity_ * 2 + 7) / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes;
        pipe_.InitBuffer(mask_, maskBytes_);
        auto off = offsets_.Get<uint32_t>();
        // Eight scalar seeds suffice; construct the exact repeated byte-offset
        // sequence with logarithmic vector doubling instead of thousands of
        // per-core scalar stores on every public call.
        for (uint32_t i = 0; i < 8; ++i) off.SetValue(i, (i / 2) * 4);
        Sync<HardEvent::S_V>();
        for (uint32_t ready = 8; ready < outCapacity_ * 2; ready *= 2) {
            const uint32_t count = Small(ready, outCapacity_ * 2 - ready);
            Adds(off.ReinterpretCast<int32_t>()[ready], off.ReinterpretCast<int32_t>(), static_cast<int32_t>(ready * 2),
                 count);
            PipeBarrier<PIPE_V>();
        }
        Duplicate(mask_.Get<uint16_t>(), static_cast<uint16_t>(0xAAAA), maskBytes_ / 2);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum,
                       end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        const uint32_t rows = t_->M / t_->nSegments;
        for (uint64_t unit = begin; unit < end; ++unit) {
            const uint64_t first = t_->outputGroup == 1 ? unit / t_->featureTiles : unit * t_->outputGroup;
            const uint64_t last = Small(first + t_->outputGroup, t_->totalTasks);
            const uint64_t cb = t_->outputGroup == 1 ? unit % t_->featureTiles * t_->tileK : 0;
            const uint64_t ce = t_->outputGroup == 1 ? Small(cb + t_->tileK, t_->K) : t_->K;
            for (uint64_t col = cb; col < ce; col += t_->tileK) {
                const uint32_t width = Small(t_->tileK, t_->K - col);
                if (rows <= t_->tileRows && (last - first) * rows <= t_->tileRows &&
                    first / t_->nSegments == (last - 1) / t_->nSegments) {
                    if (!LoadProve(first * rows * t_->K + col, (last - first) * rows, width, rows)) return false;
                    auto value = Tree(low_.Get<int32_t>(), rows, last - first, width);
                    Store(value, first * t_->K + col, last - first, width, rows);
                } else {
                    if (!ProcessLongTasks(first, last, col, rows)) return false;
                }
            }
        }
        return true;
    }

  private:
    __aicore__ inline bool ProcessLongTasks(uint64_t first, uint64_t last, uint64_t col, uint32_t rows) {
        const uint32_t width = Small(t_->tileK, t_->K - col);
        for (uint64_t task = first; task < last; ++task) {
            auto acc = acc_.Get<int32_t>();
            Duplicate(acc, Reduce == SegmentMin ? INT32_MAX : Reduce == SegmentMax ? INT32_MIN : 0, width);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < rows; row += t_->tileRows) {
                const uint32_t active = Small(rows - row, t_->tileRows);
                if (!LoadProve((task * rows + row) * t_->K + col, active, width, rows)) return false;
                auto value = Tree(low_.Get<int32_t>(), active, 1, width);
                if constexpr (Reduce == SegmentMin)
                    AscendC::Min(acc, acc, value, width);
                else if constexpr (Reduce == SegmentMax)
                    AscendC::Max(acc, acc, value, width);
                else
                    Add(acc, acc, value, width);
                PipeBarrier<PIPE_V>();
                Sync<HardEvent::V_MTE2>();
            }
            Store(acc, task * t_->K + col, 1, width, rows);
        }
        return true;
    }
    __aicore__ static inline uint64_t Small(uint64_t a, uint64_t b) { return a < b ? a : b; }
    // The common Host has already proved the CURRENT source version lies in
    // [-bound,bound] using an exact NPU 64-bit checker. A false or invalidated
    // proof dispatches the full-width kernel. No source value is cast to float.
    __aicore__ inline bool LoadProve(uint64_t offset, uint32_t rows, uint32_t width, uint32_t segmentRows) {
        auto input = input_.Get<int64_t>();
        const uint32_t count = rows * width;
        if (width == t_->K)
            DataCopy(input, src_[offset], count);
        else
            DataCopyPad(input, src_[offset],
                        {static_cast<uint16_t>(rows), width * 8, static_cast<uint32_t>((t_->K - width) * 8), 0, 0},
                        DataCopyPadExtParams<int64_t>(false, 0, 0, 0));
        Sync<HardEvent::MTE2_V>();
        uint64_t reserved = 0;
        GatherMask(low_.Get<int32_t>(), input_.Get<int32_t>(), 1, true, count * 2, {1, 1, 8, 0}, reserved);
        PipeBarrier<PIPE_V>();
        return true;
    }
    __aicore__ inline LocalTensor<int32_t> Tree(LocalTensor<int32_t> work, uint32_t rows, uint32_t groups,
                                                uint32_t width) {
        auto next = input_.Get<int32_t>();
        bool scratch = false;
        while (rows > 1) {
            const auto stage = PairStage(rows, groups, width);
            const auto span = stage.span, pairs = stage.pairs;
            const auto rp = stage.repeat;
            for (uint32_t first = 0; first < pairs; first += kVectorRepeatLimit) {
                const uint8_t repeat = Small(kVectorRepeatLimit, pairs - first);
                for (uint32_t c = 0; c < span; c += kVectorLanes) {
                    auto a = work[first * span * 2 + c], b = work[first * span * 2 + span + c],
                         d = next[first * span + c];
                    if constexpr (Reduce == SegmentMin)
                        AscendC::Min(d, a, b, Small(kVectorLanes, span - c), repeat, rp);
                    else if constexpr (Reduce == SegmentMax)
                        AscendC::Max(d, a, b, Small(kVectorLanes, span - c), repeat, rp);
                    else
                        Add(d, a, b, Small(kVectorLanes, span - c), repeat, rp);
                }
            }
            PipeBarrier<PIPE_V>();
            // A long single segment can end in a non-power-of-two row tile. Carry its
            // final unpaired row into the next level instead of dropping that row.
            if (rows & 1) {
                Adds(next[(rows / 2) * width], work[(rows - 1) * width], static_cast<int32_t>(0), width);
                PipeBarrier<PIPE_V>();
            }
            work = next;
            scratch = !scratch;
            next = scratch ? low_.Get<int32_t>() : input_.Get<int32_t>();
            rows = (rows + 1) / 2;
        }
        return work;
    }
    __aicore__ inline void Store(LocalTensor<int32_t> value, uint64_t offset, uint32_t groups, uint32_t width,
                                 uint32_t rows) {
        const uint32_t count = groups * width;
        // Before packing, packed_ is scratch for mean correction. After Gather,
        // the complete low-word tile is dead and holds the interleaved sign words.
        // A short batch has tileRows >= 2*groups; a long segment has groups=1.
        // Do not use value as scratch: a long segment passes the one-row acc_.
        auto temp = packed_.Get<int32_t>();
        if constexpr (Reduce == SegmentMean) {
            uint32_t shift = 0;
            for (uint32_t n = rows; n > 1; n /= 2) ++shift;
            ShiftRight(temp, value, static_cast<int32_t>(31), count);
            PipeBarrier<PIPE_V>();
            Muls(temp, temp, -static_cast<int32_t>(rows - 1), count);
            PipeBarrier<PIPE_V>();
            Add(value, value, temp, count);
            PipeBarrier<PIPE_V>();
            ShiftRight(value, value, static_cast<int32_t>(shift), count);
            PipeBarrier<PIPE_V>();
        }
        auto packed = packed_.Get<int32_t>();
        Gather(packed, value, offsets_.Get<uint32_t>(), 0, count * 2);
        PipeBarrier<PIPE_V>();
        temp = low_.Get<int32_t>();
        ShiftRight(temp, packed, static_cast<int32_t>(31), count * 2);
        PipeBarrier<PIPE_V>();
        Select(packed.ReinterpretCast<float>(), mask_.Get<uint8_t>(), temp.ReinterpretCast<float>(),
               packed.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count * 2);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_MTE3>();
        if (width == t_->K)
            DataCopy(out_[offset], packed.ReinterpretCast<int64_t>(), groups * width);
        else
            DataCopyPad(out_[offset], packed.ReinterpretCast<int64_t>(),
                        {static_cast<uint16_t>(groups), width * 8, 0, static_cast<uint32_t>((t_->K - width) * 8), 0});
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::V_MTE2>();
    }
    const SegmentCsrTilingData *t_;
    uint32_t capacity_, outCapacity_, maskBytes_;
    GlobalTensor<int64_t> src_, out_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_, low_, acc_, packed_, offsets_, mask_;
};

// Proven bounded int64 values reduce as exact int32 low words. Long uniform
// segments retain complete feature rows and overlap two dense 64-KiB input
// slots. Full-width values never enter this route without the existing proof.
template <SegmentReduce Reduce> class SegmentInt64NarrowLong {
  public:
    __aicore__ inline void Init(GM_ADDR source, GM_ADDR destination, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(source), t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(int64_t))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(destination), t.outNumel);
        pipe_.InitBuffer(input_, 2, t.tileRows * t.tileK * 8);
        pipe_.InitBuffer(low_, t.tileRows * t.tileK * 4);
        pipe_.InitBuffer(acc_, t.tileK * 4);
        pipe_.InitBuffer(packed_, t.tileK * 8);
        pipe_.InitBuffer(offsets_, t.tileK * 8);
        const uint32_t maskBytes = ((t.tileK * 2 + 7) / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes;
        pipe_.InitBuffer(mask_, maskBytes);
        auto off = offsets_.Get<uint32_t>();
        for (uint32_t i = 0; i < 8; ++i) off.SetValue(i, (i / 2) * 4);
        Sync<HardEvent::S_V>();
        for (uint32_t ready = 8; ready < t.tileK * 2; ready *= 2) {
            Adds(off.ReinterpretCast<int32_t>()[ready], off.ReinterpretCast<int32_t>(), static_cast<int32_t>(ready * 2),
                 Small(ready, t.tileK * 2 - ready));
            PipeBarrier<PIPE_V>();
        }
        Duplicate(mask_.Get<uint16_t>(), static_cast<uint16_t>(0xAAAA), maskBytes / 2);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalTasks / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalTasks / t_->usedBlockNum;
        const uint32_t rows = t_->M / t_->nSegments, width = t_->tileK;
        for (uint64_t task = begin; task < end; ++task) {
            auto acc = acc_.Get<int32_t>();
            Duplicate(acc, Reduce == SegmentMin ? INT32_MAX : Reduce == SegmentMax ? INT32_MIN : 0, width);
            PipeBarrier<PIPE_V>();
            CopyIn(task, 0, Small(rows, t_->tileRows));
            for (uint32_t row = 0; row < rows; row += t_->tileRows) {
                auto input = input_.DeQue<int64_t>();
                const uint32_t active = Small(t_->tileRows, rows - row);
                const uint32_t next = row + active;
                if (next < rows) CopyIn(task, next, Small(t_->tileRows, rows - next));
                auto low = low_.Get<int32_t>();
                uint64_t reserved = 0;
                GatherMask(low, input.template ReinterpretCast<int32_t>(), 1, true, active * width * 2, {1, 1, 8, 0},
                           reserved);
                PipeBarrier<PIPE_V>();
                for (uint32_t remaining = active; remaining > 1; remaining /= 2) {
                    const uint32_t count = remaining / 2 * width;
                    Combine(low, low, low[count], count);
                }
                Combine(acc, acc, low, width);
                input_.FreeTensor(input);
            }
            Store(task, rows, width);
        }
    }

  private:
    __aicore__ static inline uint32_t Small(uint32_t a, uint32_t b) { return a < b ? a : b; }
    __aicore__ inline void CopyIn(uint64_t task, uint32_t row, uint32_t count) {
        auto values = input_.AllocTensor<int64_t>();
        DataCopy(values, src_[(task * (t_->M / t_->nSegments) + row) * t_->K], count * t_->tileK);
        input_.EnQue(values);
    }
    __aicore__ inline void Combine(LocalTensor<int32_t> dst, LocalTensor<int32_t> left, LocalTensor<int32_t> right,
                                   uint32_t count) {
        CombineInt32<Reduce>(dst, left, right, count);
    }
    __aicore__ inline void Store(uint64_t task, uint32_t rows, uint32_t width) {
        auto value = acc_.Get<int32_t>();
        auto temporary = low_.Get<int32_t>();
        if constexpr (Reduce == SegmentMean) {
            uint32_t shift = 0;
            for (uint32_t n = rows; n > 1; n /= 2) ++shift;
            ShiftRight(temporary, value, static_cast<int32_t>(31), width);
            PipeBarrier<PIPE_V>();
            Muls(temporary, temporary, -static_cast<int32_t>(rows - 1), width);
            PipeBarrier<PIPE_V>();
            Add(value, value, temporary, width);
            PipeBarrier<PIPE_V>();
            ShiftRight(value, value, static_cast<int32_t>(shift), width);
            PipeBarrier<PIPE_V>();
        }
        auto packed = packed_.Get<int32_t>();
        Gather(packed, value, offsets_.Get<uint32_t>(), 0, width * 2);
        PipeBarrier<PIPE_V>();
        // Only the full low-word tile is big enough for both sign words per value.
        ShiftRight(temporary, packed, static_cast<int32_t>(31), width * 2);
        PipeBarrier<PIPE_V>();
        Select(packed.ReinterpretCast<float>(), mask_.Get<uint8_t>(), temporary.ReinterpretCast<float>(),
               packed.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, width * 2);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_MTE3>();
        DataCopy(out_[task * width], packed.ReinterpretCast<int64_t>(), width);
        Sync<HardEvent::MTE3_V>();
    }
    const SegmentCsrTilingData *t_;
    GlobalTensor<int64_t> src_, out_;
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> input_;
    TBuf<TPosition::VECCALC> low_, acc_, packed_, offsets_, mask_;
};

template <SegmentReduce Reduce>
__global__ __aicore__ void SegmentInt64NarrowLongKernel(GM_ADDR source, GM_ADDR out, SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentInt64NarrowLong<Reduce> op;
    op.Init(source, out, t);
    op.Process();
}

// Short proven segments share a dense full-feature-row transfer. Input queue
// slots alternate while the current tile is reduced. Its consumed int64 bytes
// are legal tree scratch until the queue tensor is freed after Store.
template <SegmentReduce Reduce> class SegmentInt64NarrowPipeline {
  public:
    __aicore__ inline void Init(GM_ADDR source, GM_ADDR destination, const SegmentCsrTilingData &t) {
        t_ = &t;
        const uint32_t inputElements = t.outputGroup * t.tileRows * t.tileK;
        const uint32_t outputs = t.outputGroup * t.tileK;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(source), t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(int64_t))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(destination), t.outNumel);
        pipe_.InitBuffer(input_, 2, inputElements * 8);
        pipe_.InitBuffer(low_, inputElements * 4);
        pipe_.InitBuffer(packed_, 2, outputs * 8);
        pipe_.InitBuffer(offsets_, outputs * 8);
        const uint32_t maskBytes = ((outputs * 2 + 7) / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes;
        pipe_.InitBuffer(mask_, maskBytes);
        auto offsets = offsets_.Get<uint32_t>();
        for (uint32_t i = 0; i < 8; ++i) offsets.SetValue(i, (i / 2) * 4);
        Sync<HardEvent::S_V>();
        for (uint32_t ready = 8; ready < outputs * 2; ready *= 2) {
            Adds(offsets.ReinterpretCast<int32_t>()[ready], offsets.ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(ready * 2), Small(ready, outputs * 2 - ready));
            PipeBarrier<PIPE_V>();
        }
        Duplicate(mask_.Get<uint16_t>(), static_cast<uint16_t>(0xAAAA), maskBytes / 2);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        if (begin == end) return;
        Load(begin);
        const uint32_t rows = t_->tileRows, width = t_->tileK;
        for (uint64_t unit = begin; unit < end; ++unit) {
            auto input = input_.DeQue<int64_t>();
            if (unit + 1 < end) Load(unit + 1);
            const uint64_t first = unit * t_->outputGroup;
            const uint32_t groups = Small(t_->outputGroup, t_->totalTasks - first);
            auto low = low_.Get<int32_t>();
            uint64_t reserved = 0;
            GatherMask(low, input.template ReinterpretCast<int32_t>(), 1, true, groups * rows * width * 2, {1, 1, 8, 0},
                       reserved);
            PipeBarrier<PIPE_V>();
            auto value = Tree(low, input.template ReinterpretCast<int32_t>(), rows, groups, width);
            Store(value, first, groups, rows, width);
            input_.FreeTensor(input);
        }
    }

  private:
    __aicore__ static inline uint32_t Small(uint64_t a, uint64_t b) { return static_cast<uint32_t>(a < b ? a : b); }
    __aicore__ inline void Load(uint64_t unit) {
        const uint64_t first = unit * t_->outputGroup;
        const uint32_t groups = Small(t_->outputGroup, t_->totalTasks - first);
        auto input = input_.AllocTensor<int64_t>();
        DataCopy(input, src_[first * t_->tileRows * t_->tileK], groups * t_->tileRows * t_->tileK);
        input_.EnQue(input);
    }
    __aicore__ inline LocalTensor<int32_t> Tree(LocalTensor<int32_t> work, LocalTensor<int32_t> borrowed, uint32_t rows,
                                                uint32_t groups, uint32_t width) {
        auto next = borrowed;
        bool inBorrowed = false;
        while (rows > 1) {
            const auto stage = PairStage(rows, groups, width);
            const auto span = stage.span, pairs = stage.pairs;
            const auto repeatParams = stage.repeat;
            for (uint32_t first = 0; first < pairs; first += kVectorRepeatLimit) {
                const uint8_t repeats = Small(kVectorRepeatLimit, pairs - first);
                for (uint32_t col = 0; col < span; col += kVectorLanes) {
                    auto dst = next[first * span + col];
                    auto left = work[first * span * 2 + col];
                    auto right = work[first * span * 2 + span + col];
                    if constexpr (Reduce == SegmentMin)
                        AscendC::Min(dst, left, right, Small(kVectorLanes, span - col), repeats, repeatParams);
                    else if constexpr (Reduce == SegmentMax)
                        AscendC::Max(dst, left, right, Small(kVectorLanes, span - col), repeats, repeatParams);
                    else
                        Add(dst, left, right, Small(kVectorLanes, span - col), repeats, repeatParams);
                }
            }
            PipeBarrier<PIPE_V>();
            work = next;
            inBorrowed = !inBorrowed;
            next = inBorrowed ? low_.Get<int32_t>() : borrowed;
            rows /= 2;
        }
        return work;
    }
    __aicore__ inline void Store(LocalTensor<int32_t> value, uint64_t first, uint32_t groups, uint32_t rows,
                                 uint32_t width) {
        const uint32_t count = groups * width;
        auto packed = packed_.AllocTensor<int32_t>();
        auto temporary = packed;
        if constexpr (Reduce == SegmentMean) {
            uint32_t shift = 0;
            for (uint32_t n = rows; n > 1; n /= 2) ++shift;
            ShiftRight(temporary, value, static_cast<int32_t>(31), count);
            PipeBarrier<PIPE_V>();
            Muls(temporary, temporary, -static_cast<int32_t>(rows - 1), count);
            PipeBarrier<PIPE_V>();
            Add(value, value, temporary, count);
            PipeBarrier<PIPE_V>();
            ShiftRight(value, value, static_cast<int32_t>(shift), count);
            PipeBarrier<PIPE_V>();
        }
        Gather(packed, value, offsets_.Get<uint32_t>(), 0, count * 2);
        PipeBarrier<PIPE_V>();
        // Every segment has at least two rows, hence low_ has >=2*count words.
        temporary = low_.Get<int32_t>();
        ShiftRight(temporary, packed, static_cast<int32_t>(31), count * 2);
        PipeBarrier<PIPE_V>();
        Select(packed.ReinterpretCast<float>(), mask_.Get<uint8_t>(), temporary.ReinterpretCast<float>(),
               packed.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count * 2);
        PipeBarrier<PIPE_V>();
        // A slot remains owned by MTE3 until FreeTensor's event is satisfied on
        // its next allocation. A second slot lets the next reduction overlap this
        // store without a whole-pipeline MTE3 -> V wait after every tile.
        packed_.EnQue(packed);
        auto output = packed_.DeQue<int32_t>();
        DataCopy(out_[first * width], output.template ReinterpretCast<int64_t>(), count);
        packed_.FreeTensor(output);
    }
    const SegmentCsrTilingData *t_;
    GlobalTensor<int64_t> src_, out_;
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> input_;
    TQue<TPosition::VECOUT, 2> packed_;
    TBuf<TPosition::VECCALC> low_, offsets_, mask_;
};

template <SegmentReduce Reduce>
__global__ __aicore__ void SegmentInt64NarrowPipelineKernel(GM_ADDR source, GM_ADDR out, SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentInt64NarrowPipeline<Reduce> op;
    op.Init(source, out, t);
    op.Process();
}
} // namespace opsgnn::csr

// ============================================================================
// Proven INT64 narrow entry
// ============================================================================

template <opsgnn::csr::SegmentReduce Reduce>
__global__ __aicore__ void SegmentInt64ProvenKernel(GM_ADDR src, GM_ADDR out, SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    opsgnn::csr::SegmentInt64Narrow<Reduce> op;
    op.Init(src, out, t);
    op.Process();
}

// ============================================================================
// Ordered uniform floating reductions
// ============================================================================

namespace opsgnn::csr {
using namespace AscendC;

template <typename T, SegmentReduce Reduce, bool WholeCast = false, bool Bounded = false>
class SegmentCsrOrderedUniform {
    static constexpr bool LowPrecision = !std::is_same<T, float>::value;

  public:
    __aicore__ inline void Run(GM_ADDR source, GM_ADDR destination, const SegmentCsrTilingData &t) {
        Init(source, destination, t);
        const uint64_t begin = GetBlockIdx() * t.totalUnits / t.usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t.totalUnits / t.usedBlockNum;
        const uint64_t steps = (end - begin) * chunks_;
        if (!steps) return;
        Load(begin, 0);
        auto acc = accBuffer_.Get<float>();
        for (uint64_t step = 0; step < steps; ++step) {
            const uint64_t unit = begin + step / chunks_;
            const uint64_t chunk = step % chunks_;
            const uint64_t first = unit * t.outputGroup;
            const uint32_t groups = static_cast<uint32_t>(Small(t.outputGroup, t.totalTasks - first));
            const uint32_t active = static_cast<uint32_t>(Small(t.tileRows, rows_ - chunk * t.tileRows));
            auto sourceTile = input_.template DeQue<T>();
            // The other queue slot is filled while V consumes this slot. This
            // also crosses output-group boundaries when a segment fits one tile.
            if (step + 1 < steps) Load(begin + (step + 1) / chunks_, (step + 1) % chunks_);
            if (!chunk && !Bounded) {
                if constexpr (!LowPrecision) WaitOutput();
                Duplicate(acc, 0.0f, groups * width_);
                PipeBarrier<PIPE_V>();
            }
            Accumulate(sourceTile, active, groups);
            input_.FreeTensor(sourceTile);
            if (chunk + 1 == chunks_) Store(first, groups * width_);
        }
        WaitOutput();
    }

  private:
    __aicore__ inline void Init(GM_ADDR source, GM_ADDR destination, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(source), t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(T))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(destination), t.outNumel);
        width_ = static_cast<uint32_t>(t.K);
        rows_ = t.M / t.nSegments;
        chunks_ = (rows_ + t.tileRows - 1) / t.tileRows;
        count_ = rows_ > 0 ? rows_ : 1;
        powerMean_ = !(count_ & (count_ - 1));
        if constexpr (std::is_same<T, half>::value) powerMean_ = powerMean_ && count_ <= kHalfPowerMeanLimit;
        const uint32_t elements = t.outputGroup * t.tileRows * width_;
        const uint32_t capacity = elements;
        const uint32_t outputs = t.outputGroup * width_;
        pipe_.InitBuffer(input_, 2, capacity * sizeof(T));
        pipe_.InitBuffer(accBuffer_, outputs * 4);
        if constexpr (LowPrecision) {
            pipe_.InitBuffer(rowBuffer_, (WholeCast ? capacity : outputs) * 4);
            pipe_.InitBuffer(outputBuffer_, outputs * sizeof(T));
        }
        if constexpr (Reduce == SegmentMean) {
            if (!powerMean_) pipe_.InitBuffer(denominatorBuffer_, outputs * 4);
        }
        vmte3_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
        mte3v_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
    }
    __aicore__ inline void AccumulateWholeTile(LocalTensor<T> sourceTile, uint32_t active, uint32_t groups) {
        auto acc = accBuffer_.Get<float>();
        // Promote a complete narrow-feature segment tile once.
        // The subsequent FP32 additions still visit every row in
        // native CPU order unless Host proved every current source
        // value finite with magnitude at most 6, for complete
        // power-of-two segments of at most 16 rows.
        auto row = rowBuffer_.Get<float>();
        Cast(row, sourceTile, RoundMode::CAST_NONE, groups * active * width_);
        PipeBarrier<PIPE_V>();
        if constexpr (Bounded) {
            auto value = Tree(row, sourceTile.template ReinterpretCast<float>(), active, groups);
            // CPU starts from +0. Adding +0 also preserves that
            // zero-sign behavior when all inputs are negative zero.
            Adds(acc, value, 0.0f, groups * width_);
            PipeBarrier<PIPE_V>();
        } else {
            const BinaryRepeatParams ordered{1,
                                             1,
                                             1,
                                             static_cast<uint8_t>(width_ / 8),
                                             static_cast<uint8_t>(width_ / 8),
                                             static_cast<uint8_t>(active * width_ / 8)};
            for (uint32_t r = 0; r < active; ++r) {
                for (uint32_t c = 0; c < width_; c += kVectorLanes)
                    Add(acc[c], acc[c], row[r * width_ + c], Small(kVectorLanes, width_ - c),
                        static_cast<uint8_t>(groups), ordered);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void Accumulate(LocalTensor<T> sourceTile, uint32_t active, uint32_t groups) {
        auto acc = accBuffer_.Get<float>();
        if constexpr (LowPrecision) {
            if constexpr (WholeCast) {
                AccumulateWholeTile(sourceTile, active, groups);
            } else {
                auto row = rowBuffer_.Get<float>();
                const UnaryRepeatParams cast{1, 1, static_cast<uint8_t>(width_ / 8),
                                             static_cast<uint8_t>(active * width_ / 16)};
                for (uint32_t r = 0; r < active; ++r) {
                    for (uint32_t c = 0; c < width_; c += kVectorLanes)
                        Cast(row[c], sourceTile[r * width_ + c], RoundMode::CAST_NONE, Small(kVectorLanes, width_ - c),
                             static_cast<uint8_t>(groups), cast);
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, row, groups * width_);
                    PipeBarrier<PIPE_V>();
                }
            }
        } else {
            const BinaryRepeatParams ordered{1,
                                             1,
                                             1,
                                             static_cast<uint8_t>(width_ / 8),
                                             static_cast<uint8_t>(width_ / 8),
                                             static_cast<uint8_t>(active * width_ / 8)};
            for (uint32_t r = 0; r < active; ++r) {
                for (uint32_t c = 0; c < width_; c += kVectorLanes)
                    Add(acc[c], acc[c], sourceTile[r * width_ + c], Small(kVectorLanes, width_ - c),
                        static_cast<uint8_t>(groups), ordered);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ static inline uint64_t Small(uint64_t a, uint64_t b) { return a < b ? a : b; }
    __aicore__ inline LocalTensor<float> Tree(LocalTensor<float> input, LocalTensor<float> scratch, uint32_t rows,
                                              uint32_t groups) {
        auto current = input;
        auto next = scratch;
        bool inScratch = false;
        while (rows > 1) {
            uint32_t stripe = 1;
            while (stripe * 2 <= rows / 2 && stripe * 2 * width_ <= kVectorLanes) stripe *= 2;
            const uint32_t span = stripe * width_;
            const uint32_t pairs = groups * (rows / 2) / stripe;
            const BinaryRepeatParams pair{1,
                                          1,
                                          1,
                                          static_cast<uint8_t>(span / 8),
                                          static_cast<uint8_t>(2 * span / 8),
                                          static_cast<uint8_t>(2 * span / 8)};
            for (uint32_t first = 0; first < pairs; first += kVectorRepeatLimit) {
                const uint8_t repeats = static_cast<uint8_t>(Small(kVectorRepeatLimit, pairs - first));
                for (uint32_t c = 0; c < span; c += kVectorLanes)
                    Add(next[first * span + c], current[2 * first * span + c], current[2 * first * span + span + c],
                        Small(kVectorLanes, span - c), repeats, pair);
            }
            PipeBarrier<PIPE_V>();
            current = next;
            inScratch = !inScratch;
            next = inScratch ? input : scratch;
            rows /= 2;
        }
        return current;
    }
    __aicore__ inline void WaitOutput() {
        if (outputPending_) {
            WaitFlag<HardEvent::MTE3_V>(mte3v_);
            outputPending_ = false;
        }
    }
    __aicore__ inline void Load(uint64_t unit, uint64_t chunk) {
        auto input = input_.template AllocTensor<T>();
        const uint64_t first = unit * t_->outputGroup;
        const uint32_t groups = static_cast<uint32_t>(Small(t_->outputGroup, t_->totalTasks - first));
        const uint64_t base = chunk * t_->tileRows;
        const uint32_t active = static_cast<uint32_t>(Small(t_->tileRows, rows_ - base));
        const uint64_t offset = (first * rows_ + base) * width_;
        if (active == rows_)
            DataCopy(input, src_[offset], groups * active * width_);
        else {
            const uint64_t gap = (rows_ - active) * width_ * sizeof(T);
            if (gap <= UINT32_MAX)
                DataCopyPad(input, src_[offset],
                            {static_cast<uint16_t>(groups), active * width_ * static_cast<uint32_t>(sizeof(T)),
                             static_cast<uint32_t>(gap), 0, 0},
                            DataCopyPadExtParams<T>(false, 0, 0, 0));
            else
                for (uint32_t group = 0; group < groups; ++group)
                    DataCopy(input[group * active * width_], src_[offset + group * rows_ * width_], active * width_);
        }
        input_.EnQue(input);
    }
    __aicore__ inline void StoreHalfMean(uint64_t first, uint32_t outputs) {
        auto narrowed = outputBuffer_.Get<T>();
        if (count_ > 1) {
            Muls(narrowed, narrowed, static_cast<half>(1.0f / CountFloat(count_)), outputs);
            PipeBarrier<PIPE_V>();
        }
        CopyOutput(first, narrowed, outputs);
    }
    __aicore__ inline bool PrepareMeanNumerator(uint64_t first, uint32_t outputs) {
        if constexpr (LowPrecision) {
            auto acc = accBuffer_.Get<float>();
            auto narrowed = outputBuffer_.Get<T>();
            Cast(narrowed, acc, RoundMode::CAST_RINT, outputs);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<T, half>::value) {
                if (powerMean_) {
                    StoreHalfMean(first, outputs);
                    return true;
                }
            }
            Cast(acc, narrowed, RoundMode::CAST_NONE, outputs);
            PipeBarrier<PIPE_V>();
        }
        return false;
    }
    __aicore__ inline void DivideMeanOutput(uint32_t outputs) {
        auto acc = accBuffer_.Get<float>();
        if (powerMean_) {
            if (count_ > 1) {
                Muls(acc, acc, 1.0f / CountFloat(count_), outputs);
                PipeBarrier<PIPE_V>();
            }
        } else {
            auto denominator = denominatorBuffer_.Get<float>();
            Duplicate(denominator, CountFloat(count_), outputs);
            PipeBarrier<PIPE_V>();
            if constexpr (LowPrecision) {
                auto narrowed = outputBuffer_.Get<T>();
                Cast(narrowed, denominator, RoundMode::CAST_RINT, outputs);
                PipeBarrier<PIPE_V>();
                Cast(denominator, narrowed, RoundMode::CAST_NONE, outputs);
                PipeBarrier<PIPE_V>();
            }
            Div(acc, acc, denominator, outputs);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Store(uint64_t first, uint32_t outputs) {
        // Reuse the output staging buffer only after its previous DMA completes.
        WaitOutput();
        if constexpr (Reduce == SegmentMean) {
            if (PrepareMeanNumerator(first, outputs)) return;
            DivideMeanOutput(outputs);
        }
        auto acc = accBuffer_.Get<float>();
        if constexpr (LowPrecision) {
            auto output = outputBuffer_.Get<T>();
            Cast(output, acc, RoundMode::CAST_RINT, outputs);
            PipeBarrier<PIPE_V>();
            CopyOutput(first, output, outputs);
        } else
            CopyOutput(first, acc, outputs);
    }
    __aicore__ inline void CopyOutput(uint64_t first, LocalTensor<T> value, uint32_t outputs) {
        SetFlag<HardEvent::V_MTE3>(vmte3_);
        WaitFlag<HardEvent::V_MTE3>(vmte3_);
        DataCopy(out_[first * width_], value, outputs);
        SetFlag<HardEvent::MTE3_V>(mte3v_);
        outputPending_ = true;
    }
    const SegmentCsrTilingData *t_;
    GlobalTensor<T> src_, out_;
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> input_;
    TBuf<TPosition::VECCALC> rowBuffer_, accBuffer_, outputBuffer_, denominatorBuffer_;
    uint64_t rows_, chunks_, count_;
    uint32_t width_;
    bool powerMean_ = false, outputPending_ = false;
    event_t vmte3_, mte3v_;
};

template <typename T, SegmentReduce Reduce, bool WholeCast = false, bool Bounded = false>
__global__ __aicore__ void SegmentCsrOrderedUniformKernel(GM_ADDR source, GM_ADDR output, SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentCsrOrderedUniform<T, Reduce, WholeCast, Bounded> op;
    op.Run(source, output, t);
}

} // namespace opsgnn::csr

// ============================================================================
// Short uniform floating reductions and public ACL launch
// ============================================================================

namespace opsgnn::csr {
using namespace AscendC;

// Compile the geometry of a complete narrow segment into its scalar address
// calculations and vector repeat descriptors. FP32 visits rows in CPU order;
// FP16 uses the same striped FP32 tree only after the whole-source bounded
// proof. Eight/sixteen-row FP16 tiles trade one input slot for larger groups;
// other geometries retain the existing two-slot input pipeline.
template <typename T, SegmentReduce Reduce, uint32_t Rows, uint32_t Width> class SegmentCsrShortUniform {
    static constexpr bool IsHalf = std::is_same<T, half>::value;
    static constexpr uint32_t InputBuffers = IsHalf && Rows >= 8 ? 1 : 2;

  public:
    __aicore__ inline void Run(GM_ADDR source, GM_ADDR destination, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(source), t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(T))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(destination), t.outNumel);
        const uint32_t outputs = t.outputGroup * Width;
        pipe_.InitBuffer(input_, InputBuffers, outputs * Rows * sizeof(T));
        if constexpr (!IsHalf) pipe_.InitBuffer(acc_, outputs * sizeof(float));
        if constexpr (IsHalf) {
            pipe_.InitBuffer(cast_, outputs * Rows * sizeof(float));
            pipe_.InitBuffer(output_, outputs * sizeof(T));
        }
        vmte3_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
        mte3v_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
        const uint64_t begin = GetBlockIdx() * t.totalUnits / t.usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t.totalUnits / t.usedBlockNum;
        if (begin == end) return;
        Load(begin);
        auto acc = Accumulator();
        for (uint64_t unit = begin; unit < end; ++unit) {
            const uint64_t first = unit * t.outputGroup;
            const uint32_t groups = static_cast<uint32_t>(Small(t.outputGroup, t.totalTasks - first));
            auto sourceTile = input_.template DeQue<T>();
            if constexpr (InputBuffers == 2) {
                if (unit + 1 < end) Load(unit + 1);
            }
            if constexpr (IsHalf) {
                auto promoted = cast_.Get<float>();
                Cast(promoted, sourceTile, RoundMode::CAST_NONE, groups * Rows * Width);
                PipeBarrier<PIPE_V>();
                auto value = Tree<Rows>(promoted, sourceTile.template ReinterpretCast<float>(), groups);
                // Preserve native CPU's initial +0 for an all-negative-zero row.
                Adds(acc, value, 0.0f, groups * Width);
                PipeBarrier<PIPE_V>();
            } else {
                WaitOutput();
                Duplicate(acc, 0.0f, groups * Width);
                PipeBarrier<PIPE_V>();
                Ordered<0>(acc, sourceTile, groups);
            }
            input_.FreeTensor(sourceTile);
            if constexpr (InputBuffers == 1) {
                // Tree reuses sourceTile as FP32 scratch until Adds finishes.
                // FreeTensor/AllocTensor retain the V-to-MTE2 queue dependency
                // before refilling that same slot. The next load can overlap
                // the current output cast and store, which use other buffers.
                if (unit + 1 < end) Load(unit + 1);
            }
            Store(first, groups * Width);
        }
        WaitOutput();
    }

  private:
    __aicore__ inline LocalTensor<float> Accumulator() {
        // The half tree has finished with the promoted input before Adds.
        // Reuse its prefix for the final sum; Store copies that prefix into
        // the separate output staging buffer before the next vector Cast.
        if constexpr (IsHalf)
            return cast_.Get<float>();
        else
            return acc_.Get<float>();
    }
    __aicore__ static inline uint64_t Small(uint64_t a, uint64_t b) { return a < b ? a : b; }
    template <uint32_t ActiveRows> __aicore__ static inline constexpr uint32_t Stripe() {
        uint32_t stripe = 1;
        while (stripe * 2 <= ActiveRows / 2 && stripe * 2 * Width <= kVectorLanes) stripe *= 2;
        return stripe;
    }
    template <uint32_t ActiveRows>
    __aicore__ inline LocalTensor<float> Tree(LocalTensor<float> current, LocalTensor<float> next, uint32_t groups) {
        if constexpr (ActiveRows == 1)
            return current;
        else {
            constexpr uint32_t stripe = Stripe<ActiveRows>();
            constexpr uint32_t span = stripe * Width;
            const uint32_t pairs = groups * (ActiveRows / 2) / stripe;
            const BinaryRepeatParams pair{1,
                                          1,
                                          1,
                                          static_cast<uint8_t>(span / 8),
                                          static_cast<uint8_t>(2 * span / 8),
                                          static_cast<uint8_t>(2 * span / 8)};
            for (uint32_t first = 0; first < pairs; first += kVectorRepeatLimit) {
                const uint8_t repeats = static_cast<uint8_t>(Small(kVectorRepeatLimit, pairs - first));
                Add(next[first * span], current[2 * first * span], current[2 * first * span + span], span, repeats,
                    pair);
            }
            PipeBarrier<PIPE_V>();
            return Tree<ActiveRows / 2>(next, current, groups);
        }
    }
    template <uint32_t Row>
    __aicore__ inline void Ordered(LocalTensor<float> acc, LocalTensor<T> source, uint32_t groups) {
        if constexpr (Row < Rows) {
            const BinaryRepeatParams ordered{1, 1, 1, Width / 8, Width / 8, Rows * Width / 8};
            Add(acc, acc, source[Row * Width], Width, static_cast<uint8_t>(groups), ordered);
            PipeBarrier<PIPE_V>();
            Ordered<Row + 1>(acc, source, groups);
        }
    }
    __aicore__ inline void Load(uint64_t unit) {
        auto input = input_.template AllocTensor<T>();
        const uint64_t first = unit * t_->outputGroup;
        const uint32_t groups = static_cast<uint32_t>(Small(t_->outputGroup, t_->totalTasks - first));
        DataCopy(input, src_[first * Rows * Width], groups * Rows * Width);
        input_.EnQue(input);
    }
    __aicore__ inline void WaitOutput() {
        if (outputPending_) {
            WaitFlag<HardEvent::MTE3_V>(mte3v_);
            outputPending_ = false;
        }
    }
    __aicore__ inline void Store(uint64_t first, uint32_t outputs) {
        WaitOutput();
        auto acc = Accumulator();
        if constexpr (IsHalf) {
            auto output = output_.Get<T>();
            Cast(output, acc, RoundMode::CAST_RINT, outputs);
            PipeBarrier<PIPE_V>();
            if constexpr (Reduce == SegmentMean && Rows > 1) {
                // Native CPU narrows the numerator before the half division.
                Muls(output, output, static_cast<half>(1.0f / Rows), outputs);
                PipeBarrier<PIPE_V>();
            }
            CopyOutput(first, output, outputs);
        } else {
            if constexpr (Reduce == SegmentMean && Rows > 1) {
                Muls(acc, acc, 1.0f / Rows, outputs);
                PipeBarrier<PIPE_V>();
            }
            CopyOutput(first, acc, outputs);
        }
    }
    __aicore__ inline void CopyOutput(uint64_t first, LocalTensor<T> output, uint32_t outputs) {
        SetFlag<HardEvent::V_MTE3>(vmte3_);
        WaitFlag<HardEvent::V_MTE3>(vmte3_);
        DataCopy(out_[first * Width], output, outputs);
        SetFlag<HardEvent::MTE3_V>(mte3v_);
        outputPending_ = true;
    }
    const SegmentCsrTilingData *t_;
    GlobalTensor<T> src_, out_;
    TPipe pipe_;
    TQue<TPosition::VECIN, InputBuffers> input_;
    TBuf<TPosition::VECCALC> acc_, cast_, output_;
    bool outputPending_ = false;
    event_t vmte3_, mte3v_;
};

template <typename T, SegmentReduce Reduce, uint32_t Rows, uint32_t Width>
__global__ __aicore__ void SegmentCsrShortUniformKernel(GM_ADDR source, GM_ADDR output, SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentCsrShortUniform<T, Reduce, Rows, Width> op;
    op.Run(source, output, t);
}

template <typename T, uint32_t Rows, uint32_t Width>
inline aclError LaunchCsrShortUniformGeometry(void *src, void *out, SegmentCsrTilingData t, aclrtStream stream) {
    // Public CANN Host-argument launch returns lookup/submission errors
    // directly. The parameter aggregate follows this kernel's explicit
    // pointer, pointer, by-value tiling signature, without hidden arguments.
    struct Arguments {
        void *source;
        void *output;
        SegmentCsrTilingData tiling;
    };
    static_assert(std::is_standard_layout<Arguments>::value, "short CSR argument layout");
    static_assert(std::is_trivially_copyable<Arguments>::value, "short CSR argument copyability");
    static_assert(alignof(Arguments) == 8 && sizeof(Arguments) == 296, "short CSR argument size");
    static_assert(offsetof(Arguments, source) == 0 && offsetof(Arguments, output) == 8 &&
                      offsetof(Arguments, tiling) == 16,
                  "short CSR argument offsets");
    Arguments arguments{src, out, t};
    const void *symbol =
        t.reduceCode == SegmentSum
            ? reinterpret_cast<const void *>(&SegmentCsrShortUniformKernel<T, SegmentSum, Rows, Width>)
            : reinterpret_cast<const void *>(&SegmentCsrShortUniformKernel<T, SegmentMean, Rows, Width>);
    aclrtFuncHandle function = nullptr;
    aclError status = aclrtGetFuncBySymbol(symbol, &function);
    if (status == ACL_SUCCESS)
        status = aclrtLaunchKernelWithHostArgs(function, t.usedBlockNum, stream, nullptr, &arguments, sizeof(arguments),
                                               nullptr, 0);
    // Failed public ACL calls also populate thread lastError. Consume that
    // duplicate only on failure, preserving the exact returned error for the
    // caller and any framework retry. Successful launches need no extra poll.
    if (status != ACL_SUCCESS) (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    return status;
}

template <typename T, uint32_t Width>
inline aclError LaunchCsrShortUniformWidth(void *src, void *out, SegmentCsrTilingData t, aclrtStream stream) {
    switch (t.tileRows) {
#define CSR_SHORT_ROWS(R)                                                                                              \
    case R:                                                                                                            \
        return LaunchCsrShortUniformGeometry<T, R, Width>(src, out, t, stream)
        CSR_SHORT_ROWS(1);
        CSR_SHORT_ROWS(2);
        CSR_SHORT_ROWS(4);
        CSR_SHORT_ROWS(8);
        CSR_SHORT_ROWS(16);
#undef CSR_SHORT_ROWS
    }
    return ACL_ERROR_INVALID_PARAM;
}

} // namespace opsgnn::csr

// ============================================================================
// Native FP16 uniform extrema
// ============================================================================

namespace opsgnn::csr {
using namespace AscendC;

// Only value extrema use native half arithmetic. FP32 opmath initialization
// narrows to +/-Inf in half, so nonempty values match CPU even for all-NaN or
// opposite-infinity segments. Arguments retain the separate FP32 strict path.
template <SegmentReduce Reduce> class SegmentCsrHalfExtreme {
  public:
    __aicore__ inline void Run(GM_ADDR source, GM_ADDR destination, const SegmentCsrTilingData &t) {
        Init(source, destination, t);
        const uint64_t begin = GetBlockIdx() * t.totalUnits / t.usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t.totalUnits / t.usedBlockNum;
        const uint64_t steps = (end - begin) * chunks_;
        if (!steps) return;
        Load(begin, 0);
        auto acc = acc_.Get<half>();
        for (uint64_t step = 0; step < steps; ++step) {
            const uint64_t unit = begin + step / chunks_;
            const uint64_t chunk = step % chunks_;
            const uint64_t first = unit * t.outputGroup;
            const uint32_t groups = static_cast<uint32_t>(Small(t.outputGroup, t.totalTasks - first));
            auto input = input_.template DeQue<half>();
            if (step + 1 < steps) Load(begin + (step + 1) / chunks_, (step + 1) % chunks_);
            if (!chunk) {
                Duplicate(acc, Initial(), groups * width_);
                PipeBarrier<PIPE_V>();
            }
            const uint32_t count = groups * t.tileRows * width_;
            auto mask = mask_.Get<uint8_t>();
            Compare(mask, input, input, CMPMODE::EQ,
                    (count + kHalfVectorLanes - 1) / kHalfVectorLanes * kHalfVectorLanes);
            PipeBarrier<PIPE_V>();
            Select(input, mask, input, Initial(), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
            PipeBarrier<PIPE_V>();
            auto value = Tree(input, t.tileRows, groups);
            if constexpr (Reduce == SegmentMin)
                Min(acc, acc, value, groups * width_);
            else
                Max(acc, acc, value, groups * width_);
            PipeBarrier<PIPE_V>();
            input_.FreeTensor(input);
            if (chunk + 1 == chunks_) {
                SetFlag<HardEvent::V_MTE3>(vmte3_);
                WaitFlag<HardEvent::V_MTE3>(vmte3_);
                DataCopy(out_[first * width_], acc, groups * width_);
                SetFlag<HardEvent::MTE3_V>(mte3v_);
                WaitFlag<HardEvent::MTE3_V>(mte3v_);
            }
        }
    }

  private:
    __aicore__ inline void Init(GM_ADDR source, GM_ADDR destination, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(source), t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(half))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(destination), t.outNumel);
        width_ = static_cast<uint32_t>(t.K);
        rows_ = t.M / t.nSegments;
        chunks_ = rows_ / t.tileRows;
        const uint32_t elements = t.outputGroup * t.tileRows * width_;
        // Compare reads a complete 128-half repeat; its padded lanes never
        // participate in a reduction or a GM store.
        const uint32_t capacity = (elements + kHalfVectorLanes - 1) / kHalfVectorLanes * kHalfVectorLanes;
        pipe_.InitBuffer(input_, 2, capacity * 2);
        pipe_.InitBuffer(tree_, capacity);
        pipe_.InitBuffer(mask_, (capacity / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes);
        pipe_.InitBuffer(acc_, t.outputGroup * width_ * 2);
        vmte3_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
        mte3v_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
    }
    __aicore__ static inline uint64_t Small(uint64_t a, uint64_t b) { return a < b ? a : b; }
    __aicore__ static inline half Initial() {
        return static_cast<half>(Reduce == SegmentMin ? __builtin_inff() : -__builtin_inff());
    }
    __aicore__ inline void Load(uint64_t unit, uint64_t chunk) {
        auto input = input_.template AllocTensor<half>();
        const uint64_t first = unit * t_->outputGroup;
        const uint32_t groups = static_cast<uint32_t>(Small(t_->outputGroup, t_->totalTasks - first));
        const uint64_t base = chunk * t_->tileRows;
        const uint64_t offset = (first * rows_ + base) * width_;
        if (t_->tileRows == rows_)
            DataCopy(input, src_[offset], groups * t_->tileRows * width_);
        else {
            const uint64_t gap = (rows_ - t_->tileRows) * width_ * 2;
            if (gap <= UINT32_MAX)
                DataCopyPad(
                    input, src_[offset],
                    {static_cast<uint16_t>(groups), t_->tileRows * width_ * 2, static_cast<uint32_t>(gap), 0, 0},
                    DataCopyPadExtParams<half>(false, 0, 0, 0));
            else
                for (uint32_t group = 0; group < groups; ++group)
                    DataCopy(input[group * t_->tileRows * width_], src_[offset + group * rows_ * width_],
                             t_->tileRows * width_);
        }
        input_.EnQue(input);
    }
    __aicore__ inline LocalTensor<half> Tree(LocalTensor<half> input, uint32_t rows, uint32_t groups) {
        auto current = input;
        auto next = tree_.Get<half>();
        bool scratch = false;
        while (rows > 1) {
            uint32_t stripe = 1;
            while (stripe * 2 <= rows / 2 && stripe * 2 * width_ <= kHalfVectorLanes) stripe *= 2;
            const uint32_t span = stripe * width_;
            const uint32_t pairs = groups * (rows / 2) / stripe;
            const BinaryRepeatParams pair{1,
                                          1,
                                          1,
                                          static_cast<uint8_t>(span / 16),
                                          static_cast<uint8_t>(2 * span / 16),
                                          static_cast<uint8_t>(2 * span / 16)};
            for (uint32_t first = 0; first < pairs; first += kVectorRepeatLimit) {
                const uint8_t repeats = static_cast<uint8_t>(Small(kVectorRepeatLimit, pairs - first));
                for (uint32_t c = 0; c < span; c += kHalfVectorLanes) {
                    if constexpr (Reduce == SegmentMin)
                        Min(next[first * span + c], current[2 * first * span + c], current[2 * first * span + span + c],
                            Small(kHalfVectorLanes, span - c), repeats, pair);
                    else
                        Max(next[first * span + c], current[2 * first * span + c], current[2 * first * span + span + c],
                            Small(kHalfVectorLanes, span - c), repeats, pair);
                }
            }
            PipeBarrier<PIPE_V>();
            current = next;
            scratch = !scratch;
            next = scratch ? input : tree_.Get<half>();
            rows /= 2;
        }
        return current;
    }
    const SegmentCsrTilingData *t_;
    GlobalTensor<half> src_, out_;
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> input_;
    TBuf<TPosition::VECCALC> tree_, mask_, acc_;
    uint32_t width_;
    uint64_t rows_, chunks_;
    event_t vmte3_, mte3v_;
};

template <SegmentReduce Reduce>
__global__ __aicore__ void SegmentCsrHalfExtremeKernel(GM_ADDR src, GM_ADDR out, SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentCsrHalfExtreme<Reduce> op;
    op.Run(src, out, t);
}

} // namespace opsgnn::csr

// ============================================================================
// Launch the immutable Host plan
// ============================================================================

namespace {
using namespace opsgnn::csr;

template <typename T, bool WholeCast = false, bool Bounded = false>
void LaunchOrderedCsr(void *src, void *out, const SegmentCsrTilingData &t, aclrtStream stream) {
    if (t.reduceCode == SegmentSum)
        SegmentCsrOrderedUniformKernel<T, SegmentSum, WholeCast, Bounded>
            <<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)out, t);
    else
        SegmentCsrOrderedUniformKernel<T, SegmentMean, WholeCast, Bounded>
            <<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)out, t);
}

template <typename T> bool LaunchHalfCsr(void *src, void *out, const SegmentCsrPlan &plan, aclrtStream stream) {
    const auto &t = plan.tiling;
    if constexpr (std::is_same<T, half>::value) {
        if (plan.path == SegmentCsrKernelPath::OrderedWholeCast) {
            LaunchOrderedCsr<T, true>(src, out, t, stream);
            return true;
        }
        if (plan.path == SegmentCsrKernelPath::OrderedBoundedTree) {
            LaunchOrderedCsr<T, true, true>(src, out, t, stream);
            return true;
        }
        if (plan.path == SegmentCsrKernelPath::HalfExtreme) {
            if (t.reduceCode == SegmentMin)
                SegmentCsrHalfExtremeKernel<SegmentMin>
                    <<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)out, t);
            else
                SegmentCsrHalfExtremeKernel<SegmentMax>
                    <<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)out, t);
            return true;
        }
    }
    return false;
}

#define CSR_LAUNCH_REDUCE(Kernel)                                                                                      \
    do {                                                                                                               \
        switch (t.reduceCode) {                                                                                        \
        case SegmentSum:                                                                                               \
            Kernel<SegmentSum><<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)out, t);                    \
            break;                                                                                                     \
        case SegmentMean:                                                                                              \
            Kernel<SegmentMean><<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)out, t);                   \
            break;                                                                                                     \
        case SegmentMin:                                                                                               \
            Kernel<SegmentMin><<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)out, t);                    \
            break;                                                                                                     \
        case SegmentMax:                                                                                               \
            Kernel<SegmentMax><<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)out, t);                    \
            break;                                                                                                     \
        }                                                                                                              \
    } while (false)

template <typename T>
aclError LaunchIntegerCsr(void *src, int64_t *index, void *out, int64_t *arg, const SegmentCsrPlan &plan,
                          aclrtStream stream) {
    const auto &t = plan.tiling;
    if constexpr (std::is_same<T, int32_t>::value) {
        switch (plan.path) {
        case SegmentCsrKernelPath::Int32Pipeline:
            CSR_LAUNCH_REDUCE(SegmentInt32PipelineKernel);
            break;
        case SegmentCsrKernelPath::Int32Long:
            CSR_LAUNCH_REDUCE(SegmentInt32LongKernel);
            break;
        case SegmentCsrKernelPath::Int32Batch:
            CSR_LAUNCH_REDUCE(SegmentInt32BatchKernel);
            break;
        default:
            return SegmentCsrGeneral(src, index, out, arg, plan, stream);
        }
        return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    }
    if constexpr (std::is_same<T, int64_t>::value) {
        switch (plan.path) {
        case SegmentCsrKernelPath::Int64Pipeline:
            CSR_LAUNCH_REDUCE(SegmentInt64NarrowPipelineKernel);
            break;
        case SegmentCsrKernelPath::Int64Long:
            CSR_LAUNCH_REDUCE(SegmentInt64NarrowLongKernel);
            break;
        case SegmentCsrKernelPath::Int64Narrow:
            CSR_LAUNCH_REDUCE(SegmentInt64ProvenKernel);
            break;
        default:
            return SegmentCsrGeneral(src, index, out, arg, plan, stream);
        }
        return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    }
    return SegmentCsrGeneral(src, index, out, arg, plan, stream);
}
#undef CSR_LAUNCH_REDUCE
template <typename T>
aclError LaunchPlannedCsr(void *src, int64_t *index, void *out, int64_t *arg, const SegmentCsrPlan &plan,
                          aclrtStream stream) {
    const auto &t = plan.tiling;
    if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value) {
        if (plan.path == SegmentCsrKernelPath::ShortUniform) {
            switch (t.K) {
            case kShortUniformColumns:
                return LaunchCsrShortUniformWidth<T, kShortUniformColumns>(src, out, t, stream);
            case kMediumUniformColumns:
                return LaunchCsrShortUniformWidth<T, kMediumUniformColumns>(src, out, t, stream);
            case kWideUniformColumns:
                return LaunchCsrShortUniformWidth<T, kWideUniformColumns>(src, out, t, stream);
            default:
                return ACL_ERROR_INVALID_PARAM;
            }
        }
    }
    if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        if (plan.path == SegmentCsrKernelPath::OrderedUniform) {
            LaunchOrderedCsr<T>(src, out, t, stream);
            return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
        }
    }
    if (LaunchHalfCsr<T>(src, out, plan, stream)) return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    return LaunchIntegerCsr<T>(src, index, out, arg, plan, stream);
}

} // namespace

aclError SegmentCsr(void *src, int64_t *index, void *out, int64_t *arg, const SegmentCsrPlan &plan,
                    aclrtStream stream) {
    if (!plan.tiling.usedBlockNum) return ACL_SUCCESS;
    switch (plan.tiling.dtypeCode) {
    case SEGMENT_CSR_FLOAT32:
        return LaunchPlannedCsr<float>(src, index, out, arg, plan, stream);
    case SEGMENT_CSR_FLOAT16:
        return LaunchPlannedCsr<half>(src, index, out, arg, plan, stream);
    case SEGMENT_CSR_BFLOAT16:
        return LaunchPlannedCsr<bfloat16_t>(src, index, out, arg, plan, stream);
    case SEGMENT_CSR_INT8:
        return LaunchPlannedCsr<int8_t>(src, index, out, arg, plan, stream);
    case SEGMENT_CSR_UINT8:
        return LaunchPlannedCsr<uint8_t>(src, index, out, arg, plan, stream);
    case SEGMENT_CSR_INT32:
        return LaunchPlannedCsr<int32_t>(src, index, out, arg, plan, stream);
    case SEGMENT_CSR_INT64:
        return LaunchPlannedCsr<int64_t>(src, index, out, arg, plan, stream);
    default:
        return ACL_ERROR_INVALID_PARAM;
    }
}

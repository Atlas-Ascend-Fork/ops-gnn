/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "segment_coo.h"
#include "kernel_operator.h"
#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "segment_coo_reduce.h"

namespace opsgnn {

// ============================================================================
// Exact integer device reductions
// ============================================================================

using namespace AscendC;

struct UnitBounds {
    uint64_t first, last, columnBegin, columnEnd;
};
__aicore__ inline UnitBounds DescribeUnit(const SegmentTilingData &t, uint64_t unit) {
    const uint64_t first = t.outputGroup == 1 ? unit / t.featureTiles : unit * t.outputGroup;
    const uint64_t last = first + t.outputGroup < t.totalTasks ? first + t.outputGroup : t.totalTasks;
    const uint64_t begin = t.outputGroup == 1 ? unit % t.featureTiles * t.tileK : 0;
    const uint64_t end = t.outputGroup == 1 && begin + t.tileK < t.K ? begin + t.tileK : t.K;
    return {first, last, begin, end};
}

// Byte reductions and value+arg int64 use scalar integer arithmetic on AIV.
// DMA stages complete padded rows; ownership groups keep partial 32-byte output
// blocks on one core. No source integer is converted to floating point.
template <typename T, SegmentReduce Reduce, bool HasArg> class SegmentIntegerScalar {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, const SegmentTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), t.srcNumel);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), t.outNumel);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), t.indexNumel);
        if constexpr (HasArg)
            arg_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(arg), t.outNumel);
        pipe_.InitBuffer(input_, t.tileRows * t.tileK * sizeof(T));
        pipe_.InitBuffer(output_, t.tileK * sizeof(T));
        pipe_.InitBuffer(sums_, t.tileK * sizeof(uint64_t));
        pipe_.InitBuffer(args_, t.tileK * sizeof(int64_t));
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        for (uint64_t unit = begin; unit < end; ++unit) {
            const uint64_t first = t_->outputGroup == 1 ? unit / t_->featureTiles : unit * t_->outputGroup;
            const uint64_t last = Min(first + t_->outputGroup, t_->totalTasks);
            const uint64_t colBegin = t_->outputGroup == 1 ? unit % t_->featureTiles * t_->tileK : 0;
            const uint64_t colEnd = t_->outputGroup == 1 ? Min(colBegin + t_->tileK, t_->K) : t_->K;
            for (uint64_t task = first; task < last; ++task)
                for (uint64_t col = colBegin; col < colEnd; col += t_->tileK)
                    Tile(task, col);
        }
    }

  private:
    __aicore__ static inline uint64_t Min(uint64_t a, uint64_t b) {
        return a < b ? a : b;
    }

    __aicore__ inline void AccumulateRows(LocalTensor<T> input, uint64_t start, uint64_t first, uint32_t count,
                                          uint32_t valid, uint32_t width) {
        auto sums = sums_.Get<uint64_t>();
        auto args = args_.Get<int64_t>();
        for (uint32_t r = 0; r < count; ++r) {
            for (uint32_t c = 0; c < valid; ++c) {
                const int64_t next = static_cast<int64_t>(input.GetValue(r * width + c));
                const int64_t old = static_cast<int64_t>(sums.GetValue(c));
                if constexpr (Reduce == SegmentSum || Reduce == SegmentMean) {
                    // Unsigned arithmetic defines two's-complement modulo wrap.
                    sums.SetValue(c, sums.GetValue(c) + static_cast<uint64_t>(next));
                } else if (Reduce == SegmentMin ? next < old : next > old) {
                    sums.SetValue(c, static_cast<uint64_t>(next));
                    args.SetValue(c, start + r);
                }
            }
        }
    }
    __aicore__ inline int64_t MeanValue(int64_t value, uint64_t rows) {
        if constexpr (Reduce == SegmentMean) {
            if (rows == 0)
                return static_cast<T>(value);
            const int64_t denominator = static_cast<T>(rows);
            if (denominator != 0)
                return static_cast<int64_t>(static_cast<T>(value)) / denominator;
            // Explicit platform-compatible result; never execute integer division by zero.
            return 0;
        }
        return value;
    }
    __aicore__ static inline int64_t InitialValue() {
        if constexpr (Reduce == SegmentMin)
            return sizeof(T) == 8 ? INT64_MAX : (std::is_same<T, uint8_t>::value ? 255 : 127);
        if constexpr (Reduce == SegmentMax)
            return sizeof(T) == 8 ? INT64_MIN : (std::is_same<T, uint8_t>::value ? 0 : -128);
        return 0;
    }
    __aicore__ inline void Tile(uint64_t task, uint64_t col) {
        const uint64_t batch = task / t_->nSegments, seg = task % t_->nSegments;
        uint64_t first, last;
        FindSegmentBounds(index_, *t_, batch, seg, first, last);
        const uint32_t valid = static_cast<uint32_t>(Min(t_->tileK, t_->K - col));
        const uint32_t align = 32 / sizeof(T), width = (valid + align - 1) / align * align;
        const uint64_t destination = task * t_->K + col;
        auto input = input_.Get<T>();
        auto output = output_.Get<T>();
        auto sums = sums_.Get<uint64_t>();
        auto args = args_.Get<int64_t>();
        if (t_->hasInitial) {
            DataCopyPad(output, out_[last > first && first == 0 ? batch * t_->nSegments * t_->K + col : destination],
                        {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
                        DataCopyPadExtParams<T>(true, 0, static_cast<uint8_t>(width - valid), 0));
            SyncEvent<HardEvent::MTE2_S>();
        }
        for (uint32_t c = 0; c < valid; ++c) {
            sums.SetValue(c, static_cast<uint64_t>(t_->hasInitial ? static_cast<int64_t>(output.GetValue(c))
                                                               : InitialValue()));
            args.SetValue(c, t_->M);
        }
        for (uint64_t start = first; start < last; start += t_->tileRows) {
            const uint32_t count = static_cast<uint32_t>(Min(t_->tileRows, last - start));
            const uint64_t sourceOffset = (batch * t_->M + start) * t_->K + col;
            LoadPaddedRows(input, src_, sourceOffset, count, valid, width, t_->K);
            SyncEvent<HardEvent::MTE2_S>();
            AccumulateRows(input, start, first, count, valid, width);
            SyncEvent<HardEvent::S_MTE2>();
        }
        for (uint32_t c = 0; c < valid; ++c) {
            int64_t value = static_cast<int64_t>(sums.GetValue(c));
            value = MeanValue(value, last - first);
            if constexpr (Reduce == SegmentMin || Reduce == SegmentMax)
                if (!t_->hasInitial && value == InitialValue())
                    value = 0;
            output.SetValue(c, static_cast<T>(value));
        }
        SyncEvent<HardEvent::S_MTE3>();
        DataCopyPad(out_[destination], output, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        if constexpr (HasArg)
            DataCopyPad(arg_[destination], args, {1, valid * 8, 0, 0, 0});
        SyncEvent<HardEvent::MTE3_S>();
    }
    const SegmentTilingData *t_;
    GlobalTensor<T> src_, out_;
    GlobalTensor<int64_t> index_, arg_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_, output_, sums_, args_;
};

// Full-range int64 sum/mean. Four base-2^16 digits are exactly representable
// in FP32; at most 128 rows give digit sums < 2^23, so every operation here is
// an exact integer operation. No whole int64 is ever numerically cast to FP32.
// Digit sums are reassembled with 32-bit integer carries before tile merges.
template <SegmentReduce Reduce> class SegmentInt64Vector {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, const SegmentTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(src), t.srcNumel);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(out), t.outNumel);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), t.indexNumel);
        capacity_ = t.tileK * t.tileRows;
        // Every scratch tensor has a spare 32-byte zero block used by Gather.
        outputCapacity_ = t.tileK * t.outputGroup;
        const uint32_t words = 2 * outputCapacity_ + 8;
        pipe_.InitBuffer(input_, capacity_ * 8);
        pipe_.InitBuffer(digits_, capacity_ * 16);
        pipe_.InitBuffer(tree_, capacity_ * 8);
        pipe_.InitBuffer(a_, (2 * capacity_ + 8) * 4);
        pipe_.InitBuffer(b_, words * 4);
        pipe_.InitBuffer(c_, words * 4);
        pipe_.InitBuffer(offsets_, words * 12);
        pipe_.InitBuffer(acc_, t.tileK * 8);
        pipe_.InitBuffer(mask_, ((4 * capacity_ + 7) / 8 + 31) / 32 * 32);
        auto offsets = offsets_.Get<uint32_t>();
        for (uint32_t i = 0; i < 2 * outputCapacity_; ++i) {
            offsets.SetValue(i, i * 8);
            offsets.SetValue(words + i, (i % 2 ? i - 1 : 2 * outputCapacity_) * 4);
            offsets.SetValue(words * 2 + i, (i % 2 ? 2 * outputCapacity_ : i + 1) * 4);
        }
        SyncEvent<HardEvent::S_V>();
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        for (uint64_t unit = begin; unit < end; ++unit) {
            const auto bounds = DescribeUnit(*t_, unit);
            for (uint64_t col = bounds.columnBegin; col < bounds.columnEnd; col += t_->tileK) {
                const uint64_t rows = t_->indexMode == kUniformIndex ? t_->M / t_->nSegments : 0;
                const uint32_t width = static_cast<uint32_t>(Min(t_->tileK, t_->K - col));
                if (!t_->hasInitial && rows && rows <= t_->tileRows && (rows & (rows - 1)) == 0 && width % 8 == 0 &&
                    (bounds.last - bounds.first) * rows <= t_->tileRows &&
                    bounds.first / t_->nSegments == (bounds.last - 1) / t_->nSegments) {
                    Batch(bounds.first, bounds.last - bounds.first, col, rows, width);
                } else {
                    for (uint64_t task = bounds.first; task < bounds.last; ++task)
                        Tile(task, col);
                }
            }
        }
    }

  private:
    __aicore__ static inline uint64_t Min(uint64_t a, uint64_t b) {
        return a < b ? a : b;
    }

    __aicore__ inline void Load(uint64_t offset, uint32_t rows, uint32_t valid, uint32_t width) {
        if (width == t_->K && valid == width)
            DataCopy(input_.Get<int64_t>(), src_[offset], rows * width);
        else
            LoadPaddedRows(input_.Get<int64_t>(), src_, offset, rows, valid, width, t_->K);
        SyncEvent<HardEvent::MTE2_V>();
    }
    __aicore__ inline LocalTensor<float> Digits(uint32_t elements) {
        auto work = digits_.Get<float>();
        auto adjusted = a_.Get<float>();
        auto mask = mask_.Get<uint8_t>();
        // Cast through signed 16-bit lanes is exact. Correct their sign to
        // unsigned digits, including the most significant digit; packing later
        // reconstructs the original two's-complement bit pattern.
        Cast(work, input_.Get<int16_t>(), RoundMode::CAST_NONE, elements * 4);
        PipeBarrier<PIPE_V>();
        for (uint32_t start = 0; start < elements * 4; start += capacity_ * 2) {
            const uint32_t count = static_cast<uint32_t>(Min(capacity_ * 2, elements * 4 - start));
            Adds(adjusted, work[start], 65536.0f, count);
            CompareScalar(mask, work[start], 0.0f, CMPMODE::LT, count);
            PipeBarrier<PIPE_V>();
            Select(work[start], mask, adjusted, work[start], SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            PipeBarrier<PIPE_V>();
        }
        return work;
    }
    __aicore__ inline LocalTensor<float> Tree(LocalTensor<float> work, uint32_t rows, uint32_t groups, uint32_t width) {
        auto next = tree_.Get<float>();
        bool scratch = false;
        const uint32_t lanes = width * 4;
        while (rows > 1) {
            const BinaryRepeatParams pair{1,
                                          1,
                                          1,
                                          static_cast<uint8_t>(lanes / 8),
                                          static_cast<uint8_t>(lanes / 4),
                                          static_cast<uint8_t>(lanes / 4)};
            const uint32_t pairs = groups * rows / 2;
            for (uint32_t start = 0; start < pairs; start += 255)
                for (uint32_t col = 0; col < lanes; col += 64)
                    Add(next[start * lanes + col], work[start * 2 * lanes + col], work[start * 2 * lanes + lanes + col],
                        Min(64, lanes - col), static_cast<uint8_t>(Min(255, pairs - start)), pair);
            PipeBarrier<PIPE_V>();
            work = next;
            scratch = !scratch;
            next = scratch ? digits_.Get<float>() : tree_.Get<float>();
            rows /= 2;
        }
        return work;
    }
    __aicore__ inline LocalTensor<int32_t> Pack(LocalTensor<float> values, uint32_t elements) {
        const uint32_t words = elements * 2;
        auto integral = values.ReinterpretCast<int32_t>();
        Cast(integral, values, RoundMode::CAST_RINT, elements * 4);
        PipeBarrier<PIPE_V>();
        auto lower = a_.Get<uint32_t>(), upper = b_.Get<uint32_t>();
        auto carry = c_.Get<uint32_t>(), offsets = offsets_.Get<uint32_t>();
        Gather(lower, integral.ReinterpretCast<uint32_t>(), offsets, 0, words);
        Gather(upper, integral.ReinterpretCast<uint32_t>(), offsets, 4, words);
        PipeBarrier<PIPE_V>();
        ShiftRight(carry, lower, static_cast<uint32_t>(16), words);
        PipeBarrier<PIPE_V>();
        Add(upper.ReinterpretCast<int32_t>(), upper.ReinterpretCast<int32_t>(), carry.ReinterpretCast<int32_t>(),
            words);
        PipeBarrier<PIPE_V>();
        ShiftRight(carry, upper, static_cast<uint32_t>(16), words);
        ShiftLeft(upper, upper, static_cast<uint32_t>(16), words);
        ShiftLeft(lower, lower, static_cast<uint32_t>(16), words);
        PipeBarrier<PIPE_V>();
        ShiftRight(lower, lower, static_cast<uint32_t>(16), words);
        PipeBarrier<PIPE_V>();
        Or(lower.ReinterpretCast<uint16_t>(), lower.ReinterpretCast<uint16_t>(), upper.ReinterpretCast<uint16_t>(),
           words * 2);
        PipeBarrier<PIPE_V>();
        // Transfer each low-word overflow into its own high word only.
        SyncEvent<HardEvent::V_S>();
        carry.SetValue(2 * outputCapacity_, 0);
        SyncEvent<HardEvent::S_V>();
        Gather(upper, carry, offsets[2 * outputCapacity_ + 8], 0, words);
        PipeBarrier<PIPE_V>();
        Add(lower.ReinterpretCast<int32_t>(), lower.ReinterpretCast<int32_t>(), upper.ReinterpretCast<int32_t>(),
            words);
        PipeBarrier<PIPE_V>();
        return lower.ReinterpretCast<int32_t>();
    }
    __aicore__ inline void Merge(LocalTensor<int32_t> value, uint32_t words) {
        auto acc = acc_.Get<int32_t>();
        auto both = b_.Get<uint16_t>(), either = c_.Get<uint16_t>();
        auto sum = input_.Get<int32_t>();
        Add(sum, acc, value, words);
        And(both, acc.ReinterpretCast<uint16_t>(), value.ReinterpretCast<uint16_t>(), words * 2);
        Or(either, acc.ReinterpretCast<uint16_t>(), value.ReinterpretCast<uint16_t>(), words * 2);
        PipeBarrier<PIPE_V>();
        Not(value.ReinterpretCast<uint16_t>(), sum.ReinterpretCast<uint16_t>(), words * 2);
        PipeBarrier<PIPE_V>();
        And(either, either, value.ReinterpretCast<uint16_t>(), words * 2);
        PipeBarrier<PIPE_V>();
        Or(both, both, either, words * 2);
        PipeBarrier<PIPE_V>();
        ShiftRight(both.ReinterpretCast<uint32_t>(), both.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(31),
                   words);
        PipeBarrier<PIPE_V>();
        auto offsets = offsets_.Get<uint32_t>();
        SyncEvent<HardEvent::V_S>();
        both.ReinterpretCast<uint32_t>().SetValue(2 * outputCapacity_, 0);
        SyncEvent<HardEvent::S_V>();
        Gather(either.ReinterpretCast<uint32_t>(), both.ReinterpretCast<uint32_t>(), offsets[2 * outputCapacity_ + 8],
               0, words);
        PipeBarrier<PIPE_V>();
        Add(acc, sum, either.ReinterpretCast<int32_t>(), words);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void AddMeanBias(LocalTensor<int64_t> value64, uint32_t elements, uint32_t shift) {
        const uint32_t words = elements * 2;
        auto value = value64.ReinterpretCast<int32_t>();
        auto sum = b_.Get<int32_t>(), temp = c_.Get<int32_t>();
        auto bits = input_.Get<uint32_t>();
        auto offsets = offsets_.Get<uint32_t>();
        const uint32_t offsetWidth = 2 * outputCapacity_ + 8;
        // For a negative numerator, add (2^shift-1) exactly before the
        // arithmetic right shift: C++ truncation, not floor rounding.
        ShiftRight(sum, value, static_cast<int32_t>(31), words);
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_S>();
        sum.SetValue(2 * outputCapacity_, 0);
        SyncEvent<HardEvent::S_V>();
        Gather(temp.ReinterpretCast<uint32_t>(), sum.ReinterpretCast<uint32_t>(), offsets[2 * offsetWidth], 0, words);
        PipeBarrier<PIPE_V>();
        ShiftRight(temp.ReinterpretCast<uint32_t>(), temp.ReinterpretCast<uint32_t>(),
                   static_cast<uint32_t>(32 - shift), words);
        PipeBarrier<PIPE_V>();
        Add(sum, value, temp, words);
        And(bits.ReinterpretCast<uint16_t>(), value.ReinterpretCast<uint16_t>(), temp.ReinterpretCast<uint16_t>(),
            words * 2);
        Or(temp.ReinterpretCast<uint16_t>(), value.ReinterpretCast<uint16_t>(), temp.ReinterpretCast<uint16_t>(),
           words * 2);
        PipeBarrier<PIPE_V>();
        Not(value.ReinterpretCast<uint16_t>(), sum.ReinterpretCast<uint16_t>(), words * 2);
        PipeBarrier<PIPE_V>();
        And(temp.ReinterpretCast<uint16_t>(), temp.ReinterpretCast<uint16_t>(), value.ReinterpretCast<uint16_t>(),
            words * 2);
        PipeBarrier<PIPE_V>();
        Or(temp.ReinterpretCast<uint16_t>(), temp.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(),
           words * 2);
        PipeBarrier<PIPE_V>();
        ShiftRight(temp.ReinterpretCast<uint32_t>(), temp.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(31),
                   words);
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_S>();
        temp.SetValue(2 * outputCapacity_, 0);
        SyncEvent<HardEvent::S_V>();
        Gather(bits, temp.ReinterpretCast<uint32_t>(), offsets[offsetWidth], 0, words);
        PipeBarrier<PIPE_V>();
        Add(value, sum, bits.ReinterpretCast<int32_t>(), words);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ShiftMean(LocalTensor<int64_t> value64, uint32_t elements, uint32_t shift) {
        const uint32_t words = elements * 2;
        auto value = value64.ReinterpretCast<int32_t>();
        auto sum = b_.Get<int32_t>(), temp = c_.Get<int32_t>();
        auto bits = input_.Get<uint32_t>();
        auto offsets = offsets_.Get<uint32_t>();
        const uint32_t offsetWidth = 2 * outputCapacity_ + 8;
        ShiftRight(sum.ReinterpretCast<uint32_t>(), value.ReinterpretCast<uint32_t>(), shift, words);
        ShiftLeft(temp.ReinterpretCast<uint32_t>(), value.ReinterpretCast<uint32_t>(),
                  static_cast<uint32_t>(32 - shift), words);
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_S>();
        temp.SetValue(2 * outputCapacity_, 0);
        SyncEvent<HardEvent::S_V>();
        Gather(bits, temp.ReinterpretCast<uint32_t>(), offsets[2 * offsetWidth], 0, words);
        PipeBarrier<PIPE_V>();
        Or(sum.ReinterpretCast<uint16_t>(), sum.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(),
           words * 2);
        ShiftRight(temp, value, static_cast<int32_t>(shift), words);
        Duplicate(mask_.Get<uint16_t>(), static_cast<uint16_t>(0xAAAA), (words + 15) / 16);
        PipeBarrier<PIPE_V>();
        // Low word uses logical shift + next-word bits; high word uses
        // arithmetic shift. Selection reinterprets bits, never values.
        Select(value.ReinterpretCast<float>(), mask_.Get<uint8_t>(), temp.ReinterpretCast<float>(),
               sum.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, words);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Mean(LocalTensor<int64_t> value64, uint32_t elements, uint64_t rows) {
        if constexpr (Reduce == SegmentMean) {
            if (rows > 1 && rows <= (1ULL << 31) && (rows & (rows - 1)) == 0) {
                uint32_t shift = 0;
                for (uint64_t n = rows; n > 1; n /= 2)
                    ++shift;
                AddMeanBias(value64, elements, shift);
                ShiftMean(value64, elements, shift);
            } else if (rows != 1) {
                SyncEvent<HardEvent::V_S>();
                for (uint32_t c = 0; c < elements; ++c)
                    value64.SetValue(c, value64.GetValue(c) / static_cast<int64_t>(rows ? rows : 1));
                SyncEvent<HardEvent::S_V>();
            }
        }
    }
    __aicore__ inline void Store(LocalTensor<int64_t> value, uint64_t output, uint32_t groups, uint32_t valid,
                                 uint32_t width) {
        SyncEvent<HardEvent::V_MTE3>();
        StorePaddedRows(out_, output, value, groups, valid, width, t_->K);
        SyncEvent<HardEvent::MTE3_V>();
        SyncEvent<HardEvent::V_MTE2>();
    }
    __aicore__ inline void Batch(uint64_t task, uint32_t groups, uint64_t col, uint32_t rows, uint32_t width) {
        Load(task * rows * t_->K + col, rows * groups, width, width);
        auto result = Pack(Tree(Digits(rows * groups * width), rows, groups, width), groups * width);
        Mean(result.template ReinterpretCast<int64_t>(), groups * width, rows);
        Store(result.template ReinterpretCast<int64_t>(), task * t_->K + col, groups, width, width);
    }
    __aicore__ inline void Tile(uint64_t task, uint64_t col) {
        const uint64_t batch = task / t_->nSegments, seg = task % t_->nSegments;
        uint64_t first, last;
        FindSegmentBounds(index_, *t_, batch, seg, first, last);
        const uint32_t valid = static_cast<uint32_t>(Min(t_->tileK, t_->K - col));
        // Full 64-lane mask for exact negative-digit correction on arch22.
        const uint32_t width = (valid + 15) / 16 * 16;
        auto acc = acc_.Get<int32_t>();
        Duplicate(acc, 0, width * 2);
        PipeBarrier<PIPE_V>();
        if (t_->hasInitial) {
            DataCopyPad(acc.ReinterpretCast<int64_t>(),
                        out_[last > first && first == 0 ? batch * t_->nSegments * t_->K + col : task * t_->K + col],
                        {1, valid * 8, 0, 0, 0},
                        DataCopyPadExtParams<int64_t>(true, 0, static_cast<uint8_t>((valid + 3) / 4 * 4 - valid), 0));
            SyncEvent<HardEvent::MTE2_V>();
        }
        for (uint64_t start = first; start < last;) {
            uint32_t rows = t_->tileRows;
            while (rows > last - start)
                rows /= 2;
            Load((batch * t_->M + start) * t_->K + col, rows, valid, width);
            auto work = Digits(rows * width);
            auto result = Pack(Tree(work, rows, 1, width), width);
            Merge(result, width * 2);
            SyncEvent<HardEvent::V_MTE2>();
            start += rows;
        }
        Mean(acc.ReinterpretCast<int64_t>(), valid, last - first);
        Store(acc.ReinterpretCast<int64_t>(), task * t_->K + col, 1, valid, width);
    }
    const SegmentTilingData *t_;
    uint32_t capacity_, outputCapacity_;
    GlobalTensor<int64_t> src_, out_, index_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_, digits_, tree_, a_, b_, c_, offsets_, acc_, mask_;
};

using namespace AscendC;
// Lexicographic signed-high/unsigned-low 32-bit comparisons preserve every
// int64 bit, including adjacent integers above 2^53 and both signed limits.
template <SegmentReduce Reduce> class SegmentInt64Extreme {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, const SegmentTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(src), t.srcNumel);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(out), t.outNumel);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), t.indexNumel);
        const uint32_t capacity = t.tileRows * t.tileK;
        pipe_.InitBuffer(input_, capacity * 8);
        pipe_.InitBuffer(tree_, capacity * 4);
        pipe_.InitBuffer(candidate_, capacity * 4);
        pipe_.InitBuffer(acc_, t.tileK * 8);
        pipe_.InitBuffer(bias_, t.tileK * 8);
        pipe_.InitBuffer(winner_, 2048);
        pipe_.InitBuffer(equal_, 2048);
        pipe_.InitBuffer(scratchMask_, 2048);
        pipe_.InitBuffer(even_, 2048);
        auto bias = bias_.Get<int32_t>();
        for (uint32_t i = 0; i < t.tileK * 2; ++i)
            bias.SetValue(i, i % 2 ? 0 : INT32_MIN);
        SyncEvent<HardEvent::S_V>();
        Duplicate(even_.Get<uint16_t>(), static_cast<uint16_t>(0x5555), 1024);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        for (uint64_t unit = begin; unit < end; ++unit) {
            const auto bounds = DescribeUnit(*t_, unit);
            for (uint64_t col = bounds.columnBegin; col < bounds.columnEnd; col += t_->tileK) {
                const uint64_t rows = t_->indexMode == kUniformIndex ? t_->M / t_->nSegments : 0;
                const uint32_t width = static_cast<uint32_t>(Min(t_->tileK, t_->K - col));
                if (!t_->hasInitial && rows >= 2 && rows <= t_->tileRows && (rows & (rows - 1)) == 0 &&
                    width % 32 == 0 && (bounds.last - bounds.first) * rows <= t_->tileRows &&
                    bounds.first / t_->nSegments == (bounds.last - 1) / t_->nSegments) {
                    Batch(bounds.first, bounds.last - bounds.first, col, rows, width);
                } else
                    for (uint64_t task = bounds.first; task < bounds.last; ++task)
                        Tile(task, col);
            }
        }
    }

  private:
    __aicore__ static inline uint64_t Min(uint64_t a, uint64_t b) {
        return a < b ? a : b;
    }

    __aicore__ inline void Bias(LocalTensor<int32_t> value, uint32_t rows, uint32_t width) {
        const uint32_t words = width * 2;
        const BinaryRepeatParams params{1, 1, 1, static_cast<uint8_t>(words / 8), static_cast<uint8_t>(words / 8), 0};
        for (uint32_t start = 0; start < rows; start += 255)
            for (uint32_t col = 0; col < words; col += 64)
                Add(value[start * words + col], value[start * words + col], bias_.Get<int32_t>()[col], 64,
                    static_cast<uint8_t>(Min(255, rows - start)), params);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline LocalTensor<int32_t> Load(uint64_t offset, uint32_t rows, uint32_t valid, uint32_t width) {
        if (width == t_->K && valid == width)
            DataCopy(input_.Get<int64_t>(), src_[offset], rows * width);
        else
            LoadPaddedRows(input_.Get<int64_t>(), src_, offset, rows, valid, width, t_->K);
        SyncEvent<HardEvent::MTE2_V>();
        auto work = input_.Get<int32_t>();
        Bias(work, rows, width);
        return work;
    }
    __aicore__ inline void ResolvePairMask(LocalTensor<uint16_t> win, LocalTensor<uint16_t> equal, uint32_t count) {
        auto temp = scratchMask_.Get<uint16_t>();
        auto even = even_.Get<uint16_t>();
        // One compare bit per 32-bit word. Resolve equal high words by
        // the unsigned-low comparison; duplicate the winner to both words.
        ShiftRight(temp, win, static_cast<uint16_t>(1), count);
        ShiftRight(equal, equal, static_cast<uint16_t>(1), count);
        PipeBarrier<PIPE_V>();
        And(equal, equal, even, count);
        PipeBarrier<PIPE_V>();
        And(win, win, equal, count);
        PipeBarrier<PIPE_V>();
        Not(equal, equal, count);
        PipeBarrier<PIPE_V>();
        And(temp, temp, equal, count);
        PipeBarrier<PIPE_V>();
        Or(win, win, temp, count);
        PipeBarrier<PIPE_V>();
        And(win, win, even, count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(temp, win, static_cast<uint16_t>(1), count);
        PipeBarrier<PIPE_V>();
        Or(win, win, temp, count);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Pair(LocalTensor<int32_t> dst, LocalTensor<int32_t> left, LocalTensor<int32_t> right,
                                uint32_t pairs, uint32_t words, uint32_t srcStride, uint32_t dstStride) {
        auto candidate = candidate_.Get<int32_t>();
        auto win = winner_.Get<uint16_t>(), equal = equal_.Get<uint16_t>();
        const BinaryRepeatParams arithmetic{1,
                                            1,
                                            1,
                                            static_cast<uint8_t>(dstStride / 8),
                                            static_cast<uint8_t>(srcStride / 8),
                                            static_cast<uint8_t>(srcStride / 8)};
        const BinaryRepeatParams compare{
            1, 1, 1, 1, static_cast<uint8_t>(dstStride / 8), static_cast<uint8_t>(srcStride / 8)};
        const BinaryRepeatParams equalParams{
            1, 1, 1, 1, static_cast<uint8_t>(srcStride / 8), static_cast<uint8_t>(srcStride / 8)};
        for (uint32_t start = 0; start < pairs; start += 255) {
            const uint8_t repeats = static_cast<uint8_t>(Min(255, pairs - start));
            const uint32_t count = repeats * 4;
            for (uint32_t col = 0; col < words; col += 64) {
                const uint32_t source = start * srcStride + col, output = start * dstStride + col;
                if constexpr (Reduce == SegmentMin)
                    AscendC::Min(candidate[output], left[source], right[source], 64, repeats, arithmetic);
                else
                    AscendC::Max(candidate[output], left[source], right[source], 64, repeats, arithmetic);
                PipeBarrier<PIPE_V>();
                Compare(win.ReinterpretCast<uint8_t>(), candidate[output], right[source], CMPMODE::EQ, 64, repeats,
                        compare);
                Compare(equal.ReinterpretCast<uint8_t>(), left[source], right[source], CMPMODE::EQ, 64, repeats,
                        equalParams);
                PipeBarrier<PIPE_V>();
                ResolvePairMask(win, equal, count);
                Select(dst.ReinterpretCast<float>()[output], win.ReinterpretCast<uint8_t>(),
                       right.ReinterpretCast<float>()[source], left.ReinterpretCast<float>()[source],
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, 64, repeats, arithmetic);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline LocalTensor<int32_t> Tree(LocalTensor<int32_t> work, uint32_t rows, uint32_t groups,
                                                uint32_t width) {
        auto next = tree_.Get<int32_t>();
        bool scratch = false;
        const uint32_t words = width * 2;
        while (rows > 1) {
            Pair(next, work, work[words], groups * rows / 2, words, words * 2, words);
            work = next;
            scratch = !scratch;
            next = scratch ? input_.Get<int32_t>() : tree_.Get<int32_t>();
            rows /= 2;
        }
        return work;
    }
    __aicore__ inline void Finish(LocalTensor<int32_t> value, uint32_t groups, uint32_t width) {
        Bias(value, groups, width);
    }
    __aicore__ inline void Store(LocalTensor<int32_t> value, uint64_t output, uint32_t groups, uint32_t valid,
                                 uint32_t width) {
        Finish(value, groups, width);
        if (!t_->hasInitial) {
            SyncEvent<HardEvent::V_S>();
            auto result = value.ReinterpretCast<int64_t>();
            const int64_t identity = Reduce == SegmentMin ? INT64_MAX : INT64_MIN;
            for (uint32_t lane = 0; lane < groups * width; ++lane)
                if (result.GetValue(lane) == identity)
                    result.SetValue(lane, 0);
            SyncEvent<HardEvent::S_V>();
        }
        SyncEvent<HardEvent::V_MTE3>();
        StorePaddedRows(out_, output, value.ReinterpretCast<int64_t>(), groups, valid, width, t_->K);
        SyncEvent<HardEvent::MTE3_V>();
        SyncEvent<HardEvent::V_MTE2>();
    }
    __aicore__ inline void Batch(uint64_t task, uint32_t groups, uint64_t col, uint32_t rows, uint32_t width) {
        auto work = Load(task * rows * t_->K + col, rows * groups, width, width);
        Store(Tree(work, rows, groups, width), task * t_->K + col, groups, width, width);
    }
    __aicore__ inline void Tile(uint64_t task, uint64_t col) {
        const uint64_t batch = task / t_->nSegments, seg = task % t_->nSegments;
        uint64_t first, last;
        FindSegmentBounds(index_, *t_, batch, seg, first, last);
        const uint32_t valid = static_cast<uint32_t>(Min(t_->tileK, t_->K - col));
        const uint32_t width = (valid + 31) / 32 * 32;
        auto acc = acc_.Get<int32_t>();
        Duplicate(acc, Reduce == SegmentMin ? INT32_MAX : INT32_MIN, width * 2);
        PipeBarrier<PIPE_V>();
        if (t_->hasInitial) {
            DataCopyPad(acc.ReinterpretCast<int64_t>(),
                        out_[last > first && first == 0 ? batch * t_->nSegments * t_->K + col : task * t_->K + col],
                        {1, valid * 8, 0, 0, 0},
                        DataCopyPadExtParams<int64_t>(true, 0, static_cast<uint8_t>((valid + 3) / 4 * 4 - valid), 0));
            SyncEvent<HardEvent::MTE2_V>();
            Bias(acc, 1, width);
        }
        for (uint64_t start = first; start < last;) {
            uint32_t rows = t_->tileRows;
            while (rows > last - start)
                rows /= 2;
            auto work = Load((batch * t_->M + start) * t_->K + col, rows, valid, width);
            auto value = Tree(work, rows, 1, width);
            Pair(acc, acc, value, 1, width * 2, width * 2, width * 2);
            SyncEvent<HardEvent::V_MTE2>();
            start += rows;
        }
        if (first == last && !t_->hasInitial) {
            Duplicate(acc, 0, width * 2);
            PipeBarrier<PIPE_V>();
            Bias(acc, 1, width);
        }
        Store(acc, task * t_->K + col, 1, valid, width);
    }
    const SegmentTilingData *t_;
    GlobalTensor<int64_t> src_, out_, index_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_, tree_, candidate_, acc_, bias_, winner_, equal_, scratchMask_, even_;
};

using namespace AscendC;
template <SegmentReduce Reduce> class SegmentInt64Narrow {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR out, const SegmentTilingData &t) {
        t_ = &t;
        capacity_ = t.tileRows * t.tileK;
        outCapacity_ = t.outputGroup * t.tileK;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(src), t.srcNumel);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(out), t.outNumel);
        pipe_.InitBuffer(input_, capacity_ * 8);
        pipe_.InitBuffer(low_, capacity_ * 4);
        pipe_.InitBuffer(temp_, outCapacity_ * 8);
        pipe_.InitBuffer(acc_, t.tileK * 4);
        pipe_.InitBuffer(packed_, outCapacity_ * 8);
        pipe_.InitBuffer(offsets_, outCapacity_ * 8);
        maskBytes_ = ((outCapacity_ * 2 + 7) / 8 + 31) / 32 * 32;
        pipe_.InitBuffer(mask_, maskBytes_);
        auto off = offsets_.Get<uint32_t>();
        for (uint32_t i = 0; i < outCapacity_ * 2; i++)
            off.SetValue(i, (i / 2) * 4);
        SyncEvent<HardEvent::S_V>();
    }

  private:
    __aicore__ inline bool ProcessLongSegment(uint64_t task, uint64_t col, uint32_t rows, uint32_t width) {
        auto acc = acc_.Get<int32_t>();
        Duplicate(acc, Reduce == SegmentMin ? INT32_MAX : Reduce == SegmentMax ? INT32_MIN : 0, width);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < rows; row += t_->tileRows) {
            const uint32_t active = Small(rows - row, t_->tileRows);
            if (!LoadProve((task * rows + row) * t_->K + col, active, width, rows))
                return false;
            auto value = Tree(low_.Get<int32_t>(), active, 1, width);
            if constexpr (Reduce == SegmentMin)
                AscendC::Min(acc, acc, value, width);
            else if constexpr (Reduce == SegmentMax)
                AscendC::Max(acc, acc, value, width);
            else
                Add(acc, acc, value, width);
            PipeBarrier<PIPE_V>();
            SyncEvent<HardEvent::V_MTE2>();
        }
        Store(acc, task * t_->K + col, 1, width, rows);
        return true;
    }
    __aicore__ inline bool ProcessColumnGroup(uint64_t first, uint64_t last, uint64_t col, uint32_t rows) {
        const uint32_t width = Small(t_->tileK, t_->K - col);
        if (rows <= t_->tileRows && (last - first) * rows <= t_->tileRows &&
            first / t_->nSegments == (last - 1) / t_->nSegments) {
            if (!LoadProve(first * rows * t_->K + col, (last - first) * rows, width, rows))
                return false;
            auto value = Tree(low_.Get<int32_t>(), rows, last - first, width);
            Store(value, first * t_->K + col, last - first, width, rows);
            return true;
        }
        for (uint64_t task = first; task < last; ++task) {
            if (!ProcessLongSegment(task, col, rows, width))
                return false;
        }
        return true;
    }

  public:
    __aicore__ inline bool Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        const uint32_t rows = t_->M / t_->nSegments;
        for (uint64_t unit = begin; unit < end; ++unit) {
            const auto bounds = DescribeUnit(*t_, unit);
            for (uint64_t col = bounds.columnBegin; col < bounds.columnEnd; col += t_->tileK) {
                if (!ProcessColumnGroup(bounds.first, bounds.last, col, rows))
                    return false;
            }
        }
        return true;
    }

  private:
    __aicore__ static inline uint64_t Small(uint64_t a, uint64_t b) {
        return a < b ? a : b;
    }

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
        SyncEvent<HardEvent::MTE2_V>();
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
            const uint32_t pairs = groups * rows / 2;
            const BinaryRepeatParams rp{1,
                                        1,
                                        1,
                                        static_cast<uint8_t>(width / 8),
                                        static_cast<uint8_t>(width / 4),
                                        static_cast<uint8_t>(width / 4)};
            for (uint32_t first = 0; first < pairs; first += 255) {
                const uint8_t repeat = Small(255, pairs - first);
                for (uint32_t c = 0; c < width; c += 64) {
                    auto a = work[first * width * 2 + c], b = work[first * width * 2 + width + c],
                         d = next[first * width + c];
                    if constexpr (Reduce == SegmentMin)
                        AscendC::Min(d, a, b, Small(64, width - c), repeat, rp);
                    else if constexpr (Reduce == SegmentMax)
                        AscendC::Max(d, a, b, Small(64, width - c), repeat, rp);
                    else
                        Add(d, a, b, Small(64, width - c), repeat, rp);
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
        auto temp = temp_.Get<int32_t>();
        if constexpr (Reduce == SegmentMean) {
            uint32_t shift = 0;
            for (uint32_t n = rows; n > 1; n /= 2)
                ++shift;
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
        ShiftRight(temp, packed, static_cast<int32_t>(31), count * 2);
        Duplicate(mask_.Get<uint16_t>(), static_cast<uint16_t>(0xAAAA), (count * 2 + 15) / 16);
        PipeBarrier<PIPE_V>();
        Select(packed.ReinterpretCast<float>(), mask_.Get<uint8_t>(), temp.ReinterpretCast<float>(),
               packed.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count * 2);
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_MTE3>();
        if (width == t_->K)
            DataCopy(out_[offset], packed.ReinterpretCast<int64_t>(), groups * width);
        else
            DataCopyPad(out_[offset], packed.ReinterpretCast<int64_t>(),
                        {static_cast<uint16_t>(groups), width * 8, 0, static_cast<uint32_t>((t_->K - width) * 8), 0});
        SyncEvent<HardEvent::MTE3_V>();
        SyncEvent<HardEvent::V_MTE2>();
    }
    const SegmentTilingData *t_;
    uint32_t capacity_, outCapacity_, maskBytes_;
    GlobalTensor<int64_t> src_, out_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_, low_, temp_, acc_, packed_, offsets_, mask_;
};

using namespace AscendC;

// Only a complete current-index proof permits this path. Each tree handles
// adjacent equal-length segments independently, with native modular int32 adds.
// No integer value is converted to floating point, including signed division.
template <SegmentReduce Reduce> class SegmentInt32Batch {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR out, const SegmentTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer((__gm__ int32_t *)src, t.srcNumel);
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
            DataCopyPad(
                work, src_[first * rows * t_->K + col],
                {static_cast<uint16_t>(groups * rows), width * 4, static_cast<uint32_t>((t_->K - width) * 4), 0, 0},
                DataCopyPadExtParams<int32_t>(false, 0, 0, 0));
            SyncEvent<HardEvent::MTE2_V>();
            auto next = scratch_.Get<int32_t>();
            bool inScratch = false;
            for (uint32_t remaining = rows; remaining > 1; remaining /= 2) {
                const uint32_t pairs = groups * remaining / 2;
                const BinaryRepeatParams pair{1,
                                              1,
                                              1,
                                              static_cast<uint8_t>(width / 8),
                                              static_cast<uint8_t>(width / 4),
                                              static_cast<uint8_t>(width / 4)};
                for (uint32_t base = 0; base < pairs; base += 255) {
                    for (uint32_t c = 0; c < width; c += 64) {
                        Add(next[base * width + c], work[2 * base * width + c], work[(2 * base + 1) * width + c],
                            Min(64, width - c), static_cast<uint8_t>(Min(255, pairs - base)), pair);
                    }
                }
                PipeBarrier<PIPE_V>();
                work = next;
                inScratch = !inScratch;
                next = inScratch ? input_.Get<int32_t>() : scratch_.Get<int32_t>();
            }
            if constexpr (Reduce == SegmentMean)
                Divide(work, groups * width, rows);
            SyncEvent<HardEvent::V_MTE3>();
            DataCopyPad(out_[first * t_->K + col], work,
                        {static_cast<uint16_t>(groups), width * 4, 0, static_cast<uint32_t>((t_->K - width) * 4), 0});
            SyncEvent<HardEvent::MTE3_V>();
            SyncEvent<HardEvent::V_MTE2>();
        }
    }

  private:
    __aicore__ static inline uint64_t Min(uint64_t a, uint64_t b) {
        return a < b ? a : b;
    }

    __aicore__ inline void Divide(LocalTensor<int32_t> value, uint32_t count, uint32_t rows) {
        if (rows == 1)
            return;
        uint32_t shift = 0;
        for (uint32_t n = rows; n > 1; n /= 2)
            ++shift;
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
    const SegmentTilingData *t_;
    GlobalTensor<int32_t> src_, out_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_, scratch_, sign_, bias_;
};

template <SegmentReduce Reduce>
__global__ __aicore__ void SegmentInt32BatchKernel(GM_ADDR src, GM_ADDR out, SegmentTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentInt32Batch<Reduce> op;
    op.Init(src, out, t);
    op.Process();
}

// ============================================================================
// Kernel dispatch and dtype launchers
// ============================================================================

using namespace AscendC;
template <typename T> struct IntegerReduction {
    template <SegmentReduce Reduce, bool HasArg>
    __aicore__ static inline void Run(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg,
                                      const SegmentTilingData &t) {
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
                      (std::is_same<T, int64_t>::value && HasArg)) {
            SegmentIntegerScalar<T, Reduce, HasArg> op;
            op.Init(src, index, out, arg, t);
            op.Process();
        } else if constexpr (std::is_same<T, int64_t>::value && Reduce >= SegmentMin) {
            SegmentInt64Extreme<Reduce> op;
            op.Init(src, index, out, t);
            op.Process();
        } else if constexpr (std::is_same<T, int64_t>::value) {
            SegmentInt64Vector<Reduce> op;
            op.Init(src, index, out, t);
            op.Process();
        }
    }
};

// Host has already validated eligibility and finalized every tiling field.
void LaunchInt32Batch(void *src, void *out, const SegmentTilingData &t, aclrtStream stream) {
    if (t.reduceCode == SegmentSum)
        SegmentInt32BatchKernel<SegmentSum><<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)out, t);
    else
        SegmentInt32BatchKernel<SegmentMean><<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)out, t);
}

// Host proves uniform segments, bounded values and value-only output.
struct Int64NarrowReduction {
    template <SegmentReduce Reduce, bool>
    __aicore__ static inline void Run(GM_ADDR src, GM_ADDR, GM_ADDR out, GM_ADDR, const SegmentTilingData &t) {
        SegmentInt64Narrow<Reduce> op;
        op.Init(src, out, t);
        op.Process();
    }
};

// ============================================================================
// Exact INT64 source validation
// ============================================================================

using namespace AscendC;
namespace {
constexpr uint32_t kValidationElements = 4096;
class ValidateInt64Range {
  private:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR flags, int64_t elements) {
        input.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(src), elements);
        output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flags), GetBlockNum() * 8);
        pipe.InitBuffer(sourceBuffer, kValidationElements * 8);
        pipe.InitBuffer(lowBuffer, kValidationElements * 4);
        pipe.InitBuffer(highBuffer, kValidationElements * 4);
        pipe.InitBuffer(workBuffer, kValidationElements * 4);
        pipe.InitBuffer(maskBuffer, kValidationElements / 4);
    }
    __aicore__ inline void LoadChunk(int64_t offset, uint32_t count, uint32_t rounded) {
        auto source = sourceBuffer.Get<int64_t>();
        auto low = lowBuffer.Get<int32_t>();
        auto high = highBuffer.Get<int32_t>();
        // Zero padding is inside the tested interval for every valid bound.
        Duplicate(sourceBuffer.Get<int32_t>(), 0, rounded * 2);
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_MTE2>();
        DataCopyPad(source, input[offset], {1, count * 8, 0, 0, 0},
                    DataCopyPadExtParams<int64_t>(true, 0, static_cast<uint8_t>((4 - count % 4) % 4), 0));
        SyncEvent<HardEvent::MTE2_V>();
        uint64_t reserved = 0;
        GatherMask(low, sourceBuffer.Get<int32_t>(), 1, true, rounded * 2, {1, 1, 8, 0}, reserved);
        GatherMask(high, sourceBuffer.Get<int32_t>(), 2, true, rounded * 2, {1, 1, 8, 0}, reserved);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CheckValues(uint32_t rounded, int64_t bound) {
        auto low = lowBuffer.Get<int32_t>();
        auto high = highBuffer.Get<int32_t>();
        auto work = workBuffer.Get<int32_t>();
        auto valid = maskBuffer.Get<uint8_t>();
        auto within = valid[kValidationElements / 8];
        ShiftRight(work, low, static_cast<int32_t>(31), rounded);
        PipeBarrier<PIPE_V>();
        Compare(valid, high, work, CMPMODE::EQ, rounded);
        PipeBarrier<PIPE_V>();
        Mins(work, low, static_cast<int32_t>(bound), rounded);
        PipeBarrier<PIPE_V>();
        Maxs(work, work, -static_cast<int32_t>(bound), rounded);
        PipeBarrier<PIPE_V>();
        Compare(within, low, work, CMPMODE::EQ, rounded);
        PipeBarrier<PIPE_V>();
        And(valid.ReinterpretCast<uint16_t>(), valid.ReinterpretCast<uint16_t>(), within.ReinterpretCast<uint16_t>(),
            rounded / 16);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool AllPredicatesValid(uint32_t rounded) {
        auto valid = maskBuffer.Get<uint8_t>();
        // These are packed 16-bit PREDICATES, not source integers: -1
        // means all sixteen original values passed the exact bit checks.
        auto predicate = workBuffer.Get<float>();
        auto result = highBuffer.Get<float>();
        Cast(predicate, valid.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, rounded / 16);
        PipeBarrier<PIPE_V>();
        Adds(predicate, predicate, 1.0f, rounded / 16);
        PipeBarrier<PIPE_V>();
        Abs(predicate, predicate, rounded / 16);
        PipeBarrier<PIPE_V>();
        ReduceMax(result, predicate, result[64], rounded / 16, false);
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_S>();
        const bool proved = result.GetValue(0) == 0.0f;
        SyncEvent<HardEvent::S_V>();
        return proved;
    }
    __aicore__ inline void StoreProof(bool proved) {
        SyncEvent<HardEvent::V_S>();
        auto result = lowBuffer.Get<int32_t>();
        for (uint32_t i = 0; i < 8; ++i)
            result.SetValue(i, i == 0 && proved ? 1 : 0);
        SyncEvent<HardEvent::S_MTE3>();
        // A full 32-byte store per core prevents partial-block writer overlap.
        DataCopy(output[GetBlockIdx() * 8], result, 8);
    }

  public:
    __aicore__ inline void Run(GM_ADDR src, GM_ADDR flags, int64_t elements, int64_t bound) {
        Init(src, flags, elements);
        bool proved = bound >= 0 && bound <= INT32_MAX;
        const int64_t begin = GetBlockIdx() * elements / GetBlockNum();
        const int64_t end = (GetBlockIdx() + 1) * elements / GetBlockNum();
        for (int64_t offset = begin; proved && offset < end; offset += kValidationElements) {
            const uint32_t count = end - offset < kValidationElements ? end - offset : kValidationElements;
            const uint32_t rounded = (count + 63) / 64 * 64;
            LoadChunk(offset, count, rounded);
            CheckValues(rounded, bound);
            proved = AllPredicatesValid(rounded);
        }
        StoreProof(proved);
    }

  private:
    GlobalTensor<int64_t> input;
    GlobalTensor<int32_t> output;
    TPipe pipe;
    TBuf<TPosition::VECCALC> sourceBuffer, lowBuffer, highBuffer, workBuffer, maskBuffer;
};
__global__ __aicore__ void ValidateInt64Kernel(GM_ADDR src, GM_ADDR flags, int64_t elements, int64_t bound) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ValidateInt64Range validation;
    validation.Run(src, flags, elements, bound);
}
} // namespace

void SegmentCooValidateInt64(void *src, int64_t numel, int64_t absBound, int32_t *perCoreValid, int blocks,
                             aclrtStream stream) {
    ValidateInt64Kernel<<<blocks, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)perCoreValid, numel, absBound);
}

// ============================================================================
// CPU extrema argument state and initial NaN repair
// ============================================================================

using namespace AscendC;
namespace {
constexpr uint32_t kFeatureSlice = 64, kSourceRows = 64;

// A separate strict-first state reproduces the fixed CPU arg carry. It must
// never be updated from the main kernel's task-defined last-tie arg output.
template <typename Raw, uint32_t Dtype> class ExtremaRepair {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, GM_ADDR initial,
                                const SegmentTilingData &t) {
        t_ = t;
        hasArg_ = t.hasArg && arg != nullptr;
        hasInitial_ = t.hasInitial && initial != nullptr;
        isMin_ = t.reduceCode == 2;
        src_.SetGlobalBuffer((__gm__ Raw *)src, t.srcNumel);
        index_.SetGlobalBuffer((__gm__ int64_t *)index, t.indexNumel);
        out_.SetGlobalBuffer((__gm__ Raw *)out, t.outNumel);
        if (hasArg_)
            arg_.SetGlobalBuffer((__gm__ int64_t *)arg, t.outNumel);
        if (hasInitial_)
            initial_.SetGlobalBuffer((__gm__ Raw *)initial, t.outNumel);
        pipe_.InitBuffer(sourceBuf_, kSourceRows * kFeatureSlice * sizeof(Raw));
        pipe_.InitBuffer(initialBuf_, kFeatureSlice * sizeof(Raw));
        pipe_.InitBuffer(outputBuf_, kFeatureSlice * sizeof(Raw));
        pipe_.InitBuffer(argBuf_, kFeatureSlice * sizeof(int64_t));
        pipe_.InitBuffer(carryBuf_, kFeatureSlice * sizeof(int64_t));
        pipe_.InitBuffer(keyBuf_, kFeatureSlice * sizeof(uint64_t));
        pipe_.InitBuffer(seedNanBuf_, kFeatureSlice);
    }
    __aicore__ inline void Process() {
        const uint64_t slices = (t_.K + kFeatureSlice - 1) / kFeatureSlice;
        for (uint64_t slice = GetBlockIdx(); slice < slices; slice += t_.usedBlockNum)
            ProcessSlice(slice);
    }

  private:
    __aicore__ inline bool IsNan(Raw raw) const {
        const uint64_t bits = static_cast<uint64_t>(raw);
        if constexpr (Dtype == 0)
            return (bits & 0x7f800000ULL) == 0x7f800000ULL && (bits & 0x007fffffULL);
        else if constexpr (Dtype == 1)
            return (bits & 0x7c00ULL) == 0x7c00ULL && (bits & 0x03ffULL);
        else if constexpr (Dtype == 2)
            return (bits & 0x7f80ULL) == 0x7f80ULL && (bits & 0x007fULL);
        else
            return false;
    }
    // Raw IEEE keys preserve exact ordering, including subnormals. Zero signs
    // compare equal; NaNs are excluded separately. Integer keys retain all bits.
    __aicore__ inline uint64_t Key(Raw raw) const {
        const uint64_t bits = static_cast<uint64_t>(raw);
        constexpr uint32_t width = sizeof(Raw) * 8;
        constexpr uint64_t sign = uint64_t(1) << (width - 1);
        if constexpr (Dtype <= 2) {
            constexpr uint64_t mask = (uint64_t(1) << width) - 1;
            if ((bits & (sign - 1)) == 0)
                return sign;
            return bits & sign ? (~bits) & mask : bits ^ sign;
        } else if constexpr (Dtype == 4)
            return bits;
        else
            return bits ^ sign;
    }
    __aicore__ inline Raw FiniteInitial() const {
        if constexpr (Dtype == 0)
            return Raw(isMin_ ? 0x7f7fffffULL : 0xff7fffffULL);
        else if constexpr (Dtype == 1)
            return Raw(isMin_ ? 0x7bffULL : 0xfbffULL);
        else if constexpr (Dtype == 2)
            return Raw(isMin_ ? 0x7f7fULL : 0xff7fULL);
        else if constexpr (Dtype == 3)
            return Raw(isMin_ ? 0x7fULL : 0x80ULL);
        else if constexpr (Dtype == 4)
            return Raw(isMin_ ? 0xffULL : 0ULL);
        else if constexpr (Dtype == 5)
            return Raw(isMin_ ? 0x7fffffffULL : 0x80000000ULL);
        else
            return Raw(isMin_ ? 0x7fffffffffffffffULL : 0x8000000000000000ULL);
    }
    template <typename Value>
    __aicore__ inline void Load(LocalTensor<Value> local, GlobalTensor<Value> &global, uint64_t offset,
                                uint32_t count) {
        const uint32_t bytes = count * sizeof(Value);
        const uint8_t pad = (((bytes + 31) & ~31U) - bytes) / sizeof(Value);
        DataCopyPad(local, global[offset], {1, bytes, 0, 0, 0}, DataCopyPadExtParams<Value>(true, 0, pad, Value(0)));
        SyncEvent<HardEvent::MTE2_S>();
    }
    template <typename Value>
    __aicore__ inline void Store(GlobalTensor<Value> &global, uint64_t offset, LocalTensor<Value> local,
                                 uint32_t count) {
        SyncEvent<HardEvent::S_MTE3>();
        DataCopyPad(global[offset], local, {1, uint32_t(count * sizeof(Value)), 0, 0, 0});
        // Complete partial writes before reusing UB or touching adjacent blocks.
        SyncEvent<HardEvent::MTE3_S>();
    }
    __aicore__ inline int64_t IndexAt(uint64_t batch, uint64_t row) const {
        return index_.GetValue(batch * t_.indexBatchStride + row * t_.indexRowStride);
    }
    __aicore__ inline uint64_t SegmentEnd(uint64_t batch, uint64_t begin, int64_t label) const {
        // Host rejects INT64_MAX labels, so label + 1 is exact and in range.
        return IndexLowerBound(index_, batch * t_.indexBatchStride, t_.M, label + 1, begin + 1, t_.indexRowStride);
    }
    __aicore__ inline void LoadRows(uint64_t batch, uint64_t row, uint32_t rows) {
        const uint64_t offset = (batch * t_.M + row) * t_.K + featureBegin_;
        LoadPaddedRows(sourceBuf_.Get<Raw>(), src_, offset, rows, featureCount_, sourcePitch_, t_.K);
        SyncEvent<HardEvent::MTE2_S>();
    }
    __aicore__ inline void UpdateCarry(uint64_t batch, uint64_t begin, uint64_t end) {
        auto values = sourceBuf_.Get<Raw>();
        auto keys = keyBuf_.Get<uint64_t>();
        auto carry = carryBuf_.Get<int64_t>();
        auto nan = seedNanBuf_.Get<uint8_t>();
        for (uint64_t start = begin; start < end;) {
            const uint32_t rows = end - start < kSourceRows ? end - start : kSourceRows;
            LoadRows(batch, start, rows);
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t f = 0; f < featureCount_; ++f) {
                    if (nan.GetValue(f))
                        continue;
                    const Raw raw = values.GetValue(uint64_t(r) * sourcePitch_ + f);
                    if (IsNan(raw))
                        continue;
                    const uint64_t candidate = Key(raw), current = keys.GetValue(f);
                    if (!(isMin_ ? candidate < current : candidate > current))
                        continue;
                    keys.SetValue(f, candidate);
                    outputBuf_.Get<Raw>().SetValue(f, raw);
                    if (hasArg_)
                        carry.SetValue(f, static_cast<int64_t>(start + r));
                }
            }
            start += rows;
        }
    }
    __aicore__ inline void EmptySegment(uint64_t batch, uint64_t label) {
        const uint64_t offset = (batch * t_.nSegments + label) * t_.K + featureBegin_;
        if (!hasInitial_) {
            auto output = outputBuf_.Get<Raw>();
            for (uint32_t f = 0; f < featureCount_; ++f)
                output.SetValue(f, Raw(0));
            Store(out_, offset, output, featureCount_);
        }
        if (hasArg_) {
            auto args = argBuf_.Get<int64_t>();
            for (uint32_t f = 0; f < featureCount_; ++f)
                args.SetValue(f, static_cast<int64_t>(t_.M));
            Store(arg_, offset, args, featureCount_);
        }
    }
    __aicore__ inline void RepairSegment(uint64_t batch, uint64_t label, uint64_t begin, uint64_t end) {
        auto initial = initialBuf_.Get<Raw>();
        auto output = outputBuf_.Get<Raw>();
        auto keys = keyBuf_.Get<uint64_t>();
        auto nan = seedNanBuf_.Get<uint8_t>();
        if (hasInitial_) {
            const uint64_t seedLabel = begin == 0 ? 0 : label;
            Load(initial, initial_, (batch * t_.nSegments + seedLabel) * t_.K + featureBegin_, featureCount_);
        }
        for (uint32_t f = 0; f < featureCount_; ++f) {
            const Raw seed = hasInitial_ ? initial.GetValue(f) : FiniteInitial();
            output.SetValue(f, seed);
            keys.SetValue(f, Key(seed));
            nan.SetValue(f, static_cast<uint8_t>(IsNan(seed)));
        }
        UpdateCarry(batch, begin, end);
        if (!hasInitial_)
            for (uint32_t f = 0; f < featureCount_; ++f)
                if (output.GetValue(f) == FiniteInitial())
                    output.SetValue(f, Raw(0));
        const uint64_t offset = (batch * t_.nSegments + label) * t_.K + featureBegin_;
        Store(out_, offset, output, featureCount_);
        if (hasArg_)
            Store(arg_, offset, carryBuf_.Get<int64_t>(), featureCount_);
    }
    __aicore__ inline void ProcessSlice(uint64_t slice) {
        featureBegin_ = slice * kFeatureSlice;
        featureCount_ = t_.K - featureBegin_ < kFeatureSlice ? t_.K - featureBegin_ : kFeatureSlice;
        sourcePitch_ = ((featureCount_ * sizeof(Raw) + 31) & ~31U) / sizeof(Raw);
        // CPU args(K) lives outside both segment and batch loops.
        if (hasArg_) {
            auto carry = carryBuf_.Get<int64_t>();
            for (uint32_t f = 0; f < featureCount_; ++f)
                carry.SetValue(f, 0);
        }
        const uint64_t uniformRows = t_.indexMode == kUniformIndex ? t_.M / t_.nSegments : 0;
        for (uint64_t batch = 0; batch < t_.E_1; ++batch) {
            uint64_t nextLabel = 0;
            for (uint64_t begin = 0; begin < t_.M;) {
                const uint64_t label = uniformRows ? begin / uniformRows : IndexAt(batch, begin);
                const uint64_t end = uniformRows ? begin + uniformRows : SegmentEnd(batch, begin, label);
                for (; nextLabel < label; ++nextLabel)
                    EmptySegment(batch, nextLabel);
                RepairSegment(batch, label, begin, end);
                nextLabel = label + 1;
                begin = end;
            }
            for (; nextLabel < t_.nSegments; ++nextLabel)
                EmptySegment(batch, nextLabel);
        }
    }
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> sourceBuf_, initialBuf_, outputBuf_, argBuf_, carryBuf_, keyBuf_, seedNanBuf_;
    GlobalTensor<Raw> src_, out_, initial_;
    GlobalTensor<int64_t> index_, arg_;
    SegmentTilingData t_;
    uint64_t featureBegin_ = 0;
    uint32_t featureCount_ = 0, sourcePitch_ = 0;
    bool hasArg_ = false, hasInitial_ = false, isMin_ = false;
};

template <typename Raw, uint32_t Dtype>
__global__ __aicore__ void RepairExtremaKernel(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, GM_ADDR initial,
                                               SegmentTilingData t) {
    ExtremaRepair<Raw, Dtype> kernel;
    kernel.Init(src, index, out, arg, initial, t);
    kernel.Process();
}
template <typename Raw, uint32_t Dtype>
void LaunchRepair(void *src, void *index, void *out, void *arg, void *initial, const SegmentTilingData &t,
                  aclrtStream stream) {
    auto launch = t;
    const uint64_t slices = (t.K + kFeatureSlice - 1) / kFeatureSlice;
    launch.usedBlockNum = slices < t.usedBlockNum ? slices : t.usedBlockNum;
    if ((t.K * sizeof(Raw)) % 32 || (t.K * sizeof(int64_t)) % 32 || reinterpret_cast<uintptr_t>(out) % 32 ||
        (t.hasArg && reinterpret_cast<uintptr_t>(arg) % 32))
        launch.usedBlockNum = 1;
    RepairExtremaKernel<Raw, Dtype><<<launch.usedBlockNum, nullptr, stream>>>(
        (GM_ADDR)src, (GM_ADDR)index, (GM_ADDR)out, (GM_ADDR)arg, (GM_ADDR)initial, launch);
}
} // namespace

void SegmentCooRepairExtrema(void *src, void *index, void *out, void *arg, void *initial, const SegmentTilingData &t,
                             aclrtStream stream) {
    if (!(t.sourceFlags & kRepairExtrema) || t.reduceCode < 2 || t.reduceCode > 3 || !t.K || !t.E_1 ||
        !t.nSegments)
        return;
    switch (t.dtypeCode) {
        case 0:
            LaunchRepair<uint32_t, 0>(src, index, out, arg, initial, t, stream);
            break;
        case 1:
            LaunchRepair<uint16_t, 1>(src, index, out, arg, initial, t, stream);
            break;
        case 2:
            LaunchRepair<uint16_t, 2>(src, index, out, arg, initial, t, stream);
            break;
        case 3:
            LaunchRepair<uint8_t, 3>(src, index, out, arg, initial, t, stream);
            break;
        case 4:
            LaunchRepair<uint8_t, 4>(src, index, out, arg, initial, t, stream);
            break;
        case 5:
            LaunchRepair<uint32_t, 5>(src, index, out, arg, initial, t, stream);
            break;
        case 6:
            LaunchRepair<uint64_t, 6>(src, index, out, arg, initial, t, stream);
            break;
    }
}

// ============================================================================
// Public launch entry point
// ============================================================================


// CPU entry point compiled by Ascend C for the kernel launch syntax.
// Arithmetic planning is complete before this function is queued.
void SegmentCoo(void *src, int64_t *index, void *out, int64_t *arg, int64_t *initialRaw, const SegmentCooPlan &plan,
                aclrtStream stream) {
    const auto &t = plan.tiling;
    if (!t.usedBlockNum)
        return;
    if (t.sourceFlags & kRepairExtrema) {
        auto ordered = t;
        ordered.usedBlockNum = plan.availableBlocks;
        SegmentCooRepairExtrema(src, index, out, arg, initialRaw, ordered, stream);
        return;
    }
    if (plan.path == SegmentKernelPath::Int32Batch) {
        LaunchInt32Batch(src, out, t, stream);
    } else if (plan.path == SegmentKernelPath::Int64Narrow) {
        DispatchTyped<Int64NarrowReduction>(src, index, out, arg, t, stream);
    } else {
        switch (t.dtypeCode) {
            case SEGMENT_COO_FLOAT32:
            case SEGMENT_COO_FLOAT16:
            case SEGMENT_COO_BFLOAT16:
            case SEGMENT_COO_INT32:
                LaunchSegmentCooVector(src, index, out, arg, t, stream);
                break;
            case SEGMENT_COO_INT8:
                DispatchTyped<IntegerReduction<int8_t>>(src, index, out, arg, t, stream);
                break;
            case SEGMENT_COO_UINT8:
                DispatchTyped<IntegerReduction<uint8_t>>(src, index, out, arg, t, stream);
                break;
            case SEGMENT_COO_INT64:
                DispatchTyped<IntegerReduction<int64_t>>(src, index, out, arg, t, stream);
                break;
        }
    }
}

} // namespace opsgnn

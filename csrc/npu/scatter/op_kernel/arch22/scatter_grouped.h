/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "scatter_common.h"

namespace opsgnn::scatter_a2 {
// Every bucket/channel accumulator visits its original row coordinates in
// increasing order. A tile is an input transfer unit, never a partial sum.
template <uint32_t Kind, uint32_t Reduction, bool Scalar, bool Waves = false> class GroupedScatter {
    using Numeric = Number<Kind>;
    using Raw = typename Numeric::Raw;
    using Value = typename Numeric::Value;
    using T = typename std::conditional<Kind == Float32, float,
                                        typename std::conditional<Kind == Float16, half, bfloat16_t>::type>::type;
    using Operation = Reducer<Kind, Reduction == Mean ? Sum : Reduction>;
    static constexpr bool Extreme = Reduction == Min || Reduction == Max;
    static constexpr uint32_t MaxWaveOutputs = 8;
    struct BucketRows {
        uint64_t batch, first, last;
    };

  public:
    __aicore__ inline explicit GroupedScatter(TPipe &pipe) : pipe_(pipe) {}
    __aicore__ inline void Run(GM_ADDR src, GM_ADDR rowptr, GM_ADDR rows, GM_ADDR out, GM_ADDR arg, GM_ADDR validation,
                               const GroupTiling &t) {
        t_ = &t;
        if (!Validated(validation))
            return;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ Raw *>(src), t.srcElements);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ Raw *>(out), t.outElements);
        rowptr_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(rowptr), t.buckets + 1);
        rows_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(rows), t.length);
        if constexpr (Extreme)
            arg_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(arg), t.outElements);
        InitBuffers();
        for (uint64_t unit = GetBlockIdx(); unit < t.units; unit += GetBlockNum())
            ProcessUnit(unit);
    }

  private:
    __aicore__ inline bool Validated(GM_ADDR validation) {
        if (!t_->validationBlocks)
            return true;
        const uint32_t words = t_->validationBlocks * kGroupValidationWordsPerCore;
        GlobalTensor<int32_t> flags;
        flags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(validation), words);
        pipe_.InitBuffer(validation_, t_->validationBlocks * kGroupValidationBytesPerCore);
        auto local = validation_.Get<int32_t>();
        DataCopy(local, flags, words);
        Sync<HardEvent::MTE2_S>();
        for (uint32_t core = 0; core < t_->validationBlocks; ++core)
            if (local.GetValue(core * kGroupValidationWordsPerCore) != 1)
                return false;
        Sync<HardEvent::S_MTE2>();
        return true;
    }
    __aicore__ inline void InitBuffers() {
        const uint32_t width = t_->tileWidth, rows = t_->tileRows, groups = t_->outputGroup;
        pipe_.InitBuffer(source_, rows * width * sizeof(Raw));
        pipe_.InitBuffer(output_, groups * width * sizeof(Raw));
        pipe_.InitBuffer(rowIds_, (rows * 8 + kDmaBytes - 1) / kDmaBytes * kDmaBytes);
        pipe_.InitBuffer(accumulator_, width * sizeof(Value));
        pipe_.InitBuffer(groupPointers_, ((groups + 1) * sizeof(int64_t) + kDmaBytes - 1) / kDmaBytes * kDmaBytes);
        if constexpr (!Scalar && Kind != Float32)
            pipe_.InitBuffer(work_, rows * width * sizeof(float));
        if constexpr (Extreme) {
            pipe_.InitBuffer(arguments_, groups * width * sizeof(int64_t));
            if constexpr (!Scalar) {
                pipe_.InitBuffer(argumentFloat_, width * sizeof(float));
                pipe_.InitBuffer(rowPosition_, width * sizeof(float));
                pipe_.InitBuffer(mask_, (width + kMaskElementsPerDmaBlock - 1) / kMaskElementsPerDmaBlock * kDmaBytes);
            }
        }
        if constexpr (!Scalar && Reduction == Mean)
            pipe_.InitBuffer(denominator_, width * sizeof(float));
    }
    __aicore__ inline void ProcessUnit(uint64_t unit) {
        const uint64_t first = unit / t_->unitFeatureTiles * t_->outputGroup;
        const uint32_t groups = Smaller(t_->outputGroup, t_->totalTasks - first);
        const uint64_t feature = unit % t_->unitFeatureTiles * t_->tileWidth;
        if (t_->unitFeatureTiles == 1) {
            for (uint64_t col = 0; col < t_->after; col += t_->tileWidth)
                ProcessGroup(first, groups, col);
        } else
            ProcessGroup(first, groups, feature);
    }
    __aicore__ inline void CopyInitial(uint64_t first, uint32_t groups, uint64_t feature, uint32_t valid) {
        if (!t_->hasOut)
            return;
        const uint64_t gap = (t_->after - valid) * sizeof(Raw);
        const uint32_t transfer = (valid * sizeof(Raw) + kDmaBytes - 1) / kDmaBytes * kDmaBytes;
        const uint32_t stride = (t_->tileWidth * sizeof(Raw) - transfer) / kDmaBytes;
        if (gap <= UINT32_MAX)
            DataCopyPad(output_.Get<Raw>(), out_[first * t_->after + feature],
                        {static_cast<uint16_t>(groups), valid * static_cast<uint32_t>(sizeof(Raw)),
                         static_cast<uint32_t>(gap), stride, 0},
                        DataCopyPadExtParams<Raw>(false, 0, 0, 0));
        else
            for (uint32_t group = 0; group < groups; ++group)
                DataCopyPad(output_.Get<Raw>()[group * t_->tileWidth], out_[(first + group) * t_->after + feature],
                            {1, valid * static_cast<uint32_t>(sizeof(Raw)), 0, 0, 0},
                            DataCopyPadExtParams<Raw>(false, 0, 0, 0));
        if constexpr (Scalar)
            Sync<HardEvent::MTE2_S>();
        else
            Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline void ProcessGroup(uint64_t first, uint32_t groups, uint64_t feature) {
        const uint32_t valid = Smaller(t_->tileWidth, t_->after - feature);
        CopyInitial(first, groups, feature, valid);
        const uint64_t bucket = first % t_->buckets;
        if (bucket + groups <= t_->buckets)
            ProcessPrefetchedGroup(first / t_->buckets, bucket, groups, feature, valid);
        else
            for (uint32_t group = 0; group < groups; ++group)
                ProcessBucket(first + group, group, feature, valid);
        Sync<HardEvent::S_MTE2>();
        StoreGroup(first, groups, feature, valid);
    }
    __aicore__ inline void ProcessPrefetchedGroup(uint64_t batch, uint64_t bucket, uint32_t groups, uint64_t feature,
                                                  uint32_t valid) {
        auto pointers = groupPointers_.Get<int64_t>();
        DataCopyPad(pointers, rowptr_[bucket], {1, (groups + 1) * 8, 0, 0, 0},
                    DataCopyPadExtParams<int64_t>(false, 0, 0, 0));
        Sync<HardEvent::MTE2_S>();
        const uint64_t first = pointers.GetValue(0), last = pointers.GetValue(groups);
        if (last - first <= t_->tileRows) {
            ProcessLoadedGroup({batch, first, last}, groups, feature, valid);
            return;
        }
        for (uint32_t group = 0; group < groups; ++group)
            ProcessBucketRows({batch, static_cast<uint64_t>(pointers.GetValue(group)),
                               static_cast<uint64_t>(pointers.GetValue(group + 1))},
                              group, feature, valid);
    }
    __aicore__ inline void ProcessLoadedGroup(const BucketRows &range, uint32_t groups, uint64_t feature,
                                              uint32_t valid) {
        const uint32_t rows = range.last - range.first;
        if (rows) {
            LoadRows(range.batch, range.first, rows, feature, valid);
            PrepareFloatRows(rows);
        }
        if constexpr (Waves && Kind == Float32 && Reduction == Sum && !Scalar) {
            if (!t_->hasOut && valid == t_->tileWidth && groups <= MaxWaveOutputs) {
                AccumulateLoadedSumWaves(range, groups);
                Sync<HardEvent::S_MTE2>();
                Sync<HardEvent::V_MTE2>();
                return;
            }
        }
        auto pointers = groupPointers_.Get<int64_t>();
        for (uint32_t group = 0; group < groups; ++group) {
            const uint32_t first = pointers.GetValue(group) - range.first;
            const uint32_t count = pointers.GetValue(group + 1) - pointers.GetValue(group);
            if constexpr (Scalar) {
                InitializeScalar(group, valid);
                AccumulateScalar(group, first, count, valid);
                FinishScalar(group, valid, count);
            } else {
                InitializeFloat(group);
                AccumulateFloatRows(FloatRows(), first, count);
                FinishFloat(group, count);
            }
        }
        Sync<HardEvent::S_MTE2>();
        if constexpr (!Scalar)
            Sync<HardEvent::V_MTE2>();
    }
    __aicore__ inline void AccumulateLoadedSumWaves(const BucketRows &range, uint32_t groups) {
        auto pointers = groupPointers_.Get<int64_t>();
        uint32_t first[MaxWaveOutputs], count[MaxWaveOutputs], maximum = 0;
        for (uint32_t group = 0; group < groups; ++group) {
            first[group] = pointers.GetValue(group) - range.first;
            count[group] = pointers.GetValue(group + 1) - pointers.GetValue(group);
            if (count[group] > maximum)
                maximum = count[group];
        }
        auto output = output_.Get<float>();
        auto source = source_.Get<float>();
        // Each bucket starts at +0 and keeps the original FP32 recurrence.
        // Distinct buckets have disjoint accumulators, so only the next row
        // of the same bucket needs the barrier at the end of each wave.
        Duplicate(output, 0.0f, groups * t_->tileWidth);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < maximum; ++row) {
            for (uint32_t group = 0; group < groups; ++group) {
                if (row >= count[group])
                    continue;
                auto acc = output[group * t_->tileWidth];
                Add(acc, acc, source[(first[group] + row) * t_->tileWidth], t_->tileWidth);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void CopyFloat(LocalTensor<float> destination, LocalTensor<float> source) {
        Copy(destination.template ReinterpretCast<int32_t>(), source.template ReinterpretCast<int32_t>(),
             kVectorMaskElements, static_cast<uint8_t>(t_->tileWidth / kVectorMaskElements), {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void InitializeScalar(uint32_t group, uint32_t valid) {
        auto acc = accumulator_.Get<Value>();
        auto initial = output_.Get<Raw>()[group * t_->tileWidth];
        for (uint32_t col = 0; col < valid; ++col) {
            acc.SetValue(col, t_->hasOut ? Numeric::Decode(initial.GetValue(col))
                                         : (t_->srcElements ? Operation::Initial() : static_cast<Value>(0)));
            if constexpr (Extreme)
                arguments_.Get<int64_t>().SetValue(group * t_->tileWidth + col, t_->length);
        }
    }
    __aicore__ inline void InitializeFloat(uint32_t group) {
        auto acc = accumulator_.Get<float>();
        if (t_->hasOut) {
            if constexpr (Kind == Float32)
                CopyFloat(acc, output_.Get<float>()[group * t_->tileWidth]);
            else
                Cast(acc, output_.Get<T>()[group * t_->tileWidth], RoundMode::CAST_NONE, t_->tileWidth);
        } else
            Duplicate(acc, t_->srcElements ? Operation::Initial() : 0.0f, t_->tileWidth);
        if constexpr (Extreme)
            Duplicate(argumentFloat_.Get<float>(), static_cast<float>(static_cast<int32_t>(t_->length)), t_->tileWidth);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadRows(uint64_t batch, uint64_t first, uint32_t rows, uint64_t feature, uint32_t valid) {
        auto ids = rowIds_.Get<int64_t>();
        DataCopyPad(ids, rows_[first], {1, rows * 8, 0, 0, 0}, DataCopyPadExtParams<int64_t>(false, 0, 0, 0));
        Sync<HardEvent::MTE2_S>();
        const uint64_t firstRow = static_cast<uint64_t>(ids.GetValue(0));
        bool contiguous = static_cast<uint64_t>(ids.GetValue(rows - 1)) - firstRow + 1 == rows;
        // A prefetched group concatenates different buckets; only a complete
        // coordinate check can establish one contiguous source DMA span.
        for (uint32_t row = 1; contiguous && row < rows; ++row)
            contiguous = static_cast<uint64_t>(ids.GetValue(row)) == firstRow + row;
        const uint64_t gap = (t_->after - valid) * sizeof(Raw);
        const uint32_t transfer = (valid * sizeof(Raw) + kDmaBytes - 1) / kDmaBytes * kDmaBytes;
        const uint32_t stride = (t_->tileWidth * sizeof(Raw) - transfer) / kDmaBytes;
        if (contiguous && gap <= UINT32_MAX)
            DataCopyPad(source_.Get<Raw>(), src_[(batch * t_->length + firstRow) * t_->after + feature],
                        {static_cast<uint16_t>(rows), valid * static_cast<uint32_t>(sizeof(Raw)),
                         static_cast<uint32_t>(gap), stride, 0},
                        DataCopyPadExtParams<Raw>(false, 0, 0, 0));
        else
            for (uint32_t row = 0; row < rows; ++row)
                DataCopyPad(source_.Get<Raw>()[row * t_->tileWidth],
                            src_[(batch * t_->length + ids.GetValue(row)) * t_->after + feature],
                            {1, valid * static_cast<uint32_t>(sizeof(Raw)), 0, 0, 0},
                            DataCopyPadExtParams<Raw>(false, 0, 0, 0));
        if constexpr (Scalar)
            Sync<HardEvent::MTE2_S>();
        else
            Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline void AccumulateScalar(uint32_t group, uint32_t first, uint32_t rows, uint32_t valid) {
        auto source = source_.Get<Raw>();
        auto acc = accumulator_.Get<Value>();
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < valid; ++col) {
                Value value = acc.GetValue(col);
                const bool changed =
                    Operation::Update(value, Numeric::Decode(source.GetValue((first + row) * t_->tileWidth + col)));
                acc.SetValue(col, value);
                if constexpr (Extreme)
                    if (changed)
                        arguments_.Get<int64_t>().SetValue(group * t_->tileWidth + col,
                                                           rowIds_.Get<int64_t>().GetValue(first + row));
            }
        }
    }
    __aicore__ inline void PrepareFloatRows(uint32_t rows) {
        if constexpr (!Scalar && Kind != Float32) {
            Cast(work_.Get<float>(), source_.Get<T>(), RoundMode::CAST_NONE, rows * t_->tileWidth);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline LocalTensor<float> FloatRows() {
        if constexpr (Kind == Float32)
            return source_.Get<float>();
        else
            return work_.Get<float>();
    }
    __aicore__ inline void FloatExtreme(LocalTensor<float> source, int64_t row) {
        auto acc = accumulator_.Get<float>();
        auto mask = mask_.Get<uint8_t>();
        Compare(mask, source, acc, Reduction == Min ? CMPMODE::LT : CMPMODE::GT, t_->tileWidth);
        Duplicate(rowPosition_.Get<float>(), static_cast<float>(static_cast<int32_t>(row)), t_->tileWidth);
        PipeBarrier<PIPE_V>();
        Select(acc, mask, source, acc, SELMODE::VSEL_TENSOR_TENSOR_MODE, t_->tileWidth);
        Select(argumentFloat_.Get<float>(), mask, rowPosition_.Get<float>(), argumentFloat_.Get<float>(),
               SELMODE::VSEL_TENSOR_TENSOR_MODE, t_->tileWidth);
    }
    __aicore__ inline void AccumulateFloatRows(LocalTensor<float> source, uint32_t first, uint32_t rows) {
        auto acc = accumulator_.Get<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            auto values = source[(first + row) * t_->tileWidth];
            if constexpr (Reduction == Sum || Reduction == Mean)
                Add(acc, acc, values, t_->tileWidth);
            else if constexpr (Reduction == Mul) {
                AscendC::Mul(acc, acc, values, t_->tileWidth);
                if constexpr (Kind != Float32) {
                    PipeBarrier<PIPE_V>();
                    Cast(source_.Get<T>(), acc, RoundMode::CAST_RINT, t_->tileWidth);
                    PipeBarrier<PIPE_V>();
                    Cast(acc, source_.Get<T>(), RoundMode::CAST_NONE, t_->tileWidth);
                }
            } else
                FloatExtreme(values, rowIds_.Get<int64_t>().GetValue(first + row));
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline Value ScalarMean(Value value, uint64_t rows) {
        if constexpr (Reduction != Mean)
            return value;
        else {
            const uint64_t count = t_->indexElements == 1 && rows ? 1 : rows;
            const Value denominator = Numeric::Decode(static_cast<Raw>(count));
            const Value divisor = denominator < 1 ? 1 : denominator;
            const Value numerator = Numeric::Decode(Numeric::Encode(value));
            const Value quotient = numerator / divisor;
            return quotient - (numerator < 0 && numerator % divisor != 0 ? 1 : 0);
        }
    }
    __aicore__ inline void FinishScalar(uint32_t group, uint32_t valid, uint64_t rows) {
        auto acc = accumulator_.Get<Value>();
        auto output = output_.Get<Raw>()[group * t_->tileWidth];
        for (uint32_t col = 0; col < valid; ++col) {
            Value value = ScalarMean(acc.GetValue(col), rows);
            if constexpr (Extreme)
                if (!t_->hasOut && value == Operation::Initial())
                    value = 0;
            output.SetValue(col, Numeric::Encode(value));
        }
    }
    __aicore__ inline void FloatMean(uint64_t rows) {
        auto acc = accumulator_.Get<float>();
        if constexpr (Kind != Float32) {
            Cast(source_.Get<T>(), acc, RoundMode::CAST_RINT, t_->tileWidth);
            PipeBarrier<PIPE_V>();
            Cast(acc, source_.Get<T>(), RoundMode::CAST_NONE, t_->tileWidth);
            PipeBarrier<PIPE_V>();
        }
        const uint64_t physicalCount = t_->indexElements == 1 && rows ? 1 : rows;
        const float count = static_cast<float>(static_cast<int32_t>(Smaller(physicalCount, kExactFloatIndexLimit)));
        const float narrowed = Numeric::Decode(Numeric::Encode(count));
        Duplicate(denominator_.Get<float>(), narrowed < 1 ? 1.0f : narrowed, t_->tileWidth);
        PipeBarrier<PIPE_V>();
        Div(acc, acc, denominator_.Get<float>(), t_->tileWidth);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void FinishFloat(uint32_t group, uint64_t rows) {
        auto acc = accumulator_.Get<float>();
        if constexpr (Reduction == Mean)
            FloatMean(rows);
        if constexpr (Extreme) {
            if (!t_->hasOut) {
                CompareScalar(mask_.Get<uint8_t>(), acc, Operation::Initial(), CMPMODE::NE, t_->tileWidth);
                PipeBarrier<PIPE_V>();
                Select(acc, mask_.Get<uint8_t>(), acc, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, t_->tileWidth);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t col = 0; col < t_->tileWidth; col += kVectorMaskElements)
                Cast(arguments_.Get<int64_t>()[group * t_->tileWidth + col], argumentFloat_.Get<float>()[col],
                     RoundMode::CAST_RINT, kVectorMaskElements);
        }
        if constexpr (Kind == Float32)
            CopyFloat(output_.Get<float>()[group * t_->tileWidth], acc);
        else
            Cast(output_.Get<T>()[group * t_->tileWidth], acc, RoundMode::CAST_RINT, t_->tileWidth);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ProcessBucket(uint64_t task, uint32_t group, uint64_t feature, uint32_t valid) {
        const uint64_t bucket = task % t_->buckets;
        ProcessBucketRows({task / t_->buckets, static_cast<uint64_t>(rowptr_.GetValue(bucket)),
                           static_cast<uint64_t>(rowptr_.GetValue(bucket + 1))},
                          group, feature, valid);
    }
    __aicore__ inline void ProcessBucketRows(const BucketRows &range, uint32_t group, uint64_t feature,
                                             uint32_t valid) {
        if constexpr (Scalar)
            InitializeScalar(group, valid);
        else
            InitializeFloat(group);
        for (uint64_t row = range.first; row < range.last; row += t_->tileRows) {
            const uint32_t count = Smaller(t_->tileRows, range.last - row);
            LoadRows(range.batch, row, count, feature, valid);
            if constexpr (Scalar)
                AccumulateScalar(group, 0, count, valid);
            else {
                PrepareFloatRows(count);
                AccumulateFloatRows(FloatRows(), 0, count);
            }
            Sync<HardEvent::S_MTE2>();
            if constexpr (!Scalar)
                Sync<HardEvent::V_MTE2>();
        }
        if constexpr (Scalar)
            FinishScalar(group, valid, range.last - range.first);
        else {
            FinishFloat(group, range.last - range.first);
            if constexpr (Reduction == Mean && Kind != Float32)
                Sync<HardEvent::V_MTE2>();
        }
    }
    template <typename U>
    __aicore__ inline void StoreRows(GlobalTensor<U> &destination, LocalTensor<U> source, uint64_t offset,
                                     uint32_t groups, uint32_t valid) {
        const uint32_t transfer = (valid * sizeof(U) + kDmaBytes - 1) / kDmaBytes * kDmaBytes;
        const uint32_t stride = (t_->tileWidth * sizeof(U) - transfer) / kDmaBytes;
        const uint64_t gap = (t_->after - valid) * sizeof(U);
        if (gap <= UINT32_MAX)
            DataCopyPad(destination[offset], source,
                        {static_cast<uint16_t>(groups), valid * static_cast<uint32_t>(sizeof(U)), stride,
                         static_cast<uint32_t>(gap), 0});
        else
            for (uint32_t group = 0; group < groups; ++group)
                DataCopyPad(destination[offset + group * t_->after], source[group * t_->tileWidth],
                            {1, valid * static_cast<uint32_t>(sizeof(U)), 0, 0, 0});
    }
    __aicore__ inline void StoreGroup(uint64_t first, uint32_t groups, uint64_t feature, uint32_t valid) {
        if constexpr (Scalar)
            Sync<HardEvent::S_MTE3>();
        else
            Sync<HardEvent::V_MTE3>();
        StoreRows(out_, output_.Get<Raw>(), first * t_->after + feature, groups, valid);
        if constexpr (Extreme)
            StoreRows(arg_, arguments_.Get<int64_t>(), first * t_->after + feature, groups, valid);
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_MTE2>();
    }
    const GroupTiling *t_;
    GlobalTensor<Raw> src_, out_;
    GlobalTensor<int64_t> rowptr_, rows_, arg_;
    TPipe &pipe_;
    TBuf<TPosition::VECCALC> source_, output_, rowIds_, accumulator_, work_, arguments_;
    TBuf<TPosition::VECCALC> argumentFloat_, rowPosition_, mask_, denominator_, groupPointers_, validation_;
};
} // namespace opsgnn::scatter_a2

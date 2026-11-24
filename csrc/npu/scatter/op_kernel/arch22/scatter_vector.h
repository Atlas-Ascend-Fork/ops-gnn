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
// A core owns one aligned feature tile for every bucket. Sequential row DMA
// preserves source order and strict arg ties without an atomic operation.
template <uint32_t Kind, uint32_t Reduction> class VectorScatter {
    using T = typename std::conditional<Kind == Float32, float,
                                        typename std::conditional<Kind == Float16, half, bfloat16_t>::type>::type;
    using Operation = Reducer<Kind, Reduction>;
    using Raw = typename Number<Kind>::Raw;

  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, GM_ADDR workspace,
                                GM_ADDR argWorkspace, GM_ADDR counts, const Tiling &t) {
        t_ = &t;
        if (t.reduce == Mean)
            counts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(counts), t.buckets);
        rawSource_.SetGlobalBuffer(reinterpret_cast<__gm__ Raw *>(src), t.srcElements);
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), t.srcElements);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), t.outElements);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), t.indexElements);
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace), t.outElements);
        if constexpr (Operation::extreme) {
            arg_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(arg), t.outElements);
            argWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(argWorkspace), t.outElements);
        }
        pipe_.InitBuffer(source_, kOutputTileElements * sizeof(T));
        pipe_.InitBuffer(accumulator_, kOutputTileElements * sizeof(float));
        pipe_.InitBuffer(work_, kOutputTileElements * sizeof(float));
        pipe_.InitBuffer(argument_, kOutputTileElements * sizeof(float));
        pipe_.InitBuffer(position_, kOutputTileElements * sizeof(float));
        pipe_.InitBuffer(argument64_, kOutputTileElements * sizeof(int64_t));
        pipe_.InitBuffer(mask_,
                         (kOutputTileElements + kMaskElementsPerDmaBlock - 1) / kMaskElementsPerDmaBlock * kDmaBytes);
    }
    __aicore__ inline void Process(uint32_t stage) {
        if (stage == 0 && t_->reduce == Mean)
            InitializeScatterCounts(counts_, position_.Get<int32_t>(), t_->buckets);
        if (stage == 1) {
            for (uint64_t unit = GetBlockIdx(); unit < t_->units; unit += GetBlockNum())
                UpdateTile(unit);
            return;
        }
        const uint64_t tiles = (t_->outElements + kOutputTileElements - 1) / kOutputTileElements;
        for (uint64_t tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum()) {
            const uint64_t offset = tile * kOutputTileElements;
            const uint32_t valid = Smaller(kOutputTileElements, t_->outElements - offset);
            if (stage == 0)
                InitializeTile(offset, valid);
            else
                FinalizeTile(offset, valid);
        }
    }

  private:
    __aicore__ inline void InitializeTile(uint64_t offset, uint32_t valid) {
        auto acc = accumulator_.Get<float>();
        if (t_->hasOut) {
            if constexpr (Kind == Float32)
                LoadScatterTile(acc, out_, offset, valid);
            else
                LoadScatterTile(source_.Get<T>(), out_, offset, valid);
            Sync<HardEvent::MTE2_V>();
            if constexpr (Kind != Float32)
                Cast(acc, source_.Get<T>(), RoundMode::CAST_NONE, kOutputTileElements);
        } else
            Duplicate(acc, t_->srcElements ? Operation::Initial() : 0.0f, kOutputTileElements);
        SanitizeInitial();
        if constexpr (Operation::extreme)
            Duplicate(argument_.Get<float>(), static_cast<float>(static_cast<int32_t>(t_->length)),
                      kOutputTileElements);
        Sync<HardEvent::V_MTE3>();
        StoreScatterTile(workspace_, acc, offset, valid);
        if constexpr (Operation::extreme)
            StoreScatterTile(argWorkspace_, argument_.Get<float>(), offset, valid);
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::V_MTE2>();
    }
    __aicore__ inline void SanitizeInitial() {
        if constexpr (Operation::extreme) {
            if (t_->path == Atomic && t_->hasOut) {
                auto acc = accumulator_.Get<float>();
                auto mask = mask_.Get<uint8_t>();
                PipeBarrier<PIPE_V>();
                Compare(mask, acc, acc, CMPMODE::EQ, kOutputTileElements);
                PipeBarrier<PIPE_V>();
                const float blocker = Reduction == Min ? -__builtin_inff() : __builtin_inff();
                Select(acc, mask, acc, blocker, SELMODE::VSEL_TENSOR_SCALAR_MODE, kOutputTileElements);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline LocalTensor<float> SourceValues(uint32_t count = kFeatureTileElements) {
        if constexpr (Kind == Float32)
            return source_.Get<float>();
        else {
            auto work = work_.Get<float>();
            Cast(work, source_.Get<T>(), RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            return work;
        }
    }
    __aicore__ inline void ReduceRow(uint64_t row) {
        auto acc = accumulator_.Get<float>();
        auto source = SourceValues();
        if constexpr (Reduction == Sum)
            Add(acc, acc, source, kFeatureTileElements);
        else if constexpr (Reduction == Mul) {
            AscendC::Mul(acc, acc, source, kFeatureTileElements);
            if constexpr (Kind != Float32) {
                PipeBarrier<PIPE_V>();
                Cast(source_.Get<T>(), acc, RoundMode::CAST_RINT, kFeatureTileElements);
                PipeBarrier<PIPE_V>();
                Cast(acc, source_.Get<T>(), RoundMode::CAST_NONE, kFeatureTileElements);
            }
        } else {
            auto mask = mask_.Get<uint8_t>();
            auto argument = argument_.Get<float>();
            Compare(mask, source, acc, Reduction == Min ? CMPMODE::LT : CMPMODE::GT, kFeatureTileElements);
            Duplicate(position_.Get<float>(), static_cast<float>(static_cast<int32_t>(row)), kFeatureTileElements);
            PipeBarrier<PIPE_V>();
            Select(acc, mask, source, acc, SELMODE::VSEL_TENSOR_TENSOR_MODE, kFeatureTileElements);
            Select(argument, mask, position_.Get<float>(), argument, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   kFeatureTileElements);
        }
    }
    __aicore__ inline void UpdateRow(uint64_t batch, uint64_t row, uint64_t feature, uint32_t valid) {
        const int64_t bucket = index_.GetValue(t_->indexElements == 1 ? 0 : row);
        if (bucket < 0 || static_cast<uint64_t>(bucket) >= t_->buckets)
            return;
        const uint64_t offset = (batch * t_->buckets + bucket) * t_->after + feature;
        LoadScatterTile(source_.Get<T>(), src_, (batch * t_->length + row) * t_->after + feature, valid);
        LoadScatterTile(accumulator_.Get<float>(), workspace_, offset, valid);
        if constexpr (Operation::extreme)
            LoadScatterTile(argument_.Get<float>(), argWorkspace_, offset, valid);
        Sync<HardEvent::MTE2_V>();
        ReduceRow(row);
        Sync<HardEvent::V_MTE3>();
        StoreScatterTile(workspace_, accumulator_.Get<float>(), offset, valid);
        if constexpr (Operation::extreme)
            StoreScatterTile(argWorkspace_, argument_.Get<float>(), offset, valid);
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::V_MTE2>();
    }
    __aicore__ inline void UpdateTile(uint64_t unit) {
        const uint64_t tiles = (t_->after + kFeatureTileElements - 1) / kFeatureTileElements;
        const uint64_t batch = unit / tiles, feature = unit % tiles * kFeatureTileElements;
        const uint32_t valid = Smaller(kFeatureTileElements, t_->after - feature);
        for (uint64_t row = 0; row < t_->length; ++row)
            UpdateRow(batch, row, feature, valid);
    }
    __aicore__ inline void ClearIdentity() {
        if constexpr (Operation::extreme) {
            if (!t_->hasOut) {
                auto acc = accumulator_.Get<float>();
                auto mask = mask_.Get<uint8_t>();
                CompareScalar(mask, acc, Operation::Initial(), CMPMODE::NE, kOutputTileElements);
                PipeBarrier<PIPE_V>();
                Select(acc, mask, acc, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, kOutputTileElements);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void RestoreUnchanged(uint64_t offset, uint32_t valid) {
        if (!t_->hasOut) {
            ClearIdentity();
            return;
        }
        LoadScatterTile(source_.Get<T>(), out_, offset, valid);
        Sync<HardEvent::MTE2_V>();
        auto initial = SourceValues(kOutputTileElements);
        auto mask = mask_.Get<uint8_t>();
        auto acc = accumulator_.Get<float>();
        CompareScalar(mask, argument_.Get<float>(), static_cast<float>(static_cast<int32_t>(t_->length)), CMPMODE::EQ,
                      kOutputTileElements);
        PipeBarrier<PIPE_V>();
        Select(acc, mask, initial, acc, SELMODE::VSEL_TENSOR_TENSOR_MODE, kOutputTileElements);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void RestoreZeroWord(uint64_t offset, uint32_t valid, uint32_t word) {
        const uint64_t zeroMask = mask_.Get<uint64_t>().GetValue(word);
        if (!zeroMask)
            return;
        auto arguments = argument64_.Get<int64_t>();
        auto raw = Kind == Float32 ? accumulator_.Get<Raw>() : source_.Get<Raw>();
        const uint32_t first = word * kVectorMaskElements;
        const uint32_t count = Smaller(kVectorMaskElements, valid - first);
        for (uint32_t lane = 0; lane < count; ++lane) {
            if (!((zeroMask >> lane) & 1U))
                continue;
            const uint64_t row = static_cast<uint64_t>(arguments.GetValue(first + lane));
            if (row >= t_->length)
                continue;
            const uint64_t destination = offset + first + lane;
            const uint64_t batch = destination / (t_->buckets * t_->after);
            raw.SetValue(first + lane,
                         rawSource_.GetValue((batch * t_->length + row) * t_->after + destination % t_->after));
        }
    }
    __aicore__ inline void RestoreSignedZero(uint64_t offset, uint32_t valid) {
        CompareScalar(mask_.Get<uint8_t>(), accumulator_.Get<float>(), 0.0f, CMPMODE::EQ, kOutputTileElements);
        Sync<HardEvent::V_S>();
        for (uint32_t word = 0; word < (valid + kVectorMaskElements - 1) / kVectorMaskElements; ++word)
            RestoreZeroWord(offset, valid, word);
    }
    __aicore__ inline void ConvertArguments() {
        // A2 float-to-int64 destination masks contain at most 64 lanes.
        for (uint32_t first = 0; first < kOutputTileElements; first += kVectorMaskElements)
            Cast(argument64_.Get<int64_t>()[first], argument_.Get<float>()[first], RoundMode::CAST_RINT,
                 kVectorMaskElements);
    }
    __aicore__ inline void FinalizeAtomic(uint64_t offset, uint32_t valid) {
        RestoreUnchanged(offset, valid);
        ConvertArguments();
        if constexpr (Kind != Float32)
            Cast(source_.Get<T>(), accumulator_.Get<float>(), RoundMode::CAST_RINT, kOutputTileElements);
        PipeBarrier<PIPE_V>();
        RestoreSignedZero(offset, valid);
        Sync<HardEvent::S_MTE3>();
        Sync<HardEvent::V_MTE3>();
        if constexpr (Kind == Float32)
            StoreScatterTile(out_, accumulator_.Get<float>(), offset, valid);
        else
            StoreScatterTile(out_, source_.Get<T>(), offset, valid);
        StoreScatterTile(arg_, argument64_.Get<int64_t>(), offset, valid);
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::V_MTE2>();
    }
    __aicore__ inline void MeanDenominator(uint64_t offset, uint32_t valid) {
        auto counts = position_.Get<float>();
        for (uint32_t first = 0; first < valid;) {
            const uint64_t position = offset + first;
            const uint64_t bucket = position / t_->after % t_->buckets;
            const uint32_t width = Smaller(t_->after - position % t_->after, valid - first);
            const float count = static_cast<float>(counts_.GetValue(bucket));
            Duplicate(counts[first], count, width);
            first += width;
        }
        PipeBarrier<PIPE_V>();
        if constexpr (Kind != Float32) {
            Cast(source_.Get<T>(), counts, RoundMode::CAST_RINT, kOutputTileElements);
            PipeBarrier<PIPE_V>();
            Cast(counts, source_.Get<T>(), RoundMode::CAST_NONE, kOutputTileElements);
            PipeBarrier<PIPE_V>();
        }
        Maxs(counts, counts, 1.0f, kOutputTileElements);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void FinalizeMean(uint64_t offset, uint32_t valid) {
        auto acc = accumulator_.Get<float>();
        // CPU mean first materializes its source-dtype sum, including overflow.
        if constexpr (Kind != Float32) {
            Cast(source_.Get<T>(), acc, RoundMode::CAST_RINT, kOutputTileElements);
            PipeBarrier<PIPE_V>();
            Cast(acc, source_.Get<T>(), RoundMode::CAST_NONE, kOutputTileElements);
            PipeBarrier<PIPE_V>();
        }
        MeanDenominator(offset, valid);
        Div(acc, acc, position_.Get<float>(), kOutputTileElements);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void FinalizeTile(uint64_t offset, uint32_t valid) {
        auto acc = accumulator_.Get<float>();
        LoadScatterTile(acc, workspace_, offset, valid);
        if constexpr (Operation::extreme)
            LoadScatterTile(argument_.Get<float>(), argWorkspace_, offset, valid);
        Sync<HardEvent::MTE2_V>();
        if constexpr (Operation::extreme) {
            if (t_->path == Atomic) {
                FinalizeAtomic(offset, valid);
                return;
            }
        }
        ClearIdentity();
        if (t_->reduce == Mean)
            FinalizeMean(offset, valid);
        if constexpr (Kind != Float32)
            Cast(source_.Get<T>(), acc, RoundMode::CAST_RINT, kOutputTileElements);
        if constexpr (Operation::extreme)
            ConvertArguments();
        Sync<HardEvent::V_MTE3>();
        if constexpr (Kind == Float32)
            StoreScatterTile(out_, acc, offset, valid);
        else
            StoreScatterTile(out_, source_.Get<T>(), offset, valid);
        if constexpr (Operation::extreme)
            StoreScatterTile(arg_, argument64_.Get<int64_t>(), offset, valid);
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::V_MTE2>();
    }
    const Tiling *t_;
    GlobalTensor<T> src_, out_;
    GlobalTensor<Raw> rawSource_;
    GlobalTensor<int64_t> index_, arg_;
    GlobalTensor<int32_t> counts_;
    GlobalTensor<float> workspace_, argWorkspace_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> source_, accumulator_, work_, argument_, position_, argument64_, mask_;
};
} // namespace opsgnn::scatter_a2

namespace opsgnn::scatter_a2 {

// Compact source tiles issue one batch of independent atomic writes. The arg
// pass compares a complete tile before storing any row, preserving strict CPU
// first-winner semantics without a DMA/vector rendezvous for every source row.
template <uint32_t Kind, uint32_t Reduction> class AtomicScatter {
    using T = typename std::conditional<Kind == Float32, float,
                                        typename std::conditional<Kind == Float16, half, bfloat16_t>::type>::type;
    static constexpr bool Extreme = Reduction == Min || Reduction == Max;

  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR workspace, GM_ADDR argWorkspace,
                                const Tiling &t, bool arguments) {
        t_ = &t;
        arguments_ = arguments;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), t.srcElements);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), t.outElements);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), t.indexElements);
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace), t.outElements);
        if constexpr (Extreme)
            argWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(argWorkspace), t.outElements);
        const uint32_t elements = t.atomicWidth * t.atomicRows;
        pipe_.InitBuffer(source_, elements * sizeof(T));
        if constexpr (Kind != Float32)
            pipe_.InitBuffer(work_, elements * sizeof(float));
        pipe_.InitBuffer(indices_, ((t.atomicRows * 8 + kDmaBytes - 1) / kDmaBytes) * kDmaBytes);
        if constexpr (Extreme) {
            pipe_.InitBuffer(mask_, ((elements + kMaskElementsPerDmaBlock - 1) / kMaskElementsPerDmaBlock) * kDmaBytes);
            if (arguments)
                InitArgumentBuffers(elements);
        }
    }
    __aicore__ inline void Process() {
        if (arguments_)
            SetAtomicMin<float>();
        else if constexpr (Reduction == Min)
            SetAtomicMin<float>();
        else
            SetAtomicMax<float>();
        for (uint64_t unit = GetBlockIdx(); unit < t_->units; unit += GetBlockNum())
            ProcessTile(DescribeSourceTile(unit, t_));
        Sync<HardEvent::MTE3_S>();
        SetAtomicNone();
    }

  private:
    __aicore__ inline void InitArgumentBuffers(uint32_t elements) {
        pipe_.InitBuffer(extrema_, elements * sizeof(float));
        pipe_.InitBuffer(position_, elements * sizeof(float));
        if (t_->hasOut) {
            pipe_.InitBuffer(initial_, elements * sizeof(T));
            if constexpr (Kind != Float32)
                pipe_.InitBuffer(initialWork_, elements * sizeof(float));
        }
    }
    __aicore__ inline uint64_t Destination(const SourceTile &tile, uint32_t row) {
        const int64_t bucket = indices_.Get<int64_t>().GetValue(t_->indexElements == 1 ? 0 : row);
        if (bucket < 0 || static_cast<uint64_t>(bucket) >= t_->buckets)
            return UINT64_MAX;
        return (tile.batch * t_->buckets + bucket) * t_->after + tile.feature;
    }
    __aicore__ inline void LoadSource(const SourceTile &tile) {
        const uint64_t offset = (tile.batch * t_->length + tile.first) * t_->after + tile.feature;
        const uint32_t transferred = (tile.valid * sizeof(T) + kDmaBytes - 1) / kDmaBytes * kDmaBytes;
        const uint32_t stride = (t_->atomicWidth * sizeof(T) - transferred) / kDmaBytes;
        const uint64_t gap = (t_->after - tile.valid) * sizeof(T);
        const auto padding = DataCopyPadExtParams<T>(false, 0, 0, 0);
        if (gap <= UINT32_MAX)
            DataCopyPad(source_.Get<T>(), src_[offset],
                        {static_cast<uint16_t>(tile.rows), tile.valid * static_cast<uint32_t>(sizeof(T)),
                         static_cast<uint32_t>(gap), stride, 0},
                        padding);
        else
            for (uint32_t row = 0; row < tile.rows; ++row)
                DataCopyPad(source_.Get<T>()[row * t_->atomicWidth], src_[offset + row * t_->after],
                            {1, tile.valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}, padding);
        const uint32_t indexCount = t_->indexElements == 1 ? 1 : tile.rows;
        DataCopyPad(indices_.Get<int64_t>(), index_[t_->indexElements == 1 ? 0 : tile.first],
                    {1, indexCount * 8, 0, 0, 0}, DataCopyPadExtParams<int64_t>(false, 0, 0, 0));
        Sync<HardEvent::MTE2_S>();
    }
    __aicore__ inline void LoadTargets(const SourceTile &tile) {
        for (uint32_t row = 0; row < tile.rows; ++row) {
            const uint64_t destination = Destination(tile, row);
            if (destination == UINT64_MAX)
                continue;
            DataCopyPad(extrema_.Get<float>()[row * t_->atomicWidth], workspace_[destination],
                        {1, tile.valid * 4, 0, 0, 0}, DataCopyPadExtParams<float>(false, 0, 0, 0));
            if (t_->hasOut)
                DataCopyPad(initial_.Get<T>()[row * t_->atomicWidth], out_[destination],
                            {1, tile.valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
                            DataCopyPadExtParams<T>(false, 0, 0, 0));
        }
    }
    __aicore__ inline LocalTensor<float> FloatSource(uint32_t elements) {
        if constexpr (Kind == Float32)
            return source_.Get<float>();
        else {
            Cast(work_.Get<float>(), source_.Get<T>(), RoundMode::CAST_NONE, elements);
            PipeBarrier<PIPE_V>();
            return work_.Get<float>();
        }
    }
    __aicore__ inline LocalTensor<float> FloatInitial(uint32_t elements) {
        if constexpr (Kind == Float32)
            return initial_.Get<float>();
        else {
            Cast(initialWork_.Get<float>(), initial_.Get<T>(), RoundMode::CAST_NONE, elements);
            PipeBarrier<PIPE_V>();
            return initialWork_.Get<float>();
        }
    }
    __aicore__ inline void Sanitize(LocalTensor<float> source, uint32_t count) {
        if constexpr (Extreme) {
            auto mask = mask_.Get<uint8_t>();
            Compare(mask, source, source, CMPMODE::EQ, count);
            PipeBarrier<PIPE_V>();
            Select(source, mask, source, Reduction == Min ? __builtin_inff() : -__builtin_inff(),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void PrepareArguments(LocalTensor<float> source, const SourceTile &tile) {
        const uint32_t count = tile.rows * t_->atomicWidth;
        auto positions = position_.Get<float>();
        auto mask = mask_.Get<uint8_t>();
        const float sentinel = static_cast<float>(static_cast<int32_t>(t_->length));
        for (uint32_t row = 0; row < tile.rows; ++row)
            Duplicate(positions[row * t_->atomicWidth], static_cast<float>(static_cast<int32_t>(tile.first + row)),
                      t_->atomicWidth);
        Compare(mask, source, extrema_.Get<float>(), CMPMODE::EQ, count);
        PipeBarrier<PIPE_V>();
        Select(positions, mask, positions, sentinel, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        PipeBarrier<PIPE_V>();
        if (t_->hasOut)
            Compare(mask, source, FloatInitial(count), Reduction == Min ? CMPMODE::LT : CMPMODE::GT, count);
        else
            CompareScalar(mask, source, Number<Kind>::Limit(Reduction == Min),
                          Reduction == Min ? CMPMODE::LT : CMPMODE::GT, count);
        PipeBarrier<PIPE_V>();
        Select(positions, mask, positions, sentinel, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    }
    __aicore__ inline void WriteRows(LocalTensor<float> source, const SourceTile &tile) {
        for (uint32_t row = 0; row < tile.rows; ++row) {
            const uint64_t destination = Destination(tile, row);
            if (destination == UINT64_MAX)
                continue;
            if constexpr (Extreme) {
                if (arguments_) {
                    DataCopyPad(argWorkspace_[destination], position_.Get<float>()[row * t_->atomicWidth],
                                {1, tile.valid * 4, 0, 0, 0});
                    continue;
                }
            }
            DataCopyPad(workspace_[destination], source[row * t_->atomicWidth], {1, tile.valid * 4, 0, 0, 0});
        }
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::S_MTE2>();
    }
    __aicore__ inline void ProcessTile(const SourceTile &tile) {
        LoadSource(tile);
        if constexpr (Extreme)
            if (arguments_)
                LoadTargets(tile);
        Sync<HardEvent::MTE2_V>();
        auto source = FloatSource(tile.rows * t_->atomicWidth);
        if constexpr (Extreme) {
            if (arguments_)
                PrepareArguments(source, tile);
            else
                Sanitize(source, tile.rows * t_->atomicWidth);
        }
        Sync<HardEvent::V_MTE3>();
        WriteRows(source, tile);
        Sync<HardEvent::V_MTE2>();
    }
    const Tiling *t_;
    bool arguments_;
    GlobalTensor<T> src_, out_;
    GlobalTensor<int64_t> index_;
    GlobalTensor<float> workspace_, argWorkspace_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> source_, work_, indices_, extrema_, initial_, initialWork_, position_, mask_;
};
} // namespace opsgnn::scatter_a2

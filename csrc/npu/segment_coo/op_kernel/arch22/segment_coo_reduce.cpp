/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "segment_coo_reduce.h"
#include <type_traits>

namespace opsgnn {

// ============================================================================
// Shared floating and INT32 device reduction
// ============================================================================

using namespace AscendC;

// Column widths, row tiles and ownership come from the shared capacity planner.
// No branch is selected by a particular input K.
template <typename T, SegmentReduce Reduce, bool HasArg> class SegmentReduceDevice {
    static constexpr bool Floating =
        std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value;
    static constexpr bool Extreme = Reduce == SegmentMin || Reduce == SegmentMax;
    static constexpr bool Compensated = Floating && !Extreme;
    static constexpr bool BatchType = !Extreme && (std::is_same<T, float>::value || std::is_same<T, half>::value);
    static constexpr bool ExtremeBatchType =
        Extreme && !HasArg && (std::is_same<T, float>::value || std::is_same<T, half>::value);
    using Work = typename std::conditional<Floating, float, int32_t>::type;

  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, const SegmentTilingData &t) {
        t_ = &t;
        // Finite FP16 values are exact FP32 multiples of 2^-24. With at
        // most 2^32 source rows their absolute sum is below 2^48, leaving
        // ample FP32 range for the unchanged compensated tree and tile
        // merges. No initial out participates in this bounded path. NaN/Inf
        // propagation and final nonfinite-residual cleanup remain unchanged.
        if constexpr (std::is_same<T, half>::value && Compensated)
            boundedUnscaled_ = !t.hasInitial && t.M <= (1ULL << 32);
        if constexpr (std::is_same<T, float>::value && Compensated)
            boundedUnscaled_ = !t.hasInitial && (t.sourceFlags & kSourceRangeProven);
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), t.srcNumel);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), t.indexNumel);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), t.outNumel);
        if constexpr (HasArg)
            arg_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(arg), t.outNumel);
        const uint32_t k = t.tileK;
        pipe_.InitBuffer(srcBuf_, t.tileRows * k * sizeof(T));
        pipe_.InitBuffer(accBuf_, k * sizeof(Work));
        pipe_.InitBuffer(outBuf_, k * sizeof(T));
        if constexpr (Floating && !std::is_same<T, float>::value)
            pipe_.InitBuffer(workBuf_, t.tileRows * k * sizeof(Work));
        if constexpr (ExtremeBatchType && std::is_same<T, float>::value) {
            if (t.batchReduce)
                pipe_.InitBuffer(workBuf_, (t.tileRows / 2) * k * sizeof(float));
        }
        InitCompensationBuffers(t, k);
        if constexpr (Extreme && !HasArg)
            pipe_.InitBuffer(maskBuf_, ((t.tileRows * k + 7) / 8 + 31) / 32 * 32);
        InitArgBuffers(k);
        mte2v_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
        vmte2_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
        vmte3_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
        mte3v_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
        vs_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_S>());
        sv_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::S_V>());
        smte3_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::S_MTE3>());
    }

  private:
    __aicore__ inline void InitCompensationBuffers(const SegmentTilingData &t, uint32_t k) {
        if constexpr (Compensated) {
            const uint32_t scratch = (t.sourceFlags & kCpuOrderedFloat) ? k : (t.tileRows / 2 ? t.tileRows / 2 : 1) * k;
            // Input rows have implicit zero low parts. Only the first tree
            // level's surviving half needs stored corrections.
            if constexpr (std::is_same<T, half>::value) {
                // In the batch plan Cast has consumed srcBuf before any low
                // part is written. Its half tile holds half a float tile.
                if (!t.batchReduce || t.tileRows < 2)
                    pipe_.InitBuffer(correctionBuf_, scratch * sizeof(float));
            } else
                pipe_.InitBuffer(correctionBuf_, scratch * sizeof(float));
            pipe_.InitBuffer(sumScratchBuf_, scratch * sizeof(float));
            pipe_.InitBuffer(zScratchBuf_, scratch * sizeof(float));
            pipe_.InitBuffer(accCorrectionBuf_, k * sizeof(float));
            pipe_.InitBuffer(finiteMaskBuf_, ((scratch + 7) / 8 + 31) / 32 * 32);
        }
    }
    __aicore__ inline void InitArgBuffers(uint32_t k) {
        if constexpr (HasArg) {
            pipe_.InitBuffer(extremeScratchBuf_, k * sizeof(Work));
            pipe_.InitBuffer(argFBuf_, k * sizeof(float));
            pipe_.InitBuffer(arg64Buf_, k * sizeof(int64_t));
            const uint32_t maskLanes = k;
            const uint32_t maskBytes = ((maskLanes + 7) / 8 + 31) / 32 * 32;
            pipe_.InitBuffer(maskBuf_, maskBytes);
            pipe_.InitBuffer(nanMaskBuf_, maskBytes);
        }
    }

  public:
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        for (uint64_t unit = begin; unit < end; ++unit) {
            ProcessUnit(unit);
        }
    }

  private:
    __aicore__ inline void ProcessUnit(uint64_t unit) {
        if (t_->outputGroup == 1) {
            if constexpr (BatchType) {
                if (t_->batchReduce && t_->indexMode == kUniformIndex) {
                    const uint64_t task = unit / t_->featureTiles;
                    ProcessGroupedTile(task, task + 1, (unit % t_->featureTiles) * t_->tileK);
                    return;
                }
            }
            ProcessTile(unit / t_->featureTiles, (unit % t_->featureTiles) * t_->tileK);
        } else {
            const uint64_t first = unit * t_->outputGroup;
            const uint64_t last = Min64(first + t_->outputGroup, t_->totalTasks);
            for (uint64_t col = 0; col < t_->K; col += t_->tileK) {
                // Initial-out updates and wide arg positions retain the
                // original one-task ownership and synchronization rules.
                // The grouped path below is only an optimization for
                // independent segments and falls back when its layout
                // checks do not hold.
                if (!t_->hasInitial)
                    ProcessGroupedTile(first, last, col);
                else
                    for (uint64_t task = first; task < last; ++task)
                        ProcessTile(task, col);
            }
        }
    }

  private:
    __aicore__ static inline uint64_t Min64(uint64_t a, uint64_t b) {
        return a < b ? a : b;
    }
    // CPU torch_scatter 2.1.2 divides the wrapped source-dtype numerator
    // by a source-dtype count, using C++ truncation toward zero.
    __aicore__ static inline int64_t IntegerMean(int64_t value, uint64_t rows) {
        const int64_t count = static_cast<T>(rows ? rows : 1);
        return count ? value / count : 0;
    }
    __aicore__ static inline Work FiniteExtreme() {
        if constexpr (std::is_same<T, half>::value)
            return Reduce == SegmentMin ? 65504.0f : -65504.0f;
        if constexpr (std::is_same<T, bfloat16_t>::value)
            return Reduce == SegmentMin ? 3.3895313892515355e38f : -3.3895313892515355e38f;
        if constexpr (Floating)
            return Reduce == SegmentMin ? 3.4028234663852886e38f : -3.4028234663852886e38f;
        return Reduce == SegmentMin ? INT32_MAX : INT32_MIN;
    }
    __aicore__ inline void NormalizeExtremeValue(LocalTensor<Work> values, uint32_t count) {
        if constexpr (Extreme) {
            if (t_->hasInitial)
                return;
            const uint32_t rounded = (count + 63) / 64 * 64;
            auto mask = maskBuf_.Get<uint8_t>();
            if constexpr (Floating) {
                CompareScalar(mask, values, FiniteExtreme(), Reduce == SegmentMin ? CMPMODE::LT : CMPMODE::GT,
                              rounded);
            } else {
                CompareScalar(mask, values, FiniteExtreme(), CMPMODE::EQ, rounded);
                PipeBarrier<PIPE_V>();
                Not(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(), rounded / 16);
            }
            PipeBarrier<PIPE_V>();
            Select(values.template ReinterpretCast<float>(), mask, values.template ReinterpretCast<float>(),
                   0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, rounded);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void NormalizeExtremePosition(LocalTensor<Work> values, LocalTensor<float> positions,
                                                    uint32_t count) {
        if constexpr (Floating && HasArg) {
            const uint32_t rounded = (count + 63) / 64 * 64;
            auto mask = maskBuf_.Get<uint8_t>();
            Compare(mask, values, values, CMPMODE::EQ, rounded);
            PipeBarrier<PIPE_V>();
            Select(positions, mask, positions, CountFloat(t_->M), SELMODE::VSEL_TENSOR_SCALAR_MODE, rounded);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ static inline float CountFloat(uint64_t value) {
        // Scalar conversion accepts signed int32; split without narrowing the count.
        return static_cast<float>(static_cast<int32_t>(value >> 32)) * 4294967296.0f +
               static_cast<float>(static_cast<int32_t>(value & 0x7fffffffULL)) +
               ((value & 0x80000000ULL) ? 2147483648.0f : 0.0f);
    }
    __aicore__ inline void SetCompensatedScale(uint64_t terms) {
        if (boundedUnscaled_) {
            inputScale_ = 1.0f;
            outputScale_ = 1.0f;
            return;
        }
        inputScale_ = 0.5f;
        outputScale_ = 2.0f;
        for (uint64_t bound = terms; bound > 1; bound = (bound + 1) / 2) {
            inputScale_ *= 0.5f;
            outputScale_ *= 2.0f;
        }
    }
    __aicore__ inline void ScaleCompensatedRows(LocalTensor<float> work, uint32_t count) {
        if (boundedUnscaled_)
            return;
        Muls(work, work, inputScale_, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Bounds(uint64_t batch, uint64_t seg, uint64_t &first, uint64_t &last) {
        if (t_->indexMode == kUniformIndex) {
            const uint64_t rows = t_->M / t_->nSegments;
            first = seg * rows;
            last = first + rows;
            return;
        }
        if (cached_ && batch == lastBatch_ && seg == lastSegment_) {
            first = lastFirst_;
            last = lastEnd_;
            return;
        }
        const uint64_t base = batch * t_->indexBatchStride;
        bool found = false;
        // Preserve the per-segment endpoint proof and adjacent-segment cache.
        if (t_->M % t_->nSegments == 0) {
            const uint64_t rows = t_->M / t_->nSegments;
            const uint64_t a = seg * rows, b = a + rows;
            if (rows && index_.GetValue(base + a * t_->indexRowStride) == static_cast<int64_t>(seg) &&
                index_.GetValue(base + (b - 1) * t_->indexRowStride) == static_cast<int64_t>(seg) &&
                (a == 0 || index_.GetValue(base + (a - 1) * t_->indexRowStride) < static_cast<int64_t>(seg)) &&
                (b == t_->M || index_.GetValue(base + b * t_->indexRowStride) > static_cast<int64_t>(seg))) {
                first = a;
                last = b;
                found = true;
            }
        }
        if (!found) {
            first = cached_ && batch == lastBatch_ && seg == lastSegment_ + 1
                        ? lastEnd_
                        : IndexLowerBound(index_, base, t_->M, seg, 0, t_->indexRowStride);
            last = IndexLowerBound(index_, base, t_->M, seg + 1, first, t_->indexRowStride);
        }
        cached_ = true;
        lastBatch_ = batch;
        lastSegment_ = seg;
        lastFirst_ = first;
        lastEnd_ = last;
    }
    __aicore__ inline void LoadRows(uint64_t offset, uint32_t rows, uint32_t valid, uint32_t width) {
        auto src = srcBuf_.Get<T>();
        if constexpr (BatchType || ExtremeBatchType) {
            if ((t_->batchReduce || OrderedFloat()) && valid == width && width == t_->K) {
                // Preserve the aligned full-row transfer of the active fast paths.
                DataCopy(src, src_[offset], rows * width);
                SyncEvent<HardEvent::MTE2_V>(mte2v_);
                return;
            }
        }
        LoadPaddedRows(src, src_, offset, rows, valid, width, t_->K);
        SyncEvent<HardEvent::MTE2_V>(mte2v_);
    }
    __aicore__ inline void SanitizeExtremeRows(LocalTensor<Work> values, uint32_t count) {
        if constexpr (Floating && Extreme && !HasArg) {
            auto mask = maskBuf_.Get<uint8_t>();
            const uint32_t rounded = (count + 63) / 64 * 64;
            // A full tile mask permits one vector submission, even for many
            // rows. NaNs cannot displace an ordinary numeric extremum.
            Compare(mask, values, values, CMPMODE::EQ, rounded);
            PipeBarrier<PIPE_V>();
            Select(values, mask, values, Reduce == SegmentMin ? __builtin_inff() : -__builtin_inff(),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, rounded);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline LocalTensor<Work> WorkRows(uint32_t rows, uint32_t width) {
        if constexpr (std::is_same<T, Work>::value)
            return srcBuf_.Get<Work>();
        else {
            auto work = workBuf_.Get<Work>();
            Cast(work, srcBuf_.Get<T>(), RoundMode::CAST_NONE, rows * width);
            PipeBarrier<PIPE_V>();
            return work;
        }
    }
    __aicore__ inline LocalTensor<Work> WorkRow(uint32_t row, uint32_t width, LocalTensor<Work> work) {
        return work[row * width];
    }
    __aicore__ inline void Combine(LocalTensor<Work> dst, LocalTensor<Work> a, LocalTensor<Work> b, uint32_t count) {
        if constexpr (Floating && Extreme && HasArg) {
            auto temp = extremeScratchBuf_.Get<float>();
            auto mask = maskBuf_.Get<uint8_t>();
            const uint32_t rounded = (count + 63) / 64 * 64;
            if constexpr (Reduce == SegmentMin)
                Min(temp, a, b, count);
            else
                Max(temp, a, b, count);
            PipeBarrier<PIPE_V>();
            // The accumulator may alias dst, so preserve it until both NaN
            // masks have selected the original numeric operand.
            if (!t_->hasInitial) {
                Compare(mask, a, a, CMPMODE::EQ, rounded);
                PipeBarrier<PIPE_V>();
                Select(temp, mask, temp, b, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
                PipeBarrier<PIPE_V>();
            }
            Compare(mask, b, b, CMPMODE::EQ, rounded);
            PipeBarrier<PIPE_V>();
            Select(dst, mask, temp, a, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            PipeBarrier<PIPE_V>();
        } else {
            if constexpr (Reduce == SegmentMin)
                Min(dst, a, b, count);
            else if constexpr (Reduce == SegmentMax)
                Max(dst, a, b, count);
            else
                Add(dst, a, b, count);
            PipeBarrier<PIPE_V>();
        }
    }
    // Odd row counts carry the final row into the first partial. All column tails
    // are handled by the AscendC count API, including multi-repeat widths.
    __aicore__ inline void Tree(LocalTensor<Work> work, uint32_t rows, uint32_t width) {
        while (rows > 1) {
            const uint32_t halfRows = rows / 2;
            Combine(work, work, work[halfRows * width], halfRows * width);
            if (rows % 2)
                Combine(work, work, work[(rows - 1) * width], width);
            rows = halfRows;
        }
    }
    // Error-free TwoSum at each tree node and at tile boundaries. The low parts
    // prevent long sums and cancellation from inheriting FP32 accumulation drift.
    // b is consumed, which is safe for the discarded half of a tree level.
    // Fresh inputs have implicit +0 low parts; an odd first-level carry only
    // has a fresh b. Preserve the same rounded operations for every residual.
    template <bool FreshInputs = false, bool FreshB = false>
    __aicore__ inline void CompensatedCombineInto(LocalTensor<float> a, LocalTensor<float> ca, LocalTensor<float> b,
                                                  LocalTensor<float> cb, LocalTensor<float> sum, uint32_t count) {
        auto z = zScratchBuf_.Get<float>();
        Add(sum, a, b, count);
        PipeBarrier<PIPE_V>();
        Sub(z, sum, a, count);
        PipeBarrier<PIPE_V>();
        if constexpr (!FreshInputs) {
            if constexpr (FreshB)
                Adds(ca, ca, 0.0f, count);
            else
                Add(ca, ca, cb, count);
            PipeBarrier<PIPE_V>();
        }
        // Finish b-z before reusing z for sum-z. The arithmetic dependency
        // graph is unchanged, while no upper-half correction storage is used.
        Sub(b, b, z, count);
        PipeBarrier<PIPE_V>();
        Sub(z, sum, z, count);
        PipeBarrier<PIPE_V>();
        Sub(a, a, z, count);
        PipeBarrier<PIPE_V>();
        Add(a, a, b, count);
        PipeBarrier<PIPE_V>();
        if constexpr (FreshInputs)
            Adds(ca, a, 0.0f, count);
        else
            Add(ca, ca, a, count);
        PipeBarrier<PIPE_V>();
    }
    template <bool FreshInputs = false, bool FreshB = false>
    __aicore__ inline void CompensatedCombine(LocalTensor<float> a, LocalTensor<float> ca, LocalTensor<float> b,
                                              LocalTensor<float> cb, uint32_t count) {
        auto sum = sumScratchBuf_.Get<float>();
        CompensatedCombineInto<FreshInputs, FreshB>(a, ca, b, cb, sum, count);
        Adds(a, sum, 0.0f, count);
        PipeBarrier<PIPE_V>();
    }
    // Addition keeps a nonfinite high part nonfinite in every ancestor. Its
    // meaningless low residual can therefore be cleared once at finalization,
    // instead of paying Abs/Compare/Select at every TwoSum tree node.
    __aicore__ inline void ClearNonfiniteResidual(LocalTensor<float> high, LocalTensor<float> low, uint32_t count) {
        auto z = zScratchBuf_.Get<float>();
        // IEEE nonfinite high parts propagate normally; their Inf-Inf rounding
        // residual is not meaningful. Remove that residual explicitly.
        Abs(z, high, count);
        PipeBarrier<PIPE_V>();
        auto mask = finiteMaskBuf_.Get<uint8_t>();
        const uint32_t full = count / 64 * 64;
        for (uint32_t start = 0; start < full; start += 16320) {
            const uint32_t n = static_cast<uint32_t>(Min64(16320, full - start));
            const uint8_t repeats = n / 64;
            CompareScalar(mask, z[start], 3.402823466e38f, CMPMODE::LE, 64, repeats, {1, 1, 1, 8});
            PipeBarrier<PIPE_V>();
            Select(low[start], mask, low[start], 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 64, repeats,
                   {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        if (count > full) {
            CompareScalar(mask, z[full], 3.402823466e38f, CMPMODE::LE, 64, 1, {1, 1, 1, 8});
            PipeBarrier<PIPE_V>();
            // Compare reads a full repeat; Select masks the write. Do not clear
            // a following tree row's low part when the logical width is small.
            Select(low[full], mask, low[full], 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count - full, 1,
                   {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline LocalTensor<float> Correction() {
        if constexpr (std::is_same<T, half>::value) {
            if (t_->batchReduce && t_->tileRows >= 2)
                return srcBuf_.Get<float>();
        }
        return correctionBuf_.Get<float>();
    }
    __aicore__ inline void CompensatedTree(LocalTensor<float> work, uint32_t rows, uint32_t width,
                                           bool initialize = false) {
        auto low = Correction();
        auto high = work;
        auto next = sumScratchBuf_.Get<float>();
        bool inScratch = false;
        bool fresh = true;
        while (rows > 1) {
            const uint32_t halfRows = rows / 2;
            if (fresh)
                CompensatedCombineInto<true>(high, low, high[halfRows * width], low, next, halfRows * width);
            else
                CompensatedCombineInto(high, low, high[halfRows * width], low[halfRows * width], next,
                                       halfRows * width);
            // Merge the surviving upper tail without changing other partials.
            if (rows % 2) {
                // Preserve the odd-row carry without aliasing the other partials.
                auto saved = high[(rows - 1) * width];
                if (fresh)
                    CompensatedCombineInto<false, true>(next, low, saved, low, high, width);
                else
                    CompensatedCombineInto(next, low, saved, low[(rows - 1) * width], high, width);
                Adds(next, high, 0.0f, width);
                PipeBarrier<PIPE_V>();
            }
            high = next;
            inScratch = !inScratch;
            next = inScratch ? work : sumScratchBuf_.Get<float>();
            rows = halfRows;
            fresh = false;
        }
        if (fresh) {
            Duplicate(low, 0.0f, width);
            PipeBarrier<PIPE_V>();
        }
        if (initialize) {
            Adds(accBuf_.Get<float>(), high, 0.0f, width);
            Adds(accCorrectionBuf_.Get<float>(), low, 0.0f, width);
            PipeBarrier<PIPE_V>();
        } else {
            // Tile-to-accumulator TwoSum uses sumScratch as its destination;
            // its b operand must remain disjoint until the residual is formed.
            if (inScratch) {
                Adds(work, high, 0.0f, width);
                PipeBarrier<PIPE_V>();
                high = work;
            }
            CompensatedCombine(accBuf_.Get<float>(), accCorrectionBuf_.Get<float>(), high, low, width);
        }
    }
    __aicore__ inline void UpdateArg(LocalTensor<Work> acc, LocalTensor<Work> row, float position, uint32_t width) {
        auto mask = maskBuf_.Get<uint8_t>();
        const uint32_t rounded = (width + 63) / 64 * 64;
        if constexpr (Floating) {
            Compare(mask, row, acc, Reduce == SegmentMin ? CMPMODE::LT : CMPMODE::GT, rounded);
            PipeBarrier<PIPE_V>();
            Not(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(), rounded / 16);
        } else {
            auto candidate = extremeScratchBuf_.Get<Work>();
            Combine(candidate, acc, row, width);
            Compare(mask, candidate, acc, CMPMODE::EQ, rounded);
        }
        PipeBarrier<PIPE_V>();
        auto arg = argFBuf_.Get<float>();
        // True mask lanes retain the previous argument, including equal values.
        Select(arg, mask, arg, position, SELMODE::VSEL_TENSOR_SCALAR_MODE, rounded);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void FillWideArg(uint32_t valid) {
        SyncEvent<HardEvent::V_S>(vs_);
        auto arg = arg64Buf_.Get<int64_t>();
        for (uint32_t col = 0; col < valid; ++col)
            arg.SetValue(col, static_cast<int64_t>(t_->M));
        SyncEvent<HardEvent::S_V>(sv_);
    }
    __aicore__ inline void MergeWideArg(uint64_t base, uint32_t valid) {
        SyncEvent<HardEvent::V_S>(vs_);
        auto relative = argFBuf_.Get<float>();
        auto arg = arg64Buf_.Get<int64_t>();
        for (uint32_t col = 0; col < valid; ++col) {
            const float pos = relative.GetValue(col);
            if (pos >= 0)
                arg.SetValue(col, static_cast<int64_t>(base) + static_cast<int64_t>(pos));
        }
        SyncEvent<HardEvent::S_V>(sv_);
    }
    __aicore__ inline void PrepareNarrowArg(uint64_t first, uint64_t rows, uint32_t width) {
        auto arg = arg64Buf_.Get<int64_t>();
        auto relative = argFBuf_.Get<float>();
        if (!rows)
            Duplicate(relative, static_cast<float>(static_cast<int32_t>(t_->M)), width);
        else {
            Adds(relative, relative, static_cast<float>(static_cast<int32_t>(first)), width);
            PipeBarrier<PIPE_V>();
            if (t_->hasInitial) {
                auto mask = maskBuf_.Get<uint8_t>();
                const uint8_t repeats = (width + 63) / 64;
                CompareScalar(mask, relative, static_cast<float>(static_cast<int32_t>(first)), CMPMODE::GE, 64, repeats,
                              {1, 1, 1, 8});
                PipeBarrier<PIPE_V>();
                Select(relative, mask, relative, static_cast<float>(static_cast<int32_t>(t_->M)),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE, 64, repeats, {1, 1, 1, 8, 8, 1});
            }
        }
        PipeBarrier<PIPE_V>();
        if (rows)
            NormalizeExtremePosition(accBuf_.Get<Work>(), relative, width);
        // The f32->i64 instruction supports a 64-element destination mask.
        for (uint32_t col = 0; col < width; col += 64)
            Cast(arg[col], relative[col], RoundMode::CAST_RINT, static_cast<uint32_t>(Min64(64, width - col)));
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_MTE3>(vmte3_);
    }
    __aicore__ inline void PrepareWideArg(uint64_t first, uint64_t rows, uint32_t valid, bool wide) {
        auto arg = arg64Buf_.Get<int64_t>();
        auto relative = argFBuf_.Get<float>();
        if (!wide) {
            SyncEvent<HardEvent::V_S>(vs_);
            for (uint32_t col = 0; col < valid; ++col) {
                const float pos = rows ? relative.GetValue(col) : -1.0f;
                arg.SetValue(col, pos < 0 ? static_cast<int64_t>(t_->M)
                                          : static_cast<int64_t>(first) + static_cast<int64_t>(pos));
            }
        }
        if constexpr (Floating) {
            if (rows) {
                SyncEvent<HardEvent::V_S>(vs_);
                auto values = accBuf_.Get<Work>();
                for (uint32_t col = 0; col < valid; ++col) {
                    const Work value = values.GetValue(col);
                    if (value != value)
                        arg.SetValue(col, static_cast<int64_t>(t_->M));
                }
            }
        }
        SyncEvent<HardEvent::S_MTE3>(smte3_);
    }
    __aicore__ inline void StoreArg(uint64_t offset, uint64_t first, uint64_t rows, uint32_t valid, uint32_t width,
                                    bool wide) {
        if (!wide && t_->M <= (1ULL << 24))
            PrepareNarrowArg(first, rows, width);
        else
            PrepareWideArg(first, rows, valid, wide);
        DataCopyPad(arg_[offset], arg64Buf_.Get<int64_t>(),
                    {1, valid * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0});
        SyncEvent<HardEvent::MTE3_V>(mte3v_);
    }
    // NaNs are ignored. An all-NaN segment follows the fixed CPU reference's
    // zero value, while a real +/-Inf remains a valid nonempty extremum.
    // Only final +/-Inf candidates trigger source reads, so finite reductions
    // do not rescan their input. The bit test also handles BF16 without a
    // scalar floating-point conversion.
    __aicore__ inline bool SourceIsNumber(uint64_t offset) {
        if constexpr (std::is_same<T, float>::value) {
            GlobalTensor<uint32_t> bits;
            bits.SetGlobalBuffer(
                const_cast<__gm__ uint32_t *>(reinterpret_cast<__gm__ const uint32_t *>(src_.GetPhyAddr())));
            const uint32_t value = bits.GetValue(offset);
            return (value & 0x7f800000U) != 0x7f800000U || !(value & 0x007fffffU);
        } else {
            GlobalTensor<uint16_t> bits;
            bits.SetGlobalBuffer(
                const_cast<__gm__ uint16_t *>(reinterpret_cast<__gm__ const uint16_t *>(src_.GetPhyAddr())));
            const uint16_t value = bits.GetValue(offset);
            constexpr uint16_t exponent = std::is_same<T, half>::value ? 0x7c00 : 0x7f80;
            constexpr uint16_t mantissa = std::is_same<T, half>::value ? 0x03ff : 0x007f;
            return (value & exponent) != exponent || !(value & mantissa);
        }
    }
    __aicore__ inline bool SegmentHasNumber(uint64_t task, uint64_t column) {
        const uint64_t batch = task / t_->nSegments;
        uint64_t first, last;
        Bounds(batch, task % t_->nSegments, first, last);
        for (uint64_t row = first; row < last; ++row) {
            if (SourceIsNumber((batch * t_->M + row) * t_->K + column))
                return true;
        }
        return false;
    }
    __aicore__ inline void NormalizeNanWord(LocalTensor<Work> values, uint64_t output, uint32_t count,
                                            uint32_t physicalWidth, uint32_t word, uint64_t bits) {
        if (physicalWidth == 0 || bits == 0)
            return;
        for (uint32_t bit = 0; bit < 64 && word * 64 + bit < count; ++bit) {
            if (!(bits & (1ULL << bit)))
                continue;
            const uint32_t lane = word * 64 + bit;
            const uint64_t task = output / t_->K + lane / physicalWidth;
            const uint64_t column = output % t_->K + lane % physicalWidth;
            if (!SegmentHasNumber(task, column))
                values.SetValue(lane, 0.0f);
        }
    }
    __aicore__ inline void NormalizeAllNan(LocalTensor<Work> values, uint64_t output, uint32_t count,
                                           uint32_t physicalWidth) {
        if constexpr (Floating && Extreme && !HasArg) {
            if (t_->hasInitial)
                return;
            auto mask = maskBuf_.Get<uint8_t>();
            const uint32_t rounded = (count + 63) / 64 * 64;
            CompareScalar(mask, values, Reduce == SegmentMin ? __builtin_inff() : -__builtin_inff(), CMPMODE::EQ,
                          rounded);
            PipeBarrier<PIPE_V>();
            SyncEvent<HardEvent::V_S>(vs_);
            auto flags = mask.ReinterpretCast<uint64_t>();
            for (uint32_t word = 0; word < (count + 63) / 64; ++word)
                NormalizeNanWord(values, output, count, physicalWidth, word, flags.GetValue(word));
            SyncEvent<HardEvent::S_V>(sv_);
        }
    }
    __aicore__ inline bool OrderedFloat() const {
        return Compensated && (t_->sourceFlags & kCpuOrderedFloat);
    }
    __aicore__ inline void StoreOrderedValue(uint64_t offset, uint64_t rows, uint32_t valid, uint32_t width) {
        if constexpr (Compensated) {
            auto acc = accBuf_.Get<float>();
            if constexpr (Reduce == SegmentMean) {
                if (rows) {
                    // CPU Reducer<T, MEAN>::write narrows the numerator and
                    // count to T before the promoted division and final cast.
                    if constexpr (!std::is_same<T, float>::value) {
                        Cast(outBuf_.Get<T>(), acc, RoundMode::CAST_RINT, width);
                        PipeBarrier<PIPE_V>();
                        Cast(acc, outBuf_.Get<T>(), RoundMode::CAST_NONE, width);
                        PipeBarrier<PIPE_V>();
                    }
                    auto denominator = zScratchBuf_.Get<float>();
                    Duplicate(denominator, CountFloat(rows), width);
                    PipeBarrier<PIPE_V>();
                    if constexpr (!std::is_same<T, float>::value) {
                        Cast(outBuf_.Get<T>(), denominator, RoundMode::CAST_RINT, width);
                        PipeBarrier<PIPE_V>();
                        Cast(denominator, outBuf_.Get<T>(), RoundMode::CAST_NONE, width);
                        PipeBarrier<PIPE_V>();
                    }
                    Div(acc, acc, denominator, width);
                    PipeBarrier<PIPE_V>();
                }
            }
            if constexpr (std::is_same<T, float>::value) {
                SyncEvent<HardEvent::V_MTE3>(vmte3_);
                DataCopyPad(out_[offset], acc, {1, valid * 4, 0, 0, 0});
            } else {
                Cast(outBuf_.Get<T>(), acc, RoundMode::CAST_RINT, width);
                PipeBarrier<PIPE_V>();
                SyncEvent<HardEvent::V_MTE3>(vmte3_);
                DataCopyPad(out_[offset], outBuf_.Get<T>(), {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
            }
            SyncEvent<HardEvent::MTE3_V>(mte3v_);
        }
    }
    __aicore__ inline void OrderedRows(LocalTensor<float> work, uint32_t rows, uint32_t width) {
        if constexpr (Compensated) {
            auto acc = accBuf_.Get<float>();
            for (uint32_t row = 0; row < rows; ++row) {
                Add(acc, acc, work[row * width], width);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void OrderedTile(uint64_t task, uint64_t col) {
        if constexpr (Compensated) {
            const uint64_t batch = task / t_->nSegments, seg = task % t_->nSegments;
            uint64_t first = 0, last = 0;
            Bounds(batch, seg, first, last);
            const uint64_t rows = last - first;
            const uint32_t valid = static_cast<uint32_t>(Min64(t_->tileK, t_->K - col));
            const uint32_t align = 32 / sizeof(T), width = (valid + align - 1) / align * align;
            const uint64_t output = task * t_->K + col;
            auto acc = accBuf_.Get<float>();
            Duplicate(acc, 0.0f, width);
            PipeBarrier<PIPE_V>();
            if (!rows) {
                if (!t_->hasInitial)
                    StoreOrderedValue(output, 0, valid, width);
                return;
            }
            if (t_->hasInitial) {
                DataCopyPad(srcBuf_.Get<T>(), out_[first == 0 ? batch * t_->nSegments * t_->K + col : output],
                            {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
                            DataCopyPadExtParams<T>(true, 0, static_cast<uint8_t>(width - valid), 0));
                SyncEvent<HardEvent::MTE2_V>(mte2v_);
                Adds(acc, WorkRow(0, width, WorkRows(1, width)), 0.0f, width);
                PipeBarrier<PIPE_V>();
                SyncEvent<HardEvent::V_MTE2>(vmte2_);
            }
            for (uint64_t row = first; row < last; row += t_->tileRows) {
                const uint32_t count = static_cast<uint32_t>(Min64(t_->tileRows, last - row));
                LoadRows((batch * t_->M + row) * t_->K + col, count, valid, width);
                OrderedRows(WorkRows(count, width), count, width);
                SyncEvent<HardEvent::V_MTE2>(vmte2_);
            }
            StoreOrderedValue(output, rows, valid, width);
        }
    }

    __aicore__ inline void FinalizeFloatingValue(uint64_t rows, uint32_t width) {
        auto acc = accBuf_.Get<float>();
        if constexpr (Compensated) {
            ClearNonfiniteResidual(acc, accCorrectionBuf_.Get<float>(), width);
            Add(acc, acc, accCorrectionBuf_.Get<float>(), width);
            PipeBarrier<PIPE_V>();
            if constexpr (Reduce == SegmentMean) {
                if (rows) {
                    auto denominator = sumScratchBuf_.Get<float>();
                    Duplicate(denominator, CountFloat(rows) * inputScale_, width);
                    PipeBarrier<PIPE_V>();
                    // Division avoids an upward-rounded reciprocal turning a
                    // representable max-float mean into infinity (e.g. M=7).
                    Div(acc, acc, denominator, width);
                    PipeBarrier<PIPE_V>();
                }
            } else {
                if (!boundedUnscaled_) {
                    Muls(acc, acc, outputScale_, width);
                    PipeBarrier<PIPE_V>();
                }
            }
        } else if constexpr (Reduce == SegmentMean) {
            if (rows) {
                Muls(acc, acc, 1.0f / CountFloat(rows), width);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void StoreValue(uint64_t offset, uint64_t rows, uint32_t valid, uint32_t width) {
        auto acc = accBuf_.Get<Work>();
        NormalizeExtremeValue(acc, width);
        if constexpr (Floating) {
            FinalizeFloatingValue(rows, width);
            if constexpr (std::is_same<T, float>::value) {
                SyncEvent<HardEvent::V_MTE3>(vmte3_);
                DataCopyPad(out_[offset], acc, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
            } else {
                auto output = outBuf_.Get<T>();
                Cast(output, acc, RoundMode::CAST_RINT, width);
                PipeBarrier<PIPE_V>();
                SyncEvent<HardEvent::V_MTE3>(vmte3_);
                DataCopyPad(out_[offset], output, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
            }
        } else if constexpr (std::is_same<T, int32_t>::value && Reduce != SegmentMean) {
            SyncEvent<HardEvent::V_MTE3>(vmte3_);
            DataCopyPad(out_[offset], acc, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        } else {
            auto output = outBuf_.Get<T>();
            SyncEvent<HardEvent::V_S>(vs_);
            for (uint32_t col = 0; col < valid; ++col) {
                // Wrap numerator/count in T, then truncate signed division toward zero.
                int64_t value = static_cast<T>(acc.GetValue(col));
                if constexpr (Reduce == SegmentMean)
                    value = IntegerMean(value, rows);
                output.SetValue(col, static_cast<T>(value));
            }
            SyncEvent<HardEvent::S_MTE3>(smte3_);
            DataCopyPad(out_[offset], output, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        }
        SyncEvent<HardEvent::MTE3_V>(mte3v_);
    }
    // Partition an ownership group into the largest adjacent source spans
    // that fit the planner's row capacity.  The entire ownership group remains
    // on one AIV, including subgroups and fallback tasks, so an unaligned
    // output row never gains a writer on a different physical 32-byte block.
    __aicore__ inline void ProcessTaskRange(uint64_t first, uint64_t last, uint64_t col) {
        for (uint64_t task = first; task < last; ++task)
            ProcessTile(task, col);
    }
    __aicore__ inline void ProcessOrderedGroup(uint64_t firstTask, uint64_t count, uint64_t rows) {
        const uint32_t width = static_cast<uint32_t>(t_->K);
        const uint32_t groupWidth = static_cast<uint32_t>(count) * width;
        auto input = srcBuf_.Get<T>();
        if (count == 1) {
            LoadRows(firstTask * rows * t_->K, static_cast<uint32_t>(rows), width, width);
        } else {
            // DMA gathers corresponding rows of independent
            // segments into one wide feature row. The source gap
            // is measured in bytes, without vector-repeat limits.
            const uint32_t sourceGap = static_cast<uint32_t>((rows - 1) * width * sizeof(T));
            for (uint32_t row = 0; row < rows; ++row)
                DataCopyPad(input[row * groupWidth], src_[(firstTask * rows + row) * t_->K],
                            {static_cast<uint16_t>(count), width * static_cast<uint32_t>(sizeof(T)), sourceGap, 0, 0},
                            DataCopyPadExtParams<T>(false, 0, 0, 0));
            SyncEvent<HardEvent::MTE2_V>(mte2v_);
        }
        auto work = WorkRows(static_cast<uint32_t>(rows), groupWidth);
        Duplicate(accBuf_.Get<float>(), 0.0f, groupWidth);
        PipeBarrier<PIPE_V>();
        // One vector instruction covers all independent segments
        // at each row; each accumulator still follows CPU order.
        OrderedRows(work, static_cast<uint32_t>(rows), groupWidth);
        StoreOrderedValue(firstTask * t_->K, rows, groupWidth, groupWidth);
        SyncEvent<HardEvent::V_MTE2>(vmte2_);
    }
    __aicore__ inline bool TryOrderedGroup(uint64_t firstTask, uint64_t count, uint64_t col) {
        if constexpr (Compensated) {
            if (!OrderedFloat() || t_->hasInitial || t_->indexMode != kUniformIndex || col != 0 ||
                (t_->K * sizeof(T)) % 32 != 0 || count * t_->K > t_->tileK)
                return false;
            const uint64_t rows = t_->M / t_->nSegments;
            if (count * rows * t_->K > uint64_t(t_->tileRows) * t_->tileK || count * rows > 65535)
                return false;
            ProcessOrderedGroup(firstTask, count, rows);
            return true;
        }
        return false;
    }
    __aicore__ inline bool TryUniformBatch(uint64_t firstTask, uint64_t count, uint64_t col) {
        if constexpr (BatchType || ExtremeBatchType) {
            // The host proof covers every segment, so the batch needs neither
            // per-segment GM reads nor a second uniformity scan in local arrays.
            if (t_->indexMode == kUniformIndex && t_->batchReduce && (count >= 2 || BatchType)) {
                const uint32_t rows = static_cast<uint32_t>(t_->M / t_->nSegments);
                const uint32_t width = static_cast<uint32_t>(Min64(t_->tileK, t_->K - col));
                LoadRows(firstTask * rows * t_->K + col, count * rows, width, width);
                auto work = WorkRows(count * rows, width);
                SanitizeExtremeRows(work, count * rows * width);
                if constexpr (BatchType)
                    BatchedCompensatedTree(work, rows, count, width, firstTask * t_->K + col);
                else
                    BatchedExtremeTree(work, rows, count, width, firstTask * t_->K);
                return true;
            }
        }
        return false;
    }
    __aicore__ inline void ProcessBoundedTasks(uint64_t firstTask, uint64_t count, uint64_t col) {
        uint64_t starts[32] = {0};
        uint64_t ends[32] = {0};
        for (uint64_t i = 0; i < count; ++i) {
            const uint64_t task = firstTask + i;
            Bounds(task / t_->nSegments, task % t_->nSegments, starts[i], ends[i]);
        }
        uint64_t first = 0;
        while (first < count) {
            const uint64_t task = firstTask + first;
            const uint64_t batch = task / t_->nSegments;
            if (ends[first] - starts[first] > t_->tileRows) {
                // A single long segment is reduced by the established tiled
                // path; it does not prevent neighboring short segments from
                // sharing their own transfers.
                ProcessTile(task, col);
                ++first;
                continue;
            }
            uint64_t last = first + 1;
            while (last < count && (firstTask + last) / t_->nSegments == batch && starts[last] == ends[last - 1] &&
                   ends[last] - starts[first] <= t_->tileRows)
                ++last;
            ProcessBoundedGroup(task, last - first, batch, starts + first, ends + first, col);
            first = last;
        }
    }
    __aicore__ inline void ProcessGroupedTile(uint64_t firstTask, uint64_t lastTask, uint64_t col) {
        const uint64_t count = lastTask - firstTask;
        if (!count || count > 32) {
            ProcessTaskRange(firstTask, lastTask, col);
            return;
        }
        if (TryOrderedGroup(firstTask, count, col))
            return;
        if constexpr (HasArg) {
            if (t_->M > (1ULL << 24)) {
                ProcessTaskRange(firstTask, lastTask, col);
                return;
            }
        }
        if (!TryUniformBatch(firstTask, count, col))
            ProcessBoundedTasks(firstTask, count, col);
    }
    // Adjacent-row pairs stay inside each verified power-of-two segment.
    // Repeat strides read even/odd rows directly and write compact partials;
    // no separate high/low packing passes or batch buffers are needed.
    template <bool Fresh>
    __aicore__ inline void PairLevel(LocalTensor<float> high, LocalTensor<float> low, LocalTensor<float> nextHigh,
                                     LocalTensor<float> nextLow, uint32_t pairs, uint32_t width) {
        auto z = zScratchBuf_.Get<float>();
        const uint8_t dense = static_cast<uint8_t>(width / 8);
        const uint8_t paired = static_cast<uint8_t>(2 * width / 8);
        const BinaryRepeatParams fromPair{1, 1, 1, dense, paired, paired};
        const BinaryRepeatParams mixed{1, 1, 1, dense, dense, paired};
        const BinaryRepeatParams toPair{1, 1, 1, paired, paired, dense};
        const BinaryRepeatParams packed{1, 1, 1, dense, dense, dense};
        for (uint32_t first = 0; first < pairs; first += 255) {
            const uint8_t repeats = static_cast<uint8_t>(Min64(255, pairs - first));
            for (uint32_t col = 0; col < width; col += 64) {
                const uint64_t mask = Min64(64, width - col);
                const uint32_t input = 2 * first * width + col;
                const uint32_t output = first * width + col;
                auto a = high[input];
                auto b = high[input + width];
                auto sum = nextHigh[output];
                auto temp = z[output];
                auto residual = nextLow[output];
                Add(sum, a, b, mask, repeats, fromPair);
                PipeBarrier<PIPE_V>();
                Sub(temp, sum, a, mask, repeats, mixed);
                PipeBarrier<PIPE_V>();
                Sub(b, b, temp, mask, repeats, toPair);
                PipeBarrier<PIPE_V>();
                Sub(temp, sum, temp, mask, repeats, packed);
                PipeBarrier<PIPE_V>();
                Sub(a, a, temp, mask, repeats, toPair);
                PipeBarrier<PIPE_V>();
                if constexpr (Fresh) {
                    Add(residual, a, b, mask, repeats, fromPair);
                    PipeBarrier<PIPE_V>();
                } else {
                    Add(temp, a, b, mask, repeats, fromPair);
                    PipeBarrier<PIPE_V>();
                    Add(residual, low[input], low[input + width], mask, repeats, fromPair);
                    PipeBarrier<PIPE_V>();
                    Add(residual, residual, temp, mask, repeats, packed);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }
    // Pair equally sized row stripes inside each segment. For a narrow row,
    // several rows occupy one 64-float vector without a packing pass. Stripe
    // size divides every power-of-two segment; no pair crosses its boundary.
    struct PairGeometry {
        uint32_t stripe;
        uint32_t pairs;
    };
    __aicore__ static inline PairGeometry MakePairGeometry(uint32_t rows, uint32_t groups, uint32_t width) {
        uint32_t stripe = 1;
        uint32_t shift = 0;
        while (stripe * 2 <= rows / 2 && stripe * 2 * width <= 64) {
            stripe *= 2;
            ++shift;
        }
        return {stripe, (groups * (rows / 2)) >> shift};
    }
    __aicore__ inline void BatchedExtremeTree(LocalTensor<float> work, uint32_t rows, uint32_t groups, uint32_t width,
                                              uint64_t output) {
        // FP16's source tile is dead after its conversion to FP32; its bytes
        // hold exactly half as many float partials. FP32 gets a separately
        // budgeted half-tile scratch. Ping-pong avoids compacting in place
        // across vector repeats with overlapping source/destination ranges.
        auto scratch = std::is_same<T, float>::value ? workBuf_.Get<float>() : srcBuf_.Get<float>();
        auto high = work;
        auto next = scratch;
        bool inScratch = false;
        while (rows > 1) {
            const auto geometry = MakePairGeometry(rows, groups, width);
            const uint32_t stripe = geometry.stripe;
            const uint32_t span = stripe * width;
            const uint32_t pairs = geometry.pairs;
            const BinaryRepeatParams pair{1,
                                          1,
                                          1,
                                          static_cast<uint8_t>(span / 8),
                                          static_cast<uint8_t>(2 * span / 8),
                                          static_cast<uint8_t>(2 * span / 8)};
            for (uint32_t first = 0; first < pairs; first += 255) {
                const uint8_t repeats = static_cast<uint8_t>(Min64(255, pairs - first));
                for (uint32_t col = 0; col < span; col += 64) {
                    const uint64_t mask = Min64(64, span - col);
                    const uint32_t input = 2 * first * span + col;
                    const uint32_t dest = first * span + col;
                    if constexpr (Reduce == SegmentMin)
                        Min(next[dest], high[input], high[input + span], mask, repeats, pair);
                    else
                        Max(next[dest], high[input], high[input + span], mask, repeats, pair);
                }
            }
            PipeBarrier<PIPE_V>();
            high = next;
            inScratch = !inScratch;
            next = inScratch ? work : scratch;
            rows /= 2;
        }
        StoreBatchedResult(high, groups * width, output, inScratch);
    }
    __aicore__ inline void ProcessOrderedBatch(LocalTensor<float> work, uint32_t rows, uint32_t groups, uint32_t width,
                                               uint64_t output) {
        for (uint32_t group = 0; group < groups; ++group) {
            Duplicate(accBuf_.Get<float>(), 0.0f, width);
            PipeBarrier<PIPE_V>();
            OrderedRows(work[group * rows * width], rows, width);
            StoreOrderedValue(output + group * t_->K, rows, width, width);
        }
        SyncEvent<HardEvent::V_MTE2>(vmte2_);
    }
    __aicore__ inline void ProcessTwoRowBatch(LocalTensor<float> work, uint32_t groups, uint32_t width,
                                              uint64_t output) {
        // With exactly two source values there is no accumulation chain:
        // one FP32 addition already provides the rounded sum. Mean scales
        // first to avoid overflowing a finite average. FP16 inputs are
        // already promoted by WorkRows. Longer trees retain TwoSum.
        if constexpr (Reduce == SegmentMean) {
            Muls(work, work, 0.5f, 2 * groups * width);
            PipeBarrier<PIPE_V>();
        }
        auto result = sumScratchBuf_.Get<float>();
        const BinaryRepeatParams pair{1,
                                      1,
                                      1,
                                      static_cast<uint8_t>(width / 8),
                                      static_cast<uint8_t>(2 * width / 8),
                                      static_cast<uint8_t>(2 * width / 8)};
        for (uint32_t col = 0; col < width; col += 64)
            Add(result[col], work[col], work[width + col], Min64(64, width - col), static_cast<uint8_t>(groups), pair);
        PipeBarrier<PIPE_V>();
        StoreBatchedResult(result, groups * width, output, false, width);
    }
    struct CompensatedValues {
        LocalTensor<float> high;
        LocalTensor<float> low;
    };
    __aicore__ inline CompensatedValues ReduceCompensatedBatch(LocalTensor<float> work, uint32_t rows, uint32_t groups,
                                                               uint32_t width) {
        auto high = work;
        auto low = Correction();
        auto nextHigh = sumScratchBuf_.Get<float>();
        auto nextLow = low;
        // After the first level the source tile is dead. Reuse its lower half
        // for highs and upper half for lows, alternating with the dedicated
        // sum/correction buffers. All input/output regions remain disjoint.
        const uint32_t halfCapacity = (t_->tileRows / 2) * t_->tileK;
        bool fresh = true;
        bool outputToScratch = true;
        while (rows > 1) {
            const auto geometry = MakePairGeometry(rows, groups, width);
            const uint32_t stripe = geometry.stripe;
            const uint32_t pairs = geometry.pairs;
            if (fresh)
                PairLevel<true>(high, low, nextHigh, nextLow, pairs, stripe * width);
            else
                PairLevel<false>(high, low, nextHigh, nextLow, pairs, stripe * width);
            high = nextHigh;
            low = nextLow;
            outputToScratch = !outputToScratch;
            nextHigh = outputToScratch ? sumScratchBuf_.Get<float>() : work;
            nextLow = outputToScratch ? Correction() : work[halfCapacity];
            rows /= 2;
            fresh = false;
        }
        return {high, low};
    }
    __aicore__ inline void BatchedCompensatedTree(LocalTensor<float> work, uint32_t rows, uint32_t groups,
                                                  uint32_t width, uint64_t output) {
        if (OrderedFloat()) {
            ProcessOrderedBatch(work, rows, groups, width, output);
            return;
        }
        if (rows == 2) {
            ProcessTwoRowBatch(work, groups, width, output);
            return;
        }
        float scale = 0.5f, inverseScale = 2.0f;
        for (uint32_t n = rows; n > 1; n /= 2) {
            scale *= 0.5f;
            inverseScale *= 2.0f;
        }
        // Finite half inputs are exact FP32 multiples of 2^-24. In this
        // bounded, no-out batch their complete compensated tree cannot reach
        // FP32 overflow/underflow, so power-of-two input/output scaling is
        // redundant. Keep FP32 promotion, TwoSum, and nonfinite cleanup.
        const bool boundedInput = (std::is_same<T, half>::value && rows <= 256) ||
                                  (std::is_same<T, float>::value && (t_->sourceFlags & kSourceRangeProven));
        if (!boundedInput) {
            Muls(work, work, scale, rows * groups * width);
            PipeBarrier<PIPE_V>();
        }
        const auto result = ReduceCompensatedBatch(work, rows, groups, width);
        auto high = result.high;
        auto low = result.low;
        const uint32_t count = groups * width;
        ClearNonfiniteResidual(high, low, count);
        Add(high, high, low, count);
        PipeBarrier<PIPE_V>();
        if constexpr (Reduce == SegmentMean) {
            Muls(high, high, boundedInput ? 2.0f * scale : 2.0f, count);
            PipeBarrier<PIPE_V>();
        } else if (!boundedInput) {
            Muls(high, high, inverseScale, count);
            PipeBarrier<PIPE_V>();
        }
        StoreBatchedResult(high, count, output, false, width);
    }
    __aicore__ inline void WriteBatchedRows(LocalTensor<T> result, uint32_t count, uint64_t output, uint32_t width) {
        if (width == 0) {
            DataCopy(out_[output], result, count);
            return;
        }
        if (width == t_->K) {
            DataCopy(out_[output], result, count);
            return;
        }
        {
            const uint64_t gap = (t_->K - width) * sizeof(T);
            if (gap <= UINT32_MAX) {
                DataCopyPad(out_[output], result,
                            {static_cast<uint16_t>(count / width), width * static_cast<uint32_t>(sizeof(T)), 0,
                             static_cast<uint32_t>(gap), 0});
            } else {
                for (uint32_t row = 0; row < count / width; ++row)
                    DataCopy(out_[output + row * t_->K], result[row * width], width);
            }
        }
    }
    __aicore__ inline void StoreBatchedResult(LocalTensor<float> high, uint32_t count, uint64_t output,
                                              bool inScratch = false, uint32_t rowWidth = 0) {
        NormalizeExtremeValue(high, count);
        if constexpr (std::is_same<T, float>::value) {
            SyncEvent<HardEvent::V_MTE3>(vmte3_);
            WriteBatchedRows(high, count, output, rowWidth);
        } else {
            auto result = srcBuf_.Get<T>();
            if constexpr (ExtremeBatchType) {
                // A final partial in srcBuf must not be cast into overlapping
                // half addresses. The consumed float input provides room.
                if (inScratch)
                    result = workBuf_.Get<T>();
            }
            Cast(result, high, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            SyncEvent<HardEvent::V_MTE3>(vmte3_);
            WriteBatchedRows(result, count, output, rowWidth);
        }
        SyncEvent<HardEvent::MTE3_V>(mte3v_);
        SyncEvent<HardEvent::V_MTE2>(vmte2_);
    }
    // Every bound in this subgroup belongs to the same batch, neighboring
    // intervals are contiguous, and their combined row span fits tileRows.
    // The reductions reuse the same compensated tree and arg routines as
    // ProcessTile; only the source load/cast is shared.
    __aicore__ inline bool TryBoundedBatch(LocalTensor<Work> work, uint64_t firstTask, uint64_t count, uint32_t width,
                                           const uint64_t *starts, const uint64_t *ends) {
        if constexpr (BatchType || ExtremeBatchType) {
            const uint64_t rows = ends[0] - starts[0];
            bool uniform = t_->batchReduce && count >= 2 && rows >= 2 && (rows & (rows - 1)) == 0 && width == t_->K;
            for (uint64_t i = 1; uniform && i < count; ++i)
                uniform = ends[i] - starts[i] == rows;
            if (uniform) {
                if constexpr (BatchType)
                    BatchedCompensatedTree(work, static_cast<uint32_t>(rows), static_cast<uint32_t>(count), width,
                                           firstTask * t_->K);
                else
                    BatchedExtremeTree(work, static_cast<uint32_t>(rows), static_cast<uint32_t>(count), width,
                                       firstTask * t_->K);
                return true;
            }
        }
        return false;
    }
    __aicore__ inline bool ReduceBoundedSegment(LocalTensor<Work> work, uint32_t rowOffset, uint64_t rows,
                                                uint32_t width, uint64_t output, uint32_t valid) {
        auto acc = accBuf_.Get<Work>();
        if constexpr (Extreme && std::is_same<T, int32_t>::value && !HasArg) {
            // Integer extrema are associative: reduce each independent segment
            // in a tree, then apply the same finite-identity cleanup at StoreValue.
            auto segment = work[rowOffset * width];
            Tree(segment, static_cast<uint32_t>(rows), width);
            Adds(acc, segment, static_cast<Work>(0), width);
            PipeBarrier<PIPE_V>();
        } else if constexpr (Extreme) {
            Duplicate(acc, FiniteExtreme(), width);
            if constexpr (HasArg)
                Duplicate(argFBuf_.Get<float>(), -1.0f, width);
            PipeBarrier<PIPE_V>();
            for (uint64_t row = 0; row < rows; ++row) {
                auto value = WorkRow(rowOffset + static_cast<uint32_t>(row), width, work);
                if constexpr (HasArg)
                    UpdateArg(acc, value, static_cast<float>(static_cast<int32_t>(row)), width);
                Combine(acc, acc, value, width);
            }
        } else if constexpr (Compensated) {
            if (OrderedFloat()) {
                Duplicate(acc, 0.0f, width);
                PipeBarrier<PIPE_V>();
                OrderedRows(work[rowOffset * width], static_cast<uint32_t>(rows), width);
                StoreOrderedValue(output, rows, valid, width);
                return true;
            }
            ScaleCompensatedRows(work[rowOffset * width], static_cast<uint32_t>(rows) * width);
            CompensatedTree(work[rowOffset * width], static_cast<uint32_t>(rows), width, true);
        } else if constexpr ((Floating || std::is_same<T, int32_t>::value) && !HasArg) {
            Tree(work[rowOffset * width], static_cast<uint32_t>(rows), width);
            Combine(acc, acc, work[rowOffset * width], width);
        } else {
            for (uint64_t row = 0; row < rows; ++row) {
                auto value = WorkRow(rowOffset + static_cast<uint32_t>(row), width, work);
                Combine(acc, acc, value, width);
            }
        }
        return false;
    }
    __aicore__ inline void ProcessBoundedSegment(LocalTensor<Work> work, uint64_t firstTask, uint64_t i,
                                                 const uint64_t *starts, const uint64_t *ends, uint64_t col,
                                                 uint32_t valid, uint32_t width) {
        auto acc = accBuf_.Get<Work>();
        const uint64_t task = firstTask + i;
        const uint64_t rows = ends[i] - starts[i];
        const uint32_t rowOffset = static_cast<uint32_t>(starts[i] - starts[0]);
        const uint64_t output = task * t_->K + col;
        // A nonempty compensated group initializes both accumulators
        // from its first tree result. Empty segments still need zeros.
        const bool initializeAccumulator = !Compensated || !rows;
        if constexpr (Compensated) {
            SetCompensatedScale(rows);
            if (initializeAccumulator) {
                Duplicate(accCorrectionBuf_.Get<float>(), 0.0f, width);
                PipeBarrier<PIPE_V>();
            }
        }
        if (initializeAccumulator) {
            Duplicate(acc, static_cast<Work>(0), width);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (HasArg)
            Duplicate(argFBuf_.Get<float>(), -1.0f, width);
        if (!rows) {
            StoreValue(output, 0, valid, width);
            if constexpr (HasArg)
                StoreArg(output, starts[i], 0, valid, width, false);
            return;
        }

        if (ReduceBoundedSegment(work, rowOffset, rows, width, output, valid))
            return;
        if constexpr (HasArg)
            StoreArg(output, starts[i], rows, valid, width, false);
        StoreValue(output, rows, valid, width);
    }
    __aicore__ inline void ProcessBoundedGroup(uint64_t firstTask, uint64_t count, uint64_t batch,
                                               const uint64_t *starts, const uint64_t *ends, uint64_t col) {
        const uint64_t totalRows = ends[count - 1] - starts[0];
        const uint32_t valid = static_cast<uint32_t>(Min64(t_->tileK, t_->K - col));
        const uint32_t align = 32 / sizeof(T);
        const uint32_t width = (valid + align - 1) / align * align;
        if (totalRows)
            LoadRows((batch * t_->M + starts[0]) * t_->K + col, static_cast<uint32_t>(totalRows), valid, width);
        // No DMA or cast instruction is submitted for an all-empty group.
        auto work = totalRows ? WorkRows(static_cast<uint32_t>(totalRows), width) : srcBuf_.Get<Work>();
        if (totalRows)
            SanitizeExtremeRows(work, static_cast<uint32_t>(totalRows) * width);
        if constexpr (HasArg)
            Duplicate(argFBuf_.Get<float>(), -1.0f, width);
        if (TryBoundedBatch(work, firstTask, count, width, starts, ends))
            return;
        for (uint64_t i = 0; i < count; ++i)
            ProcessBoundedSegment(work, firstTask, i, starts, ends, col, valid, width);
        // Do not overwrite srcBuf until all vector users in this group have
        // consumed it.  The next LoadRows waits on this event as well.
        SyncEvent<HardEvent::V_MTE2>(vmte2_);
    }
    struct TileInfo {
        uint64_t batch, first, rows, col, output;
        uint32_t valid, width;
        bool wide;
    };
    __aicore__ inline TileInfo DescribeTile(uint64_t task, uint64_t col) {
        const uint64_t batch = task / t_->nSegments, seg = task % t_->nSegments;
        uint64_t first = 0, last = 0;
        Bounds(batch, seg, first, last);
        const uint64_t rows = last - first;
        const uint32_t valid = static_cast<uint32_t>(Min64(t_->tileK, t_->K - col));
        const uint32_t align = 32 / sizeof(T);
        const uint32_t width = (valid + align - 1) / align * align;
        const uint64_t output = task * t_->K + col;
        return {batch, first, rows, col, output, valid, width, rows > (1ULL << 24)};
    }
    __aicore__ inline void InitializeTile(const TileInfo &tile) {
        auto acc = accBuf_.Get<Work>();
        // The first no-out, nonempty compensated tile overwrites high/low.
        // Keep the established initialization for empty and initial-out cases.
        const bool initializeAccumulator = !Compensated || !tile.rows || t_->hasInitial;
        if constexpr (Compensated) {
            // A power-of-two scale is exact for normal values and bounds every
            // intermediate by half FLT_MAX, even when a finite mean's raw sum
            // would overflow. The final conversion determines true output overflow.
            SetCompensatedScale(tile.rows + (t_->hasInitial ? 1 : 0));
            if (initializeAccumulator) {
                Duplicate(accCorrectionBuf_.Get<float>(), 0.0f, tile.width);
                PipeBarrier<PIPE_V>();
            }
        }
        if constexpr (HasArg) {
            if (tile.wide)
                FillWideArg(tile.valid);
            Duplicate(argFBuf_.Get<float>(), -1.0f, tile.width);
        }
        if (initializeAccumulator) {
            Duplicate(acc, Extreme ? FiniteExtreme() : static_cast<Work>(0), tile.width);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void LoadInitialValue(const TileInfo &tile) {
        auto acc = accBuf_.Get<Work>();
        if (t_->hasInitial) {
            DataCopyPad(srcBuf_.Get<T>(),
                        out_[tile.first == 0 ? tile.batch * t_->nSegments * t_->K + tile.col : tile.output],
                        {1, tile.valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
                        DataCopyPadExtParams<T>(true, 0, static_cast<uint8_t>(tile.width - tile.valid), 0));
            SyncEvent<HardEvent::MTE2_V>(mte2v_);
            auto initial = WorkRow(0, tile.width, WorkRows(1, tile.width));
            Adds(acc, initial, static_cast<Work>(0), tile.width);
            PipeBarrier<PIPE_V>();
            if constexpr (Compensated) {
                Muls(acc, acc, inputScale_, tile.width);
                PipeBarrier<PIPE_V>();
            }
            SyncEvent<HardEvent::V_MTE2>(vmte2_);
        }
    }
    __aicore__ inline void ReduceRowTile(LocalTensor<Work> work, const TileInfo &tile, uint64_t rowBase, uint32_t count,
                                         uint32_t begin) {
        auto acc = accBuf_.Get<Work>();
        if constexpr (Compensated) {
            ScaleCompensatedRows(work, count * tile.width);
            CompensatedTree(work, count, tile.width, !rowBase && !t_->hasInitial);
        } else if constexpr ((Floating || std::is_same<T, int32_t>::value) && !HasArg) {
            Tree(work, count, tile.width);
            Combine(acc, acc, work, tile.width);
        } else {
            for (uint32_t row = begin; row < count; ++row) {
                auto value = WorkRow(row, tile.width, work);
                if constexpr (HasArg)
                    UpdateArg(acc, value, static_cast<float>(static_cast<int32_t>(tile.wide ? row : rowBase + row)),
                              tile.width);
                Combine(acc, acc, value, tile.width);
            }
        }
    }
    __aicore__ inline void ProcessRowTile(const TileInfo &tile, uint64_t rowBase) {
        auto acc = accBuf_.Get<Work>();
        const uint32_t count = static_cast<uint32_t>(Min64(t_->tileRows, tile.rows - rowBase));
        LoadRows((tile.batch * t_->M + tile.first + rowBase) * t_->K + tile.col, count, tile.valid, tile.width);
        auto work = WorkRows(count, tile.width);
        SanitizeExtremeRows(work, count * tile.width);
        if constexpr (HasArg) {
            if (tile.wide) {
                Duplicate(argFBuf_.Get<float>(), -1.0f, tile.width);
                PipeBarrier<PIPE_V>();
            }
        }
        const uint32_t begin = 0;
        ReduceRowTile(work, tile, rowBase, count, begin);
        if constexpr (HasArg) {
            if (tile.wide)
                MergeWideArg(tile.first + rowBase, tile.valid);
        }
        SyncEvent<HardEvent::V_MTE2>(vmte2_);
    }
    __aicore__ inline void ProcessTile(uint64_t task, uint64_t col) {
        if constexpr (Compensated) {
            if (OrderedFloat()) {
                OrderedTile(task, col);
                return;
            }
        }
        const auto tile = DescribeTile(task, col);
        InitializeTile(tile);
        if (!tile.rows) {
            if (!t_->hasInitial)
                StoreValue(tile.output, 0, tile.valid, tile.width);
            if constexpr (HasArg)
                StoreArg(tile.output, tile.first, 0, tile.valid, tile.width, false);
            return;
        }
        LoadInitialValue(tile);
        for (uint64_t rowBase = 0; rowBase < tile.rows; rowBase += t_->tileRows)
            ProcessRowTile(tile, rowBase);
        if constexpr (HasArg)
            StoreArg(tile.output, tile.first, tile.rows, tile.valid, tile.width, tile.wide);
        StoreValue(tile.output, tile.rows, tile.valid, tile.width);
    }
    const SegmentTilingData *t_ = nullptr;
    GlobalTensor<T> src_, out_;
    GlobalTensor<int64_t> index_, arg_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> srcBuf_, workBuf_, accBuf_, outBuf_;
    TBuf<TPosition::VECCALC> argFBuf_, arg64Buf_, maskBuf_, nanMaskBuf_;
    TBuf<TPosition::VECCALC> extremeScratchBuf_;
    TBuf<TPosition::VECCALC> correctionBuf_, sumScratchBuf_, zScratchBuf_, accCorrectionBuf_, finiteMaskBuf_;
    float inputScale_ = 1.0f, outputScale_ = 1.0f;
    bool boundedUnscaled_ = false;
    event_t mte2v_, vmte2_, vmte3_, mte3v_, vs_, sv_, smte3_;
    bool cached_ = false;
    uint64_t lastBatch_ = 0, lastSegment_ = 0, lastFirst_ = 0, lastEnd_ = 0;
};


namespace {
template <typename T> struct VectorReduction {
    template <SegmentReduce Reduce, bool HasArg>
    __aicore__ static inline void Run(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg,
                                      const SegmentTilingData &tiling) {
        SegmentReduceDevice<T, Reduce, HasArg> op;
        op.Init(src, index, out, arg, tiling);
        op.Process();
    }
};
} // namespace

void LaunchSegmentCooVector(void *src, int64_t *index, void *out, int64_t *arg, const SegmentTilingData &tiling,
                            aclrtStream stream) {
    switch (tiling.dtypeCode) {
        case SEGMENT_COO_FLOAT32:
            DispatchTyped<VectorReduction<float>>(src, index, out, arg, tiling, stream);
            break;
        case SEGMENT_COO_FLOAT16:
            DispatchTyped<VectorReduction<half>>(src, index, out, arg, tiling, stream);
            break;
        case SEGMENT_COO_BFLOAT16:
            DispatchTyped<VectorReduction<bfloat16_t>>(src, index, out, arg, tiling, stream);
            break;
        case SEGMENT_COO_INT32:
            DispatchTyped<VectorReduction<int32_t>>(src, index, out, arg, tiling, stream);
            break;
    }
}

} // namespace opsgnn

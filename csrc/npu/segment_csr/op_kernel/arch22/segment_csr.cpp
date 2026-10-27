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

// ============================================================================
// General floating reductions
// ============================================================================

using namespace AscendC;
namespace opsgnn::csr {

// Column widths, row tiles and ownership come from the shared capacity planner.
// No branch is selected by a particular input K.
template <typename T, SegmentReduce Reduce, bool HasArg> class SegmentReduceDevice {
    static constexpr bool Floating =
        std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value;
    static constexpr bool Extreme = Reduce == SegmentMin || Reduce == SegmentMax;
    static constexpr bool OrderedFloat = Floating && !Extreme;
    static constexpr bool BatchType = !Extreme && (std::is_same<T, float>::value || std::is_same<T, half>::value);
    static constexpr bool ExtremeBatchType =
        Extreme && !HasArg && (std::is_same<T, float>::value || std::is_same<T, half>::value);
    using Work = typename std::conditional<Floating, float, int32_t>::type;

  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), t.srcNumel);
        // Streaming tensors larger than the queried cache cannot stay
        // resident. The hint changes only reads; output keeps NORMAL.
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(T))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), t.indexNumel);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), t.outNumel);
        if constexpr (HasArg) arg_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(arg), t.outNumel);
        InitBuffers(t);
        mte2v_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
        vmte2_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
        vmte3_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
        mte3v_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
        vs_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_S>());
        sv_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::S_V>());
        smte3_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::S_MTE3>());
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        for (uint64_t unit = begin; unit < end; ++unit) {
            if (t_->outputGroup == 1)
                ProcessSingleUnit(unit);
            else
                ProcessOutputGroup(unit);
        }
    }

  private:
    __aicore__ inline void ProcessSingleUnit(uint64_t unit) {
        const uint64_t task = unit / t_->featureTiles;
        const uint64_t col = (unit % t_->featureTiles) * t_->tileK;
        if constexpr (BatchType) {
            if (t_->batchReduce && t_->indexMode == 2) {
                ProcessGroupedTile(task, task + 1, col);
                return;
            }
        }
        ProcessTile(task, col);
    }
    __aicore__ inline void ProcessOutputGroup(uint64_t unit) {
        const uint64_t first = unit * t_->outputGroup;
        const uint64_t last = Min64(first + t_->outputGroup, t_->totalTasks);
        for (uint64_t col = 0; col < t_->K; col += t_->tileK) {
            // All tasks in an ownership group remain on one core, including
            // initial-output updates and the general fallback.
            if (!t_->hasInitial)
                ProcessGroupedTile(first, last, col);
            else
                for (uint64_t task = first; task < last; ++task) ProcessTile(task, col);
        }
    }
    __aicore__ inline void InitBuffers(const SegmentCsrTilingData &t) {
        const uint32_t k = t.tileK;
        const uint32_t outputRows = 1;
        pipe_.InitBuffer(srcBuf_, t.tileRows * k * sizeof(T));
        pipe_.InitBuffer(accBuf_, k * sizeof(Work));
        pipe_.InitBuffer(outBuf_, outputRows * k * sizeof(T));
        if constexpr (Floating && !std::is_same<T, float>::value)
            pipe_.InitBuffer(workBuf_, t.tileRows * k * sizeof(Work));
        if constexpr (!Floating && !std::is_same<T, int32_t>::value) pipe_.InitBuffer(rowBuf_, k * sizeof(Work));
        if constexpr (std::is_same<T, int16_t>::value) pipe_.InitBuffer(workBuf_, k * sizeof(float));
        if constexpr (ExtremeBatchType && std::is_same<T, float>::value) {
            if (t.batchReduce) pipe_.InitBuffer(workBuf_, (t.tileRows / 2) * k * sizeof(float));
        }
        if constexpr (OrderedFloat) {
            const uint32_t scratch = (t.tileRows / 2 ? t.tileRows / 2 : 1) * k;
            // Input rows have implicit zero low parts. Only the first tree
            // level's surviving half needs stored corrections.
            if constexpr (std::is_same<T, half>::value) {
                // In the batch plan Cast has consumed srcBuf before any low
                // part is written. Its half tile holds half a float tile.
                if (!t.batchReduce || t.tileRows < 2) pipe_.InitBuffer(correctionBuf_, scratch * sizeof(float));
            } else
                pipe_.InitBuffer(correctionBuf_, scratch * sizeof(float));
            pipe_.InitBuffer(sumScratchBuf_, scratch * sizeof(float));
            pipe_.InitBuffer(zScratchBuf_, scratch * sizeof(float));
            pipe_.InitBuffer(accCorrectionBuf_, k * sizeof(float));
            pipe_.InitBuffer(finiteMaskBuf_,
                             ((scratch + 7) / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes);
        }
        if constexpr (Extreme && !HasArg)
            pipe_.InitBuffer(maskBuf_,
                             ((t.tileRows * k + 7) / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes);
        if constexpr (HasArg) {
            pipe_.InitBuffer(extremeScratchBuf_, k * sizeof(float));
            pipe_.InitBuffer(argFBuf_, outputRows * k * sizeof(float));
            pipe_.InitBuffer(arg64Buf_, outputRows * k * sizeof(int64_t));
            const uint32_t maskLanes = k;
            const uint32_t maskBytes = ((maskLanes + 7) / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes;
            pipe_.InitBuffer(maskBuf_, maskBytes);
            pipe_.InitBuffer(nanMaskBuf_, maskBytes);
        }
    }
    __aicore__ static inline uint64_t Min64(uint64_t a, uint64_t b) { return a < b ? a : b; }
    // CPU torch_scatter 2.1.2 divides the wrapped source-dtype numerator
    // by a source-dtype count, using C++ truncation toward zero.
    __aicore__ static inline int64_t IntegerMean(int64_t value, uint64_t rows) {
        const int32_t nativeCount = static_cast<int32_t>(rows);
        const int64_t count = static_cast<T>(nativeCount > 0 ? nativeCount : 1);
        return count ? value / count : 0;
    }
    __aicore__ static inline Work ExtremeLimit() {
        // torch_scatter uses opmath FP32 extrema initialization for half/BF16.
        if constexpr (Floating)
            return Reduce == SegmentMin ? 0x1.fffffep127f : -0x1.fffffep127f;
        else
            return Reduce == SegmentMin ? INT32_MAX : INT32_MIN;
    }
    template <HardEvent Event> __aicore__ inline void Sync(event_t id) {
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }
    __aicore__ inline uint64_t IndexBase(uint64_t batch) {
        if (t_->indexPrefixRank == 0) return batch * t_->indexBatchStride;
        uint64_t base = 0;
        for (int32_t axis = t_->indexPrefixRank - 1; axis >= 0; --axis) {
            base += (batch % t_->indexPrefixSizes[axis]) * t_->indexPrefixStrides[axis];
            batch /= t_->indexPrefixSizes[axis];
        }
        return base;
    }
    __aicore__ inline void Bounds(uint64_t batch, uint64_t seg, uint64_t &first, uint64_t &last) {
        if (t_->indexMode == 2) {
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
        const uint64_t base = IndexBase(batch);
        first = index_.GetValue(base + seg * t_->indexRowStride);
        last = index_.GetValue(base + (seg + 1) * t_->indexRowStride);
        cached_ = true;
        lastBatch_ = batch;
        lastSegment_ = seg;
        lastFirst_ = first;
        lastEnd_ = last;
    }
    __aicore__ inline void LoadRows(uint64_t offset, uint32_t rows, uint32_t valid, uint32_t width) {
        auto src = srcBuf_.Get<T>();
        const uint64_t gap = (t_->K - valid) * sizeof(T);
        // DataCopyPad's explicit padding is limited to one 32B block. Arg
        // trees use wider physical rows, so place the remaining gap in the UB
        // destination stride instead of requesting an illegal large pad.
        const uint32_t transferWidth =
            (valid + kDmaAlignBytes / sizeof(T) - 1) / (kDmaAlignBytes / sizeof(T)) * (kDmaAlignBytes / sizeof(T));
        const uint32_t dstStride = (width - transferWidth) * sizeof(T) / kDmaAlignBytes;
        const DataCopyPadExtParams<T> pad(true, 0, static_cast<uint8_t>(transferWidth - valid), 0);
        if (valid == width && width == t_->K) {
            // Every physical source row is contiguous and 32-byte aligned.
            // A single dense DMA avoids per-row block issue overhead even
            // when a long or irregular segment uses the generic reducer.
            DataCopy(src, src_[offset], rows * width);
            Sync<HardEvent::MTE2_V>(mte2v_);
            return;
        }
        if (gap <= UINT32_MAX) {
            DataCopyPad(src, src_[offset],
                        {static_cast<uint16_t>(rows), valid * static_cast<uint32_t>(sizeof(T)),
                         static_cast<uint32_t>(gap), dstStride, 0},
                        pad);
        } else {
            for (uint32_t row = 0; row < rows; ++row)
                DataCopyPad(src[row * width], src_[offset + row * t_->K],
                            {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}, pad);
        }
        Sync<HardEvent::MTE2_V>(mte2v_);
    }
    __aicore__ inline void SanitizeExtremeRows(LocalTensor<Work> values, uint32_t count) {
        if constexpr (Floating && Extreme && !HasArg) {
            auto mask = maskBuf_.Get<uint8_t>();
            const uint32_t rounded = (count + kVectorLanes - 1) / kVectorLanes * kVectorLanes;
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
        else if constexpr (Floating) {
            auto work = workBuf_.Get<Work>();
            Cast(work, srcBuf_.Get<T>(), RoundMode::CAST_NONE, rows * width);
            PipeBarrier<PIPE_V>();
            return work;
        } else
            return rowBuf_.Get<Work>();
    }
    __aicore__ inline LocalTensor<Work> WorkRow(uint32_t row, uint32_t width, LocalTensor<Work> work) {
        if constexpr (Floating || std::is_same<T, Work>::value)
            return work[row * width];
        else {
            auto source = srcBuf_.Get<T>()[row * width];
            auto floats = workBuf_.Get<float>();
            Cast(floats, source, RoundMode::CAST_NONE, width);
            PipeBarrier<PIPE_V>();
            Cast(work, floats, RoundMode::CAST_RINT, width);
            PipeBarrier<PIPE_V>();
            return work;
        }
    }
    __aicore__ inline void Combine(LocalTensor<Work> dst, LocalTensor<Work> a, LocalTensor<Work> b, uint32_t count) {
        if constexpr (Floating && Extreme && HasArg) {
            auto temp = extremeScratchBuf_.Get<float>();
            auto mask = maskBuf_.Get<uint8_t>();
            const uint32_t rounded = (count + kVectorLanes - 1) / kVectorLanes * kVectorLanes;
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
            if (rows % 2) Combine(work, work, work[(rows - 1) * width], width);
            rows = halfRows;
        }
    }
    __aicore__ inline LocalTensor<float> Correction() {
        if constexpr (std::is_same<T, half>::value) {
            if (t_->batchReduce && t_->tileRows >= 2) return srcBuf_.Get<float>();
        }
        return correctionBuf_.Get<float>();
    }
    __aicore__ inline void UpdateArg(LocalTensor<Work> acc, LocalTensor<Work> row, float position, uint32_t width) {
        // Called BEFORE the value merge. Equal candidates preserve the earliest
        // argument. Integer comparisons use exact min/max plus equality because
        // arch22 vector Compare supports only equality for int32.
        auto mask = maskBuf_.Get<uint8_t>();
        auto candidate = extremeScratchBuf_.Get<Work>();
        auto arg = argFBuf_.Get<float>();
        const uint32_t rounded = (width + kVectorLanes - 1) / kVectorLanes * kVectorLanes;
        if constexpr (Floating) {
            Compare(mask, row, acc, Reduce == SegmentMin ? CMPMODE::LT : CMPMODE::GT, rounded);
            PipeBarrier<PIPE_V>();
            Not(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(), rounded / 16);
        } else {
            if constexpr (Reduce == SegmentMin)
                Min(candidate, row, acc, rounded);
            else
                Max(candidate, row, acc, rounded);
            PipeBarrier<PIPE_V>();
            Compare(mask, candidate, acc, CMPMODE::EQ, rounded);
        }
        PipeBarrier<PIPE_V>();
        Select(arg, mask, arg, position, SELMODE::VSEL_TENSOR_SCALAR_MODE, rounded);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void FillWideArg(uint32_t valid) {
        Sync<HardEvent::V_S>(vs_);
        auto arg = arg64Buf_.Get<int64_t>();
        for (uint32_t col = 0; col < valid; ++col) arg.SetValue(col, static_cast<int64_t>(t_->M));
        Sync<HardEvent::S_V>(sv_);
    }
    __aicore__ inline void MergeWideArg(uint64_t base, uint32_t valid) {
        Sync<HardEvent::V_S>(vs_);
        auto relative = argFBuf_.Get<float>();
        auto arg = arg64Buf_.Get<int64_t>();
        for (uint32_t col = 0; col < valid; ++col) {
            const float pos = relative.GetValue(col);
            if (pos >= 0) arg.SetValue(col, static_cast<int64_t>(base) + static_cast<int64_t>(pos));
        }
        Sync<HardEvent::S_V>(sv_);
    }
    __aicore__ inline void PrepareVectorArg(uint64_t first, uint64_t rows, uint32_t width) {
        auto arg = arg64Buf_.Get<int64_t>();
        auto relative = argFBuf_.Get<float>();
        if (!rows)
            Duplicate(relative, static_cast<float>(static_cast<int32_t>(t_->M)), width);
        else {
            Adds(relative, relative, static_cast<float>(static_cast<int32_t>(first)), width);
            PipeBarrier<PIPE_V>();
            auto mask = maskBuf_.Get<uint8_t>();
            const uint8_t repeats = (width + kVectorLanes - 1) / kVectorLanes;
            CompareScalar(mask, relative, static_cast<float>(static_cast<int32_t>(first)), CMPMODE::GE, kVectorLanes,
                          repeats, {1, 1, 1, 8});
            PipeBarrier<PIPE_V>();
            Select(relative, mask, relative, static_cast<float>(static_cast<int32_t>(t_->M)),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, kVectorLanes, repeats, {1, 1, 1, 8, 8, 1});
        }
        PipeBarrier<PIPE_V>();
        // The f32->i64 instruction supports a 64-element destination mask.
        for (uint32_t col = 0; col < width; col += kVectorLanes)
            Cast(arg[col], relative[col], RoundMode::CAST_RINT,
                 static_cast<uint32_t>(Min64(kVectorLanes, width - col)));
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PrepareScalarArg(uint64_t first, uint64_t rows, uint32_t valid) {
        auto arg = arg64Buf_.Get<int64_t>();
        auto relative = argFBuf_.Get<float>();
        Sync<HardEvent::V_S>(vs_);
        for (uint32_t col = 0; col < valid; ++col) {
            const float pos = rows ? relative.GetValue(col) : -1.0f;
            arg.SetValue(col, pos < 0 ? static_cast<int64_t>(t_->M)
                                      : static_cast<int64_t>(first) + static_cast<int64_t>(pos));
        }
    }
    __aicore__ inline void ClearNanArg(uint64_t rows, uint32_t valid) {
        auto arg = arg64Buf_.Get<int64_t>();
        if constexpr (Floating) {
            if (!rows) return;
            Sync<HardEvent::V_S>(vs_);
            auto values = accBuf_.Get<Work>();
            for (uint32_t col = 0; col < valid; ++col) {
                const Work value = values.GetValue(col);
                if (value != value) arg.SetValue(col, static_cast<int64_t>(t_->M));
            }
        }
    }
    __aicore__ inline void StoreArg(uint64_t offset, uint64_t first, uint64_t rows, uint32_t valid, uint32_t width,
                                    bool wide) {
        if (!wide && t_->M <= (1ULL << 24)) {
            PrepareVectorArg(first, rows, width);
            Sync<HardEvent::V_MTE3>(vmte3_);
        } else {
            if (!wide) PrepareScalarArg(first, rows, valid);
            ClearNanArg(rows, valid);
            Sync<HardEvent::S_MTE3>(smte3_);
        }
        DataCopyPad(arg_[offset], arg64Buf_.Get<int64_t>(),
                    {1, valid * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0});
        Sync<HardEvent::MTE3_V>(mte3v_);
    }
    // CPU initializes extrema in opmath dtype and ignores unordered inputs.
    // Clamp infinity sentinels to the FP32 finite limit before final narrowing.
    __aicore__ inline void NormalizeAllNan(LocalTensor<Work> values, uint64_t, uint32_t count, uint32_t) {
        if constexpr (Floating && Extreme) {
            if constexpr (Reduce == SegmentMin)
                Mins(values, values, ExtremeLimit(), count);
            else
                Maxs(values, values, ExtremeLimit(), count);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void DivideMean(LocalTensor<float> values, uint32_t count, uint64_t rows) {
        auto denominator = zScratchBuf_.Get<float>();
        if constexpr (!std::is_same<T, float>::value) {
            // The CPU narrows the completed numerator before division.
            auto narrowed = t_->batchReduce ? srcBuf_.Get<T>() : outBuf_.Get<T>();
            Cast(narrowed, values, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            Cast(values, narrowed, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
        Duplicate(denominator, CountFloat(rows > 0 ? rows : 1), count);
        PipeBarrier<PIPE_V>();
        if constexpr (!std::is_same<T, float>::value) {
            auto narrowed = t_->batchReduce ? srcBuf_.Get<T>() : outBuf_.Get<T>();
            Cast(narrowed, denominator, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            Cast(denominator, narrowed, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
        Div(values, values, denominator, count);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void StoreValue(uint64_t offset, uint64_t rows, uint32_t valid, uint32_t width) {
        auto acc = accBuf_.Get<Work>();
        NormalizeAllNan(acc, offset, valid, width);
        if constexpr (Floating) {
            if constexpr (Reduce == SegmentMean) DivideMean(acc, width, rows);
            if constexpr (std::is_same<T, float>::value) {
                Sync<HardEvent::V_MTE3>(vmte3_);
                DataCopyPad(out_[offset], acc, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
            } else {
                auto output = outBuf_.Get<T>();
                Cast(output, acc, RoundMode::CAST_RINT, width);
                PipeBarrier<PIPE_V>();
                Sync<HardEvent::V_MTE3>(vmte3_);
                DataCopyPad(out_[offset], output, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
            }
        } else if constexpr (std::is_same<T, int32_t>::value && Reduce != SegmentMean) {
            Sync<HardEvent::V_MTE3>(vmte3_);
            DataCopyPad(out_[offset], acc, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        } else {
            auto output = outBuf_.Get<T>();
            Sync<HardEvent::V_S>(vs_);
            for (uint32_t col = 0; col < valid; ++col) {
                // Wrap numerator/count in T and use C++ truncation toward zero.
                int64_t value = static_cast<T>(acc.GetValue(col));
                if constexpr (Reduce == SegmentMean) value = IntegerMean(value, rows);
                output.SetValue(col, static_cast<T>(value));
            }
            Sync<HardEvent::S_MTE3>(smte3_);
            DataCopyPad(out_[offset], output, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        }
        Sync<HardEvent::MTE3_V>(mte3v_);
    }
    // Partition an ownership group into the largest adjacent source spans
    // that fit the planner's row capacity.  The entire ownership group remains
    // on one AIV, including subgroups and fallback tasks, so an unaligned
    // output row never gains a writer on a different physical 32-byte block.
    __aicore__ inline bool TryUniformGroup(uint64_t firstTask, uint64_t count, uint64_t col) {
        if constexpr (BatchType || ExtremeBatchType) {
            // The host proof covers every segment, so the batch needs neither
            // per-segment GM reads nor a second uniformity scan in local arrays.
            if (t_->indexMode == 2 && t_->batchReduce && (count >= 2 || BatchType)) {
                const uint32_t rows = static_cast<uint32_t>(t_->M / t_->nSegments);
                const uint32_t width = static_cast<uint32_t>(Min64(t_->tileK, t_->K - col));
                LoadRows(firstTask * rows * t_->K + col, count * rows, width, width);
                auto work = WorkRows(count * rows, width);
                SanitizeExtremeRows(work, count * rows * width);
                if constexpr (BatchType)
                    BatchedOrderedSum(work, rows, count, width, firstTask * t_->K + col);
                else
                    BatchedExtremeTree(work, rows, count, width, firstTask * t_->K);
                return true;
            }
        }
        return false;
    }
    __aicore__ inline void ProcessGroupedTile(uint64_t firstTask, uint64_t lastTask, uint64_t col) {
        const uint64_t count = lastTask - firstTask;
        if (!count || count > kMaxGroupedSegments) {
            for (uint64_t task = firstTask; task < lastTask; ++task) ProcessTile(task, col);
            return;
        }
        if constexpr (HasArg) {
            // StoreArg's wide path merges absolute row offsets after each
            // tile.  Keep that path's established ordering until a dedicated
            // wide grouped arg buffer is available.
            if (t_->M > (1ULL << 24)) {
                for (uint64_t task = firstTask; task < lastTask; ++task) ProcessTile(task, col);
                return;
            }
        }
        if (TryUniformGroup(firstTask, count, col)) return;
        uint64_t starts[kMaxGroupedSegments] = {0};
        uint64_t ends[kMaxGroupedSegments] = {0};
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
    // Pair equally sized row stripes inside each segment. For a narrow row,
    // several rows occupy one 64-float vector without a packing pass. Stripe
    // size divides every power-of-two segment; no pair crosses its boundary.
    __aicore__ static inline uint32_t PairStripeRows(uint32_t rows, uint32_t width) {
        uint32_t stripe = 1;
        while (stripe * 2 <= rows / 2 && stripe * 2 * width <= kVectorLanes) stripe *= 2;
        return stripe;
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
            uint32_t stripe = PairStripeRows(rows, width);
            // A pair stage always consumes at least one row per stripe.
            if (stripe == 0) stripe = 1;
            const uint32_t span = stripe * width;
            const uint32_t pairs = groups * (rows / 2) / stripe;
            const BinaryRepeatParams pair{1,
                                          1,
                                          1,
                                          static_cast<uint8_t>(span / 8),
                                          static_cast<uint8_t>(2 * span / 8),
                                          static_cast<uint8_t>(2 * span / 8)};
            for (uint32_t first = 0; first < pairs; first += kVectorRepeatLimit) {
                const uint8_t repeats = static_cast<uint8_t>(Min64(kVectorRepeatLimit, pairs - first));
                for (uint32_t col = 0; col < span; col += kVectorLanes) {
                    const uint64_t mask = Min64(kVectorLanes, span - col);
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
    // Every lane follows the CPU opmath recurrence in original row order.
    // Segment batching only changes which independent lanes share instructions.
    __aicore__ inline void BatchedOrderedSum(LocalTensor<float> work, uint32_t rows, uint32_t groups, uint32_t width,
                                             uint64_t output) {
        const uint32_t count = groups * width;
        auto result = sumScratchBuf_.Get<float>();
        Duplicate(result, 0.0f, count);
        PipeBarrier<PIPE_V>();
        const uint32_t sourceStride = rows * width / 8;
        if (groups == 1) {
            for (uint32_t row = 0; row < rows; ++row) {
                Add(result, result, work[row * width], width);
                PipeBarrier<PIPE_V>();
            }
        } else if (sourceStride <= kVectorStrideLimit && width / 8 <= kVectorStrideLimit) {
            const BinaryRepeatParams ordered{1,
                                             1,
                                             1,
                                             static_cast<uint8_t>(width / 8),
                                             static_cast<uint8_t>(width / 8),
                                             static_cast<uint8_t>(sourceStride)};
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t col = 0; col < width; col += kVectorLanes)
                    Add(result[col], result[col], work[row * width + col], Min64(kVectorLanes, width - col),
                        static_cast<uint8_t>(groups), ordered);
                PipeBarrier<PIPE_V>();
            }
        } else {
            // Gather supports byte offsets wider than the vector repeat-stride
            // field. Offsets are exact INT32 UB addresses, never source values.
            LocalTensor<int32_t> offsets = Correction().template ReinterpretCast<int32_t>();
            for (uint32_t group = 0; group < groups; ++group)
                CreateVecIndex(offsets[group * width], static_cast<int32_t>(group * rows * width), width);
            PipeBarrier<PIPE_V>();
            ShiftLeft(offsets.ReinterpretCast<uint32_t>(), offsets.ReinterpretCast<uint32_t>(), uint32_t(2), count);
            PipeBarrier<PIPE_V>();
            auto values = zScratchBuf_.Get<float>();
            for (uint32_t row = 0; row < rows; ++row) {
                Gather(values, work, offsets.ReinterpretCast<uint32_t>(), row * width * 4, count);
                PipeBarrier<PIPE_V>();
                Add(result, result, values, count);
                PipeBarrier<PIPE_V>();
            }
        }
        if constexpr (Reduce == SegmentMean) DivideMean(result, count, rows);
        StoreBatchedResult(result, count, output, false, width);
    }
    __aicore__ inline void WriteBatchedRows(LocalTensor<T> result, uint32_t count, uint64_t output, uint32_t width) {
        // Zero width denotes a contiguous result in the batching protocol.
        if (width == 0) {
            DataCopy(out_[output], result, count);
            return;
        }
        if (width == t_->K) {
            DataCopy(out_[output], result, count);
            return;
        }
        const uint32_t rows = count / width;
        const uint64_t gap = (t_->K - width) * sizeof(T);
        if (gap <= UINT32_MAX) {
            DataCopyPad(out_[output], result,
                        {static_cast<uint16_t>(rows), width * static_cast<uint32_t>(sizeof(T)), 0,
                         static_cast<uint32_t>(gap), 0});
        } else {
            for (uint32_t row = 0; row < rows; ++row) DataCopy(out_[output + row * t_->K], result[row * width], width);
        }
    }
    __aicore__ inline void StoreBatchedResult(LocalTensor<float> high, uint32_t count, uint64_t output,
                                              bool inScratch = false, uint32_t rowWidth = 0) {
        NormalizeAllNan(high, output, count, rowWidth ? rowWidth : static_cast<uint32_t>(t_->K));
        if constexpr (std::is_same<T, float>::value) {
            Sync<HardEvent::V_MTE3>(vmte3_);
            WriteBatchedRows(high, count, output, rowWidth);
        } else {
            auto result = srcBuf_.Get<T>();
            if constexpr (ExtremeBatchType) {
                // A final partial in srcBuf must not be cast into overlapping
                // half addresses. The consumed float input provides room.
                if (inScratch) result = workBuf_.Get<T>();
            }
            Cast(result, high, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            Sync<HardEvent::V_MTE3>(vmte3_);
            WriteBatchedRows(result, count, output, rowWidth);
        }
        Sync<HardEvent::MTE3_V>(mte3v_);
        Sync<HardEvent::V_MTE2>(vmte2_);
    }
    // Every bound in this subgroup belongs to the same batch, neighboring
    // intervals are contiguous, and their combined row span fits tileRows.
    // The reductions reuse the same ordered accumulator and arg routines as
    // ProcessTile; only the source load/cast is shared.
    __aicore__ inline bool TryBatchedBounds(LocalTensor<Work> work, uint32_t width, uint64_t firstTask, uint64_t count,
                                            const uint64_t *starts, const uint64_t *ends) {
        if constexpr (BatchType || ExtremeBatchType) {
            const uint64_t rows = ends[0] - starts[0];
            bool uniform = t_->batchReduce && count >= 2 && rows >= 2 && (rows & (rows - 1)) == 0 && width == t_->K;
            for (uint64_t i = 1; uniform && i < count; ++i) uniform = ends[i] - starts[i] == rows;
            if (uniform) {
                if constexpr (BatchType)
                    BatchedOrderedSum(work, static_cast<uint32_t>(rows), static_cast<uint32_t>(count), width,
                                      firstTask * t_->K);
                else
                    BatchedExtremeTree(work, static_cast<uint32_t>(rows), static_cast<uint32_t>(count), width,
                                       firstTask * t_->K);
                return true;
            }
        }
        return false;
    }
    __aicore__ inline void ReduceGroupRows(LocalTensor<Work> work, uint32_t rowOffset, uint64_t rows, uint32_t width) {
        auto acc = accBuf_.Get<Work>();
        if constexpr (Extreme) {
            Duplicate(acc, ExtremeLimit(), width);
            PipeBarrier<PIPE_V>();
            if constexpr (!HasArg) {
                // Value extrema may reduce in a fixed vector tree. The
                // ordered arg path below separately preserves the first
                // strict improvement required by the native CPU reference.
                Tree(work[rowOffset * width], static_cast<uint32_t>(rows), width);
                Combine(acc, acc, work[rowOffset * width], width);
            } else {
                Duplicate(argFBuf_.Get<float>(), -1.0f, width);
                PipeBarrier<PIPE_V>();
                for (uint64_t row = 0; row < rows; ++row) {
                    auto value = WorkRow(rowOffset + static_cast<uint32_t>(row), width, work);
                    UpdateArg(acc, value, static_cast<float>(static_cast<int32_t>(row)), width);
                    Combine(acc, acc, value, width);
                }
            }
        } else if constexpr (OrderedFloat) {
            for (uint64_t row = 0; row < rows; ++row) {
                Add(acc, acc, work[(rowOffset + row) * width], width);
                PipeBarrier<PIPE_V>();
            }
        } else if constexpr ((Floating || std::is_same<T, int32_t>::value) && !HasArg) {
            Tree(work[rowOffset * width], static_cast<uint32_t>(rows), width);
            Combine(acc, acc, work[rowOffset * width], width);
        } else {
            for (uint64_t row = 0; row < rows; ++row) {
                auto value = WorkRow(rowOffset + static_cast<uint32_t>(row), width, work);
                Combine(acc, acc, value, width);
            }
        }
    }
    __aicore__ inline void ProcessBoundedGroup(uint64_t firstTask, uint64_t count, uint64_t batch,
                                               const uint64_t *starts, const uint64_t *ends, uint64_t col) {
        const uint64_t totalRows = ends[count - 1] - starts[0];
        const uint32_t valid = static_cast<uint32_t>(Min64(t_->tileK, t_->K - col));
        const uint32_t align = kDmaAlignBytes / sizeof(T);
        const uint32_t width = (valid + align - 1) / align * align;
        if (totalRows)
            LoadRows((batch * t_->M + starts[0]) * t_->K + col, static_cast<uint32_t>(totalRows), valid, width);
        // No DMA or cast instruction is submitted for an all-empty group.
        auto work = totalRows ? WorkRows(static_cast<uint32_t>(totalRows), width) : srcBuf_.Get<Work>();
        if (totalRows) SanitizeExtremeRows(work, static_cast<uint32_t>(totalRows) * width);
        auto acc = accBuf_.Get<Work>();
        if constexpr (HasArg) Duplicate(argFBuf_.Get<float>(), -1.0f, width);
        if (TryBatchedBounds(work, width, firstTask, count, starts, ends)) return;

        for (uint64_t i = 0; i < count; ++i) {
            const uint64_t task = firstTask + i;
            const uint64_t rows = ends[i] - starts[i];
            const uint32_t rowOffset = static_cast<uint32_t>(starts[i] - starts[0]);
            const uint64_t output = task * t_->K + col;
            // Every independent segment starts from zero, including empty spans.
            if constexpr (OrderedFloat) {
                Duplicate(accCorrectionBuf_.Get<float>(), 0.0f, width);
                PipeBarrier<PIPE_V>();
            }
            Duplicate(acc, static_cast<Work>(0), width);
            PipeBarrier<PIPE_V>();
            if constexpr (HasArg) Duplicate(argFBuf_.Get<float>(), -1.0f, width);
            if (!rows) {
                StoreValue(output, 0, valid, width);
                if constexpr (HasArg) StoreArg(output, starts[i], 0, valid, width, false);
                continue;
            }

            ReduceGroupRows(work, rowOffset, rows, width);
            if constexpr (HasArg) StoreArg(output, starts[i], rows, valid, width, false);
            StoreValue(output, rows, valid, width);
        }
        // Do not overwrite srcBuf until all vector users in this group have
        // consumed it.  The next LoadRows waits on this event as well.
        Sync<HardEvent::V_MTE2>(vmte2_);
    }
    __aicore__ inline void ReduceTileRows(LocalTensor<Work> work, uint32_t count, uint32_t width, uint64_t rowBase,
                                          bool wide) {
        auto acc = accBuf_.Get<Work>();
        if constexpr (OrderedFloat) {
            for (uint32_t row = 0; row < count; ++row) {
                Add(acc, acc, work[row * width], width);
                PipeBarrier<PIPE_V>();
            }
        } else if constexpr ((Floating || std::is_same<T, int32_t>::value) && !HasArg) {
            Tree(work, count, width);
            Combine(acc, acc, work, width);
        } else {
            for (uint32_t row = 0; row < count; ++row) {
                auto value = WorkRow(row, width, work);
                if constexpr (HasArg)
                    UpdateArg(acc, value, static_cast<float>(static_cast<int32_t>(wide ? row : rowBase + row)), width);
                Combine(acc, acc, value, width);
            }
        }
    }
    __aicore__ inline void InitTile(bool wide, uint32_t valid, uint32_t width) {
        auto acc = accBuf_.Get<Work>();
        // Each CPU-order recurrence starts from zero.
        if constexpr (OrderedFloat) {
            // Scratch initialization is independent of input allocation history.
            Duplicate(accCorrectionBuf_.Get<float>(), 0.0f, width);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (HasArg) {
            if (wide) FillWideArg(valid);
            Duplicate(argFBuf_.Get<float>(), -1.0f, width);
        }
        Duplicate(acc, static_cast<Work>(0), width);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ProcessTile(uint64_t task, uint64_t col) {
        const uint64_t batch = task / t_->nSegments, seg = task % t_->nSegments;
        uint64_t first = 0, last = 0;
        Bounds(batch, seg, first, last);
        const uint64_t rows = last - first;
        const uint32_t valid = static_cast<uint32_t>(Min64(t_->tileK, t_->K - col));
        const uint32_t align = kDmaAlignBytes / sizeof(T);
        const uint32_t width = (valid + align - 1) / align * align;
        const uint64_t output = task * t_->K + col;
        auto acc = accBuf_.Get<Work>();
        const bool wide = rows > (1ULL << 24);
        InitTile(wide, valid, width);
        if (!rows) {
            if (!t_->hasInitial) StoreValue(output, 0, valid, width);
            if constexpr (HasArg) StoreArg(output, first, 0, valid, width, false);
            return;
        }

        for (uint64_t rowBase = 0; rowBase < rows; rowBase += t_->tileRows) {
            const uint32_t count = static_cast<uint32_t>(Min64(t_->tileRows, rows - rowBase));
            LoadRows((batch * t_->M + first + rowBase) * t_->K + col, count, valid, width);
            auto work = WorkRows(count, width);
            SanitizeExtremeRows(work, count * width);
            if constexpr (HasArg) {
                if (wide) {
                    Duplicate(argFBuf_.Get<float>(), -1.0f, width);
                    PipeBarrier<PIPE_V>();
                }
            }
            if constexpr (Extreme) {
                if (!rowBase && !t_->hasInitial) {
                    Duplicate(acc, ExtremeLimit(), width);
                    if constexpr (HasArg) Duplicate(argFBuf_.Get<float>(), -1.0f, width);
                    PipeBarrier<PIPE_V>();
                }
            }
            ReduceTileRows(work, count, width, rowBase, wide);
            if constexpr (HasArg) {
                if (wide) MergeWideArg(first + rowBase, valid);
            }
            Sync<HardEvent::V_MTE2>(vmte2_);
        }
        if constexpr (HasArg) StoreArg(output, first, rows, valid, width, wide);
        StoreValue(output, rows, valid, width);
    }
    const SegmentCsrTilingData *t_ = nullptr;
    GlobalTensor<T> src_, out_;
    GlobalTensor<int64_t> index_, arg_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> srcBuf_, workBuf_, accBuf_, outBuf_, rowBuf_;
    TBuf<TPosition::VECCALC> argFBuf_, arg64Buf_, maskBuf_, nanMaskBuf_;
    TBuf<TPosition::VECCALC> extremeScratchBuf_;
    TBuf<TPosition::VECCALC> correctionBuf_, sumScratchBuf_, zScratchBuf_, accCorrectionBuf_, finiteMaskBuf_;
    event_t mte2v_, vmte2_, vmte3_, mte3v_, vs_, sv_, smte3_;
    bool cached_ = false;
    uint64_t lastBatch_ = 0, lastSegment_ = 0, lastFirst_ = 0, lastEnd_ = 0;
};

} // namespace opsgnn::csr

// ============================================================================
// Full-width and byte integer reductions
// ============================================================================

namespace opsgnn::csr {
using namespace AscendC;

struct CsrTaskRange {
    uint64_t first, last, beginColumn, endColumn;
};
__aicore__ inline CsrTaskRange TaskRange(const SegmentCsrTilingData &t, uint64_t unit) {
    const uint64_t first = t.outputGroup == 1 ? unit / t.featureTiles : unit * t.outputGroup;
    const uint64_t last = first + t.outputGroup < t.totalTasks ? first + t.outputGroup : t.totalTasks;
    const uint64_t begin = t.outputGroup == 1 ? unit % t.featureTiles * t.tileK : 0;
    const uint64_t end = t.outputGroup == 1 && begin + t.tileK < t.K ? begin + t.tileK : t.K;
    return {first, last, begin, end};
}

template <typename Kernel> __aicore__ inline void ProcessIntegerTiles(const SegmentCsrTilingData &t, Kernel &kernel) {
    const uint64_t begin = GetBlockIdx() * t.totalUnits / t.usedBlockNum;
    const uint64_t end = (GetBlockIdx() + 1) * t.totalUnits / t.usedBlockNum;
    for (uint64_t unit = begin; unit < end; ++unit) {
        const auto range = TaskRange(t, unit);
        for (uint64_t col = range.beginColumn; col < range.endColumn; col += t.tileK) kernel.ProcessRange(range, col);
    }
}

// Byte reductions and value+arg int64 use scalar integer arithmetic on AIV.
// DMA stages complete padded rows; ownership groups keep partial 32-byte output
// blocks on one core. No source integer is converted to floating point.
template <typename T, SegmentReduce Reduce, bool HasArg> class SegmentIntegerScalar {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(T))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out), t.outNumel);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), t.indexNumel);
        if constexpr (HasArg) arg_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(arg), t.outNumel);
        pipe_.InitBuffer(input_, t.tileRows * t.tileK * sizeof(T));
        pipe_.InitBuffer(output_, t.tileK * sizeof(T));
        pipe_.InitBuffer(sums_, t.tileK * sizeof(uint64_t));
        pipe_.InitBuffer(args_, t.tileK * sizeof(int64_t));
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * t_->totalUnits / t_->usedBlockNum;
        const uint64_t end = (GetBlockIdx() + 1) * t_->totalUnits / t_->usedBlockNum;
        for (uint64_t unit = begin; unit < end; ++unit) {
            const auto range = TaskRange(*t_, unit);
            const auto first = range.first, last = range.last;
            const auto colBegin = range.beginColumn, colEnd = range.endColumn;
            for (uint64_t task = first; task < last; ++task)
                for (uint64_t col = colBegin; col < colEnd; col += t_->tileK) Tile(task, col);
        }
    }

  private:
    __aicore__ static inline uint64_t Min(uint64_t a, uint64_t b) { return a < b ? a : b; }
    __aicore__ inline uint64_t Lower(uint64_t batch, uint64_t key) {
        return static_cast<uint64_t>(index_.GetValue(batch * t_->indexBatchStride + key));
    }

    __aicore__ inline void LoadIntegerRows(uint64_t sourceOffset, uint32_t count, uint32_t valid, uint32_t width) {
        auto input = input_.Get<T>();
        const uint64_t sourceGap = (t_->K - valid) * sizeof(T);
        const auto pad = DataCopyPadExtParams<T>(true, 0, static_cast<uint8_t>(width - valid), 0);
        if (sourceGap <= UINT32_MAX) {
            DataCopyPad(input, src_[sourceOffset],
                        {static_cast<uint16_t>(count), valid * static_cast<uint32_t>(sizeof(T)),
                         static_cast<uint32_t>(sourceGap), 0, 0},
                        pad);
        } else {
            for (uint32_t row = 0; row < count; ++row)
                DataCopyPad(input[row * width], src_[sourceOffset + row * t_->K],
                            {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}, pad);
        }
    }
    __aicore__ inline void StoreIntegerTile(uint64_t destination, uint64_t first, uint64_t last, uint32_t valid) {
        auto sums = sums_.Get<uint64_t>();
        auto output = output_.Get<T>();
        auto args = args_.Get<int64_t>();
        for (uint32_t c = 0; c < valid; ++c) {
            int64_t value = static_cast<int64_t>(sums.GetValue(c));
            if constexpr (Reduce == SegmentMean) {
                // Reducer::write receives an int count before converting to T.
                const int32_t nativeCount = static_cast<int32_t>(last - first);
                const int64_t denominator = static_cast<T>(nativeCount > 0 ? nativeCount : 1);
                if (denominator) {
                    const int64_t numerator = static_cast<T>(value);
                    // The only possible signed division overflow is byte min/-1;
                    // division is promoted to int64 before narrowing the result.
                    value = numerator / denominator;
                } else {
                    // The pinned ARM CPU extension returns zero for its C++
                    // zero-divisor case. Reproduce the observed result explicitly
                    // without issuing a device division by zero.
                    value = 0;
                }
            }
            output.SetValue(c, static_cast<T>(value));
        }
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(out_[destination], output, {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        if constexpr (HasArg) DataCopyPad(arg_[destination], args, {1, valid * 8, 0, 0, 0});
        Sync<HardEvent::MTE3_S>();
    }
    __aicore__ inline void Tile(uint64_t task, uint64_t col) {
        const uint64_t batch = task / t_->nSegments, seg = task % t_->nSegments;
        const uint64_t first = t_->indexMode == 2 ? seg * (t_->M / t_->nSegments) : Lower(batch, seg);
        const uint64_t last = t_->indexMode == 2 ? first + t_->M / t_->nSegments : Lower(batch, seg + 1);
        const uint32_t valid = static_cast<uint32_t>(Min(t_->tileK, t_->K - col));
        const uint32_t align = kDmaAlignBytes / sizeof(T), width = (valid + align - 1) / align * align;
        const uint64_t destination = task * t_->K + col;
        auto input = input_.Get<T>();
        auto output = output_.Get<T>();
        auto sums = sums_.Get<uint64_t>();
        auto args = args_.Get<int64_t>();
        if (t_->hasInitial) {
            DataCopyPad(output, out_[last > first && first == 0 ? batch * t_->nSegments * t_->K + col : destination],
                        {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
                        DataCopyPadExtParams<T>(true, 0, static_cast<uint8_t>(width - valid), 0));
            Sync<HardEvent::MTE2_S>();
        }
        for (uint32_t c = 0; c < valid; ++c) {
            sums.SetValue(c, t_->hasInitial ? static_cast<uint64_t>(static_cast<int64_t>(output.GetValue(c))) : 0);
            args.SetValue(c, t_->M);
        }
        for (uint64_t start = first; start < last; start += t_->tileRows) {
            const uint32_t count = static_cast<uint32_t>(Min(t_->tileRows, last - start));
            const uint64_t sourceOffset = (batch * t_->M + start) * t_->K + col;
            LoadIntegerRows(sourceOffset, count, valid, width);
            Sync<HardEvent::MTE2_S>();
            for (uint32_t r = 0; r < count; ++r) {
                for (uint32_t c = 0; c < valid; ++c) {
                    const int64_t next = static_cast<int64_t>(input.GetValue(r * width + c));
                    const int64_t old = static_cast<int64_t>(sums.GetValue(c));
                    if constexpr (Reduce == SegmentSum || Reduce == SegmentMean) {
                        // Unsigned arithmetic defines two's-complement modulo wrap.
                        sums.SetValue(c, sums.GetValue(c) + static_cast<uint64_t>(next));
                    } else if ((!t_->hasInitial && start == first && r == 0) ||
                               (Reduce == SegmentMin ? next < old : next > old)) {
                        sums.SetValue(c, static_cast<uint64_t>(next));
                        args.SetValue(c, start + r);
                    }
                }
            }
            Sync<HardEvent::S_MTE2>();
        }
        StoreIntegerTile(destination, first, last, valid);
    }
    const SegmentCsrTilingData *t_;
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
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(src), t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(int64_t))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
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
        pipe_.InitBuffer(mask_, ((4 * capacity_ + 7) / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes);
        auto offsets = offsets_.Get<uint32_t>();
        for (uint32_t i = 0; i < 2 * outputCapacity_; ++i) {
            offsets.SetValue(i, i * 8);
            offsets.SetValue(words + i, (i % 2 ? i - 1 : 2 * outputCapacity_) * 4);
            offsets.SetValue(words * 2 + i, (i % 2 ? 2 * outputCapacity_ : i + 1) * 4);
        }
        Sync<HardEvent::S_V>();
    }
    __aicore__ inline void Process() { ProcessIntegerTiles(*t_, *this); }
    __aicore__ inline void ProcessRange(const CsrTaskRange &range, uint64_t col) {
        const auto first = range.first, last = range.last;
        const uint64_t rows = t_->indexMode == 2 ? t_->M / t_->nSegments : 0;
        const uint32_t width = static_cast<uint32_t>(Min(t_->tileK, t_->K - col));
        if (!t_->hasInitial && rows && rows <= t_->tileRows && (rows & (rows - 1)) == 0 && width % 8 == 0 &&
            (last - first) * rows <= t_->tileRows && first / t_->nSegments == (last - 1) / t_->nSegments) {
            Batch(first, last - first, col, rows, width);
        } else {
            for (uint64_t task = first; task < last; ++task) Tile(task, col);
        }
    }

  private:
    __aicore__ static inline uint64_t Min(uint64_t a, uint64_t b) { return a < b ? a : b; }
    __aicore__ inline uint64_t Lower(uint64_t batch, uint64_t key) {
        return static_cast<uint64_t>(index_.GetValue(batch * t_->indexBatchStride + key));
    }

    __aicore__ inline void Load(uint64_t offset, uint32_t rows, uint32_t valid, uint32_t width) {
        CopyInt64Rows(input_.Get<int64_t>(), src_, {offset, t_->K, rows, valid, width});
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
            for (uint32_t start = 0; start < pairs; start += kVectorRepeatLimit)
                for (uint32_t col = 0; col < lanes; col += kVectorLanes)
                    Add(next[start * lanes + col], work[start * 2 * lanes + col], work[start * 2 * lanes + lanes + col],
                        Min(kVectorLanes, lanes - col), static_cast<uint8_t>(Min(kVectorRepeatLimit, pairs - start)),
                        pair);
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
        Sync<HardEvent::V_S>();
        carry.SetValue(2 * outputCapacity_, 0);
        Sync<HardEvent::S_V>();
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
        Sync<HardEvent::V_S>();
        both.ReinterpretCast<uint32_t>().SetValue(2 * outputCapacity_, 0);
        Sync<HardEvent::S_V>();
        Gather(either.ReinterpretCast<uint32_t>(), both.ReinterpretCast<uint32_t>(), offsets[2 * outputCapacity_ + 8],
               0, words);
        PipeBarrier<PIPE_V>();
        Add(acc, sum, either.ReinterpretCast<int32_t>(), words);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ShiftMeanWords(LocalTensor<int32_t> value, uint32_t words, uint32_t shift) {
        auto sum = b_.Get<int32_t>(), temp = c_.Get<int32_t>();
        auto bits = input_.Get<uint32_t>();
        auto offsets = offsets_.Get<uint32_t>();
        const uint32_t offsetWidth = 2 * outputCapacity_ + 8;
        ShiftRight(sum.ReinterpretCast<uint32_t>(), value.ReinterpretCast<uint32_t>(), shift, words);
        ShiftLeft(temp.ReinterpretCast<uint32_t>(), value.ReinterpretCast<uint32_t>(),
                  static_cast<uint32_t>(32 - shift), words);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_S>();
        temp.SetValue(2 * outputCapacity_, 0);
        Sync<HardEvent::S_V>();
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
    __aicore__ inline void PowerOfTwoMean(LocalTensor<int64_t> value64, uint32_t elements, uint32_t shift) {
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
        Sync<HardEvent::V_S>();
        sum.SetValue(2 * outputCapacity_, 0);
        Sync<HardEvent::S_V>();
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
        Sync<HardEvent::V_S>();
        temp.SetValue(2 * outputCapacity_, 0);
        Sync<HardEvent::S_V>();
        Gather(bits, temp.ReinterpretCast<uint32_t>(), offsets[offsetWidth], 0, words);
        PipeBarrier<PIPE_V>();
        Add(value, sum, bits.ReinterpretCast<int32_t>(), words);
        PipeBarrier<PIPE_V>();
        ShiftMeanWords(value, words, shift);
    }
    __aicore__ inline void Mean(LocalTensor<int64_t> value64, uint32_t elements, uint64_t rows) {
        if constexpr (Reduce == SegmentMean) {
            const int32_t nativeCount = static_cast<int32_t>(rows);
            rows = nativeCount > 0 ? static_cast<uint64_t>(nativeCount) : 1;
            if (rows > 1 && rows <= (1ULL << 31) && (rows & (rows - 1)) == 0) {
                uint32_t shift = 0;
                for (uint64_t n = rows; n > 1; n /= 2) ++shift;
                PowerOfTwoMean(value64, elements, shift);
            } else if (rows != 1) {
                Sync<HardEvent::V_S>();
                for (uint32_t c = 0; c < elements; ++c)
                    value64.SetValue(c, value64.GetValue(c) / static_cast<int64_t>(rows ? rows : 1));
                Sync<HardEvent::S_V>();
            }
        }
    }
    __aicore__ inline void Store(LocalTensor<int64_t> value, uint64_t output, uint32_t groups, uint32_t valid,
                                 uint32_t width) {
        Sync<HardEvent::V_MTE3>();
        const uint64_t gap = (t_->K - valid) * 8;
        if (gap <= UINT32_MAX) {
            DataCopyPad(out_[output], value,
                        {static_cast<uint16_t>(groups), valid * 8, (width - valid) / 4, static_cast<uint32_t>(gap), 0});
        } else {
            for (uint32_t row = 0; row < groups; ++row)
                DataCopyPad(out_[output + row * t_->K], value[row * width], {1, valid * 8, 0, 0, 0});
        }
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::V_MTE2>();
    }
    __aicore__ inline void Batch(uint64_t task, uint32_t groups, uint64_t col, uint32_t rows, uint32_t width) {
        Load(task * rows * t_->K + col, rows * groups, width, width);
        auto result = Pack(Tree(Digits(rows * groups * width), rows, groups, width), groups * width);
        Mean(result.template ReinterpretCast<int64_t>(), groups * width, rows);
        Store(result.template ReinterpretCast<int64_t>(), task * t_->K + col, groups, width, width);
    }
    __aicore__ inline void Tile(uint64_t task, uint64_t col) {
        const uint64_t batch = task / t_->nSegments, seg = task % t_->nSegments;
        const uint64_t first = t_->indexMode == 2 ? seg * (t_->M / t_->nSegments) : Lower(batch, seg);
        const uint64_t last = t_->indexMode == 2 ? first + t_->M / t_->nSegments : Lower(batch, seg + 1);
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
            Sync<HardEvent::MTE2_V>();
        }
        for (uint64_t start = first; start < last;) {
            uint32_t rows = t_->tileRows;
            while (rows > last - start) rows /= 2;
            Load((batch * t_->M + start) * t_->K + col, rows, valid, width);
            auto work = Digits(rows * width);
            auto result = Pack(Tree(work, rows, 1, width), width);
            Merge(result, width * 2);
            Sync<HardEvent::V_MTE2>();
            start += rows;
        }
        Mean(acc.ReinterpretCast<int64_t>(), valid, last - first);
        Store(acc.ReinterpretCast<int64_t>(), task * t_->K + col, 1, valid, width);
    }
    const SegmentCsrTilingData *t_;
    uint32_t capacity_, outputCapacity_;
    GlobalTensor<int64_t> src_, out_, index_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_, digits_, tree_, a_, b_, c_, offsets_, acc_, mask_;
};
} // namespace opsgnn::csr

// ============================================================================
// Full-width INT64 extrema
// ============================================================================

namespace opsgnn::csr {
using namespace AscendC;
// Lexicographic signed-high/unsigned-low 32-bit comparisons preserve every
// int64 bit, including adjacent integers above 2^53 and both signed limits.
template <SegmentReduce Reduce> class SegmentInt64Extreme {
  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, const SegmentCsrTilingData &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(src), t.srcNumel);
        if (t.l2Bytes && t.srcNumel > t.l2Bytes / sizeof(int64_t))
            src_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
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
        for (uint32_t i = 0; i < t.tileK * 2; ++i) bias.SetValue(i, i % 2 ? 0 : INT32_MIN);
        Sync<HardEvent::S_V>();
        Duplicate(even_.Get<uint16_t>(), static_cast<uint16_t>(0x5555), 1024);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() { ProcessIntegerTiles(*t_, *this); }
    __aicore__ inline void ProcessRange(const CsrTaskRange &range, uint64_t col) {
        const auto first = range.first, last = range.last;
        const uint64_t rows = t_->indexMode == 2 ? t_->M / t_->nSegments : 0;
        const uint32_t width = static_cast<uint32_t>(Min(t_->tileK, t_->K - col));
        if (!t_->hasInitial && rows >= 2 && rows <= t_->tileRows && (rows & (rows - 1)) == 0 &&
            width % kInt64VectorLanes == 0 && (last - first) * rows <= t_->tileRows &&
            first / t_->nSegments == (last - 1) / t_->nSegments) {
            Batch(first, last - first, col, rows, width);
        } else
            for (uint64_t task = first; task < last; ++task) Tile(task, col);
    }

  private:
    __aicore__ static inline uint64_t Min(uint64_t a, uint64_t b) { return a < b ? a : b; }
    __aicore__ inline uint64_t Lower(uint64_t batch, uint64_t key) {
        return static_cast<uint64_t>(index_.GetValue(batch * t_->indexBatchStride + key));
    }

    __aicore__ inline void Bias(LocalTensor<int32_t> value, uint32_t rows, uint32_t width) {
        const uint32_t words = width * 2;
        const BinaryRepeatParams params{1, 1, 1, static_cast<uint8_t>(words / 8), static_cast<uint8_t>(words / 8), 0};
        for (uint32_t start = 0; start < rows; start += kVectorRepeatLimit)
            for (uint32_t col = 0; col < words; col += kVectorLanes)
                Add(value[start * words + col], value[start * words + col], bias_.Get<int32_t>()[col], kVectorLanes,
                    static_cast<uint8_t>(Min(kVectorRepeatLimit, rows - start)), params);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline LocalTensor<int32_t> Load(uint64_t offset, uint32_t rows, uint32_t valid, uint32_t width) {
        CopyInt64Rows(input_.Get<int64_t>(), src_, {offset, t_->K, rows, valid, width});
        auto work = input_.Get<int32_t>();
        Bias(work, rows, width);
        return work;
    }
    __aicore__ inline void ResolveWordWinner(uint32_t count) {
        auto win = winner_.Get<uint16_t>(), equal = equal_.Get<uint16_t>();
        auto temp = scratchMask_.Get<uint16_t>(), even = even_.Get<uint16_t>();
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
        for (uint32_t start = 0; start < pairs; start += kVectorRepeatLimit) {
            const uint8_t repeats = static_cast<uint8_t>(Min(kVectorRepeatLimit, pairs - start));
            const uint32_t count = repeats * 4;
            for (uint32_t col = 0; col < words; col += kVectorLanes) {
                const uint32_t source = start * srcStride + col, output = start * dstStride + col;
                if constexpr (Reduce == SegmentMin)
                    AscendC::Min(candidate[output], left[source], right[source], kVectorLanes, repeats, arithmetic);
                else
                    AscendC::Max(candidate[output], left[source], right[source], kVectorLanes, repeats, arithmetic);
                PipeBarrier<PIPE_V>();
                Compare(win.ReinterpretCast<uint8_t>(), candidate[output], right[source], CMPMODE::EQ, kVectorLanes,
                        repeats, compare);
                Compare(equal.ReinterpretCast<uint8_t>(), left[source], right[source], CMPMODE::EQ, kVectorLanes,
                        repeats, equalParams);
                PipeBarrier<PIPE_V>();
                ResolveWordWinner(count);
                Select(dst.ReinterpretCast<float>()[output], win.ReinterpretCast<uint8_t>(),
                       right.ReinterpretCast<float>()[source], left.ReinterpretCast<float>()[source],
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, kVectorLanes, repeats, arithmetic);
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
        Sync<HardEvent::V_MTE3>();
        const uint64_t gap = (t_->K - valid) * 8;
        if (gap <= UINT32_MAX) {
            DataCopyPad(out_[output], value.ReinterpretCast<int64_t>(),
                        {static_cast<uint16_t>(groups), valid * 8, (width - valid) / 4, static_cast<uint32_t>(gap), 0});
        } else {
            for (uint32_t row = 0; row < groups; ++row)
                DataCopyPad(out_[output + row * t_->K], value.ReinterpretCast<int64_t>()[row * width],
                            {1, valid * 8, 0, 0, 0});
        }
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::V_MTE2>();
    }
    __aicore__ inline void Batch(uint64_t task, uint32_t groups, uint64_t col, uint32_t rows, uint32_t width) {
        auto work = Load(task * rows * t_->K + col, rows * groups, width, width);
        Store(Tree(work, rows, groups, width), task * t_->K + col, groups, width, width);
    }
    __aicore__ inline void Tile(uint64_t task, uint64_t col) {
        const uint64_t batch = task / t_->nSegments, seg = task % t_->nSegments;
        const uint64_t first = t_->indexMode == 2 ? seg * (t_->M / t_->nSegments) : Lower(batch, seg);
        const uint64_t last = t_->indexMode == 2 ? first + t_->M / t_->nSegments : Lower(batch, seg + 1);
        const uint32_t valid = static_cast<uint32_t>(Min(t_->tileK, t_->K - col));
        const uint32_t width = (valid + kInt64VectorLanes - 1) / kInt64VectorLanes * kInt64VectorLanes;
        auto acc = acc_.Get<int32_t>();
        Duplicate(acc, Reduce == SegmentMin ? INT32_MAX : INT32_MIN, width * 2);
        PipeBarrier<PIPE_V>();
        if (t_->hasInitial) {
            DataCopyPad(acc.ReinterpretCast<int64_t>(),
                        out_[last > first && first == 0 ? batch * t_->nSegments * t_->K + col : task * t_->K + col],
                        {1, valid * 8, 0, 0, 0},
                        DataCopyPadExtParams<int64_t>(true, 0, static_cast<uint8_t>((valid + 3) / 4 * 4 - valid), 0));
            Sync<HardEvent::MTE2_V>();
            Bias(acc, 1, width);
        }
        for (uint64_t start = first; start < last;) {
            uint32_t rows = t_->tileRows;
            while (rows > last - start) rows /= 2;
            auto work = Load((batch * t_->M + start) * t_->K + col, rows, valid, width);
            auto value = Tree(work, rows, 1, width);
            Pair(acc, acc, value, 1, width * 2, width * 2, width * 2);
            Sync<HardEvent::V_MTE2>();
            start += rows;
        }
        if (first == last && !t_->hasInitial) {
            Duplicate(acc, 0, width * 2);
            PipeBarrier<PIPE_V>();
            Bias(acc, 1, width);
        }
        Store(acc, task * t_->K + col, 1, valid, width);
    }
    const SegmentCsrTilingData *t_;
    GlobalTensor<int64_t> src_, out_, index_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_, tree_, candidate_, acc_, bias_, winner_, equal_, scratchMask_, even_;
};
} // namespace opsgnn::csr

// ============================================================================
// CPU-compatible extrema argument carry
// ============================================================================

namespace opsgnn::csr {
using namespace AscendC;

// torch_scatter CPU keeps one args(K), initialized to zero, across every
// flattened batch/segment. A nonempty segment with no strict improvement over
// the opmath initializer retains that earlier argument. This can intentionally
// be outside the current segment; reproduce the pinned CPU behavior safely.
// Empty segments retain their separately initialized M sentinel and do not
// reset the carry. This pass is needed only by value+arg entry points.
template <typename T, SegmentReduce Reduce> class SegmentCsrArgCarry {
    static constexpr uint32_t Width = kVectorLanes;

  public:
    __aicore__ inline void Run(GM_ADDR ptr, GM_ADDR output, GM_ADDR argument, const SegmentCsrTilingData &t) {
        GlobalTensor<T> values;
        GlobalTensor<int64_t> pointers, arguments;
        values.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output), t.outNumel);
        arguments.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(argument), t.outNumel);
        pointers.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ptr), t.indexNumel);
        TPipe pipe;
        TBuf<TPosition::VECCALC> valueBuffer, argBuffer, carryBuffer;
        pipe.InitBuffer(valueBuffer, Width * sizeof(T));
        pipe.InitBuffer(argBuffer, Width * sizeof(int64_t));
        pipe.InitBuffer(carryBuffer, Width * sizeof(int64_t));
        auto value = valueBuffer.Get<T>();
        auto arg = argBuffer.Get<int64_t>();
        auto carry = carryBuffer.Get<int64_t>();
        const uint64_t tiles = (t.K + Width - 1) / Width;
        const uint64_t begin = GetBlockIdx() * tiles / GetBlockNum();
        const uint64_t end = (GetBlockIdx() + 1) * tiles / GetBlockNum();
        for (uint64_t tile = begin; tile < end; ++tile) {
            const uint64_t column = tile * Width;
            const uint32_t valid = t.K - column < Width ? t.K - column : Width;
            for (uint32_t c = 0; c < valid; ++c) carry.SetValue(c, 0);
            for (uint64_t task = 0; task < t.totalTasks; ++task) {
                const uint64_t batch = task / t.nSegments, segment = task % t.nSegments;
                const uint64_t base = batch * t.indexBatchStride + segment;
                if (pointers.GetValue(base) == pointers.GetValue(base + 1)) continue;
                const uint64_t offset = task * t.K + column;
                DataCopyPad(value, values[offset], {1, valid * static_cast<uint32_t>(sizeof(T)), 0, 0, 0},
                            DataCopyPadExtParams<T>(false, 0, 0, 0));
                DataCopyPad(arg, arguments[offset], {1, valid * 8, 0, 0, 0},
                            DataCopyPadExtParams<int64_t>(false, 0, 0, 0));
                Sync<HardEvent::MTE2_S>();
                bool changed = false;
                for (uint32_t c = 0; c < valid; ++c) {
                    if (IsInitializer(value, c)) {
                        arg.SetValue(c, carry.GetValue(c));
                        changed = true;
                    } else
                        carry.SetValue(c, arg.GetValue(c));
                }
                if (changed) {
                    Sync<HardEvent::S_MTE3>();
                    DataCopyPad(arguments[offset], arg, {1, valid * 8, 0, 0, 0});
                    Sync<HardEvent::MTE3_S>();
                }
                Sync<HardEvent::S_MTE2>();
            }
        }
    }

  private:
    __aicore__ static inline bool IsInitializer(LocalTensor<T> value, uint32_t c) {
        if constexpr (std::is_same<T, float>::value) {
            const auto bits = value.template ReinterpretCast<uint32_t>().GetValue(c);
            return bits == (Reduce == SegmentMin ? 0x7f7fffffU : 0xff7fffffU);
        } else if constexpr (std::is_same<T, half>::value) {
            const auto bits = value.template ReinterpretCast<uint16_t>().GetValue(c);
            return bits == (Reduce == SegmentMin ? 0x7c00U : 0xfc00U);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            const auto bits = value.template ReinterpretCast<uint16_t>().GetValue(c);
            return bits == (Reduce == SegmentMin ? 0x7f80U : 0xff80U);
        } else if constexpr (std::is_same<T, uint8_t>::value) {
            return value.GetValue(c) == (Reduce == SegmentMin ? 255 : 0);
        } else if constexpr (std::is_same<T, int8_t>::value) {
            return value.GetValue(c) == (Reduce == SegmentMin ? 127 : -128);
        } else if constexpr (std::is_same<T, int32_t>::value) {
            return value.GetValue(c) == (Reduce == SegmentMin ? INT32_MAX : INT32_MIN);
        } else {
            return value.GetValue(c) == (Reduce == SegmentMin ? INT64_MAX : INT64_MIN);
        }
    }
};

template <typename T, SegmentReduce Reduce>
__global__ __aicore__ void SegmentCsrArgCarryKernel(GM_ADDR ptr, GM_ADDR output, GM_ADDR argument,
                                                    SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SegmentCsrArgCarry<T, Reduce> op;
    op.Run(ptr, output, argument, t);
}
} // namespace opsgnn::csr

// ============================================================================
// INT32 uniform reductions
// ============================================================================

// ============================================================================
// Generic dtype/reduce dispatch
// ============================================================================

using namespace AscendC;
using namespace opsgnn::csr;
template <typename T, SegmentReduce Reduce, bool HasArg>
__aicore__ inline void Run(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, const SegmentCsrTilingData &t) {
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
    } else {
        SegmentReduceDevice<T, Reduce, HasArg> op;
        op.Init(src, index, out, arg, t);
        op.Process();
    }
}

template <typename T, SegmentReduce Reduce>
__global__ __aicore__ void SegmentCsrKernel(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg,
                                            SegmentCsrTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (Reduce == SegmentMin || Reduce == SegmentMax) {
        if (t.hasArg) {
            Run<T, Reduce, true>(src, index, out, arg, t);
            return;
        }
    }
    Run<T, Reduce, false>(src, index, out, arg, t);
}

template <typename T>
void Dispatch(void *src, int64_t *index, void *out, int64_t *arg, SegmentCsrTilingData t, uint32_t argCarryBlocks,
              aclrtStream stream) {
#define LAUNCH(R)                                                                                                      \
    SegmentCsrKernel<T, R>                                                                                             \
        <<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)index, (GM_ADDR)out, (GM_ADDR)arg, t)
    switch (t.reduceCode) {
    case SegmentSum:
        LAUNCH(SegmentSum);
        break;
    case SegmentMean:
        LAUNCH(SegmentMean);
        break;
    case SegmentMin:
        LAUNCH(SegmentMin);
        break;
    case SegmentMax:
        LAUNCH(SegmentMax);
        break;
    }
#undef LAUNCH
    if (t.hasArg && t.reduceCode >= SegmentMin) {
        // Distinct cores own complete 32-byte argument blocks. A row whose
        // width is not a multiple of four int64 lanes uses one owner.
        if (t.reduceCode == SegmentMin)
            SegmentCsrArgCarryKernel<T, SegmentMin>
                <<<argCarryBlocks, nullptr, stream>>>((GM_ADDR)index, (GM_ADDR)out, (GM_ADDR)arg, t);
        else
            SegmentCsrArgCarryKernel<T, SegmentMax>
                <<<argCarryBlocks, nullptr, stream>>>((GM_ADDR)index, (GM_ADDR)out, (GM_ADDR)arg, t);
    }
}

// ============================================================================
// Exact source and pointer validation
// ============================================================================

using namespace AscendC;
namespace {
constexpr uint32_t kValidationElements = 4096;
class ValidateHalfRange {
  public:
    __aicore__ inline void Run(GM_ADDR src, GM_ADDR flags, int64_t elements) {
        input.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(src), elements);
        output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flags), GetBlockNum() * 8);

        pipe.InitBuffer(sourceBuffer, kValidationElements * sizeof(half));
        pipe.InitBuffer(maskBuffer, kValidationElements / 8);
        pipe.InitBuffer(predicateBuffer, kValidationElements / 16 * sizeof(float));
        pipe.InitBuffer(resultBuffer, 2048);
        const int64_t begin = GetBlockIdx() * elements / GetBlockNum();
        const int64_t end = (GetBlockIdx() + 1) * elements / GetBlockNum();
        bool proved = true;
        for (int64_t offset = begin; proved && offset < end; offset += kValidationElements) {
            const uint32_t count = end - offset < kValidationElements ? end - offset : kValidationElements;
            proved = CheckTile(offset, count);
        }
        Sync<HardEvent::V_S>();
        auto flagsLocal = resultBuffer.Get<int32_t>();
        for (uint32_t i = 0; i < 8; ++i) flagsLocal.SetValue(i, i == 0 && proved ? 1 : 0);
        Sync<HardEvent::S_MTE3>();
        // Every core owns an entire 32-byte block, including empty partitions.
        DataCopy(output[GetBlockIdx() * 8], flagsLocal, 8);
    }

  private:
    __aicore__ inline bool CheckTile(int64_t offset, uint32_t count) {
        auto source = sourceBuffer.Get<half>();
        auto mask = maskBuffer.Get<uint8_t>();
        auto predicate = predicateBuffer.Get<float>();
        auto result = resultBuffer.Get<float>();
        const uint32_t rounded = (count + kHalfVectorLanes - 1) / kHalfVectorLanes * kHalfVectorLanes;
        // Compare consumes complete 128-half repeats. Initialize every padded
        // element, including padding beyond DataCopyPad's final 32-byte block.
        Duplicate(source, static_cast<half>(0.0f), rounded);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_MTE2>();
        DataCopyPad(source, input[offset], {1, count * 2, 0, 0, 0},
                    DataCopyPadExtParams<half>(true, 0, static_cast<uint8_t>((16 - count % 16) % 16), 0));
        Sync<HardEvent::MTE2_V>();
        Abs(source, source, rounded);
        PipeBarrier<PIPE_V>();
        // A max-of-magnitudes check could ignore NaNs. Require every original
        // value's comparison bit instead; both NaN and infinity fail <= 6.
        CompareScalar(mask, source, static_cast<half>(6.0f), CMPMODE::LE, rounded);
        PipeBarrier<PIPE_V>();
        const uint32_t words = rounded / 16;
        auto bits = mask.ReinterpretCast<uint16_t>();
        Not(bits, bits, words);
        PipeBarrier<PIPE_V>();
        Cast(predicate, bits.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, words);
        PipeBarrier<PIPE_V>();
        Abs(predicate, predicate, words);
        PipeBarrier<PIPE_V>();
        ReduceMax(result, predicate, result[kVectorLanes], words, false);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_S>();
        const bool proved = result.GetValue(0) == 0.0f;
        Sync<HardEvent::S_V>();
        return proved;
    }
    GlobalTensor<half> input;
    GlobalTensor<int32_t> output;
    TPipe pipe;
    TBuf<TPosition::VECCALC> sourceBuffer, maskBuffer, predicateBuffer, resultBuffer;
};
__global__ __aicore__ void ValidateHalfKernel(GM_ADDR src, GM_ADDR flags, int64_t elements) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ValidateHalfRange validation;
    validation.Run(src, flags, elements);
}

class ValidateInt64Range {
  public:
    __aicore__ inline void Run(GM_ADDR src, GM_ADDR flags, int64_t elements, int64_t bound) {
        input.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(src), elements);
        output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flags), GetBlockNum() * 8);

        pipe.InitBuffer(sourceBuffer, kValidationElements * 8);
        pipe.InitBuffer(lowBuffer, kValidationElements * 4);
        pipe.InitBuffer(highBuffer, kValidationElements * 4);
        pipe.InitBuffer(workBuffer, kValidationElements * 4);
        pipe.InitBuffer(maskBuffer, kValidationElements / 4);
        bool proved = bound >= 0 && bound <= INT32_MAX;
        const int64_t begin = GetBlockIdx() * elements / GetBlockNum();
        const int64_t end = (GetBlockIdx() + 1) * elements / GetBlockNum();
        for (int64_t offset = begin; proved && offset < end; offset += kValidationElements) {
            const uint32_t count = end - offset < kValidationElements ? end - offset : kValidationElements;
            proved = CheckTile(offset, count, bound);
        }
        Sync<HardEvent::V_S>();
        auto result = lowBuffer.Get<int32_t>();
        for (uint32_t i = 0; i < 8; ++i) result.SetValue(i, i == 0 && proved ? 1 : 0);
        Sync<HardEvent::S_MTE3>();
        // A full 32-byte store per core prevents partial-block writer overlap.
        DataCopy(output[GetBlockIdx() * 8], result, 8);
    }

  private:
    __aicore__ inline bool CheckWords(uint32_t rounded, int64_t bound) {
        auto low = lowBuffer.Get<int32_t>();
        auto high = highBuffer.Get<int32_t>();
        auto work = workBuffer.Get<int32_t>();
        auto valid = maskBuffer.Get<uint8_t>();
        auto within = valid[kValidationElements / 8];
        uint64_t reserved = 0;
        GatherMask(low, sourceBuffer.Get<int32_t>(), 1, true, rounded * 2, {1, 1, 8, 0}, reserved);
        GatherMask(high, sourceBuffer.Get<int32_t>(), 2, true, rounded * 2, {1, 1, 8, 0}, reserved);
        PipeBarrier<PIPE_V>();
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
        ReduceMax(result, predicate, result[kVectorLanes], rounded / 16, false);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_S>();
        const bool proved = result.GetValue(0) == 0.0f;
        Sync<HardEvent::S_V>();
        return proved;
    }
    __aicore__ inline bool CheckTile(int64_t offset, uint32_t count, int64_t bound) {
        auto source = sourceBuffer.Get<int64_t>();
        const uint32_t rounded = (count + kVectorLanes - 1) / kVectorLanes * kVectorLanes;
        // Zero padding is inside the tested interval for every valid bound.
        Duplicate(sourceBuffer.Get<int32_t>(), 0, rounded * 2);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_MTE2>();
        DataCopyPad(source, input[offset], {1, count * 8, 0, 0, 0},
                    DataCopyPadExtParams<int64_t>(true, 0, static_cast<uint8_t>((4 - count % 4) % 4), 0));
        Sync<HardEvent::MTE2_V>();
        return CheckWords(rounded, bound);
    }
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

void SegmentCsrValidateHalf(void *src, int64_t numel, int32_t *perCoreValid, int blocks, aclrtStream stream) {
    ValidateHalfKernel<<<blocks, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)perCoreValid, numel);
}

void SegmentCsrValidateInt64(void *src, int64_t numel, int64_t absBound, int32_t *perCoreValid, int blocks,
                             aclrtStream stream) {
    ValidateInt64Kernel<<<blocks, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)perCoreValid, numel, absBound);
}

namespace {
class ValidateCsrPointers {
  public:
    __aicore__ inline void Run(GM_ADDR ptr, GM_ADDR flags, int64_t batches, int64_t segments, int64_t length) {
        GlobalTensor<int64_t> input;
        GlobalTensor<int32_t> output;
        const uint64_t count = static_cast<uint64_t>(batches) * (segments + 1);
        input.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ptr), count);
        output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flags), GetBlockNum() * 8);
        TPipe pipe;
        TBuf<TPosition::VECCALC> inBuffer, outBuffer;
        pipe.InitBuffer(inBuffer, 512 * sizeof(int64_t));
        pipe.InitBuffer(outBuffer, kDmaAlignBytes);
        auto values = inBuffer.Get<int64_t>();
        auto result = outBuffer.Get<int32_t>();
        const uint64_t begin = GetBlockIdx() * count / GetBlockNum();
        const uint64_t end = (GetBlockIdx() + 1) * count / GetBlockNum();
        const int64_t rows = segments > 0 ? length / segments : 0;
        bool valid = true;
        bool uniform = segments > 0 && length % segments == 0;
        int64_t previous = begin && begin % (segments + 1) ? input.GetValue(begin - 1) : 0;
        for (uint64_t base = begin; base < end; base += 512) {
            const uint32_t active = end - base < 512 ? end - base : 512;
            DataCopyPad(values, input[base], {1, active * 8, 0, 0, 0}, DataCopyPadExtParams<int64_t>(false, 0, 0, 0));
            Sync<HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < active; ++i) {
                const uint64_t coordinate = (base + i) % (segments + 1);
                const int64_t value = values.GetValue(i);
                valid = valid && value >= 0 && value <= length && (!coordinate || previous <= value);
                uniform = uniform && value == static_cast<int64_t>(coordinate) * rows;
                previous = value;
            }
            Sync<HardEvent::S_MTE2>();
        }
        for (uint32_t i = 0; i < 8; ++i) result.SetValue(i, i == 0 ? valid : (i == 1 ? uniform && valid : 0));
        Sync<HardEvent::S_MTE3>();
        DataCopy(output[GetBlockIdx() * 8], result, 8);
    }

  private:
};
__global__ __aicore__ void ValidateCsrPointersKernel(GM_ADDR ptr, GM_ADDR flags, int64_t batches, int64_t segments,
                                                     int64_t length) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ValidateCsrPointers validation;
    validation.Run(ptr, flags, batches, segments, length);
}
} // namespace

void SegmentCsrValidatePointers(int64_t *indptr, int64_t pointerBatches, int64_t segments, int64_t length,
                                int32_t *perCoreFlags, int blocks, aclrtStream stream) {
    ValidateCsrPointersKernel<<<blocks, nullptr, stream>>>((GM_ADDR)indptr, (GM_ADDR)perCoreFlags, pointerBatches,
                                                           segments, length);
}

aclError SegmentCsrGeneral(void *src, int64_t *index, void *out, int64_t *arg, const SegmentCsrPlan &plan,
                           aclrtStream stream) {
    const auto &t = plan.tiling;
#define CSR_GENERAL(T) Dispatch<T>(src, index, out, arg, t, plan.argCarryBlocks, stream)
    switch (t.dtypeCode) {
    case SEGMENT_CSR_FLOAT32:
        CSR_GENERAL(float);
        break;
    case SEGMENT_CSR_FLOAT16:
        CSR_GENERAL(half);
        break;
    case SEGMENT_CSR_BFLOAT16:
        CSR_GENERAL(bfloat16_t);
        break;
    case SEGMENT_CSR_INT8:
        CSR_GENERAL(int8_t);
        break;
    case SEGMENT_CSR_UINT8:
        CSR_GENERAL(uint8_t);
        break;
    case SEGMENT_CSR_INT32:
        CSR_GENERAL(int32_t);
        break;
    case SEGMENT_CSR_INT64:
        CSR_GENERAL(int64_t);
        break;
    default:
        return ACL_ERROR_INVALID_PARAM;
    }
#undef CSR_GENERAL
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

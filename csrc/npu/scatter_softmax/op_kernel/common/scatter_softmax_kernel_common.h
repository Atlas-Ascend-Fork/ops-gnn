/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_SOFTMAX_KERNEL_COMMON_H
#define OPS_GNN_SCATTER_SOFTMAX_KERNEL_COMMON_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_tiling.h"

/**
 * Shared vocabulary of the grouped-softmax device kernels: the pass tags, the
 * pipeline event set, the UB buffer layout and the fp32 widening helpers.
 *
 * Everything here is dtype independent; the per-pass arithmetic lives in
 * scatter_softmax_kernel_phase.h and the kernel entries in
 * scatter_softmax_kernel.h.
 */

namespace ScatterSoftmaxNpu {

using namespace AscendC;

// One device pass. The row/tile driver and the buffer layout are shared, so a
// pass only contributes its own arithmetic.
//
// The grouped log-sum-exp reuses four of these passes: the maximum and the
// exponential sum are the ones of the softmax family, `Finalize` writes the
// group rows of the reduced output, and `GradElement` is its elementwise
// backward. Neither of the last two reduces, so neither owns a workspace.
//
// The grouped standard deviation adds two reduction passes of its own: `Sum`
// accumulates the group sum and the member count, `Deviation` accumulates the
// squared distance to the group mean. It shares `Finalize` and `GradElement`
// with the log-sum-exp, whose arithmetic is selected by the mode.
enum ScatterSoftmaxPhase {
    kPhaseMaximum = 0,
    kPhaseExp = 1,
    kPhaseNormalize = 2,
    kPhaseGradSum = 3,
    kPhaseGrad = 4,
    kPhaseFinalize = 5,
    kPhaseGradElement = 6,
    kPhaseSum = 7,
    kPhaseDeviation = 8,
};

// Compile-time arithmetic of the shared phase bodies.
//
// `kModeEdgeCsr` is the softmax arithmetic driven by a CSR pointer vector
// instead of per-row group labels: the segment that owns a row is resolved from
// `indptr`, so the edge softmax runs the same three forward passes and the same
// two backward passes as `kModeSoftmax`.
enum ScatterSoftmaxMode {
    kModeSoftmax = 0,
    kModeLogSoftmax = 1,
    kModeLogSumExp = 2,
    kModeStd = 3,
    kModeEdgeCsr = 4,
};

__aicore__ constexpr bool IsLogDomain(ScatterSoftmaxMode mode)
{
    return mode == kModeLogSoftmax || mode == kModeLogSumExp;
}

// The two softmax modes store `exp(src - maximum)` for the normalize pass and
// weight the group gradient by the saved output in the backward.
__aicore__ constexpr bool IsSoftmaxDomain(ScatterSoftmaxMode mode)
{
    return mode == kModeSoftmax || mode == kModeEdgeCsr;
}

// Passes that reduce into a (batch, group, feature) workspace slot.
template <ScatterSoftmaxPhase Phase>
__aicore__ constexpr bool IsReductionPass()
{
    return Phase == kPhaseMaximum || Phase == kPhaseExp ||
           Phase == kPhaseGradSum || Phase == kPhaseSum ||
           Phase == kPhaseDeviation;
}

// Passes that read the reduction workspace of an earlier pass.
template <ScatterSoftmaxPhase Phase>
__aicore__ constexpr bool ReadsWorkspace()
{
    return Phase == kPhaseExp || Phase == kPhaseNormalize ||
           Phase == kPhaseGrad || Phase == kPhaseDeviation;
}

// Passes that keep the reduction workspace resident in UB. The finalize pass
// walks the workspace in row order once and never revisits a slot, so it reads
// the flushed copy from GM instead of reserving the whole thing in UB.
template <ScatterSoftmaxPhase Phase>
__aicore__ constexpr bool UsesUbWorkspace()
{
    return IsReductionPass<Phase>() || ReadsWorkspace<Phase>();
}

// Passes driven by the group labels of every row. The finalize pass walks the
// dense (batch, group) rows of the output and needs no labels.
template <ScatterSoftmaxPhase Phase>
__aicore__ constexpr bool UsesGroups()
{
    return Phase != kPhaseFinalize;
}

// A group-shaped tensor (the saved output or its gradient) is small when the
// reduction workspace fits the accumulator budget. Its rows are then few enough
// to keep resident in UB, which the elementwise backward needs: every source row
// reads one row of the output and one row of its gradient, and with a small
// group count those reads all land on the same few lines and never overlap. The
// tiling reserves two fp32 copies of the workspace for this pass already.
__aicore__ inline bool GroupsFitUb(const ScatterSoftmaxTilingData& tiling)
{
    return tiling.localAccum != 0 && tiling.afterPadded <= tiling.colTile;
}

// The per-group member counts of the sum pass fit the per-core UB buffer when
// the dense counter array does. They then stay resident for the whole pass and
// are flushed with one atomic per group instead of one per row.
__aicore__ inline bool CountsFitUb(const ScatterSoftmaxTilingData& tiling)
{
    return tiling.countElems != 0 &&
           tiling.countElems * sizeof(float) <= SCATTER_SOFTMAX_COUNT_BUDGET_BYTES;
}

// Count slot of a group: the workspace is one fp32 counter per group, padded to
// whole 32-byte blocks, so a per-row atomic targets the block that holds it.
__aicore__ inline uint64_t CountBase(int32_t group)
{
    return static_cast<uint64_t>(group);
}

// Elements one 256-byte vector repeat covers in fp32. The mask API rounds its
// element count down to whole repeats, so the standard-deviation backward rounds
// the count up and lets the last repeat read the padding the row buffer keeps.
constexpr uint32_t kStdCompareRepeat = 64;

__aicore__ inline uint32_t CompareRepeatCount(uint32_t elements)
{
    return (elements + kStdCompareRepeat - 1) / kStdCompareRepeat * kStdCompareRepeat;
}

template <HardEvent Event>
__aicore__ inline void SyncEvent(event_t id)
{
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

// Elements covered by a tile that starts at `begin`.
__aicore__ inline uint32_t TileLength(uint64_t total, uint64_t begin, uint32_t tile)
{
    const uint64_t remaining = total - begin;
    const uint64_t limit = static_cast<uint64_t>(tile);
    return static_cast<uint32_t>(remaining < limit ? remaining : limit);
}

// UB allocations and vector destinations must be 32-byte based.
__aicore__ inline uint32_t AlignedBytes(uint64_t elements, uint64_t elementBytes)
{
    const uint64_t bytes = elements * elementBytes;
    return static_cast<uint32_t>((bytes + 31) / 32 * 32);
}

// A row of `after` features is not 32-byte aligned, so it always travels as one
// byte-counted burst.
__aicore__ inline DataCopyExtParams Bytes1D(uint32_t bytes)
{
    return DataCopyExtParams{1, bytes, 0, 0, 0};
}

template <typename T>
__aicore__ inline DataCopyPadExtParams<T> NoPadding()
{
    return DataCopyPadExtParams<T>(false, 0, 0, 0);
}

// Event pair per pipeline hazard. The four ids are reused for every batch:
// each SyncEvent call sets and immediately waits, so the ordering is exact and
// no id stays pending across a reuse.
struct PipelineSync {
    event_t loadToCompute;   // MTE2 -> V
    event_t computeToLoad;   // V -> MTE2
    event_t computeToStore;  // V -> MTE3
    event_t storeToCompute;  // MTE3 -> V (and, through issue order, MTE2)

    __aicore__ inline void Init(TPipe& pipe)
    {
        loadToCompute = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        computeToLoad = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
        computeToStore = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
        storeToCompute = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
    }

    __aicore__ inline void Release(TPipe& pipe)
    {
        pipe.ReleaseEventID<HardEvent::MTE2_V>(loadToCompute);
        pipe.ReleaseEventID<HardEvent::V_MTE2>(computeToLoad);
        pipe.ReleaseEventID<HardEvent::V_MTE3>(computeToStore);
        pipe.ReleaseEventID<HardEvent::MTE3_V>(storeToCompute);
    }
};

// Buffers and global tensors a pass may touch. Every pass shares the layout so
// the row/tile driver is written once; a pass simply leaves the rest unused.
// The row buffers hold `rowBatch` tiles of `colTile` elements each.
template <typename T>
struct PhaseTensors {
    TBuf<TPosition::VECCALC> narrowBuffer;
    TBuf<TPosition::VECCALC> narrowBuffer2;
    TBuf<TPosition::VECCALC> narrowBuffer3;
    TBuf<TPosition::VECCALC> wideBuffer;
    TBuf<TPosition::VECCALC> wideBuffer2;
    TBuf<TPosition::VECCALC> wideBuffer3;
    TBuf<TPosition::VECCALC> accumBuffer;
    TBuf<TPosition::VECCALC> workspaceBuffer;
    TBuf<TPosition::VECCALC> groupCacheBuffer;
    TBuf<TPosition::VECCALC> groupCacheBuffer2;
    TBuf<TPosition::VECCALC> groupBuffer;
    TBuf<TPosition::VECCALC> countBuffer;
    TBuf<TPosition::VECCALC> oneHotBuffer;
    LocalTensor<T> narrow;
    LocalTensor<T> narrow2;
    LocalTensor<T> narrow3;
    LocalTensor<float> wide;
    LocalTensor<float> wide2;
    LocalTensor<float> wide3;
    LocalTensor<float> accum;
    LocalTensor<float> cached;
    LocalTensor<float> cachedOut;
    LocalTensor<float> cachedGrad;
    LocalTensor<int32_t> groups;
    // The per-core member counts of the sum pass, resident in UB while the
    // whole dense count workspace fits (see CountsFitUb).
    LocalTensor<float> counts;
    // One 32-byte count block per lane, used by the per-row counting fallback.
    LocalTensor<float> onehot;
    // Edge softmax only: the CSR segment cursor of `LoadGroups`. The group of a
    // row is the segment that contains it, and rows are visited in order, so the
    // cursor advances forwards only and reads one boundary per segment instead
    // of one label per row.
    uint64_t csrSegment;
    uint64_t csrNext;
    uint64_t csrColumn;
    bool csrStarted;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<T> srcGm;
    GlobalTensor<T> gradGm;
    GlobalTensor<T> outGm;
    GlobalTensor<T> gradSrcGm;
    GlobalTensor<float> maximumGm;
    GlobalTensor<float> sumGm;
    // Grouped standard deviation only: the fp32 workspace of the squared
    // deviations and the per-group member counts.
    GlobalTensor<float> deviationGm;
    GlobalTensor<float> countGm;
};

// The sum pass counts the members of every group while it accumulates the group
// sums: a count workspace that fits stays in UB for the whole pass, and one that
// does not counts row by row with a 32-byte atomic built here.
template <typename T>
__aicore__ inline void InitCountBuffers(PhaseTensors<T>& tensors, TPipe& pipe,
                                        const ScatterSoftmaxTilingData& tiling)
{
    if (CountsFitUb(tiling)) {
        pipe.InitBuffer(tensors.countBuffer, AlignedBytes(tiling.countElems, sizeof(float)));
        tensors.counts = tensors.countBuffer.template Get<float>();
        return;
    }
    if (tiling.countElems == 0) {
        return;
    }
    pipe.InitBuffer(tensors.oneHotBuffer,
                    AlignedBytes(SCATTER_SOFTMAX_COUNT_ALIGN * SCATTER_SOFTMAX_COUNT_ALIGN,
                                 sizeof(float)));
    tensors.onehot = tensors.oneHotBuffer.template Get<float>();
}

// The saved output and its incoming gradient stay in UB for the whole backward
// pass while the group count allows it: one row of each is read per source row,
// and a small group count makes those reads hit the same few lines. The
// standard-deviation backward reads three group rows, so it goes to GM for all
// of them instead of reserving a third resident copy.
template <ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void InitGroupCache(PhaseTensors<T>& tensors, TPipe& pipe,
                                      const ScatterSoftmaxTilingData& tiling)
{
    if constexpr (Mode != kModeStd) {
        if (GroupsFitUb(tiling)) {
            pipe.InitBuffer(tensors.groupCacheBuffer,
                            AlignedBytes(tiling.workspaceElems, sizeof(float)));
            pipe.InitBuffer(tensors.groupCacheBuffer2,
                            AlignedBytes(tiling.workspaceElems, sizeof(float)));
            tensors.cachedOut = tensors.groupCacheBuffer.template Get<float>();
            tensors.cachedGrad = tensors.groupCacheBuffer2.template Get<float>();
        }
    }
}

template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void InitTensors(PhaseTensors<T>& tensors, TPipe& pipe,
                                   const ScatterSoftmaxTilingData& tiling)
{
    const uint64_t rowElements =
        static_cast<uint64_t>(tiling.colTile) * tiling.rowBatch;
    pipe.InitBuffer(tensors.narrowBuffer, AlignedBytes(rowElements, sizeof(T)));
    pipe.InitBuffer(tensors.narrowBuffer2, AlignedBytes(rowElements, sizeof(T)));
    pipe.InitBuffer(tensors.wideBuffer, AlignedBytes(rowElements, sizeof(float)));
    pipe.InitBuffer(tensors.wideBuffer2, AlignedBytes(rowElements, sizeof(float)));
    // The standard-deviation backward compares one fp32 tile against zero with a
    // whole-repeat count (the mask API rounds the count down to full 64-element
    // repeats), so its last tile reads up to one repeat past `colTile` and the
    // buffer keeps that much room.
    const uint64_t wideExtra =
        (Phase == kPhaseGradElement && Mode == kModeStd) ? kStdCompareRepeat : 0;
    pipe.InitBuffer(tensors.wideBuffer3,
                    AlignedBytes(rowElements + wideExtra, sizeof(float)));
    pipe.InitBuffer(tensors.groupBuffer, AlignedBytes(tiling.rowBatch, sizeof(int32_t)));
    tensors.narrow = tensors.narrowBuffer.template Get<T>();
    tensors.narrow2 = tensors.narrowBuffer2.template Get<T>();
    tensors.wide = tensors.wideBuffer.template Get<float>();
    tensors.wide2 = tensors.wideBuffer2.template Get<float>();
    tensors.wide3 = tensors.wideBuffer3.template Get<float>();
    tensors.groups = tensors.groupBuffer.template Get<int32_t>();
    tensors.csrSegment = 0;
    tensors.csrNext = 0;
    tensors.csrColumn = 0;
    tensors.csrStarted = false;
    if constexpr (Phase == kPhaseSum) {
        InitCountBuffers(tensors, pipe, tiling);
    }
    if constexpr (Phase == kPhaseGradElement) {
        // The elementwise gradient holds the source, the saved output and the
        // incoming gradient at the same time, so it needs a third dtype buffer
        // on top of the two the other passes use. The row budget is sized for
        // 20 bytes per element, so this pass reaches 24 in the worst case: with
        // the 80 KB budget that is at most 96 KB, well inside the UB.
        pipe.InitBuffer(tensors.narrowBuffer3, AlignedBytes(rowElements, sizeof(T)));
        tensors.narrow3 = tensors.narrowBuffer3.template Get<T>();
        InitGroupCache<Mode>(tensors, pipe, tiling);
    }
    if (tiling.localAccum != 0 && UsesUbWorkspace<Phase>()) {
        // The workspace is resident in UB for the whole pass: the reduction
        // phases accumulate into `accum`, the passes that read it use `cached`.
        pipe.InitBuffer(tensors.accumBuffer,
                        AlignedBytes(tiling.workspaceElems, sizeof(float)));
        tensors.accum = tensors.accumBuffer.template Get<float>();
        if (ReadsWorkspace<Phase>()) {
            pipe.InitBuffer(tensors.workspaceBuffer,
                            AlignedBytes(tiling.workspaceElems, sizeof(float)));
            tensors.cached = tensors.workspaceBuffer.template Get<float>();
        }
    }
}

// Widens `count` source elements into fp32. T == float still goes through a
// vector instruction so every dtype shares one downstream code path.
template <typename T>
__aicore__ inline void WidenToFp32(const LocalTensor<float>& dst, const LocalTensor<T>& src, uint32_t count)
{
    if (count == 0) {
        return;
    }
    if constexpr (std::is_same<T, float>::value) {
        Adds(dst, src, 0.0F, static_cast<int32_t>(count));
    } else {
        Cast(dst, src, RoundMode::CAST_NONE, static_cast<int32_t>(count));
    }
}

// Narrows `count` fp32 elements back to the source dtype, rounding to nearest.
template <typename T>
__aicore__ inline void NarrowFromFp32(const LocalTensor<T>& dst, const LocalTensor<float>& src,
                                     uint32_t count)
{
    if (count == 0) {
        return;
    }
    if constexpr (std::is_same<T, float>::value) {
        Adds(dst, src, 0.0F, static_cast<int32_t>(count));
    } else {
        Cast(dst, src, RoundMode::CAST_RINT, static_cast<int32_t>(count));
    }
}

// A group label is only usable when it lands inside the requested range; the
// host already rejects out-of-range values, so this is a defensive guard.
__aicore__ inline bool GroupValid(const ScatterSoftmaxTilingData& tiling, int32_t group)
{
    return group >= 0 && static_cast<uint64_t>(group) < tiling.groupCount;
}

// Workspace row of (row, group): the group labels are not sorted, so the
// reduction target is a function of the label, not of the row.
__aicore__ inline uint64_t WorkBase(const ScatterSoftmaxTilingData& tiling, uint64_t row, int32_t group)
{
    const uint64_t batch = row / tiling.dimLength;
    return (batch * tiling.groupCount + static_cast<uint64_t>(group)) * tiling.afterPadded;
}

// Output row of (row, group). The reduced output stores one row per
// (batch, group) pair, so its stride is `after` instead of `afterPadded`.
__aicore__ inline uint64_t OutBase(const ScatterSoftmaxTilingData& tiling, uint64_t row, int32_t group)
{
    const uint64_t batch = row / tiling.dimLength;
    return (batch * tiling.groupCount + static_cast<uint64_t>(group)) * tiling.after;
}

// Maps the rows of one batch to their reduction slot without dividing per row.
//
// `WorkBase` divides the row by `dimLength` for every row, and a 64-bit
// division on the scalar unit costs more than the rest of the loop body: the
// batch pass is scalar bound exactly because of it. A batch walks its rows in
// order, so the slice offset advances by one workspace block whenever the row
// crosses a slice boundary, which is one comparison per row and one division
// per batch. `rowStride` selects the workspace row (`afterPadded`) or the
// reduced output row (`after`).
class BatchRowSlices {
public:
    __aicore__ inline BatchRowSlices(const ScatterSoftmaxTilingData& tiling, uint64_t base,
                                     uint64_t rowStride)
        : offset_((base / tiling.dimLength) * tiling.groupCount * rowStride),
          next_((base / tiling.dimLength + 1) * tiling.dimLength),
          block_(tiling.groupCount * rowStride),
          length_(tiling.dimLength),
          rowStride_(rowStride)
    {
    }

    // Offset of the (batch, group) row that starts at `row`. Rows must be
    // requested in increasing order; a gap (a row that was skipped) is walked.
    __aicore__ inline uint64_t Row(uint64_t row, int32_t group)
    {
        while (row >= next_) {
            offset_ += block_;
            next_ += length_;
        }
        return offset_ + static_cast<uint64_t>(group) * rowStride_;
    }

private:
    uint64_t offset_;
    uint64_t next_;
    uint64_t block_;
    uint64_t length_;
    uint64_t rowStride_;
};

__aicore__ inline bool IsParticipating(const ScatterSoftmaxTilingData& tiling)
{
    return tiling.blockNum != 0 &&
           static_cast<uint64_t>(GetBlockIdx()) < tiling.blockNum &&
           tiling.rows != 0 && tiling.after != 0;
}

}  // namespace ScatterSoftmaxNpu

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_COMMON_H

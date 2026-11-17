/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_SOFTMAX_KERNEL_PHASE_H
#define OPS_GNN_SCATTER_SOFTMAX_KERNEL_PHASE_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"

/**
 * Per-batch bodies of the grouped-softmax device kernels.
 *
 * A core owns `rowChunk` consecutive rows, buffers `rowBatch` of them at a time
 * and walks the feature axis in 32-byte aligned tiles. Each pass contributes
 * only its own arithmetic; the transfers, the pipeline synchronisation and the
 * reduction bookkeeping are shared by all five passes.
 *
 * The group of a row comes from `index`, which is not sorted, so the reduced
 * values of every (batch, group, feature) slot go through the MTE3 atomic unit
 * - or, when the whole reduction workspace fits in UB, through a private
 * per-core accumulator flushed once at the end of the pass, with the workspace
 * read from a resident copy instead of from GM row by row.
 *
 * The compile-time `Log` switch selects the log-softmax variant of the same
 * pipeline (scatter_log_softmax): the maximum pass and the exponential sum are
 * identical, the normalize pass writes `(src - maximum) - ln(sum)` and reads
 * `src` instead of the stored exponential, and the backward reduces the plain
 * gradient sum instead of `out * grad_out` because the log-domain Jacobian
 * only needs `exp(out)` as a weight after the reduction.
 */

namespace ScatterSoftmaxNpu {

// ---------------------------------------------------------------------------
// Per-batch data movement.
// ---------------------------------------------------------------------------

// Loads the group labels of the batch. Consecutive rows map to consecutive
// columns while the batch stays inside one indexed slice, which is the common
// case; a batch that straddles a slice boundary falls back to scalar reads.
template <typename T>
__aicore__ inline void LoadGroups(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                  uint64_t base, uint32_t count, PipelineSync& sync){
    const uint64_t column = base % tiling.dimLength;
    if (column + count <= tiling.dimLength) {
        SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
        DataCopyPad(t.groups, t.indexGm[column], Bytes1D(count * sizeof(int32_t)),
                    NoPadding<int32_t>());
        SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
        return;
    }
    for (uint32_t k = 0; k < count; ++k) {
        t.groups.SetValue(k, t.indexGm.GetValue((base + k) % tiling.dimLength));
    }
}

// `count` row starts of the source-shaped tensors, one burst per row.
template <typename T>
__aicore__ inline void CopyRowsIn(LocalTensor<T> dst, GlobalTensor<T> src,
                                  const ScatterSoftmaxTilingData& tiling, uint64_t base,
                                  uint32_t count, uint64_t col, uint32_t valid)
{
    for (uint32_t k = 0; k < count; ++k) {
        DataCopyPad(dst[k * tiling.colTile], src[(base + k) * tiling.after + col],
                    Bytes1D(valid * sizeof(T)), NoPadding<T>());
    }
}

template <typename T>
__aicore__ inline void CopyRowsOut(GlobalTensor<T> dst, LocalTensor<T> src,
                                   const ScatterSoftmaxTilingData& tiling,
                                   LocalTensor<int32_t> groups, uint64_t base,
                                   uint32_t count, uint64_t col, uint32_t valid)
{
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        DataCopyPad(dst[(base + k) * tiling.after + col], src[k * tiling.colTile],
                    Bytes1D(valid * sizeof(T)));
    }
}

// Reads the fp32 workspace row each batched row reduces into.
__aicore__ inline void CopyWorkspaceIn(LocalTensor<float> dst, GlobalTensor<float> src,
                                       const ScatterSoftmaxTilingData& tiling,
                                       LocalTensor<int32_t> groups, uint64_t base,
                                       uint32_t count, uint64_t col, uint32_t width)
{
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        DataCopy(dst[k * tiling.colTile], src[WorkBase(tiling, base + k, group) + col],
                 static_cast<int32_t>(width));
    }
}

// Same row, from the UB-resident copy. A per-row GM read of the workspace is
// the one thing a small group count cannot amortise: every row of a million-row
// single-group shape hits the same 512-byte line, and those reads do not
// overlap. The copy costs one vector instruction and no GM round trip.
__aicore__ inline void CopyWorkspaceCached(LocalTensor<float> dst, LocalTensor<float> cached,
                                           const ScatterSoftmaxTilingData& tiling,
                                           LocalTensor<int32_t> groups, uint64_t base,
                                           uint32_t count, uint64_t col, uint32_t width)
{
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        Adds(dst[k * tiling.colTile], cached[WorkBase(tiling, base + k, group) + col], 0.0F,
             static_cast<int32_t>(width));
    }
}

// Brings the workspace of an earlier pass into UB once, before the row loop.
template <ScatterSoftmaxPhase Phase, bool Log, typename T>
__aicore__ inline void LoadWorkspace(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                     PipelineSync& sync)
{
    if (tiling.localAccum == 0 || !ReadsWorkspace<Phase>()) {
        return;
    }
    const int32_t count = static_cast<int32_t>(tiling.workspaceElems);
    if constexpr (Phase == kPhaseExp) {
        DataCopy(t.cached, t.maximumGm, count);
    } else if constexpr (Log && Phase == kPhaseNormalize) {
        // The log-domain normalization subtracts the maximum and multiplies by
        // nothing: it needs the maximum and the sum at the same time, so the
        // cached copy holds the maximum and the (otherwise idle) accumulator
        // buffer holds the sum. Both copies stay inside the two-workspace
        // reservation the exp pass already asks for.
        DataCopy(t.cached, t.maximumGm, count);
        DataCopy(t.accum, t.sumGm, count);
    } else {
        DataCopy(t.cached, t.sumGm, count);
    }
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
}

// ---------------------------------------------------------------------------
// Reduction bookkeeping: private accumulator, or one atomic per row.
// ---------------------------------------------------------------------------

// Folds the batch into the private fp32 accumulator. `accum` indexes the
// workspace layout directly, so the labels never have to be sorted.
template <bool IsMaximum, typename T>
__aicore__ inline void AccumulateRows(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                      LocalTensor<float> values, uint64_t base,
                                      uint32_t count, uint64_t col, uint32_t width)
{
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = t.groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        const uint64_t offset = WorkBase(tiling, base + k, group) + col;
        if constexpr (IsMaximum) {
            Max(t.accum[offset], t.accum[offset], values[k * tiling.colTile],
                static_cast<int32_t>(width));
        } else {
            Add(t.accum[offset], t.accum[offset], values[k * tiling.colTile],
                static_cast<int32_t>(width));
        }
    }
}

// One atomic write per row of the workspace, issued once per core.
template <typename T>
__aicore__ inline void StoreAccumulator(PhaseTensors<T>& t, GlobalTensor<float> dst,
                                        const ScatterSoftmaxTilingData& tiling)
{
    const uint64_t workspaceRows = tiling.workspaceElems / tiling.afterPadded;
    for (uint64_t row = 0; row < workspaceRows; ++row) {
        const uint64_t offset = row * tiling.afterPadded;
        DataCopy(dst[offset], t.accum[offset], static_cast<int32_t>(tiling.afterPadded));
    }
}

template <ScatterSoftmaxPhase Phase, typename T>
__aicore__ inline void InitAccumulator(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling)
{
    if constexpr (IsReductionPass<Phase>()) {
        if (tiling.localAccum == 0) {
            return;
        }
        const float neutral = (Phase == kPhaseMaximum) ? SCATTER_SOFTMAX_NEUTRAL_MAXIMUM : 0.0F;
        Duplicate(t.accum, neutral, static_cast<int32_t>(tiling.workspaceElems));
    }
}

template <ScatterSoftmaxPhase Phase, typename T>
__aicore__ inline void FlushAccumulator(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                        PipelineSync& sync)
{
    if constexpr (IsReductionPass<Phase>()) {
        if (tiling.localAccum == 0) {
            return;
        }
        SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
        if constexpr (Phase == kPhaseMaximum) {
            SetAtomicMax<float>();
            StoreAccumulator(t, t.maximumGm, tiling);
        } else {
            SetAtomicAdd<float>();
            StoreAccumulator(t, t.sumGm, tiling);
        }
        DisableDmaAtomic();
        SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
    }
}

// ---------------------------------------------------------------------------
// Per-batch bodies, one per pass.
// ---------------------------------------------------------------------------

// maximum[b][g][a] = max over the group. Padded lanes keep the neutral value.
template <typename T>
__aicore__ inline void MaximumBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                    PipelineSync& sync, uint64_t base, uint32_t count,
                                    uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    CopyRowsIn(t.narrow, t.srcGm, tiling, base, count, col, valid);
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
    for (uint32_t k = 0; k < count; ++k) {
        const uint64_t offset = k * tiling.colTile;
        // The neutral fill only pads the lanes the row does not cover; a tile
        // that is full to `valid == width` has none and skips it.
        if (valid < width) {
            Duplicate(t.wide[offset], SCATTER_SOFTMAX_NEUTRAL_MAXIMUM,
                      static_cast<int32_t>(tiling.colTile));
        }
        WidenToFp32(t.wide[offset], t.narrow[offset], valid);
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    if (tiling.localAccum != 0) {
        AccumulateRows<true>(t, tiling, t.wide, base, count, col, width);
    } else {
        SetAtomicMax<float>();
        for (uint32_t k = 0; k < count; ++k) {
            const int32_t group = t.groups.GetValue(k);
            if (!GroupValid(tiling, group)) {
                continue;
            }
            DataCopy(t.maximumGm[WorkBase(tiling, base + k, group) + col],
                     t.wide[k * tiling.colTile], static_cast<int32_t>(width));
        }
        DisableDmaAtomic();
    }
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

// sum[b][g][a] += exp(src - maximum). In log mode the exponential itself is
// never stored: the normalize pass recomputes the shifted source, so only the
// sum crosses the pass boundary.
template <bool Log, typename T>
__aicore__ inline void ExpBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                PipelineSync& sync, uint64_t base, uint32_t count,
                                uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    CopyRowsIn(t.narrow, t.srcGm, tiling, base, count, col, valid);
    if (tiling.localAccum != 0) {
        CopyWorkspaceCached(t.wide2, t.cached, tiling, t.groups, base, count, col, width);
    } else {
        CopyWorkspaceIn(t.wide2, t.maximumGm, tiling, t.groups, base, count, col, width);
    }
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
    // Padded lanes would evaluate exp(0 - neutral) and overflow; they are
    // neutralized before the subtraction instead of after the exponential.
    for (uint32_t k = 0; k < count; ++k) {
        const uint64_t offset = k * tiling.colTile;
        // Padded lanes stay zero so they add nothing to the sum; a full tile
        // has none and skips the whole-tile fill.
        if (valid < width) {
            Duplicate(t.wide[offset], 0.0F, static_cast<int32_t>(tiling.colTile));
        }
        WidenToFp32(t.wide[offset], t.narrow[offset], valid);
        Sub(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
        Exp(t.wide[offset], t.wide[offset], static_cast<int32_t>(valid));
        if constexpr (!Log) {
            NarrowFromFp32(t.narrow2[offset], t.wide[offset], valid);
        }
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    if constexpr (!Log) {
        CopyRowsOut(t.outGm, t.narrow2, tiling, t.groups, base, count, col, valid);
    }
    if (tiling.localAccum != 0) {
        AccumulateRows<false>(t, tiling, t.wide, base, count, col, width);
    } else {
        SetAtomicAdd<float>();
        for (uint32_t k = 0; k < count; ++k) {
            const int32_t group = t.groups.GetValue(k);
            if (!GroupValid(tiling, group)) {
                continue;
            }
            DataCopy(t.sumGm[WorkBase(tiling, base + k, group) + col],
                     t.wide[k * tiling.colTile], static_cast<int32_t>(width));
        }
        DisableDmaAtomic();
    }
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

// out = exp(src - maximum) / sum in the softmax mode, and
// out = (src - maximum) - ln(sum) in the log mode. A zero sum propagates as NaN
// from the division itself; in log mode it turns the output into NaN through
// `-inf - (-inf)`, which is what `torch.log_softmax` reports for a group of
// `-inf`.
template <bool Log, typename T>
__aicore__ inline void NormalizeBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                      PipelineSync& sync, uint64_t base, uint32_t count,
                                      uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    if constexpr (Log) {
        // The stored exponential is not needed: the log output is recomputed
        // from the source, which also keeps a group with an underflowed
        // exponential sum from collapsing to `-inf` (see the kernel header).
        CopyRowsIn(t.narrow, t.srcGm, tiling, base, count, col, valid);
        if (tiling.localAccum != 0) {
            CopyWorkspaceCached(t.wide2, t.cached, tiling, t.groups, base, count, col, width);
            CopyWorkspaceCached(t.wide3, t.accum, tiling, t.groups, base, count, col, width);
        } else {
            CopyWorkspaceIn(t.wide2, t.maximumGm, tiling, t.groups, base, count, col, width);
            CopyWorkspaceIn(t.wide3, t.sumGm, tiling, t.groups, base, count, col, width);
        }
    } else {
        CopyRowsIn(t.narrow, t.outGm, tiling, base, count, col, valid);
        if (tiling.localAccum != 0) {
            CopyWorkspaceCached(t.wide2, t.cached, tiling, t.groups, base, count, col, width);
        } else {
            CopyWorkspaceIn(t.wide2, t.sumGm, tiling, t.groups, base, count, col, width);
        }
    }
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = t.groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        const uint64_t offset = k * tiling.colTile;
        WidenToFp32(t.wide[offset], t.narrow[offset], valid);
        if constexpr (Log) {
            Sub(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
            Ln(t.wide3[offset], t.wide3[offset], static_cast<int32_t>(valid));
            Sub(t.wide[offset], t.wide[offset], t.wide3[offset], static_cast<int32_t>(valid));
        } else {
            Div(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
        }
        NarrowFromFp32(t.narrow[offset], t.wide[offset], valid);
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    CopyRowsOut(t.outGm, t.narrow, tiling, t.groups, base, count, col, valid);
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

// sum[b][g][a] += out * grad_out, or += grad_out in log mode: the log-domain
// Jacobian weights the reduced gradient by the probability afterwards, so the
// reduction itself is the plain group sum of `grad_out`.
template <bool Log, typename T>
__aicore__ inline void GradSumBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                    PipelineSync& sync, uint64_t base, uint32_t count,
                                    uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    if constexpr (!Log) {
        CopyRowsIn(t.narrow, t.outGm, tiling, base, count, col, valid);
    }
    CopyRowsIn(t.narrow2, t.gradGm, tiling, base, count, col, valid);
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
    for (uint32_t k = 0; k < count; ++k) {
        const uint64_t offset = k * tiling.colTile;
        // Padded lanes contribute zero to the sum; a full tile skips the fill.
        if (valid < width) {
            Duplicate(t.wide3[offset], 0.0F, static_cast<int32_t>(tiling.colTile));
        }
        if constexpr (Log) {
            WidenToFp32(t.wide3[offset], t.narrow2[offset], valid);
        } else {
            WidenToFp32(t.wide3[offset], t.narrow[offset], valid);
            WidenToFp32(t.wide[offset], t.narrow2[offset], valid);
            Mul(t.wide3[offset], t.wide3[offset], t.wide[offset], static_cast<int32_t>(valid));
        }
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    if (tiling.localAccum != 0) {
        AccumulateRows<false>(t, tiling, t.wide3, base, count, col, width);
    } else {
        SetAtomicAdd<float>();
        for (uint32_t k = 0; k < count; ++k) {
            const int32_t group = t.groups.GetValue(k);
            if (!GroupValid(tiling, group)) {
                continue;
            }
            DataCopy(t.sumGm[WorkBase(tiling, base + k, group) + col],
                     t.wide3[k * tiling.colTile], static_cast<int32_t>(width));
        }
        DisableDmaAtomic();
    }
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

// grad_src = out * (grad_out - sum[b][g][a]) in the softmax mode, and
// grad_src = grad_out - exp(out) * sum[b][g][a] in the log mode.
template <bool Log, typename T>
__aicore__ inline void GradBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                 PipelineSync& sync, uint64_t base, uint32_t count,
                                 uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    CopyRowsIn(t.narrow, t.outGm, tiling, base, count, col, valid);
    CopyRowsIn(t.narrow2, t.gradGm, tiling, base, count, col, valid);
    if (tiling.localAccum != 0) {
        CopyWorkspaceCached(t.wide3, t.cached, tiling, t.groups, base, count, col, width);
    } else {
        CopyWorkspaceIn(t.wide3, t.sumGm, tiling, t.groups, base, count, col, width);
    }
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = t.groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        const uint64_t offset = k * tiling.colTile;
        WidenToFp32(t.wide[offset], t.narrow[offset], valid);
        WidenToFp32(t.wide2[offset], t.narrow2[offset], valid);
        if constexpr (Log) {
            // dL/dx = grad_out - exp(out) * sum(grad_out): the group reduces the
            // plain gradient sum, and only then is it weighted by the
            // probability that `exp(out)` recovers.
            Exp(t.wide[offset], t.wide[offset], static_cast<int32_t>(valid));
            Mul(t.wide[offset], t.wide[offset], t.wide3[offset], static_cast<int32_t>(valid));
            Sub(t.wide2[offset], t.wide2[offset], t.wide[offset], static_cast<int32_t>(valid));
            NarrowFromFp32(t.narrow2[offset], t.wide2[offset], valid);
        } else {
            Sub(t.wide2[offset], t.wide2[offset], t.wide3[offset], static_cast<int32_t>(valid));
            Mul(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
            NarrowFromFp32(t.narrow2[offset], t.wide[offset], valid);
        }
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    CopyRowsOut(t.gradSrcGm, t.narrow2, tiling, t.groups, base, count, col, valid);
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

template <ScatterSoftmaxPhase Phase, bool Log, typename T>
__aicore__ inline void RunBatch(PhaseTensors<T>& tensors, const ScatterSoftmaxTilingData& tiling,
                                PipelineSync& sync, uint64_t base, uint32_t count,
                                uint64_t col, uint32_t valid, uint32_t width)
{
    if constexpr (Phase == kPhaseMaximum) {
        MaximumBatch(tensors, tiling, sync, base, count, col, valid, width);
    } else if constexpr (Phase == kPhaseExp) {
        ExpBatch<Log>(tensors, tiling, sync, base, count, col, valid, width);
    } else if constexpr (Phase == kPhaseNormalize) {
        NormalizeBatch<Log>(tensors, tiling, sync, base, count, col, valid, width);
    } else if constexpr (Phase == kPhaseGradSum) {
        GradSumBatch<Log>(tensors, tiling, sync, base, count, col, valid, width);
    } else {
        GradBatch<Log>(tensors, tiling, sync, base, count, col, valid, width);
    }
}

// Shared batch/tile driver: a core owns `rowChunk` consecutive rows, buffers
// `rowBatch` of them at a time and walks every row along the feature axis in
// 32-byte aligned tiles.
template <ScatterSoftmaxPhase Phase, bool Log, typename T>
__aicore__ inline void RunRows(PhaseTensors<T>& tensors, const ScatterSoftmaxTilingData& tiling,
                               PipelineSync& sync)
{
    const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * tiling.rowChunk;
    uint64_t stop = start + tiling.rowChunk;
    if (stop > tiling.rows) {
        stop = tiling.rows;
    }
    const uint32_t batch = (tiling.rowBatch == 0) ? 1 : tiling.rowBatch;
    for (uint64_t base = start; base < stop; base += batch) {
        uint32_t count = static_cast<uint32_t>(stop - base);
        if (count > batch) {
            count = batch;
        }
        LoadGroups(tensors, tiling, base, count, sync);
        for (uint64_t col = 0; col < tiling.after; col += tiling.colTile) {
            RunBatch<Phase, Log>(tensors, tiling, sync, base, count, col,
                                 TileLength(tiling.after, col, tiling.colTile),
                                 TileLength(tiling.afterPadded, col, tiling.colTile));
        }
    }
}

}  // namespace ScatterSoftmaxNpu

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_PHASE_H

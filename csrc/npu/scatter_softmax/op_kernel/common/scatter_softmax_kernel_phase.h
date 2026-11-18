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
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_io.h"

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
 * The compile-time mode selects the arithmetic of the same pipeline:
 *
 *   - `kModeSoftmax` / `kModeLogSoftmax` (scatter_softmax, scatter_log_softmax):
 *     the maximum pass and the exponential sum are shared, the normalize pass
 *     writes the `src`-shaped result, and the backward reduces the group
 *     gradient before weighting it;
 *   - `kModeLogSumExp` (scatter_logsumexp): the same maximum and exponential
 *     passes, except that a `+inf` member contributes 1 to the sum, followed by
 *     a finalize pass that writes `maximum + ln(sum)` once per group and an
 *     elementwise backward `grad_out[group] * exp(src - out[group])`.
 */

namespace ScatterSoftmaxNpu {

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

// sum[b][g][a] += exp(src - maximum). In the softmax mode the exponential is
// kept for the normalize pass; the log-domain operators never store it, so only
// the sum crosses the pass boundary.
//
// The log-sum-exp mode clamps the subtracted maximum to the largest finite
// float. A group that holds `+inf` keeps `+inf` as its maximum, so `+inf - +inf`
// would be NaN and poison the sum; with the clamped pivot the infinite member
// contributes `exp(+inf) = +inf`, so the sum is `+inf` and the group result
// stays `+inf`, while its finite members contribute `exp(-inf) = 0`. Finite
// groups are unaffected: their maximum is already below the clamp.
template <ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void ExpBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                PipelineSync& sync, uint64_t base, uint32_t count,
                                uint64_t col, uint32_t valid, uint32_t width)
{
    constexpr bool kStore = (Mode == kModeSoftmax);
    constexpr float kFiniteMax = 3.4028234663852886e38F;
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
        if constexpr (Mode == kModeLogSumExp) {
            Mins(t.wide2[offset], t.wide2[offset], kFiniteMax,
                 static_cast<int32_t>(valid));
        }
        Sub(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
        Exp(t.wide[offset], t.wide[offset], static_cast<int32_t>(valid));
        if constexpr (kStore) {
            NarrowFromFp32(t.narrow2[offset], t.wide[offset], valid);
        }
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    if constexpr (kStore) {
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
template <ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void NormalizeBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                      PipelineSync& sync, uint64_t base, uint32_t count,
                                      uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    if constexpr (Mode != kModeSoftmax) {
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
        if constexpr (Mode != kModeSoftmax) {
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
template <ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void GradSumBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                    PipelineSync& sync, uint64_t base, uint32_t count,
                                    uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    if constexpr (Mode == kModeSoftmax) {
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
        if constexpr (Mode != kModeSoftmax) {
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
template <ScatterSoftmaxMode Mode, typename T>
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
        if constexpr (Mode != kModeSoftmax) {
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

// out[b][g][a] = maximum[b][g][a] + ln(sum[b][g][a]).
//
// The reduced output rows are exactly the workspace rows, so the pass walks
// them in order and needs no group labels. An empty group keeps the neutral
// maximum and a zero sum, and `neutral + ln(0)` is `-inf`, which is what
// `torch_scatter` and `torch.logsumexp` report for a group with no member.
template <typename T>
__aicore__ inline void FinalizeBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                     PipelineSync& sync, uint64_t base, uint32_t count,
                                     uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    for (uint32_t k = 0; k < count; ++k) {
        const uint64_t offset = (base + k) * tiling.afterPadded + col;
        DataCopy(t.wide[k * tiling.colTile], t.maximumGm[offset],
                 static_cast<int32_t>(width));
        DataCopy(t.wide2[k * tiling.colTile], t.sumGm[offset],
                 static_cast<int32_t>(width));
    }
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
    for (uint32_t k = 0; k < count; ++k) {
        const uint64_t offset = k * tiling.colTile;
        Ln(t.wide2[offset], t.wide2[offset], static_cast<int32_t>(valid));
        Add(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
        NarrowFromFp32(t.narrow[offset], t.wide[offset], valid);
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    for (uint32_t k = 0; k < count; ++k) {
        DataCopyPad(t.outGm[(base + k) * tiling.after + col],
                    t.narrow[k * tiling.colTile], Bytes1D(valid * sizeof(T)));
    }
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

// grad_src = grad_out[group] * exp(src - out[group]).
//
// One pass, no workspace: the saved log-sum-exp output recovers
// `exp(src - out)`, and the group gradient is read at the row of the label. A
// `-inf` group gives `-inf - (-inf) = NaN`, a `+inf` member gives
// `+inf - (+inf) = NaN` while its finite siblings give `exp(-inf) = 0`, exactly
// like the gradient of `torch.logsumexp`.
template <typename T>
__aicore__ inline void GradElementBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                        PipelineSync& sync, uint64_t base, uint32_t count,
                                        uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    CopyRowsIn(t.narrow, t.srcGm, tiling, base, count, col, valid);
    if (!GroupsFitUb(tiling)) {
        CopyGroupRowsIn(t.narrow2, t.outGm, tiling, t.groups, base, count, col, valid);
        CopyGroupRowsIn(t.narrow3, t.gradGm, tiling, t.groups, base, count, col, valid);
    }
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = t.groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        const uint64_t offset = k * tiling.colTile;
        WidenToFp32(t.wide[offset], t.narrow[offset], valid);
        if (GroupsFitUb(tiling)) {
            // Both group rows come from the resident copy; the same GM address
            // read once per source row is what the cache removes.
            const uint64_t slot = WorkBase(tiling, base + k, group) + col;
            Adds(t.wide2[offset], t.cachedOut[slot], 0.0F, static_cast<int32_t>(valid));
            Sub(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
            Exp(t.wide[offset], t.wide[offset], static_cast<int32_t>(valid));
            Adds(t.wide3[offset], t.cachedGrad[slot], 0.0F, static_cast<int32_t>(valid));
            Mul(t.wide[offset], t.wide[offset], t.wide3[offset], static_cast<int32_t>(valid));
        } else {
            WidenToFp32(t.wide2[offset], t.narrow2[offset], valid);
            Sub(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
            Exp(t.wide[offset], t.wide[offset], static_cast<int32_t>(valid));
            WidenToFp32(t.wide2[offset], t.narrow3[offset], valid);
            Mul(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
        }
        NarrowFromFp32(t.narrow[offset], t.wide[offset], valid);
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    CopyRowsOut(t.gradSrcGm, t.narrow, tiling, t.groups, base, count, col, valid);
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void RunBatch(PhaseTensors<T>& tensors, const ScatterSoftmaxTilingData& tiling,
                                PipelineSync& sync, uint64_t base, uint32_t count,
                                uint64_t col, uint32_t valid, uint32_t width)
{
    if constexpr (Phase == kPhaseMaximum) {
        MaximumBatch(tensors, tiling, sync, base, count, col, valid, width);
    } else if constexpr (Phase == kPhaseExp) {
        ExpBatch<Mode>(tensors, tiling, sync, base, count, col, valid, width);
    } else if constexpr (Phase == kPhaseNormalize) {
        NormalizeBatch<Mode>(tensors, tiling, sync, base, count, col, valid, width);
    } else if constexpr (Phase == kPhaseGradSum) {
        GradSumBatch<Mode>(tensors, tiling, sync, base, count, col, valid, width);
    } else if constexpr (Phase == kPhaseGrad) {
        GradBatch<Mode>(tensors, tiling, sync, base, count, col, valid, width);
    } else if constexpr (Phase == kPhaseFinalize) {
        FinalizeBatch(tensors, tiling, sync, base, count, col, valid, width);
    } else {
        GradElementBatch(tensors, tiling, sync, base, count, col, valid, width);
    }
}

// Shared batch/tile driver: a core owns `rowChunk` consecutive rows, buffers
// `rowBatch` of them at a time and walks every row along the feature axis in
// 32-byte aligned tiles.
template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode, typename T>
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
        if constexpr (UsesGroups<Phase>()) {
            LoadGroups(tensors, tiling, base, count, sync);
        }
        for (uint64_t col = 0; col < tiling.after; col += tiling.colTile) {
            RunBatch<Phase, Mode>(tensors, tiling, sync, base, count, col,
                                 TileLength(tiling.after, col, tiling.colTile),
                                 TileLength(tiling.afterPadded, col, tiling.colTile));
        }
    }
}

}  // namespace ScatterSoftmaxNpu

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_PHASE_H

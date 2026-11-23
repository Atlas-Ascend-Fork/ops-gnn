/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_SOFTMAX_KERNEL_REDUCE_H
#define OPS_GNN_SCATTER_SOFTMAX_KERNEL_REDUCE_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"

/**
 * Reduction bookkeeping of the grouped kernels: how one tile of a reduction
 * pass reaches its (batch, group, feature) slot, and how the private
 * accumulator of a core is opened and flushed.
 *
 * The group of a row comes from `index`, which is not sorted, so the slot of a
 * row is a function of its label rather than of its position. A pass writes its
 * tile either through the MTE3 atomic unit - one atomic per row - or into the
 * private accumulator of the core when the whole workspace is resident in UB,
 * which is flushed once at the end of the pass. Both writers live here so every
 * reduction pass, including the ones of the standard deviation, is written once.
 */

namespace ScatterSoftmaxNpu {

// Folds the batch into the private fp32 accumulator. `accum` indexes the
// workspace layout directly, so the labels never have to be sorted.
template <bool IsMaximum, typename T>
__aicore__ inline void AccumulateRows(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                      LocalTensor<float> values, uint64_t base,
                                      uint32_t count, uint64_t col, uint32_t width)
{
    BatchRowSlices slices(tiling, base, tiling.afterPadded);
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = t.groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        const uint64_t offset = slices.Row(base + k, group) + col;
        if constexpr (IsMaximum) {
            Max(t.accum[offset], t.accum[offset], values[k * tiling.colTile],
                static_cast<int32_t>(width));
        } else {
            Add(t.accum[offset], t.accum[offset], values[k * tiling.colTile],
                static_cast<int32_t>(width));
        }
    }
}

// Writes one tile of a reduced workspace: into the private accumulator of the
// core, or as one atomic per row. `IsMaximum` picks the atomic operation, so a
// maximum pass and an additive pass share the whole body.
template <bool IsMaximum, typename T>
__aicore__ inline void StoreReduced(PhaseTensors<T>& t, GlobalTensor<float> dst,
                                    const ScatterSoftmaxTilingData& tiling, uint64_t base,
                                    uint32_t count, uint64_t col, uint32_t width,
                                    LocalTensor<float> values)
{
    if (tiling.localAccum != 0) {
        AccumulateRows<IsMaximum>(t, tiling, values, base, count, col, width);
        return;
    }
    if constexpr (IsMaximum) {
        SetAtomicMax<float>();
    } else {
        SetAtomicAdd<float>();
    }
    BatchRowSlices slices(tiling, base, tiling.afterPadded);
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = t.groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        DataCopy(dst[slices.Row(base + k, group) + col], values[k * tiling.colTile],
                 static_cast<int32_t>(width));
    }
    DisableDmaAtomic();
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
    if constexpr (Phase == kPhaseSum) {
        // The sum pass also counts the members of every group; the counts a
        // core keeps in UB start empty, and the per-row fallback builds its
        // one-hot atomic blocks once, before the first row of the pass.
        if (CountsFitUb(tiling)) {
            Duplicate(t.counts, 0.0F, static_cast<int32_t>(tiling.countElems));
        } else if (tiling.countElems != 0) {
            constexpr int32_t lanes = static_cast<int32_t>(SCATTER_SOFTMAX_COUNT_ALIGN);
            Duplicate(t.onehot, 0.0F, lanes * lanes);
            // The lanes are filled by the scalar unit, so a full barrier orders
            // the vector fill before them and the blocks before the first atomic
            // write of the pass. Both barriers run once per core.
            PipeBarrier<PIPE_ALL>();
            for (int32_t lane = 0; lane < lanes; ++lane) {
                t.onehot.SetValue(lane * lanes + lane, 1.0F);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
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
        } else if constexpr (Phase == kPhaseDeviation) {
            SetAtomicAdd<float>();
            StoreAccumulator(t, t.deviationGm, tiling);
        } else {
            SetAtomicAdd<float>();
            StoreAccumulator(t, t.sumGm, tiling);
        }
        DisableDmaAtomic();
        SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
    }
}

}  // namespace ScatterSoftmaxNpu

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_REDUCE_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_SOFTMAX_KERNEL_IO_H
#define OPS_GNN_SCATTER_SOFTMAX_KERNEL_IO_H

#include <cstdint>

#include "kernel_operator.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"

/**
 * GM/UB data movement of the grouped-softmax device kernels.
 *
 * A core owns `rowChunk` consecutive rows, buffers `rowBatch` of them at a time
 * and walks the feature axis in 32-byte aligned tiles. Every pass moves the same
 * shapes through these helpers; the arithmetic stays in
 * scatter_softmax_kernel_phase.h.
 *
 * The group of a row comes from `index`, which is not sorted, so a row of the
 * reduced output (or of its gradient) is fetched at the row its label selects.
 * When the whole reduction workspace fits in UB it is kept resident instead of
 * being read from GM row by row.
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

// `count` rows of a group-shaped tensor (the reduced output or its gradient):
// one burst per row, at the row the label selects.
template <typename T>
__aicore__ inline void CopyGroupRowsIn(LocalTensor<T> dst, GlobalTensor<T> src,
                                       const ScatterSoftmaxTilingData& tiling,
                                       LocalTensor<int32_t> groups, uint64_t base,
                                       uint32_t count, uint64_t col, uint32_t valid)
{
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        DataCopyPad(dst[k * tiling.colTile], src[OutBase(tiling, base + k, group) + col],
                    Bytes1D(valid * sizeof(T)), NoPadding<T>());
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
template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void LoadWorkspace(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                     PipelineSync& sync)
{
    if (tiling.localAccum == 0 || !ReadsWorkspace<Phase>() ||
        !UsesUbWorkspace<Phase>()) {
        return;
    }
    const int32_t count = static_cast<int32_t>(tiling.workspaceElems);
    if constexpr (Phase == kPhaseExp) {
        DataCopy(t.cached, t.maximumGm, count);
    } else if constexpr (Mode != kModeSoftmax && Phase == kPhaseNormalize) {
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

// Brings the saved output and its gradient into UB once, before the row loop.
// Only the elementwise backward reads them, and only when every row fits a
// single tile: with a small group count each source row would otherwise read the
// same line of both tensors from GM, and those reads do not overlap.
template <typename T>
__aicore__ inline void LoadGroupCaches(PhaseTensors<T>& t,
                                       const ScatterSoftmaxTilingData& tiling,
                                       PipelineSync& sync)
{
    if (!GroupsFitUb(tiling)) {
        return;
    }
    const uint32_t width = static_cast<uint32_t>(tiling.after);
    for (uint64_t row = 0; row < tiling.outRows; ++row) {
        const uint64_t source = row * tiling.after;
        const uint64_t target = row * tiling.afterPadded;
        DataCopyPad(t.narrow, t.outGm[source], Bytes1D(width * sizeof(T)),
                    NoPadding<T>());
        DataCopyPad(t.narrow2, t.gradGm[source], Bytes1D(width * sizeof(T)),
                    NoPadding<T>());
        SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
        WidenToFp32(t.cachedOut[target], t.narrow, width);
        WidenToFp32(t.cachedGrad[target], t.narrow2, width);
        SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    }
}

}  // namespace ScatterSoftmaxNpu

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_IO_H

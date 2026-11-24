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

// True when the rows of a batch form one contiguous span in both the global
// tensor and the row buffers, which needs a tile that covers a whole row and a
// workspace row that is not padded. The rows of a source-shaped tensor and of
// the reduced output are then consecutive along `dim`, so the whole batch
// travels as a single burst instead of one burst per row: a narrow row copies
// 128 bytes, which is far below what one MTE2/MTE3 issue is worth.
__aicore__ inline bool BatchIsContiguous(const ScatterSoftmaxTilingData& tiling, uint64_t col,
                                         uint32_t valid)
{
    return col == 0 && valid == tiling.after && tiling.colTile == tiling.after &&
           tiling.afterPadded == tiling.after;
}

// True when every label of the batch selects a group, which lets a whole-batch
// store run without the per-row guard the row-wise path applies.
__aicore__ inline bool BatchGroupsValid(const ScatterSoftmaxTilingData& tiling,
                                        LocalTensor<int32_t> groups, uint32_t count)
{
    for (uint32_t k = 0; k < count; ++k) {
        if (!GroupValid(tiling, groups.GetValue(k))) {
            return false;
        }
    }
    return true;
}

// Loads the group labels of the batch. Consecutive rows map to consecutive
// columns while the batch stays inside one indexed slice, which is the common
// case; a batch that straddles a slice boundary falls back to scalar reads.
//
// In the edge-softmax CSR mode the label of a row is the segment that contains
// it. The batch walks the rows in order, so the segment cursor only moves
// forwards: a segment boundary costs one scalar pointer read, not one per row.
// The first batch of a core is located with a binary search over `indptr`
// instead of a linear walk, so a core that starts deep inside the segment range
// pays O(log segments) reads.
template <ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void LoadGroups(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                  uint64_t base, uint32_t count, PipelineSync& sync){
    if constexpr (Mode == kModeEdgeCsr) {
        const uint64_t column = base % tiling.dimLength;
        if (column == 0) {
            t.csrSegment = 0;
            t.csrNext = static_cast<uint64_t>(t.indexGm.GetValue(1));
            t.csrStarted = true;
        } else if (!t.csrStarted || column < t.csrColumn) {
            // First batch of this core, or a new batch slice: locate the segment
            // whose start is the largest one not past `column`.
            uint64_t low = 0;
            uint64_t high = tiling.groupCount - 1;
            while (low < high) {
                const uint64_t middle = (low + high + 1) / 2;
                if (static_cast<uint64_t>(t.indexGm.GetValue(middle)) <= column) {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            t.csrSegment = low;
            t.csrNext = static_cast<uint64_t>(t.indexGm.GetValue(low + 1));
            t.csrStarted = true;
        }
        for (uint32_t k = 0; k < count; ++k) {
            const uint64_t row = column + k;
            // Empty segments share their start with the next one; the cursor
            // steps over them until it reaches the segment that owns `row`.
            while (row >= t.csrNext && t.csrSegment + 1 < tiling.groupCount) {
                ++t.csrSegment;
                t.csrNext = static_cast<uint64_t>(t.indexGm.GetValue(t.csrSegment + 1));
            }
            t.groups.SetValue(k, static_cast<int32_t>(t.csrSegment));
        }
        t.csrColumn = column + count;
        return;
    }
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
    BatchRowSlices slices(tiling, base, tiling.after);
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        DataCopyPad(dst[k * tiling.colTile], src[slices.Row(base + k, group) + col],
                    Bytes1D(valid * sizeof(T)), NoPadding<T>());
    }
}

// `count` row starts of the source-shaped tensors: one burst per row, or one
// burst for the whole batch when its rows are contiguous (see
// BatchIsContiguous).
template <typename T>
__aicore__ inline void CopyRowsIn(LocalTensor<T> dst, GlobalTensor<T> src,
                                  const ScatterSoftmaxTilingData& tiling, uint64_t base,
                                  uint32_t count, uint64_t col, uint32_t valid)
{
    if (BatchIsContiguous(tiling, col, valid)) {
        DataCopyPad(dst, src[base * tiling.after],
                    Bytes1D(count * static_cast<uint32_t>(tiling.after) * sizeof(T)),
                    NoPadding<T>());
        return;
    }
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
    if (BatchIsContiguous(tiling, col, valid) &&
        BatchGroupsValid(tiling, groups, count)) {
        DataCopyPad(dst[base * tiling.after], src,
                    Bytes1D(count * static_cast<uint32_t>(tiling.after) * sizeof(T)));
        return;
    }
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        DataCopyPad(dst[(base + k) * tiling.after + col], src[k * tiling.colTile],
                    Bytes1D(valid * sizeof(T)));
    }
}

// Walks the reduction-workspace rows of one batch: it skips the rows whose
// label selects no group and yields both the row index and the workspace offset
// of every other row. A batch divides once and then compares per row, which is
// what a 64-bit division per row would otherwise dominate.
class BatchWorkspaceRows {
public:
    __aicore__ inline BatchWorkspaceRows(const ScatterSoftmaxTilingData& tiling,
                                         LocalTensor<int32_t> groups, uint64_t base,
                                         uint64_t col, uint32_t count)
        : slices_(tiling, base, tiling.afterPadded), tiling_(tiling), groups_(groups),
          base_(base), col_(col), count_(count)
    {
    }

    // Yields the next row whose label selects a group, or false at the end.
    __aicore__ inline bool Next(uint32_t& row, uint64_t& offset)
    {
        while (next_ < count_) {
            const int32_t group = groups_.GetValue(next_);
            if (!GroupValid(tiling_, group)) {
                ++next_;
                continue;
            }
            row = next_++;
            offset = slices_.Row(base_ + row, group) + col_;
            return true;
        }
        return false;
    }

private:
    BatchRowSlices slices_;
    const ScatterSoftmaxTilingData& tiling_;
    LocalTensor<int32_t> groups_;
    uint64_t base_;
    uint64_t col_;
    uint32_t count_;
    uint32_t next_ = 0;
};

// Reads the fp32 workspace row each batched row reduces into.
__aicore__ inline void CopyWorkspaceIn(LocalTensor<float> dst, GlobalTensor<float> src,
                                       const ScatterSoftmaxTilingData& tiling,
                                       LocalTensor<int32_t> groups, uint64_t base,
                                       uint32_t count, uint64_t col, uint32_t width)
{
    BatchWorkspaceRows rows(tiling, groups, base, col, count);
    uint32_t k = 0;
    uint64_t offset = 0;
    while (rows.Next(k, offset)) {
        DataCopy(dst[k * tiling.colTile], src[offset], static_cast<int32_t>(width));
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
    BatchWorkspaceRows rows(tiling, groups, base, col, count);
    uint32_t k = 0;
    uint64_t offset = 0;
    while (rows.Next(k, offset)) {
        Adds(dst[k * tiling.colTile], cached[offset], 0.0F, static_cast<int32_t>(width));
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
    } else if constexpr (IsLogDomain(Mode) && Phase == kPhaseNormalize) {
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

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_SOFTMAX_KERNEL_STD_H
#define OPS_GNN_SCATTER_SOFTMAX_KERNEL_STD_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_io.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_reduce.h"

/**
 * Passes of the grouped standard deviation (`scatter_std`).
 *
 * The operator keeps the row/tile driver, the pipeline and the reduction
 * bookkeeping of the grouped-softmax family; what it adds are the two full
 * passes that turn the group sums into a group variance, the member count that
 * divides them, and the two passes that share the reduced output with the
 * grouped log-sum-exp:
 *
 *   - `Sum` accumulates the group sum over the source and counts the members of
 *     every group in the same pass;
 *   - `Deviation` accumulates `(src - mean)^2`, with the mean rebuilt from the
 *     group sum and the count;
 *   - `FinalizeStd` writes `sqrt(deviation / (n - correction))` once per group
 *     and leaves the mean in place of the group sum for the backward;
 *   - `GradElementStd` is the elementwise backward
 *     `grad_out[group] * (src - mean[group]) / ((n - correction) * std)`, with a
 *     gradient of 0 where the saved output is 0, as `torch.std` reports.
 */

namespace ScatterSoftmaxNpu {

// Flushes the member counts of a core. The workspace holds one fp32 counter per
// group, so the flush is one atomic add of the whole dense block: the counts of
// all the cores accumulate per group, with nothing to step over.
template <typename T>
__aicore__ inline void FlushCounts(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                   PipelineSync& sync)
{
    if (!CountsFitUb(tiling)) {
        return;
    }
    // The counts were incremented by the scalar unit, so the store waits for
    // every pipe; the barrier runs once per core.
    PipeBarrier<PIPE_ALL>();
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    SetAtomicAdd<float>();
    DataCopy(t.countGm, t.counts, static_cast<int32_t>(tiling.countElems));
    DisableDmaAtomic();
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

// Counts the members of every group over the rows of batch 0, which is the one
// batch that has to be counted: every batch repeats the same index vector.
//
// The counts a core sees are kept in its own dense UB array while the whole
// count workspace fits there, and flushed once at the end of the pass, which
// turns one atomic per row into one atomic per group. A count workspace too
// large for the buffer falls back to one 32-byte atomic per row; the workspace
// is dense, so those atomics stay within a compact, cache-resident range.
template <typename T>
__aicore__ inline void CountRows(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                 PipelineSync& sync, uint64_t base, uint32_t count)
{
    if (CountsFitUb(tiling)) {
        for (uint32_t k = 0; k < count; ++k) {
            if (base + k >= tiling.dimLength) {
                return;
            }
            const int32_t group = t.groups.GetValue(k);
            if (!GroupValid(tiling, group)) {
                continue;
            }
            const uint64_t slot = CountBase(group);
            t.counts.SetValue(slot, t.counts.GetValue(slot) + 1.0F);
        }
        return;
    }
    // The one-hot atomic blocks were built by InitAccumulator, once per core.
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    SetAtomicAdd<float>();
    for (uint32_t k = 0; k < count; ++k) {
        if (base + k >= tiling.dimLength) {
            break;
        }
        const int32_t group = t.groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        const uint64_t block = static_cast<uint64_t>(group) / SCATTER_SOFTMAX_COUNT_ALIGN *
                               SCATTER_SOFTMAX_COUNT_ALIGN;
        const uint32_t lane = static_cast<uint32_t>(group) % SCATTER_SOFTMAX_COUNT_ALIGN;
        DataCopy(t.countGm[block],
                 t.onehot[static_cast<int32_t>(lane * SCATTER_SOFTMAX_COUNT_ALIGN)],
                 static_cast<int32_t>(SCATTER_SOFTMAX_COUNT_ALIGN));
    }
    DisableDmaAtomic();
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

// sum[b][g][a] += src, and the member count of every group. This is the first
// of the two full passes of the standard deviation; the mean is derived from
// the sum and the count by the pass that needs it instead of being stored.
template <typename T>
__aicore__ inline void SumBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                PipelineSync& sync, uint64_t base, uint32_t count,
                                uint64_t col, uint32_t valid, uint32_t width)
{
    if (col == 0 && base < tiling.dimLength) {
        CountRows(t, tiling, sync, base, count);
    }
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    CopyRowsIn(t.narrow, t.srcGm, tiling, base, count, col, valid);
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
    for (uint32_t k = 0; k < count; ++k) {
        const uint64_t offset = k * tiling.colTile;
        // Padded lanes stay zero so they add nothing to the group sum; a full
        // tile has none and skips the whole-tile fill.
        if (valid < width) {
            Duplicate(t.wide[offset], 0.0F, static_cast<int32_t>(tiling.colTile));
        }
        WidenToFp32(t.wide[offset], t.narrow[offset], valid);
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    StoreReduced<false>(t, t.sumGm, tiling, base, count, col, width, t.wide);
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

// deviation[b][g][a] += (src - sum[b][g][a] / n[g])^2, the second full pass of
// the standard deviation.
//
// The mean is rebuilt per row from the group sum and the group count, which
// keeps the pass to one read of the sum workspace. Centering on the actual mean
// - rather than expanding `sum(x^2) - n * mean^2` - is what keeps the pass
// accurate for data with a large offset, where the expanded form cancels its
// own significant digits.
template <typename T>
__aicore__ inline void DeviationBatch(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                      PipelineSync& sync, uint64_t base, uint32_t count,
                                      uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    CopyRowsIn(t.narrow, t.srcGm, tiling, base, count, col, valid);
    if (tiling.localAccum != 0) {
        CopyWorkspaceCached(t.wide2, t.cached, tiling, t.groups, base, count, col, width);
    } else {
        CopyWorkspaceIn(t.wide2, t.sumGm, tiling, t.groups, base, count, col, width);
    }
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = t.groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        const uint64_t offset = k * tiling.colTile;
        if (valid < width) {
            Duplicate(t.wide[offset], 0.0F, static_cast<int32_t>(tiling.colTile));
        }
        WidenToFp32(t.wide[offset], t.narrow[offset], valid);
        // A group without members owns no row, so its count is positive here.
        const float members = t.countGm.GetValue(CountBase(group));
        Muls(t.wide2[offset], t.wide2[offset], 1.0F / members, static_cast<int32_t>(valid));
        Sub(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
        Mul(t.wide[offset], t.wide[offset], t.wide[offset], static_cast<int32_t>(valid));
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    StoreReduced<false>(t, t.deviationGm, tiling, base, count, col, width, t.wide);
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

// Reads the group sum and the accumulated deviation of the batch: one span when
// its rows are contiguous, one burst per row otherwise.
template <typename T>
__aicore__ inline void LoadDeviationRows(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                         PipelineSync& sync, uint64_t base, uint32_t count,
                                         uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    if (BatchIsContiguous(tiling, col, valid)) {
        const int32_t span = static_cast<int32_t>(count * tiling.afterPadded);
        DataCopy(t.wide, t.sumGm[base * tiling.afterPadded], span);
        DataCopy(t.wide2, t.deviationGm[base * tiling.afterPadded], span);
    } else {
        for (uint32_t k = 0; k < count; ++k) {
            const uint64_t offset = (base + k) * tiling.afterPadded + col;
            DataCopy(t.wide[k * tiling.colTile], t.sumGm[offset], static_cast<int32_t>(width));
            DataCopy(t.wide2[k * tiling.colTile], t.deviationGm[offset],
                     static_cast<int32_t>(width));
        }
    }
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
}

// Writes the mean over the group sum the backward reads, and the deviation to
// the output.
template <typename T>
__aicore__ inline void StoreDeviationRows(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                          PipelineSync& sync, uint64_t base, uint32_t count,
                                          uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    if (BatchIsContiguous(tiling, col, valid)) {
        DataCopy(t.sumGm[base * tiling.afterPadded], t.wide3,
                 static_cast<int32_t>(count * tiling.afterPadded));
        DataCopyPad(t.outGm[base * tiling.after], t.narrow,
                    Bytes1D(count * static_cast<uint32_t>(tiling.after) * sizeof(T)));
    } else {
        // The mean replaces the group sum it was derived from.
        for (uint32_t k = 0; k < count; ++k) {
            DataCopy(t.sumGm[(base + k) * tiling.afterPadded + col], t.wide3[k * tiling.colTile],
                     static_cast<int32_t>(width));
        }
        for (uint32_t k = 0; k < count; ++k) {
            DataCopyPad(t.outGm[(base + k) * tiling.after + col], t.narrow[k * tiling.colTile],
                        Bytes1D(valid * sizeof(T)));
        }
    }
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

// out[b][g][a] = sqrt(deviation / (n - correction)), and the mean replaces the
// group sum it was derived from so the backward reuses it. A group without
// spread - no member, or the single member of the unbiased deviation - has no
// standard deviation and is reported as NaN, like `torch.std`; a group without
// members has no mean either and stores NaN in the mean workspace.
template <typename T>
__aicore__ inline void FinalizeStd(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                   PipelineSync& sync, uint64_t base, uint32_t count,
                                   uint64_t col, uint32_t valid, uint32_t width)
{
    LoadDeviationRows(t, tiling, sync, base, count, col, valid, width);
    for (uint32_t k = 0; k < count; ++k) {
        const uint64_t offset = k * tiling.colTile;
        // Output rows are (batch, group) pairs, so the label follows from the
        // row index and the counts of the group are read directly.
        const int32_t group = static_cast<int32_t>((base + k) % tiling.groupCount);
        const float members = t.countGm.GetValue(CountBase(group));
        const float divisor = members - tiling.correction;
        if (members > 0.0F) {
            // The mean replaces the group sum in the row buffer. Padded lanes
            // stay zero so the store of a full row never picks up the value of
            // the previous row.
            if (valid < width) {
                Duplicate(t.wide3[offset], 0.0F, static_cast<int32_t>(tiling.colTile));
            }
            Muls(t.wide3[offset], t.wide[offset], 1.0F / members, static_cast<int32_t>(valid));
        } else {
            // A group without members has no mean. The whole row is written so
            // the store below never reads a lane the pass did not assign; the
            // mean workspace is what the pybind forward returns, so leaving the
            // row uninitialised would expose it.
            Duplicate(t.wide3[offset], SCATTER_SOFTMAX_NAN, static_cast<int32_t>(tiling.colTile));
        }
        if (divisor > 0.0F) {
            Muls(t.wide[offset], t.wide2[offset], 1.0F / divisor, static_cast<int32_t>(valid));
            Sqrt(t.wide[offset], t.wide[offset], static_cast<int32_t>(valid));
        } else {
            Duplicate(t.wide[offset], SCATTER_SOFTMAX_NAN, static_cast<int32_t>(tiling.colTile));
        }
        NarrowFromFp32(t.narrow[offset], t.wide[offset], valid);
    }
    StoreDeviationRows(t, tiling, sync, base, count, col, valid, width);
}

// grad_src = grad_out[group] * (src - mean[group]) / ((n - correction) * std).
// The mean was kept by the finalize pass and the count turns the saved output
// into the `(n - correction) * std` denominator.
template <typename T>
__aicore__ inline void GradElementStd(PhaseTensors<T>& t, const ScatterSoftmaxTilingData& tiling,
                                      PipelineSync& sync, uint64_t base, uint32_t count,
                                      uint64_t col, uint32_t valid, uint32_t width)
{
    SyncEvent<HardEvent::V_MTE2>(sync.computeToLoad);
    CopyRowsIn(t.narrow, t.srcGm, tiling, base, count, col, valid);
    CopyGroupRowsIn(t.narrow2, t.outGm, tiling, t.groups, base, count, col, valid);
    CopyGroupRowsIn(t.narrow3, t.gradGm, tiling, t.groups, base, count, col, valid);
    CopyWorkspaceIn(t.wide2, t.sumGm, tiling, t.groups, base, count, col, width);
    SyncEvent<HardEvent::MTE2_V>(sync.loadToCompute);
    for (uint32_t k = 0; k < count; ++k) {
        const int32_t group = t.groups.GetValue(k);
        if (!GroupValid(tiling, group)) {
            continue;
        }
        const uint64_t offset = k * tiling.colTile;
        const float divisor = t.countGm.GetValue(CountBase(group)) - tiling.correction;
        WidenToFp32(t.wide[offset], t.narrow[offset], valid);
        Sub(t.wide[offset], t.wide[offset], t.wide2[offset], static_cast<int32_t>(valid));
        WidenToFp32(t.wide3[offset], t.narrow3[offset], valid);
        Mul(t.wide[offset], t.wide[offset], t.wide3[offset], static_cast<int32_t>(valid));
        WidenToFp32(t.wide3[offset], t.narrow2[offset], valid);
        Muls(t.wide3[offset], t.wide3[offset], divisor, static_cast<int32_t>(valid));
        Div(t.wide[offset], t.wide[offset], t.wide3[offset], static_cast<int32_t>(valid));
        // ``torch.std`` reports a gradient of 0 for a group without spread: it
        // zeroes the variance gradient before propagating it, so the quotient
        // above would be 0 / 0. The saved output of the row was widened into
        // `wide3`, and its source buffer is free again, so one comparison per
        // tile carries the mask that selects the zero. The comparison is `== 0`
        // rather than `!= 0` because an unordered comparison with the NaN of an
        // empty or single-member group must keep the NaN; only the exact zero is
        // replaced. The mask API only writes whole 256-byte repeats, so the
        // count is rounded up and the extra lanes come from the padding the row
        // buffer reserves.
        LocalTensor<uint8_t> zeroStd = t.narrow2.template ReinterpretCast<uint8_t>();
        CompareScalar(zeroStd, t.wide3[offset], 0.0F, CMPMODE::EQ, CompareRepeatCount(valid));
        Duplicate(t.wide2[offset], 0.0F, static_cast<int32_t>(valid));
        Select(t.wide2[offset], zeroStd, t.wide2[offset], t.wide[offset],
               SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(valid));
        NarrowFromFp32(t.narrow[offset], t.wide2[offset], valid);
    }
    SyncEvent<HardEvent::V_MTE3>(sync.computeToStore);
    CopyRowsOut(t.gradSrcGm, t.narrow, tiling, t.groups, base, count, col, valid);
    SyncEvent<HardEvent::MTE3_V>(sync.storeToCompute);
}

}  // namespace ScatterSoftmaxNpu

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_STD_H

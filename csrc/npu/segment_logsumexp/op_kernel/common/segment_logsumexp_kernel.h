/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * AscendC device implementation of the CSR log-sum-exp shared by arch22
 * (A2/A3) and arch35 (950).
 *
 * A CSR segment is a contiguous row range, so nothing has to be reordered and
 * no atomic unit is involved. There are two schedules:
 *
 *  * one core per (segment, feature window) when there are enough segments to
 *    fill the machine: the maximum, the exponential sum and the logarithm all
 *    stay in the unified buffer of that core, in a single kernel launch;
 *  * a two-level reduction when the segments are too few: the rows of a segment
 *    are cut into ``rowChunks`` pieces, and four kernels split the work over
 *    every core, exchanging the per-chunk maximum and exponential sum through
 *    an fp32 workspace:
 *
 *      phase 1  per (segment, feature window, row chunk) maximum
 *      phase 2  maximum over the chunks of a window, clamped  -> shift
 *      phase 3  per chunk sum of exp(src - shift)
 *      phase 4  sum over the chunks, shift + ln(total), store
 *
 * Each phase is a separate launch on one stream, which is the only cross-core
 * ordering that is needed.
 *
 * The shift is clamped into [-FLT_MAX, +FLT_MAX] so it always stays finite: an
 * all ``-inf`` segment then reduces to ``-FLT_MAX + ln(0) = -inf`` and a
 * segment holding ``+inf`` to ``FLT_MAX + ln(+inf) = +inf``, which is what
 * ``torch.logsumexp`` returns for those inputs. The same arithmetic makes an
 * empty segment (no rows, so ``ln(0)``) come out as ``-inf``.
 */

#ifndef OPS_GNN_SEGMENT_LOGSUMEXP_KERNEL_H
#define OPS_GNN_SEGMENT_LOGSUMEXP_KERNEL_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "segment_logsumexp/op_kernel/common/segment_logsumexp_launch.h"

namespace SegmentLogSumExpNpu {

using namespace AscendC;

// Reduction steps. The per-chunk phases walk ``tiling.tasks``, the per-window
// phases walk ``tiling.baseTasks``; kFused does a whole window on one core.
enum class Phase {
    kLocalMax = 0,
    kReduceShift = 1,
    kLocalSum = 2,
    kFinish = 3,
    kFused = 4,
};

template <HardEvent Event>
__aicore__ inline void Order(event_t id)
{
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

// Features covered by a tile that starts at `begin`.
__aicore__ inline uint32_t TileWidth(uint64_t total, uint64_t begin, uint32_t tile)
{
    const uint64_t remaining = total - begin;
    const uint64_t limit = static_cast<uint64_t>(tile);
    return static_cast<uint32_t>(remaining < limit ? remaining : limit);
}

// Unified-buffer allocations and vector destinations are 32-byte based.
__aicore__ inline uint32_t UbBytes(uint64_t elements, uint64_t elementBytes)
{
    const uint64_t bytes = elements * elementBytes;
    return static_cast<uint32_t>((bytes + 31) / 32 * 32);
}

// A row slice is not 32-byte aligned in global memory, so it always travels as
// one byte-counted burst.
__aicore__ inline DataCopyExtParams Bytes1D(uint32_t bytes)
{
    return DataCopyExtParams{1, bytes, 0, 0, 0};
}

// Rows of the next burst: capped by the staging capacity and the rows left.
__aicore__ inline uint32_t NextBatch(uint32_t capacity, uint64_t left)
{
    return static_cast<uint32_t>(left < static_cast<uint64_t>(capacity) ? left : capacity);
}

// Event pair per pipeline hazard. The four ids are reused by every row: each
// Order call sets and immediately waits, so the ordering is exact and no id
// stays pending across a reuse.
struct RowPipeline {
    event_t loadToReduce;   // MTE2 -> V
    event_t reduceToLoad;   // V -> MTE2
    event_t reduceToStore;  // V -> MTE3
    event_t storeToReuse;   // MTE3 -> V (and, through issue order, MTE2)

    __aicore__ inline void Init(TPipe& pipe)
    {
        loadToReduce = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        reduceToLoad = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
        reduceToStore = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
        storeToReuse = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
    }

    __aicore__ inline void Release(TPipe& pipe)
    {
        pipe.ReleaseEventID<HardEvent::MTE2_V>(loadToReduce);
        pipe.ReleaseEventID<HardEvent::V_MTE2>(reduceToLoad);
        pipe.ReleaseEventID<HardEvent::V_MTE3>(reduceToStore);
        pipe.ReleaseEventID<HardEvent::MTE3_V>(storeToReuse);
    }
};

template <typename T>
struct TaskTensors {
    TBuf<TPosition::VECCALC> dtypeBuffer;   // staging area of one output row
    TBuf<TPosition::VECCALC> batchBuffer;   // staged source rows of one burst
    TBuf<TPosition::VECCALC> rowBuffer;     // one source row widened to fp32
    TBuf<TPosition::VECCALC> shiftBuffer;   // per-feature maximum / shift
    TBuf<TPosition::VECCALC> totalBuffer;   // per-feature exponential sum
    LocalTensor<T> row;
    LocalTensor<T> tile;
    LocalTensor<float> wideRow;
    LocalTensor<float> shift;
    LocalTensor<float> total;
    GlobalTensor<T> src;
    GlobalTensor<T> out;
    GlobalTensor<int32_t> indptr;
    GlobalTensor<float> splitMax;
    GlobalTensor<float> splitSum;
    uint32_t tileStride;   // elements between two staged rows
    uint32_t rowBatch;     // staged rows per burst
    uint32_t srcRowBytes;  // bytes between two source rows
};

template <typename T>
__aicore__ inline void InitTensors(TaskTensors<T>& tensors, TPipe& pipe,
                                   const SegmentLogSumExpTilingData& tiling,
                                   GM_ADDR src, GM_ADDR indptr, GM_ADDR out,
                                   GM_ADDR splitMax, GM_ADDR splitSum)
{
    const uint32_t dtypeRowBytes = UbBytes(tiling.colTile, sizeof(T));
    pipe.InitBuffer(tensors.dtypeBuffer, dtypeRowBytes);
    pipe.InitBuffer(tensors.batchBuffer, SEGMENT_LOGSUMEXP_BATCH_BYTES);
    pipe.InitBuffer(tensors.rowBuffer, UbBytes(tiling.colTile, sizeof(float)));
    pipe.InitBuffer(tensors.shiftBuffer, UbBytes(tiling.colTile, sizeof(float)));
    pipe.InitBuffer(tensors.totalBuffer, UbBytes(tiling.colTile, sizeof(float)));
    tensors.row = tensors.dtypeBuffer.template Get<T>();
    tensors.tile = tensors.batchBuffer.template Get<T>();
    tensors.wideRow = tensors.rowBuffer.template Get<float>();
    tensors.shift = tensors.shiftBuffer.template Get<float>();
    tensors.total = tensors.totalBuffer.template Get<float>();
    tensors.tileStride = dtypeRowBytes / sizeof(T);
    const uint32_t perBurst = static_cast<uint32_t>(SEGMENT_LOGSUMEXP_BATCH_BYTES / dtypeRowBytes);
    const uint32_t capped =
        perBurst < SEGMENT_LOGSUMEXP_MAX_BATCH_ROWS ? perBurst : SEGMENT_LOGSUMEXP_MAX_BATCH_ROWS;
    // Rows are staged together only when the window spans the whole row: the
    // burst then walks contiguous memory and needs no source stride. A narrower
    // window keeps one row per transfer, because its row gap would have to be
    // expressed as a source stride whose unit differs between the architectures.
    tensors.rowBatch = tiling.colTiles > 1 ? 1 : capped;
    tensors.srcRowBytes = static_cast<uint32_t>(tiling.features * sizeof(T));
    tensors.src.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src), tiling.rows * tiling.features);
    tensors.out.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), tiling.segments * tiling.features);
    tensors.indptr.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indptr), tiling.segments + 1);
    tensors.splitMax.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(splitMax),
                                     tiling.tasks * tiling.colTile);
    tensors.splitSum.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(splitSum),
                                     tiling.tasks * tiling.colTile);
}

// Widens `count` source elements into fp32. T == float still goes through a
// vector instruction so every dtype shares one downstream code path.
template <typename T>
__aicore__ inline void WidenToFp32(const LocalTensor<float>& dst, const LocalTensor<T>& src,
                                   uint32_t count)
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

// `count` feature windows of consecutive source rows, loaded as one strided
// burst. Every staged row starts on a 32-byte boundary because DataCopyPad pads
// each block to one, which is what the row loop then reads.
template <typename T>
__aicore__ inline void LoadRows(TaskTensors<T>& tensors, uint64_t offset, uint32_t count,
                                uint32_t valid, RowPipeline& events)
{
    const uint32_t bytes = static_cast<uint32_t>(valid * sizeof(T));
    const DataCopyExtParams params{
        static_cast<uint16_t>(count), bytes, tensors.srcRowBytes - bytes, 0, 0};
    Order<HardEvent::V_MTE2>(events.reduceToLoad);
    DataCopyPad(tensors.tile, tensors.src[offset], params, DataCopyPadExtParams<T>(false, 0, 0, 0));
    Order<HardEvent::MTE2_V>(events.loadToReduce);
}

// The reduced value is written to the padded feature lanes as well, but only
// the `valid` lanes are ever stored back to global memory.
template <typename T>
__aicore__ inline void StoreRow(TaskTensors<T>& tensors, uint64_t offset, uint32_t valid,
                                RowPipeline& events)
{
    Order<HardEvent::V_MTE3>(events.reduceToStore);
    DataCopyPad(tensors.out[offset], tensors.row,
                Bytes1D(static_cast<uint32_t>(valid * sizeof(T))));
    Order<HardEvent::MTE3_V>(events.storeToReuse);
}

// Whole fp32 rows of the phase workspace, which are colTile-aligned.
__aicore__ inline void LoadSplit(const GlobalTensor<float>& src, const LocalTensor<float>& dst,
                                 uint64_t offset, uint32_t width, RowPipeline& events)
{
    Order<HardEvent::V_MTE2>(events.reduceToLoad);
    DataCopyPad(dst, src[offset], Bytes1D(static_cast<uint32_t>(width * sizeof(float))),
                DataCopyPadExtParams<float>(false, 0, 0, 0));
    Order<HardEvent::MTE2_V>(events.loadToReduce);
}

__aicore__ inline void StoreSplit(const GlobalTensor<float>& dst, uint64_t offset,
                                  const LocalTensor<float>& src, uint32_t width,
                                  RowPipeline& events)
{
    Order<HardEvent::V_MTE3>(events.reduceToStore);
    DataCopyPad(dst[offset], src, Bytes1D(static_cast<uint32_t>(width * sizeof(float))));
    Order<HardEvent::MTE3_V>(events.storeToReuse);
}

// exp(src - shift) only stays in range for a finite shift.
__aicore__ inline void ClampShift(const LocalTensor<float>& shift, uint32_t width)
{
    Mins(shift, shift, SEGMENT_LOGSUMEXP_LARGEST_FLOAT, static_cast<int32_t>(width));
    Maxs(shift, shift, -SEGMENT_LOGSUMEXP_LARGEST_FLOAT, static_cast<int32_t>(width));
}

// Geometry of a (segment, feature window) pair.
struct WindowGeometry {
    uint64_t segment;
    uint64_t window;      // index of the feature window inside the row
    uint64_t slot;        // first workspace entry of this window
    uint64_t outOffset;   // element offset of the output row
    uint32_t valid;
};

// Geometry of one (segment, feature window, row chunk) task.
struct SplitGeometry {
    WindowGeometry win;
    uint64_t first;   // first source row of the chunk
    uint64_t rows;    // rows of the chunk
    uint32_t chunk;
};

__aicore__ inline WindowGeometry ResolveWindow(const SegmentLogSumExpTilingData& tiling,
                                               uint64_t baseTask)
{
    WindowGeometry geo;
    geo.segment = baseTask / tiling.colTiles;
    geo.window = baseTask % tiling.colTiles;
    geo.slot = (geo.segment * tiling.colTiles + geo.window) * tiling.rowChunks * tiling.colTile;
    geo.valid = TileWidth(tiling.features, geo.window * tiling.colTile,
                          static_cast<uint32_t>(tiling.colTile));
    geo.outOffset = geo.segment * tiling.features + geo.window * tiling.colTile;
    return geo;
}

template <typename T>
__aicore__ inline SplitGeometry ResolveSplit(const TaskTensors<T>& tensors,
                                             const SegmentLogSumExpTilingData& tiling,
                                             uint64_t task)
{
    SplitGeometry geo;
    const uint64_t perSegment = tiling.colTiles * tiling.rowChunks;
    const uint64_t segment = task / perSegment;
    const uint64_t within = task % perSegment;
    geo.chunk = static_cast<uint32_t>(within % tiling.rowChunks);
    geo.win = ResolveWindow(tiling, segment * tiling.colTiles + within / tiling.rowChunks);

    // A segment of `length` rows is cut into rowChunks pieces of `per` rows. The
    // last piece, and every piece of a segment shorter than rowChunks, is
    // clamped so that it never reads past the segment.
    const int32_t begin = tensors.indptr.GetValue(geo.win.segment);
    const int32_t end = tensors.indptr.GetValue(geo.win.segment + 1);
    const uint64_t length = end > begin ? static_cast<uint64_t>(end - begin) : 0;
    const uint64_t per = (length + tiling.rowChunks - 1) / tiling.rowChunks;
    const uint64_t stop = static_cast<uint64_t>(begin) + (geo.chunk + 1) * per;
    geo.first = static_cast<uint64_t>(begin) + geo.chunk * per;
    const uint64_t limit = stop < static_cast<uint64_t>(end) ? stop : static_cast<uint64_t>(end);
    geo.rows = geo.first < limit ? limit - geo.first : 0;
    return geo;
}

// Accumulates over the rows of a chunk: the per-feature maximum into `shift`, or
// sum(exp(src - shift)) into `total`.
template <bool kSum, typename T>
__aicore__ inline void ReduceChunkRows(TaskTensors<T>& tensors,
                                       const SegmentLogSumExpTilingData& tiling,
                                       RowPipeline& events, const SplitGeometry& geo)
{
    const uint32_t width = static_cast<uint32_t>(tiling.colTile);
    if constexpr (kSum) {
        Duplicate(tensors.total, 0.0F, static_cast<int32_t>(width));
    } else {
        Duplicate(tensors.shift, SEGMENT_LOGSUMEXP_NEGATIVE_INFINITY, static_cast<int32_t>(width));
    }
    const uint64_t base = geo.first * tiling.features + geo.win.window * tiling.colTile;
    uint64_t row = 0;
    while (row < geo.rows) {
        const uint32_t count = NextBatch(tensors.rowBatch, geo.rows - row);
        LoadRows(tensors, base + row * tiling.features, count, geo.win.valid, events);
        for (uint32_t i = 0; i < count; ++i) {
            WidenToFp32(tensors.wideRow, tensors.tile[i * tensors.tileStride], geo.win.valid);
            if constexpr (kSum) {
                Sub(tensors.wideRow, tensors.wideRow, tensors.shift,
                    static_cast<int32_t>(geo.win.valid));
                Exp(tensors.wideRow, tensors.wideRow, static_cast<int32_t>(geo.win.valid));
                Add(tensors.total, tensors.total, tensors.wideRow,
                    static_cast<int32_t>(geo.win.valid));
            } else {
                Max(tensors.shift, tensors.shift, tensors.wideRow,
                    static_cast<int32_t>(geo.win.valid));
            }
        }
        row += count;
    }
}

// shift + ln(total) in the source dtype, stored to the output row.
template <typename T>
__aicore__ inline void FinishRow(TaskTensors<T>& tensors, const WindowGeometry& win,
                                 RowPipeline& events)
{
    Ln(tensors.total, tensors.total, static_cast<int32_t>(win.valid));
    Add(tensors.total, tensors.total, tensors.shift, static_cast<int32_t>(win.valid));
    NarrowFromFp32(tensors.row, tensors.total, win.valid);
    StoreRow(tensors, win.outOffset, win.valid, events);
}

// Whole window on one core: the maximum, the exponential sum and the logarithm
// never leave the unified buffer, so nothing crosses a kernel boundary.
template <typename T>
__aicore__ inline void FusedPass(TaskTensors<T>& tensors,
                                 const SegmentLogSumExpTilingData& tiling,
                                 RowPipeline& events, uint64_t baseTask)
{
    const SplitGeometry geo = ResolveSplit(tensors, tiling, baseTask);
    ReduceChunkRows<false>(tensors, tiling, events, geo);
    ClampShift(tensors.shift, static_cast<uint32_t>(tiling.colTile));
    ReduceChunkRows<true>(tensors, tiling, events, geo);
    FinishRow(tensors, geo.win, events);
}

// phase 1: per-chunk maximum, stored to the workspace.
template <typename T>
__aicore__ inline void LocalMaxPass(TaskTensors<T>& tensors,
                                    const SegmentLogSumExpTilingData& tiling,
                                    RowPipeline& events, uint64_t task)
{
    const SplitGeometry geo = ResolveSplit(tensors, tiling, task);
    ReduceChunkRows<false>(tensors, tiling, events, geo);
    StoreSplit(tensors.splitMax, geo.win.slot + geo.chunk * tiling.colTile, tensors.shift,
               static_cast<uint32_t>(tiling.colTile), events);
}

// phase 2: maximum over the chunks of one window, clamped into the shift.
template <typename T>
__aicore__ inline void ReduceShiftPass(TaskTensors<T>& tensors,
                                       const SegmentLogSumExpTilingData& tiling,
                                       RowPipeline& events, uint64_t baseTask)
{
    const WindowGeometry win = ResolveWindow(tiling, baseTask);
    const uint32_t width = static_cast<uint32_t>(tiling.colTile);
    Duplicate(tensors.shift, SEGMENT_LOGSUMEXP_NEGATIVE_INFINITY, static_cast<int32_t>(width));
    for (uint64_t chunk = 0; chunk < tiling.rowChunks; ++chunk) {
        LoadSplit(tensors.splitMax, tensors.wideRow, win.slot + chunk * tiling.colTile, width,
                  events);
        Max(tensors.shift, tensors.shift, tensors.wideRow, static_cast<int32_t>(width));
    }
    ClampShift(tensors.shift, width);
    // The result goes back into the first chunk entry, which this core owns, so
    // the reduction is in place.
    StoreSplit(tensors.splitMax, win.slot, tensors.shift, width, events);
}

// phase 3: per-chunk sum of exp(src - shift), stored to the workspace.
template <typename T>
__aicore__ inline void LocalSumPass(TaskTensors<T>& tensors,
                                    const SegmentLogSumExpTilingData& tiling,
                                    RowPipeline& events, uint64_t task)
{
    const SplitGeometry geo = ResolveSplit(tensors, tiling, task);
    const uint32_t width = static_cast<uint32_t>(tiling.colTile);
    // The shift of the window lives in its first chunk entry: phase 2 wrote it
    // there when the rows were split, phase 1 wrote it when they were not.
    LoadSplit(tensors.splitMax, tensors.shift, geo.win.slot, width, events);
    ClampShift(tensors.shift, width);
    ReduceChunkRows<true>(tensors, tiling, events, geo);
    StoreSplit(tensors.splitSum, geo.win.slot + geo.chunk * tiling.colTile, tensors.total, width,
               events);
}

// phase 4: sum over the chunks, then shift + ln(total) into the output.
template <typename T>
__aicore__ inline void FinishPass(TaskTensors<T>& tensors,
                                  const SegmentLogSumExpTilingData& tiling,
                                  RowPipeline& events, uint64_t baseTask)
{
    const WindowGeometry win = ResolveWindow(tiling, baseTask);
    const uint32_t width = static_cast<uint32_t>(tiling.colTile);
    Duplicate(tensors.total, 0.0F, static_cast<int32_t>(width));
    for (uint64_t chunk = 0; chunk < tiling.rowChunks; ++chunk) {
        LoadSplit(tensors.splitSum, tensors.wideRow, win.slot + chunk * tiling.colTile, width,
                  events);
        Add(tensors.total, tensors.total, tensors.wideRow, static_cast<int32_t>(width));
    }
    LoadSplit(tensors.splitMax, tensors.shift, win.slot, width, events);
    // A window without rows has total == 0, so ln(0) = -inf and the clamped
    // shift leaves the result at -inf; a window holding +inf keeps it at +inf.
    FinishRow(tensors, win, events);
}

// The per-chunk phases walk the split grid, the per-window phases the window
// grid.
__aicore__ inline bool IsPerWindow(Phase phase)
{
    return phase == Phase::kReduceShift || phase == Phase::kFinish || phase == Phase::kFused;
}

template <Phase phase, typename T>
__aicore__ inline void RunTask(TaskTensors<T>& tensors, const SegmentLogSumExpTilingData& tiling,
                               RowPipeline& events, uint64_t task)
{
    if constexpr (phase == Phase::kLocalMax) {
        LocalMaxPass(tensors, tiling, events, task);
    } else if constexpr (phase == Phase::kReduceShift) {
        ReduceShiftPass(tensors, tiling, events, task);
    } else if constexpr (phase == Phase::kLocalSum) {
        LocalSumPass(tensors, tiling, events, task);
    } else if constexpr (phase == Phase::kFinish) {
        FinishPass(tensors, tiling, events, task);
    } else {
        FusedPass(tensors, tiling, events, task);
    }
}

template <Phase phase, typename T>
__aicore__ inline void RunPhase(GM_ADDR src, GM_ADDR indptr, GM_ADDR out, GM_ADDR splitMax,
                                GM_ADDR splitSum, const SegmentLogSumExpTilingData& tiling)
{
    const bool perWindow = IsPerWindow(phase);
    const uint32_t blocks = perWindow ? tiling.baseBlockNum : tiling.blockNum;
    if (tiling.tasks == 0 || static_cast<uint32_t>(GetBlockIdx()) >= blocks) {
        return;
    }
    TaskTensors<T> tensors;
    TPipe pipe;
    InitTensors(tensors, pipe, tiling, src, indptr, out, splitMax, splitSum);
    RowPipeline events;
    events.Init(pipe);
    const uint64_t tasks = perWindow ? tiling.baseTasks : tiling.tasks;
    for (uint64_t task = static_cast<uint64_t>(GetBlockIdx()); task < tasks;
         task += static_cast<uint64_t>(blocks)) {
        RunTask<phase>(tensors, tiling, events, task);
    }
    events.Release(pipe);
}

template <Phase phase, typename T>
__attribute__((aiv)) __global__ __aicore__ void SegmentLogSumExpKernel(
    GM_ADDR src, GM_ADDR indptr, GM_ADDR out, GM_ADDR splitMax, GM_ADDR splitSum,
    SegmentLogSumExpTilingData tiling)
{
    RunPhase<phase, T>(src, indptr, out, splitMax, splitSum, tiling);
}

// ---------------------------------------------------------------------------
// Host-side launch (compiled by the same device-capable translation unit).
// ---------------------------------------------------------------------------
namespace Launch {

template <typename T>
inline void LaunchPhase(Phase phase, uint32_t blocks, GM_ADDR src, GM_ADDR indptr, GM_ADDR out,
                        GM_ADDR splitMax, GM_ADDR splitSum,
                        const SegmentLogSumExpTilingData& tiling, aclrtStream stream)
{
    switch (phase) {
        case Phase::kLocalMax:
            SegmentLogSumExpKernel<Phase::kLocalMax, T><<<blocks, nullptr, stream>>>(
                src, indptr, out, splitMax, splitSum, tiling);
            break;
        case Phase::kReduceShift:
            SegmentLogSumExpKernel<Phase::kReduceShift, T><<<blocks, nullptr, stream>>>(
                src, indptr, out, splitMax, splitSum, tiling);
            break;
        case Phase::kLocalSum:
            SegmentLogSumExpKernel<Phase::kLocalSum, T><<<blocks, nullptr, stream>>>(
                src, indptr, out, splitMax, splitSum, tiling);
            break;
        case Phase::kFinish:
            SegmentLogSumExpKernel<Phase::kFinish, T><<<blocks, nullptr, stream>>>(
                src, indptr, out, splitMax, splitSum, tiling);
            break;
        default:
            SegmentLogSumExpKernel<Phase::kFused, T><<<blocks, nullptr, stream>>>(
                src, indptr, out, splitMax, splitSum, tiling);
            break;
    }
}

// A segment that is split is reduced in four phases on the caller's stream: the
// kernel boundary is the only synchronisation between them. When the segments
// already fill the machine, one fused kernel per window is cheaper than the
// workspace round trip.
template <typename T>
inline void Forward(const void* src, const int32_t* indptr, void* out, float* splitMax,
                    float* splitSum, const SegmentLogSumExpTilingData& tiling, aclrtStream stream)
{
    GM_ADDR srcAddr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src));
    GM_ADDR indptrAddr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(indptr));
    GM_ADDR outAddr = reinterpret_cast<uint8_t*>(out);
    GM_ADDR splitMaxAddr = reinterpret_cast<uint8_t*>(splitMax);
    GM_ADDR splitSumAddr = reinterpret_cast<uint8_t*>(splitSum);

    if (tiling.rowChunks == 1) {
        LaunchPhase<T>(Phase::kFused, tiling.baseBlockNum, srcAddr, indptrAddr, outAddr,
                       splitMaxAddr, splitSumAddr, tiling, stream);
        return;
    }
    LaunchPhase<T>(Phase::kLocalMax, tiling.blockNum, srcAddr, indptrAddr, outAddr, splitMaxAddr,
                   splitSumAddr, tiling, stream);
    LaunchPhase<T>(Phase::kReduceShift, tiling.baseBlockNum, srcAddr, indptrAddr, outAddr,
                   splitMaxAddr, splitSumAddr, tiling, stream);
    LaunchPhase<T>(Phase::kLocalSum, tiling.blockNum, srcAddr, indptrAddr, outAddr, splitMaxAddr,
                   splitSumAddr, tiling, stream);
    LaunchPhase<T>(Phase::kFinish, tiling.baseBlockNum, srcAddr, indptrAddr, outAddr, splitMaxAddr,
                   splitSumAddr, tiling, stream);
}

}  // namespace Launch

}  // namespace SegmentLogSumExpNpu

namespace opsgnn {

void SegmentLogSumExp(const void* src, const int32_t* indptr, void* out,
                      float* splitMax, float* splitSum,
                      SegmentLogSumExpDType dtype,
                      const SegmentLogSumExpTilingData& tiling,
                      aclrtStream stream)
{
    switch (dtype) {
        case SEGMENT_LOGSUMEXP_F16:
            SegmentLogSumExpNpu::Launch::Forward<half>(src, indptr, out, splitMax, splitSum,
                                                      tiling, stream);
            break;
        case SEGMENT_LOGSUMEXP_F32:
            SegmentLogSumExpNpu::Launch::Forward<float>(src, indptr, out, splitMax, splitSum,
                                                       tiling, stream);
            break;
        case SEGMENT_LOGSUMEXP_BF16:
            SegmentLogSumExpNpu::Launch::Forward<bfloat16_t>(src, indptr, out, splitMax, splitSum,
                                                            tiling, stream);
            break;
        default:
            break;
    }
}

}  // namespace opsgnn

#endif  // OPS_GNN_SEGMENT_LOGSUMEXP_KERNEL_H

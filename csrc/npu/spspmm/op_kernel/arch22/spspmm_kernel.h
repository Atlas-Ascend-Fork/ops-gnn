/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// A2/A3 (arch22) implementation of the spspmm pipeline.
//
// dav-2201 has no SIMT execution, so the phases of the operator run as classic
// AscendC kernels instead of ``VF_CALL`` thread functions:
//
//   1. narrow the int64 COO indices to int32 and validate them,
//   2. build both CSR row pointers from a row histogram plus a three-pass scan,
//   3. count the candidate products of every output row,
//   4. merge the candidate columns per row with an ordered k-way merge,
//   5. compact the candidates into the coalesced COO result.
//
// Every global-memory write leaves through ``DataCopyPad``. A scalar store on
// this hardware rewrites the whole 32-byte block that holds the value, which is
// why these kernels never store to global memory that way; ``DataCopyPad``
// writes exactly the bytes it is given, so neighbouring elements of one block may
// be written by different AIVs. The parallel passes below still partition their
// outputs on whole 32-byte blocks, which keeps every range aligned, and the
// sticky flags are raised with an aligned atomic add of a whole block.
#pragma once
#include <cstdint>

#include "kernel_operator.h"

#include "../common/spspmm_common.h"

namespace opsgnn {
namespace spspmm_a2 {

// Elements moved per transfer, and per-row buffers of the row-parallel phases.
constexpr int64_t kEdgeTile = 2048;
constexpr int64_t kRowTile = 1024;
constexpr int64_t kScanTile = 2048;
constexpr int64_t kCopyTile = 2048;

// A-row entries handled with UB-resident cursors. Longer rows are merged by a
// dedicated single-block pass with the cursors in global memory, which keeps the
// row working set bounded without limiting the input.
constexpr int64_t kCursorCap = 8192;
constexpr int64_t kStageCap = 4096;

// Elements of one 32-byte global block. The parallel passes partition their
// rows or edges on these boundaries, so no block is shared by two AIVs.
constexpr int64_t kInt32Group = 8;
constexpr int64_t kInt64Group = 4;

// Candidate regions are rounded up to this many slots, so the int32 columns, the
// f32 values and the f16 values of one region all start on a block boundary.
constexpr int64_t kSlotAlign = 16;

// Elements of one block of the COO output: 16 int64 rows or columns, or 16
// values of either dtype.
constexpr int64_t kCopyGroup = 16;

// Sentinel of the ordered merge: no live cursor is left in the row.
constexpr int32_t kMaxCol = 0x7FFFFFFF;

__aicore__ inline int64_t MinInt64(int64_t left, int64_t right)
{
    return left < right ? left : right;
}

__aicore__ inline uint32_t MinUint32(uint32_t left, uint32_t right)
{
    return left < right ? left : right;
}

// Start of block `index` when [0, total) is cut into `count` ranges. A launched
// kernel always runs at least one block, but GetBlockNum is a runtime value the
// analyzer cannot prove non-zero, so the divisor is guarded here.
__aicore__ inline int64_t Partition(int64_t total, uint32_t index, uint32_t count)
{
    const uint32_t divisor = count == 0 ? 1u : count;
    return static_cast<int64_t>(static_cast<uint64_t>(total) * static_cast<uint64_t>(index) /
                                static_cast<uint64_t>(divisor));
}

__aicore__ inline uint32_t LaunchBlocks()
{
    return AscendC::GetBlockNum() == 0 ? 1u : static_cast<uint32_t>(AscendC::GetBlockNum());
}

// First item of block `index` when `total` items are cut into `count` ranges of
// whole `group`s. Both ends of a range stay on a group boundary, except the last
// range, which stops at `total`.
__aicore__ inline int64_t GroupStart(int64_t total, int64_t group, uint32_t index, uint32_t count)
{
    const int64_t step = group > 0 ? group : 1;
    const int64_t groups = (total + step - 1) / step;
    const int64_t start = Partition(groups, index, count) * step;
    return start < total ? start : total;
}

__aicore__ inline int64_t GroupEnd(int64_t total, int64_t group, uint32_t index, uint32_t count)
{
    return GroupStart(total, group, index + 1, count);
}

// Group-aligned range of this block when ``total`` items are split over the
// launched blocks; every phase that writes whole 32-byte blocks uses it.
struct GroupRange {
    int64_t begin;
    int64_t end;
};

__aicore__ inline GroupRange RowRange(int64_t total)
{
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = LaunchBlocks();
    return {GroupStart(total, kInt32Group, block, blocks),
            GroupEnd(total, kInt32Group, block, blocks)};
}

// The scalar pipe and a copy pipe only see each other's results through an
// explicit flag.
template <AscendC::HardEvent Event>
__aicore__ inline void Sync()
{
    AscendC::SetFlag<Event>(EVENT_ID0);
    AscendC::WaitFlag<Event>(EVENT_ID0);
}

// Elements of one 32-byte block of the scan workspace for a chunk total.
template <typename T>
__aicore__ constexpr int64_t ChunkSumStride()
{
    return sizeof(T) == sizeof(int32_t) ? SPSSPMM_CHUNK_SUM_STRIDE_INT32
                                        : SPSSPMM_CHUNK_SUM_STRIDE_INT64;
}

template <typename T>
__aicore__ inline AscendC::DataCopyExtParams BytesParams(int64_t elements)
{
    return AscendC::DataCopyExtParams{
        1, static_cast<uint32_t>(elements * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
}

template <typename T>
__aicore__ inline AscendC::DataCopyPadExtParams<T> LoadParams()
{
    return AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0};
}

template <typename T>
__aicore__ inline float ToFloat(T value)
{
    return static_cast<float>(value);
}

template <typename T>
__aicore__ inline T FromFloat(float value)
{
    return static_cast<T>(value);
}

// Cursor accessors: a merge runs either on UB-resident cursors (the common case)
// or on the global cursor buffer (rows longer than ``kCursorCap``).
__aicore__ inline int32_t CursorAt(AscendC::LocalTensor<int32_t>& cursors, int64_t index)
{
    return cursors.GetValue(static_cast<uint32_t>(index));
}

__aicore__ inline int32_t CursorAt(AscendC::GlobalTensor<int32_t>& cursors, int64_t index)
{
    return cursors.GetValue(static_cast<uint64_t>(index));
}

__aicore__ inline void SetCursor(AscendC::LocalTensor<int32_t>& cursors, int64_t index, int32_t value)
{
    cursors.SetValue(static_cast<uint32_t>(index), value);
}

__aicore__ inline void SetCursor(AscendC::GlobalTensor<int32_t>& cursors, int64_t index, int32_t value)
{
    cursors.SetValue(static_cast<uint64_t>(index), value);
}

// Raise one word of the sticky status block. The block is 32 bytes and starts at
// word 0 of the status tensor, so the atomic update never straddles a block and
// concurrent AIVs accumulate instead of overwriting each other.
__aicore__ inline void RaiseWord(const AscendC::GlobalTensor<int32_t>& status,
                                 const AscendC::LocalTensor<int32_t>& flag, int32_t word)
{
    for (int64_t i = 0; i < SPSSPMM_ERROR_WORDS; ++i) {
        flag.SetValue(static_cast<uint32_t>(i), i == static_cast<int64_t>(word) ? 1 : 0);
    }
    Sync<AscendC::HardEvent::S_MTE3>();
    AscendC::SetAtomicAdd<int32_t>();
    AscendC::DataCopyPad(status, flag, BytesParams<int32_t>(SPSSPMM_ERROR_WORDS));
    Sync<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
}

__aicore__ inline void RaiseError(const AscendC::GlobalTensor<int32_t>& status,
                                  const AscendC::LocalTensor<int32_t>& flag)
{
    RaiseWord(status, flag, static_cast<int32_t>(SPSSPMM_ERROR_FLAG_WORD));
}

// Store `count` staged candidates at `dst`.
template <typename T>
__aicore__ inline void FlushStage(const AscendC::GlobalTensor<int32_t>& candidateCols,
                                  const AscendC::GlobalTensor<T>& candidateVals, int64_t dst,
                                  uint32_t count, const AscendC::LocalTensor<int32_t>& stagedCols,
                                  const AscendC::LocalTensor<T>& stagedVals)
{
    Sync<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(candidateCols[dst], stagedCols, BytesParams<int32_t>(count));
    AscendC::DataCopyPad(candidateVals[dst], stagedVals, BytesParams<T>(count));
    Sync<AscendC::HardEvent::MTE3_S>();
}

// Store `count` per-row counts at `dst`, keeping the local tile reusable.
__aicore__ inline void FlushCounts(const AscendC::GlobalTensor<int32_t>& uniqueCounts, int64_t dst,
                                   uint32_t count, const AscendC::LocalTensor<int32_t>& tile)
{
    if (count == 0) {
        return;
    }
    Sync<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(uniqueCounts[dst], tile, BytesParams<int32_t>(count));
    Sync<AscendC::HardEvent::MTE3_S>();
}

// One COO side: narrow int64 (row, column) to int32, reject a coordinate outside
// its matrix and a row id that is not ascending, and replace a rejected
// coordinate with (0, 0) so the following passes stay inside their buffers.
// ``begin`` and ``end`` are 32-byte groups of edges, so the two output arrays are
// written by one AIV per block.
__aicore__ inline bool ConvertSide(const AscendC::GlobalTensor<int64_t>& rowIn,
                                   const AscendC::GlobalTensor<int64_t>& colIn,
                                   const AscendC::GlobalTensor<int32_t>& rowOut,
                                   const AscendC::GlobalTensor<int32_t>& colOut, int32_t rowLimit,
                                   int32_t colLimit, int64_t begin, int64_t end,
                                   const AscendC::LocalTensor<int64_t>& rowTile,
                                   const AscendC::LocalTensor<int64_t>& colTile,
                                   const AscendC::LocalTensor<int32_t>& rowOutTile,
                                   const AscendC::LocalTensor<int32_t>& colOutTile)
{
    bool failed = false;
    int64_t previous = begin > 0 ? rowIn.GetValue(static_cast<uint64_t>(begin - 1)) : -1;
    for (int64_t offset = begin; offset < end; offset += kEdgeTile) {
        const uint32_t count = static_cast<uint32_t>(MinInt64(end - offset, kEdgeTile));
        AscendC::DataCopyPad(rowTile, rowIn[offset], BytesParams<int64_t>(count),
                             LoadParams<int64_t>());
        AscendC::DataCopyPad(colTile, colIn[offset], BytesParams<int64_t>(count),
                             LoadParams<int64_t>());
        Sync<AscendC::HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < count; ++i) {
            const int64_t row = rowTile.GetValue(i);
            const int64_t col = colTile.GetValue(i);
            const bool badRow = row < 0 || row >= rowLimit;
            const bool badCol = col < 0 || col >= colLimit;
            failed = failed || badRow || badCol || row < previous;
            previous = row;
            rowOutTile.SetValue(i, badRow ? 0 : static_cast<int32_t>(row));
            colOutTile.SetValue(i, badCol ? 0 : static_cast<int32_t>(col));
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(rowOut[offset], rowOutTile, BytesParams<int32_t>(count));
        AscendC::DataCopyPad(colOut[offset], colOutTile, BytesParams<int32_t>(count));
        Sync<AscendC::HardEvent::MTE3_S>();
        Sync<AscendC::HardEvent::S_MTE2>();
    }
    return failed;
}

// First index of the sorted ``rows`` array whose value is at least ``target``.
__aicore__ inline int64_t LowerBound(const AscendC::GlobalTensor<int32_t>& rows, int64_t count,
                                     int32_t target)
{
    int64_t low = 0;
    int64_t high = count;
    while (low < high) {
        const int64_t mid = low + (high - low) / 2;
        if (rows.GetValue(static_cast<uint64_t>(mid)) < target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Count the edges of the rows owned by one block. The row ids are sorted, so a
// binary search finds the first edge of the block and every row forms one run.
// The block owns whole 32-byte groups of ``counts``.
__aicore__ inline bool CountRows(const AscendC::GlobalTensor<int32_t>& rows,
                                 const AscendC::GlobalTensor<int32_t>& counts, int32_t edges,
                                 int32_t limit, int64_t begin, int64_t end,
                                 const AscendC::LocalTensor<int32_t>& tile)
{
    int64_t cursor = LowerBound(rows, edges, static_cast<int32_t>(begin));
    bool failed = begin == 0 && cursor > 0;
    for (int64_t rowBase = begin; rowBase < end; rowBase += kRowTile) {
        const uint32_t width = static_cast<uint32_t>(MinInt64(end - rowBase, kRowTile));
        for (uint32_t i = 0; i < width; ++i) {
            const int64_t row = rowBase + i;
            int32_t count = 0;
            while (cursor < edges && rows.GetValue(static_cast<uint64_t>(cursor)) == row) {
                ++cursor;
                ++count;
            }
            tile.SetValue(i, count);
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(counts[rowBase], tile, BytesParams<int32_t>(width));
        Sync<AscendC::HardEvent::MTE3_S>();
        Sync<AscendC::HardEvent::S_MTE2>();
    }
    return failed || (end == limit && cursor < edges);
}

// Coalesced input has strictly ascending columns inside every row; duplicates
// would silently double count because the merge assumes one entry per column.
__aicore__ inline bool ValidateRows(const AscendC::GlobalTensor<int32_t>& rowPtr,
                                    const AscendC::GlobalTensor<int32_t>& colInd, int64_t begin,
                                    int64_t end, int32_t nnz)
{
    bool failed = false;
    for (int64_t row = begin; row < end; ++row) {
        const int64_t from = rowPtr.GetValue(static_cast<uint64_t>(row));
        const int64_t to = rowPtr.GetValue(static_cast<uint64_t>(row + 1));
        if (from < 0 || to < from || to > nnz) {
            failed = true;
            continue;
        }
        int32_t previous = -1;
        for (int64_t p = from; p < to; ++p) {
            const int32_t col = colInd.GetValue(static_cast<uint64_t>(p));
            failed = failed || col <= previous;
            previous = col;
        }
    }
    return failed;
}

// Exclusive prefix sum of the block's chunks, written into ``offsets`` and
// closed by the chunk total in ``chunkSums``. Each total occupies one 32-byte
// block of ``chunkSums``, so the AIVs that own neighbouring chunks never write
// into the same block.
template <typename CountT, typename OffsetT>
__aicore__ inline void ScanLocal(const AscendC::GlobalTensor<CountT>& counts,
                                 const AscendC::GlobalTensor<OffsetT>& offsets,
                                 const AscendC::GlobalTensor<OffsetT>& chunkSums, int32_t count,
                                 int32_t numChunks, const AscendC::LocalTensor<CountT>& countTile,
                                 const AscendC::LocalTensor<OffsetT>& offTile)
{
    const uint32_t blocks = LaunchBlocks();
    const int64_t c0 = Partition(numChunks, AscendC::GetBlockIdx(), blocks);
    const int64_t c1 = Partition(numChunks, AscendC::GetBlockIdx() + 1, blocks);
    const int64_t stride = ChunkSumStride<OffsetT>();
    for (int64_t chunk = c0; chunk < c1; ++chunk) {
        const int64_t chunkEnd =
            MinInt64(chunk * SPSSPMM_SCAN_CHUNK + SPSSPMM_SCAN_CHUNK, count);
        OffsetT running = 0;
        for (int64_t offset = chunk * SPSSPMM_SCAN_CHUNK; offset < chunkEnd; offset += kScanTile) {
            const uint32_t width = static_cast<uint32_t>(MinInt64(chunkEnd - offset, kScanTile));
            AscendC::DataCopyPad(countTile, counts[offset], BytesParams<CountT>(width),
                                 LoadParams<CountT>());
            Sync<AscendC::HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < width; ++i) {
                offTile.SetValue(i, running);
                running += static_cast<OffsetT>(countTile.GetValue(i));
            }
            Sync<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(offsets[offset], offTile, BytesParams<OffsetT>(width));
            Sync<AscendC::HardEvent::MTE3_S>();
            Sync<AscendC::HardEvent::S_MTE2>();
        }
        offTile.SetValue(0, running);
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(chunkSums[chunk * stride], offTile, BytesParams<OffsetT>(1));
        Sync<AscendC::HardEvent::MTE3_S>();
        Sync<AscendC::HardEvent::S_MTE2>();
    }
}

// Prefix sum of the chunk totals, plus the closing scan boundary. One block owns
// the whole pass, so the closing write in ``offsets`` has no neighbour.
template <typename OffsetT>
__aicore__ inline void ScanChunks(const AscendC::GlobalTensor<OffsetT>& chunkSums,
                                  const AscendC::GlobalTensor<OffsetT>& chunkPrefix,
                                  const AscendC::GlobalTensor<OffsetT>& offsets, int32_t count,
                                  int32_t numChunks, const AscendC::LocalTensor<OffsetT>& prefixTile)
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    const int64_t stride = ChunkSumStride<OffsetT>();
    OffsetT running = 0;
    for (int32_t base = 0; base < numChunks; base += kScanTile) {
        const uint32_t width =
            MinUint32(static_cast<uint32_t>(kScanTile), static_cast<uint32_t>(numChunks - base));
        for (uint32_t i = 0; i < width; ++i) {
            prefixTile.SetValue(i, running);
            running += chunkSums.GetValue(static_cast<uint64_t>((base + static_cast<int32_t>(i)) *
                                                               stride));
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(chunkPrefix[base], prefixTile, BytesParams<OffsetT>(width));
        Sync<AscendC::HardEvent::MTE3_S>();
        Sync<AscendC::HardEvent::S_MTE2>();
    }
    prefixTile.SetValue(0, running);
    Sync<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(offsets[count], prefixTile, BytesParams<OffsetT>(1));
    Sync<AscendC::HardEvent::MTE3_S>();
}

// Add the per-chunk base to every chunk of the block.
template <typename OffsetT>
__aicore__ inline void ScanAdd(const AscendC::GlobalTensor<OffsetT>& offsets,
                               const AscendC::GlobalTensor<OffsetT>& chunkPrefix, int32_t count,
                               int32_t numChunks, const AscendC::LocalTensor<OffsetT>& offTile)
{
    const uint32_t blocks = LaunchBlocks();
    const int64_t c0 = Partition(numChunks, AscendC::GetBlockIdx(), blocks);
    const int64_t c1 = Partition(numChunks, AscendC::GetBlockIdx() + 1, blocks);
    for (int64_t chunk = c0; chunk < c1; ++chunk) {
        const OffsetT base = chunkPrefix.GetValue(static_cast<uint64_t>(chunk));
        const int64_t chunkEnd =
            MinInt64(chunk * SPSSPMM_SCAN_CHUNK + SPSSPMM_SCAN_CHUNK, count);
        for (int64_t offset = chunk * SPSSPMM_SCAN_CHUNK; offset < chunkEnd; offset += kScanTile) {
            const uint32_t width = static_cast<uint32_t>(MinInt64(chunkEnd - offset, kScanTile));
            AscendC::DataCopyPad(offTile, offsets[offset], BytesParams<OffsetT>(width),
                                 LoadParams<OffsetT>());
            Sync<AscendC::HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < width; ++i) {
                offTile.SetValue(i, offTile.GetValue(i) + base);
            }
            Sync<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(offsets[offset], offTile, BytesParams<OffsetT>(width));
            Sync<AscendC::HardEvent::MTE3_S>();
            Sync<AscendC::HardEvent::S_MTE2>();
        }
    }
}

// Upper bound of the intermediate products of every output row: the row merges
// one sorted B row per A entry, so its candidate slots are the sum of those
// B-row lengths. The real unique count never exceeds this bound. Every row is
// rounded up to ``kSlotAlign`` slots so the candidate region of a row starts on a
// block boundary of both the columns and the values.
__aicore__ inline void ProductCountRows(const AscendC::GlobalTensor<int32_t>& rowPtrA,
                                        const AscendC::GlobalTensor<int32_t>& colIndA,
                                        const AscendC::GlobalTensor<int32_t>& rowPtrB,
                                        const AscendC::GlobalTensor<int64_t>& counts,
                                        int64_t begin, int64_t end,
                                        const AscendC::LocalTensor<int64_t>& tile)
{
    for (int64_t rowBase = begin; rowBase < end; rowBase += kRowTile) {
        const uint32_t width = static_cast<uint32_t>(MinInt64(end - rowBase, kRowTile));
        for (uint32_t i = 0; i < width; ++i) {
            const int64_t row = rowBase + i;
            const int64_t from = rowPtrA.GetValue(static_cast<uint64_t>(row));
            const int64_t to = rowPtrA.GetValue(static_cast<uint64_t>(row) + 1);
            int64_t products = 0;
            for (int64_t p = from; p < to; ++p) {
                const int32_t bRow = colIndA.GetValue(static_cast<uint64_t>(p));
                products += rowPtrB.GetValue(static_cast<uint64_t>(bRow) + 1) -
                            rowPtrB.GetValue(static_cast<uint64_t>(bRow));
            }
            const int64_t remainder = products % kSlotAlign;
            tile.SetValue(i, remainder == 0 ? products : products + kSlotAlign - remainder);
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(counts[rowBase], tile, BytesParams<int64_t>(width));
        Sync<AscendC::HardEvent::MTE3_S>();
        Sync<AscendC::HardEvent::S_MTE2>();
    }
}

}  // namespace spspmm_a2
}  // namespace opsgnn

#include "spspmm_merge.h"

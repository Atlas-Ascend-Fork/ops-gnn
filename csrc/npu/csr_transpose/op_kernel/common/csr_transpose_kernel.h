/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_CSR_TRANSPOSE_KERNEL_H
#define OPS_GNN_CSR_TRANSPOSE_KERNEL_H

#include <acl/acl.h>
#include <cstdint>

#include "kernel_operator.h"

#include "csr_transpose/op_kernel/common/csr_transpose_launch.h"

/**
 * CSR -> CSC structural transpose.
 *
 * The operator turns an (N, M) CSR layout into an (M, N) CSR layout while
 * keeping the edge count: every edge (row, column) becomes (column, row). No
 * value is moved and no duplicate is merged, so only the two index arrays are
 * rewritten.
 *
 * Three passes run in order on one stream:
 *
 *   1. count   -- histogram the column ids, one contiguous column range per AIV;
 *   2. indptr  -- exclusive prefix sum of the histogram;
 *   3. write   -- replay the edges in input order and append each row id at the
 *                 cursor of its column.
 *
 * The write pass derives the row id of an edge from `indptr` with a running row
 * cursor while it scans, so the call needs no workspace of the size of the edge
 * array and no separate expansion pass.
 *
 * Every AIV owns a disjoint column range in all passes, so no cross-core atomic
 * is needed: the order in which an AIV visits its columns is fixed, and within a
 * column the edges are visited in input order, which keeps the row ids of every
 * column ascending. The result therefore does not depend on the number of AIVs or
 * on the scheduler.
 *
 * The price of that partition is that each AIV has to look at every edge to
 * find the ones that belong to its columns: the scan is O(edges) per AIV, so the
 * wall time grows with the edge count but not with the AIV count. The
 * alternative - a histogram per AIV in global memory - would remove the
 * amplification at the cost of a `cores * columns` workspace, which the
 * small-index operators of this library deliberately avoid.
 *
 * Three hardware and alignment details shape the code below:
 *
 *   - DataCopyPad moves whole 32-byte blocks at the boundaries of a range, so the
 *     start of a range is kept on a `kAlignment` multiple; only the length may be
 *     unaligned.
 *   - a scalar store into global memory rewrites the 32-byte block that holds the
 *     value, so two AIVs that store into one block lose one of the values. The
 *     write pass therefore moves the end of a column range forward to the next
 *     column whose output position starts a block, which gives every AIV whole
 *     blocks of the output.
 *   - the pipe queues only see each other's results through an explicit flag, so
 *     every transition between the scalar pipe and a copy pipe is a SetFlag /
 *     WaitFlag pair.
 */

namespace opsgnn {

namespace csr_transpose_detail {

// 4096 int64 elements per UB tile (32 KiB). Tiles and cursor windows start at a
// multiple of kAlignment elements, so every DataCopyPad source and destination
// stays 32-byte aligned.
constexpr uint32_t kTileElements = 4096;
constexpr uint32_t kAlignment = static_cast<uint32_t>(kTransposeAlignment);

// Pipe bookkeeping and alignment head room on top of the data tiles.
constexpr uint32_t kUbReservedBytes = 32768;

// The scalar pipe and the copy pipes only see each other's results through an
// explicit flag; a plain PipeBarrier is not enough for these transitions.
template <AscendC::HardEvent Event>
__aicore__ inline void Sync()
{
    AscendC::SetFlag<Event>(EVENT_ID0);
    AscendC::WaitFlag<Event>(EVENT_ID0);
}

// Round an element offset up to the next 32-byte boundary.
__aicore__ inline int64_t AlignUp(int64_t value)
{
    return (value + static_cast<int64_t>(kAlignment) - 1) / static_cast<int64_t>(kAlignment) *
           static_cast<int64_t>(kAlignment);
}

// Int64 elements of UB that stay usable after `tileCount` tiles and the pipe head
// room, in whole 32-byte blocks.
__aicore__ inline uint32_t LocalCapacity(uint32_t ubBytes, uint32_t tileCount)
{
    const uint64_t reserved = static_cast<uint64_t>(kUbReservedBytes) +
                              static_cast<uint64_t>(tileCount) * kTileElements * sizeof(int64_t);
    if (static_cast<uint64_t>(ubBytes) <= reserved) {
        return kAlignment;
    }
    uint32_t elements = static_cast<uint32_t>((static_cast<uint64_t>(ubBytes) - reserved) / sizeof(int64_t));
    elements -= elements % kAlignment;
    return elements > kAlignment ? elements : kAlignment;
}

// Elements of the current tile: the tile capacity, or what is left of the range.
__aicore__ inline uint32_t TileCount(int64_t remaining)
{
    return static_cast<uint32_t>(remaining < static_cast<int64_t>(kTileElements)
                                     ? remaining
                                     : static_cast<int64_t>(kTileElements));
}

// Elements of the current window: the window capacity, or what is left of the range.
__aicore__ inline uint32_t WindowWidth(int64_t remaining, uint32_t capacity)
{
    return static_cast<uint32_t>(remaining < static_cast<int64_t>(capacity) ? remaining
                                                                            : static_cast<int64_t>(capacity));
}

__aicore__ inline AscendC::DataCopyExtParams BytesParams(uint32_t elements)
{
    return AscendC::DataCopyExtParams{1, elements * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
}

__aicore__ inline AscendC::DataCopyPadExtParams<int64_t> LoadParams()
{
    return AscendC::DataCopyPadExtParams<int64_t>{false, 0, 0, 0};
}

// Slot of one AIV's range total in `blockTotals`: one 32-byte block per AIV, so
// two totals never share the block a scalar store rewrites.
__aicore__ inline uint64_t TotalSlot(uint32_t block)
{
    return static_cast<uint64_t>(block) * static_cast<uint64_t>(kTransposeBlockStride);
}

// Start of the range of AIV `index` when [0, total) is cut into `count` ranges,
// i.e. `total * index / count`. A launched kernel always runs at least one AIV,
// but GetBlockNum is a runtime value the analyzer cannot prove non-zero
// (codecheck G.EXP.22), so the divisor is guarded in this one place.
__aicore__ inline int64_t SplitBoundary(int64_t total, uint32_t index, uint32_t count)
{
    const uint32_t divisor = count == 0 ? 1u : count;
    return static_cast<int64_t>(static_cast<uint64_t>(total) * static_cast<uint64_t>(index) / divisor);
}

// Columns owned by one AIV in the counting passes: [begin, end). The ranges tile
// [0, numColumns) without overlap and every start is aligned; only the very last
// end, numColumns, may be unaligned.
__aicore__ inline void ColumnRange(uint32_t block, uint32_t blocks, int64_t numColumns, int64_t& begin, int64_t& end)
{
    begin = AlignUp(SplitBoundary(numColumns, block, blocks));
    end = AlignUp(SplitBoundary(numColumns, block + 1, blocks));
    begin = begin < numColumns ? begin : numColumns;
    end = end < numColumns ? end : numColumns;
}

// First column in [target, numColumns) whose output position starts a 32-byte
// block, or numColumns when there is none. Pointers are non-decreasing, so an
// unaligned value is skipped by jumping to the next column that has a different
// pointer, which keeps long runs of empty columns cheap.
__aicore__ inline int64_t BlockAlignedColumn(const AscendC::GlobalTensor<int64_t>& outIndptr, int64_t target,
                                            int64_t numColumns)
{
    if (target >= numColumns) {
        return numColumns;
    }
    int64_t value = outIndptr.GetValue(static_cast<uint64_t>(target));
    while ((value & static_cast<int64_t>(kAlignment - 1)) != 0) {
        int64_t low = target + 1;
        int64_t high = numColumns;
        while (low < high) {
            const int64_t mid = low + (high - low) / 2;
            if (outIndptr.GetValue(static_cast<uint64_t>(mid)) <= value) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if (low >= numColumns) {
            return numColumns;
        }
        target = low;
        value = outIndptr.GetValue(static_cast<uint64_t>(target));
    }
    return target;
}

// Count the columns of [window, window + width) into `cursor` by walking every
// edge tile once, and return the total of the window. The tile is reloaded per
// window, which is what keeps the UB budget at one tile plus one cursor.
__aicore__ inline int64_t CountWindow(AscendC::GlobalTensor<int64_t> indexGm, int64_t numEdges, int64_t window,
                                      uint32_t width, AscendC::LocalTensor<int64_t> cursor,
                                      AscendC::LocalTensor<int64_t> tile)
{
    for (uint32_t i = 0; i < width; ++i) {
        cursor.SetValue(i, 0);
    }
    Sync<AscendC::HardEvent::S_MTE2>();
    for (int64_t offset = 0; offset < numEdges; offset += kTileElements) {
        const uint32_t count = TileCount(numEdges - offset);
        AscendC::DataCopyPad(tile, indexGm[offset], BytesParams(count), LoadParams());
        Sync<AscendC::HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < count; ++i) {
            const int64_t column = tile.GetValue(i);
            if (column >= window && column < window + static_cast<int64_t>(width)) {
                const uint32_t slot = static_cast<uint32_t>(column - window);
                cursor.SetValue(slot, cursor.GetValue(slot) + 1);
            }
        }
        Sync<AscendC::HardEvent::S_MTE2>();
    }
    int64_t total = 0;
    for (uint32_t i = 0; i < width; ++i) {
        total += cursor.GetValue(i);
    }
    return total;
}

}  // namespace csr_transpose_detail

// Pass 1: histogram of the column ids. Each AIV walks the edges once per column
// window, counts only the columns of that window, publishes the window as
// `counts` and keeps the total of its whole range for `blockTotals`.
extern "C" __global__ __aicore__ void csr_transpose_count(
    __gm__ int64_t* indices, __gm__ int64_t* counts, __gm__ int64_t* blockTotals,
    int64_t numEdges, int64_t numColumns, uint32_t ubBytes)
{
    using namespace csr_transpose_detail;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // The row of one 32-byte block at the base of UB carries the range total: a
    // scalar global store loses values on this hardware, so the total leaves as an
    // 8-byte DataCopyPad from a block-aligned source like every other write. The
    // packed head room of LocalCapacity covers the extra block.
    const uint32_t capacity = LocalCapacity(ubBytes, 1);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, kAlignment * sizeof(int64_t) + capacity * sizeof(int64_t) +
                                 kTileElements * sizeof(int64_t));
    AscendC::LocalTensor<int64_t> totalTile = buffer.Get<int64_t>();
    AscendC::LocalTensor<int64_t> cursor = totalTile[kAlignment];
    AscendC::LocalTensor<int64_t> tile = cursor[capacity];

    AscendC::GlobalTensor<int64_t> indexGm;
    AscendC::GlobalTensor<int64_t> countGm;
    AscendC::GlobalTensor<int64_t> totalGm;
    const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blocks = static_cast<uint32_t>(AscendC::GetBlockNum());
    indexGm.SetGlobalBuffer(indices, numEdges);
    countGm.SetGlobalBuffer(counts, numColumns);
    totalGm.SetGlobalBuffer(blockTotals,
                            static_cast<uint64_t>(blocks) * static_cast<uint64_t>(kTransposeBlockStride));

    int64_t begin = 0;
    int64_t end = 0;
    ColumnRange(block, blocks, numColumns, begin, end);

    int64_t total = 0;
    for (int64_t window = begin; window < end; window += capacity) {
        const uint32_t width = WindowWidth(end - window, capacity);
        total += CountWindow(indexGm, numEdges, window, width, cursor, tile);
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(countGm[window], cursor, BytesParams(width));
        Sync<AscendC::HardEvent::MTE3_S>();
    }
    // One total per 32-byte block: the AIVs publish their totals concurrently and
    // each store only owns the block it lands in.
    totalTile.SetValue(0, total);
    Sync<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(totalGm[TotalSlot(block)], totalTile, BytesParams(1));
}

// Pass 2: exclusive prefix sum of the histogram, written as `outIndptr`. Each AIV
// starts from the sum of the range totals of the AIVs before it, so no AIV reads a
// pointer written by another one. The range that reaches numColumns also emits the
// closing boundary, and it always holds at least one column: the host caps the AIV
// count by `numColumns / kAlignment`, and neighbouring column ranges are at least
// `kAlignment` apart.
extern "C" __global__ __aicore__ void csr_transpose_indptr(
    __gm__ int64_t* counts, __gm__ int64_t* blockTotals, __gm__ int64_t* outIndptr, int64_t numColumns)
{
    using namespace csr_transpose_detail;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, (2 * kTileElements + kAlignment) * sizeof(int64_t));
    AscendC::LocalTensor<int64_t> countTile = buffer.Get<int64_t>();
    AscendC::LocalTensor<int64_t> ptrTile = countTile[kTileElements];

    AscendC::GlobalTensor<int64_t> countGm;
    AscendC::GlobalTensor<int64_t> totalGm;
    AscendC::GlobalTensor<int64_t> outGm;
    const uint32_t blocks = static_cast<uint32_t>(AscendC::GetBlockNum());
    countGm.SetGlobalBuffer(counts, numColumns);
    totalGm.SetGlobalBuffer(blockTotals,
                            static_cast<uint64_t>(blocks) * static_cast<uint64_t>(kTransposeBlockStride));
    outGm.SetGlobalBuffer(outIndptr, numColumns + 1);

    const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
    int64_t begin = 0;
    int64_t end = 0;
    ColumnRange(block, blocks, numColumns, begin, end);
    if (begin >= end) {
        return;
    }

    int64_t running = 0;
    for (uint32_t previous = 0; previous < block; ++previous) {
        running += totalGm.GetValue(TotalSlot(previous));
    }

    const bool closingRange = end == numColumns;
    for (int64_t window = begin; window < end; window += kTileElements) {
        const uint32_t width = TileCount(end - window);
        AscendC::DataCopyPad(countTile, countGm[window], BytesParams(width), LoadParams());
        Sync<AscendC::HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < width; ++i) {
            ptrTile.SetValue(i, running);
            running += countTile.GetValue(i);
        }
        uint32_t writeCount = width;
        if (closingRange && window + static_cast<int64_t>(width) == end) {
            ptrTile.SetValue(width, running);
            ++writeCount;
        }
        Sync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(outGm[window], ptrTile, BytesParams(writeCount));
        Sync<AscendC::HardEvent::MTE3_S>();
        Sync<AscendC::HardEvent::MTE3_MTE2>();
    }
}

// Pass 3: replay the edges in input order into the output layout. Each AIV owns a
// contiguous column range whose ends start a 32-byte block of the output, and it
// sorts the edges of that range into UB before writing them back as whole tiles:
//
//   - `cursor` keeps the output position of every column of the current window;
//   - `cache`  holds the rows of the positions of the current position chunk.
//
// The positions of one column range are contiguous, so a chunk is a contiguous
// run of `outIndices` that can be stored with one DataCopyPad. A chunk is a
// multiple of `kAlignment` long and starts at a block boundary, which keeps every
// AIV inside whole blocks of the output: a store only owns the block it lands in,
// so a block shared by two AIVs would lose values.
//
// A range whose positions do not fit one chunk is rescanned once per chunk; a
// column wider than the chunk thus runs through the same code as a narrow one.
// The row of an edge comes from a cursor over `indptr` that advances while the
// edges are replayed, so the pass needs no expansion of the row array and only
// one edge tile in UB. The row cache owns the base of the UB block because an
// MTE store wants its source at the start of the buffer; the cursors and the
// edge tile follow it.
namespace csr_transpose_detail {

// State of one AIV in the write pass: the global views, the UB tiles and the two
// capacities that came from the UB budget. Keeping them in one struct keeps the
// helpers of the pass short.
struct WriteContext {
    AscendC::GlobalTensor<int64_t> indptr;
    AscendC::GlobalTensor<int64_t> indices;
    AscendC::GlobalTensor<int64_t> outPtr;
    AscendC::GlobalTensor<int64_t> outIndices;
    AscendC::LocalTensor<int64_t> cache;
    AscendC::LocalTensor<int64_t> cursor;
    AscendC::LocalTensor<int64_t> indexTile;
    int64_t numRows;
    int64_t numEdges;
    uint32_t capacityColumns;
    uint32_t capacityPositions;
};

// Seed `cursor` with the output position of every column of the window.
__aicore__ inline void LoadPositionCursor(WriteContext& context, int64_t window, uint32_t width)
{
    for (uint32_t i = 0; i < width; ++i) {
        context.cursor.SetValue(i, context.outPtr.GetValue(static_cast<uint64_t>(window + static_cast<int64_t>(i))));
    }
}

// Append `row` at the cursor of column `slot`, and keep it when the claimed
// position belongs to the chunk that is being filled.
__aicore__ inline void PlaceRow(WriteContext& context, uint32_t slot, int64_t chunkBase, uint32_t limit,
                                int64_t row)
{
    const int64_t position = context.cursor.GetValue(slot);
    context.cursor.SetValue(slot, position + 1);
    if (position >= chunkBase && position < chunkBase + static_cast<int64_t>(limit)) {
        context.cache.SetValue(static_cast<uint32_t>(position - chunkBase), row);
    }
}

// Replay the edges in input order and keep the rows whose output position falls in
// [chunkBase, chunkBase + limit). The row of an edge comes from a cursor over
// `indptr` that advances while the edges are scanned, so no row array is built.
__aicore__ inline void ReplayChunk(WriteContext& context, int64_t window, uint32_t width, int64_t chunkBase,
                                   uint32_t limit)
{
    int64_t currentRow = 0;
    int64_t boundary = context.indptr.GetValue(1);
    Sync<AscendC::HardEvent::S_MTE2>();
    for (int64_t offset = 0; offset < context.numEdges; offset += kTileElements) {
        const uint32_t count = TileCount(context.numEdges - offset);
        AscendC::DataCopyPad(context.indexTile, context.indices[offset], BytesParams(count), LoadParams());
        Sync<AscendC::HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < count; ++i) {
            const int64_t edge = offset + static_cast<int64_t>(i);
            while (currentRow + 1 < context.numRows && edge >= boundary) {
                ++currentRow;
                boundary = context.indptr.GetValue(static_cast<uint64_t>(currentRow + 1));
            }
            const int64_t column = context.indexTile.GetValue(i);
            if (column >= window && column < window + static_cast<int64_t>(width)) {
                PlaceRow(context, static_cast<uint32_t>(column - window), chunkBase, limit, currentRow);
            }
        }
        Sync<AscendC::HardEvent::S_MTE2>();
    }
}

// Fill and store one position chunk of the window: the rows are collected in
// `cache` and leave as a single DataCopyPad of whole blocks.
__aicore__ inline void WriteChunk(WriteContext& context, int64_t window, uint32_t width, int64_t chunkBase,
                                  uint32_t limit)
{
    LoadPositionCursor(context, window, width);
    ReplayChunk(context, window, width, chunkBase, limit);
    Sync<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(context.outIndices[static_cast<uint64_t>(chunkBase)], context.cache, BytesParams(limit));
    Sync<AscendC::HardEvent::MTE3_S>();
}

// All chunks of one column window. A window whose columns hold no edge is skipped
// without scanning the edges.
__aicore__ inline void WriteWindow(WriteContext& context, int64_t window, int64_t end)
{
    const uint32_t width = WindowWidth(end - window, context.capacityColumns);
    const int64_t base = context.outPtr.GetValue(static_cast<uint64_t>(window));
    const int64_t span = context.outPtr.GetValue(static_cast<uint64_t>(window + static_cast<int64_t>(width))) - base;
    if (span <= 0) {
        return;
    }
    const int64_t step = static_cast<int64_t>(context.capacityPositions);
    for (int64_t chunkBase = base; chunkBase < base + span; chunkBase += step) {
        const uint32_t limit = WindowWidth(base + span - chunkBase, context.capacityPositions);
        WriteChunk(context, window, width, chunkBase, limit);
    }
}

}  // namespace csr_transpose_detail

extern "C" __global__ __aicore__ void csr_transpose_write(
    __gm__ int64_t* indptr, __gm__ int64_t* indices, __gm__ int64_t* outIndptr, __gm__ int64_t* outIndices,
    int64_t numRows, int64_t numEdges, int64_t numColumns, uint32_t ubBytes)
{
    using namespace csr_transpose_detail;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // The UB block is split three ways: rows, column cursors and the edge tile
    // that LocalCapacity already reserved.
    const uint32_t usable = LocalCapacity(ubBytes, 1);
    WriteContext context;
    context.capacityColumns =
        usable > 2 * kAlignment ? usable / 4 / kAlignment * kAlignment : kAlignment;
    context.capacityPositions =
        usable > context.capacityColumns ? usable - context.capacityColumns : kAlignment;
    context.numRows = numRows;
    context.numEdges = numEdges;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, usable * sizeof(int64_t) + kTileElements * sizeof(int64_t));
    context.cache = buffer.Get<int64_t>();
    context.cursor = context.cache[context.capacityPositions];
    context.indexTile = context.cursor[context.capacityColumns];

    context.indptr.SetGlobalBuffer(indptr, numRows + 1);
    context.indices.SetGlobalBuffer(indices, numEdges);
    context.outPtr.SetGlobalBuffer(outIndptr, numColumns + 1);
    context.outIndices.SetGlobalBuffer(outIndices, numEdges);

    const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blocks = static_cast<uint32_t>(AscendC::GetBlockNum());
    const int64_t begin = BlockAlignedColumn(context.outPtr, SplitBoundary(numColumns, block, blocks), numColumns);
    const int64_t end = BlockAlignedColumn(context.outPtr, SplitBoundary(numColumns, block + 1, blocks), numColumns);
    if (begin >= end) {
        return;
    }
    for (int64_t window = begin; window < end; window += static_cast<int64_t>(context.capacityColumns)) {
        WriteWindow(context, window, end);
    }
}

void CsrTransposeCount(const int64_t* indices, int64_t* counts, int64_t* blockTotals, int64_t numEdges,
                       int64_t numColumns, uint32_t blockNum, uint32_t ubBytes, aclrtStream stream)
{
    csr_transpose_count<<<blockNum, nullptr, stream>>>(const_cast<int64_t*>(indices), counts, blockTotals,
                                                       numEdges, numColumns, ubBytes);
}

void CsrTransposeIndptr(const int64_t* counts, const int64_t* blockTotals, int64_t* outIndptr,
                        int64_t numColumns, uint32_t blockNum, aclrtStream stream)
{
    csr_transpose_indptr<<<blockNum, nullptr, stream>>>(const_cast<int64_t*>(counts),
                                                        const_cast<int64_t*>(blockTotals), outIndptr, numColumns);
}

void CsrTransposeWrite(const int64_t* indptr, const int64_t* indices, const int64_t* outIndptr,
                       int64_t* outIndices, int64_t numRows, int64_t numEdges, int64_t numColumns,
                       uint32_t blockNum, uint32_t ubBytes, aclrtStream stream)
{
    csr_transpose_write<<<blockNum, nullptr, stream>>>(const_cast<int64_t*>(indptr),
                                                       const_cast<int64_t*>(indices),
                                                       const_cast<int64_t*>(outIndptr), outIndices, numRows,
                                                       numEdges, numColumns, ubBytes);
}

}  // namespace opsgnn

#endif  // OPS_GNN_CSR_TRANSPOSE_KERNEL_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Ordered merge and COO compaction of the arch22 spspmm kernels. The shared
// helpers (UB tiles, cursor access, 32-byte grouping and bounds) live in
// ``spspmm_kernel.h``, which includes this header after them; keeping the two
// device headers separate holds each file below the 500-line file limit.
#pragma once
#include "../common/spspmm_common.h"

namespace opsgnn {
namespace spspmm_a2 {

// Smallest column still reachable through the live cursors of one output row,
// or ``kMaxCol`` when every cursor has reached the end of its B row.
template <bool Cached, typename CursorT>
__aicore__ inline int32_t FindMinColumn(const AscendC::GlobalTensor<int32_t>& rowPtrB,
                                        const AscendC::GlobalTensor<int32_t>& colIndB,
                                        const AscendC::GlobalTensor<int32_t>& colIndA,
                                        const AscendC::LocalTensor<int32_t>& aCols,
                                        CursorT& cursors, int64_t cursorBase, int64_t begin,
                                        int64_t count)
{
    int32_t minCol = kMaxCol;
    for (int64_t i = 0; i < count; ++i) {
        const int32_t bRow = Cached ? aCols.GetValue(static_cast<uint32_t>(i))
                                    : colIndA.GetValue(static_cast<uint64_t>(begin + i));
        const int32_t cursor = CursorAt(cursors, cursorBase + i);
        if (cursor >= rowPtrB.GetValue(static_cast<uint64_t>(bRow) + 1)) {
            continue;
        }
        const int32_t col = colIndB.GetValue(static_cast<uint64_t>(cursor));
        minCol = col < minCol ? col : minCol;
    }
    return minCol;
}

// Sum every product of one output row that lands on ``minCol`` and advance the
// cursors past the consumed entries.
template <bool Cached, typename T, typename CursorT>
__aicore__ inline float AccumulateColumn(const AscendC::GlobalTensor<int32_t>& rowPtrB,
                                         const AscendC::GlobalTensor<int32_t>& colIndB,
                                         const AscendC::GlobalTensor<int32_t>& colIndA,
                                         const AscendC::GlobalTensor<T>& valA,
                                         const AscendC::GlobalTensor<T>& valB,
                                         const AscendC::LocalTensor<int32_t>& aCols, CursorT& cursors,
                                         int64_t cursorBase, int64_t begin, int64_t count,
                                         int32_t minCol)
{
    float sum = 0.0F;
    for (int64_t i = 0; i < count; ++i) {
        const int64_t position = begin + i;
        const int32_t bRow = Cached ? aCols.GetValue(static_cast<uint32_t>(i))
                                    : colIndA.GetValue(static_cast<uint64_t>(position));
        const int32_t bEnd = rowPtrB.GetValue(static_cast<uint64_t>(bRow) + 1);
        int32_t cursor = CursorAt(cursors, cursorBase + i);
        if (cursor >= bEnd || colIndB.GetValue(static_cast<uint64_t>(cursor)) != minCol) {
            continue;
        }
        const float lhs = ToFloat(valA.GetValue(static_cast<uint64_t>(position)));
        while (cursor < bEnd && colIndB.GetValue(static_cast<uint64_t>(cursor)) == minCol) {
            sum += lhs * ToFloat(valB.GetValue(static_cast<uint64_t>(cursor)));
            ++cursor;
        }
        SetCursor(cursors, cursorBase + i, cursor);
    }
    return sum;
}

// Ordered merge of one output row. ``Cached`` selects the UB-resident A-row
// cache; the global-cursor fallback reads the A entries straight from memory.
// The row owns its aligned candidate region, so its staged writes never meet
// another AIV.
template <bool Cached, typename T, typename CursorT>
__aicore__ inline int32_t MergeRow(const AscendC::GlobalTensor<int32_t>& rowPtrB,
                                   const AscendC::GlobalTensor<int32_t>& colIndB,
                                   const AscendC::GlobalTensor<int32_t>& colIndA,
                                   const AscendC::GlobalTensor<T>& valA,
                                   const AscendC::GlobalTensor<T>& valB, int64_t begin, int64_t end,
                                   int64_t cursorBase, CursorT& cursors,
                                   const AscendC::LocalTensor<int32_t>& aCols,
                                   const AscendC::LocalTensor<int32_t>& stagedCols,
                                   const AscendC::LocalTensor<T>& stagedVals, int64_t outBase,
                                   int64_t limit, const AscendC::GlobalTensor<int32_t>& candidateCols,
                                   const AscendC::GlobalTensor<T>& candidateVals, bool& failed)
{
    const int64_t count = end - begin;
    // Both variants start every cursor at the first entry of its B row; the
    // cached one keeps the B row ids in UB so the merge reads them once.
    for (int64_t i = 0; i < count; ++i) {
        const int32_t bRow = colIndA.GetValue(static_cast<uint64_t>(begin + i));
        if constexpr (Cached) {
            aCols.SetValue(static_cast<uint32_t>(i), bRow);
        }
        SetCursor(cursors, cursorBase + i, rowPtrB.GetValue(static_cast<uint64_t>(bRow)));
    }
    int32_t produced = 0;
    uint32_t staged = 0;
    int64_t flushed = 0;
    for (;;) {
        const int32_t minCol =
            FindMinColumn<Cached>(rowPtrB, colIndB, colIndA, aCols, cursors, cursorBase, begin, count);
        if (minCol == kMaxCol) {
            break;
        }
        const float sum = AccumulateColumn<Cached, T>(
            rowPtrB, colIndB, colIndA, valA, valB, aCols, cursors, cursorBase, begin, count, minCol);
        if (static_cast<int64_t>(produced) >= limit) {
            failed = true;
            break;
        }
        stagedCols.SetValue(staged, minCol);
        stagedVals.SetValue(staged, FromFloat<T>(sum));
        ++staged;
        ++produced;
        if (staged < static_cast<uint32_t>(kStageCap)) {
            continue;
        }
        FlushStage(candidateCols, candidateVals, outBase + flushed, staged, stagedCols, stagedVals);
        flushed += static_cast<int64_t>(staged);
        staged = 0;
    }
    if (staged > 0) {
        FlushStage(candidateCols, candidateVals, outBase + flushed, staged, stagedCols, stagedVals);
    }
    return produced;
}

// Merge the rows [begin, end) with UB-resident cursors. The block owns whole
// 32-byte groups of ``uniqueCounts`` and reports them in group-sized writes. A
// row above ``kCursorCap`` is skipped here: its count stays zero and the spill
// pass rewrites it after this kernel, so the two writes never overlap in time.
template <typename T>
__aicore__ inline bool ComputeRows(const AscendC::GlobalTensor<int32_t>& rowPtrA,
                                   const AscendC::GlobalTensor<int32_t>& colIndA,
                                   const AscendC::GlobalTensor<T>& valA,
                                   const AscendC::GlobalTensor<int32_t>& rowPtrB,
                                   const AscendC::GlobalTensor<int32_t>& colIndB,
                                   const AscendC::GlobalTensor<T>& valB,
                                   const AscendC::GlobalTensor<int64_t>& productOffsets,
                                   const AscendC::GlobalTensor<int32_t>& candidateCols,
                                   const AscendC::GlobalTensor<T>& candidateVals,
                                   const AscendC::GlobalTensor<int32_t>& uniqueCounts, int64_t begin,
                                   int64_t end, AscendC::LocalTensor<int32_t>& cursors,
                                   const AscendC::LocalTensor<int32_t>& aCols,
                                   const AscendC::LocalTensor<int32_t>& stagedCols,
                                   const AscendC::LocalTensor<T>& stagedVals,
                                   const AscendC::GlobalTensor<int32_t>& status,
                                   const AscendC::LocalTensor<int32_t>& flag,
                                   const AscendC::LocalTensor<int32_t>& countTile)
{
    bool failed = false;
    int64_t groupStart = begin;
    uint32_t inGroup = 0;
    for (int64_t row = begin; row < end; ++row) {
        const int64_t from = rowPtrA.GetValue(static_cast<uint64_t>(row));
        const int64_t to = rowPtrA.GetValue(static_cast<uint64_t>(row) + 1);
        if (to - from > kCursorCap) {
            countTile.SetValue(inGroup, 0);
            RaiseWord(status, flag, static_cast<int32_t>(SPSSPMM_LONG_ROW_WORD));
        } else {
            const int64_t outBase = productOffsets.GetValue(static_cast<uint64_t>(row));
            const int64_t limit = productOffsets.GetValue(static_cast<uint64_t>(row) + 1) - outBase;
            const int32_t produced =
                MergeRow<true, T>(rowPtrB, colIndB, colIndA, valA, valB, from, to, 0, cursors, aCols,
                                  stagedCols, stagedVals, outBase, limit, candidateCols,
                                  candidateVals, failed);
            countTile.SetValue(inGroup, produced);
        }
        ++inGroup;
        if (inGroup < static_cast<uint32_t>(kInt32Group)) {
            continue;
        }
        FlushCounts(uniqueCounts, groupStart, inGroup, countTile);
        groupStart = row + 1;
        inGroup = 0;
    }
    FlushCounts(uniqueCounts, groupStart, inGroup, countTile);
    return failed;
}

// Rows longer than ``kCursorCap`` are merged by this pass on block 0 with the
// global cursor buffer. No other AIV touches that buffer or those rows, so its
// scalar stores are safe, and it runs after the group writes of ``ComputeRows``.
template <typename T>
__aicore__ inline bool SpillRows(const AscendC::GlobalTensor<int32_t>& rowPtrA,
                                 const AscendC::GlobalTensor<int32_t>& colIndA,
                                 const AscendC::GlobalTensor<T>& valA,
                                 const AscendC::GlobalTensor<int32_t>& rowPtrB,
                                 const AscendC::GlobalTensor<int32_t>& colIndB,
                                 const AscendC::GlobalTensor<T>& valB,
                                 const AscendC::GlobalTensor<int64_t>& productOffsets,
                                 const AscendC::GlobalTensor<int32_t>& candidateCols,
                                 const AscendC::GlobalTensor<T>& candidateVals,
                                 const AscendC::GlobalTensor<int32_t>& uniqueCounts,
                                 AscendC::GlobalTensor<int32_t>& cursors, int64_t rows,
                                 const AscendC::LocalTensor<int32_t>& aCols,
                                 const AscendC::LocalTensor<int32_t>& stagedCols,
                                 const AscendC::LocalTensor<T>& stagedVals,
                                 const AscendC::LocalTensor<int32_t>& countTile)
{
    bool failed = false;
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t from = rowPtrA.GetValue(static_cast<uint64_t>(row));
        const int64_t to = rowPtrA.GetValue(static_cast<uint64_t>(row) + 1);
        if (to - from <= kCursorCap) {
            continue;
        }
        const int64_t outBase = productOffsets.GetValue(static_cast<uint64_t>(row));
        const int64_t limit = productOffsets.GetValue(static_cast<uint64_t>(row) + 1) - outBase;
        const int32_t produced =
            MergeRow<false, T>(rowPtrB, colIndB, colIndA, valA, valB, from, to, from, cursors, aCols,
                               stagedCols, stagedVals, outBase, limit, candidateCols, candidateVals,
                               failed);
        countTile.SetValue(0, produced);
        FlushCounts(uniqueCounts, row, 1, countTile);
    }
    return failed;
}

// First row whose output ends after ``position``; ``rowPtr`` is the exclusive
// prefix sum of the unique counts, so it is non-decreasing.
__aicore__ inline int64_t FindRow(const AscendC::GlobalTensor<int64_t>& rowPtr, int64_t rows,
                                  int64_t position)
{
    int64_t low = 0;
    int64_t high = rows;
    while (low < high) {
        const int64_t mid = low + (high - low) / 2;
        if (rowPtr.GetValue(static_cast<uint64_t>(mid) + 1) <= position) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Compact the per-row candidates into the exact COO output. The rows are already
// column-ascending, so the emitted order is row-major. The pass parallelises on
// the output elements instead of the rows: each AIV owns whole 32-byte groups of
// the three output arrays, and walks the rows that intersect its range.
template <typename T>
__aicore__ inline void CopyRange(const AscendC::GlobalTensor<int64_t>& productOffsets,
                                 const AscendC::GlobalTensor<int32_t>& candidateCols,
                                 const AscendC::GlobalTensor<T>& candidateVals,
                                 const AscendC::GlobalTensor<int64_t>& rowPtrC,
                                 const AscendC::GlobalTensor<int64_t>& indexC,
                                 const AscendC::GlobalTensor<T>& valueC, int64_t total, int64_t rows,
                                 int64_t begin, int64_t end,
                                 const AscendC::LocalTensor<int32_t>& colTile,
                                 const AscendC::LocalTensor<T>& valTile,
                                 const AscendC::LocalTensor<int64_t>& rowTile,
                                 const AscendC::LocalTensor<int64_t>& colOutTile)
{
    if (begin >= end) {
        return;
    }
    int64_t row = FindRow(rowPtrC, rows, begin);
    for (int64_t base = begin; base < end; base += kCopyTile) {
        const uint32_t width = static_cast<uint32_t>(MinInt64(end - base, kCopyTile));
        int64_t position = base;
        uint32_t copied = 0;
        while (copied < width && row < rows) {
            const int64_t rowBegin = rowPtrC.GetValue(static_cast<uint64_t>(row));
            const int64_t rowEnd = rowPtrC.GetValue(static_cast<uint64_t>(row) + 1);
            if (rowEnd <= position) {
                ++row;
                continue;
            }
            const uint32_t count = static_cast<uint32_t>(
                MinInt64(rowEnd - position, static_cast<int64_t>(width - copied)));
            const int64_t src = productOffsets.GetValue(static_cast<uint64_t>(row)) +
                                (position - rowBegin);
            AscendC::DataCopyPad(colTile, candidateCols[src], BytesParams<int32_t>(count),
                                 LoadParams<int32_t>());
            AscendC::DataCopyPad(valTile, candidateVals[src], BytesParams<T>(count),
                                 LoadParams<T>());
            Sync<AscendC::HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < count; ++i) {
                rowTile.SetValue(i, row);
                colOutTile.SetValue(i, static_cast<int64_t>(colTile.GetValue(i)));
            }
            Sync<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(indexC[position], rowTile, BytesParams<int64_t>(count));
            AscendC::DataCopyPad(indexC[total + position], colOutTile, BytesParams<int64_t>(count));
            AscendC::DataCopyPad(valueC[position], valTile, BytesParams<T>(count));
            Sync<AscendC::HardEvent::MTE3_S>();
            Sync<AscendC::HardEvent::S_MTE2>();
            position += static_cast<int64_t>(count);
            copied += count;
            if (position >= rowEnd) {
                ++row;
            }
        }
    }
}

}  // namespace spspmm_a2
}  // namespace opsgnn

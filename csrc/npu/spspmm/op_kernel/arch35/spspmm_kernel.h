/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Adapted from cann/ops-sparse sparse/spgemm/arch35 (Huawei Technologies Co., Ltd.):
// the row-parallel candidate counting, the three-pass prefix sum and the ordered
// per-row merge follow that implementation, restricted to the dtypes and the
// fixed-value layout this operator supports.
#pragma once
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "spspmm_tiling.h"

namespace {

__simt_callee__ __aicore__ inline int64_t GlobalThreadId()
{
    return static_cast<int64_t>(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() +
           AscendC::Simt::GetThreadIdx();
}

__simt_callee__ __aicore__ inline int64_t GlobalThreadStride()
{
    return static_cast<int64_t>(AscendC::Simt::GetBlockNum()) * AscendC::Simt::GetThreadNum();
}

// The public COO indices are int64 and may be out of range; every kernel sees the
// narrowed copy, and an invalid entry is reported through ``error`` and replaced
// by zero so the following passes stay inside their buffers.
__simt_vf__ __aicore__ __launch_bounds__(SPSSPMM_THREADS) void ConvertIndicesSimt(
    __gm__ const int64_t* rowA, __gm__ const int64_t* colA,
    __gm__ const int64_t* rowB, __gm__ const int64_t* colB,
    __gm__ int32_t* rowA32, __gm__ int32_t* colA32,
    __gm__ int32_t* rowB32, __gm__ int32_t* colB32,
    SpspmmShape shape, __gm__ int32_t* error)
{
    const int64_t tid = GlobalThreadId();
    const int64_t stride = GlobalThreadStride();
    for (int64_t e = tid; e < shape.nnzA; e += stride) {
        int64_t row = rowA[e];
        int64_t col = colA[e];
        if (row < 0 || row >= shape.m || col < 0 || col >= shape.k) {
            asc_atomic_add(error, 1);
            row = 0;
            col = 0;
        }
        if (e > 0 && rowA[e] < rowA[e - 1]) {
            asc_atomic_add(error, 1);
        }
        rowA32[e] = static_cast<int32_t>(row);
        colA32[e] = static_cast<int32_t>(col);
    }
    for (int64_t e = tid; e < shape.nnzB; e += stride) {
        int64_t row = rowB[e];
        int64_t col = colB[e];
        if (row < 0 || row >= shape.k || col < 0 || col >= shape.n) {
            asc_atomic_add(error, 1);
            row = 0;
            col = 0;
        }
        if (e > 0 && rowB[e] < rowB[e - 1]) {
            asc_atomic_add(error, 1);
        }
        rowB32[e] = static_cast<int32_t>(row);
        colB32[e] = static_cast<int32_t>(col);
    }
}

// The rows are sorted, so a histogram plus a prefix sum is the CSR row pointer.
__simt_vf__ __aicore__ __launch_bounds__(SPSSPMM_THREADS) void HistogramSimt(
    __gm__ const int32_t* rows, __gm__ int32_t* counts, int32_t edges, int32_t limit,
    __gm__ int32_t* error)
{
    const int64_t tid = GlobalThreadId();
    const int64_t stride = GlobalThreadStride();
    for (int64_t e = tid; e < edges; e += stride) {
        const int32_t row = rows[e];
        if (row < 0 || row >= limit) {
            asc_atomic_add(error, 1);
            continue;
        }
        asc_atomic_add(&counts[row], static_cast<int32_t>(1));
    }
}

// Coalesced input has strictly ascending columns inside every row; duplicates
// would silently double count because the merge assumes one entry per column.
__simt_vf__ __aicore__ __launch_bounds__(SPSSPMM_THREADS) void ValidateRowsSimt(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, int32_t rows, int32_t nnz,
    __gm__ int32_t* error)
{
    const int64_t tid = GlobalThreadId();
    const int64_t stride = GlobalThreadStride();
    for (int64_t row = tid; row < rows; row += stride) {
        const int32_t begin = rowPtr[row];
        const int32_t end = rowPtr[row + 1];
        if (begin < 0 || end < begin || end > nnz) {
            asc_atomic_add(error, 1);
            continue;
        }
        int32_t previous = -1;
        for (int32_t p = begin; p < end; ++p) {
            const int32_t col = colInd[p];
            if (col <= previous) {
                asc_atomic_add(error, 1);
            }
            previous = col;
        }
    }
}

template <typename CountT, typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(SPSSPMM_THREADS) void ScanLocalSimt(
    __gm__ const CountT* counts, __gm__ OffsetT* offsets, __gm__ OffsetT* chunkSums,
    int32_t count, int32_t chunkSize, int32_t numChunks)
{
    const int64_t tid = GlobalThreadId();
    const int64_t stride = GlobalThreadStride();
    for (int64_t chunk = tid; chunk < numChunks; chunk += stride) {
        const int64_t begin = chunk * chunkSize;
        const int64_t end = begin + chunkSize < count ? begin + chunkSize : count;
        OffsetT running = 0;
        for (int64_t i = begin; i < end; ++i) {
            offsets[i] = running;
            running += static_cast<OffsetT>(counts[i]);
        }
        chunkSums[chunk] = running;
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(SPSSPMM_THREADS) void ScanChunksSimt(
    __gm__ const OffsetT* chunkSums, __gm__ OffsetT* chunkPrefix, __gm__ OffsetT* offsets,
    int32_t count, int32_t numChunks)
{
    if (GlobalThreadId() != 0) {
        return;
    }
    OffsetT running = 0;
    for (int32_t chunk = 0; chunk < numChunks; ++chunk) {
        chunkPrefix[chunk] = running;
        running += chunkSums[chunk];
    }
    // The last chunk total is the length of the whole scan, so the closing
    // boundary is known here and does not need another kernel.
    offsets[count] = running;
}

template <typename OffsetT>
__simt_vf__ __aicore__ __launch_bounds__(SPSSPMM_THREADS) void ScanAddSimt(
    __gm__ OffsetT* offsets, __gm__ const OffsetT* chunkPrefix,
    int32_t count, int32_t chunkSize, int32_t numChunks)
{
    const int64_t tid = GlobalThreadId();
    const int64_t stride = GlobalThreadStride();
    for (int64_t chunk = tid; chunk < numChunks; chunk += stride) {
        const int64_t begin = chunk * chunkSize;
        const int64_t end = begin + chunkSize < count ? begin + chunkSize : count;
        const OffsetT base = chunkPrefix[chunk];
        for (int64_t i = begin; i < end; ++i) {
            offsets[i] += base;
        }
    }
}

// Upper bound of the intermediate products of every output row: the row merges
// one sorted B row per A entry, so its candidate slots are the sum of those
// B-row lengths. The real unique count never exceeds this bound.
__simt_vf__ __aicore__ __launch_bounds__(SPSSPMM_THREADS) void ProductCountSimt(
    __gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA,
    __gm__ const int32_t* rowPtrB, __gm__ int64_t* counts, SpspmmShape shape)
{
    const int64_t tid = GlobalThreadId();
    const int64_t stride = GlobalThreadStride();
    for (int64_t row = tid; row < shape.m; row += stride) {
        const int32_t begin = rowPtrA[row];
        const int32_t end = rowPtrA[row + 1];
        int64_t products = 0;
        for (int32_t p = begin; p < end; ++p) {
            const int32_t bRow = colIndA[p];
            products += static_cast<int64_t>(rowPtrB[bRow + 1]) - rowPtrB[bRow];
        }
        counts[row] = products;
    }
}

template <typename T>
__simt_callee__ __aicore__ inline float SpspmmToFloat(T value)
{
    return static_cast<float>(value);
}

template <typename T>
__simt_callee__ __aicore__ inline T SpspmmFromFloat(float value)
{
    return static_cast<T>(value);
}

// Smallest column still reachable through the live cursors of one output row,
// or SPSSPMM_MAX_COL when every cursor has reached the end of its B row.
__simt_callee__ __aicore__ inline int32_t FindMinColumn(
    __gm__ const int32_t* rowPtrB, __gm__ const int32_t* colIndB,
    __gm__ const int32_t* colIndA, __gm__ const int32_t* cursors,
    int32_t begin, int32_t end)
{
    int32_t minCol = SPSSPMM_MAX_COL;
    for (int32_t p = begin; p < end; ++p) {
        const int32_t cursor = cursors[p];
        if (cursor < rowPtrB[colIndA[p] + 1] && colIndB[cursor] < minCol) {
            minCol = colIndB[cursor];
        }
    }
    return minCol;
}

// Sum every product of one output row that lands on ``minCol`` and advance the
// cursors past the consumed entries.
template <typename T>
__simt_callee__ __aicore__ inline float AccumulateColumn(
    __gm__ const int32_t* rowPtrB, __gm__ const int32_t* colIndB,
    __gm__ const int32_t* colIndA, __gm__ const T* valA, __gm__ const T* valB,
    __gm__ int32_t* cursors, int32_t begin, int32_t end, int32_t minCol)
{
    float sum = 0.0F;
    for (int32_t p = begin; p < end; ++p) {
        const int32_t bEnd = rowPtrB[colIndA[p] + 1];
        int32_t cursor = cursors[p];
        while (cursor < bEnd && colIndB[cursor] == minCol) {
            sum += SpspmmToFloat<T>(valA[p]) * SpspmmToFloat<T>(valB[cursor]);
            ++cursor;
        }
        cursors[p] = cursor;
    }
    return sum;
}

// One thread owns one output row. Every A entry of the row walks its own sorted
// B row through ``cursors``; repeatedly taking the smallest live column and
// summing every product that hits it emits the unique, ascending columns in one
// pass, without atomics or shared memory.
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(SPSSPMM_THREADS) void ComputeSimt(
    __gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA, __gm__ const T* valA,
    __gm__ const int32_t* rowPtrB, __gm__ const int32_t* colIndB, __gm__ const T* valB,
    __gm__ int32_t* cursors, __gm__ const int64_t* productOffsets,
    __gm__ int32_t* candidateCols, __gm__ T* candidateVals, __gm__ int32_t* uniqueCounts,
    __gm__ int32_t* error, int32_t numRows)
{
    const int64_t tid = GlobalThreadId();
    const int64_t stride = GlobalThreadStride();
    for (int64_t row = tid; row < numRows; row += stride) {
        const int32_t begin = rowPtrA[row];
        const int32_t end = rowPtrA[row + 1];
        const int64_t outBase = productOffsets[row];
        const int64_t limit = productOffsets[row + 1] - outBase;
        int32_t count = 0;
        if (begin < end) {
            for (int32_t p = begin; p < end; ++p) {
                cursors[p] = rowPtrB[colIndA[p]];
            }
            for (;;) {
                const int32_t minCol = FindMinColumn(rowPtrB, colIndB, colIndA, cursors, begin, end);
                if (minCol == SPSSPMM_MAX_COL) {
                    break;
                }
                const float sum =
                    AccumulateColumn<T>(rowPtrB, colIndB, colIndA, valA, valB, cursors, begin,
                                        end, minCol);
                if (static_cast<int64_t>(count) >= limit) {
                    asc_atomic_add(error, 1);
                    break;
                }
                candidateCols[outBase + count] = minCol;
                candidateVals[outBase + count] = SpspmmFromFloat<T>(sum);
                ++count;
            }
        }
        uniqueCounts[row] = count;
    }
}

// Compact the per-row candidate slots into the exact COO output. The rows are
// already column-ascending, so the emitted order is row-major.
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(SPSSPMM_THREADS) void CopySimt(
    __gm__ const int64_t* productOffsets, __gm__ const int32_t* candidateCols,
    __gm__ const T* candidateVals, __gm__ const int32_t* uniqueCounts,
    __gm__ const int64_t* rowPtrC, __gm__ int64_t* indexC, __gm__ T* valueC,
    int64_t total, SpspmmShape shape)
{
    const int64_t tid = GlobalThreadId();
    const int64_t stride = GlobalThreadStride();
    for (int64_t row = tid; row < shape.m; row += stride) {
        const int32_t count = uniqueCounts[row];
        if (count <= 0) {
            continue;
        }
        const int64_t dstBase = rowPtrC[row];
        const int64_t srcBase = productOffsets[row];
        for (int32_t j = 0; j < count; ++j) {
            indexC[dstBase + j] = static_cast<int64_t>(row);
            indexC[total + dstBase + j] = static_cast<int64_t>(candidateCols[srcBase + j]);
            valueC[dstBase + j] = candidateVals[srcBase + j];
        }
    }
}

}  // namespace

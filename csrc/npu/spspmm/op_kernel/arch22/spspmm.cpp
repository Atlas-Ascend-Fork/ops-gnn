/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// arch22 (A2/A3) compilation unit of spspmm: device entry points and the
// ``<<<>>>`` launches the Host schedules on the current stream.
#include "spspmm_kernel.h"

#include "common/aiv_launch_utils.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220
#error "arch22/spspmm.cpp must be compiled for dav-2201 (A2/A3)"
#endif

namespace {

using namespace opsgnn::spspmm_a2;

// AIV count of the widest parallel phase; A2 and A3 expose fewer units, so the
// launch helper clamps it to the hardware.
constexpr uint32_t kBlocks = 40;

uint32_t BlocksFor(uint64_t workItems)
{
    const uint64_t wanted = workItems == 0 ? 1 : workItems;
    return ops_gnn::sparse::ResolveAivCoreNum(static_cast<uint32_t>(wanted < kBlocks ? wanted : kBlocks));
}

// Three-pass prefix sum of ``count`` entries: the chunk count and the block
// count derived from it. Every scan launcher starts from this plan.
struct ScanPlan {
    int32_t chunks;
    uint32_t blocks;
};

ScanPlan PlanScan(int32_t count)
{
    const int32_t chunks = (count + SPSSPMM_SCAN_CHUNK - 1) / SPSSPMM_SCAN_CHUNK;
    return {chunks, BlocksFor(static_cast<uint64_t>(chunks))};
}

// UB tiles of the merge pass. The entry owns the pipe and this struct only
// takes a view of it, following the kernel-struct layout used by ``coalesce``.
template <typename T>
struct ComputeUb {
    AscendC::TPipe* pipe = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cursorBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> columnBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stagedColBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stagedValBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> countBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> flag;

    __aicore__ inline void Init(AscendC::TPipe* owner)
    {
        pipe = owner;
        pipe->InitBuffer(cursorBuf, kCursorCap * sizeof(int32_t));
        pipe->InitBuffer(columnBuf, kCursorCap * sizeof(int32_t));
        pipe->InitBuffer(stagedColBuf, kStageCap * sizeof(int32_t));
        pipe->InitBuffer(stagedValBuf, kStageCap * sizeof(T));
        pipe->InitBuffer(countBuf, kInt32Group * sizeof(int32_t));
        pipe->InitBuffer(flag, SPSSPMM_ERROR_WORDS * sizeof(int32_t));
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> Cursors()
    {
        return cursorBuf.Get<int32_t>();
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> Columns()
    {
        return columnBuf.Get<int32_t>();
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> StagedCols()
    {
        return stagedColBuf.Get<int32_t>();
    }

    __aicore__ inline AscendC::LocalTensor<T> StagedVals()
    {
        return stagedValBuf.Get<T>();
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> Counts()
    {
        return countBuf.Get<int32_t>();
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> Flags()
    {
        return flag.Get<int32_t>();
    }
};

// Device views of one compute launch; the entry binds them once and the row
// loop only reads them.
template <typename T>
struct ComputeGm {
    AscendC::GlobalTensor<int32_t> rowPtrA;
    AscendC::GlobalTensor<int32_t> colIndA;
    AscendC::GlobalTensor<T> valA;
    AscendC::GlobalTensor<int32_t> rowPtrB;
    AscendC::GlobalTensor<int32_t> colIndB;
    AscendC::GlobalTensor<T> valB;
    AscendC::GlobalTensor<int64_t> offset;
    AscendC::GlobalTensor<int32_t> candCol;
    AscendC::GlobalTensor<T> candVal;
    AscendC::GlobalTensor<int32_t> unique;
    AscendC::GlobalTensor<int32_t> cursor;
    AscendC::GlobalTensor<int32_t> status;

    __aicore__ inline void Bind(__gm__ const int32_t* rowPtrA_, __gm__ const int32_t* colIndA_,
                                __gm__ const T* valA_, __gm__ const int32_t* rowPtrB_,
                                __gm__ const int32_t* colIndB_, __gm__ const T* valB_,
                                __gm__ int32_t* cursor_, __gm__ const int64_t* offset_,
                                __gm__ int32_t* candCol_, __gm__ T* candVal_, __gm__ int32_t* unique_,
                                __gm__ int32_t* status_, SpspmmShape shape)
    {
        rowPtrA.SetGlobalBuffer(const_cast<__gm__ int32_t*>(rowPtrA_), static_cast<int64_t>(shape.m) + 1);
        colIndA.SetGlobalBuffer(const_cast<__gm__ int32_t*>(colIndA_), shape.nnzA);
        valA.SetGlobalBuffer(const_cast<__gm__ T*>(valA_), shape.nnzA);
        rowPtrB.SetGlobalBuffer(const_cast<__gm__ int32_t*>(rowPtrB_), static_cast<int64_t>(shape.k) + 1);
        colIndB.SetGlobalBuffer(const_cast<__gm__ int32_t*>(colIndB_), shape.nnzB);
        valB.SetGlobalBuffer(const_cast<__gm__ T*>(valB_), shape.nnzB);
        cursor.SetGlobalBuffer(cursor_, shape.nnzA);
        offset.SetGlobalBuffer(const_cast<__gm__ int64_t*>(offset_), static_cast<int64_t>(shape.m) + 1);
        candCol.SetGlobalBuffer(candCol_, SPSSPMM_MAX_PRODUCTS);
        candVal.SetGlobalBuffer(candVal_, SPSSPMM_MAX_PRODUCTS);
        unique.SetGlobalBuffer(unique_, shape.m);
        status.SetGlobalBuffer(status_, SPSSPMM_ERROR_WORDS);
    }
};

// Device views of one conversion launch; the entry binds them once.
struct ConvertGm {
    AscendC::GlobalTensor<int64_t> rowA;
    AscendC::GlobalTensor<int64_t> colA;
    AscendC::GlobalTensor<int64_t> rowB;
    AscendC::GlobalTensor<int64_t> colB;
    AscendC::GlobalTensor<int32_t> rowA32;
    AscendC::GlobalTensor<int32_t> colA32;
    AscendC::GlobalTensor<int32_t> rowB32;
    AscendC::GlobalTensor<int32_t> colB32;
    AscendC::GlobalTensor<int32_t> status;

    __aicore__ inline void Bind(__gm__ const int64_t* rowA_, __gm__ const int64_t* colA_,
                                __gm__ const int64_t* rowB_, __gm__ const int64_t* colB_,
                                __gm__ int32_t* rowA32_, __gm__ int32_t* colA32_,
                                __gm__ int32_t* rowB32_, __gm__ int32_t* colB32_,
                                __gm__ int32_t* status_, SpspmmShape shape)
    {
        rowA.SetGlobalBuffer(const_cast<__gm__ int64_t*>(rowA_), shape.nnzA);
        colA.SetGlobalBuffer(const_cast<__gm__ int64_t*>(colA_), shape.nnzA);
        rowB.SetGlobalBuffer(const_cast<__gm__ int64_t*>(rowB_), shape.nnzB);
        colB.SetGlobalBuffer(const_cast<__gm__ int64_t*>(colB_), shape.nnzB);
        rowA32.SetGlobalBuffer(rowA32_, shape.nnzA);
        colA32.SetGlobalBuffer(colA32_, shape.nnzA);
        rowB32.SetGlobalBuffer(rowB32_, shape.nnzB);
        colB32.SetGlobalBuffer(colB32_, shape.nnzB);
        status.SetGlobalBuffer(status_, SPSSPMM_ERROR_WORDS);
    }
};

}  // namespace

extern "C" __global__ __aicore__ void spspmm_convert(
    __gm__ const int64_t* rowA, __gm__ const int64_t* colA, __gm__ const int64_t* rowB,
    __gm__ const int64_t* colB, __gm__ int32_t* rowA32, __gm__ int32_t* colA32,
    __gm__ int32_t* rowB32, __gm__ int32_t* colB32, SpspmmShape shape, __gm__ int32_t* status)
{
    using namespace opsgnn::spspmm_a2;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowIn, colIn, rowOut, colOut, flag;
    pipe.InitBuffer(rowIn, kEdgeTile * sizeof(int64_t));
    pipe.InitBuffer(colIn, kEdgeTile * sizeof(int64_t));
    pipe.InitBuffer(rowOut, kEdgeTile * sizeof(int32_t));
    pipe.InitBuffer(colOut, kEdgeTile * sizeof(int32_t));
    pipe.InitBuffer(flag, SPSSPMM_ERROR_WORDS * sizeof(int32_t));

    ConvertGm gm;
    gm.Bind(rowA, colA, rowB, colB, rowA32, colA32, rowB32, colB32, status, shape);

    const GroupRange a = RowRange(shape.nnzA);
    const GroupRange b = RowRange(shape.nnzB);

    const AscendC::LocalTensor<int64_t> rowInTile = rowIn.Get<int64_t>();
    const AscendC::LocalTensor<int64_t> colInTile = colIn.Get<int64_t>();
    const AscendC::LocalTensor<int32_t> rowOutTile = rowOut.Get<int32_t>();
    const AscendC::LocalTensor<int32_t> colOutTile = colOut.Get<int32_t>();

    bool failed = false;
    if (a.begin < a.end) {
        failed = ConvertSide(gm.rowA, gm.colA, gm.rowA32, gm.colA32, shape.m, shape.k, a.begin,
                             a.end, rowInTile, colInTile, rowOutTile, colOutTile);
    }
    if (b.begin < b.end) {
        failed = ConvertSide(gm.rowB, gm.colB, gm.rowB32, gm.colB32, shape.k, shape.n, b.begin,
                             b.end, rowInTile, colInTile, rowOutTile, colOutTile) ||
                 failed;
    }
    if (failed) {
        RaiseError(gm.status, flag.Get<int32_t>());
    }
}

extern "C" __global__ __aicore__ void spspmm_histogram(__gm__ const int32_t* rows,
                                                       __gm__ int32_t* counts, int32_t edges,
                                                       int32_t limit, __gm__ int32_t* status)
{
    using namespace opsgnn::spspmm_a2;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tile, flag;
    pipe.InitBuffer(tile, kRowTile * sizeof(int32_t));
    pipe.InitBuffer(flag, SPSSPMM_ERROR_WORDS * sizeof(int32_t));

    AscendC::GlobalTensor<int32_t> rowGm;
    AscendC::GlobalTensor<int32_t> countGm;
    AscendC::GlobalTensor<int32_t> statusGm;
    rowGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(rows), edges);
    countGm.SetGlobalBuffer(counts, limit);
    statusGm.SetGlobalBuffer(status, SPSSPMM_ERROR_WORDS);

    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = LaunchBlocks();
    const int64_t begin = GroupStart(limit, kInt32Group, block, blocks);
    const int64_t end = GroupEnd(limit, kInt32Group, block, blocks);
    if (begin >= end) {
        return;
    }
    if (CountRows(rowGm, countGm, edges, limit, begin, end, tile.Get<int32_t>())) {
        RaiseError(statusGm, flag.Get<int32_t>());
    }
}

extern "C" __global__ __aicore__ void spspmm_validate(__gm__ const int32_t* rowPtr,
                                                      __gm__ const int32_t* colInd, int32_t rows,
                                                      int32_t nnz, __gm__ int32_t* status)
{
    using namespace opsgnn::spspmm_a2;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> flag;
    pipe.InitBuffer(flag, SPSSPMM_ERROR_WORDS * sizeof(int32_t));

    AscendC::GlobalTensor<int32_t> ptrGm;
    AscendC::GlobalTensor<int32_t> indGm;
    AscendC::GlobalTensor<int32_t> statusGm;
    ptrGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(rowPtr), static_cast<int64_t>(rows) + 1);
    indGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(colInd), nnz);
    statusGm.SetGlobalBuffer(status, SPSSPMM_ERROR_WORDS);

    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = LaunchBlocks();
    const int64_t begin = Partition(rows, block, blocks);
    const int64_t end = Partition(rows, block + 1, blocks);
    if (ValidateRows(ptrGm, indGm, begin, end, nnz)) {
        RaiseError(statusGm, flag.Get<int32_t>());
    }
}

// --- three-pass prefix sum ---------------------------------------------------
// Nine entry points cover the three (count, offset) type pairs of the Host.

#define SPSSPMM_SCAN_KERNELS(SUFFIX, COUNT_T, OFFSET_T)                                             \
    extern "C" __global__ __aicore__ void spspmm_scan_local_##SUFFIX(                               \
        __gm__ const COUNT_T* counts, __gm__ OFFSET_T* offsets, __gm__ OFFSET_T* chunkSums,          \
        int32_t count, int32_t numChunks)                                                            \
    {                                                                                                \
        using namespace opsgnn::spspmm_a2;                                                           \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                              \
        AscendC::TPipe pipe;                                                                         \
        AscendC::TBuf<AscendC::TPosition::VECCALC> countBuf;                                         \
        AscendC::TBuf<AscendC::TPosition::VECCALC> offBuf;                                           \
        pipe.InitBuffer(countBuf, kScanTile * sizeof(COUNT_T));                                      \
        pipe.InitBuffer(offBuf, kScanTile * sizeof(OFFSET_T));                                        \
        AscendC::GlobalTensor<COUNT_T> countGm;                                                       \
        AscendC::GlobalTensor<OFFSET_T> offGm;                                                        \
        AscendC::GlobalTensor<OFFSET_T> sumGm;                                                        \
        countGm.SetGlobalBuffer(const_cast<__gm__ COUNT_T*>(counts), count);                          \
        offGm.SetGlobalBuffer(offsets, static_cast<int64_t>(count) + 1);                                                    \
        sumGm.SetGlobalBuffer(chunkSums, numChunks * ChunkSumStride<OFFSET_T>());                    \
        ScanLocal<COUNT_T, OFFSET_T>(countGm, offGm, sumGm, count, numChunks,                         \
                                     countBuf.Get<COUNT_T>(), offBuf.Get<OFFSET_T>());                \
    }                                                                                                \
    extern "C" __global__ __aicore__ void spspmm_scan_chunks_##SUFFIX(                               \
        __gm__ const OFFSET_T* chunkSums, __gm__ OFFSET_T* chunkPrefix, __gm__ OFFSET_T* offsets,     \
        int32_t count, int32_t numChunks)                                                            \
    {                                                                                                \
        using namespace opsgnn::spspmm_a2;                                                           \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                              \
        AscendC::TPipe pipe;                                                                         \
        AscendC::TBuf<AscendC::TPosition::VECCALC> prefixBuf;                                        \
        pipe.InitBuffer(prefixBuf, kScanTile * sizeof(OFFSET_T));                                    \
        AscendC::GlobalTensor<OFFSET_T> sumGm;                                                        \
        AscendC::GlobalTensor<OFFSET_T> prefixGm;                                                     \
        AscendC::GlobalTensor<OFFSET_T> offGm;                                                        \
        sumGm.SetGlobalBuffer(const_cast<__gm__ OFFSET_T*>(chunkSums),                                \
                              numChunks * ChunkSumStride<OFFSET_T>());                                \
        prefixGm.SetGlobalBuffer(chunkPrefix, numChunks);                                             \
        offGm.SetGlobalBuffer(offsets, static_cast<int64_t>(count) + 1);                                                    \
        ScanChunks<OFFSET_T>(sumGm, prefixGm, offGm, count, numChunks, prefixBuf.Get<OFFSET_T>());     \
    }                                                                                                \
    extern "C" __global__ __aicore__ void spspmm_scan_add_##SUFFIX(                                 \
        __gm__ OFFSET_T* offsets, __gm__ const OFFSET_T* chunkPrefix, int32_t count,                 \
        int32_t numChunks)                                                                            \
    {                                                                                                \
        using namespace opsgnn::spspmm_a2;                                                           \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                              \
        AscendC::TPipe pipe;                                                                         \
        AscendC::TBuf<AscendC::TPosition::VECCALC> offBuf;                                           \
        pipe.InitBuffer(offBuf, kScanTile * sizeof(OFFSET_T));                                       \
        AscendC::GlobalTensor<OFFSET_T> offGm;                                                        \
        AscendC::GlobalTensor<OFFSET_T> prefixGm;                                                     \
        offGm.SetGlobalBuffer(offsets, static_cast<int64_t>(count) + 1);                                                    \
        prefixGm.SetGlobalBuffer(const_cast<__gm__ OFFSET_T*>(chunkPrefix), numChunks);               \
        ScanAdd<OFFSET_T>(offGm, prefixGm, count, numChunks, offBuf.Get<OFFSET_T>());                 \
    }

SPSSPMM_SCAN_KERNELS(i32, int32_t, int32_t)
SPSSPMM_SCAN_KERNELS(i64, int64_t, int64_t)
SPSSPMM_SCAN_KERNELS(i32_to_i64, int32_t, int64_t)

#undef SPSSPMM_SCAN_KERNELS

extern "C" __global__ __aicore__ void spspmm_product_count(
    __gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA, __gm__ const int32_t* rowPtrB,
    __gm__ int64_t* counts, SpspmmShape shape)
{
    using namespace opsgnn::spspmm_a2;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tile;
    pipe.InitBuffer(tile, kRowTile * sizeof(int64_t));

    AscendC::GlobalTensor<int32_t> rowPtrAGm;
    AscendC::GlobalTensor<int32_t> colIndAGm;
    AscendC::GlobalTensor<int32_t> rowPtrBGm;
    AscendC::GlobalTensor<int64_t> countGm;
    rowPtrAGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(rowPtrA), static_cast<int64_t>(shape.m) + 1);
    colIndAGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(colIndA), shape.nnzA);
    rowPtrBGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(rowPtrB), static_cast<int64_t>(shape.k) + 1);
    countGm.SetGlobalBuffer(counts, shape.m);

    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = LaunchBlocks();
    const int64_t begin = GroupStart(shape.m, kInt64Group, block, blocks);
    const int64_t end = GroupEnd(shape.m, kInt64Group, block, blocks);
    if (begin >= end) {
        return;
    }
    ProductCountRows(rowPtrAGm, colIndAGm, rowPtrBGm, countGm, begin, end, tile.Get<int64_t>());
}

// Shared body of the two compute entry points: the UB-cursor merge of the short
// rows, then the guard of the global-cursor spill pass.
template <typename T>
__aicore__ inline void ComputeEntry(__gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA,
                                    __gm__ const T* valA, __gm__ const int32_t* rowPtrB,
                                    __gm__ const int32_t* colIndB, __gm__ const T* valB,
                                    __gm__ int32_t* cursors, __gm__ const int64_t* productOffsets,
                                    __gm__ int32_t* candidateCols, __gm__ T* candidateVals,
                                    __gm__ int32_t* uniqueCounts, __gm__ int32_t* status,
                                    SpspmmShape shape)
{
    using namespace opsgnn::spspmm_a2;

    AscendC::TPipe pipe;
    ComputeUb<T> ub;
    ub.Init(&pipe);
    ComputeGm<T> gm;
    gm.Bind(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, cursors, productOffsets,
            candidateCols, candidateVals, uniqueCounts, status, shape);

    const GroupRange range = RowRange(shape.m);
    if (range.begin >= range.end) {
        return;
    }
    AscendC::LocalTensor<int32_t> cursorsTile = ub.Cursors();
    const bool failed = ComputeRows<T>(
        gm.rowPtrA, gm.colIndA, gm.valA, gm.rowPtrB, gm.colIndB, gm.valB, gm.offset, gm.candCol,
        gm.candVal, gm.unique, range.begin, range.end, cursorsTile, ub.Columns(),
        ub.StagedCols(), ub.StagedVals(), gm.status, ub.Flags(), ub.Counts());
    if (failed) {
        RaiseError(gm.status, ub.Flags());
    }
}

extern "C" __global__ __aicore__ void spspmm_compute_f32(
    __gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA, __gm__ const float* valA,
    __gm__ const int32_t* rowPtrB, __gm__ const int32_t* colIndB, __gm__ const float* valB,
    __gm__ int32_t* cursors, __gm__ const int64_t* productOffsets, __gm__ int32_t* candidateCols,
    __gm__ float* candidateVals, __gm__ int32_t* uniqueCounts, __gm__ int32_t* status,
    SpspmmShape shape)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ComputeEntry<float>(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, cursors, productOffsets,
                        candidateCols, candidateVals, uniqueCounts, status, shape);
}

extern "C" __global__ __aicore__ void spspmm_compute_f16(
    __gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA, __gm__ const half* valA,
    __gm__ const int32_t* rowPtrB, __gm__ const int32_t* colIndB, __gm__ const half* valB,
    __gm__ int32_t* cursors, __gm__ const int64_t* productOffsets, __gm__ int32_t* candidateCols,
    __gm__ half* candidateVals, __gm__ int32_t* uniqueCounts, __gm__ int32_t* status,
    SpspmmShape shape)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ComputeEntry<half>(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, cursors, productOffsets,
                       candidateCols, candidateVals, uniqueCounts, status, shape);
}

// The spill pass runs on one AIV and only when the compute pass recorded a long
// row, so an input whose rows all fit the UB cursors pays a single scalar read.
template <typename T>
__aicore__ inline void SpillEntry(__gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA,
                                  __gm__ const T* valA, __gm__ const int32_t* rowPtrB,
                                  __gm__ const int32_t* colIndB, __gm__ const T* valB,
                                  __gm__ int32_t* cursors, __gm__ const int64_t* productOffsets,
                                  __gm__ int32_t* candidateCols, __gm__ T* candidateVals,
                                  __gm__ int32_t* uniqueCounts, __gm__ int32_t* status,
                                  SpspmmShape shape)
{
    using namespace opsgnn::spspmm_a2;

    ComputeGm<T> gm;
    gm.Bind(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, cursors, productOffsets,
            candidateCols, candidateVals, uniqueCounts, status, shape);
    if (gm.status.GetValue(static_cast<uint64_t>(SPSSPMM_LONG_ROW_WORD)) == 0) {
        return;
    }

    AscendC::TPipe pipe;
    ComputeUb<T> ub;
    ub.Init(&pipe);

    const bool failed = SpillRows<T>(gm.rowPtrA, gm.colIndA, gm.valA, gm.rowPtrB, gm.colIndB,
                                     gm.valB, gm.offset, gm.candCol, gm.candVal, gm.unique,
                                     gm.cursor, shape.m, ub.Columns(), ub.StagedCols(),
                                     ub.StagedVals(), ub.Counts());
    if (failed) {
        RaiseError(gm.status, ub.Flags());
    }
}

extern "C" __global__ __aicore__ void spspmm_spill_f32(
    __gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA, __gm__ const float* valA,
    __gm__ const int32_t* rowPtrB, __gm__ const int32_t* colIndB, __gm__ const float* valB,
    __gm__ int32_t* cursors, __gm__ const int64_t* productOffsets, __gm__ int32_t* candidateCols,
    __gm__ float* candidateVals, __gm__ int32_t* uniqueCounts, __gm__ int32_t* status,
    SpspmmShape shape)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpillEntry<float>(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, cursors, productOffsets,
                      candidateCols, candidateVals, uniqueCounts, status, shape);
}

extern "C" __global__ __aicore__ void spspmm_spill_f16(
    __gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA, __gm__ const half* valA,
    __gm__ const int32_t* rowPtrB, __gm__ const int32_t* colIndB, __gm__ const half* valB,
    __gm__ int32_t* cursors, __gm__ const int64_t* productOffsets, __gm__ int32_t* candidateCols,
    __gm__ half* candidateVals, __gm__ int32_t* uniqueCounts, __gm__ int32_t* status,
    SpspmmShape shape)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpillEntry<half>(rowPtrA, colIndA, valA, rowPtrB, colIndB, valB, cursors, productOffsets,
                     candidateCols, candidateVals, uniqueCounts, status, shape);
}

template <typename T>
__aicore__ inline void CopyEntry(__gm__ const int64_t* productOffsets,
                                 __gm__ const int32_t* candidateCols, __gm__ const T* candidateVals,
                                 __gm__ const int32_t* uniqueCounts, __gm__ const int64_t* rowPtrC,
                                 __gm__ int64_t* indexC, __gm__ T* valueC, int64_t total,
                                 SpspmmShape shape)
{
    using namespace opsgnn::spspmm_a2;

    // The per-row counts are already folded into ``rowPtrC``, so the compaction
    // pass only needs the row boundaries.
    (void)uniqueCounts;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> colBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> colOutBuf;
    pipe.InitBuffer(colBuf, kCopyTile * sizeof(int32_t));
    pipe.InitBuffer(valBuf, kCopyTile * sizeof(T));
    pipe.InitBuffer(rowBuf, kCopyTile * sizeof(int64_t));
    pipe.InitBuffer(colOutBuf, kCopyTile * sizeof(int64_t));

    AscendC::GlobalTensor<int64_t> offsetGm;
    AscendC::GlobalTensor<int32_t> candColGm;
    AscendC::GlobalTensor<T> candValGm;
    AscendC::GlobalTensor<int64_t> rowPtrGm;
    AscendC::GlobalTensor<int64_t> indexGm;
    AscendC::GlobalTensor<T> valueGm;
    offsetGm.SetGlobalBuffer(const_cast<__gm__ int64_t*>(productOffsets), static_cast<int64_t>(shape.m) + 1);
    candColGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(candidateCols), SPSSPMM_MAX_PRODUCTS);
    candValGm.SetGlobalBuffer(const_cast<__gm__ T*>(candidateVals), SPSSPMM_MAX_PRODUCTS);
    rowPtrGm.SetGlobalBuffer(const_cast<__gm__ int64_t*>(rowPtrC), static_cast<int64_t>(shape.m) + 1);
    indexGm.SetGlobalBuffer(indexC, 2 * total);
    valueGm.SetGlobalBuffer(valueC, total);

    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = LaunchBlocks();
    const int64_t begin = GroupStart(total, kCopyGroup, block, blocks);
    const int64_t end = GroupEnd(total, kCopyGroup, block, blocks);
    CopyRange<T>(offsetGm, candColGm, candValGm, rowPtrGm, indexGm, valueGm, total, shape.m, begin,
                 end, colBuf.Get<int32_t>(), valBuf.Get<T>(), rowBuf.Get<int64_t>(),
                 colOutBuf.Get<int64_t>());
}

extern "C" __global__ __aicore__ void spspmm_copy_f32(
    __gm__ const int64_t* productOffsets, __gm__ const int32_t* candidateCols,
    __gm__ const float* candidateVals, __gm__ const int32_t* uniqueCounts,
    __gm__ const int64_t* rowPtrC, __gm__ int64_t* indexC, __gm__ float* valueC, int64_t total,
    SpspmmShape shape)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CopyEntry<float>(productOffsets, candidateCols, candidateVals, uniqueCounts, rowPtrC, indexC,
                     valueC, total, shape);
}

extern "C" __global__ __aicore__ void spspmm_copy_f16(
    __gm__ const int64_t* productOffsets, __gm__ const int32_t* candidateCols,
    __gm__ const half* candidateVals, __gm__ const int32_t* uniqueCounts,
    __gm__ const int64_t* rowPtrC, __gm__ int64_t* indexC, __gm__ half* valueC, int64_t total,
    SpspmmShape shape)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CopyEntry<half>(productOffsets, candidateCols, candidateVals, uniqueCounts, rowPtrC, indexC,
                    valueC, total, shape);
}

namespace opsgnn {

void SpspmmConvertIndices(const void* rowA, const void* colA, const void* rowB, const void* colB,
                          void* rowA32, void* colA32, void* rowB32, void* colB32, SpspmmShape shape,
                          void* status, void* stream)
{
    if (shape.nnzA == 0 && shape.nnzB == 0) {
        return;
    }
    const uint32_t blocks = BlocksFor(static_cast<uint64_t>(shape.nnzA) + shape.nnzB);
    spspmm_convert<<<blocks, nullptr, stream>>>(
        reinterpret_cast<const int64_t*>(rowA), reinterpret_cast<const int64_t*>(colA),
        reinterpret_cast<const int64_t*>(rowB), reinterpret_cast<const int64_t*>(colB),
        reinterpret_cast<int32_t*>(rowA32), reinterpret_cast<int32_t*>(colA32),
        reinterpret_cast<int32_t*>(rowB32), reinterpret_cast<int32_t*>(colB32), shape,
        reinterpret_cast<int32_t*>(status));
}

void SpspmmHistogram(const void* rows, void* counts, int32_t edges, int32_t limit, void* status,
                     void* stream)
{
    if (edges == 0 || limit <= 0) {
        return;
    }
    const uint32_t blocks = BlocksFor(static_cast<uint64_t>(limit));
    spspmm_histogram<<<blocks, nullptr, stream>>>(reinterpret_cast<const int32_t*>(rows),
                                                  reinterpret_cast<int32_t*>(counts), edges, limit,
                                                  reinterpret_cast<int32_t*>(status));
}

void SpspmmValidateRows(const void* rowPtr, const void* colInd, int32_t rows, int32_t nnz,
                        void* status, void* stream)
{
    if (rows == 0) {
        return;
    }
    const uint32_t blocks = BlocksFor(static_cast<uint64_t>(rows));
    spspmm_validate<<<blocks, nullptr, stream>>>(reinterpret_cast<const int32_t*>(rowPtr),
                                                 reinterpret_cast<const int32_t*>(colInd), rows, nnz,
                                                 reinterpret_cast<int32_t*>(status));
}

// Three-pass prefix sum shared by the three count/offset type pairs; the
// per-type kernels are the only part that differs.
template <typename InT, typename OutT>
void LaunchScan(const void* counts, void* offsets, void* chunkSums, void* chunkPrefix,
                int32_t count, void* stream)
{
    const ScanPlan plan = PlanScan(count);
    if (plan.chunks <= 0) {
        return;
    }
    const InT* input = reinterpret_cast<const InT*>(counts);
    OutT* offset = reinterpret_cast<OutT*>(offsets);
    OutT* sums = reinterpret_cast<OutT*>(chunkSums);
    OutT* prefix = reinterpret_cast<OutT*>(chunkPrefix);
    if constexpr (sizeof(InT) == sizeof(OutT)) {
        if constexpr (sizeof(InT) == sizeof(int32_t)) {
            spspmm_scan_local_i32<<<plan.blocks, nullptr, stream>>>(input, offset, sums, count,
                                                                    plan.chunks);
            spspmm_scan_chunks_i32<<<1, nullptr, stream>>>(sums, prefix, offset, count, plan.chunks);
            spspmm_scan_add_i32<<<plan.blocks, nullptr, stream>>>(offset, prefix, count, plan.chunks);
        } else {
            spspmm_scan_local_i64<<<plan.blocks, nullptr, stream>>>(input, offset, sums, count,
                                                                    plan.chunks);
            spspmm_scan_chunks_i64<<<1, nullptr, stream>>>(sums, prefix, offset, count, plan.chunks);
            spspmm_scan_add_i64<<<plan.blocks, nullptr, stream>>>(offset, prefix, count, plan.chunks);
        }
    } else {
        spspmm_scan_local_i32_to_i64<<<plan.blocks, nullptr, stream>>>(input, offset, sums, count,
                                                                       plan.chunks);
        spspmm_scan_chunks_i32_to_i64<<<1, nullptr, stream>>>(sums, prefix, offset, count,
                                                              plan.chunks);
        spspmm_scan_add_i32_to_i64<<<plan.blocks, nullptr, stream>>>(offset, prefix, count,
                                                                     plan.chunks);
    }
}

void SpspmmScanInt32(const void* counts, void* offsets, void* chunkSums, void* chunkPrefix,
                     int32_t count, void* stream)
{
    LaunchScan<int32_t, int32_t>(counts, offsets, chunkSums, chunkPrefix, count, stream);
}

void SpspmmScanInt64(const void* counts, void* offsets, void* chunkSums, void* chunkPrefix,
                     int32_t count, void* stream)
{
    LaunchScan<int64_t, int64_t>(counts, offsets, chunkSums, chunkPrefix, count, stream);
}

void SpspmmScanInt32ToInt64(const void* counts, void* offsets, void* chunkSums, void* chunkPrefix,
                            int32_t count, void* stream)
{
    LaunchScan<int32_t, int64_t>(counts, offsets, chunkSums, chunkPrefix, count, stream);
}

void SpspmmProductCount(const void* rowPtrA, const void* colIndA, const void* rowPtrB, void* counts,
                        SpspmmShape shape, void* stream)
{
    if (shape.m == 0) {
        return;
    }
    const uint32_t blocks = BlocksFor(static_cast<uint64_t>(shape.m));
    spspmm_product_count<<<blocks, nullptr, stream>>>(
        reinterpret_cast<const int32_t*>(rowPtrA), reinterpret_cast<const int32_t*>(colIndA),
        reinterpret_cast<const int32_t*>(rowPtrB), reinterpret_cast<int64_t*>(counts), shape);
}

void SpspmmCompute(const void* rowPtrA, const void* colIndA, const void* valA,
                   const void* rowPtrB, const void* colIndB, const void* valB, void* cursors,
                   const void* productOffsets, void* candidateCols, void* candidateVals,
                   void* uniqueCounts, void* status, SpspmmShape shape, int dtype, void* stream)
{
    if (shape.m == 0) {
        return;
    }
    const uint32_t blocks = BlocksFor(static_cast<uint64_t>(shape.m));
    if (dtype == 0) {
        spspmm_compute_f32<<<blocks, nullptr, stream>>>(
            reinterpret_cast<const int32_t*>(rowPtrA), reinterpret_cast<const int32_t*>(colIndA),
            reinterpret_cast<const float*>(valA), reinterpret_cast<const int32_t*>(rowPtrB),
            reinterpret_cast<const int32_t*>(colIndB), reinterpret_cast<const float*>(valB),
            reinterpret_cast<int32_t*>(cursors), reinterpret_cast<const int64_t*>(productOffsets),
            reinterpret_cast<int32_t*>(candidateCols), reinterpret_cast<float*>(candidateVals),
            reinterpret_cast<int32_t*>(uniqueCounts), reinterpret_cast<int32_t*>(status), shape);
        spspmm_spill_f32<<<1, nullptr, stream>>>(
            reinterpret_cast<const int32_t*>(rowPtrA), reinterpret_cast<const int32_t*>(colIndA),
            reinterpret_cast<const float*>(valA), reinterpret_cast<const int32_t*>(rowPtrB),
            reinterpret_cast<const int32_t*>(colIndB), reinterpret_cast<const float*>(valB),
            reinterpret_cast<int32_t*>(cursors), reinterpret_cast<const int64_t*>(productOffsets),
            reinterpret_cast<int32_t*>(candidateCols), reinterpret_cast<float*>(candidateVals),
            reinterpret_cast<int32_t*>(uniqueCounts), reinterpret_cast<int32_t*>(status), shape);
    } else {
        spspmm_compute_f16<<<blocks, nullptr, stream>>>(
            reinterpret_cast<const int32_t*>(rowPtrA), reinterpret_cast<const int32_t*>(colIndA),
            reinterpret_cast<const half*>(valA), reinterpret_cast<const int32_t*>(rowPtrB),
            reinterpret_cast<const int32_t*>(colIndB), reinterpret_cast<const half*>(valB),
            reinterpret_cast<int32_t*>(cursors), reinterpret_cast<const int64_t*>(productOffsets),
            reinterpret_cast<int32_t*>(candidateCols), reinterpret_cast<half*>(candidateVals),
            reinterpret_cast<int32_t*>(uniqueCounts), reinterpret_cast<int32_t*>(status), shape);
        spspmm_spill_f16<<<1, nullptr, stream>>>(
            reinterpret_cast<const int32_t*>(rowPtrA), reinterpret_cast<const int32_t*>(colIndA),
            reinterpret_cast<const half*>(valA), reinterpret_cast<const int32_t*>(rowPtrB),
            reinterpret_cast<const int32_t*>(colIndB), reinterpret_cast<const half*>(valB),
            reinterpret_cast<int32_t*>(cursors), reinterpret_cast<const int64_t*>(productOffsets),
            reinterpret_cast<int32_t*>(candidateCols), reinterpret_cast<half*>(candidateVals),
            reinterpret_cast<int32_t*>(uniqueCounts), reinterpret_cast<int32_t*>(status), shape);
    }
}

void SpspmmCopy(const void* productOffsets, const void* candidateCols, const void* candidateVals,
                const void* uniqueCounts, const void* rowPtrC, void* indexC, void* valueC,
                int64_t total, SpspmmShape shape, int dtype, void* stream)
{
    if (shape.m == 0 || total == 0) {
        return;
    }
    const uint32_t blocks = BlocksFor(static_cast<uint64_t>(total));
    if (dtype == 0) {
        spspmm_copy_f32<<<blocks, nullptr, stream>>>(
            reinterpret_cast<const int64_t*>(productOffsets),
            reinterpret_cast<const int32_t*>(candidateCols),
            reinterpret_cast<const float*>(candidateVals),
            reinterpret_cast<const int32_t*>(uniqueCounts), reinterpret_cast<const int64_t*>(rowPtrC),
            reinterpret_cast<int64_t*>(indexC), reinterpret_cast<float*>(valueC), total, shape);
    } else {
        spspmm_copy_f16<<<blocks, nullptr, stream>>>(
            reinterpret_cast<const int64_t*>(productOffsets),
            reinterpret_cast<const int32_t*>(candidateCols),
            reinterpret_cast<const half*>(candidateVals),
            reinterpret_cast<const int32_t*>(uniqueCounts), reinterpret_cast<const int64_t*>(rowPtrC),
            reinterpret_cast<int64_t*>(indexC), reinterpret_cast<half*>(valueC), total, shape);
    }
}

}  // namespace opsgnn

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "spspmm_kernel.h"

#include "../common/spspmm_common.h"

#include "common/aiv_launch_utils.h"

namespace {

uint32_t BlocksFor(uint64_t workItems)
{
    const uint64_t blocks = (workItems + SPSSPMM_THREADS - 1) / SPSSPMM_THREADS;
    return ops_gnn::sparse::ResolveAivCoreNum(static_cast<uint32_t>(blocks == 0 ? 1 : blocks));
}

__attribute__((aiv)) __global__ __aicore__ void SpspmmConvertKernel(
    __gm__ const int64_t* rowA, __gm__ const int64_t* colA, __gm__ const int64_t* rowB,
    __gm__ const int64_t* colB, __gm__ int32_t* rowA32, __gm__ int32_t* colA32,
    __gm__ int32_t* rowB32, __gm__ int32_t* colB32, SpspmmShape shape, __gm__ int32_t* error)
{
    AscendC::Simt::VF_CALL<ConvertIndicesSimt>(AscendC::Simt::Dim3{SPSSPMM_THREADS}, rowA, colA, rowB,
                                               colB, rowA32, colA32, rowB32, colB32, shape, error);
}

__attribute__((aiv)) __global__ __aicore__ void SpspmmHistogramKernel(
    __gm__ const int32_t* rows, __gm__ int32_t* counts, int32_t edges, int32_t limit,
    __gm__ int32_t* error)
{
    AscendC::Simt::VF_CALL<HistogramSimt>(AscendC::Simt::Dim3{SPSSPMM_THREADS}, rows, counts, edges,
                                          limit, error);
}

__attribute__((aiv)) __global__ __aicore__ void SpspmmValidateKernel(
    __gm__ const int32_t* rowPtr, __gm__ const int32_t* colInd, int32_t rows, int32_t nnz,
    __gm__ int32_t* error)
{
    AscendC::Simt::VF_CALL<ValidateRowsSimt>(AscendC::Simt::Dim3{SPSSPMM_THREADS}, rowPtr, colInd,
                                             rows, nnz, error);
}

template <typename CountT, typename OffsetT>
__attribute__((aiv)) __global__ __aicore__ void SpspmmScanLocalKernel(
    __gm__ const CountT* counts, __gm__ OffsetT* offsets, __gm__ OffsetT* chunkSums,
    int32_t count, int32_t chunkSize, int32_t numChunks)
{
    AscendC::Simt::VF_CALL<ScanLocalSimt<CountT, OffsetT>>(
        AscendC::Simt::Dim3{SPSSPMM_THREADS}, counts, offsets, chunkSums, count, chunkSize,
        numChunks);
}

template <typename OffsetT>
__attribute__((aiv)) __global__ __aicore__ void SpspmmScanChunksKernel(
    __gm__ const OffsetT* chunkSums, __gm__ OffsetT* chunkPrefix, __gm__ OffsetT* offsets,
    int32_t count, int32_t numChunks)
{
    AscendC::Simt::VF_CALL<ScanChunksSimt<OffsetT>>(AscendC::Simt::Dim3{SPSSPMM_THREADS},
                                                    chunkSums, chunkPrefix, offsets, count, numChunks);
}

template <typename OffsetT>
__attribute__((aiv)) __global__ __aicore__ void SpspmmScanAddKernel(
    __gm__ OffsetT* offsets, __gm__ const OffsetT* chunkPrefix, int32_t count, int32_t chunkSize,
    int32_t numChunks)
{
    AscendC::Simt::VF_CALL<ScanAddSimt<OffsetT>>(AscendC::Simt::Dim3{SPSSPMM_THREADS}, offsets,
                                                 chunkPrefix, count, chunkSize, numChunks);
}

__attribute__((aiv)) __global__ __aicore__ void SpspmmProductCountKernel(
    __gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA, __gm__ const int32_t* rowPtrB,
    __gm__ int64_t* counts, SpspmmShape shape)
{
    AscendC::Simt::VF_CALL<ProductCountSimt>(AscendC::Simt::Dim3{SPSSPMM_THREADS}, rowPtrA, colIndA,
                                             rowPtrB, counts, shape);
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void SpspmmComputeKernel(
    __gm__ const int32_t* rowPtrA, __gm__ const int32_t* colIndA, __gm__ const T* valA,
    __gm__ const int32_t* rowPtrB, __gm__ const int32_t* colIndB, __gm__ const T* valB,
    __gm__ int32_t* cursors, __gm__ const int64_t* productOffsets, __gm__ int32_t* candidateCols,
    __gm__ T* candidateVals, __gm__ int32_t* uniqueCounts, __gm__ int32_t* error,
    SpspmmShape shape)
{
    AscendC::Simt::VF_CALL<ComputeSimt<T>>(AscendC::Simt::Dim3{SPSSPMM_THREADS}, rowPtrA, colIndA,
                                           valA, rowPtrB, colIndB, valB, cursors, productOffsets,
                                           candidateCols, candidateVals, uniqueCounts, error,
                                           shape.m);
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void SpspmmCopyKernel(
    __gm__ const int64_t* productOffsets, __gm__ const int32_t* candidateCols,
    __gm__ const T* candidateVals, __gm__ const int32_t* uniqueCounts,
    __gm__ const int64_t* rowPtrC, __gm__ int64_t* indexC, __gm__ T* valueC, int64_t total,
    SpspmmShape shape)
{
    AscendC::Simt::VF_CALL<CopySimt<T>>(AscendC::Simt::Dim3{SPSSPMM_THREADS}, productOffsets,
                                        candidateCols, candidateVals, uniqueCounts, rowPtrC, indexC,
                                        valueC, total, shape);
}

template <typename CountT, typename OffsetT>
void LaunchScan(const CountT* counts, OffsetT* offsets, OffsetT* chunkSums, OffsetT* chunkPrefix,
                int32_t count, void* stream)
{
    const int32_t numChunks = (count + SPSSPMM_SCAN_CHUNK - 1) / SPSSPMM_SCAN_CHUNK;
    if (numChunks <= 0) {
        return;
    }
    const uint32_t blocks = BlocksFor(static_cast<uint64_t>(numChunks));
    SpspmmScanLocalKernel<CountT, OffsetT><<<blocks, nullptr, stream>>>(
        counts, offsets, chunkSums, count, SPSSPMM_SCAN_CHUNK, numChunks);
    SpspmmScanChunksKernel<OffsetT><<<1, nullptr, stream>>>(chunkSums, chunkPrefix, offsets, count,
                                                            numChunks);
    SpspmmScanAddKernel<OffsetT><<<blocks, nullptr, stream>>>(offsets, chunkPrefix, count,
                                                              SPSSPMM_SCAN_CHUNK, numChunks);
}

}  // namespace

namespace opsgnn {

void SpspmmConvertIndices(const void* rowA, const void* colA, const void* rowB, const void* colB,
                          void* rowA32, void* colA32, void* rowB32, void* colB32,
                          SpspmmShape shape, void* error, void* stream)
{
    const uint64_t work = static_cast<uint64_t>(shape.nnzA) + static_cast<uint64_t>(shape.nnzB);
    if (work == 0) {
        return;
    }
    SpspmmConvertKernel<<<BlocksFor(work), nullptr, stream>>>(
        reinterpret_cast<const int64_t*>(rowA), reinterpret_cast<const int64_t*>(colA),
        reinterpret_cast<const int64_t*>(rowB), reinterpret_cast<const int64_t*>(colB),
        reinterpret_cast<int32_t*>(rowA32), reinterpret_cast<int32_t*>(colA32),
        reinterpret_cast<int32_t*>(rowB32), reinterpret_cast<int32_t*>(colB32), shape,
        reinterpret_cast<int32_t*>(error));
}

void SpspmmHistogram(const void* rows, void* counts, int32_t edges, int32_t limit,
                     void* error, void* stream)
{
    if (edges == 0) {
        return;
    }
    SpspmmHistogramKernel<<<BlocksFor(static_cast<uint64_t>(edges)), nullptr, stream>>>(
        reinterpret_cast<const int32_t*>(rows), reinterpret_cast<int32_t*>(counts), edges, limit,
        reinterpret_cast<int32_t*>(error));
}

void SpspmmValidateRows(const void* rowPtr, const void* colInd, int32_t rows, int32_t nnz,
                        void* error, void* stream)
{
    if (rows == 0) {
        return;
    }
    SpspmmValidateKernel<<<BlocksFor(static_cast<uint64_t>(rows)), nullptr, stream>>>(
        reinterpret_cast<const int32_t*>(rowPtr), reinterpret_cast<const int32_t*>(colInd), rows, nnz,
        reinterpret_cast<int32_t*>(error));
}

void SpspmmScanInt32(const void* counts, void* offsets, void* chunkSums, void* chunkPrefix,
                     int32_t count, void* stream)
{
    LaunchScan(reinterpret_cast<const int32_t*>(counts), reinterpret_cast<int32_t*>(offsets),
               reinterpret_cast<int32_t*>(chunkSums), reinterpret_cast<int32_t*>(chunkPrefix), count, stream);
}

void SpspmmScanInt64(const void* counts, void* offsets, void* chunkSums, void* chunkPrefix,
                     int32_t count, void* stream)
{
    LaunchScan(reinterpret_cast<const int64_t*>(counts), reinterpret_cast<int64_t*>(offsets),
               reinterpret_cast<int64_t*>(chunkSums), reinterpret_cast<int64_t*>(chunkPrefix), count, stream);
}

void SpspmmScanInt32ToInt64(const void* counts, void* offsets, void* chunkSums, void* chunkPrefix,
                            int32_t count, void* stream)
{
    LaunchScan(reinterpret_cast<const int32_t*>(counts), reinterpret_cast<int64_t*>(offsets),
               reinterpret_cast<int64_t*>(chunkSums), reinterpret_cast<int64_t*>(chunkPrefix), count, stream);
}

void SpspmmProductCount(const void* rowPtrA, const void* colIndA, const void* rowPtrB,
                        void* counts, SpspmmShape shape, void* stream)
{
    if (shape.m == 0) {
        return;
    }
    SpspmmProductCountKernel<<<BlocksFor(static_cast<uint64_t>(shape.m)), nullptr, stream>>>(
        reinterpret_cast<const int32_t*>(rowPtrA), reinterpret_cast<const int32_t*>(colIndA),
        reinterpret_cast<const int32_t*>(rowPtrB), reinterpret_cast<int64_t*>(counts), shape);
}

void SpspmmCompute(const void* rowPtrA, const void* colIndA, const void* valA,
                   const void* rowPtrB, const void* colIndB, const void* valB, void* cursors,
                   const void* productOffsets, void* candidateCols, void* candidateVals,
                   void* uniqueCounts, void* error, SpspmmShape shape, int dtype, void* stream)
{
    if (shape.m == 0) {
        return;
    }
    const uint32_t blocks = BlocksFor(static_cast<uint64_t>(shape.m));
    if (dtype == 0) {
        SpspmmComputeKernel<float><<<blocks, nullptr, stream>>>(
            reinterpret_cast<const int32_t*>(rowPtrA), reinterpret_cast<const int32_t*>(colIndA),
            reinterpret_cast<const float*>(valA), reinterpret_cast<const int32_t*>(rowPtrB),
            reinterpret_cast<const int32_t*>(colIndB), reinterpret_cast<const float*>(valB),
            reinterpret_cast<int32_t*>(cursors), reinterpret_cast<const int64_t*>(productOffsets),
            reinterpret_cast<int32_t*>(candidateCols), reinterpret_cast<float*>(candidateVals),
            reinterpret_cast<int32_t*>(uniqueCounts), reinterpret_cast<int32_t*>(error), shape);
    } else {
        SpspmmComputeKernel<half><<<blocks, nullptr, stream>>>(
            reinterpret_cast<const int32_t*>(rowPtrA), reinterpret_cast<const int32_t*>(colIndA),
            reinterpret_cast<const half*>(valA), reinterpret_cast<const int32_t*>(rowPtrB),
            reinterpret_cast<const int32_t*>(colIndB), reinterpret_cast<const half*>(valB),
            reinterpret_cast<int32_t*>(cursors), reinterpret_cast<const int64_t*>(productOffsets),
            reinterpret_cast<int32_t*>(candidateCols), reinterpret_cast<half*>(candidateVals),
            reinterpret_cast<int32_t*>(uniqueCounts), reinterpret_cast<int32_t*>(error), shape);
    }
}

void SpspmmCopy(const void* productOffsets, const void* candidateCols, const void* candidateVals,
                const void* uniqueCounts, const void* rowPtrC, void* indexC, void* valueC,
                int64_t total, SpspmmShape shape, int dtype, void* stream)
{
    if (shape.m == 0 || total == 0) {
        return;
    }
    const uint32_t blocks = BlocksFor(static_cast<uint64_t>(shape.m));
    if (dtype == 0) {
        SpspmmCopyKernel<float><<<blocks, nullptr, stream>>>(
            reinterpret_cast<const int64_t*>(productOffsets),
            reinterpret_cast<const int32_t*>(candidateCols), reinterpret_cast<const float*>(candidateVals),
            reinterpret_cast<const int32_t*>(uniqueCounts), reinterpret_cast<const int64_t*>(rowPtrC),
            reinterpret_cast<int64_t*>(indexC), reinterpret_cast<float*>(valueC), total, shape);
    } else {
        SpspmmCopyKernel<half><<<blocks, nullptr, stream>>>(
            reinterpret_cast<const int64_t*>(productOffsets),
            reinterpret_cast<const int32_t*>(candidateCols), reinterpret_cast<const half*>(candidateVals),
            reinterpret_cast<const int32_t*>(uniqueCounts), reinterpret_cast<const int64_t*>(rowPtrC),
            reinterpret_cast<int64_t*>(indexC), reinterpret_cast<half*>(valueC), total, shape);
    }
}

}  // namespace opsgnn

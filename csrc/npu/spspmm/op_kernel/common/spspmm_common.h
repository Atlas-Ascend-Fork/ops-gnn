/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>

// Public contract shared by the arch22 (A2/A3) and arch35 (950) kernels. The
// Host schedules one pipeline for both architectures: narrow and validate the
// COO indices, build both CSR row pointers, count the candidate products, merge
// them per row and compact the result. Only the device side differs.

// Prefix sums run in three passes so that no kernel has to serialise over every
// row: a per-chunk scan, a scan of the chunk totals, then the chunk offsets.
constexpr int32_t SPSSPMM_SCAN_CHUNK = 8192;

// Candidate products are materialised per row before the unique columns are
// compacted. The bound is data dependent, so the host refuses an input whose
// product count would need more than this many slots instead of failing inside
// the allocator.
constexpr int64_t SPSSPMM_MAX_PRODUCTS = 1LL << 30;

// The sticky status flags occupy one 32-byte block: the arch22 kernels raise them
// with an aligned atomic add of the whole block, so several AIVs can raise a flag
// without overwriting each other. Word 0 is the input/limit error the host reads
// back; word 1 records that the arch22 merge met a row longer than its UB cursor
// capacity, which makes its spill pass run instead of returning immediately.
constexpr int64_t SPSSPMM_ERROR_WORDS = 8;
constexpr int64_t SPSSPMM_ERROR_FLAG_WORD = 0;
constexpr int64_t SPSSPMM_LONG_ROW_WORD = 1;

// A chunk total occupies one 32-byte block of the scan workspace: eight int32 or
// four int64 elements. The arch22 prefix sum stores one total per block, so
// neighbouring chunks never share a block that two AIVs rewrite concurrently; the
// arch35 scan writes the totals contiguously and ignores the padding.
constexpr int64_t SPSSPMM_CHUNK_SUM_STRIDE_INT32 = 8;
constexpr int64_t SPSSPMM_CHUNK_SUM_STRIDE_INT64 = 4;

// The public COO indices are int64; the host narrows them to int32 once it has
// validated that every dimension fits, and the device kernels only see int32.
struct SpspmmShape {
    int32_t m;
    int32_t k;
    int32_t n;
    int32_t nnzA;
    int32_t nnzB;
};

namespace opsgnn {

// int64 COO indices (row block followed by the column block) -> int32 pairs,
// with the range and the row-major order validated on the device.
void SpspmmConvertIndices(const void* rowA, const void* colA, const void* rowB, const void* colB,
                          void* rowA32, void* colA32, void* rowB32, void* colB32,
                          SpspmmShape shape, void* status, void* stream);

// Row histogram of a sorted row-id array; the result is prefix-summed into CSR.
void SpspmmHistogram(const void* rows, void* counts, int32_t edges, int32_t limit,
                     void* status, void* stream);

// Rejects a row whose columns are not strictly ascending (duplicate coordinates).
void SpspmmValidateRows(const void* rowPtr, const void* colInd, int32_t rows, int32_t nnz,
                        void* status, void* stream);

// Exclusive prefix sum over ``count`` entries, closing boundary included in the
// offsets array (``count + 1`` elements).
void SpspmmScanInt32(const void* counts, void* offsets, void* chunkSums, void* chunkPrefix,
                     int32_t count, void* stream);
void SpspmmScanInt64(const void* counts, void* offsets, void* chunkSums, void* chunkPrefix,
                     int32_t count, void* stream);
void SpspmmScanInt32ToInt64(const void* counts, void* offsets, void* chunkSums, void* chunkPrefix,
                            int32_t count, void* stream);

// Candidate products per output row, as an int64 upper bound.
void SpspmmProductCount(const void* rowPtrA, const void* colIndA, const void* rowPtrB,
                        void* counts, SpspmmShape shape, void* stream);

// dtype ids follow the rest of the extension: float=0, half=1.
void SpspmmCompute(const void* rowPtrA, const void* colIndA, const void* valA,
                   const void* rowPtrB, const void* colIndB, const void* valB, void* cursors,
                   const void* productOffsets, void* candidateCols, void* candidateVals,
                   void* uniqueCounts, void* status, SpspmmShape shape, int dtype, void* stream);

void SpspmmCopy(const void* productOffsets, const void* candidateCols, const void* candidateVals,
                const void* uniqueCounts, const void* rowPtrC, void* indexC, void* valueC,
                int64_t total, SpspmmShape shape, int dtype, void* stream);

}  // namespace opsgnn

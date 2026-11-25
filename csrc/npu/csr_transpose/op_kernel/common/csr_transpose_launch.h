/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_CSR_TRANSPOSE_LAUNCH_H
#define OPS_GNN_CSR_TRANSPOSE_LAUNCH_H

#include <acl/acl.h>
#include <cstdint>

namespace opsgnn {

// Slots between two consecutive AIV totals in the `blockTotals` workspace: one
// 32-byte block per AIV. A copy only owns the block it lands in, so neighbouring
// totals must not share one.
constexpr int64_t kTransposeBlockStride = 32 / sizeof(int64_t);

// Elements in one 32-byte global block. Every range that reaches global memory
// starts on such a boundary, and the column split never inserts a boundary
// closer than that, so the range of one AIV never shares a block with another.
constexpr int64_t kTransposeAlignment = kTransposeBlockStride;

// Histogram the column ids into `counts` (length numColumns), one contiguous
// column range per AIV, and publish each range total in `blockTotals`.
void CsrTransposeCount(const int64_t* indices, int64_t* counts, int64_t* blockTotals, int64_t numEdges,
                       int64_t numColumns, uint32_t blockNum, uint32_t ubBytes, aclrtStream stream);

// Exclusive prefix sum of `counts` into `outIndptr` (length numColumns + 1).
void CsrTransposeIndptr(const int64_t* counts, const int64_t* blockTotals, int64_t* outIndptr,
                        int64_t numColumns, uint32_t blockNum, aclrtStream stream);

// Append every row id at the cursor of its column and produce `outIndices`.
// The row id of an edge is taken from `indptr` while the edges are replayed.
void CsrTransposeWrite(const int64_t* indptr, const int64_t* indices, const int64_t* outIndptr,
                       int64_t* outIndices, int64_t numRows, int64_t numEdges, int64_t numColumns,
                       uint32_t blockNum, uint32_t ubBytes, aclrtStream stream);

}  // namespace opsgnn

#endif  // OPS_GNN_CSR_TRANSPOSE_LAUNCH_H

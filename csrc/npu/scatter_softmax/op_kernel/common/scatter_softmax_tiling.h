/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_SOFTMAX_TILING_H
#define OPS_GNN_SCATTER_SOFTMAX_TILING_H

#include <cstdint>

// Largest number of trailing features a single vector tile covers. Every tile
// start is a multiple of this constant, so a GM atomic destination keeps the
// 32-byte alignment MTE3 requires. The host shrinks it for narrow rows.
constexpr uint32_t SCATTER_SOFTMAX_MAX_COL_TILE = 2048;

// `colTile` is also the stride between the per-row UB slices of a batch, so it
// stays a multiple of this constant: 16 elements are 32 bytes of fp16/bf16 and
// 64 bytes of fp32, which keeps every slice base 32-byte aligned.
constexpr uint32_t SCATTER_SOFTMAX_ROW_ALIGN = 16;

// One fp32 workspace row is padded to SCATTER_SOFTMAX_WORKSPACE_ALIGN elements
// (32 bytes) so that every atomic write is a whole number of 32-byte blocks.
constexpr uint64_t SCATTER_SOFTMAX_WORKSPACE_ALIGN = 8;

// Neutral value of the fp32 per-group maximum. The host fills the workspace
// with the same constant, so a padded feature lane can never win a maximum.
constexpr float SCATTER_SOFTMAX_NEUTRAL_MAXIMUM = -3.4028234663852886e38F;

// Rows a core buffers at once, and the UB budget those row buffers may use.
// A batch hides the GM round trip of one row behind the transfers of the rest,
// which is what keeps narrow rows off the "one synchronous row at a time" floor.
constexpr uint32_t SCATTER_SOFTMAX_MAX_ROW_BATCH = 8;
constexpr uint64_t SCATTER_SOFTMAX_ROW_BUDGET_BYTES = 80 * 1024;

// Upper bound of the per-core fp32 workspace copy. When the whole reduction
// workspace fits here it is kept in UB for the duration of a pass: the
// reduction phases accumulate in place and flush once per core, and every pass
// that reads the workspace reads it once instead of once per row.
//
// The per-row read is what dominates small group counts. A million rows that
// all reduce into one group re-read the same 512-byte workspace line a million
// times, and on 910B4 that costs about 46 ms per pass - more than a bandwidth
// bound pass ever would - while the same pass with 16 groups, whose rows hit 16
// different lines, costs 4 ms. Caching the workspace in UB removes the
// repeated read; the arithmetic was never the problem.
//
// The budget stays well below the row budget on purpose: a workspace copy that
// eats into it shrinks `rowBatch`, and a batch of eight rows is worth more than
// skipping the atomic writes.
constexpr uint64_t SCATTER_SOFTMAX_ACC_BUDGET_BYTES = 16 * 1024;

enum ScatterSoftmaxDType {
    SCATTER_SOFTMAX_F16 = 0,
    SCATTER_SOFTMAX_F32 = 1,
    SCATTER_SOFTMAX_BF16 = 2,
};

// Canonical shape decomposition of a contiguous `src` reduced along `dim`:
//   batch      = prod(src.size(0 .. dim-1))
//   dimLength  = src.size(dim)                indexed by `index`
//   after      = prod(src.size(dim+1 .. -1))
// Element (b, j, a) therefore lives at (b * dimLength + j) * after + a.
//
// The `out*` fields describe the reduced output of the grouped log-sum-exp,
// whose rows are (batch, group) pairs instead of (batch, dimLength) rows. The
// softmax family keeps outRows == rows, so the shared driver can walk either
// layout with the same code.
struct ScatterSoftmaxTilingData {
    uint64_t batch;
    uint64_t dimLength;
    uint64_t groupCount;   // dim_size
    uint64_t after;
    uint64_t afterPadded;  // after rounded up to SCATTER_SOFTMAX_WORKSPACE_ALIGN
    uint64_t rows;         // batch * dimLength
    uint64_t srcNumel;     // rows * after
    uint64_t workspaceElems;  // batch * groupCount * afterPadded
    uint64_t rowChunk;     // consecutive rows owned by one core
    uint32_t colTile;      // vector tile along the feature axis
    uint32_t blockNum;
    uint32_t rowBatch;     // consecutive rows buffered per iteration (1..8)
    uint32_t localAccum;   // 1: reduction workspace resident in UB for a pass
    uint64_t outRows;      // batch * groupCount (log-sum-exp output rows)
    uint64_t outRowChunk;  // output rows owned by one core
    uint32_t outBlockNum;
    uint32_t outRowBatch;
};

#endif  // OPS_GNN_SCATTER_SOFTMAX_TILING_H

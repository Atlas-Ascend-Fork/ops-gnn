/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SEGMENT_LOGSUMEXP_TILING_H
#define OPS_GNN_SEGMENT_LOGSUMEXP_TILING_H

#include <cstdint>

// Largest number of features a single vector tile reduces. The host shrinks it
// so that a tile stays inside the unified buffer of one core.
constexpr uint64_t SEGMENT_LOGSUMEXP_MAX_COL_TILE = 1024;

// A tile width is always a multiple of this constant (32 bytes of fp32), so
// every vector instruction of the row loop starts on an aligned buffer base.
constexpr uint64_t SEGMENT_LOGSUMEXP_COL_ALIGN = 8;

// The row loop stages several rows per transfer: a single row is only a few
// hundred bytes, so one burst per row would pay the MTE2 latency on every
// iteration. ``*_BATCH_BYTES`` bounds the staging buffer and ``*_MAX_BATCH_ROWS``
// keeps the burst count small for narrow rows.
constexpr uint64_t SEGMENT_LOGSUMEXP_BATCH_BYTES = 16384;
constexpr uint32_t SEGMENT_LOGSUMEXP_MAX_BATCH_ROWS = 64;

// A segment is reduced in ``rowChunks`` pieces so that a few wide segments can
// still fill every core. The split is bounded so that each piece keeps enough
// rows to amortise the extra passes.
constexpr uint32_t SEGMENT_LOGSUMEXP_MAX_ROW_CHUNKS = 64;
constexpr uint64_t SEGMENT_LOGSUMEXP_MIN_ROWS_PER_CHUNK = 16;

// Neutral value of the running maximum. ``-inf`` keeps the reduction correct
// for every finite input, and the shift is clamped before it is used.
constexpr float SEGMENT_LOGSUMEXP_NEGATIVE_INFINITY = -__builtin_inff();

// Clamp bounds of the shift. ``-FLT_MAX`` and ``+FLT_MAX`` keep the shift
// finite while leaving every finite reduction untouched.
constexpr float SEGMENT_LOGSUMEXP_LARGEST_FLOAT = 3.4028234663852886e38F;

// Canonical shape decomposition of ``src``: the CSR rows live along axis 0 and
// ``features`` is the product of the trailing axes. Element (row, a) of the
// source is therefore at ``row * features + a`` and the output of segment ``s``
// at ``s * features + a``.
struct SegmentLogSumExpTilingData {
    uint64_t rows;          // src.size(0), also indptr[-1]
    uint64_t segments;      // indptr.numel() - 1
    uint64_t features;      // product of the trailing axes
    uint64_t colTile;       // features reduced per tile
    uint64_t colTiles;      // ceil(features / colTile)
    uint64_t rowChunks;     // row pieces a segment is split into
    uint64_t baseTasks;     // segments * colTiles, one per (segment, feature window)
    uint64_t tasks;         // baseTasks * rowChunks, one per reduction task
    uint32_t blockNum;      // cores used by the per-chunk phases
    uint32_t baseBlockNum;  // cores used by the per-window phases
};

#endif  // OPS_GNN_SEGMENT_LOGSUMEXP_TILING_H

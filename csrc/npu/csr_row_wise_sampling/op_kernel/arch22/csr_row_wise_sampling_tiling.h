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

#ifndef CSR_ROW_WISE_SAMPLING_TILING_H
#define CSR_ROW_WISE_SAMPLING_TILING_H

#include <cstdint>

// Tiling layout for the multi-core weighted (probability/mask) CSR row-wise
// sampling kernels: the per-row pick-count helper
// (csr_row_wise_sampling_num_picks_*) and the main sampler
// (csr_row_wise_sampling_*). Field order must match the Init() of both
// kernels. Same design as the uniform/biased v2 kernels (ADR-0011): the host
// derives a nnz-balanced row_split and a per-row out_starts prefix from the
// helper's output, so blocks write disjoint output ranges and every buffer
// is allocated exactly.
constexpr uint32_t kTilingHeaderWordsWeighted = 9;

struct CsrRowWiseSamplingWeightedTiling {
  uint32_t num_rows;        // number of seed rows to sample
  uint32_t num_samples;     // fanout (0 when select_all)
  uint32_t replace;         // 1 = with replacement
  uint32_t has_data;        // 1 = CSR data array present
  uint32_t seed;            // base RNG seed for this launch
  uint32_t select_all;      // 1 = num_samples == -1 (pick every prob > 0)
  uint32_t num_total_rows;  // total rows of the CSR matrix (bounds check)
  uint32_t ub_available;    // per-core UB budget in bytes (runtime query)
  uint32_t write_row_pos;   // 1 = out_rows carries the row position (fused)
};

// UB budget reserved for the tiling/meta staging area.
constexpr uint32_t kUbReservedBytesWeighted = 2 * 1024;

// Upper bound on launched blocks. row_split / out_starts are built and
// uploaded by the host from that many partitions, so the host clamps the
// queried vector-core count to this cap; past the cap the op degrades
// gracefully (fewer blocks than cores).
constexpr uint32_t kMaxSamplingBlocksWeighted = 64;

// RNG constants (xorshift32 with Knuth golden-ratio row hashing) — keep in
// sync with the other sampling kernels.
constexpr uint32_t kGoldenRatioHashWeighted = 2654435761u;    // 2^32 / phi
constexpr uint32_t kGoldenRatioOffsetWeighted = 0x9e3779b9u;  // frac
constexpr uint32_t kRngFallbackSeedWeighted = 0x12345678u;    // nonzero

// UB window equation (single source of truth, shared by kernel and host).
// Window instances: idx window queue (db) + data window queue (db) + output
// queue (db) + probability buffer + cdf buffer + no-replace generation
// markers + 3 output staging buffers. Meta staging (out_ptr) is a fixed byte
// budget below.
constexpr uint32_t kQueueDepthWeighted = 2;   // double buffering
constexpr uint32_t kMetaRowsWeighted = 256;   // meta staging rows per flush
constexpr uint32_t kWindowInstancesWeighted =
    2 * kQueueDepthWeighted + 2 * kQueueDepthWeighted +
    2 * kQueueDepthWeighted + 1 + 1 + 1 + 3;
constexpr uint32_t kMetaUbReserveWeighted = (kMetaRowsWeighted + 1) * 8;

#endif  // CSR_ROW_WISE_SAMPLING_TILING_H

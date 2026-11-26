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

namespace opsgnn {

// Reduce codes mirror the Python interface (sum/mean/min/max).
enum CoalesceReduce : uint32_t {
    kCoalesceSum = 0,
    kCoalesceMean = 1,
    kCoalesceMin = 2,
    kCoalesceMax = 3,
};

enum CoalesceDtype : uint32_t {
    kCoalesceFloat32 = 0,
    kCoalesceFloat16 = 1,
};

// arch22-private Host/Kernel protocol. The kernel reduces contiguous sorted
// segments delimited by `indptr`; the Host has already sorted the keys and
// gathered edge_attr into that order, so segment `g` owns the contiguous rows
// [indptr[g], indptr[g+1]) of the (E, C) source. Rebuild Host and kernel
// together on any change.
struct CoalesceTilingData {
    uint64_t E;          // number of input edges
    uint64_t U;          // number of unique (deduplicated) edges
    uint64_t C;          // flattened feature width of edge_attr (0 when absent)
    uint64_t srcNumel;   // E * C
    uint64_t outNumel;   // U * C
    uint32_t reduce;     // CoalesceReduce
    uint32_t dtype;      // CoalesceDtype
    uint32_t usedBlockNum;
    uint32_t tileRows;   // rows loaded per UB tile
    uint32_t width;      // C padded to a 32-byte vector width
    uint32_t hasAttr;    // 0 => compaction only, no reduction
};

} // namespace opsgnn

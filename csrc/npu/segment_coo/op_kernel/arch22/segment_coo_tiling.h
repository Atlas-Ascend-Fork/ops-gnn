/*
 * Copyright (c) 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>
#include <cstddef>

enum SegmentCooReduce : int32_t {
    SEGMENT_COO_SUM = 0,
    SEGMENT_COO_MEAN = 1,
    SEGMENT_COO_MIN = 2,
    SEGMENT_COO_MAX = 3,
};

enum SegmentCooDtype : int32_t {
    SEGMENT_COO_FLOAT32 = 0,
    SEGMENT_COO_FLOAT16 = 1,
    SEGMENT_COO_BFLOAT16 = 2,
    SEGMENT_COO_INT8 = 3,
    SEGMENT_COO_UINT8 = 4,
    SEGMENT_COO_INT32 = 5,
    SEGMENT_COO_INT64 = 6,
};

namespace opsgnn {

constexpr int64_t kSegmentCooPackedElementLimit = 2147483647LL;
constexpr int64_t kSegmentCooPackedChannelAlignment = 8;

// Stable bit values shared by Host predicates and device numerical policies.
enum SegmentSourceFlag : uint32_t {
    kSourceRangeProven = 1U << 0,
    kCpuOrderedFloat = 1U << 1,
    kRepairExtrema = 1U << 2,
};
enum SegmentIndexMode : uint32_t { kSortedIndex = 1, kUniformIndex = 2 };
enum SegmentBatchMode : uint32_t { kTaskReduction = 0, kValueBatch = 1 };

enum SegmentReduce : uint32_t {
    SegmentSum = SEGMENT_COO_SUM,
    SegmentMean = SEGMENT_COO_MEAN,
    SegmentMin = SEGMENT_COO_MIN,
    SegmentMax = SEGMENT_COO_MAX
};
enum SegmentDType : uint32_t {
    SegmentFloat32 = SEGMENT_COO_FLOAT32,
    SegmentFloat16 = SEGMENT_COO_FLOAT16,
    SegmentBFloat16 = SEGMENT_COO_BFLOAT16,
    SegmentInt8 = SEGMENT_COO_INT8,
    SegmentUInt8 = SEGMENT_COO_UINT8,
    SegmentInt32 = SEGMENT_COO_INT32,
    SegmentInt64 = SEGMENT_COO_INT64
};
// arch22-private Host/device protocol; rebuild the kernel and Host together.
struct SegmentTilingData {
    uint64_t srcNumel, outNumel, E_1, M, K, nSegments, indexNumel;
    uint64_t totalTasks, featureTiles, totalUnits;
    // Reserved, zero-filled ABI slots. Host expands broadcast index prefixes.
    uint64_t indexPrefixSizes[8], indexPrefixStrides[8];
    uint64_t indexBatchStride, indexRowStride;
    uint32_t reduceCode, dtypeCode, usedBlockNum, hasInitial, hasArg;
    uint32_t tileK, tileRows, outputGroup, indexMode, indexPrefixRank;
    // arch22: 0=ordinary reduction, 1=value batch.
    uint32_t batchReduce;
    // SegmentSourceFlag mask; never interpret the entire word as one predicate.
    uint32_t sourceFlags;
};

// Preserve the launch ABI. Intentional changes require a matching Host/device rebuild.
static_assert(sizeof(SegmentTilingData) == 272, "unexpected segment tiling size");
static_assert(alignof(SegmentTilingData) == 8, "unexpected segment tiling alignment");
static_assert(offsetof(SegmentTilingData, srcNumel) == 0, "unexpected srcNumel offset");
static_assert(offsetof(SegmentTilingData, outNumel) == 8, "unexpected outNumel offset");
static_assert(offsetof(SegmentTilingData, E_1) == 16, "unexpected E_1 offset");
static_assert(offsetof(SegmentTilingData, M) == 24, "unexpected M offset");
static_assert(offsetof(SegmentTilingData, K) == 32, "unexpected K offset");
static_assert(offsetof(SegmentTilingData, nSegments) == 40, "unexpected nSegments offset");
static_assert(offsetof(SegmentTilingData, indexNumel) == 48, "unexpected indexNumel offset");
static_assert(offsetof(SegmentTilingData, totalTasks) == 56, "unexpected totalTasks offset");
static_assert(offsetof(SegmentTilingData, featureTiles) == 64, "unexpected featureTiles offset");
static_assert(offsetof(SegmentTilingData, totalUnits) == 72, "unexpected totalUnits offset");
static_assert(offsetof(SegmentTilingData, indexPrefixSizes) == 80, "unexpected indexPrefixSizes offset");
static_assert(offsetof(SegmentTilingData, indexPrefixStrides) == 144, "unexpected indexPrefixStrides offset");
static_assert(offsetof(SegmentTilingData, indexBatchStride) == 208, "unexpected indexBatchStride offset");
static_assert(offsetof(SegmentTilingData, indexRowStride) == 216, "unexpected indexRowStride offset");
static_assert(offsetof(SegmentTilingData, reduceCode) == 224, "unexpected reduceCode offset");
static_assert(offsetof(SegmentTilingData, dtypeCode) == 228, "unexpected dtypeCode offset");
static_assert(offsetof(SegmentTilingData, usedBlockNum) == 232, "unexpected usedBlockNum offset");
static_assert(offsetof(SegmentTilingData, hasInitial) == 236, "unexpected hasInitial offset");
static_assert(offsetof(SegmentTilingData, hasArg) == 240, "unexpected hasArg offset");
static_assert(offsetof(SegmentTilingData, tileK) == 244, "unexpected tileK offset");
static_assert(offsetof(SegmentTilingData, tileRows) == 248, "unexpected tileRows offset");
static_assert(offsetof(SegmentTilingData, outputGroup) == 252, "unexpected outputGroup offset");
static_assert(offsetof(SegmentTilingData, indexMode) == 256, "unexpected indexMode offset");
static_assert(offsetof(SegmentTilingData, indexPrefixRank) == 260, "unexpected indexPrefixRank offset");
static_assert(offsetof(SegmentTilingData, batchReduce) == 264, "unexpected batchReduce offset");

static_assert(offsetof(SegmentTilingData, sourceFlags) == 268, "unexpected sourceFlags offset");

// Host-only selection; these values are not part of the device ABI.
enum class SegmentKernelPath : uint32_t { General, Int32Batch, Int64Narrow };
struct SegmentCooPlan {
    SegmentTilingData tiling{};
    SegmentKernelPath path = SegmentKernelPath::General;
    uint32_t availableBlocks = 0; // Extrema repair uses the full device core count.
    int32_t hasOut = 0;           // Caller-owned output lifetime, distinct from hasInitial.
};

} // namespace opsgnn

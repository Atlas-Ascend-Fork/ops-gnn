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
#include "segment_csr_constants.h"

enum SegmentCsrReduce : int32_t {
    SEGMENT_CSR_SUM = 0,
    SEGMENT_CSR_MEAN = 1,
    SEGMENT_CSR_MIN = 2,
    SEGMENT_CSR_MAX = 3,
};

enum SegmentCsrDtype : int32_t {
    SEGMENT_CSR_FLOAT32 = 0,
    SEGMENT_CSR_FLOAT16 = 1,
    SEGMENT_CSR_BFLOAT16 = 2,
    SEGMENT_CSR_INT8 = 3,
    SEGMENT_CSR_UINT8 = 4,
    SEGMENT_CSR_INT32 = 5,
    SEGMENT_CSR_INT64 = 6,
};

constexpr int64_t kSegmentCsrPackedElementLimit = 2147483647LL;
constexpr int64_t kSegmentCsrPackedChannelAlignment = 8;

namespace opsgnn::csr {
enum SegmentReduce : uint32_t { SegmentSum = 0, SegmentMean = 1, SegmentMin = 2, SegmentMax = 3 };
enum SegmentDType : uint32_t {
    SegmentFloat32 = 0,
    SegmentFloat16 = 1,
    SegmentBFloat16 = 2,
    SegmentInt8 = 3,
    SegmentUInt8 = 4,
    SegmentInt32 = 5,
    SegmentInt64 = 6,
    SegmentInt16 = 7
};
} // namespace opsgnn::csr
// arch22-private Host/Kernel protocol; rebuild the kernel and Host together.
struct SegmentCsrTilingData {
    uint64_t srcNumel, outNumel, E_1, M, K, nSegments, indexNumel;
    uint64_t totalTasks, featureTiles, totalUnits;
    uint64_t indexPrefixSizes[8], indexPrefixStrides[8];
    uint64_t indexBatchStride, indexRowStride;
    uint32_t reduceCode, dtypeCode, usedBlockNum, hasInitial, hasArg;
    uint32_t tileK, tileRows, outputGroup, indexMode, indexPrefixRank;
    // Grouped value scheduling flag; arguments use the strict sequential path.
    uint32_t batchReduce;
    // dtype-specific proof: exact INT64 narrowing or finite FP16 |x| <= 6.
    uint32_t boundedSource;
    // Queried device L2 capacity; zero preserves the normal cache policy.
    uint64_t l2Bytes;
};

// Private CSR ABI: rebuild Host and kernel together on changes.
static_assert(sizeof(SegmentCsrTilingData) == 280, "unexpected segment tiling size");
static_assert(alignof(SegmentCsrTilingData) == 8, "unexpected segment tiling alignment");
static_assert(offsetof(SegmentCsrTilingData, srcNumel) == 0, "unexpected srcNumel offset");
static_assert(offsetof(SegmentCsrTilingData, outNumel) == 8, "unexpected outNumel offset");
static_assert(offsetof(SegmentCsrTilingData, E_1) == 16, "unexpected E_1 offset");
static_assert(offsetof(SegmentCsrTilingData, M) == 24, "unexpected M offset");
static_assert(offsetof(SegmentCsrTilingData, K) == 32, "unexpected K offset");
static_assert(offsetof(SegmentCsrTilingData, nSegments) == 40, "unexpected nSegments offset");
static_assert(offsetof(SegmentCsrTilingData, indexNumel) == 48, "unexpected indexNumel offset");
static_assert(offsetof(SegmentCsrTilingData, totalTasks) == 56, "unexpected totalTasks offset");
static_assert(offsetof(SegmentCsrTilingData, featureTiles) == 64, "unexpected featureTiles offset");
static_assert(offsetof(SegmentCsrTilingData, totalUnits) == 72, "unexpected totalUnits offset");
static_assert(offsetof(SegmentCsrTilingData, indexPrefixSizes) == 80, "unexpected indexPrefixSizes offset");
static_assert(offsetof(SegmentCsrTilingData, indexPrefixStrides) == 144, "unexpected indexPrefixStrides offset");
static_assert(offsetof(SegmentCsrTilingData, indexBatchStride) == 208, "unexpected indexBatchStride offset");
static_assert(offsetof(SegmentCsrTilingData, indexRowStride) == 216, "unexpected indexRowStride offset");
static_assert(offsetof(SegmentCsrTilingData, reduceCode) == 224, "unexpected reduceCode offset");
static_assert(offsetof(SegmentCsrTilingData, dtypeCode) == 228, "unexpected dtypeCode offset");
static_assert(offsetof(SegmentCsrTilingData, usedBlockNum) == 232, "unexpected usedBlockNum offset");
static_assert(offsetof(SegmentCsrTilingData, hasInitial) == 236, "unexpected hasInitial offset");
static_assert(offsetof(SegmentCsrTilingData, hasArg) == 240, "unexpected hasArg offset");
static_assert(offsetof(SegmentCsrTilingData, tileK) == 244, "unexpected tileK offset");
static_assert(offsetof(SegmentCsrTilingData, tileRows) == 248, "unexpected tileRows offset");
static_assert(offsetof(SegmentCsrTilingData, outputGroup) == 252, "unexpected outputGroup offset");
static_assert(offsetof(SegmentCsrTilingData, indexMode) == 256, "unexpected indexMode offset");
static_assert(offsetof(SegmentCsrTilingData, indexPrefixRank) == 260, "unexpected indexPrefixRank offset");
static_assert(offsetof(SegmentCsrTilingData, batchReduce) == 264, "unexpected batchReduce offset");

static_assert(offsetof(SegmentCsrTilingData, boundedSource) == 268, "unexpected boundedSource offset");

static_assert(offsetof(SegmentCsrTilingData, l2Bytes) == 272, "unexpected l2Bytes offset");

// Host-only selection, outside the 280-byte device ABI. The Host finalizes all
// geometry before queue submission; the ASC launcher consumes this plan unchanged.
enum class SegmentCsrKernelPath : uint32_t {
    General,
    ShortUniform,
    OrderedUniform,
    OrderedWholeCast,
    OrderedBoundedTree,
    HalfExtreme,
    Int32Pipeline,
    Int32Long,
    Int32Batch,
    Int64Pipeline,
    Int64Long,
    Int64Narrow,
};
struct SegmentCsrPlan {
    SegmentCsrTilingData tiling{};
    SegmentCsrKernelPath path = SegmentCsrKernelPath::General;
    uint32_t argCarryBlocks = 0;
};

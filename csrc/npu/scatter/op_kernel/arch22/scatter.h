/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <acl/acl.h>
#include <cstdint>

namespace opsgnn::scatter_a2 {
enum Reduce : uint32_t { Sum = 0, Mul = 1, Mean = 2, Min = 3, Max = 4 };
enum Dtype : uint32_t { Float16 = 0, BFloat16 = 1, Float32 = 2, Int8 = 3, Int16 = 4, Int32 = 5, UInt8 = 6 };
enum Path : uint32_t { OutputOwner = 0, FeatureOwner = 1, AliasedOrdered = 2, Atomic = 3, BucketTable = 4 };
constexpr uint32_t kDmaBytes = 32;
constexpr uint32_t kFeatureTileElements = 64;
constexpr uint32_t kScalarOutputElements = 64;
constexpr uint32_t kVectorMaskElements = 64;
constexpr uint32_t kBucketConversionElements = 64;
constexpr uint32_t kBucketTableAlignment = 64;
constexpr uint32_t kMaskElementsPerDmaBlock = kDmaBytes * 8;
constexpr uint32_t kMaxAtomicFeatureElements = 2048;
constexpr uint32_t kMaxAtomicSourceRows = 16;
constexpr uint32_t kMaxBucketTableElements = 32768;
constexpr uint32_t kExactFloatIndexLimit = 1U << 24;
constexpr uint32_t kMinAtomicSourceElements = 4096;
constexpr uint32_t kOutputTileElements = 1024;
constexpr uint32_t kUbReserveBytes = 8192;
constexpr uint32_t kBucketSourceRows = 256;
constexpr uint32_t kIndexStatsTileElements = 512;
constexpr uint32_t kIndexStatsElementsPerCore = kDmaBytes / sizeof(int64_t);

// Private Host/device protocol. Element addresses and scheduling products are 64 bit.
struct Tiling {
    uint64_t before, length, after, buckets, srcElements, outElements, indexElements, units;
    uint32_t reduce, dtype, blocks, hasOut, compactIndex, path, atomicWidth, atomicRows;
    uint64_t sizes[8], sourceStrides[8], outputStrides[8], sourceSpan, outputSpan;
    uint32_t rank, dim;
    uint64_t indexStrides[8], indexSpan;
    uint32_t indexMutable, countWidth, argumentRows;
};
static_assert(sizeof(Tiling) == 400, "unexpected Scatter A2 tiling layout");
} // namespace opsgnn::scatter_a2

namespace opsgnn::scatter_a2 {
constexpr uint32_t kGroupMaxFeatureElements = 2048;
constexpr uint32_t kGroupMaxSourceRows = 32;
constexpr uint32_t kGroupDefaultOutputs = 4;
constexpr uint32_t kGroupValidationBytesPerCore = 32;
constexpr uint32_t kGroupValidationWordsPerCore = kGroupValidationBytesPerCore / sizeof(int32_t);

struct GroupTiling {
    uint64_t length, after, buckets, srcElements, outElements, indexElements;
    uint64_t totalTasks, featureTiles, unitFeatureTiles, units;
    uint32_t dtype, reduce, blocks, hasOut, validationBlocks, tileWidth, tileRows, outputGroup;
};
static_assert(sizeof(GroupTiling) == 112, "unexpected grouped Scatter tiling size");
} // namespace opsgnn::scatter_a2

aclError ScatterA2(void *src, int64_t *index, void *out, int64_t *arg, float *workspace, float *argWorkspace,
                   int32_t *counts, const opsgnn::scatter_a2::Tiling &tiling, aclrtStream stream);

// Each core writes [minimum, maximum, 0, 0] in a private 32-byte block.
aclError ScatterA2IndexStats(int64_t *index, uint64_t elements, int64_t *stats, uint32_t blocks, aclrtStream stream);

// Exact comparison: core i writes its boolean result at flags[i * 8].
aclError ScatterA2IndexEqual(int64_t *index, int64_t *snapshot, uint64_t elements, int32_t *flags, uint32_t blocks,
                             aclrtStream stream);

aclError ScatterA2Grouped(void *src, int64_t *rowptr, int64_t *rows, void *out, int64_t *arg, int32_t *validation,
                          const opsgnn::scatter_a2::GroupTiling &tiling, aclrtStream stream);

// The private output is not public until full kernel completion and exact EQ.
aclError ScatterA2FusedPrivateSum(void *src, int64_t *index, int64_t *snapshot, int64_t *rowptr, int64_t *rows,
                                  void *out, int32_t *flags, int32_t *mappedFlags,
                                  const opsgnn::scatter_a2::GroupTiling &tiling, aclrtStream stream);

// Keep GPU flags for the ordinary guarded Owner and mirror to registered Host.
aclError ScatterA2IndexEqualMapped(int64_t *index, int64_t *snapshot, uint64_t elements, int32_t *flags,
                                   int32_t *mappedFlags, uint32_t blocks, aclrtStream stream);

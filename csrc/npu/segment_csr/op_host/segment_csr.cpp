/*
 * Copyright (c) 2026 Starlink_. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * Host layer of the segment_csr family (torch_scatter-compatible) for NPU.
 *
 * Canonical decomposition of contiguous src with dim = indptr.dim() - 1:
 *   E1 = prod(src.size(0 .. dim-1))   batch rows
 *   M  = src.size(dim)                reduce-dim length (indexed by indptr)
 *   K  = prod(src.size(dim+1 ..))     trailing channels
 *   nSeg = indptr.size(-1) - 1
 * indptr is INT64 (task-book requirement); its leading dims broadcast to
 * src's leading dims. All leading dims equal to 1 share one indptr vector
 * (stride 0), otherwise indptr is expanded+materialized per batch row.
 *
 * Launches on the CURRENT NPU stream. arch22 cold validation synchronizes to
 * read small validation predicates; versioned cache hits remain asynchronous.
 * The task benchmark times the full interface, including any validation.
 *
 * dtype facts (register-path item size, vector lane count) come from the
 * kernel header's KindRegisterItemSize / KindVecLanes so this layer never
 * duplicates them.
 */

#include "segment_csr.h"

#include <c10/util/Optional.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>
#include <limits>

#ifdef OPSGNN_DAV_2201
#include "npu/segment_csr/op_kernel/arch22/segment_csr.h"
#include "tiling/platform/platform_ascendc.h"
#include "acl/acl_rt.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include <ATen/MemoryOverlap.h>
#include <ATen/record_function.h>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <mutex>
#include <cmath>
#include <functional>
#include <tuple>
#include <stdexcept>
#else
#include "npu/segment_csr/op_kernel/arch35/segment_csr.h"
#endif

#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#ifdef OPSGNN_DAV_2201
// Arch22 Host planning. Device kernels consume immutable launch geometry.
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 */

namespace {
namespace csr_host {
using namespace opsgnn::csr;
struct PlanInputs {
    int64_t batches, length, channels, segments, indexBatches;
    SegmentCsrReduce reduce;
    SegmentCsrDtype dtype;
    int32_t hasOut, blocks;
    int32_t hasArg = 0, boundedSource = 0;
    uint64_t uniformRows = 0, ubBytes = kArch22UbCapacityBytes, l2Bytes = 0;
    uint32_t tileK = kVectorLanes, tileRows = kDefaultTileRows, outputGroup = 1, batchReduce = 0;
};

inline uint64_t AlignDma(uint64_t bytes) { return (bytes + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes; }
inline uint32_t ItemBytes(uint32_t dtype) {
    return dtype == SegmentFloat32 || dtype == SegmentInt32
               ? 4
               : (dtype == SegmentInt64 ? 8 : (dtype == SegmentInt8 || dtype == SegmentUInt8 ? 1 : 2));
}
inline bool FloatInput(uint32_t dtype) {
    return dtype == SegmentFloat32 || dtype == SegmentFloat16 || dtype == SegmentBFloat16;
}
// Exactly the buffers allocated by SegmentReduceDevice::Init, plus explicit reserve.
inline uint64_t TileBytes(uint32_t dtype, bool hasArg, uint32_t cols, uint32_t rows, bool compensated = false,
                          bool batchReduce = false) {
    const uint32_t item = ItemBytes(dtype);
    uint64_t bytes = AlignDma(uint64_t(rows) * cols * item) + AlignDma(cols * 4ULL) + AlignDma(uint64_t(cols) * item);
    if (FloatInput(dtype) && dtype != SegmentFloat32) bytes += AlignDma(uint64_t(rows) * cols * 4);
    if (!FloatInput(dtype) && dtype != SegmentInt32) bytes += AlignDma(cols * 4ULL);
    if (dtype == SegmentInt16) bytes += AlignDma(cols * 4ULL);
    if (hasArg)
        bytes += AlignDma(cols * 4ULL) + AlignDma(cols * 8ULL) + 2 * AlignDma((cols + 7ULL) / 8);
    else if (!compensated)
        bytes += AlignDma((uint64_t(rows) * cols + 7) / 8); // Full source-tile NaN mask.
    if (hasArg) bytes += AlignDma(cols * 4ULL);             // NaN-preserving extrema merge scratch.
    if (compensated) {
        const uint64_t scratch = uint64_t(std::max(1U, rows / 2)) * cols;
        // First-level source lows are implicit zero; all three tree buffers
        // (correction, sum and z) need at most the surviving half of the tile.
        // Batch FP16 reuses the consumed half source tile for float lows.
        const uint64_t buffers = batchReduce && dtype == SegmentFloat16 && rows >= 2 ? 2 : 3;
        bytes += buffers * AlignDma(scratch * 4) + AlignDma(cols * 4ULL) + AlignDma((scratch + 7) / 8);
    }
    // Value-only floating extrema use disjoint ping-pong partials. Half input
    // storage can be reused after Cast; float input needs a half-tile buffer.
    if (batchReduce && !compensated && dtype == SegmentFloat32) bytes += AlignDma(uint64_t(rows / 2) * cols * 4);
    return bytes;
}
struct TilePlan {
    uint32_t cols, rows, outputGroup;
    uint64_t bytes;
};
inline TilePlan PlanTile(uint64_t k, uint32_t dtype, bool hasArg, uint64_t ubBytes, bool compensated = false,
                         bool groupAligned = false, bool batchReduce = false) {
    if (!k || ubBytes <= kUbReserveBytes) throw std::invalid_argument("invalid K or UB capacity");
    // Compare on arch22 ignores a partial repeat. Reserve full 64-lane Work/arg
    // regions even when DMA and arithmetic process only the logical padded row.
    const uint32_t align = kVectorLanes;
    // Bound DMA/vector instruction counts independently of the full feature width.
    uint32_t cols = static_cast<uint32_t>(std::min<uint64_t>((k + align - 1) / align * align, 4096));
    const uint64_t budget = ubBytes - kUbReserveBytes;
    // Prefer enough rows for amortized DMA. Neither limit is an input-shape whitelist.
    while (cols > align && TileBytes(dtype, hasArg, cols, kMinAmortizedTileRows, compensated, batchReduce) > budget)
        cols = std::max(align, (cols / 2 / align) * align);
    if (TileBytes(dtype, hasArg, cols, 1, compensated, batchReduce) > budget)
        throw std::invalid_argument("UB is too small for a minimum segment tile");
    uint32_t rows = 256;
    while (TileBytes(dtype, hasArg, cols, rows, compensated, batchReduce) > budget) --rows;
    const uint64_t valueGroup =
        kDmaAlignBytes / std::gcd(uint64_t(kDmaAlignBytes), (k % kDmaAlignBytes) * ItemBytes(dtype));
    const uint64_t argGroup =
        hasArg ? kDmaAlignBytes / std::gcd(uint64_t(kDmaAlignBytes), (k % kDmaAlignBytes) * 8) : 1;
    // Adjacent segments can share one strided source transfer when a complete
    // feature row is 32-byte aligned.  Keep the old ownership/alignment group
    // for tails, but permit a bounded group of eight for aligned K as well;
    // the device verifies contiguous bounds before using the grouped path and
    // falls back to one task at a time for irregular CSR pointers.  This is a
    // layout-derived choice, not a K-value whitelist.
    uint64_t outputGroup = std::lcm(valueGroup, argGroup);
    if (groupAligned && k * uint64_t(ItemBytes(dtype)) % kDmaAlignBytes == 0 && outputGroup < 8) outputGroup = 8;
    return {cols, rows, static_cast<uint32_t>(outputGroup),
            TileBytes(dtype, hasArg, cols, rows, compensated, batchReduce) + kUbReserveBytes};
}
// Transfer-aware instruction-count model for proven uniform, aligned 2D
// sum/mean. Enumerate vector-aligned column tiles together with their exact UB
// row capacity and segment group; account for every extra tile's DMA/store,
// vector submissions and dependent-stage barriers. This is a scheduling
// heuristic, not a cycle prediction; measured transfers remain authoritative.
inline TilePlan PlanCompensatedColumns(uint64_t k, uint32_t dtype, uint64_t ubBytes, uint32_t segmentRows,
                                       uint64_t tasks, uint32_t cores) {
    auto best = PlanTile(k, dtype, false, ubBytes, true, false, true);
    best.outputGroup = 0;
    uint64_t bestCost = UINT64_MAX;
    uint32_t levels = 0;
    for (uint32_t n = segmentRows; n > 1; n /= 2) ++levels;
    const uint32_t stages = segmentRows == 2 ? 1 : 6 + 8 * (levels - 1);
    for (uint32_t cols = best.cols;; cols = std::max(kVectorLanes, (cols / 2 / kVectorLanes) * kVectorLanes)) {
        auto plan = PlanTile(cols, dtype, false, ubBytes, true, false, true);
        const uint32_t width = static_cast<uint32_t>(std::min<uint64_t>(k, plan.cols));
        const uint64_t groups =
            std::min<uint64_t>({kMaxGroupedSegments, plan.rows / segmentRows, tasks / (uint64_t(cores) * 2)});
        if (groups >= 1 && 2 * width / 8 <= kVectorStrideLimit) {
            const uint64_t tiles = (k + plan.cols - 1) / plan.cols;
            const uint64_t batches = (tasks + groups - 1) / groups;
            // Contiguous full rows use one dense transfer. Column tiles
            // submit strided row blocks on both load and store. Charge their
            // fragmentation by actual row bytes and active group rows, not
            // by dtype/K names. Eight bytes per row is a calibrated relative
            // issue-cost weight (not a hardware transaction size or cycles).
            const auto tileCost = [&](uint64_t logicalWidth) {
                const uint64_t rowBytes = logicalWidth * ItemBytes(dtype);
                const uint64_t blocks = groups * (uint64_t(segmentRows) + 1);
                const uint64_t strided = tiles > 1 ? (blocks * 8 + rowBytes - 1) / rowBytes : 0;
                return 2 + stages * ((logicalWidth + kVectorLanes - 1) / kVectorLanes + 1) + strided;
            };
            const uint64_t lastWidth = k - (tiles - 1) * plan.cols;
            const uint64_t cost = batches * ((tiles - 1) * tileCost(width) + tileCost(lastWidth));
            // Retain the existing multi-segment choice whenever it fits.
            // A one-segment tree is a capacity fallback, still consuming the
            // validator's full uniform proof and a complete power-of-two span.
            const bool multiple = groups >= 2;
            const bool bestMultiple = best.outputGroup >= 2;
            if ((multiple && !bestMultiple) || (multiple == bestMultiple && cost < bestCost)) {
                bestCost = cost;
                plan.outputGroup = static_cast<uint32_t>(groups);
                best = plan;
            }
        }
        if (cols == kVectorLanes) break;
    }
    return best;
}

// Every input to MakeSegmentPlan participates in the key. Index/out dtypes and
// address guards are normalized by the common Host; no address is cached here.
struct SegmentPlanKey {
    uint64_t batches, rows, features, segments, uniformRows, ubBytes;
    uint32_t cores, dtype;
    SegmentReduce reduce;
    bool hasArg, hasInitial, direct2D;
    bool operator==(const SegmentPlanKey &other) const {
        return std::tie(batches, rows, features, segments, uniformRows, ubBytes, cores, dtype, reduce, hasArg,
                        hasInitial, direct2D) == std::tie(other.batches, other.rows, other.features, other.segments,
                                                          other.uniformRows, other.ubBytes, other.cores, other.dtype,
                                                          other.reduce, other.hasArg, other.hasInitial, other.direct2D);
    }
};
struct SegmentPlan {
    TilePlan tile;
    uint32_t batchReduce;
};

inline void RefineBatchValues(const SegmentPlanKey &key, uint64_t averageRows, bool compensated,
                              SegmentPlan &selected) {
    auto &plan = selected.tile;
    auto &batchReduce = selected.batchReduce;
    // Paired vector repeats consume adjacent rows without packing. Both row
    // strides must fit the hardware repeat field; multiple complete segments
    // must fit UB, with enough groups left to occupy every vector core.
    const bool batchValueExtreme = !key.hasArg && (key.reduce == SegmentMin || key.reduce == SegmentMax);
    if (key.direct2D && (compensated || batchValueExtreme) && !key.hasInitial &&
        (key.dtype == SegmentFloat32 || key.dtype == SegmentFloat16) && averageRows >= 2 &&
        (averageRows & (averageRows - 1)) == 0) {
        auto candidate = PlanTile(key.features, key.dtype, false, key.ubBytes, compensated, false, true);
        const uint64_t groups =
            std::min<uint64_t>({kMaxGroupedSegments, candidate.rows / averageRows,
                                key.batches * static_cast<uint64_t>(key.segments) / (uint64_t(key.cores) * 2)});
        if (key.features <= candidate.cols && key.features % 8 == 0 && 2 * key.features / 8 <= kVectorStrideLimit &&
            groups >= 2) {
            candidate.outputGroup = static_cast<uint32_t>(groups);
            plan = candidate;
            batchReduce = 1;
        }
    }
}

inline void RefineCompensatedColumns(const SegmentPlanKey &key, uint64_t averageRows, bool compensated,
                                     SegmentPlan &selected) {
    auto &plan = selected.tile;
    auto &batchReduce = selected.batchReduce;
    // Column splitting is limited to the validator's complete uniform proof.
    // Every output row and feature tile owns whole 32-byte blocks. Existing
    // irregular, initial-out and extrema planners remain unchanged.
    if (key.direct2D && compensated && key.uniformRows &&
        (key.dtype == SegmentFloat32 || key.dtype == SegmentFloat16) && averageRows >= 2 && averageRows <= 256 &&
        (averageRows & (averageRows - 1)) == 0 && key.features * ItemBytes(key.dtype) % kDmaAlignBytes == 0) {
        const auto candidate =
            PlanCompensatedColumns(key.features, key.dtype, key.ubBytes, static_cast<uint32_t>(averageRows),
                                   key.batches * static_cast<uint64_t>(key.segments), key.cores);
        if (candidate.outputGroup >= 1) {
            plan = candidate;
            batchReduce = 1;
        }
    }
}

inline void RefineNarrowFeatures(const SegmentPlanKey &key, uint64_t averageRows, bool compensated,
                                 SegmentPlan &selected) {
    auto &plan = selected.tile;
    auto &batchReduce = selected.batchReduce;
    // In the proven uniform batch path the logical row is also the physical
    // UB row. Narrow aligned features can therefore use twice as many rows
    // without changing any per-segment TwoSum operation or rounding order.
    // Keep the generic >=64-lane layout whenever a scalar/out fallback exists.
    if (key.direct2D && compensated && !key.hasInitial && !key.hasArg && key.uniformRows >= 2 &&
        (key.uniformRows & (key.uniformRows - 1)) == 0 && key.features < kVectorLanes &&
        key.features * ItemBytes(key.dtype) % kDmaAlignBytes == 0 &&
        (key.dtype == SegmentFloat32 || key.dtype == SegmentFloat16)) {
        const uint32_t cols = static_cast<uint32_t>(key.features);
        uint32_t rows = 512;
        while (rows > 1 && TileBytes(key.dtype, false, cols, rows, true, true) + kUbReserveBytes > key.ubBytes)
            rows /= 2;
        const uint64_t groups = std::min<uint64_t>(
            {kMaxGroupedSegments, rows / key.uniformRows, key.batches * key.segments / (uint64_t(key.cores) * 2)});
        if (groups >= 2 && groups > plan.outputGroup) {
            plan = {cols, rows, static_cast<uint32_t>(groups),
                    TileBytes(key.dtype, false, cols, rows, true, true) + kUbReserveBytes};
            batchReduce = 1;
        }
    }
}

inline void RefineLongSegments(const SegmentPlanKey &key, uint64_t averageRows, bool compensated,
                               SegmentPlan &selected) {
    auto &plan = selected.tile;
    auto &batchReduce = selected.batchReduce;
    // A long proven segment can fit one compensated tree by splitting its
    // aligned feature columns. This avoids repeated tree finalization/merge
    // when the generic wide row cannot hold the complete segment in UB.
    if (key.direct2D && compensated && !key.hasInitial && !key.hasArg && key.uniformRows > plan.rows &&
        key.uniformRows <= 512 && (key.uniformRows & (key.uniformRows - 1)) == 0 && key.features > 0 &&
        key.features * ItemBytes(key.dtype) % kDmaAlignBytes == 0 &&
        (key.dtype == SegmentFloat32 || key.dtype == SegmentFloat16)) {
        const uint32_t cols = static_cast<uint32_t>(std::min<uint64_t>(kLongFloatTileColumns, key.features));
        const uint32_t rows = 512;
        const uint64_t bytes = TileBytes(key.dtype, false, cols, rows, true, true) + kUbReserveBytes;
        if (bytes <= key.ubBytes) {
            plan = {cols, rows, 1, bytes};
            batchReduce = 1;
        }
    }
}

inline SegmentPlan MakeSegmentPlan(const SegmentPlanKey &key) {
    // Argument reductions retain strict row order and first-improvement ties.
    // The later argCarryBlocks pass propagates state across segments/batches;
    // value-tree refinements must not replace this sequential geometry.
    if (key.hasArg) return {PlanTile(key.features, key.dtype, true, key.ubBytes), 0};
    // Preserve this backend's compensated floating sum/mean workspace.
    const bool compensated = FloatInput(key.dtype) && (key.reduce == SegmentSum || key.reduce == SegmentMean);
    // Grouping amortizes DMA, provided there are enough segments to give every
    // AIV at least one complete group.
    // Group ownership only amortizes its scalar bound checks when a typical
    // segment occupies a material fraction of the planned row tile.  Derive
    // that cutoff from the actual UB plan instead of an input-K whitelist;
    // short segments keep the ordinary per-task ownership and its parallelism.
    const uint64_t averageRows = key.segments ? key.rows / static_cast<uint64_t>(key.segments) : 0;
    const auto basePlan = PlanTile(key.features, key.dtype, key.hasArg, key.ubBytes, compensated, false);
    const bool groupLengthOk = averageRows * 4 >= basePlan.rows;
    const bool groupAligned = !key.hasInitial && groupLengthOk &&
                              key.batches * static_cast<uint64_t>(key.segments) >= uint64_t(key.cores) * 8;
    auto plan = groupAligned ? PlanTile(key.features, key.dtype, key.hasArg, key.ubBytes, compensated, true) : basePlan;
    SegmentPlan selected{plan, 0};
    RefineBatchValues(key, averageRows, compensated, selected);
    RefineCompensatedColumns(key, averageRows, compensated, selected);
    RefineNarrowFeatures(key, averageRows, compensated, selected);
    RefineLongSegments(key, averageRows, compensated, selected);
    return selected;
}

inline SegmentPlan CachedSegmentPlan(const SegmentPlanKey &key) {
    // A small per-thread ring avoids locking or heap allocation on the hot path.
    // Repeated shapes hit mostRecent directly; misses retain ordinary planning.
    struct Entry {
        SegmentPlanKey key{};
        SegmentPlan plan{};
    };
    constexpr size_t capacity = 16;
    struct Cache {
        std::array<Entry, capacity> entries{};
        size_t size = 0, next = 0, mostRecent = 0;
    };
    static thread_local Cache cache;
    if (cache.size && cache.entries[cache.mostRecent].key == key) return cache.entries[cache.mostRecent].plan;
    for (size_t i = 0; i < cache.size; ++i) {
        if (cache.entries[i].key == key) {
            cache.mostRecent = i;
            return cache.entries[i].plan;
        }
    }
    const auto plan = MakeSegmentPlan(key);
    cache.entries[cache.next] = {key, plan};
    cache.mostRecent = cache.next;
    cache.next = (cache.next + 1) % capacity;
    cache.size = std::min(cache.size + 1, capacity);
    return plan;
}
void ConfigureIntegerTiles(PlanInputs &p) {
    p.tileK =
        p.dtype == SEGMENT_CSR_INT64
            ? std::min<uint64_t>((p.channels + kInt64VectorLanes - 1) / kInt64VectorLanes * kInt64VectorLanes,
                                 kIntegerTileColumns)
            : std::min<uint64_t>((p.channels + kVectorLanes - 1) / kVectorLanes * kVectorLanes, kIntegerTileColumns);
    p.tileRows = kIntegerTileRows;
    if (p.dtype == SEGMENT_CSR_INT64 && p.reduce < SEGMENT_CSR_MIN && !p.hasArg) {
        // Exact int64 digit decomposition budgets every live UB buffer.
        for (;;) {
            const uint64_t groups = p.uniformRows && p.uniformRows <= p.tileRows
                                        ? std::min<uint64_t>(kMaxGroupedSegments, p.tileRows / p.uniformRows)
                                        : kInt64FallbackGroups;
            const uint64_t bytes = uint64_t(p.tileK) * p.tileRows * kInt64DecompInputBytesPerElem +
                                   uint64_t(p.tileK) * groups * kInt64DecompOutputBytesPerElem +
                                   p.tileK * kInt64AccumulatorBytesPerColumn + kUbReserveBytes;
            if (bytes <= p.ubBytes) break;
            p.tileRows /= 2;
        }
    }
    if (p.dtype == SEGMENT_CSR_INT64 && p.reduce >= SEGMENT_CSR_MIN && !p.hasArg) {
        while (uint64_t(p.tileK) * p.tileRows * kExtremaInputBytesPerElem + p.tileK * kExtremaBytesPerColumn +
                   kExtremaReserveBytes >
               p.ubBytes)
            p.tileRows /= 2;
    }
    const uint64_t item = p.dtype == SEGMENT_CSR_INT64 ? 8 : 1;
    p.outputGroup = kDmaAlignBytes / std::gcd(uint64_t(kDmaAlignBytes), uint64_t(p.channels) * item);
    if (p.uniformRows && p.uniformRows <= p.tileRows && (p.uniformRows & (p.uniformRows - 1)) == 0 &&
        p.channels % 8 == 0 && !p.hasOut) {
        const uint64_t groups = std::min<uint64_t>({kMaxGroupedSegments, p.tileRows / p.uniformRows,
                                                    uint64_t(p.batches * p.segments) / (std::max(1, p.blocks) * 2ULL)});
        p.outputGroup = std::max<uint64_t>(p.outputGroup, std::max<uint64_t>(1, groups));
    }
}

void ConfigureTiles(PlanInputs &p) {
    if (p.dtype == SEGMENT_CSR_INT64 || p.dtype == SEGMENT_CSR_INT8 || p.dtype == SEGMENT_CSR_UINT8) {
        ConfigureIntegerTiles(p);
        return;
    }
    const auto selected =
        CachedSegmentPlan({static_cast<uint64_t>(p.batches), static_cast<uint64_t>(p.length),
                           static_cast<uint64_t>(p.channels), static_cast<uint64_t>(p.segments), p.uniformRows,
                           p.ubBytes, static_cast<uint32_t>(std::max(1, p.blocks)), static_cast<uint32_t>(p.dtype),
                           static_cast<SegmentReduce>(p.reduce), p.hasArg != 0, p.hasOut != 0, p.batches == 1});
    p.tileK = selected.tile.cols;
    p.tileRows = selected.tile.rows;
    p.outputGroup = selected.tile.outputGroup;
    p.batchReduce = selected.batchReduce;
}

SegmentCsrTilingData MakeBaseTiling(const PlanInputs &p) {
    SegmentCsrTilingData t{};
    t.E_1 = p.batches;
    t.M = p.length;
    t.K = p.channels;
    t.nSegments = p.segments;
    t.srcNumel = p.batches * p.length * p.channels;
    t.outNumel = p.batches * p.segments * p.channels;
    t.indexNumel = p.indexBatches * (p.segments + 1);
    t.indexBatchStride = p.indexBatches == 1 ? 0 : p.segments + 1;
    t.indexRowStride = 1;
    t.indexMode = p.uniformRows ? 2 : 0;
    t.reduceCode = p.reduce;
    t.dtypeCode = p.dtype;
    // CSR always overwrites out, including empty segments.
    t.hasInitial = 0; // CSR overwrites every output; out is never a reduction seed.
    t.hasArg = p.hasArg;
    t.tileK = p.tileK;
    t.tileRows = p.tileRows;
    t.outputGroup = p.outputGroup;
    t.batchReduce = p.batchReduce;
    t.boundedSource = p.boundedSource;
    t.l2Bytes = p.l2Bytes;
    t.totalTasks = p.batches * p.segments;
    t.featureTiles = (p.channels + p.tileK - 1) / p.tileK;
    t.totalUnits =
        p.outputGroup == 1 ? t.totalTasks * t.featureTiles : (t.totalTasks + p.outputGroup - 1) / p.outputGroup;
    t.usedBlockNum = std::min<uint64_t>(std::max(1, p.blocks), t.totalUnits);
    return t;
}
bool SelectShortUniform(SegmentCsrPlan &plan) {
    auto &t = plan.tiling;
    const bool isHalf = t.dtypeCode == SEGMENT_CSR_FLOAT16;
    const uint32_t itemBytes = isHalf ? 2 : 4;
    if (t.indexMode != 2 || t.reduceCode > SegmentMean || !t.nSegments ||
        (t.K != kShortUniformColumns && t.K != kMediumUniformColumns && t.K != kWideUniformColumns))
        return false;
    const uint64_t rows = t.M / t.nSegments;
    if (!rows || rows > 16 || (rows & (rows - 1))) return false;
    if (isHalf) {
        if (!t.boundedSource) return false;
    }
    // Match all live UB allocations. Half reuses the promoted-input prefix
    // as its accumulator, so only its separate half output adds per-group space.
    constexpr uint64_t budget = kArch22UbCapacityBytes - kUbReserveBytes;
    const uint32_t inputBuffers = isHalf && rows >= 8 ? 1 : 2;
    const uint64_t bytesPerGroup = t.K * (rows * (inputBuffers * itemBytes + (isHalf ? 4 : 0)) + (isHalf ? 2 : 4));
    const uint64_t balance = std::max<uint64_t>(1, t.totalTasks / (std::max(1U, t.usedBlockNum) * 2ULL));
    t.tileK = t.K;
    t.tileRows = static_cast<uint32_t>(rows);
    t.outputGroup = static_cast<uint32_t>(std::min<uint64_t>({kVectorRepeatLimit, budget / bytesPerGroup, balance}));
    t.totalUnits = (t.totalTasks + t.outputGroup - 1) / t.outputGroup;
    t.usedBlockNum = std::min<uint64_t>(t.usedBlockNum, t.totalUnits);
    plan.path = SegmentCsrKernelPath::ShortUniform;
    return true;
}
std::pair<uint32_t, uint32_t> SelectOrderedChunks(const SegmentCsrTilingData &t, uint64_t rows, uint32_t outputBytes,
                                                  bool wholeCast) {
    const bool isFloat = t.dtypeCode == SEGMENT_CSR_FLOAT32;
    const uint32_t itemBytes = isFloat ? 4 : 2;
    const uint64_t budget = kArch22UbCapacityBytes - kUbReserveBytes;
    uint64_t bestNumerator = UINT64_MAX, bestDenominator = 1;
    uint32_t selectedChunk = 1, selectedGroups = 1;
    const uint64_t balance = std::max<uint64_t>(1, t.totalTasks / (std::max(1U, t.usedBlockNum) * 2ULL));
    const uint32_t strideElements = isFloat ? 8 : 16;
    if (wholeCast) {
        selectedChunk = rows;
        const uint64_t bytesPerGroup = t.K * (uint64_t(rows) * (2 * itemBytes + 4) + outputBytes - 4);
        selectedGroups =
            static_cast<uint32_t>(std::min<uint64_t>({kVectorRepeatLimit, budget / bytesPerGroup, balance}));
    } else {
        for (uint32_t chunk = 1; chunk <= rows && uint64_t(chunk) * t.K / strideElements <= kVectorStrideLimit;
             chunk *= 2) {
            const uint64_t bytesPerGroup = t.K * (uint64_t(chunk) * 2 * itemBytes + outputBytes);
            const uint32_t groups =
                static_cast<uint32_t>(std::min<uint64_t>({kVectorRepeatLimit, budget / bytesPerGroup, balance}));
            if (!groups) break;
            // Compare scalar/vector submission cost at equal input volume. Each
            // extra group amortizes ordered row instructions; extra rows amortize
            // DMA/queue bookkeeping. All candidates retain identical arithmetic.
            const uint64_t rowOps = (t.K + kVectorLanes - 1) / kVectorLanes + (isFloat ? 1 : 3);
            const uint64_t numerator = kOrderedChunkIssueCost + rowOps * chunk;
            const uint64_t denominator = uint64_t(groups) * chunk;
            if (bestNumerator == UINT64_MAX || numerator * bestDenominator <= bestNumerator * denominator) {
                bestNumerator = numerator;
                bestDenominator = denominator;
                selectedChunk = chunk;
                selectedGroups = groups;
            }
        }
    }
    return {selectedChunk, selectedGroups};
}
bool OrderedMeanUsesPowerOfTwo(uint64_t rows, bool isHalf) {
    const uint64_t count = rows > 0 ? rows : 1;
    bool powerMean = !(count & (count - 1));
    if (isHalf) powerMean = powerMean && count <= kHalfPowerMeanLimit;
    return powerMean;
}
bool SelectOrderedUniform(SegmentCsrPlan &plan) {
    auto &t = plan.tiling;
    const bool isHalf = t.dtypeCode == SEGMENT_CSR_FLOAT16;
    const bool isFloat = t.dtypeCode == SEGMENT_CSR_FLOAT32;
    const uint32_t itemBytes = isFloat ? 4 : 2;
    if (t.indexMode != 2 || t.reduceCode > SegmentMean || !t.K || t.K > 256 || t.K * itemBytes % kDmaAlignBytes ||
        !t.nSegments)
        return false;
    const uint64_t rows = t.M / t.nSegments;
    if (!rows) return false;
    const bool powerMean = OrderedMeanUsesPowerOfTwo(rows, isHalf);
    const bool wholeCast = isHalf && t.K <= kWholeCastColumnLimit && rows <= 16;
    const bool boundedTree = wholeCast && !(rows & (rows - 1)) && t.boundedSource;
    const uint32_t outputBytes =
        4 + (isFloat ? 0 : 4 + itemBytes) + (t.reduceCode == SegmentMean && !powerMean ? 4 : 0);
    const auto chunks = SelectOrderedChunks(t, rows, outputBytes, wholeCast);
    const auto selectedChunk = chunks.first, selectedGroups = chunks.second;
    t.tileK = t.K;
    t.tileRows = selectedChunk;
    t.outputGroup = selectedGroups;
    t.totalUnits = (t.totalTasks + t.outputGroup - 1) / t.outputGroup;
    t.usedBlockNum = std::min<uint64_t>(t.usedBlockNum, t.totalUnits);
    plan.path = boundedTree
                    ? SegmentCsrKernelPath::OrderedBoundedTree
                    : (wholeCast ? SegmentCsrKernelPath::OrderedWholeCast : SegmentCsrKernelPath::OrderedUniform);
    return true;
}
bool SelectHalfExtreme(SegmentCsrPlan &plan) {
    auto &t = plan.tiling;
    if (t.indexMode != 2 || t.hasArg || t.reduceCode < SegmentMin || !t.K || t.K > 256 || t.K % 16 || !t.nSegments)
        return false;
    const uint64_t rows = t.M / t.nSegments;
    if (!rows || (rows & (rows - 1))) return false;
    const uint64_t budget = kArch22UbCapacityBytes - kUbReserveBytes;
    const uint64_t balance = std::max<uint64_t>(1, t.totalTasks / (std::max(1U, t.usedBlockNum) * 2ULL));
    uint64_t bestNumerator = UINT64_MAX, bestDenominator = 1;
    uint32_t bestRows = 1, bestGroups = 1;
    uint32_t levels = 0;
    for (uint32_t chunk = 1; chunk <= rows && chunk <= 512; chunk *= 2, ++levels) {
        const uint64_t bytesPerGroupTimes8 =
            t.K * (uint64_t(chunk) * kHalfExtremaInputBitsPerElem + kHalfExtremaOutputBytesPerElem * 8);
        uint32_t groups =
            static_cast<uint32_t>(std::min<uint64_t>({kVectorRepeatLimit, budget * 8 / bytesPerGroupTimes8, balance}));
        while (groups) {
            const uint64_t capacity =
                (uint64_t(groups) * chunk * t.K + kHalfVectorLanes - 1) / kHalfVectorLanes * kHalfVectorLanes;
            const uint64_t bytes = capacity * kHalfExtremaInputBytesPerElem +
                                   ((capacity / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes) +
                                   groups * t.K * kHalfExtremaOutputBytesPerElem;
            if (bytes <= budget) break;
            --groups;
        }
        if (!groups) break;
        const uint64_t numerator = 68 + 2 * levels;
        const uint64_t denominator = uint64_t(groups) * chunk;
        if (bestNumerator == UINT64_MAX || numerator * bestDenominator <= bestNumerator * denominator) {
            bestNumerator = numerator;
            bestDenominator = denominator;
            bestRows = chunk;
            bestGroups = groups;
        }
    }
    t.tileK = t.K;
    t.tileRows = bestRows;
    t.outputGroup = bestGroups;
    t.totalUnits = (t.totalTasks + bestGroups - 1) / bestGroups;
    t.usedBlockNum = std::min<uint64_t>(t.usedBlockNum, t.totalUnits);
    plan.path = SegmentCsrKernelPath::HalfExtreme;
    return true;
}
void SetPipelinePlan(SegmentCsrPlan &plan, uint64_t rows, uint32_t groups, SegmentCsrKernelPath path) {
    auto &t = plan.tiling;
    t.tileK = static_cast<uint32_t>(t.K);
    t.tileRows = static_cast<uint32_t>(rows);
    t.outputGroup = groups;
    t.featureTiles = 1;
    t.totalUnits = (t.totalTasks + groups - 1) / groups;
    t.usedBlockNum = std::min<uint64_t>(t.usedBlockNum, t.totalUnits);
    plan.path = path;
}
bool SelectInt32Pipeline(SegmentCsrPlan &plan, uint64_t rows) {
    auto &t = plan.tiling;
    // Complete short segments: two <=64-KiB input slots, half-size
    // ping-pong scratch, two outputs and (for mean) two correction planes.
    // Include every live plane and an 8-KiB reserve in the 192-KiB UB bound.
    if (t.indexMode == 2 && !t.hasInitial && !t.hasArg && rows >= 2 && rows <= kPipelineRowLimit &&
        (rows & (rows - 1)) == 0 && t.K && t.K % 8 == 0 && t.K <= 256) {
        const uint64_t groupInput = rows * t.K;
        const uint64_t balance = std::max<uint64_t>(1, t.totalTasks / (t.usedBlockNum * 2ULL));
        uint32_t groups =
            static_cast<uint32_t>(std::min<uint64_t>(kPipelineSlotBytes / sizeof(int32_t) / groupInput, balance));
        for (; groups; --groups) {
            const uint64_t inputs = uint64_t(groups) * groupInput;
            const uint64_t outputs = uint64_t(groups) * t.K;
            const uint64_t bytes =
                inputs * kInt32PipelineInputBytesPerElem + outputs * kInt32PipelineOutputBytesPerElem + kUbReserveBytes;
            if (bytes <= kArch22UbCapacityBytes) break;
        }
        if (groups) {
            SetPipelinePlan(plan, rows, groups, SegmentCsrKernelPath::Int32Pipeline);
            return true;
        }
    }
    return false;
}

bool SelectInt32Long(SegmentCsrPlan &plan, uint64_t rows) {
    auto &t = plan.tiling;
    // A minimum of 64 full feature rows fit in each 64-KiB queue slot.
    // This capacity-derived route covers long uniform segments without
    // fragmenting every row into narrow feature DMA blocks.
    constexpr uint32_t slotElements = kPipelineSlotBytes / sizeof(int32_t);
    if (t.indexMode == 2 && !t.hasInitial && !t.hasArg && rows >= kLongSegmentMinRows && rows <= INT32_MAX &&
        (rows & (rows - 1)) == 0 && t.K % 8 == 0 && t.K <= slotElements / kLongInt32MinSlotRows) {
        t.tileK = static_cast<uint32_t>(t.K);
        t.tileRows = 1;
        while (t.tileRows * 2 <= slotElements / t.tileK) t.tileRows *= 2;
        t.outputGroup = 1;
        t.featureTiles = 1;
        t.totalUnits = t.totalTasks;
        t.usedBlockNum = std::min<uint64_t>(t.usedBlockNum, t.totalUnits);
        plan.path = SegmentCsrKernelPath::Int32Long;
        return true;
    }
    return false;
}
void SelectInt32(SegmentCsrPlan &plan) {
    auto &t = plan.tiling;
    const uint64_t rows = t.nSegments ? t.M / t.nSegments : 0;
    if (SelectInt32Pipeline(plan, rows)) return;
    if (SelectInt32Long(plan, rows)) return;
    // Input plus ping-pong scratch use <=144 KiB; mean temporaries add at
    // most 32 KiB, retaining >=16 KiB of the supported 192 KiB UB. There is
    // no data-dependent speculation: indexMode=2 is the Host's full proof.
    // One native int32 vector per feature slice. Extra independent segments
    // share its repeats instead of issuing 2-4 instructions across wide rows.
    uint32_t cols = static_cast<uint32_t>(std::min<uint64_t>(kVectorLanes, t.K));
    // A complete uniform segment can trade channel parallelism for enough UB
    // rows, avoiding the per-segment generic loop for long narrow reductions.
    while (cols > 8 && rows > kInt32BatchRowBudget / cols) cols = (cols / 2 / 8) * 8;
    const uint32_t capacityRows = cols ? kInt32BatchRowBudget / cols : 0;
    // Two mean scratch planes total at most 4096 * 8 = 32 KiB.
    const uint32_t maxGroups = cols ? kInt32MeanOutputBudget / cols : 0;
    if (t.indexMode == 2 && !t.hasInitial && !t.hasArg && t.K % 8 == 0 && t.K <= UINT32_MAX / 4 && rows &&
        (rows & (rows - 1)) == 0 && rows <= capacityRows) {
        t.tileK = cols;
        t.tileRows = capacityRows;
        t.outputGroup = static_cast<uint32_t>(std::min<uint64_t>(maxGroups, capacityRows / rows));
        t.featureTiles = (t.K + cols - 1) / cols;
        t.totalUnits = (t.totalTasks + t.outputGroup - 1) / t.outputGroup * t.featureTiles;
        t.usedBlockNum = std::min<uint64_t>(t.usedBlockNum, t.totalUnits);
        plan.path = SegmentCsrKernelPath::Int32Batch;
        return;
    }
}
bool SelectInt64Pipeline(SegmentCsrPlan &plan, uint64_t rows) {
    auto &t = plan.tiling;
    // Short complete segments: two <=64-KiB input slots + <=32-KiB low
    // words + two packed-output slots + one offsets plane + mask + reserve.
    // Budget all live buffers rather than assuming the maximum slot size fits
    // together with the larger outputs of 2/4-row segments.
    if (rows <= kPipelineRowLimit && t.K <= 256) {
        const uint64_t groupInput = rows * t.K;
        const uint64_t balance = std::max<uint64_t>(1, t.totalTasks / (t.usedBlockNum * 2ULL));
        uint32_t groups =
            static_cast<uint32_t>(std::min<uint64_t>(kPipelineSlotBytes / sizeof(int64_t) / groupInput, balance));
        for (; groups; --groups) {
            const uint64_t inputs = uint64_t(groups) * groupInput;
            const uint64_t outputs = uint64_t(groups) * t.K;
            const uint64_t mask = ((outputs * 2 + 7) / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes;
            const uint64_t bytes = inputs * kInt64PipelineInputBytesPerElem +
                                   outputs * kInt64PipelineOutputBytesPerElem + mask + kUbReserveBytes;
            if (bytes <= kArch22UbCapacityBytes) break;
        }
        if (groups) {
            SetPipelinePlan(plan, rows, groups, SegmentCsrKernelPath::Int64Pipeline);
            return true;
        }
    }
    return false;
}

void SelectInt64(SegmentCsrPlan &plan) {
    auto &t = plan.tiling;
    const uint64_t rows = t.nSegments ? t.M / t.nSegments : 0;
    if (!t.boundedSource || t.hasInitial || t.hasArg || t.indexMode != 2 || t.E_1 != 1 || rows < 2 ||
        rows > INT32_MAX || (rows & (rows - 1)) || t.K % 8 || t.K > UINT32_MAX / 8) {
        return;
    }
    if (SelectInt64Pipeline(plan, rows)) return;
    // Two 64-KiB int64 DMA slots, one 32-KiB low-word plane and at most
    // 5 KiB of output metadata fit within the supported 192-KiB UB.
    // Require at least 32 full feature rows per DMA slot.
    constexpr uint32_t slotElements = kPipelineSlotBytes / sizeof(int64_t);
    if (rows >= kLongSegmentMinRows && t.K <= slotElements / kLongInt64MinSlotRows) {
        t.tileK = static_cast<uint32_t>(t.K);
        t.tileRows = 1;
        while (t.tileRows * 2 <= slotElements / t.tileK) t.tileRows *= 2;
        t.outputGroup = 1;
        t.featureTiles = 1;
        t.totalUnits = t.totalTasks;
        t.usedBlockNum = std::min<uint64_t>(t.usedBlockNum, t.totalUnits);
        plan.path = SegmentCsrKernelPath::Int64Long;
        return;
    }
    // A native int32 vector is 64 lanes. Keeping one vector-wide feature tile
    // increases independent segment batching instead of issuing a separate
    // arithmetic instruction for every 64-lane slice of a wide feature row.
    // Dense/strided DMA both retain complete 32-byte ownership.
    t.tileK = std::min<uint64_t>((t.K + kInt64VectorLanes - 1) / kInt64VectorLanes * kInt64VectorLanes, kVectorLanes);
    const uint64_t tasksPerCore = std::max<uint64_t>(1, t.totalTasks / (t.usedBlockNum * 2ULL));
    uint32_t selectedRows = 1, selectedGroups = 1;
    for (uint32_t candidate = 512; candidate >= 1; --candidate) {
        const uint32_t groups =
            std::min<uint64_t>({kInt64NarrowGroupLimit, std::max<uint64_t>(1, candidate / rows), tasksPerCore});
        const uint64_t output = uint64_t(groups) * t.tileK;
        // Mean uses packed output as pre-pack scratch; after Gather, the dead
        // reduced-value buffer holds sign words. No dedicated temporary plane.
        const uint64_t bytes = uint64_t(candidate) * t.tileK * kInt64NarrowInputBytesPerElem +
                               output * kInt64NarrowOutputBytesPerElem + t.tileK * 4 +
                               ((output * 2 + 7) / 8 + kDmaAlignBytes - 1) / kDmaAlignBytes * kDmaAlignBytes +
                               kUbReserveBytes;
        if (bytes <= kArch22UbCapacityBytes) {
            selectedRows = candidate;
            selectedGroups = groups;
            break;
        }
    }
    // Batches consist of whole power-of-two segments. Single long segments
    // use the exact odd-row carry in the integer tree.
    t.tileRows = selectedRows;
    t.outputGroup = selectedGroups;
    t.featureTiles = (t.K + t.tileK - 1) / t.tileK;
    t.totalUnits =
        t.outputGroup == 1 ? t.totalTasks * t.featureTiles : (t.totalTasks + t.outputGroup - 1) / t.outputGroup;
    t.usedBlockNum = std::min<uint64_t>(t.usedBlockNum, t.totalUnits);
    plan.path = SegmentCsrKernelPath::Int64Narrow;
}
SegmentCsrPlan MakeLaunchPlan(PlanInputs inputs) {
    ConfigureTiles(inputs);
    SegmentCsrPlan plan{MakeBaseTiling(inputs)};
    if (!plan.tiling.usedBlockNum) return plan;
    switch (inputs.dtype) {
    case SEGMENT_CSR_FLOAT16:
    case SEGMENT_CSR_FLOAT32:
        if (SelectShortUniform(plan) || SelectOrderedUniform(plan)) break;
        if (inputs.dtype == SEGMENT_CSR_FLOAT16) SelectHalfExtreme(plan);
        break;
    case SEGMENT_CSR_BFLOAT16:
        SelectOrderedUniform(plan);
        break;
    case SEGMENT_CSR_INT32:
        SelectInt32(plan);
        break;
    case SEGMENT_CSR_INT64:
        SelectInt64(plan);
        break;
    default:
        break;
    }
    const auto &t = plan.tiling;
    if (t.hasArg && t.reduceCode >= SegmentMin) {
        // Each carry worker owns complete 32-byte argument blocks.
        const uint32_t tiles = static_cast<uint32_t>((t.K + kVectorLanes - 1) / kVectorLanes);
        plan.argCarryBlocks = t.K % 4 ? 1 : std::min(tiles, t.usedBlockNum);
    }
    return plan;
}
} // namespace csr_host
} // namespace
#endif

namespace ops_gnn {
namespace {
#ifdef OPSGNN_DAV_2201
constexpr int KIND_F32 = SEGMENT_CSR_FLOAT32, KIND_F16 = SEGMENT_CSR_FLOAT16, KIND_BF16 = SEGMENT_CSR_BFLOAT16,
              KIND_I8 = SEGMENT_CSR_INT8, KIND_U8 = SEGMENT_CSR_UINT8, KIND_I32 = SEGMENT_CSR_INT32,
              KIND_I64 = SEGMENT_CSR_INT64;
#endif

struct SegDims {
    int64_t dim;
    int64_t e1;
    int64_t m;
    int64_t k;
    int64_t nSeg;
    int64_t indptrLast;
};

SegDims ComputeDims(const torch::Tensor &src, const torch::Tensor &indptr) {
    SegDims d;
    d.dim = indptr.dim() - 1;
    d.e1 = 1;
    for (const auto i : c10::irange(d.dim)) { d.e1 *= src.size(i); }
    d.m = src.size(d.dim);
    d.indptrLast = indptr.size(d.dim);
    d.nSeg = d.indptrLast - 1;
    d.k = 1;
    for (const auto i : c10::irange(static_cast<int64_t>(d.dim + 1), src.dim())) { d.k *= src.size(i); }
    return d;
}

void Validate(const torch::Tensor &src, const torch::Tensor &indptr, const SegDims &d) {
    TORCH_CHECK(indptr.scalar_type() == at::kLong, "indptr must be int64, got ", indptr.scalar_type());
    TORCH_CHECK(indptr.dim() >= 1, "indptr must have at least 1 dim");
    TORCH_CHECK(src.dim() >= indptr.dim(), "src.dim() (", src.dim(), ") must be >= indptr.dim() (", indptr.dim(), ")");
    TORCH_CHECK(d.indptrLast >= 1, "indptr last dim must be >= 1");
    for (const auto i : c10::irange(d.dim)) {
        TORCH_CHECK(indptr.size(i) == 1 || indptr.size(i) == src.size(i), "indptr leading dim ", i, " (",
                    indptr.size(i), ") must be 1 or match src (", src.size(i), ")");
    }
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "segment_csr(NPU) requires NPU tensors, got ",
                src.device());
    TORCH_CHECK(indptr.device() == src.device(), "src and indptr must live on the same device");
}

int KindFor(const torch::Tensor &src) {
    switch (src.scalar_type()) {
    case at::kFloat:
        return KIND_F32;
    case at::kHalf:
        return KIND_F16;
    case at::kBFloat16:
        return KIND_BF16;
    case at::kChar:
        return KIND_I8;
    case at::kByte:
        return KIND_U8;
    case at::kInt:
        return KIND_I32;
    case at::kLong:
        return KIND_I64;
    default:
        TORCH_CHECK(false, "segment_csr unsupported dtype: ", src.scalar_type());
        return -1;
    }
}

#ifndef OPSGNN_DAV_2201
// 16-byte vector lane count for the perf dtypes; 0 keeps the scalar path.
// Lane counts live in KindVecLanes (kernel header, single source).
int VecElemsFor(int kind, uint64_t k) {
    const int lanes = KindVecLanes(kind);
    return lanes > 0 && k % static_cast<uint64_t>(lanes) == 0 ? lanes : 0;
}

#endif

struct ReductionConfig {
    int op;
    int isMean;
};

ReductionConfig ParseReduction(const std::string &reduce) {
    if (reduce == "sum" || reduce == "add") { return {SEGMENT_CSR_SUM, 0}; }
    if (reduce == "mean") { return {SEGMENT_CSR_SUM, 1}; }
    if (reduce == "min") { return {SEGMENT_CSR_MIN, 0}; }
    TORCH_CHECK(reduce == "max", "reduce value '", reduce, "' is not valid (use sum, add, mean, min or max)");
    return {SEGMENT_CSR_MAX, 0};
}

torch::Tensor PreparePointers(const torch::Tensor &src, const torch::Tensor &indptr, const SegDims &dims,
                              uint64_t &stride) {
    auto result = indptr.is_contiguous() ? indptr : indptr.contiguous();
    bool shared = true;
    for (const auto i : c10::irange(dims.dim)) {
        if (indptr.size(i) != 1) {
            shared = false;
            break;
        }
    }
    stride = 0;
    if (!shared) {
        std::vector<int64_t> shape;
        for (const auto i : c10::irange(dims.dim)) { shape.push_back(src.size(i)); }
        shape.push_back(dims.indptrLast);
        result = result.expand(shape).contiguous();
        stride = static_cast<uint64_t>(dims.indptrLast);
    }
    return result;
}

#ifdef OPSGNN_DAV_2201
void ValidateInterleavedOutput(const std::vector<std::pair<int64_t, int64_t>> &axes, int64_t elements) {
    if (axes.size() == 2) {
        const auto divisor = std::gcd(axes[0].first, axes[1].first);
        TORCH_CHECK(divisor > 0, "out must have positive strides");
        const bool overlap = axes[0].second > axes[1].first / divisor && axes[1].second > axes[0].first / divisor;
        TORCH_CHECK(!overlap, "out must have non-overlapping storage");
        return;
    }
    // Enumerate metadata addresses only, never source/output values.
    std::vector<int64_t> offsets{0};
    offsets.reserve(static_cast<size_t>(elements));
    for (const auto &axis : axes) {
        const size_t previous = offsets.size();
        offsets.resize(previous * static_cast<size_t>(axis.second));
        for (int64_t i = 1; i < axis.second; ++i)
            for (size_t j = 0; j < previous; ++j)
                offsets[static_cast<size_t>(i) * previous + j] = offsets[j] + i * axis.first;
    }
    std::sort(offsets.begin(), offsets.end());
    TORCH_CHECK(std::adjacent_find(offsets.begin(), offsets.end()) == offsets.end(),
                "out must have non-overlapping storage");
}

void ValidateOutputLayout(const torch::Tensor &out) {
    TORCH_CHECK(out.layout() == at::kStrided, "out must have strided layout");
    if (!out.numel()) return;
    // Sorted spans prove most layouts; legal interleaved views use an exact
    // check when this sufficient condition does not hold.
    std::vector<std::pair<int64_t, int64_t>> axes;
    for (int64_t d = 0; d < out.dim(); ++d)
        if (out.size(d) > 1) axes.emplace_back(out.stride(d), out.size(d));
    std::sort(axes.begin(), axes.end());
    int64_t span = 1;
    bool separated = true;
    for (const auto &axis : axes) {
        TORCH_CHECK(axis.first > 0, "out must have non-overlapping storage");
        separated = separated && axis.first >= span;
        span += (axis.second - 1) * axis.first;
    }
    if (!separated) ValidateInterleavedOutput(axes, out.numel());
}

#endif

torch::Tensor PrepareOutput(const torch::Tensor &src, const std::vector<int64_t> &shape,
                            const c10::optional<torch::Tensor> &outOpt) {
    if (!outOpt.has_value() || !outOpt->defined()) { return torch::empty(shape, src.options()); }
    const auto &out = *outOpt;
    TORCH_CHECK(out.scalar_type() == src.scalar_type(), "out dtype (", out.scalar_type(), ") must match src (",
                src.scalar_type(), ")");
    TORCH_CHECK(out.sizes() == c10::IntArrayRef(shape), "out shape ", out.sizes(), " must be ",
                c10::IntArrayRef(shape));
    TORCH_CHECK(out.device() == src.device(), "out must live on src's device");
#ifdef OPSGNN_DAV_2201
    ValidateOutputLayout(out);
    TORCH_CHECK(!out.is_inference() || c10::InferenceMode::is_enabled(),
                "inference out may only be updated in inference_mode");
    if (!out.is_contiguous() || reinterpret_cast<uintptr_t>(out.data_ptr()) % opsgnn::csr::kDmaAlignBytes)
        return torch::empty(shape, src.options());
#endif
    return out.is_contiguous() ? out : out.contiguous();
}

std::vector<torch::Tensor> ResultTensors(torch::Tensor out, torch::Tensor argOut, int op) {
    if (argOut.defined()) { return {out, argOut}; }
    return {out};
}

std::vector<torch::Tensor> EmptyOutput(torch::Tensor out, torch::Tensor argOut,
                                       const c10::optional<torch::Tensor> &outOpt, const SegDims &dims, int op) {
    if (!outOpt.has_value() || !outOpt->defined()) { out.zero_(); }
    if (argOut.defined()) { argOut.fill_(dims.m); }
    if (outOpt.has_value() && outOpt->defined()) { out = *outOpt; }
    return ResultTensors(out, argOut, op);
}

#ifndef OPSGNN_DAV_2201
bool UseRegisterKernel(int kind, int op, const SegDims &dims, uint64_t stride) {
    const uint32_t itemSize = static_cast<uint32_t>(KindRegisterItemSize(kind));
    const uint64_t rowBytes = static_cast<uint64_t>(dims.k) * itemSize;
    const bool minMaxAligned = op == SEGMENT_CSR_SUM || itemSize == 2 || itemSize == 4 || rowBytes % 256 == 0;
    return itemSize != 0 && dims.e1 == 1 && stride == 0 && dims.m != 0 && minMaxAligned && rowBytes % 32 == 0 &&
           dims.k <= 1024 && dims.m <= 0x7FFFFFFFll;
}

void LaunchReduction(const torch::Tensor &src, const torch::Tensor &ptr, torch::Tensor &out, torch::Tensor &argOut,
                     const SegDims &dims, uint64_t stride, int kind, const ReductionConfig &reduction) {
    const auto e1 = static_cast<uint64_t>(dims.e1);
    const auto m = static_cast<uint64_t>(dims.m);
    const auto nSeg = static_cast<uint32_t>(dims.nSeg);
    const auto k = static_cast<uint32_t>(dims.k);
    if (e1 == 0 || nSeg == 0 || k == 0) { return; }
    aclrtStream stream = c10_npu::getCurrentNPUStream(src.get_device()).stream();
    const void *srcPtr = src.const_data_ptr();
    const int64_t *indptrPtr = ptr.const_data_ptr<int64_t>();
    void *outPtr = out.mutable_data_ptr();
    int64_t *argPtr = argOut.defined() ? argOut.data_ptr<int64_t>() : nullptr;
    if (UseRegisterKernel(kind, reduction.op, dims, stride)) {
        SegmentCsrVector(kind, srcPtr, indptrPtr, outPtr, argPtr, reduction.op, m, nSeg, k, stream, reduction.isMean);
        return;
    }
    const int vecElems = VecElemsFor(kind, k);
    const uint64_t laneCount = vecElems > 0 ? k / static_cast<uint64_t>(vecElems) : k;
    const uint64_t total = e1 * nSeg * laneCount;
    TORCH_CHECK(total < 0xFFFFFFFFull, "segment_csr work items exceed 2^32 (unsupported shape)");
    SegmentCsr(kind, vecElems, srcPtr, indptrPtr, outPtr, argPtr, reduction.op, reduction.isMean, e1, m, nSeg, k,
               stride, stream);
}

#else
// Tensor versions alone do not track .data or independent set_(storage)
// aliases. A storage-address key also detects .data rebinding. Observe normal
// ATen dispatcher calls as a conservative generation
// boundary. Internal CSR allocations/validation are excluded on this thread;
// another thread's operations still invalidate proofs. No tensor data is kept.
std::atomic<uint64_t> mutationEpoch{0};
std::atomic<uint64_t> activeExternalOps{0};
thread_local unsigned internalCsrDepth = 0;
thread_local bool csrCanObserveMutations = true;
struct ExternalOperation : at::ObserverContext {};
std::unique_ptr<at::ObserverContext> ObserveOperationStart(const at::RecordFunction &operation) {
    if (internalCsrDepth) return nullptr;
    // OpCommand may emit these known labels on its asynchronous worker, where
    // the calling thread's TLS guard is not present. Our output-alias writes
    // have already invalidated overlapping proofs explicitly.
    const char *name = operation.name();
    if (name && (!std::strcmp(name, "segment_csr") || !std::strcmp(name, "segment_csr_validate_pointers") ||
                 !std::strcmp(name, "segment_csr_validate_half") || !std::strcmp(name, "segment_csr_validate_int64")))
        return nullptr;
    activeExternalOps.fetch_add(1, std::memory_order_acq_rel);
    mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
    return std::make_unique<ExternalOperation>();
}
void ObserveOperationEnd(const at::RecordFunction &, at::ObserverContext *context) {
    if (!context) return;
    mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
    activeExternalOps.fetch_sub(1, std::memory_order_acq_rel);
}
struct ProofMutationObserver {
    at::CallbackHandle handle;
    ProofMutationObserver()
        : handle(at::addGlobalCallback(at::RecordFunctionCallback(ObserveOperationStart, ObserveOperationEnd)
                                           .needsInputs(false)
                                           .needsOutputs(false)
                                           .samplingProb(1.0)
                                           .scopes({at::RecordScope::FUNCTION}))) {}
};
struct InternalCsrOperations {
    const bool wasRecording;
    const bool previousObservation;
    const bool writesOutput;
    // This guard is thread-local. It avoids constructing RecordFunction
    // records for our own allocations/validation, whose mutations are already
    // covered explicitly; external operations on all other threads still run
    // their callbacks. Restore the incoming recording state on return.
    at::DisableRecordFunctionGuard internalRecords;
    explicit InternalCsrOperations(bool suppliedOutput)
        : wasRecording(at::isRecordFunctionEnabled()), previousObservation(csrCanObserveMutations),
          writesOutput(suppliedOutput) {
        csrCanObserveMutations = wasRecording;
        if (!wasRecording && !internalCsrDepth) mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
        if (writesOutput) {
            // An independent storage alias need not share a Tensor version.
            // Cover the entire write interval, including other threads that
            // validate after our targeted cache erase but before submission.
            activeExternalOps.fetch_add(1, std::memory_order_acq_rel);
            mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
        }
        ++internalCsrDepth;
    }
    ~InternalCsrOperations() {
        if (writesOutput) {
            mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
            activeExternalOps.fetch_sub(1, std::memory_order_acq_rel);
        }
        --internalCsrDepth;
        csrCanObserveMutations = previousObservation;
    }
};
bool CanReuseProof(uint64_t epoch) {
    return csrCanObserveMutations && activeExternalOps.load(std::memory_order_acquire) == 0 &&
           mutationEpoch.load(std::memory_order_acquire) == epoch;
}

// Snapshot before any input materialization. A completed external write must
// not stamp an older contiguous copy with its newer mutation generation.
struct ProofSnapshot {
    uint64_t epoch;
    int64_t version;
    const void *data;
};
ProofSnapshot CaptureProofSnapshot(const torch::Tensor &tensor, uint64_t epoch) {
    return {epoch, tensor.is_inference() ? -1 : tensor._version(), tensor.const_data_ptr()};
}
bool ProofSnapshotIsCurrent(const torch::Tensor &tensor, const ProofSnapshot &snapshot) {
    return snapshot.version >= 0 && !tensor.is_inference() && tensor._version() == snapshot.version &&
           tensor.const_data_ptr() == snapshot.data && CanReuseProof(snapshot.epoch);
}

// The arch22 path validates the actual INT64 pointer contents on the device.
// Cache immutable facts only, keyed by weak Tensor identity, version and layout.
struct PointerFacts {
    int64_t uniformRows = 0;
};
struct PointerEntry {
    c10::weak_intrusive_ptr<c10::TensorImpl> tensor;
    int64_t version, length, offset;
    const void *data;
    uint64_t epoch;
    c10::Device device;
    std::vector<int64_t> sizes, strides;
    PointerFacts facts;
};
struct PointerCache {
    std::mutex mutex;
    std::map<const c10::TensorImpl *, PointerEntry> entries;
};
PointerCache &PointerProofCache() {
    static PointerCache cache;
    return cache;
}
struct DeviceCaps {
    int32_t blocks;
    uint64_t ubBytes;
    uint64_t l2Bytes = 0;
};
DeviceCaps QueryDeviceCaps(int device) {
    static thread_local std::map<int, DeviceCaps> cache;
    auto found = cache.find(device);
    if (found != cache.end()) return found->second;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    DeviceCaps caps{static_cast<int32_t>(std::max<uint32_t>(1, platform->GetCoreNumAiv())), 0};
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, caps.ubBytes);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, caps.l2Bytes);
    int64_t runtimeCores = 0;
    if (aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &runtimeCores) == ACL_SUCCESS && runtimeCores > 0)
        caps.blocks = static_cast<int32_t>(runtimeCores);
    TORCH_CHECK(caps.ubBytes > opsgnn::csr::kUbReserveBytes, "invalid UB capacity");
    cache.emplace(device, caps);
    return caps;
}
torch::Tensor Aligned(torch::Tensor value) {
    value = value.contiguous();
    if (reinterpret_cast<uintptr_t>(value.data_ptr()) % opsgnn::csr::kDmaAlignBytes) value = value.clone();
    return value;
}
bool PointerProofMatches(const PointerEntry &e, const torch::Tensor &original, const SegDims &dims,
                         const ProofSnapshot &snapshot) {
    return e.tensor.lock().get() == original.unsafeGetTensorImpl() && e.version == snapshot.version &&
           e.epoch == snapshot.epoch && e.length == dims.m && e.offset == original.storage_offset() &&
           e.data == snapshot.data && e.device == original.device() && original.sizes().equals(e.sizes) &&
           original.strides().equals(e.strides) && ProofSnapshotIsCurrent(original, snapshot);
}

PointerFacts CheckPointerValues(const torch::Tensor &packed, const SegDims &dims) {
    const int blocks = QueryDeviceCaps(packed.get_device()).blocks;
    auto flags = torch::empty({blocks * 8}, packed.options().dtype(at::kInt));
    const auto current = c10_npu::getCurrentNPUStream(packed.get_device());
    auto *const pointerAddress = packed.data_ptr<int64_t>();
    auto *const flagAddress = flags.data_ptr<int32_t>();
    const auto pointerBatches = packed.numel() / dims.indptrLast;
    std::array<c10::Storage, 2> owners{packed.storage(), flags.storage()};
    for (const auto &owner : owners) c10_npu::NPUCachingAllocator::recordStream(owner.data_ptr(), current);
    const auto stream = current.stream(false);
    at_npu::native::OpCommand::RunOpApi(
        "segment_csr_validate_pointers",
        [pointerAddress, flagAddress, pointerBatches, owners = std::move(owners), dims, blocks, stream]() {
            SegmentCsrValidatePointers(pointerAddress, pointerBatches, dims.nSeg, dims.m, flagAddress, blocks, stream);
            return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
        });
    const auto host = flags.cpu();
    const auto *data = host.data_ptr<int32_t>();
    bool uniform = dims.nSeg > 0 && dims.m > 0 && dims.m % dims.nSeg == 0;
    for (int i = 0; i < blocks; ++i) {
        TORCH_CHECK(data[i * 8] == 1, "indptr must be nondecreasing and within [0, src.size(dim)]");
        uniform = uniform && data[i * 8 + 1] == 1;
    }
    PointerFacts facts{uniform ? dims.m / dims.nSeg : 0};
    return facts;
}

PointerFacts ValidatePointers(const torch::Tensor &original, const torch::Tensor &packed, const SegDims &dims,
                              const ProofSnapshot &snapshot) {
    if (!packed.numel()) return {};
    auto &state = PointerProofCache();
    auto &mutex = state.mutex;
    auto &cache = state.entries;
    const auto identity = original.unsafeGetTensorImpl();
    const int64_t version = snapshot.version;
    const uint64_t epoch = snapshot.epoch;
    // Own a metadata-only copy; never keep a reference into the shared map.
    // Supplied-out writes and explicit invalidation advance the global epoch,
    // so they invalidate this thread's recent proof as well as shared entries.
    static thread_local c10::optional<PointerEntry> recent;
    if (version >= 0) {
        if (recent && PointerProofMatches(*recent, original, dims, snapshot)) return recent->facts;
        std::lock_guard<std::mutex> lock(mutex);
        auto found = cache.find(identity);
        if (found != cache.end() && PointerProofMatches(found->second, original, dims, snapshot)) {
            recent = found->second;
            return recent->facts;
        }
    }
    const PointerFacts facts = CheckPointerValues(packed, dims);
    if (version >= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        if (ProofSnapshotIsCurrent(original, snapshot)) {
            if (cache.size() >= 128) cache.erase(cache.begin());
            auto inserted = cache.insert_or_assign(
                identity, PointerEntry{c10::weak_intrusive_ptr<c10::TensorImpl>(original.getIntrusivePtr()), version,
                                       dims.m, original.storage_offset(), snapshot.data, epoch, original.device(),
                                       original.sizes().vec(), original.strides().vec(), facts});
            recent = inserted.first->second;
        }
    }
    return facts;
}

struct SourceRangeEntry {
    c10::weak_intrusive_ptr<c10::TensorImpl> tensor;
    int64_t version, criterion, offset;
    const void *data;
    uint64_t epoch;
    c10::Device device;
    at::ScalarType dtype;
    std::vector<int64_t> sizes, strides;
    bool bounded;
};

struct SourceCache {
    std::mutex mutex;
    std::map<const c10::TensorImpl *, SourceRangeEntry> entries;
};
SourceCache &SourceProofCache() {
    static SourceCache cache;
    return cache;
}

// set_(untyped_storage()) aliases need not share Tensor version counters.
// Invalidate facts for every cached alias that this operator is about to write,
// including cached tensors that are not inputs to the current invocation.
template <typename Cache> void InvalidateStorageProofs(Cache &state, const torch::Tensor &output) {
    const auto begin = reinterpret_cast<uintptr_t>(output.storage().data_ptr().get());
    const auto end = begin + output.storage().nbytes();
    std::lock_guard<std::mutex> lock(state.mutex);
    for (auto it = state.entries.begin(); it != state.entries.end();) {
        const auto tensor = it->second.tensor.lock();
        bool erase = !tensor;
        if (tensor && tensor->device() == output.device()) {
            const auto cachedBegin = reinterpret_cast<uintptr_t>(tensor->storage().data_ptr().get());
            const auto cachedEnd = cachedBegin + tensor->storage().nbytes();
            erase = begin < cachedEnd && cachedBegin < end;
        }
        if (erase)
            it = state.entries.erase(it);
        else
            ++it;
    }
}

bool CheckSourceRange(const torch::Tensor &src, int64_t bound) {
    const bool half = src.scalar_type() == at::kHalf;
    TORCH_INTERNAL_ASSERT(half || src.scalar_type() == at::kLong);
    TORCH_INTERNAL_ASSERT(!half || bound == 6);
    auto input = src.contiguous();
    if (reinterpret_cast<uintptr_t>(input.data_ptr()) % opsgnn::csr::kDmaAlignBytes) input = input.clone();
    const int blocks = std::max<uint32_t>(1, platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    // Every core owns one complete 32-byte flag block, including DMA padding.
    auto flags = torch::empty({blocks * 8}, src.options().dtype(at::kInt));
    const auto current = c10_npu::getCurrentNPUStream(src.device().index());
    void *const inputAddress = input.data_ptr();
    auto *const flagAddress = flags.data_ptr<int32_t>();
    const auto elements = input.numel();
    std::array<c10::Storage, 2> owners{input.storage(), flags.storage()};
    for (const auto &owner : owners) c10_npu::NPUCachingAllocator::recordStream(owner.data_ptr(), current);
    const auto stream = current.stream(false);
    at_npu::native::OpCommand::RunOpApi(
        half ? "segment_csr_validate_half" : "segment_csr_validate_int64",
        [inputAddress, flagAddress, elements, owners = std::move(owners), bound, blocks, stream, half]() {
            if (half)
                SegmentCsrValidateHalf(inputAddress, elements, flagAddress, blocks, stream);
            else
                SegmentCsrValidateInt64(inputAddress, elements, bound, flagAddress, blocks, stream);
            return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
        });
    const auto host = flags.cpu();
    const auto *valid = host.data_ptr<int32_t>();
    for (int i = 0; i < blocks; ++i)
        if (valid[i * 8] != 1) return false;
    return true;
}

bool SourceProofMatches(const SourceRangeEntry &e, const torch::Tensor &src, int64_t criterion,
                        const ProofSnapshot &snapshot) {
    return e.tensor.lock().get() == src.unsafeGetTensorImpl() && e.version == snapshot.version &&
           e.epoch == snapshot.epoch && e.criterion == criterion && e.offset == src.storage_offset() &&
           e.data == snapshot.data && e.device == src.device() && e.dtype == src.scalar_type() &&
           src.sizes().equals(e.sizes) && src.strides().equals(e.strides) && ProofSnapshotIsCurrent(src, snapshot);
}

bool ProveSourceRange(const torch::Tensor &src, const torch::Tensor &packed, int64_t length, int64_t reduce,
                      const ProofSnapshot &snapshot) {
    if (!packed.numel() || !length) return false;
    const bool half = src.scalar_type() == at::kHalf;
    if (!half && length > INT32_MAX) return false;
    // A half proof is independent of segment length; the launcher separately
    // requires complete power-of-two segments of at most sixteen rows.
    // dtype remains part of the cache key, so criterion 6 cannot reuse an
    // INT64 proof with the same numerical bound.
    const int64_t criterion = half ? 6 : (reduce <= SEGMENT_CSR_MEAN ? INT32_MAX / length : INT32_MAX);
    auto &state = SourceProofCache();
    auto &mutex = state.mutex;
    auto &cache = state.entries;
    const auto *identity = src.unsafeGetTensorImpl();
    const int64_t version = snapshot.version;
    const uint64_t epoch = snapshot.epoch;
    // Keep only the last genuinely verified predicate and its complete key.
    // The weak identity does not retain the tensor or its device allocation.
    static thread_local c10::optional<SourceRangeEntry> recent;
    if (version >= 0) {
        if (recent && SourceProofMatches(*recent, src, criterion, snapshot)) return recent->bounded;
        std::lock_guard<std::mutex> lock(mutex);
        auto found = cache.find(identity);
        if (found != cache.end() && SourceProofMatches(found->second, src, criterion, snapshot)) {
            recent = found->second;
            return recent->bounded;
        }
    }
    // Validation statistics are computed on the current NPU stream. Cache only
    // the predicate and invalidation metadata, never values or reduction results.
    // A fresh tensor/version pays this proof inside the public interface.
    at::NoGradGuard noGrad;
    // Validate the exact packed tensor passed to the kernel. Proving the
    // original again could authorize an earlier, larger-valued source copy.
    const bool bounded = CheckSourceRange(packed, criterion);
    if (version >= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        if (ProofSnapshotIsCurrent(src, snapshot)) {
            if (cache.size() >= 128) cache.erase(cache.begin());
            auto inserted = cache.insert_or_assign(
                identity, SourceRangeEntry{c10::weak_intrusive_ptr<c10::TensorImpl>(src.getIntrusivePtr()), version,
                                           criterion, src.storage_offset(), snapshot.data, epoch, src.device(),
                                           src.scalar_type(), src.sizes().vec(), src.strides().vec(), bounded});
            recent = inserted.first->second;
        }
    }
    return bounded;
}

SegmentCsrPlan CachedLaunchPlan(const csr_host::PlanInputs &inputs, int device) {
    using Key = std::array<int64_t, 14>;
    const Key key{inputs.batches,
                  inputs.length,
                  inputs.channels,
                  inputs.segments,
                  inputs.indexBatches,
                  inputs.dtype,
                  inputs.reduce,
                  inputs.hasArg,
                  static_cast<int64_t>(inputs.uniformRows),
                  device,
                  inputs.blocks,
                  static_cast<int64_t>(inputs.ubBytes),
                  static_cast<int64_t>(inputs.l2Bytes),
                  inputs.boundedSource};
    struct Entry {
        Key key{};
        SegmentCsrPlan plan{};
    };
    static thread_local std::array<Entry, 16> cache;
    static thread_local size_t size = 0, next = 0, recent = 0;
    bool hit = size && cache[recent].key == key;
    if (!hit)
        for (size_t i = 0; i < size; ++i)
            if (cache[i].key == key) {
                recent = i;
                hit = true;
                break;
            }
    return [&] {
        if (hit) return cache[recent].plan;
        const auto selected = csr_host::MakeLaunchPlan(inputs);
        cache[next] = {key, selected};
        recent = next;
        next = (next + 1) % cache.size();
        size = std::min(size + 1, cache.size());
        return selected;
    }();
}

struct CsrInvocation {
    void *sourceAddress;
    int64_t *pointerAddress;
    void *outputAddress;
    int64_t *argAddress;
    std::array<c10::Storage, 4> owners;
    SegmentCsrPlan plan;
};
void QueueReduction(CsrInvocation invocation, aclrtStream stream, bool callerOut) {
    auto &owners = invocation.owners;
    const auto &plan = invocation.plan;
    const auto sourceAddress = invocation.sourceAddress;
    const auto pointerAddress = invocation.pointerAddress;
    const auto outputAddress = invocation.outputAddress;
    const auto argAddress = invocation.argAddress;
    const bool shortFloat = plan.path == SegmentCsrKernelPath::ShortUniform;
    // Only the native caching allocator can recycle a fresh output on
    // its allocation stream without synchronizing. Keep uncached/custom
    // allocation deleters on the framework's normal release path.
    static const c10::DeleterFnPtr cachedDeleter = []() -> c10::DeleterFnPtr {
        auto *allocator = c10_npu::NPUCachingAllocator::get();
        if (allocator->name() != "native") return nullptr;
        // The native zero-byte allocation has no device allocation and
        // always carries its cache deleter, including uncached mode.
        return allocator->allocate(0).get_deleter();
    }();
    const bool releaseOutput =
        !shortFloat && cachedDeleter && !callerOut && owners[2].data_ptr().get_deleter() == cachedDeleter;
    const bool releaseArg =
        cachedDeleter && argAddress != nullptr && owners[3].data_ptr().get_deleter() == cachedDeleter;
    // RunOpApiV2 synchronously moves this non-const function into the
    // framework queue before returning. Avoid the extra heap allocation
    // and Storage-reference copy made by the older by-value API.
    std::function<int()> handler = [sourceAddress, pointerAddress, outputAddress, argAddress,
                                    owners = std::move(owners), plan, stream, releaseOutput, releaseArg]() mutable {
        // The queued worker calls this handler in place. After successful
        // submission, release selected fresh outputs here for allocator
        // reuse. Short-float outputs keep their owner through the separate
        // release thread; retained Python results stay independent either
        // way. Inputs and supplied out keep their recorded-stream lifetime.
        // Failures retain every owner for a possible synchronous/OOM retry.
        const auto status = SegmentCsr(sourceAddress, pointerAddress, outputAddress, argAddress, plan, stream);
        if (status == ACL_SUCCESS) {
            if (releaseOutput) owners[2] = c10::Storage{};
            if (releaseArg) owners[3] = c10::Storage{};
        }
        return static_cast<int>(status);
    };
    at_npu::native::OpCommand::RunOpApiV2("segment_csr", handler);
}

void LaunchReduction22(torch::Tensor src, torch::Tensor ptr, torch::Tensor out, torch::Tensor arg, const SegDims &dims,
                       uint64_t stride, int kind, const ReductionConfig &reduction, PointerFacts facts, bool bounded,
                       bool callerOut, bool materializedInputs) {
    if (!out.numel()) return;
    const auto caps = QueryDeviceCaps(src.get_device());
    csr_host::PlanInputs inputs{dims.e1,
                                dims.m,
                                dims.k,
                                dims.nSeg,
                                stride ? dims.e1 : 1,
                                static_cast<SegmentCsrReduce>(reduction.isMean ? SEGMENT_CSR_MEAN : reduction.op),
                                static_cast<SegmentCsrDtype>(kind),
                                0,
                                caps.blocks};
    inputs.uniformRows = facts.uniformRows;
    inputs.hasArg = arg.defined();
    inputs.boundedSource = bounded;
    inputs.ubBytes = caps.ubBytes;
    inputs.l2Bytes = caps.l2Bytes;
    const auto plan = CachedLaunchPlan(inputs, src.get_device());
    const auto current = c10_npu::getCurrentNPUStream(src.get_device());
    // Tensor references do not freeze their addresses: .data rebinding changes
    // the shared TensorImpl. Snapshot addresses now and retain the matching
    // Storage objects independently until the kernel has been submitted.
    // A raw in-place replacement of a retained StorageImpl's allocation is
    // different from Tensor rebinding and requires its own synchronization.
    void *const sourceAddress = src.data_ptr();
    auto *const pointerAddress = ptr.data_ptr<int64_t>();
    void *const outputAddress = out.data_ptr();
    auto *const argAddress = arg.defined() ? arg.data_ptr<int64_t>() : nullptr;
    std::array<c10::Storage, 4> owners{src.storage(), ptr.storage(), out.storage(),
                                       arg.defined() ? arg.storage() : c10::Storage{}};
    // Inputs and a supplied out may have been allocated on another stream.
    // Fresh outputs/args already belong to this allocation stream. Recording
    // those unnecessarily delays allocator reuse in long asynchronous batches
    // (torch_npu 2.7.1); the local owners retain their original storages until
    // the direct launch below has submitted its device work.
    for (size_t i = 0; i < 2; ++i) c10_npu::NPUCachingAllocator::recordStream(owners[i].data_ptr(), current);
    if (callerOut) c10_npu::NPUCachingAllocator::recordStream(owners[2].data_ptr(), current);
    // The finalized short path retains a fresh output's queue owner
    // through the framework release stage. This permits allocator reclamation
    // to overlap later public calls when Python immediately drops the result.
    // Other small working sets overlap Host preparation and submission.
    // Large buffers use direct submission to avoid keeping many distinct
    // outputs alive in the Host task queue.
    const uint64_t sourceBytes = src.nbytes();
    const uint64_t outputBytes = out.nbytes() + (arg.defined() ? arg.nbytes() : 0);
    const uint64_t queueBudget =
        caps.l2Bytes ? std::min<uint64_t>(64ULL * 1024 * 1024, caps.l2Bytes / 3) : 64ULL * 1024 * 1024;
    if (sourceBytes <= queueBudget && outputBytes <= queueBudget - sourceBytes) {
        // Materialization was issued through ATen. Drain its Host submissions
        // before returning so later TensorImpl rebinding cannot redirect the
        // producer of a packed input. Device work remains asynchronous.
        const auto stream = current.stream(materializedInputs);
        TORCH_CHECK(stream != nullptr, "segment_csr failed to drain the NPU submission queue");
        QueueReduction({sourceAddress, pointerAddress, outputAddress, argAddress, std::move(owners), plan}, stream,
                       callerOut);
        return;
    }
    // The official torch_npu queue-to-ACL bridge drains previously enqueued
    // host submissions, then returns this current stream's handle. It does
    // not wait for normal device work to finish. Append CSR directly so later
    // public ATen calls preserve the same stream ordering without another
    // RunOpApi worker handoff (also used by the existing arch35 CSR path).
    const auto stream = current.stream();
    TORCH_CHECK(stream != nullptr, "segment_csr failed to drain the NPU submission queue");
    const auto launchError = SegmentCsr(sourceAddress, pointerAddress, outputAddress, argAddress, plan, stream);
    TORCH_CHECK(launchError == ACL_SUCCESS, "segment_csr kernel launch failed with ACL error ", launchError);
}

#endif

SegDims CheckedDimensions(const torch::Tensor &src, const torch::Tensor &indptr) {
    TORCH_CHECK(indptr.dim() >= 1, "indptr must have at least 1 dim");
    TORCH_CHECK(src.dim() >= indptr.dim(), "src.dim() must be >= indptr.dim()");
    const SegDims dims = ComputeDims(src, indptr);
    Validate(src, indptr, dims);
    TORCH_CHECK(src.layout() == at::kStrided && indptr.layout() == at::kStrided,
                "src and indptr must have strided layout");
#ifndef OPSGNN_DAV_2201
    TORCH_CHECK(dims.nSeg < std::numeric_limits<uint32_t>::max() && dims.k <= std::numeric_limits<uint32_t>::max(),
                "segment count or channel count exceeds the kernel limit");
#endif
    return dims;
}

void FinishOutput(torch::Tensor &out, const c10::optional<torch::Tensor> &outOpt) {
    if (outOpt.has_value() && outOpt->defined() && out.data_ptr() != outOpt->data_ptr()) {
        outOpt->copy_(out);
        out = *outOpt;
    }
#ifdef OPSGNN_DAV_2201
    else if (outOpt.has_value() && outOpt->defined()) {
        out.unsafeGetTensorImpl()->bump_version();
    }
#endif
}

#ifdef OPSGNN_DAV_2201
bool NeedsSourceProof(int kind, int reduceCode, bool returnArg, uint64_t rows, int64_t channels) {
    const bool proveInt64 = kind == KIND_I64 && (reduceCode <= SEGMENT_CSR_MEAN || !returnArg);
    // Shape selects whether a numeric proof is useful, never whether the
    // bounded tree is authorized. Check every value of the exact packed input
    // on the current stream before setting boundedSource.
    const bool proveHalf = kind == KIND_F16 && reduceCode <= SEGMENT_CSR_MEAN && rows > 0 && rows <= 16 &&
                           !(rows & (rows - 1)) && channels > 0 && channels <= opsgnn::csr::kWholeCastColumnLimit &&
                           channels % 16 == 0;
    return proveInt64 || proveHalf;
}

void ProtectAliasedInputs(torch::Tensor &srcC, torch::Tensor &indptrC, const torch::Tensor &out,
                          const c10::optional<torch::Tensor> &outOpt) {
    if (outOpt.has_value() && outOpt->defined()) {
        if (at::get_overlap_status(srcC, out) != at::MemOverlapStatus::No) srcC = srcC.clone();
        if (at::get_overlap_status(indptrC, out) != at::MemOverlapStatus::No) indptrC = indptrC.clone();
        InvalidateStorageProofs(PointerProofCache(), *outOpt);
        InvalidateStorageProofs(SourceProofCache(), *outOpt);
    }
}

bool RecordOriginalInputs(const torch::Tensor &src, const torch::Tensor &indptr, const torch::Tensor &srcC,
                          const torch::Tensor &indptrC, const ProofSnapshot &sourceSnapshot,
                          const ProofSnapshot &pointerSnapshot) {
    const bool copiedSource = srcC.const_data_ptr() != sourceSnapshot.data;
    const bool copiedPointers = indptrC.const_data_ptr() != pointerSnapshot.data;
    if (copiedSource || copiedPointers) {
        const auto current = c10_npu::getCurrentNPUStream(src.get_device());
        // The original temporary inputs may be freed after the packed copies
        // are submitted but before their device reads finish.
        if (copiedSource) c10_npu::NPUCachingAllocator::recordStream(src.storage().data_ptr(), current);
        if (copiedPointers) c10_npu::NPUCachingAllocator::recordStream(indptr.storage().data_ptr(), current);
    }
    return copiedSource || copiedPointers;
}

#endif

} // namespace

void initialize_segment_csr_proof_observer() {
#ifdef OPSGNN_DAV_2201
    // Register while the extension module is initialized, before GIL-released
    // public calls. ATen global callback removal requires exclusive shutdown;
    // retain this tiny registration for process lifetime instead of removing
    // it from an unsafe static-destructor ordering. This does not touch NPU.
    static const auto *observer = new ProofMutationObserver();
    (void)observer;
#endif
}

void invalidate_segment_csr_proofs() {
#ifdef OPSGNN_DAV_2201
    mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
#endif
}

std::vector<torch::Tensor> segment_csr(torch::Tensor src, torch::Tensor indptr, c10::optional<torch::Tensor> outOpt,
                                       const std::string &reduce, bool returnArg) {
#ifdef OPSGNN_DAV_2201
    const InternalCsrOperations internalOperations(outOpt.has_value() && outOpt->defined());
#endif
#ifndef OPSGNN_DAV_2201
    // Existing arch35 extrema kernels require an arg buffer even when Python
    // returns values only. Keep the established 950 invocation unchanged.
    returnArg = true;
#endif
    const auto reduction = ParseReduction(reduce);
    const SegDims dims = CheckedDimensions(src, indptr);
    const int kind = KindFor(src);
    const c10::DeviceGuard deviceGuard(src.device());
#ifdef OPSGNN_DAV_2201
    const uint64_t proofEpoch = mutationEpoch.load(std::memory_order_acquire);
    const ProofSnapshot sourceSnapshot = CaptureProofSnapshot(src, proofEpoch);
    const ProofSnapshot pointerSnapshot = CaptureProofSnapshot(indptr, proofEpoch);
#endif
    torch::Tensor srcC = src.is_contiguous() ? src : src.contiguous();
    uint64_t stride = 0;
    torch::Tensor indptrC = PreparePointers(src, indptr, dims, stride);
    std::vector<int64_t> outShape(src.sizes().begin(), src.sizes().end());
    outShape[dims.dim] = dims.nSeg;
    torch::Tensor out = PrepareOutput(src, outShape, outOpt);
#ifdef OPSGNN_DAV_2201
    srcC = Aligned(srcC);
    indptrC = Aligned(indptrC);
    const auto facts = ValidatePointers(indptr, indptrC, dims, pointerSnapshot);
    const int reduceCode = reduction.isMean ? SEGMENT_CSR_MEAN : reduction.op;
    const bool bounded = NeedsSourceProof(kind, reduceCode, returnArg, facts.uniformRows, dims.k) &&
                         ProveSourceRange(src, srcC, dims.m, reduceCode, sourceSnapshot);
    ProtectAliasedInputs(srcC, indptrC, out, outOpt);
#endif
    torch::Tensor argOut;
    if (reduction.op != SEGMENT_CSR_SUM && returnArg) {
        argOut = torch::empty(outShape, src.options().dtype(at::kLong));
    }
    if (src.numel() == 0) { return EmptyOutput(out, argOut, outOpt, dims, reduction.op); }
#ifdef OPSGNN_DAV_2201
    const bool materialized = RecordOriginalInputs(src, indptr, srcC, indptrC, sourceSnapshot, pointerSnapshot);
    LaunchReduction22(srcC, indptrC, out, argOut, dims, stride, kind, reduction, facts, bounded,
                      outOpt.has_value() && outOpt->defined() && out.data_ptr() == outOpt->data_ptr(), materialized);
#else
    LaunchReduction(srcC, indptrC, out, argOut, dims, stride, kind, reduction);
#endif
    FinishOutput(out, outOpt);
    return ResultTensors(out, argOut, reduction.op);
}

} // namespace ops_gnn

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
#include <cstdint>

namespace opsgnn::csr {
// A2 AIV capacities and instruction fields, shared by Host budgeting and kernels.
constexpr uint64_t kArch22UbCapacityBytes = 192 * 1024;
constexpr uint64_t kUbReserveBytes = 8 * 1024;
constexpr uint32_t kDmaAlignBytes = 32;
constexpr uint32_t kVectorLanes = 64; // FP32 / INT32 lanes in a 256-byte vector.
constexpr uint32_t kHalfVectorLanes = 2 * kVectorLanes;
constexpr uint32_t kInt64VectorLanes = kVectorLanes / 2;
constexpr uint32_t kVectorRepeatLimit = 255;
constexpr uint32_t kVectorStrideLimit = 255; // Repeat strides, in DMA blocks.
constexpr uint32_t kPipelineSlotBytes = 64 * 1024;

// Capacity search bounds; these are scheduling choices, not input restrictions.
constexpr uint32_t kIntegerTileColumns = 128;
constexpr uint32_t kIntegerTileRows = 128;
constexpr uint32_t kDefaultTileRows = 64;
constexpr uint32_t kMinAmortizedTileRows = 32;
constexpr uint32_t kLongFloatTileColumns = 32;
constexpr uint32_t kShortUniformColumns = 16;
constexpr uint32_t kMediumUniformColumns = 32;
constexpr uint32_t kWideUniformColumns = 64;
constexpr uint32_t kWholeCastColumnLimit = 64;
constexpr uint32_t kPipelineRowLimit = 64;
constexpr uint32_t kLongSegmentMinRows = 128;
constexpr uint32_t kInt64NarrowGroupLimit = 128;
constexpr uint32_t kMaxGroupedSegments = 32;
constexpr uint32_t kInt64FallbackGroups = 4;
constexpr uint32_t kInt32BatchRowBudget = 24576;  // Input elements; input + half scratch <= 144 KiB.
constexpr uint32_t kInt32MeanOutputBudget = 4096; // Two 4-byte mean planes <= 32 KiB.
constexpr uint32_t kLongInt32MinSlotRows = 64;
constexpr uint32_t kLongInt64MinSlotRows = 32;
constexpr uint32_t kHalfPowerMeanLimit = 16384; // Reciprocal remains a normal FP16 power of two.
constexpr uint32_t kOrderedChunkIssueCost = 64; // Relative DMA/queue submission weight, not cycles.

// INT64 digit sum: input(8) + four FP32 digits(16) + half tree(8) + a(8),
// plus four mask bits per input element, conservatively rounded up to one byte.
constexpr uint32_t kInt64DecompInputBytesPerElem = 8 + 16 + 8 + 8 + 1;
// Output planes b(8), c(8), and three two-word Gather-offset planes(24).
constexpr uint32_t kInt64DecompOutputBytesPerElem = 8 + 8 + 24;
constexpr uint32_t kInt64AccumulatorBytesPerColumn = 8;
// INT64 extrema: input(8), half tree(4), candidate(4); acc(8), bias(8).
constexpr uint32_t kExtremaInputBytesPerElem = 8 + 4 + 4;
constexpr uint32_t kExtremaBytesPerColumn = 8 + 8;
constexpr uint64_t kExtremaReserveBytes = 16 * 1024; // Masks and aligned scratch blocks.
// Two input/output queue slots, with live tree/mean or low-word/offset planes.
constexpr uint32_t kInt32PipelineInputBytesPerElem = 2 * 4 + 2;
constexpr uint32_t kInt32PipelineOutputBytesPerElem = 2 * 4 + 4 + 4;
constexpr uint32_t kInt64PipelineInputBytesPerElem = 2 * 8 + 4;
constexpr uint32_t kInt64PipelineOutputBytesPerElem = 2 * 8 + 8;
// Narrow INT64: input(8), low words(4); packed output(8), values(4), offsets(4).
constexpr uint32_t kInt64NarrowInputBytesPerElem = 8 + 4;
constexpr uint32_t kInt64NarrowOutputBytesPerElem = 8 + 4 + 4;
// FP16 extrema: two half input slots(4), half-size tree(1), one mask bit.
constexpr uint32_t kHalfExtremaInputBytesPerElem = 2 * 2 + 1;
constexpr uint32_t kHalfExtremaInputBitsPerElem = kHalfExtremaInputBytesPerElem * 8 + 1;
constexpr uint32_t kHalfExtremaOutputBytesPerElem = 2;
} // namespace opsgnn::csr

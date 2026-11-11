/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0.
 * Please refer to the License for details.
 */

#pragma once

#include <cstdint>

constexpr uint32_t GATHER_COO_MAX_FEATURE_TILE = 2048;
constexpr uint32_t GATHER_COO_SLAB_BYTES = 64 * 1024;
constexpr uint32_t GATHER_COO_SLAB_MAX_ROWS = 128;

struct GatherCooTilingData {
    uint64_t batchCount;
    uint64_t sourceRows;
    uint64_t indexRows;
    // featureCount is expressed in the kernel storage type. For 8-byte
    // source elements, one logical element occupies two uint32_t lanes.
    uint64_t featureCount;
    uint64_t logicalFeatureCount;
    uint32_t elementBytes;
    uint64_t totalRows;
    // The public index is int64, but the kernel reads it as two uint32 lanes
    // because scalar GetValue<int64_t> is not supported consistently by the
    // Ascend 950 vector core path.
    uint64_t indexStorageElements;
    uint64_t sourceStorageElements;
    uint64_t outputStorageElements;
    uint64_t rowsPerCore;
    uint64_t extraCoreCount;
    uint32_t featureTile;
    uint32_t coreNum;
    uint32_t useRunCache;
    uint32_t outputSlabRows;
};

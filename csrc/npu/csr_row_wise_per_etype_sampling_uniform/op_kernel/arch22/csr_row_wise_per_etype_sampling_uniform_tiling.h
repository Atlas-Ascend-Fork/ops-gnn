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
namespace csr_per_etype {

struct CsrRowWisePerEtypeSamplingUniformTiling {
    uint32_t numRows;
    uint32_t numSamples;
    uint32_t replace;
    uint32_t hasData;
    uint32_t seed;
    uint32_t selectAll;
    uint32_t numTotalRows;
    uint32_t ubAvailable;
    uint32_t dtypeBytes;
    uint32_t pickStride;
};

constexpr uint32_t kUbInstances = 16;

constexpr uint32_t kGoldenRatioHash = 2654435761u;
constexpr uint32_t kGoldenRatioOffset = 0x9e3779b9u;
constexpr uint32_t kRngFallbackSeed = 0x12345678u;

}  // namespace csr_per_etype
}  // namespace opsgnn

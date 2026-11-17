/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Shared device-side Philox4x32-10 counter-based RNG for the random walk kernels.
 */

#pragma once

#include <cstdint>

#include "simt_api/device_functions.h"

namespace opsgnn {
namespace philox {

constexpr uint32_t PHILOX_M0 = 0xD2511F53U;
constexpr uint32_t PHILOX_M1 = 0xCD9E8D57U;
constexpr uint32_t PHILOX_W0 = 0x9E3779B9U;
constexpr uint32_t PHILOX_W1 = 0xBB67AE85U;

struct PhiloxState {
    uint32_t key0;
    uint32_t key1;
    uint32_t counter0;
    uint32_t counter1;
    uint32_t counter2;
    uint32_t counter3;
    uint32_t values[4];
    uint32_t lane;
};

__simt_callee__ inline void PhiloxRefill(PhiloxState &state)
{
    uint32_t c0 = state.counter0;
    uint32_t c1 = state.counter1;
    uint32_t c2 = state.counter2;
    uint32_t c3 = state.counter3;
    uint32_t key0 = state.key0;
    uint32_t key1 = state.key1;

    for (uint32_t round = 0; round < 10; ++round) {
        const uint32_t hi0 = __umulhi(PHILOX_M0, c0);
        const uint32_t lo0 = PHILOX_M0 * c0;
        const uint32_t hi1 = __umulhi(PHILOX_M1, c2);
        const uint32_t lo1 = PHILOX_M1 * c2;
        c0 = hi1 ^ c1 ^ key0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ key1;
        c3 = lo0;
        key0 += PHILOX_W0;
        key1 += PHILOX_W1;
    }

    state.values[0] = c0;
    state.values[1] = c1;
    state.values[2] = c2;
    state.values[3] = c3;
    state.lane = 0;
    state.counter0 += 1;
    if (state.counter0 == 0) {
        state.counter1 += 1;
    }
}

__simt_callee__ inline uint32_t PhiloxNext(PhiloxState &state)
{
    if (state.lane >= 4) {
        PhiloxRefill(state);
    }
    return state.values[state.lane++];
}

__simt_callee__ inline PhiloxState MakePhiloxState(uint64_t seed, uint64_t offset, uint64_t walkIndex)
{
    PhiloxState state;
    state.key0 = static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(offset);
    state.key1 = static_cast<uint32_t>(seed >> 32) ^ static_cast<uint32_t>(offset >> 32);
    state.counter0 = 0;
    state.counter1 = 0;
    state.counter2 = static_cast<uint32_t>(walkIndex);
    state.counter3 = static_cast<uint32_t>(walkIndex >> 32);
    state.lane = 4;
    return state;
}

} // namespace philox
} // namespace opsgnn

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

#include <acl/acl.h>
#include <cstdint>

constexpr uint32_t FPS_REDUCE_WORK_BYTES = 4096;
constexpr uint32_t FPS_REDUCE_RESULT_BYTES = 32;
constexpr uint32_t FPS_MAX_TILE_POINTS = 16384;

template <typename T>
struct FpsLaunchArgs {
    const T* src_soa;
    const int64_t* ptr;
    const int64_t* out_ptr;
    const int64_t* start;
    T* dist;
    float* best_dist;
    int64_t* best_index;
    int64_t* out;
    int64_t num_points;
    int64_t feature_dim;
    int64_t batch_size;
    int64_t single_out_size;
    int64_t single_start;
    uint32_t tile_points;
    uint32_t tile_bytes;
    uint32_t promoted_tile_bytes;
    uint32_t cascade_tile_bytes;
    uint32_t center_bytes;
};

// DType: 0 = float32, 1 = float16 stored as raw uint16_t on the host.
template <typename T, int DType = 0>
void Fps(const T* src, const FpsLaunchArgs<T>& args, aclrtStream stream);

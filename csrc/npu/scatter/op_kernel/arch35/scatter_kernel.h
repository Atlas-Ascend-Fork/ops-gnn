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

#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/common_functions.h"

#include "scatter_tiling.h"

namespace ScatterNpu {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 256;
constexpr uint32_t VECTOR_COPY_BYTES = 16 * 1024;
constexpr uint32_t ZERO_COPY_BYTES = 128 * 1024;
constexpr uint32_t PROMOTE_TILE_ELEMENTS = 4096;
constexpr uint32_t PROMOTE_SLOT_FLOATS =
    PROMOTE_TILE_ELEMENTS + PROMOTE_TILE_ELEMENTS / 2;
constexpr uint32_t MINMAX_ARG_SLOT_FLOATS = 15360;
constexpr uint32_t MINMAX_FINAL_SLOT_FLOATS = 11264;


template <typename T>
__simt_callee__ __aicore__ inline T ZeroValue()
{
    return static_cast<T>(0);
}

template <typename T>
__simt_callee__ __aicore__ inline T OneValue()
{
    return static_cast<T>(1);
}

template <typename T>
__simt_callee__ __aicore__ inline T HighestValue()
{
    if constexpr (IsSameType<T, float>::value) {
        return static_cast<T>(3.402823466e38F);
    } else if constexpr (IsSameType<T, half>::value) {
        return static_cast<T>(65504.0F);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        return static_cast<T>(3.38953139e38F);
    } else if constexpr (IsSameType<T, int8_t>::value) {
        return static_cast<T>(127);
    } else if constexpr (IsSameType<T, int16_t>::value) {
        return static_cast<T>(32767);
    } else if constexpr (IsSameType<T, int32_t>::value) {
        return static_cast<T>(2147483647);
    } else {
        return static_cast<T>(255);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T LowestValue()
{
    if constexpr (IsSameType<T, float>::value) {
        return static_cast<T>(-3.402823466e38F);
    } else if constexpr (IsSameType<T, half>::value) {
        return static_cast<T>(-65504.0F);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        return static_cast<T>(-3.38953139e38F);
    } else if constexpr (IsSameType<T, int8_t>::value) {
        return static_cast<T>(-128);
    } else if constexpr (IsSameType<T, int16_t>::value) {
        return static_cast<T>(-32768);
    } else if constexpr (IsSameType<T, int32_t>::value) {
        return static_cast<T>(-2147483647 - 1);
    } else {
        return static_cast<T>(0);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T AddValue(T lhs, T rhs)
{
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        return static_cast<T>(static_cast<float>(lhs) + static_cast<float>(rhs));
    } else {
        return static_cast<T>(lhs + rhs);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T MulValue(T lhs, T rhs)
{
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        return static_cast<T>(static_cast<float>(lhs) * static_cast<float>(rhs));
    } else {
        return static_cast<T>(lhs * rhs);
    }
}

template <typename T, typename CountT>
__simt_callee__ __aicore__ inline T MeanValue(T value, CountT count)
{
    if constexpr (IsSameType<T, uint8_t>::value || IsSameType<T, int8_t>::value ||
                  IsSameType<T, int16_t>::value) {
        // torch_scatter counts in the source dtype. Reproduce its wrapping
        // with unsigned arithmetic and clamp wrapped negative counts to one.
        constexpr bool unsignedCount = IsSameType<T, uint8_t>::value;
        uint32_t maxCount = static_cast<uint32_t>(HighestValue<T>());
        uint32_t mask = unsignedCount ? maxCount : (maxCount << 1U) | 1U;
        uint32_t wrapped = static_cast<uint32_t>(count) & mask;
        count = static_cast<CountT>(wrapped <= maxCount ? wrapped : 1U);
    }
    if (count < static_cast<CountT>(1)) {
        count = static_cast<CountT>(1);
    }
    if constexpr (IsSameType<T, float>::value || IsSameType<T, half>::value ||
                  IsSameType<T, bfloat16_t>::value) {
        return static_cast<T>(static_cast<float>(value) / static_cast<float>(count));
    } else if constexpr (IsSameType<T, uint8_t>::value) {
        return static_cast<T>(static_cast<uint32_t>(value) / static_cast<uint32_t>(count));
    } else {
        int64_t wide = static_cast<int64_t>(value);
        int64_t quotient = wide / static_cast<int64_t>(count);
        int64_t remainder = wide % static_cast<int64_t>(count);
        if (remainder != 0 && wide < 0) {
            --quotient; // C++ truncates toward zero; torch_scatter requires floor.
        }
        return static_cast<T>(quotient);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T InitialValue(uint32_t reduce, bool emptyInput)
{
    if (emptyInput) {
        return ZeroValue<T>();
    }
    if (reduce == SCATTER_MUL) {
        return OneValue<T>();
    }
    if (reduce == SCATTER_MIN) {
        return HighestValue<T>();
    }
    if (reduce == SCATTER_MAX) {
        return LowestValue<T>();
    }
    return ZeroValue<T>();
}

template <typename T>
__simt_callee__ __aicore__ inline void PrepareLane(
    uint64_t outBase, uint64_t outputDim, uint64_t afterDim,
    uint64_t dimLength, uint32_t reduce, bool emptyInput, bool hasOut,
    __gm__ T* out, __gm__ int32_t* count, __gm__ int64_t* argOut)
{
    for (uint64_t n = 0; n < outputDim; ++n) {
        uint64_t offset = outBase + n * afterDim;
        if (!hasOut) {
            out[offset] = InitialValue<T>(reduce, emptyInput);
        }
        if (reduce == SCATTER_MEAN) {
            count[offset] = 0;
        }
        if (reduce == SCATTER_MIN || reduce == SCATTER_MAX) {
            argOut[offset] = static_cast<int64_t>(dimLength);
        }
    }
}

__simt_callee__ __aicore__ inline uint64_t LaneIndexOffset(
    uint64_t sourceOffset, uint64_t element, uint64_t indexDimLength,
    uint32_t indexMode)
{
    if (indexMode == SCATTER_INDEX_DIM_VECTOR) {
        return indexDimLength == 1 ? 0 : element;
    }
    return sourceOffset;
}

template <typename T>
__simt_callee__ __aicore__ inline void UpdateLaneValue(
    uint64_t outOffset, uint64_t element, uint32_t reduce, T value,
    __gm__ T* out, __gm__ int32_t* count, __gm__ int64_t* argOut)
{
    T current = out[outOffset];
    if (reduce == SCATTER_SUM || reduce == SCATTER_MEAN) {
        out[outOffset] = AddValue<T>(current, value);
        if (reduce == SCATTER_MEAN) {
            ++count[outOffset];
        }
        return;
    }
    if (reduce == SCATTER_MUL) {
        out[outOffset] = MulValue<T>(current, value);
        return;
    }
    bool replace = reduce == SCATTER_MIN ? value <= current : value >= current;
    if (replace) { // Task contract: the later writer wins ties.
        out[outOffset] = value;
        argOut[outOffset] = static_cast<int64_t>(element);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline void AccumulateLane(
    uint64_t before, uint64_t inner, uint64_t outBase,
    uint64_t dimLength, uint64_t afterDim, uint64_t outputDim,
    uint64_t indexDimLength, uint32_t indexMode, uint32_t reduce,
    __gm__ T* src, __gm__ int64_t* index, __gm__ T* out,
    __gm__ int32_t* count, __gm__ int64_t* argOut)
{
    for (uint64_t element = 0; element < dimLength; ++element) {
        uint64_t sourceOffset =
            (before * dimLength + element) * afterDim + inner;
        uint64_t indexOffset = LaneIndexOffset(
            sourceOffset, element, indexDimLength, indexMode);
        int64_t target = index[indexOffset];
        if (target < 0 || static_cast<uint64_t>(target) >= outputDim) {
            continue;
        }
        uint64_t outOffset =
            outBase + static_cast<uint64_t>(target) * afterDim;
        UpdateLaneValue<T>(outOffset, element, reduce, src[sourceOffset],
                           out, count, argOut);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline void FinalizeLane(
    uint64_t outBase, uint64_t outputDim, uint64_t afterDim,
    uint64_t dimLength, uint32_t reduce, bool hasOut,
    __gm__ T* out, __gm__ int32_t* count, __gm__ int64_t* argOut)
{
    for (uint64_t n = 0; n < outputDim; ++n) {
        uint64_t offset = outBase + n * afterDim;
        if (reduce == SCATTER_MEAN) {
            out[offset] = MeanValue<T, int32_t>(out[offset], count[offset]);
        } else if (!hasOut &&
                   (reduce == SCATTER_MIN || reduce == SCATTER_MAX) &&
                   argOut[offset] == static_cast<int64_t>(dimLength)) {
            out[offset] = ZeroValue<T>();
        }
    }
}

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void LaneOwnerCompute(
    uint64_t laneStart, uint64_t laneEnd, uint64_t dimLength,
    uint64_t afterDim, uint64_t outputDim, uint64_t srcNumel,
    uint64_t indexDimLength, uint32_t indexMode, uint32_t reduce, bool hasOut,
    __gm__ T* src, __gm__ int64_t* index, __gm__ T* out,
    __gm__ int32_t* count, __gm__ int64_t* argOut)
{
    for (uint64_t lane = laneStart + static_cast<uint64_t>(threadIdx.x);
         lane < laneEnd; lane += static_cast<uint64_t>(blockDim.x)) {
        uint64_t before = lane / afterDim;
        uint64_t inner = lane - before * afterDim;
        uint64_t outBase = before * outputDim * afterDim + inner;
        PrepareLane<T>(outBase, outputDim, afterDim, dimLength, reduce,
                       srcNumel == 0, hasOut, out, count, argOut);
        AccumulateLane<T>(before, inner, outBase, dimLength, afterDim,
                          outputDim, indexDimLength, indexMode, reduce,
                          src, index, out, count, argOut);
        FinalizeLane<T>(outBase, outputDim, afterDim, dimLength, reduce,
                        hasOut, out, count, argOut);
    }
}

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void AtomicSumCompute(
    uint64_t sourceStart, uint64_t sourceEnd, uint64_t dimLength,
    uint64_t afterDim, uint64_t outputDim, uint64_t indexDimLength,
    uint32_t indexMode, bool isMean, __gm__ T* src, __gm__ int64_t* index,
    __gm__ T* out, __gm__ T* count)
{
    for (uint64_t sourceOffset = sourceStart + static_cast<uint64_t>(threadIdx.x);
         sourceOffset < sourceEnd;
         sourceOffset += static_cast<uint64_t>(blockDim.x)) {
        uint64_t row = sourceOffset / afterDim;
        uint64_t inner = sourceOffset - row * afterDim;
        uint64_t before = row / dimLength;
        uint64_t e = row - before * dimLength;
        uint64_t indexOffset = sourceOffset;
        if (indexMode == SCATTER_INDEX_DIM_VECTOR) {
            indexOffset = indexDimLength == 1 ? 0 : e;
        }
        int64_t target = index[indexOffset];
        if (target < 0 || static_cast<uint64_t>(target) >= outputDim) {
            continue;
        }
        uint64_t outOffset = (before * outputDim + static_cast<uint64_t>(target)) *
                             afterDim + inner;
        asc_atomic_add(&out[outOffset], src[sourceOffset]);
        if (isMean) {
            asc_atomic_add(&count[outOffset], OneValue<T>());
        }
    }
}

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void MeanFinalizeCompute(
    uint64_t outputStart, uint64_t outputEnd, __gm__ T* out,
    __gm__ T* count)
{
    for (uint64_t offset = outputStart + static_cast<uint64_t>(threadIdx.x);
         offset < outputEnd; offset += static_cast<uint64_t>(blockDim.x)) {
        out[offset] = MeanValue<T, T>(out[offset], count[offset]);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void CompactCountCompute(
    uint64_t rowStart, uint64_t rowEnd, uint64_t dimLength,
    uint64_t outputDim, uint64_t indexDimLength, __gm__ int64_t* index,
    __gm__ int32_t* count, int64_t hotTarget)
{
    for (uint64_t row = rowStart + static_cast<uint64_t>(threadIdx.x);
         row < rowEnd; row += static_cast<uint64_t>(blockDim.x)) {
        uint64_t before = row / dimLength;
        uint64_t e = row - before * dimLength;
        uint64_t indexOffset = indexDimLength == 1 ? 0 : e;
        int64_t target = index[indexOffset];
        if (target < 0 || static_cast<uint64_t>(target) >= outputDim) {
            continue;
        }
        if (hotTarget >= 0 && target == hotTarget) {
            continue;
        }
        uint64_t group = before * outputDim + static_cast<uint64_t>(target);
        asc_atomic_add(&count[group], static_cast<int32_t>(1));
    }
}

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void
CompactMeanFinalizeCompute(uint64_t outputStart, uint64_t outputEnd,
                           uint64_t afterDim, __gm__ T* out,
                           __gm__ int32_t* count)
{
    for (uint64_t offset = outputStart + static_cast<uint64_t>(threadIdx.x);
         offset < outputEnd; offset += static_cast<uint64_t>(blockDim.x)) {
        uint64_t group = offset / afterDim;
        out[offset] = MeanValue<T, int32_t>(out[offset], count[group]);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void
MinMaxInitCompute(uint64_t outputStart, uint64_t outputEnd, uint32_t reduce,
                  __gm__ float* values, __gm__ int32_t* argOut)
{
    float initial = reduce == SCATTER_MIN ? 3.402823466e38F :
                                            -3.402823466e38F;
    for (uint64_t offset = outputStart + static_cast<uint64_t>(threadIdx.x);
         offset < outputEnd; offset += static_cast<uint64_t>(blockDim.x)) {
        values[offset] = initial;
        argOut[offset] = -1;
    }
}

#include "scatter_kernel_hot.h"

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void scatter_kernel(
    GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR count, GM_ADDR argOut,
    ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0) {
        return;
    }

    if constexpr (IsSameType<T, half>::value ||
                  IsSameType<T, bfloat16_t>::value ||
                  IsSameType<T, float>::value ||
                  IsSameType<T, int32_t>::value) {
        if (tiling.path == SCATTER_ATOMIC) {
            uint64_t start = tiling.srcNumel * block / blocks;
            uint64_t end = tiling.srcNumel * (block + 1) / blocks;
            asc_vf_call<AtomicSumCompute<T>>(
                dim3(THREAD_NUM), start, end, tiling.dimLength, tiling.afterDim,
                tiling.outputDim, tiling.indexDimLength, tiling.indexMode,
                tiling.reduce == SCATTER_MEAN,
                (__gm__ T*)src, (__gm__ int64_t*)index, (__gm__ T*)out,
                (__gm__ T*)count);
            return;
        }
    }

    uint64_t start = tiling.laneCount * block / blocks;
    uint64_t end = tiling.laneCount * (block + 1) / blocks;
    asc_vf_call<LaneOwnerCompute<T>>(
        dim3(THREAD_NUM), start, end, tiling.dimLength, tiling.afterDim,
        tiling.outputDim, tiling.srcNumel, tiling.indexDimLength,
        tiling.indexMode, tiling.reduce, tiling.hasOut != 0, (__gm__ T*)src,
        (__gm__ int64_t*)index, (__gm__ T*)out, (__gm__ int32_t*)count,
        (__gm__ int64_t*)argOut);
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void scatter_mean_finalize_kernel(
    GM_ADDR out, GM_ADDR count, ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0) {
        return;
    }
    uint64_t start = tiling.outNumel * block / blocks;
    uint64_t end = tiling.outNumel * (block + 1) / blocks;
    asc_vf_call<MeanFinalizeCompute<T>>(
        dim3(THREAD_NUM), start, end, (__gm__ T*)out, (__gm__ T*)count);
}

#include "scatter_kernel_atomic.h"

#include "scatter_kernel_minmax.h"

} // namespace ScatterNpu

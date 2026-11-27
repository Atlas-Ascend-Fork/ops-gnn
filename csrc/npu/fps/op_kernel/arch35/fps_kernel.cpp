/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../common/fps_kernel.h"
#include "../common/fps_vector_kernel.h"

#include "kernel_operator.h"
#include "simt_api/asc_fp16.h"

using namespace AscendC;

namespace {

constexpr int64_t FPS_SIMT_THREADS = 256;
constexpr int64_t FPS_CPU_FLOAT_VECTOR_WIDTH = 8;
constexpr int64_t FPS_CPU_SUM_ILP = 4;
constexpr int64_t FPS_CPU_SUM_LEVELS = 4;

template <typename T>
__simt_callee__ __aicore__ inline void InitFallbackDistance(
    __gm__ T* dist, int64_t index)
{
    dist[index] = static_cast<T>(__builtin_inff());
}

template <>
__simt_callee__ __aicore__ inline void InitFallbackDistance<half>(
    __gm__ half* dist, int64_t index)
{
    dist[index] = __float2half_rn(__builtin_inff());
}

__simt_callee__ __aicore__ inline float LoadFloatSquare(
    __gm__ const float* src, int64_t point, int64_t center,
    int64_t featureDim, int64_t feature)
{
    const float delta = src[point * featureDim + feature] -
                        src[center * featureDim + feature];
    // torch_cluster CPU materializes pow(2) before sum.  Volatile preserves
    // that multiplication rounding boundary instead of contracting to FMA.
    volatile float square = delta * delta;
    return square;
}

struct FloatDistanceInput {
    __gm__ const float* src;
    int64_t point;
    int64_t center;
    int64_t featureDim;
};

using FloatCascadeAccumulators =
    float[FPS_CPU_SUM_LEVELS][FPS_CPU_SUM_ILP][FPS_CPU_FLOAT_VECTOR_WIDTH];

__simt_callee__ __aicore__ inline int64_t FloatCascadeLevelPower(
    int64_t groupCount)
{
    int64_t ceilLog2 = 0;
    int64_t powerOfTwo = 1;
    while (powerOfTwo < groupCount) {
        powerOfTwo <<= 1;
        ++ceilLog2;
    }
    int64_t levelPower = ceilLog2 / FPS_CPU_SUM_LEVELS;
    if (levelPower < 4) {
        levelPower = 4;
    }
    return levelPower;
}

__simt_callee__ __aicore__ inline void AccumulateFloatGroup(
    const FloatDistanceInput& input, int64_t group,
    FloatCascadeAccumulators& accum)
{
    for (int64_t row = 0; row < FPS_CPU_SUM_ILP; ++row) {
        const int64_t base =
            (group * FPS_CPU_SUM_ILP + row) * FPS_CPU_FLOAT_VECTOR_WIDTH;
        for (int64_t lane = 0; lane < FPS_CPU_FLOAT_VECTOR_WIDTH; ++lane) {
            accum[0][row][lane] += LoadFloatSquare(
                input.src, input.point, input.center, input.featureDim,
                base + lane);
        }
    }
}

__simt_callee__ __aicore__ inline void CarryFloatCascade(
    FloatCascadeAccumulators& accum, int64_t group,
    int64_t levelPower, int64_t levelMask)
{
    for (int64_t level = 1; level < FPS_CPU_SUM_LEVELS; ++level) {
        for (int64_t row = 0; row < FPS_CPU_SUM_ILP; ++row) {
            for (int64_t lane = 0; lane < FPS_CPU_FLOAT_VECTOR_WIDTH;
                 ++lane) {
                accum[level][row][lane] += accum[level - 1][row][lane];
                accum[level - 1][row][lane] = 0.0f;
            }
        }
        const int64_t mask = levelMask << (level * levelPower);
        if ((group & mask) != 0) {
            break;
        }
    }
}

__simt_callee__ __aicore__ inline void FoldFloatCascadeLevels(
    FloatCascadeAccumulators& accum)
{
    for (int64_t level = 1; level < FPS_CPU_SUM_LEVELS; ++level) {
        for (int64_t row = 0; row < FPS_CPU_SUM_ILP; ++row) {
            for (int64_t lane = 0; lane < FPS_CPU_FLOAT_VECTOR_WIDTH;
                 ++lane) {
                accum[0][row][lane] += accum[level][row][lane];
            }
        }
    }
}

__simt_callee__ __aicore__ inline void AccumulateFloatRemainingChunks(
    const FloatDistanceInput& input, FloatCascadeAccumulators& accum,
    int64_t groupCount, int64_t vectorChunks)
{
    for (int64_t chunk = groupCount * FPS_CPU_SUM_ILP;
         chunk < vectorChunks; ++chunk) {
        const int64_t base = chunk * FPS_CPU_FLOAT_VECTOR_WIDTH;
        for (int64_t lane = 0; lane < FPS_CPU_FLOAT_VECTOR_WIDTH; ++lane) {
            accum[0][0][lane] += LoadFloatSquare(
                input.src, input.point, input.center, input.featureDim,
                base + lane);
        }
    }
    for (int64_t row = 1; row < FPS_CPU_SUM_ILP; ++row) {
        for (int64_t lane = 0; lane < FPS_CPU_FLOAT_VECTOR_WIDTH; ++lane) {
            accum[0][0][lane] += accum[0][row][lane];
        }
    }
}

__simt_callee__ __aicore__ inline float FinishFloatCascadeDistance(
    const FloatDistanceInput& input, const FloatCascadeAccumulators& accum,
    int64_t vectorChunks)
{
    float value = 0.0f;
    for (int64_t feature = vectorChunks * FPS_CPU_FLOAT_VECTOR_WIDTH;
         feature < input.featureDim; ++feature) {
        value += LoadFloatSquare(
            input.src, input.point, input.center, input.featureDim, feature);
    }
    for (int64_t lane = 0; lane < FPS_CPU_FLOAT_VECTOR_WIDTH; ++lane) {
        value += accum[0][0][lane];
    }
    return value;
}

__simt_callee__ __aicore__ inline float CpuFloatCascadeDistance(
    __gm__ const float* src, int64_t point, int64_t center,
    int64_t featureDim)
{
    // F<8 uses PyTorch's scalar row_sum with four interleaved partials.
    // The large-N launcher reaches this fallback even for these dimensions.
    if (featureDim < FPS_CPU_FLOAT_VECTOR_WIDTH) {
        float partial[FPS_CPU_SUM_ILP] = {};
        const int64_t grouped = (featureDim / FPS_CPU_SUM_ILP) * FPS_CPU_SUM_ILP;
        for (int64_t feature = 0; feature < grouped; ++feature) {
            partial[feature % FPS_CPU_SUM_ILP] += LoadFloatSquare(
                src, point, center, featureDim, feature);
        }
        for (int64_t feature = grouped; feature < featureDim; ++feature) {
            partial[0] += LoadFloatSquare(src, point, center, featureDim, feature);
        }
        for (int64_t row = 1; row < FPS_CPU_SUM_ILP; ++row) {
            partial[0] += partial[row];
        }
        return partial[0];
    }
    // PyTorch 2.9's CPU sum kernel is dispatched through its AVX2 path on the
    // validation host: 8 float lanes, four interleaved accumulators, and four
    // cascade levels. Reproduce SumKernel.cpp's row_sum/multi_row_sum order.
    FloatCascadeAccumulators accum = {};
    const int64_t vectorChunks = featureDim / FPS_CPU_FLOAT_VECTOR_WIDTH;
    const int64_t groupCount = vectorChunks / FPS_CPU_SUM_ILP;
    const int64_t levelPower = FloatCascadeLevelPower(groupCount);
    const int64_t levelStep = static_cast<int64_t>(1) << levelPower;
    const int64_t levelMask = levelStep - 1;
    const FloatDistanceInput input{src, point, center, featureDim};

    int64_t group = 0;
    while (group + levelStep <= groupCount) {
        for (int64_t step = 0; step < levelStep; ++step, ++group) {
            AccumulateFloatGroup(input, group, accum);
        }
        CarryFloatCascade(accum, group, levelPower, levelMask);
    }
    while (group < groupCount) {
        AccumulateFloatGroup(input, group, accum);
        ++group;
    }
    FoldFloatCascadeLevels(accum);
    AccumulateFloatRemainingChunks(input, accum, groupCount, vectorChunks);
    return FinishFloatCascadeDistance(input, accum, vectorChunks);
}

template <typename T>
__simt_callee__ __aicore__ inline float UpdateFallbackDistance(
    __gm__ const T* src, __gm__ T* dist, int64_t point, int64_t center,
    int64_t featureDim)
{
    float value = 0.0f;
    for (int64_t feature = 0; feature < featureDim; ++feature) {
        const float delta = src[point * featureDim + feature] -
                            src[center * featureDim + feature];
        value += delta * delta;
    }
    const float previous = dist[point];
    const float nearest = value < previous ? value : previous;
    dist[point] = static_cast<T>(nearest);
    return nearest;
}

template <>
__simt_callee__ __aicore__ inline float UpdateFallbackDistance<float>(
    __gm__ const float* src, __gm__ float* dist, int64_t point,
    int64_t center, int64_t featureDim)
{
    const float value = CpuFloatCascadeDistance(
        src, point, center, featureDim);
    const float previous = dist[point];
    const float nearest = value < previous ? value : previous;
    dist[point] = nearest;
    return nearest;
}

template <>
__simt_callee__ __aicore__ inline float UpdateFallbackDistance<half>(
    __gm__ const half* src, __gm__ half* dist, int64_t point,
    int64_t center, int64_t featureDim)
{
    float value = 0.0f;
    for (int64_t feature = 0; feature < featureDim; ++feature) {
        // torch_cluster CPU evaluates subtraction and pow in fp16, then its
        // reduction accumulates the fp16 squares in fp32 and casts the final
        // sum back to fp16.  Keep that exact operation boundary here.
        const float pointValue =
            __half2float(src[point * featureDim + feature]);
        const float centerValue =
            __half2float(src[center * featureDim + feature]);
        const half delta = __float2half_rn(pointValue - centerValue);
        const float deltaFloat = __half2float(delta);
        const half square =
            __float2half_rn(deltaFloat * deltaFloat);
        value += __half2float(square);
    }
    const half current = __float2half_rn(value);
    const half previous = dist[point];
    const half nearest = __half2float(current) < __half2float(previous)
        ? current : previous;
    dist[point] = nearest;
    return __half2float(nearest);
}

__simt_callee__ __aicore__ inline int64_t SelectFallbackBlockIndex(
    __gm__ const float* bestDist, __gm__ const int64_t* bestIndex)
{
    float blockBest = bestDist[0];
    int64_t blockBestIndex = bestIndex[0];
    for (int64_t candidate = 1; candidate < FPS_SIMT_THREADS; ++candidate) {
        if (bestDist[candidate] > blockBest ||
            (bestDist[candidate] == blockBest &&
             bestIndex[candidate] < blockBestIndex)) {
            blockBest = bestDist[candidate];
            blockBestIndex = bestIndex[candidate];
        }
    }
    return blockBestIndex;
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(FPS_SIMT_THREADS)
void FpsFallbackSimt(
    __gm__ const T* src, __gm__ const int64_t* ptr,
    __gm__ const int64_t* outPtr, __gm__ const int64_t* start,
    __gm__ T* dist, __gm__ float* bestDist,
    __gm__ int64_t* bestIndex, __gm__ int64_t* out,
    int64_t numPoints, int64_t featureDim, int64_t batchSize,
    int64_t singleOutSize, int64_t singleStart)
{
    const int64_t lane = static_cast<int64_t>(Simt::GetThreadIdx());
    const bool batched = ptr != nullptr;
    for (int64_t batch = 0; batch < batchSize; ++batch) {
        const int64_t srcStart = batched ? ptr[batch] : 0;
        const int64_t srcEnd = batched ? ptr[batch + 1] : numPoints;
        const int64_t outStart = batched ? outPtr[batch] : 0;
        const int64_t outEnd = batched ? outPtr[batch + 1] : singleOutSize;
        const int64_t startOffset = batched ? start[batch] : singleStart;

        for (int64_t point = srcStart + lane; point < srcEnd;
             point += FPS_SIMT_THREADS) {
            InitFallbackDistance<T>(dist, point);
        }
        if (lane == 0 && outStart < outEnd) {
            out[outStart] = srcStart + startOffset;
        }
        asc_syncthreads();

        for (int64_t sample = outStart + 1; sample < outEnd; ++sample) {
            const int64_t center = out[sample - 1];
            float laneBest = -1.0f;
            int64_t laneBestIndex = srcStart + lane;
            for (int64_t point = srcStart + lane; point < srcEnd;
                 point += FPS_SIMT_THREADS) {
                const float nearest = UpdateFallbackDistance<T>(
                    src, dist, point, center, featureDim);
                if (nearest > laneBest) {
                    laneBest = nearest;
                    laneBestIndex = point;
                }
            }
            bestDist[lane] = laneBest;
            bestIndex[lane] = laneBestIndex;
            asc_syncthreads();

            if (lane == 0) {
                out[sample] = SelectFallbackBlockIndex(bestDist, bestIndex);
            }
            asc_syncthreads();
        }
    }
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void FpsFallbackKernel(
    __gm__ const T* src, __gm__ const int64_t* ptr,
    __gm__ const int64_t* outPtr, __gm__ const int64_t* start,
    __gm__ T* dist, __gm__ float* bestDist,
    __gm__ int64_t* bestIndex, __gm__ int64_t* out,
    int64_t numPoints, int64_t featureDim, int64_t batchSize,
    int64_t singleOutSize, int64_t singleStart)
{
    Simt::VF_CALL<FpsFallbackSimt<T>>(
        Simt::Dim3{FPS_SIMT_THREADS}, src, ptr, outPtr, start, dist,
        bestDist, bestIndex, out, numPoints, featureDim, batchSize,
        singleOutSize, singleStart);
}


template <typename T>
__attribute__((aiv)) __global__ __aicore__ void FpsKernel(
    __gm__ const T* src, __gm__ const T* srcSoa,
    __gm__ const uint32_t* ptr, __gm__ const uint32_t* outPtr,
    __gm__ const uint32_t* start, __gm__ T* dist,
    __gm__ uint32_t* out, uint64_t numPoints, uint64_t featureDim,
    uint64_t batchSize, uint64_t singleOutSize, uint64_t singleStart,
    uint32_t tilePoints, uint32_t tileBytes, uint32_t promotedTileBytes,
    uint32_t cascadeTileBytes, uint32_t centerBytes)
{
    TPipe pipe;
    opsgnn::fps::FpsVectorKernel<T> op;
    op.Init(src, srcSoa, ptr, outPtr, start, dist, out, numPoints,
            featureDim, batchSize, singleOutSize, singleStart, tilePoints,
            tileBytes, promotedTileBytes, cascadeTileBytes, centerBytes, &pipe);
    op.Process();
}

}  // namespace

template <typename T, int DType>
void Fps(const T* src, const FpsLaunchArgs<T>& args, aclrtStream stream)
{
    // Unaligned int64 output ranges from neighboring batches can share a
    // 32-byte GM transaction. Process batches on one AIV core so those scalar
    // writes cannot race; the official performance cases are single-batch.
    const uint32_t blockNum = 1;
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(args.ptr);
    const uint32_t* outPtr = reinterpret_cast<const uint32_t*>(args.out_ptr);
    const uint32_t* start = reinterpret_cast<const uint32_t*>(args.start);
    uint32_t* out = reinterpret_cast<uint32_t*>(args.out);
    if constexpr (DType == 0) {
        // torch.addcmul on the CPU golden path uses fused fp32 accumulation.
        // The SIMT expression below compiles to the same arithmetic contract;
        // retain it for feature dimensions above the performance matrix (the
        // measured cases stop at F=64), where tiny accumulated roundoff can
        // otherwise change a late discrete FPS index.
        if (FpsUsesFallback(DType, args.max_batch_points, args.feature_dim)) {
            FpsFallbackKernel<float><<<1, nullptr, stream>>>(
                reinterpret_cast<const float*>(src), args.ptr, args.out_ptr,
                args.start, reinterpret_cast<float*>(args.dist),
                args.best_dist, args.best_index, args.out, args.num_points,
                args.feature_dim, args.batch_size, args.single_out_size,
                args.single_start);
            return;
        }
        FpsKernel<float><<<blockNum, nullptr, stream>>>(
            reinterpret_cast<const float*>(src),
            reinterpret_cast<const float*>(args.src_soa), ptr, outPtr, start,
            reinterpret_cast<float*>(args.dist), out, args.num_points,
            args.feature_dim, args.batch_size, args.single_out_size,
            args.single_start, args.tile_points, args.tile_bytes,
            args.promoted_tile_bytes, args.cascade_tile_bytes,
            args.center_bytes);
    } else {
        if (FpsUsesFallback(DType, args.max_batch_points, args.feature_dim)) {
            FpsFallbackKernel<half><<<1, nullptr, stream>>>(
                reinterpret_cast<const half*>(src), args.ptr, args.out_ptr,
                args.start, reinterpret_cast<half*>(args.dist),
                args.best_dist, args.best_index, args.out, args.num_points,
                args.feature_dim, args.batch_size, args.single_out_size,
                args.single_start);
            return;
        }
        FpsKernel<half><<<blockNum, nullptr, stream>>>(
            reinterpret_cast<const half*>(src),
            reinterpret_cast<const half*>(args.src_soa), ptr, outPtr, start,
            reinterpret_cast<half*>(args.dist), out, args.num_points,
            args.feature_dim, args.batch_size, args.single_out_size,
            args.single_start, args.tile_points, args.tile_bytes,
            args.promoted_tile_bytes, args.cascade_tile_bytes,
            args.center_bytes);
    }
}

template void Fps<float, 0>(const float*, const FpsLaunchArgs<float>&,
                            aclrtStream);
template void Fps<uint16_t, 1>(const uint16_t*,
                               const FpsLaunchArgs<uint16_t>&, aclrtStream);

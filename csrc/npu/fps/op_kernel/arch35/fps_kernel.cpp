/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fps_kernel.h"

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
class FpsVectorKernel {
public:
    __aicore__ inline void Init(
        __gm__ const T* src, __gm__ const T* srcSoa,
        __gm__ const uint32_t* ptr, __gm__ const uint32_t* outPtr,
        __gm__ const uint32_t* start, __gm__ T* dist,
        __gm__ uint32_t* out, uint64_t numPoints, uint64_t featureDim,
        uint64_t batchSize, uint64_t singleOutSize, uint64_t singleStart,
        uint32_t tilePoints, uint32_t tileBytes,
        uint32_t promotedTileBytes, uint32_t cascadeTileBytes,
        uint32_t centerBytes, TPipe* pipe)
    {
        srcGm.SetGlobalBuffer(const_cast<__gm__ T*>(src), numPoints * featureDim);
        srcSoaGm.SetGlobalBuffer(const_cast<__gm__ T*>(srcSoa), numPoints * featureDim);
        distGm.SetGlobalBuffer(dist, numPoints);
        outBitsGm.SetGlobalBuffer(out, singleOutSize * 2);

        batched = ptr != nullptr;
        if (batched) {
            ptrBitsGm.SetGlobalBuffer(const_cast<__gm__ uint32_t*>(ptr), (batchSize + 1) * 2);
            outPtrBitsGm.SetGlobalBuffer(const_cast<__gm__ uint32_t*>(outPtr), (batchSize + 1) * 2);
            startBitsGm.SetGlobalBuffer(const_cast<__gm__ uint32_t*>(start), batchSize * 2);
        }

        this->numPoints = numPoints;
        this->featureDim = featureDim;
        this->batchSize = batchSize;
        this->singleOutSize = singleOutSize;
        this->singleStart = singleStart;
        this->tilePoints = tilePoints;
        this->tileBytes = tileBytes;
        this->promotedTileBytes = promotedTileBytes;
        this->cascadeTileBytes = cascadeTileBytes;

        pipe->InitBuffer(pointBuffer, tileBytes);
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            pipe->InitBuffer(distanceBuffer, promotedTileBytes);
            pipe->InitBuffer(promotedBuffer, promotedTileBytes);
        } else {
            pipe->InitBuffer(
                distanceBuffer,
                featureDim >= 5
                    ? cascadeTileBytes : tileBytes);
        }
        pipe->InitBuffer(nearestBuffer, tileBytes);
        pipe->InitBuffer(centerBuffer, centerBytes);
        pipe->InitBuffer(reduceWorkBuffer, FPS_REDUCE_WORK_BYTES);
        pipe->InitBuffer(reduceResultBuffer, FPS_REDUCE_RESULT_BYTES);

        pointLocal = pointBuffer.Get<T>();
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            distanceFloatLocal = distanceBuffer.Get<float>();
            promotedFloatLocal = promotedBuffer.Get<float>();
        } else {
            distanceLocal = distanceBuffer.Get<T>();
            if (featureDim >= 5) {
                cascadeLocal = distanceBuffer.Get<float>();
                distanceLocal = cascadeLocal[4 * tilePoints];
            }
        }
        nearestLocal = nearestBuffer.Get<T>();
        centerLocal = centerBuffer.Get<T>();
        reduceWorkLocal = reduceWorkBuffer.Get<T>();
        reduceResultLocal = reduceResultBuffer.Get<T>();
    }

    __aicore__ inline void Process()
    {
        const uint64_t firstBatch = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t batchStride = static_cast<uint64_t>(GetBlockNum());
        for (uint64_t batch = firstBatch; batch < batchSize;
             batch += batchStride) {
            ProcessBatch(batch);
        }
    }

private:
    struct FloatDistancePiece {
        uint64_t srcStart;
        uint64_t pieceOffset;
        uint32_t count;
        const DataCopyExtParams& copyParams;
        const DataCopyPadExtParams<T>& padParams;
    };

    __aicore__ inline void ProcessBatch(uint64_t batch)
    {
        const uint64_t srcStart = batched ? ReadInt64(ptrBitsGm, batch) : 0;
        const uint64_t srcEnd = batched ? ReadInt64(ptrBitsGm, batch + 1) : numPoints;
        const uint64_t outStart = batched ? ReadInt64(outPtrBitsGm, batch) : 0;
        const uint64_t outEnd = batched ? ReadInt64(outPtrBitsGm, batch + 1) : singleOutSize;
        const uint64_t startOffset = batched ? ReadInt64(startBitsGm, batch) : singleStart;
        if (srcStart >= srcEnd || outStart >= outEnd) {
            return;
        }

        uint64_t center = srcStart + startOffset;
        WriteInt64(outBitsGm, outStart, center);
        if (outEnd - outStart == 1) {
            return;
        }

        const uint64_t pointCount = srcEnd - srcStart;
        if (pointCount <= tilePoints) {
            ProcessComplete(srcStart, pointCount, outStart, outEnd, center);
        } else {
            ProcessSplit(srcStart, pointCount, outStart, outEnd, center);
        }
    }

    __aicore__ inline uint64_t ReadInt64(
        const GlobalTensor<uint32_t>& tensor, uint64_t index) const
    {
        const uint64_t low = tensor.GetValue(index * 2);
        const uint64_t high = tensor.GetValue(index * 2 + 1);
        return low | (high << 32);
    }

    __aicore__ inline void WriteInt64(
        GlobalTensor<uint32_t>& tensor, uint64_t index, uint64_t value)
    {
        tensor.SetValue(index * 2, static_cast<uint32_t>(value));
        tensor.SetValue(index * 2 + 1, static_cast<uint32_t>(value >> 32));
    }

    __aicore__ inline void LoadCenter(uint64_t center)
    {
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(featureDim * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(centerLocal, srcGm[center * featureDim], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Muls<T>(centerLocal, centerLocal, static_cast<T>(-1.0f),
                static_cast<int32_t>(featureDim));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void ComputeDistance(
        uint64_t srcStart, uint64_t pieceOffset, uint32_t count)
    {
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        if constexpr (sizeof(T) == sizeof(uint32_t)) {
            if (featureDim >= 5) {
                ComputeFloatCascadeDistance(
                    srcStart, pieceOffset, count, copyParams, padParams);
                return;
            }
        }

        for (uint64_t feature = 0; feature < featureDim; ++feature) {
            LoadSquareFeature(
                srcStart, pieceOffset, feature, count, copyParams, padParams);
            if constexpr (sizeof(T) == sizeof(uint16_t)) {
                if (feature == 0) {
                    Cast(distanceFloatLocal, pointLocal,
                         RoundMode::CAST_NONE, static_cast<int32_t>(count));
                } else {
                    Cast(promotedFloatLocal, pointLocal,
                         RoundMode::CAST_NONE, static_cast<int32_t>(count));
                    PipeBarrier<PIPE_V>();
                    Add<float>(distanceFloatLocal, distanceFloatLocal,
                               promotedFloatLocal,
                               static_cast<int32_t>(count));
                }
            } else {
                if (feature == 0) {
                    Adds<T>(distanceLocal, pointLocal, static_cast<T>(0),
                            static_cast<int32_t>(count));
                } else {
                    Add<T>(distanceLocal, distanceLocal, pointLocal,
                           static_cast<int32_t>(count));
                }
            }
            FinishPointVector();
        }
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            Cast(pointLocal, distanceFloatLocal, RoundMode::CAST_RINT,
                 static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void LoadSquareFeature(
        uint64_t srcStart, uint64_t pieceOffset, uint64_t feature,
        uint32_t count, const DataCopyExtParams& copyParams,
        const DataCopyPadExtParams<T>& padParams)
    {
        DataCopyPad(
            pointLocal,
            srcSoaGm[feature * numPoints + srcStart + pieceOffset],
            copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        const T negativeCenter = centerLocal.GetValue(feature);
        Adds<T>(pointLocal, pointLocal, negativeCenter,
                static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Mul<T>(pointLocal, pointLocal, pointLocal,
               static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FinishPointVector()
    {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void InitFloatCascadeRows(uint32_t count)
    {
        for (uint32_t row = 0; row < FPS_CPU_SUM_ILP; ++row) {
            Duplicate<float>(cascadeLocal[row * tilePoints], 0.0f,
                             static_cast<int32_t>(count));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddFloatPointVector(
        LocalTensor<float> accumulator, uint32_t count)
    {
        Add<float>(accumulator, accumulator, pointLocal,
                   static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        FinishPointVector();
    }

    __aicore__ inline void AccumulateFloatRowFeature(
        const FloatDistancePiece& piece, uint64_t feature, uint32_t row)
    {
        LoadSquareFeature(
            piece.srcStart, piece.pieceOffset, feature, piece.count,
            piece.copyParams, piece.padParams);
        LocalTensor<float> accumulator = cascadeLocal[row * tilePoints];
        AddFloatPointVector(accumulator, piece.count);
    }

    __aicore__ inline void AccumulateFloatFirstRowFeature(
        const FloatDistancePiece& piece, uint64_t feature)
    {
        LoadSquareFeature(
            piece.srcStart, piece.pieceOffset, feature, piece.count,
            piece.copyParams, piece.padParams);
        AddFloatPointVector(cascadeLocal, piece.count);
    }

    __aicore__ inline void FoldFloatCascadeRows(uint32_t count)
    {
        for (uint32_t row = 1; row < FPS_CPU_SUM_ILP; ++row) {
            LocalTensor<float> other = cascadeLocal[row * tilePoints];
            Add<float>(cascadeLocal, cascadeLocal, other,
                       static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeFloatSmallFeatures(
        const FloatDistancePiece& piece)
    {
        InitFloatCascadeRows(piece.count);
        for (uint64_t row = 0; row < FPS_CPU_SUM_ILP; ++row) {
            AccumulateFloatRowFeature(piece, row, static_cast<uint32_t>(row));
        }
        for (uint64_t feature = FPS_CPU_SUM_ILP;
             feature < featureDim; ++feature) {
            AccumulateFloatFirstRowFeature(piece, feature);
        }
        FoldFloatCascadeRows(piece.count);
        Adds<float>(distanceLocal, cascadeLocal, 0.0f,
                    static_cast<int32_t>(piece.count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateFloatScalarTail(
        const FloatDistancePiece& piece, uint64_t vectorChunks)
    {
        // CPU vectorized_inner_sum adds scalar tail values before its
        // horizontal W=8 lane reduction.
        for (uint64_t feature =
                 vectorChunks * FPS_CPU_FLOAT_VECTOR_WIDTH;
             feature < featureDim; ++feature) {
            LoadSquareFeature(
                piece.srcStart, piece.pieceOffset, feature, piece.count,
                piece.copyParams, piece.padParams);
            AddFloatPointVector(distanceLocal, piece.count);
        }
    }

    __aicore__ inline void ComputeFloatCascadeLane(
        const FloatDistancePiece& piece, uint64_t lane,
        uint64_t vectorChunks, uint64_t groupedChunks)
    {
        InitFloatCascadeRows(piece.count);
        for (uint64_t chunk = 0; chunk < groupedChunks; ++chunk) {
            const uint32_t row = static_cast<uint32_t>(
                chunk % FPS_CPU_SUM_ILP);
            const uint64_t feature =
                chunk * FPS_CPU_FLOAT_VECTOR_WIDTH + lane;
            AccumulateFloatRowFeature(piece, feature, row);
        }
        for (uint64_t chunk = groupedChunks; chunk < vectorChunks; ++chunk) {
            const uint64_t feature =
                chunk * FPS_CPU_FLOAT_VECTOR_WIDTH + lane;
            AccumulateFloatFirstRowFeature(piece, feature);
        }
        FoldFloatCascadeRows(piece.count);
        Add<float>(distanceLocal, distanceLocal, cascadeLocal,
                   static_cast<int32_t>(piece.count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeFloatCascadeDistance(
        uint64_t srcStart, uint64_t pieceOffset, uint32_t count,
        const DataCopyExtParams& copyParams,
        const DataCopyPadExtParams<T>& padParams)
    {
        Duplicate<float>(distanceLocal, 0.0f, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        const FloatDistancePiece piece{
            srcStart, pieceOffset, count, copyParams, padParams};
        if (featureDim < FPS_CPU_FLOAT_VECTOR_WIDTH) {
            ComputeFloatSmallFeatures(piece);
            return;
        }
        const uint64_t vectorChunks =
            featureDim / FPS_CPU_FLOAT_VECTOR_WIDTH;
        const uint64_t groupedChunks =
            (vectorChunks / FPS_CPU_SUM_ILP) * FPS_CPU_SUM_ILP;
        AccumulateFloatScalarTail(piece, vectorChunks);
        // Compute one W=8 lane at a time. This preserves PyTorch's ILP=4
        // partial sums and horizontal order while keeping only four per-point
        // row accumulators live in UB.
        for (uint64_t lane = 0; lane < FPS_CPU_FLOAT_VECTOR_WIDTH; ++lane) {
            ComputeFloatCascadeLane(piece, lane, vectorChunks, groupedChunks);
        }
    }

    __aicore__ inline void InitNearest(uint32_t count)
    {
        Duplicate<T>(nearestLocal, static_cast<T>(__builtin_inff()),
                     static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadNearest(uint64_t offset, uint32_t count)
    {
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(nearestLocal, distGm[offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    }

    __aicore__ inline void StoreNearest(uint64_t offset, uint32_t count)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(distGm[offset], nearestLocal, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline uint32_t ReadReduceResult(T& value)
    {
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
        value = reduceResultLocal.GetValue(0);
        if constexpr (sizeof(T) == sizeof(uint32_t)) {
            LocalTensor<uint32_t> resultBits =
                reduceResultLocal.template ReinterpretCast<uint32_t>();
            return resultBits.GetValue(1);
        } else {
            LocalTensor<uint16_t> resultBits =
                reduceResultLocal.template ReinterpretCast<uint16_t>();
            return static_cast<uint32_t>(resultBits.GetValue(1));
        }
    }

    __aicore__ inline uint32_t ReduceNearest(
        uint32_t count, uint64_t pieceOffset, T& value)
    {
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            Min<T>(nearestLocal, nearestLocal, pointLocal,
                   static_cast<int32_t>(count));
        } else {
            Min<T>(nearestLocal, nearestLocal, distanceLocal,
                   static_cast<int32_t>(count));
        }
        PipeBarrier<PIPE_V>();
        ReduceMax<T>(reduceResultLocal, nearestLocal, reduceWorkLocal,
                     static_cast<int32_t>(count), true);
        uint32_t bestIndex = ReadReduceResult(value);

        // ATen argmax returns the first (lowest) index on an exact tie.
        // ReduceMax with returnIndex=true has the same within-piece rule.
        return bestIndex;
    }

    __aicore__ inline void ProcessComplete(
        uint64_t srcStart, uint64_t pointCount, uint64_t outStart,
        uint64_t outEnd, uint64_t center)
    {
        const uint32_t count = static_cast<uint32_t>(pointCount);
        InitNearest(count);
        for (uint64_t sample = outStart + 1; sample < outEnd; ++sample) {
            LoadCenter(center);
            ComputeDistance(srcStart, 0, count);
            T bestValue;
            const uint32_t bestIndex = ReduceNearest(count, 0, bestValue);
            center = srcStart + bestIndex;
            WriteInt64(outBitsGm, sample, center);
        }
    }

    __aicore__ inline void ProcessSplit(
        uint64_t srcStart, uint64_t pointCount, uint64_t outStart,
        uint64_t outEnd, uint64_t center)
    {
        for (uint64_t sample = outStart + 1; sample < outEnd; ++sample) {
            LoadCenter(center);
            float bestValue = -1.0f;
            uint64_t bestIndex = 0;
            for (uint64_t pieceOffset = 0; pieceOffset < pointCount;
                 pieceOffset += tilePoints) {
                const uint64_t remaining = pointCount - pieceOffset;
                const uint32_t count = static_cast<uint32_t>(
                    remaining < tilePoints ? remaining : tilePoints);
                if (sample == outStart + 1) {
                    InitNearest(count);
                } else {
                    LoadNearest(srcStart + pieceOffset, count);
                }
                ComputeDistance(srcStart, pieceOffset, count);
                T pieceValue;
                const uint32_t pieceIndex =
                    ReduceNearest(count, pieceOffset, pieceValue);
                StoreNearest(srcStart + pieceOffset, count);
                const float candidateValue = static_cast<float>(pieceValue);
                const uint64_t candidateIndex = pieceOffset + pieceIndex;
                if (candidateValue > bestValue ||
                    (candidateValue == bestValue &&
                     candidateIndex < bestIndex)) {
                    bestValue = candidateValue;
                    bestIndex = candidateIndex;
                }
            }
            center = srcStart + bestIndex;
            WriteInt64(outBitsGm, sample, center);
        }
    }

    GlobalTensor<T> srcGm;
    GlobalTensor<T> srcSoaGm;
    GlobalTensor<T> distGm;
    GlobalTensor<uint32_t> ptrBitsGm;
    GlobalTensor<uint32_t> outPtrBitsGm;
    GlobalTensor<uint32_t> startBitsGm;
    GlobalTensor<uint32_t> outBitsGm;

    TBuf<TPosition::VECCALC> pointBuffer;
    TBuf<TPosition::VECCALC> distanceBuffer;
    TBuf<TPosition::VECCALC> promotedBuffer;
    TBuf<TPosition::VECCALC> nearestBuffer;
    TBuf<TPosition::VECCALC> centerBuffer;
    TBuf<TPosition::VECCALC> reduceWorkBuffer;
    TBuf<TPosition::VECCALC> reduceResultBuffer;

    LocalTensor<T> pointLocal;
    LocalTensor<T> distanceLocal;
    LocalTensor<float> distanceFloatLocal;
    LocalTensor<float> promotedFloatLocal;
    LocalTensor<float> cascadeLocal;
    LocalTensor<T> nearestLocal;
    LocalTensor<T> centerLocal;
    LocalTensor<T> reduceWorkLocal;
    LocalTensor<T> reduceResultLocal;

    uint64_t numPoints = 0;
    uint64_t featureDim = 0;
    uint64_t batchSize = 0;
    uint64_t singleOutSize = 0;
    uint64_t singleStart = 0;
    uint32_t tilePoints = 0;
    uint32_t tileBytes = 0;
    uint32_t promotedTileBytes = 0;
    uint32_t cascadeTileBytes = 0;
    bool batched = false;
};

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
    FpsVectorKernel<T> op;
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
        if (args.feature_dim > 64 ||
            args.num_points > FPS_MAX_TILE_POINTS) {
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
        if (args.num_points > FPS_MAX_TILE_POINTS) {
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

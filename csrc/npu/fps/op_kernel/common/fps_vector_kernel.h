/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_FPS_VECTOR_KERNEL_H
#define OPS_GNN_FPS_VECTOR_KERNEL_H

#include "fps_kernel.h"
#include "kernel_operator.h"

namespace opsgnn {
namespace fps {

using namespace AscendC;

constexpr int64_t FPS_CPU_FLOAT_VECTOR_WIDTH = 8;
constexpr int64_t FPS_CPU_SUM_ILP = 4;
constexpr int64_t FPS_CPU_SUM_LEVELS = 4;

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

}  // namespace fps
}  // namespace opsgnn

#endif  // OPS_GNN_FPS_VECTOR_KERNEL_H

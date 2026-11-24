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

using namespace AscendC;

namespace {

constexpr int64_t FPS_CPU_FLOAT_VECTOR_WIDTH = 8;
constexpr int64_t FPS_CPU_SUM_ILP = 4;
constexpr int64_t FPS_CPU_SUM_LEVELS = 4;
constexpr uint32_t FALLBACK_FEAT_CHUNK = 1024U;
constexpr uint32_t FALLBACK_PT_TILE = 256U;
constexpr uint32_t FALLBACK_FAST_PT = 128U;
constexpr uint32_t FALLBACK_FAST_BUF_ELEMS = 8192U;

__aicore__ inline uint32_t Align32BytesElems(uint32_t count, uint32_t elemBytes)
{
    if (elemBytes == 0U || elemBytes > 32U) {
        return count;
    }
    const uint32_t elemsPer32 = 32U / elemBytes;
    return ((count + elemsPer32 - 1U) / elemsPer32) * elemsPer32;
}



__aicore__ inline float LoadUbSquare(
    const LocalTensor<float>& ptLoc, const LocalTensor<float>& ctrLoc, int64_t feature)
{
    const float delta = ptLoc.GetValue(static_cast<uint32_t>(feature)) -
                        ctrLoc.GetValue(static_cast<uint32_t>(feature));
    volatile float square = delta * delta;
    return square;
}

__aicore__ inline int64_t UbCascadeLevelPower(int64_t groupCount)
{
    int64_t ceilLog2 = 0;
    int64_t powerOfTwo = 1;
    while (powerOfTwo < groupCount) {
        powerOfTwo <<= 1;
        ++ceilLog2;
    }
    int64_t levelPower = ceilLog2 / FPS_CPU_SUM_LEVELS;
    return (levelPower < 4) ? 4 : levelPower;
}

__aicore__ inline float ComputeUbSmallFloat(
    const LocalTensor<float>& ptLoc, const LocalTensor<float>& ctrLoc, int64_t featureDim)
{
    float partial[FPS_CPU_SUM_ILP] = {};
    const int64_t grouped = (featureDim / FPS_CPU_SUM_ILP) * FPS_CPU_SUM_ILP;
    for (int64_t f = 0; f < grouped; ++f) {
        partial[f % FPS_CPU_SUM_ILP] += LoadUbSquare(ptLoc, ctrLoc, f);
    }
    for (int64_t f = grouped; f < featureDim; ++f) {
        partial[0] += LoadUbSquare(ptLoc, ctrLoc, f);
    }
    for (int64_t r = 1; r < FPS_CPU_SUM_ILP; ++r) {
        partial[0] += partial[r];
    }
    return partial[0];
}

__aicore__ inline void CarryUbCascade(
    float accum[FPS_CPU_SUM_LEVELS][FPS_CPU_SUM_ILP][FPS_CPU_FLOAT_VECTOR_WIDTH], int64_t group,
    int64_t levelPower, int64_t levelMask)
{
    for (int64_t lvl = 1; lvl < FPS_CPU_SUM_LEVELS; ++lvl) {
        for (int64_t r = 0; r < FPS_CPU_SUM_ILP; ++r) {
            for (int64_t l = 0; l < FPS_CPU_FLOAT_VECTOR_WIDTH; ++l) {
                accum[lvl][r][l] += accum[lvl - 1][r][l];
                accum[lvl - 1][r][l] = 0.0f;
            }
        }
        if ((group & (levelMask << (lvl * levelPower))) != 0) {
            break;
        }
    }
}

__aicore__ inline float FoldUbCascade(
    const LocalTensor<float>& ptLoc, const LocalTensor<float>& ctrLoc, int64_t featureDim,
    int64_t vectorChunks, int64_t groupCount,
    float accum[FPS_CPU_SUM_LEVELS][FPS_CPU_SUM_ILP][FPS_CPU_FLOAT_VECTOR_WIDTH])
{
    for (int64_t lvl = 1; lvl < FPS_CPU_SUM_LEVELS; ++lvl) {
        for (int64_t r = 0; r < FPS_CPU_SUM_ILP; ++r) {
            for (int64_t l = 0; l < FPS_CPU_FLOAT_VECTOR_WIDTH; ++l) {
                accum[0][r][l] += accum[lvl][r][l];
            }
        }
    }
    for (int64_t c = groupCount * FPS_CPU_SUM_ILP; c < vectorChunks; ++c) {
        for (int64_t l = 0; l < FPS_CPU_FLOAT_VECTOR_WIDTH; ++l) {
            accum[0][0][l] += LoadUbSquare(ptLoc, ctrLoc, c * FPS_CPU_FLOAT_VECTOR_WIDTH + l);
        }
    }
    for (int64_t r = 1; r < FPS_CPU_SUM_ILP; ++r) {
        for (int64_t l = 0; l < FPS_CPU_FLOAT_VECTOR_WIDTH; ++l) {
            accum[0][0][l] += accum[0][r][l];
        }
    }
    float value = 0.0f;
    for (int64_t f = vectorChunks * FPS_CPU_FLOAT_VECTOR_WIDTH; f < featureDim; ++f) {
        value += LoadUbSquare(ptLoc, ctrLoc, f);
    }
    for (int64_t l = 0; l < FPS_CPU_FLOAT_VECTOR_WIDTH; ++l) {
        value += accum[0][0][l];
    }
    return value;
}

__aicore__ inline float ComputeUbFloatCascade(
    const LocalTensor<float>& ptLoc, const LocalTensor<float>& ctrLoc, int64_t featureDim)
{
    if (featureDim < FPS_CPU_FLOAT_VECTOR_WIDTH) {
        return ComputeUbSmallFloat(ptLoc, ctrLoc, featureDim);
    }
    float accum[FPS_CPU_SUM_LEVELS][FPS_CPU_SUM_ILP][FPS_CPU_FLOAT_VECTOR_WIDTH] = {};
    const int64_t vectorChunks = featureDim / FPS_CPU_FLOAT_VECTOR_WIDTH;
    const int64_t groupCount = vectorChunks / FPS_CPU_SUM_ILP;
    const int64_t levelPower = UbCascadeLevelPower(groupCount);
    const int64_t levelStep = static_cast<int64_t>(1) << levelPower;
    const int64_t levelMask = levelStep - 1;
    int64_t group = 0;
    while (group < groupCount) {
        for (int64_t row = 0; row < FPS_CPU_SUM_ILP; ++row) {
            const int64_t base = (group * FPS_CPU_SUM_ILP + row) * FPS_CPU_FLOAT_VECTOR_WIDTH;
            for (int64_t lane = 0; lane < FPS_CPU_FLOAT_VECTOR_WIDTH; ++lane) {
                accum[0][row][lane] += LoadUbSquare(ptLoc, ctrLoc, base + lane);
            }
        }
        ++group;
        if ((group % levelStep) == 0) {
            CarryUbCascade(accum, group, levelPower, levelMask);
        }
    }
    return FoldUbCascade(ptLoc, ctrLoc, featureDim, vectorChunks, groupCount, accum);
}

template <typename T>
__aicore__ inline float ComputeRowSqDist(
    const GlobalTensor<T>& srcGm, uint64_t point, uint64_t center, uint64_t featureDim, bool useCascade,
    LocalTensor<T>& ptRawLoc, LocalTensor<T>& ctrRawLoc, LocalTensor<float>& ptF32Loc,
    LocalTensor<float>& ctrF32Loc, LocalTensor<float>& redLoc, LocalTensor<float>& workLoc)
{
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        if (useCascade && featureDim <= FALLBACK_FEAT_CHUNK) {
            const DataCopyExtParams cp{1U, static_cast<uint32_t>(featureDim * sizeof(float)), 0U, 0U, 0U};
            const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
            DataCopyPad(ptRawLoc, srcGm[point * featureDim], cp, pad);
            DataCopyPad(ctrRawLoc, srcGm[center * featureDim], cp, pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            const float res = ComputeUbFloatCascade(ptRawLoc, ctrRawLoc, static_cast<int64_t>(featureDim));
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            return res;
        }
    }
    float total = 0.0F;
    for (uint64_t fBase = 0; fBase < featureDim; fBase += FALLBACK_FEAT_CHUNK) {
        const uint64_t rem = featureDim - fBase;
        const uint32_t fLen = static_cast<uint32_t>(rem < FALLBACK_FEAT_CHUNK ? rem : FALLBACK_FEAT_CHUNK);
        const DataCopyExtParams cp{1U, fLen * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        const DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(ptRawLoc, srcGm[point * featureDim + fBase], cp, pad);
        DataCopyPad(ctrRawLoc, srcGm[center * featureDim + fBase], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        const int32_t n = static_cast<int32_t>(fLen);
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            Sub(ptRawLoc, ptRawLoc, ctrRawLoc, n);
            PipeBarrier<PIPE_V>();
            Mul(ptRawLoc, ptRawLoc, ptRawLoc, n);
            PipeBarrier<PIPE_V>();
            Cast(ptF32Loc, ptRawLoc, RoundMode::CAST_NONE, n);
        } else {
            Sub(ptF32Loc, ptRawLoc, ctrRawLoc, n);
            PipeBarrier<PIPE_V>();
            Mul(ptF32Loc, ptF32Loc, ptF32Loc, n);
        }
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(redLoc, ptF32Loc, workLoc, n);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        total += redLoc.GetValue(0);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    }
    return total;
}

template <typename T>
__aicore__ inline void InitFallbackDistTile(
    GlobalTensor<T>& distGm, uint64_t srcStart, uint64_t pointCount, LocalTensor<T>& tileLoc)
{
    const uint32_t infCount = static_cast<uint32_t>(pointCount < FALLBACK_PT_TILE ? pointCount : FALLBACK_PT_TILE);
    Duplicate<T>(tileLoc, static_cast<T>(__builtin_inff()), static_cast<int32_t>(infCount));
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    for (uint64_t off = 0; off < pointCount; off += FALLBACK_PT_TILE) {
        const uint64_t rem = pointCount - off;
        const uint32_t len = static_cast<uint32_t>(rem < FALLBACK_PT_TILE ? rem : FALLBACK_PT_TILE);
        const DataCopyExtParams copyP{1U, len * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(distGm[srcStart + off], tileLoc, copyP);
    }
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline void UpdateFastDistAndReduce(
    GlobalTensor<float>& distGm, uint64_t dstOffset, uint32_t len, LocalTensor<float>& tileLoc,
    LocalTensor<float>& distSrcLoc, LocalTensor<float>& redLoc, LocalTensor<float>& workLoc,
    float& bestDist, uint64_t& bestPoint)
{
    const DataCopyExtParams dP{1U, len * 4U, 0U, 0U, 0U};
    Min<float>(tileLoc, tileLoc, distSrcLoc, static_cast<int32_t>(len));
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    DataCopyPad(distGm[dstOffset], tileLoc, dP);
    ReduceMax<float>(redLoc, tileLoc, workLoc, static_cast<int32_t>(len), true);
    SetFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    const float candDist = redLoc.GetValue(0);
    const uint32_t candIdx = redLoc.ReinterpretCast<uint32_t>().GetValue(1);
    if (candDist > bestDist) {
        bestDist = candDist;
        bestPoint = dstOffset + candIdx;
    }
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void RunFallbackFastFloatSmallF(
    const GlobalTensor<float>& srcGm, GlobalTensor<float>& distGm, __gm__ int64_t* out, uint64_t srcStart,
    uint64_t pointCount, uint64_t outStart, uint64_t outEnd, uint64_t center, uint64_t featureDim,
    LocalTensor<float>& ptF32Loc, LocalTensor<float>& ctrF32Loc, LocalTensor<float>& redLoc,
    LocalTensor<float>& workLoc, LocalTensor<float>& tileLoc)
{
    InitFallbackDistTile(distGm, srcStart, pointCount, tileLoc);
    const uint32_t fLen = static_cast<uint32_t>(featureDim);
    const uint32_t strideElems = Align32BytesElems(fLen, 4U);
    const uint8_t repStride = static_cast<uint8_t>(strideElems / 8U);
    const DataCopyPadExtParams<float> padP{false, 0U, 0U, 0.0F};
    const DataCopyExtParams ctrP{1U, fLen * 4U, 0U, 0U, 0U};
    const BinaryRepeatParams subP{1U, 1U, 1U, repStride, repStride, 0U};
    const BinaryRepeatParams mulP{1U, 1U, 1U, repStride, repStride, repStride};

    for (uint64_t sample = outStart + 1; sample < outEnd; ++sample) {
        DataCopyPad(ctrF32Loc, srcGm[center * featureDim], ctrP, padP);
        float bestDist = -1.0F;
        uint64_t bestPoint = srcStart;
        for (uint64_t off = 0; off < pointCount; off += FALLBACK_FAST_PT) {
            const uint64_t rem = pointCount - off;
            const uint32_t len = static_cast<uint32_t>(rem < FALLBACK_FAST_PT ? rem : FALLBACK_FAST_PT);
            const DataCopyExtParams ptP{static_cast<uint16_t>(len), fLen * 4U, 0U, 0U, 0U};
            const DataCopyExtParams dP{1U, len * 4U, 0U, 0U, 0U};
            DataCopyPad(ptF32Loc, srcGm[(srcStart + off) * featureDim], ptP, padP);
            DataCopyPad(tileLoc, distGm[srcStart + off], dP, padP);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

            Sub<float>(ptF32Loc, ptF32Loc, ctrF32Loc, fLen, static_cast<uint8_t>(len), subP);
            PipeBarrier<PIPE_V>();
            Mul<float>(ptF32Loc, ptF32Loc, ptF32Loc, fLen, static_cast<uint8_t>(len), mulP);
            PipeBarrier<PIPE_V>();
            WholeReduceSum<float, true>(
                redLoc, ptF32Loc, static_cast<int32_t>(fLen), static_cast<int32_t>(len), 1, 1, repStride);
            PipeBarrier<PIPE_V>();
            UpdateFastDistAndReduce(
                distGm, srcStart + off, len, tileLoc, redLoc, redLoc, workLoc, bestDist, bestPoint);
        }
        center = bestPoint;
        out[sample] = static_cast<int64_t>(center);
    }
}

__aicore__ inline void RunFallbackFastFloatLargeF(
    const GlobalTensor<float>& srcGm, GlobalTensor<float>& distGm, __gm__ int64_t* out, uint64_t srcStart,
    uint64_t pointCount, uint64_t outStart, uint64_t outEnd, uint64_t center, uint64_t featureDim,
    LocalTensor<float>& ptF32Loc, LocalTensor<float>& ctrF32Loc, LocalTensor<float>& redLoc,
    LocalTensor<float>& workLoc, LocalTensor<float>& tileLoc)
{
    InitFallbackDistTile(distGm, srcStart, pointCount, tileLoc);
    const uint32_t fLen = static_cast<uint32_t>(featureDim);
    const uint32_t strideElems = Align32BytesElems(fLen, 4U);
    const uint32_t maxPts = FALLBACK_FAST_BUF_ELEMS / strideElems;
    const uint32_t tilePts = (maxPts > 64U) ? 64U : maxPts;
    const uint8_t repStride = static_cast<uint8_t>(strideElems / 8U);
    const DataCopyPadExtParams<float> padP{false, 0U, 0U, 0.0F};
    const DataCopyExtParams ctrP{1U, fLen * 4U, 0U, 0U, 0U};

    for (uint64_t sample = outStart + 1; sample < outEnd; ++sample) {
        DataCopyPad(ctrF32Loc, srcGm[center * featureDim], ctrP, padP);
        float bestDist = -1.0F;
        uint64_t bestPoint = srcStart;
        for (uint64_t off = 0; off < pointCount; off += tilePts) {
            const uint64_t rem = pointCount - off;
            const uint32_t len = static_cast<uint32_t>(rem < tilePts ? rem : tilePts);
            const DataCopyExtParams ptP{static_cast<uint16_t>(len), fLen * 4U, 0U, 0U, 0U};
            const DataCopyExtParams dP{1U, len * 4U, 0U, 0U, 0U};
            DataCopyPad(ptF32Loc, srcGm[(srcStart + off) * featureDim], ptP, padP);
            DataCopyPad(tileLoc, distGm[srcStart + off], dP, padP);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

            for (uint32_t p = 0; p < len; ++p) {
                Sub<float>(ptF32Loc[p * strideElems], ptF32Loc[p * strideElems], ctrF32Loc, static_cast<int32_t>(fLen));
            }
            PipeBarrier<PIPE_V>();
            Mul<float>(ptF32Loc, ptF32Loc, ptF32Loc, static_cast<int32_t>(len * strideElems));
            PipeBarrier<PIPE_V>();
            Duplicate<float>(workLoc, 0.0F, static_cast<int32_t>(len));
            PipeBarrier<PIPE_V>();
            for (uint32_t fOff = 0; fOff < fLen; fOff += 64U) {
                const uint32_t chunk = (fOff + 64U > fLen) ? (fLen - fOff) : 64U;
                WholeReduceSum<float, true>(
                    redLoc, ptF32Loc[fOff], static_cast<int32_t>(chunk), static_cast<int32_t>(len), 1, 1, repStride);
                PipeBarrier<PIPE_V>();
                Add<float>(workLoc, workLoc, redLoc, static_cast<int32_t>(len));
                PipeBarrier<PIPE_V>();
            }
            UpdateFastDistAndReduce(
                distGm, srcStart + off, len, tileLoc, workLoc, redLoc, workLoc, bestDist, bestPoint);
        }
        center = bestPoint;
        out[sample] = static_cast<int64_t>(center);
    }
}

template <typename T>
__aicore__ inline void UpdateFallbackTile(
    const GlobalTensor<T>& srcGm, GlobalTensor<T>& distGm, uint64_t tileStart, uint32_t len, uint64_t center,
    uint64_t featureDim, bool useCascade, LocalTensor<T>& ptRawLoc, LocalTensor<T>& ctrRawLoc,
    LocalTensor<float>& ptF32Loc, LocalTensor<float>& ctrF32Loc, LocalTensor<float>& redLoc,
    LocalTensor<float>& workLoc, LocalTensor<T>& tileLoc, float& bestDist, uint64_t& bestPoint)
{
    const DataCopyPadExtParams<T> padP{false, 0U, 0U, static_cast<T>(0)};
    const DataCopyExtParams copyP{1U, len * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
    DataCopyPad(tileLoc, distGm[tileStart], copyP, padP);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    for (uint32_t i = 0; i < len; ++i) {
        const uint64_t p = tileStart + i;
        const float curDist = ComputeRowSqDist(
            srcGm, p, center, featureDim, useCascade, ptRawLoc, ctrRawLoc, ptF32Loc, ctrF32Loc, redLoc, workLoc);
        const float prevDist = static_cast<float>(tileLoc.GetValue(i));
        const T curVal = static_cast<T>(curDist);
        const float minDist = (static_cast<float>(curVal) < prevDist) ? static_cast<float>(curVal) : prevDist;
        if (static_cast<float>(curVal) < prevDist) {
            tileLoc.SetValue(i, curVal);
        }
        if (minDist > bestDist) {
            bestDist = minDist;
            bestPoint = p;
        }
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    DataCopyPad(distGm[tileStart], tileLoc, copyP);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void RunFallbackBatch(
    const GlobalTensor<T>& srcGm, GlobalTensor<T>& distGm, __gm__ int64_t* out, uint64_t srcStart,
    uint64_t pointCount, uint64_t outStart, uint64_t outEnd, uint64_t center, uint64_t featureDim,
    LocalTensor<T>& ptRawLoc, LocalTensor<T>& ctrRawLoc, LocalTensor<float>& ptF32Loc,
    LocalTensor<float>& ctrF32Loc, LocalTensor<float>& redLoc, LocalTensor<float>& workLoc, LocalTensor<T>& tileLoc)
{
    const bool useCascade = (outEnd - outStart <= 8ULL);
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        if (!useCascade && featureDim <= 64ULL) {
            RunFallbackFastFloatSmallF(
                srcGm, distGm, out, srcStart, pointCount, outStart, outEnd, center, featureDim,
                ptF32Loc, ctrF32Loc, redLoc, workLoc, tileLoc);
            return;
        }
        if (!useCascade && featureDim <= FALLBACK_FEAT_CHUNK) {
            RunFallbackFastFloatLargeF(
                srcGm, distGm, out, srcStart, pointCount, outStart, outEnd, center, featureDim,
                ptF32Loc, ctrF32Loc, redLoc, workLoc, tileLoc);
            return;
        }
    }
    InitFallbackDistTile(distGm, srcStart, pointCount, tileLoc);
    for (uint64_t sample = outStart + 1; sample < outEnd; ++sample) {
        float bestDist = -1.0F;
        uint64_t bestPoint = srcStart;
        for (uint64_t off = 0; off < pointCount; off += FALLBACK_PT_TILE) {
            const uint64_t rem = pointCount - off;
            const uint32_t len = static_cast<uint32_t>(rem < FALLBACK_PT_TILE ? rem : FALLBACK_PT_TILE);
            UpdateFallbackTile(
                srcGm, distGm, srcStart + off, len, center, featureDim, useCascade,
                ptRawLoc, ctrRawLoc, ptF32Loc, ctrF32Loc, redLoc, workLoc, tileLoc, bestDist, bestPoint);
        }
        center = bestPoint;
        out[sample] = static_cast<int64_t>(center);
    }
}

template <typename T>
__aicore__ inline void RunFallback(
    __gm__ const T* src, __gm__ const int64_t* ptr, __gm__ const int64_t* outPtr, __gm__ const int64_t* start,
    __gm__ T* dist, __gm__ int64_t* out, uint64_t numPoints, uint64_t featureDim, uint64_t batchSize,
    uint64_t singleOutSize, uint64_t singleStart)
{
    GlobalTensor<T> srcGm;
    GlobalTensor<T> distGm;
    srcGm.SetGlobalBuffer(const_cast<__gm__ T*>(src), numPoints * featureDim);
    distGm.SetGlobalBuffer(dist, numPoints);

    TPipe pipe;
    TBuf<TPosition::VECCALC> ptRawBuf;
    TBuf<TPosition::VECCALC> ctrRawBuf;
    TBuf<TPosition::VECCALC> ptF32Buf;
    TBuf<TPosition::VECCALC> ctrF32Buf;
    TBuf<TPosition::VECCALC> redBuf;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> tileBuf;

    pipe.InitBuffer(ptRawBuf, FALLBACK_FAST_BUF_ELEMS * sizeof(T));
    pipe.InitBuffer(ctrRawBuf, FALLBACK_FEAT_CHUNK * sizeof(T));
    pipe.InitBuffer(ptF32Buf, FALLBACK_FAST_BUF_ELEMS * sizeof(float));
    pipe.InitBuffer(ctrF32Buf, FALLBACK_FEAT_CHUNK * sizeof(float));
    pipe.InitBuffer(redBuf, FALLBACK_FAST_PT * sizeof(float));
    pipe.InitBuffer(workBuf, 256U * sizeof(float));
    pipe.InitBuffer(tileBuf, Align32BytesElems(FALLBACK_PT_TILE, sizeof(T)) * sizeof(T));

    LocalTensor<T> ptRawLoc = ptRawBuf.Get<T>();
    LocalTensor<T> ctrRawLoc = ctrRawBuf.Get<T>();
    LocalTensor<float> ptF32Loc = ptF32Buf.Get<float>();
    LocalTensor<float> ctrF32Loc = ctrF32Buf.Get<float>();
    LocalTensor<float> redLoc = redBuf.Get<float>();
    LocalTensor<float> workLoc = workBuf.Get<float>();
    LocalTensor<T> tileLoc = tileBuf.Get<T>();

    const bool batched = (ptr != nullptr);
    for (uint64_t batch = 0; batch < batchSize; ++batch) {
        const uint64_t srcStart = batched ? static_cast<uint64_t>(ptr[batch]) : 0ULL;
        const uint64_t srcEnd = batched ? static_cast<uint64_t>(ptr[batch + 1]) : numPoints;
        const uint64_t outStart = batched ? static_cast<uint64_t>(outPtr[batch]) : 0ULL;
        const uint64_t outEnd = batched ? static_cast<uint64_t>(outPtr[batch + 1]) : singleOutSize;
        const uint64_t startOffset = batched ? static_cast<uint64_t>(start[batch]) : singleStart;
        if (srcStart >= srcEnd || outStart >= outEnd) {
            continue;
        }
        const uint64_t center = srcStart + startOffset;
        out[outStart] = static_cast<int64_t>(center);
        if (outEnd - outStart == 1) {
            continue;
        }
        RunFallbackBatch(
            srcGm, distGm, out, srcStart, srcEnd - srcStart, outStart, outEnd, center, featureDim,
            ptRawLoc, ctrRawLoc, ptF32Loc, ctrF32Loc, redLoc, workLoc, tileLoc);
    }
}

} // namespace

template <typename T>
__global__ __aicore__ void FpsKernel(
    __gm__ const T* src, __gm__ const T* srcSoa, __gm__ const uint32_t* ptr, __gm__ const uint32_t* outPtr,
    __gm__ const uint32_t* start, __gm__ T* dist, __gm__ uint32_t* out, uint64_t numPoints, uint64_t featureDim,
    uint64_t batchSize, uint64_t singleOutSize, uint64_t singleStart, uint32_t tilePoints, uint32_t tileBytes,
    uint32_t promotedTileBytes, uint32_t cascadeTileBytes, uint32_t centerBytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    opsgnn::fps::FpsVectorKernel<T> op;
    op.Init(
        src, srcSoa, ptr, outPtr, start, dist, out, numPoints, featureDim, batchSize, singleOutSize, singleStart,
        tilePoints, tileBytes, promotedTileBytes, cascadeTileBytes, centerBytes, &pipe);
    op.Process();
}

template <typename T>
__global__ __aicore__ void FpsFallbackKernel(
    __gm__ const T* src, __gm__ const int64_t* ptr, __gm__ const int64_t* outPtr, __gm__ const int64_t* start,
    __gm__ T* dist, __gm__ int64_t* out, uint64_t numPoints, uint64_t featureDim, uint64_t batchSize,
    uint64_t singleOutSize, uint64_t singleStart)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunFallback(src, ptr, outPtr, start, dist, out, numPoints, featureDim, batchSize, singleOutSize, singleStart);
}

template <typename T, int DType>
void Fps(const T* src, const FpsLaunchArgs<T>& args, aclrtStream stream)
{
    const uint32_t* p = reinterpret_cast<const uint32_t*>(args.ptr);
    const uint32_t* op = reinterpret_cast<const uint32_t*>(args.out_ptr);
    const uint32_t* st = reinterpret_cast<const uint32_t*>(args.start);
    uint32_t* o = reinterpret_cast<uint32_t*>(args.out);
    const bool isFloat = (DType == 0);
    if (FpsUsesFallback(DType, args.num_points, args.feature_dim)) {
        if (isFloat) {
            FpsFallbackKernel<float><<<1, nullptr, stream>>>(
                reinterpret_cast<const float*>(src), args.ptr, args.out_ptr, args.start,
                reinterpret_cast<float*>(args.dist), args.out, args.num_points, args.feature_dim, args.batch_size,
                args.single_out_size, args.single_start);
        } else {
            FpsFallbackKernel<half><<<1, nullptr, stream>>>(
                reinterpret_cast<const half*>(src), args.ptr, args.out_ptr, args.start,
                reinterpret_cast<half*>(args.dist), args.out, args.num_points, args.feature_dim, args.batch_size,
                args.single_out_size, args.single_start);
        }
        return;
    }
    if (isFloat) {
        FpsKernel<float><<<1, nullptr, stream>>>(
            reinterpret_cast<const float*>(src), reinterpret_cast<const float*>(args.src_soa), p, op, st,
            reinterpret_cast<float*>(args.dist), o, args.num_points, args.feature_dim, args.batch_size,
            args.single_out_size, args.single_start, args.tile_points, args.tile_bytes, args.promoted_tile_bytes,
            args.cascade_tile_bytes, args.center_bytes);
    } else {
        FpsKernel<half><<<1, nullptr, stream>>>(
            reinterpret_cast<const half*>(src), reinterpret_cast<const half*>(args.src_soa), p, op, st,
            reinterpret_cast<half*>(args.dist), o, args.num_points, args.feature_dim, args.batch_size,
            args.single_out_size, args.single_start, args.tile_points, args.tile_bytes, args.promoted_tile_bytes,
            args.cascade_tile_bytes, args.center_bytes);
    }
}

template void Fps<float, 0>(const float*, const FpsLaunchArgs<float>&, aclrtStream);
template void Fps<uint16_t, 1>(const uint16_t*, const FpsLaunchArgs<uint16_t>&, aclrtStream);

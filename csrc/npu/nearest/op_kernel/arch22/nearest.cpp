/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "nearest/op_kernel/arch35/nearest.h"

#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220
#error "arch22/nearest.cpp must be compiled for dav-2201 (A2/A3)"
#endif

using namespace AscendC;

namespace {

constexpr uint32_t PREP_CHUNK = 4096U;
constexpr uint32_t TRANS_M_TILE = 64U;
constexpr uint32_t TRANS_F_TILE = 64U;
constexpr uint32_t SEARCH_Y_TILE = 1024U;
constexpr uint32_t MAX_FEAT_UB = 4096U;

__aicore__ inline uint32_t Align32BytesElems(uint32_t count, uint32_t elemSize)
{
    if (elemSize == 0U || elemSize > 32U) {
        return count;
    }
    const uint32_t perBlock = 32U / elemSize;
    return ((count + perBlock - 1U) / perBlock) * perBlock;
}

template <typename T>
__aicore__ inline float ReadAsFloat(const LocalTensor<T>& buf, uint32_t idx)
{
    return static_cast<float>(buf.GetValue(idx));
}

template <>
__aicore__ inline float ReadAsFloat<half>(const LocalTensor<half>& buf, uint32_t idx)
{
    return static_cast<float>(buf.GetValue(idx));
}

template <typename T>
__aicore__ inline void ConvertXChunk(
    const GlobalTensor<T>& xGm, const GlobalTensor<float>& dstGm, LocalTensor<T>& inLoc,
    LocalTensor<float>& outLoc, uint64_t off, uint32_t len)
{
    const DataCopyExtParams inParams{1U, len * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
    const DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
    DataCopyPad(inLoc, xGm[off], inParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    const uint32_t aligned = Align32BytesElems(len, static_cast<uint32_t>(sizeof(T)));
    Cast(outLoc, inLoc, RoundMode::CAST_NONE, aligned);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    const DataCopyExtParams outParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(dstGm[off], outLoc, outParams);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void TransposeYTile(
    const GlobalTensor<T>& yGm, const GlobalTensor<float>& yTGm, LocalTensor<T>& yTile,
    LocalTensor<float>& yTrans, uint32_t mBase, uint32_t mLen, uint32_t fBase, uint32_t fLen,
    uint32_t mTotal, uint32_t fTotal)
{
    const uint32_t inStrideBytes = (fTotal - fLen) * static_cast<uint32_t>(sizeof(T));
    const DataCopyExtParams inParams{
        static_cast<uint16_t>(mLen), fLen * static_cast<uint32_t>(sizeof(T)), inStrideBytes, 0U, 0U};
    const DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
    DataCopyPad(yTile, yGm[static_cast<uint64_t>(mBase) * fTotal + fBase], inParams, padParams);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    const uint32_t fStride = Align32BytesElems(fLen, static_cast<uint32_t>(sizeof(T)));
    const uint32_t mStride = Align32BytesElems(mLen, static_cast<uint32_t>(sizeof(float)));
    for (uint32_t d = 0U; d < fLen; ++d) {
        for (uint32_t r = 0U; r < mLen; ++r) {
            yTrans.SetValue(d * mStride + r, ReadAsFloat(yTile, r * fStride + d));
        }
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

    const uint32_t outBytes = mLen * static_cast<uint32_t>(sizeof(float));
    const uint32_t outDstStride = (mTotal - mLen) * static_cast<uint32_t>(sizeof(float));
    const DataCopyExtParams outParams{static_cast<uint16_t>(fLen), outBytes, 0U, outDstStride, 0U};
    DataCopyPad(yTGm[static_cast<uint64_t>(fBase) * mTotal + mBase], yTrans, outParams);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void ComputePrepareNorms(
    const GlobalTensor<T>& xGm, const GlobalTensor<T>& yGm, GlobalTensor<float>& normGm, LocalTensor<T>& inLoc,
    LocalTensor<float>& outLoc, const NearestTiling& t, uint32_t coreIdx, uint32_t coreNum)
{
    const uint32_t totalPts = t.n + t.m;
    for (uint32_t p = coreIdx; p < totalPts; p += coreNum) {
        const bool isX = (p < t.n);
        const uint32_t row = isX ? p : (p - t.n);
        float norm = 0.0F;
        for (uint32_t fBase = 0U; fBase < t.features; fBase += PREP_CHUNK) {
            const uint32_t fLen = (fBase + PREP_CHUNK > t.features) ? (t.features - fBase) : PREP_CHUNK;
            const DataCopyExtParams inP{1U, fLen * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            const DataCopyPadExtParams<T> padP{false, 0U, 0U, static_cast<T>(0)};
            if (isX) {
                DataCopyPad(inLoc, xGm[static_cast<uint64_t>(row) * t.features + fBase], inP, padP);
            } else {
                DataCopyPad(inLoc, yGm[static_cast<uint64_t>(row) * t.features + fBase], inP, padP);
            }
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            for (uint32_t d = 0U; d < fLen; ++d) {
                const float v = ReadAsFloat(inLoc, d);
                volatile float square = v * v;
                norm = norm + square;
            }
        }
        outLoc.SetValue(0, norm);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopyPad(normGm[p], outLoc, DataCopyExtParams{1U, 4U, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    }
}

template <typename T>
__aicore__ inline void RunPrepare(GM_ADDR xAddr, GM_ADDR yAddr, GM_ADDR workAddr, NearestTiling t)
{
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint64_t nx = static_cast<uint64_t>(t.n) * t.features;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> xF32Gm;
    GlobalTensor<float> yTGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xAddr));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(yAddr));
    xF32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workAddr));
    yTGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workAddr) + nx);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bufIn;
    TBuf<TPosition::VECCALC> bufOut;
    pipe.InitBuffer(bufIn, PREP_CHUNK * sizeof(float));
    pipe.InitBuffer(bufOut, PREP_CHUNK * sizeof(float));
    LocalTensor<T> inLoc = bufIn.Get<T>();
    LocalTensor<float> outLoc = bufOut.Get<float>();

    if constexpr (sizeof(T) == sizeof(half)) {
        const uint64_t numChunks = (nx + PREP_CHUNK - 1U) / PREP_CHUNK;
        for (uint64_t c = coreIdx; c < numChunks; c += coreNum) {
            const uint64_t off = c * PREP_CHUNK;
            const uint32_t len = static_cast<uint32_t>((off + PREP_CHUNK > nx) ? (nx - off) : PREP_CHUNK);
            ConvertXChunk<T>(xGm, xF32Gm, inLoc, outLoc, off, len);
        }
    }

    if (t.features >= 5U) {
        GlobalTensor<float> normGm;
        normGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workAddr) + nx + static_cast<uint64_t>(t.m) * t.features);
        ComputePrepareNorms<T>(xGm, yGm, normGm, inLoc, outLoc, t, coreIdx, coreNum);
    }

    const uint32_t mTiles = (t.m + TRANS_M_TILE - 1U) / TRANS_M_TILE;
    const uint32_t fTiles = (t.features + TRANS_F_TILE - 1U) / TRANS_F_TILE;
    const uint32_t totalTiles = mTiles * fTiles;
    for (uint32_t tile = coreIdx; tile < totalTiles; tile += coreNum) {
        const uint32_t mt = tile / fTiles;
        const uint32_t ft = tile % fTiles;
        const uint32_t mBase = mt * TRANS_M_TILE;
        const uint32_t fBase = ft * TRANS_F_TILE;
        const uint32_t mLen = (mBase + TRANS_M_TILE > t.m) ? (t.m - mBase) : TRANS_M_TILE;
        const uint32_t fLen = (fBase + TRANS_F_TILE > t.features) ? (t.features - fBase) : TRANS_F_TILE;
        TransposeYTile<T>(yGm, yTGm, inLoc, outLoc, mBase, mLen, fBase, fLen, t.m, t.features);
    }
}

__aicore__ inline uint32_t FindBatch(uint32_t row, __gm__ const int64_t* ptr, uint32_t batches)
{
    uint32_t lo = 0U;
    uint32_t hi = batches;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2U;
        if (ptr[mid + 1U] <= static_cast<int64_t>(row)) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__aicore__ inline bool ResolveRowBatchRange(
    uint32_t row, __gm__ const int64_t* ptrX, __gm__ const int64_t* ptrY, const NearestTiling& t,
    uint32_t& yBegin, uint32_t& yEnd)
{
    if (t.batches == 0U) {
        yBegin = 0U;
        yEnd = t.m;
        return true;
    }
    const uint32_t batch = FindBatch(row, ptrX, t.batches);
    if (batch >= t.batches) {
        return false;
    }
    const int64_t a = ptrY[batch];
    const int64_t b = ptrY[batch + 1U];
    if (a < 0 || b <= a || b > static_cast<int64_t>(t.m)) {
        return false;
    }
    yBegin = static_cast<uint32_t>(a);
    yEnd = static_cast<uint32_t>(b);
    return true;
}

__aicore__ inline void AccumulatePanelDot(
    const GlobalTensor<float>& xGm, const GlobalTensor<float>& yTGm, const GlobalTensor<float>& normGm,
    uint32_t row, uint32_t yOff, uint32_t lenAligned, const NearestTiling& t, const DataCopyExtParams& yParams,
    LocalTensor<float>& xLoc, LocalTensor<float>& yLoc, LocalTensor<float>& tmpLoc, LocalTensor<float>& distLoc)
{
    const DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    for (uint32_t beginK = 0U; beginK < t.features;) {
        uint32_t count = t.features - beginK;
        count = (count >= 896U) ? 448U : ((count > 448U) ? ((count + 1U) / 2U) : count);
        Duplicate(tmpLoc, 0.0F, lenAligned);
        PipeBarrier<PIPE_V>();
        for (uint32_t fOff = 0U; fOff < count; fOff += MAX_FEAT_UB) {
            const uint32_t fLen = (fOff + MAX_FEAT_UB > count) ? (count - fOff) : MAX_FEAT_UB;
            const DataCopyExtParams xParams{1U, fLen * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(xLoc, xGm[static_cast<uint64_t>(row) * t.features + beginK + fOff], xParams, padParams);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            for (uint32_t d = 0U; d < fLen; ++d) {
                const float xVal = xLoc.GetValue(d);
                const uint64_t yAddr = static_cast<uint64_t>(beginK + fOff + d) * t.m + yOff;
                DataCopyPad(yLoc, yTGm[yAddr], yParams, padParams);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Axpy(tmpLoc, yLoc, xVal, lenAligned);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
        }
        PipeBarrier<PIPE_V>();
        Axpy(distLoc, tmpLoc, -2.0F, lenAligned);
        PipeBarrier<PIPE_V>();
        beginK += count;
    }
    DataCopyPad(xLoc, normGm[row], DataCopyExtParams{1U, 4U, 0U, 0U, 0U}, padParams);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    const float normX = xLoc.GetValue(0);
    DataCopyPad(yLoc, normGm[t.n + yOff], yParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Adds(distLoc, distLoc, normX, lenAligned);
    PipeBarrier<PIPE_V>();
    Add(distLoc, distLoc, yLoc, lenAligned);
}

__aicore__ inline void AccumulateSqDiff(
    const GlobalTensor<float>& xGm, const GlobalTensor<float>& yTGm, uint32_t row, uint32_t yOff,
    uint32_t lenAligned, const NearestTiling& t, const DataCopyExtParams& yParams, LocalTensor<float>& xLoc,
    LocalTensor<float>& yLoc, LocalTensor<float>& tmpLoc, LocalTensor<float>& distLoc)
{
    const DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    for (uint32_t fBase = 0U; fBase < t.features; fBase += MAX_FEAT_UB) {
        const uint32_t fLen = (fBase + MAX_FEAT_UB > t.features) ? (t.features - fBase) : MAX_FEAT_UB;
        const DataCopyExtParams xParams{1U, fLen * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(xLoc, xGm[static_cast<uint64_t>(row) * t.features + fBase], xParams, padParams);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        for (uint32_t d = 0U; d < fLen; ++d) {
            const float xVal = xLoc.GetValue(d);
            const uint64_t yAddr = static_cast<uint64_t>(fBase + d) * t.m + yOff;
            DataCopyPad(yLoc, yTGm[yAddr], yParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Adds(tmpLoc, yLoc, -xVal, lenAligned);
            PipeBarrier<PIPE_V>();
            Mul(tmpLoc, tmpLoc, tmpLoc, lenAligned);
            PipeBarrier<PIPE_V>();
            Add(distLoc, distLoc, tmpLoc, lenAligned);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
    }
}

__aicore__ inline void ComputeChunkMin(
    const GlobalTensor<float>& xGm, const GlobalTensor<float>& yTGm, const GlobalTensor<float>& normGm,
    uint32_t row, uint32_t yOff, uint32_t len, const NearestTiling& t, LocalTensor<float>& xLoc,
    LocalTensor<float>& yLoc, LocalTensor<float>& tmpLoc, LocalTensor<float>& distLoc, LocalTensor<float>& redLoc,
    LocalTensor<float>& workLoc, float& chunkMin, uint32_t& chunkArgMin)
{
    const uint32_t lenAligned = Align32BytesElems(len, static_cast<uint32_t>(sizeof(float)));
    Duplicate(distLoc, 0.0F, lenAligned);
    PipeBarrier<PIPE_V>();

    const DataCopyExtParams yParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    if (t.features >= 5U) {
        AccumulatePanelDot(xGm, yTGm, normGm, row, yOff, lenAligned, t, yParams, xLoc, yLoc, tmpLoc, distLoc);
    } else {
        AccumulateSqDiff(xGm, yTGm, row, yOff, lenAligned, t, yParams, xLoc, yLoc, tmpLoc, distLoc);
    }

    PipeBarrier<PIPE_V>();
    ReduceMin<float>(redLoc, distLoc, workLoc, len, true);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);

    chunkMin = redLoc.GetValue(0);
    const uint32_t rawIdx = redLoc.ReinterpretCast<uint32_t>().GetValue(1);
    chunkArgMin = yOff + rawIdx;
}

__aicore__ inline void SearchRow(
    const GlobalTensor<float>& xGm, const GlobalTensor<float>& yTGm, const GlobalTensor<float>& normGm,
    GlobalTensor<int64_t>& outGm, __gm__ const int64_t* ptrX, __gm__ const int64_t* ptrY, uint32_t row,
    const NearestTiling& t, LocalTensor<float>& xLoc, LocalTensor<float>& yLoc, LocalTensor<float>& tmpLoc,
    LocalTensor<float>& distLoc, LocalTensor<float>& redLoc, LocalTensor<float>& workLoc, LocalTensor<int64_t>& outLoc)
{
    uint32_t yBegin = 0U;
    uint32_t yEnd = 0U;
    if (!ResolveRowBatchRange(row, ptrX, ptrY, t, yBegin, yEnd)) {
        return;
    }
    float bestMin = 0.0F;
    uint32_t bestIdx = yBegin;
    bool hasBest = false;
    for (uint32_t off = yBegin; off < yEnd; off += SEARCH_Y_TILE) {
        const uint32_t len = (off + SEARCH_Y_TILE > yEnd) ? (yEnd - off) : SEARCH_Y_TILE;
        float chunkMin = 0.0F;
        uint32_t chunkIdx = off;
        ComputeChunkMin(
            xGm, yTGm, normGm, row, off, len, t, xLoc, yLoc, tmpLoc, distLoc, redLoc, workLoc, chunkMin, chunkIdx);
        if (!hasBest || chunkMin < bestMin) {
            bestMin = chunkMin;
            bestIdx = chunkIdx;
            hasBest = true;
        }
    }
    outLoc.SetValue(0, static_cast<int64_t>(bestIdx));
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    DataCopyPad(outGm[row], outLoc, DataCopyExtParams{1U, 8U, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
}

__aicore__ inline void RunSearch(
    GM_ADDR xF32Addr, GM_ADDR yTAddr, GM_ADDR normAddr, GM_ADDR ptrXAddr, GM_ADDR ptrYAddr, GM_ADDR outAddr,
    NearestTiling t)
{
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());

    GlobalTensor<float> xGm;
    GlobalTensor<float> yTGm;
    GlobalTensor<float> normGm;
    GlobalTensor<int64_t> outGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xF32Addr));
    yTGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yTAddr));
    normGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(normAddr));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(outAddr));

    __gm__ const int64_t* ptrX = reinterpret_cast<__gm__ const int64_t*>(ptrXAddr);
    __gm__ const int64_t* ptrY = reinterpret_cast<__gm__ const int64_t*>(ptrYAddr);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bufX;
    TBuf<TPosition::VECCALC> bufY;
    TBuf<TPosition::VECCALC> bufTmp;
    TBuf<TPosition::VECCALC> bufDist;
    TBuf<TPosition::VECCALC> bufRed;
    TBuf<TPosition::VECCALC> bufWork;
    TBuf<TPosition::VECCALC> bufOut;

    const uint32_t xBufElems = (t.features < MAX_FEAT_UB) ? Align32BytesElems(t.features, 4U) : MAX_FEAT_UB;
    pipe.InitBuffer(bufX, xBufElems * sizeof(float));
    pipe.InitBuffer(bufY, SEARCH_Y_TILE * sizeof(float));
    pipe.InitBuffer(bufTmp, SEARCH_Y_TILE * sizeof(float));
    pipe.InitBuffer(bufDist, SEARCH_Y_TILE * sizeof(float));
    pipe.InitBuffer(bufRed, 64U * sizeof(float));
    pipe.InitBuffer(bufWork, 256U * sizeof(float));
    pipe.InitBuffer(bufOut, 32U);

    LocalTensor<float> xLoc = bufX.Get<float>();
    LocalTensor<float> yLoc = bufY.Get<float>();
    LocalTensor<float> tmpLoc = bufTmp.Get<float>();
    LocalTensor<float> distLoc = bufDist.Get<float>();
    LocalTensor<float> redLoc = bufRed.Get<float>();
    LocalTensor<float> workLoc = bufWork.Get<float>();
    LocalTensor<int64_t> outLoc = bufOut.Get<int64_t>();

    for (uint32_t row = coreIdx; row < t.n; row += coreNum) {
        SearchRow(
            xGm, yTGm, normGm, outGm, ptrX, ptrY, row, t, xLoc, yLoc, tmpLoc, distLoc, redLoc, workLoc, outLoc);
    }
}

} // namespace

extern "C" __global__ __aicore__ void nearest_prepare_fp16(
    GM_ADDR xAddr, GM_ADDR yAddr, GM_ADDR workAddr, NearestTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunPrepare<half>(xAddr, yAddr, workAddr, t);
}

extern "C" __global__ __aicore__ void nearest_prepare_fp32(
    GM_ADDR xAddr, GM_ADDR yAddr, GM_ADDR workAddr, NearestTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunPrepare<float>(xAddr, yAddr, workAddr, t);
}

extern "C" __global__ __aicore__ void nearest_search(
    GM_ADDR xF32Addr, GM_ADDR yTAddr, GM_ADDR normAddr, GM_ADDR ptrXAddr, GM_ADDR ptrYAddr, GM_ADDR outAddr,
    NearestTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunSearch(xF32Addr, yTAddr, normAddr, ptrXAddr, ptrYAddr, outAddr, t);
}

namespace opsgnn {

void Nearest(
    const void* x, const void* y, const int64_t* ptr_x, const int64_t* ptr_y, float* workspace, int64_t* out,
    bool is_half, const NearestTiling& tiling, aclrtStream stream)
{
    if (tiling.n == 0U || tiling.m == 0U || tiling.cores == 0U) {
        return;
    }
    const uint64_t nx = static_cast<uint64_t>(tiling.n) * tiling.features;
    const uint64_t ny = static_cast<uint64_t>(tiling.m) * tiling.features;
    uint8_t* xGm = reinterpret_cast<uint8_t*>(const_cast<void*>(x));
    uint8_t* yGm = reinterpret_cast<uint8_t*>(const_cast<void*>(y));
    uint8_t* workGm = reinterpret_cast<uint8_t*>(workspace);
    uint8_t* yTGm = reinterpret_cast<uint8_t*>(workspace + nx);
    uint8_t* normGm = reinterpret_cast<uint8_t*>(workspace + nx + ny);
    uint8_t* xF32Gm = is_half ? workGm : xGm;

    if (is_half) {
        nearest_prepare_fp16<<<tiling.cores, nullptr, stream>>>(xGm, yGm, workGm, tiling);
    } else {
        nearest_prepare_fp32<<<tiling.cores, nullptr, stream>>>(xGm, yGm, workGm, tiling);
    }
    nearest_search<<<tiling.cores, nullptr, stream>>>(
        xF32Gm, yTGm, normGm, reinterpret_cast<uint8_t*>(const_cast<int64_t*>(ptr_x)),
        reinterpret_cast<uint8_t*>(const_cast<int64_t*>(ptr_y)), reinterpret_cast<uint8_t*>(out), tiling);
}

} // namespace opsgnn

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
#include "csr_row_wise_per_etype_sampling_uniform/op_kernel/arch22/csr_row_wise_per_etype_sampling_uniform_tiling.h"

using namespace AscendC;

namespace opsgnn {
namespace csr_per_etype {

__aicore__ inline uint32_t Xorshift32(uint32_t& x)
{
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__aicore__ inline uint32_t RandBelow(uint32_t& state, uint32_t n)
{
    if (n == 0) {
        return 0;
    }
    uint32_t r = Xorshift32(state);
    return static_cast<uint32_t>((static_cast<uint64_t>(r) * n) >> 32);
}

template <typename IdT>
class KernelCsrRowWisePerEtypeSamplingUniform {
public:
    __aicore__ inline KernelCsrRowWisePerEtypeSamplingUniform() {}

    __aicore__ inline void Init(
        GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
        GM_ADDR outPtr, GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outData,
        GM_ADDR rowSplit, GM_ADDR outStarts, GM_ADDR pickScratch,
        const CsrRowWisePerEtypeSamplingUniformTiling& tiling, TPipe* pipe)
    {
        numRows_ = tiling.numRows;
        numSamples_ = tiling.numSamples;
        replace_ = tiling.replace;
        hasData_ = tiling.hasData;
        seed_ = tiling.seed;
        selectAll_ = tiling.selectAll;
        numTotalRows_ = tiling.numTotalRows;
        pickStride_ = tiling.pickStride;

        indptrGm_.SetGlobalBuffer((__gm__ IdT*)indptr, numTotalRows_ + 1);
        indicesGm_.SetGlobalBuffer((__gm__ IdT*)indices);
        if (hasData_) {
            dataGm_.SetGlobalBuffer((__gm__ IdT*)data);
        }
        rowsGm_.SetGlobalBuffer((__gm__ IdT*)rows, numRows_);
        (void)outPtr;
        outRowsGm_.SetGlobalBuffer((__gm__ IdT*)outRows);
        outColsGm_.SetGlobalBuffer((__gm__ IdT*)outCols);
        outDataGm_.SetGlobalBuffer((__gm__ IdT*)outData);
        if (pickScratch != nullptr) {
            pickScratchGm_.SetGlobalBuffer((__gm__ IdT*)pickScratch);
        }

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        blockIdx_ = blockIdx;
        const __gm__ uint32_t* rowSplitP = (const __gm__ uint32_t*)rowSplit;
        const __gm__ uint32_t* outStartsP = (const __gm__ uint32_t*)outStarts;
        rowBegin_ = rowSplitP[blockIdx];
        rowEnd_ = rowSplitP[blockIdx + 1];
        outStart_ = outStartsP[blockIdx];

        const uint32_t ubAvailable = tiling.ubAvailable;
        windowElems_ = ubAvailable / kUbInstances / sizeof(IdT);
        pipe->InitBuffer(winIdxQ_, kQueueDepth, windowElems_ * sizeof(IdT));
        pipe->InitBuffer(winDataQ_, kQueueDepth, windowElems_ * sizeof(IdT));
        pipe->InitBuffer(pickBuf_, windowElems_ * sizeof(uint32_t));
        pipe->InitBuffer(outRBuf_, windowElems_ * sizeof(IdT));
        pipe->InitBuffer(outCBuf_, windowElems_ * sizeof(IdT));
        pipe->InitBuffer(outEBuf_, windowElems_ * sizeof(IdT));
        pipe->InitBuffer(outQ_, kQueueDepth, windowElems_ * sizeof(IdT));
    }

    __aicore__ inline void Process()
    {
        if (rowBegin_ >= rowEnd_) {
            return;
        }
        uint32_t offset = 0;
        for (uint32_t i = rowBegin_; i < rowEnd_; ++i) {
            IdT rid = rowsGm_.GetValue(i);
            if (rid < 0 || rid >= static_cast<IdT>(numTotalRows_)) {
                continue;
            }
            IdT off = indptrGm_.GetValue(static_cast<uint32_t>(rid));
            IdT end = indptrGm_.GetValue(static_cast<uint32_t>(rid) + 1);
            uint32_t deg = static_cast<uint32_t>(end - off);
            uint32_t numPicks = selectAll_
                ? deg
                : (replace_ ? (deg == 0 ? 0u : numSamples_)
                            : (deg < numSamples_ ? deg : numSamples_));
            if (numPicks > 0) {
                uint32_t state = seed_ ^ (i * kGoldenRatioHash + kGoldenRatioOffset);
                if (state == 0) {
                    state = kRngFallbackSeed;
                }
                offset += SampleRow(outStart_ + offset, rid, off, deg, numPicks, state);
            }
        }
    }

private:
    __aicore__ inline uint32_t SampleRow(
        uint32_t outPos, IdT rid, IdT off, uint32_t deg, uint32_t numPicks, uint32_t& state)
    {
        if (deg <= windowElems_ && numPicks <= windowElems_) {
            return SampleRowThroughUb(outPos, rid, off, deg, numPicks, state);
        }
        return SampleRowDirectGm(outPos, rid, off, deg, numPicks, state);
    }

    __aicore__ inline void BuildUbPicks(
        uint32_t deg, uint32_t numPicks, uint32_t& state,
        LocalTensor<uint32_t>& picks)
    {
        if (selectAll_ || (!replace_ && numPicks == deg)) {
            for (uint32_t j = 0; j < numPicks; ++j) {
                picks.SetValue(j, j);
            }
        } else if (replace_) {
            for (uint32_t j = 0; j < numPicks; ++j) {
                picks.SetValue(j, RandBelow(state, deg));
            }
        } else {
            for (uint32_t j = 0; j < numPicks; ++j) {
                picks.SetValue(j, j);
            }
            for (uint32_t i2 = numPicks; i2 < deg; ++i2) {
                uint32_t j = RandBelow(state, i2 + 1);
                if (j < numPicks) {
                    picks.SetValue(j, i2);
                }
            }
        }
    }

    __aicore__ inline uint32_t SampleRowThroughUb(
        uint32_t outPos, IdT rid, IdT off, uint32_t deg, uint32_t numPicks, uint32_t& state)
    {
        const uint32_t copyBytes = deg * sizeof(IdT);
        DataCopyExtParams cp{1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};

        LocalTensor<IdT> winIdx = winIdxQ_.AllocTensor<IdT>();
        DataCopyPad(winIdx, indicesGm_[off], cp, pad);
        winIdxQ_.EnQue(winIdx);
        winIdx = winIdxQ_.DeQue<IdT>();

        LocalTensor<IdT> winData;
        if (hasData_) {
            winData = winDataQ_.AllocTensor<IdT>();
            DataCopyPad(winData, dataGm_[off], cp, pad);
            winDataQ_.EnQue(winData);
            winData = winDataQ_.DeQue<IdT>();
        }

        LocalTensor<uint32_t> picks = pickBuf_.Get<uint32_t>();
        BuildUbPicks(deg, numPicks, state, picks);

        LocalTensor<IdT> outR = outRBuf_.Get<IdT>();
        LocalTensor<IdT> outC = outCBuf_.Get<IdT>();
        LocalTensor<IdT> outE = outEBuf_.Get<IdT>();
        for (uint32_t j = 0; j < numPicks; ++j) {
            uint32_t local = picks.GetValue(j);
            outR.SetValue(j, rid);
            outC.SetValue(j, winIdx.GetValue(local));
            outE.SetValue(j, hasData_ ? winData.GetValue(local)
                                      : static_cast<IdT>(off + local));
        }
        CopyOutStaged(outR, outRowsGm_[outPos], numPicks);
        CopyOutStaged(outC, outColsGm_[outPos], numPicks);
        CopyOutStaged(outE, outDataGm_[outPos], numPicks);

        winIdxQ_.FreeTensor(winIdx);
        if (hasData_) {
            winDataQ_.FreeTensor(winData);
        }
        return numPicks;
    }

    __aicore__ inline bool PrepareNoReplacePicks(
        uint32_t deg, uint32_t numPicks, uint32_t& state,
        LocalTensor<uint32_t>& ubPicks, uint32_t& gmPicksBase)
    {
        gmPicksBase = 0;
        const bool useUbPicks = numPicks <= windowElems_;
        if (useUbPicks) {
            ubPicks = pickBuf_.Get<uint32_t>();
            for (uint32_t j = 0; j < numPicks; ++j) {
                ubPicks.SetValue(j, j);
            }
        } else {
            gmPicksBase = blockIdx_ * pickStride_;
            for (uint32_t j = 0; j < numPicks; ++j) {
                pickScratchGm_.SetValue(gmPicksBase + j, static_cast<IdT>(j));
            }
        }
        for (uint32_t i2 = numPicks; i2 < deg; ++i2) {
            uint32_t j = RandBelow(state, i2 + 1);
            if (j < numPicks) {
                if (useUbPicks) {
                    ubPicks.SetValue(j, i2);
                } else {
                    pickScratchGm_.SetValue(gmPicksBase + j, static_cast<IdT>(i2));
                }
            }
        }
        return useUbPicks;
    }

    __aicore__ inline uint32_t ResolveDirectLocal(
        uint32_t done, uint32_t j, uint32_t deg, bool identity,
        bool useUbPicks, uint32_t gmPicksBase,
        LocalTensor<uint32_t>& ubPicks, uint32_t& state)
    {
        if (identity) return done + j;
        if (replace_) return RandBelow(state, deg);
        if (useUbPicks) return ubPicks.GetValue(done + j);
        return static_cast<uint32_t>(pickScratchGm_.GetValue(gmPicksBase + done + j));
    }

    __aicore__ inline uint32_t SampleRowDirectGm(
        uint32_t outPos, IdT rid, IdT off, uint32_t deg, uint32_t numPicks, uint32_t& state)
    {
        const bool identity = selectAll_ || (!replace_ && numPicks == deg);
        LocalTensor<uint32_t> ubPicks;
        uint32_t gmPicksBase = 0;
        bool useUbPicks = false;
        if (!identity && !replace_) {
            useUbPicks = PrepareNoReplacePicks(deg, numPicks, state, ubPicks, gmPicksBase);
        }

        uint32_t done = 0;
        while (done < numPicks) {
            const uint32_t count = (numPicks - done < windowElems_) ? (numPicks - done) : windowElems_;
            LocalTensor<IdT> outR = outRBuf_.Get<IdT>();
            LocalTensor<IdT> outC = outCBuf_.Get<IdT>();
            LocalTensor<IdT> outE = outEBuf_.Get<IdT>();
            for (uint32_t j = 0; j < count; ++j) {
                const uint32_t local = ResolveDirectLocal(
                    done, j, deg, identity, useUbPicks, gmPicksBase, ubPicks, state);
                const IdT picked = off + static_cast<IdT>(local);
                outR.SetValue(j, rid);
                outC.SetValue(j, indicesGm_.GetValue(static_cast<uint64_t>(picked)));
                outE.SetValue(j, hasData_ ? dataGm_.GetValue(static_cast<uint64_t>(picked)) : picked);
            }
            CopyOutStaged(outR, outRowsGm_[outPos + done], count);
            CopyOutStaged(outC, outColsGm_[outPos + done], count);
            CopyOutStaged(outE, outDataGm_[outPos + done], count);
            done += count;
        }
        return numPicks;
    }

    __aicore__ inline void CopyOutStaged(
        LocalTensor<IdT>& staging, GlobalTensor<IdT> dst, uint32_t count)
    {
        LocalTensor<IdT> out = outQ_.AllocTensor<IdT>();
        for (uint32_t j = 0; j < count; ++j) {
            out.SetValue(j, staging.GetValue(j));
        }
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(IdT)), 0, 0, 0};
        DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
        outQ_.EnQue(out);
        LocalTensor<IdT> ready = outQ_.DeQue<IdT>();
        DataCopyPad(dst, ready, cp);
        outQ_.FreeTensor(ready);
    }

    static constexpr uint32_t kQueueDepth = 2;

    GlobalTensor<IdT> indptrGm_, indicesGm_, dataGm_, rowsGm_;
    GlobalTensor<IdT> outRowsGm_, outColsGm_, outDataGm_;
    GlobalTensor<IdT> pickScratchGm_;
    TQue<TPosition::VECIN, kQueueDepth> winIdxQ_, winDataQ_;
    TQue<TPosition::VECOUT, kQueueDepth> outQ_;
    TBuf<TPosition::VECCALC> pickBuf_;
    TBuf<TPosition::VECCALC> outRBuf_, outCBuf_, outEBuf_;
    uint32_t numRows_ = 0, numSamples_ = 0, replace_ = 0, hasData_ = 0;
    uint32_t seed_ = 0, selectAll_ = 0, numTotalRows_ = 0, outStart_ = 0;
    uint32_t rowBegin_ = 0, rowEnd_ = 0, windowElems_ = 0;
    uint32_t blockIdx_ = 0, pickStride_ = 0;
};

}  // namespace csr_per_etype
}  // namespace opsgnn

extern "C" __global__ __aicore__ void csr_row_wise_per_etype_sampling_uniform_kernel(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
    GM_ADDR outPtr, GM_ADDR outRows, GM_ADDR outCols, GM_ADDR outData,
    GM_ADDR rowSplit, GM_ADDR outStarts, GM_ADDR pickScratch,
    opsgnn::csr_per_etype::CsrRowWisePerEtypeSamplingUniformTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.dtypeBytes == 4) {
        opsgnn::csr_per_etype::KernelCsrRowWisePerEtypeSamplingUniform<int32_t> op;
        TPipe pipe;
        op.Init(indptr, indices, data, rows, outPtr, outRows, outCols, outData,
                rowSplit, outStarts, pickScratch, tiling, &pipe);
        op.Process();
    } else {
        opsgnn::csr_per_etype::KernelCsrRowWisePerEtypeSamplingUniform<int64_t> op;
        TPipe pipe;
        op.Init(indptr, indices, data, rows, outPtr, outRows, outCols, outData,
                rowSplit, outStarts, pickScratch, tiling, &pipe);
        op.Process();
    }
}

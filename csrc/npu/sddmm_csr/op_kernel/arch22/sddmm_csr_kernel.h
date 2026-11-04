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

#include "sddmm_csr_tiling.h"
#include "kernel_operator.h"

template <typename DType>
class SddmmCsrKernel {
public:
    __aicore__ inline explicit SddmmCsrKernel(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR lhsData, GM_ADDR rhsData, GM_ADDR indptrData,
                                GM_ADDR indicesData, GM_ADDR outData,
                                const __gm__ SddmmCsrTilingData* tiling)
    {
        op_ = tiling->op;
        lhsTarget_ = tiling->lhsTarget;
        rhsTarget_ = tiling->rhsTarget;
        featDim_ = tiling->featDim;
        numRows_ = tiling->numRows;
        numEdges_ = tiling->numEdges;
        batchSize_ = tiling->batchSize;
        featDimAligned_ = tiling->featDimAligned;
        featDimAlignedF32_ = tiling->featDimAlignedF32;

        if (!InitPartition(lhsData, rhsData, indptrData, indicesData, outData, tiling)) {
            return;
        }
        InitIndexEvents();

        if (op_ == SDDMM_CSR_OP_DOT) {
            InitDotBuffers();
        } else if (op_ == SDDMM_CSR_OP_COPY_LHS || op_ == SDDMM_CSR_OP_COPY_RHS) {
            InitCopyBuffers();
        } else {
            InitBinaryBuffers();
        }
    }

    __aicore__ inline void Process()
    {
        if (startEdge_ >= endEdge_) {
            return;
        }
        if (op_ == SDDMM_CSR_OP_DOT) {
            ProcessDot();
        } else if (op_ == SDDMM_CSR_OP_COPY_LHS || op_ == SDDMM_CSR_OP_COPY_RHS) {
            ProcessCopy();
        } else {
            ProcessBinary();
        }
        if (idxEventsAllocated_) {
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_S>(idxMte2S_);
            pipe_->ReleaseEventID<AscendC::HardEvent::S_MTE2>(idxSToMte2_);
        }
    }

private:
    __aicore__ inline bool UseLhs() const
    {
        return op_ != SDDMM_CSR_OP_COPY_RHS;
    }

    __aicore__ inline bool UseRhs() const
    {
        return op_ != SDDMM_CSR_OP_COPY_LHS;
    }

    __aicore__ inline uint32_t FindRow(uint32_t edge) const
    {
        uint32_t low = 0;
        uint32_t high = numRows_;
        while (low + 1 < high) {
            uint32_t mid = (low + high) / 2;
            if (static_cast<uint32_t>(indptrGm.GetValue(mid)) <= edge) {
                low = mid;
            } else {
                high = mid;
            }
        }
        return low;
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> LoadIndices(
        uint32_t target, AscendC::TBuf<AscendC::TPosition::VECCALC>& buffer,
        uint32_t edgeStart, uint32_t count, uint32_t& rowState,
        uint32_t& rowEndState)
    {
        AscendC::LocalTensor<int32_t> indices = buffer.Get<int32_t>();
        if (target == SDDMM_CSR_TARGET_DST) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(idxSToMte2_);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(idxSToMte2_);
            AscendC::DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(indices, indicesGm[edgeStart], params, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(idxMte2S_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(idxMte2S_);
            return indices;
        }
        if (target == SDDMM_CSR_TARGET_EDGE) {
            for (uint32_t i = 0; i < count; ++i) {
                indices.SetValue(i, static_cast<int32_t>(edgeStart + i));
            }
            return indices;
        }
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t edge = edgeStart + i;
            while (edge >= rowEndState) {
                ++rowState;
                rowEndState = static_cast<uint32_t>(indptrGm.GetValue(rowState + 1));
            }
            indices.SetValue(i, static_cast<int32_t>(rowState));
        }
        return indices;
    }

    __aicore__ inline bool InitPartition(GM_ADDR lhsData, GM_ADDR rhsData,
                                         GM_ADDR indptrData, GM_ADDR indicesData,
                                         GM_ADDR outData,
                                         const __gm__ SddmmCsrTilingData* tiling)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        startEdge_ = blockIdx * tiling->edgesPerCore;
        endEdge_ = (blockIdx + 1) * tiling->edgesPerCore;
        if (endEdge_ > numEdges_) {
            endEdge_ = numEdges_;
        }
        if (startEdge_ >= endEdge_) {
            return false;
        }

        if (lhsData != nullptr) {
            lhsGm.SetGlobalBuffer((__gm__ DType*)lhsData);
        }
        if (rhsData != nullptr) {
            rhsGm.SetGlobalBuffer((__gm__ DType*)rhsData);
        }
        outGm.SetGlobalBuffer((__gm__ DType*)outData);
        indptrGm.SetGlobalBuffer((__gm__ int32_t*)indptrData, numRows_ + 1);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indicesData, numEdges_);

        if (UseLhs() && lhsTarget_ == SDDMM_CSR_TARGET_SRC) {
            lhsRow_ = FindRow(startEdge_);
            lhsRowEnd_ = static_cast<uint32_t>(indptrGm.GetValue(lhsRow_ + 1));
        }
        if (UseRhs() && rhsTarget_ == SDDMM_CSR_TARGET_SRC) {
            rhsRow_ = FindRow(startEdge_);
            rhsRowEnd_ = static_cast<uint32_t>(indptrGm.GetValue(rhsRow_ + 1));
        }
        return true;
    }

    __aicore__ inline void InitIndexEvents()
    {
        pipe_->InitBuffer(lhsIdxBuf_, batchSize_ * sizeof(int32_t));
        pipe_->InitBuffer(rhsIdxBuf_, batchSize_ * sizeof(int32_t));
        if (lhsTarget_ == SDDMM_CSR_TARGET_DST ||
            rhsTarget_ == SDDMM_CSR_TARGET_DST) {
            idxMte2S_ = static_cast<int32_t>(
                pipe_->AllocEventID<AscendC::HardEvent::MTE2_S>());
            idxSToMte2_ = static_cast<int32_t>(
                pipe_->AllocEventID<AscendC::HardEvent::S_MTE2>());
            idxEventsAllocated_ = true;
        }
    }

    __aicore__ inline void InitDotBuffers()
    {
        uint32_t featBytes = featDimAligned_ * sizeof(DType);
        pipe_->InitBuffer(lhsQueue, SDDMM_CSR_BUFFER_NUM, featBytes);
        pipe_->InitBuffer(rhsQueue, SDDMM_CSR_BUFFER_NUM, featBytes);
        pipe_->InitBuffer(mulF32BatchBuf, batchSize_ * featDimAlignedF32_ * sizeof(float));
        if constexpr (sizeof(DType) != sizeof(float)) {
            pipe_->InitBuffer(lhsF32Buf, featDimAlignedF32_ * sizeof(float));
            pipe_->InitBuffer(rhsF32Buf, featDimAlignedF32_ * sizeof(float));
        }
        pipe_->InitBuffer(partSumBuf, batchSize_ * sizeof(float));
        if constexpr (sizeof(DType) != sizeof(float)) {
            pipe_->InitBuffer(outF32Buf, batchSize_ * sizeof(float));
        }
        pipe_->InitBuffer(outQueue, 1, batchSize_ * sizeof(DType));
    }

    __aicore__ inline void InitBinaryBuffers()
    {
        uint32_t featBytes = featDimAligned_ * sizeof(DType);
        pipe_->InitBuffer(lhsQueue, SDDMM_CSR_BUFFER_NUM, featBytes);
        pipe_->InitBuffer(rhsQueue, SDDMM_CSR_BUFFER_NUM, featBytes);
        pipe_->InitBuffer(outQueue, SDDMM_CSR_BUFFER_NUM, featBytes);
        if constexpr (sizeof(DType) != sizeof(float)) {
            pipe_->InitBuffer(lhsF32Buf, featDimAlignedF32_ * sizeof(float));
            pipe_->InitBuffer(rhsF32Buf, featDimAlignedF32_ * sizeof(float));
            pipe_->InitBuffer(outF32Buf, featDimAlignedF32_ * sizeof(float));
        }
    }

    __aicore__ inline void InitCopyBuffers()
    {
        pipe_->InitBuffer(featQueue, SDDMM_CSR_BUFFER_NUM, featDimAligned_ * sizeof(DType));
        evtM2M3_ = static_cast<int32_t>(
            pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
        evtM3M2_ = static_cast<int32_t>(
            pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    }

    __aicore__ inline void GatherFeature(
        AscendC::GlobalTensor<DType>& featureGm, uint32_t index,
        AscendC::TQue<AscendC::TPosition::VECIN, SDDMM_CSR_BUFFER_NUM>& queue)
    {
        AscendC::LocalTensor<DType> local = queue.AllocTensor<DType>();
        AscendC::DataCopyExtParams params{
            1, static_cast<uint32_t>(featDim_ * sizeof(DType)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DType> pad{false, 0, 0, DType(0.0f)};
        AscendC::DataCopyPad(local, featureGm[index * featDim_], params, pad);
        queue.EnQue(local);
    }

    __aicore__ inline void PrepareDotBatch(uint32_t batchStart, uint32_t actualBatch,
                                           AscendC::LocalTensor<int32_t>& lhsIdx,
                                           AscendC::LocalTensor<int32_t>& rhsIdx,
                                           AscendC::LocalTensor<DType>& lhsCur,
                                           AscendC::LocalTensor<DType>& rhsCur,
                                           AscendC::LocalTensor<float>& mulBatch)
    {
        lhsIdx = LoadIndices(lhsTarget_, lhsIdxBuf_, batchStart, actualBatch,
                             lhsRow_, lhsRowEnd_);
        rhsIdx = LoadIndices(rhsTarget_, rhsIdxBuf_, batchStart, actualBatch,
                             rhsRow_, rhsRowEnd_);
        GatherFeature(lhsGm, static_cast<uint32_t>(lhsIdx.GetValue(0)), lhsQueue);
        GatherFeature(rhsGm, static_cast<uint32_t>(rhsIdx.GetValue(0)), rhsQueue);
        lhsCur = lhsQueue.DeQue<DType>();
        rhsCur = rhsQueue.DeQue<DType>();
        mulBatch = mulF32BatchBuf.Get<float>();
    }

    __aicore__ inline void PrefetchDotFeatures(uint32_t index, uint32_t actualBatch,
                                               AscendC::LocalTensor<int32_t>& lhsIdx,
                                               AscendC::LocalTensor<int32_t>& rhsIdx)
    {
        if (index + 1 < actualBatch) {
            GatherFeature(lhsGm, static_cast<uint32_t>(lhsIdx.GetValue(index + 1)), lhsQueue);
            GatherFeature(rhsGm, static_cast<uint32_t>(rhsIdx.GetValue(index + 1)), rhsQueue);
        }
    }

    __aicore__ inline void ComputeDotProducts(uint32_t actualBatch,
                                              AscendC::LocalTensor<int32_t>& lhsIdx,
                                              AscendC::LocalTensor<int32_t>& rhsIdx,
                                              AscendC::LocalTensor<DType>& lhsCur,
                                              AscendC::LocalTensor<DType>& rhsCur,
                                              AscendC::LocalTensor<float>& mulBatch)
    {
        for (uint32_t i = 0; i < actualBatch; ++i) {
            PrefetchDotFeatures(i, actualBatch, lhsIdx, rhsIdx);
            if constexpr (sizeof(DType) == sizeof(float)) {
                AscendC::LocalTensor<float> mulDst = mulBatch[i * featDimAlignedF32_];
                AscendC::Mul<float>(mulDst, lhsCur, rhsCur, featDim_);
            } else {
                AscendC::LocalTensor<float> lhsF32 = lhsF32Buf.Get<float>();
                AscendC::LocalTensor<float> rhsF32 = rhsF32Buf.Get<float>();
                AscendC::Cast<float, DType>(lhsF32, lhsCur,
                                            AscendC::RoundMode::CAST_NONE, featDim_);
                AscendC::Cast<float, DType>(rhsF32, rhsCur,
                                            AscendC::RoundMode::CAST_NONE, featDim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::LocalTensor<float> mulDst = mulBatch[i * featDimAlignedF32_];
                AscendC::Mul<float>(mulDst, lhsF32, rhsF32, featDim_);
            }
            lhsQueue.FreeTensor(lhsCur);
            rhsQueue.FreeTensor(rhsCur);
            if (i + 1 < actualBatch) {
                lhsCur = lhsQueue.DeQue<DType>();
                rhsCur = rhsQueue.DeQue<DType>();
            }
        }
    }

    __aicore__ inline void CopyDotOutput(uint32_t batchStart, uint32_t actualBatch,
                                         AscendC::LocalTensor<DType>& outLocal)
    {
        outQueue.EnQue(outLocal);
        AscendC::LocalTensor<DType> outReady = outQueue.DeQue<DType>();
        AscendC::DataCopyExtParams outParams{
            1, static_cast<uint32_t>(actualBatch * sizeof(DType)), 0, 0, 0};
        AscendC::DataCopyPad(outGm[batchStart], outReady, outParams);
        outQueue.FreeTensor(outReady);
    }

    __aicore__ inline void WriteDotOutput(uint32_t batchStart, uint32_t actualBatch,
                                          AscendC::LocalTensor<float>& mulBatch)
    {
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (sizeof(DType) == sizeof(float)) {
            AscendC::LocalTensor<DType> outLocal = outQueue.AllocTensor<DType>();
            BatchReduceSum(outLocal, mulBatch, actualBatch);
            CopyDotOutput(batchStart, actualBatch, outLocal);
        } else {
            AscendC::LocalTensor<float> outF32 = outF32Buf.Get<float>();
            BatchReduceSum(outF32, mulBatch, actualBatch);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<DType> outLocal = outQueue.AllocTensor<DType>();
            AscendC::Cast<DType, float>(outLocal, outF32,
                                        AscendC::RoundMode::CAST_ROUND, actualBatch);
            CopyDotOutput(batchStart, actualBatch, outLocal);
        }
    }

    __aicore__ inline void ProcessDot()
    {
        for (uint32_t batchStart = startEdge_; batchStart < endEdge_; batchStart += batchSize_) {
            uint32_t actualBatch = batchSize_;
            if (batchStart + actualBatch > endEdge_) {
                actualBatch = endEdge_ - batchStart;
            }

            AscendC::LocalTensor<int32_t> lhsIdx;
            AscendC::LocalTensor<int32_t> rhsIdx;
            AscendC::LocalTensor<DType> lhsCur;
            AscendC::LocalTensor<DType> rhsCur;
            AscendC::LocalTensor<float> mulBatch;
            PrepareDotBatch(batchStart, actualBatch, lhsIdx, rhsIdx,
                            lhsCur, rhsCur, mulBatch);
            ComputeDotProducts(actualBatch, lhsIdx, rhsIdx, lhsCur, rhsCur, mulBatch);
            WriteDotOutput(batchStart, actualBatch, mulBatch);
        }
    }

    __aicore__ inline void BatchReduceSum(AscendC::LocalTensor<float>& dst,
                                          AscendC::LocalTensor<float>& src,
                                          uint32_t batchCount)
    {
        if (featDim_ <= SDDMM_CSR_MAX_REDUCE_MASK_FLOAT) {
            AscendC::WholeReduceSum<float, true>(
                dst, src, static_cast<int32_t>(featDim_),
                static_cast<int32_t>(batchCount), 1, 1,
                static_cast<int32_t>(featDimAlignedF32_ / 8));
            return;
        }
        uint32_t offset = 0;
        uint32_t remaining = featDim_;
        bool firstChunk = true;
        while (remaining > 0) {
            uint32_t chunkLen = (remaining > SDDMM_CSR_MAX_REDUCE_MASK_FLOAT)
                                    ? SDDMM_CSR_MAX_REDUCE_MASK_FLOAT : remaining;
            AscendC::LocalTensor<float> srcChunk = src[offset];
            if (firstChunk) {
                AscendC::WholeReduceSum<float, true>(
                    dst, srcChunk, static_cast<int32_t>(chunkLen),
                    static_cast<int32_t>(batchCount), 1, 1,
                    static_cast<int32_t>(featDimAlignedF32_ / 8));
                firstChunk = false;
            } else {
                AscendC::LocalTensor<float> partSum = partSumBuf.Get<float>();
                AscendC::WholeReduceSum<float, true>(
                    partSum, srcChunk, static_cast<int32_t>(chunkLen),
                    static_cast<int32_t>(batchCount), 1, 1,
                    static_cast<int32_t>(featDimAlignedF32_ / 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add<float>(dst, dst, partSum, batchCount);
                AscendC::PipeBarrier<PIPE_V>();
            }
            offset += chunkLen;
            remaining -= chunkLen;
        }
    }

    template <typename T>
    __aicore__ inline void DispatchBinary(AscendC::LocalTensor<T>& dst,
                                          const AscendC::LocalTensor<T>& lhs,
                                          const AscendC::LocalTensor<T>& rhs,
                                          uint32_t count)
    {
        switch (op_) {
            case SDDMM_CSR_OP_ADD: AscendC::Add<T>(dst, lhs, rhs, count); break;
            case SDDMM_CSR_OP_SUB: AscendC::Sub<T>(dst, lhs, rhs, count); break;
            case SDDMM_CSR_OP_MUL: AscendC::Mul<T>(dst, lhs, rhs, count); break;
            case SDDMM_CSR_OP_DIV: AscendC::Div<T>(dst, lhs, rhs, count); break;
            default: break;
        }
    }

    __aicore__ inline void ProcessBinary()
    {
        AscendC::DataCopyExtParams rowParams{
            1, static_cast<uint32_t>(featDim_ * sizeof(DType)), 0, 0, 0};
        for (uint32_t batchStart = startEdge_; batchStart < endEdge_; batchStart += batchSize_) {
            uint32_t actualBatch = batchSize_;
            if (batchStart + actualBatch > endEdge_) {
                actualBatch = endEdge_ - batchStart;
            }

            AscendC::LocalTensor<int32_t> lhsIdx = LoadIndices(
                lhsTarget_, lhsIdxBuf_, batchStart, actualBatch, lhsRow_, lhsRowEnd_);
            AscendC::LocalTensor<int32_t> rhsIdx = LoadIndices(
                rhsTarget_, rhsIdxBuf_, batchStart, actualBatch, rhsRow_, rhsRowEnd_);
            for (uint32_t i = 0; i < actualBatch; ++i) {
                uint32_t edge = batchStart + i;
                uint32_t lhsIndex = static_cast<uint32_t>(lhsIdx.GetValue(i));
                uint32_t rhsIndex = static_cast<uint32_t>(rhsIdx.GetValue(i));
                GatherFeature(lhsGm, lhsIndex, lhsQueue);
                GatherFeature(rhsGm, rhsIndex, rhsQueue);
                AscendC::LocalTensor<DType> lhsRow = lhsQueue.DeQue<DType>();
                AscendC::LocalTensor<DType> rhsRow = rhsQueue.DeQue<DType>();
                AscendC::LocalTensor<DType> outLocal = outQueue.AllocTensor<DType>();

                if constexpr (sizeof(DType) == sizeof(float)) {
                    DispatchBinary<DType>(outLocal, lhsRow, rhsRow, featDim_);
                } else {
                    AscendC::LocalTensor<float> lhsF32 = lhsF32Buf.Get<float>();
                    AscendC::LocalTensor<float> rhsF32 = rhsF32Buf.Get<float>();
                    AscendC::LocalTensor<float> outF32 = outF32Buf.Get<float>();
                    AscendC::Cast<float, DType>(lhsF32, lhsRow,
                                                AscendC::RoundMode::CAST_NONE, featDim_);
                    AscendC::Cast<float, DType>(rhsF32, rhsRow,
                                                AscendC::RoundMode::CAST_NONE, featDim_);
                    AscendC::PipeBarrier<PIPE_V>();
                    DispatchBinary<float>(outF32, lhsF32, rhsF32, featDim_);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast<DType, float>(outLocal, outF32,
                                                AscendC::RoundMode::CAST_ROUND, featDim_);
                }

                outQueue.EnQue(outLocal);
                AscendC::LocalTensor<DType> outReady = outQueue.DeQue<DType>();
                AscendC::DataCopyPad(outGm[edge * featDim_], outReady, rowParams);
                lhsQueue.FreeTensor(lhsRow);
                rhsQueue.FreeTensor(rhsRow);
                outQueue.FreeTensor(outReady);
            }
        }
    }

    template <bool UseLhs>
    __aicore__ inline void ProcessCopyImpl()
    {
        AscendC::GlobalTensor<DType>& featureGm = UseLhs ? lhsGm : rhsGm;
        uint32_t target = UseLhs ? lhsTarget_ : rhsTarget_;
        uint32_t& rowState = UseLhs ? lhsRow_ : rhsRow_;
        uint32_t& rowEndState = UseLhs ? lhsRowEnd_ : rhsRowEnd_;
        AscendC::TBuf<AscendC::TPosition::VECCALC>& idxBuffer =
            UseLhs ? lhsIdxBuf_ : rhsIdxBuf_;

        AscendC::DataCopyExtParams featParams{
            1, static_cast<uint32_t>(featDim_ * sizeof(DType)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DType> featPad{false, 0, 0, DType(0.0f)};

        for (uint32_t batchStart = startEdge_; batchStart < endEdge_; batchStart += batchSize_) {
            uint32_t actualBatch = batchSize_;
            if (batchStart + actualBatch > endEdge_) {
                actualBatch = endEdge_ - batchStart;
            }
            AscendC::LocalTensor<int32_t> edgeIdx = LoadIndices(
                target, idxBuffer, batchStart, actualBatch, rowState, rowEndState);
            for (uint32_t i = 0; i < actualBatch; ++i) {
                uint32_t edge = batchStart + i;
                uint32_t index = static_cast<uint32_t>(edgeIdx.GetValue(i));
                AscendC::LocalTensor<DType> local = featQueue.AllocTensor<DType>();
                AscendC::DataCopyPad(local, featureGm[index * featDim_],
                                     featParams, featPad);
                featQueue.EnQue(local);
                AscendC::LocalTensor<DType> ready = featQueue.DeQue<DType>();
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtM2M3_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtM2M3_);
                AscendC::DataCopyPad(outGm[edge * featDim_], ready, featParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtM3M2_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtM3M2_);
                featQueue.FreeTensor(ready);
            }
        }
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtM2M3_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evtM3M2_);
    }

    __aicore__ inline void ProcessCopy()
    {
        if (op_ == SDDMM_CSR_OP_COPY_LHS) {
            ProcessCopyImpl<true>();
        } else {
            ProcessCopyImpl<false>();
        }
    }

private:
    AscendC::TPipe* pipe_;

    AscendC::GlobalTensor<DType> lhsGm;
    AscendC::GlobalTensor<DType> rhsGm;
    AscendC::GlobalTensor<DType> outGm;
    AscendC::GlobalTensor<int32_t> indptrGm;
    AscendC::GlobalTensor<int32_t> indicesGm;

    AscendC::TQue<AscendC::TPosition::VECIN, SDDMM_CSR_BUFFER_NUM> lhsQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, SDDMM_CSR_BUFFER_NUM> rhsQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, SDDMM_CSR_BUFFER_NUM> featQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, SDDMM_CSR_BUFFER_NUM> outQueue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> lhsIdxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rhsIdxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mulF32BatchBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partSumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outF32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lhsF32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rhsF32Buf;

    uint32_t op_ = SDDMM_CSR_OP_DOT;
    uint32_t lhsTarget_ = SDDMM_CSR_TARGET_SRC;
    uint32_t rhsTarget_ = SDDMM_CSR_TARGET_DST;
    uint32_t featDim_ = 0;
    uint32_t numRows_ = 0;
    uint32_t numEdges_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t featDimAligned_ = 0;
    uint32_t featDimAlignedF32_ = 0;
    uint32_t startEdge_ = 0;
    uint32_t endEdge_ = 0;
    uint32_t lhsRow_ = 0;
    uint32_t lhsRowEnd_ = 0;
    uint32_t rhsRow_ = 0;
    uint32_t rhsRowEnd_ = 0;
    int32_t evtM2M3_ = 0;
    int32_t evtM3M2_ = 0;
    int32_t idxMte2S_ = 0;
    int32_t idxSToMte2_ = 0;
    bool idxEventsAllocated_ = false;
};

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

#include "unified_spmm_tiling.h"
#include <cstdint>
#include "kernel_operator.h"

constexpr uint32_t UNIFIED_BUFFER_NUM = 2;
constexpr uint32_t UNIFIED_UB_RESERVED = 2048;
constexpr uint32_t UNIFIED_ALIGN_BYTES = 32;

template <typename DType>
class UnifiedSpmmKernel {
public:
    __aicore__ inline void Init(GM_ADDR featureData, GM_ADDR outputData,
        GM_ADDR indptrData, GM_ADDR indicesData, GM_ADDR rowSplitData,
        const __gm__ UnifiedSpmmTilingData* tiling, AscendC::TPipe* pipe)
    {
        M = tiling->numDstRows;
        K = tiling->numFeatureRows;
        N = tiling->featureDim;
        nnz = tiling->nonZeroCount;
        reduce = tiling->reduce;
        message = tiling->message;
        hasNan = tiling->hasNan;
        batchCount = tiling->batchCount;
        startRow = 0;
        localRowCount = 0;

        uint32_t ubBytes = tiling->ubBytes;
        if (IsInvalidTiling(ubBytes) || !InitWorkRange(rowSplitData)) return;
        InitGlobalBuffers(featureData, outputData, indptrData, indicesData);
        InitLocalBuffers(ubBytes, pipe);
    }

    __aicore__ inline void Process()
    {
        uint32_t endRow = startRow + localRowCount;
        for (uint32_t row = startRow; row < endRow; ++row) {
            uint32_t edgeBegin = indptrGm.GetValue(row);
            uint32_t edgeEnd = indptrGm.GetValue(row + 1);
            for (uint32_t batch = 0; batch < batchCount; ++batch) {
                ProcessRow(row, edgeBegin, edgeEnd, batch);
            }
        }
    }

private:
    __aicore__ inline bool IsInvalidTiling(uint32_t ubBytes) const
    {
        if (ubBytes <= UNIFIED_UB_RESERVED || N == 0 || batchCount == 0) {
            return true;
        }
        uint32_t maxAlignedN =
            (0xffffffffU - (UNIFIED_ALIGN_BYTES - 1)) / sizeof(DType);
        return N > maxAlignedN || batchCount > 0xffffffffU / N ||
               K > 0xffffffffU / N || M > 0xffffffffU / N ||
               M == 0xffffffffU;
    }

    __aicore__ inline bool InitWorkRange(GM_ADDR rowSplitData)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        rowSplitGm.SetGlobalBuffer((__gm__ uint32_t*)rowSplitData, blockNum + 1);
        startRow = rowSplitGm.GetValue(blockIdx);
        uint32_t endRow = rowSplitGm.GetValue(blockIdx + 1);
        if (endRow < startRow || endRow > M) return false;
        localRowCount = endRow - startRow;
        return true;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR featureData,
        GM_ADDR outputData, GM_ADDR indptrData, GM_ADDR indicesData)
    {
        featureGm.SetGlobalBuffer((__gm__ DType*)featureData,
                                  K * batchCount * N);
        outputGm.SetGlobalBuffer((__gm__ DType*)outputData,
                                 M * batchCount * N);
        indptrGm.SetGlobalBuffer((__gm__ uint32_t*)indptrData, M + 1);
        indicesGm.SetGlobalBuffer((__gm__ uint32_t*)indicesData, nnz);
        if constexpr (sizeof(DType) == sizeof(uint16_t)) {
            featureBits16Gm.SetGlobalBuffer((__gm__ uint16_t*)featureData,
                                            K * batchCount * N);
        } else {
            featureBits32Gm.SetGlobalBuffer((__gm__ uint32_t*)featureData,
                                            K * batchCount * N);
        }
    }

    __aicore__ inline void InitLocalBuffers(uint32_t ubBytes,
                                             AscendC::TPipe* pipe)
    {
        rowBytes = N * sizeof(DType);
        rowAlignedBytes = (rowBytes + UNIFIED_ALIGN_BYTES - 1) /
                          UNIFIED_ALIGN_BYTES * UNIFIED_ALIGN_BYTES;
        rowAlignedElems = rowAlignedBytes / sizeof(DType);
        uint32_t fixedBytes = UNIFIED_BUFFER_NUM * rowAlignedBytes;
        if constexpr (sizeof(DType) == sizeof(uint16_t)) {
            rowF32Bytes = rowAlignedElems * sizeof(float);
            fixedBytes += 2 * rowF32Bytes;
        }
        uint32_t available = ubBytes - UNIFIED_UB_RESERVED;
        uint32_t remaining = available > fixedBytes ? available - fixedBytes : 0;
        batchSize = remaining / (UNIFIED_BUFFER_NUM * rowAlignedBytes);
        if (batchSize == 0) {
            localRowCount = 0;
            return;
        }
        pipe->InitBuffer(accumQueue, UNIFIED_BUFFER_NUM, rowAlignedBytes);
        pipe->InitBuffer(featureQueue, UNIFIED_BUFFER_NUM, batchSize * rowAlignedBytes);
        if constexpr (sizeof(DType) == sizeof(uint16_t)) {
            pipe->InitBuffer(featureF32Buffer, rowF32Bytes);
            pipe->InitBuffer(accumF32Buffer, rowF32Bytes);
        }
    }

    __aicore__ inline void ProcessRow(uint32_t row, uint32_t edgeBegin,
                                      uint32_t edgeEnd, uint32_t batch)
    {
        AscendC::LocalTensor<DType> accum = accumQueue.AllocTensor<DType>();
        uint32_t degree = edgeEnd - edgeBegin;
        if (degree == 0) {
            AscendC::Duplicate<DType>(accum, DType(0.0f), rowAlignedElems);
        } else if constexpr (sizeof(DType) == sizeof(uint16_t)) {
            ProcessFp16(accum, edgeBegin, edgeEnd, batch);
        } else {
            InitAccumulator(accum);
            ProcessBatches(accum, edgeBegin, edgeEnd, batch);
        }
        RepairNan(accum, edgeBegin, edgeEnd, batch);
        accumQueue.EnQue(accum);
        CopyOut(row, batch);
    }

    __aicore__ inline void ProcessFp16(AscendC::LocalTensor<DType>& accum,
                                       uint32_t edgeBegin, uint32_t edgeEnd,
                                       uint32_t batch)
    {
        AscendC::LocalTensor<float> accumF32 = accumF32Buffer.Get<float>();
        if (reduce == UNIFIED_SPMM_REDUCE_MAX) {
            AscendC::Duplicate<float>(accumF32, -__builtin_huge_valf(), rowAlignedElems);
        } else if (reduce == UNIFIED_SPMM_REDUCE_MIN) {
            AscendC::Duplicate<float>(accumF32, __builtin_huge_valf(), rowAlignedElems);
        } else {
            AscendC::Duplicate<float>(accumF32, 0.0f, rowAlignedElems);
        }

        uint32_t chunkCount = (edgeEnd - edgeBegin) / batchSize +
                              ((edgeEnd - edgeBegin) % batchSize != 0);
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            uint32_t begin = edgeBegin + chunk * batchSize;
            uint32_t count = batchSize > edgeEnd - begin ? edgeEnd - begin : batchSize;
            CopyInBatch(begin, count, batch);
            AscendC::LocalTensor<DType> features = featureQueue.DeQue<DType>();
            AscendC::LocalTensor<float> featureF32 = featureF32Buffer.Get<float>();
            for (uint32_t i = 0; i < count; ++i) {
                AscendC::Cast<float, DType>(featureF32, features[i * rowAlignedElems],
                                            AscendC::RoundMode::CAST_NONE, rowAlignedElems);
                ReduceF32(accumF32, featureF32);
            }
            featureQueue.FreeTensor(features);
        }
        AscendC::Cast<DType, float>(accum, accumF32,
                                    AscendC::RoundMode::CAST_ROUND, rowAlignedElems);
    }

    __aicore__ inline void InitAccumulator(AscendC::LocalTensor<DType>& accum)
    {
        if (reduce == UNIFIED_SPMM_REDUCE_MAX) {
            AscendC::Duplicate<float>(accum, -__builtin_huge_valf(), rowAlignedElems);
        } else if (reduce == UNIFIED_SPMM_REDUCE_MIN) {
            AscendC::Duplicate<float>(accum, __builtin_huge_valf(), rowAlignedElems);
        } else {
            AscendC::Duplicate<float>(accum, 0.0f, rowAlignedElems);
        }
    }

    __aicore__ inline void ProcessBatches(AscendC::LocalTensor<DType>& accum,
                                           uint32_t edgeBegin, uint32_t edgeEnd,
                                           uint32_t batch)
    {
        uint32_t chunkCount = (edgeEnd - edgeBegin) / batchSize +
                              ((edgeEnd - edgeBegin) % batchSize != 0);
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            uint32_t begin = edgeBegin + chunk * batchSize;
            uint32_t count = batchSize > edgeEnd - begin ? edgeEnd - begin : batchSize;
            CopyInBatch(begin, count, batch);
            AscendC::LocalTensor<DType> features = featureQueue.DeQue<DType>();
            for (uint32_t i = 0; i < count; ++i) {
                AscendC::LocalTensor<DType> feature = features[i * rowAlignedElems];
                if (reduce == UNIFIED_SPMM_REDUCE_MAX) {
                    AscendC::Max(accum, accum, feature, rowAlignedElems);
                } else if (reduce == UNIFIED_SPMM_REDUCE_MIN) {
                    AscendC::Min(accum, accum, feature, rowAlignedElems);
                } else {
                    AscendC::Add(accum, accum, feature, rowAlignedElems);
                }
            }
            featureQueue.FreeTensor(features);
        }
    }

    __aicore__ inline void ReduceF32(AscendC::LocalTensor<float>& accum,
                                     AscendC::LocalTensor<float>& feature)
    {
        if (reduce == UNIFIED_SPMM_REDUCE_MAX) {
            AscendC::Max(accum, accum, feature, rowAlignedElems);
        } else if (reduce == UNIFIED_SPMM_REDUCE_MIN) {
            AscendC::Min(accum, accum, feature, rowAlignedElems);
        } else {
            AscendC::Add(accum, accum, feature, rowAlignedElems);
        }
    }

    __aicore__ inline void CopyInBatch(uint32_t edgeBegin, uint32_t count,
                                       uint32_t batch)
    {
        AscendC::LocalTensor<DType> local = featureQueue.AllocTensor<DType>();
        AscendC::DataCopyExtParams copy = {1, rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DType> pad = {
            true, 0, static_cast<uint8_t>(rowAlignedElems - N), DType(0.0f)};
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t edge = edgeBegin + i;
            uint32_t source =
                message == UNIFIED_SPMM_COPY_RHS ? edge : indicesGm.GetValue(edge);
            AscendC::DataCopyPad<DType>(local[i * rowAlignedElems],
                                        featureGm[(source * batchCount + batch) * N],
                                        copy, pad);
        }
        featureQueue.EnQue(local);
    }

    __aicore__ inline void RepairNan(AscendC::LocalTensor<DType>& accum,
                                      uint32_t edgeBegin, uint32_t edgeEnd,
                                      uint32_t batch)
    {
        if (!hasNan || edgeBegin == edgeEnd) return;
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t edge = edgeBegin; edge < edgeEnd; ++edge) {
            uint32_t source =
                message == UNIFIED_SPMM_COPY_RHS ? edge : indicesGm.GetValue(edge);
            for (uint32_t feature = 0; feature < N; ++feature) {
                uint32_t offset = (source * batchCount + batch) * N + feature;
                bool isNan;
                if constexpr (sizeof(DType) == sizeof(uint16_t)) {
                    uint16_t bits = featureBits16Gm.GetValue(offset);
                    isNan = (bits & 0x7c00U) == 0x7c00U && (bits & 0x03ffU) != 0;
                } else {
                    uint32_t bits = featureBits32Gm.GetValue(offset);
                    isNan = (bits & 0x7f800000U) == 0x7f800000U &&
                            (bits & 0x007fffffU) != 0;
                }
                if (isNan) accum.SetValue(feature, featureGm.GetValue(offset));
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t batch)
    {
        AscendC::LocalTensor<DType> accum = accumQueue.DeQue<DType>();
        AscendC::DataCopyExtParams copy = {1, rowBytes, 0, 0, 0};
        AscendC::DataCopyPad<DType>(outputGm[(row * batchCount + batch) * N],
                                    accum, copy);
        accumQueue.FreeTensor(accum);
    }

    uint32_t M, K, N, nnz, reduce, message, hasNan, batchCount;
    uint32_t startRow, localRowCount, rowBytes, rowAlignedBytes, rowAlignedElems;
    uint32_t rowF32Bytes, batchSize;
    AscendC::TQue<AscendC::TPosition::VECOUT, UNIFIED_BUFFER_NUM> accumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, UNIFIED_BUFFER_NUM> featureQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> featureF32Buffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumF32Buffer;
    AscendC::GlobalTensor<DType> featureGm, outputGm;
    AscendC::GlobalTensor<uint16_t> featureBits16Gm;
    AscendC::GlobalTensor<uint32_t> featureBits32Gm;
    AscendC::GlobalTensor<uint32_t> indptrGm, indicesGm, rowSplitGm;
};

extern "C" __global__ __aicore__ void unified_spmm(GM_ADDR featureData,
    GM_ADDR outputData, GM_ADDR indptrData, GM_ADDR indicesData,
    GM_ADDR rowSplitData, GM_ADDR tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    const __gm__ UnifiedSpmmTilingData* tiling =
        (const __gm__ UnifiedSpmmTilingData*)tilingData;
    if (tiling->dtype == UNIFIED_SPMM_DTYPE_FP32) {
        UnifiedSpmmKernel<float> processor;
        processor.Init(featureData, outputData, indptrData, indicesData,
                       rowSplitData, tiling, &pipe);
        processor.Process();
    } else {
        UnifiedSpmmKernel<half> processor;
        processor.Init(featureData, outputData, indptrData, indicesData,
                       rowSplitData, tiling, &pipe);
        processor.Process();
    }
}

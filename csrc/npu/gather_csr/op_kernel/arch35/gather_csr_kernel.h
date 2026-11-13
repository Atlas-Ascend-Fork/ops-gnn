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

#include "gather_csr_tiling.h"

using namespace AscendC;

constexpr uint32_t GATHER_CSR_REPEAT_BUFFER_BYTES = 64 * 1024;

class GatherCsrKernel {
public:
    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ int64_t* indptr, __gm__ uint8_t* out,
                                const GatherCsrTilingData& tiling, TPipe* pipe)
    {
        batchCount_ = tiling.batchCount;
        segmentCount_ = tiling.segmentCount;
        outputRows_ = tiling.outputRows;
        featureBytes_ = tiling.featureBytes;
        totalSegments_ = tiling.totalSegments;
        tileBytes_ = tiling.tileBytes;
        activeCoreNum_ = tiling.activeCoreNum;
        scheduleMode_ = tiling.scheduleMode;

        const uint64_t srcBytes = batchCount_ * segmentCount_ * featureBytes_;
        const uint64_t indptrElements = batchCount_ * (segmentCount_ + 1);
        const uint64_t outBytes = batchCount_ * outputRows_ * featureBytes_;
        srcGm_.SetGlobalBuffer(src, srcBytes);
        indptrGm_.SetGlobalBuffer(indptr, indptrElements);
        outGm_.SetGlobalBuffer(out, outBytes);

        const uint64_t totalJobs = scheduleMode_ == GATHER_CSR_OUTPUT_MAJOR
            ? batchCount_ * outputRows_ : totalSegments_;
        const uint64_t jobsPerCore = totalJobs / activeCoreNum_ +
            static_cast<uint64_t>(totalJobs % activeCoreNum_ != 0);
        jobBegin_ = GetBlockIdx() * jobsPerCore;
        jobEnd_ = jobBegin_ + jobsPerCore;
        if (jobEnd_ > totalJobs) {
            jobEnd_ = totalJobs;
        }

        pipe->InitBuffer(featureBuffer_, tileBytes_);
        pipe->InitBuffer(repeatBuffer_, GATHER_CSR_REPEAT_BUFFER_BYTES);
        mte2ToV_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        mte2ToMte3_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
        vToMte3_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        mte3ToV_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        mte3ToMte2_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
    }

    __aicore__ inline void Process()
    {
        LocalTensor<uint8_t> featureLocal = featureBuffer_.Get<uint8_t>();
        LocalTensor<uint8_t> repeatLocal = repeatBuffer_.Get<uint8_t>();
        if (scheduleMode_ == GATHER_CSR_OUTPUT_MAJOR) {
            ProcessOutputMajor(featureLocal, repeatLocal);
            return;
        }
        ProcessSegmentMajor(featureLocal, repeatLocal);
    }

private:
    __aicore__ inline void ProcessSegmentMajor(const LocalTensor<uint8_t>& featureLocal,
                                                const LocalTensor<uint8_t>& repeatLocal)
    {
        for (uint64_t job = jobBegin_; job < jobEnd_; ++job) {
            const uint64_t batch = job / segmentCount_;
            const uint64_t segment = job - batch * segmentCount_;
            const uint64_t ptrBase = batch * (segmentCount_ + 1);
            const int64_t rowStart = indptrGm_.GetValue(ptrBase + segment);
            const int64_t rowEnd = indptrGm_.GetValue(ptrBase + segment + 1);
            if (rowStart >= rowEnd) {
                continue;
            }
            CopySegmentRows(featureLocal, repeatLocal, batch, segment, rowStart, rowEnd);
        }
    }

    __aicore__ inline void ProcessOutputMajor(const LocalTensor<uint8_t>& featureLocal,
                                               const LocalTensor<uint8_t>& repeatLocal)
    {
        uint64_t globalRow = jobBegin_;
        while (globalRow < jobEnd_) {
            const uint64_t batch = globalRow / outputRows_;
            int64_t row = static_cast<int64_t>(globalRow - batch * outputRows_);
            const int64_t localBatchEnd = static_cast<int64_t>(
                (jobEnd_ < (batch + 1) * outputRows_ ? jobEnd_ : (batch + 1) * outputRows_) -
                batch * outputRows_);
            const uint64_t ptrBase = batch * (segmentCount_ + 1);
            const int64_t firstRow = indptrGm_.GetValue(ptrBase);
            if (row < firstRow) {
                row = firstRow < localBatchEnd ? firstRow : localBatchEnd;
                globalRow = batch * outputRows_ + static_cast<uint64_t>(row);
                continue;
            }

            // Locate the first non-empty segment whose end pointer is after this row.
            uint64_t low = 0;
            uint64_t high = segmentCount_;
            while (low < high) {
                const uint64_t mid = low + (high - low) / 2;
                if (indptrGm_.GetValue(ptrBase + mid + 1) <= row) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            if (low >= segmentCount_) {
                globalRow = (batch + 1) * outputRows_;
                continue;
            }
            const int64_t segmentEnd = indptrGm_.GetValue(ptrBase + low + 1);
            const int64_t runEnd = segmentEnd < localBatchEnd ? segmentEnd : localBatchEnd;
            CopySegmentRows(featureLocal, repeatLocal, batch, low, row, runEnd);
            globalRow = batch * outputRows_ + static_cast<uint64_t>(runEnd);
        }
    }

    __aicore__ inline void CopySegmentRows(const LocalTensor<uint8_t>& featureLocal,
                                            const LocalTensor<uint8_t>& repeatLocal,
                                            uint64_t batch, uint64_t segment,
                                            int64_t rowStart, int64_t rowEnd)
    {
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToV_);

        const uint64_t srcBase = (batch * segmentCount_ + segment) * featureBytes_;
        const uint64_t batchOutBase = batch * outputRows_ * featureBytes_;
        for (uint64_t featureOffset = 0; featureOffset < featureBytes_; featureOffset += tileBytes_) {
            const uint64_t remaining = featureBytes_ - featureOffset;
            const uint32_t copyBytes = static_cast<uint32_t>(remaining < tileBytes_ ? remaining : tileBytes_);
            DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
            DataCopyPad(featureLocal, srcGm_[srcBase + featureOffset], copyParams, padParams);

            const uint32_t dataBlockBytes = static_cast<uint32_t>(GetDataBlockSizeInBytes());
            if (copyBytes == featureBytes_ && copyBytes % dataBlockBytes == 0 &&
                copyBytes <= GATHER_CSR_REPEAT_BUFFER_BYTES) {
                SetFlag<HardEvent::MTE2_V>(mte2ToV_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
                CopyRowsBatched(featureLocal, repeatLocal, batchOutBase,
                                copyBytes, rowStart, rowEnd);
            } else {
                SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
                WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
                for (int64_t row = rowStart; row < rowEnd; ++row) {
                    const uint64_t outOffset = batchOutBase +
                        static_cast<uint64_t>(row) * featureBytes_ + featureOffset;
                    DataCopyPad(outGm_[outOffset], featureLocal, copyParams);
                }
                SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
                WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            }
        }
    }

    __aicore__ inline void CopyRowsBatched(const LocalTensor<uint8_t>& featureLocal,
                                           const LocalTensor<uint8_t>& repeatLocal,
                                           uint64_t batchOutBase, uint32_t copyBytes,
                                           int64_t rowStart, int64_t rowEnd)
    {
        if (copyBytes == 0) {
            return;
        }
        const uint32_t maxRows = GATHER_CSR_REPEAT_BUFFER_BYTES / copyBytes;
        const uint64_t segmentRows = static_cast<uint64_t>(rowEnd - rowStart);
        const uint32_t preparedRows = segmentRows < maxRows ? static_cast<uint32_t>(segmentRows) : maxRows;
        LocalTensor<uint32_t> featureWords = featureLocal.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> repeatWords = repeatLocal.ReinterpretCast<uint32_t>();
        const uint32_t wordsPerRow = copyBytes / sizeof(uint32_t);
        // Both tensors are in UB: keep this on the vector pipe, not UB-to-L1 MTE.
        Copy(repeatWords, featureWords, wordsPerRow);
        PipeBarrier<PIPE_V>();
        // Double the prepared rows each step to avoid one UB copy per output row.
        uint32_t filledRows = 1;
        while (filledRows < preparedRows) {
            const uint32_t copyRows = (preparedRows - filledRows) < filledRows
                ? (preparedRows - filledRows) : filledRows;
            Copy(repeatWords[filledRows * wordsPerRow], repeatWords, copyRows * wordsPerRow);
            PipeBarrier<PIPE_V>();
            filledRows += copyRows;
        }
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);

        int64_t row = rowStart;
        while (row < rowEnd) {
            const uint64_t rowsRemaining = static_cast<uint64_t>(rowEnd - row);
            const uint32_t rowCount = rowsRemaining < preparedRows
                ? static_cast<uint32_t>(rowsRemaining) : preparedRows;
            const uint64_t outOffset = batchOutBase + static_cast<uint64_t>(row) * featureBytes_;
            DataCopyExtParams outputParams{1, rowCount * copyBytes, 0, 0, 0};
            DataCopyPad(outGm_[outOffset], repeatLocal, outputParams);
            row += rowCount;
        }
    }

    TBuf<TPosition::VECCALC> featureBuffer_;
    TBuf<TPosition::VECCALC> repeatBuffer_;
    GlobalTensor<uint8_t> srcGm_;
    GlobalTensor<int64_t> indptrGm_;
    GlobalTensor<uint8_t> outGm_;
    event_t mte2ToV_ = static_cast<event_t>(0);
    event_t mte2ToMte3_ = static_cast<event_t>(0);
    event_t vToMte3_ = static_cast<event_t>(0);
    event_t mte3ToV_ = static_cast<event_t>(0);
    event_t mte3ToMte2_ = static_cast<event_t>(0);
    uint64_t batchCount_ = 0;
    uint64_t segmentCount_ = 0;
    uint64_t outputRows_ = 0;
    uint64_t featureBytes_ = 0;
    uint64_t totalSegments_ = 0;
    uint64_t jobBegin_ = 0;
    uint64_t jobEnd_ = 0;
    uint32_t tileBytes_ = 0;
    uint32_t activeCoreNum_ = 0;
    uint32_t scheduleMode_ = GATHER_CSR_SEGMENT_MAJOR;
};

__attribute__((aiv)) __global__ __aicore__ void gather_csr_kernel(
    __gm__ uint8_t* src, __gm__ int64_t* indptr, __gm__ uint8_t* out,
    const GatherCsrTilingData tiling)
{
    TPipe pipe;
    GatherCsrKernel op;
    op.Init(src, indptr, out, tiling, &pipe);
    op.Process();
}

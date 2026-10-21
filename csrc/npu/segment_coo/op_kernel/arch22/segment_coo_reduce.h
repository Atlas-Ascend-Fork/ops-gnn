/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef OPS_GNN_SEGMENT_COO_REDUCE_H
#define OPS_GNN_SEGMENT_COO_REDUCE_H

#include "segment_coo.h"
#include "kernel_operator.h"
#include <cstdint>
#include <type_traits>

namespace opsgnn {

// ============================================================================
// Shared index, transfer and synchronization helpers
// ============================================================================

using namespace AscendC;

template <HardEvent Event> __aicore__ inline void SyncEvent(event_t id = EVENT_ID0) {
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

// Host normalizes broadcast prefixes into contiguous COO rows. Mode 1 reads
// those rows; mode 2 has a complete uniform-segment proof. Keep search bounds
// explicit so the vector kernel can reuse its adjacent-segment cache.
__aicore__ inline uint64_t IndexLowerBound(const GlobalTensor<int64_t> &index, uint64_t base, uint64_t length,
                                           int64_t label, uint64_t left = 0, uint64_t rowStride = 1) {
    uint64_t right = length;
    while (left < right) {
        const uint64_t mid = left + (right - left) / 2;
        if (index.GetValue(base + mid * rowStride) < label)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}
__aicore__ inline void FindSegmentBounds(const GlobalTensor<int64_t> &index, const SegmentTilingData &t, uint64_t batch,
                                         uint64_t segment, uint64_t &first, uint64_t &last) {
    if (t.indexMode == kUniformIndex) {
        const uint64_t rows = t.M / t.nSegments;
        first = segment * rows;
        last = first + rows;
    } else {
        const uint64_t base = batch * t.indexBatchStride;
        first = IndexLowerBound(index, base, t.M, segment);
        last = IndexLowerBound(index, base, t.M, segment + 1);
    }
}

// These helpers only issue DMA. Callers keep their existing dense fast paths,
// ownership and event order. Padding is at most one 32-byte block; extra UB
// pitch uses the destination stride. Oversized GM strides use separate rows.
template <typename T>
__aicore__ inline void LoadPaddedRows(LocalTensor<T> local, GlobalTensor<T> &global, uint64_t offset, uint32_t rows,
                                      uint32_t valid, uint32_t width, uint64_t gmWidth) {
    constexpr uint32_t block = 32 / sizeof(T);
    const uint32_t transfer = (valid + block - 1) / block * block;
    const uint64_t gap = (gmWidth - valid) * sizeof(T);
    const auto pad = DataCopyPadExtParams<T>(true, 0, static_cast<uint8_t>(transfer - valid), 0);
    if (gap <= UINT32_MAX) {
        DataCopyPad(local, global[offset],
                    {static_cast<uint16_t>(rows), valid * uint32_t(sizeof(T)), static_cast<uint32_t>(gap),
                     (width - transfer) / block, 0},
                    pad);
    } else {
        for (uint32_t row = 0; row < rows; ++row)
            DataCopyPad(local[uint64_t(row) * width], global[offset + uint64_t(row) * gmWidth],
                        {1, valid * uint32_t(sizeof(T)), 0, 0, 0}, pad);
    }
}
template <typename T>
__aicore__ inline void StorePaddedRows(GlobalTensor<T> &global, uint64_t offset, LocalTensor<T> local, uint32_t rows,
                                       uint32_t valid, uint32_t width, uint64_t gmWidth) {
    const uint64_t gap = (gmWidth - valid) * sizeof(T);
    if (gap <= UINT32_MAX) {
        DataCopyPad(global[offset], local,
                    {static_cast<uint16_t>(rows), valid * uint32_t(sizeof(T)),
                     (width - valid) * uint32_t(sizeof(T)) / 32, static_cast<uint32_t>(gap), 0});
    } else {
        for (uint32_t row = 0; row < rows; ++row)
            DataCopyPad(global[offset + uint64_t(row) * gmWidth], local[uint64_t(row) * width],
                        {1, valid * uint32_t(sizeof(T)), 0, 0, 0});
    }
}


template <typename Runner, SegmentReduce Reduce>
__global__ __aicore__ void SegmentTypedKernel(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg,
                                              SegmentTilingData t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (Reduce == SegmentMin || Reduce == SegmentMax) {
        if (t.hasArg) {
            Runner::template Run<Reduce, true>(src, index, out, arg, t);
            return;
        }
    }
    Runner::template Run<Reduce, false>(src, index, out, arg, t);
}

template <typename Runner>
inline void DispatchTyped(void *src, int64_t *index, void *out, int64_t *arg, SegmentTilingData t, aclrtStream stream) {
#define LAUNCH(R)                                                                                                      \
    SegmentTypedKernel<Runner, R>                                                                                      \
        <<<t.usedBlockNum, nullptr, stream>>>((GM_ADDR)src, (GM_ADDR)index, (GM_ADDR)out, (GM_ADDR)arg, t)
    switch (t.reduceCode) {
        case SegmentSum:
            LAUNCH(SegmentSum);
            break;
        case SegmentMean:
            LAUNCH(SegmentMean);
            break;
        case SegmentMin:
            LAUNCH(SegmentMin);
            break;
        case SegmentMax:
            LAUNCH(SegmentMax);
            break;
    }
#undef LAUNCH
}


// Launch the general floating/INT32 implementation in its own translation unit.
void LaunchSegmentCooVector(void *src, int64_t *index, void *out, int64_t *arg, const SegmentTilingData &tiling,
                            aclrtStream stream);

} // namespace opsgnn

#endif // OPS_GNN_SEGMENT_COO_REDUCE_H

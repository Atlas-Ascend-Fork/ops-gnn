/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_SOFTMAX_KERNEL_COMMON_H
#define OPS_GNN_SCATTER_SOFTMAX_KERNEL_COMMON_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_tiling.h"

/**
 * Shared vocabulary of the grouped-softmax device kernels: the pass tags, the
 * pipeline event set, the UB buffer layout and the fp32 widening helpers.
 *
 * Everything here is dtype independent; the per-pass arithmetic lives in
 * scatter_softmax_kernel_phase.h and the kernel entries in
 * scatter_softmax_kernel.h.
 */

namespace ScatterSoftmaxNpu {

using namespace AscendC;

// One device pass. The row/tile driver and the buffer layout are shared, so a
// pass only contributes its own arithmetic.
enum ScatterSoftmaxPhase {
    kPhaseMaximum = 0,
    kPhaseExp = 1,
    kPhaseNormalize = 2,
    kPhaseGradSum = 3,
    kPhaseGrad = 4,
};

// Passes that reduce into a (batch, group, feature) workspace slot.
template <ScatterSoftmaxPhase Phase>
__aicore__ constexpr bool IsReductionPass()
{
    return Phase == kPhaseMaximum || Phase == kPhaseExp || Phase == kPhaseGradSum;
}

// Passes that read the reduction workspace of an earlier pass.
template <ScatterSoftmaxPhase Phase>
__aicore__ constexpr bool ReadsWorkspace()
{
    return Phase == kPhaseExp || Phase == kPhaseNormalize || Phase == kPhaseGrad;
}

template <HardEvent Event>
__aicore__ inline void SyncEvent(event_t id)
{
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

// Elements covered by a tile that starts at `begin`.
__aicore__ inline uint32_t TileLength(uint64_t total, uint64_t begin, uint32_t tile)
{
    const uint64_t remaining = total - begin;
    const uint64_t limit = static_cast<uint64_t>(tile);
    return static_cast<uint32_t>(remaining < limit ? remaining : limit);
}

// UB allocations and vector destinations must be 32-byte based.
__aicore__ inline uint32_t AlignedBytes(uint64_t elements, uint64_t elementBytes)
{
    const uint64_t bytes = elements * elementBytes;
    return static_cast<uint32_t>((bytes + 31) / 32 * 32);
}

// A row of `after` features is not 32-byte aligned, so it always travels as one
// byte-counted burst.
__aicore__ inline DataCopyExtParams Bytes1D(uint32_t bytes)
{
    return DataCopyExtParams{1, bytes, 0, 0, 0};
}

template <typename T>
__aicore__ inline DataCopyPadExtParams<T> NoPadding()
{
    return DataCopyPadExtParams<T>(false, 0, 0, 0);
}

// Event pair per pipeline hazard. The four ids are reused for every batch:
// each SyncEvent call sets and immediately waits, so the ordering is exact and
// no id stays pending across a reuse.
struct PipelineSync {
    event_t loadToCompute;   // MTE2 -> V
    event_t computeToLoad;   // V -> MTE2
    event_t computeToStore;  // V -> MTE3
    event_t storeToCompute;  // MTE3 -> V (and, through issue order, MTE2)

    __aicore__ inline void Init(TPipe& pipe)
    {
        loadToCompute = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        computeToLoad = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
        computeToStore = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
        storeToCompute = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
    }

    __aicore__ inline void Release(TPipe& pipe)
    {
        pipe.ReleaseEventID<HardEvent::MTE2_V>(loadToCompute);
        pipe.ReleaseEventID<HardEvent::V_MTE2>(computeToLoad);
        pipe.ReleaseEventID<HardEvent::V_MTE3>(computeToStore);
        pipe.ReleaseEventID<HardEvent::MTE3_V>(storeToCompute);
    }
};

// Buffers and global tensors a pass may touch. Every pass shares the layout so
// the row/tile driver is written once; a pass simply leaves the rest unused.
// The row buffers hold `rowBatch` tiles of `colTile` elements each.
template <typename T>
struct PhaseTensors {
    TBuf<TPosition::VECCALC> narrowBuffer;
    TBuf<TPosition::VECCALC> narrowBuffer2;
    TBuf<TPosition::VECCALC> wideBuffer;
    TBuf<TPosition::VECCALC> wideBuffer2;
    TBuf<TPosition::VECCALC> wideBuffer3;
    TBuf<TPosition::VECCALC> accumBuffer;
    TBuf<TPosition::VECCALC> workspaceBuffer;
    TBuf<TPosition::VECCALC> groupBuffer;
    LocalTensor<T> narrow;
    LocalTensor<T> narrow2;
    LocalTensor<float> wide;
    LocalTensor<float> wide2;
    LocalTensor<float> wide3;
    LocalTensor<float> accum;
    LocalTensor<float> cached;
    LocalTensor<int32_t> groups;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<T> srcGm;
    GlobalTensor<T> gradGm;
    GlobalTensor<T> outGm;
    GlobalTensor<T> gradSrcGm;
    GlobalTensor<float> maximumGm;
    GlobalTensor<float> sumGm;
};

template <ScatterSoftmaxPhase Phase, typename T>
__aicore__ inline void InitTensors(PhaseTensors<T>& tensors, TPipe& pipe,
                                   const ScatterSoftmaxTilingData& tiling)
{
    const uint64_t rowElements =
        static_cast<uint64_t>(tiling.colTile) * tiling.rowBatch;
    pipe.InitBuffer(tensors.narrowBuffer, AlignedBytes(rowElements, sizeof(T)));
    pipe.InitBuffer(tensors.narrowBuffer2, AlignedBytes(rowElements, sizeof(T)));
    pipe.InitBuffer(tensors.wideBuffer, AlignedBytes(rowElements, sizeof(float)));
    pipe.InitBuffer(tensors.wideBuffer2, AlignedBytes(rowElements, sizeof(float)));
    pipe.InitBuffer(tensors.wideBuffer3, AlignedBytes(rowElements, sizeof(float)));
    pipe.InitBuffer(tensors.groupBuffer, AlignedBytes(tiling.rowBatch, sizeof(int32_t)));
    tensors.narrow = tensors.narrowBuffer.template Get<T>();
    tensors.narrow2 = tensors.narrowBuffer2.template Get<T>();
    tensors.wide = tensors.wideBuffer.template Get<float>();
    tensors.wide2 = tensors.wideBuffer2.template Get<float>();
    tensors.wide3 = tensors.wideBuffer3.template Get<float>();
    tensors.groups = tensors.groupBuffer.template Get<int32_t>();
    if (tiling.localAccum != 0) {
        // The workspace is resident in UB for the whole pass: the reduction
        // phases accumulate into `accum`, the passes that read it use `cached`.
        pipe.InitBuffer(tensors.accumBuffer,
                        AlignedBytes(tiling.workspaceElems, sizeof(float)));
        tensors.accum = tensors.accumBuffer.template Get<float>();
        if (ReadsWorkspace<Phase>()) {
            pipe.InitBuffer(tensors.workspaceBuffer,
                            AlignedBytes(tiling.workspaceElems, sizeof(float)));
            tensors.cached = tensors.workspaceBuffer.template Get<float>();
        }
    }
}

// Widens `count` source elements into fp32. T == float still goes through a
// vector instruction so every dtype shares one downstream code path.
template <typename T>
__aicore__ inline void WidenToFp32(const LocalTensor<float>& dst, const LocalTensor<T>& src, uint32_t count)
{
    if (count == 0) {
        return;
    }
    if constexpr (std::is_same<T, float>::value) {
        Adds(dst, src, 0.0F, static_cast<int32_t>(count));
    } else {
        Cast(dst, src, RoundMode::CAST_NONE, static_cast<int32_t>(count));
    }
}

// Narrows `count` fp32 elements back to the source dtype, rounding to nearest.
template <typename T>
__aicore__ inline void NarrowFromFp32(const LocalTensor<T>& dst, const LocalTensor<float>& src, uint32_t count)
{
    if (count == 0) {
        return;
    }
    if constexpr (std::is_same<T, float>::value) {
        Adds(dst, src, 0.0F, static_cast<int32_t>(count));
    } else {
        Cast(dst, src, RoundMode::CAST_RINT, static_cast<int32_t>(count));
    }
}

// A group label is only usable when it lands inside the requested range; the
// host already rejects out-of-range values, so this is a defensive guard.
__aicore__ inline bool GroupValid(const ScatterSoftmaxTilingData& tiling, int32_t group)
{
    return group >= 0 && static_cast<uint64_t>(group) < tiling.groupCount;
}

// Workspace row of (row, group): the group labels are not sorted, so the
// reduction target is a function of the label, not of the row.
__aicore__ inline uint64_t WorkBase(const ScatterSoftmaxTilingData& tiling, uint64_t row, int32_t group)
{
    const uint64_t batch = row / tiling.dimLength;
    return (batch * tiling.groupCount + static_cast<uint64_t>(group)) * tiling.afterPadded;
}

__aicore__ inline bool IsParticipating(const ScatterSoftmaxTilingData& tiling)
{
    return tiling.blockNum != 0 &&
           static_cast<uint64_t>(GetBlockIdx()) < tiling.blockNum &&
           tiling.rows != 0 && tiling.after != 0;
}

}  // namespace ScatterSoftmaxNpu

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_COMMON_H

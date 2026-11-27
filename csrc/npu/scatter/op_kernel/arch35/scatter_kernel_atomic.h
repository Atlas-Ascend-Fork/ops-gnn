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

__attribute__((aiv)) __global__ __aicore__ void scatter_compact_count_kernel(
    GM_ADDR index, GM_ADDR count, ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0) {
        return;
    }
    uint64_t rows = tiling.beforeDim * tiling.dimLength;
    uint64_t start = rows * block / blocks;
    uint64_t end = rows * (block + 1) / blocks;
    asc_vf_call<CompactCountCompute>(
        dim3(THREAD_NUM), start, end, tiling.dimLength, tiling.outputDim,
        tiling.indexDimLength, (__gm__ int64_t*)index, (__gm__ int32_t*)count,
        tiling.hotTarget);
}

__attribute__((aiv)) __global__ __aicore__ void
scatter_compact_hot_count_kernel(GM_ADDR index, GM_ADDR count,
                                 ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0 || tiling.hotTarget < 0) {
        return;
    }
    int32_t localCount = 0;
    uint64_t rows = tiling.beforeDim * tiling.dimLength;
    for (uint64_t row = block; row < rows; row += blocks) {
        uint64_t e = row % tiling.dimLength;
        uint64_t indexOffset = tiling.indexDimLength == 1 ? 0 : e;
        if (((__gm__ int64_t*)index)[indexOffset] == tiling.hotTarget) {
            ++localCount;
        }
    }

    GlobalTensor<int32_t> countGm;
    countGm.SetGlobalBuffer((__gm__ int32_t*)count,
                            tiling.beforeDim * tiling.outputDim);
    TPipe pipe;
    TBuf<TPosition::VECCALC> countBuffer;
    pipe.InitBuffer(countBuffer, 32);
    LocalTensor<int32_t> local = countBuffer.Get<int32_t>();
    event_t ready = static_cast<event_t>(
        GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t done = static_cast<event_t>(
        GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    Duplicate(local, localCount, 1);
    SetFlag<HardEvent::V_MTE3>(ready);
    WaitFlag<HardEvent::V_MTE3>(ready);
    SetAtomicAdd<int32_t>();
    DataCopy(countGm[static_cast<uint64_t>(tiling.hotTarget)], local, 1);
    DisableDmaAtomic();
    SetFlag<HardEvent::MTE3_MTE2>(done);
    WaitFlag<HardEvent::MTE3_MTE2>(done);
    PipeBarrier<PIPE_ALL>();
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(ready);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(done);
}

template <typename T>
__aicore__ inline void FinalizeCompactMeanGroup(
    uint64_t group, ScatterTilingData tiling, __gm__ int32_t* count,
    GlobalTensor<T>& outGm, LocalTensor<T> localBase,
    DoubleBufferEvents& events, uint64_t& sequence)
{
    int32_t divisor = count[group];
    divisor = divisor < 1 ? 1 : divisor;
    T scale = static_cast<T>(1.0F / static_cast<float>(divisor));
    constexpr uint32_t tileElements = VECTOR_COPY_BYTES / sizeof(T);
    uint64_t outputBase = group * tiling.afterDim;
    for (uint64_t inner = 0; inner < tiling.afterDim; inner += tileElements) {
        uint32_t slot = static_cast<uint32_t>(sequence & 1U);
        events.WaitSlot(slot);
        LocalTensor<T> local = localBase[slot * tileElements];
        uint32_t length = TileLength(tiling.afterDim, inner, tileElements);
        DataCopy(local, outGm[outputBase + inner], length);
        SetFlag<HardEvent::MTE2_V>(events.load[slot]);
        WaitFlag<HardEvent::MTE2_V>(events.load[slot]);
        Muls(local, local, scale, static_cast<int32_t>(length));
        SetFlag<HardEvent::V_MTE3>(events.compute[slot]);
        WaitFlag<HardEvent::V_MTE3>(events.compute[slot]);
        DataCopy(outGm[outputBase + inner], local, length);
        SetFlag<HardEvent::MTE3_MTE2>(events.done[slot]);
        events.issued[slot] = true;
        ++sequence;
    }
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void
scatter_compact_mean_finalize_kernel(GM_ADDR out, GM_ADDR count,
                                     ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0 || tiling.afterDim == 0) {
        return;
    }

    GlobalTensor<T> outGm;
    outGm.SetGlobalBuffer((__gm__ T*)out, tiling.outNumel);
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer;
    pipe.InitBuffer(dataBuffer, 2 * VECTOR_COPY_BYTES);
    LocalTensor<T> localBase = dataBuffer.Get<T>();
    DoubleBufferEvents events;
    events.Init();
    uint64_t sequence = 0;
    uint64_t groups = tiling.beforeDim * tiling.outputDim;
    for (uint64_t group = block; group < groups; group += blocks) {
        FinalizeCompactMeanGroup<T>(group, tiling, (__gm__ int32_t*)count,
                                    outGm, localBase, events, sequence);
    }
    events.Finish();
}

// Ascend 950PR does not perform MTE3 atomic add for fp16 destinations.  The
// compact-row performance path widens each source tile, atomically accumulates
// into an fp32 workspace, then narrows once after all rows are complete.

__aicore__ inline void AccumulateF16Row(
    ScatterRow row, ScatterTilingData tiling, GlobalTensor<half>& srcGm,
    GlobalTensor<float>& accumGm, LocalTensor<float> storage,
    DoubleBufferEvents& events, uint64_t& sequence)
{
    for (uint64_t inner = 0; inner < tiling.afterDim;
         inner += PROMOTE_TILE_ELEMENTS) {
        uint32_t slot = static_cast<uint32_t>(sequence & 1U);
        events.WaitSlot(slot);
        LocalTensor<float> wide = storage[slot * PROMOTE_SLOT_FLOATS];
        LocalTensor<half> raw =
            storage[slot * PROMOTE_SLOT_FLOATS + PROMOTE_TILE_ELEMENTS]
                .ReinterpretCast<half>();
        uint32_t length =
            TileLength(tiling.afterDim, inner, PROMOTE_TILE_ELEMENTS);
        DataCopy(raw, srcGm[row.sourceBase + inner], length);
        SetFlag<HardEvent::MTE2_V>(events.load[slot]);
        WaitFlag<HardEvent::MTE2_V>(events.load[slot]);
        Cast(wide, raw, RoundMode::CAST_NONE, static_cast<int32_t>(length));
        SetFlag<HardEvent::V_MTE3>(events.compute[slot]);
        WaitFlag<HardEvent::V_MTE3>(events.compute[slot]);
        SetAtomicAdd<float>();
        DataCopy(accumGm[row.outputBase + inner], wide, length);
        DisableDmaAtomic();
        SetFlag<HardEvent::MTE3_MTE2>(events.done[slot]);
        events.issued[slot] = true;
        ++sequence;
    }
}

__attribute__((aiv)) __global__ __aicore__ void
scatter_f16_float_atomic_kernel(GM_ADDR src, GM_ADDR index, GM_ADDR accum,
                                ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0 || tiling.afterDim == 0) {
        return;
    }

    GlobalTensor<half> srcGm;
    GlobalTensor<float> accumGm;
    srcGm.SetGlobalBuffer((__gm__ half*)src, tiling.srcNumel);
    accumGm.SetGlobalBuffer((__gm__ float*)accum, tiling.outNumel);

    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer;
    pipe.InitBuffer(dataBuffer, 2 * PROMOTE_SLOT_FLOATS * sizeof(float));
    LocalTensor<float> storage = dataBuffer.Get<float>();
    DoubleBufferEvents events;
    events.Init();
    uint64_t sequence = 0;
    uint64_t rows = tiling.beforeDim * tiling.dimLength;
    for (uint64_t row = block; row < rows; row += blocks) {
        ScatterRow mapping =
            ResolveScatterRow(row, (__gm__ int64_t*)index, tiling);
        if (!mapping.valid ||
            (tiling.hotTarget >= 0 && mapping.target == tiling.hotTarget)) {
            continue;
        }
        AccumulateF16Row(mapping, tiling, srcGm, accumGm, storage,
                         events, sequence);
    }
    events.Finish();
}

__aicore__ inline float FinalizeF16Scale(
    uint64_t group, ScatterTilingData tiling, __gm__ int32_t* count)
{
    if (tiling.reduce != SCATTER_MEAN) {
        return 1.0F;
    }
    int32_t divisor = count[group];
    divisor = divisor < 1 ? 1 : divisor;
    return 1.0F / static_cast<float>(divisor);
}

__aicore__ inline void FinalizeF16Group(
    uint64_t group, ScatterTilingData tiling, float scale,
    GlobalTensor<float>& accumGm, GlobalTensor<half>& outGm,
    LocalTensor<float> storage, DoubleBufferEvents& events,
    uint64_t& sequence)
{
    uint64_t outputBase = group * tiling.afterDim;
    for (uint64_t inner = 0; inner < tiling.afterDim;
         inner += PROMOTE_TILE_ELEMENTS) {
        uint32_t slot = static_cast<uint32_t>(sequence & 1U);
        events.WaitSlot(slot);
        LocalTensor<float> wide = storage[slot * PROMOTE_SLOT_FLOATS];
        LocalTensor<half> narrow =
            storage[slot * PROMOTE_SLOT_FLOATS + PROMOTE_TILE_ELEMENTS]
                .ReinterpretCast<half>();
        uint32_t length =
            TileLength(tiling.afterDim, inner, PROMOTE_TILE_ELEMENTS);
        DataCopy(wide, accumGm[outputBase + inner], length);
        SetFlag<HardEvent::MTE2_V>(events.load[slot]);
        WaitFlag<HardEvent::MTE2_V>(events.load[slot]);
        if (tiling.reduce == SCATTER_MEAN) {
            Muls(wide, wide, scale, static_cast<int32_t>(length));
            PipeBarrier<PIPE_V>();
        }
        Cast(narrow, wide, RoundMode::CAST_NONE, static_cast<int32_t>(length));
        SetFlag<HardEvent::V_MTE3>(events.compute[slot]);
        WaitFlag<HardEvent::V_MTE3>(events.compute[slot]);
        DataCopy(outGm[outputBase + inner], narrow, length);
        SetFlag<HardEvent::MTE3_MTE2>(events.done[slot]);
        events.issued[slot] = true;
        ++sequence;
    }
}

__attribute__((aiv)) __global__ __aicore__ void
scatter_f16_float_finalize_kernel(GM_ADDR accum, GM_ADDR out, GM_ADDR count,
                                  ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0 || tiling.afterDim == 0) {
        return;
    }

    GlobalTensor<float> accumGm;
    GlobalTensor<half> outGm;
    accumGm.SetGlobalBuffer((__gm__ float*)accum, tiling.outNumel);
    outGm.SetGlobalBuffer((__gm__ half*)out, tiling.outNumel);
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer;
    pipe.InitBuffer(dataBuffer, 2 * PROMOTE_SLOT_FLOATS * sizeof(float));
    LocalTensor<float> storage = dataBuffer.Get<float>();
    DoubleBufferEvents events;
    events.Init();
    uint64_t sequence = 0;
    uint64_t groups = tiling.beforeDim * tiling.outputDim;
    for (uint64_t group = block; group < groups; group += blocks) {
        float scale = FinalizeF16Scale(
            group, tiling, (__gm__ int32_t*)count);
        FinalizeF16Group(group, tiling, scale, accumGm, outGm, storage,
                         events, sequence);
    }
    events.Finish();
}

// Compact min/max is split into value reduction and arg selection.  All
// values are reduced in fp32 so the same MTE3 atomic path works for fp16 and
// fp32 on Ascend 950PR.  A later pass atomically maximizes the matching source
// position, implementing the task document's later-writer tie rule.

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void scatter_vector_zero_kernel(
    GM_ADDR values, GM_ADDR count, ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0) {
        return;
    }

    GlobalTensor<T> valuesGm;
    valuesGm.SetGlobalBuffer((__gm__ T*)values, tiling.outNumel);

    TPipe pipe;
    TBuf<TPosition::VECCALC> zeroBuffer;
    pipe.InitBuffer(zeroBuffer, ZERO_COPY_BYTES);
    LocalTensor<T> zero = zeroBuffer.Get<T>();
    constexpr uint32_t tileElements = ZERO_COPY_BYTES / sizeof(T);
    Duplicate(zero, static_cast<T>(0), tileElements);
    event_t ready =
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(ready);
    WaitFlag<HardEvent::V_MTE3>(ready);

    if (tiling.hasOut == 0) {
        for (uint64_t offset = block * tileElements; offset < tiling.outNumel;
             offset += blocks * tileElements) {
            uint32_t length = static_cast<uint32_t>(
                tiling.outNumel - offset > tileElements ?
                    tileElements : tiling.outNumel - offset);
            DataCopy(valuesGm[offset], zero, length);
        }
    }

    if (tiling.reduce == SCATTER_MEAN) {
        uint64_t countElements = tiling.beforeDim * tiling.outputDim;
        GlobalTensor<int32_t> countGm;
        countGm.SetGlobalBuffer((__gm__ int32_t*)count, countElements);
        LocalTensor<int32_t> zeroCount =
            zero.template ReinterpretCast<int32_t>();
        constexpr uint32_t countTileElements =
            ZERO_COPY_BYTES / sizeof(int32_t);
        for (uint64_t offset = block * countTileElements;
             offset < countElements;
             offset += blocks * countTileElements) {
            uint32_t length = static_cast<uint32_t>(
                countElements - offset > countTileElements ?
                    countTileElements : countElements - offset);
            DataCopyExtParams copyParams{1, length * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(countGm[offset], zeroCount, copyParams);
        }
    }

    PipeBarrier<PIPE_ALL>();
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(ready);
}

template <typename T>
__aicore__ inline void CopyAtomicTile(
    GlobalTensor<T>& srcGm, GlobalTensor<T>& outGm,
    LocalTensor<T> localBase, CopyBufferEvents& events, uint64_t source,
    uint64_t destination, uint32_t length, uint32_t slot,
    uint32_t tileElements)
{
    events.WaitSlot(slot);
    LocalTensor<T> local = localBase[slot * tileElements];
    DataCopy(local, srcGm[source], length);
    SetFlag<HardEvent::MTE2_MTE3>(events.ready[slot]);
    WaitFlag<HardEvent::MTE2_MTE3>(events.ready[slot]);
    SetAtomicAdd<T>();
    DataCopy(outGm[destination], local, length);
    DisableDmaAtomic();
    SetFlag<HardEvent::MTE3_MTE2>(events.complete[slot]);
    events.active[slot] = true;
}

template <typename T>
__aicore__ inline void CopyAtomicRow(
    GlobalTensor<T>& srcGm, GlobalTensor<T>& outGm,
    LocalTensor<T> localBase, CopyBufferEvents& events,
    ScatterRow rowInfo, ScatterTilingData tiling, uint64_t& sequence)
{
    constexpr uint32_t tileElements = VECTOR_COPY_BYTES / sizeof(T);
    for (uint64_t inner = 0; inner < tiling.afterDim;
         inner += tileElements, ++sequence) {
        uint32_t slot = static_cast<uint32_t>(sequence & 1U);
        uint32_t length = TileLength(tiling.afterDim, inner, tileElements);
        CopyAtomicTile<T>(srcGm, outGm, localBase, events,
                          rowInfo.sourceBase + inner,
                          rowInfo.outputBase + inner, length, slot,
                          tileElements);
    }
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void scatter_vector_atomic_kernel(
    GM_ADDR src, GM_ADDR index, GM_ADDR out, ScatterTilingData tiling)
{
    uint64_t block = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blocks = static_cast<uint64_t>(tiling.blockNum);
    if (block >= blocks || blocks == 0 || tiling.afterDim == 0) {
        return;
    }
    GlobalTensor<T> srcGm;
    GlobalTensor<T> outGm;
    srcGm.SetGlobalBuffer((__gm__ T*)src, tiling.srcNumel);
    outGm.SetGlobalBuffer((__gm__ T*)out, tiling.outNumel);
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer;
    pipe.InitBuffer(dataBuffer, 2 * VECTOR_COPY_BYTES);
    LocalTensor<T> localBase = dataBuffer.Get<T>();
    CopyBufferEvents events;
    events.Init();
    uint64_t sequence = 0;
    uint64_t rows = tiling.beforeDim * tiling.dimLength;
    for (uint64_t row = block; row < rows; row += blocks) {
        ScatterRow rowInfo = ResolveScatterRow(
            row, (__gm__ int64_t*)index, tiling);
        if (rowInfo.valid) {
            CopyAtomicRow<T>(srcGm, outGm, localBase, events, rowInfo,
                             tiling, sequence);
        }
    }
    events.Finish();
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void
scatter_vector_hot_atomic_kernel(GM_ADDR src, GM_ADDR index, GM_ADDR out,
                                 ScatterTilingData tiling)
{
    RunHotPromotedKernel<T>(src, index, out, tiling, SCATTER_SUM);
}

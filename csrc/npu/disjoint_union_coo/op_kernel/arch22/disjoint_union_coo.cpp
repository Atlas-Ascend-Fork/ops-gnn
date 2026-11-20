/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "disjoint_union_coo/op_kernel/arch22/disjoint_union_coo.h"

#include <type_traits>

#include "kernel_operator.h"

namespace {
using namespace AscendC;

constexpr uint32_t kUbReservedBytes = 2048;
constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kTileBuffers = 3;  // row, col and data tiles
constexpr uint32_t kColTile = 1;
constexpr uint32_t kDataTile = 2;
constexpr uint32_t kNarrowBytes = sizeof(int32_t);
// 32-bit lanes holding one element of the widest supported index type.
constexpr uint32_t kWideLanes = sizeof(int64_t) / kNarrowBytes;
constexpr int32_t kEventId = EVENT_ID0;

// Whether T needs the narrow/shift/widen sequence (arch22 has no 64-bit add).
template <typename T>
constexpr bool kIsWideIndex = std::is_same<T, int64_t>::value;

// Tile capacity in elements. The int64 path also reserves one 32-bit scratch
// tile for the narrow/shift/widen sequence. Every tile is 32-byte aligned, so
// the three local buffers stay independently addressable by DataCopy.
template <typename T>
__aicore__ inline uint32_t TileElements(uint32_t ubBytes)
{
    constexpr uint32_t kScratchBytes = kIsWideIndex<T> ? kNarrowBytes : 0u;
    constexpr uint32_t kBytesPerElement = kTileBuffers * sizeof(T) + kScratchBytes;
    const uint32_t elementsPerBlock = kAlignBytes / sizeof(T);
    const uint32_t usable = ubBytes > kUbReservedBytes ? ubBytes - kUbReservedBytes : 0;
    const uint32_t elements = usable / (kBytesPerElement * elementsPerBlock) * elementsPerBlock;
    return elements > 0 ? elements : elementsPerBlock;
}

// Concatenate one COO matrix into the disjoint-union result at its offset.
// Elements owned by an AIV are the contiguous range
// [block * n / blocks, (block + 1) * n / blocks).
template <typename T>
class DisjointUnionCooKernel {
public:
    // Bind the global buffers of one matrix and size the UB tiles.
    __aicore__ inline void Init(__gm__ T* inRow, __gm__ T* inCol, __gm__ T* inData, __gm__ T* outRow,
                                __gm__ T* outCol, __gm__ T* outData, int64_t offSrc, int64_t offDst,
                                int64_t offElm, uint32_t n, uint32_t totalNnz, uint32_t hasDataLocal,
                                uint32_t hasDataGlobal, uint32_t ubBytes, AscendC::TPipe* pipe)
    {
        n_ = n;
        totalNnz_ = totalNnz;
        offElm_ = offElm;
        offSrc_ = static_cast<T>(offSrc);
        offDst_ = static_cast<T>(offDst);
        offElmLocal_ = static_cast<T>(offElm);
        hasDataLocal_ = hasDataLocal;
        hasDataGlobal_ = hasDataGlobal;

        inRowGm_.SetGlobalBuffer(inRow, n);
        inColGm_.SetGlobalBuffer(inCol, n);
        inDataGm_.SetGlobalBuffer(inData, n);
        outRowGm_.SetGlobalBuffer(outRow, totalNnz);
        outColGm_.SetGlobalBuffer(outCol, totalNnz);
        outDataGm_.SetGlobalBuffer(outData, totalNnz);

        tileElements_ = TileElements<T>(ubBytes);
        pipe->InitBuffer(buffer_, tileElements_ * (kTileBuffers * sizeof(T) +
                                                   (kIsWideIndex<T> ? kNarrowBytes : 0u)));
        rowLocal_ = buffer_.Get<T>();
        colLocal_ = rowLocal_[kColTile * tileElements_];
        dataLocal_ = rowLocal_[kDataTile * tileElements_];
        if constexpr (kIsWideIndex<T>) {
            // The 32-bit scratch tile follows the three wide tiles.
            scratchLocal_ = rowLocal_.template ReinterpretCast<int32_t>()[tileElements_ * kTileBuffers *
                                                                           kWideLanes];
        }
    }

    // Walk the edge range owned by this AIV, one UB tile at a time.
    __aicore__ inline void Process()
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t launched = AscendC::GetBlockNum();
        const uint32_t blocks = launched > 0 ? launched : 1;
        const uint32_t start = static_cast<uint32_t>(static_cast<uint64_t>(block) * n_ / blocks);
        const uint32_t end = static_cast<uint32_t>(static_cast<uint64_t>(block + 1) * n_ / blocks);
        for (uint32_t offset = start; offset < end; offset += tileElements_) {
            const uint32_t remaining = end - offset;
            const uint32_t count = remaining < tileElements_ ? remaining : tileElements_;
            CopyIn(offset, count);
            Compute(offset, count);
            CopyOut(offset, count);
            // The tile buffers are reused by the next iteration, so this store
            // must drain before the next load overwrites them.
            if (offset + count < end) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEventId);
            }
        }
    }

private:
    // Load the row, column and (when present) data slices of one tile.
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        AscendC::DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, T(0)};
        AscendC::DataCopyPad(rowLocal_, inRowGm_[offset], copyParams, padParams);
        AscendC::DataCopyPad(colLocal_, inColGm_[offset], copyParams, padParams);
        if (hasDataLocal_ != 0) {
            AscendC::DataCopyPad(dataLocal_, inDataGm_[offset], copyParams, padParams);
        }
    }

    // Shift the row, column and data tiles by their accumulated offsets.
    __aicore__ inline void Compute(uint32_t offset, uint32_t count)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEventId);
        AddOffset(rowLocal_, offSrc_, count);
        AddOffset(colLocal_, offDst_, count);
        if (hasDataGlobal_ != 0) {
            if (hasDataLocal_ != 0) {
                AddOffset(dataLocal_, offElmLocal_, count);
            } else {
                FillEdgeIndex(offset, count);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEventId);
    }

    // Shift one tile by a constant. arch22 has no 64-bit vector add, so wide
    // tiles move through the vector unit as narrow/shift/widen: a shifted index
    // or edge id always stays within the concatenated extent (capped at
    // INT32_MAX), so narrowing to the low word is lossless for any input that
    // can form a valid COO.
    __aicore__ inline void AddOffset(AscendC::LocalTensor<T>& tile, T offset, uint32_t count)
    {
        const int32_t elements = static_cast<int32_t>(count);
        if constexpr (kIsWideIndex<T>) {
            AscendC::Cast(scratchLocal_, tile, AscendC::RoundMode::CAST_NONE, elements);
            AscendC::Adds(scratchLocal_, scratchLocal_, static_cast<int32_t>(offset), elements);
            AscendC::Cast(tile, scratchLocal_, AscendC::RoundMode::CAST_NONE, elements);
        } else {
            AscendC::Adds(tile, tile, offset, elements);
        }
    }

    // A matrix without an explicit data array contributes its edge index, i.e.
    // the global position offElm + local index.
    __aicore__ inline void FillEdgeIndex(uint32_t offset, uint32_t count)
    {
        const T first = static_cast<T>(offElm_ + static_cast<int64_t>(offset));
        const int32_t elements = static_cast<int32_t>(count);
        if constexpr (kIsWideIndex<T>) {
            for (uint32_t i = 0; i < count; ++i) {
                dataLocal_.SetValue(i, static_cast<T>(first + static_cast<T>(i)));
            }
        } else {
            AscendC::ArithProgression(dataLocal_, first, static_cast<T>(1), elements);
        }
    }

    // Store the shifted tiles at their global position. The destination range is
    // clipped to the allocated output so a malformed call can never write past
    // the buffer.
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        const uint32_t destination = static_cast<uint32_t>(offElm_) + offset;
        const uint32_t room = totalNnz_ > destination ? totalNnz_ - destination : 0;
        const uint32_t stores = count < room ? count : room;
        if (stores == 0) {
            return;
        }
        AscendC::DataCopyExtParams copyParams{1, stores * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(outRowGm_[destination], rowLocal_, copyParams);
        AscendC::DataCopyPad(outColGm_[destination], colLocal_, copyParams);
        if (hasDataGlobal_ != 0) {
            AscendC::DataCopyPad(outDataGm_[destination], dataLocal_, copyParams);
        }
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    AscendC::LocalTensor<T> rowLocal_;
    AscendC::LocalTensor<T> colLocal_;
    AscendC::LocalTensor<T> dataLocal_;
    AscendC::LocalTensor<int32_t> scratchLocal_;
    AscendC::GlobalTensor<T> inRowGm_;
    AscendC::GlobalTensor<T> inColGm_;
    AscendC::GlobalTensor<T> inDataGm_;
    AscendC::GlobalTensor<T> outRowGm_;
    AscendC::GlobalTensor<T> outColGm_;
    AscendC::GlobalTensor<T> outDataGm_;
    T offSrc_{0};
    T offDst_{0};
    T offElmLocal_{0};
    int64_t offElm_{0};
    uint32_t n_{0};
    uint32_t totalNnz_{0};
    uint32_t hasDataLocal_{0};
    uint32_t hasDataGlobal_{0};
    uint32_t tileElements_{0};
};

// Kernel body shared by the int32 and int64 entries.
template <typename T>
__aicore__ inline void DisjointUnionCooEntry(__gm__ T* inRow, __gm__ T* inCol, __gm__ T* inData,
                                             __gm__ T* outRow, __gm__ T* outCol, __gm__ T* outData,
                                             int64_t offSrc, int64_t offDst, int64_t offElm, uint32_t n,
                                             uint32_t totalNnz, uint32_t hasDataLocal,
                                             uint32_t hasDataGlobal, uint32_t ubBytes)
{
    AscendC::TPipe pipe;
    DisjointUnionCooKernel<T> kernel;
    kernel.Init(inRow, inCol, inData, outRow, outCol, outData, offSrc, offDst, offElm, n, totalNnz,
                hasDataLocal, hasDataGlobal, ubBytes, &pipe);
    kernel.Process();
}

}  // namespace

// Both index types share the body above, so the AIV entry is emitted once per
// type from a single definition.
#define OPSGNN_DEFINE_DISJOINT_UNION_COO_ENTRY(indexType, functionName)                            \
    extern "C" __global__ __aicore__ void functionName(                                            \
        __gm__ indexType* inRow, __gm__ indexType* inCol, __gm__ indexType* inData,                \
        __gm__ indexType* outRow, __gm__ indexType* outCol, __gm__ indexType* outData, int64_t offSrc, \
        int64_t offDst, int64_t offElm, uint32_t n, uint32_t totalNnz, uint32_t hasDataLocal,      \
        uint32_t hasDataGlobal, uint32_t ubBytes)                                                  \
    {                                                                                              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                            \
        DisjointUnionCooEntry<indexType>(inRow, inCol, inData, outRow, outCol, outData, offSrc,     \
                                         offDst, offElm, n, totalNnz, hasDataLocal, hasDataGlobal, \
                                         ubBytes);                                                 \
    }

OPSGNN_DEFINE_DISJOINT_UNION_COO_ENTRY(int32_t, disjoint_union_coo_int32)
OPSGNN_DEFINE_DISJOINT_UNION_COO_ENTRY(int64_t, disjoint_union_coo_int64)

namespace opsgnn {

namespace {

// Launch one concatenation pass for a single matrix on the current stream.
template <typename T>
void LaunchOnStream(uint32_t blocks, aclrtStream stream, const T* inRow, const T* inCol, const T* inData,
                    T* outRow, T* outCol, T* outData, int64_t offSrc, int64_t offDst, int64_t offElm,
                    uint32_t n, uint32_t totalNnz, uint32_t hasDataLocal, uint32_t hasDataGlobal,
                    uint32_t ubBytes)
{
    if constexpr (std::is_same<T, int32_t>::value) {
        disjoint_union_coo_int32<<<blocks, nullptr, stream>>>(
            const_cast<T*>(inRow), const_cast<T*>(inCol), const_cast<T*>(inData), outRow, outCol, outData,
            offSrc, offDst, offElm, n, totalNnz, hasDataLocal, hasDataGlobal, ubBytes);
    } else {
        disjoint_union_coo_int64<<<blocks, nullptr, stream>>>(
            const_cast<T*>(inRow), const_cast<T*>(inCol), const_cast<T*>(inData), outRow, outCol, outData,
            offSrc, offDst, offElm, n, totalNnz, hasDataLocal, hasDataGlobal, ubBytes);
    }
}

}  // namespace

// Launch one concatenation pass for int32 indices on the current stream.
void DisjointUnionCooInt32(uint32_t blocks, aclrtStream stream, const int32_t* inRow, const int32_t* inCol,
                           const int32_t* inData, int32_t* outRow, int32_t* outCol, int32_t* outData,
                           int64_t offSrc, int64_t offDst, int64_t offElm, uint32_t n, uint32_t totalNnz,
                           uint32_t hasDataLocal, uint32_t hasDataGlobal, uint32_t ubBytes)
{
    LaunchOnStream<int32_t>(blocks, stream, inRow, inCol, inData, outRow, outCol, outData, offSrc, offDst,
                            offElm, n, totalNnz, hasDataLocal, hasDataGlobal, ubBytes);
}

// Launch one concatenation pass for int64 indices on the current stream.
void DisjointUnionCooInt64(uint32_t blocks, aclrtStream stream, const int64_t* inRow, const int64_t* inCol,
                           const int64_t* inData, int64_t* outRow, int64_t* outCol, int64_t* outData,
                           int64_t offSrc, int64_t offDst, int64_t offElm, uint32_t n, uint32_t totalNnz,
                           uint32_t hasDataLocal, uint32_t hasDataGlobal, uint32_t ubBytes)
{
    LaunchOnStream<int64_t>(blocks, stream, inRow, inCol, inData, outRow, outCol, outData, offSrc, offDst,
                            offElm, n, totalNnz, hasDataLocal, hasDataGlobal, ubBytes);
}

}  // namespace opsgnn

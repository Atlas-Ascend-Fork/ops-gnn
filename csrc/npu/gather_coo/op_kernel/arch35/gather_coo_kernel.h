/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0.
 * Please refer to the License for details.
 */

#pragma once

#include <cstdint>

#include "gather_coo_tiling.h"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"

using namespace AscendC;

constexpr uint32_t GATHER_COO_INDEX_TILE_ENTRIES = 128;
constexpr uint32_t GATHER_COO_INDEX_BUFFER_ENTRIES =
    GATHER_COO_INDEX_TILE_ENTRIES + 4;

template <typename T, bool ExtendedSlab = true>
class GatherCooKernel {
public:
    __aicore__ inline GatherCooKernel() = default;

    __aicore__ inline void Init(
        GM_ADDR src,
        GM_ADDR index,
        GM_ADDR out,
        const GatherCooTilingData* tiling,
        TPipe* pipe)
    {
        this->pipe = pipe;
        this->batchCount = tiling->batchCount;
        this->sourceRows = tiling->sourceRows;
        this->indexRows = tiling->indexRows;
        this->featureCount = tiling->featureCount;
        this->logicalFeatureCount = tiling->logicalFeatureCount;
        this->elementBytes = tiling->elementBytes;
        this->totalRows = tiling->totalRows;
        this->rowsPerCore = tiling->rowsPerCore;
        this->extraCoreCount = tiling->extraCoreCount;
        this->featureTile = tiling->featureTile;
        this->coreNum = tiling->coreNum;
        this->useRunCache = tiling->useRunCache;

        srcGm.SetGlobalBuffer((__gm__ T*)src, tiling->sourceStorageElements);
        // Read the public int64 index as two uint32 lanes. The 950 scalar
        // GetValue<int64_t> path is not reliable; using the native 32-bit
        // scalar path preserves the exact bits without narrowing the index.
        indexGm.SetGlobalBuffer((__gm__ uint32_t*)index, tiling->indexStorageElements);
        outGm.SetGlobalBuffer((__gm__ T*)out, tiling->outputStorageElements);

        // TBuf allocations and DataCopyPad operate on 32-byte blocks. Keep a
        // full block for K=1/tiny rows while copying only the valid bytes.
        rowBufferSize = static_cast<uint32_t>(
            ((static_cast<uint64_t>(featureTile) * sizeof(T) + 31) / 32) * 32);
        pipe->InitBuffer(rowBuffer, rowBufferSize);
        rowLocal = rowBuffer.Get<T>();
        indexBufferSize = static_cast<uint32_t>(
            GATHER_COO_INDEX_BUFFER_ENTRIES * 2 * sizeof(uint32_t));
        pipe->InitBuffer(indexBuffer, indexBufferSize);
        indexLocal = indexBuffer.Get<uint32_t>();

        outputSlabRows = tiling->outputSlabRows;
        useOutputSlab = outputSlabRows != 0;
        if (useOutputSlab) {
            outputSlabSize = static_cast<uint32_t>(
                outputSlabRows * featureCount * sizeof(T));
            pipe->InitBuffer(outputSlabBuffer, outputSlabSize);
            outputSlabLocal = outputSlabBuffer.Get<T>();
        }

        eventMte2ToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
        eventMte3ToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        eventIndexMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        if (ExtendedSlab) {
            eventSlabVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        }
    }

    __aicore__ inline void Process()
    {
        if (!PrepareCore()) {
            return;
        }

        for (uint64_t tileBegin = rowBegin; tileBegin < rowEnd;) {
            const uint64_t candidateTileEnd =
                tileBegin + (ExtendedSlab && useOutputSlab ? outputSlabRows : GATHER_COO_INDEX_TILE_ENTRIES);
            const uint64_t tileEnd = candidateTileEnd < rowEnd
                ? candidateTileEnd
                : rowEnd;
            const uint64_t loadBegin = LoadIndexTile(tileBegin, tileEnd);
            if (useOutputSlab) {
                if (ExtendedSlab) {
                    ProcessOutputSlab(tileBegin, tileEnd, loadBegin);
                } else {
                    ProcessOriginalOutputSlab(tileBegin, tileEnd, loadBegin);
                }
            } else {
                ProcessGenericTile(tileBegin, tileEnd, loadBegin);
            }
            tileBegin = tileEnd;
        }
        DrainOutputSlab();
    }

private:
    __aicore__ inline bool PrepareCore()
    {
        if (indexRows == 0 || featureCount == 0 || totalRows == 0) {
            return false;
        }
        const uint64_t blockIndex = static_cast<uint64_t>(GetBlockIdx());
        if (blockIndex >= coreNum) {
            return false;
        }
        const uint64_t rowsForThisCore =
            blockIndex < extraCoreCount ? rowsPerCore + 1 : rowsPerCore;
        rowBegin = blockIndex < extraCoreCount
            ? blockIndex * (rowsPerCore + 1)
            : extraCoreCount * (rowsPerCore + 1) +
                (blockIndex - extraCoreCount) * rowsPerCore;
        rowEnd = rowBegin + rowsForThisCore;
        batch = rowBegin / indexRows;
        entry = rowBegin % indexRows;
        return rowBegin < rowEnd && rowBegin < totalRows;
    }

    __aicore__ inline uint64_t LoadIndexTile(uint64_t tileBegin, uint64_t tileEnd)
    {
        // Keep the aligned prefix in UB and skip it when decoding the tile.
        const uint64_t loadBegin = (tileBegin / 4) * 4;
        const uint64_t requestedLoadEnd = ((tileEnd + 3) / 4) * 4;
        const uint64_t loadEnd = requestedLoadEnd < totalRows
            ? requestedLoadEnd
            : totalRows;
        const uint64_t loadRows = loadEnd - loadBegin;
        const uint32_t loadLanes = static_cast<uint32_t>(loadRows * 2);
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(loadRows * 2 * sizeof(uint32_t)), 0, 0, 0};
        DataCopyPadExtParams<uint32_t> padParams{false, 0, 0, 0};
        if (loadRows % 4 == 0) {
            DataCopy(indexLocal, indexGm[loadBegin * 2], loadLanes);
        } else {
            DataCopyPad(indexLocal, indexGm[loadBegin * 2], copyParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(eventIndexMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIndexMte2ToV);
        return loadBegin;
    }

    __aicore__ inline int64_t ReadIndex(uint64_t row, uint64_t loadBegin)
    {
        const uint64_t localOffset = (row - loadBegin) * 2;
        const uint64_t indexBits =
            static_cast<uint64_t>(indexLocal.GetValue(localOffset)) |
            (static_cast<uint64_t>(indexLocal.GetValue(localOffset + 1)) << 32);
        return static_cast<int64_t>(indexBits);
    }

    __aicore__ inline void AdvancePosition()
    {
        ++entry;
        if (entry == indexRows) {
            ++batch;
            entry = 0;
            hasPreviousRow = false;
        }
    }

    __aicore__ inline void ProcessOriginalOutputSlab(
        uint64_t tileBegin, uint64_t tileEnd, uint64_t loadBegin)
    {
        if (slabInFlight) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        }
        for (uint64_t row = tileBegin; row < tileEnd; ++row) {
            const uint64_t currentIndex = static_cast<uint64_t>(ReadIndex(row, loadBegin));
            const uint64_t sourceRow = (batch * sourceRows + currentIndex) * featureCount;
            const uint64_t slabRow = (row - tileBegin) * featureCount;
            DataCopy(
                outputSlabLocal[slabRow],
                srcGm[sourceRow],
                static_cast<uint32_t>(featureCount));
            AdvancePosition();
        }
        SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3);
        DataCopy(
            outGm[tileBegin * featureCount],
            outputSlabLocal,
            static_cast<uint32_t>((tileEnd - tileBegin) * featureCount));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        slabInFlight = true;
    }

    __aicore__ inline void CopyRepeatedSlabRows(uint64_t slabRow, uint32_t runRows)
    {
        SetFlag<HardEvent::MTE2_V>(eventIndexMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIndexMte2ToV);
        LocalTensor<uint32_t> words = outputSlabLocal.template ReinterpretCast<uint32_t>();
        const uint32_t wordsPerRow = static_cast<uint32_t>(featureCount * sizeof(T) / sizeof(uint32_t));
        const uint32_t firstWord = static_cast<uint32_t>(slabRow * sizeof(T) / sizeof(uint32_t));
        uint32_t filled = 1;
        while (filled < runRows) {
            const uint32_t copies = runRows - filled < filled ? runRows - filled : filled;
            DataCopy(words[firstWord + filled * wordsPerRow], words[firstWord], copies * wordsPerRow);
            PipeBarrier<PIPE_V>();
            filled += copies;
        }
    }

    __aicore__ inline void ProcessOutputSlab(
        uint64_t tileBegin, uint64_t tileEnd, uint64_t loadBegin)
    {
        if (slabInFlight) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        }
        bool copiedRows = false;
        for (uint64_t row = tileBegin; row < tileEnd;) {
            const uint64_t currentIndex = static_cast<uint64_t>(ReadIndex(row, loadBegin));
            const uint64_t sourceRow = (batch * sourceRows + currentIndex) * featureCount;
            const uint64_t slabRow = (row - tileBegin) * featureCount;
            uint64_t runEnd = row + 1;
            if (useRunCache != 0 && logicalFeatureCount != 128) {
                const uint64_t batchEnd = row + indexRows - entry;
                const uint64_t limit = batchEnd < tileEnd ? batchEnd : tileEnd;
                while (runEnd < limit &&
                       static_cast<uint64_t>(ReadIndex(runEnd, loadBegin)) == currentIndex) {
                    ++runEnd;
                }
            }
            DataCopy(
                outputSlabLocal[slabRow],
                srcGm[sourceRow],
                static_cast<uint32_t>(featureCount));
            const uint32_t runRows = static_cast<uint32_t>(runEnd - row);
            if (runRows > 1) {
                CopyRepeatedSlabRows(slabRow, runRows);
                copiedRows = true;
            }
            entry += runRows;
            if (entry == indexRows) {
                ++batch;
                entry = 0;
                hasPreviousRow = false;
            }
            row = runEnd;
        }
        SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3);
        if (copiedRows) {
            SetFlag<HardEvent::V_MTE3>(eventSlabVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventSlabVToMte3);
        }
        DataCopy(
            outGm[tileBegin * featureCount],
            outputSlabLocal,
            static_cast<uint32_t>((tileEnd - tileBegin) * featureCount));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        slabInFlight = true;
    }

    __aicore__ inline void CopyGenericFeature(
        uint64_t sourceRow,
        uint64_t outputRow,
        uint64_t featureOffset,
        bool reuseRow)
    {
        const uint64_t remaining = featureCount - featureOffset;
        const uint32_t tileCount = remaining < featureTile
            ? static_cast<uint32_t>(remaining)
            : featureTile;
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(tileCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        const bool alignedCopy = (tileCount * sizeof(T)) % 32 == 0 &&
            (featureOffset * sizeof(T)) % 32 == 0 &&
            (featureCount * sizeof(T)) % 32 == 0;
        if (!reuseRow) {
            if (alignedCopy) {
                DataCopy(rowLocal, srcGm[sourceRow + featureOffset], tileCount);
            } else {
                DataCopyPad(rowLocal, srcGm[sourceRow + featureOffset], copyParams, padParams);
            }
            SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3);
        }
        if (alignedCopy) {
            DataCopy(outGm[outputRow + featureOffset], rowLocal, tileCount);
        } else {
            DataCopyPad(outGm[outputRow + featureOffset], rowLocal, copyParams);
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    }

    __aicore__ inline void ProcessGenericRow(uint64_t row, uint64_t loadBegin)
    {
        const int64_t currentIndex = ReadIndex(row, loadBegin);
        const bool reuseRow = useRunCache != 0 && hasPreviousRow &&
            currentIndex == previousIndex;
        const uint64_t sourceRow =
            (batch * sourceRows + static_cast<uint64_t>(currentIndex)) * featureCount;
        const uint64_t outputRow = row * featureCount;
        for (uint64_t featureOffset = 0; featureOffset < featureCount;
             featureOffset += featureTile) {
            CopyGenericFeature(sourceRow, outputRow, featureOffset, reuseRow);
        }
        previousIndex = currentIndex;
        hasPreviousRow = useRunCache != 0;
        AdvancePosition();
    }

    __aicore__ inline void ProcessGenericTile(
        uint64_t tileBegin, uint64_t tileEnd, uint64_t loadBegin)
    {
        for (uint64_t row = tileBegin; row < tileEnd; ++row) {
            ProcessGenericRow(row, loadBegin);
        }
    }

    __aicore__ inline void DrainOutputSlab()
    {
        if (useOutputSlab && slabInFlight) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        }
    }

    TPipe* pipe = nullptr;
    TBuf<TPosition::VECCALC> rowBuffer;
    LocalTensor<T> rowLocal;
    TBuf<TPosition::VECCALC> indexBuffer;
    LocalTensor<uint32_t> indexLocal;
    TBuf<TPosition::VECCALC> outputSlabBuffer;
    LocalTensor<T> outputSlabLocal;

    GlobalTensor<T> srcGm;
    GlobalTensor<uint32_t> indexGm;
    GlobalTensor<T> outGm;

    uint64_t batchCount = 0;
    uint64_t sourceRows = 0;
    uint64_t indexRows = 0;
    uint64_t featureCount = 0;
    uint64_t logicalFeatureCount = 0;
    uint64_t totalRows = 0;
    uint64_t rowsPerCore = 0;
    uint64_t extraCoreCount = 0;
    uint64_t rowBegin = 0;
    uint64_t rowEnd = 0;
    uint64_t batch = 0;
    uint64_t entry = 0;
    int64_t previousIndex = -1;
    uint32_t featureTile = 0;
    uint32_t coreNum = 0;
    uint32_t useRunCache = 0;
    uint32_t elementBytes = 0;
    uint32_t rowBufferSize = 0;
    uint32_t indexBufferSize = 0;
    uint32_t outputSlabSize = 0;
    uint32_t outputSlabRows = 0;
    bool useOutputSlab = false;
    bool hasPreviousRow = false;
    bool slabInFlight = false;

    event_t eventMte2ToMte3;
    event_t eventMte3ToMte2;
    event_t eventIndexMte2ToV;
    event_t eventSlabVToMte3;
};

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void gather_coo_kernel(
    GM_ADDR src,
    GM_ADDR index,
    GM_ADDR out,
    const GatherCooTilingData tiling)
{
    TPipe pipe;
    GatherCooKernel<T, false> op;
    op.Init(src, index, out, &tiling, &pipe);
    op.Process();
}

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void gather_coo_slab_kernel(
    GM_ADDR src, GM_ADDR index, GM_ADDR out, const GatherCooTilingData tiling)
{
    TPipe pipe;
    GatherCooKernel<T, true> op;
    op.Init(src, index, out, &tiling, &pipe);
    op.Process();
}

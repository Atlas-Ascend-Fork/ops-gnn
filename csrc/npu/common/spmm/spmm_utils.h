/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SPARSE_COMMON_SPMM_UTILS_H
#define OPS_GNN_SPARSE_COMMON_SPMM_UTILS_H

#include <cstdint>

#ifdef OPSGNN_ENABLE_SPMM_KERNEL
#include "kernel_operator.h"
#endif

#ifdef OPSGNN_ENABLE_SPMM_HOST
#include <algorithm>
#include <limits>
#include <optional>
#include <vector>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "common/spmm/host_proof_cache.h"
#endif

#include "tiling/platform/platform_ascendc.h"

namespace ops_gnn {
namespace sparse {

constexpr uint32_t SPMM_BUFFER_NUM = 2;
constexpr uint32_t SPMM_UB_RESERVED = 2048;
constexpr uint32_t SPMM_ALIGN_BYTES = 32;

#ifdef OPSGNN_ENABLE_SPMM_HOST

using SpmmLauncher = void (*)(uint32_t, aclrtStream, void*, void*, void*,
                              void*, void*, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t);
using BspmmLauncher = void (*)(uint32_t, aclrtStream, void*, void*, void*,
                               void*, void*, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t);

struct SpmmHostShape {
    int64_t rows;
    int64_t sources;
    int64_t batches;
    int64_t features;
    int64_t edges;
};

inline void CheckSpmmInputs(const torch::Tensor& indptr,
                            const torch::Tensor& indices,
                            const torch::Tensor& x, const char* api)
{
    for (const auto& tensor : {indptr, indices, x}) {
        TORCH_CHECK(tensor.defined() &&
                    tensor.device().type() == c10::DeviceType::PrivateUse1,
                    api, ": inputs must be NPU tensors");
        TORCH_CHECK(tensor.device() == x.device(),
                    api, ": inputs must be on the same NPU");
    }
    TORCH_CHECK((x.dim() == 2 || x.dim() == 3) && x.scalar_type() == at::kHalf,
                api, ": x must be a two- or three-dimensional float16 tensor");
    TORCH_CHECK(!x.requires_grad(), api, ": autograd is not supported");
    TORCH_CHECK(indptr.dim() == 1 && indices.dim() == 1 && indptr.numel() >= 1,
                api, ": CSR tensors must be one-dimensional; indptr must be nonempty");
    TORCH_CHECK((indptr.scalar_type() == at::kInt || indptr.scalar_type() == at::kLong) &&
                indices.scalar_type() == indptr.scalar_type(),
                api, ": CSR tensors must have the same int32/int64 dtype");
}

inline SpmmHostShape GetSpmmShape(const torch::Tensor& indptr,
                                  const torch::Tensor& indices,
                                  const torch::Tensor& x)
{
    return {indptr.numel() - 1, x.size(0), x.dim() == 3 ? x.size(1) : 1,
            x.size(-1), indices.numel()};
}

inline void CheckSpmmLimits(const SpmmHostShape& shape,
                            int64_t indptrNumel, const char* api)
{
    const uint64_t limit = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(shape.rows < static_cast<int64_t>(limit) &&
                shape.sources <= static_cast<int64_t>(limit) &&
                shape.batches >= 0 && static_cast<uint64_t>(shape.batches) <= limit &&
                shape.features > 0 && static_cast<uint64_t>(shape.features) <= limit / 2 &&
                shape.edges <= static_cast<int64_t>(limit),
                api, ": dimensions exceed uint32 limits or feature_dim is zero");
    uint64_t rowElements = static_cast<uint64_t>(shape.batches) * shape.features;
    TORCH_CHECK(rowElements <= limit / 2 &&
                static_cast<uint64_t>(shape.sources) * rowElements <= limit / 2 &&
                static_cast<uint64_t>(shape.rows) * rowElements <= limit / 2 &&
                static_cast<uint64_t>(indptrNumel) <= limit / 4 &&
                static_cast<uint64_t>(shape.edges) <= limit / 4,
                api, ": address byte offsets exceed uint32 limits");
}

inline void CheckSpmmOut(const std::optional<torch::Tensor>& out,
                         const torch::Tensor& x, const torch::Tensor& indptr,
                         const torch::Tensor& indices,
                         const SpmmHostShape& shape, const char* api)
{
    if (!out) return;
    TORCH_CHECK(out->defined(), api, ": out must be defined");
    bool shapeMatches = out->dim() == x.dim() &&
                        out->size(0) == shape.rows &&
                        out->size(-1) == shape.features;
    if (x.dim() == 3) {
        shapeMatches = shapeMatches && out->size(1) == shape.batches;
    }
    TORCH_CHECK(out->device() == x.device() &&
                out->scalar_type() == at::kHalf && shapeMatches,
                api, ": out must match output shape, dtype and device");
    TORCH_CHECK(out->is_contiguous() && !out->requires_grad(),
                api, ": out must be contiguous and must not require gradients");
    TORCH_CHECK(!out->is_alias_of(x) && !out->is_alias_of(indptr) &&
                !out->is_alias_of(indices),
                api, ": out must not alias any input storage");
}

inline void CheckSpmmCsr(const torch::Tensor& indptr,
                         const torch::Tensor& indices,
                         const SpmmHostShape& shape, const char* api)
{
    CheckCachedSpmmCsr(indptr, indices, shape.rows, shape.sources, shape.edges, api);
}

inline uint64_t GetSpmmPlatform(const SpmmHostShape& shape,
                                uint32_t& blocks, const char* api)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, api, ": platform information unavailable");
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    TORCH_CHECK(ubBytes > SPMM_UB_RESERVED &&
                ubBytes <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                api, ": invalid platform UB capacity");
    uint64_t maxRowBytes = ((ubBytes - SPMM_UB_RESERVED) / 4 /
                            SPMM_ALIGN_BYTES) * SPMM_ALIGN_BYTES;
    TORCH_CHECK(static_cast<uint64_t>(shape.features) <= maxRowBytes / 2,
                api, ": feature_dim exceeds UB limit ", maxRowBytes / 2);
    uint32_t cores = platform->GetCoreNumAiv();
    TORCH_CHECK(cores > 0, api, ": no AIV cores available");
    blocks = static_cast<uint32_t>(std::min<int64_t>(cores, shape.rows));
    return ubBytes;
}

inline torch::Tensor LaunchSpmmCopyLhs(const torch::Tensor& indptr,
    const torch::Tensor& indices, const torch::Tensor& x,
    const std::optional<torch::Tensor>& out, const SpmmHostShape& shape,
    uint32_t blocks, uint64_t ubBytes, SpmmLauncher launch,
    BspmmLauncher batchLaunch)
{
    std::vector<int64_t> outputShape = x.dim() == 3 ?
        std::vector<int64_t>{shape.rows, shape.batches, shape.features} :
        std::vector<int64_t>{shape.rows, shape.features};
    auto result = out ? *out : torch::empty(outputShape, x.options());
    if (shape.rows == 0 || shape.batches == 0) return result;
    auto split = torch::arange(static_cast<int64_t>(blocks) + 1,
                               indptr.options().dtype(at::kLong));
    split = at::floor_divide(split * shape.rows, blocks).to(at::kInt);
    auto ptr = indptr.to(at::kInt).contiguous();
    auto idx = indices.to(at::kInt).contiguous();
    auto features = x.contiguous();
    uint32_t hasNan = CachedHasNan(x);
    auto stream = c10_npu::getCurrentNPUStream(x.get_device()).stream();
    if (x.dim() == 3) {
        batchLaunch(blocks, stream, features.data_ptr(), result.data_ptr(),
                    ptr.data_ptr(), idx.data_ptr(), split.data_ptr(),
                    shape.rows, shape.sources, shape.features, shape.edges,
                    shape.batches, ubBytes, hasNan);
    } else {
        launch(blocks, stream, features.data_ptr(), result.data_ptr(),
               ptr.data_ptr(), idx.data_ptr(), split.data_ptr(), shape.rows,
               shape.sources, shape.features, shape.edges, ubBytes, hasNan);
    }
    return result;
}

inline torch::Tensor RunSpmmCopyLhs(const torch::Tensor& indptr,
    const torch::Tensor& indices, const torch::Tensor& x,
    const std::optional<torch::Tensor>& out, const char* api,
    SpmmLauncher launch, BspmmLauncher batchLaunch)
{
    CheckSpmmInputs(indptr, indices, x, api);
    c10::DeviceGuard guard(x.device());
    SpmmHostShape shape = GetSpmmShape(indptr, indices, x);
    CheckSpmmLimits(shape, indptr.numel(), api);
    CheckSpmmOut(out, x, indptr, indices, shape, api);
    SpmmProofScope proofs(out.has_value());
    CheckSpmmCsr(indptr, indices, shape, api);
    uint32_t blocks = 0;
    uint64_t ubBytes = GetSpmmPlatform(shape, blocks, api);
    return LaunchSpmmCopyLhs(indptr, indices, x, out, shape, blocks,
                             ubBytes, launch, batchLaunch);
}

#endif  // OPSGNN_ENABLE_SPMM_HOST

#ifdef OPSGNN_ENABLE_SPMM_KERNEL

enum class SpmmReduce : uint8_t {
    SUM,
    MAX,
    MIN,
};

template <SpmmReduce Reduce>
struct SpmmReducer {
    __aicore__ static inline half InitialValue(bool empty)
    {
        if (empty || Reduce == SpmmReduce::SUM) return half(0.0f);
        if constexpr (Reduce == SpmmReduce::MAX) {
            return half(-__builtin_inff());
        }
        return half(__builtin_inff());
    }

    __aicore__ static inline void Apply(AscendC::LocalTensor<half>& accum,
                                         AscendC::LocalTensor<half>& value,
                                         uint32_t count)
    {
        if constexpr (Reduce == SpmmReduce::MAX) {
            AscendC::Max(accum, accum, value, count);
        } else if constexpr (Reduce == SpmmReduce::MIN) {
            AscendC::Min(accum, accum, value, count);
        } else {
            AscendC::Add(accum, accum, value, count);
        }
    }
};

template <SpmmReduce Reduce>
class SpmmCopyLhsKernel {
public:
    __aicore__ inline void Init(GM_ADDR featureData, GM_ADDR outputData,
        GM_ADDR indptrData, GM_ADDR indicesData, GM_ADDR rowSplitData,
        uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
        uint32_t nonZeroCount, uint32_t ubBytes, uint32_t hasNan,
        AscendC::TPipe* pipe)
    {
        Init(featureData, outputData, indptrData, indicesData, rowSplitData,
             numDstRows, numSrcRows, featureDim, nonZeroCount, 1, ubBytes,
             hasNan, pipe);
    }

    __aicore__ inline void Init(GM_ADDR featureData, GM_ADDR outputData,
        GM_ADDR indptrData, GM_ADDR indicesData, GM_ADDR rowSplitData,
        uint32_t numDstRows, uint32_t numSrcRows, uint32_t featureDim,
        uint32_t nonZeroCount, uint32_t batches, uint32_t ubBytes,
        uint32_t hasNan, AscendC::TPipe* pipe)
    {
        M = numDstRows;
        K = numSrcRows;
        N = featureDim;
        nnz = nonZeroCount;
        batchCount = batches;
        this->hasNan = hasNan;
        startRow = 0;
        localRowCount = 0;
        if (IsInvalidTiling(ubBytes) || !InitWorkRange(rowSplitData)) return;
        InitGlobalBuffers(featureData, outputData, indptrData, indicesData);
        InitLocalBuffers(ubBytes, pipe);
    }

    __aicore__ inline void Process()
    {
        uint32_t rowEnd = startRow + localRowCount;
        for (uint32_t row = startRow; row < rowEnd; ++row) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline bool IsInvalidTiling(uint32_t ubBytes) const
    {
        if (ubBytes <= SPMM_UB_RESERVED || N == 0 || batchCount == 0) {
            return true;
        }
        uint32_t maxN = ((ubBytes - SPMM_UB_RESERVED) / 4 /
                         SPMM_ALIGN_BYTES * SPMM_ALIGN_BYTES) / sizeof(half);
        if (N > maxN || batchCount > 0xffffffffU / N) return true;
        uint32_t rowElements = batchCount * N;
        return K > 0xffffffffU / rowElements ||
               M > 0xffffffffU / rowElements || M == 0xffffffffU;
    }

    __aicore__ inline bool InitWorkRange(GM_ADDR rowSplitData)
    {
        uint32_t block = AscendC::GetBlockIdx();
        uint32_t blocks = AscendC::GetBlockNum();
        rowSplitGm.SetGlobalBuffer((__gm__ uint32_t*)rowSplitData, blocks + 1);
        startRow = rowSplitGm.GetValue(block);
        uint32_t endRow = rowSplitGm.GetValue(block + 1);
        if (endRow < startRow || endRow > M) return false;
        localRowCount = endRow - startRow;
        return true;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR featureData,
        GM_ADDR outputData, GM_ADDR indptrData, GM_ADDR indicesData)
    {
        uint32_t inputSize = K * batchCount * N;
        uint32_t outputSize = M * batchCount * N;
        featureGm.SetGlobalBuffer((__gm__ half*)featureData, inputSize);
        featureBitsGm.SetGlobalBuffer((__gm__ uint16_t*)featureData, inputSize);
        outputGm.SetGlobalBuffer((__gm__ half*)outputData, outputSize);
        indptrGm.SetGlobalBuffer((__gm__ uint32_t*)indptrData, M + 1);
        indicesGm.SetGlobalBuffer((__gm__ uint32_t*)indicesData, nnz);
    }

    __aicore__ inline void InitLocalBuffers(uint32_t ubBytes,
                                             AscendC::TPipe* pipe)
    {
        rowBytes = N * sizeof(half);
        rowAlignedBytes = (rowBytes + SPMM_ALIGN_BYTES - 1) /
                          SPMM_ALIGN_BYTES * SPMM_ALIGN_BYTES;
        rowAlignedElems = rowAlignedBytes / sizeof(half);
        rightPadding = rowAlignedElems - N;
        uint32_t accumBytes = SPMM_BUFFER_NUM * rowAlignedBytes;
        uint32_t available = ubBytes - SPMM_UB_RESERVED;
        uint32_t remaining = available > accumBytes ? available - accumBytes : 0;
        edgeBatchSize = remaining / (SPMM_BUFFER_NUM * rowAlignedBytes);
        if (edgeBatchSize == 0) {
            localRowCount = 0;
            return;
        }
        pipe->InitBuffer(accumQueue, SPMM_BUFFER_NUM, rowAlignedBytes);
        pipe->InitBuffer(featureQueue, SPMM_BUFFER_NUM,
                         edgeBatchSize * rowAlignedBytes);
    }

    __aicore__ inline void ProcessRow(uint32_t row)
    {
        uint32_t begin = indptrGm.GetValue(row);
        uint32_t end = indptrGm.GetValue(row + 1);
        for (uint32_t batch = 0; batch < batchCount; ++batch) {
            AscendC::LocalTensor<half> accum = accumQueue.AllocTensor<half>();
            AscendC::Duplicate<half>(
                accum, SpmmReducer<Reduce>::InitialValue(begin == end),
                rowAlignedElems);
            for (uint32_t edge = begin; edge < end; edge += edgeBatchSize) {
                uint32_t count = edgeBatchSize < end - edge ?
                                 edgeBatchSize : end - edge;
                CopyIn(edge, count, batch);
                Compute(accum, count);
            }
            RepairNan(accum, begin, end, batch);
            accumQueue.EnQue(accum);
            CopyOut(row, batch);
        }
    }

    __aicore__ inline void CopyIn(uint32_t edgeStart, uint32_t count,
                                  uint32_t batch)
    {
        AscendC::LocalTensor<half> features = featureQueue.AllocTensor<half>();
        AscendC::DataCopyExtParams copy = {1, rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> pad = {
            true, 0, static_cast<uint8_t>(rightPadding), half(0.0f)};
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t source = indicesGm.GetValue(edgeStart + i);
            uint32_t offset = (source * batchCount + batch) * N;
            AscendC::DataCopyPad<half>(features[i * rowAlignedElems],
                                       featureGm[offset], copy, pad);
        }
        featureQueue.EnQue(features);
    }

    __aicore__ inline void Compute(AscendC::LocalTensor<half>& accum,
                                    uint32_t count)
    {
        AscendC::LocalTensor<half> features = featureQueue.DeQue<half>();
        for (uint32_t i = 0; i < count; ++i) {
            AscendC::LocalTensor<half> value = features[i * rowAlignedElems];
            SpmmReducer<Reduce>::Apply(accum, value, rowAlignedElems);
        }
        featureQueue.FreeTensor(features);
    }

    __aicore__ inline void RepairNan(AscendC::LocalTensor<half>& accum,
                                      uint32_t begin, uint32_t end,
                                      uint32_t batch)
    {
        if (!hasNan || begin == end) return;
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t edge = begin; edge < end; ++edge) {
            uint32_t source = indicesGm.GetValue(edge);
            uint32_t base = (source * batchCount + batch) * N;
            for (uint32_t feature = 0; feature < N; ++feature) {
                uint16_t bits = featureBitsGm.GetValue(base + feature);
                if ((bits & 0x7c00U) == 0x7c00U && (bits & 0x03ffU) != 0) {
                    accum.SetValue(feature, featureGm.GetValue(base + feature));
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t batch)
    {
        AscendC::LocalTensor<half> accum = accumQueue.DeQue<half>();
        AscendC::DataCopyExtParams copy = {1, rowBytes, 0, 0, 0};
        uint32_t offset = (row * batchCount + batch) * N;
        AscendC::DataCopyPad<half>(outputGm[offset], accum, copy);
        accumQueue.FreeTensor(accum);
    }

    uint32_t M, K, N, nnz, batchCount, hasNan;
    uint32_t startRow, localRowCount, edgeBatchSize;
    uint32_t rowBytes, rowAlignedBytes, rowAlignedElems, rightPadding;
    AscendC::TQue<AscendC::TPosition::VECOUT, SPMM_BUFFER_NUM> accumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, SPMM_BUFFER_NUM> featureQueue;
    AscendC::GlobalTensor<half> featureGm, outputGm;
    AscendC::GlobalTensor<uint16_t> featureBitsGm;
    AscendC::GlobalTensor<uint32_t> indptrGm, indicesGm, rowSplitGm;
};

template <SpmmReduce Reduce>
using BspmmCopyLhsKernel = SpmmCopyLhsKernel<Reduce>;

#endif  // OPSGNN_ENABLE_SPMM_KERNEL

}  // namespace sparse
}  // namespace ops_gnn


#ifdef OPSGNN_ENABLE_SPMM_KERNEL

#define OPSGNN_DEFINE_SPMM_KERNEL(functionName, reduceKind)                 \
extern "C" __global__ __aicore__ void functionName(                       \
    GM_ADDR featureData, GM_ADDR outputData, GM_ADDR indptrData,           \
    GM_ADDR indicesData, GM_ADDR rowSplitData, uint32_t numDstRows,        \
    uint32_t numSrcRows, uint32_t featureDim, uint32_t nonZeroCount,       \
    uint32_t ubBytes, uint32_t hasNan)                                     \
{                                                                          \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                        \
    AscendC::TPipe pipe;                                                   \
    ops_gnn::sparse::SpmmCopyLhsKernel<reduceKind> processor;              \
    processor.Init(featureData, outputData, indptrData, indicesData,       \
                   rowSplitData, numDstRows, numSrcRows, featureDim,       \
                   nonZeroCount, ubBytes, hasNan, &pipe);                  \
    processor.Process();                                                   \
}

#define OPSGNN_DEFINE_BSPMM_KERNEL(functionName, reduceKind)               \
extern "C" __global__ __aicore__ void functionName(                       \
    GM_ADDR featureData, GM_ADDR outputData, GM_ADDR indptrData,           \
    GM_ADDR indicesData, GM_ADDR rowSplitData, uint32_t numDstRows,        \
    uint32_t numSrcRows, uint32_t featureDim, uint32_t nonZeroCount,       \
    uint32_t batchCount, uint32_t ubBytes, uint32_t hasNan)                \
{                                                                          \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                        \
    AscendC::TPipe pipe;                                                   \
    ops_gnn::sparse::BspmmCopyLhsKernel<reduceKind> processor;             \
    processor.Init(featureData, outputData, indptrData, indicesData,       \
                   rowSplitData, numDstRows, numSrcRows, featureDim,       \
                   nonZeroCount, batchCount, ubBytes, hasNan, &pipe);      \
    processor.Process();                                                   \
}

#endif  // OPSGNN_ENABLE_SPMM_KERNEL

#endif  // OPS_GNN_SPARSE_COMMON_SPMM_UTILS_H

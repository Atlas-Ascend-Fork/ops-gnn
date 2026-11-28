/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "csr_row_wise_per_etype_sampling_uniform.h"

#include <algorithm>
#include <acl/acl.h>
#include <cstdint>
#include <limits>
#include <vector>

#include "csr_row_wise_per_etype_sampling_uniform/op_kernel/arch22/csr_row_wise_per_etype_sampling_uniform.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {
namespace csr_per_etype {
namespace {

constexpr uint32_t kUbReservedBytes = 2 * 1024;

void CheckNpuTensor(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), "csr_row_wise_per_etype_sampling_uniform: ", name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "csr_row_wise_per_etype_sampling_uniform: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.dim() == 1, "csr_row_wise_per_etype_sampling_uniform: ", name, " must be 1-D");
}

std::vector<uint32_t> BuildBalancedPartitions(
    const std::vector<uint32_t>& weights, uint32_t numParts)
{
    std::vector<uint32_t> boundaries(numParts + 1, 0);
    if (numParts == 0) {
        return boundaries;
    }
    const uint32_t itemCount = static_cast<uint32_t>(weights.size());
    boundaries[numParts] = itemCount;
    if (itemCount == 0) {
        return boundaries;
    }
    if (itemCount <= numParts) {
        for (uint32_t i = 0; i <= itemCount; ++i) {
            boundaries[i] = i;
        }
        for (uint32_t i = itemCount + 1; i <= numParts; ++i) {
            boundaries[i] = itemCount;
        }
        return boundaries;
    }

    std::vector<double> prefix(itemCount + 1, 0.0);
    for (uint32_t i = 0; i < itemCount; ++i) {
        prefix[i + 1] = prefix[i] + weights[i];
    }
    const double total = prefix[itemCount];
    if (total <= 0.0) {
        for (uint32_t part = 1; part < numParts; ++part) {
            boundaries[part] = part * itemCount / numParts;
        }
        return boundaries;
    }
    for (uint32_t part = 1; part < numParts; ++part) {
        const double target = total * part / numParts;
        auto it = std::lower_bound(prefix.begin(), prefix.end(), target);
        boundaries[part] = static_cast<uint32_t>(it - prefix.begin());
    }
    for (uint32_t part = 1; part < numParts; ++part) {
        if (boundaries[part] < boundaries[part - 1]) {
            boundaries[part] = boundaries[part - 1];
        }
        if (boundaries[part] > itemCount) {
            boundaries[part] = itemCount;
        }
    }
    return boundaries;
}

template <typename T>
torch::Tensor Uint32ToTensor(const std::vector<uint32_t>& values, const torch::Tensor& like)
{
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(like.device());
    auto cpu = torch::from_blob(const_cast<uint32_t*>(values.data()),
        {static_cast<int64_t>(values.size())}, options.device(torch::kCPU)).clone();
    return cpu.to(like.options().dtype(c10::CppTypeToScalarType<T>::value));
}


struct CsrSlice {
    torch::Tensor indptr;
    torch::Tensor indices;
    torch::Tensor localData;
};

CsrSlice BuildCsrSliceOnHost(const int64_t* rowHost, const int64_t* colHost,
    int64_t off, int64_t len, int64_t numNodes, const torch::Tensor& like)
{
    std::vector<int64_t> counts(static_cast<size_t>(numNodes), 0);
    for (int64_t i = 0; i < len; ++i) {
        ++counts[static_cast<size_t>(rowHost[off + i])];
    }
    std::vector<int64_t> indptr(static_cast<size_t>(numNodes) + 1, 0);
    for (int64_t k = 0; k < numNodes; ++k) {
        indptr[static_cast<size_t>(k) + 1] =
            indptr[static_cast<size_t>(k)] + counts[static_cast<size_t>(k)];
    }
    std::vector<int64_t> cursor(indptr.begin(), indptr.end() - 1);
    std::vector<int64_t> indices(static_cast<size_t>(len), 0);
    std::vector<int64_t> localData(static_cast<size_t>(len), 0);
    for (int64_t i = 0; i < len; ++i) {
        const int64_t pos = cursor[static_cast<size_t>(rowHost[off + i])]++;
        indices[static_cast<size_t>(pos)] = colHost[off + i];
        localData[static_cast<size_t>(pos)] = i;
    }
    auto cpuLong = torch::TensorOptions().dtype(torch::kLong);
    auto toDevice = [&](std::vector<int64_t>& v) {
        return torch::from_blob(v.data(), {static_cast<int64_t>(v.size())}, cpuLong)
            .clone().to(like.options());
    };
    return {toDevice(indptr), toDevice(indices), toDevice(localData)};
}

struct SampledSlice {
    torch::Tensor outPtr;
    torch::Tensor outRows;
    torch::Tensor outCols;
    torch::Tensor outData;
};

SampledSlice MakeEmptySlice(const torch::Tensor& indptr, int64_t numRows)
{
    auto options = indptr.options();
    SampledSlice s;
    s.outPtr = torch::zeros({numRows + 1}, options);
    s.outRows = torch::empty({0}, options);
    s.outCols = torch::empty({0}, options);
    s.outData = torch::empty({0}, options);
    return s;
}

std::vector<uint32_t> ComputePickCounts(const torch::Tensor& indptr,
    const torch::Tensor& rows, int64_t numSamples, bool replace,
    uint32_t& maxReservoirPicks)
{
    const bool selectAll = numSamples == -1;
    const bool doReplace = replace && !selectAll;
    const int64_t numRows = rows.numel();
    const int64_t numTotalRows = indptr.numel() - 1;
    auto rowsLong = rows.to(torch::kLong);
    auto rowsValid = (rowsLong >= 0) & (rowsLong < numTotalRows);
    auto rowsClamped = rowsLong.clamp(0, numTotalRows - 1);
    auto degrees = (indptr.index({rowsClamped + 1}) - indptr.index({rowsClamped})).to(torch::kLong);
    degrees = torch::where(rowsValid, degrees, torch::zeros_like(degrees));
    auto degreesCpu = degrees.to(torch::kCPU).contiguous();
    const int64_t* degPtr = degreesCpu.data_ptr<int64_t>();
    std::vector<uint32_t> picks(static_cast<size_t>(numRows));
    maxReservoirPicks = 0;
    for (int64_t i = 0; i < numRows; ++i) {
        TORCH_CHECK(degPtr[i] >= 0,
                    "csr_row_wise_per_etype_sampling_uniform: negative row degree at row ", i);
        const uint32_t degree = static_cast<uint32_t>(degPtr[i]);
        picks[i] = selectAll ? degree
                 : doReplace ? (degree == 0 ? 0u : static_cast<uint32_t>(numSamples))
                             : std::min(static_cast<uint32_t>(numSamples), degree);
        if (!selectAll && !doReplace && picks[i] < degree) {
            maxReservoirPicks = std::max(maxReservoirPicks, picks[i]);
        }
    }
    return picks;
}

struct SlicePlan {
    std::vector<uint32_t> rowSplit;
    std::vector<uint32_t> outStarts;
    std::vector<uint32_t> prefix;
    uint64_t maxOutput = 0;
};

SlicePlan BuildSlicePlan(const std::vector<uint32_t>& picks, uint32_t blocks)
{
    SlicePlan plan;
    plan.rowSplit = BuildBalancedPartitions(picks, blocks);
    std::vector<uint64_t> prefix64(picks.size() + 1, 0);
    for (size_t i = 0; i < picks.size(); ++i) {
        prefix64[i + 1] = prefix64[i] + picks[i];
    }
    plan.maxOutput = prefix64.back();
    TORCH_CHECK(plan.maxOutput <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                "csr_row_wise_per_etype_sampling_uniform: output size exceeds int32 range");
    plan.prefix.assign(prefix64.begin(), prefix64.end());
    plan.outStarts.assign(blocks + 1, 0);
    for (uint32_t b = 0; b <= blocks; ++b) {
        plan.outStarts[b] = plan.prefix[plan.rowSplit[b]];
    }
    return plan;
}

CsrRowWisePerEtypeSamplingUniformTiling MakeTiling(
    int64_t numRows, int64_t numSamples, bool selectAll, bool doReplace,
    int64_t seed, int64_t numTotalRows, uint32_t ubAvailable,
    uint32_t elemSize, uint32_t pickStride)
{
    CsrRowWisePerEtypeSamplingUniformTiling tiling{};
    tiling.numRows = static_cast<uint32_t>(numRows);
    tiling.numSamples = selectAll ? 0u : static_cast<uint32_t>(numSamples);
    tiling.replace = doReplace ? 1u : 0u;
    tiling.hasData = 1u;
    tiling.seed = static_cast<uint32_t>(seed);
    tiling.selectAll = selectAll ? 1u : 0u;
    tiling.numTotalRows = static_cast<uint32_t>(numTotalRows);
    tiling.ubAvailable = ubAvailable;
    tiling.dtypeBytes = elemSize;
    tiling.pickStride = pickStride;
    return tiling;
}

SampledSlice SampleOneEtype(
    const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::Tensor& localData, const torch::Tensor& rows,
    int64_t numSamples, bool replace, int64_t seed,
    uint32_t blocks, uint32_t ubAvailable, uint32_t elemSize,
    aclrtStream stream)
{
    const bool selectAll = numSamples == -1;
    const bool doReplace = replace && !selectAll;
    const int64_t numRows = rows.numel();
    const int64_t numTotalRows = indptr.numel() - 1;
    if (numRows == 0 || numTotalRows == 0) {
        return MakeEmptySlice(indptr, numRows);
    }

    uint32_t maxReservoirPicks = 0;
    const auto picks = ComputePickCounts(indptr, rows, numSamples, replace, maxReservoirPicks);
    TORCH_CHECK(elemSize > 0, "csr_row_wise_per_etype_sampling_uniform: invalid element size");
    const uint32_t elemBytes = elemSize == 0 ? 1u : elemSize;
    const uint32_t windowElems = ubAvailable / kUbInstances / elemBytes;
    const bool needPickScratch = maxReservoirPicks > windowElems;
    const auto plan = BuildSlicePlan(picks, blocks);
    if (plan.maxOutput == 0) {
        return MakeEmptySlice(indptr, numRows);
    }

    auto options = indptr.options();
    auto rowSplit = Uint32ToTensor<uint32_t>(plan.rowSplit, indptr);
    auto outStartsTensor = Uint32ToTensor<uint32_t>(plan.outStarts, indptr);
    auto outPtr = indptr.scalar_type() == at::ScalarType::Long
        ? Uint32ToTensor<int64_t>(plan.prefix, indptr)
        : Uint32ToTensor<int32_t>(plan.prefix, indptr);
    auto outRows = torch::empty({static_cast<int64_t>(plan.maxOutput)}, options);
    auto outCols = torch::empty({static_cast<int64_t>(plan.maxOutput)}, options);
    auto outData = torch::empty({static_cast<int64_t>(plan.maxOutput)}, options);
    auto pickScratch = needPickScratch
        ? torch::zeros({static_cast<int64_t>(blocks) * maxReservoirPicks}, options)
        : torch::empty({0}, options);

    const auto tiling = MakeTiling(numRows, numSamples, selectAll, doReplace, seed,
        numTotalRows, ubAvailable, elemSize,
        needPickScratch ? maxReservoirPicks : 0u);

    void* pickScratchPtr = needPickScratch ? pickScratch.data_ptr() : nullptr;
    TORCH_CHECK(aclrtSynchronizeDevice() == ACL_SUCCESS, "pre-launch device sync failed");
    LaunchUniformSampling(blocks, stream, indptr.data_ptr(), indices.data_ptr(),
        localData.data_ptr(), rows.data_ptr(), outPtr.data_ptr(), outRows.data_ptr(),
        outCols.data_ptr(), outData.data_ptr(), rowSplit.data_ptr(),
        outStartsTensor.data_ptr(), pickScratchPtr, tiling);
    TORCH_CHECK(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "post-launch stream sync failed");

    SampledSlice s;
    s.outPtr = outPtr;
    s.outRows = outRows;
    s.outCols = outCols;
    s.outData = outData;
    return s;
}

}  // namespace
}  // namespace csr_per_etype

using namespace csr_per_etype;

struct EtypeContext {
    int64_t numRows = 0;
    int64_t nnz = 0;
    int64_t numNodes = 0;
    bool hasData = false;
    uint32_t blocks = 1;
    uint32_t ubAvailable = 0;
    uint32_t elemSize = 0;
    aclrtStream stream = nullptr;
};

void ValidateEtypeInputs(const torch::Tensor& row, const torch::Tensor& col,
    const torch::Tensor& rows, const std::vector<int64_t>& eid2etypeOffset,
    const std::vector<int64_t>& numSamples, const c10::optional<torch::Tensor>& data,
    EtypeContext& ctx)
{
    CheckNpuTensor(row, "row");
    CheckNpuTensor(col, "col");
    CheckNpuTensor(rows, "rows");
    TORCH_CHECK(row.scalar_type() == at::ScalarType::Int ||
                row.scalar_type() == at::ScalarType::Long,
                "csr_row_wise_per_etype_sampling_uniform: row dtype must be int32 or int64");
    TORCH_CHECK(col.scalar_type() == row.scalar_type() &&
                rows.scalar_type() == row.scalar_type(),
                "csr_row_wise_per_etype_sampling_uniform: row/col/rows dtypes must match");
    TORCH_CHECK(row.device() == col.device() && row.device() == rows.device(),
                "csr_row_wise_per_etype_sampling_uniform: all inputs must be on the same NPU");
    TORCH_CHECK(row.numel() == col.numel(),
                "csr_row_wise_per_etype_sampling_uniform: row and col must have the same length");
    TORCH_CHECK(eid2etypeOffset.size() == numSamples.size() + 1,
                "csr_row_wise_per_etype_sampling_uniform: eid2etype_offset must have "
                "one entry per edge type plus the total");
    TORCH_CHECK(!eid2etypeOffset.empty() && eid2etypeOffset.front() == 0,
                "csr_row_wise_per_etype_sampling_uniform: eid2etype_offset must start at 0");
    ctx.nnz = row.numel();
    ctx.numRows = rows.numel();
    TORCH_CHECK(eid2etypeOffset.back() == ctx.nnz,
                "csr_row_wise_per_etype_sampling_uniform: last eid2etype_offset must equal nnz");
    for (size_t e = 0; e < numSamples.size(); ++e) {
        TORCH_CHECK(eid2etypeOffset[e] <= eid2etypeOffset[e + 1],
                    "csr_row_wise_per_etype_sampling_uniform: eid2etype_offset must be non-decreasing");
        TORCH_CHECK(numSamples[e] >= -1,
                    "csr_row_wise_per_etype_sampling_uniform: num_samples must be >= -1");
    }
    if (data.has_value()) {
        CheckNpuTensor(data.value(), "data");
        TORCH_CHECK(data.value().scalar_type() == row.scalar_type(),
                    "csr_row_wise_per_etype_sampling_uniform: data dtype must match row");
        TORCH_CHECK(data.value().device() == row.device(),
                    "csr_row_wise_per_etype_sampling_uniform: data must be on the same NPU");
        TORCH_CHECK(data.value().numel() == ctx.nnz,
                    "csr_row_wise_per_etype_sampling_uniform: data length must match row");
        ctx.hasData = true;
    }
}

void ResolveEtypeLaunchConfig(const torch::Tensor& row, EtypeContext& ctx)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr,
                "csr_row_wise_per_etype_sampling_uniform: failed to initialize AscendC platform");
    ctx.blocks = std::max<uint32_t>(platform->GetCoreNumAiv(), 1);
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    ctx.ubAvailable = ubBytes > kUbReservedBytes
        ? static_cast<uint32_t>(ubBytes - kUbReservedBytes) : 0;
    TORCH_CHECK(ctx.ubAvailable > 0, "csr_row_wise_per_etype_sampling_uniform: invalid UB size");
    ctx.elemSize = static_cast<uint32_t>(row.element_size());
    TORCH_CHECK(ctx.elemSize > 0, "csr_row_wise_per_etype_sampling_uniform: invalid element size");
    ctx.stream = c10_npu::getCurrentNPUStream(row.get_device()).stream();
}

void ValidateIdRange(const torch::Tensor& row, const torch::Tensor& col, EtypeContext& ctx)
{
    TORCH_CHECK(row.min().item<int64_t>() >= 0,
                "csr_row_wise_per_etype_sampling_uniform: row ids must be non-negative");
    TORCH_CHECK(col.min().item<int64_t>() >= 0,
                "csr_row_wise_per_etype_sampling_uniform: col ids must be non-negative");
    ctx.numNodes = row.max().item<int64_t>() + 1;
    TORCH_CHECK(ctx.numNodes <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "csr_row_wise_per_etype_sampling_uniform: too many nodes");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SampleAllEtypes(
    const torch::Tensor& row, const torch::Tensor& col, const torch::Tensor& rows,
    const std::vector<int64_t>& eid2etypeOffset,
    const std::vector<int64_t>& numSamples, bool replace,
    const c10::optional<torch::Tensor>& data, int64_t seed, const EtypeContext& ctx)
{
    auto options = row.options();
    auto empty = torch::empty({0}, options);
    auto rowCpu = row.to(torch::kCPU).to(torch::kLong).contiguous();
    auto colCpu = col.to(torch::kCPU).to(torch::kLong).contiguous();
    aclrtSynchronizeDevice();
    const int64_t* rowHost = rowCpu.data_ptr<int64_t>();
    const int64_t* colHost = colCpu.data_ptr<int64_t>();
    std::vector<torch::Tensor> outRowsVec;
    std::vector<torch::Tensor> outColsVec;
    std::vector<torch::Tensor> outDataVec;
    const int64_t numEtypes = static_cast<int64_t>(numSamples.size());
    for (int64_t e = 0; e < numEtypes; ++e) {
        const int64_t off = eid2etypeOffset[e];
        const int64_t len = eid2etypeOffset[e + 1] - off;
        if (len == 0 || numSamples[e] == 0) {
            continue;
        }
        auto csr = BuildCsrSliceOnHost(rowHost, colHost, off, len, ctx.numNodes, row);
        const uint32_t seedE = static_cast<uint32_t>(seed) +
            static_cast<uint32_t>(e) * 0x9e3779b9u;
        auto sampled = SampleOneEtype(csr.indptr, csr.indices, csr.localData, rows,
            numSamples[e], replace, static_cast<int64_t>(seedE),
            ctx.blocks, ctx.ubAvailable, ctx.elemSize, ctx.stream);
        if (sampled.outRows.numel() == 0) {
            continue;
        }
        auto global = sampled.outData.to(torch::kLong) + off;
        torch::Tensor outData;
        if (ctx.hasData) {
            outData = data.value().index_select(0, global);
        } else {
            outData = global.to(row.scalar_type());
        }
        outRowsVec.push_back(sampled.outRows);
        outColsVec.push_back(sampled.outCols);
        outDataVec.push_back(outData);
    }

    if (outRowsVec.empty()) {
        return std::make_tuple(empty, torch::empty({0}, options), torch::empty({0}, options));
    }
    return std::make_tuple(torch::cat(outRowsVec), torch::cat(outColsVec),
                           torch::cat(outDataVec));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CsrRowWisePerEtypeSamplingUniform(
    const torch::Tensor& row, const torch::Tensor& col, const torch::Tensor& rows,
    const std::vector<int64_t>& eid2etypeOffset,
    const std::vector<int64_t>& numSamples,
    bool replace, const c10::optional<torch::Tensor>& data, int64_t seed)
{
    EtypeContext ctx;
    ValidateEtypeInputs(row, col, rows, eid2etypeOffset, numSamples, data, ctx);
    auto options = row.options();
    auto empty = torch::empty({0}, options);
    if (ctx.numRows == 0 || ctx.nnz == 0) {
        return std::make_tuple(empty, torch::empty({0}, options), torch::empty({0}, options));
    }
    ValidateIdRange(row, col, ctx);
    ResolveEtypeLaunchConfig(row, ctx);
    return SampleAllEtypes(row, col, rows, eid2etypeOffset, numSamples, replace,
                           data, seed, ctx);
}

}  // namespace opsgnn

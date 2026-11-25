/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "csr_row_wise_sampling_uniform_fused.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "csr_row_wise_sampling_uniform_fused/op_kernel/arch22/csr_row_wise_sampling_uniform_fused.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {
namespace csr_uniform_fused {
namespace {

constexpr uint32_t kUbReservedBytes = 2 * 1024;

void CheckNpuTensor(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), "csr_row_wise_sampling_uniform_fused: ", name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "csr_row_wise_sampling_uniform_fused: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.dim() == 1, "csr_row_wise_sampling_uniform_fused: ", name, " must be 1-D");
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

void UpdateSeedMapping(const torch::Tensor& seedMapping, const torch::Tensor& rowsLong,
    const torch::Tensor& rowsValid, int64_t numRows)
{
    auto positions = torch::arange(numRows, seedMapping.options().dtype(torch::kLong));
    auto validRows = rowsLong.masked_select(rowsValid);
    auto validPos = positions.masked_select(rowsValid).to(seedMapping.scalar_type());
    seedMapping.index_copy_(0, validRows, validPos);
}

void ValidateSignature(const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::Tensor& rows, int64_t numSamples)
{
    CheckNpuTensor(indptr, "indptr");
    CheckNpuTensor(indices, "indices");
    CheckNpuTensor(rows, "rows");
    TORCH_CHECK(indptr.scalar_type() == at::ScalarType::Int ||
                indptr.scalar_type() == at::ScalarType::Long,
                "csr_row_wise_sampling_uniform_fused: indptr dtype must be int32 or int64");
    TORCH_CHECK(indices.scalar_type() == indptr.scalar_type(),
                "csr_row_wise_sampling_uniform_fused: indices dtype must match indptr");
    TORCH_CHECK(rows.scalar_type() == indptr.scalar_type(),
                "csr_row_wise_sampling_uniform_fused: rows dtype must match indptr");
    TORCH_CHECK(indptr.device() == indices.device() && indptr.device() == rows.device(),
                "csr_row_wise_sampling_uniform_fused: all inputs must be on the same NPU");
    TORCH_CHECK(numSamples >= -1, "csr_row_wise_sampling_uniform_fused: num_samples must be >= -1");
}

bool CheckOptionalData(const c10::optional<torch::Tensor>& data, const torch::Tensor& indptr,
    const torch::Tensor& indices)
{
    if (!data.has_value()) {
        return false;
    }
    CheckNpuTensor(data.value(), "data");
    TORCH_CHECK(data.value().scalar_type() == indptr.scalar_type(),
                "csr_row_wise_sampling_uniform_fused: data dtype must match indptr");
    TORCH_CHECK(data.value().numel() == indices.numel(),
                "csr_row_wise_sampling_uniform_fused: data must have the same length as indices");
    return true;
}

bool CheckOptionalSeedMapping(const c10::optional<torch::Tensor>& seedMapping,
    const torch::Tensor& indptr, int64_t numTotalRows)
{
    if (!seedMapping.has_value()) {
        return false;
    }
    CheckNpuTensor(seedMapping.value(), "seed_mapping");
    TORCH_CHECK(seedMapping.value().scalar_type() == indptr.scalar_type(),
                "csr_row_wise_sampling_uniform_fused: seed_mapping dtype must match indptr");
    TORCH_CHECK(seedMapping.value().device() == indptr.device(),
                "csr_row_wise_sampling_uniform_fused: seed_mapping must be on the same NPU");
    TORCH_CHECK(seedMapping.value().numel() >= numTotalRows,
                "csr_row_wise_sampling_uniform_fused: seed_mapping is shorter than the graph rows");
    return true;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
EmptySamplingResult(const torch::Tensor& indptr, const torch::Tensor& rows,
    const c10::optional<torch::Tensor>& seedMapping, bool hasSeedMapping,
    const torch::Tensor& rowsLong, const torch::Tensor& rowsValid, int64_t numRows)
{
    auto options = indptr.options();
    auto outPtr = torch::zeros({numRows + 1}, options);
    c10::optional<torch::Tensor> mappingOut = c10::nullopt;
    c10::optional<torch::Tensor> newSeedNodes = c10::nullopt;
    if (hasSeedMapping) {
        if (numRows > 0) {
            UpdateSeedMapping(seedMapping.value(), rowsLong, rowsValid, numRows);
        }
        mappingOut = seedMapping.value();
        newSeedNodes = rows.clone();
    }
    return std::make_tuple(outPtr, torch::empty({0}, options), torch::empty({0}, options),
                           torch::empty({0}, options), mappingOut, newSeedNodes);
}

std::vector<uint32_t> ComputePickCounts(const torch::Tensor& indptr, const torch::Tensor& rowsClamped,
    const torch::Tensor& rowsValid, int64_t numRows, int64_t numSamples, bool selectAll,
    bool doReplace, uint32_t& maxReservoirPicks)
{
    auto degrees = (indptr.index({rowsClamped + 1}) - indptr.index({rowsClamped})).to(torch::kLong);
    degrees = torch::where(rowsValid, degrees, torch::zeros_like(degrees));
    auto degreesCpu = degrees.to(torch::kCPU).contiguous();
    const int64_t* degPtr = degreesCpu.data_ptr<int64_t>();
    std::vector<uint32_t> picks(static_cast<size_t>(numRows));
    maxReservoirPicks = 0;
    for (int64_t i = 0; i < numRows; ++i) {
        TORCH_CHECK(degPtr[i] >= 0,
                    "csr_row_wise_sampling_uniform_fused: negative row degree at row ", i);
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

struct LaunchPlan {
    std::vector<uint32_t> rowSplit;
    std::vector<uint32_t> outStarts;
    std::vector<uint32_t> prefix;
    uint64_t maxOutput = 0;
};

LaunchPlan BuildLaunchPlan(const std::vector<uint32_t>& picks, uint32_t blocks)
{
    LaunchPlan plan;
    plan.rowSplit = BuildBalancedPartitions(picks, blocks);
    plan.prefix.assign(picks.size() + 1, 0);
    for (size_t i = 0; i < picks.size(); ++i) {
        plan.prefix[i + 1] = plan.prefix[i] + picks[i];
    }
    plan.outStarts.assign(blocks + 1, 0);
    for (uint32_t block = 0; block <= blocks; ++block) {
        plan.outStarts[block] = plan.prefix[plan.rowSplit[block]];
    }
    plan.maxOutput = plan.outStarts[blocks];
    return plan;
}

struct SamplingOutputs {
    torch::Tensor outPtr;
    torch::Tensor outRows;
    torch::Tensor outCols;
    torch::Tensor outData;
    torch::Tensor rowSplit;
    torch::Tensor outStarts;
    torch::Tensor pickScratch;
};

SamplingOutputs AllocateOutputs(const torch::Tensor& indptr, const LaunchPlan& plan, uint32_t blocks,
    uint32_t maxReservoirPicks, bool needPickScratch)
{
    auto options = indptr.options();
    const int64_t maxOutput = static_cast<int64_t>(plan.maxOutput);
    SamplingOutputs outputs;
    outputs.rowSplit = Uint32ToTensor<uint32_t>(plan.rowSplit, indptr);
    outputs.outStarts = Uint32ToTensor<uint32_t>(plan.outStarts, indptr);
    outputs.outPtr = indptr.scalar_type() == at::ScalarType::Long
        ? Uint32ToTensor<int64_t>(plan.prefix, indptr)
        : Uint32ToTensor<int32_t>(plan.prefix, indptr);
    outputs.outRows = torch::zeros({maxOutput}, options);
    outputs.outCols = torch::zeros({maxOutput}, options);
    outputs.outData = torch::zeros({maxOutput}, options);
    outputs.pickScratch = needPickScratch
        ? torch::zeros({static_cast<int64_t>(blocks) * maxReservoirPicks}, options)
        : torch::empty({0}, options);
    return outputs;
}

struct UbPlan {
    uint32_t blocks = 0;
    uint32_t ubAvailable = 0;
    uint32_t elemSize = 0;
    uint32_t windowElems = 0;
};

UbPlan QueryUbPlan(const torch::Tensor& indptr)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr,
                "csr_row_wise_sampling_uniform_fused: failed to initialize AscendC platform");
    UbPlan plan;
    plan.blocks = std::max<uint32_t>(platform->GetCoreNumAiv(), 1);
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    plan.ubAvailable = ubBytes > kUbReservedBytes
        ? static_cast<uint32_t>(ubBytes - kUbReservedBytes) : 0;
    TORCH_CHECK(plan.ubAvailable > 0, "csr_row_wise_sampling_uniform_fused: invalid UB size");
    plan.elemSize = static_cast<uint32_t>(indptr.element_size());
    plan.windowElems = plan.ubAvailable / csr_uniform_fused::kUbInstances / plan.elemSize;
    return plan;
}

std::tuple<c10::optional<torch::Tensor>, c10::optional<torch::Tensor>> ApplySeedMapping(
    const c10::optional<torch::Tensor>& seedMapping, bool hasSeedMapping,
    const torch::Tensor& rows, const torch::Tensor& rowsLong, const torch::Tensor& rowsValid,
    int64_t numRows)
{
    c10::optional<torch::Tensor> mappingOut = c10::nullopt;
    c10::optional<torch::Tensor> newSeedNodes = c10::nullopt;
    if (!hasSeedMapping) {
        return std::make_tuple(mappingOut, newSeedNodes);
    }
    UpdateSeedMapping(seedMapping.value(), rowsLong, rowsValid, numRows);
    mappingOut = seedMapping.value();
    newSeedNodes = rows.clone();
    return std::make_tuple(mappingOut, newSeedNodes);
}

struct TilingParams {
    int64_t numRows = 0;
    int64_t numSamples = 0;
    int64_t numTotalRows = 0;
    int64_t seed = 0;
    bool selectAll = false;
    bool doReplace = false;
    bool hasData = false;
    bool needPickScratch = false;
    uint32_t ubAvailable = 0;
    uint32_t elemSize = 0;
    uint32_t maxReservoirPicks = 0;
};

CsrRowWiseSamplingUniformFusedTiling BuildTiling(const TilingParams& params)
{
    csr_uniform_fused::CsrRowWiseSamplingUniformFusedTiling tiling{};
    tiling.numRows = static_cast<uint32_t>(params.numRows);
    tiling.numSamples = params.selectAll ? 0u : static_cast<uint32_t>(params.numSamples);
    tiling.replace = params.doReplace ? 1u : 0u;
    tiling.hasData = params.hasData ? 1u : 0u;
    tiling.seed = static_cast<uint32_t>(params.seed);
    tiling.selectAll = params.selectAll ? 1u : 0u;
    tiling.numTotalRows = static_cast<uint32_t>(params.numTotalRows);
    tiling.ubAvailable = params.ubAvailable;
    tiling.dtypeBytes = params.elemSize;
    tiling.pickStride = params.needPickScratch ? params.maxReservoirPicks : 0u;
    return tiling;
}

void LaunchSampling(const torch::Tensor& indptr, const torch::Tensor& indices,
    const c10::optional<torch::Tensor>& data, const torch::Tensor& rows,
    const SamplingOutputs& outputs, const CsrRowWiseSamplingUniformFusedTiling& tiling,
    uint32_t blocks, bool hasData, bool needPickScratch)
{
    aclrtStream stream = c10_npu::getCurrentNPUStream(indptr.get_device()).stream();
    void* dataPtr = hasData ? data.value().data_ptr() : nullptr;
    void* pickScratchPtr = needPickScratch ? outputs.pickScratch.data_ptr() : nullptr;
    csr_uniform_fused::LaunchUniformSampling(blocks, stream, indptr.data_ptr(), indices.data_ptr(), dataPtr,
        rows.data_ptr(), outputs.outPtr.data_ptr(), outputs.outRows.data_ptr(), outputs.outCols.data_ptr(),
        outputs.outData.data_ptr(), outputs.rowSplit.data_ptr(), outputs.outStarts.data_ptr(),
        pickScratchPtr, tiling);
}

}  // namespace
}  // namespace csr_uniform_fused

using namespace csr_uniform_fused;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
CsrRowWiseSamplingUniformFused(const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::Tensor& rows, int64_t numSamples, bool replace,
    const c10::optional<torch::Tensor>& data,
    const c10::optional<torch::Tensor>& seedMapping, int64_t seed)
{
    ValidateSignature(indptr, indices, rows, numSamples);
    const bool hasData = CheckOptionalData(data, indptr, indices);
    const bool selectAll = numSamples == -1;
    const bool doReplace = replace && !selectAll;
    const int64_t numRows = rows.numel();
    const int64_t numTotalRows = indptr.numel() - 1;
    TORCH_CHECK(numTotalRows >= 0,
                "csr_row_wise_sampling_uniform_fused: indptr must have at least one element");
    const bool hasSeedMapping = CheckOptionalSeedMapping(seedMapping, indptr, numTotalRows);

    auto rowsLong = rows.to(torch::kLong);
    torch::Tensor rowsValid = (rowsLong >= 0) & (rowsLong < numTotalRows);
    torch::Tensor rowsClamped = numTotalRows > 0
        ? rowsLong.clamp(0, numTotalRows - 1) : torch::zeros_like(rowsLong);
    if (numRows == 0 || numTotalRows == 0 || numSamples == 0) {
        return EmptySamplingResult(indptr, rows, seedMapping, hasSeedMapping, rowsLong, rowsValid, numRows);
    }

    uint32_t maxReservoirPicks = 0;
    const std::vector<uint32_t> picks = ComputePickCounts(indptr, rowsClamped, rowsValid, numRows,
        numSamples, selectAll, doReplace, maxReservoirPicks);

    const UbPlan ub = QueryUbPlan(indptr);
    const bool needPickScratch = maxReservoirPicks > ub.windowElems;

    const LaunchPlan plan = BuildLaunchPlan(picks, ub.blocks);
    TORCH_CHECK(plan.maxOutput <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                "csr_row_wise_sampling_uniform_fused: output size exceeds int32 range");
    const SamplingOutputs outputs = AllocateOutputs(indptr, plan, ub.blocks, maxReservoirPicks,
                                                    needPickScratch);

    TilingParams params{};
    params.numRows = numRows;
    params.numSamples = numSamples;
    params.numTotalRows = numTotalRows;
    params.seed = seed;
    params.selectAll = selectAll;
    params.doReplace = doReplace;
    params.hasData = hasData;
    params.needPickScratch = needPickScratch;
    params.ubAvailable = ub.ubAvailable;
    params.elemSize = ub.elemSize;
    params.maxReservoirPicks = maxReservoirPicks;
    const auto tiling = BuildTiling(params);
    LaunchSampling(indptr, indices, data, rows, outputs, tiling, ub.blocks, hasData, needPickScratch);

    const auto mapping = ApplySeedMapping(seedMapping, hasSeedMapping, rows, rowsLong, rowsValid,
                                          numRows);
    return std::make_tuple(outputs.outPtr, outputs.outRows, outputs.outCols, outputs.outData,
                           std::get<0>(mapping), std::get<1>(mapping));
}

}  // namespace opsgnn

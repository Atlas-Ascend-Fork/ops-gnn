/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "csr_row_wise_sampling.h"

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "csr_row_wise_sampling/op_kernel/arch22/csr_row_wise_sampling.h"
#include "csr_row_wise_sampling/op_kernel/arch22/csr_row_wise_sampling_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {
namespace csr_weighted {
namespace {

constexpr uint32_t kUbReservedBytes = 2 * 1024;

void CheckNpuTensor(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), "csr_row_wise_sampling: ", name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "csr_row_wise_sampling: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.dim() == 1, "csr_row_wise_sampling: ", name, " must be 1-D");
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

torch::Tensor MakeTiling(const std::array<uint32_t, kTilingHeaderWordsWeighted>& values,
    const torch::Tensor& like)
{
    auto cpu = torch::from_blob(const_cast<uint32_t*>(values.data()),
        {static_cast<int64_t>(values.size())},
        torch::TensorOptions().dtype(torch::kInt32)).clone();
    return cpu.to(like.device());
}

struct SamplingContext {
    int64_t numRows = 0;
    int64_t numTotalRows = 0;
    int64_t numSamples = 0;
    bool selectAll = false;
    bool doReplace = false;
    bool hasData = false;
    bool isInt32 = false;
    uint32_t fanout = 0;
    uint32_t blocks = 1;
    uint32_t ubAvailable = 0;
    aclrtStream stream = nullptr;
    void* dataPtr = nullptr;
};

void ValidateCsrInputs(const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::Tensor& rows, int64_t numSamples, bool replace,
    const c10::optional<torch::Tensor>& data, const torch::Tensor& prob,
    SamplingContext& ctx)
{
    CheckNpuTensor(indptr, "indptr");
    CheckNpuTensor(indices, "indices");
    CheckNpuTensor(rows, "rows");
    CheckNpuTensor(prob, "prob");
    TORCH_CHECK(indptr.scalar_type() == at::ScalarType::Int ||
                indptr.scalar_type() == at::ScalarType::Long,
                "csr_row_wise_sampling: indptr dtype must be int32 or int64");
    TORCH_CHECK(indices.scalar_type() == indptr.scalar_type() &&
                rows.scalar_type() == indptr.scalar_type(),
                "csr_row_wise_sampling: indptr/indices/rows dtypes must match");
    TORCH_CHECK(indptr.device() == indices.device() &&
                indptr.device() == rows.device() &&
                indptr.device() == prob.device(),
                "csr_row_wise_sampling: all inputs must be on the same NPU");
    TORCH_CHECK(prob.scalar_type() == at::ScalarType::Float,
                "csr_row_wise_sampling: prob must be float32");
    TORCH_CHECK(prob.numel() == indices.numel(),
                "csr_row_wise_sampling: prob length must match indices");
    TORCH_CHECK(numSamples >= -1,
                "csr_row_wise_sampling: num_samples must be >= -1");

    ctx.numSamples = numSamples;
    ctx.selectAll = numSamples == -1;
    ctx.doReplace = replace && !ctx.selectAll;
    ctx.numRows = rows.numel();
    ctx.numTotalRows = indptr.numel() - 1;
    TORCH_CHECK(ctx.numTotalRows >= 0,
                "csr_row_wise_sampling: indptr must have at least one element");
    ctx.hasData = false;
    if (data.has_value()) {
        CheckNpuTensor(data.value(), "data");
        TORCH_CHECK(data.value().scalar_type() == indptr.scalar_type(),
                    "csr_row_wise_sampling: data dtype must match indptr");
        TORCH_CHECK(data.value().device() == indptr.device(),
                    "csr_row_wise_sampling: data must be on the same NPU");
        TORCH_CHECK(data.value().numel() == indices.numel(),
                    "csr_row_wise_sampling: data length must match indices");
        ctx.hasData = true;
    }
}

void PrepareLaunch(const torch::Tensor& indptr,
    const c10::optional<torch::Tensor>& data, SamplingContext& ctx)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr,
                "csr_row_wise_sampling: failed to initialize AscendC platform");
    ctx.blocks = std::max<uint32_t>(platform->GetCoreNumAiv(), 1);
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    ctx.ubAvailable = ubBytes > kUbReservedBytes
        ? static_cast<uint32_t>(ubBytes - kUbReservedBytes) : 0;
    TORCH_CHECK(ctx.ubAvailable > 0, "csr_row_wise_sampling: invalid UB size");
    ctx.isInt32 = indptr.scalar_type() == at::ScalarType::Int;
    ctx.stream = c10_npu::getCurrentNPUStream(indptr.get_device()).stream();
    ctx.dataPtr = ctx.hasData ? data.value().data_ptr() : nullptr;
    ctx.fanout = ctx.selectAll ? 0u : static_cast<uint32_t>(ctx.numSamples);
}

std::vector<uint32_t> CountPicksPerRow(const torch::Tensor& indptr,
    const torch::Tensor& rows, const torch::Tensor& prob,
    const SamplingContext& ctx)
{
    auto options = indptr.options();
    auto picks = torch::empty({ctx.numRows}, options.dtype(torch::kInt32));
    std::array<uint32_t, kTilingHeaderWordsWeighted> pickTiling = {
        static_cast<uint32_t>(ctx.numRows), ctx.fanout,
        static_cast<uint32_t>(ctx.doReplace ? 1 : 0),
        static_cast<uint32_t>(ctx.hasData ? 1 : 0), 0u,
        static_cast<uint32_t>(ctx.selectAll ? 1 : 0),
        static_cast<uint32_t>(ctx.numTotalRows), 0u, 0u,
    };
    auto pickTilingDev = MakeTiling(pickTiling, indptr);
    TORCH_CHECK(aclrtSynchronizeDevice() == ACL_SUCCESS,
                "csr_row_wise_sampling: pre-launch sync failed");
    LaunchNumPicks(ctx.isInt32, ctx.blocks, ctx.stream, rows.data_ptr(),
        indptr.data_ptr(), ctx.dataPtr, prob.data_ptr(), picks.data_ptr(),
        pickTilingDev.data_ptr());
    TORCH_CHECK(aclrtSynchronizeStream(ctx.stream) == ACL_SUCCESS,
                "csr_row_wise_sampling: num-picks sync failed");

    auto picksCpu = picks.to(torch::kCPU).contiguous();
    const int32_t* pickPtr = picksCpu.data_ptr<int32_t>();
    std::vector<uint32_t> pickVec(static_cast<size_t>(ctx.numRows));
    for (int64_t i = 0; i < ctx.numRows; ++i) {
        pickVec[static_cast<size_t>(i)] = static_cast<uint32_t>(pickPtr[i]);
    }
    return pickVec;
}

void BuildRowPlan(const std::vector<uint32_t>& pickVec, uint32_t blocks,
    std::vector<uint32_t>& rowSplitVec, std::vector<uint32_t>& prefix,
    uint64_t& maxOutput)
{
    rowSplitVec = BuildBalancedPartitions(pickVec, blocks);
    std::vector<uint64_t> prefix64(pickVec.size() + 1, 0);
    for (size_t i = 0; i < pickVec.size(); ++i) {
        prefix64[i + 1] = prefix64[i] + pickVec[i];
    }
    maxOutput = prefix64.back();
    TORCH_CHECK(maxOutput <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                "csr_row_wise_sampling: output size exceeds int32 range");
    prefix.assign(prefix64.begin(), prefix64.end());
}

}  // namespace
}  // namespace csr_weighted

using namespace csr_weighted;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CsrRowWiseSampling(const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::Tensor& rows, int64_t numSamples, bool replace,
    const c10::optional<torch::Tensor>& data, const torch::Tensor& prob,
    int64_t seed)
{
    SamplingContext ctx;
    ValidateCsrInputs(indptr, indices, rows, numSamples, replace, data, prob, ctx);
    auto options = indptr.options();
    auto empty = torch::empty({0}, options);
    if (ctx.numRows == 0 || ctx.numTotalRows == 0 || numSamples == 0) {
        return std::make_tuple(empty, torch::empty({0}, options), torch::empty({0}, options));
    }

    PrepareLaunch(indptr, data, ctx);
    const auto pickVec = CountPicksPerRow(indptr, rows, prob, ctx);
    std::vector<uint32_t> rowSplitVec;
    std::vector<uint32_t> prefix;
    uint64_t maxOutput = 0;
    BuildRowPlan(pickVec, ctx.blocks, rowSplitVec, prefix, maxOutput);
    if (maxOutput == 0) {
        return std::make_tuple(empty, torch::empty({0}, options), torch::empty({0}, options));
    }

    auto rowSplit = Uint32ToTensor<uint32_t>(rowSplitVec, indptr);
    auto outStartsTensor = Uint32ToTensor<uint32_t>(prefix, indptr);
    auto outPtr = torch::empty({ctx.numRows + 1}, options);
    auto outRows = torch::empty({static_cast<int64_t>(maxOutput)}, options);
    auto outCols = torch::empty({static_cast<int64_t>(maxOutput)}, options);
    auto outData = torch::empty({static_cast<int64_t>(maxOutput)}, options);

    std::array<uint32_t, kTilingHeaderWordsWeighted> mainTiling = {
        static_cast<uint32_t>(ctx.numRows), ctx.fanout,
        static_cast<uint32_t>(ctx.doReplace ? 1 : 0),
        static_cast<uint32_t>(ctx.hasData ? 1 : 0),
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(ctx.selectAll ? 1 : 0),
        static_cast<uint32_t>(ctx.numTotalRows), ctx.ubAvailable, 0u,
    };
    auto mainTilingDev = MakeTiling(mainTiling, indptr);
    TORCH_CHECK(aclrtSynchronizeDevice() == ACL_SUCCESS,
                "csr_row_wise_sampling: pre-launch sync failed");
    LaunchWeightedSampling(ctx.isInt32, ctx.blocks, ctx.stream, indptr.data_ptr(), indices.data_ptr(),
        ctx.dataPtr, rows.data_ptr(), prob.data_ptr(), outPtr.data_ptr(), outRows.data_ptr(),
        outCols.data_ptr(), outData.data_ptr(), rowSplit.data_ptr(),
        outStartsTensor.data_ptr(), mainTilingDev.data_ptr());
    TORCH_CHECK(aclrtSynchronizeStream(ctx.stream) == ACL_SUCCESS,
                "csr_row_wise_sampling: sampler sync failed");

    return std::make_tuple(outRows, outCols, outData);
}

}  // namespace opsgnn

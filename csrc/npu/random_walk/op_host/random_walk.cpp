/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "random_walk.h"

#include "random_walk/common/philox_host.h"
#include "random_walk/op_kernel/arch35/random_walk.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace opsgnn {

namespace {

constexpr uint64_t UINT32_RANGE = 1ULL << 32;

void CheckInt64Vector(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Long, name, " must have dtype torch.int64");
    TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

uint32_t ProbabilityThreshold(long double probability)
{
    if (probability <= 0.0L) {
        return 0;
    }
    if (probability >= 1.0L) {
        return std::numeric_limits<uint32_t>::max();
    }
    uint64_t acceptedCount = static_cast<uint64_t>(
        std::floor(probability * static_cast<long double>(UINT32_RANGE)));
    TORCH_CHECK(acceptedCount > 0,
                "random_walk: normalized acceptance probability ",
                static_cast<double>(probability),
                " is below the 1/2**32 resolution of the 32-bit sampler; "
                "p and q are too extreme to preserve relative transition weights");
    return static_cast<uint32_t>(acceptedCount - 1);
}

void ValidateRandomWalkInputs(
    const torch::Tensor& rowptr, const torch::Tensor& col, const torch::Tensor& start,
    int64_t walkLength, double p, double q)
{
    CheckInt64Vector(rowptr, "rowptr");
    CheckInt64Vector(col, "col");
    CheckInt64Vector(start, "start");
    TORCH_CHECK(rowptr.get_device() == col.get_device() && rowptr.get_device() == start.get_device(),
                "rowptr, col and start must be on the same NPU device");
    TORCH_CHECK(rowptr.numel() >= 1, "rowptr must contain at least one element");
    TORCH_CHECK(walkLength >= 0, "walk_length must be non-negative");
    TORCH_CHECK(walkLength <= std::numeric_limits<uint32_t>::max(), "walk_length exceeds uint32 range");
    TORCH_CHECK(col.numel() <= std::numeric_limits<uint32_t>::max(),
                "random_walk currently supports at most 2**32-1 edges");
    TORCH_CHECK(std::isfinite(p) && p > 0.0, "p must be finite and greater than zero");
    TORCH_CHECK(std::isfinite(q) && q > 0.0, "q must be finite and greater than zero");
}

RandomWalkLaunchParams CreateLaunchParams(
    int64_t startCount, int64_t walkLength, double p, double q, bool returnEdgeIndices,
    bool neighborsSorted, c10::DeviceIndex deviceIndex)
{
    const long double returnWeight = 1.0L / static_cast<long double>(p);
    const long double neighborWeight = 1.0L;
    const long double distantWeight = 1.0L / static_cast<long double>(q);
    const long double maxWeight = std::max(std::max(returnWeight, neighborWeight), distantWeight);
    TORCH_CHECK(maxWeight > 0.0L && std::isfinite(maxWeight),
                "normalized probability denominator must be positive and finite");

    RandomWalkLaunchParams params{};
    params.startCount = static_cast<uint64_t>(startCount);
    params.walkLength = static_cast<uint32_t>(walkLength);
    params.returnThreshold = ProbabilityThreshold(returnWeight / maxWeight);
    params.neighborThreshold = ProbabilityThreshold(neighborWeight / maxWeight);
    params.distantThreshold = ProbabilityThreshold(distantWeight / maxWeight);
    params.node2vec = p != 1.0 || q != 1.0;
    params.writeEdge = returnEdgeIndices;
    params.neighborsSorted = neighborsSorted;
    auto [seed, offset] = AcquirePhiloxState(deviceIndex);
    params.seed = seed;
    params.offset = offset;
    return params;
}

std::pair<torch::Tensor, torch::Tensor> CreateOutputTensors(
    const torch::Tensor& start, int64_t walkLength, bool returnEdgeIndices)
{
    const int64_t startCount = start.numel();
    torch::Tensor nodeOut = walkLength == 0
        ? start.reshape({startCount, 1}).clone()
        : torch::empty({startCount, walkLength + 1}, start.options());
    torch::Tensor edgeOut = torch::empty({startCount, returnEdgeIndices ? walkLength : 0}, start.options());
    return {nodeOut, edgeOut};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> random_walk_npu(
    const torch::Tensor& rowptr,
    const torch::Tensor& col,
    const torch::Tensor& start,
    int64_t walkLength,
    double p,
    double q,
    bool returnEdgeIndices,
    bool neighborsSorted)
{
    ValidateRandomWalkInputs(rowptr, col, start, walkLength, p, q);
    const int64_t startCount = start.numel();
    auto [nodeOut, edgeOut] = CreateOutputTensors(start, walkLength, returnEdgeIndices);
    if (startCount == 0 || walkLength == 0) {
        return std::make_tuple(nodeOut, edgeOut);
    }

    auto params = CreateLaunchParams(
        startCount, walkLength, p, q, returnEdgeIndices, neighborsSorted, start.get_device());

    // Flush queued PyTorch preprocessing before launching directly on the underlying ACL stream.
    aclrtStream stream = c10_npu::getCurrentNPUStream(start.get_device()).stream();
    RandomWalk(
        rowptr.data_ptr<int64_t>(),
        col.data_ptr<int64_t>(),
        start.data_ptr<int64_t>(),
        nodeOut.data_ptr<int64_t>(),
        returnEdgeIndices ? edgeOut.data_ptr<int64_t>() : nullptr,
        params,
        stream);

    return std::make_tuple(nodeOut, edgeOut);
}

}  // namespace opsgnn

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Host entry for torch_sparse::random_walk (NPU).
 */

#include "torch_sparse_random_walk.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>
#include <c10/core/DeviceGuard.h>

#include <ATen/core/Generator.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "common/torch_sparse_shared_checks.h"
#include "torch_sparse_random_walk/op_kernel/arch35/torch_sparse_random_walk.h"

namespace opsgnn {

namespace {

// Philox refill produces 4 uint32 draws; the generator bookkeeping advances
// by the same increment (mirrors the ops-gnn random_walk host).
constexpr uint64_t RNG_CALL_INCREMENT = 4;

std::pair<uint64_t, uint64_t> AcquirePhiloxState(c10::DeviceIndex deviceIndex)
{
    const at::Generator &generator = at_npu::detail::getDefaultNPUGenerator(deviceIndex);
    auto *generatorImpl = at::check_generator<at_npu::NPUGeneratorImpl>(generator);
    std::lock_guard<std::mutex> lock(generatorImpl->mutex_);
    return generatorImpl->philox_engine_inputs(RNG_CALL_INCREMENT);
}

void ValidateWalkInputs(
    const torch::Tensor &rowptr,
    const torch::Tensor &col,
    const torch::Tensor &start,
    int64_t walkLength)
{
    torch_sparse_utils::CheckInt64Vector(rowptr, "rowptr");
    torch_sparse_utils::CheckInt64Vector(col, "col");
    torch_sparse_utils::CheckInt64Vector(start, "start");
    torch_sparse_utils::CheckSameDevice(rowptr, col, "rowptr", "col");
    torch_sparse_utils::CheckSameDevice(rowptr, start, "rowptr", "start");
    TORCH_CHECK(rowptr.numel() >= 1, "rowptr must contain at least one element");
    TORCH_CHECK(walkLength >= 0, "walk_length must be non-negative");
    TORCH_CHECK(walkLength < std::numeric_limits<int64_t>::max(),
                "walk_length + 1 exceeds int64 range");
    torch_sparse_utils::CheckOutputSize(start.numel(), walkLength + 1, sizeof(int64_t));
}

TorchSparseRandomWalkLaunchParams BuildLaunchParams(
    const torch::Tensor &rowptr,
    const torch::Tensor &col,
    const torch::Tensor &start,
    int64_t walkLength)
{
    TorchSparseRandomWalkLaunchParams params{};
    params.startCount = static_cast<uint64_t>(start.numel());
    params.walkLength = static_cast<uint64_t>(walkLength);
    params.nnz = static_cast<uint64_t>(col.numel());
    params.nodeCount = static_cast<uint64_t>(rowptr.numel() - 1);
    return params;
}

torch::Tensor AllocateOutput(const torch::Tensor &start, int64_t walkLength)
{
    const int64_t startCount = start.numel();
    // The kernel writes every slot (start column + one per step, with -1
    // propagation), so no pre-fill is needed.
    return torch::empty({startCount, walkLength + 1}, start.options());
}

} // namespace

torch::Tensor torch_sparse_random_walk_npu(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor start,
    int64_t walkLength)
{
    ValidateWalkInputs(rowptr, col, start, walkLength);

    const c10::OptionalDeviceGuard deviceGuard(rowptr.device());

    rowptr = rowptr.contiguous();
    col = col.contiguous();
    start = start.contiguous();

    torch::Tensor out = AllocateOutput(start, walkLength);
    TorchSparseRandomWalkLaunchParams params = BuildLaunchParams(rowptr, col, start, walkLength);
    if (start.numel() > 0 && walkLength > 0) {
        std::tie(params.seed, params.offset) = AcquirePhiloxState(start.get_device());
    }

    const uint32_t availableCores = torch_sparse_utils::ResolveAivCoreCount();
    aclrtStream stream = c10_npu::getCurrentNPUStream(start.get_device()).stream();
    LaunchTorchSparseRandomWalkKernel(
        rowptr.data_ptr<int64_t>(),
        col.data_ptr<int64_t>(),
        start.data_ptr<int64_t>(),
        out.data_ptr<int64_t>(),
        params,
        availableCores,
        stream);
    return out;
}

torch::Tensor torch_sparse_random_walk_with_rand_npu(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor start,
    int64_t walkLength,
    torch::Tensor rand)
{
    ValidateWalkInputs(rowptr, col, start, walkLength);

    const c10::OptionalDeviceGuard deviceGuard(rowptr.device());

    TORCH_CHECK(rand.defined(), "rand must be defined");
    TORCH_CHECK(rand.device().type() == c10::DeviceType::PrivateUse1,
                "rand must be an NPU tensor");
    torch_sparse_utils::CheckSameDevice(rowptr, rand, "rowptr", "rand");
    TORCH_CHECK(rand.scalar_type() == at::ScalarType::Float,
                "rand must have dtype torch.float32");
    TORCH_CHECK(rand.dim() == 2, "rand must be two-dimensional [num_starts, walk_length]");
    TORCH_CHECK(rand.size(0) == start.numel() && rand.size(1) == walkLength,
                "rand must have shape [num_starts, walk_length]");
    rand = rand.contiguous();

    rowptr = rowptr.contiguous();
    col = col.contiguous();
    start = start.contiguous();

    torch::Tensor out = AllocateOutput(start, walkLength);
    TorchSparseRandomWalkLaunchParams params = BuildLaunchParams(rowptr, col, start, walkLength);
    params.seed = 0;
    params.offset = 0;

    const uint32_t availableCores = torch_sparse_utils::ResolveAivCoreCount();
    aclrtStream stream = c10_npu::getCurrentNPUStream(start.get_device()).stream();
    LaunchTorchSparseRandomWalkExactKernel(
        rowptr.data_ptr<int64_t>(),
        col.data_ptr<int64_t>(),
        start.data_ptr<int64_t>(),
        rand.data_ptr<float>(),
        out.data_ptr<int64_t>(),
        params,
        availableCores,
        stream);
    return out;
}

} // namespace opsgnn

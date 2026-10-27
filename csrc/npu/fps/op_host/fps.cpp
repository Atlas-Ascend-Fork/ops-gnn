/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fps.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include <c10/core/DeviceGuard.h>
#include "fps/op_kernel/arch35/fps_kernel.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

void CheckSource(const torch::Tensor& src)
{
    TORCH_CHECK(src.defined(), "fps: src must be defined");
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
                "fps: src must be an NPU tensor");
    TORCH_CHECK(src.dim() == 2, "fps: native src must be a 2D tensor");
    TORCH_CHECK(src.is_contiguous(), "fps: native src must be contiguous");
    TORCH_CHECK(src.scalar_type() == torch::kFloat ||
                    src.scalar_type() == torch::kHalf,
                "fps: src dtype must be float16 or float32");
    TORCH_CHECK(src.size(0) <= std::numeric_limits<int32_t>::max(),
                "fps: src has too many points");
}

void CheckIndexTensor(const torch::Tensor& tensor, const torch::Tensor& src,
                      const char* name)
{
    TORCH_CHECK(tensor.defined(), "fps: ", name, " must be defined");
    TORCH_CHECK(tensor.device() == src.device(), "fps: ", name,
                " must be on the same NPU as src");
    TORCH_CHECK(tensor.scalar_type() == torch::kLong, "fps: ", name,
                " must have dtype int64");
    TORCH_CHECK(tensor.dim() == 1, "fps: ", name, " must be one-dimensional");
    TORCH_CHECK(tensor.is_contiguous(), "fps: ", name, " must be contiguous");
}

uint32_t AlignUp32(uint64_t value)
{
    constexpr uint32_t alignment = 32;
    return static_cast<uint32_t>(((value + alignment - 1) / alignment) * alignment);
}

uint32_t SelectTilePoints(const torch::Tensor& src, uint64_t ub_bytes)
{
    const uint64_t element_bytes = src.element_size();
    const uint64_t center_bytes = AlignUp32(
        static_cast<uint64_t>(src.size(1)) * element_bytes);
    const uint64_t fixed_bytes = center_bytes + FPS_REDUCE_WORK_BYTES +
                                 FPS_REDUCE_RESULT_BYTES + 4096;
    // FP16 uses two fp16 vectors plus two fp32 vectors (12 bytes/point).
    // FP32 F>=8 reproduces PyTorch CPU's W=8/ILP=4 reduction one lane at a
    // time with four row accumulators plus scalar tail, point, and nearest.
    const bool fp32_cascade = src.scalar_type() == torch::kFloat &&
                              src.size(1) >= 5 && src.size(1) <= 64;
    const uint64_t bytes_per_point = fp32_cascade ? 28 : 12;
    TORCH_CHECK(ub_bytes > fixed_bytes + bytes_per_point * 256,
                "fps: insufficient UB capacity");
    uint64_t available_points =
        (ub_bytes - fixed_bytes) / bytes_per_point;
    // Keep vector tiles aligned to the native 256-element reduction width.
    const uint64_t vector_points = 256;
    available_points = (available_points / vector_points) * vector_points;
    TORCH_CHECK(available_points >= vector_points,
                "fps: insufficient UB capacity for one vector tile");
    const uint64_t requested = std::min<uint64_t>(
        FPS_MAX_TILE_POINTS, static_cast<uint64_t>(src.size(0)));
    const uint64_t aligned_requested =
        ((requested + vector_points - 1) / vector_points) * vector_points;
    return static_cast<uint32_t>(
        std::min<uint64_t>(available_points, aligned_requested));
}

template <typename T, int DType>
void Launch(const torch::Tensor& src, const torch::Tensor& src_soa,
            const torch::Tensor& ptr,
            const torch::Tensor& out_ptr, const torch::Tensor& start,
            torch::Tensor& dist, torch::Tensor& best_dist,
            torch::Tensor& best_index, torch::Tensor& out, int64_t batch_size,
            int64_t single_out_size, int64_t single_start)
{
    FpsLaunchArgs<T> args{};
    const bool batched = ptr.defined();
    args.src_soa = static_cast<const T*>(src_soa.data_ptr());
    args.ptr = batched ? ptr.data_ptr<int64_t>() : nullptr;
    args.out_ptr = batched ? out_ptr.data_ptr<int64_t>() : nullptr;
    args.start = batched ? start.data_ptr<int64_t>() : nullptr;
    args.dist = static_cast<T*>(dist.data_ptr());
    args.best_dist = best_dist.data_ptr<float>();
    args.best_index = best_index.data_ptr<int64_t>();
    args.out = out.data_ptr<int64_t>();
    args.num_points = src.size(0);
    args.feature_dim = src.size(1);
    args.batch_size = batch_size;
    args.single_out_size = single_out_size;
    args.single_start = single_start;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "fps: platform information unavailable");
    uint64_t ub_bytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
    args.tile_points = SelectTilePoints(src, ub_bytes);
    args.tile_bytes = AlignUp32(
        static_cast<uint64_t>(args.tile_points) * src.element_size());
    args.promoted_tile_bytes = AlignUp32(
        static_cast<uint64_t>(args.tile_points) * sizeof(float));
    args.cascade_tile_bytes = AlignUp32(
        static_cast<uint64_t>(args.tile_points) * 5 * sizeof(float));
    args.center_bytes = AlignUp32(
        static_cast<uint64_t>(src.size(1)) * src.element_size());

    aclrtStream stream = c10_npu::getCurrentNPUStream(src.device().index()).stream();
    Fps<T, DType>(static_cast<const T*>(src.data_ptr()), args, stream);
}

torch::Tensor AllocateAndLaunch(const torch::Tensor& src,
                                const torch::Tensor& ptr,
                                const torch::Tensor& out_ptr,
                                const torch::Tensor& start,
                                int64_t out_size, int64_t batch_size,
                                int64_t single_start)
{
    auto long_options = torch::TensorOptions()
                            .dtype(torch::kLong)
                            .device(src.device());
    const torch::Tensor src_soa = src.transpose(0, 1).contiguous();
    torch::Tensor dist = torch::empty({src.size(0)}, src.options());
    torch::Tensor best_dist = torch::empty(
        {256}, torch::TensorOptions().dtype(torch::kFloat).device(src.device()));
    torch::Tensor best_index = torch::empty({256}, long_options);
    torch::Tensor out = torch::empty({out_size}, long_options);

    if (src.scalar_type() == torch::kFloat) {
        Launch<float, 0>(src, src_soa, ptr, out_ptr, start, dist, best_dist,
                         best_index, out, batch_size, out_size, single_start);
    } else {
        Launch<uint16_t, 1>(src, src_soa, ptr, out_ptr, start, dist,
                            best_dist, best_index, out, batch_size, out_size,
                            single_start);
    }
    return out;
}

}  // namespace

torch::Tensor fps_random_start(torch::Tensor degree)
{
    TORCH_CHECK(degree.defined(), "fps: degree must be defined");
    TORCH_CHECK(degree.device().type() == c10::DeviceType::PrivateUse1,
                "fps: degree must be an NPU tensor");
    TORCH_CHECK(degree.scalar_type() == torch::kLong,
                "fps: degree must have dtype int64");
    TORCH_CHECK(degree.dim() == 1,
                "fps: degree must be one-dimensional");
    c10::DeviceGuard guard(degree.device());
    const torch::Tensor degree_cpu = degree.cpu().contiguous();
    torch::Tensor start_cpu = torch::empty(
        {degree.numel()},
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    const int64_t* degree_data = degree_cpu.data_ptr<int64_t>();
    int64_t* start_data = start_cpu.data_ptr<int64_t>();
    for (int64_t batch = 0; batch < degree.numel(); ++batch) {
        TORCH_CHECK(degree_data[batch] >= 0,
                    "fps: degree must be non-negative");
        start_data[batch] = degree_data[batch] == 0
            ? 0
            : static_cast<int64_t>(std::rand()) % degree_data[batch];
    }
    return start_cpu.to(degree.device());
}

torch::Tensor fps_npu_single(torch::Tensor src, int64_t out_size,
                             int64_t start_index)
{
    CheckSource(src);
    c10::DeviceGuard guard(src.device());
    const int64_t num_points = src.size(0);
    TORCH_CHECK(num_points > 0, "fps: native single-batch input must be non-empty");
    TORCH_CHECK(out_size >= 1 && out_size <= num_points,
                "fps: out_size must be in [1, src.size(0)]");
    TORCH_CHECK(start_index >= 0 && start_index < num_points,
                "fps: start_index is out of range");
    if (out_size == 1) {
        return torch::full({1}, start_index,
                           torch::TensorOptions()
                               .dtype(torch::kLong)
                               .device(src.device()));
    }
    return AllocateAndLaunch(src, torch::Tensor(), torch::Tensor(),
                             torch::Tensor(), out_size, 1, start_index);
}

torch::Tensor fps_npu_batch(torch::Tensor src, torch::Tensor ptr,
                            torch::Tensor out_ptr, torch::Tensor start)
{
    CheckSource(src);
    c10::DeviceGuard guard(src.device());
    CheckIndexTensor(ptr, src, "ptr");
    CheckIndexTensor(out_ptr, src, "out_ptr");
    CheckIndexTensor(start, src, "start");
    TORCH_CHECK(ptr.numel() >= 2, "fps: ptr must contain at least two values");
    const int64_t batch_size = ptr.numel() - 1;
    TORCH_CHECK(out_ptr.numel() == batch_size + 1,
                "fps: out_ptr length must equal ptr length");
    TORCH_CHECK(start.numel() == batch_size,
                "fps: start length must equal the batch count");

    const torch::Tensor ptr_cpu = ptr.cpu().contiguous();
    const torch::Tensor out_ptr_cpu = out_ptr.cpu().contiguous();
    const torch::Tensor start_cpu = start.cpu().contiguous();
    const int64_t* ptr_data = ptr_cpu.data_ptr<int64_t>();
    const int64_t* out_ptr_data = out_ptr_cpu.data_ptr<int64_t>();
    const int64_t* start_data = start_cpu.data_ptr<int64_t>();
    TORCH_CHECK(ptr_data[0] == 0 && ptr_data[batch_size] == src.size(0),
                "fps: ptr must start at 0 and end at src.size(0)");
    TORCH_CHECK(out_ptr_data[0] == 0,
                "fps: out_ptr must start at 0");
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        TORCH_CHECK(ptr_data[batch] <= ptr_data[batch + 1],
                    "fps: ptr must be non-decreasing");
        TORCH_CHECK(out_ptr_data[batch] <= out_ptr_data[batch + 1],
                    "fps: out_ptr must be non-decreasing");
        const int64_t degree = ptr_data[batch + 1] - ptr_data[batch];
        const int64_t count = out_ptr_data[batch + 1] - out_ptr_data[batch];
        TORCH_CHECK(count >= 0 && count <= degree,
                    "fps: invalid sample count for batch ", batch);
        if (degree > 0 && count > 0) {
            TORCH_CHECK(start_data[batch] >= 0 && start_data[batch] < degree,
                        "fps: start index is out of range for batch ", batch);
        }
    }
    const int64_t out_size = out_ptr_data[batch_size];
    if (out_size == 0) {
        return torch::empty({0}, ptr.options());
    }
    return AllocateAndLaunch(src, ptr, out_ptr, start, out_size, batch_size, 0);
}

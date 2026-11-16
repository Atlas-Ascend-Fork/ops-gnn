/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "nearest.h"

#include <algorithm>
#include <limits>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <tiling/platform/platform_ascendc.h>

#include "nearest/op_kernel/arch35/nearest.h"

namespace opsgnn {
namespace {

void ValidatePointerLayout(const at::Tensor& ptr_x, const at::Tensor& ptr_y,
                           int64_t n, int64_t m)
{
    const auto host_x = ptr_x.cpu();
    const auto host_y = ptr_y.cpu();
    const int64_t* px = host_x.data_ptr<int64_t>();
    const int64_t* py = host_y.data_ptr<int64_t>();
    const int64_t len = host_x.numel();
    auto check_layout = [len](const int64_t* v, int64_t total, const char* name) {
        TORCH_CHECK(v[0] == 0, "nearest: ", name, "[0] must be 0, got ", v[0]);
        TORCH_CHECK(v[len - 1] == total, "nearest: ", name,
                    " must end with the point count ", total, ", got ", v[len - 1]);
        for (int64_t i = 1; i < len; ++i) {
            TORCH_CHECK(v[i] >= v[i - 1], "nearest: ", name,
                        " must be non-decreasing at index ", i, ", got ",
                        v[i - 1], " -> ", v[i]);
        }
    };
    check_layout(px, n, "ptr_x");
    check_layout(py, m, "ptr_y");
    for (int64_t i = 1; i < len; ++i) {
        TORCH_CHECK(py[i] != py[i - 1] || n == 0 || px[i] == px[i - 1],
                    "nearest: y segment ", i - 1, " is empty while its x segment is not");
    }
}

uint32_t ValidateInputs(const at::Tensor& x, const at::Tensor& y,
                        const c10::optional<at::Tensor>& ptr_x,
                        const c10::optional<at::Tensor>& ptr_y)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "nearest: x must be an NPU tensor");
    TORCH_CHECK(x.device() == y.device(), "nearest: x and y must share a device");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == y.size(1),
                "nearest: x and y must have shape [points, features]");
    TORCH_CHECK(x.scalar_type() == y.scalar_type() &&
                (x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat),
                "nearest: matching float16 or float32 inputs required");
    TORCH_CHECK(x.size(1) > 0, "nearest: features must be positive");
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int32_t>::max() &&
                y.size(0) <= std::numeric_limits<int32_t>::max() &&
                x.size(1) <= std::numeric_limits<int32_t>::max(),
                "nearest: dimensions exceed the supported int32 range");
    TORCH_CHECK(ptr_x.has_value() == ptr_y.has_value(),
                "nearest: both pointers must be supplied together");
    uint32_t batches = 0;
    if (ptr_x.has_value()) {
        for (const auto& ptr : {*ptr_x, *ptr_y}) {
            TORCH_CHECK(ptr.device() == x.device() && ptr.scalar_type() == at::kLong &&
                        ptr.dim() == 1 && ptr.is_contiguous(),
                        "nearest: pointers must be contiguous int64 on the input NPU");
        }
        TORCH_CHECK(ptr_x->numel() == ptr_y->numel() && ptr_x->numel() >= 2 &&
                    ptr_x->numel() <= std::numeric_limits<int32_t>::max(),
                    "nearest: pointers must have equal length >= 2");
        ValidatePointerLayout(*ptr_x, *ptr_y, x.size(0), y.size(0));
        batches = static_cast<uint32_t>(ptr_x->numel() - 1);
    }
    return batches;
}

NearestTiling MakeTiling(const at::Tensor& x, const at::Tensor& y, uint32_t batches)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "nearest: CANN platform unavailable");
    uint32_t cores = platform->GetCoreNumAiv();
    TORCH_CHECK(cores > 0, "nearest: no AIV cores available");
    uint64_t ub_bytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
    NearestTiling tiling{static_cast<uint32_t>(x.size(0)), static_cast<uint32_t>(y.size(0)),
                         static_cast<uint32_t>(x.size(1)), batches, cores, ub_bytes, 0, 0};
    uint64_t pairs = static_cast<uint64_t>(tiling.n) * tiling.m;
    bool small_dot = tiling.features >= 32 && pairs <= 1200 &&
                     pairs * tiling.features <= 1000000;
    if (tiling.features >= 5 && tiling.features <= 768 && batches == 0 && !small_dot &&
        x.size(0) <= std::numeric_limits<uint32_t>::max() / 4 &&
        y.size(0) <= std::numeric_limits<uint32_t>::max() / 4) {
        uint32_t tx = tiling.features <= 64 ? 64 : (tiling.features <= 256 ? 32 : 16);
        uint64_t fixed = static_cast<uint64_t>(tx) * (4 * (tiling.features + 1) + 16) + 4096;
        uint64_t capacity = ub_bytes > fixed ? (ub_bytes - fixed) / (4 * (tiling.features + 1)) : 0;
        uint32_t ty = 64;
        if (capacity >= ty) {
            while (2 * ty <= capacity && ty < 8192) ty *= 2;
            tiling.tile_x = tx;
            tiling.tile_y = ty;
        }
    }
    return tiling;
}
} // namespace

at::Tensor nearest_npu(const at::Tensor& x, const at::Tensor& y,
                       const c10::optional<at::Tensor>& ptr_x,
                       const c10::optional<at::Tensor>& ptr_y)
{
    uint32_t batches = ValidateInputs(x, y, ptr_x, ptr_y);
    c10_npu::OptionalNPUGuard guard(x.device());
    if (x.size(0) == 0) {
        return at::empty({0}, x.options().dtype(at::kLong));
    }
    TORCH_CHECK(y.size(0) > 0, "nearest: y must contain a point when x is nonempty");
    // A scalar FP16 query needs no feature reduction. Use the device-wide
    // primitives instead of scanning a large candidate set with one warp.
    if (x.scalar_type() == at::kHalf && x.size(0) == 1 && x.size(1) == 1 &&
        y.size(0) >= 32768 && !ptr_x.has_value()) {
        auto delta = y.to(at::kFloat).reshape({-1}) -
                     x.to(at::kFloat).reshape({});
        delta.mul_(delta);
        return delta.argmin().reshape({1});
    }
    TORCH_CHECK(x.numel() <= std::numeric_limits<int64_t>::max() / 4 - y.numel() - x.size(0) - y.size(0),
                "nearest: workspace size overflow");
    auto out = at::empty({x.size(0)}, x.options().dtype(at::kLong));
    auto xc = x.contiguous();
    auto yc = y.contiguous();
    auto work = at::empty({x.numel() + y.numel() + x.size(0) + y.size(0)},
                          x.options().dtype(at::kFloat));
    NearestTiling tiling = MakeTiling(x, y, batches);
    auto stream = c10_npu::getCurrentNPUStream(x.get_device());
    Nearest(xc.data_ptr(), yc.data_ptr(),
            ptr_x ? ptr_x->data_ptr<int64_t>() : nullptr,
            ptr_y ? ptr_y->data_ptr<int64_t>() : nullptr,
            work.data_ptr<float>(), out.data_ptr<int64_t>(),
            x.scalar_type() == at::kHalf, tiling, stream.stream());
    return out;
}

}  // namespace opsgnn

TORCH_LIBRARY_IMPL(torch_cluster, PrivateUse1, m)
{
    m.impl("nearest", TORCH_FN(opsgnn::nearest_npu));
}

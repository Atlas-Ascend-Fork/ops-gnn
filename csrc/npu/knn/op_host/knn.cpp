/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "knn.h"

#include <acl/acl.h>
#include <cstdint>

#include "knn/op_kernel/arch35/knn_kernel.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace opsgnn {

namespace {

struct BatchPointers {
    torch::Tensor ptr_x;
    torch::Tensor ptr_y;
    int64_t batch_size = 1;
    bool batched = false;
};

void ValidatePtr(const torch::Tensor& ptr_cpu, int64_t length,
                 const char* name) {
    const int64_t count = ptr_cpu.numel();
    const int64_t* data = ptr_cpu.data_ptr<int64_t>();
    TORCH_CHECK(count >= 2, "knn: ", name,
                " must contain at least two elements");
    TORCH_CHECK(data[0] == 0, "knn: ", name, " must start with zero");
    TORCH_CHECK(data[count - 1] == length, "knn: ", name,
                " must end with the input length");
    for (int64_t i = 0; i < count; ++i) {
        TORCH_CHECK(data[i] >= 0 && data[i] <= length, "knn: ", name,
                    " contains an out-of-range value");
        if (i > 0) {
            TORCH_CHECK(data[i - 1] <= data[i], "knn: ", name,
                        " must be non-decreasing");
        }
    }
}

BatchPointers ResolveBatchPointers(
    const torch::Tensor& x, const torch::Tensor& y,
    const std::optional<torch::Tensor>& ptr_x,
    const std::optional<torch::Tensor>& ptr_y, int64_t k) {
    const bool has_x = ptr_x.has_value() && ptr_x->defined();
    const bool has_y = ptr_y.has_value() && ptr_y->defined();
    TORCH_CHECK(has_x == has_y,
                "knn: ptr_x and ptr_y must either both be set or both be None");

    BatchPointers result;
    if (!has_x) {
        TORCH_CHECK(k <= x.size(0), "knn: k must not exceed x.size(0)");
        return result;
    }

    TORCH_CHECK(ptr_x->device() == x.device(),
                "knn: ptr_x must be on the same device as x");
    TORCH_CHECK(ptr_y->device() == y.device(),
                "knn: ptr_y must be on the same device as y");
    result.ptr_x = ptr_x->contiguous();
    result.ptr_y = ptr_y->contiguous();
    TORCH_CHECK(result.ptr_x.scalar_type() == torch::kLong &&
                    result.ptr_y.scalar_type() == torch::kLong,
                "knn: batch pointers must be int64");
    TORCH_CHECK(result.ptr_x.dim() == 1 && result.ptr_y.dim() == 1,
                "knn: batch pointers must be one-dimensional");
    TORCH_CHECK(result.ptr_x.numel() == result.ptr_y.numel(),
                "knn: ptr_x and ptr_y must have equal lengths");

    auto ptr_x_cpu = result.ptr_x.cpu().contiguous();
    auto ptr_y_cpu = result.ptr_y.cpu().contiguous();
    ValidatePtr(ptr_x_cpu, x.size(0), "ptr_x");
    ValidatePtr(ptr_y_cpu, y.size(0), "ptr_y");
    result.batch_size = result.ptr_x.numel() - 1;
    TORCH_CHECK(result.batch_size > 0, "knn: batch_size must be positive");

    const int64_t* px = ptr_x_cpu.data_ptr<int64_t>();
    const int64_t* py = ptr_y_cpu.data_ptr<int64_t>();
    for (int64_t batch = 0; batch < result.batch_size; ++batch) {
        const int64_t x_count = px[batch + 1] - px[batch];
        const int64_t y_count = py[batch + 1] - py[batch];
        if (y_count > 0) {
            TORCH_CHECK(k <= x_count,
                        "knn: k exceeds the number of x points in batch ",
                        batch);
        }
    }
    result.batched = result.batch_size > 1;
    return result;
}

aclrtStream ResolveStream(int64_t stream_handle) {
    aclrtStream stream = reinterpret_cast<aclrtStream>(stream_handle);
    TORCH_CHECK(stream != nullptr,
                "knn: a valid current NPU stream handle is required");
    return stream;
}

template <typename T, int DType>
void DispatchKernel(const torch::Tensor& x, const torch::Tensor& y,
                    const BatchPointers& batch, torch::Tensor& output,
                    int64_t k, bool cosine, float* workspace,
                    int64_t* tree, float* bounds,
                    aclrtStream stream) {
    const int64_t pairs = y.size(0) * k;
    int64_t* output_base = output.data_ptr<int64_t>();
    Knn<T, DType>(
        static_cast<T*>(x.data_ptr()), static_cast<T*>(y.data_ptr()),
        batch.batched ? batch.ptr_x.data_ptr<int64_t>() : nullptr,
        batch.batched ? batch.ptr_y.data_ptr<int64_t>() : nullptr,
        output_base, output_base + pairs, k, x.size(0), y.size(0),
        x.size(1), batch.batch_size, cosine, workspace, tree, bounds, stream);
}

void ValidateInputs(const torch::Tensor& x, const torch::Tensor& y, int64_t k) {
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2,
                "knn: x and y must be two-dimensional");
    TORCH_CHECK(x.device() == y.device(),
                "knn: x and y must be on the same device");
    TORCH_CHECK(
        x.device().type() == c10::DeviceType::PrivateUse1,
        "knn: x and y must be NPU tensors");
    TORCH_CHECK(x.size(1) == y.size(1),
                "knn: x and y feature dimensions must match");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                "knn: x and y dtypes must match");
    TORCH_CHECK(x.scalar_type() == torch::kFloat ||
                    x.scalar_type() == torch::kHalf,
                "knn: supported dtypes are float16 and float32");
    TORCH_CHECK(k > 0, "knn: k must be positive");
}

void AllocateTreeStorage(const torch::Tensor& x, torch::Tensor& storage,
                         torch::Tensor& bounds) {
    const int64_t bbox_features = x.scalar_type() == torch::kHalf ? 1 : x.size(1);
    storage = torch::empty(
        {x.size(0) * (2 * KNN_TREE_NODE_FIELDS + 3)},
        x.options().dtype(torch::kLong));
    bounds = torch::empty(
        {2 * x.size(0), 2 * bbox_features + 2},
        x.options().dtype(torch::kFloat));
}

}  // namespace

torch::Tensor knn_npu(
    torch::Tensor x, torch::Tensor y,
    std::optional<torch::Tensor> ptr_x,
    std::optional<torch::Tensor> ptr_y, int64_t k, bool cosine,
    int64_t num_workers, int64_t stream_handle) {
    // Retained for torch_cluster API compatibility; NPU scheduling uses AIV
    // cores and the current stream, not the CPU worker-thread count.
    (void)num_workers;
    ValidateInputs(x, y, k);

    x = x.contiguous();
    y = y.contiguous();
    c10_npu::OptionalNPUGuard device_guard(x.device());

    if (x.size(0) == 0 || y.size(0) == 0) {
        return torch::empty(
            {2, 0}, torch::TensorOptions().dtype(torch::kLong).device(x.device()));
    }

    BatchPointers batch = ResolveBatchPointers(x, y, ptr_x, ptr_y, k);
    auto output = torch::empty(
        {2, y.size(0) * k},
        torch::TensorOptions().dtype(torch::kLong).device(x.device()));
    aclrtStream stream = ResolveStream(stream_handle);
    // Only large-k calls need dynamic Top-K distances. The workspace is
    // proportional to the output, not a full query-candidate distance matrix,
    // and uses the current-stream NPU allocator's asynchronous lifetime.
    torch::Tensor workspace;
    float* workspace_ptr = nullptr;
    if (k > KNN_LOCAL_TOPK_LIMIT) {
        workspace = torch::empty({y.size(0), k}, x.options().dtype(torch::kFloat));
        workspace_ptr = workspace.data_ptr<float>();
    }

    // CPU-reference tie ordering is constructed on NPU from this call's x.
    // Trees contain at most 2*N nodes, plus permutation/inverse/leaf maps.
    // FP16 reference splitting only uses coordinate 0; FP32 uses all F.
    torch::Tensor tree_storage;
    torch::Tensor tree_bounds;
    int64_t* tree_ptr = nullptr;
    float* bounds_ptr = nullptr;
    if (!cosine) {
        AllocateTreeStorage(x, tree_storage, tree_bounds);
        tree_ptr = tree_storage.data_ptr<int64_t>();
        bounds_ptr = tree_bounds.data_ptr<float>();
    }

    if (x.scalar_type() == torch::kFloat) {
        DispatchKernel<float, 0>(
            x, y, batch, output, k, cosine, workspace_ptr, tree_ptr, bounds_ptr, stream);
    } else {
        DispatchKernel<uint16_t, 1>(
            x, y, batch, output, k, cosine, workspace_ptr, tree_ptr, bounds_ptr, stream);
    }

    aclError ret = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    TORCH_CHECK(ret == ACL_SUCCESS, "knn: kernel launch failed: ", ret);
    return output;
}

}  // namespace opsgnn

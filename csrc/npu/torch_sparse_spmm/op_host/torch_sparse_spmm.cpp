/**
 * Copyright (c) 2026 Eternal_ZCL.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "torch_sparse_spmm.h"
#include <algorithm>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <c10/core/DeviceGuard.h>
#include <tiling/platform/platform_ascendc.h>
#include "../op_kernel/arch35/torch_sparse_spmm.h"

namespace opsgnn {
using at::Tensor;
namespace {
int Type(at::ScalarType t)
{
    switch (t) {
        case at::kFloat:
            return 0;
        case at::kHalf:
            return 1;
        case at::kBFloat16:
            return 2;
        case at::kChar:
            return 3;
        case at::kInt:
            return 4;
        default:
            TORCH_CHECK(false, "unsupported SpMM dtype");
    }
}
void Device(const Tensor& x)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "SpMM requires NPU tensors");
}
uint32_t Blocks()
{
    static uint32_t n = []() {
        auto p = platform_ascendc::PlatformAscendCManager::GetInstance();
        return p ? std::max(1u, p->GetCoreNumAiv()) : 1u;
    }();
    return n;
}
void Indices(const Tensor& ptr, const Tensor& col)
{
    Device(ptr);
    Device(col);
    TORCH_CHECK(ptr.device() == col.device() && ptr.scalar_type() == at::kLong && col.scalar_type() == at::kLong,
                "indices must be int64 on the same NPU");
    TORCH_CHECK(ptr.dim() == 1 && ptr.numel() > 0 && col.dim() == 1, "invalid CSR shape");
}
SpmmShape Shape(const Tensor& ptr, const Tensor& col, const Tensor& mat, int64_t width, int reduce)
{
    Indices(ptr, col);
    Device(mat);
    TORCH_CHECK(mat.device() == ptr.device() && mat.dim() >= 2, "mat must have shape [...,N,K] on the same NPU");
    TORCH_CHECK(reduce >= 0 && reduce <= 3, "unsupported reduce");
    TORCH_CHECK(Type(mat.scalar_type()) < 4, "int32 input is unsupported");
    int64_t b = 1;
    for (int64_t i = 0; i < mat.dim() - 2; ++i) {
        b *= mat.size(i);
    }
    return {ptr.numel() - 1, mat.size(-2), mat.size(-1), col.numel(), b, width, reduce};
}
}  // namespace

std::tuple<Tensor, Tensor> TorchSparseSpmmForward(Tensor ptr, Tensor col, c10::optional<Tensor> value, Tensor mat,
                                                  int reduce, int outType)
{
    int64_t width = 1;
    if (value) {
        Device(*value);
        TORCH_CHECK(value->device() == mat.device() && value->scalar_type() == mat.scalar_type(),
                    "value dtype/device must match mat");
        TORCH_CHECK(value->dim() >= 1 && value->size(0) == col.numel(), "value length must equal E");
        for (int64_t i = 1; i < value->dim(); ++i) {
            width *= value->size(i);
        }
    }
    auto s = Shape(ptr, col, mat, width, reduce);
    TORCH_CHECK(width == 1 || width == s.k, "value must contain 1 or K weights per edge");
    int input = Type(mat.scalar_type());
    TORCH_CHECK((input == 0 && outType == 0) || (input == 1 && (outType == 0 || outType == 1)) ||
                    (input == 2 && (outType == 0 || outType == 2)) || (input == 3 && (outType == 0 || outType == 4)),
                "unsupported input/output dtype combination");
    c10::DeviceGuard guard(mat.device());
    ptr = ptr.contiguous();
    col = col.contiguous();
    mat = mat.contiguous();
    // col is the upstream placeholder when has_value=false, never a tensor of ones.
    Tensor v = value ? value->contiguous() : col;
    auto sizes = mat.sizes().vec();
    sizes[sizes.size() - 2] = s.m;
    at::ScalarType ot = outType == 0 ? at::kFloat : outType == 1 ? at::kHalf : outType == 2 ? at::kBFloat16 : at::kInt;
    auto out = at::empty(sizes, mat.options().dtype(ot));
    auto arg = at::empty(reduce >= 2 ? sizes : std::vector<int64_t>{0}, col.options());
    // stream() drains torch_npu's host task queue. Obtain it AFTER contiguous
    // copies, otherwise our direct launch can overtake their queued dispatch.
    auto stream = c10_npu::getCurrentNPUStream(mat.device().index()).stream();
    if (out.numel()) {
        ::TorchSparseSpmm(ptr.data_ptr(), col.data_ptr(), v.data_ptr(), mat.data_ptr(), out.data_ptr(), arg.data_ptr(), s,
                        input, outType, value.has_value(), Blocks(), stream);
    }
    return {out, arg};
}
Tensor TorchSparseSpmmValidate(Tensor ptr, Tensor col, int64_t n)
{
    Indices(ptr, col);
    TORCH_CHECK(n >= 0, "N must be nonnegative");
    c10::DeviceGuard guard(ptr.device());
    ptr = ptr.contiguous();
    col = col.contiguous();
    auto status = at::empty({Blocks()}, ptr.options().dtype(at::kInt));
    ::TorchSparseSpmmValidate(ptr.data_ptr(), col.data_ptr(), status.data_ptr(), ptr.numel() - 1, n, col.numel(), Blocks(),
                            c10_npu::getCurrentNPUStream(ptr.device().index()).stream());
    return status;
}
Tensor TorchSparseSpmmValueGrad(Tensor ptr, Tensor col, Tensor mat, Tensor grad, int reduce, int64_t width)
{
    auto s = Shape(ptr, col, mat, width, reduce);
    TORCH_CHECK(width == 1 || width == s.k, "value width must be 1 or K");
    TORCH_CHECK(reduce < 2 && mat.is_floating_point(), "value backward supports floating sum/mean");
    auto expected = mat.sizes().vec();
    expected[expected.size() - 2] = s.m;
    TORCH_CHECK(grad.device() == mat.device() && grad.is_floating_point() && grad.sizes().vec() == expected,
                "grad must be floating [...,M,K] on the same NPU as mat");
    c10::DeviceGuard guard(mat.device());
    grad = grad.to(at::kFloat).contiguous();
    ptr = ptr.contiguous();
    col = col.contiguous();
    mat = mat.contiguous();
    auto out = at::empty({s.edges, width}, mat.options().dtype(at::kFloat));
    if (out.numel()) {
        ::TorchSparseSpmmValueGrad(ptr.data_ptr(), col.data_ptr(), mat.data_ptr(), grad.data_ptr(), out.data_ptr(), s,
                                Type(mat.scalar_type()), Blocks(),
                                c10_npu::getCurrentNPUStream(mat.device().index()).stream());
    }
    return out;
}
Tensor TorchSparseSpmmConvert(Tensor src, int64_t output, bool toPtr)
{
    Device(src);
    TORCH_CHECK(src.scalar_type() == at::kLong && src.dim() == 1 && output >= 0,
                "conversion requires int64 vector and nonnegative size");
    TORCH_CHECK(toPtr || src.numel() > 0, "CSR ptr must not be empty");
    c10::DeviceGuard guard(src.device());
    src = src.contiguous();
    auto out = at::empty({output}, src.options());
    if (output) {
        ::TorchSparseSpmmConvert(src.data_ptr(), out.data_ptr(), src.numel(), output, toPtr, Blocks(),
                           c10_npu::getCurrentNPUStream(src.device().index()).stream());
    }
    return out;
}
// Internal gather: perm is the stable argsort permutation constructed by the adapter.
Tensor TorchSparseSpmmPermuteValues(Tensor value, Tensor perm)
{
    Device(value);
    Device(perm);
    TORCH_CHECK(value.device() == perm.device() && value.dim() >= 1 && perm.dim() == 1 &&
                    perm.scalar_type() == at::kLong && perm.numel() == value.size(0),
                "invalid value permutation metadata");
    int type = Type(value.scalar_type());
    TORCH_CHECK(type < 4, "unsupported edge-value dtype");
    c10::DeviceGuard guard(value.device());
    value = value.contiguous();
    perm = perm.contiguous();
    int64_t width = 1;
    for (int64_t i = 1; i < value.dim(); ++i) {
        width *= value.size(i);
    }
    auto out = at::empty(value.sizes(), value.options());
    auto stream = c10_npu::getCurrentNPUStream(value.device().index()).stream();
    if (out.numel()) {
        ::TorchSparseSpmmPermuteValues(perm.data_ptr(), value.data_ptr(), out.data_ptr(), out.numel(), width, type, Blocks(),
                                     stream);
    }
    return out;
}
// Read host-side metadata through Tensor's public C++ interface. Inference
// tensors have no counter and must never populate a mutation-sensitive cache.
c10::optional<int64_t> TorchSparseSpmmTensorVersion(const Tensor& tensor)
{
    if (tensor.is_inference()) {
        return c10::nullopt;
    }
    return tensor._version();
}
}  // namespace opsgnn

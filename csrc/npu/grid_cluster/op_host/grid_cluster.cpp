/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "grid_cluster.h"
#include "grid_cluster/op_kernel/arch35/grid_cluster.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {
at::Tensor grid_cluster_npu(at::Tensor pos, at::Tensor size,
    std::optional<at::Tensor> optionalStart, std::optional<at::Tensor> optionalEnd)
{
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) > 0, "grid: pos must be [N,D], D > 0");
    auto dtype = pos.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat || dtype == at::kBFloat16 ||
        dtype == at::kChar || dtype == at::kByte || dtype == at::kShort || dtype == at::kInt,
        "grid: unsupported pos dtype");
    const auto dims = pos.size(1);
    auto check = [&](const at::Tensor& tensor) {
        TORCH_CHECK(tensor.device() == pos.device(), "grid: inputs must be on the same device");
        TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == dims, "grid: expected vector [D]");
    };
    check(size);
    if (optionalStart) check(*optionalStart);
    if (optionalEnd) check(*optionalEnd);
    c10_npu::OptionalNPUGuard guard(pos.device());
    TORCH_CHECK(size.gt(0).logical_and(at::isfinite(size)).all().item<bool>(),
        "grid: size must be positive and finite in every dimension");
    auto out = at::empty({pos.size(0)}, pos.options().dtype(at::kLong));
    if (pos.size(0) == 0) return out;
    auto p = pos.contiguous();
    auto s = size.contiguous();
    // aclnnMinDim/MaxDim do not support narrow integers on 950. int32
    // represents their complete range exactly; cast only the reduced vectors back.
    auto reduction = ((!optionalStart || !optionalEnd) &&
        (dtype == at::kChar || dtype == at::kByte || dtype == at::kShort)) ? p.to(at::kInt) : p;
    auto start = optionalStart ? optionalStart->contiguous() : std::get<0>(at::min(reduction, 0)).to(dtype);
    auto end = optionalEnd ? optionalEnd->contiguous() : std::get<0>(at::max(reduction, 0)).to(dtype);
    if (s.scalar_type() == dtype && start.scalar_type() == dtype && end.scalar_type() == dtype) {
        auto stream = c10_npu::getCurrentNPUStream(pos.device().index());
        GridCluster(p.data_ptr(), s.data_ptr(), start.data_ptr(), end.data_ptr(),
            out.data_ptr<int64_t>(), pos.size(0), dims, static_cast<int>(dtype), stream);
        return out;
    }
    // Mixed dtypes follow ATen promotion, including subtraction before division.
    auto count = at::true_divide(end - start, s).to(at::kLong) + 1;
    auto prefix = at::cat({at::ones({1}, out.options()), at::cumprod(count, 0)}, 0).narrow(0, 0, dims);
    return (at::true_divide(p - start.unsqueeze(0), s.unsqueeze(0)).to(at::kLong) * prefix).sum(1);
}

} // namespace opsgnn

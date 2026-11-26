/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ind2ptr/op_host/ind2ptr.h"
#include "ind2ptr/op_kernel/common/ind2ptr_launch.h"

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

// Launch AscendC kernel on the current NPU stream and return (async).
torch::Tensor ind2ptr(torch::Tensor ind, int64_t M)
{
    TORCH_CHECK(ind.dim() == 1, "ind2ptr: ind must be 1-D");
    TORCH_CHECK(ind.scalar_type() == at::kLong, "ind2ptr: ind must be int64 (torch.long)");
    TORCH_CHECK(ind.device().type() == c10::DeviceType::PrivateUse1,
                "ind2ptr: ind must be an NPU tensor");
    TORCH_CHECK(M >= 0, "ind2ptr: M must be non-negative");

    c10_npu::OptionalNPUGuard device_guard(ind.device());

    auto out = torch::empty({M + 1}, ind.options());
    if (ind.numel() == 0) {
        return out.zero_();
    }

    if (!ind.is_contiguous()) {
        ind = ind.contiguous();
    }

    const torch::Tensor indHost = ind.cpu();
    const int64_t* indValues = indHost.data_ptr<int64_t>();
    const int64_t count = indHost.numel();
    for (int64_t i = 0; i < count; ++i) {
        const int64_t value = indValues[i];
        TORCH_CHECK(value >= 0 && value < M,
                    "ind2ptr: ind[", i, "]=", value, " out of range [0, ", M, ")");
        TORCH_CHECK(i == 0 || indValues[i - 1] <= value,
                    "ind2ptr: ind must be non-decreasing, ind[", i - 1, "]=", indValues[i - 1],
                    " > ind[", i, "]=", value);
    }

    auto stream = c10_npu::getCurrentNPUStream(ind.device().index());
    Ind2Ptr(ind.data_ptr<int64_t>(), out.data_ptr<int64_t>(), M, ind.numel(), stream);
    return out;
}

}  // namespace opsgnn

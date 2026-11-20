/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ptr2ind/op_host/ptr2ind.h"
#include "ptr2ind/op_kernel/arch35/ptr2ind.h"

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

// Launch AscendC kernel on the current NPU stream and return (async).
torch::Tensor ptr2ind(torch::Tensor ptr, int64_t E)
{
    TORCH_CHECK(ptr.dim() == 1, "ptr2ind: ptr must be 1-D");
    TORCH_CHECK(ptr.scalar_type() == at::kLong, "ptr2ind: ptr must be int64 (torch.long)");
    TORCH_CHECK(E >= 0, "ptr2ind: E must be non-negative");
    TORCH_CHECK(ptr.numel() >= 1, "ptr2ind: ptr must have at least one element");

    c10_npu::OptionalNPUGuard device_guard(ptr.device());

    if (!ptr.is_contiguous()) {
        ptr = ptr.contiguous();
    }

    // The kernel derives GM write addresses directly from ptr values, so an
    // invalid CSR pointer must be rejected here (same contract as knn/radius:
    // start at 0, non-decreasing, end at E, values kept in [0, E]).
    const auto ptr_cpu = ptr.cpu();
    const int64_t count = ptr_cpu.numel();
    const int64_t* p = ptr_cpu.data_ptr<int64_t>();
    TORCH_CHECK(p[0] == 0, "ptr2ind: ptr must start with 0");
    TORCH_CHECK(p[count - 1] == E, "ptr2ind: ptr must end with num_edges");
    for (int64_t i = 0; i < count; ++i) {
        TORCH_CHECK(p[i] >= 0 && p[i] <= E, "ptr2ind: ptr contains an out-of-range value");
        if (i > 0) {
            TORCH_CHECK(p[i - 1] <= p[i], "ptr2ind: ptr must be non-decreasing");
        }
    }

    auto out = torch::empty({E}, ptr.options());
    if (E == 0) {
        return out;
    }

    auto stream = c10_npu::getCurrentNPUStream(ptr.device().index());
    Ptr2Ind(ptr.data_ptr<int64_t>(), out.data_ptr<int64_t>(), ptr.numel() - 1, E, stream);
    return out;
}

}  // namespace opsgnn

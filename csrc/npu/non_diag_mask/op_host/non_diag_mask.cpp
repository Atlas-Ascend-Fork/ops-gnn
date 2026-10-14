/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Host entry for torch_sparse::non_diag_mask (NPU).
 */

#include "non_diag_mask.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <c10/core/DeviceGuard.h>

#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "common/torch_sparse_shared_checks.h"
#include "non_diag_mask/op_kernel/arch35/non_diag_mask_kernel.h"

namespace opsgnn {

torch::Tensor non_diag_mask_npu(
    torch::Tensor row,
    torch::Tensor col,
    int64_t M,
    int64_t N,
    int64_t k)
{
    torch_sparse_utils::CheckInt64Vector(row, "row");
    torch_sparse_utils::CheckInt64Vector(col, "col");
    torch_sparse_utils::CheckSameDevice(row, col, "row", "col");
    TORCH_CHECK(row.numel() == col.numel(),
                "row and col must have the same number of elements");
    TORCH_CHECK(M >= 0, "M must be non-negative");
    TORCH_CHECK(N >= 0, "N must be non-negative");

    const c10::OptionalDeviceGuard deviceGuard(row.device());

    row = row.contiguous();
    col = col.contiguous();

    // Task-book semantics: negative diagonal length is clamped to zero.
    const int64_t rawDiag = k >= 0 ? std::min(M, N - k) : std::min(M + k, N);
    const int64_t numDiag = std::max<int64_t>(rawDiag, 0);
    const int64_t edgeCount = row.numel();
    TORCH_CHECK(edgeCount <= std::numeric_limits<int64_t>::max() - numDiag,
                "non_diag_mask output length exceeds int64 range");

    torch::Tensor mask = torch::empty(
        {edgeCount + numDiag}, row.options().dtype(torch::kBool));

    if (edgeCount + numDiag > 0) {
        const uint32_t availableCores =
            torch_sparse_utils::ResolveAivCoreCount();
        // NPUStream::stream() is torch_npu's queue-to-ACL bridge; the kernel
        // is appended to the caller's stream and stays asynchronous.
        aclrtStream stream = c10_npu::getCurrentNPUStream(row.get_device()).stream();
        const NonDiagMaskLaunchParams params{
            row.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(),
            reinterpret_cast<uint8_t*>(mask.data_ptr<bool>()),
            static_cast<uint64_t>(edgeCount),
            static_cast<uint64_t>(numDiag),
            k,
            M,
            N,
            availableCores,
            stream};
        LaunchNonDiagMaskKernel(params);
    }
    return mask;
}

} // namespace opsgnn

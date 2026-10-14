/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Shared host-side input validation helpers (no device synchronization).
 */

#pragma once

#include <cstdint>
#include <torch/extension.h>
#include <limits>

#include <c10/core/DeviceType.h>
#include "tiling/platform/platform_ascendc.h"

namespace opsgnn {

namespace torch_sparse_utils {

inline uint32_t ResolveAivCoreCount()
{
    auto *platform =
        platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr,
                "torch_sparse_utils: failed to get AscendC platform information");
    const uint32_t reportedCores = platform->GetCoreNumAiv();
    return reportedCores == 0 ? 1U : reportedCores;
}

inline void CheckInt64Vector(const torch::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Long,
                name, " must have dtype torch.int64");
    TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
}

inline void CheckSameDevice(const torch::Tensor &first, const torch::Tensor &second,
                            const char *firstName, const char *secondName)
{
    TORCH_CHECK(first.get_device() == second.get_device(),
                firstName, " and ", secondName, " must be on the same NPU device");
}

inline void CheckOutputSize(int64_t rows, int64_t columns, int64_t elementSize)
{
    TORCH_CHECK(elementSize > 0, "element size must be positive");
    const int64_t maxElements = std::numeric_limits<int64_t>::max() / elementSize;
    TORCH_CHECK(columns >= 0 && rows >= 0 &&
                    (columns == 0 || rows <= maxElements / columns),
                "output size exceeds the supported int64 byte range");
}

} // namespace torch_sparse_utils

} // namespace opsgnn

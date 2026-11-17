/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Shared host-side acquisition of the default NPU generator's Philox state.
 */

#pragma once

#include <cstdint>
#include <mutex>
#include <utility>

#include <ATen/core/Generator.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>

namespace opsgnn {

constexpr uint64_t RNG_CALL_INCREMENT = 4;

inline std::pair<uint64_t, uint64_t> AcquirePhiloxState(c10::DeviceIndex deviceIndex)
{
    const at::Generator &generator = at_npu::detail::getDefaultNPUGenerator(deviceIndex);
    auto *generatorImpl = at::check_generator<at_npu::NPUGeneratorImpl>(generator);
    std::lock_guard<std::mutex> lock(generatorImpl->mutex_);
    return generatorImpl->philox_engine_inputs(RNG_CALL_INCREMENT);
}

} // namespace opsgnn

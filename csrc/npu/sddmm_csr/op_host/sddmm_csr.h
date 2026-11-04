/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <optional>
#include <string>
#include <torch/extension.h>

namespace opsgnn {

torch::Tensor SddmmCsrForward(const torch::Tensor& indptr, const torch::Tensor& indices,
                              const std::optional<torch::Tensor>& lhs,
                              const std::optional<torch::Tensor>& rhs,
                              const std::string& op, int64_t lhsTarget, int64_t rhsTarget,
                              const std::optional<torch::Tensor>& out);

}  // namespace opsgnn

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_LOG_SOFTMAX_LAUNCH_H
#define OPS_GNN_SCATTER_LOG_SOFTMAX_LAUNCH_H

#include <acl/acl.h>
#include <cstdint>

#include "scatter_softmax/op_kernel/common/scatter_softmax_tiling.h"

namespace opsgnn {

// Grouped log-softmax forward, evaluated entirely in fp32 on the device:
//
//   maximum = max over the group of src
//   out     = (src - maximum) - log(sum over the group of exp(src - maximum))
//
// `maximum` must hold SCATTER_SOFTMAX_NEUTRAL_MAXIMUM on entry and
// `sumWorkspace` must hold zeros; both hold
// batch * groupCount * afterPadded floats - the same workspace
// scatter_softmax uses. `out` has the same shape and dtype as `src`. The three
// internal passes run in order on `stream`.
void ScatterLogSoftmaxForward(void* src, const int32_t* index, void* out,
                              float* maximum, float* sumWorkspace,
                              ScatterSoftmaxDType dtype,
                              const ScatterSoftmaxTilingData& tiling,
                              aclrtStream stream);

// Analytic backward of the grouped log-softmax:
//
//   gradSrc = gradOut - exp(out) * sum over the group of gradOut
//
// `out` is the forward result, so `exp(out)` is the softmax probability that
// the log-domain Jacobian weights the reduced gradient with. `sumWorkspace`
// must hold zeros on entry; `gradSrc` has the same shape and dtype as `out`.
void ScatterLogSoftmaxBackward(const void* out, const void* gradOut,
                               const int32_t* index, void* gradSrc,
                               float* sumWorkspace, ScatterSoftmaxDType dtype,
                               const ScatterSoftmaxTilingData& tiling,
                               aclrtStream stream);

}  // namespace opsgnn

#endif  // OPS_GNN_SCATTER_LOG_SOFTMAX_LAUNCH_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_LOGSUMEXP_LAUNCH_H
#define OPS_GNN_SCATTER_LOGSUMEXP_LAUNCH_H

#include <acl/acl.h>
#include <cstdint>

#include "scatter_softmax/op_kernel/common/scatter_softmax_tiling.h"

namespace opsgnn {

// Grouped log-sum-exp forward, evaluated entirely in fp32 on the device:
//
//   maximum = max over the group of src
//   sum     = sum over the group of exp(src - min(maximum, FLT_MAX))
//   out     = maximum + ln(sum)
//
// `out` holds `tiling.outRows * tiling.after` elements - one row per
// (batch, group) pair. `maximum` must hold SCATTER_SOFTMAX_NEUTRAL_MAXIMUM on
// entry and `sumWorkspace` must hold zeros; both hold
// batch * groupCount * afterPadded floats, the workspace scatter_softmax uses.
// The three internal passes run in order on `stream`.
void ScatterLogSumExpForward(void* src, const int32_t* index, void* out,
                             float* maximum, float* sumWorkspace,
                             ScatterSoftmaxDType dtype,
                             const ScatterSoftmaxTilingData& tiling,
                             aclrtStream stream);

// Analytic backward of the grouped log-sum-exp:
//
//   gradSrc = gradOut[group] * exp(src - out[group])
//
// `src` is the forward input, `out` the forward result; both `out` and
// `gradOut` hold one row per (batch, group) pair, while `gradSrc` has the shape
// and dtype of `src`. One elementwise pass runs on `stream`.
void ScatterLogSumExpBackward(const void* src, const void* out, const void* gradOut,
                              const int32_t* index, void* gradSrc,
                              ScatterSoftmaxDType dtype,
                              const ScatterSoftmaxTilingData& tiling,
                              aclrtStream stream);

}  // namespace opsgnn

#endif  // OPS_GNN_SCATTER_LOGSUMEXP_LAUNCH_H

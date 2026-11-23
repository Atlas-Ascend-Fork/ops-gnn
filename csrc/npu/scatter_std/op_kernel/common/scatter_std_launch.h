/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_STD_LAUNCH_H
#define OPS_GNN_SCATTER_STD_LAUNCH_H

#include <acl/acl.h>
#include <cstdint>

#include "scatter_softmax/op_kernel/common/scatter_softmax_tiling.h"

namespace opsgnn {

// Grouped standard deviation forward, evaluated entirely in fp32 on the device:
//
//   count     = number of rows of every group
//   sum       = sum over the group of src
//   mean      = sum / count
//   deviation = sum over the group of (src - mean)^2
//   out       = sqrt(deviation / (count - correction))
//
// `out` holds `tiling.outRows * tiling.after` elements - one row per
// (batch, group) pair. `sumWorkspace` and `deviation` must hold zeros on entry
// and each hold batch * groupCount * afterPadded floats, the workspace
// scatter_softmax uses; `countWorkspace` holds one fp32 counter per group
// (countElems floats) and is written by the first pass. On
// return `sumWorkspace` holds the group means, which the backward reuses. The
// three internal passes run in order on `stream`.
void ScatterStdForward(void* src, const int32_t* index, void* out,
                       float* sumWorkspace, float* deviation, float* countWorkspace,
                       ScatterSoftmaxDType dtype,
                       const ScatterSoftmaxTilingData& tiling,
                       aclrtStream stream);

// Analytic backward of the grouped standard deviation:
//
//   gradSrc = gradOut[group] * (src - mean[group]) / ((n - correction) * std)
//
// `src` is the forward input, `out` the forward result, and `meanWorkspace` /
// `countWorkspace` are the two tensors the forward returned. `out` and `gradOut`
// hold one row per (batch, group) pair, while `gradSrc` has the shape and dtype
// of `src`. One elementwise pass runs on `stream`.
void ScatterStdBackward(const void* src, const void* out, const void* gradOut,
                        const int32_t* index, void* gradSrc, float* meanWorkspace,
                        float* countWorkspace, ScatterSoftmaxDType dtype,
                        const ScatterSoftmaxTilingData& tiling, aclrtStream stream);

}  // namespace opsgnn

#endif  // OPS_GNN_SCATTER_STD_LAUNCH_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_EDGE_SOFTMAX_LAUNCH_H
#define OPS_GNN_EDGE_SOFTMAX_LAUNCH_H

#include <acl/acl.h>
#include <cstdint>

#include "scatter_softmax/op_kernel/common/scatter_softmax_tiling.h"

namespace opsgnn {

// Softmax over the edges of every CSR segment, evaluated entirely in fp32 on
// the device:
//
//   maximum = max over the segment of src
//   out     = exp(src - maximum) / sum over the segment of exp(src - maximum)
//
// `indptr` holds `groupCount + 1` int32 boundaries; `maximum` must hold
// SCATTER_SOFTMAX_NEUTRAL_MAXIMUM on entry and `sumWorkspace` must hold zeros.
// Both workspaces hold batch * groupCount * afterPadded floats and `out` has the
// same shape and dtype as `src`. The three passes run in order on `stream`.
void EdgeSoftmaxForward(void* src, const int32_t* indptr, void* out,
                        float* maximum, float* sumWorkspace,
                        ScatterSoftmaxDType dtype,
                        const ScatterSoftmaxTilingData& tiling,
                        aclrtStream stream);

// Analytic backward of the edge softmax:
//
//   gradSrc = out * (gradOut - sum over the segment of out * gradOut)
//
// `sumWorkspace` must hold zeros on entry; `gradSrc` has the same shape and
// dtype as `out`.
void EdgeSoftmaxBackward(const void* out, const void* gradOut,
                         const int32_t* indptr, void* gradSrc,
                         float* sumWorkspace, ScatterSoftmaxDType dtype,
                         const ScatterSoftmaxTilingData& tiling,
                         aclrtStream stream);

}  // namespace opsgnn

#endif  // OPS_GNN_EDGE_SOFTMAX_LAUNCH_H

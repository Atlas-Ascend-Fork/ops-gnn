/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SEGMENT_LOGSUMEXP_LAUNCH_H
#define OPS_GNN_SEGMENT_LOGSUMEXP_LAUNCH_H

#include <acl/acl.h>
#include <cstdint>

#include "segment_logsumexp/op_kernel/common/segment_logsumexp_tiling.h"

namespace opsgnn {

enum SegmentLogSumExpDType {
    SEGMENT_LOGSUMEXP_F16 = 0,
    SEGMENT_LOGSUMEXP_F32 = 1,
    SEGMENT_LOGSUMEXP_BF16 = 2,
};

// Numerically stable log-sum-exp over CSR segments:
//
//   out[s] = log(sum over rows j in [indptr[s], indptr[s + 1]) of exp(src[j]))
//          = shift + log(sum of exp(src[j] - shift))
//
// Every segment is reduced in fp32, so a long float16/bfloat16 segment is not
// limited by the 65504 representation limit of its source dtype. ``indptr`` is
// an int32 device vector of ``segments + 1`` entries; rows before ``indptr[0]``
// belong to no segment and are never read. A segment with no rows yields
// ``-inf``. ``out`` is a device tensor of ``segments * features`` elements.
//
// ``splitMax`` and ``splitSum`` are fp32 workspaces of ``2 * tasks * colTile``
// elements: the per-chunk maximum and exponential sum that the phases exchange.
// They stay on the device and are released by the caller.
void SegmentLogSumExp(const void* src, const int32_t* indptr, void* out,
                      float* splitMax, float* splitSum,
                      SegmentLogSumExpDType dtype,
                      const SegmentLogSumExpTilingData& tiling,
                      aclrtStream stream);

}  // namespace opsgnn

#endif  // OPS_GNN_SEGMENT_LOGSUMEXP_LAUNCH_H

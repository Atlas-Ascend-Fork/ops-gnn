/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * AscendC device implementation of the edge softmax shared by arch22 (A2/A3)
 * and arch35 (950).
 *
 * The operator is the grouped softmax of the scatter family driven by a CSR
 * pointer vector: a segment is a contiguous row range, so the label of a row is
 * the segment that contains it and no per-row index vector is materialised. The
 * phases, the row batching, the fp32 workspace and the atomic reduction are the
 * ones of `scatter_softmax` (see scatter_softmax_kernel.h); only the compile-time
 * mode differs, and it selects the CSR group cursor of `LoadGroups`:
 *
 *   Maximum    for every row: maximum[max(src)]              -> fp32 maximum
 *   Exp        for every row: out = exp(src - maximum), sum sum += out
 *   Normalize  for every row: out /= sum
 *   GradSum    for every row: sum sum += out * grad_out
 *   Grad       for every row: grad_src = out * (grad_out - sum)
 *
 * A row that starts deep inside the segment range is located with a binary
 * search over `indptr`; from there the cursor only moves forwards, so a segment
 * boundary costs one scalar pointer read rather than one label read per edge.
 */

#ifndef OPS_GNN_EDGE_SOFTMAX_KERNEL_H
#define OPS_GNN_EDGE_SOFTMAX_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "edge_softmax/op_kernel/common/edge_softmax_launch.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_entry.h"

SCATTER_GROUPED_SOFTMAX_OPERATOR(EdgeSoftmax, EdgeLaunch, kModeEdgeCsr)

#endif  // OPS_GNN_EDGE_SOFTMAX_KERNEL_H

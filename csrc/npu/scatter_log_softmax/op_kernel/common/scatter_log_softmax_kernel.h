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
 * AscendC device implementation of the grouped log-softmax shared by arch22
 * (A2/A3) and arch35 (950).
 *
 * The operator is the log-domain sibling of scatter_softmax and reuses its
 * five-pass pipeline - the same tiling, the same fp32 reduction workspace and
 * the same per-group accumulation (MTE3 atomic unit, or a private per-core
 * accumulator when the workspace fits in UB) - with the compile-time log mode
 * of the shared phase bodies turned on:
 *
 *   Maximum    for every row: maximum[max(src)]              -> fp32 maximum
 *   Exp        for every row: sum += exp(src - maximum)
 *   Normalize  for every row: out = (src - maximum) - ln(sum)
 *   GradSum    for every row: sum += grad_out
 *   Grad       for every row: grad_src = grad_out - exp(out) * sum
 *
 * The normalization takes the logarithm of the group sum instead of dividing
 * the stored exponential: `(src - maximum) - ln(sum)` is finite whenever the
 * group holds a finite maximum, so a group whose `exp` values underflow keeps a
 * usable result, and the exponential never has to be written and read back
 * between the second and the third pass.
 *
 * Two consequences of the log form shape the remaining passes:
 *
 *   - the exponential pass only contributes the sum, so `Log` skips its store;
 *   - the log-domain Jacobian is `d out_j / d src_i = delta_ij - p_i`, so the
 *     backward reduces the plain group sum of `grad_out` and weights it by the
 *     probability afterwards. `exp(out)` recovers that probability from the
 *     saved log output, so no forward probability tensor is kept.
 *
 * A group of `-inf` therefore ends up as `-inf - ln(0) = NaN`, exactly like
 * `torch.log_softmax`, and a `+inf` member turns the feature column into NaN
 * through the shared maximum pass.
 *
 * The operator itself is emitted by the generator in
 * scatter_softmax_kernel_entry.h: the kernel entries, the launcher and the two
 * exported host entry points are the ones of scatter_softmax with the log mode
 * switched on, so this file only names them.
 */

#ifndef OPS_GNN_SCATTER_LOG_SOFTMAX_KERNEL_H
#define OPS_GNN_SCATTER_LOG_SOFTMAX_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "scatter_log_softmax/op_kernel/common/scatter_log_softmax_launch.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_entry.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_phase.h"

SCATTER_GROUPED_SOFTMAX_OPERATOR(ScatterLogSoftmax, LogLaunch, true)

#endif  // OPS_GNN_SCATTER_LOG_SOFTMAX_KERNEL_H

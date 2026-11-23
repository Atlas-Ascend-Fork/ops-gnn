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
 * AscendC device implementation of the grouped standard deviation shared by
 * arch22 (A2/A3) and arch35 (950).
 *
 * The operator is the second-moment sibling of the grouped log-sum-exp: it
 * keeps that operator's tiling, its fp32 reduction workspace and its per-group
 * accumulation (MTE3 atomic unit, or a private per-core accumulator when the
 * workspace fits in UB), and writes one row per (batch, group) pair:
 *
 *   Sum        for every row:    sum[group] += src, count[group] += 1
 *   Deviation  for every row:    deviation[group] += (src - sum[group] / n)^2
 *   Finalize   for every group:  mean = sum[group] / n  (kept for the backward)
 *                                out  = sqrt(deviation[group] / (n - correction))
 *   GradElement for every row:   grad_src = grad_out[group] * (src - mean[group])
 *                                          / ((n - correction) * out[group])
 *
 * Two full passes over the source and one pass over the group rows: the first
 * accumulates the group sums and the member counts, the second re-centers every
 * row on the mean of its group. The mean is derived from the sum and the count
 * rather than being stored, and the deviation is accumulated around the actual
 * mean instead of being expanded into `sum(x^2) - n * mean^2`, which is what
 * keeps a group like `1000 +/- 0.01` from cancelling its own significant digits.
 *
 * Edge cases follow `torch.std`:
 *
 *   - an empty group is NaN;
 *   - the single member of an unbiased deviation (n - 1 == 0) is NaN, while the
 *     biased one is 0;
 *   - a group that holds `NaN` or `+/-inf` is NaN, because the mean and the
 *     centered deviation are;
 *   - the backward of a group without spread divides zero by zero and is NaN.
 *
 * The operator itself is emitted by the generator in
 * scatter_softmax_kernel_entry.h, so this file only names it.
 */

#ifndef OPS_GNN_SCATTER_STD_KERNEL_H
#define OPS_GNN_SCATTER_STD_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_entry.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_phase.h"
#include "scatter_std/op_kernel/common/scatter_std_launch.h"

SCATTER_GROUPED_STD_OPERATOR(ScatterStd, StdLaunch)

#endif  // OPS_GNN_SCATTER_STD_KERNEL_H

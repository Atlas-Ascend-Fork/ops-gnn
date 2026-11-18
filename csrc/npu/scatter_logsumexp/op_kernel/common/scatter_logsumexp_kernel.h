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
 * AscendC device implementation of the grouped log-sum-exp shared by arch22
 * (A2/A3) and arch35 (950).
 *
 * The operator is the reduction sibling of scatter_softmax: it keeps that
 * operator's tiling, its fp32 reduction workspace and its per-group
 * accumulation (MTE3 atomic unit, or a private per-core accumulator when the
 * workspace fits in UB), but writes one row per (batch, group) pair instead of
 * one row per source row:
 *
 *   Maximum    for every row:      maximum[max(src)]           -> fp32 maximum
 *   Exp        for every row:      sum += exp(src - maximum)
 *   Finalize   for every group:    out = maximum + ln(sum)
 *   GradElement for every row:     grad_src = grad_out[group] * exp(src - out[group])
 *
 * The reduced output is `outRows = batch * groupCount` rows, so the finalize
 * pass walks its own core split (see the `out*` tiling fields) and reads no
 * group labels: the dense workspace rows are exactly the output rows.
 *
 * Edge cases follow `torch.logsumexp` and the elementwise backward of
 * `torch_scatter`'s combination:
 *
 *   - an empty group keeps the neutral maximum and a zero sum, so its output is
 *     `-inf`;
 *   - a group of `-inf` has a zero exponential sum as well and is `-inf` too;
 *   - a group that holds `+inf` is `+inf`: the exponential pass clamps the
 *     subtracted maximum to the largest finite float, which makes the infinite
 *     member contribute `exp(+inf)` and every finite member `exp(-inf) = 0`
 *     instead of the NaN a `+inf - +inf` would produce;
 *   - a group that holds NaN is NaN through the exponential sum;
 *   - the backward of an all-`-inf` group is NaN (`-inf - (-inf)`), the
 *     gradient of a `+inf` member is NaN, and the gradient of its finite
 *     siblings is zero.
 *
 * The operator itself is emitted by the generator in
 * scatter_softmax_kernel_entry.h, so this file only names it.
 */

#ifndef OPS_GNN_SCATTER_LOGSUMEXP_KERNEL_H
#define OPS_GNN_SCATTER_LOGSUMEXP_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "scatter_logsumexp/op_kernel/common/scatter_logsumexp_launch.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_entry.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_phase.h"

SCATTER_GROUPED_LOGSUMEXP_OPERATOR(ScatterLogSumExp, LogSumExpLaunch)

#endif  // OPS_GNN_SCATTER_LOGSUMEXP_KERNEL_H

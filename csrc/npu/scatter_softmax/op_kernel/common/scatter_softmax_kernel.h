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
 * AscendC device implementation of the grouped softmax shared by arch22
 * (A2/A3) and arch35 (950).
 *
 * The group of a row is given by `index`, which is not sorted, so the reduced
 * values of every (batch, group, feature) slot are accumulated by the MTE3
 * atomic unit (or, when the whole reduction workspace fits in UB, by a private
 * per-core accumulator that is flushed once at the end of the pass):
 *
 *   Maximum    for every row: maximum[max(src)]              -> fp32 maximum
 *   Exp        for every row: out = exp(src - maximum), sum sum += out
 *   Normalize  for every row: out /= sum
 *   GradSum    for every row: sum sum += out * grad_out
 *   Grad       for every row: grad_src = out * (grad_out - sum)
 *
 * Two scheduling decisions keep the passes off the "one synchronous row at a
 * time" floor that a naive per-row loop hits on a 64-core part:
 *
 *   1. a core owns a contiguous chunk of rows, so `rowBatch` consecutive rows
 *      are loaded, reduced and stored as one batch - the GM round trip of the
 *      first row is hidden behind the transfers of the rest, and the three
 *      pipeline stages synchronise once per batch instead of once per row;
 *   2. when `batch * groupCount * afterPadded` fits the accumulator budget, the
 *      reduction workspace lives in UB for the whole pass: the reduction
 *      phases accumulate in place and touch the workspace once per core, and
 *      every pass that reads it reads it once instead of once per row. A
 *      million-row single-group shape would otherwise re-read the same 512-byte
 *      line a million times, which costs more than the arithmetic by an order
 *      of magnitude.
 *
 * The fp32 workspace pads each row to a 32-byte multiple so every atomic
 * transfer is aligned. All arithmetic is done in fp32 regardless of the source
 * dtype, which keeps a long fp16 group (exp-sum above 65504) from collapsing.
 *
 * Every vector instruction writes at a 32-byte aligned buffer base: when a tile
 * leaves lanes past the meaningful ones, they are filled by duplicating the
 * neutral value over the whole tile before the meaningful lanes are overwritten.
 * A full tile has no such lanes and skips the fill.
 *
 * The five passes are shared with the log-domain sibling (scatter_log_softmax),
 * which instantiates the same bodies with the compile-time `Log` mode enabled
 * instead of carrying a second copy of the row driver and the reduction
 * bookkeeping. Both operators are emitted by the generator in
 * scatter_softmax_kernel_entry.h, so the phase entries, the launcher and the
 * exported host entry points are written once.
 */

#ifndef OPS_GNN_SCATTER_SOFTMAX_KERNEL_H
#define OPS_GNN_SCATTER_SOFTMAX_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_entry.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_phase.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_launch.h"

SCATTER_GROUPED_SOFTMAX_OPERATOR(ScatterSoftmax, Launch, kModeSoftmax)

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_H

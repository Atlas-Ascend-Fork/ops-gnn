/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file csr_row_wise_sampling_entry_kernel.h
 * @brief Kernel entry points for weighted CSR row-wise sampling.
 */

#pragma once

#include "csr_row_wise_sampling/op_kernel/arch22/csr_row_wise_sampling_kernel.h"

extern "C" __global__ __aicore__ void csr_row_wise_sampling_int32_f32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows, GM_ADDR prob,
    GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
    GM_ADDR row_split, GM_ADDR out_starts, GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KernelCsrRowWiseSampling<int32_t> op;
  TPipe pipe;
  op.Init(indptr, indices, data, rows, prob, out_ptr, out_rows, out_cols,
          out_idxs, row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_row_wise_sampling_int64_f32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows, GM_ADDR prob,
    GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
    GM_ADDR row_split, GM_ADDR out_starts, GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KernelCsrRowWiseSampling<int64_t> op;
  TPipe pipe;
  op.Init(indptr, indices, data, rows, prob, out_ptr, out_rows, out_cols,
          out_idxs, row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_row_wise_sampling_num_picks_int32(
    GM_ADDR rows, GM_ADDR indptr, GM_ADDR data, GM_ADDR prob, GM_ADDR out,
    GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KernelCsrRowWiseSamplingNumPicks<int32_t> op;
  TPipe pipe;
  op.Init(rows, indptr, data, prob, out, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_row_wise_sampling_num_picks_int64(
    GM_ADDR rows, GM_ADDR indptr, GM_ADDR data, GM_ADDR prob, GM_ADDR out,
    GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KernelCsrRowWiseSamplingNumPicks<int64_t> op;
  TPipe pipe;
  op.Init(rows, indptr, data, prob, out, tiling_ptr, &pipe);
  op.Process();
}

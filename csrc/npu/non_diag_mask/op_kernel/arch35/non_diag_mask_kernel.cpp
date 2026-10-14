/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Ascend C (SIMT) kernel for torch_sparse::non_diag_mask on Ascend 950 (arch35).
 *
 * Semantics (task-book / official-test reference):
 *   num_diag = max(0, k >= 0 ? min(M, N - k) : min(M + k, N))
 *   mask     = cat([ row + k != col , zeros(num_diag, bool) ])
 */

#include "non_diag_mask_kernel.h"

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"
#include "utils/debug/asc_assert.h"

namespace {
constexpr uint32_t NON_DIAG_MASK_THREADS_PER_BLOCK = 256;

__simt_vf__ __aicore__ LAUNCH_BOUND(NON_DIAG_MASK_THREADS_PER_BLOCK) void NonDiagMaskSimt(
    __gm__ const int64_t *row,
    __gm__ const int64_t *col,
    __gm__ uint8_t *mask,
    uint64_t total,
    uint64_t edgeCount,
    int64_t diagOffset,
    int64_t rowCount,
    int64_t columnCount)
{
    const uint32_t blockIndex = blockIdx.x;
    const uint32_t blockCount = gridDim.x;
    const uint32_t threadIndex = threadIdx.x;
    const uint32_t threadCount = blockDim.x;
    const uint64_t globalThread =
        static_cast<uint64_t>(blockIndex) * threadCount + threadIndex;
    const uint64_t globalStride =
        static_cast<uint64_t>(blockCount) * threadCount;

    for (uint64_t i = globalThread; i < total; i += globalStride) {
        uint8_t value = 0;
        if (i < edgeCount) {
            const int64_t r = row[i];
            const int64_t c = col[i];
            if (r < 0 || r >= rowCount || c < 0 || c >= columnCount) {
                ascendc_assert(false, "non_diag_mask: COO index out of range");
                __trap();  // Remains active when assertion printing is disabled.
                return;
            }
            // A difference of two legal nonnegative int64 indices is representable.
            value = (c - r) != diagOffset ? 1 : 0;
        }
        mask[i] = value;
    }
}

__attribute__((aiv)) __global__ __aicore__ void NonDiagMaskKernel(
    __gm__ const int64_t *row,
    __gm__ const int64_t *col,
    __gm__ uint8_t *mask,
    uint64_t total,
    uint64_t edgeCount,
    int64_t diagOffset,
    int64_t rowCount,
    int64_t columnCount)
{
    asc_vf_call<NonDiagMaskSimt>(
        dim3{NON_DIAG_MASK_THREADS_PER_BLOCK, 1, 1},
        row, col, mask, total, edgeCount, diagOffset, rowCount, columnCount);
}
} // namespace

namespace opsgnn {

void LaunchNonDiagMaskKernel(const NonDiagMaskLaunchParams &params)
{
    const uint64_t total = params.edgeCount + params.numDiag;
    if (total == 0) {
        return;
    }

    const uint64_t requiredCores =
        (total + NON_DIAG_MASK_THREADS_PER_BLOCK - 1) / NON_DIAG_MASK_THREADS_PER_BLOCK;
    uint32_t launchBlocks = static_cast<uint32_t>(
        requiredCores < params.availableCores ? requiredCores : params.availableCores);
    launchBlocks = launchBlocks == 0 ? 1 : launchBlocks;

    NonDiagMaskKernel<<<launchBlocks, nullptr, params.stream>>>(
        params.row, params.col, params.mask, total, params.edgeCount, params.diagOffset,
        params.rowCount, params.columnCount);
}

} // namespace opsgnn

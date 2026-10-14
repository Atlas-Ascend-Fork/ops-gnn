/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Ascend C (SIMT) kernel entry for torch_sparse::random_walk on Ascend 950 (arch35).
 */

#pragma once

#include <acl/acl.h>
#include <cstdint>

namespace opsgnn {

struct TorchSparseRandomWalkLaunchParams {
    uint64_t startCount;   // S
    uint64_t walkLength;   // L (number of steps; output has L+1 columns)
    uint64_t nnz;          // col.numel()
    uint64_t nodeCount;    // rowptr.numel() - 1
    uint64_t seed;         // from torch_npu default generator philox inputs
    uint64_t offset;
};

// Uniform random walk with the kernel-internal Philox RNG.
void LaunchTorchSparseRandomWalkKernel(
    const int64_t *rowptr,
    const int64_t *col,
    const int64_t *start,
    int64_t *out,
    const TorchSparseRandomWalkLaunchParams &params,
    uint32_t availableCores,
    aclrtStream stream);

// Task-semantic verification consumes caller-supplied [S, L] float32 rand.
// Float32 multiplication and truncation are clamped within the current row.
// Legal degree-zero nodes terminate with a -1 suffix. Public CPU/NPU RNG
// streams are not aligned; upstream dead-end behavior is not reproduced.
void LaunchTorchSparseRandomWalkExactKernel(
    const int64_t *rowptr,
    const int64_t *col,
    const int64_t *start,
    const float *rand,
    int64_t *out,
    const TorchSparseRandomWalkLaunchParams &params,
    uint32_t availableCores,
    aclrtStream stream);

} // namespace opsgnn

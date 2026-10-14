/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Ascend C (SIMT) kernel entry for torch_sparse::non_diag_mask on Ascend 950 (arch35).
 */

#pragma once

#include <acl/acl.h>
#include <cstdint>

namespace opsgnn {

struct NonDiagMaskLaunchParams {
    const int64_t *row;
    const int64_t *col;
    uint8_t *mask;
    uint64_t edgeCount;   // row.numel()
    uint64_t numDiag;     // clamped diagonal length appended after the edges
    int64_t diagOffset;   // k
    int64_t rowCount;     // M
    int64_t columnCount;  // N
    uint32_t availableCores;
    aclrtStream stream;
};

// Validates COO bounds and computes (col[i] - row[i] != k) without signed overflow.
// Invalid contents raise an asynchronous device error at stream synchronization.
// Writes every
// position of the E + num_diag output, so the caller can allocate the output
// with torch::empty (no pre-zeroing kernel needed).
void LaunchNonDiagMaskKernel(const NonDiagMaskLaunchParams &params);

} // namespace opsgnn

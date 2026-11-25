/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <acl/acl.h>
#include <cstdint>

#include "csr_row_wise_sampling_uniform_fused/op_kernel/arch22/csr_row_wise_sampling_uniform_fused_tiling.h"

namespace opsgnn {
namespace csr_uniform_fused {

void LaunchUniformSampling(uint32_t blocks, aclrtStream stream,
    void* indptr, void* indices, void* data, void* rows,
    void* outPtr, void* outRows, void* outCols, void* outData,
    void* rowSplit, void* outStarts, void* pickScratch,
    const CsrRowWiseSamplingUniformFusedTiling& tiling);

}  // namespace csr_uniform_fused
}  // namespace opsgnn

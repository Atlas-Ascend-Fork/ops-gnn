/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "csr_row_wise_per_etype_sampling_uniform_kernel.h"
#include "csr_row_wise_per_etype_sampling_uniform.h"

namespace opsgnn {
namespace csr_per_etype {

void LaunchUniformSampling(uint32_t blocks, aclrtStream stream,
    void* indptr, void* indices, void* data, void* rows,
    void* outPtr, void* outRows, void* outCols, void* outData,
    void* rowSplit, void* outStarts, void* pickScratch,
    const CsrRowWisePerEtypeSamplingUniformTiling& tiling)
{
    csr_row_wise_per_etype_sampling_uniform_kernel<<<blocks, nullptr, stream>>>(
        static_cast<uint8_t*>(indptr), static_cast<uint8_t*>(indices),
        static_cast<uint8_t*>(data), static_cast<uint8_t*>(rows),
        static_cast<uint8_t*>(outPtr), static_cast<uint8_t*>(outRows),
        static_cast<uint8_t*>(outCols), static_cast<uint8_t*>(outData),
        static_cast<uint8_t*>(rowSplit), static_cast<uint8_t*>(outStarts),
        static_cast<uint8_t*>(pickScratch), tiling);
}

}  // namespace csr_per_etype
}  // namespace opsgnn

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sddmm_csr_kernel.h"
#include "sddmm_csr.h"

extern "C" __global__ __aicore__ void sddmm_csr(GM_ADDR lhs, GM_ADDR rhs,
                                            GM_ADDR indptr, GM_ADDR indices,
                                            GM_ADDR out, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    const __gm__ SddmmCsrTilingData* tilingData =
        (const __gm__ SddmmCsrTilingData*)tiling;
    if (tilingData->dtype == SDDMM_CSR_DTYPE_FP32) {
        SddmmCsrKernel<float> processor(&pipe);
        processor.Init(lhs, rhs, indptr, indices, out, tilingData);
        processor.Process();
    } else {
        SddmmCsrKernel<half> processor(&pipe);
        processor.Init(lhs, rhs, indptr, indices, out, tilingData);
        processor.Process();
    }
}

namespace opsgnn {

void SddmmCsr(uint32_t blocks, aclrtStream stream, void* lhs, void* rhs,
              void* indptr, void* indices, void* out, void* tiling)
{
    sddmm_csr<<<blocks, nullptr, stream>>>(
        static_cast<uint8_t*>(lhs), static_cast<uint8_t*>(rhs),
        static_cast<uint8_t*>(indptr), static_cast<uint8_t*>(indices),
        static_cast<uint8_t*>(out), static_cast<uint8_t*>(tiling));
}

}  // namespace opsgnn

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_csr/op_kernel/arch35/gather_csr_kernel.h"
#include "gather_csr/op_kernel/arch35/gather_csr.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220
#error "arch22/gather_csr.cpp must be compiled for dav-2201 (A2/A3)"
#endif

namespace opsgnn {

void GatherCsr(void* src, int64_t* indptr, void* out,
                           const GatherCsrTilingData& tiling, aclrtStream stream)
{
    gather_csr_kernel<<<tiling.activeCoreNum, nullptr, stream>>>(
        static_cast<uint8_t*>(src), indptr, static_cast<uint8_t*>(out), tiling);
}

}  // namespace opsgnn

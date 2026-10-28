/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matrix_exp_kernel.h"
#include "matrix_exp.h"

namespace opsgnn {

void MatrixExp(const uint8_t* a, uint8_t* out, const uint8_t* ws, const uint8_t* eye,
               const MatrixExpTilingData& tiling, uint32_t dtypeTag, aclrtStream stream)
{
    // MIX_AIC_1_2 下 blockNum 即 AIC pair 数，与 tiling.coreNum（GetCoreNumAic）一致。
    uint32_t coreNum = tiling.coreNum == 0 ? 1 : tiling.coreNum;
    // GM_ADDR 形参不接受 const 指针，调用侧统一去 const（只读语义由布局契约保证）。
    uint8_t* aGm = const_cast<uint8_t*>(a);
    uint8_t* wsGm = const_cast<uint8_t*>(ws);
    uint8_t* eyeGm = const_cast<uint8_t*>(eye);
    if (dtypeTag == 0) {
        matrix_exp_kernel<float><<<coreNum, nullptr, stream>>>(aGm, out, wsGm, eyeGm, tiling);
    } else if (dtypeTag == 1) {
        matrix_exp_kernel<half><<<coreNum, nullptr, stream>>>(aGm, out, wsGm, eyeGm, tiling);
    } else {
        matrix_exp_kernel<bfloat16_t><<<coreNum, nullptr, stream>>>(aGm, out, wsGm, eyeGm, tiling);
    }
}

}  // namespace opsgnn

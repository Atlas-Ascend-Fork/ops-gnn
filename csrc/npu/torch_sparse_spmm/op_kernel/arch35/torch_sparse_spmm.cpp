/**
 * Copyright (c) 2026 Eternal_ZCL.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Adapted from cann/ops-sparse sparse/spmm/arch35/spmm_kernel.cpp,
// commit 9a514470c2cc99e469c34a615048eb8b30a66694 (Huawei Technologies Co., Ltd.).
#include "torch_sparse_spmm.h"
#include "torch_sparse_spmm_kernel.h"

void TorchSparseSpmmPermuteValues(void* perm, void* value, void* out, int64_t elements, int64_t width, int type,
                                  uint32_t blocks, void* stream)
{
#define PERMUTE(T)                                                                                                     \
    PermuteValuesKernel<T><<<blocks, nullptr, stream>>>(reinterpret_cast<int64_t*>(perm), reinterpret_cast<T*>(value), \
                                                        reinterpret_cast<T*>(out), elements, width)
    switch (type) {
        case 0:
            PERMUTE(float);
            break;
        case 1:
            PERMUTE(half);
            break;
        case 2:
            PERMUTE(bfloat16_t);
            break;
        case 3:
            PERMUTE(int8_t);
            break;
    }
#undef PERMUTE
}

void TorchSparseSpmm(void* ptr, void* col, void* val, void* mat, void* out, void* arg, SpmmShape s, int input, int output,
                     bool weighted, uint32_t blocks, void* stream)
{
    // dtype ids: float=0, half=1, bfloat16=2, int8=3, int32=4.
    if (input == 0) {
        TypedLaunch<float, float, float>(ptr, col, val, mat, out, arg, s, weighted, blocks, stream);
    } else if (input == 1 && output == 0) {
        TypedLaunch<half, float, float>(ptr, col, val, mat, out, arg, s, weighted, blocks, stream);
    } else if (input == 1) {
        TypedLaunch<half, half, float>(ptr, col, val, mat, out, arg, s, weighted, blocks, stream);
    } else if (input == 2 && output == 0) {
        TypedLaunch<bfloat16_t, float, float>(ptr, col, val, mat, out, arg, s, weighted, blocks, stream);
    } else if (input == 2) {
        TypedLaunch<bfloat16_t, bfloat16_t, float>(ptr, col, val, mat, out, arg, s, weighted, blocks, stream);
    } else if (output == 4) {
        TypedLaunch<int8_t, int32_t, int32_t>(ptr, col, val, mat, out, arg, s, weighted, blocks, stream);
    } else {
        TypedLaunch<int8_t, float, float>(ptr, col, val, mat, out, arg, s, weighted, blocks, stream);
    }
}

void TorchSparseSpmmValueGrad(void* ptr, void* col, void* mat, void* grad, void* out, SpmmShape s, int type,
                              uint32_t blocks, void* stream)
{
#define VG(T)                                                                                                         \
    ValueGradKernel<T><<<blocks, nullptr, stream>>>(reinterpret_cast<int64_t*>(ptr), reinterpret_cast<int64_t*>(col), \
                                                    reinterpret_cast<T*>(mat), reinterpret_cast<float*>(grad),        \
                                                    reinterpret_cast<float*>(out), s)
    if (type == 0) {
        VG(float);
    } else if (type == 1) {
        VG(half);
    } else {
        VG(bfloat16_t);
    }
#undef VG
}

void TorchSparseSpmmValidate(void* ptr, void* col, void* status, int64_t m, int64_t n, int64_t e, uint32_t blocks, void* stream)
{
    ValidateKernel<<<blocks, nullptr, stream>>>(reinterpret_cast<int64_t*>(ptr), reinterpret_cast<int64_t*>(col),
                                                reinterpret_cast<int32_t*>(status), m, n, e);
}

void TorchSparseSpmmConvert(void* src, void* dst, int64_t size, int64_t output, bool toPtr, uint32_t blocks, void* stream)
{
    ConvertKernel<<<blocks, nullptr, stream>>>(reinterpret_cast<int64_t*>(src), reinterpret_cast<int64_t*>(dst), size,
                                               output, toPtr);
}

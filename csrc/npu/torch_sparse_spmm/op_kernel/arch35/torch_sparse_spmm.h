/**
 * Copyright (c) 2026 Eternal_ZCL.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "torch_sparse_spmm_tiling.h"

void TorchSparseSpmm(void* ptr, void* col, void* value, void* mat, void* out, void* arg, SpmmShape shape, int inputType,
                     int outputType, bool weighted, uint32_t blocks, void* stream);
void TorchSparseSpmmValueGrad(void* ptr, void* col, void* mat, void* grad, void* out, SpmmShape shape, int type,
                              uint32_t blocks, void* stream);
void TorchSparseSpmmValidate(void* ptr, void* col, void* status, int64_t m, int64_t n, int64_t edges, uint32_t blocks,
                             void* stream);
void TorchSparseSpmmConvert(void* src, void* out, int64_t length, int64_t output, bool toPtr, uint32_t blocks, void* stream);
void TorchSparseSpmmPermuteValues(void* perm, void* value, void* out, int64_t elements, int64_t width, int type,
                                  uint32_t blocks, void* stream);

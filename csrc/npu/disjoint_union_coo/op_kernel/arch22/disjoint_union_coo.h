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

namespace opsgnn {

// One launch concatenates a single COO matrix into the disjoint-union result.
// Every source/dest id is shifted by the accumulated row/column count of the
// preceding matrices, and the data array (or the implicit edge index) by the
// accumulated nnz. All offsets are constant for a launch, so they travel as
// scalar kernel arguments instead of a device tiling buffer.
//
// `inData` may point at any readable buffer when `hasDataLocal` is 0, and
// `outData` at any writable buffer when `hasDataGlobal` is 0; both are only
// dereferenced when the corresponding flag is set.
void DisjointUnionCooInt32(uint32_t blocks, aclrtStream stream, const int32_t* inRow,
                           const int32_t* inCol, const int32_t* inData, int32_t* outRow,
                           int32_t* outCol, int32_t* outData, int64_t offSrc,
                           int64_t offDst, int64_t offElm, uint32_t n, uint32_t totalNnz,
                           uint32_t hasDataLocal, uint32_t hasDataGlobal, uint32_t ubBytes);

void DisjointUnionCooInt64(uint32_t blocks, aclrtStream stream, const int64_t* inRow,
                           const int64_t* inCol, const int64_t* inData, int64_t* outRow,
                           int64_t* outCol, int64_t* outData, int64_t offSrc,
                           int64_t offDst, int64_t offElm, uint32_t n, uint32_t totalNnz,
                           uint32_t hasDataLocal, uint32_t hasDataGlobal, uint32_t ubBytes);

}  // namespace opsgnn

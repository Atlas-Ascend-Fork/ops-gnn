/*
 * Copyright (c) 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <acl/acl.h>
#include "segment_csr_tiling.h"
aclError SegmentCsr(void *src, int64_t *index, void *out, int64_t *arg, const SegmentCsrPlan &plan, aclrtStream stream);

// Asynchronous exact NPU range proof. perCoreValid needs blocks*8 int32 values;
// the result for core i is perCoreValid[i*8], with a private 32-byte store block.
void SegmentCsrValidateInt64(void *src, int64_t numel, int64_t absBound, int32_t *perCoreValid, int blocks,
                             aclrtStream stream);

// Exact whole-source FP16 predicate: every element is finite and abs(x) <= 6.
// The same private 32-byte per-core result layout as the INT64 validator.
void SegmentCsrValidateHalf(void *src, int64_t numel, int32_t *perCoreValid, int blocks, aclrtStream stream);

// Each core writes a private 8-int32 block: [valid, fullUniform, 0,...].
// The uniform proof checks every pointer equals row * (length/segments).
void SegmentCsrValidatePointers(int64_t *indptr, int64_t pointerBatches, int64_t segments, int64_t length,
                                int32_t *perCoreFlags, int blocks, aclrtStream stream);

// Internal fallback shared by uniform launch selection and general kernels.
aclError SegmentCsrGeneral(void *src, int64_t *index, void *out, int64_t *arg, const SegmentCsrPlan &plan,
                           aclrtStream stream);

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
#include "coalesce_tiling.h"

namespace opsgnn {

// Reduce each contiguous sorted segment of a gathered edge_attr matrix into one
// output row. `indptr` holds U + 1 ascending offsets into the (E, C) source.
// Returns the launch status; the operation is asynchronous on `stream`.
aclError CoalesceSegmentReduce(void *src, int64_t *indptr, void *out, const CoalesceTilingData &t,
                               aclrtStream stream);

} // namespace opsgnn

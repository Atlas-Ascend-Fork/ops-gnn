/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// arch22 (A2/A3) compilation unit of the CSR log-sum-exp. The reduction and its
// host launcher live in the shared header because they only use primitives with
// identical semantics on dav-2201 and dav-3510: DataCopyPad, the fp32 vector
// math library (Max/Mins/Maxs/Sub/Exp/Add/Ln) and the per-core task walk.
#include "segment_logsumexp/op_kernel/common/segment_logsumexp_kernel.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220
#error "arch22/segment_logsumexp.cpp must be compiled for dav-2201 (A2/A3)"
#endif

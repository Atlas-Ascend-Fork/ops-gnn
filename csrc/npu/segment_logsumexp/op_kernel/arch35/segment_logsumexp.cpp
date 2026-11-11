/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// arch35 (Ascend 950) compilation unit of the CSR log-sum-exp. It includes the
// same shared header as arch22 on purpose: the reduction only uses DataCopyPad,
// the fp32 vector math library and a per-core task walk, which dav-3510
// provides with the same semantics, so a second device implementation would
// only diverge over time.
#include "segment_logsumexp/op_kernel/common/segment_logsumexp_kernel.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 310
#error "arch35/segment_logsumexp.cpp must be compiled for dav-3510 (Ascend 950)"
#endif

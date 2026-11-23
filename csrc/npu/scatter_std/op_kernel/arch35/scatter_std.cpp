/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// arch35 (Ascend 950) compilation unit of the grouped standard deviation. It
// includes the same shared header as arch22 on purpose: the kernels only use
// MTE3 atomic max/add, DataCopyPad and fp32 vector math, which dav-3510
// implements with the same semantics, so a second device implementation would
// only diverge over time.
#include "scatter_std/op_kernel/common/scatter_std_kernel.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 310
#error "arch35/scatter_std.cpp must be compiled for dav-3510 (Ascend 950)"
#endif

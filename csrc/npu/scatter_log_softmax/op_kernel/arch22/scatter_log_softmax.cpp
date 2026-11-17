/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// arch22 (A2/A3) compilation unit of the grouped log-softmax. The five kernels
// and their host launcher live in the shared header because they only use
// primitives whose semantics are identical on dav-2201 and dav-3510: MTE3
// atomic max/add, DataCopyPad and the fp32 vector math library.
#include "scatter_log_softmax/op_kernel/common/scatter_log_softmax_kernel.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220
#error "arch22/scatter_log_softmax.cpp must be compiled for dav-2201 (A2/A3)"
#endif

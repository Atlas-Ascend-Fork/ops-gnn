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
#include "radius_tiling.h"

namespace opsgnn {

// DType template parameter selects the device-side widening rule:
//   0 = float32, 1 = fp16 raw uint16, 2 = bf16 raw uint16.
// tiling_gm points to a GM-resident RadiusTilingData (H2D-copied by the
// host); structs are never passed by value through the SIMT launch path.
template <typename T>
struct RadiusLaunchArgs {
    int64_t* ptr_x;
    int64_t* ptr_y;
    int64_t* unique_cell_ids;
    int64_t* cell_offsets;
    T* reordered_x;
    int64_t* reordered_orig_idx;
    int64_t* counts;
    int64_t* out_row1;
    uint8_t* config_ptr;
    uint8_t* tiling_gm;
};

template <typename T, int DType = 0>
void Radius(T* x, T* y, const RadiusLaunchArgs<T>& args,
                        aclrtStream stream);

}  // namespace opsgnn

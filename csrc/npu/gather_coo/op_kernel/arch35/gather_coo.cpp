/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0.
 * Please refer to the License for details.
 */

#include "gather_coo_kernel.h"
#include "gather_coo.h"

namespace opsgnn {

template <typename T>
void GatherCoo(
    T* src,
    int64_t* index,
    T* out,
    const GatherCooTilingData& tiling,
    aclrtStream stream)
{
    if (tiling.outputSlabRows != 0 &&
        !(tiling.logicalFeatureCount == 128 && tiling.elementBytes <= 4)) {
        gather_coo_slab_kernel<T><<<tiling.coreNum, nullptr, stream>>>(
            reinterpret_cast<uint8_t*>(src), reinterpret_cast<uint8_t*>(index),
            reinterpret_cast<uint8_t*>(out), tiling);
        return;
    }
    gather_coo_kernel<T><<<tiling.coreNum, nullptr, stream>>>(
        reinterpret_cast<uint8_t*>(src),
        reinterpret_cast<uint8_t*>(index),
        reinterpret_cast<uint8_t*>(out),
        tiling);
}

template void GatherCoo<uint8_t>(
    uint8_t*, int64_t*, uint8_t*, const GatherCooTilingData&, aclrtStream);
template void GatherCoo<uint16_t>(
    uint16_t*, int64_t*, uint16_t*, const GatherCooTilingData&, aclrtStream);
template void GatherCoo<uint32_t>(
    uint32_t*, int64_t*, uint32_t*, const GatherCooTilingData&, aclrtStream);

}  // namespace opsgnn

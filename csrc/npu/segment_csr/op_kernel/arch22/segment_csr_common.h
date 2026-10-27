/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "segment_csr.h"
#include "kernel_operator.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace opsgnn::csr {
using namespace AscendC;

__aicore__ inline float CountFloat(uint64_t value) {
    // Validated pointer differences fit INT64_MAX. Scalar conversion accepts
    // signed int32, so split the full count without an intermediate int32 count.
    return static_cast<float>(static_cast<int32_t>(value >> 32)) * 4294967296.0f +
           static_cast<float>(static_cast<int32_t>(value & 0x7fffffffULL)) +
           ((value & 0x80000000ULL) ? 2147483648.0f : 0.0f);
}

template <HardEvent Event> __aicore__ inline void Sync() {
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}

struct CsrRowCopy {
    uint64_t offset, features;
    uint32_t rows, valid, width;
};

__aicore__ inline void CopyInt64Rows(LocalTensor<int64_t> destination, GlobalTensor<int64_t> &source,
                                     const CsrRowCopy &copy) {
    if (copy.width == copy.features && copy.valid == copy.width) {
        DataCopy(destination, source[copy.offset], copy.rows * copy.width);
        Sync<HardEvent::MTE2_V>();
        return;
    }
    const uint32_t transfer = (copy.valid + 3) / 4 * 4;
    const uint64_t gap = (copy.features - copy.valid) * sizeof(int64_t);
    const auto pad = DataCopyPadExtParams<int64_t>(true, 0, static_cast<uint8_t>(transfer - copy.valid), 0);
    if (gap <= UINT32_MAX) {
        DataCopyPad(destination, source[copy.offset],
                    {static_cast<uint16_t>(copy.rows), copy.valid * 8, static_cast<uint32_t>(gap),
                     (copy.width - transfer) / 4, 0},
                    pad);
    } else {
        for (uint32_t row = 0; row < copy.rows; ++row)
            DataCopyPad(destination[row * copy.width], source[copy.offset + row * copy.features],
                        {1, copy.valid * 8, 0, 0, 0}, pad);
    }
    Sync<HardEvent::MTE2_V>();
}
} // namespace opsgnn::csr

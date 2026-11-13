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

struct SegmentMaxCsrTilingData {
    uint32_t srcLength;
    uint32_t E_1;
    uint32_t nSegments;
    uint32_t M;
    uint32_t K;
    uint32_t indptrPhysicalNum;
    uint32_t coreDataNum;
    uint32_t ALIGN_NUM;
    uint32_t aivNum;
    uint32_t rowsPerAiv;
    uint32_t rowsLastAiv;
    uint32_t KloopTime;
    uint32_t hasOptionalOut;
    uint32_t strideIndptr;
};
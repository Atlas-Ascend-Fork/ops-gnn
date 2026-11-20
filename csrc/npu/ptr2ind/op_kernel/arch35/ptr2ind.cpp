/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "kernel_operator.h"
#include "common/aiv_launch_utils.h"
#include "ptr2ind/op_kernel/arch35/ptr2ind.h"

// SIMT threads launched per AIV via VF_CALL Dim3.
constexpr uint32_t PTR2IND_SIMT_THREADS = 256;

// numRows == ptr length - 1 (CSR row count).
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void Ptr2IndKernelSimt(__gm__ int64_t* ptr, __gm__ int64_t* out,
                                                                int64_t numRows, int64_t numEdges)
{
    const int64_t threadNum = AscendC::Simt::GetThreadNum();
    const int64_t blockNum = AscendC::Simt::GetBlockNum();
    const int64_t globalThreadIdx =
        AscendC::Simt::GetBlockIdx() * threadNum + AscendC::Simt::GetThreadIdx();
    const int64_t totalThreadNum = blockNum * threadNum;

    // One logical work item per CSR row; stride when rows exceed total SIMT threads.
    // Defense in depth: clamp the write range to [0, numEdges) so a pointer that
    // slipped past host validation can never address GM outside the output.
    for (int64_t row = globalThreadIdx; row < numRows; row += totalThreadNum) {
        int64_t idx = ptr[row];
        int64_t nextIdx = ptr[row + 1];
        if (idx < 0) {
            idx = 0;
        }
        if (nextIdx > numEdges) {
            nextIdx = numEdges;
        }
        for (int64_t i = idx; i < nextIdx; i++) {
            out[i] = row;
        }
    }
}

__attribute__((aiv)) __global__ __aicore__ void Ptr2IndKernel(__gm__ int64_t* ptr, __gm__ int64_t* out,
                                                              int64_t numRows, int64_t numEdges)
{
    AscendC::Simt::VF_CALL<Ptr2IndKernelSimt>(AscendC::Simt::Dim3{PTR2IND_SIMT_THREADS}, ptr, out,
                                              numRows, numEdges);
}

namespace opsgnn {

void Ptr2Ind(const int64_t* ptr, int64_t* out, int64_t numRows, int64_t numEdges, aclrtStream stream)
{
    // AscendC SIMT: each AIV runs PTR2IND_SIMT_THREADS threads; cover row in [0, numRows).
    const uint32_t needCoreNum =
        static_cast<uint32_t>((static_cast<uint64_t>(numRows) + PTR2IND_SIMT_THREADS - 1) /
                              PTR2IND_SIMT_THREADS);
    const uint32_t coreNum = ops_gnn::sparse::ResolveAivCoreNum(needCoreNum);

    Ptr2IndKernel<<<coreNum, nullptr, stream>>>(const_cast<int64_t*>(ptr), out, numRows, numEdges);
}

}  // namespace opsgnn

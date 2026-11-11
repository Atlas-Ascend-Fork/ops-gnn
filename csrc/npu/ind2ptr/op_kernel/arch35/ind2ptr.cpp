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
#include "ind2ptr/op_kernel/arch35/ind2ptr.h"

namespace {

bool UseInd2PtrSearch(int64_t numel, int64_t numRows)
{
    if (numel < 1048576) {
        return false;
    }
    uint64_t searchDepth = 0;
    for (uint64_t remaining = static_cast<uint64_t>(numel); remaining > 0; remaining >>= 1) {
        ++searchDepth;
    }
    const uint64_t outputItems = static_cast<uint64_t>(numRows) + 1;
    return static_cast<uint64_t>(numel) / outputItems >= 8 * searchDepth;
}

}

// SIMT threads launched per AIV via VF_CALL Dim3.
constexpr uint32_t IND2PTR_SIMT_THREADS = 256;

__simt_vf__ __aicore__ LAUNCH_BOUND(256) void Ind2PtrSearchSimt(
    __gm__ const int64_t* ind, __gm__ int64_t* out, int64_t M, int64_t numel)
{
    const int64_t tid = AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
        AscendC::Simt::GetThreadIdx();
    const int64_t stride = AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum();
    for (int64_t row = tid; row <= M; row += stride) {
        int64_t low = 0;
        int64_t high = numel;
        while (low < high) {
            const int64_t mid = low + (high - low) / 2;
            if (ind[mid] < row) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        out[row] = low;
    }
}

__attribute__((aiv)) __global__ __aicore__ void Ind2PtrSearchKernel(
    __gm__ const int64_t* ind, __gm__ int64_t* out, int64_t M, int64_t numel)
{
    AscendC::Simt::VF_CALL<Ind2PtrSearchSimt>(
        AscendC::Simt::Dim3{IND2PTR_SIMT_THREADS}, ind, out, M, numel);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) void Ind2PtrBoundarySimt(
    __gm__ int64_t* ind, __gm__ int64_t* out, int64_t M, int64_t numel)
{
    const int64_t threadNum = AscendC::Simt::GetThreadNum();
    const int64_t blockNum = AscendC::Simt::GetBlockNum();
    const int64_t globalThreadIdx =
        AscendC::Simt::GetBlockIdx() * threadNum + AscendC::Simt::GetThreadIdx();
    const int64_t totalThreadNum = blockNum * threadNum;

    for (int64_t row = globalThreadIdx; row <= ind[0]; row += totalThreadNum) {
        out[row] = 0;
    }
    for (int64_t row = ind[numel - 1] + 1 + globalThreadIdx; row <= M; row += totalThreadNum) {
        out[row] = numel;
    }
    constexpr uint32_t chunkSize = 4;
    const uint32_t stride = static_cast<uint32_t>(totalThreadNum) * chunkSize;
    const uint32_t count = static_cast<uint32_t>(numel);
    for (uint32_t begin = static_cast<uint32_t>(globalThreadIdx) * chunkSize + 1;
         begin < count; begin += stride) {
        const uint32_t end = begin + chunkSize < count ? begin + chunkSize : count;
        const int64_t firstRow = ind[begin - 1];
        const int64_t lastRow = ind[end - 1];
        for (int64_t row = firstRow + 1; row <= lastRow; ++row) {
            uint32_t low = begin;
            uint32_t high = end - 1;
            while (low < high) {
                const uint32_t mid = low + (high - low) / 2;
                if (ind[mid] < row) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            out[row] = low;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) void Ind2PtrKernelSimt(__gm__ int64_t* ind, __gm__ int64_t* out,
                                                                int64_t M, int64_t numel)
{
    const int64_t threadNum = AscendC::Simt::GetThreadNum();
    const int64_t blockNum = AscendC::Simt::GetBlockNum();
    const int64_t globalThreadIdx =
        AscendC::Simt::GetBlockIdx() * threadNum + AscendC::Simt::GetThreadIdx();
    const int64_t totalThreadNum = blockNum * threadNum;

    // Logical work indices are [0, numel]. Stride over all AIV SIMT threads when
    // workItems > coreNum * IND2PTR_SIMT_THREADS.
    for (int64_t workIdx = globalThreadIdx; workIdx <= numel; workIdx += totalThreadNum) {
        if (workIdx == 0) {
            for (int64_t i = 0; i <= ind[0]; i++) {
                out[i] = 0;
            }
        } else if (workIdx < numel) {
            for (int64_t i = ind[workIdx - 1]; i < ind[workIdx]; i++) {
                out[i + 1] = workIdx;
            }
        } else {
            for (int64_t i = ind[numel - 1] + 1; i < M + 1; i++) {
                out[i] = numel;
            }
        }
    }
}

template <bool ParallelBoundary>
__attribute__((aiv)) __global__ __aicore__ void Ind2PtrKernel(__gm__ int64_t* ind, __gm__ int64_t* out,
                                                              int64_t M, int64_t numel)
{
    if (ParallelBoundary) {
        AscendC::Simt::VF_CALL<Ind2PtrBoundarySimt>(
            AscendC::Simt::Dim3{IND2PTR_SIMT_THREADS}, ind, out, M, numel);
    } else {
        AscendC::Simt::VF_CALL<Ind2PtrKernelSimt>(
            AscendC::Simt::Dim3{IND2PTR_SIMT_THREADS}, ind, out, M, numel);
    }
}

namespace opsgnn {

void Ind2Ptr(const int64_t* ind, int64_t* out, int64_t M, int64_t numel, aclrtStream stream)
{
    const uint64_t outputItems = static_cast<uint64_t>(M) + 1;
    if (UseInd2PtrSearch(numel, M)) {
        const uint64_t requested = (outputItems + IND2PTR_SIMT_THREADS - 1) / IND2PTR_SIMT_THREADS;
        const uint32_t available = ops_gnn::sparse::ResolveAivCoreNum(0xffffffffU);
        const uint32_t cores = static_cast<uint32_t>(requested < available ? requested : available);
        Ind2PtrSearchKernel<<<cores, nullptr, stream>>>(ind, out, M, numel);
        return;
    }
    // AscendC SIMT: each AIV runs IND2PTR_SIMT_THREADS threads; cover workIdx in [0, numel].
    const uint64_t workItems = static_cast<uint64_t>(numel) + 1;
    const uint32_t needCoreNum =
        static_cast<uint32_t>((workItems + IND2PTR_SIMT_THREADS - 1) / IND2PTR_SIMT_THREADS);
    const uint32_t coreNum = ops_gnn::sparse::ResolveAivCoreNum(needCoreNum);

    if (numel >= 32768 && numel <= 2147483647 && static_cast<uint64_t>(numel) / outputItems >= 64) {
        Ind2PtrKernel<true><<<coreNum, nullptr, stream>>>(const_cast<int64_t*>(ind), out, M, numel);
    } else {
        Ind2PtrKernel<false><<<coreNum, nullptr, stream>>>(const_cast<int64_t*>(ind), out, M, numel);
    }
}

}  // namespace opsgnn

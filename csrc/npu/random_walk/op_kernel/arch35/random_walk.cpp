/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "random_walk.h"

#include "random_walk/common/philox.h"
#include "kernel_operator.h"
#include "simt_api/device_functions.h"
#include "tiling/platform/platform_ascendc.h"

using opsgnn::philox::MakePhiloxState;
using opsgnn::philox::PhiloxNext;
using opsgnn::philox::PhiloxState;

namespace {

// Uniform walks are a short-register-pressure path and benefit from more AIV
// blocks. The biased path needs the proven 1024-thread arch35 configuration.
constexpr uint32_t UNIFORM_THREADS_PER_BLOCK = 256;
constexpr uint32_t NODE2VEC_THREADS_PER_BLOCK = 1024;
constexpr uint32_t SMALL_DEGREE_LINEAR_LIMIT = 16;
constexpr uint32_t UINT32_MAX_VALUE = 0xFFFFFFFFU;

__simt_callee__ inline uint32_t SampleIndex(PhiloxState& state, uint32_t degree)
{
    return __umulhi(PhiloxNext(state), degree);
}

__simt_callee__ inline bool IsNeighbor(
    __gm__ const int64_t* rowptr, __gm__ const int64_t* col, int64_t node, int64_t target,
    bool neighborsSorted)
{
    int64_t begin = rowptr[node];
    int64_t end = rowptr[node + 1];
    if (!neighborsSorted || end - begin <= SMALL_DEGREE_LINEAR_LIMIT) {
        for (int64_t edge = begin; edge < end; ++edge) {
            if (col[edge] == target) {
                return true;
            }
        }
        return false;
    }

    int64_t originalEnd = end;
    while (begin < end) {
        int64_t middle = begin + ((end - begin) >> 1);
        int64_t value = col[middle];
        if (value < target) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin < originalEnd && col[begin] == target;
}

struct Node2VecStepResult {
    int64_t edge;
    int64_t candidate;
};

__simt_callee__ inline bool AcceptNode2VecCandidate(
    __gm__ const int64_t* rowptr, __gm__ const int64_t* col, int64_t candidate,
    int64_t previous, uint32_t draw, uint32_t returnThreshold, uint32_t neighborThreshold,
    uint32_t distantThreshold, bool neighborsSorted)
{
    if (returnThreshold == UINT32_MAX_VALUE && neighborThreshold == UINT32_MAX_VALUE &&
        distantThreshold == UINT32_MAX_VALUE) {
        return true;
    }
    if (candidate == previous) {
        return draw <= returnThreshold;
    }
    if (IsNeighbor(rowptr, col, candidate, previous, neighborsSorted)) {
        return draw <= neighborThreshold;
    }
    return draw <= distantThreshold;
}

__simt_callee__ inline Node2VecStepResult SampleNode2VecStep(
    __gm__ const int64_t* rowptr, __gm__ const int64_t* col, int64_t current, int64_t previous,
    uint32_t step, PhiloxState& random, uint32_t returnThreshold, uint32_t neighborThreshold,
    uint32_t distantThreshold, bool neighborsSorted)
{
    const int64_t begin = rowptr[current];
    const int64_t end = rowptr[current + 1];
    const int64_t degree64 = end - begin;
    Node2VecStepResult result{ -1, current };
    if (degree64 <= 0) {
        return result;
    }
    if (step == 0 || degree64 == 1) {
        const uint32_t index = degree64 == 1 ? 0 : SampleIndex(random, static_cast<uint32_t>(degree64));
        result.edge = begin + static_cast<int64_t>(index);
        result.candidate = col[result.edge];
        return result;
    }
    const uint32_t degree = static_cast<uint32_t>(degree64);
    while (true) {
        result.edge = begin + static_cast<int64_t>(SampleIndex(random, degree));
        result.candidate = col[result.edge];
        const uint32_t draw = PhiloxNext(random);
        if (AcceptNode2VecCandidate(rowptr, col, result.candidate, previous, draw,
                                    returnThreshold, neighborThreshold, distantThreshold,
                                    neighborsSorted)) {
            return result;
        }
    }
}

template <bool WRITE_EDGE>
__simt_vf__ __aicore__ LAUNCH_BOUND(UNIFORM_THREADS_PER_BLOCK) void RandomWalkUniformSimt(
    __gm__ const int64_t* rowptr, __gm__ const int64_t* col, __gm__ const int64_t* start,
    __gm__ int64_t* nodeOut, __gm__ int64_t* edgeOut, uint64_t startCount, uint32_t walkLength,
    uint64_t seed, uint64_t offset)
{
    uint64_t blockIndex = AscendC::Simt::GetBlockIdx();
    uint64_t blockCount = AscendC::Simt::GetBlockNum();
    uint64_t threadIndex = AscendC::Simt::GetThreadIdx();
    uint64_t threadCount = AscendC::Simt::GetThreadNum();
    uint64_t globalThread = blockIndex * threadCount + threadIndex;
    uint64_t globalStride = blockCount * threadCount;

    for (uint64_t walk = globalThread; walk < startCount; walk += globalStride) {
        PhiloxState random = MakePhiloxState(seed, offset, walk);
        int64_t current = start[walk];
        uint64_t nodeBase = walk * (static_cast<uint64_t>(walkLength) + 1);
        uint64_t edgeBase = walk * static_cast<uint64_t>(walkLength);
        nodeOut[nodeBase] = current;

        for (uint32_t step = 0; step < walkLength; ++step) {
            int64_t begin = rowptr[current];
            int64_t end = rowptr[current + 1];
            int64_t edge = -1;
            if (end > begin) {
                uint32_t degree = static_cast<uint32_t>(end - begin);
                edge = begin + static_cast<int64_t>(SampleIndex(random, degree));
                current = col[edge];
            }
            nodeOut[nodeBase + step + 1] = current;
            if constexpr (WRITE_EDGE) {
                edgeOut[edgeBase + step] = edge;
            }
        }
    }
}

template <bool WRITE_EDGE, uint32_t LAUNCH_THREADS>
__simt_vf__ __aicore__ LAUNCH_BOUND(LAUNCH_THREADS) void RandomWalkNode2VecSimt(
    __gm__ const int64_t* rowptr, __gm__ const int64_t* col, __gm__ const int64_t* start,
    __gm__ int64_t* nodeOut, __gm__ int64_t* edgeOut, uint64_t startCount, uint32_t walkLength,
    uint64_t seed, uint64_t offset, uint32_t returnThreshold, uint32_t neighborThreshold,
    uint32_t distantThreshold, bool neighborsSorted)
{
    uint64_t blockIndex = AscendC::Simt::GetBlockIdx();
    uint64_t blockCount = AscendC::Simt::GetBlockNum();
    uint64_t threadIndex = AscendC::Simt::GetThreadIdx();
    uint64_t threadCount = AscendC::Simt::GetThreadNum();
    uint64_t globalThread = blockIndex * threadCount + threadIndex;
    uint64_t globalStride = blockCount * threadCount;

    for (uint64_t walk = globalThread; walk < startCount; walk += globalStride) {
        PhiloxState random = MakePhiloxState(seed, offset, walk);
        int64_t previous = start[walk];
        int64_t current = previous;
        uint64_t nodeBase = walk * (static_cast<uint64_t>(walkLength) + 1);
        uint64_t edgeBase = walk * static_cast<uint64_t>(walkLength);
        nodeOut[nodeBase] = current;

        for (uint32_t step = 0; step < walkLength; ++step) {
            const Node2VecStepResult result = SampleNode2VecStep(
                rowptr, col, current, previous, step, random, returnThreshold,
                neighborThreshold, distantThreshold, neighborsSorted);

            nodeOut[nodeBase + step + 1] = result.candidate;
            if constexpr (WRITE_EDGE) {
                edgeOut[edgeBase + step] = result.edge;
            }
            previous = current;
            current = result.candidate;
        }
    }
}

__attribute__((aiv)) __global__ __aicore__ void RandomWalkUniformNodeKernel(
    __gm__ const int64_t* rowptr, __gm__ const int64_t* col, __gm__ const int64_t* start,
    __gm__ int64_t* nodeOut, uint64_t startCount, uint32_t walkLength, uint64_t seed, uint64_t offset)
{
    AscendC::Simt::VF_CALL<RandomWalkUniformSimt<false>>(
        AscendC::Simt::Dim3{UNIFORM_THREADS_PER_BLOCK}, rowptr, col, start, nodeOut, nullptr,
        startCount, walkLength, seed, offset);
}

__attribute__((aiv)) __global__ __aicore__ void RandomWalkUniformEdgeKernel(
    __gm__ const int64_t* rowptr, __gm__ const int64_t* col, __gm__ const int64_t* start,
    __gm__ int64_t* nodeOut, __gm__ int64_t* edgeOut, uint64_t startCount, uint32_t walkLength,
    uint64_t seed, uint64_t offset)
{
    AscendC::Simt::VF_CALL<RandomWalkUniformSimt<true>>(
        AscendC::Simt::Dim3{UNIFORM_THREADS_PER_BLOCK}, rowptr, col, start, nodeOut, edgeOut,
        startCount, walkLength, seed, offset);
}

__attribute__((aiv)) __global__ __aicore__ void RandomWalkNode2VecNodeKernel(
    __gm__ const int64_t* rowptr, __gm__ const int64_t* col, __gm__ const int64_t* start,
    __gm__ int64_t* nodeOut, uint64_t startCount, uint32_t walkLength, uint64_t seed, uint64_t offset,
    uint32_t returnThreshold, uint32_t neighborThreshold, uint32_t distantThreshold,
    bool neighborsSorted)
{
    AscendC::Simt::VF_CALL<RandomWalkNode2VecSimt<false, NODE2VEC_THREADS_PER_BLOCK>>(
        AscendC::Simt::Dim3{NODE2VEC_THREADS_PER_BLOCK}, rowptr, col, start, nodeOut, nullptr,
        startCount, walkLength, seed, offset, returnThreshold, neighborThreshold, distantThreshold,
        neighborsSorted);
}

__attribute__((aiv)) __global__ __aicore__ void RandomWalkNode2VecEdgeKernel(
    __gm__ const int64_t* rowptr, __gm__ const int64_t* col, __gm__ const int64_t* start,
    __gm__ int64_t* nodeOut, __gm__ int64_t* edgeOut, uint64_t startCount, uint32_t walkLength,
    uint64_t seed, uint64_t offset, uint32_t returnThreshold, uint32_t neighborThreshold,
    uint32_t distantThreshold, bool neighborsSorted)
{
    AscendC::Simt::VF_CALL<RandomWalkNode2VecSimt<true, NODE2VEC_THREADS_PER_BLOCK>>(
        AscendC::Simt::Dim3{NODE2VEC_THREADS_PER_BLOCK}, rowptr, col, start, nodeOut, edgeOut,
        startCount, walkLength, seed, offset, returnThreshold, neighborThreshold, distantThreshold,
        neighborsSorted);
}

}  // namespace

namespace opsgnn {

void RandomWalk(const int64_t* rowptr, const int64_t* col, const int64_t* start,
                            int64_t* nodeOut, int64_t* edgeOut, const RandomWalkLaunchParams& params,
                            aclrtStream stream)
{
    if (params.startCount == 0 || params.walkLength == 0) {
        return;
    }
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t availableCores = platform->GetCoreNumAiv();
    uint32_t threadsPerBlock = params.node2vec ? NODE2VEC_THREADS_PER_BLOCK : UNIFORM_THREADS_PER_BLOCK;
    uint64_t requiredCores = (params.startCount + threadsPerBlock - 1) / threadsPerBlock;
    uint32_t blockDim = static_cast<uint32_t>(requiredCores < availableCores ? requiredCores : availableCores);
    blockDim = blockDim == 0 ? 1 : blockDim;

    if (!params.node2vec) {
        if (params.writeEdge) {
            RandomWalkUniformEdgeKernel<<<blockDim, nullptr, stream>>>(
                rowptr, col, start, nodeOut, edgeOut, params.startCount, params.walkLength,
                params.seed, params.offset);
        } else {
            RandomWalkUniformNodeKernel<<<blockDim, nullptr, stream>>>(
                rowptr, col, start, nodeOut, params.startCount, params.walkLength,
                params.seed, params.offset);
        }
    } else if (params.writeEdge) {
        RandomWalkNode2VecEdgeKernel<<<blockDim, nullptr, stream>>>(
            rowptr, col, start, nodeOut, edgeOut, params.startCount, params.walkLength,
            params.seed, params.offset, params.returnThreshold, params.neighborThreshold,
            params.distantThreshold, params.neighborsSorted);
    } else {
        RandomWalkNode2VecNodeKernel<<<blockDim, nullptr, stream>>>(
            rowptr, col, start, nodeOut, params.startCount, params.walkLength, params.seed,
            params.offset, params.returnThreshold, params.neighborThreshold,
            params.distantThreshold, params.neighborsSorted);
    }
}

}  // namespace opsgnn

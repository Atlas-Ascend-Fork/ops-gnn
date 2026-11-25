/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Ascend C (SIMT) kernel for torch_sparse::random_walk (uniform) on Ascend 950 (arch35).
 *
 * Walk semantics follow the task-book adjacency and unwritten-value contract:
 *   cur = start[n]; out[n,0] = cur;
 *   for l in [0, L):
 *     begin = rowptr[cur]; end = rowptr[cur+1]
 *     if begin == end: leave the remaining output slots at -1
 *     cur = col[begin + int64(rand[n,l] * (end - begin))]
 *     out[n,l+1] = cur
 *
 * RNG record (task-book section 7 requires documenting algorithm / seed /
 * consumption order):
 *   - Algorithm: Philox4x32-10 (counter based), 4 uint32 per refill.
 *   - Seed: key = philox_engine_inputs() of the torch_npu default generator,
 *     i.e. controlled by torch.manual_seed; offset stays in counter0/1 and is
 *     never folded into the key (seed and offset must not collide).
 *   - Consumption order: one PhiloxState per walk n (counter2/3 = n), values
 *     consumed sequentially across steps l; selection uses
 *     idx = floor(u * degree / 2^32), using a split product (u is one uint32 draw).
 *   - Deterministic paths (degree == 1, walk_length 0) never observe the RNG,
 *     so CPU / GPU / NPU agree element-wise on the official test graphs.
 *
 * Input contracts are checked on the device on every call, including empty
 * starts and zero-length walks. Invalid contents interrupt the stream through
 * the public Ascend C SIMT __trap API; synchronization reports the device error.
 * Legal degree-zero nodes terminate the walk and leave a -1 suffix.
 */

#include "torch_sparse_random_walk.h"

#include "random_walk/common/philox.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"
#include "utils/debug/asc_assert.h"

using opsgnn::philox::MakePhiloxState;
using opsgnn::philox::PhiloxNext;
using opsgnn::philox::PhiloxState;

namespace {
constexpr uint32_t WALK_THREADS_PER_BLOCK = 256;

// One (rowptr, col) transition. `rand01` selects the neighbor:
//   Philox mode : idx = floor(u32 * degree / 2^32)
//   Inject mode : float32 multiply/truncate, bounded within the current row
// Every dependent gather has its own bounds check, independent of the parallel
// whole-input validation below. No invalid CSR/start/col can reach an OOB load.
__simt_callee__ inline int64_t WalkStep(
    __gm__ const int64_t *rowptr,
    __gm__ const int64_t *col,
    int64_t current,
    uint64_t nnz,
    uint64_t nodeCount,
    bool injected,
    float rand01,
    uint32_t randBits)
{
    if (current < 0 || static_cast<uint64_t>(current) >= nodeCount) {
        ascendc_assert(false, "random_walk: current node out of range");
        __trap();
        return -1;
    }
    const int64_t begin = rowptr[current];
    const int64_t end = rowptr[current + 1];
    if (begin < 0 || end < begin || static_cast<uint64_t>(end) > nnz) {
        ascendc_assert(false, "random_walk: invalid CSR row interval");
        __trap();
        return -1;
    }
    const int64_t degree = end - begin;
    if (degree == 0) {
        return -1;
    }

    int64_t idx = 0;
    if (injected) {
        // Task semantics: float32 multiply and truncation, with an in-row clamp.
        if (!(rand01 >= 0.0f && rand01 < 1.0f)) {
            ascendc_assert(false, "random_walk: rand must be finite and in [0, 1)");
            __trap();
            return -1;
        }
        const float scaled = rand01 * static_cast<float>(degree);
        // float32 multiplication can round up to degree for large rows.
        idx = scaled >= static_cast<float>(degree) ? degree - 1 : static_cast<int64_t>(scaled);
    } else {
        const uint64_t degree64 = static_cast<uint64_t>(degree);
        // floor(u32 * degree / 2**32), without truncating a large degree or
        // requiring a uint128 product. Both terms fit for int64 CSR indices.
        idx = static_cast<int64_t>((degree64 >> 32) * randBits +
            (((degree64 & 0xffffffffULL) * randBits) >> 32));
    }

    const uint64_t offset = static_cast<uint64_t>(begin) + static_cast<uint64_t>(idx);
    if (offset >= nnz) {
        ascendc_assert(false, "random_walk: neighbor offset out of range");
        __trap();
        return -1;
    }
    const int64_t next = col[offset];
    if (next < 0 || static_cast<uint64_t>(next) >= nodeCount) {
        ascendc_assert(false, "random_walk: col index out of range");
        __trap();
        return -1;
    }
    return next;
}

// Shared signature of the kernel entry points and the SIMT device helpers
// below. The Philox launch passes nullptr for `rand`; the injected launch
// passes caller-supplied rand values.
#define RANDOM_WALK_KERNEL_PARAMETERS                                                             \
    __gm__ const int64_t *rowptr, __gm__ const int64_t *col, __gm__ const int64_t *start,         \
        __gm__ const float *rand, __gm__ int64_t *out, uint64_t startCount,                       \
        uint64_t walkLength, uint64_t nnz, uint64_t nodeCount, uint64_t seed, uint64_t offset


// Content validation runs on every launch that produces a walk result. The
// host short-circuits zero-workload calls (S == 0 or L == 0) before launch,
// because their output never reads rowptr/col/rand. The compute
// section also guards each gather, so it does not require a cross-block
// barrier with these parallel scans. Each validator returns false after
// trapping so the caller stops exactly where the fused loop used to return.
__simt_callee__ inline bool ValidateRowptr(
    __gm__ const int64_t *rowptr,
    uint64_t nnz,
    uint64_t nodeCount,
    uint64_t globalThread,
    uint64_t globalStride)
{
    for (uint64_t i = globalThread; i <= nodeCount; i += globalStride) {
        const int64_t value = rowptr[i];
        if (value < 0 || static_cast<uint64_t>(value) > nnz ||
            (i == 0 && value != 0) ||
            (i == nodeCount && static_cast<uint64_t>(value) != nnz) ||
            (i < nodeCount && rowptr[i + 1] < value)) {
            ascendc_assert(false, "random_walk: rowptr must start at 0, end at nnz and be nondecreasing");
            __trap();
            return false;
        }
    }
    return true;
}

__simt_callee__ inline bool ValidateCol(
    __gm__ const int64_t *col,
    uint64_t nnz,
    uint64_t nodeCount,
    uint64_t globalThread,
    uint64_t globalStride)
{
    for (uint64_t i = globalThread; i < nnz; i += globalStride) {
        const int64_t value = col[i];
        if (value < 0 || static_cast<uint64_t>(value) >= nodeCount) {
            ascendc_assert(false, "random_walk: col index out of range");
            __trap();
            return false;
        }
    }
    return true;
}

template <bool INJECTED>
__simt_callee__ inline bool ValidateRand(
    __gm__ const float *rand,
    uint64_t startCount,
    uint64_t walkLength,
    uint64_t globalThread,
    uint64_t globalStride)
{
    if constexpr (INJECTED) {
        const uint64_t randCount = startCount * walkLength;
        for (uint64_t i = globalThread; i < randCount; i += globalStride) {
            const float value = rand[i];
            if (!(value >= 0.0f && value < 1.0f)) {
                ascendc_assert(false, "random_walk: rand must be finite and in [0, 1)");
                __trap();
                return false;
            }
        }
    }
    return true;
}

template <bool INJECTED>
__simt_callee__ inline bool RunWalks(
    RANDOM_WALK_KERNEL_PARAMETERS,
    uint64_t globalThread,
    uint64_t globalStride)
{
    for (uint64_t walk = globalThread; walk < startCount; walk += globalStride) {
        PhiloxState rng = MakePhiloxState(seed, offset, walk);
        int64_t current = start[walk];
        if (current < 0 || static_cast<uint64_t>(current) >= nodeCount) {
            ascendc_assert(false, "random_walk: start index out of range");
            __trap();
            return false;
        }
        const uint64_t base = walk * (static_cast<uint64_t>(walkLength) + 1);
        out[base] = current;

        for (uint64_t step = 0; step < walkLength; ++step) {
            if (current < 0) {
                out[base + step + 1] = -1;
                continue;
            }
            float rand01 = 0.0f;
            uint32_t randBits = 0;
            if constexpr (INJECTED) {
                rand01 = rand[walk * static_cast<uint64_t>(walkLength) + step];
            } else {
                randBits = PhiloxNext(rng);
            }
            current = WalkStep(
                rowptr, col, current, nnz, nodeCount, INJECTED, rand01, randBits);
            out[base + step + 1] = current;
        }
    }
    return true;
}

template <bool INJECTED>
__simt_vf__ __aicore__ LAUNCH_BOUND(WALK_THREADS_PER_BLOCK) void RandomWalkSimt(
    RANDOM_WALK_KERNEL_PARAMETERS)
{
    const uint32_t blockIndex = blockIdx.x;
    const uint32_t blockCount = gridDim.x;
    const uint32_t threadIndex = threadIdx.x;
    const uint32_t threadCount = blockDim.x;
    const uint64_t globalThread =
        static_cast<uint64_t>(blockIndex) * threadCount + threadIndex;
    const uint64_t globalStride =
        static_cast<uint64_t>(blockCount) * threadCount;

    if (!ValidateRowptr(rowptr, nnz, nodeCount, globalThread, globalStride)) {
        return;
    }
    if (!ValidateCol(col, nnz, nodeCount, globalThread, globalStride)) {
        return;
    }
    if (!ValidateRand<INJECTED>(rand, startCount, walkLength, globalThread, globalStride)) {
        return;
    }
    RunWalks<INJECTED>(
        rowptr, col, start, rand, out, startCount, walkLength, nnz, nodeCount, seed,
        offset, globalThread, globalStride);
}

__attribute__((aiv)) __global__ __aicore__ void TorchSparseRandomWalkPhiloxKernel(RANDOM_WALK_KERNEL_PARAMETERS)
{
    asc_vf_call<RandomWalkSimt<false>>(
        dim3{WALK_THREADS_PER_BLOCK, 1, 1},
        rowptr, col, start, rand, out, startCount, walkLength, nnz, nodeCount, seed, offset);
}

__attribute__((aiv)) __global__ __aicore__ void TorchSparseRandomWalkInjectedKernel(RANDOM_WALK_KERNEL_PARAMETERS)
{
    asc_vf_call<RandomWalkSimt<true>>(
        dim3{WALK_THREADS_PER_BLOCK, 1, 1},
        rowptr, col, start, rand, out, startCount, walkLength, nnz, nodeCount, seed, offset);
}

uint32_t ResolveBlockDim(uint64_t workSize, uint32_t availableCores)
{
    const uint64_t requiredCores =
        (workSize + WALK_THREADS_PER_BLOCK - 1) / WALK_THREADS_PER_BLOCK;
    uint32_t launchBlocks = static_cast<uint32_t>(
        requiredCores < availableCores ? requiredCores : availableCores);
    return launchBlocks == 0 ? 1 : launchBlocks;
}

uint64_t ResolveWorkSize(const opsgnn::TorchSparseRandomWalkLaunchParams &params, bool injected)
{
    uint64_t total = params.nodeCount + 1;
    total = total < params.nnz ? params.nnz : total;
    total = total < params.startCount ? params.startCount : total;
    if (injected) {
        const uint64_t randCount = params.startCount * params.walkLength;
        total = total < randCount ? randCount : total;
    }
    return total;
}
} // namespace

namespace opsgnn {

void LaunchTorchSparseRandomWalkKernel(
    const int64_t *rowptr,
    const int64_t *col,
    const int64_t *start,
    int64_t *out,
    const TorchSparseRandomWalkLaunchParams &params,
    uint32_t availableCores,
    aclrtStream stream)
{
    TorchSparseRandomWalkPhiloxKernel<<<ResolveBlockDim(
        ResolveWorkSize(params, false), availableCores), nullptr, stream>>>(
        rowptr, col, start, nullptr, out, params.startCount, params.walkLength, params.nnz,
        params.nodeCount, params.seed, params.offset);
}

void LaunchTorchSparseRandomWalkExactKernel(
    const int64_t *rowptr,
    const int64_t *col,
    const int64_t *start,
    const float *rand,
    int64_t *out,
    const TorchSparseRandomWalkLaunchParams &params,
    uint32_t availableCores,
    aclrtStream stream)
{
    TorchSparseRandomWalkInjectedKernel<<<ResolveBlockDim(
        ResolveWorkSize(params, true), availableCores), nullptr, stream>>>(
        rowptr, col, start, rand, out, params.startCount, params.walkLength, params.nnz,
        params.nodeCount, params.seed, params.offset);
}

} // namespace opsgnn

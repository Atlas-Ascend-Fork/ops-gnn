/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "radius.h"
#include <cmath>
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

template <typename T, int DType>
__simt_callee__ __aicore__ inline float RadiusToFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__simt_callee__ __aicore__ inline float RadiusToFloat<uint16_t, 1>(uint16_t value)
{
    return __half2float(__ushort_as_half(value));
}

template <>
__simt_callee__ __aicore__ inline float RadiusToFloat<uint16_t, 2>(uint16_t value)
{
    return __bfloat162float(__ushort_as_bfloat16(value));
}
// Ascending top-K insert for a single candidate index. Keeps top[] sorted in
// ascending x-index order so the merged result matches the CPU reference order
// (official test_radius.py torch.equal). Returns true when the candidate was
// accepted.
__simt_callee__ __aicore__ inline bool TopInsert(int64_t* top, int64_t& topLen,
                                                 int64_t maxNeighbors,
                                                 int64_t idx) {
    if (topLen < maxNeighbors) {
        int64_t pos = topLen;
        while (pos > 0 && top[pos - 1] > idx) {
            top[pos] = top[pos - 1];
            --pos;
        }
        top[pos] = idx;
        ++topLen;
        return true;
    }
    if (idx < top[topLen - 1]) {
        int64_t pos = topLen - 1;
        while (pos > 0 && top[pos - 1] > idx) {
            top[pos] = top[pos - 1];
            --pos;
        }
        top[pos] = idx;
        return true;
    }
    return false;
}

// Find the batch containing query q from a prefix-sum pointer, via binary
// search. Returns the batch id in [0, batch_size). Mirrors the host BatchOf;
// kept as a separate kernel-side copy because __simt_callee__ code cannot
// call into the host translation unit.
__simt_callee__ __aicore__ inline int64_t FindBatch(int64_t q,
                                                    __gm__ int64_t* ptr_y,
                                                    int64_t batch_size) {
    int64_t lo = 0;
    int64_t hi = batch_size;
    while (lo < hi) {
        const int64_t mid = lo + ((hi - lo) >> 1);
        if (ptr_y[mid + 1] <= q) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// AABB min-distance pre-check: a whole cell cannot contain a neighbor when its
// closest corner is farther than r. Returns true when the cell may contribute.
__simt_callee__ __aicore__ inline bool CellMayOverlap(float y0, float y1, float y2,
                                                      int64_t probeX,
                                                      int64_t probeY,
                                                      int64_t probeZ, float cs,
                                                      float o0, float o1, float o2,
                                                      float r2) {
    float cx0 = o0 + static_cast<float>(probeX) * cs;
    float cx1 = cx0 + cs;
    float cy0 = o1 + static_cast<float>(probeY) * cs;
    float cy1 = cy0 + cs;
    float cz0 = o2 + static_cast<float>(probeZ) * cs;
    float cz1 = cz0 + cs;
    float dx = (y0 < cx0) ? (cx0 - y0) : ((y0 > cx1) ? (y0 - cx1) : 0.0f);
    float dy = (y1 < cy0) ? (cy0 - y1) : ((y1 > cy1) ? (y1 - cy1) : 0.0f);
    float dz = (y2 < cz0) ? (cz0 - y2) : ((y2 > cz1) ? (y2 - cz1) : 0.0f);
    return dx * dx + dy * dy + dz * dz <= r2;
}

// Scan the points of one cell (in x-index order) and insert in-radius
// candidates into the ascending top-K array. Early-exits the cell when the
// sorted-array max can no longer be improved by later points.
template <typename T, int DType>
__simt_callee__ __aicore__ inline void ScanCellPoints(
    __gm__ T* reordered_x, __gm__ int64_t* reordered_orig_idx,
    int64_t p, int64_t end, float y0, float y1, float y2, float cs, float r2,
    int64_t maxNeighbors, int64_t q, int64_t featureDim,
    int64_t ignoreSame, int64_t* top, int64_t& topLen) {
    int64_t topMax = (topLen > 0) ? top[topLen - 1] : (1LL << 62);
    // Per-axis early reject using SQUARED offsets: a point whose single-axis
    // squared offset exceeds r2 can never be in-radius (euclidean >= any
    // axis). Compare against r2 directly -- NOT cs^2 (= cell_size^2 =
    // float(r)^2) -- because cs can be smaller than the true radius when
    // float(r)^2 < float(r*r), which would drop legitimate boundary
    // candidates. Avoiding sqrt keeps the bisheng SIMT linker happy.
    for (; p < end; ++p) {
        int64_t idx = reordered_orig_idx[p];
        if (topLen >= maxNeighbors && idx >= topMax) {
            break;  // cell ascending: remaining points can't improve
        }
        float xv0 = RadiusToFloat<T, DType>(reordered_x[p * featureDim]);
        float d0 = xv0 - y0;
        if (d0 * d0 > r2) continue;
        float xv1 = RadiusToFloat<T, DType>(reordered_x[p * featureDim + 1]);
        float d1 = xv1 - y1;
        if (d1 * d1 > r2) continue;
        float dist2 = d0 * d0 + d1 * d1;
        if (dist2 > r2) continue;
        if (featureDim > 2) {
            float xv2 = RadiusToFloat<T, DType>(reordered_x[p * featureDim + 2]);
            float d2 = xv2 - y2;
            if (d2 * d2 > r2) continue;
            dist2 += d2 * d2;
            if (dist2 > r2) continue;
        }
        if (ignoreSame != 0 && idx == q) {
            continue;
        }
        TopInsert(top, topLen, maxNeighbors, idx);
        if (topLen > 0) {
            topMax = top[topLen - 1];
        }
    }
}

// Grid query path: compute the query's cell, probe its 27 neighbor cells, and
// maintain the ascending top-K over in-radius candidates.
// Probe the 27 neighbor cells of the query's cell and insert in-radius
// candidates into the ascending top-K array. Returns the top-K length.
template <typename T, int DType>
__simt_callee__ __aicore__ inline int64_t ProbeNeighborCells(
    __gm__ int64_t* cell_offsets, __gm__ T* reordered_x,
    __gm__ int64_t* reordered_orig_idx, int64_t cellX, int64_t cellY,
    int64_t cellZ, int64_t dim0, int64_t dim1, int64_t dim2,
    int64_t batchKey, float y0, float y1, float y2, float cs, float o0,
    float o1, float o2, float r2, int64_t maxNeighbors, int64_t q,
    int64_t featureDim, int64_t ignoreSame, int64_t* top) {
    constexpr int8_t kOffX[27] = {0, -1, 1, 0, 0, 0, 0, -1, -1, 1, 1,
                                  -1, -1, 1, 1, 0, 0, 0, 0, -1, -1,
                                  -1, -1, 1, 1, 1, 1};
    constexpr int8_t kOffY[27] = {0, 0, 0, -1, 1, 0, 0, -1, 1, -1, 1,
                                  0, 0, 0, 0, -1, -1, 1, 1, -1, -1,
                                  1, 1, -1, -1, 1, 1};
    constexpr int8_t kOffZ[27] = {0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0,
                                  -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
                                  -1, 1, -1, 1, -1, 1};
    int64_t topLen = 0;
    for (int64_t oi = 0; oi < 27; ++oi) {
        int64_t probeX = cellX + kOffX[oi];
        int64_t probeY = cellY + kOffY[oi];
        int64_t probeZ = cellZ + kOffZ[oi];
        if (probeX < 0 || probeX >= dim0) continue;
        if (probeY < 0 || probeY >= dim1) continue;
        if (probeZ < 0 || probeZ >= dim2) continue;
        if (!CellMayOverlap(y0, y1, y2, probeX, probeY, probeZ, cs,
                            o0, o1, o2, r2)) {
            continue;
        }
        int64_t cellId = (probeX * dim1 + probeY) * dim2 + probeZ;
        int64_t key = batchKey + cellId;
        int64_t start = cell_offsets[key];
        int64_t end = cell_offsets[key + 1];
        if (start >= end) continue;  // empty cell
        ScanCellPoints<T, DType>(reordered_x, reordered_orig_idx, start, end,
                                 y0, y1, y2, cs, r2, maxNeighbors, q,
                                 featureDim, ignoreSame, top, topLen);
    }
    return topLen;
}

template <typename T, int DType>
__simt_callee__ __aicore__ inline int64_t GridQuery(
    __gm__ T* x, __gm__ T* y, __gm__ int64_t* cell_offsets,
    __gm__ T* reordered_x, __gm__ int64_t* reordered_orig_idx,
    __gm__ uint8_t* config_ptr, const RadiusTilingData& tiling, int64_t q,
    int64_t batch, int64_t* top) {
    __gm__ float* cfg = reinterpret_cast<__gm__ float*>(config_ptr);
    float cellSize = cfg[0];
    // Guard the divisor for codecheck G.EXP.22 (cellSize comes from a GM
    // buffer the analyzer cannot prove non-zero; the host always writes > 0).
    const float safeCellSize = (cellSize > 0.0f) ? cellSize : 1e-9f;
    float origin0 = cfg[1];
    float origin1 = cfg[2];
    float origin2 = cfg[3];
    __gm__ int64_t* dims = reinterpret_cast<__gm__ int64_t*>(cfg + 4);
    int64_t dim0 = dims[0];
    int64_t dim1 = dims[1];
    int64_t dim2 = dims[2];
    int64_t gridCellCount = dim0 * dim1 * dim2;

    const int64_t featureDim = tiling.feature_dim;
    float y0 = RadiusToFloat<T, DType>(y[q * featureDim]);
    float y1 = RadiusToFloat<T, DType>(y[q * featureDim + 1]);
    float y2 = RadiusToFloat<T, DType>(y[q * featureDim + 2]);

    int64_t cellX = static_cast<int64_t>((y0 - origin0) / safeCellSize);
    int64_t cellY = static_cast<int64_t>((y1 - origin1) / safeCellSize);
    int64_t cellZ = static_cast<int64_t>((y2 - origin2) / safeCellSize);
    if (cellX < 0) cellX = 0;
    if (cellX >= dim0) cellX = dim0 - 1;
    if (cellY < 0) cellY = 0;
    if (cellY >= dim1) cellY = dim1 - 1;
    if (cellZ < 0) cellZ = 0;
    if (cellZ >= dim2) cellZ = dim2 - 1;

    int64_t batchKey = batch * gridCellCount;
    return ProbeNeighborCells<T, DType>(
        cell_offsets, reordered_x, reordered_orig_idx, cellX, cellY, cellZ,
        dim0, dim1, dim2, batchKey, y0, y1, y2, cellSize, origin0, origin1,
        origin2, tiling.r2, tiling.max_num_neighbors, q, featureDim,
        tiling.ignore_same_index, top);
}

// Brute-force scan of the query's batch with early-exit (used when the grid is
// ineligible or the input is not 3D).
template <typename T, int DType>
__simt_callee__ __aicore__ inline int64_t BruteForceQuery(
    __gm__ T* x, __gm__ T* y, __gm__ int64_t* ptr_x, __gm__ int64_t* ptr_y,
    __gm__ int64_t* out_row1,
    const RadiusTilingData& tiling, int64_t q) {
    int64_t xStart = 0;
    int64_t xEnd = tiling.n;
    if (tiling.batch_size > 1) {
        int64_t batch = FindBatch(q, ptr_y, tiling.batch_size);
        xStart = ptr_x[batch];
        xEnd = ptr_x[batch + 1];
    }
    int64_t kept = 0;
    const int64_t maxNeighbors = tiling.max_num_neighbors;
    const int64_t featureDim = tiling.feature_dim;
    for (int64_t i = xStart; i < xEnd && kept < maxNeighbors; ++i) {
        float dist2 = 0.0f;
        for (int64_t d = 0; d < featureDim; ++d) {
            float xv = RadiusToFloat<T, DType>(x[i * featureDim + d]);
            float yv = RadiusToFloat<T, DType>(y[q * featureDim + d]);
            float diff = xv - yv;
            dist2 += diff * diff;
        }
        if (dist2 > tiling.r2) continue;
        if (tiling.ignore_same_index != 0 && i == q) continue;
        out_row1[q * maxNeighbors + kept] = i;
        ++kept;
    }
    return kept;
}

template <typename T, int DType>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) void RadiusKernelSimt(
    __gm__ T* x, __gm__ T* y,
    __gm__ int64_t* ptr_x, __gm__ int64_t* ptr_y,
    __gm__ int64_t* unique_cell_ids, __gm__ int64_t* cell_offsets,
    __gm__ T* reordered_x, __gm__ int64_t* reordered_orig_idx,
    __gm__ int64_t* counts,
    __gm__ int64_t* out_row1, __gm__ uint8_t* config_ptr,
    __gm__ uint8_t* tiling_ptr)
{
    // Tiling arrives through GM, never as a by-value SIMT argument: aggregate
    // staging through VF_CALL proved unreliable on first launches (950PR,
    // CANN 9.1.0-beta.3). Each thread reads the 96-byte struct once.
    //
    // NOTE: aggregate copy-initialization from a __gm__ pointer
    // (const RadiusTilingData t = *reinterpret_cast<__gm__ RadiusTilingData*>(p))
    // is rejected by the bisheng SIMT compiler ("no matching constructor for
    // RadiusTilingData"), so the local struct is populated field-by-field.
    // Struct layout is fixed by radius_tiling.h and matches the host-side blob.
    __gm__ RadiusTilingData* tiling_gm =
        reinterpret_cast<__gm__ RadiusTilingData*>(tiling_ptr);
    RadiusTilingData tiling;
    tiling.n = tiling_gm->n;
    tiling.m = tiling_gm->m;
    tiling.feature_dim = tiling_gm->feature_dim;
    tiling.batch_size = tiling_gm->batch_size;
    tiling.max_num_neighbors = tiling_gm->max_num_neighbors;
    tiling.r2 = tiling_gm->r2;
    tiling.ignore_same_index = tiling_gm->ignore_same_index;
    tiling.core_num = tiling_gm->core_num;
    tiling.workspace_pairs = tiling_gm->workspace_pairs;
    tiling.use_grid = tiling_gm->use_grid;
    tiling.unique_cells = tiling_gm->unique_cells;
    tiling.config_ptr = tiling_gm->config_ptr;
    // NOTE: the old grid-stride mapping q = blockIdx*threadNum + threadIdxInBlock
    // with totalThreadNum = blockNum*threadNum only uses the first m threads,
    // i.e. the first m/threadNum blocks. For m=8192, threadNum=1024 that is 8 of
    // 56 AIV blocks -> 7/8 of the machine idle (measured: 4226us kernel with 56
    // blocks where only 8 did work). We instead spread queries across ALL blocks:
    // block b owns a contiguous chunk [b*queriesPerBlock, ...), threads stride
    // within that chunk. Every AIV core now participates.
    const int64_t blockIdx = static_cast<int64_t>(AscendC::Simt::GetBlockIdx());
    const int64_t threadIdxInBlock =
        static_cast<int64_t>(AscendC::Simt::GetThreadIdx());
    const int64_t threadNum = static_cast<int64_t>(AscendC::Simt::GetThreadNum());
    const int64_t blockNum = static_cast<int64_t>(AscendC::Simt::GetBlockNum());
    const int64_t maxNeighbors = tiling.max_num_neighbors;

    // Each block processes a contiguous range of queries so that all launched
    // blocks (AIV cores) share the m queries evenly. Guard the divisor for
    // codecheck G.EXP.22 (GetBlockNum is a runtime value the analyzer cannot
    // prove non-zero; a launched kernel always has >= 1 block).
    const int64_t safeBlockNum = (blockNum > 0) ? blockNum : 1;
    const int64_t queriesPerBlock = (tiling.m + safeBlockNum - 1) / safeBlockNum;
    int64_t qStart = blockIdx * queriesPerBlock;
    int64_t qEnd = qStart + queriesPerBlock;
    if (qEnd > tiling.m) qEnd = tiling.m;

    for (int64_t q = qStart + threadIdxInBlock; q < qEnd; q += threadNum) {
        int64_t batch =
            (tiling.batch_size > 1) ? FindBatch(q, ptr_y, tiling.batch_size) : 0;

        if (tiling.use_grid == 1) {
            int64_t top[64];
            int64_t topLen =
                GridQuery<T, DType>(x, y, cell_offsets, reordered_x,
                                    reordered_orig_idx, config_ptr, tiling,
                                    q, batch, top);
            for (int64_t i = 0; i < topLen; ++i) {
                out_row1[q * maxNeighbors + i] = top[i];
            }
            counts[q] = topLen;
        } else {
            counts[q] =
                BruteForceQuery<T, DType>(x, y, ptr_x, ptr_y,
                                          out_row1, tiling, q);
        }
    }
}

template <typename T, int DType>
__attribute__((aiv)) __global__ __aicore__ void RadiusKernel(
    __gm__ T* x, __gm__ T* y,
    __gm__ int64_t* ptr_x, __gm__ int64_t* ptr_y,
    __gm__ int64_t* unique_cell_ids, __gm__ int64_t* cell_offsets,
    __gm__ T* reordered_x, __gm__ int64_t* reordered_orig_idx,
    __gm__ int64_t* counts,
    __gm__ int64_t* out_row1, __gm__ uint8_t* config_ptr,
    __gm__ uint8_t* tiling_ptr)
{
    AscendC::Simt::VF_CALL<RadiusKernelSimt<T, DType>>(
        AscendC::Simt::Dim3{1024}, x, y, ptr_x, ptr_y, unique_cell_ids,
        cell_offsets, reordered_x, reordered_orig_idx, counts,
        out_row1, config_ptr, tiling_ptr);
}

namespace opsgnn {

template <typename T, int DType>
void Radius(T* x, T* y, const RadiusLaunchArgs<T>& args,
                        aclrtStream stream)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    RadiusKernel<T, DType><<<coreNum, nullptr, stream>>>(
        x, y, args.ptr_x, args.ptr_y, args.unique_cell_ids,
        args.cell_offsets, args.reordered_x, args.reordered_orig_idx,
        args.counts, args.out_row1, args.config_ptr,
        args.tiling_gm);
}

template void Radius<float, 0>(float*, float*,
                                           const RadiusLaunchArgs<float>&, aclrtStream);

template void Radius<uint16_t, 1>(uint16_t*, uint16_t*,
                                              const RadiusLaunchArgs<uint16_t>&, aclrtStream);

template void Radius<uint16_t, 2>(uint16_t*, uint16_t*,
                                              const RadiusLaunchArgs<uint16_t>&, aclrtStream);

}  // namespace opsgnn

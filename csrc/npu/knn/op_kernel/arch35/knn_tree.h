/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// NPU implementation of the partition/order rules used by torch_cluster's
// bundled nanoflann. Adapted from the BSD-licensed KDTreeBaseClass algorithm.
// Copyright 2011-16 Jose Luis Blanco (joseluisblancoc@gmail.com).
// See the repository-root LICENSE for the complete BSD license.
#pragma once

namespace opsgnn {
namespace KnnCpuOrder {

constexpr int NODE_WIDTH = KNN_TREE_NODE_FIELDS;

template <int DType>
__simt_callee__ __aicore__ inline float Round(float value) {
    if constexpr (DType == 1) {
        return static_cast<float>(__float2half_rn(value));
    }
    return value;
}

template <typename T, int DType>
__simt_callee__ __aicore__ inline float Read(T value) {
    if constexpr (DType == 1) {
        return static_cast<float>(__ushort_as_half(value));
    }
    return static_cast<float>(value);
}

template <typename T, int DType>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void InitTreeSimt(
    __gm__ T* x, __gm__ int64_t* ptr_x, __gm__ int64_t* tree,
    __gm__ float* bounds, int64_t n, int64_t f, int64_t batches) {
    const int64_t tid = AscendC::Simt::GetBlockIdx() * 256 + AscendC::Simt::GetThreadIdx();
    const int64_t stride = AscendC::Simt::GetBlockNum() * 256;
    __gm__ int64_t* permutation = tree + 2 * n * NODE_WIDTH;
    for (int64_t i = tid; i < n; i += stride) permutation[i] = i;
    const int64_t bf = DType == 1 ? 1 : f;
    const int64_t width = 2 * bf + 2;
    for (int64_t item = tid; item < batches * bf; item += stride) {
        const int64_t batch = item / bf;
        const int64_t feature = item % bf;
        const int64_t start = batches > 1 ? ptr_x[batch] : 0;
        const int64_t end = batches > 1 ? ptr_x[batch + 1] : n;
        if (end == start) continue;
        const int64_t root = 2 * start;
        float low = Read<T, DType>(x[start * f + feature]);
        float high = low;
        for (int64_t i = start + 1; i < end; ++i) {
            const float value = Read<T, DType>(x[i * f + feature]);
            if (value < low) low = value;
            if (value > high) high = value;
        }
        bounds[root * width + 2 * feature] = low;
        bounds[root * width + 2 * feature + 1] = high;
        if (feature == 0) {
            tree[root * NODE_WIDTH] = start;
            tree[root * NODE_WIDTH + 1] = end;
            tree[root * NODE_WIDTH + 4] = -1;
            tree[root * NODE_WIDTH + 5] = 0;
        }
    }
}

template <typename T, int DType>
__simt_callee__ __aicore__ inline void Range(
    __gm__ T* x, __gm__ int64_t* permutation, int64_t start,
    int64_t end, int64_t f, int64_t feature, float& low, float& high) {
    low = Read<T, DType>(x[permutation[start] * f + feature]);
    high = low;
    for (int64_t i = start + 1; i < end; ++i) {
        const float value = Read<T, DType>(x[permutation[i] * f + feature]);
        if (value < low) low = value;
        if (value > high) high = value;
    }
}

// All cooperative helpers are called by every thread in the block. UB
// reductions preserve min/max exactly; no distance arithmetic is reduced here.
template <typename T, int DType>
__simt_callee__ __aicore__ inline void CooperativeRange(
    __gm__ T* x, __gm__ int64_t* permutation, int64_t start,
    int64_t end, int64_t f, int64_t feature, __ubuf__ float* scratch,
    float& low, float& high) {
    const int64_t tid = AscendC::Simt::GetThreadIdx();
    const int64_t chunk = (end - start + 255) / 256;
    const int64_t first = start + tid * chunk;
    const int64_t last = first + chunk < end ? first + chunk : end;
    float a = 3.402823466e+38F;
    float b = -3.402823466e+38F;
    for (int64_t i = first; i < last; ++i) {
        const float v = Read<T, DType>(x[permutation[i] * f + feature]);
        if (v < a) a = v;
        if (v > b) b = v;
    }
    scratch[tid] = a;
    scratch[256 + tid] = b;
    asc_syncthreads();
    // Adjacent range pairs preserve the earliest value on equal min/max,
    // including the sign of zero, without a 256-element lane-zero loop.
    for (int64_t stride = 1; stride < 256; stride *= 2) {
        const int64_t i = tid * (2 * stride);
        if (i < 256) {
            const float other_low = scratch[i + stride];
            const float other_high = scratch[256 + i + stride];
            if (other_low < scratch[i]) scratch[i] = other_low;
            if (other_high > scratch[256 + i]) scratch[256 + i] = other_high;
        }
        asc_syncthreads();
    }
    low = scratch[0];
    high = scratch[256];
    asc_syncthreads();
}

template <typename T, int DType>
__simt_callee__ __aicore__ inline int64_t CountLeftPartition(
    __gm__ T* x, __gm__ int64_t* permutation, int64_t first,
    int64_t last, int64_t f, int64_t feature, float cut, bool inclusive,
    __ubuf__ int64_t* scratch) {
    const int64_t tid = AscendC::Simt::GetThreadIdx();
    int64_t total = 0;
    for (int64_t i = first; i < last; ++i) {
        const float v = Read<T, DType>(x[permutation[i] * f + feature]);
        if (inclusive ? v <= cut : v < cut) ++total;
    }
    scratch[tid] = total;
    asc_syncthreads();
    for (int64_t stride = 1; stride < 256; stride *= 2) {
        const int64_t i = tid * (2 * stride);
        if (i < 256) scratch[i] += scratch[i + stride];
        asc_syncthreads();
    }
    if (tid == 0) scratch[512] = scratch[0];
    asc_syncthreads();
    return scratch[512];
}

template <typename T, int DType>
__simt_callee__ __aicore__ inline void PartitionOffsets(
    __gm__ T* x, __gm__ int64_t* permutation, int64_t first,
    int64_t last, int64_t f, int64_t feature, float cut, bool inclusive,
    int64_t boundary, __ubuf__ int64_t* scratch, int64_t& lp, int64_t& rp) {
    const int64_t tid = AscendC::Simt::GetThreadIdx();
    int64_t lcount = 0;
    int64_t rcount = 0;
    for (int64_t i = first; i < last; ++i) {
        const float v = Read<T, DType>(x[permutation[i] * f + feature]);
        const bool belongs_left = inclusive ? v <= cut : v < cut;
        if (i < boundary && !belongs_left) ++lcount;
        if (i >= boundary && belongs_left) ++rcount;
    }
    scratch[tid] = lcount;
    scratch[256 + tid] = rcount;
    asc_syncthreads();
    for (int64_t stride = 1; stride < 256; stride *= 2) {
        const int64_t add_left = tid >= stride ? scratch[tid - stride] : 0;
        const int64_t add_right = tid >= stride ? scratch[256 + tid - stride] : 0;
        // All prior-level neighbor reads finish before any lane overwrites.
        asc_syncthreads();
        scratch[tid] += add_left;
        scratch[256 + tid] += add_right;
        asc_syncthreads();
    }
    if (tid == 0) scratch[513] = scratch[255];
    lp = scratch[tid] - lcount;
    rp = scratch[256 + tid] - rcount;
}

template <typename T, int DType>
__simt_callee__ __aicore__ inline int64_t CooperativePartition(
    __gm__ T* x, __gm__ int64_t* permutation, int64_t start,
    int64_t end, int64_t f, int64_t feature, float cut, bool inclusive,
    __ubuf__ int64_t* scratch, __gm__ int64_t* left_wrong,
    __gm__ int64_t* right_wrong) {
    const int64_t tid = AscendC::Simt::GetThreadIdx();
    const int64_t chunk = (end - start + 255) / 256;
    const int64_t first = start + tid * chunk;
    const int64_t last = first + chunk < end ? first + chunk : end;
    const int64_t count = CountLeftPartition<T, DType>(
        x, permutation, first, last, f, feature, cut, inclusive, scratch);
    const int64_t boundary = start + count;
    int64_t lp, rp;
    PartitionOffsets<T, DType>(x, permutation, first, last, f, feature,
                              cut, inclusive, boundary, scratch, lp, rp);
    for (int64_t i = first; i < last; ++i) {
        const float v = Read<T, DType>(x[permutation[i] * f + feature]);
        const bool belongs_left = inclusive ? v <= cut : v < cut;
        if (i < boundary && !belongs_left) left_wrong[lp++] = i;
        if (i >= boundary && belongs_left) right_wrong[rp++] = i;
    }
    asc_threadfence_block();
    asc_syncthreads();
    const int64_t swaps = scratch[513];
    // Exactly the original two-pointer swaps: left positions ascend while
    // their corresponding right positions descend. All pairs are disjoint.
    for (int64_t i = tid; i < swaps; i += 256) {
        const int64_t a = left_wrong[i];
        const int64_t b = right_wrong[swaps - 1 - i];
        const int64_t value = permutation[a];
        permutation[a] = permutation[b];
        permutation[b] = value;
    }
    asc_threadfence_block();
    asc_syncthreads();
    return count;
}

template <typename T, int DType, bool Cooperative>
__simt_callee__ __aicore__ inline void NodeRange(
    __gm__ T* x, __gm__ int64_t* permutation, int64_t left, int64_t right,
    int64_t f, int64_t feature, __ubuf__ float* values, float& low, float& high) {
    if constexpr (Cooperative) {
        CooperativeRange<T, DType>(x, permutation, left, right, f, feature, values, low, high);
    } else {
        Range<T, DType>(x, permutation, left, right, f, feature, low, high);
    }
}

struct Split {
    int64_t feature = 0;
    float value = 0.0F;
};

template <typename T, int DType, bool Cooperative>
__simt_callee__ __aicore__ inline Split SelectSplit(
    __gm__ T* x, __gm__ int64_t* permutation, int64_t left, int64_t right,
    int64_t f, __gm__ float* bbox, __ubuf__ float* values) {
    Split result;
    float low = 0.0F;
    float high = 0.0F;
    bool have_range = false;
    // Half(1)-Half(1e-5) rounds to Half(1), so FP16 retains feature zero.
    if constexpr (DType == 0) {
        float max_span = bbox[1] - bbox[0];
        for (int64_t d = 1; d < f; ++d) {
            const float span = bbox[2 * d + 1] - bbox[2 * d];
            if (span > max_span) max_span = span;
        }
        float max_spread = -1.0F;
        for (int64_t d = 0; d < f; ++d) {
            const float span = bbox[2 * d + 1] - bbox[2 * d];
            if (span > (1.0F - 0.00001F) * max_span) {
                float candidate_low, candidate_high;
                NodeRange<T, DType, Cooperative>(x, permutation, left, right, f, d,
                                                 values, candidate_low, candidate_high);
                const float spread = candidate_high - candidate_low;
                if (spread > max_spread) {
                    max_spread = spread;
                    result.feature = d;
                    low = candidate_low;
                    high = candidate_high;
                    have_range = true;
                }
            }
        }
    }
    if (!have_range) {
        NodeRange<T, DType, Cooperative>(
            x, permutation, left, right, f, result.feature, values, low, high);
    }
    result.value = Round<DType>(
        Round<DType>(bbox[2 * result.feature] + bbox[2 * result.feature + 1]) * 0.5F);
    if (result.value < low) result.value = low;
    else if (result.value > high) result.value = high;
    return result;
}

__simt_callee__ __aicore__ inline void WriteChildNodes(
    __gm__ int64_t* tree, int64_t node, int64_t middle, int64_t a, int64_t feature) {
    __gm__ int64_t* data = tree + node * NODE_WIDTH;
    const int64_t b = a + 1;
    data[2] = a;
    data[3] = b;
    data[6] = feature;
    tree[a * NODE_WIDTH] = data[0];
    tree[a * NODE_WIDTH + 1] = middle;
    tree[b * NODE_WIDTH] = middle;
    tree[b * NODE_WIDTH + 1] = data[1];
    tree[a * NODE_WIDTH + 4] = tree[b * NODE_WIDTH + 4] = node;
    tree[a * NODE_WIDTH + 5] = tree[b * NODE_WIDTH + 5] = data[5] + 1;
}

template <typename T, int DType, bool Cooperative>
__simt_callee__ __aicore__ inline void FinalizeChildren(
    __gm__ T* x, __gm__ int64_t* tree, __gm__ float* bounds,
    int64_t n, int64_t f, int64_t node, int64_t middle, int64_t a,
    const Split& split, __ubuf__ float* values) {
    const int64_t tid = Cooperative ? AscendC::Simt::GetThreadIdx() : 0;
    const int64_t stride = Cooperative ? 256 : 1;
    const int64_t bf = DType == 1 ? 1 : f;
    const int64_t width = 2 * bf + 2;
    const int64_t b = a + 1;
    __gm__ int64_t* data = tree + node * NODE_WIDTH;
    __gm__ int64_t* permutation = tree + 2 * n * NODE_WIDTH;
    __gm__ float* bbox = bounds + node * width;
    if (tid == 0) WriteChildNodes(tree, node, middle, a, split.feature);
    for (int64_t d = tid; d < 2 * bf; d += stride) {
        bounds[a * width + d] = bbox[d];
        bounds[b * width + d] = bbox[d];
    }
    if constexpr (Cooperative) {
        asc_threadfence_block();
        asc_syncthreads();
    }
    if (tid == 0) {
        bounds[a * width + 2 * split.feature + 1] = split.value;
        bounds[b * width + 2 * split.feature] = split.value;
    }
    float low, high;
    NodeRange<T, DType, Cooperative>(
        x, permutation, data[0], middle, f, split.feature, values, low, high);
    if (tid == 0) bbox[width - 2] = high;
    NodeRange<T, DType, Cooperative>(
        x, permutation, middle, data[1], f, split.feature, values, low, high);
    if (tid == 0) bbox[width - 1] = low;
    if constexpr (Cooperative) {
        asc_threadfence_block();
        asc_syncthreads();
    }
}

template <typename T, int DType>
__simt_callee__ __aicore__ inline void CooperativeNode(
    __gm__ T* x, __gm__ int64_t* tree, __gm__ float* bounds,
    int64_t n, int64_t f, int64_t node, int64_t& next,
    __ubuf__ int64_t* scratch, __gm__ int64_t* left_wrong,
    __gm__ int64_t* right_wrong) {
    const int64_t tid = AscendC::Simt::GetThreadIdx();
    const int64_t bf = DType == 1 ? 1 : f;
    const int64_t width = 2 * bf + 2;
    __gm__ int64_t* permutation = tree + 2 * n * NODE_WIDTH;
    __gm__ int64_t* data = tree + node * NODE_WIDTH;
    __gm__ float* bbox = bounds + node * width;
    __ubuf__ float* values = reinterpret_cast<__ubuf__ float*>(scratch + 528);
    const int64_t left = data[0];
    const int64_t right = data[1];
    const int64_t count = right - left;
    if (count <= 10) {
        if (tid == 0) data[2] = data[3] = data[6] = -1;
        for (int64_t i = left + tid; i < right; i += 256) {
            permutation[n + permutation[i]] = i;
            permutation[2 * n + permutation[i]] = node;
        }
        asc_threadfence_block();
        asc_syncthreads();
        return;
    }
    const Split split = SelectSplit<T, DType, true>(x, permutation, left, right, f, bbox, values);
    const int64_t lim1 = CooperativePartition<T, DType>(
        x, permutation, left, right, f, split.feature, split.value, false, scratch, left_wrong, right_wrong);
    const int64_t lim2 = CooperativePartition<T, DType>(
        x, permutation, left, right, f, split.feature, split.value, true, scratch, left_wrong, right_wrong);
    const int64_t middle = left + (lim1 > count / 2 ? lim1 :
                                     (lim2 < count / 2 ? lim2 : count / 2));
    const int64_t a = next;
    next += 2;
    FinalizeChildren<T, DType, true>(x, tree, bounds, n, f, node, middle, a, split, values);
}

template <typename T, int DType, bool Inclusive>
__simt_callee__ __aicore__ inline int64_t SerialPartition(
    __gm__ T* x, __gm__ int64_t* permutation, int64_t left,
    int64_t count, int64_t f, const Split& split, int64_t l) {
    int64_t r = count - 1;
    for (;;) {
        while (l <= r) {
            const float value = Read<T, DType>(x[permutation[left + l] * f + split.feature]);
            if (!(Inclusive ? value <= split.value : value < split.value)) break;
            ++l;
        }
        while (r && l <= r) {
            const float value = Read<T, DType>(x[permutation[left + r] * f + split.feature]);
            if (!(Inclusive ? value > split.value : value >= split.value)) break;
            --r;
        }
        if (l > r || !r) break;
        const int64_t tmp = permutation[left + l];
        permutation[left + l] = permutation[left + r];
        permutation[left + r] = tmp;
        ++l;
        --r;
    }
    return l;
}

template <typename T, int DType>
__simt_callee__ __aicore__ inline void SerialNode(
    __gm__ T* x, __gm__ int64_t* tree, __gm__ float* bounds,
    int64_t n, int64_t f, int64_t node, __gm__ int64_t* next_node) {
    __gm__ int64_t* permutation = tree + 2 * n * NODE_WIDTH;
    __gm__ int64_t* data = tree + node * NODE_WIDTH;
    const int64_t width = 2 * (DType == 1 ? 1 : f) + 2;
    __gm__ float* bbox = bounds + node * width;
    const int64_t left = data[0];
    const int64_t right = data[1];
    const int64_t count = right - left;
    if (count <= 10) {
        data[2] = data[3] = data[6] = -1;
        for (int64_t i = left; i < right; ++i) {
            permutation[n + permutation[i]] = i;
            permutation[2 * n + permutation[i]] = node;
        }
        return;
    }
    const Split split = SelectSplit<T, DType, false>(
        x, permutation, left, right, f, bbox, nullptr);
    const int64_t lim1 = SerialPartition<T, DType, false>(x, permutation, left, count, f, split, 0);
    const int64_t lim2 = SerialPartition<T, DType, true>(x, permutation, left, count, f, split, lim1);
    const int64_t middle = left + (lim1 > count / 2 ? lim1 :
                                     (lim2 < count / 2 ? lim2 : count / 2));
    const int64_t a = asc_atomic_add(next_node, static_cast<int64_t>(2));
    FinalizeChildren<T, DType, false>(x, tree, bounds, n, f, node, middle, a, split, nullptr);
}

template <typename T, int DType>
__simt_callee__ __aicore__ inline void BuildBatchTree(
    __gm__ T* x, __gm__ int64_t* ptr_x, __gm__ int64_t* tree,
    __gm__ float* bounds, __ubuf__ int64_t* scratch,
    int64_t n, int64_t f, int64_t batches, int64_t batch) {
    const int64_t tid = AscendC::Simt::GetThreadIdx();
    const int64_t bf = DType == 1 ? 1 : f;
    const int64_t start = batches > 1 ? ptr_x[batch] : 0;
    const int64_t end = batches > 1 ? ptr_x[batch + 1] : n;
    if (start == end) return;
    const int64_t root = 2 * start;
    // Large effective-low-dimensional trees amortize more cooperative
    // nodes. FP16 splitting uses only coordinate zero, even when f>3.
    const int64_t cooperative_nodes =
        end - start >= 8192 && bf <= 3
            ? (end - start >= 16384 ? 63 : 31) : 7;
    // Prefix63: at most 127 nodes/1016 fields, scratch >=229376 fields.
    // Prefix31: at most 63 nodes/504 fields, scratch >=114688 fields.
    // Prefix7: at most 120 fields, scratch >=154 fields for non-leaves.
    __gm__ int64_t* left_wrong = tree + root * NODE_WIDTH +
                                2 * (end - start) * (NODE_WIDTH - 1);
    __gm__ int64_t* right_wrong = left_wrong + (end - start);
    int64_t next = root + 1;
    int64_t prefix_end = root;
    for (; prefix_end < next && prefix_end < root + cooperative_nodes; ++prefix_end) {
        CooperativeNode<T, DType>(x, tree, bounds, n, f, prefix_end, next,
                                  scratch, left_wrong, right_wrong);
    }
    // A depth frontier contains disjoint point ranges. Child IDs may be
    // allocated in any order; only each node's permutation is observable.
    __gm__ int64_t* next_node = tree + root * NODE_WIDTH + 7;
    if (tid == 0) asc_stcg(next_node, next);
    asc_threadfence_block();
    asc_syncthreads();
    int64_t level_start = prefix_end;
    int64_t level_end = next;
    while (level_start < level_end) {
        for (int64_t node = level_start + tid; node < level_end; node += 256) {
            SerialNode<T, DType>(x, tree, bounds, n, f, node, next_node);
        }
        // Publish all children before anyone consumes the next frontier.
        // Bypass DCache for the atomic counter. The second barrier prevents
        // a fast lane updating it before a slower lane has read this level.
        asc_threadfence_block();
        asc_syncthreads();
        level_start = level_end;
        level_end = asc_ldcg(next_node);
        asc_syncthreads();
    }
}

template <typename T, int DType>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void BuildTreeSimt(
    __gm__ T* x, __gm__ int64_t* ptr_x, __gm__ int64_t* tree,
    __gm__ float* bounds, __ubuf__ int64_t* scratch,
    int64_t n, int64_t f, int64_t batches) {
    for (int64_t batch = AscendC::Simt::GetBlockIdx(); batch < batches;
         batch += AscendC::Simt::GetBlockNum()) {
        BuildBatchTree<T, DType>(x, ptr_x, tree, bounds, scratch, n, f, batches, batch);
    }
}

template <typename T, int DType>
__simt_callee__ __aicore__ inline bool Earlier(
    int64_t index, int64_t other, __gm__ T* query,
    __gm__ int64_t* tree, __gm__ float* bounds, int64_t n, int64_t f) {
    if (other < 0) return true;
    if (tree == nullptr) return index < other;
    __gm__ int64_t* inverse = tree + 2 * n * NODE_WIDTH + n;
    __gm__ int64_t* leaves = inverse + n;
    int64_t a = leaves[index];
    int64_t b = leaves[other];
    if (a == b) return inverse[index] < inverse[other];
    while (tree[a * NODE_WIDTH + 5] > tree[b * NODE_WIDTH + 5]) a = tree[a * NODE_WIDTH + 4];
    while (tree[b * NODE_WIDTH + 5] > tree[a * NODE_WIDTH + 5]) b = tree[b * NODE_WIDTH + 4];
    while (tree[a * NODE_WIDTH + 4] != tree[b * NODE_WIDTH + 4]) {
        a = tree[a * NODE_WIDTH + 4];
        b = tree[b * NODE_WIDTH + 4];
    }
    const int64_t parent = tree[a * NODE_WIDTH + 4];
    const int64_t width = 2 * (DType == 1 ? 1 : f) + 2;
    const int64_t feature = tree[parent * NODE_WIDTH + 6];
    const float value = Read<T, DType>(query[feature]);
    const float d1 = Round<DType>(value - bounds[parent * width + width - 2]);
    const float d2 = Round<DType>(value - bounds[parent * width + width - 1]);
    const bool left_first = Round<DType>(d1 + d2) < 0.0F;
    return a == tree[parent * NODE_WIDTH + (left_first ? 2 : 3)];
}

template <typename T, int DType>
__attribute__((aiv)) __global__ __aicore__ void InitTreeKernel(
    __gm__ T* x, __gm__ int64_t* ptr_x, __gm__ int64_t* tree,
    __gm__ float* bounds, int64_t n, int64_t f, int64_t batches) {
    AscendC::Simt::VF_CALL<InitTreeSimt<T, DType>>(
        AscendC::Simt::Dim3(256), x, ptr_x, tree, bounds, n, f, batches);
}

template <typename T, int DType>
__attribute__((aiv)) __global__ __aicore__ void BuildTreeKernel(
    __gm__ T* x, __gm__ int64_t* ptr_x, __gm__ int64_t* tree,
    __gm__ float* bounds, int64_t n, int64_t f, int64_t batches) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> shared;
    pipe.InitBuffer(shared, 8192);
    auto local = shared.Get<int64_t>();
    AscendC::Simt::VF_CALL<BuildTreeSimt<T, DType>>(
        AscendC::Simt::Dim3(256), x, ptr_x, tree, bounds,
        (__ubuf__ int64_t*)local.GetPhyAddr(), n, f, batches);
}

}  // namespace KnnCpuOrder
}  // namespace opsgnn

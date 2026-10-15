/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "knn_kernel.h"

#include "kernel_operator.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_sync_functions.h"
#include "tiling/platform/platform_ascendc.h"
#include "knn_tree.h"

// L2 arithmetic follows nanoflann's BSD-licensed L2_Adaptor.
// Copyright 2011-16 Jose Luis Blanco (joseluisblancoc@gmail.com).
// See docs/en/api_reference.md for KNN source scope and root LICENSE for the BSD license.
namespace opsgnn {

// Translation-unit-local device helpers; only Knn is exposed to the host.
namespace {

constexpr uint32_t KNN_SIMT_THREADS = 256;
constexpr float KNN_MAX_DISTANCE = 3.402823466e+38F;

template <typename T, int DType>
__simt_callee__ __aicore__ inline float KnnToFloat(T value) {
    return static_cast<float>(value);
}

template <>
__simt_callee__ __aicore__ inline float KnnToFloat<uint16_t, 1>(
    uint16_t value) {
    // __half2float expands IEEE-754 edge handling into scalar integer
    // instructions in the CANN 9.1 SIMT headers.  A direct cast from the
    // native half type lowers to the hardware conversion instruction and is
    // materially faster in this conversion-heavy distance loop.
    return static_cast<float>(__ushort_as_half(value));
}

__simt_callee__ __aicore__ inline int64_t FindBatch(
    int64_t query, __gm__ int64_t* ptr_y, int64_t batch_size) {
    int64_t low = 0;
    int64_t high = batch_size;
    while (low < high) {
        const int64_t mid = low + ((high - low) >> 1);
        if (ptr_y[mid + 1] <= query) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

template <typename T, int DType>
struct QueryOrder {
    __gm__ T* query;
    __gm__ int64_t* tree;
    __gm__ float* bounds;
    int64_t n;
    int64_t f;
};

template <typename T, int DType>
__simt_callee__ __aicore__ inline bool IsBetter(
    float distance, int64_t index, float other_distance,
    int64_t other_index, const QueryOrder<T, DType>& order) {
    if (distance < other_distance) return true;
    if (distance > other_distance) return false;
    return KnnCpuOrder::Earlier<T, DType>(
        index, other_index, order.query, order.tree, order.bounds, order.n, order.f);
}

template <int KMax, typename T, int DType, typename DistancePtr, typename IndexPtr>
__simt_callee__ __aicore__ inline void InsertCandidate(
    float distance, int64_t index, int64_t k, DistancePtr best_distance,
    IndexPtr best_index, int64_t& length, const QueryOrder<T, DType>& order) {
    int64_t position;
    if (length < k) {
        position = length++;
    } else {
        if (!IsBetter(distance, index, best_distance[k - 1], best_index[k - 1], order)) return;
        position = k - 1;
    }
    while (position > 0 &&
           IsBetter(distance, index, best_distance[position - 1], best_index[position - 1], order)) {
        best_distance[position] = best_distance[position - 1];
        best_index[position] = best_index[position - 1];
        --position;
    }
    best_distance[position] = distance;
    best_index[position] = index;
}

template <typename T, int DType, int FeatureCache>
__simt_callee__ __aicore__ inline float SquaredDifference(
    __gm__ T* candidate, __gm__ T* query, const T* cached, int64_t feature) {
    const float a = KnnToFloat<T, DType>(candidate[feature]);
    float b;
    if constexpr (FeatureCache > 0) b = KnnToFloat<T, DType>(cached[feature]);
    else b = KnnToFloat<T, DType>(query[feature]);
    const float difference = KnnCpuOrder::Round<DType>(b - a);
    return KnnCpuOrder::Round<DType>(difference * difference);
}

template <typename T, int DType, int FeatureCache>
__simt_callee__ __aicore__ inline float PrepareQuery(
    __gm__ T* query_ptr, T* query_values, int64_t feature_dim, int64_t cosine) {
    float y_norm2 = 0.0F;
    if constexpr (FeatureCache > 0) {
        for (int64_t feature = 0; feature < FeatureCache; ++feature) {
            const T raw_value = query_ptr[feature];
            query_values[feature] = raw_value;
            if (cosine != 0) {
                const float value = KnnToFloat<T, DType>(raw_value);
                y_norm2 += value * value;
            }
        }
    } else if (cosine != 0) {
        for (int64_t feature = 0; feature < feature_dim; ++feature) {
            const float y_value = KnnToFloat<T, DType>(query_ptr[feature]);
            y_norm2 += y_value * y_value;
        }
    }
    return y_norm2;
}

template <typename T, int DType, int FeatureCache>
__simt_callee__ __aicore__ inline float L2Distance(
    __gm__ T* candidate_ptr, __gm__ T* query_ptr,
    const T* query_values, int64_t feature_dim) {
    // Preserve all four-term and tail rounding operations of the CPU oracle.
    float distance = 0.0F;
    const int64_t dimensions = FeatureCache > 0 ? FeatureCache : feature_dim;
    int64_t feature = 0;
    for (; feature + 3 < dimensions; feature += 4) {
        const float p0 = SquaredDifference<T, DType, FeatureCache>(
            candidate_ptr, query_ptr, query_values, feature);
        const float p1 = SquaredDifference<T, DType, FeatureCache>(
            candidate_ptr, query_ptr, query_values, feature + 1);
        const float p2 = SquaredDifference<T, DType, FeatureCache>(
            candidate_ptr, query_ptr, query_values, feature + 2);
        const float p3 = SquaredDifference<T, DType, FeatureCache>(
            candidate_ptr, query_ptr, query_values, feature + 3);
        float group = KnnCpuOrder::Round<DType>(p0 + p1);
        group = KnnCpuOrder::Round<DType>(group + p2);
        group = KnnCpuOrder::Round<DType>(group + p3);
        distance = KnnCpuOrder::Round<DType>(distance + group);
    }
    for (; feature < dimensions; ++feature) {
        const float square = SquaredDifference<T, DType, FeatureCache>(
            candidate_ptr, query_ptr, query_values, feature);
        distance = KnnCpuOrder::Round<DType>(distance + square);
    }
    return distance;
}

template <typename T, int DType, int FeatureCache>
__simt_callee__ __aicore__ inline float CosineDistance(
    __gm__ T* candidate_ptr, __gm__ T* query_ptr,
    const T* query_values, int64_t feature_dim, float y_norm2) {
    float dot = 0.0F;
    float x_norm2 = 0.0F;
    const int64_t dimensions = FeatureCache > 0 ? FeatureCache : feature_dim;
    for (int64_t feature = 0; feature < dimensions; ++feature) {
        const float x_value = KnnToFloat<T, DType>(candidate_ptr[feature]);
        float y_value;
        if constexpr (FeatureCache > 0) y_value = KnnToFloat<T, DType>(query_values[feature]);
        else y_value = KnnToFloat<T, DType>(query_ptr[feature]);
        dot += x_value * y_value;
        x_norm2 += x_value * x_value;
    }
    const float norm_product = x_norm2 * y_norm2;
    return norm_product > 0.0F ? 1.0F - dot * rsqrtf(norm_product) : 1.0F;
}

template <int KMax>
__simt_callee__ __aicore__ inline void InitializeTopK(
    int64_t k, float* best_distance, int64_t* best_index) {
    if constexpr (KMax > 0) {
        for (int64_t rank = 0; rank < k; ++rank) {
            best_distance[rank] = KNN_MAX_DISTANCE;
            best_index[rank] = -1;
        }
    }
}

template <typename T, int DType, int KMax, int FeatureCache>
__simt_callee__ __aicore__ inline void ProcessQuery(
    __gm__ T* x, __gm__ T* y, __gm__ int64_t* ptr_x,
    __gm__ int64_t* ptr_y, __gm__ int64_t* row,
    __gm__ int64_t* col, int64_t query, int64_t k, int64_t n,
    int64_t feature_dim, int64_t batch_size, int64_t cosine,
    __gm__ float* workspace, __gm__ int64_t* tree, __gm__ float* bounds) {
    const QueryOrder<T, DType> order{y + query * feature_dim, tree, bounds, n, feature_dim};
    float best_distance[KMax > 0 ? KMax : 1];
    int64_t best_index[KMax > 0 ? KMax : 1];
    InitializeTopK<KMax>(k, best_distance, best_index);
    int64_t x_start = 0;
    int64_t x_end = n;
    if (batch_size > 1) {
        const int64_t batch = FindBatch(query, ptr_y, batch_size);
        x_start = ptr_x[batch];
        x_end = ptr_x[batch + 1];
    }
    // Cache raw query features once per query, retaining the general fallback.
    T query_values[FeatureCache > 0 ? FeatureCache : 1];
    const float y_norm2 = PrepareQuery<T, DType, FeatureCache>(
        y + query * feature_dim, query_values, feature_dim, cosine);

    int64_t length = 0;
    for (int64_t candidate = x_start; candidate < x_end; ++candidate) {
        __gm__ T* candidate_ptr = x + candidate * feature_dim;
        __gm__ T* query_ptr = y + query * feature_dim;
        float distance = cosine == 0
            ? L2Distance<T, DType, FeatureCache>(candidate_ptr, query_ptr, query_values, feature_dim)
            : CosineDistance<T, DType, FeatureCache>(
                  candidate_ptr, query_ptr, query_values, feature_dim, y_norm2);
        if (!(distance == distance)) {
            distance = KNN_MAX_DISTANCE;
        }
        if constexpr (KMax > 0) {
            InsertCandidate<KMax>(distance, candidate, k, best_distance,
                                  best_index, length, order);
        } else {
            // Large k has no fixed local-array capacity. Each query owns
            // disjoint distance and output-index slices; length ensures no
            // entry is read before an earlier insertion initialized it.
            InsertCandidate<0>(distance, candidate, k,
                                  workspace + query * k,
                                  col + query * k, length, order);
        }
    }

    const int64_t output_offset = query * k;
    for (int64_t rank = 0; rank < k; ++rank) {
        row[output_offset + rank] = query;
        if constexpr (KMax > 0) {
            col[output_offset + rank] = best_index[rank];
        }
    }
}

template <typename T, int DType, int KMax, int FeatureCache>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void KnnKernelSimt(
    __gm__ T* x, __gm__ T* y, __gm__ int64_t* ptr_x,
    __gm__ int64_t* ptr_y, __gm__ int64_t* row, __gm__ int64_t* col,
    __gm__ float* workspace, __gm__ int64_t* tree, __gm__ float* bounds,
    int64_t k, int64_t n, int64_t m, int64_t f, int64_t batches, int64_t cosine) {
    const int64_t block_count = AscendC::Simt::GetBlockNum();
    const int64_t block = AscendC::Simt::GetBlockIdx();
    const int64_t queries_per_block = (m + block_count - 1) / block_count;
    const int64_t start = block * queries_per_block;
    int64_t end = start + queries_per_block;
    if (end > m) end = m;
    for (int64_t query = start + AscendC::Simt::GetThreadIdx(); query < end;
         query += AscendC::Simt::GetThreadNum()) {
        ProcessQuery<T, DType, KMax, FeatureCache>(
            x, y, ptr_x, ptr_y, row, col, query, k, n, f, batches, cosine,
            workspace, tree, bounds);
    }
}

template <typename T, int DType, int KMax, int FeatureCache>
__attribute__((aiv)) __global__ __aicore__ void KnnKernel(
    __gm__ T* x, __gm__ T* y, __gm__ int64_t* ptr_x,
    __gm__ int64_t* ptr_y, __gm__ int64_t* row, __gm__ int64_t* col,
    __gm__ float* workspace, __gm__ int64_t* tree, __gm__ float* bounds,
    int64_t k, int64_t n, int64_t m, int64_t f, int64_t batches, int64_t cosine) {
    AscendC::Simt::VF_CALL<KnnKernelSimt<T, DType, KMax, FeatureCache>>(
        AscendC::Simt::Dim3(KNN_SIMT_THREADS), x, y, ptr_x, ptr_y, row, col,
        workspace, tree, bounds, k, n, m, f, batches, cosine);
}

template <typename T, int DType, int KMax>
void LaunchSelected(
    T* x, T* y, int64_t* ptr_x, int64_t* ptr_y, int64_t* row, int64_t* col,
    float* workspace, int64_t* tree, float* bounds,
    int64_t k, int64_t n, int64_t m, int64_t f, int64_t batches,
    bool cosine, uint32_t cores, aclrtStream stream) {
    if constexpr (KMax > 0 && KMax <= 32) {
        if (f == 3) {
            KnnKernel<T, DType, KMax, 3><<<cores, nullptr, stream>>>(
                x, y, ptr_x, ptr_y, row, col, workspace, tree, bounds,
                k, n, m, f, batches, cosine ? 1 : 0);
            return;
        }
        if (f == 16) {
            KnnKernel<T, DType, KMax, 16><<<cores, nullptr, stream>>>(
                x, y, ptr_x, ptr_y, row, col, workspace, tree, bounds,
                k, n, m, f, batches, cosine ? 1 : 0);
            return;
        }
        if (f == 32) {
            KnnKernel<T, DType, KMax, 32><<<cores, nullptr, stream>>>(
                x, y, ptr_x, ptr_y, row, col, workspace, tree, bounds,
                k, n, m, f, batches, cosine ? 1 : 0);
            return;
        }
    }
    KnnKernel<T, DType, KMax, 0><<<cores, nullptr, stream>>>(
        x, y, ptr_x, ptr_y, row, col, workspace, tree, bounds,
        k, n, m, f, batches, cosine ? 1 : 0);
}

}  // namespace

template <typename T, int DType>
void Knn(
    T* x, T* y, int64_t* ptr_x, int64_t* ptr_y, int64_t* row, int64_t* col,
    int64_t k, int64_t n, int64_t m, int64_t f, int64_t batches, bool cosine,
    float* workspace, int64_t* tree, float* bounds, aclrtStream stream) {
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t cores = platform->GetCoreNumAiv();
    if (cores == 0) cores = 1;
    if (m < static_cast<int64_t>(cores)) cores = static_cast<uint32_t>(m);
    if (!cosine) {
        KnnCpuOrder::InitTreeKernel<T, DType><<<cores, nullptr, stream>>>(
            x, ptr_x, tree, bounds, n, f, batches);
        const uint32_t build_cores = batches < static_cast<int64_t>(cores)
                                         ? static_cast<uint32_t>(batches) : cores;
        KnnCpuOrder::BuildTreeKernel<T, DType><<<build_cores, nullptr, stream>>>(
            x, ptr_x, tree, bounds, n, f, batches);
    }
#define KNN_LAUNCH(CAPACITY) \
    LaunchSelected<T, DType, CAPACITY>(x, y, ptr_x, ptr_y, row, col, workspace, \
        tree, bounds, k, n, m, f, batches, cosine, cores, stream)
    if (k > KNN_LOCAL_TOPK_LIMIT) KNN_LAUNCH(0);
    else if (k == 1) KNN_LAUNCH(1);
    else if (k <= 4) KNN_LAUNCH(4);
    else if (k <= 8) KNN_LAUNCH(8);
    else if (k <= 16) KNN_LAUNCH(16);
    else if (k <= 32) KNN_LAUNCH(32);
    else KNN_LAUNCH(64);
#undef KNN_LAUNCH
}

template void Knn<float, 0>(
    float*, float*, int64_t*, int64_t*, int64_t*, int64_t*, int64_t,
    int64_t, int64_t, int64_t, int64_t, bool, float*, int64_t*, float*, aclrtStream);
template void Knn<uint16_t, 1>(
    uint16_t*, uint16_t*, int64_t*, int64_t*, int64_t*, int64_t*,
    int64_t, int64_t, int64_t, int64_t, int64_t, bool, float*, int64_t*, float*, aclrtStream);

}  // namespace opsgnn

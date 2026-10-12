/**
 * Copyright (c) 2026 Eternal_ZCL.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Adapted from cann/ops-sparse sparse/spmm/arch35/spmm_kernel.cpp,
// commit 9a514470c2cc99e469c34a615048eb8b30a66694 (Huawei Technologies Co., Ltd.).
#pragma once
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "torch_sparse_spmm_tiling.h"

template <class T>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) void PermuteValuesCompute(__gm__ int64_t* perm, __gm__ T* value, __gm__ T* out,
                                                                   int64_t elements, int64_t width)
{
    int64_t tid = static_cast<int64_t>(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() +
                  AscendC::Simt::GetThreadIdx();
    int64_t stride = static_cast<int64_t>(AscendC::Simt::GetBlockNum()) * AscendC::Simt::GetThreadNum();
    for (int64_t id = tid; id < elements; id += stride) {
        out[id] = value[perm[id / width] * width + id % width];
    }
}

template <class T>
__attribute__((aiv)) __global__ __aicore__ void PermuteValuesKernel(__gm__ int64_t* perm, __gm__ T* value,
                                                                    __gm__ T* out, int64_t elements, int64_t width)
{
    AscendC::Simt::VF_CALL<PermuteValuesCompute<T>>(AscendC::Simt::Dim3{512}, perm, value, out, elements, width);
}

template <class T, class O, class A, int R, class I>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) void CsrCompute(__gm__ int64_t* ptr, __gm__ int64_t* col, __gm__ T* val,
                                                         __gm__ T* mat, __gm__ O* out, __gm__ int64_t* arg, SpmmShape s,
                                                         bool weighted)
{
    I tid =
        static_cast<I>(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    I stride = static_cast<I>(AscendC::Simt::GetBlockNum()) * AscendC::Simt::GetThreadNum();
    I m = s.m, n = s.n, features = s.k, batches = s.batches, width = s.valueWidth, edges = s.edges;
    I featureShift = 0;
    for (I remaining = features; remaining > 1; remaining >>= 1) {
        ++featureShift;
    }
    bool powerOfTwo = features > 0 && (features & (features - 1)) == 0;
    for (I id = tid; id < batches * m * features; id += stride) {
        I k = powerOfTwo ? id & (features - 1) : id % features;
        I linearRow = powerOfTwo ? id >> featureShift : id / features;
        I row = batches == 1 ? linearRow : linearRow % m, b = batches == 1 ? 0 : linearRow / m;
        I begin = ptr[row], end = ptr[row + 1], best = edges;
        A acc = 0;
        for (I e = begin; e < end; ++e) {
            I column = col[e];
            A v = static_cast<A>(mat[(b * n + column) * features + k]);
            if (weighted) {
                v *= static_cast<A>(val[e * width + (width == 1 ? 0 : k)]);
            }
            if constexpr (R < 2) {
                acc += v;
            } else if (v == v && (best == s.edges || (R == 2 ? v < acc : v > acc))) {
                acc = v;
                best = e;
            }
        }
        if constexpr (R == 1) {
            if (end > begin) {
                acc /= static_cast<A>(end - begin);
            }
        }
        out[id] = static_cast<O>(acc);
        if constexpr (R >= 2) {
            arg[id] = best;
        }
    }
}

template <class T, class O, class A, int R, class I>
__attribute__((aiv)) __global__ __aicore__ void CsrKernel(__gm__ int64_t* ptr, __gm__ int64_t* col, __gm__ T* val,
                                                          __gm__ T* mat, __gm__ O* out, __gm__ int64_t* arg,
                                                          SpmmShape s, bool weighted)
{
    AscendC::Simt::VF_CALL<CsrCompute<T, O, A, R, I>>(AscendC::Simt::Dim3{512}, ptr, col, val, mat, out, arg, s,
                                                      weighted);
}

template <class T, class O, class A>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void CsrComputeFour(__gm__ int64_t* ptr, __gm__ int64_t* col, __gm__ T* val,
                                                             __gm__ T* mat, __gm__ O* out, SpmmShape s, bool weighted)
{
    // Four adjacent features share row/column traversal and scalar edge weights.
    // This path is selected only when flattened offsets safely fit int32.
    int32_t tid = AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    int32_t stride = AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum();
    int32_t m = s.m, n = s.n, k = s.k, batches = s.batches, width = s.valueWidth;
    int32_t groups = k / 4;
    int32_t group_shift = 0;
    for (int32_t remaining = groups; remaining > 1; remaining >>= 1) {
        ++group_shift;
    }
    bool powerOfTwo = groups > 0 && (groups & (groups - 1)) == 0;
    for (int32_t id = tid; id < batches * m * groups; id += stride) {
        int32_t feature = (powerOfTwo ? id & (groups - 1) : id % groups) * 4;
        int32_t linearRow = powerOfTwo ? id >> group_shift : id / groups;
        int32_t row = batches == 1 ? linearRow : linearRow % m, batch = batches == 1 ? 0 : linearRow / m;
        int32_t begin = ptr[row], end = ptr[row + 1];
        A accum[4] = {0, 0, 0, 0};
        for (int32_t edge = begin; edge < end; ++edge) {
            int32_t offset = (batch * n + static_cast<int32_t>(col[edge])) * k + feature;
            A scalar = weighted && width == 1 ? static_cast<A>(val[edge]) : static_cast<A>(1);
#pragma unroll
            for (int32_t j = 0; j < 4; ++j) {
                A product = static_cast<A>(mat[offset + j]);
                if (weighted) {
                    product *= width == 1 ? scalar : static_cast<A>(val[edge * width + feature + j]);
                }
                accum[j] += product;
            }
        }
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            if (s.reduce == 1 && end > begin) {
                accum[j] /= static_cast<A>(end - begin);
            }
            out[id * 4 + j] = static_cast<O>(accum[j]);
        }
    }
}

template <class T, class O, class A>
__attribute__((aiv)) __global__ __aicore__ void CsrKernelFour(__gm__ int64_t* ptr, __gm__ int64_t* col, __gm__ T* val,
                                                              __gm__ T* mat, __gm__ O* out, SpmmShape s, bool weighted)
{
    AscendC::Simt::VF_CALL<CsrComputeFour<T, O, A>>(AscendC::Simt::Dim3{256}, ptr, col, val, mat, out, s, weighted);
}

// Preserve edge order and the first non-NaN extremum in both prefetch loops.
template <class A, int R>
__simt_callee__ __aicore__ inline void AccumulateEdge(A value, A& acc, int& best, int edge, int edges)
{
    if constexpr (R < 2) {
        acc += value;
    } else if (value == value && (best == edges || (R == 2 ? value < acc : value > acc))) {
        acc = value;
        best = edge;
    }
}

// Prefetch independent edges for scalar-weight, unbatched, fixed-width CSR.
template <class T, class O, class A, int R, int K>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) void CsrPrefetchCompute(__gm__ int64_t* ptr, __gm__ int64_t* col,
                                                                  __gm__ T* val, __gm__ T* mat, __gm__ O* out,
                                                                  __gm__ int64_t* arg, int m, int edges)
{
    int tid = AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    int stride = AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum();
    for (int id = tid; id < m * K; id += stride) {
        int begin = ptr[id / K], end = ptr[id / K + 1], feature = id % K, best = edges;
        A acc = 0;
        float inverse = 1.0f;
        if constexpr (R == 1 && __is_same(A, float)) {
            if (end > begin) {
                inverse = 1.0f / static_cast<float>(end - begin);
            }
        }
        int e = begin;
        for (; e + 7 < end; e += 8) {
            int columns[8];
            A values[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                columns[j] = col[e + j];
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                values[j] = static_cast<A>(mat[columns[j] * K + feature]);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                A v = values[j] * static_cast<A>(val[e + j]);
                AccumulateEdge<A, R>(v, acc, best, e + j, edges);
            }
        }
        for (; e < end; ++e) {
            A v = static_cast<A>(mat[static_cast<int>(col[e]) * K + feature]) * static_cast<A>(val[e]);
            AccumulateEdge<A, R>(v, acc, best, e, edges);
        }
        if constexpr (R == 1) {
            if constexpr (__is_same(A, float)) {
                acc *= inverse;
            } else if (end > begin) {
                acc /= static_cast<A>(end - begin);
            }
        }
        out[id] = static_cast<O>(acc);
        if constexpr (R >= 2) {
            arg[id] = best;
        }
    }
}
template <class T, class O, class A, int R, int K>
__attribute__((aiv)) __global__ __aicore__ void CsrPrefetchKernel(__gm__ int64_t* ptr, __gm__ int64_t* col,
                                                                  __gm__ T* val, __gm__ T* mat, __gm__ O* out,
                                                                  __gm__ int64_t* arg, int m, int edges, int threads)
{
    AscendC::Simt::VF_CALL<CsrPrefetchCompute<T, O, A, R, K>>(AscendC::Simt::Dim3{static_cast<uint32_t>(threads)}, ptr,
                                                              col, val, mat, out, arg, m, edges);
}

template <class T, class O, class A>
void PrefetchLaunch(void* ptr, void* col, void* val, void* mat, void* out, void* arg, SpmmShape s, uint32_t blocks,
                    void* stream)
{
    int waves = (s.m * s.k + blocks * 1024 - 1) / (blocks * 1024);
    int threads = ((s.m * s.k + blocks * waves - 1) / (blocks * waves) + 31) / 32 * 32;
#define PREFETCH(R, K)                                                                               \
    CsrPrefetchKernel<T, O, A, R, K><<<blocks, nullptr, stream>>>(                                   \
        reinterpret_cast<int64_t*>(ptr), reinterpret_cast<int64_t*>(col), reinterpret_cast<T*>(val), \
        reinterpret_cast<T*>(mat), reinterpret_cast<O*>(out), reinterpret_cast<int64_t*>(arg), s.m, s.edges, threads)
#define WIDTH(R)              \
    switch (s.k) {            \
        case 64:              \
            PREFETCH(R, 64);  \
            break;            \
        case 128:             \
            PREFETCH(R, 128); \
            break;            \
        case 256:             \
            PREFETCH(R, 256); \
            break;            \
        case 512:             \
            PREFETCH(R, 512); \
            break;            \
    }
    switch (s.reduce) {
        case 0:
            WIDTH(0);
            break;
        case 1:
            WIDTH(1);
            break;
        case 2:
            WIDTH(2);
            break;
        case 3:
            WIDTH(3);
            break;
    }
#undef WIDTH
#undef PREFETCH
}

template <class T, class O, class A>
void GenericLaunch(void* ptr, void* col, void* val, void* mat, void* out, void* arg, SpmmShape s, bool weighted,
                   bool small, uint32_t blocks, void* stream)
{
#define CALL(R, I)                                                                                   \
    CsrKernel<T, O, A, R, I><<<blocks, nullptr, stream>>>(                                           \
        reinterpret_cast<int64_t*>(ptr), reinterpret_cast<int64_t*>(col), reinterpret_cast<T*>(val), \
        reinterpret_cast<T*>(mat), reinterpret_cast<O*>(out), reinterpret_cast<int64_t*>(arg), s, weighted)
#define RUN(R)            \
    if (small) {          \
        CALL(R, int32_t); \
    } else {              \
        CALL(R, int64_t); \
    }
    switch (s.reduce) {
        case 0:
            RUN(0);
            break;
        case 1:
            RUN(1);
            break;
        case 2:
            RUN(2);
            break;
        case 3:
            RUN(3);
            break;
    }
#undef RUN
#undef CALL
}

template <class T, class O, class A>
void TypedLaunch(void* ptr, void* col, void* val, void* mat, void* out, void* arg, SpmmShape s, bool weighted,
                 uint32_t blocks, void* stream)
{
    // Keep public indices int64, but avoid expensive 64-bit inner-loop arithmetic
    // when all flattened offsets fit int32. Leave room for the final grid stride.
    const int64_t limit = 2147483647LL - static_cast<int64_t>(blocks) * 1024;
    bool small = s.batches * s.m * s.k <= limit && s.batches * s.n * s.k <= limit && s.edges * s.valueWidth <= limit &&
                 s.m <= limit && s.n <= limit && s.edges <= limit;
    if (small && weighted && s.valueWidth == 1 && s.batches == 1 &&
        (s.k == 64 || s.k == 128 || s.k == 256 || s.k == 512)) {
        PrefetchLaunch<T, O, A>(ptr, col, val, mat, out, arg, s, blocks, stream);
        return;
    }
    if (small && s.k % 4 == 0 && s.reduce < 2 && s.batches * s.m * s.k >= 1000000) {
        CsrKernelFour<T, O, A><<<blocks, nullptr, stream>>>(
            reinterpret_cast<int64_t*>(ptr), reinterpret_cast<int64_t*>(col), reinterpret_cast<T*>(val),
            reinterpret_cast<T*>(mat), reinterpret_cast<O*>(out), s, weighted);
        return;
    }
    GenericLaunch<T, O, A>(ptr, col, val, mat, out, arg, s, weighted, small, blocks, stream);
}

template <class T>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) void ValueGradCompute(__gm__ int64_t* ptr, __gm__ int64_t* col, __gm__ T* mat,
                                                               __gm__ float* grad, __gm__ float* out, SpmmShape s)
{
    int64_t tid = static_cast<int64_t>(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() +
                  AscendC::Simt::GetThreadIdx();
    int64_t stride = static_cast<int64_t>(AscendC::Simt::GetBlockNum()) * AscendC::Simt::GetThreadNum();
    for (int64_t id = tid; id < s.edges * s.valueWidth; id += stride) {
        int64_t e = id / s.valueWidth, lo = 0, hi = s.m;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (ptr[mid + 1] <= e) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int64_t row = lo;
        float acc = 0;
        int64_t first = s.valueWidth == 1 ? 0 : id % s.valueWidth;
        int64_t last = s.valueWidth == 1 ? s.k : first + 1;
        for (int64_t b = 0; b < s.batches; ++b) {
            for (int64_t k = first; k < last; ++k) {
                acc += static_cast<float>(mat[(b * s.n + col[e]) * s.k + k]) * grad[(b * s.m + row) * s.k + k];
            }
        }
        if (s.reduce == 1 && ptr[row + 1] > ptr[row]) {
            acc /= static_cast<float>(ptr[row + 1] - ptr[row]);
        }
        out[id] = acc;
    }
}
template <class T>
__attribute__((aiv)) __global__ __aicore__ void ValueGradKernel(__gm__ int64_t* ptr, __gm__ int64_t* col, __gm__ T* mat,
                                                                __gm__ float* grad, __gm__ float* out, SpmmShape s)
{
    AscendC::Simt::VF_CALL<ValueGradCompute<T>>(AscendC::Simt::Dim3{512}, ptr, col, mat, grad, out, s);
}

// One status per AIV block, no atomics and no copying of input data to CPU.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) void ValidateCompute(__gm__ int64_t* ptr, __gm__ int64_t* col,
                                                              __gm__ int32_t* status, int64_t m, int64_t n, int64_t e)
{
    if (AscendC::Simt::GetThreadIdx() != 0) {
        return;
    }
    int64_t block = AscendC::Simt::GetBlockIdx(), blocks = AscendC::Simt::GetBlockNum();
    int32_t bad = 0;
    for (int64_t i = block; i <= m; i += blocks) {
        int64_t v = ptr[i];
        if (v < 0 || v > e || (i == 0 && v != 0) || (i == m && v != e) || (i < m && v > ptr[i + 1])) {
            bad = 1;
        }
    }
    for (int64_t i = block; i < e; i += blocks) {
        if (col[i] < 0 || col[i] >= n) {
            bad = 2;
        }
    }
    status[block] = bad;
}
__attribute__((aiv)) __global__ __aicore__ void ValidateKernel(__gm__ int64_t* ptr, __gm__ int64_t* col,
                                                               __gm__ int32_t* status, int64_t m, int64_t n, int64_t e)
{
    AscendC::Simt::VF_CALL<ValidateCompute>(AscendC::Simt::Dim3{32}, ptr, col, status, m, n, e);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) void ConvertCompute(__gm__ int64_t* src, __gm__ int64_t* dst, int64_t size,
                                                             int64_t output, bool toPtr)
{
    int64_t tid = static_cast<int64_t>(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() +
                  AscendC::Simt::GetThreadIdx();
    int64_t stride = static_cast<int64_t>(AscendC::Simt::GetBlockNum()) * AscendC::Simt::GetThreadNum();
    for (int64_t i = tid; i < output; i += stride) {
        int64_t lo = 0, hi = toPtr ? size : size - 1;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            bool next = toPtr ? src[mid] < i : src[mid + 1] <= i;
            if (next) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        dst[i] = lo;
    }
}
__attribute__((aiv)) __global__ __aicore__ void ConvertKernel(__gm__ int64_t* src, __gm__ int64_t* dst, int64_t size,
                                                              int64_t output, bool toPtr)
{
    AscendC::Simt::VF_CALL<ConvertCompute>(AscendC::Simt::Dim3{512}, src, dst, size, output, toPtr);
}

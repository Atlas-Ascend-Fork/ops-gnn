/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include <type_traits>
#include "common/aiv_launch_utils.h"
#include "grid_cluster.h"

union GridFloatBits {
    float value;
    uint32_t bits;
};

__simt_callee__ inline bool GridIsSubnormal(float value)
{
    GridFloatBits data;
    data.value = value;
    uint32_t magnitude = data.bits & 0x7fffffffU;
    return magnitude != 0 && magnitude < 0x00800000U;
}

// Scale by 2^126 without using subnormal operands in floating-point arithmetic.
__simt_callee__ inline float GridScaleTiny(float value)
{
    GridFloatBits data;
    data.value = value;
    uint32_t exponent = (data.bits >> 23) & 0xffU;
    if (exponent == 0) {
        float scaled = static_cast<float>(data.bits & 0x007fffffU) * 0x1p-23f;
        return (data.bits & 0x80000000U) ? -scaled : scaled;
    }
    if (exponent < 129) {
        data.bits += 126U << 23;
    } else if (exponent != 255) {
        data.bits = (data.bits & 0x80000000U) | 0x7f800000U;
    }
    return data.value;
}

template <typename T>
__simt_callee__ inline T GridSubtract(T value, T origin)
{
    if constexpr (std::is_same<T, int32_t>::value) {
        return static_cast<int32_t>(static_cast<uint32_t>(value) - static_cast<uint32_t>(origin));
    }
    return static_cast<T>(value - origin);
}

__simt_callee__ inline float GridDivide(float a, float b)
{
    // The host rejects zero sizes. Keep a device-side guard so neither division
    // is evaluated if an invalid size reaches the kernel.
    if (b == 0.0f) return __builtin_nanf("");
    float q = a / b;
    float residual = AscendC::Simt::Fma(-q, b, a);
    return AscendC::Simt::Fma(residual, 1.0f / b, q);
}

template <typename T, bool Round>
__simt_callee__ inline int64_t GridIndex(T value, T origin, T size)
{
    if constexpr (std::is_same<T, float>::value) {
        if (GridIsSubnormal(size)) {
            // Normalize before subtraction as well: a subnormal difference may
            // otherwise be flushed to zero. Large operands cannot cancel to a
            // subnormal, and are subtracted first to avoid inf - inf on scaling.
            GridFloatBits v, o;
            v.value = value;
            o.value = origin;
            float delta = ((v.bits & 0x7fffffffU) < 0x40800000U &&
                (o.bits & 0x7fffffffU) < 0x40800000U)
                ? GridScaleTiny(value) - GridScaleTiny(origin)
                : GridScaleTiny(GridSubtract(value, origin));
            float q = GridDivide(delta, GridScaleTiny(size));
            // CPU float-to-int64 conversion uses INT64_MIN for nonfinite or
            // out-of-range quotients. Do not rely on a device conversion here.
            if (!(q >= -0x1p63f && q < 0x1p63f)) {
                return static_cast<int64_t>(0x8000000000000000ULL);
            }
            return static_cast<int64_t>(q);
        }
    }
    T delta = GridSubtract(value, origin);
    float q = GridDivide(static_cast<float>(delta), static_cast<float>(size));
    if constexpr (Round) q = static_cast<float>(static_cast<T>(q));
    return static_cast<int64_t>(q);
}

template <typename T, bool Round>
__simt_callee__ inline int64_t GridIndexReciprocal(T value, T origin, T size, float reciprocal)
{
    if constexpr (std::is_same<T, float>::value) {
        if (GridIsSubnormal(size)) return GridIndex<T, Round>(value, origin, size);
    }
    T delta = GridSubtract(value, origin);
    float a = static_cast<float>(delta);
    float q = a * reciprocal;
    q = AscendC::Simt::Fma(AscendC::Simt::Fma(-q, static_cast<float>(size), a), reciprocal, q);
    if constexpr (Round) q = static_cast<float>(static_cast<T>(q));
    return static_cast<int64_t>(q);
}

// Match the CPU int64 tensor's low 64 bits without an overflowing wide product.
__simt_callee__ inline uint64_t GridStrideMultiply(uint64_t prefix, uint64_t radix)
{
    constexpr uint64_t maxValue = ~static_cast<uint64_t>(0);
    if (radix == 0 || prefix <= maxValue / radix) return prefix * radix;

    // Only the low two 32-bit limbs contribute to the result modulo 2^64.
    uint64_t low = static_cast<uint64_t>(static_cast<uint32_t>(prefix)) * static_cast<uint32_t>(radix);
    uint32_t high = static_cast<uint32_t>(low >> 32);
    high += static_cast<uint32_t>(prefix) * static_cast<uint32_t>(radix >> 32);
    high += static_cast<uint32_t>(prefix >> 32) * static_cast<uint32_t>(radix);
    return (static_cast<uint64_t>(high) << 32) | static_cast<uint32_t>(low);
}

template <typename T, bool Round>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) void Grid3Simt(
    __gm__ T* pos, __gm__ T* size, __gm__ T* start, __gm__ T* end,
    __gm__ int64_t* out, int64_t n)
{
    T a = start[0], b = start[1], c = start[2];
    T x = size[0], y = size[1], z = size[2];
    float rx = 1.0f / static_cast<float>(x), ry = 1.0f / static_cast<float>(y), rz = 1.0f / static_cast<float>(z);
    uint64_t sy = static_cast<uint64_t>(GridIndex<T, Round>(end[0], a, x)) + 1;
    uint64_t sz = sy * (static_cast<uint64_t>(GridIndex<T, Round>(end[1], b, y)) + 1);
    // GridKernel routes inputs with more than UINT32_MAX coordinates to
    // GridSimt, so 32-bit offsets are sufficient for this specialized path.
    uint32_t tid = AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    uint32_t step = AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum();
    for (uint32_t i = tid; i < static_cast<uint32_t>(n); i += step) {
        uint64_t v0 = static_cast<uint64_t>(GridIndexReciprocal<T, Round>(pos[3*i], a, x, rx));
        uint64_t v1 = static_cast<uint64_t>(GridIndexReciprocal<T, Round>(pos[3*i+1], b, y, ry));
        uint64_t v2 = static_cast<uint64_t>(GridIndexReciprocal<T, Round>(pos[3*i+2], c, z, rz));
        out[i] = static_cast<int64_t>(v0 + sy*v1 + sz*v2);
    }
}

template <typename T, bool Round, int D, int Active = D, bool PowerOfTwo = false>
__simt_callee__ inline void GridWideBody(
    __gm__ T* pos, __gm__ T* size, __gm__ T* start, __gm__ T* end,
    __gm__ int64_t* out, int64_t n)
{
    constexpr int W = Active == 16 ? (D == 16 ? 2 : 1) : 4;
    constexpr int K = Active / W;
    int lane = AscendC::Simt::GetThreadIdx() % W;
    T origins[K], sizes[K];
    float reciprocals[K];
    uint64_t strides[K];
    uint64_t prefix = 1;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        origins[k] = start[k*W+lane];
        sizes[k] = size[k*W+lane];
        reciprocals[k] = 1.0f / static_cast<float>(sizes[k]);
#pragma unroll
        for (int j = 0; j < W; ++j) {
            if (j == lane) strides[k] = prefix;
            uint64_t radix = static_cast<uint64_t>(GridIndex<T, Round>(end[k*W+j], start[k*W+j], size[k*W+j])) + 1;
            prefix = GridStrideMultiply(prefix, radix);
        }
    }
    uint32_t tid = (AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx()) / W;
    uint32_t step = AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum() / W;
    for (uint32_t i = tid; i < static_cast<uint32_t>(n); i += step) {
        uint64_t value = 0;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            if constexpr (PowerOfTwo) {
                // Integer inputs divided by a positive power of two: the
                // float32 quotient is exact after CPU-compatible promotion.
                T delta = GridSubtract(pos[i*D+k*W+lane], origins[k]);
                int64_t index = static_cast<int64_t>(static_cast<float>(delta) * reciprocals[k]);
                value += static_cast<uint64_t>(index) * strides[k];
            } else {
                value += static_cast<uint64_t>(GridIndexReciprocal<T, Round>(pos[i*D+k*W+lane], origins[k], sizes[k], reciprocals[k])) * strides[k];
            }
        }
        if constexpr (W == 4) value += AscendC::Simt::WarpShflDownSync(value, 2, W);
        if constexpr (W > 1) value += AscendC::Simt::WarpShflDownSync(value, 1, W);
        if (lane == 0) out[i] = static_cast<int64_t>(value);
    }
}

template <typename T, bool Round, int D>
__simt_callee__ inline void GridWideDispatch(
    __gm__ T* pos, __gm__ T* size, __gm__ T* start, __gm__ T* end,
    __gm__ int64_t* out, int64_t n)
{
    if constexpr (D == 16 && std::is_integral<T>::value) {
        bool exactPower = true;
        for (int j = 0; j < D; ++j) {
            uint32_t edge = static_cast<uint32_t>(size[j]);
            exactPower = exactPower && size[j] > 0 && (edge & (edge - 1)) == 0;
        }
        if (exactPower) {
            GridWideBody<T, Round, D, D, true>(pos, size, start, end, out, n);
            return;
        }
    }
    if constexpr (D > 16) {
        uint64_t prefix = 1;
        for (int j = 0; j < 16; ++j) {
            uint64_t radix = static_cast<uint64_t>(GridIndex<T, Round>(end[j], start[j], size[j])) + 1;
            prefix = GridStrideMultiply(prefix, radix);
        }
        // Modulo arithmetic makes all later contributions zero, for any input
        // coordinates, once the prefix product is zero. This is not relabeling.
        if (prefix == 0) {
            GridWideBody<T, Round, D, 16>(pos, size, start, end, out, n);
            return;
        }
    }
    GridWideBody<T, Round, D>(pos, size, start, end, out, n);
}

template <typename T, bool Round, int D>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void GridWideSimt(
    __gm__ T* pos, __gm__ T* size, __gm__ T* start, __gm__ T* end,
    __gm__ int64_t* out, int64_t n)
{
    GridWideDispatch<T, Round, D>(pos, size, start, end, out, n);
}

template <typename T, bool Round, int D>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) void GridWide512Simt(
    __gm__ T* pos, __gm__ T* size, __gm__ T* start, __gm__ T* end,
    __gm__ int64_t* out, int64_t n)
{
    GridWideDispatch<T, Round, D>(pos, size, start, end, out, n);
}

template <typename T, bool Round = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) void GridSimt(
    __gm__ T* pos, __gm__ T* size, __gm__ T* start, __gm__ T* end,
    __gm__ int64_t* out, int64_t n, int64_t dims)
{
    int64_t tid = AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum()
        + AscendC::Simt::GetThreadIdx();
    int64_t step = AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum();
    for (int64_t i = tid; i < n; i += step) {
        uint64_t stride = 1;
        uint64_t result = 0;
        for (int64_t d = 0; d < dims; ++d) {
            int64_t index = GridIndex<T, Round>(pos[i * dims + d], start[d], size[d]);
            int64_t count = GridIndex<T, Round>(end[d], start[d], size[d]);
            result += static_cast<uint64_t>(index) * stride;
            // Unsigned arithmetic gives defined modulo-2^64 overflow, matching int64 tensors.
            stride *= static_cast<uint64_t>(count) + 1;
        }
        out[i] = static_cast<int64_t>(result);
    }
}

template <typename T, bool Round = false>
__attribute__((aiv)) __global__ __aicore__ void GridKernel(
    __gm__ T* pos, __gm__ T* size, __gm__ T* start, __gm__ T* end,
    __gm__ int64_t* out, int64_t n, int64_t dims)
{
    if (n > 0xffffffffULL / static_cast<uint64_t>(dims)) {
        AscendC::Simt::VF_CALL<GridSimt<T, Round>>(AscendC::Simt::Dim3{256}, pos, size, start, end, out, n, dims);
        return;
    }
    if (dims == 3) {
        AscendC::Simt::VF_CALL<Grid3Simt<T, Round>>(AscendC::Simt::Dim3{512}, pos, size, start, end, out, n);
    } else if (dims == 16) {
        AscendC::Simt::VF_CALL<GridWide512Simt<T, Round, 16>>(AscendC::Simt::Dim3{512}, pos, size, start, end, out, n);
    } else if (dims == 32) {
        if constexpr (Round) {
            AscendC::Simt::VF_CALL<GridWide512Simt<T, Round, 32>>(AscendC::Simt::Dim3{512}, pos, size, start, end, out, n);
        } else {
            AscendC::Simt::VF_CALL<GridWideSimt<T, Round, 32>>(AscendC::Simt::Dim3{256}, pos, size, start, end, out, n);
        }
    } else if (dims == 64) {
        AscendC::Simt::VF_CALL<GridWideSimt<T, Round, 64>>(AscendC::Simt::Dim3{256}, pos, size, start, end, out, n);
    } else {
        AscendC::Simt::VF_CALL<GridSimt<T, Round>>(AscendC::Simt::Dim3{256}, pos, size, start, end, out, n, dims);
    }
}

void GridCluster(void* pos, void* size, void* start, void* end, int64_t* out,
    int64_t n, int64_t dims, int dtype, aclrtStream stream)
{
    uint32_t cores = ops_gnn::sparse::ResolveAivCoreNum((n + 255) / 256);
#define GRID_LAUNCH(T, R) GridKernel<T, R><<<cores, nullptr, stream>>>( \
    static_cast<T*>(pos), static_cast<T*>(size), static_cast<T*>(start), static_cast<T*>(end), out, n, dims)
    switch (dtype) {
        case 0: GRID_LAUNCH(uint8_t, false); break;
        case 1: GRID_LAUNCH(int8_t, false); break;
        case 2: GRID_LAUNCH(int16_t, false); break;
        case 3: GRID_LAUNCH(int32_t, false); break;
        case 5: GRID_LAUNCH(half, true); break;
        case 6: GRID_LAUNCH(float, false); break;
        case 15: GRID_LAUNCH(bfloat16_t, true); break;
    }
#undef GRID_LAUNCH
}

/*
 * Copyright (c) 2026 Starlink_. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * Ascend C (SIMT) kernel for the torch_scatter-compatible segment_csr
 * family (sum/add/mean/min/max) on Ascend 950 (arch35).
 *
 * Structure: one reduction engine templated on the element kind, with the
 * kind-specific surface reduced to thin SegTraits specializations on top of
 * the shared FloatTraitsBase / IntTraitsBase families. Everything the scalar
 * and vector paths have in common -- grid-stride walking, work-item decode,
 * segment bounds, empty-segment writeback, row packet loads -- lives in one
 * shared helper each; the per-path code only contains the reduction logic.
 *
 * Determinism strategy (hard requirement of the task book):
 *   Every output cell (batch e, segment s, channel k) is owned by exactly
 *   ONE thread (grid-stride enumeration of work items), and each owner
 *   reduces its segment in a FIXED row order. The 4-way unrolled vector
 *   path splits rows by residue class into four chains and merges them
 *   in a fixed order; min/max merges break ties by row index, which
 *   reproduces the sequential first-occurrence winner of the CPU
 *   reference. No atomics, no cross-thread races: identical inputs give
 *   bit-identical outputs.
 *
 * Memory strategy (see DESIGN.md):
 *   - Benchmark orientation is E1 == 1, M == rows, K == C with contiguous
 *     (rows, C) src: consecutive threads own consecutive channel lanes of
 *     one segment, so each warp issues fully coalesced 128-byte lines.
 *   - When K % VEC_LANES == 0 (4 for fp32/int32, 8 for fp16, 2 for int64)
 *     the vector path performs one aligned 16-byte load per row via the
 *     SIMT vector-load intrinsics (asc_ldcg + float4/int4/uint4) and
 *     keeps four independent accumulator chains for load-level ILP.
 *   - fp16/bf16 are widened to fp32 for all arithmetic (CPU-reference
 *     semantics: fp32 accumulation, one final RNE rounding) through the
 *     compiler intrinsics __half2float / __float2half_rn /
 *     __bfloat162float / __float2bfloat16_rn.
 */

#include "segment_csr.h"

#include "kernel_operator.h"
#include "simt_api/device_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "tiling/platform/platform_ascendc.h"
#include <limits>

namespace ops_gnn {
namespace {

constexpr uint32_t SEG_THREADS_PER_BLOCK = 256;

union FloatBits {
    float f;
    uint32_t u;
};

// ------------------------------------------------------------ fp16 codec
// Bit-level fp16 <-> fp32: avoids `half`-typed GM loads, which take a slow
// microcode path in SIMT kernels (plain uint16 loads are regular accesses).
union SimtFloatBits {
    float f;
    uint32_t u;
};

__simt_callee__ inline float HalfBitsToF32(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x03FFu;
    if (exp == 0) {
        const float magnitude = static_cast<float>(man) * 5.9604644775390625e-08f;
        return sign == 0 ? magnitude : -magnitude;
    }
    SimtFloatBits v;
    v.u = sign | ((exp == 31u ? 255u : exp + 112u) << 23) | (man << 13);
    return v.f;
}

__simt_callee__ inline uint16_t F32ToHalfBits(float x)
{
    SimtFloatBits v;
    v.f = x;
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    const uint32_t rest = v.u & 0x7FFFFFFFu;
    if (rest >= 0x7F800000u) {
        return static_cast<uint16_t>(sign | 0x7C00u | ((rest & 0x7FFFFFu) ? 0x0200u : 0u));
    }
    if (rest >= 0x477FF000u) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (rest < 0x33000000u) {
        return static_cast<uint16_t>(sign);
    }
    if (rest < 0x38800000u) {
        // A subnormal half retains the implicit fp32 leading bit. Round
        // its discarded bits to nearest, resolving exact ties to even.
        const uint32_t shift = 126u - (rest >> 23);
        const uint32_t mant = (rest & 0x007FFFFFu) | 0x00800000u;
        uint32_t sig = mant >> shift;
        const uint32_t remainder = mant & ((1u << shift) - 1u);
        const uint32_t midpoint = 1u << (shift - 1u);
        sig += (remainder > midpoint || (remainder == midpoint && (sig & 1u) != 0u)) ? 1u : 0u;
        return static_cast<uint16_t>(sign | sig);
    }
    const uint32_t mant = rest & 0x007FFFFFu;
    uint32_t h = (((rest >> 23) - 127u + 15u) << 10) | (mant >> 13);
    h += ((((mant >> 12) & 1u) != 0u) && (((mant & 0xFFFu) != 0u) || ((h & 1u) != 0u))) ? 1u : 0u;
    return static_cast<uint16_t>(sign | h);
}

// ------------------------------------------------------------ element traits
// The kind families own everything the seven dtypes share; each SegTraits
// specialization only adds storage type, packet type, vector width and the
// load/store codec.
namespace segdetail {

// fp32-accumulating kinds: extremum sentinels are +/-Inf in fp32.
struct FloatTraitsBase {
    static constexpr bool IS_INT = false;
    __simt_callee__ static inline float MinSentinel()
    {
        FloatBits v;
        v.u = 0x7F800000u;
        return v.f;
    }
    __simt_callee__ static inline float MaxSentinel()
    {
        FloatBits v;
        v.u = 0xFF800000u;
        return v.f;
    }
};

// integer kinds: int64 accumulation, wrap sentinels for min/max.
struct IntTraitsBase {
    using Acc = int64_t;
    static constexpr bool IS_INT = true;
    __simt_callee__ static inline int64_t MinSentinel() { return std::numeric_limits<int64_t>::max(); }
    __simt_callee__ static inline int64_t MaxSentinel() { return std::numeric_limits<int64_t>::min(); }
};

} // namespace segdetail

template <int KIND>
struct SegTraits;

template <>
struct SegTraits<KIND_F32> : segdetail::FloatTraitsBase {
    using Storage = float;
    using Acc = float;
    using Packet = float4;
    static constexpr int VEC_LANES = 4;
    __simt_callee__ static inline Acc Load(float v) { return v; }
    __simt_callee__ static inline float Store(float a) { return a; }
    __simt_callee__ static inline float Zero() { return 0.0f; }
};

template <>
struct SegTraits<KIND_F16> : segdetail::FloatTraitsBase {
    using Storage = uint16_t;
    using Acc = float;
    using Packet = uint4;
    static constexpr int VEC_LANES = 8;
    __simt_callee__ static inline Acc Load(uint16_t v) { return HalfBitsToF32(v); }
    __simt_callee__ static inline uint16_t Store(float a) { return F32ToHalfBits(a); }
    __simt_callee__ static inline uint16_t Zero() { return 0; }
};

template <>
struct SegTraits<KIND_BF16> : segdetail::FloatTraitsBase {
    using Storage = bfloat16_t;
    using Acc = float;
    using Packet = uint4;
    static constexpr int VEC_LANES = 0; // scalar path only (functional tier)
    __simt_callee__ static inline Acc Load(bfloat16_t v) { return __bfloat162float(v); }
    __simt_callee__ static inline bfloat16_t Store(float a) { return __float2bfloat16_rn(a); }
    __simt_callee__ static inline bfloat16_t Zero() { return __float2bfloat16_rn(0.0f); }
};

template <>
struct SegTraits<KIND_I8> : segdetail::IntTraitsBase {
    using Storage = int8_t;
    using Packet = uint4;
    static constexpr int VEC_LANES = 0;
    __simt_callee__ static inline Acc Load(int8_t v) { return static_cast<int64_t>(v); }
    __simt_callee__ static inline int8_t Store(int64_t a) { return static_cast<int8_t>(a); }
    __simt_callee__ static inline int8_t Zero() { return 0; }
};

template <>
struct SegTraits<KIND_U8> : segdetail::IntTraitsBase {
    using Storage = uint8_t;
    using Packet = uint4;
    static constexpr int VEC_LANES = 0;
    __simt_callee__ static inline Acc Load(uint8_t v) { return static_cast<int64_t>(v); }
    __simt_callee__ static inline uint8_t Store(int64_t a) { return static_cast<uint8_t>(a); }
    __simt_callee__ static inline uint8_t Zero() { return 0; }
};

template <>
struct SegTraits<KIND_I32> : segdetail::IntTraitsBase {
    using Storage = int32_t;
    using Packet = int4;
    static constexpr int VEC_LANES = 4;
    __simt_callee__ static inline Acc Load(int32_t v) { return static_cast<int64_t>(v); }
    __simt_callee__ static inline int32_t Store(int64_t a) { return static_cast<int32_t>(a); }
    __simt_callee__ static inline int32_t Zero() { return 0; }
};

template <>
struct SegTraits<KIND_I64> : segdetail::IntTraitsBase {
    using Storage = int64_t;
    using Packet = uint4;
    static constexpr int VEC_LANES = 2;
    __simt_callee__ static inline Acc Load(int64_t v) { return v; }
    __simt_callee__ static inline int64_t Store(int64_t a) { return a; }
    __simt_callee__ static inline int64_t Zero() { return 0; }
};

template <int KIND>
using SegStoreT = typename SegTraits<KIND>::Storage;
template <int KIND>
using SegAccT = typename SegTraits<KIND>::Acc;
template <int KIND>
using SegPacketT = typename SegTraits<KIND>::Packet;

// One aligned 16-byte streaming load (bypasses L1 for the streaming src).
template <int KIND>
__simt_callee__ inline SegPacketT<KIND> LoadPacket(__gm__ const SegPacketT<KIND>* addr)
{
    return asc_ldcg(const_cast<__gm__ SegPacketT<KIND>*>(addr));
}

// Vector lane unpack: lane index within one 16-byte packet.
template <int KIND>
__simt_callee__ inline SegAccT<KIND> VecLane(const SegPacketT<KIND>& v, int lane);

template <>
__simt_callee__ inline float VecLane<KIND_F32>(const float4& v, int lane)
{
    const float arr[4] = {v.x, v.y, v.z, v.w};
    return arr[lane];
}

template <>
__simt_callee__ inline float VecLane<KIND_F16>(const uint4& v, int lane)
{
    const uint32_t comp[4] = {v.x, v.y, v.z, v.w};
    const uint32_t c = comp[lane >> 1];
    const unsigned short bits = static_cast<unsigned short>((lane & 1) ? (c >> 16) : (c & 0xFFFFu));
    return HalfBitsToF32(bits);
}

template <>
__simt_callee__ inline int64_t VecLane<KIND_I32>(const int4& v, int lane)
{
    const int32_t arr[4] = {v.x, v.y, v.z, v.w};
    return static_cast<int64_t>(arr[lane]);
}

template <>
__simt_callee__ inline int64_t VecLane<KIND_I64>(const uint4& v, int lane)
{
    const uint32_t lo = (lane == 0) ? v.x : v.z;
    const uint32_t hi = (lane == 0) ? v.y : v.w;
    return static_cast<int64_t>(static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32));
}

// ------------------------------------------------------- final value writers
// CPU-reference mean for finalized sums: round both the accumulated sum and
// the row count to the source dtype BEFORE dividing (wrap-first integers,
// trunc toward zero; fp goes through the storage rounding).
template <int KIND>
__simt_callee__ inline SegStoreT<KIND> MeanDivideRounded(SegAccT<KIND> acc, int64_t rows)
{
    using Traits = SegTraits<KIND>;
    if constexpr (Traits::IS_INT) {
        const int64_t wrapped = static_cast<int64_t>(Traits::Store(acc));
        const int64_t count = Traits::Load(Traits::Store(static_cast<SegAccT<KIND>>(rows)));
        // A wrapped zero count makes CPU division undefined. Return
        // zero here instead of raising a device division exception.
        return count == 0 ? Traits::Zero() : Traits::Store(wrapped / count);
    } else {
        const float sum = Traits::Load(Traits::Store(acc));
        const float count = Traits::Load(Traits::Store(static_cast<float>(rows)));
        return Traits::Store(sum / count);
    }
}

template <int KIND>
__simt_callee__ inline void WriteReduced(
    __gm__ SegStoreT<KIND>* out, uint64_t idx, SegAccT<KIND> acc, int64_t rows, int isMean)
{
    if (isMean != 0) {
        out[idx] = MeanDivideRounded<KIND>(acc, rows);
    } else {
        out[idx] = SegTraits<KIND>::Store(acc);
    }
}

// ------------------------------------------------------- shared work decode
// Grid-stride coordinates shared by every SIMT loop body.
struct GridStride {
    uint64_t gtid;
    uint64_t gstride;
};

__simt_callee__ inline GridStride GridCoords()
{
    GridStride gs;
    gs.gtid = AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    gs.gstride = AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum();
    return gs;
}

// Segment bounds with the same clamping as the CPU reference.
struct SegRange {
    int64_t begin;
    int64_t end;
};

__simt_callee__ inline SegRange SegmentBounds(__gm__ const int64_t* indptr, uint64_t indptrBase, uint32_t s, uint64_t m)
{
    int64_t b = indptr[indptrBase + s];
    int64_t e = indptr[indptrBase + s + 1];
    if (b < 0) {
        b = 0;
    }
    if (e > static_cast<int64_t>(m)) {
        e = static_cast<int64_t>(m);
    }
    SegRange r;
    r.begin = b;
    r.end = e;
    return r;
}

// Everything both reduce paths need for one enumerated work item: the
// (batch, segment, lane) triple, the owning batch's indptr base and the
// clamped segment bounds. Host guarantees total < 2^32, so the divisions
// stay in 32-bit.
struct WorkItem {
    uint64_t batch;
    uint32_t segment;
    uint32_t lane;
    uint64_t indptrBase;
    SegRange seg;
};

__simt_callee__ inline WorkItem DecodeWorkItem(
    __gm__ const int64_t* indptr, uint32_t w, uint32_t nSeg, uint32_t laneCount, uint64_t indptrStride, uint64_t m)
{
    const uint32_t perBatch = nSeg * laneCount;
    WorkItem it;
    it.batch = w / perBatch;
    const uint32_t rem = w - static_cast<uint32_t>(it.batch) * perBatch;
    it.segment = rem / laneCount;
    it.lane = rem - it.segment * laneCount;
    it.indptrBase = indptrStride == 0 ? 0 : it.batch * indptrStride;
    it.seg = SegmentBounds(indptr, it.indptrBase, it.segment, m);
    return it;
}

// Empty-segment writeback shared by the scalar and vector paths: zero fill
// plus the out-of-range arg position (== m) of the CPU reference.
template <int KIND>
__simt_callee__ inline void FillEmptySegment(__gm__ SegStoreT<KIND>* out, __gm__ int64_t* argOut, uint64_t idx, uint64_t m)
{
    out[idx] = SegTraits<KIND>::Zero();
    if (argOut != nullptr) {
        argOut[idx] = static_cast<int64_t>(m);
    }
}

template <int KIND>
__simt_callee__ inline void ReduceScalarExtremum(
    __gm__ const SegStoreT<KIND>* base, uint32_t k, SegRange seg, int reduce,
    __gm__ SegStoreT<KIND>* out, __gm__ int64_t* argOut, uint64_t outIdx)
{
    using Traits = SegTraits<KIND>;
    using Acc = SegAccT<KIND>;
    // Sequential scan: strict comparison keeps the FIRST extremum.
    Acc best = Traits::Load(base[0]);
    int64_t bestRow = seg.begin;
    for (int64_t j = seg.begin + 1; j < seg.end; ++j) {
        const Acc v = Traits::Load(base[static_cast<uint64_t>(j - seg.begin) * k]);
        if (reduce == SEGMENT_CSR_MIN) {
            if (v < best) {
                best = v;
                bestRow = j;
            }
        } else if (v > best) {
            best = v;
            bestRow = j;
        }
    }
    out[outIdx] = Traits::Store(best);
    if (argOut != nullptr) {
        argOut[outIdx] = bestRow;
    }
}

// --------------------------------------------------------------- scalar path
template <int KIND>
__simt_callee__ inline void RunScalar(
    __gm__ const SegStoreT<KIND>* src, __gm__ const int64_t* indptr, __gm__ SegStoreT<KIND>* out,
    __gm__ int64_t* argOut, int reduce, int isMean, uint64_t m, uint32_t nSeg, uint32_t k, uint64_t indptrStride,
    uint64_t total)
{
    using Traits = SegTraits<KIND>;
    using Store = SegStoreT<KIND>;
    using Acc = SegAccT<KIND>;
    const GridStride gs = GridCoords();

    for (uint64_t w = gs.gtid; w < total; w += gs.gstride) {
        const WorkItem it = DecodeWorkItem(indptr, static_cast<uint32_t>(w), nSeg, k, indptrStride, m);
        const uint64_t outIdx = (it.batch * nSeg + it.segment) * k + it.lane;

        if (it.seg.begin >= it.seg.end) {
            FillEmptySegment<KIND>(out, argOut, outIdx, m);
            continue;
        }

        const __gm__ Store* base = src + (it.batch * m + static_cast<uint64_t>(it.seg.begin)) * k + it.lane;

        if (reduce == SEGMENT_CSR_SUM) {
            Acc acc = static_cast<Acc>(0);
            for (int64_t j = it.seg.begin; j < it.seg.end; ++j) {
                acc += Traits::Load(base[static_cast<uint64_t>(j - it.seg.begin) * k]);
            }
            WriteReduced<KIND>(out, outIdx, acc, it.seg.end - it.seg.begin, isMean);
        } else {
            ReduceScalarExtremum<KIND>(base, k, it.seg, reduce, out, argOut, outIdx);
        }
    }
}

// ----------------------------------------------------------------- vec path
// One aligned packet load for row (j - seg.begin) of the owning work item.
template <int KIND>
__simt_callee__ inline SegPacketT<KIND> LoadRowPacket(
    __gm__ const SegPacketT<KIND>* vecBase, uint64_t firstRowVec, uint64_t rowStepVec, uint64_t rowOffset)
{
    return LoadPacket<KIND>(vecBase + (firstRowVec + rowOffset * rowStepVec));
}

template <int KIND>
__simt_callee__ inline void InitializeSumChains(
    SegAccT<KIND>* a0, SegAccT<KIND>* a1, SegAccT<KIND>* a2, SegAccT<KIND>* a3)
{
    using Acc = SegAccT<KIND>;
    constexpr int VE = SegTraits<KIND>::VEC_LANES;
    for (int l = 0; l < VE; ++l) {
        a0[l] = static_cast<Acc>(0);
        a1[l] = static_cast<Acc>(0);
        a2[l] = static_cast<Acc>(0);
        a3[l] = static_cast<Acc>(0);
    }
}

template <int KIND>
__simt_callee__ inline void ReducePacketSum(
    __gm__ const SegPacketT<KIND>* vecBase, uint64_t firstRowVec, uint64_t rowStepVec,
    SegRange seg, __gm__ SegStoreT<KIND>* out, uint64_t outBase, int isMean)
{
    using Acc = SegAccT<KIND>;
    using Packet = SegPacketT<KIND>;
    constexpr int VE = SegTraits<KIND>::VEC_LANES;
    const int64_t rows = seg.end - seg.begin;
    Acc a0[VE];
    Acc a1[VE];
    Acc a2[VE];
    Acc a3[VE];
    InitializeSumChains<KIND>(a0, a1, a2, a3);
    const uint64_t full = static_cast<uint64_t>(rows) >> 2;
    for (uint64_t i = 0; i < full; ++i) {
        const Packet v0 = LoadRowPacket<KIND>(vecBase, firstRowVec, rowStepVec, i * 4 + 0);
        const Packet v1 = LoadRowPacket<KIND>(vecBase, firstRowVec, rowStepVec, i * 4 + 1);
        const Packet v2 = LoadRowPacket<KIND>(vecBase, firstRowVec, rowStepVec, i * 4 + 2);
        const Packet v3 = LoadRowPacket<KIND>(vecBase, firstRowVec, rowStepVec, i * 4 + 3);
        for (int l = 0; l < VE; ++l) {
            a0[l] += VecLane<KIND>(v0, l);
            a1[l] += VecLane<KIND>(v1, l);
            a2[l] += VecLane<KIND>(v2, l);
            a3[l] += VecLane<KIND>(v3, l);
        }
    }
    // Tail rows fold into chain 0 in row order, then a fixed
    // left-to-right merge: deterministic for every input.
    const int64_t tail = seg.begin + static_cast<int64_t>(full * 4);
    for (int64_t j = tail; j < seg.end; ++j) {
        const Packet v = LoadRowPacket<KIND>(vecBase, firstRowVec, rowStepVec, static_cast<uint64_t>(j - seg.begin));
        for (int l = 0; l < VE; ++l) {
            a0[l] += VecLane<KIND>(v, l);
        }
    }
    for (int l = 0; l < VE; ++l) {
        const Acc acc = (a0[l] + a1[l]) + (a2[l] + a3[l]);
        WriteReduced<KIND>(out, outBase + l, acc, rows, isMean);
    }
}

template <int KIND>
struct PacketExtremaState {
    SegAccT<KIND> b0[SegTraits<KIND>::VEC_LANES];
    SegAccT<KIND> b1[SegTraits<KIND>::VEC_LANES];
    SegAccT<KIND> b2[SegTraits<KIND>::VEC_LANES];
    SegAccT<KIND> b3[SegTraits<KIND>::VEC_LANES];
    int64_t r0[SegTraits<KIND>::VEC_LANES];
    int64_t r1[SegTraits<KIND>::VEC_LANES];
    int64_t r2[SegTraits<KIND>::VEC_LANES];
    int64_t r3[SegTraits<KIND>::VEC_LANES];
};

template <int KIND>
__simt_callee__ inline void InitializePacketExtrema(PacketExtremaState<KIND>& state, SegRange seg, bool isMin)
{
    using Traits = SegTraits<KIND>;
    using Acc = SegAccT<KIND>;
    constexpr int VE = Traits::VEC_LANES;
    const Acc sentinel = isMin ? Traits::MinSentinel() : Traits::MaxSentinel();
    for (int l = 0; l < VE; ++l) {
        state.b0[l] = sentinel;
        state.b1[l] = sentinel;
        state.b2[l] = sentinel;
        state.b3[l] = sentinel;
        state.r0[l] = seg.begin;
        state.r1[l] = seg.begin;
        state.r2[l] = seg.begin;
        state.r3[l] = seg.begin;
    }
}

template <int KIND>
__simt_callee__ inline void AccumulatePacketExtrema(
    PacketExtremaState<KIND>& state, __gm__ const SegPacketT<KIND>* vecBase, uint64_t firstRowVec,
    uint64_t rowStepVec, SegRange seg, bool isMin)
{
    using Acc = SegAccT<KIND>;
    using Packet = SegPacketT<KIND>;
    constexpr int VE = SegTraits<KIND>::VEC_LANES;
    const int64_t rows = seg.end - seg.begin;
    const uint64_t full = static_cast<uint64_t>(rows) >> 2;
    for (uint64_t i = 0; i < full; ++i) {
        const Packet v0 = LoadRowPacket<KIND>(vecBase, firstRowVec, rowStepVec, i * 4 + 0);
        const Packet v1 = LoadRowPacket<KIND>(vecBase, firstRowVec, rowStepVec, i * 4 + 1);
        const Packet v2 = LoadRowPacket<KIND>(vecBase, firstRowVec, rowStepVec, i * 4 + 2);
        const Packet v3 = LoadRowPacket<KIND>(vecBase, firstRowVec, rowStepVec, i * 4 + 3);
        const int64_t rowBase = seg.begin + static_cast<int64_t>(i * 4);
        for (int l = 0; l < VE; ++l) {
            const Acc x0 = VecLane<KIND>(v0, l);
            const Acc x1 = VecLane<KIND>(v1, l);
            const Acc x2 = VecLane<KIND>(v2, l);
            const Acc x3 = VecLane<KIND>(v3, l);
            if (isMin ? (x0 < state.b0[l]) : (x0 > state.b0[l])) {
                state.b0[l] = x0;
                state.r0[l] = rowBase;
            }
            if (isMin ? (x1 < state.b1[l]) : (x1 > state.b1[l])) {
                state.b1[l] = x1;
                state.r1[l] = rowBase + 1;
            }
            if (isMin ? (x2 < state.b2[l]) : (x2 > state.b2[l])) {
                state.b2[l] = x2;
                state.r2[l] = rowBase + 2;
            }
            if (isMin ? (x3 < state.b3[l]) : (x3 > state.b3[l])) {
                state.b3[l] = x3;
                state.r3[l] = rowBase + 3;
            }
        }
    }
}

template <int KIND>
__simt_callee__ inline void MergePacketExtrema(PacketExtremaState<KIND>& state, bool isMin)
{
    using Acc = SegAccT<KIND>;
    constexpr int VE = SegTraits<KIND>::VEC_LANES;
    for (int l = 0; l < VE; ++l) {
        const Acc cand[3] = {state.b1[l], state.b2[l], state.b3[l]};
        const int64_t candRow[3] = {state.r1[l], state.r2[l], state.r3[l]};
        Acc vBest = state.b0[l];
        int64_t vRow = state.r0[l];
        for (int c = 0; c < 3; ++c) {
            const bool better = isMin ? (cand[c] < vBest || (cand[c] == vBest && candRow[c] < vRow)) :
                                        (cand[c] > vBest || (cand[c] == vBest && candRow[c] < vRow));
            if (better) {
                vBest = cand[c];
                vRow = candRow[c];
            }
        }
        state.b0[l] = vBest;
        state.r0[l] = vRow;
    }
}

template <int KIND>
__simt_callee__ inline void FinishPacketExtrema(
    PacketExtremaState<KIND>& state, __gm__ const SegPacketT<KIND>* vecBase, uint64_t firstRowVec,
    uint64_t rowStepVec, SegRange seg, bool isMin, __gm__ SegStoreT<KIND>* out,
    __gm__ int64_t* argOut, uint64_t outBase)
{
    using Traits = SegTraits<KIND>;
    using Acc = SegAccT<KIND>;
    using Packet = SegPacketT<KIND>;
    constexpr int VE = Traits::VEC_LANES;
    const int64_t tail = seg.begin + ((seg.end - seg.begin) / 4) * 4;
    for (int64_t j = tail; j < seg.end; ++j) {
        const Packet v = LoadRowPacket<KIND>(vecBase, firstRowVec, rowStepVec, static_cast<uint64_t>(j - seg.begin));
        for (int l = 0; l < VE; ++l) {
            const Acc x = VecLane<KIND>(v, l);
            if (isMin ? (x < state.b0[l]) : (x > state.b0[l])) {
                state.b0[l] = x;
                state.r0[l] = j;
            }
        }
    }
    for (int l = 0; l < VE; ++l) {
        out[outBase + l] = Traits::Store(state.b0[l]);
        if (argOut != nullptr) {
            argOut[outBase + l] = state.r0[l];
        }
    }
}

template <int KIND>
__simt_callee__ inline void RunVec(
    __gm__ const SegStoreT<KIND>* src, __gm__ const int64_t* indptr, __gm__ SegStoreT<KIND>* out,
    __gm__ int64_t* argOut, int reduce, int isMean, uint64_t m, uint32_t nSeg, uint32_t k, uint64_t indptrStride,
    uint64_t total)
{
    constexpr int VE = SegTraits<KIND>::VEC_LANES;
    const uint32_t rowStepVec = k / VE;
    const GridStride gs = GridCoords();
    __gm__ const SegPacketT<KIND>* vecBase = reinterpret_cast<__gm__ const SegPacketT<KIND>*>(src);
    for (uint64_t w = gs.gtid; w < total; w += gs.gstride) {
        const WorkItem it = DecodeWorkItem(indptr, static_cast<uint32_t>(w), nSeg, rowStepVec, indptrStride, m);
        const uint64_t outBase = (it.batch * nSeg + it.segment) * k + static_cast<uint64_t>(it.lane) * VE;
        if (it.seg.begin >= it.seg.end) {
            for (int lane = 0; lane < VE; ++lane) {
                FillEmptySegment<KIND>(out, argOut, outBase + lane, m);
            }
            continue;
        }
        const uint64_t firstRowVec = (it.batch * m + static_cast<uint64_t>(it.seg.begin)) * rowStepVec + it.lane;
        if (reduce == SEGMENT_CSR_SUM) {
            ReducePacketSum<KIND>(vecBase, firstRowVec, rowStepVec, it.seg, out, outBase, isMean);
        } else {
            const bool isMin = reduce == SEGMENT_CSR_MIN;
            PacketExtremaState<KIND> state;
            InitializePacketExtrema<KIND>(state, it.seg, isMin);
            AccumulatePacketExtrema<KIND>(state, vecBase, firstRowVec, rowStepVec, it.seg, isMin);
            MergePacketExtrema<KIND>(state, isMin);
            FinishPacketExtrema<KIND>(state, vecBase, firstRowVec, rowStepVec, it.seg, isMin, out, argOut, outBase);
        }
    }
}

// ------------------------------------------------------------- SIMT wrappers
template <int KIND, bool VEC>
__simt_vf__ __aicore__ LAUNCH_BOUND(SEG_THREADS_PER_BLOCK) void SegmentCsrSimt(
    __gm__ const void* srcV, __gm__ const int64_t* indptr, __gm__ void* outV, __gm__ int64_t* argOut, int reduce,
    int isMean, uint64_t m, uint32_t nSeg, uint32_t k, uint64_t indptrStride, uint64_t total)
{
    using Store = SegStoreT<KIND>;
    __gm__ const Store* src = reinterpret_cast<__gm__ const Store*>(srcV);
    __gm__ Store* out = reinterpret_cast<__gm__ Store*>(outV);
    if constexpr (VEC) {
        RunVec<KIND>(src, indptr, out, argOut, reduce, isMean, m, nSeg, k, indptrStride, total);
    } else {
        RunScalar<KIND>(src, indptr, out, argOut, reduce, isMean, m, nSeg, k, indptrStride, total);
    }
}

template <int KIND, bool VEC>
__attribute__((aiv)) __global__ __aicore__ void SegmentCsrKernelEntry(
    __gm__ const void* src, __gm__ const int64_t* indptr, __gm__ void* out, __gm__ int64_t* argOut, int reduce,
    int isMean, uint64_t m, uint32_t nSeg, uint32_t k, uint64_t indptrStride, uint64_t total)
{
    AscendC::Simt::VF_CALL<SegmentCsrSimt<KIND, VEC>>(
        AscendC::Simt::Dim3{SEG_THREADS_PER_BLOCK}, src, indptr, out, argOut, reduce, isMean, m, nSeg, k, indptrStride,
        total);
}

uint32_t ResolveBlockDim(uint64_t totalWork)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t availableCores = platform->GetCoreNumAiv();
    const uint64_t requiredCores = (totalWork + SEG_THREADS_PER_BLOCK - 1) / SEG_THREADS_PER_BLOCK;
    uint32_t blockDim = static_cast<uint32_t>(requiredCores < availableCores ? requiredCores : availableCores);
    return blockDim == 0 ? 1 : blockDim;
}

} // namespace

void SegmentCsr(
    int kind, int vecElems, const void* src, const int64_t* indptr, void* out, int64_t* argOut, int reduce, int isMean,
    uint64_t e1, uint64_t m, uint32_t nSeg, uint32_t k, uint64_t indptrStride, aclrtStream stream)
{
    if (e1 == 0 || nSeg == 0 || k == 0) {
        return;
    }
    const uint64_t laneCount = vecElems > 0 ? static_cast<uint64_t>(k / vecElems) : k;
    const uint64_t total = e1 * nSeg * laneCount;
    if (total == 0 || total > 0xFFFFFFFFull) {
        return; // host layer guarantees total < 2^32 before calling
    }
    const uint32_t blockDim = ResolveBlockDim(total);

    // Vector eligibility is owned by the traits (VEC_LANES > 0); the host
    // only decides via vecElems whether K is aligned for this launch.
    DispatchKind(kind, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if constexpr (SegTraits<K>::VEC_LANES > 0) {
            if (vecElems > 0) {
                SegmentCsrKernelEntry<K, true>
                    <<<blockDim, nullptr, stream>>>(src, indptr, out, argOut, reduce, isMean, m, nSeg, k, indptrStride, total);
                return;
            }
        }
        SegmentCsrKernelEntry<K, false>
            <<<blockDim, nullptr, stream>>>(src, indptr, out, argOut, reduce, isMean, m, nSeg, k, indptrStride, total);
    });
}

} // namespace ops_gnn

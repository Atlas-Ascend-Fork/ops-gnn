/*
 * Copyright (c) 2026 Starlink_. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "segment_csr.h"

#include "kernel_operator.h"
#include "acl/acl_rt.h"
#include "tiling/platform/platform_ascendc.h"
#include "segment_csr_extrema.h"
#include "segment_csr_sum.h"
#include "segment_csr_uniform.h"

using namespace AscendC;
namespace ops_gnn {
namespace {
namespace R = AscendC::Reg;
constexpr R::CastTrait kWiden = {
    R::RegLayout::ZERO, R::SatMode::UNKNOWN, R::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr R::CastTrait kFloat = {
    R::RegLayout::ZERO, R::SatMode::UNKNOWN, R::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr R::CastTrait kHalf = {
    R::RegLayout::ZERO, R::SatMode::NO_SAT, R::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr R::CastTrait kTrunc = {
    R::RegLayout::ZERO, R::SatMode::NO_SAT, R::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

template <typename T>
struct RegisterType {
    using Value = T;
    using Register = R::RegTensor<T>;
};
template <>
struct RegisterType<half> {
    using Value = float;
    using Register = R::RegTensor<float>;
};
template <>
struct RegisterType<int64_t> {
    using Value = int64_t;
    using Register = R::RegTensor<int64_t, R::RegTraitNumTwo>;
};

// Widening load: 16-bit lanes are unpacked to fp32 registers on the fly.
template <typename T>
__simd_callee__ __attribute__((always_inline)) inline void LoadValue(
    typename RegisterType<T>::Register& value, __ubuf__ T* src, R::MaskReg mask)
{
    if constexpr (sizeof(T) == 2) {
        R::RegTensor<half> packed;
        R::LoadAlign<half, R::LoadDist::DIST_UNPACK_B16>(packed, src);
        R::Cast<float, half, kWiden>(value, packed, mask);
    } else {
        R::LoadAlign(value, src);
    }
}

template <typename C, int OP, bool PACK, typename V>
__simd_callee__ __attribute__((always_inline)) inline void Accumulate(
    V& acc, R::RegTensor<int32_t>& index, V& value, uint32_t row, R::MaskReg mask, R::RegTensor<int32_t>& laneRows)
{
    if constexpr (OP == 0) {
        R::Add(acc, acc, value, mask);
    } else {
        R::MaskReg better;
        R::RegTensor<int32_t> candidate;
        if constexpr (OP == 1)
            R::Compare<C, CMPMODE::LT>(better, value, acc, mask);
        else
            R::Compare<C, CMPMODE::GT>(better, value, acc, mask);
        if constexpr (PACK)
            R::Adds(candidate, laneRows, static_cast<int32_t>(row), mask);
        else
            R::Duplicate(candidate, static_cast<int32_t>(row), mask);
        R::Select(acc, value, acc, better);
        R::Select(index, candidate, index, better);
    }
}

template <typename C, int OP, typename V>
__simd_callee__ __attribute__((always_inline)) inline void Merge(
    V& acc, R::RegTensor<int32_t>& index, V& value, R::RegTensor<int32_t>& otherIndex, R::MaskReg mask)
{
    if constexpr (OP == 0)
        R::Add(acc, acc, value, mask);
    else {
        R::MaskReg better, equal;
        R::RegTensor<int32_t> firstIndex;
        R::Compare<C, CMPMODE::EQ>(equal, value, acc, mask);
        if constexpr (OP == 1)
            R::Compare<C, CMPMODE::LT>(better, value, acc, mask);
        else
            R::Compare<C, CMPMODE::GT>(better, value, acc, mask);
        R::Min(firstIndex, index, otherIndex, mask);
        R::Select(acc, value, acc, better);
        R::Select(index, otherIndex, index, better);
        R::Select(index, firstIndex, index, equal);
    }
}

// Four independent row chains hide register dependency latency. Their merge
// explicitly selects the first index on ties, including across chunk boundaries.
template <typename T, int OP, bool PACK>
__simd_callee__ __attribute__((always_inline)) inline uint32_t InitializeRegisterAccumulator(
    typename RegisterType<T>::Register& acc, R::RegTensor<int32_t>& index,
    __ubuf__ T* src, __ubuf__ typename RegisterType<T>::Value* saved,
    __ubuf__ int32_t* savedIndex, bool first, uint32_t rows, uint32_t rowBase,
    uint32_t sourceRows, R::RegTensor<int32_t>& laneRows, R::MaskReg mask)
{
    using C = typename RegisterType<T>::Value;
    uint32_t begin = 0;
    if (!first) {
        R::LoadAlign(acc, saved);
        if constexpr (OP != 0)
            R::LoadAlign(index, savedIndex);
    } else {
        R::Duplicate(acc, static_cast<C>(0), mask);
        R::Duplicate(index, static_cast<int32_t>(sourceRows), mask);
        if (OP != 0 && rows != 0) {
            LoadValue<T>(acc, src, mask);
            if constexpr (PACK)
                R::Adds(index, laneRows, static_cast<int32_t>(rowBase), mask);
            else
                R::Duplicate(index, static_cast<int32_t>(rowBase), mask);
            begin = 1;
        }
    }
    return begin;
}

template <typename T, int OP, bool PACK>
__simd_callee__ __attribute__((always_inline)) inline uint32_t ReduceLongRegisterRows(
    typename RegisterType<T>::Register& acc, R::RegTensor<int32_t>& index,
    __ubuf__ T* src, uint32_t rows, uint32_t srcStep, uint32_t rowBase, uint32_t begin,
    R::RegTensor<int32_t>& laneRows, R::MaskReg mask)
{
    using C = typename RegisterType<T>::Value;
    using V = typename RegisterType<T>::Register;
    V a1, a2, a3, value, v1, v2, v3;
    R::RegTensor<int32_t> i1, i2, i3;
    uint32_t r = begin;
    if constexpr (OP == 0) {
        R::Duplicate(a1, static_cast<C>(0), mask);
        R::Duplicate(a2, static_cast<C>(0), mask);
        R::Duplicate(a3, static_cast<C>(0), mask);
    } else {
        a1 = acc;
        a2 = acc;
        a3 = acc;
        i1 = index;
        i2 = index;
        i3 = index;
    }
    for (; r + 3 < rows; r += 4) {
        auto base = src + r * srcStep;
        LoadValue<T>(value, base, mask);
        LoadValue<T>(v1, base + srcStep, mask);
        LoadValue<T>(v2, base + 2 * srcStep, mask);
        LoadValue<T>(v3, base + 3 * srcStep, mask);
        const uint32_t row = rowBase + r * (PACK ? 2 : 1);
        Accumulate<C, OP, PACK>(acc, index, value, row, mask, laneRows);
        Accumulate<C, OP, PACK>(a1, i1, v1, row + (PACK ? 2 : 1), mask, laneRows);
        Accumulate<C, OP, PACK>(a2, i2, v2, row + (PACK ? 4 : 2), mask, laneRows);
        Accumulate<C, OP, PACK>(a3, i3, v3, row + (PACK ? 6 : 3), mask, laneRows);
    }
    Merge<C, OP>(acc, index, a1, i1, mask);
    Merge<C, OP>(a2, i2, a3, i3, mask);
    Merge<C, OP>(acc, index, a2, i2, mask);
    return r;
}

template <typename T, int OP, bool PACK>
__simd_callee__ __attribute__((always_inline)) inline void ReduceRegisterRows(
    typename RegisterType<T>::Register& acc, R::RegTensor<int32_t>& index,
    __ubuf__ T* src, uint32_t rows, uint32_t srcStep, uint32_t rowBase, uint32_t begin,
    R::RegTensor<int32_t>& laneRows, R::MaskReg mask)
{
    using C = typename RegisterType<T>::Value;
    typename RegisterType<T>::Register value;
    uint32_t r = begin;
    if (rows >= 8) {
        r = ReduceLongRegisterRows<T, OP, PACK>(acc, index, src, rows, srcStep, rowBase, begin, laneRows, mask);
    }
    for (; r < rows; ++r) {
        LoadValue<T>(value, src + r * srcStep, mask);
        Accumulate<C, OP, PACK>(
            acc, index, value, rowBase + r * (PACK ? 2 : 1), mask, laneRows);
    }
}

template <typename T, int OP, bool PACK>
__simd_callee__ __attribute__((always_inline)) inline void MergePackedChannels(
    typename RegisterType<T>::Register& acc, R::RegTensor<int32_t>& index, R::MaskReg& mask)
{
    using C = typename RegisterType<T>::Value;
    typename RegisterType<T>::Register value;
    R::RegTensor<int32_t> candidateIndex;
    if constexpr (PACK) {
        R::RegTensor<uint32_t> permutation, flip;
        R::Arange(reinterpret_cast<R::RegTensor<int32_t>&>(permutation), 0);
        R::Duplicate(flip, 32u, mask);
        R::Xor(permutation, permutation, flip, mask);
        R::Gather<C, uint32_t>(value, acc, permutation);
        if constexpr (OP != 0)
            R::Gather<int32_t, uint32_t>(candidateIndex, index, permutation);
        uint32_t halfCount = 32;
        mask = R::UpdateMask<int32_t>(halfCount);
        if constexpr (OP == 0)
            R::Add(acc, acc, value, mask);
        else
            Merge<C, OP>(acc, index, value, candidateIndex, mask);
    }
}

template <typename T>
__simd_callee__ __attribute__((always_inline)) inline void ApplyRegisterMean(
    typename RegisterType<T>::Register& acc, uint32_t fullRows, bool useMeanScale,
    float meanScale, int16_t meanShift, R::MaskReg mask)
{
    using C = typename RegisterType<T>::Value;
    if constexpr (std::is_integral<T>::value) {
        integer_mean::Divide<T>(acc, fullRows, meanShift, mask);
    } else {
        R::RegTensor<float> numerator, denominator;
        if constexpr (sizeof(T) == 2) {
            // The official reference rounds the sum before division.
            R::RegTensor<half> rounded;
            R::Cast<half, float, kHalf>(rounded, acc, mask);
            R::Cast<float, half, kWiden>(numerator, rounded, mask);
        } else if constexpr (std::is_same<T, float>::value) {
            numerator = acc;
        }
        if (useMeanScale) {
            R::Muls(numerator, numerator, meanScale, mask);
        } else {
            R::Duplicate(denominator, static_cast<float>(fullRows), mask);
            if constexpr (sizeof(T) == 2) {
                R::RegTensor<half> roundedCount;
                R::Cast<half, float, kHalf>(roundedCount, denominator, mask);
                R::Cast<float, half, kWiden>(denominator, roundedCount, mask);
            }
            R::Div(numerator, numerator, denominator, mask);
        }
        if constexpr (std::is_same<C, float>::value)
            acc = numerator;
    }
}

template <typename T, int OP>
__simd_callee__ __attribute__((always_inline)) inline void FinishRegisterAccumulator(
    typename RegisterType<T>::Register& acc, R::RegTensor<int32_t>& index,
    __ubuf__ typename RegisterType<T>::Value* saved, __ubuf__ int32_t* savedIndex,
    __ubuf__ T* out, __ubuf__ int64_t* arg, bool last, bool mean, uint32_t fullRows,
    bool useMeanScale, float meanScale, int16_t meanShift, R::MaskReg mask)
{
    if (!last) {
        R::StoreAlign(saved, acc, mask);
        if constexpr (OP != 0)
            R::StoreAlign(savedIndex, index, mask);
    } else {
        if (OP == 0 && mean && fullRows != 0) {
            ApplyRegisterMean<T>(acc, fullRows, useMeanScale, meanScale, meanShift, mask);
        }
        if constexpr (sizeof(T) == 2) {
            R::RegTensor<half> packed;
            R::Cast<half, float, kHalf>(packed, acc, mask);
            R::StoreAlign<half, R::StoreDist::DIST_PACK_B32>(out, packed, mask);
        } else {
            R::StoreAlign(out, acc, mask);
        }
        if constexpr (OP != 0) {
            R::RegTensor<int64_t, R::RegTraitNumTwo> wideIndex;
            R::Cast<int64_t, int32_t, kWiden>(wideIndex, index, mask);
            R::StoreAlign(arg, wideIndex, mask);
        }
    }
}

template <typename T, int OP, uint32_t CHANNELS = 0, bool PACK = false>
__simd_vf__ inline void ReduceTile(
    __ubuf__ T* src, __ubuf__ typename RegisterType<T>::Value* saved, __ubuf__ int32_t* savedIndex, __ubuf__ T* out,
    __ubuf__ int64_t* arg, uint32_t segments, uint32_t inputRows, uint32_t inputChannels, uint32_t rowBase,
    uint32_t fullRows, bool first, bool last, bool mean, bool useMeanScale, float meanScale, int16_t meanShift,
    uint32_t sourceRows)
{
    using V = typename RegisterType<T>::Register;
    const uint32_t channels = CHANNELS != 0 ? CHANNELS : inputChannels;
    const uint32_t rows = inputRows / (PACK ? 2 : 1);
    const uint32_t srcStep = channels * (PACK ? 2 : 1);
    R::RegTensor<int32_t> laneRows;
    if constexpr (PACK && OP != 0) {
        R::Arange(laneRows, 0);
        auto fullMask = R::CreateMask<int32_t, R::MaskPattern::ALL>();
        R::ShiftRights(laneRows, laneRows, int16_t(5), fullMask);
    }
    for (uint32_t s = 0; s < segments; ++s) {
        for (uint32_t c = 0; c < channels; c += 64) {
            uint32_t count = PACK ? 64 : channels - c < 64 ? channels - c : 64;
            auto mask = R::UpdateMask<int32_t>(count);
            V acc;
            R::RegTensor<int32_t> index;
            const uint32_t dst = s * channels + c;
            auto source = src + s * rows * srcStep + c;
            const uint32_t firstRow = rowBase + s * inputRows;
            const uint32_t begin = InitializeRegisterAccumulator<T, OP, PACK>(
                acc, index, source, saved + dst, savedIndex + dst, first, rows, firstRow, sourceRows, laneRows, mask);
            ReduceRegisterRows<T, OP, PACK>(acc, index, source, rows, srcStep, firstRow, begin, laneRows, mask);
            MergePackedChannels<T, OP, PACK>(acc, index, mask);
            FinishRegisterAccumulator<T, OP>(acc, index, saved + dst, savedIndex + dst, out + dst, arg + dst,
                last, mean, fullRows, useMeanScale, meanScale, meanShift, mask);
        }
    }
}

struct MeanParameters {
    float scale;
    bool useScale;
    int16_t shift;
};

__aicore__ inline MeanParameters MeanForRows(uint32_t rows)
{
    switch (rows) {
        case 1:
            return {1.0f, true, 0};
        case 2:
            return {0.5f, true, 1};
        case 4:
            return {0.25f, true, 2};
        case 8:
            return {0.125f, true, 3};
        case 16:
            return {0.0625f, true, 4};
        case 32:
            return {0.03125f, true, 5};
        case 64:
            return {0.015625f, true, 6};
        case 128:
            return {0.0078125f, true, 7};
        case 256:
            return {0.00390625f, true, 8};
        case 512:
            return {0.001953125f, true, 9};
        default:
            return {0.0f, false, -1};
    }
}

struct Tile {
    uint32_t segment, row, rows, segments, fullRows;
    bool first, last;
};

template <typename T>
class SegmentCsrRegisterKernel {
public:
    __aicore__ inline void Init(
        __gm__ const T* src, __gm__ const int64_t* ptr, __gm__ T* out, __gm__ int64_t* arg, uint32_t m,
        uint32_t segments, uint32_t channels, uint32_t tileBytes, int op, bool mean, TPipe* pipe)
    {
        m_ = m;
        k_ = channels;
        op_ = op;
        mean_ = mean;
        hasArg_ = (op != 0 && arg != nullptr);
        const uint32_t per = (segments + GetBlockNum() - 1) / GetBlockNum();
        begin_ = per * GetBlockIdx() < segments ? per * GetBlockIdx() : segments;
        end_ = begin_ + per < segments ? begin_ + per : segments;
        planSeg_ = begin_;
        planRow_ = 0;
        uniform_ = false;
        uniformBase_ = 0;
        uniformLength_ = 0;
        maxRows_ = tileBytes / (k_ * sizeof(T));
        maxSegments_ = 2048 / k_;
        src_.SetGlobalBuffer(const_cast<__gm__ T*>(src), static_cast<uint64_t>(m) * k_);
        ptr_.SetGlobalBuffer(const_cast<__gm__ int64_t*>(ptr), segments + 1);
        out_.SetGlobalBuffer(out, static_cast<uint64_t>(segments) * k_);
        if (op != 0 && arg != nullptr)
            arg_.SetGlobalBuffer(arg, static_cast<uint64_t>(segments) * k_);
        pipe->InitBuffer(input_, 2, tileBytes + 512);
        pipe->InitBuffer(saved_, 2048 * sizeof(typename RegisterType<T>::Value) + 256);
        pipe->InitBuffer(savedIndex_, 2048 * sizeof(int32_t) + 256);
        pipe->InitBuffer(result_, 2048 * sizeof(T) + 256);
        pipe->InitBuffer(argResult_, 2048 * sizeof(int64_t) + 512);
        pipe->InitBuffer(ptrCache_, 2048 * sizeof(int64_t));
        cached_ = end_ - begin_ + 1 < 2048 ? end_ - begin_ + 1 : 2048;
        evCopyScalar_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_S>());
        evWrite_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        evReuse_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        CacheAndValidatePointers(pipe);
    }

    __aicore__ inline void CacheAndValidatePointers(TPipe* pipe)
    {
        const auto evValidate = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_S>());
        if (begin_ < end_) {
            DataCopyPad(
                ptrCache_.Get<int64_t>(), ptr_[begin_], DataCopyExtParams{1, cached_ * 8, 0, 0, 0},
                DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(evCopyScalar_);
            WaitFlag<HardEvent::MTE2_S>(evCopyScalar_);
            const int64_t first = ptrCache_.Get<int64_t>().GetValue(0);
            const int64_t second = ptrCache_.Get<int64_t>().GetValue(1);
            if (cached_ == end_ - begin_ + 1 && first >= 0 && second >= first &&
                static_cast<uint64_t>(first) + static_cast<uint64_t>(second - first) * (end_ - begin_) <= m_) {
                uniformBase_ = static_cast<uint32_t>(first);
                uniformLength_ = static_cast<uint32_t>(second - first);
                uniform_ptr::ValidateUniformPtr(
                    (__ubuf__ int64_t*)ptrCache_.Get<int64_t>().GetPhyAddr(),
                    (__ubuf__ int32_t*)savedIndex_.Get<int32_t>().GetPhyAddr(), cached_, uniformBase_, uniformLength_);
                SetFlag<HardEvent::V_S>(evValidate);
                WaitFlag<HardEvent::V_S>(evValidate);
                uniform_ = savedIndex_.Get<int32_t>().GetValue(0) == 0;
            }
        }
    }

    __aicore__ inline uint32_t Ptr(uint32_t segment)
    {
        int64_t p =
            segment - begin_ < cached_ ? ptrCache_.Get<int64_t>().GetValue(segment - begin_) : ptr_.GetValue(segment);
        return p < 0 ? 0 : p > m_ ? m_ : static_cast<uint32_t>(p);
    }

    __aicore__ inline bool Prepare(Tile& tile)
    {
        if (planSeg_ >= end_)
            return false;
        const uint32_t b = uniform_ ? uniformBase_ + (planSeg_ - begin_) * uniformLength_ : Ptr(planSeg_);
        const uint32_t e = uniform_ ? b + uniformLength_ : Ptr(planSeg_ + 1);
        const uint32_t length = e > b ? e - b : 0;
        tile.segment = planSeg_;
        tile.row = b + planRow_;
        tile.fullRows = length;
        tile.rows = length - planRow_ < maxRows_ ? length - planRow_ : maxRows_;
        tile.first = planRow_ == 0;
        tile.last = planRow_ + tile.rows == length;
        tile.segments = 1;
        if (tile.first && tile.last) {
            GroupCompleteSegments(tile, length, b);
        }
        if (tile.last) {
            planSeg_ += tile.segments;
            planRow_ = 0;
        } else
            planRow_ += tile.rows;
        if (tile.rows != 0) {
            auto input = input_.AllocTensor<T>();
            DataCopyPad(
                input, src_[static_cast<uint64_t>(tile.row) * k_],
                DataCopyExtParams{1, tile.rows * tile.segments * k_ * sizeof(T), 0, 0, 0},
                DataCopyPadExtParams<T>{false, 0, 0, 0});
            input_.EnQue(input);
        }
        return true;
    }

    __aicore__ inline void GroupCompleteSegments(Tile& tile, uint32_t length, uint32_t b)
    {
        if (uniform_) {
            uint32_t capacity = maxSegments_;
            if (length > 0)
                capacity = maxRows_ / length;
            tile.segments = end_ - planSeg_ < maxSegments_ ? end_ - planSeg_ : maxSegments_;
            if (tile.segments > capacity)
                tile.segments = capacity;
        } else {
            while (tile.segments < maxSegments_ && planSeg_ + tile.segments < end_) {
                if (length != 0 && (tile.segments + 1) * length > maxRows_)
                    break;
                const uint32_t nb = Ptr(planSeg_ + tile.segments);
                const uint32_t ne = Ptr(planSeg_ + tile.segments + 1);
                if (nb != b + tile.segments * length || ne != nb + length)
                    break;
                ++tile.segments;
            }
        }
    }

    __aicore__ inline void Process()
    {
        Tile current{}, next{};
        bool hasCurrent = Prepare(current);
        bool hasNext = hasCurrent && Prepare(next);
        while (hasCurrent) {
            auto input = result_.Get<T>();
            if (current.rows != 0)
                input = input_.DeQue<T>();
            if (op_ == 0)
                Dispatch<0>(input, current);
            else if (op_ == 1)
                Dispatch<1>(input, current);
            else
                Dispatch<2>(input, current);
            if (current.rows != 0)
                input_.FreeTensor(input);
            if (current.last) {
                SetFlag<HardEvent::V_MTE3>(evWrite_);
                WaitFlag<HardEvent::V_MTE3>(evWrite_);
                const uint32_t elems = current.segments * k_;
                DataCopyPad(
                    out_[static_cast<uint64_t>(current.segment) * k_], result_.Get<T>(),
                    DataCopyExtParams{1, elems * sizeof(T), 0, 0, 0});
                if (hasArg_)
                    DataCopyPad(
                        arg_[static_cast<uint64_t>(current.segment) * k_], argResult_.Get<int64_t>(),
                        DataCopyExtParams{1, elems * 8, 0, 0, 0});
                SetFlag<HardEvent::MTE3_V>(evReuse_);
                WaitFlag<HardEvent::MTE3_V>(evReuse_);
            }
            current = next;
            hasCurrent = hasNext;
            hasNext = hasCurrent && Prepare(next);
        }
    }

    template <int OP>
    __aicore__ inline void Dispatch(LocalTensor<T> input, const Tile& current)
    {
        if constexpr (OP == 0) {
            if (TryNarrowHalfSum(input, current) || TryShortSum(input, current))
                return;
        }
        if constexpr (OP != 0 && sizeof(T) == 8) {
            if (TryInt64Extrema<OP>(input, current))
                return;
        }
        if constexpr (OP != 0 && sizeof(T) == 2) {
            if (TryHalfExtrema<OP>(input, current))
                return;
        }
        DispatchGeneric<OP>(input, current);
    }

    __aicore__ inline bool TryNarrowHalfSum(LocalTensor<T> input, const Tile& current)
    {
        if constexpr (sizeof(T) == 2) {
            if (k_ == 32 && current.first && current.last) {
                if (current.rows == 16) {
                    ComputeNarrowHalf<16>(input, current);
                    return true;
                }
                if (current.rows == 32) {
                    ComputeNarrowHalf<32>(input, current);
                    return true;
                }
                if (current.rows == 64) {
                    ComputeNarrowHalf<64>(input, current);
                    return true;
                }
            }
        }
        return false;
    }

    __aicore__ inline bool TryShortSum(LocalTensor<T> input, const Tile& current)
    {
        if (current.first && current.last && (current.rows == 2 || current.rows == 4 || current.rows == 8) &&
            (k_ == 64 || k_ == 128 || k_ == 256)) {
            if (current.rows == 2)
                DispatchShortSum<2>(input, current);
            else if (current.rows == 4)
                DispatchShortSum<4>(input, current);
            else
                DispatchShortSum<8>(input, current);
            return true;
        }
        return false;
    }

    template <int OP>
    __aicore__ inline bool TryInt64Extrema(LocalTensor<T> input, const Tile& current)
    {
        if (current.rows > 0) {
            ReduceInt64Tile<OP>(
                (__ubuf__ int64_t*)input.GetPhyAddr(), (__ubuf__ int64_t*)result_.Get<T>().GetPhyAddr(),
                (__ubuf__ int64_t*)argResult_.Get<int64_t>().GetPhyAddr(), current.segments, current.rows, k_,
                current.row, (__ubuf__ int64_t*)saved_.Get<int64_t>().GetPhyAddr(),
                (__ubuf__ int32_t*)savedIndex_.Get<int32_t>().GetPhyAddr(), current.first, current.last);
            return true;
        }
        return false;
    }

    template <int OP>
    __aicore__ inline bool TryHalfExtrema(LocalTensor<T> input, const Tile& current)
    {
        const uint32_t pack = k_ < 128 ? 128 / k_ : 1;
        if (current.first && current.last && current.rows > 0 && current.rows <= 65535 &&
            current.rows % pack == 0) {
            auto src = (__ubuf__ half*)input.GetPhyAddr();
            auto out = (__ubuf__ half*)result_.Get<T>().GetPhyAddr();
            auto arg = (__ubuf__ int64_t*)argResult_.Get<int64_t>().GetPhyAddr();
            if (k_ == 32)
                half_extrema::ReduceHalfPackedTile<32, OP>(
                    src, out, arg, current.segments, current.rows, current.row);
            else if (k_ == 64)
                half_extrema::ReduceHalfPackedTile<64, OP>(
                    src, out, arg, current.segments, current.rows, current.row);
            else if (k_ == 128)
                half_extrema::ReduceHalfPackedTile<128, OP>(
                    src, out, arg, current.segments, current.rows, current.row);
            else if (k_ == 256)
                half_extrema::ReduceHalfPackedTile<256, OP>(
                    src, out, arg, current.segments, current.rows, current.row);
            else {
                Compute<OP>(input, current);
            }
            return true;
        }
        return false;
    }

    template <int OP>
    __aicore__ inline void DispatchGeneric(LocalTensor<T> input, const Tile& current)
    {
        if constexpr (sizeof(T) != 8) {
            if (k_ == 32 && current.first && current.last && current.rows % 2 == 0) {
                Compute<OP, 32, true>(input, current);
                return;
            }
        }
        if (k_ == 32)
            Compute<OP, 32>(input, current);
        else if (k_ == 64)
            Compute<OP, 64>(input, current);
        else if (k_ == 128)
            Compute<OP, 128>(input, current);
        else if (k_ == 256)
            Compute<OP, 256>(input, current);
        else
            Compute<OP>(input, current);
    }



    template <int OP, uint32_t WIDTH = 0, bool PACK = false>
    __aicore__ inline void Compute(LocalTensor<T> input, const Tile& current)
    {
        MeanParameters parameters{0.0f, false, -1};
        if constexpr (OP == 0) {
            parameters = MeanForRows(current.fullRows);
        }
        ReduceTile<T, OP, WIDTH, PACK>(
            (__ubuf__ T*)input.GetPhyAddr(),
            (__ubuf__ typename RegisterType<T>::Value*)saved_.Get<uint8_t>().GetPhyAddr(),
            (__ubuf__ int32_t*)savedIndex_.Get<int32_t>().GetPhyAddr(), (__ubuf__ T*)result_.Get<T>().GetPhyAddr(),
            (__ubuf__ int64_t*)argResult_.Get<int64_t>().GetPhyAddr(), current.segments, current.rows, k_, current.row,
            current.fullRows, current.first, current.last, mean_, parameters.useScale, parameters.scale, parameters.shift, m_);
    }

    template <uint32_t ROWS, uint32_t WIDTH>
    __aicore__ inline void ComputeShortSum(LocalTensor<T> input, const Tile& current)
    {
        auto src = (__ubuf__ T*)input.GetPhyAddr();
        auto out = (__ubuf__ T*)result_.Get<T>().GetPhyAddr();
        if (mean_)
            short_sum::ReduceShortSumTile<T, ROWS, WIDTH, true>(src, out, current.segments);
        else
            short_sum::ReduceShortSumTile<T, ROWS, WIDTH, false>(src, out, current.segments);
    }

    template <uint32_t ROWS>
    __aicore__ inline void ComputeNarrowHalf(LocalTensor<T> input, const Tile& current)
    {
        auto src = (__ubuf__ half*)input.GetPhyAddr();
        auto out = (__ubuf__ half*)result_.Get<T>().GetPhyAddr();
        if (mean_)
            narrow_half_sum::ReduceNarrowHalfSum<ROWS, true>(src, out, current.segments);
        else
            narrow_half_sum::ReduceNarrowHalfSum<ROWS, false>(src, out, current.segments);
    }

    template <uint32_t ROWS>
    __aicore__ inline void DispatchShortSum(LocalTensor<T> input, const Tile& current)
    {
        if (k_ == 64)
            ComputeShortSum<ROWS, 64>(input, current);
        else if (k_ == 128)
            ComputeShortSum<ROWS, 128>(input, current);
        else
            ComputeShortSum<ROWS, 256>(input, current);
    }

private:
    TQue<TPosition::VECIN, 2> input_;
    TBuf<TPosition::VECCALC> saved_, savedIndex_, result_, argResult_, ptrCache_;
    GlobalTensor<T> src_, out_;
    GlobalTensor<int64_t> ptr_, arg_;
    uint32_t m_, k_, begin_, end_, cached_, maxRows_, maxSegments_, planSeg_, planRow_;
    int op_;
    bool mean_;
    bool hasArg_;
    bool uniform_;
    uint32_t uniformBase_, uniformLength_;
    event_t evCopyScalar_, evWrite_, evReuse_;
};

template <typename T>
__attribute__((aiv)) __global__ __aicore__ void SegmentCsrVecKernel(
    __gm__ const void* src, __gm__ const int64_t* ptr, __gm__ void* out, __gm__ int64_t* arg, uint32_t m,
    uint32_t segments, uint32_t k, uint32_t tileBytes, int op, int mean)
{
    TPipe pipe;
    SegmentCsrRegisterKernel<T> kernel;
    kernel.Init((__gm__ const T*)src, ptr, (__gm__ T*)out, arg, m, segments, k, tileBytes, op, mean != 0, &pipe);
    kernel.Process();
}

} // namespace

namespace {

// kind -> register-engine element type; kinds without a register path map to
// void and are silently skipped, exactly like the previous fall-through.
template <int KIND>
struct VecRegType {
    using type = void;
};
template <>
struct VecRegType<KIND_F32> {
    using type = float;
};
template <>
struct VecRegType<KIND_F16> {
    using type = half;
};
template <>
struct VecRegType<KIND_I32> {
    using type = int32_t;
};
template <>
struct VecRegType<KIND_I64> {
    using type = int64_t;
};

} // namespace

void SegmentCsrVector(
    int kind, const void* src, const int64_t* ptr, void* out, int64_t* arg, int op, uint64_t m, uint32_t segments,
    uint32_t k, aclrtStream stream, int mean)
{
    if (segments == 0 || k == 0 || m == 0)
        return;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    uint32_t tileBytes = static_cast<uint32_t>((ub - 80 * 1024) / 2 / 256 * 256);
    if (tileBytes > 64 * 1024)
        tileBytes = 64 * 1024;
    const uint32_t cores = platform->GetCoreNumAiv();
    const uint32_t blocks = segments < cores ? segments : cores;
    // One launch expression for every dtype: DispatchKind resolves the
    // compile-time kind, VecRegType maps it to the element type (or void).
    DispatchKind(kind, [&](auto kc) {
        using T = typename VecRegType<decltype(kc)::value>::type;
        if constexpr (!std::is_void_v<T>) {
            SegmentCsrVecKernel<T><<<blocks, nullptr, stream>>>(
                src, ptr, out, arg, m, segments, k, tileBytes, op, mean);
        }
    });
}
} // namespace ops_gnn

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "kernel_operator.h"
#include "scatter.h"
#include <type_traits>

namespace opsgnn::scatter_a2 {
using namespace AscendC;

template <HardEvent Event> __aicore__ inline void Sync() {
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
}
__aicore__ inline uint64_t Smaller(uint64_t a, uint64_t b) { return a < b ? a : b; }
// Divide first so even very large logical lengths cannot overflow core * length.
__aicore__ inline uint64_t Partition(uint64_t elements, uint32_t core, uint32_t cores) {
    if (cores == 0)
        return 0;
    return (elements / cores) * core + ((elements % cores) * core) / cores;
}
template <typename U>
__aicore__ inline void LoadScatterTile(LocalTensor<U> target, GlobalTensor<U> &input, uint64_t offset, uint32_t count) {
    DataCopyPad(target, input[offset], {1, count * static_cast<uint32_t>(sizeof(U)), 0, 0, 0},
                DataCopyPadExtParams<U>(false, 0, 0, 0));
}

template <typename U>
__aicore__ inline void StoreScatterTile(GlobalTensor<U> &output, LocalTensor<U> source, uint64_t offset,
                                        uint32_t count) {
    DataCopyPad(output[offset], source, {1, count * static_cast<uint32_t>(sizeof(U)), 0, 0, 0});
}

__aicore__ inline void InitializeScatterCounts(GlobalTensor<int32_t> &output, LocalTensor<int32_t> counts,
                                               uint64_t buckets) {
    Duplicate(counts, 0, kOutputTileElements);
    Sync<HardEvent::V_MTE3>();
    for (uint64_t first = GetBlockIdx() * kOutputTileElements; first < buckets;
         first += GetBlockNum() * kOutputTileElements) {
        const uint32_t valid = Smaller(kOutputTileElements, buckets - first);
        DataCopyPad(output[first], counts, {1, valid * 4, 0, 0, 0});
    }
    Sync<HardEvent::MTE3_V>();
}

struct SourceTile {
    uint64_t batch, first, feature;
    uint32_t rows, valid;
};

__aicore__ inline SourceTile DescribeSourceTile(uint64_t unit, const Tiling *tiling) {
    const uint64_t features = (tiling->after + tiling->atomicWidth - 1) / tiling->atomicWidth;
    const uint64_t rows = (tiling->length + tiling->atomicRows - 1) / tiling->atomicRows;
    const uint64_t feature = unit % features * tiling->atomicWidth;
    const uint64_t first = unit / features % rows * tiling->atomicRows;
    return {unit / (features * rows), first, feature,
            static_cast<uint32_t>(Smaller(tiling->atomicRows, tiling->length - first)),
            static_cast<uint32_t>(Smaller(tiling->atomicWidth, tiling->after - feature))};
}

union FloatBits {
    uint32_t bits;
    float value;
};
__aicore__ inline float AsFloat(uint32_t bits) {
    FloatBits x;
    x.bits = bits;
    return x.value;
}
__aicore__ inline uint32_t AsBits(float value) {
    FloatBits x;
    x.value = value;
    return x.bits;
}

__aicore__ inline float DecodeHalf(uint16_t raw) {
    const uint32_t sign = static_cast<uint32_t>(raw & 0x8000U) << 16;
    uint32_t exponent = (raw >> 10) & 31U;
    uint32_t mantissa = raw & 1023U;
    if (exponent == 31)
        return AsFloat(sign | 0x7f800000U | (mantissa << 13));
    if (exponent)
        return AsFloat(sign | ((exponent + 112) << 23) | (mantissa << 13));
    if (!mantissa)
        return AsFloat(sign);
    int32_t shift = 0;
    while (!(mantissa & 1024U)) {
        mantissa <<= 1;
        ++shift;
    }
    return AsFloat(sign | (static_cast<uint32_t>(113 - shift) << 23) | ((mantissa & 1023U) << 13));
}

__aicore__ inline uint16_t EncodeHalf(float value) {
    const uint32_t bits = AsBits(value), sign = (bits >> 16) & 0x8000U;
    const uint32_t exponent = (bits >> 23) & 255U, fraction = bits & 0x7fffffU;
    if (exponent == 255)
        return static_cast<uint16_t>(sign | 0x7c00U | (fraction ? 0x200U : 0));
    if (exponent > 142)
        return static_cast<uint16_t>(sign | 0x7c00U);
    if (exponent < 102)
        return static_cast<uint16_t>(sign);
    if (exponent < 113) {
        const uint32_t shift = 126 - exponent, mantissa = fraction | 0x800000U;
        const uint32_t bias = (1U << (shift - 1)) - 1 + ((mantissa >> shift) & 1U);
        return static_cast<uint16_t>(sign | ((mantissa + bias) >> shift));
    }
    const uint32_t rounded = fraction + 0xfffU + ((fraction >> 13) & 1U);
    return static_cast<uint16_t>(sign | (((exponent - 112) << 10) + (rounded >> 13)));
}

template <uint32_t Kind> struct Number {
    static constexpr bool floating = Kind <= Float32;
    using Raw = typename std::conditional<
        Kind == Float32 || Kind == Int32, uint32_t,
        typename std::conditional<Kind == Float16 || Kind == BFloat16 || Kind == Int16, uint16_t, uint8_t>::type>::type;
    using Value = typename std::conditional<floating, float, int32_t>::type;
    __aicore__ static inline Value Decode(Raw raw) {
        if constexpr (Kind == Float32)
            return AsFloat(raw);
        else if constexpr (Kind == Float16)
            return DecodeHalf(raw);
        else if constexpr (Kind == BFloat16)
            return AsFloat(static_cast<uint32_t>(raw) << 16);
        else if constexpr (Kind == Int8)
            return static_cast<int8_t>(raw);
        else if constexpr (Kind == Int16)
            return static_cast<int16_t>(raw);
        else
            return static_cast<Value>(raw);
    }
    __aicore__ static inline Raw Encode(Value value) {
        if constexpr (Kind == Float32)
            return AsBits(value);
        else if constexpr (Kind == Float16)
            return EncodeHalf(value);
        else if constexpr (Kind == BFloat16) {
            const uint32_t bits = AsBits(value);
            if ((bits & 0x7fffffffU) > 0x7f800000U)
                return 0x7fc0U;
            return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
        } else
            return static_cast<Raw>(value);
    }
    __aicore__ static inline Value Limit(bool minimum) {
        if constexpr (Kind == Float32)
            return minimum ? 0x1.fffffep127f : -0x1.fffffep127f;
        if constexpr (Kind == Float16)
            return minimum ? 65504.0f : -65504.0f;
        if constexpr (Kind == BFloat16)
            return minimum ? 0x1.fep127f : -0x1.fep127f;
        if constexpr (Kind == Int8)
            return minimum ? 127 : -128;
        if constexpr (Kind == Int16)
            return minimum ? 32767 : -32768;
        if constexpr (Kind == UInt8)
            return minimum ? 255 : 0;
        return minimum ? INT32_MAX : INT32_MIN;
    }
};

template <uint32_t Kind, uint32_t Reduction> struct Reducer {
    using Numeric = Number<Kind>;
    using Value = typename Numeric::Value;
    static constexpr bool extreme = Reduction == Min || Reduction == Max;
    __aicore__ static inline Value Initial() {
        if constexpr (extreme)
            return Numeric::Limit(Reduction == Min);
        return Reduction == Mul ? 1 : 0;
    }
    __aicore__ static inline bool Update(Value &accumulator, Value source) {
        if constexpr (Reduction == Sum) {
            if constexpr (Numeric::floating)
                accumulator += source;
            else
                accumulator = static_cast<int32_t>(static_cast<uint32_t>(accumulator) + static_cast<uint32_t>(source));
        } else if constexpr (Reduction == Mul) {
            if constexpr (Numeric::floating)
                accumulator = Numeric::Decode(Numeric::Encode(accumulator * source));
            else
                accumulator = static_cast<int32_t>(static_cast<uint32_t>(accumulator) * static_cast<uint32_t>(source));
        } else if (Reduction == Min ? source < accumulator : source > accumulator) {
            accumulator = source;
            return true;
        }
        return false;
    }
};
} // namespace opsgnn::scatter_a2

namespace opsgnn::scatter_a2 {
// Every core owns complete 32-byte output regions. This path handles arbitrary
// expanded index layouts and all integer dtypes without floating conversion.
template <uint32_t Kind, uint32_t Reduction> class ScalarScatter {
    using Numeric = Number<Kind>;
    using Raw = typename Numeric::Raw;
    using Value = typename Numeric::Value;
    using Operation = Reducer<Kind, Reduction>;

  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, const Tiling &t) {
        t_ = &t;
        src_.SetGlobalBuffer(reinterpret_cast<__gm__ Raw *>(src), t.sourceSpan);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ Raw *>(out), t.outputSpan);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), t.indexSpan);
        if constexpr (Operation::extreme)
            arg_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(arg), t.outElements);
        pipe_.InitBuffer(values_, kScalarOutputElements * sizeof(Raw));
        pipe_.InitBuffer(arguments_, kScalarOutputElements * sizeof(int64_t));
        pipe_.InitBuffer(source_, kDmaBytes);
        pipe_.InitBuffer(indexBuffer_, kDmaBytes);
    }
    __aicore__ inline void Process() {
        if (t_->path == AliasedOrdered) {
            ProcessAlias();
            return;
        }
        for (uint64_t unit = GetBlockIdx(); unit < t_->units; unit += GetBlockNum())
            ProcessTile(unit);
    }

  private:
    __aicore__ inline uint64_t IndexOffset(uint64_t source, uint64_t row) {
        if (t_->compactIndex)
            return (t_->indexElements == 1 ? 0 : row) * t_->indexStrides[0];
        uint64_t offset = 0;
        for (int32_t axis = static_cast<int32_t>(t_->rank) - 1; axis >= 0; --axis) {
            offset += (source % t_->sizes[axis]) * t_->indexStrides[axis];
            source /= t_->sizes[axis];
        }
        return offset;
    }
    __aicore__ inline int64_t Index(uint64_t source, uint64_t row) {
        const uint64_t offset = IndexOffset(source, row);
        if (!t_->indexMutable)
            return index_.GetValue(offset);
        DataCopyPad(indexBuffer_.Get<int64_t>(), index_[offset], {1, sizeof(int64_t), 0, 0, 0},
                    DataCopyPadExtParams<int64_t>(false, 0, 0, 0));
        Sync<HardEvent::MTE2_S>();
        const int64_t value = indexBuffer_.Get<int64_t>().GetValue(0);
        Sync<HardEvent::S_MTE2>();
        return value;
    }
    __aicore__ inline Value ReduceOutput(uint64_t position, int64_t &argument) {
        const uint64_t feature = position % t_->after;
        const uint64_t bucket = position / t_->after % t_->buckets;
        const uint64_t batch = position / (t_->after * t_->buckets);
        Value value = t_->hasOut ? Numeric::Decode(out_.GetValue(position)) : Operation::Initial();
        for (uint64_t row = 0; row < t_->length; ++row) {
            const uint64_t source = (batch * t_->length + row) * t_->after + feature;
            if (Index(source, row) == static_cast<int64_t>(bucket) &&
                Operation::Update(value, Numeric::Decode(src_.GetValue(source))))
                argument = row;
        }
        if (!t_->hasOut && (!t_->srcElements || (Operation::extreme && value == Operation::Initial())))
            value = 0;
        return value;
    }
    __aicore__ inline void StoreTile(uint64_t first, uint32_t count) {
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(out_[first], values_.Get<Raw>(), {1, count * static_cast<uint32_t>(sizeof(Raw)), 0, 0, 0});
        if constexpr (Operation::extreme)
            DataCopyPad(arg_[first], arguments_.Get<int64_t>(), {1, count * 8, 0, 0, 0});
        Sync<HardEvent::MTE3_S>();
    }
    __aicore__ inline void ProcessTile(uint64_t unit) {
        const uint64_t first = unit * kScalarOutputElements;
        const uint32_t count = Smaller(kScalarOutputElements, t_->outElements - first);
        auto values = values_.Get<Raw>();
        auto arguments = arguments_.Get<int64_t>();
        for (uint32_t lane = 0; lane < count; ++lane) {
            int64_t argument = t_->length;
            values.SetValue(lane, Numeric::Encode(ReduceOutput(first + lane, argument)));
            if constexpr (Operation::extreme)
                arguments.SetValue(lane, argument);
        }
        StoreTile(first, count);
    }
    __aicore__ inline Raw ReadMutable(GlobalTensor<Raw> &tensor, uint64_t offset) {
        DataCopyPad(source_.Get<Raw>(), tensor[offset], {1, sizeof(Raw), 0, 0, 0},
                    DataCopyPadExtParams<Raw>(false, 0, 0, 0));
        Sync<HardEvent::MTE2_S>();
        const Raw value = source_.Get<Raw>().GetValue(0);
        Sync<HardEvent::S_MTE2>();
        return value;
    }
    __aicore__ inline void InitializeAliasArgs() {
        if constexpr (Operation::extreme) {
            auto arguments = arguments_.Get<int64_t>();
            for (uint32_t lane = 0; lane < kScalarOutputElements; ++lane)
                arguments.SetValue(lane, t_->length);
            Sync<HardEvent::S_MTE3>();
            for (uint64_t first = 0; first < t_->outElements; first += kScalarOutputElements) {
                const uint32_t count = Smaller(kScalarOutputElements, t_->outElements - first);
                DataCopyPad(arg_[first], arguments, {1, count * 8, 0, 0, 0});
            }
            Sync<HardEvent::MTE3_S>();
        }
    }
    __aicore__ inline uint64_t StorageOffset(uint64_t logical, bool output) {
        uint64_t offset = 0;
        for (int32_t axis = static_cast<int32_t>(t_->rank) - 1; axis >= 0; --axis) {
            const uint64_t size = output && static_cast<uint32_t>(axis) == t_->dim ? t_->buckets : t_->sizes[axis];
            offset += (logical % size) * (output ? t_->outputStrides[axis] : t_->sourceStrides[axis]);
            logical /= size;
        }
        return offset;
    }
    __aicore__ inline void AliasUpdate(uint64_t source) {
        const uint64_t row = source / t_->after % t_->length;
        const int64_t bucket = Index(source, row);
        if (bucket < 0 || static_cast<uint64_t>(bucket) >= t_->buckets)
            return;
        const uint64_t batch = source / (t_->after * t_->length);
        const uint64_t destination = (batch * t_->buckets + bucket) * t_->after + source % t_->after;
        const Value next = Numeric::Decode(ReadMutable(src_, StorageOffset(source, false)));
        Value value = Numeric::Decode(ReadMutable(out_, StorageOffset(destination, true)));
        const bool changed = Operation::Update(value, next);
        if constexpr (Operation::extreme)
            if (!changed)
                return;
        values_.Get<Raw>().SetValue(0, Numeric::Encode(value));
        arguments_.Get<int64_t>().SetValue(0, row);
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(out_[StorageOffset(destination, true)], values_.Get<Raw>(), {1, sizeof(Raw), 0, 0, 0});
        if constexpr (Operation::extreme)
            if (changed)
                DataCopyPad(arg_[destination], arguments_.Get<int64_t>(), {1, 8, 0, 0, 0});
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::MTE3_S>();
    }
    __aicore__ inline void ProcessAlias() {
        InitializeAliasArgs();
        for (uint64_t source = 0; source < t_->srcElements; ++source)
            AliasUpdate(source);
    }
    const Tiling *t_;
    GlobalTensor<Raw> src_, out_;
    GlobalTensor<int64_t> index_, arg_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> values_, arguments_, source_, indexBuffer_;
};
} // namespace opsgnn::scatter_a2

namespace opsgnn::scatter_a2 {
// Integer bucket sums and the internal histogram for fused floating mean use
// an INT32 workspace. All source-value floating sums stay in original row order.
template <uint32_t Kind, bool CountOnly = false> class BucketScatter {
    using Numeric = Number<Kind>;
    using Raw = typename Numeric::Raw;
    static_assert(CountOnly || !Numeric::floating, "floating source sums require ordered owners");

  public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR workspace, const Tiling &t) {
        t_ = &t;
        if constexpr (!CountOnly) {
            src_.SetGlobalBuffer(reinterpret_cast<__gm__ Raw *>(src), t.srcElements);
            out_.SetGlobalBuffer(reinterpret_cast<__gm__ Raw *>(out), t.outElements);
        }
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), t.indexElements);
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace),
                                   CountOnly ? t.buckets : t.outElements);
        pipe_.InitBuffer(table_, t.atomicWidth * sizeof(uint32_t));
        pipe_.InitBuffer(indices_, kBucketSourceRows * sizeof(int64_t));
        if constexpr (!CountOnly) {
            pipe_.InitBuffer(source_, kBucketSourceRows * sizeof(Raw));
            pipe_.InitBuffer(output_, kBucketConversionElements * sizeof(uint32_t));
        }
    }
    __aicore__ inline void Process(uint32_t stage) {
        if constexpr (CountOnly) {
            for (uint64_t first = 0; first < t_->buckets; first += t_->atomicWidth)
                ReduceGroup(first);
        } else {
            if (stage == 1) {
                for (uint64_t first = 0; first < t_->buckets; first += t_->atomicWidth)
                    ReduceGroup(first);
            } else {
                const uint64_t tiles = (t_->outElements + kBucketConversionElements - 1) / kBucketConversionElements;
                for (uint64_t tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum())
                    ConvertTile(tile, stage);
            }
        }
    }

  private:
    __aicore__ inline uint32_t WorkBits(Raw source) { return static_cast<uint32_t>(Numeric::Decode(source)); }
    __aicore__ inline Raw OutputBits(uint32_t work) { return Numeric::Encode(static_cast<int32_t>(work)); }
    __aicore__ inline void ConvertTile(uint64_t tile, uint32_t stage) {
        const uint64_t first = tile * kBucketConversionElements;
        const uint32_t count = Smaller(kBucketConversionElements, t_->outElements - first);
        auto source = source_.Get<Raw>();
        auto output = output_.Get<uint32_t>();
        if (stage == 0 && t_->hasOut)
            DataCopyPad(source, out_[first], {1, count * static_cast<uint32_t>(sizeof(Raw)), 0, 0, 0},
                        DataCopyPadExtParams<Raw>(false, 0, 0, 0));
        if (stage == 2)
            DataCopyPad(output, workspace_[first], {1, count * 4, 0, 0, 0},
                        DataCopyPadExtParams<uint32_t>(false, 0, 0, 0));
        Sync<HardEvent::MTE2_S>();
        for (uint32_t lane = 0; lane < count; ++lane) {
            if (stage == 0)
                output.SetValue(lane, t_->hasOut ? WorkBits(source.GetValue(lane)) : 0U);
            else
                source.SetValue(lane, OutputBits(output.GetValue(lane)));
        }
        Sync<HardEvent::S_MTE3>();
        if (stage == 0)
            DataCopyPad(workspace_[first], output, {1, count * 4, 0, 0, 0});
        else
            DataCopyPad(out_[first], source, {1, count * static_cast<uint32_t>(sizeof(Raw)), 0, 0, 0});
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::MTE3_S>();
    }
    __aicore__ inline void AddRows(uint64_t bucketFirst, uint32_t buckets, uint32_t count) {
        auto table = table_.Get<uint32_t>();
        auto indices = indices_.Get<int64_t>();
        for (uint32_t lane = 0; lane < count; ++lane) {
            const int64_t bucket = indices.GetValue(t_->indexElements == 1 ? 0 : lane);
            if (bucket < static_cast<int64_t>(bucketFirst) || static_cast<uint64_t>(bucket) >= bucketFirst + buckets)
                continue;
            const uint32_t slot = bucket - bucketFirst;
            const uint32_t old = table.GetValue(slot);
            if constexpr (CountOnly)
                table.SetValue(slot, old + 1);
            else {
                const uint32_t next = WorkBits(source_.Get<Raw>().GetValue(lane));
                table.SetValue(slot, old + next);
            }
        }
    }
    __aicore__ inline void ReadRows(uint64_t first, uint32_t count) {
        if constexpr (!CountOnly)
            DataCopyPad(source_.Get<Raw>(), src_[first], {1, count * static_cast<uint32_t>(sizeof(Raw)), 0, 0, 0},
                        DataCopyPadExtParams<Raw>(false, 0, 0, 0));
        const uint32_t indices = t_->indexElements == 1 ? 1 : count;
        DataCopyPad(indices_.Get<int64_t>(), index_[t_->indexElements == 1 ? 0 : first], {1, indices * 8, 0, 0, 0},
                    DataCopyPadExtParams<int64_t>(false, 0, 0, 0));
        Sync<HardEvent::MTE2_S>();
    }
    __aicore__ inline void ReduceGroup(uint64_t bucketFirst) {
        const uint32_t buckets = Smaller(t_->atomicWidth, t_->buckets - bucketFirst);
        Duplicate(table_.Get<int32_t>(), 0, t_->atomicWidth);
        Sync<HardEvent::V_S>();
        const uint64_t begin = Partition(t_->length, GetBlockIdx(), GetBlockNum());
        const uint64_t end = Partition(t_->length, GetBlockIdx() + 1, GetBlockNum());
        for (uint64_t first = begin; first < end; first += kBucketSourceRows) {
            const uint32_t count = Smaller(kBucketSourceRows, end - first);
            ReadRows(first, count);
            AddRows(bucketFirst, buckets, count);
            Sync<HardEvent::S_MTE2>();
        }
        Sync<HardEvent::S_MTE3>();
        SetAtomicAdd<int32_t>();
        DataCopyPad(workspace_[bucketFirst], table_.Get<uint32_t>(), {1, buckets * 4, 0, 0, 0});
        Sync<HardEvent::MTE3_S>();
        SetAtomicNone();
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::S_V>();
    }
    const Tiling *t_;
    GlobalTensor<Raw> src_, out_;
    GlobalTensor<int64_t> index_;
    GlobalTensor<uint32_t> workspace_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> table_, source_, indices_, output_;
};
} // namespace opsgnn::scatter_a2

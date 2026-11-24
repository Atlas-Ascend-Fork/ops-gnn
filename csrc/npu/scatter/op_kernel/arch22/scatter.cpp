/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "scatter.h"
#include "scatter_common.h"
#include "scatter_grouped.h"
#include "scatter_vector.h"

namespace opsgnn::scatter_a2 {
template <uint32_t Kind, uint32_t Reduction>
__global__ __aicore__ void ScatterScalarKernel(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, Tiling t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ScalarScatter<Kind, Reduction> kernel;
    kernel.Init(src, index, out, arg, t);
    kernel.Process();
}
template <uint32_t Kind, uint32_t Reduction>
__global__ __aicore__ void ScatterVectorKernel(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR arg, GM_ADDR workspace,
                                               GM_ADDR argWorkspace, GM_ADDR counts, Tiling t, uint32_t stage) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    VectorScatter<Kind, Reduction> kernel;
    kernel.Init(src, index, out, arg, workspace, argWorkspace, counts, t);
    kernel.Process(stage);
}
template <uint32_t Kind, uint32_t Reduction>
__global__ __aicore__ void ScatterAtomicKernel(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR workspace,
                                               GM_ADDR argWorkspace, Tiling t, bool arguments) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AtomicScatter<Kind, Reduction> kernel;
    kernel.Init(src, index, out, workspace, argWorkspace, t, arguments);
    kernel.Process();
}
template <uint32_t Kind>
__global__ __aicore__ void ScatterBucketKernel(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR workspace, Tiling t,
                                               uint32_t stage) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BucketScatter<Kind> kernel;
    kernel.Init(src, index, out, workspace, t);
    kernel.Process(stage);
}
template <uint32_t Kind>
__global__ __aicore__ void ScatterMeanCountKernel(GM_ADDR src, GM_ADDR index, GM_ADDR out, GM_ADDR counts, Tiling t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    t.length = t.indexElements;
    t.atomicWidth = t.countWidth;
    BucketScatter<Kind, true> kernel;
    kernel.Init(src, index, out, counts, t);
    kernel.Process(1);
}
struct Launch {
    void *src, *index, *out, *arg, *workspace, *argWorkspace, *counts;
    const Tiling &tiling;
    aclrtStream stream;
};
template <uint32_t Kind, uint32_t Reduction> void LaunchVector(const Launch &launch, uint32_t stage) {
    const auto &t = launch.tiling;
    ScatterVectorKernel<Kind, Reduction><<<t.blocks, nullptr, launch.stream>>>(
        (GM_ADDR)launch.src, (GM_ADDR)launch.index, (GM_ADDR)launch.out, (GM_ADDR)launch.arg, (GM_ADDR)launch.workspace,
        (GM_ADDR)launch.argWorkspace, (GM_ADDR)launch.counts, t, stage);
}
template <uint32_t Kind, uint32_t Reduction> void LaunchAtomic(const Launch &launch) {
    const auto &t = launch.tiling;
    static_assert(Reduction == Min || Reduction == Max, "only extrema use the atomic path");
    LaunchVector<Kind, Reduction>(launch, 0);
    ScatterAtomicKernel<Kind, Reduction>
        <<<t.blocks, nullptr, launch.stream>>>((GM_ADDR)launch.src, (GM_ADDR)launch.index, (GM_ADDR)launch.out,
                                               (GM_ADDR)launch.workspace, (GM_ADDR)launch.argWorkspace, t, false);
    if constexpr (Reduction == Min || Reduction == Max) {
        Tiling arguments = t;
        arguments.atomicRows = t.argumentRows;
        arguments.units = t.before * ((t.after + t.atomicWidth - 1) / t.atomicWidth) *
                          ((t.length + t.argumentRows - 1) / t.argumentRows);
        ScatterAtomicKernel<Kind, Reduction><<<t.blocks, nullptr, launch.stream>>>(
            (GM_ADDR)launch.src, (GM_ADDR)launch.index, (GM_ADDR)launch.out, (GM_ADDR)launch.workspace,
            (GM_ADDR)launch.argWorkspace, arguments, true);
    }
    LaunchVector<Kind, Reduction>(launch, 2);
}
template <uint32_t Kind, uint32_t Reduction> void LaunchReduction(const Launch &launch) {
    const auto &t = launch.tiling;
    if constexpr (Reduction == Sum && Kind > Float32) {
        if (t.path == BucketTable) {
            for (uint32_t stage = 0; stage < 3; ++stage)
                ScatterBucketKernel<Kind>
                    <<<t.blocks, nullptr, launch.stream>>>((GM_ADDR)launch.src, (GM_ADDR)launch.index,
                                                           (GM_ADDR)launch.out, (GM_ADDR)launch.workspace, t, stage);
            return;
        }
    }
    if constexpr (Kind <= Float32) {
        if constexpr (Reduction == Min || Reduction == Max) {
            if (t.path == Atomic) {
                LaunchAtomic<Kind, Reduction>(launch);
                return;
            }
        }
        if (t.path == FeatureOwner) {
            const uint32_t first = Kind == Float32 && Reduction == Sum && t.hasOut ? 1 : 0;
            const uint32_t last = Kind == Float32 && Reduction == Sum ? 2 : 3;
            for (uint32_t stage = first; stage < last; ++stage)
                LaunchVector<Kind, Reduction>(launch, stage);
            return;
        }
    }
    ScatterScalarKernel<Kind, Reduction><<<t.blocks, nullptr, launch.stream>>>(
        (GM_ADDR)launch.src, (GM_ADDR)launch.index, (GM_ADDR)launch.out, (GM_ADDR)launch.arg, t);
}
template <uint32_t Kind> void LaunchMean(const Launch &launch) {
    const auto &t = launch.tiling;
    LaunchVector<Kind, Sum>(launch, 0);
    ScatterMeanCountKernel<Kind><<<t.blocks, nullptr, launch.stream>>>((GM_ADDR)launch.src, (GM_ADDR)launch.index,
                                                                       (GM_ADDR)launch.out, (GM_ADDR)launch.counts, t);
    LaunchVector<Kind, Sum>(launch, 1);
    LaunchVector<Kind, Sum>(launch, 2);
}
template <uint32_t Kind> void LaunchDtype(const Launch &launch) {
    switch (launch.tiling.reduce) {
    case Sum:
        LaunchReduction<Kind, Sum>(launch);
        break;
    case Mean:
        if constexpr (Kind <= Float32)
            LaunchMean<Kind>(launch);
        break;
    case Mul:
        LaunchReduction<Kind, Mul>(launch);
        break;
    case Min:
        LaunchReduction<Kind, Min>(launch);
        break;
    case Max:
        LaunchReduction<Kind, Max>(launch);
        break;
    }
}
} // namespace opsgnn::scatter_a2

aclError ScatterA2(void *src, int64_t *index, void *out, int64_t *arg, float *workspace, float *argWorkspace,
                   int32_t *counts, const opsgnn::scatter_a2::Tiling &t, aclrtStream stream) {
    using namespace opsgnn::scatter_a2;
    const Launch launch{src, index, out, arg, workspace, argWorkspace, counts, t, stream};
    switch (t.dtype) {
    case Float16:
        LaunchDtype<Float16>(launch);
        break;
    case BFloat16:
        LaunchDtype<BFloat16>(launch);
        break;
    case Float32:
        LaunchDtype<Float32>(launch);
        break;
    case Int8:
        LaunchDtype<Int8>(launch);
        break;
    case Int16:
        LaunchDtype<Int16>(launch);
        break;
    case Int32:
        LaunchDtype<Int32>(launch);
        break;
    case UInt8:
        LaunchDtype<UInt8>(launch);
        break;
    default:
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

namespace opsgnn::scatter_a2 {
constexpr uint32_t kEqualTileElements = 512;
constexpr uint32_t kWordsPerIndex = sizeof(int64_t) / sizeof(int32_t);
constexpr uint32_t kEqualWords = kEqualTileElements * kWordsPerIndex;
constexpr uint32_t kEqualMaskBytes = kEqualWords / 8;
constexpr uint32_t kEqualMaskWords = kEqualMaskBytes / sizeof(uint64_t);
static_assert(kEqualWords == 1024 && kEqualMaskBytes == 128 && kEqualMaskWords == 16);
constexpr uint32_t kEqualFlagsPerCore = kDmaBytes / sizeof(int32_t);
constexpr uint32_t kIndicesPerDmaBlock = kDmaBytes / sizeof(int64_t);

class IndexEqualShared {
  public:
    __aicore__ inline explicit IndexEqualShared(TPipe &pipe) : pipe_(pipe) {}
    __aicore__ inline void Run(GM_ADDR index, GM_ADDR snapshot, GM_ADDR flags, uint64_t elements) {
        RunMirror(index, snapshot, flags, nullptr, elements);
    }
    __aicore__ inline void RunMirror(GM_ADDR index, GM_ADDR snapshot, GM_ADDR flags, GM_ADDR mappedFlags,
                                     uint64_t elements) {
        mirror_ = mappedFlags != nullptr;
        if (mirror_)
            mappedFlags_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mappedFlags),
                                         static_cast<uint64_t>(GetBlockNum()) * kEqualFlagsPerCore);
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), elements);
        snapshot_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(snapshot), elements);
        flags_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flags),
                               static_cast<uint64_t>(GetBlockNum()) * kEqualFlagsPerCore);
        // Two 4096-byte raw tiles, a 128-byte predicate mask and a
        // private 32-byte flag block: 8352 UB bytes. No float workspace.
        pipe_.InitBuffer(inputBuffer_, kEqualTileElements * sizeof(int64_t));
        pipe_.InitBuffer(snapshotBuffer_, kEqualTileElements * sizeof(int64_t));
        pipe_.InitBuffer(maskBuffer_, kEqualMaskBytes);
        pipe_.InitBuffer(flagBuffer_, kDmaBytes);
        const uint64_t first = Partition(elements, GetBlockIdx(), GetBlockNum());
        const uint64_t last = Partition(elements, GetBlockIdx() + 1, GetBlockNum());
        bool equal = true;
        for (uint64_t offset = first; equal && offset < last;) {
            const uint32_t valid = Smaller(kEqualTileElements, last - offset);
            equal = CheckTile(offset, valid);
            offset += valid;
        }
        WriteFlag(equal);
    }

  private:
    __aicore__ inline void LoadTile(uint64_t offset, uint32_t valid) {
        // Initialize the complete comparison extent, not just the final DMA
        // block. Padding beyond the valid input must compare as zero to zero.
        Duplicate(inputBuffer_.Get<int32_t>(), 0, kEqualTileElements * 2);
        Duplicate(snapshotBuffer_.Get<int32_t>(), 0, kEqualTileElements * 2);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_MTE2>();
        const uint8_t padding = (kIndicesPerDmaBlock - valid % kIndicesPerDmaBlock) % kIndicesPerDmaBlock;
        const auto parameters = DataCopyPadExtParams<int64_t>(true, 0, padding, 0);
        DataCopyPad(inputBuffer_.Get<int64_t>(), input_[offset], {1, valid * 8, 0, 0, 0}, parameters);
        DataCopyPad(snapshotBuffer_.Get<int64_t>(), snapshot_[offset], {1, valid * 8, 0, 0, 0}, parameters);
        Sync<HardEvent::MTE2_V>();
    }

    __aicore__ inline bool CheckTile(uint64_t offset, uint32_t valid) {
        LoadTile(offset, valid);
        // Reinterpret each INT64 as two INT32 limbs. Integer EQ is exact for
        // every bit pattern, including INT64 values outside the FP64 range
        // of consecutive integers; neither limb is numerically converted.
        // CANN 9.1 dav_c220 implements int32 Compare EQ using vcmpv_eq.
        Compare(maskBuffer_.Get<uint8_t>(), inputBuffer_.Get<int32_t>(), snapshotBuffer_.Get<int32_t>(), CMPMODE::EQ,
                kEqualWords);
        Sync<HardEvent::V_S>();
        auto mask = maskBuffer_.Get<uint64_t>();
        bool equal = true;
        for (uint32_t word = 0; equal && word < kEqualMaskWords; ++word)
            equal = mask.GetValue(word) == UINT64_MAX;
        Sync<HardEvent::S_V>();
        return equal;
    }

    __aicore__ inline void WriteFlag(bool equal) {
        auto flags = flagBuffer_.Get<int32_t>();
        for (uint32_t lane = 0; lane < kEqualFlagsPerCore; ++lane)
            flags.SetValue(lane, lane == 0 && equal ? 1 : 0);
        Sync<HardEvent::S_MTE3>();
        const uint64_t offset = static_cast<uint64_t>(GetBlockIdx()) * kEqualFlagsPerCore;
        DataCopy(flags_[offset], flags, kEqualFlagsPerCore);
        if (mirror_)
            DataCopy(mappedFlags_[offset], flags, kEqualFlagsPerCore);
        Sync<HardEvent::MTE3_S>();
    }

    GlobalTensor<int64_t> input_, snapshot_;
    GlobalTensor<int32_t> flags_, mappedFlags_;
    bool mirror_ = false;
    TPipe &pipe_;
    TBuf<TPosition::VECCALC> inputBuffer_, snapshotBuffer_, maskBuffer_, flagBuffer_;
};

} // namespace opsgnn::scatter_a2

namespace opsgnn::scatter_a2 {
namespace {
__global__ __aicore__ void IndexEqualKernel(GM_ADDR index, GM_ADDR snapshot, GM_ADDR flags, uint64_t elements) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IndexEqualShared kernel(pipe);
    kernel.Run(index, snapshot, flags, elements);
}
__global__ __aicore__ void IndexEqualMappedKernel(GM_ADDR index, GM_ADDR snapshot, GM_ADDR flags, GM_ADDR mappedFlags,
                                                  uint64_t elements) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IndexEqualShared kernel(pipe);
    kernel.RunMirror(index, snapshot, flags, mappedFlags, elements);
}
} // namespace
} // namespace opsgnn::scatter_a2

aclError ScatterA2IndexEqual(int64_t *index, int64_t *snapshot, uint64_t elements, int32_t *flags, uint32_t blocks,
                             aclrtStream stream) {
    opsgnn::scatter_a2::IndexEqualKernel<<<blocks, nullptr, stream>>>((GM_ADDR)index, (GM_ADDR)snapshot, (GM_ADDR)flags,
                                                                      elements);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

aclError ScatterA2IndexEqualMapped(int64_t *index, int64_t *snapshot, uint64_t elements, int32_t *flags,
                                   int32_t *mappedFlags, uint32_t blocks, aclrtStream stream) {
    opsgnn::scatter_a2::IndexEqualMappedKernel<<<blocks, nullptr, stream>>>(
        (GM_ADDR)index, (GM_ADDR)snapshot, (GM_ADDR)flags, (GM_ADDR)mappedFlags, elements);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

namespace opsgnn::scatter_a2 {
namespace {

class IndexStats {
  public:
    __aicore__ inline void Run(GM_ADDR index, GM_ADDR stats, uint64_t count) {
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), count);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(stats), GetBlockNum() * kIndexStatsElementsPerCore);
        pipe_.InitBuffer(inputBuffer_, kIndexStatsTileElements * sizeof(int64_t));
        pipe_.InitBuffer(outputBuffer_, kDmaBytes);
        int64_t minimum = INT64_MAX, maximum = INT64_MIN;
        const uint64_t first = Partition(count, GetBlockIdx(), GetBlockNum());
        const uint64_t last = Partition(count, GetBlockIdx() + 1, GetBlockNum());
        for (uint64_t offset = first; offset < last; offset += kIndexStatsTileElements) {
            const uint32_t valid = Smaller(kIndexStatsTileElements, last - offset);
            DataCopyPad(inputBuffer_.Get<int64_t>(), input_[offset], {1, valid * 8, 0, 0, 0},
                        DataCopyPadExtParams<int64_t>(false, 0, 0, 0));
            Sync<HardEvent::MTE2_S>();
            Accumulate(valid, minimum, maximum);
            Sync<HardEvent::S_MTE2>();
        }
        auto result = outputBuffer_.Get<int64_t>();
        result.SetValue(0, minimum);
        result.SetValue(1, maximum);
        result.SetValue(2, 0);
        result.SetValue(3, 0);
        Sync<HardEvent::S_MTE3>();
        DataCopy(output_[GetBlockIdx() * kIndexStatsElementsPerCore], result, kIndexStatsElementsPerCore);
    }

  private:
    __aicore__ inline void Accumulate(uint32_t count, int64_t &minimum, int64_t &maximum) {
        auto input = inputBuffer_.Get<int64_t>();
        for (uint32_t lane = 0; lane < count; ++lane) {
            const int64_t value = input.GetValue(lane);
            minimum = value < minimum ? value : minimum;
            maximum = value > maximum ? value : maximum;
        }
    }
    GlobalTensor<int64_t> input_, output_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputBuffer_, outputBuffer_;
};

__global__ __aicore__ void IndexStatsKernel(GM_ADDR index, GM_ADDR stats, uint64_t elements) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    IndexStats kernel;
    kernel.Run(index, stats, elements);
}
} // namespace
} // namespace opsgnn::scatter_a2

aclError ScatterA2IndexStats(int64_t *index, uint64_t elements, int64_t *stats, uint32_t blocks, aclrtStream stream) {
    opsgnn::scatter_a2::IndexStatsKernel<<<blocks, nullptr, stream>>>((GM_ADDR)index, (GM_ADDR)stats, elements);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

namespace opsgnn::scatter_a2 {
__global__ __aicore__ void FusedPrivateSumKernel(GM_ADDR src, GM_ADDR index, GM_ADDR snapshot, GM_ADDR rowptr,
                                                 GM_ADDR rows, GM_ADDR out, GM_ADDR flags, GM_ADDR mappedFlags,
                                                 GroupTiling t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IndexEqualShared equality(pipe);
    equality.RunMirror(index, snapshot, flags, mappedFlags, t.indexElements);
    // No cross-core flags are consumed inside this kernel. Only the private
    // output is written; the Host publishes it after full completion and EQ.
    GroupedScatter<Float32, Sum, false, true> owner(pipe);
    owner.Run(src, rowptr, rows, out, nullptr, nullptr, t);
}
} // namespace opsgnn::scatter_a2

aclError ScatterA2FusedPrivateSum(void *src, int64_t *index, int64_t *snapshot, int64_t *rowptr, int64_t *rows,
                                  void *out, int32_t *flags, int32_t *mappedFlags,
                                  const opsgnn::scatter_a2::GroupTiling &tiling, aclrtStream stream) {
    opsgnn::scatter_a2::FusedPrivateSumKernel<<<tiling.blocks, nullptr, stream>>>(
        (GM_ADDR)src, (GM_ADDR)index, (GM_ADDR)snapshot, (GM_ADDR)rowptr, (GM_ADDR)rows, (GM_ADDR)out, (GM_ADDR)flags,
        (GM_ADDR)mappedFlags, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

namespace opsgnn::scatter_a2 {
template <uint32_t Kind, uint32_t Reduction, bool Scalar>
__global__ __aicore__ void GroupedScatterKernel(GM_ADDR src, GM_ADDR rowptr, GM_ADDR rows, GM_ADDR out, GM_ADDR arg,
                                                GM_ADDR validation, GroupTiling t) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GroupedScatter<Kind, Reduction, Scalar> kernel(pipe);
    kernel.Run(src, rowptr, rows, out, arg, validation, t);
}
struct GroupLaunch {
    void *src, *rowptr, *rows, *out, *arg, *validation;
    const GroupTiling &tiling;
    aclrtStream stream;
};
template <uint32_t Kind, uint32_t Reduction> void LaunchGrouped(const GroupLaunch &launch) {
    const auto &t = launch.tiling;
    constexpr bool Integer = Kind > Float32;
    constexpr bool Extreme = Reduction == Min || Reduction == Max;
    if constexpr (Integer) {
        GroupedScatterKernel<Kind, Reduction, true><<<t.blocks, nullptr, launch.stream>>>(
            (GM_ADDR)launch.src, (GM_ADDR)launch.rowptr, (GM_ADDR)launch.rows, (GM_ADDR)launch.out, (GM_ADDR)launch.arg,
            (GM_ADDR)launch.validation, t);
    } else {
        if constexpr (Extreme) {
            if (t.length > kExactFloatIndexLimit) {
                GroupedScatterKernel<Kind, Reduction, true><<<t.blocks, nullptr, launch.stream>>>(
                    (GM_ADDR)launch.src, (GM_ADDR)launch.rowptr, (GM_ADDR)launch.rows, (GM_ADDR)launch.out,
                    (GM_ADDR)launch.arg, (GM_ADDR)launch.validation, t);
                return;
            }
        }
        GroupedScatterKernel<Kind, Reduction, false><<<t.blocks, nullptr, launch.stream>>>(
            (GM_ADDR)launch.src, (GM_ADDR)launch.rowptr, (GM_ADDR)launch.rows, (GM_ADDR)launch.out, (GM_ADDR)launch.arg,
            (GM_ADDR)launch.validation, t);
    }
}
template <uint32_t Kind> void DispatchGrouped(const GroupLaunch &launch) {
    switch (launch.tiling.reduce) {
    case Sum:
        LaunchGrouped<Kind, Sum>(launch);
        break;
    case Mul:
        LaunchGrouped<Kind, Mul>(launch);
        break;
    case Mean:
        LaunchGrouped<Kind, Mean>(launch);
        break;
    case Min:
        LaunchGrouped<Kind, Min>(launch);
        break;
    case Max:
        LaunchGrouped<Kind, Max>(launch);
        break;
    }
}
} // namespace opsgnn::scatter_a2

aclError ScatterA2Grouped(void *src, int64_t *rowptr, int64_t *rows, void *out, int64_t *arg, int32_t *validation,
                          const opsgnn::scatter_a2::GroupTiling &tiling, aclrtStream stream) {
    using namespace opsgnn::scatter_a2;
    const GroupLaunch launch{src, rowptr, rows, out, arg, validation, tiling, stream};
    switch (tiling.dtype) {
    case Float16:
        DispatchGrouped<Float16>(launch);
        break;
    case BFloat16:
        DispatchGrouped<BFloat16>(launch);
        break;
    case Float32:
        DispatchGrouped<Float32>(launch);
        break;
    case Int8:
        DispatchGrouped<Int8>(launch);
        break;
    case Int16:
        DispatchGrouped<Int16>(launch);
        break;
    case Int32:
        DispatchGrouped<Int32>(launch);
        break;
    case UInt8:
        DispatchGrouped<UInt8>(launch);
        break;
    default:
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

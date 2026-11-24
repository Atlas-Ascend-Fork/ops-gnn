/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SCATTER_SOFTMAX_KERNEL_ENTRY_H
#define OPS_GNN_SCATTER_SOFTMAX_KERNEL_ENTRY_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_io.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_phase.h"

/**
 * Kernel-entry plumbing shared by the grouped softmax, log-sum-exp and standard
 * deviation operators: the per-phase global-tensor binding and the
 * buffer/pipeline bootstrap around the row driver.
 *
 * Every entry uses one parameter list so the boilerplate lives here once; the
 * arguments a phase does not use are null and are never bound. `Mode` selects
 * the arithmetic of the phase bodies (see scatter_softmax_kernel_phase.h), so
 * scatter_softmax, scatter_log_softmax, scatter_logsumexp and scatter_std share
 * this file and differ only in the kernels they instantiate.
 */

namespace ScatterSoftmaxNpu {

// The launch API takes untyped addresses, so every pointer a launcher forwards
// is converted the same way.
inline GM_ADDR AsGmAddr(const void* pointer)
{
    return const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(pointer));
}

// Which reduction arrays a pass reads. The questions are separate constant
// expressions so the binding below is a short sequence of guarded calls instead
// of one long condition per array.
//
// The log-domain normalization is recomputed from the source, so that pass
// reads `src` and the maximum as well; the softmax path only needs the
// exponential it stored in the previous pass.
template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode>
__aicore__ constexpr bool BindsSource()
{
    if constexpr (Phase == kPhaseMaximum || Phase == kPhaseExp || Phase == kPhaseSum ||
                  Phase == kPhaseDeviation || Phase == kPhaseGradElement) {
        return true;
    }
    return IsLogDomain(Mode) && Phase == kPhaseNormalize;
}

template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode>
__aicore__ constexpr bool BindsMaximum()
{
    if constexpr (Phase == kPhaseMaximum || Phase == kPhaseExp) {
        return true;
    }
    // The standard deviation has its own finalize pass and never reads it.
    if constexpr (Mode == kModeStd) {
        return false;
    }
    return Phase == kPhaseFinalize || (IsLogDomain(Mode) && Phase == kPhaseNormalize);
}

// Every pass except the maximum reads the group sums: the softmax family reads
// the reduction it accumulated, and the standard deviation reads the group sums
// in its deviation pass and the mean the finalize pass derived from them.
template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode>
__aicore__ constexpr bool BindsSum()
{
    if constexpr (Phase == kPhaseGradElement) {
        return Mode == kModeStd;
    }
    return Phase != kPhaseMaximum;
}

template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode>
__aicore__ constexpr bool BindsDeviation()
{
    return Mode == kModeStd && (Phase == kPhaseDeviation || Phase == kPhaseFinalize);
}

template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode>
__aicore__ constexpr bool BindsCount()
{
    return Mode == kModeStd && (Phase == kPhaseSum || Phase == kPhaseDeviation ||
                                Phase == kPhaseFinalize || Phase == kPhaseGradElement);
}

// Binds the tensors a pass reduces into, or reads the reduction workspace from.
template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void BindReductionTensors(PhaseTensors<T>& tensors,
                                            const ScatterSoftmaxTilingData& tiling,
                                            GM_ADDR src, GM_ADDR index,
                                            GM_ADDR maximum, GM_ADDR sum,
                                            GM_ADDR deviation, GM_ADDR count)
{
    // The grouped operators index the labels of every row; the edge softmax
    // indexes the CSR boundaries of every segment, one more than the segments.
    const uint64_t labelElems =
        (Mode == kModeEdgeCsr) ? tiling.groupCount + 1 : tiling.dimLength;
    tensors.indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index), labelElems);
    if constexpr (BindsSource<Phase, Mode>()) {
        tensors.srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src), tiling.srcNumel);
    }
    if constexpr (BindsMaximum<Phase, Mode>()) {
        tensors.maximumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(maximum),
                                          tiling.workspaceElems);
    }
    if constexpr (BindsSum<Phase, Mode>()) {
        tensors.sumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum), tiling.workspaceElems);
    }
    if constexpr (BindsDeviation<Phase, Mode>()) {
        tensors.deviationGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deviation),
                                            tiling.workspaceElems);
    }
    if constexpr (BindsCount<Phase, Mode>()) {
        tensors.countGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(count),
                                        tiling.countElems);
    }
}

// Binds the reduced result, its upstream gradient and the gradient output. The
// grouped softmax writes a `src`-shaped result while the grouped reductions
// write one row per (batch, group) pair, which is what `outElems` selects.
template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void BindResultTensors(PhaseTensors<T>& tensors,
                                         const ScatterSoftmaxTilingData& tiling,
                                         GM_ADDR out, GM_ADDR gradOut, GM_ADDR gradSrc)
{
    const uint64_t outElems = (Mode == kModeLogSumExp || Mode == kModeStd)
        ? tiling.outRows * tiling.after : tiling.srcNumel;
    if constexpr (Phase != kPhaseMaximum) {
        tensors.outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), outElems);
    }
    if constexpr (Phase == kPhaseGradSum || Phase == kPhaseGrad ||
                  Phase == kPhaseGradElement) {
        tensors.gradGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradOut), outElems);
    }
    if constexpr (Phase == kPhaseGrad || Phase == kPhaseGradElement) {
        tensors.gradSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradSrc), tiling.srcNumel);
    }
}

// Binds the global tensors a phase touches.
template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void BindTensors(PhaseTensors<T>& tensors,
                                   const ScatterSoftmaxTilingData& tiling,
                                   GM_ADDR src, GM_ADDR index, GM_ADDR maximum,
                                   GM_ADDR sum, GM_ADDR deviation, GM_ADDR count,
                                   GM_ADDR out, GM_ADDR gradOut, GM_ADDR gradSrc)
{
    BindReductionTensors<Phase, Mode, T>(tensors, tiling, src, index, maximum, sum,
                                         deviation, count);
    BindResultTensors<Phase, Mode, T>(tensors, tiling, out, gradOut, gradSrc);
}

// The buffers, the pipeline events and the row/batch driver are identical for
// all passes, so only the phase tag, the mode and the tensor list differ.
template <ScatterSoftmaxPhase Phase, ScatterSoftmaxMode Mode, typename T>
__aicore__ inline void RunPhase(GM_ADDR src, GM_ADDR index, GM_ADDR maximum, GM_ADDR sum,
                                GM_ADDR deviation, GM_ADDR count, GM_ADDR out,
                                GM_ADDR gradOut, GM_ADDR gradSrc,
                                ScatterSoftmaxTilingData tiling)
{
    if (!IsParticipating(tiling)) {
        return;
    }
    PhaseTensors<T> tensors;
    TPipe pipe;
    InitTensors<Phase, Mode>(tensors, pipe, tiling);
    BindTensors<Phase, Mode>(tensors, tiling, src, index, maximum, sum, deviation, count,
                             out, gradOut, gradSrc);
    PipelineSync sync;
    sync.Init(pipe);
    InitAccumulator<Phase>(tensors, tiling);
    LoadWorkspace<Phase, Mode>(tensors, tiling, sync);
    if constexpr (Phase == kPhaseGradElement && Mode != kModeStd) {
        LoadGroupCaches<T>(tensors, tiling, sync);
    }
    RunRows<Phase, Mode>(tensors, tiling, sync);
    FlushAccumulator<Phase>(tensors, tiling, sync);
    if constexpr (Phase == kPhaseSum) {
        FlushCounts(tensors, tiling, sync);
    }
    sync.Release(pipe);
}

// Compile-time dtype tags used to dispatch a launch body. The device compiler
// gives `bfloat16_t` no default constructor, so the tags are empty structs that
// only carry the element type instead of values of it.
template <typename T>
struct DTypeTag {
    using Type = T;
};

// Runs one dtype-specific launch body, with the element type recovered from the
// tag it receives. The two exported entry points of every operator share this
// switch instead of keeping a per-operator copy of it. It is host code - it only
// selects a kernel instantiation - so it must not carry the device annotation.
template <typename Body>
inline void DispatchDType(ScatterSoftmaxDType dtype, Body&& body)
{
    switch (dtype) {
        case SCATTER_SOFTMAX_F16:
            body(DTypeTag<half>{});
            break;
        case SCATTER_SOFTMAX_F32:
            body(DTypeTag<float>{});
            break;
        case SCATTER_SOFTMAX_BF16:
            body(DTypeTag<bfloat16_t>{});
            break;
        default:
            break;
    }
}

}  // namespace ScatterSoftmaxNpu

// ---------------------------------------------------------------------------
// Whole-operator generator.
//
// The phase kernels, their launcher and the two exported entry points are
// identical in shape for every grouped softmax operator: only the kernel names,
// the compile-time arithmetic mode and the launch namespace differ. Generating
// them from one description keeps the operators from drifting apart - the
// arithmetic itself stays in the shared phase bodies.
//
//   Prefix  kernel and export name prefix (Prefix##MaxKernel, Prefix##Forward)
//   Name    pass name inside that prefix
//   Phase   pass tag forwarded to the driver
//   Mode    compile-time arithmetic forwarded to the phase bodies
// ---------------------------------------------------------------------------
#define SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Name, Phase, Mode)                        \
    template <typename T>                                                                \
    __attribute__((aiv)) __global__ __aicore__ void Prefix##Name##Kernel(                \
        GM_ADDR src, GM_ADDR index, GM_ADDR maximum, GM_ADDR sum, GM_ADDR deviation,      \
        GM_ADDR count, GM_ADDR out, GM_ADDR gradOut, GM_ADDR gradSrc,                     \
        ScatterSoftmaxTilingData tiling)                                                  \
    {                                                                                    \
        RunPhase<Phase, Mode, T>(src, index, maximum, sum, deviation, count, out,          \
                                 gradOut, gradSrc, tiling);                                \
    }

// The maximum and the exponential sum: the two passes the softmax family and the
// grouped log-sum-exp run before their own last pass. The launcher bodies the
// macro is expanded in provide `srcBytes`, `indexBytes`, `maximumBytes`,
// `sumBytes`, `outBytes`, `tiling`, `stream` and the element type `T`.
#define SCATTER_GROUPED_REDUCE_PASSES(Prefix)                                            \
    Prefix##MaxKernel<T><<<tiling.blockNum, nullptr, stream>>>(                          \
        srcBytes, indexBytes, maximumBytes, nullptr, nullptr, nullptr, nullptr, nullptr,  \
        nullptr, tiling);                                                                \
    Prefix##ExpKernel<T><<<tiling.blockNum, nullptr, stream>>>(                          \
        srcBytes, indexBytes, maximumBytes, sumBytes, nullptr, nullptr, outBytes, nullptr, \
        nullptr, tiling);

// Exported forward entry: the dtype switch and the argument list are the same
// for every operator, so this wrapper is written once and expanded per prefix.
#define SCATTER_GROUPED_EXPORT_FORWARD(Prefix, Ns)                                       \
    namespace opsgnn {                                                                   \
    void Prefix##Forward(void* src, const int32_t* index, void* out,                     \
                         float* maximum, float* sumWorkspace,                             \
                         ScatterSoftmaxDType dtype,                                      \
                         const ScatterSoftmaxTilingData& tiling, aclrtStream stream)      \
    {                                                                                    \
        ScatterSoftmaxNpu::DispatchDType(dtype, [&](auto tag) {                           \
            using T = typename decltype(tag)::Type;                                       \
            ScatterSoftmaxNpu::Ns::Forward<T>(src, index, out, maximum, sumWorkspace,     \
                                              tiling, stream);                            \
        });                                                                              \
    }                                                                                    \
    }

#define SCATTER_GROUPED_SOFTMAX_OPERATOR(Prefix, Ns, Mode)                               \
    namespace ScatterSoftmaxNpu {                                                        \
    using namespace AscendC;                                                             \
                                                                                         \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Max, kPhaseMaximum, Mode)                     \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Exp, kPhaseExp, Mode)                         \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Normalize, kPhaseNormalize, Mode)             \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, GradSum, kPhaseGradSum, Mode)                 \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Grad, kPhaseGrad, Mode)                       \
                                                                                         \
    /* Every pass takes the same argument list; the phases a mode does not use */        \
    /* are never bound (see BindTensors), so one sequence serves every operator. */      \
    namespace Ns {                                                                       \
        template <typename T>                                                            \
        inline void Forward(void* src, const int32_t* index, void* out,                  \
                            float* maximum, float* sumWorkspace,                         \
                            const ScatterSoftmaxTilingData& tiling, aclrtStream stream)   \
        {                                                                                \
            GM_ADDR indexBytes = AsGmAddr(index);                                        \
            GM_ADDR srcBytes = AsGmAddr(src);                                            \
            GM_ADDR outBytes = AsGmAddr(out);                                            \
            GM_ADDR maximumBytes = AsGmAddr(maximum);                                    \
            GM_ADDR sumBytes = AsGmAddr(sumWorkspace);                                   \
            SCATTER_GROUPED_REDUCE_PASSES(Prefix)                                        \
            Prefix##NormalizeKernel<T><<<tiling.blockNum, nullptr, stream>>>(            \
                srcBytes, indexBytes, maximumBytes, sumBytes, nullptr, nullptr, outBytes, \
                nullptr, nullptr, tiling);                                               \
        }                                                                                \
                                                                                         \
        template <typename T>                                                            \
        inline void Backward(const void* out, const void* gradOut,                       \
                             const int32_t* index, void* gradSrc,                        \
                             float* sumWorkspace,                                        \
                             const ScatterSoftmaxTilingData& tiling,                     \
                             aclrtStream stream)                                         \
        {                                                                                \
            GM_ADDR indexBytes = AsGmAddr(index);                                        \
            GM_ADDR outBytes = AsGmAddr(out);                                            \
            GM_ADDR gradOutBytes = AsGmAddr(gradOut);                                    \
            GM_ADDR sumBytes = AsGmAddr(sumWorkspace);                                   \
            Prefix##GradSumKernel<T><<<tiling.blockNum, nullptr, stream>>>(              \
                nullptr, indexBytes, nullptr, sumBytes, nullptr, nullptr, outBytes,       \
                gradOutBytes, nullptr, tiling);                                          \
            Prefix##GradKernel<T><<<tiling.blockNum, nullptr, stream>>>(                 \
                nullptr, indexBytes, nullptr, sumBytes, nullptr, nullptr, outBytes,       \
                gradOutBytes, AsGmAddr(gradSrc), tiling);                                \
        }                                                                                \
    }  /* namespace Ns */                                                                \
    }  /* namespace ScatterSoftmaxNpu */                                                  \
                                                                                         \
    SCATTER_GROUPED_EXPORT_FORWARD(Prefix, Ns)                                            \
                                                                                         \
    namespace opsgnn {                                                                   \
    void Prefix##Backward(const void* out, const void* gradOut,                          \
                          const int32_t* index, void* gradSrc,                           \
                          float* sumWorkspace, ScatterSoftmaxDType dtype,                \
                          const ScatterSoftmaxTilingData& tiling,                        \
                          aclrtStream stream)                                            \
    {                                                                                    \
        ScatterSoftmaxNpu::DispatchDType(dtype, [&](auto tag) {                           \
            using T = typename decltype(tag)::Type;                                       \
            ScatterSoftmaxNpu::Ns::Backward<T>(                                          \
                out, gradOut, index, gradSrc, sumWorkspace, tiling, stream);             \
        });                                                                              \
    }                                                                                    \
    }  /* namespace opsgnn */

// ---------------------------------------------------------------------------
// Whole-operator generator of the grouped log-sum-exp.
//
// The reduced output has one row per (batch, group) pair instead of one row per
// source row, so its final pass runs on its own core split (`outRows`) and the
// forward launches three passes: maximum, exponential sum and finalize. The
// backward is a single elementwise pass and needs no reduction workspace.
// ---------------------------------------------------------------------------
#define SCATTER_GROUPED_LOGSUMEXP_OPERATOR(Prefix, Ns)                                     \
    namespace ScatterSoftmaxNpu {                                                          \
    using namespace AscendC;                                                               \
                                                                                           \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Max, kPhaseMaximum, kModeLogSumExp)             \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Exp, kPhaseExp, kModeLogSumExp)                 \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Finalize, kPhaseFinalize, kModeLogSumExp)       \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, GradElement, kPhaseGradElement, kModeLogSumExp) \
                                                                                           \
    namespace Ns {                                                                         \
        template <typename T>                                                              \
        inline void Forward(void* src, const int32_t* index, void* out,                    \
                            float* maximum, float* sumWorkspace,                           \
                            const ScatterSoftmaxTilingData& tiling, aclrtStream stream)    \
        {                                                                                  \
            /* The reduced output is one row per (batch, group), so the finalize */        \
            /* pass walks `outRows` on its own core split. */                              \
            ScatterSoftmaxTilingData finalizeTiling = tiling;                              \
            finalizeTiling.rows = tiling.outRows;                                          \
            finalizeTiling.rowChunk = tiling.outRowChunk;                                  \
            finalizeTiling.rowBatch = tiling.outRowBatch;                                  \
            finalizeTiling.blockNum = tiling.outBlockNum;                                  \
            GM_ADDR indexBytes = AsGmAddr(index);                                          \
            GM_ADDR srcBytes = AsGmAddr(src);                                              \
            GM_ADDR outBytes = AsGmAddr(out);                                              \
            GM_ADDR maximumBytes = AsGmAddr(maximum);                                      \
            GM_ADDR sumBytes = AsGmAddr(sumWorkspace);                                     \
            SCATTER_GROUPED_REDUCE_PASSES(Prefix)                                          \
            Prefix##FinalizeKernel<T><<<tiling.outBlockNum, nullptr, stream>>>(            \
                nullptr, indexBytes, maximumBytes, sumBytes, nullptr, nullptr, outBytes,   \
                nullptr, nullptr, finalizeTiling);                                         \
        }                                                                                  \
                                                                                           \
        template <typename T>                                                              \
        inline void Backward(const void* src, const void* out, const void* gradOut,        \
                             const int32_t* index, void* gradSrc,                          \
                             const ScatterSoftmaxTilingData& tiling,                       \
                             aclrtStream stream)                                           \
        {                                                                                  \
            GM_ADDR indexBytes = AsGmAddr(index);                                          \
            GM_ADDR srcBytes = AsGmAddr(src);                                              \
            GM_ADDR outBytes = AsGmAddr(out);                                              \
            GM_ADDR gradOutBytes = AsGmAddr(gradOut);                                      \
            Prefix##GradElementKernel<T><<<tiling.blockNum, nullptr, stream>>>(            \
                srcBytes, indexBytes, nullptr, nullptr, nullptr, nullptr, outBytes,         \
                gradOutBytes, AsGmAddr(gradSrc), tiling);                                   \
        }                                                                                  \
    }  /* namespace Ns */                                                                  \
    }  /* namespace ScatterSoftmaxNpu */                                                   \
                                                                                           \
    SCATTER_GROUPED_EXPORT_FORWARD(Prefix, Ns)                                             \
                                                                                           \
    namespace opsgnn {                                                                     \
    void Prefix##Backward(const void* src, const void* out, const void* gradOut,           \
                          const int32_t* index, void* gradSrc,                             \
                          ScatterSoftmaxDType dtype,                                       \
                          const ScatterSoftmaxTilingData& tiling, aclrtStream stream)      \
    {                                                                                      \
        ScatterSoftmaxNpu::DispatchDType(dtype, [&](auto tag) {                            \
            using T = typename decltype(tag)::Type;                                        \
            ScatterSoftmaxNpu::Ns::Backward<T>(src, out, gradOut, index, gradSrc,          \
                                               tiling, stream);                            \
        });                                                                                \
    }                                                                                      \
    }  /* namespace opsgnn */

// ---------------------------------------------------------------------------
// Whole-operator generator of the grouped standard deviation.
//
// The forward runs two passes over the source - a group sum that also counts
// the members of every group, and a centered squared-deviation sum - and
// finishes with one pass over the `outRows` group rows that writes
// `sqrt(deviation / (n - correction))` while keeping the mean in place of the
// group sum. The backward is a single elementwise pass.
// ---------------------------------------------------------------------------
#define SCATTER_GROUPED_STD_OPERATOR(Prefix, Ns)                                           \
    namespace ScatterSoftmaxNpu {                                                          \
    using namespace AscendC;                                                               \
                                                                                           \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Sum, kPhaseSum, kModeStd)                       \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Deviation, kPhaseDeviation, kModeStd)           \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Finalize, kPhaseFinalize, kModeStd)             \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, GradElement, kPhaseGradElement, kModeStd)       \
                                                                                           \
    namespace Ns {                                                                         \
        template <typename T>                                                              \
        inline void Forward(void* src, const int32_t* index, void* out,                    \
                            float* sumWorkspace, float* deviation, float* count,           \
                            const ScatterSoftmaxTilingData& tiling, aclrtStream stream)    \
        {                                                                                  \
            GM_ADDR indexBytes = AsGmAddr(index);                                          \
            GM_ADDR srcBytes = AsGmAddr(src);                                              \
            GM_ADDR outBytes = AsGmAddr(out);                                              \
            GM_ADDR sumBytes = AsGmAddr(sumWorkspace);                                     \
            GM_ADDR deviationBytes = AsGmAddr(deviation);                                  \
            GM_ADDR countBytes = AsGmAddr(count);                                          \
            Prefix##SumKernel<T><<<tiling.blockNum, nullptr, stream>>>(                    \
                srcBytes, indexBytes, nullptr, sumBytes, nullptr, countBytes, nullptr,     \
                nullptr, nullptr, tiling);                                                 \
            Prefix##DeviationKernel<T><<<tiling.blockNum, nullptr, stream>>>(              \
                srcBytes, indexBytes, nullptr, sumBytes, deviationBytes, countBytes,       \
                nullptr, nullptr, nullptr, tiling);                                        \
            /* The reduced rows are (batch, group) pairs, so the finalize pass walks */    \
            /* `outRows` on its own core split. */                                         \
            ScatterSoftmaxTilingData finalizeTiling = tiling;                              \
            finalizeTiling.rows = tiling.outRows;                                          \
            finalizeTiling.rowChunk = tiling.outRowChunk;                                  \
            finalizeTiling.rowBatch = tiling.outRowBatch;                                  \
            finalizeTiling.blockNum = tiling.outBlockNum;                                  \
            Prefix##FinalizeKernel<T><<<tiling.outBlockNum, nullptr, stream>>>(            \
                nullptr, indexBytes, nullptr, sumBytes, deviationBytes, countBytes,        \
                outBytes, nullptr, nullptr, finalizeTiling);                               \
        }                                                                                  \
                                                                                           \
        template <typename T>                                                              \
        inline void Backward(const void* src, const void* out, const void* gradOut,        \
                             const int32_t* index, void* gradSrc, float* meanWorkspace,    \
                             float* count, const ScatterSoftmaxTilingData& tiling,          \
                             aclrtStream stream)                                           \
        {                                                                                  \
            GM_ADDR indexBytes = AsGmAddr(index);                                          \
            GM_ADDR srcBytes = AsGmAddr(src);                                              \
            GM_ADDR outBytes = AsGmAddr(out);                                              \
            GM_ADDR gradOutBytes = AsGmAddr(gradOut);                                      \
            GM_ADDR meanBytes = AsGmAddr(meanWorkspace);                                   \
            GM_ADDR countBytes = AsGmAddr(count);                                          \
            Prefix##GradElementKernel<T><<<tiling.blockNum, nullptr, stream>>>(            \
                srcBytes, indexBytes, nullptr, meanBytes, nullptr, countBytes, outBytes,   \
                gradOutBytes, AsGmAddr(gradSrc), tiling);                                  \
        }                                                                                  \
    }  /* namespace Ns */                                                                  \
    }  /* namespace ScatterSoftmaxNpu */                                                   \
                                                                                           \
    namespace opsgnn {                                                                     \
    void Prefix##Forward(void* src, const int32_t* index, void* out,                       \
                         float* sumWorkspace, float* deviation, float* count,               \
                         ScatterSoftmaxDType dtype,                                        \
                         const ScatterSoftmaxTilingData& tiling, aclrtStream stream)        \
    {                                                                                      \
        ScatterSoftmaxNpu::DispatchDType(dtype, [&](auto tag) {                             \
            using T = typename decltype(tag)::Type;                                        \
            ScatterSoftmaxNpu::Ns::Forward<T>(src, index, out, sumWorkspace, deviation,     \
                                              count, tiling, stream);                      \
        });                                                                                \
    }                                                                                      \
                                                                                           \
    void Prefix##Backward(const void* src, const void* out, const void* gradOut,            \
                          const int32_t* index, void* gradSrc, float* meanWorkspace,        \
                          float* count, ScatterSoftmaxDType dtype,                         \
                          const ScatterSoftmaxTilingData& tiling, aclrtStream stream)       \
    {                                                                                      \
        ScatterSoftmaxNpu::DispatchDType(dtype, [&](auto tag) {                             \
            using T = typename decltype(tag)::Type;                                         \
            ScatterSoftmaxNpu::Ns::Backward<T>(src, out, gradOut, index, gradSrc,           \
                                               meanWorkspace, count, tiling, stream);       \
        });                                                                                 \
    }                                                                                      \
    }

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_ENTRY_H
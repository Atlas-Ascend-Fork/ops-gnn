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
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_phase.h"

/**
 * Kernel-entry plumbing shared by the grouped-softmax operators: the per-phase
 * global-tensor binding and the buffer/pipeline bootstrap around the row driver.
 *
 * Every entry uses one parameter list so the boilerplate lives here once; the
 * arguments a phase does not use are null and are never bound. `Log` selects
 * the log-softmax arithmetic of the phase bodies (see
 * scatter_softmax_kernel_phase.h), so scatter_softmax and scatter_log_softmax
 * share this file and differ only in the tag they instantiate.
 */

namespace ScatterSoftmaxNpu {

// Binds the global tensors a phase touches.
template <ScatterSoftmaxPhase Phase, bool Log, typename T>
__aicore__ inline void BindTensors(PhaseTensors<T>& tensors,
                                   const ScatterSoftmaxTilingData& tiling,
                                   GM_ADDR src, GM_ADDR index, GM_ADDR maximum,
                                   GM_ADDR sum, GM_ADDR out, GM_ADDR gradOut,
                                   GM_ADDR gradSrc)
{
    tensors.indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index), tiling.dimLength);
    // The log-domain normalization is recomputed from the source, so that pass
    // reads `src` and the maximum as well; the softmax path only needs the
    // exponential it stored in the previous pass.
    if constexpr (Phase == kPhaseMaximum || Phase == kPhaseExp ||
                  (Log && Phase == kPhaseNormalize)) {
        tensors.srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src), tiling.srcNumel);
        tensors.maximumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(maximum),
                                          tiling.workspaceElems);
    }
    if constexpr (Phase != kPhaseMaximum) {
        tensors.sumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum), tiling.workspaceElems);
        tensors.outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), tiling.srcNumel);
    }
    if constexpr (Phase == kPhaseGradSum || Phase == kPhaseGrad) {
        tensors.gradGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradOut), tiling.srcNumel);
    }
    if constexpr (Phase == kPhaseGrad) {
        tensors.gradSrcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradSrc), tiling.srcNumel);
    }
}

// The buffers, the pipeline events and the row/batch driver are identical for
// all five kernels, so only the phase tag, the mode and the tensor list differ.
template <ScatterSoftmaxPhase Phase, bool Log, typename T>
__aicore__ inline void RunPhase(GM_ADDR src, GM_ADDR index, GM_ADDR maximum, GM_ADDR sum,
                                GM_ADDR out, GM_ADDR gradOut, GM_ADDR gradSrc,
                                ScatterSoftmaxTilingData tiling)
{
    if (!IsParticipating(tiling)) {
        return;
    }
    PhaseTensors<T> tensors;
    TPipe pipe;
    InitTensors<Phase>(tensors, pipe, tiling);
    BindTensors<Phase, Log>(tensors, tiling, src, index, maximum, sum, out, gradOut, gradSrc);
    PipelineSync sync;
    sync.Init(pipe);
    InitAccumulator<Phase>(tensors, tiling);
    LoadWorkspace<Phase, Log>(tensors, tiling, sync);
    RunRows<Phase, Log>(tensors, tiling, sync);
    FlushAccumulator<Phase>(tensors, tiling, sync);
    sync.Release(pipe);
}

}  // namespace ScatterSoftmaxNpu

// ---------------------------------------------------------------------------
// Whole-operator generator.
//
// The five phase kernels, their launcher and the two exported entry points are
// identical in shape for every grouped-softmax operator: only the kernel names,
// the compile-time arithmetic mode and the launch namespace differ. Generating
// them from one description keeps `scatter_softmax` and `scatter_log_softmax`
// from drifting apart - the arithmetic itself stays in the shared phase bodies.
//
//   Prefix   kernel and export name prefix (Prefix##MaxKernel, Prefix##Forward)
//   Ns       launch namespace inside ScatterSoftmaxNpu
//   LogMode  compile-time mode forwarded to the phase bodies
// ---------------------------------------------------------------------------
#define SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Name, Phase, LogMode)                     \
    template <typename T>                                                                \
    __attribute__((aiv)) __global__ __aicore__ void Prefix##Name##Kernel(                \
        GM_ADDR src, GM_ADDR index, GM_ADDR maximum, GM_ADDR sum, GM_ADDR out,            \
        GM_ADDR gradOut, GM_ADDR gradSrc, ScatterSoftmaxTilingData tiling)                \
    {                                                                                    \
        RunPhase<Phase, LogMode, T>(src, index, maximum, sum, out, gradOut, gradSrc,      \
                                    tiling);                                             \
    }

#define SCATTER_GROUPED_SOFTMAX_OPERATOR(Prefix, Ns, LogMode)                            \
    namespace ScatterSoftmaxNpu {                                                        \
    using namespace AscendC;                                                             \
                                                                                         \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Max, kPhaseMaximum, LogMode)                   \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Exp, kPhaseExp, LogMode)                       \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Normalize, kPhaseNormalize, LogMode)           \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, GradSum, kPhaseGradSum, LogMode)               \
    SCATTER_GROUPED_SOFTMAX_KERNEL(Prefix, Grad, kPhaseGrad, LogMode)                     \
                                                                                         \
    /* Every pass takes the same argument list; the phases a mode does not use */        \
    /* are never bound (see BindTensors), so one sequence serves both operators. */       \
    namespace Ns {                                                                        \
        template <typename T>                                                            \
        inline void Forward(void* src, const int32_t* index, void* out,                  \
                            float* maximum, float* sumWorkspace,                         \
                            const ScatterSoftmaxTilingData& tiling, aclrtStream stream)   \
        {                                                                                \
            GM_ADDR indexBytes = const_cast<uint8_t*>(                                    \
                reinterpret_cast<const uint8_t*>(index));                                \
            GM_ADDR srcBytes = reinterpret_cast<uint8_t*>(src);                          \
            GM_ADDR outBytes = reinterpret_cast<uint8_t*>(out);                          \
            GM_ADDR maximumBytes = reinterpret_cast<uint8_t*>(maximum);                  \
            GM_ADDR sumBytes = reinterpret_cast<uint8_t*>(sumWorkspace);                 \
            Prefix##MaxKernel<T><<<tiling.blockNum, nullptr, stream>>>(                  \
                srcBytes, indexBytes, maximumBytes, nullptr, nullptr, nullptr, nullptr,   \
                tiling);                                                                 \
            Prefix##ExpKernel<T><<<tiling.blockNum, nullptr, stream>>>(                  \
                srcBytes, indexBytes, maximumBytes, sumBytes, outBytes, nullptr,          \
                nullptr, tiling);                                                        \
            Prefix##NormalizeKernel<T><<<tiling.blockNum, nullptr, stream>>>(            \
                srcBytes, indexBytes, maximumBytes, sumBytes, outBytes, nullptr,          \
                nullptr, tiling);                                                        \
        }                                                                                \
                                                                                         \
        template <typename T>                                                            \
        inline void Backward(const void* out, const void* gradOut,                       \
                             const int32_t* index, void* gradSrc,                        \
                             float* sumWorkspace,                                        \
                             const ScatterSoftmaxTilingData& tiling,                     \
                             aclrtStream stream)                                         \
        {                                                                                \
            GM_ADDR indexBytes = const_cast<uint8_t*>(                                    \
                reinterpret_cast<const uint8_t*>(index));                                \
            GM_ADDR outBytes = const_cast<uint8_t*>(                                      \
                reinterpret_cast<const uint8_t*>(out));                                  \
            GM_ADDR gradOutBytes = const_cast<uint8_t*>(                                  \
                reinterpret_cast<const uint8_t*>(gradOut));                              \
            GM_ADDR sumBytes = reinterpret_cast<uint8_t*>(sumWorkspace);                 \
            Prefix##GradSumKernel<T><<<tiling.blockNum, nullptr, stream>>>(              \
                nullptr, indexBytes, nullptr, sumBytes, outBytes, gradOutBytes, nullptr,  \
                tiling);                                                                 \
            Prefix##GradKernel<T><<<tiling.blockNum, nullptr, stream>>>(                 \
                nullptr, indexBytes, nullptr, sumBytes, outBytes, gradOutBytes,           \
                reinterpret_cast<uint8_t*>(gradSrc), tiling);                            \
        }                                                                                \
    }  /* namespace Ns */                                                                \
    }  /* namespace ScatterSoftmaxNpu */                                                 \
                                                                                         \
    namespace opsgnn {                                                                   \
    void Prefix##Forward(void* src, const int32_t* index, void* out,                     \
                         float* maximum, float* sumWorkspace,                             \
                         ScatterSoftmaxDType dtype,                                      \
                         const ScatterSoftmaxTilingData& tiling, aclrtStream stream)      \
    {                                                                                    \
        switch (dtype) {                                                                 \
            case SCATTER_SOFTMAX_F16:                                                    \
                ScatterSoftmaxNpu::Ns::Forward<half>(                                    \
                    src, index, out, maximum, sumWorkspace, tiling, stream);              \
                break;                                                                   \
            case SCATTER_SOFTMAX_F32:                                                    \
                ScatterSoftmaxNpu::Ns::Forward<float>(                                   \
                    src, index, out, maximum, sumWorkspace, tiling, stream);              \
                break;                                                                   \
            case SCATTER_SOFTMAX_BF16:                                                   \
                ScatterSoftmaxNpu::Ns::Forward<bfloat16_t>(                              \
                    src, index, out, maximum, sumWorkspace, tiling, stream);              \
                break;                                                                   \
            default:                                                                     \
                break;                                                                   \
        }                                                                                \
    }                                                                                    \
                                                                                         \
    void Prefix##Backward(const void* out, const void* gradOut,                          \
                          const int32_t* index, void* gradSrc,                           \
                          float* sumWorkspace, ScatterSoftmaxDType dtype,                 \
                          const ScatterSoftmaxTilingData& tiling,                         \
                          aclrtStream stream)                                             \
    {                                                                                    \
        switch (dtype) {                                                                 \
            case SCATTER_SOFTMAX_F16:                                                    \
                ScatterSoftmaxNpu::Ns::Backward<half>(                                   \
                    out, gradOut, index, gradSrc, sumWorkspace, tiling, stream);           \
                break;                                                                   \
            case SCATTER_SOFTMAX_F32:                                                    \
                ScatterSoftmaxNpu::Ns::Backward<float>(                                  \
                    out, gradOut, index, gradSrc, sumWorkspace, tiling, stream);           \
                break;                                                                   \
            case SCATTER_SOFTMAX_BF16:                                                   \
                ScatterSoftmaxNpu::Ns::Backward<bfloat16_t>(                             \
                    out, gradOut, index, gradSrc, sumWorkspace, tiling, stream);           \
                break;                                                                   \
            default:                                                                     \
                break;                                                                   \
        }                                                                                \
    }                                                                                    \
    }  /* namespace opsgnn */

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_ENTRY_H

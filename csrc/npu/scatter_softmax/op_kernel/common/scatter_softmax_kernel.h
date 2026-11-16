/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * AscendC device implementation of the grouped softmax shared by arch22
 * (A2/A3) and arch35 (950).
 *
 * The group of a row is given by `index`, which is not sorted, so the reduced
 * values of every (batch, group, feature) slot are accumulated by the MTE3
 * atomic unit (or, when the whole reduction workspace fits in UB, by a private
 * per-core accumulator that is flushed once at the end of the pass):
 *
 *   Maximum    for every row: maximum[max(src)]              -> fp32 maximum
 *   Exp        for every row: out = exp(src - maximum), sum sum += out
 *   Normalize  for every row: out /= sum
 *   GradSum    for every row: sum sum += out * grad_out
 *   Grad       for every row: grad_src = out * (grad_out - sum)
 *
 * Two scheduling decisions keep the passes off the "one synchronous row at a
 * time" floor that a naive per-row loop hits on a 64-core part:
 *
 *   1. a core owns a contiguous chunk of rows, so `rowBatch` consecutive rows
 *      are loaded, reduced and stored as one batch - the GM round trip of the
 *      first row is hidden behind the transfers of the rest, and the three
 *      pipeline stages synchronise once per batch instead of once per row;
 *   2. when `batch * groupCount * afterPadded` fits the accumulator budget, the
 *      reduction workspace lives in UB for the whole pass: the reduction
 *      phases accumulate in place and touch the workspace once per core, and
 *      every pass that reads it reads it once instead of once per row. A
 *      million-row single-group shape would otherwise re-read the same 512-byte
 *      line a million times, which costs more than the arithmetic by an order
 *      of magnitude.
 *
 * The fp32 workspace pads each row to a 32-byte multiple so every atomic
 * transfer is aligned. All arithmetic is done in fp32 regardless of the source
 * dtype, which keeps a long fp16 group (exp-sum above 65504) from collapsing.
 *
 * Every vector instruction writes at a 32-byte aligned buffer base: when a tile
 * leaves lanes past the meaningful ones, they are filled by duplicating the
 * neutral value over the whole tile before the meaningful lanes are overwritten.
 * A full tile has no such lanes and skips the fill.
 */

#ifndef OPS_GNN_SCATTER_SOFTMAX_KERNEL_H
#define OPS_GNN_SCATTER_SOFTMAX_KERNEL_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_common.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_kernel_phase.h"
#include "scatter_softmax/op_kernel/common/scatter_softmax_launch.h"

namespace ScatterSoftmaxNpu {

using namespace AscendC;

// ---------------------------------------------------------------------------
// Kernel entries. arch22 and arch35 compile the same bodies.
// ---------------------------------------------------------------------------
// Binds the global tensors a phase touches. Every entry shares one parameter
// list so the boilerplate lives here once; the arguments a phase does not use
// are null and are never bound.
template <ScatterSoftmaxPhase Phase, typename T>
__aicore__ inline void BindTensors(PhaseTensors<T>& tensors,
                                   const ScatterSoftmaxTilingData& tiling,
                                   GM_ADDR src, GM_ADDR index, GM_ADDR maximum,
                                   GM_ADDR sum, GM_ADDR out, GM_ADDR gradOut,
                                   GM_ADDR gradSrc)
{
    tensors.indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index), tiling.dimLength);
    if constexpr (Phase == kPhaseMaximum || Phase == kPhaseExp) {
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
// all five kernels, so only the phase tag and the tensor list differ.
template <ScatterSoftmaxPhase Phase, typename T>
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
    BindTensors<Phase>(tensors, tiling, src, index, maximum, sum, out, gradOut, gradSrc);
    PipelineSync sync;
    sync.Init(pipe);
    InitAccumulator<Phase>(tensors, tiling);
    LoadWorkspace<Phase>(tensors, tiling, sync);
    RunRows<Phase>(tensors, tiling, sync);
    FlushAccumulator<Phase>(tensors, tiling, sync);
    sync.Release(pipe);
}

#define SCATTER_SOFTMAX_KERNEL(Name, Phase)                                              \
    template <typename T>                                                               \
    __attribute__((aiv)) __global__ __aicore__ void Name(                               \
        GM_ADDR src, GM_ADDR index, GM_ADDR maximum, GM_ADDR sum, GM_ADDR out,           \
        GM_ADDR gradOut, GM_ADDR gradSrc, ScatterSoftmaxTilingData tiling)               \
    {                                                                                   \
        RunPhase<Phase, T>(src, index, maximum, sum, out, gradOut, gradSrc, tiling);     \
    }

SCATTER_SOFTMAX_KERNEL(ScatterSoftmaxMaxKernel, kPhaseMaximum)
SCATTER_SOFTMAX_KERNEL(ScatterSoftmaxExpKernel, kPhaseExp)
SCATTER_SOFTMAX_KERNEL(ScatterSoftmaxNormalizeKernel, kPhaseNormalize)
SCATTER_SOFTMAX_KERNEL(ScatterSoftmaxGradSumKernel, kPhaseGradSum)
SCATTER_SOFTMAX_KERNEL(ScatterSoftmaxGradKernel, kPhaseGrad)

// ---------------------------------------------------------------------------
// Host-side launch (compiled by the same device-capable translation unit).
// ---------------------------------------------------------------------------
namespace Launch {

template <typename T>
inline void Forward(void* src, const int32_t* index, void* out,
                    float* maximum, float* sumWorkspace,
                    const ScatterSoftmaxTilingData& tiling, aclrtStream stream)
{
    ScatterSoftmaxMaxKernel<T><<<tiling.blockNum, nullptr, stream>>>(
        reinterpret_cast<uint8_t*>(src),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(index)),
        reinterpret_cast<uint8_t*>(maximum),
        nullptr, nullptr, nullptr, nullptr, tiling);
    ScatterSoftmaxExpKernel<T><<<tiling.blockNum, nullptr, stream>>>(
        reinterpret_cast<uint8_t*>(src),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(index)),
        reinterpret_cast<uint8_t*>(maximum),
        reinterpret_cast<uint8_t*>(sumWorkspace), reinterpret_cast<uint8_t*>(out),
        nullptr, nullptr, tiling);
    ScatterSoftmaxNormalizeKernel<T><<<tiling.blockNum, nullptr, stream>>>(
        nullptr,
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(index)),
        nullptr, reinterpret_cast<uint8_t*>(sumWorkspace), reinterpret_cast<uint8_t*>(out),
        nullptr, nullptr, tiling);
}

template <typename T>
inline void Backward(const void* out, const void* gradOut,
                     const int32_t* index, void* gradSrc,
                     float* sumWorkspace,
                     const ScatterSoftmaxTilingData& tiling, aclrtStream stream)
{
    ScatterSoftmaxGradSumKernel<T><<<tiling.blockNum, nullptr, stream>>>(
        nullptr,
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(index)),
        nullptr, reinterpret_cast<uint8_t*>(sumWorkspace),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(out)),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(gradOut)), nullptr, tiling);
    ScatterSoftmaxGradKernel<T><<<tiling.blockNum, nullptr, stream>>>(
        nullptr,
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(index)),
        nullptr, reinterpret_cast<uint8_t*>(sumWorkspace),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(out)),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(gradOut)),
        reinterpret_cast<uint8_t*>(gradSrc), tiling);
}

}  // namespace Launch

}  // namespace ScatterSoftmaxNpu

namespace opsgnn {

void ScatterSoftmaxForward(void* src, const int32_t* index, void* out,
                           float* maximum, float* sumWorkspace,
                           ScatterSoftmaxDType dtype,
                           const ScatterSoftmaxTilingData& tiling,
                           aclrtStream stream)
{
    switch (dtype) {
        case SCATTER_SOFTMAX_F16:
            ScatterSoftmaxNpu::Launch::Forward<half>(
                src, index, out, maximum, sumWorkspace, tiling, stream);
            break;
        case SCATTER_SOFTMAX_F32:
            ScatterSoftmaxNpu::Launch::Forward<float>(
                src, index, out, maximum, sumWorkspace, tiling, stream);
            break;
        case SCATTER_SOFTMAX_BF16:
            ScatterSoftmaxNpu::Launch::Forward<bfloat16_t>(
                src, index, out, maximum, sumWorkspace, tiling, stream);
            break;
        default:
            break;
    }
}

void ScatterSoftmaxBackward(const void* out, const void* gradOut,
                            const int32_t* index, void* gradSrc,
                            float* sumWorkspace, ScatterSoftmaxDType dtype,
                            const ScatterSoftmaxTilingData& tiling,
                            aclrtStream stream)
{
    switch (dtype) {
        case SCATTER_SOFTMAX_F16:
            ScatterSoftmaxNpu::Launch::Backward<half>(
                out, gradOut, index, gradSrc, sumWorkspace, tiling, stream);
            break;
        case SCATTER_SOFTMAX_F32:
            ScatterSoftmaxNpu::Launch::Backward<float>(
                out, gradOut, index, gradSrc, sumWorkspace, tiling, stream);
            break;
        case SCATTER_SOFTMAX_BF16:
            ScatterSoftmaxNpu::Launch::Backward<bfloat16_t>(
                out, gradOut, index, gradSrc, sumWorkspace, tiling, stream);
            break;
        default:
            break;
    }
}

}  // namespace opsgnn

#endif  // OPS_GNN_SCATTER_SOFTMAX_KERNEL_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matrix_exp.h"
#include "matrix_exp/op_kernel/arch35/matrix_exp.h"
#include "matrix_exp/op_kernel/arch35/matrix_exp_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include <acl/acl_base.h>
#include <array>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

namespace {

constexpr int64_t MATRIX_EXP_MAX_N = 2048;

// 单位矩阵常量缓存（进程级持有，key = device<<16 | nPad16）。
// raw kernel launch 对 eye 的 GM 读取不被 caching allocator 感知：若 eye 为调用局部
// 张量，析构后 block 可能被后续分配复用，与异步入队的 kernel 读取产生竞态（实测偶发
// 读零）。静态缓存后内容一次写入终身复用，且消除创建-析构生命周期。
// 注意：不能在 launch 前紧邻处用 NPU eye 算子创建——进程内首次调用该算子时，其写入
// 的实际提交晚于紧随的 raw kernel launch（aclrtSynchronizeDevice 兜底无效，同步点
// 早于 lazy submit），kernel 读到零；实测同进程二次调用起正常。改为 host 构造 +
// 同步 H2D 拷贝，写入语义确定，彻底绕开该竞态（每 (device, nPad16) 仅一次）。
torch::Tensor GetEyeConst(const torch::TensorOptions& opts, uint32_t nPad16, int64_t device)
{
    static std::mutex mu;
    static std::unordered_map<uint64_t, torch::Tensor> cache;
    const uint64_t key = (static_cast<uint64_t>(device) << 16) | nPad16;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto t = torch::eye(static_cast<int64_t>(nPad16),
                            torch::TensorOptions().dtype(at::kFloat))
                     .to(opts.device(), /*non_blocking=*/false);
        it = cache.emplace(key, std::move(t)).first;
    }
    return it->second;
}

void ValidateInputs(const torch::Tensor& A)
{
    TORCH_CHECK(A.device().type() == c10::DeviceType::PrivateUse1, "matrix_exp: A must be an NPU tensor");
    TORCH_CHECK(A.dim() >= 2, "matrix_exp: A must have at least 2 dims, got ", A.dim());
    TORCH_CHECK(A.size(A.dim() - 2) == A.size(A.dim() - 1),
                "matrix_exp: A must be a square matrix, got (",
                A.size(A.dim() - 2), ", ", A.size(A.dim() - 1), ")");
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float ||
                A.scalar_type() == at::ScalarType::Half ||
                A.scalar_type() == at::ScalarType::BFloat16,
                "matrix_exp: unsupported dtype (float64/complex are not supported on NPU), got ",
                A.scalar_type());
    TORCH_CHECK(A.size(-1) <= MATRIX_EXP_MAX_N,
                "matrix_exp: matrix order too large, got ", A.size(-1),
                ", max supported is ", MATRIX_EXP_MAX_N);
}

torch::Tensor MatrixExpLarge(const torch::Tensor& input)
{
    const int64_t n = input.size(-1);
    TORCH_CHECK(n > MATRIX_EXP_N_MAX_UB, "matrix_exp: invalid order for the large-matrix path: ", n);
    auto a = input.to(at::kFloat).reshape({-1, n, n});
    // Scale before summing to keep the reduction in range for finite fp32 inputs.
    // A column contains at most MATRIX_EXP_MAX_N elements.
    constexpr double invNormScale = 1.0 / MATRIX_EXP_MAX_N;
    auto scaledNorm = (a.detach().abs() * invNormScale).sum(-2).amax(-1);
    auto steps = (scaledNorm.log2() - std::log2(invNormScale) - 1).ceil().clamp_min(0);
    auto finite = steps.isfinite();
    steps = torch::where(finite, steps, 0);
    // Use the same host read for the scaling count and the low-norm dispatch.
    // A negative result means every matrix has 1-norm at most 1/8.
    const int64_t maxSteps = torch::where(scaledNorm.le(0.125 * invNormScale), -1, steps).max().item<int64_t>();
    auto eye = torch::eye(n, a.options());
    if (maxSteps < 0) {
        // Degree 5 has a 1-norm truncation bound below 7e-9 on this interval.
        // Evaluate it in powers of A^2 with three batched multiplies.
        auto a2 = a.bmm(a);
        auto upper = (eye / 24.0).add(a, 1.0 / 120.0);
        auto middle = a2.bmm(upper) + (eye / 2.0).add(a, 1.0 / 6.0);
        auto result = (eye + a) + a2.bmm(middle);
        return result.reshape(input.sizes()).to(input.scalar_type());
    }
    steps = steps.view({-1, 1, 1});
    // Split the power of two so both factors remain normal in fp32.
    auto halfSteps = (steps * 0.5).floor();
    a = a * (-halfSteps).exp2() * (halfSteps - steps).exp2();
    a = torch::where(finite.view({-1, 1, 1}), a, NAN);

    // Group the degree-18 Taylor polynomial by powers of A^5.
    // Four multiplications build A^2 through A^5, then three combine the blocks.
    constexpr int degree = 18;
    constexpr int blockSize = 5;
    std::array<double, degree + 1> coeff{1.0};
    for (int k = 1; k <= degree; ++k) {
        coeff[k] = coeff[k - 1] / k;
    }
    std::array<torch::Tensor, blockSize + 1> powers{eye, a};
    for (int k = 2; k <= blockSize; ++k) {
        powers[k] = powers[k - 1].bmm(a);
    }
    torch::Tensor result;
    for (int offset = degree / blockSize * blockSize; offset >= 0; offset -= blockSize) {
        auto block = eye * coeff[offset];
        for (int k = 1; k < blockSize && offset + k <= degree; ++k) {
            block = block.add(powers[k], coeff[offset + k]);
        }
        result = result.defined() ? powers[blockSize].bmm(result) + block : block;
    }
    for (int64_t k = 0; k < maxSteps; ++k) {
        auto active = steps.gt(k);
        auto operand = torch::where(active, result, 0);
        result = torch::where(active, operand.bmm(operand), result);
    }
    return result.reshape(input.sizes()).to(input.scalar_type());
}

MatrixExpTilingData ComputeTiling(uint64_t batchSize, uint32_t n, uint32_t dtSize, uint32_t aicCanUseNum)
{
    // 均衡切分（设计文档 3.4.2）：floor + 余数均摊，避免 ceil+末核吸收的下溢与整核空闲
    uint32_t coreNum = (aicCanUseNum < batchSize) ? aicCanUseNum : static_cast<uint32_t>(batchSize);
    coreNum = coreNum >= 1 ? coreNum : 1;

    MatrixExpTilingData tiling;
    tiling.batchSize = batchSize;
    tiling.n = n;
    tiling.coreNum = coreNum;
    tiling.matricesPerCore = static_cast<uint32_t>(batchSize / coreNum);
    tiling.coresWithExtra = static_cast<uint32_t>(batchSize % coreNum);
    tiling.dtSize = dtSize;
    tiling.nSmall = 0;  // 预留字段（迭代一未使用）
    return tiling;
}

void LaunchKernel(const torch::Tensor& A, torch::Tensor& out, const MatrixExpTilingData& tiling,
                  aclrtStream stream)
{
    const uint8_t* aPtr = nullptr;
    uint8_t* outPtr = nullptr;
    uint32_t dtypeTag = 0;
    if (A.scalar_type() == at::ScalarType::Float) {
        aPtr = reinterpret_cast<const uint8_t*>(A.data_ptr<float>());
        outPtr = reinterpret_cast<uint8_t*>(out.data_ptr<float>());
        dtypeTag = 0;
    } else if (A.scalar_type() == at::ScalarType::Half) {
        aPtr = reinterpret_cast<const uint8_t*>(A.data_ptr<c10::Half>());
        outPtr = reinterpret_cast<uint8_t*>(out.data_ptr<c10::Half>());
        dtypeTag = 1;
    } else {
        aPtr = reinterpret_cast<const uint8_t*>(A.data_ptr<c10::BFloat16>());
        outPtr = reinterpret_cast<uint8_t*>(out.data_ptr<c10::BFloat16>());
        dtypeTag = 2;
    }

    // GM workspace（fp32）：每核 [A nPad² | E nPad² | tmp nPad² | meta 8]，三区均为
    // nPad(=ceil(n/16)*16) 方阵布局，meta 32B 对齐（布局契约见 arch35/matrix_exp.h
    // 的 MATRIX_EXP_WS_STRIDE_FLOATS）。
    // kernel 异步入队后 ws/eye 由 caching allocator 按 stream 顺序回收，同 stream 复用安全。
    const uint64_t perCoreFloats = MATRIX_EXP_WS_STRIDE_FLOATS(tiling.n);
    torch::Tensor ws = torch::empty(
        {static_cast<int64_t>(tiling.coreNum) * static_cast<int64_t>(perCoreFloats)},
        A.options().dtype(at::kFloat));
    // 单位矩阵常量（nPad16 方阵，尾行/行尾为零，AIV 在输出阶段组装 E = I + M 用）。
    // 经 GetEyeConst 进程级缓存持有。
    const uint32_t nPad16 = ((tiling.n + 15U) / 16U) * 16U;
    torch::Tensor eye = GetEyeConst(A.options(), nPad16, A.get_device());

    MatrixExp(aPtr, outPtr, reinterpret_cast<const uint8_t*>(ws.data_ptr<float>()),
              reinterpret_cast<const uint8_t*>(eye.data_ptr<float>()), tiling, dtypeTag, stream);
}

}  // namespace

torch::Tensor matrix_exp(torch::Tensor A, torch::Tensor out)
{
    ValidateInputs(A);

    if (!A.is_contiguous()) {
        A = A.contiguous();
    }

    const int64_t n = A.size(A.dim() - 1);
    // n == 0（如 (0, 0) 形状）视为空 batch，避免除零
    const uint64_t batchSize = (n == 0) ? 0
        : static_cast<uint64_t>(A.numel()) / static_cast<uint64_t>(n * n);

    // out 统一校验（空/非空 batch 走同一契约，见设计文档 1.4）：仅 defined() 判定是否提供，
    // 形状/dtype/设备必须与 A 一致，空输入时合法空 out 也原样返回。
    const bool hasOut = out.defined();
    if (hasOut) {
        TORCH_CHECK(out.scalar_type() == A.scalar_type(),
                    "matrix_exp: out dtype must match A, got ", out.scalar_type(),
                    " vs ", A.scalar_type());
        TORCH_CHECK(out.sizes() == A.sizes(),
                    "matrix_exp: out shape must match A, got ", out.sizes(),
                    " vs ", A.sizes());
        TORCH_CHECK(out.device() == A.device(),
                    "matrix_exp: out device must match A, got ", out.device(),
                    " vs ", A.device());
    }

    // 空 batch：与 torch 行为对齐，直接返回空张量（out 校验已在上方统一完成）
    if (batchSize == 0) {
        return hasOut ? out : A.clone();
    }

    if (n > MATRIX_EXP_N_MAX_UB) {
        auto result = MatrixExpLarge(A);
        if (hasOut) {
            out.copy_(result);
            return out;
        }
        return result;
    }

    // 非连续 out 先算到连续缓冲，再拷回视图。
    const bool directOut = hasOut && out.is_contiguous();
    auto result = directOut ? out : torch::empty(A.sizes(), A.options());

    const uint32_t dtSize = (A.scalar_type() == at::ScalarType::Float) ? 4 : 2;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t aicCanUseNum = ascendcPlatform->GetCoreNumAic();
    MatrixExpTilingData tiling = ComputeTiling(batchSize, static_cast<uint32_t>(n), dtSize, aicCanUseNum);

    aclrtStream stream = c10_npu::getCurrentNPUStream(A.get_device()).stream();
    LaunchKernel(A, result, tiling, stream);

    if (hasOut && !directOut) {
        out.copy_(result);
    }
    return hasOut ? out : result;
}

}  // namespace opsgnn

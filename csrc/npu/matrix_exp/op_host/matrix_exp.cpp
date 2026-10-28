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
#include <mutex>
#include <unordered_map>
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

namespace {

constexpr uint64_t MATRIX_EXP_MAX_N = 1024;  // 矩阵阶数硬上限，防御异常入参

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
    TORCH_CHECK(!A.is_cpu(), "matrix_exp: A must be an NPU tensor");
    TORCH_CHECK(A.dim() >= 2, "matrix_exp: A must have at least 2 dims, got ", A.dim());
    TORCH_CHECK(A.size(A.dim() - 2) == A.size(A.dim() - 1),
                "matrix_exp: A must be a square matrix, got (",
                A.size(A.dim() - 2), ", ", A.size(A.dim() - 1), ")");
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float ||
                A.scalar_type() == at::ScalarType::Half ||
                A.scalar_type() == at::ScalarType::BFloat16,
                "matrix_exp: unsupported dtype (float64/complex are not supported on NPU), got ",
                A.scalar_type());
    TORCH_CHECK(A.size(A.dim() - 1) <= MATRIX_EXP_MAX_N,
                "matrix_exp: matrix order too large, got ", A.size(A.dim() - 1),
                ", max supported is ", MATRIX_EXP_MAX_N);
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
    // 单位矩阵常量（nPad16 方阵，尾行/行尾为零，AIV 用于 E0 = I + (sigma/K)*A 与
    // Horner 每轮的 I 重置）。经 GetEyeConst 进程级缓存持有。
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
        TORCH_CHECK(!out.is_cpu(), "matrix_exp: out must be an NPU tensor");
    }

    // 空 batch：与 torch 行为对齐，直接返回空张量（out 校验已在上方统一完成）
    if (batchSize == 0) {
        return hasOut ? out : torch::empty(A.sizes(), A.options());
    }

    // TODO(算子实现，迭代三): 模板三（n > 128 分块路径）落地后放开此限制（设计文档 3.5）
    TORCH_CHECK(n <= MATRIX_EXP_N_MAX_UB,
                "matrix_exp: matrices with n > ", MATRIX_EXP_N_MAX_UB,
                " are not supported yet, got n = ", n);

    torch::Tensor result;
    bool needCopyBack = false;
    if (hasOut) {
        if (out.is_contiguous()) {
            result = out;
        } else {
            // 非连续 out：先算到连续缓冲，再拷回视图（设计文档 1.4）
            result = torch::empty(A.sizes(), A.options());
            needCopyBack = true;
        }
    } else {
        result = torch::empty(A.sizes(), A.options());
    }

    const uint32_t dtSize = (A.scalar_type() == at::ScalarType::Float) ? 4 : 2;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t aicCanUseNum = ascendcPlatform->GetCoreNumAic();
    MatrixExpTilingData tiling = ComputeTiling(batchSize, static_cast<uint32_t>(n), dtSize, aicCanUseNum);

    aclrtStream stream = c10_npu::getCurrentNPUStream(A.get_device()).stream();
    LaunchKernel(A, result, tiling, stream);

    if (needCopyBack) {
        out.copy_(result);
    }
    return hasOut ? out : result;
}

}  // namespace opsgnn

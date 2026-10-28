# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""matrix_exp 精度/行为回归。

精度验收方法学（设计文档 3.1 精度契约 + 5.2 风险项）：
    - 参考值为 **fp64** matrix_exp，而非 torch fp32。大范数（||A||_F > 64）
      病态输入下 torch fp32 自身相对 fp64 的逐元素偏差即可达 1e-2 量级
      （实测 n=128/norm=256 时 6.6e-2），两个不同的 fp32 算法（本算子
      Taylor K=9/theta=0.5 vs torch Padé-13）在该区间逐元素必然分歧，
      "与 torch fp32 逐位对拍"作为验收在数学上不可满足——设计文档 5.2
      明确规定"按 3.1 容差对 fp64 参考对拍，而非与 torch 逐位对拍"。
    - 混合判据：|out - ref| <= tol * (|ref| + max|ref|)。大范数下 exp(A)
      元素动态范围极宽（对消主导的小元素与 e^+90 量级大元素共存），纯逐
      元素相对容差在 fp32 下不可实现；绝对余量随 max|ref| 缩放后，实测
      本算子以 18~40x 余量通过各桶（<=8: 3.5e-7/1e-5，<=64: 5.7e-6/1e-4，
      <=1e3: 3.0e-5/1e-3）。

覆盖设计文档的关键路径：
    - n == 1 纯 Vector 快速路径（batch 维逐元素 Exp）
    - n 非 16 倍数（3/15/17）与 16 边界（16/32）及 n=128 上限
    - 缩放平方触发（大范数 ||A||_F > 0.5，s > 0 平方路径）与 s == 0 直接 Taylor
    - fp32 / fp16 / bf16 三 dtype（fp16/bf16 输出量化主导容差）
    - out= 变体、非连续输入、多维 batch、重复 launch 确定性
"""

import os

import pytest
import torch

import ops_gnn

# test_matrix_exp

SUPPORTED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# n 覆盖：奇数尾块(3/15/17)、16 对齐边界(16/32)、上限(128)
TEST_NS = [2, 3, 8, 15, 16, 17, 32, 128]


def _golden(a):
    """fp64 参考（设计文档 3.1 契约）：double 精度 torch.linalg.matrix_exp。

    a 为 fp16/bf16 时 double() 为精确提升，fp64 参考与实现算法无关。
    """
    return torch.linalg.matrix_exp(a.detach().cpu().double())


def _tolerance(a, dtype):
    """按 ||A||_F 分桶的精度契约（设计文档 3.1），fp16/bf16 由输出量化主导放宽。"""
    norm = a.detach().cpu().double().norm().item()
    if norm <= 8.0:
        tol = 1e-5
    elif norm <= 64.0:
        tol = 1e-4
    elif norm <= 1e3:
        tol = 1e-3
    else:
        tol = 1e-2
    if dtype == torch.float16:
        tol = max(tol, 2e-3)   # fp16 输出量化 half-ulp ~5e-4
    elif dtype == torch.bfloat16:
        tol = max(tol, 2e-2)   # bf16 输出量化 half-ulp ~2e-3
    return tol


def _assert_close(result, a, dtype):
    """混合判据 |out - ref| <= tol * (|ref| + max|ref|)。

    exp 结果在大范数下动态范围极宽（对消小元素与 e^+90 大元素共存），
    atol 随 max|ref| 缩放，见模块 docstring 的方法学说明。
    """
    expected = _golden(a)
    tol = _tolerance(a, dtype)
    scale = expected.abs().max().item()
    torch.testing.assert_close(
        result.detach().cpu().double(), expected, rtol=tol, atol=tol * scale)


def _rand_matrix(batch, n, scale, dtype, seed=0):
    """生成 batch 个 n×n 矩阵，元素级 scale 控制范数桶。"""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(*(batch, n, n), generator=g) * scale
    return a.to(dtype)


def test_n1_fast_path():
    """n == 1：exp([[a]]) = [[e^a]]，批量逐元素。"""
    for batch in (1, 128, 4096):
        a = _rand_matrix(batch, 1, 1.0, torch.float32, seed=batch)
        out = ops_gnn.matrix_exp(a.npu())
        _assert_close(out, a, torch.float32)


@pytest.mark.parametrize("n", TEST_NS)
def test_small_norm_fp32(n):
    """小范数（||A||_F < 0.5）：s == 0，纯 Horner 无平方。"""
    a = _rand_matrix(4, n, 0.5 / (n * n), torch.float32, seed=n)
    out = ops_gnn.matrix_exp(a.npu())
    assert out.device.type == "npu"
    assert out.shape == a.shape and out.dtype == torch.float32
    _assert_close(out, a, torch.float32)


@pytest.mark.parametrize("n", [2, 3, 16, 17, 32, 128])
def test_scaled_squaring_fp32(n):
    """大范数：触发 s > 0 缩放平方路径（含奇数 n 尾块）。"""
    for scale in (1.0, 4.0):
        a = _rand_matrix(4, n, scale, torch.float32, seed=n + int(scale * 10))
        out = ops_gnn.matrix_exp(a.npu())
        _assert_close(out, a, torch.float32)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_dtypes(dtype):
    """三 dtype：fp16/bf16 输入量化主导容差。"""
    a = _rand_matrix(3, 16, 0.05, dtype, seed=7)
    out = ops_gnn.matrix_exp(a.npu())
    assert out.dtype == dtype
    _assert_close(out, a, dtype)


def test_multidim_batch():
    """多维 batch：(2, 3, n, n) 展平为 6 个矩阵。"""
    g = torch.Generator().manual_seed(11)
    a = (torch.randn(2, 3, 24, 24, generator=g) * 0.1).to(torch.float32)
    out = ops_gnn.matrix_exp(a.npu())
    _assert_close(out, a, torch.float32)


def test_out_variant():
    """out= 变体：返回 out 本身且写入正确。"""
    a = _rand_matrix(2, 8, 0.2, torch.float32, seed=13)
    out = torch.empty_like(a).npu()
    ret = ops_gnn.matrix_exp(a.npu(), out=out)
    assert ret is out
    _assert_close(ret, a, torch.float32)


def test_non_contiguous_input():
    """非连续输入：转置视图（对称性不成立时仍应拷贝后计算正确）。"""
    g = torch.Generator().manual_seed(17)
    a = torch.randn(2, 12, 12, generator=g) * 0.2
    at = a.mT  # 非连续视图
    assert not at.is_contiguous()
    out = ops_gnn.matrix_exp(at.npu())
    _assert_close(out, at, torch.float32)


def test_deterministic_repeat_launch():
    """同一 stream 连续 launch：结果 bitwise 一致（重复执行稳定性）。"""
    a = _rand_matrix(8, 24, 0.5, torch.float32, seed=19)
    x = a.npu()
    r1 = ops_gnn.matrix_exp(x)
    r2 = ops_gnn.matrix_exp(x)
    assert torch.equal(r1.cpu(), r2.cpu())


def test_special_matrices():
    """零矩阵→I；单位阵→e·I；对角阵→逐对角 exp（大负对角触发大 s）。"""
    n = 8
    # 零矩阵
    a = torch.zeros(1, n, n)
    out = ops_gnn.matrix_exp(a.npu())
    _assert_close(out, a, torch.float32)
    # 单位矩阵
    a = torch.eye(n).expand(1, n, n).contiguous()
    out = ops_gnn.matrix_exp(a.npu())
    _assert_close(out, a, torch.float32)
    # 对角矩阵：正/负大值（负大范数验证缩放平方的数值稳定性）
    diag = torch.diag(torch.tensor([3.0, -3.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0]))
    a = diag.expand(2, n, n).contiguous()
    out = ops_gnn.matrix_exp(a.npu())
    _assert_close(out, a, torch.float32)


def test_batch_larger_than_cores():
    """batch 远大于核数：多轮矩阵循环复用缓冲与 flag 配对。"""
    a = _rand_matrix(64, 4, 0.3, torch.float32, seed=23)
    out = ops_gnn.matrix_exp(a.npu())
    _assert_close(out, a, torch.float32)


@pytest.mark.parametrize("n", [129, 1024])  # 超出迭代一 n<=128 支持域
def test_n_above_supported(n):
    """n > 128：迭代一明确报错（模板三未实现）。"""
    a = torch.eye(n).npu()
    with pytest.raises(RuntimeError, match="not supported yet"):
        ops_gnn.matrix_exp(a)


def test_rejects_non_square():
    """非方阵拒绝。"""
    a = torch.randn(4, 3).npu()
    with pytest.raises(RuntimeError, match="square"):
        ops_gnn.matrix_exp(a)

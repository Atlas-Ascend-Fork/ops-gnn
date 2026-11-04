# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CSR SDDMM tests covering every message op, target and dtype."""
import os
import runpy
from pathlib import Path

import pytest
import torch

sddmm_reference = runpy.run_path(Path(__file__).with_name("golden.py"))["sddmm_reference"]

_MESSAGE_OPS = ["dot", "add", "sub", "mul", "div", "copy_lhs", "copy_rhs"]
_TARGET_PAIRS = [(0, 2), (2, 0), (0, 0), (1, 1), (0, 1),
                 (1, 2), (2, 1), (2, 2), (1, 0)]


@pytest.fixture
def op():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import sddmm_csr
    return sddmm_csr


def make_csr(num_src, num_dst, num_edges, index_dtype, seed=42):
    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(num_src, (num_edges,), generator=generator)
    dst = torch.randint(num_dst, (num_edges,), generator=generator)
    order = torch.argsort(src, stable=True)
    counts = torch.bincount(src[order], minlength=num_src)
    ptr = torch.cat([
        torch.zeros(1, dtype=torch.long),
        counts.cumsum(0),
    ]).to(index_dtype)
    return ptr, dst[order].to(index_dtype)


def _operands(op, rows, dim, dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    lhs = torch.randn(rows, dim, generator=generator, dtype=dtype)
    rhs = torch.randn(rows, dim, generator=generator, dtype=dtype)
    if op == "div":
        rhs = rhs.abs() + 0.5
    return lhs, rhs


def _invoke(func, op_name, ptr, idx, lhs, rhs, lhs_target, rhs_target, out=None):
    kwargs = {"op": op_name, "lhs_target": lhs_target, "rhs_target": rhs_target}
    if op_name == "copy_rhs":
        return func(ptr, idx, None, rhs, out=out, **kwargs)
    if op_name == "copy_lhs":
        return func(ptr, idx, lhs, None, out=out, **kwargs)
    return func(ptr, idx, lhs, rhs, out=out, **kwargs)


def _reference(op, ptr, idx, lhs, rhs, lhs_target, rhs_target):
    if op == "copy_rhs":
        return sddmm_reference(ptr, idx, None, rhs, op, lhs_target, rhs_target)
    if op == "copy_lhs":
        return sddmm_reference(ptr, idx, lhs, None, op, lhs_target, rhs_target)
    return sddmm_reference(ptr, idx, lhs, rhs, op, lhs_target, rhs_target)


def assert_close(got, expected, op, dtype):
    got = got.cpu()
    expected = expected.cpu()
    if op == "copy_lhs" or op == "copy_rhs":
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        return
    if dtype == torch.float16:
        if op == "dot":
            torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2,
                                       equal_nan=True)
        else:
            torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3,
                                       equal_nan=True)
        return
    if op == "dot":
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5,
                                   equal_nan=True)
    else:
        torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6,
                                   equal_nan=True)


@pytest.mark.parametrize("op_name", _MESSAGE_OPS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_message_ops(op, op_name, dtype, index_dtype):
    ptr, idx = make_csr(13, 11, 40, index_dtype)
    lhs, rhs = _operands(op_name, 48, 5, dtype)
    expected = _reference(op_name, ptr, idx, lhs, rhs, 0, 2)
    got = _invoke(op, op_name, ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), 0, 2)
    assert_close(got, expected, op_name, dtype)


@pytest.mark.parametrize("op_name", ["dot", "add"])
@pytest.mark.parametrize("lhs_target,rhs_target", _TARGET_PAIRS)
def test_targets(op, op_name, lhs_target, rhs_target):
    ptr, idx = make_csr(9, 8, 25, torch.int64)
    lhs, rhs = _operands(op_name, 40, 4, torch.float32, seed=3)
    expected = _reference(op_name, ptr, idx, lhs, rhs, lhs_target, rhs_target)
    got = _invoke(op, op_name, ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(),
                  lhs_target, rhs_target)
    assert_close(got, expected, op_name, torch.float32)


@pytest.mark.parametrize("target", [0, 1, 2])
def test_copy_targets(op, target):
    ptr, idx = make_csr(9, 8, 17, torch.int64)
    lhs, rhs = _operands("copy_lhs", 30, 4, torch.float32, seed=5)
    expected = _reference("copy_lhs", ptr, idx, lhs, None, target, 2)
    got = _invoke(op, "copy_lhs", ptr.npu(), idx.npu(), lhs.npu(), None, target, 2)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)

    expected = _reference("copy_rhs", ptr, idx, None, rhs, 0, target)
    got = _invoke(op, "copy_rhs", ptr.npu(), idx.npu(), None, rhs.npu(), 0, target)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("op_name", ["copy_lhs", "copy_rhs"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_copy_multi_batch(op, op_name, dtype):
    ptr, idx = make_csr(512, 512, 8192, torch.int64, seed=101)
    generator = torch.Generator().manual_seed(103)
    feature = torch.randn(8192, 4096, generator=generator, dtype=dtype)
    if op_name == "copy_lhs":
        got = op(ptr.npu(), idx.npu(), feature.npu(), None, op=op_name,
                 lhs_target=1, rhs_target=2)
    else:
        got = op(ptr.npu(), idx.npu(), None, feature.npu(), op=op_name,
                 lhs_target=0, rhs_target=1)
    torch.testing.assert_close(got.cpu(), feature, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_dot_multi_batch(op, dtype):
    ptr, idx = make_csr(512, 512, 8192, torch.int64, seed=201)
    lhs, rhs = _operands("dot", 512, 1024, dtype, seed=203)
    expected = _reference("dot", ptr, idx, lhs, rhs, 0, 2)
    got = _invoke(op, "dot", ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), 0, 2)
    assert_close(got, expected, "dot", dtype)


@pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_binary_multi_batch(op, op_name, dtype):
    ptr, idx = make_csr(512, 512, 8192, torch.int64, seed=211)
    lhs, rhs = _operands(op_name, 512, 1024, dtype, seed=213)
    expected = _reference(op_name, ptr, idx, lhs, rhs, 0, 2)
    got = _invoke(op, op_name, ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), 0, 2)
    assert_close(got, expected, op_name, dtype)


@pytest.mark.parametrize("op_name", ["copy_lhs", "copy_rhs"])
@pytest.mark.parametrize("target", [0, 2])
def test_copy_multi_batch_targets(op, op_name, target):
    ptr, idx = make_csr(512, 512, 8192, torch.int64, seed=221)
    generator = torch.Generator().manual_seed(223)
    feature = torch.randn(8192, 4096, generator=generator, dtype=torch.float16)
    if op_name == "copy_lhs":
        lhs, rhs = feature, None
        lhs_target, rhs_target = target, 2
    else:
        lhs, rhs = None, feature
        lhs_target, rhs_target = 0, target
    expected = _reference(op_name, ptr, idx, lhs, rhs, lhs_target, rhs_target)
    got = _invoke(op, op_name, ptr.npu(), idx.npu(),
                  lhs.npu() if lhs is not None else None,
                  rhs.npu() if rhs is not None else None,
                  lhs_target, rhs_target)
    assert_close(got, expected, op_name, torch.float16)


@pytest.mark.parametrize("dim", [1, 7, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_feature_dims(op, dim, dtype):
    ptr, idx = make_csr(7, 9, 21, torch.int64)
    lhs, rhs = _operands("dot", 26, dim, dtype, seed=7)
    expected = _reference("dot", ptr, idx, lhs, rhs, 0, 2)
    got = _invoke(op, "dot", ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), 0, 2)
    assert_close(got, expected, "dot", dtype)


@pytest.mark.parametrize("op_name", _MESSAGE_OPS)
def test_scalar_features(op, op_name):
    ptr, idx = make_csr(6, 5, 13, torch.int32)
    lhs, rhs = _operands(op_name, 20, 1, torch.float32, seed=9)
    lhs = lhs.squeeze(-1)
    rhs = rhs.squeeze(-1)
    expected = _reference(op_name, ptr, idx, lhs, rhs, 0, 2)
    got = _invoke(op, op_name, ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), 0, 2)
    assert_close(got, expected, op_name, torch.float32)


def test_scalar_out(op):
    ptr, idx = make_csr(6, 5, 13, torch.int64, seed=31)
    lhs, rhs = _operands("add", 20, 1, torch.float32, seed=33)
    lhs = lhs.squeeze(-1)
    rhs = rhs.squeeze(-1)
    expected = _reference("add", ptr, idx, lhs, rhs, 0, 2)
    out = torch.empty(idx.numel(), dtype=torch.float32, device="npu")
    got = _invoke(op, "add", ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), 0, 2,
                  out=out)
    if got.data_ptr() != out.data_ptr():
        raise AssertionError("sddmm_csr did not return the supplied out tensor")
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-6, atol=1e-6)


def test_dot_and_elementwise_shapes(op):
    ptr, idx = make_csr(5, 4, 9, torch.int64, seed=41)
    lhs, rhs = _operands("dot", 12, 3, torch.float32, seed=43)
    dot = op(ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), op="dot")
    assert tuple(dot.shape) == (idx.numel(),)
    add = op(ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), op="add")
    assert tuple(add.shape) == (idx.numel(), 3)
    scalar_lhs = lhs[:, 0]
    scalar_rhs = rhs[:, 0]
    scalar = op(ptr.npu(), idx.npu(), scalar_lhs.npu(), scalar_rhs.npu(), op="add")
    assert tuple(scalar.shape) == (idx.numel(),)


def test_bad_argument_types(op):
    ptr = torch.tensor([0, 1], dtype=torch.int64)
    idx = torch.tensor([0], dtype=torch.int64)
    lhs = torch.ones(2, 3, dtype=torch.float32)
    rhs = torch.ones(2, 3, dtype=torch.float32)
    with pytest.raises(TypeError, match="sddmm_csr"):
        op([0, 1], idx.npu(), lhs.npu(), rhs.npu())
    with pytest.raises(TypeError, match="sddmm_csr"):
        op(ptr.npu(), [0], lhs.npu(), rhs.npu())


@pytest.mark.parametrize("op_name", ["dot", "add", "copy_lhs"])
def test_empty(op, op_name):
    ptr = torch.tensor([0, 0, 0], dtype=torch.int64, device="npu")
    idx = torch.empty(0, dtype=torch.int64, device="npu")
    lhs, rhs = _operands(op_name, 5, 7, torch.float32, seed=11)
    expected = _reference(op_name, ptr.cpu(), idx.cpu(), lhs, rhs, 0, 2)
    got = _invoke(op, op_name, ptr, idx, lhs.npu(), rhs.npu(), 0, 2)
    assert got.shape == expected.shape
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("op_name", ["dot", "add", "div"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_special_values(op, op_name, dtype):
    ptr = torch.tensor([0, 2, 4, 5])
    idx = torch.tensor([0, 1, 1, 0, 0])
    lhs = torch.tensor([[float("inf"), float("nan"), 1, -2],
                        [-float("inf"), 3, 4, float("nan")],
                        [0, 1, -1, 2],
                        [float("nan"), -3, 5, 6]], dtype=dtype)
    rhs = torch.tensor([[1, 0, float("inf"), -3],
                        [2, float("nan"), -4, 5],
                        [0, -1, 2, float("nan")],
                        [3, 4, 5, 6]], dtype=dtype)
    expected = _reference(op_name, ptr, idx, lhs, rhs, 0, 2)
    got = _invoke(op, op_name, ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), 0, 2)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0,
                               equal_nan=True)


def test_strides_out_and_repetition(op):
    ptr, idx = make_csr(11, 7, 31, torch.int64, seed=13)
    lhs = torch.randn(6, 40, dtype=torch.float16, device="npu").t()
    rhs = torch.randn(6, 40, dtype=torch.float16, device="npu").t()
    lhs_cpu, rhs_cpu = lhs.cpu(), rhs.cpu()
    expected = _reference("add", ptr, idx, lhs_cpu, rhs_cpu, 0, 2)
    out = torch.empty((idx.numel(), 6), dtype=torch.float16, device="npu")
    for _ in range(50):
        got = _invoke(op, "add", ptr.npu(), idx.npu(), lhs, rhs, 0, 2, out=out)
        if got.data_ptr() != out.data_ptr():
            raise AssertionError("sddmm_csr did not return the supplied out tensor")
        torch.testing.assert_close(got.cpu(), expected, rtol=2e-3, atol=2e-3)

    bad = torch.empty((idx.numel() + 1, 6), dtype=torch.float16, device="npu")
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        _invoke(op, "add", ptr.npu(), idx.npu(), lhs, rhs, 0, 2, out=bad)


@pytest.mark.parametrize("ptr_values,index_values", [
    ([1, 1], [0]),
    ([0, 2], [0]),
    ([0, 2, 1, 2], [0, 0]),
    ([0, -1, 1], [0]),
    ([0, 1], [-1]),
    ([0, 1], [4]),
    ([0, 1], [2**32]),
])
def test_bad_csr(op, ptr_values, index_values):
    lhs = torch.ones(4, 3, dtype=torch.float32, device="npu")
    rhs = torch.ones(4, 3, dtype=torch.float32, device="npu")
    with pytest.raises(RuntimeError, match="sddmm_csr"):
        op(torch.tensor(ptr_values, device="npu"),
           torch.tensor(index_values, device="npu"), lhs, rhs)


def test_bad_metadata(op):
    ptr = torch.tensor([0, 1, 2], device="npu")
    idx = torch.tensor([0, 1], device="npu")
    lhs = torch.ones(2, 17, device="npu", dtype=torch.float16)
    rhs = torch.ones(2, 17, device="npu", dtype=torch.float16)

    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs.cpu(), rhs.cpu())
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs.bfloat16(), rhs.bfloat16())
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs.unsqueeze(0), rhs.unsqueeze(0))
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs, rhs.float())
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr.unsqueeze(0), idx, lhs, rhs)
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr[:0], idx, lhs, rhs)
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs[:, :0], rhs[:, :0])
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs, rhs, op="bad_op")
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs, rhs, lhs_target=9)

    out = torch.empty(1, 17, device="npu", dtype=torch.float16)
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs, rhs, op="add", out=out)
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs, rhs, op="add",
           out=torch.empty(2, 17, device="npu", dtype=torch.float32))
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs, rhs, op="add",
           out=torch.empty(2, 17, device="cpu", dtype=torch.float16))
    with pytest.raises((ValueError, RuntimeError), match="sddmm_csr"):
        op(ptr, idx, lhs, rhs, op="add", out=lhs)


def test_large_random_graph(op):
    ptr, idx = make_csr(1000, 1000, 5000, torch.int64, seed=17)
    lhs, rhs = _operands("dot", 1000, 64, torch.float16, seed=19)
    expected = _reference("dot", ptr, idx, lhs, rhs, 0, 2)
    got = _invoke(op, "dot", ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), 0, 2)
    assert_close(got, expected, "dot", torch.float16)


def test_stream(op):
    ptr, idx = make_csr(6, 6, 15, torch.int64, seed=21)
    lhs, rhs = _operands("dot", 20, 4, torch.float16, seed=23)
    expected = _reference("dot", ptr, idx, lhs, rhs, 0, 2)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        got = _invoke(op, "dot", ptr.npu(), idx.npu(), lhs.npu(), rhs.npu(), 0, 2)
    torch.npu.current_stream().wait_stream(stream)
    torch.testing.assert_close(got.cpu(), expected, rtol=2e-2, atol=2e-2)

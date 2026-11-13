# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Unified SpMM tests, including two-dimensional forward parity cases.

The copy_rhs interface receives edge features already stored in CSR order.
"""
import ast
import math
import os
import runpy
from pathlib import Path

import pytest
import torch

unified_spmm_reference = runpy.run_path(Path(__file__).with_name("golden.py"))[
    "unified_spmm_reference"]


@pytest.fixture
def op():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import spmm
    return spmm


@pytest.fixture
def bop():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import bspmm
    return bspmm


def make_destination_csr(num_src, num_dst, num_edges, index_dtype, seed=42):
    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(num_src, (num_edges,), generator=generator)
    dst = torch.randint(num_dst, (num_edges,), generator=generator)
    order = torch.argsort(dst, stable=True)
    ptr = torch.cat([
        torch.zeros(1, dtype=torch.long),
        torch.bincount(dst, minlength=num_dst).cumsum(0),
    ]).to(index_dtype)
    return ptr, src[order].to(index_dtype), order


def assert_reference_result(got, expected, reduce, dtype):
    got = got.cpu()
    if reduce in ("min", "max") or dtype == torch.float32:
        assert torch.equal(got, expected)
    else:
        torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)


def test_reference():
    ptr = torch.tensor([0, 0, 2, 3])
    idx = torch.tensor([0, 2, 1])
    x = torch.tensor([[1, -2], [3, 4], [-5, 6]], dtype=torch.float16)
    assert torch.equal(unified_spmm_reference(ptr, idx, x, "sum"),
                       torch.tensor([[0, 0], [-4, 4], [3, 4]], dtype=torch.float16))
    assert torch.equal(unified_spmm_reference(ptr, idx, x, "min"),
                       torch.tensor([[0, 0], [-5, -2], [3, 4]], dtype=torch.float16))
    assert torch.equal(unified_spmm_reference(ptr, idx, x, "max"),
                       torch.tensor([[0, 0], [1, 6], [3, 4]], dtype=torch.float16))


@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("n", [1, 15, 16, 17, 33, 127])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_copy_lhs(op, reduce, dtype, n, index_dtype):
    ptr = torch.tensor([0, 0, 3, 4, 7], dtype=index_dtype)
    idx = torch.tensor([0, 2, 1, 3, 1, 2, 0], dtype=index_dtype)
    x = ((torch.arange(4 * n).reshape(4, n) % 11) - 5).to(dtype)
    expected = unified_spmm_reference(ptr, idx, x, reduce)
    got = op(ptr.npu(), idx.npu(), x.npu(), reduce=reduce)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, 17, 33])
@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
def test_copy_rhs(op, n, reduce):
    ptr = torch.tensor([0, 0, 3, 5])
    idx = torch.tensor([2, 0, 1, 1, 0])
    edge_features = ((torch.arange(5 * n).reshape(5, n) % 7) - 3).float()
    expected = unified_spmm_reference(ptr, idx, edge_features, reduce, "copy_rhs")
    got = op(ptr.npu(), idx.npu(), edge_features.npu(),
             op="copy_rhs", reduce=reduce)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_special_values(op, reduce, dtype):
    ptr = torch.tensor([0, 1, 3, 3])
    idx = torch.tensor([0, 0, 1])
    x = torch.tensor([[float("inf"), float("nan"), -2],
                      [-float("inf"), 3, 4]], dtype=dtype)
    expected = unified_spmm_reference(ptr, idx, x, reduce)
    torch.testing.assert_close(op(ptr.npu(), idx.npu(), x.npu(), reduce=reduce).cpu(),
                               expected, equal_nan=True, rtol=0, atol=0)


@pytest.mark.parametrize("m,k", [(0, 0), (0, 2), (1, 0), (5, 0), (5, 2)])
def test_empty(op, m, k):
    ptr = torch.zeros(m + 1, dtype=torch.int64, device="npu")
    idx = torch.empty(0, dtype=torch.int64, device="npu")
    x = torch.empty(k, 17, dtype=torch.float32, device="npu")
    torch.testing.assert_close(op(ptr, idx, x).cpu(), torch.zeros(m, 17))


def test_fp16_mean_large_degree(op):
    degree = 1 << 16
    ptr = torch.tensor([0, degree], dtype=torch.int64)
    idx = torch.zeros(degree, dtype=torch.int64)
    x = torch.ones((1, 1), dtype=torch.float16)
    rhs = torch.zeros((degree, 1), dtype=torch.float16)
    rhs[0, 0] = 1

    got = op(ptr.npu(), idx.npu(), x.npu(), op="mul", reduce="mean",
             rhs=rhs.npu())
    expected = torch.tensor([[1.0 / degree]], dtype=torch.float16)

    assert got.dtype == torch.float16
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


def test_noncontiguous_and_out(op):
    ptr = torch.tensor([0, 99, 2, 99, 3, 99], device="npu")[::2]
    idx = torch.tensor([0, 99, 1, 99, 0, 99], device="npu")[::2]
    x = torch.arange(34, device="npu", dtype=torch.float32).reshape(17, 2).t()
    out = torch.empty(2, 17, device="npu", dtype=torch.float32)
    expected = unified_spmm_reference(ptr.cpu(), idx.cpu(), x.cpu(), "sum")
    got = op(ptr, idx, x, out=out)
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


def test_multiple_batches(op):
    ptr = torch.tensor([0, 10001, 10001, 10002])
    idx = torch.arange(10002) % 32
    x = ((torch.arange(32 * 33).reshape(32, 33) % 5) - 2).to(torch.float16)
    expected = unified_spmm_reference(ptr, idx, x, "sum")
    torch.testing.assert_close(op(ptr.npu(), idx.npu(), x.npu()).cpu(),
                               expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduce,spmm_op", [("median", "copy_lhs"),
                                              ("sum", "copy_src")])
def test_bad_modes(op, reduce, spmm_op):
    ptr = torch.tensor([0, 1], device="npu")
    idx = torch.tensor([0], device="npu")
    x = torch.ones(1, 17, device="npu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="spmm"):
        op(ptr, idx, x, op=spmm_op, reduce=reduce)


def test_bad_copy_rhs_shape(op):
    ptr = torch.tensor([0, 2], device="npu")
    idx = torch.tensor([0, 0], device="npu")
    x = torch.ones(1, 17, device="npu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="x.size"):
        op(ptr, idx, x, op="copy_rhs")


@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
def test_copy_rhs_rejects_fp16(op, reduce):
    ptr = torch.tensor([0, 1], device="npu")
    idx = torch.tensor([0], device="npu")
    x = torch.ones(1, 17, dtype=torch.float16, device="npu")
    with pytest.raises(RuntimeError, match="copy_rhs supports only float32"):
        op(ptr, idx, x, op="copy_rhs", reduce=reduce)


@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
def test_copy_rhs_special_values(op, reduce):
    ptr = torch.tensor([0, 1, 3, 3])
    idx = torch.tensor([0, 0, 0])
    edge_features = torch.tensor([
        [float("inf"), float("nan"), -2],
        [-float("inf"), 3, 4],
        [5, -1, float("nan")],
    ], dtype=torch.float32)
    expected = unified_spmm_reference(
        ptr, idx, edge_features, reduce, "copy_rhs")
    got = op(ptr.npu(), idx.npu(), edge_features.npu(),
             op="copy_rhs", reduce=reduce)
    torch.testing.assert_close(
        got.cpu(), expected, equal_nan=True, rtol=0, atol=0)


@pytest.mark.parametrize("ptr,idx", [([1, 1], [0]), ([0, 2], [0]),
                                      ([0, 2, 1, 2], [0, 0]),
                                      ([0, -1, 1], [0]), ([0, 1], [-1]),
                                      ([0, 1], [2]), ([0, 1], [2**32]),
                                      ([0, 2**32], [0])])
def test_bad_csr(op, ptr, idx):
    with pytest.raises(RuntimeError, match="spmm"):
        op(torch.tensor(ptr, device="npu"), torch.tensor(idx, device="npu"),
           torch.ones(2, 17, dtype=torch.float32, device="npu"))


@pytest.mark.parametrize("case", ["cpu", "bf16", "rank", "dtype",
                                  "ptr_rank", "empty_ptr", "zero_dim",
                                  "out_shape", "out_dtype", "out_cpu", "alias", "out_grad"])
def test_bad_metadata(op, case):
    ptr = torch.tensor([0, 1, 2], device="npu")
    idx = torch.tensor([0, 1], device="npu")
    x = torch.ones(2, 17, device="npu", dtype=torch.float32)
    out = None
    if case == "cpu":
        x = x.cpu()
    if case == "bf16":
        x = x.bfloat16()
    if case == "rank":
        x = x.unsqueeze(0)
    if case == "dtype":
        idx = idx.int()
    if case == "ptr_rank":
        ptr = ptr.unsqueeze(0)
    if case == "empty_ptr":
        ptr = ptr[:0]
    if case == "zero_dim":
        x = x[:, :0]
    if case == "out_shape":
        out = torch.empty(1, 17, device="npu")
    if case == "out_dtype":
        out = x.half()
    if case == "out_cpu":
        out = x.cpu()
    if case == "alias":
        out = x
    if case == "out_grad":
        out = torch.empty(
            x.shape, dtype=x.dtype, device=x.device, requires_grad=True)
    with pytest.raises(RuntimeError, match="spmm"):
        op(ptr, idx, x, out=out)


def test_ub_capacity_rejected(op):
    ptr = torch.tensor([0, 1], device="npu")
    idx = torch.tensor([0], device="npu")
    x = torch.empty(1, 2**20, dtype=torch.float32, device="npu")
    with pytest.raises(RuntimeError, match="UB capacity"):
        op(ptr, idx, x)


def test_stream(op):
    stream = torch.npu.Stream()
    ptr = torch.tensor([0, 1, 2], device="npu")
    idx = torch.tensor([1, 0], device="npu")
    x = torch.ones(2, 17, device="npu", dtype=torch.float32)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        got = op(ptr, idx, x * -3) + 2
    torch.npu.current_stream().wait_stream(stream)
    torch.testing.assert_close(got.cpu(), torch.full((2, 17), -1.0))


@pytest.mark.parametrize("reduce", ["sum", "max", "min"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_reference_copy_lhs_dtype_reduce_idtype(op, reduce, dtype, index_dtype):
    ptr, idx, _ = make_destination_csr(10, 10, 30, index_dtype)
    x = torch.rand(10, 4, generator=torch.Generator().manual_seed(7)).to(dtype)
    expected = unified_spmm_reference(ptr, idx, x, reduce)
    got = op(ptr.npu(), idx.npu(), x.npu(), reduce=reduce)
    assert_reference_result(got, expected, reduce, dtype)


@pytest.mark.parametrize("feature_dim", [1, 4, 13, 64, 128, 256])
def test_reference_copy_lhs_feature_dimensions(op, feature_dim):
    ptr, idx, _ = make_destination_csr(10, 10, 30, torch.int64)
    x = torch.rand(10, feature_dim,
                   generator=torch.Generator().manual_seed(feature_dim))
    expected = unified_spmm_reference(ptr, idx, x, "sum")
    got = op(ptr.npu(), idx.npu(), x.npu())
    assert torch.equal(got.cpu(), expected)


def test_large_bipartite_copy_lhs(op):
    ptr, idx, _ = make_destination_csr(
        1000, 1200, 5000, torch.int64, seed=123)
    x = torch.rand(1000, 64, generator=torch.Generator().manual_seed(11))
    expected = unified_spmm_reference(ptr, idx, x, "sum")
    got = op(ptr.npu(), idx.npu(), x.npu())
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=1e-3)


@pytest.mark.parametrize("reduce", ["sum", "max", "min"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_copy_lhs_repeated_calls(op, reduce, dtype):
    ptr, idx, _ = make_destination_csr(20, 20, 100, torch.int64)
    x = torch.rand(20, 13, generator=torch.Generator().manual_seed(21)).to(dtype)
    ptr_npu, idx_npu, x_npu = ptr.npu(), idx.npu(), x.npu()
    results = [
        op(ptr_npu, idx_npu, x_npu, reduce=reduce).cpu()
        for _ in range(5)
    ]
    for result in results[1:]:
        assert torch.equal(result, results[0])


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_copy_rhs_asymmetric_reorder(op, index_dtype):
    # Original edges: 0->1, 0->2, 1->2, 2->0. The inbound CSR edge data
    # permutation is [3, 0, 1, 2], so edge features follow that order.
    ptr = torch.tensor([0, 1, 2, 4], dtype=index_dtype)
    idx = torch.tensor([2, 0, 0, 1], dtype=index_dtype)
    original = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
    edge_features = original[torch.tensor([3, 0, 1, 2])]
    expected = torch.tensor([[4], [1], [5]], dtype=torch.float32)
    got = op(ptr.npu(), idx.npu(), edge_features.npu(), op="copy_rhs")
    assert torch.equal(got.cpu(), expected)


@pytest.mark.parametrize("feature_dim", [1, 4, 13, 64, 128, 256])
@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
def test_copy_rhs_feature_dimensions(op, feature_dim, reduce):
    ptr, idx, _ = make_destination_csr(10, 10, 30, torch.int64)
    edge_features = torch.rand(
        30, feature_dim, generator=torch.Generator().manual_seed(feature_dim + 1)
    )
    expected = unified_spmm_reference(
        ptr, idx, edge_features, reduce, "copy_rhs")
    got = op(ptr.npu(), idx.npu(), edge_features.npu(),
             op="copy_rhs", reduce=reduce)
    assert_reference_result(got, expected, reduce, torch.float32)


@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
def test_copy_rhs_edge_cases(op, reduce):
    cases = [
        ([0, 0, 1], [[1, 2, 3]]),
        ([0, 1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ([0, 0, 1, 2], [[1, 2, 3], [4, 5, 6]]),
        ([0, 0, 0, 0, 0, 0], []),
    ]
    for ptr_values, feature_values in cases:
        ptr = torch.tensor(ptr_values, dtype=torch.int64)
        nnz = ptr_values[-1]
        idx = torch.zeros(nnz, dtype=torch.int64)
        edge_features = (torch.tensor(feature_values, dtype=torch.float32)
                         if nnz else torch.empty(0, 3, dtype=torch.float32))
        expected = unified_spmm_reference(
            ptr, idx, edge_features, reduce, "copy_rhs")
        got = op(ptr.npu(), idx.npu(), edge_features.npu(),
                 op="copy_rhs", reduce=reduce)
        assert_reference_result(got, expected, reduce, torch.float32)


@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
def test_copy_rhs_matches_copy_lhs(op, reduce):
    ptr, idx, _ = make_destination_csr(5, 5, 10, torch.int64)
    node_features = torch.rand(
        5, 4, generator=torch.Generator().manual_seed(31))
    edge_features = node_features[idx.long()]
    ptr_npu, idx_npu = ptr.npu(), idx.npu()
    lhs = op(ptr_npu, idx_npu, node_features.npu(),
             op="copy_lhs", reduce=reduce)
    rhs = op(ptr_npu, idx_npu, edge_features.npu(),
             op="copy_rhs", reduce=reduce)
    assert torch.equal(lhs.cpu(), rhs.cpu())


@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
def test_copy_rhs_noncontiguous_out_and_stability(op, reduce):
    ptr = torch.tensor([0, 99, 2, 99, 5, 99], device="npu")[::2]
    idx = torch.tensor(
        [0, 99, 1, 99, 2, 99, 0, 99, 1, 99], device="npu")[::2]
    edge_features = torch.arange(
        65, dtype=torch.float32, device="npu").reshape(13, 5).t()
    out = torch.empty(2, 13, dtype=torch.float32, device="npu")
    expected = unified_spmm_reference(
        ptr.cpu(), idx.cpu(), edge_features.cpu(), reduce, "copy_rhs")
    for _ in range(5):
        got = op(ptr, idx, edge_features, op="copy_rhs", reduce=reduce, out=out)
        assert got.data_ptr() == out.data_ptr()
        assert torch.equal(got.cpu(), expected)


@pytest.mark.parametrize("message_op", ["add", "sub", "mul", "div"])
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_binary_messages(op, message_op, reduce):
    ptr = torch.tensor([0, 2, 4])
    idx = torch.tensor([0, 2, 1, 0])
    lhs = torch.tensor([[2.0, 4.0], [3.0, 8.0], [5.0, 6.0]])
    rhs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [4.0, 2.0], [5.0, 4.0]])
    gathered = lhs[idx]
    messages = {
        "add": gathered + rhs,
        "sub": gathered - rhs,
        "mul": gathered * rhs,
        "div": gathered / rhs,
    }[message_op]
    expected_rows = []
    for row in range(2):
        values = messages[ptr[row]:ptr[row + 1]]
        if reduce == "sum":
            expected_rows.append(values.sum(0))
        elif reduce == "mean":
            expected_rows.append(values.mean(0))
        elif reduce == "min":
            expected_rows.append(values.amin(0))
        else:
            expected_rows.append(values.amax(0))
    expected = torch.stack(expected_rows)
    got = op(ptr.npu(), idx.npu(), lhs.npu(), op=message_op,
             reduce=reduce, rhs=rhs.npu())
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("message_op", ["copy_lhs", "copy_rhs"])
def test_mean_and_scalar_features(op, message_op):
    ptr = torch.tensor([0, 2, 3])
    idx = torch.tensor([0, 2, 1])
    values = torch.tensor([2.0, 4.0, 8.0])
    expected = torch.tensor([5.0, 4.0]) if message_op == "copy_lhs" else torch.tensor([3.0, 8.0])
    got = op(ptr.npu(), idx.npu(), values.npu(), op=message_op, reduce="mean")
    assert got.shape == (2,)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


def test_binary_autograd(op):
    ptr = torch.tensor([0, 2, 3], device="npu")
    idx = torch.tensor([0, 2, 1], device="npu")
    lhs = torch.tensor([[2.0], [3.0], [5.0]], device="npu", requires_grad=True)
    rhs = torch.tensor([[7.0], [11.0], [13.0]], device="npu", requires_grad=True)
    result = op(ptr, idx, lhs, op="mul", reduce="sum", rhs=rhs)
    gradient = torch.ones(result.shape, dtype=result.dtype, device=result.device)
    result.backward(gradient)
    torch.testing.assert_close(lhs.grad.cpu(), torch.tensor([[7.0], [13.0], [11.0]]))
    torch.testing.assert_close(rhs.grad.cpu(), torch.tensor([[2.0], [5.0], [3.0]]))


@pytest.mark.parametrize("message_op", ["copy_lhs", "copy_rhs"])
def test_copy_autograd(op, message_op):
    ptr = torch.tensor([0, 2, 3], device="npu")
    idx = torch.tensor([0, 0, 1], device="npu")
    features = torch.tensor([[2.0], [3.0], [5.0]], device="npu",
                            requires_grad=True)
    result = op(ptr, idx, features, op=message_op, reduce="sum")
    gradient = torch.ones(result.shape, dtype=result.dtype, device=result.device)
    result.backward(gradient)
    expected = (torch.tensor([[2.0], [1.0], [0.0]])
                if message_op == "copy_lhs" else torch.ones(3, 1))
    torch.testing.assert_close(features.grad.cpu(), expected)


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_bspmm_arbitrary_rank(bop, rank, reduce):

    ptr = torch.tensor([0, 2, 3])
    idx = torch.tensor([0, 2, 1])
    shape = (3,) + tuple(range(2, rank + 1))
    size = torch.tensor(shape).prod().item()
    x = torch.arange(size, dtype=torch.float32).reshape(shape)
    expected = unified_spmm_reference(ptr, idx, x, reduce)
    got = bop(ptr.npu(), idx.npu(), x.npu(), reduce=reduce)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_bspmm_rank_alignment(bop, reduce):

    ptr = torch.tensor([0, 2, 4])
    idx = torch.tensor([0, 2, 1, 0])
    lhs = torch.arange(3 * 3, dtype=torch.float32).reshape(3, 3, 1) + 1
    rhs = torch.arange(
        4 * 2 * 3 * 4, dtype=torch.float32).reshape(4, 2, 3, 4) + 1
    expected = unified_spmm_reference(
        ptr, idx, lhs, reduce, op="mul", rhs=rhs)
    got = bop(ptr.npu(), idx.npu(), lhs.npu(), op="mul",
                reduce=reduce, rhs=rhs.npu())
    assert got.shape == (2, 2, 3, 4)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


def test_bspmm_high_rank_autograd(bop):

    ptr = torch.tensor([0, 2, 3], device="npu")
    idx = torch.tensor([0, 2, 1], device="npu")
    lhs = torch.ones(3, 2, 1, 4, device="npu", requires_grad=True)
    rhs = torch.ones(3, 1, 3, 4, device="npu", requires_grad=True)
    result = bop(ptr, idx, lhs, op="mul", reduce="sum", rhs=rhs)
    gradient = torch.ones(result.shape, dtype=result.dtype, device=result.device)
    result.backward(gradient)
    assert lhs.grad is not None and lhs.grad.shape == lhs.shape
    assert rhs.grad is not None and rhs.grad.shape == rhs.shape


@pytest.fixture
def host_reduce_messages():
    path = Path(__file__).resolve().parents[3] / "python/ops_gnn/spmm.py"
    node = next(
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_reduce_messages"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    reduce_messages = namespace.get("_reduce_messages")
    assert callable(reduce_messages)
    return reduce_messages


def test_half_mean_accumulates_before_converting_back(host_reduce_messages):
    count = 70_000
    messages = torch.ones(count, 2, dtype=torch.float16)
    result = host_reduce_messages(torch.tensor([0, count, count]), messages, "mean")
    assert result.dtype == messages.dtype
    assert torch.equal(
        result, torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=messages.dtype)
    )


def test_half_mean_preserves_cancellation(host_reduce_messages):
    messages = torch.cat(
        [torch.full((400, 1), 1000.0), torch.full((400, 1), -1000.0)]
    ).to(torch.float16)
    result = host_reduce_messages(torch.tensor([0, 800]), messages, "mean")
    assert torch.equal(result, torch.zeros(1, 1, dtype=torch.float16))


def test_small_mean_keeps_dtype_and_gradients(host_reduce_messages):
    messages = torch.tensor(
        [[1.0], [2.0], [3.0]], dtype=torch.float16, requires_grad=True
    )
    result = host_reduce_messages(torch.tensor([0, 2, 3]), messages, "mean")
    assert torch.equal(result, torch.tensor([[1.5], [3.0]], dtype=messages.dtype))
    result.sum().backward()
    assert torch.equal(
        messages.grad, torch.tensor([[0.5], [0.5], [1.0]], dtype=messages.dtype)
    )


def test_sum_and_float32_mean_keep_existing_results(host_reduce_messages):
    messages = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    ptr = torch.tensor([0, 2, 3])
    assert torch.equal(
        host_reduce_messages(ptr, messages, "mean"), torch.tensor([[1.5], [3.0]])
    )
    assert torch.equal(
        host_reduce_messages(ptr, messages.to(torch.float16), "sum"),
        torch.tensor([[3.0], [3.0]], dtype=torch.float16),
    )


@pytest.mark.parametrize("value", [65504.0, -65504.0])
def test_half_mean_keeps_maximum_finite_values(host_reduce_messages, value):
    count = 70_000
    messages = torch.full((count, 2), value, dtype=torch.float16)
    result = host_reduce_messages(torch.tensor([0, count, count]), messages, "mean")
    assert torch.equal(
        result, torch.tensor([[value, value], [0.0, 0.0]], dtype=torch.float16)
    )


@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
@pytest.mark.parametrize("feature_shape", [(), (17,), (3, 17), (0,), (0, 17)])
@pytest.mark.parametrize("rows", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("fp32", [False, True])
def test_specialized_reference_empty_source(reduction, feature_shape, rows, dtype, fp32):
    path = Path(__file__).parents[2] / f"spmm_{reduction}" / "arch22" / "golden.py"
    reference = runpy.run_path(path)[f"spmm_{reduction}_reference"]
    indptr = torch.zeros(rows + 1, dtype=torch.int64)
    indices = torch.empty(0, dtype=torch.int64)
    features = torch.empty((0,) + feature_shape, dtype=dtype)
    expected = features.new_zeros((rows,) + feature_shape)
    actual = reference(indptr, indices, features, fp32=fp32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# Host proof cache regressions (Issue #96, A2/A3, real native operators).
# Run after building this checkout, with PYTHONPATH=python. No mocked
# extension or CPU fallback: CPU-only machines explicitly skip these
# integration tests.
PATHS = ("fp16", "fp32", "batched_fp16", "copy_rhs")


@pytest.fixture(scope="module")
def operators():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend A2/A3 NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    name = torch.npu.get_device_name()
    if not any(model in name for model in ("910B", "910C", "910_93")):
        pytest.skip(f"arch22 A2/A3 regression, found {name}")
    from ops_gnn import bspmm, spmm
    return spmm, bspmm


def inputs(path, noncontiguous=False):
    ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
    idx = torch.tensor([0, 1, 0], dtype=torch.int64)
    dtype = torch.float16 if "fp16" in path else torch.float32
    shape = (3, 2, 17) if path == "batched_fp16" else (3, 17)
    x = (torch.arange(math.prod(shape)).reshape(shape) % 7 - 3).to(dtype)
    if noncontiguous:
        def strided(value):
            storage = torch.empty(value.shape[:-1] + (value.shape[-1] * 2,),
                                  device="npu", dtype=value.dtype)
            view = storage[..., ::2]
            view.copy_(value)
            return view
        return tuple(strided(value) for value in (ptr, idx, x))
    return ptr.npu(), idx.npu(), x.npu()


def invoke(operators, path, ptr, idx, x, reduce, out=None):
    operation = operators[1] if path == "batched_fp16" else operators[0]
    return operation(ptr, idx, x, reduce=reduce,
                     op="copy_rhs" if path == "copy_rhs" else "copy_lhs", out=out)


def assert_result(result, path, ptr, idx, x, reduce):
    expected = unified_spmm_reference(
        ptr.cpu(), idx.cpu(), x.cpu(), reduce,
        "copy_rhs" if path == "copy_rhs" else "copy_lhs")
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
@pytest.mark.parametrize("alias", [False, True], ids=["inplace", "data_alias"])
def test_warm_nan_mutation(operators, path, reduce, alias):
    ptr, idx, x = inputs(path)
    target = x.data if alias else x
    for _ in range(3):
        result = invoke(operators, path, ptr, idx, x, reduce)
    assert_result(result, path, ptr, idx, x, reduce)
    # Both finite->NaN and NaN->finite must invalidate the NaN proof.
    target[0].fill_(float("nan"))
    result = invoke(operators, path, ptr, idx, x, reduce)
    assert_result(result, path, ptr, idx, x, reduce)
    target[0].fill_(2)
    result = invoke(operators, path, ptr, idx, x, reduce)
    assert_result(result, path, ptr, idx, x, reduce)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("field,message", [
    ("endpoint", "indptr must start at zero and end at nnz"),
    ("range", "indptr values out of range"),
    ("order", "indptr must be non-decreasing"),
    ("index", "indices out of range"),
])
@pytest.mark.parametrize("alias", [False, True], ids=["inplace", "data_alias"])
def test_warm_csr_mutation(operators, path, field, message, alias):
    if path == "copy_rhs" and field == "index":
        pytest.skip("copy_rhs does not dereference source index values")
    ptr, idx, x = inputs(path)
    for _ in range(3):
        invoke(operators, path, ptr, idx, x, "sum")
    target = idx if field == "index" else ptr
    target = target.data if alias else target
    if field == "endpoint":
        target[0] = 1
    elif field == "range":
        target[1] = 4
    elif field == "order":
        target[1], target[2] = 3, 2
    else:
        target[0] = -1
    for _ in range(2):
        with pytest.raises(RuntimeError, match=message):
            invoke(operators, path, ptr, idx, x, "sum")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("inference,noncontiguous,supplied_out", [
    (True, False, False), (False, True, False), (False, False, True),
])
def test_layout_inference_and_out(operators, path, inference, noncontiguous,
                                  supplied_out):
    with torch.inference_mode(inference):
        ptr, idx, x = inputs(path, noncontiguous)
        out = None
        if supplied_out:
            out = torch.empty((3,) + tuple(x.shape[1:]), device=x.device,
                              dtype=x.dtype)
        for _ in range(3):
            result = invoke(operators, path, ptr, idx, x, "max", out)
        if supplied_out:
            assert result.data_ptr() == out.data_ptr()
        x[0].fill_(float("nan"))
        result = invoke(operators, path, ptr, idx, x, "max", out)
        assert_result(result, path, ptr, idx, x, "max")
        idx[0] = -1
        if path != "copy_rhs":
            with pytest.raises(RuntimeError, match="indices out of range"):
                invoke(operators, path, ptr, idx, x, "max", out)


@pytest.mark.parametrize("path", PATHS)
def test_nondefault_stream_mutation(operators, path):
    ptr, idx, x = inputs(path)
    invoke(operators, path, ptr, idx, x, "min")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        x.data[0].fill_(float("nan"))
        result = invoke(operators, path, ptr, idx, x, "min")
    stream.synchronize()
    assert_result(result, path, ptr, idx, x, "min")


@pytest.mark.parametrize("path", PATHS)
def test_full_operator_warm_proofs(operators, path):
    ptr, idx, x = inputs(path)
    # CPU activities observe host-side ATen dispatch even for NPU tensors.
    # Profile both calls in one scope, with no unrelated tensor work between
    # them. A helper-only test cannot establish full-operator cache hits.
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        for _ in range(4):
            invoke(operators, path, ptr, idx, x, "sum")
    counts = {event.key: event.count for event in profile.key_averages()}
    assert counts.get("aten::isnan", 0) == 1, counts
    assert counts.get("aten::stack", 0) == 1, counts


@pytest.mark.parametrize("path", PATHS)
def test_fresh_input_proofs_are_recomputed(operators, path):
    benchmark = runpy.run_path(str(Path(__file__).with_name("benchmark_unified_spmm.py")))
    case = next(benchmark["iter_cases"]([path], ("fresh",)))
    cpu_inputs = [benchmark["make_fresh_inputs"](torch, case, sample) for sample in range(4)]
    device_inputs = [tuple(value.npu() for value in values) for values in cpu_inputs]
    # Keep all identities/storage alive and do no tensor work between calls.
    # This proves the benchmark's prepared-fresh workload is all cache misses,
    # rather than repeatedly timing one already validated graph.
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        results = [invoke(operators, path, *values, "sum") for values in device_inputs]
    counts = {event.key: event.count for event in profile.key_averages()}
    assert counts.get("aten::isnan", 0) == 4, counts
    assert counts.get("aten::stack", 0) == 4, counts
    assert counts.get("aten::le", 0) == 0, counts
    for result, values in zip(results, cpu_inputs):
        assert_result(result, path, *values, "sum")


@pytest.mark.parametrize("path", PATHS)
def test_out_write_invalidates_other_inputs(operators, path):
    ptr, idx, x = inputs(path)
    writer = torch.full_like(x, float("nan"))
    out_alias = x.data
    for _ in range(3):
        invoke(operators, path, ptr, idx, x, "max")
    # out aliases an input from a PREVIOUS invocation, not the writer's inputs.
    # The public result must reflect the write through the independent alias,
    # without inspecting private version counters or cache implementation.
    invoke(operators, path, ptr, idx, writer, "max", out_alias)
    result = invoke(operators, path, ptr, idx, x, "max")
    assert_result(result, path, ptr, idx, x, "max")


def test_copy_rhs_proof_does_not_bypass_copy_lhs_bounds(operators):
    ptr, idx, x = inputs("fp32")
    idx.fill_(99)
    for _ in range(3):
        invoke(operators, "copy_rhs", ptr, idx, x, "sum")
    with pytest.raises(RuntimeError, match="indices out of range"):
        invoke(operators, "fp32", ptr, idx, x, "sum")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("writer_kind", ["csr_value", "csr_tuple", "coo_value", "coo_tuple"])
def test_other_native_operator_out_alias(operators, path, writer_kind):
    from ops_gnn import _pybind

    ptr, idx, x = inputs(path)
    writer = torch.full_like(x, float("nan"))
    writer_ptr = torch.arange(4, device=x.device, dtype=torch.int64)
    writer_idx = torch.arange(3, device=x.device, dtype=torch.int64)
    out_alias = x.data
    for _ in range(3):
        invoke(operators, path, ptr, idx, x, "max")
    if writer_kind == "csr_value":
        _pybind.segment_csr_value(writer, writer_ptr, out_alias, "sum")
    elif writer_kind == "csr_tuple":
        _pybind.segment_csr(writer, writer_ptr, out_alias, "sum", True)
    elif writer_kind == "coo_value":
        _pybind.segment_coo_value(writer, writer_idx, out_alias, 3,
                                  _pybind.SEGMENT_COO_SUM)
    else:
        _pybind.segment_coo_forward(writer, writer_idx, out_alias, 3,
                                    _pybind.SEGMENT_COO_SUM, True)
    result = invoke(operators, path, ptr, idx, x, "max")
    assert torch.isnan(x.cpu()).all()
    assert_result(result, path, ptr, idx, x, "max")


def proof_alias(value, kind):
    if kind == "view":
        return value.view_as(value)
    # set_ creates an independent Tensor sharing storage, not a shared view
    # version counter. Use only public Tensor/storage APIs in these tests.
    return torch.empty(0, device=value.device, dtype=value.dtype).set_(
        value.untyped_storage(), value.storage_offset(), value.size(), value.stride())


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("alias_kind", ["view", "storage"])
def test_warm_nan_alias_mutation(operators, path, alias_kind):
    ptr, idx, x = inputs(path)
    alias = proof_alias(x, alias_kind)
    for _ in range(3):
        invoke(operators, path, ptr, idx, x, "max")
    for value in (float("nan"), 2):
        alias[0].fill_(value)
        result = invoke(operators, path, ptr, idx, x, "max")
        assert_result(result, path, ptr, idx, x, "max")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("field", ["pointer", "index"])
@pytest.mark.parametrize("alias_kind", ["view", "storage"])
def test_warm_csr_alias_mutation(operators, path, field, alias_kind):
    if path == "copy_rhs" and field == "index":
        pytest.skip("copy_rhs does not dereference source index values")
    ptr, idx, x = inputs(path)
    alias = proof_alias(ptr if field == "pointer" else idx, alias_kind)
    for _ in range(3):
        invoke(operators, path, ptr, idx, x, "sum")
    alias[0] = 1 if field == "pointer" else -1
    message = ("indptr must start at zero and end at nnz" if field == "pointer"
               else "indices out of range")
    for _ in range(2):
        with pytest.raises(RuntimeError, match=message):
            invoke(operators, path, ptr, idx, x, "sum")
    alias[0] = 0
    result = invoke(operators, path, ptr, idx, x, "sum")
    assert_result(result, path, ptr, idx, x, "sum")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("layout", ["data_rebind", "storage_offset", "stride"])
def test_warm_feature_rebind(operators, path, layout):
    ptr, idx, x = inputs(path)
    shape = tuple(x.shape)
    if layout == "storage_offset":
        replacement = torch.full((shape[0] + 1,) + shape[1:], float("nan"),
                                 device=x.device, dtype=x.dtype)[1:]
    elif layout == "stride":
        replacement = torch.full(shape[:-1] + (shape[-1] * 2,), float("nan"),
                                 device=x.device, dtype=x.dtype)[..., ::2]
    else:
        replacement = torch.full_like(x, float("nan"))
    for _ in range(3):
        invoke(operators, path, ptr, idx, x, "max")
    x.data = replacement
    result = invoke(operators, path, ptr, idx, x, "max")
    assert_result(result, path, ptr, idx, x, "max")


@pytest.mark.parametrize("path", PATHS)
def test_inference_proofs_are_recomputed(operators, path):
    with torch.inference_mode():
        ptr, idx, x = inputs(path)
        assert x.is_inference() and ptr.is_inference() and idx.is_inference()
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            for _ in range(2):
                result = invoke(operators, path, ptr, idx, x, "sum")
        counts = {event.key: event.count for event in profile.key_averages()}
        assert counts.get("aten::isnan", 0) == 2, counts
        assert counts.get("aten::stack", 0) == 2, counts
        assert_result(result, path, ptr, idx, x, "sum")


@pytest.mark.parametrize("path", PATHS)
def test_external_operation_invalidates_proofs(operators, path):
    ptr, idx, x = inputs(path)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        for _ in range(2):
            invoke(operators, path, ptr, idx, x, "sum")
        # The observer intentionally invalidates even on unrelated ATen work.
        unrelated = torch.ones(1, device=x.device)
        for _ in range(2):
            result = invoke(operators, path, ptr, idx, x, "sum")
    counts = {event.key: event.count for event in profile.key_averages()}
    assert counts.get("aten::isnan", 0) == 2, counts
    assert counts.get("aten::stack", 0) == 2, counts
    assert unrelated.item() == 1
    assert_result(result, path, ptr, idx, x, "sum")


@pytest.mark.parametrize("path", ["fp16", "fp32", "batched_fp16"])
def test_warm_csr_source_count_change(operators, path):
    ptr, idx, x = inputs(path)
    smaller = x[:1]
    for _ in range(3):
        invoke(operators, path, ptr, idx, x, "sum")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="indices out of range"):
            invoke(operators, path, ptr, idx, smaller, "sum")


@pytest.mark.parametrize("path", PATHS)
def test_csr_error_priority_after_warm_proof(operators, path):
    ptr, idx, x = inputs(path)
    for _ in range(3):
        invoke(operators, path, ptr, idx, x, "sum")
    # Simultaneous endpoint, pointer-range/order and index errors must still
    # report the original first failing criterion on repeated invalid calls.
    ptr.data.copy_(torch.tensor([1, 4, 2, 3], device=ptr.device))
    idx.data.fill_(-1)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="indptr must start at zero and end at nnz"):
            invoke(operators, path, ptr, idx, x, "sum")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("middle", [-1, 4])
def test_csr_range_error_precedes_order_and_index_errors(operators, path, middle):
    ptr, idx, x = inputs(path)
    for _ in range(2):
        invoke(operators, path, ptr, idx, x, "sum")
    # Endpoints are valid, but range and order fail together. On copy_lhs
    # the source indices are also invalid. The cold fallback must report range.
    ptr.data.copy_(torch.tensor([0, middle, 2, 3], device=ptr.device))
    idx.data.fill_(-1)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="indptr values out of range"):
            invoke(operators, path, ptr, idx, x, "sum")


@pytest.mark.parametrize("path", PATHS)
def test_many_tensor_identities(operators, path):
    # Exercise more than the cache's 256 identities through public operators.
    # This checks results after churn, not private cache size or eviction order.
    cases = [inputs(path) for _ in range(260)]
    for ptr, idx, x in cases:
        invoke(operators, path, ptr, idx, x, "max")
    ptr, idx, x = cases[0]
    del cases
    x.data[0].fill_(float("nan"))
    result = invoke(operators, path, ptr, idx, x, "max")
    assert_result(result, path, ptr, idx, x, "max")

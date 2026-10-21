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

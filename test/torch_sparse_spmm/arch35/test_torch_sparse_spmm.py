# Copyright (c) 2026 Eternal_ZCL.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Functional, validation, cache and PyG tests for torch_sparse SpMM."""

import copy
import gc
import importlib
from contextlib import nullcontext

import pytest
import torch

from golden import COMBOS, compare, golden, registered_call, sample

pytest.importorskip("torch_npu")
torch_sparse = pytest.importorskip("torch_sparse")
if not torch.npu.is_available():
    pytest.skip("requires an Ascend NPU", allow_module_level=True)

npu = importlib.import_module("ops_gnn.torch_sparse_spmm")
native_spmm = importlib.import_module("ops_gnn._pybind").torch_sparse_spmm


@pytest.fixture(scope="module")
def pyg_nn():
    pytest.importorskip("torch_geometric")
    return importlib.import_module("torch_geometric.nn")


@pytest.mark.parametrize("dtype,outdtype", COMBOS)
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("batch,feature_values", [(False, False), (True, True)])
def test_forward(dtype, outdtype, reduce, batch, feature_values):
    ptr, col, v, mat = sample(dtype, batch, feature_values)
    expected, arg = golden(ptr, col, v, mat, reduce, outdtype=outdtype)
    result = npu.csr_spmm(ptr.npu(), col.npu(), v.npu(), mat.npu(), reduce, output_dtype=outdtype)
    actual, actual_arg = result if reduce in ("min", "max") else (result, None)
    assert actual.device.type == "npu" and actual.dtype == outdtype
    tol = 0 if outdtype == torch.int32 else 0.015 if outdtype == torch.bfloat16 else 0.002
    torch.testing.assert_close(actual.cpu(), expected, rtol=tol, atol=tol)
    if actual_arg is not None:
        torch.testing.assert_close(actual_arg.cpu(), arg)


@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_unweighted_empty_and_noncontiguous(reduce):
    ptr, col, v, mat = sample(torch.float32)
    mat = mat.t().contiguous().t()
    expected, arg = golden(ptr, col, None, mat, reduce, outdtype=torch.float32)
    result = npu.csr_spmm(ptr.npu(), col.npu(), None, mat.npu(), reduce)
    actual, aa = result if reduce in ("min", "max") else (result, None)
    torch.testing.assert_close(actual.cpu(), expected)
    if aa is not None:
        torch.testing.assert_close(aa.cpu(), arg)
    for m, k in [(0, 17), (4, 0), (4, 17)]:
        p = torch.zeros(m + 1, dtype=torch.long, device="npu")
        c = torch.empty(0, dtype=torch.long, device="npu")
        b = torch.empty(2, 4, k, device="npu")
        result = npu.csr_spmm(p, c, None, b, reduce)
        out = result[0] if isinstance(result, tuple) else result
        assert out.shape == (2, m, k) and int(torch.count_nonzero(out).item()) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("weighted,feature_values", [(False, False), (True, False), (True, True)])
def test_backward(dtype, reduce, weighted, feature_values):
    ptr, col, v, mat = sample(dtype, True, feature_values)
    vm = v.npu().requires_grad_() if weighted else None
    mm = mat.npu().requires_grad_()
    result = npu.csr_spmm(ptr.npu(), col.npu(), vm, mm, reduce, output_dtype=torch.float32)
    weight = torch.randn(result.shape)
    (result * weight.npu()).sum().backward()
    rv = v.double().requires_grad_() if weighted else None
    rm = mat.double().requires_grad_()
    reference = golden(ptr, col, rv, rm, reduce, outdtype=torch.float64)[0]
    (reference * weight.double()).sum().backward()
    tol = 0.03 if dtype == torch.bfloat16 else 0.008 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(mm.grad.cpu().float(), rm.grad.to(dtype).float(), rtol=tol, atol=tol)
    if weighted:
        torch.testing.assert_close(vm.grad.cpu().float(), rv.grad.to(dtype).float(), rtol=tol, atol=tol)


def test_coo_and_sparse_tensor():
    ptr, col, v, mat = sample(torch.float32)
    row = torch.repeat_interleave(torch.arange(4), ptr[1:] - ptr[:-1])
    order = torch.tensor([6, 2, 4, 0, 5, 1, 3])
    index = torch.stack((row[order], col[order])).npu()
    vn = v[order].npu().requires_grad_()
    bn = mat.npu().requires_grad_()
    result = torch_sparse.spmm(index, vn, 4, 4, bn)
    expected = golden(ptr, col, v, mat, "sum", outdtype=torch.float32)[0]
    torch.testing.assert_close(result.cpu(), expected)
    result.sum().backward()
    assert vn.grad is not None and bn.grad is not None
    sparse = torch_sparse.SparseTensor(
        rowptr=ptr.npu(), col=col.npu(), value=v.npu(), sparse_sizes=(4, 4), is_sorted=True, trust_data=True
    )
    for reduce in ("sum", "add", "mean", "min", "max"):
        expected = golden(ptr, col, v, mat, reduce, outdtype=torch.float32)[0]
        torch.testing.assert_close(sparse.matmul(mat.npu(), reduce).cpu(), expected)
        torch.testing.assert_close(sparse.spmm(mat.npu(), reduce).cpu(), expected)


def test_registered_ops_and_metadata_errors():
    ptr, col, v, mat = sample(torch.float32)
    row = torch.repeat_interleave(torch.arange(4), ptr[1:] - ptr[:-1])
    perm = col.argsort(stable=True)
    counts = ptr[1:] - ptr[:-1]
    cp = torch.cat((torch.zeros(1, dtype=torch.long), torch.bincount(col, minlength=4).cumsum(0)))
    row, ptr, col, v, cp, perm, mat, counts = [x.npu() for x in (row, ptr, col, v, cp, perm, mat, counts)]
    mat.requires_grad_()
    v.requires_grad_()
    out = torch.ops.torch_sparse.spmm_sum(row, ptr, col, v, cp, perm, mat)
    out.sum().backward()
    assert v.grad is not None and mat.grad is not None
    with pytest.raises(ValueError, match="csr2csc"):
        torch.ops.torch_sparse.spmm_sum(row, ptr, col, v, cp, torch.arange(7, device="npu"), mat)
    with pytest.raises(ValueError, match="row"):
        torch.ops.torch_sparse.spmm_sum(None, ptr, col, v, cp, perm, mat)
    out = torch.ops.torch_sparse.spmm_mean(row, ptr, col, v, counts, cp, perm, mat)
    assert out.shape == (4, 17)


def test_validation_and_cache_invalidation():
    ptr, col, v, mat = [x.npu() for x in sample(torch.float32)]
    npu.csr_spmm(ptr, col, v, mat)
    col[0] = 99
    with pytest.raises(ValueError, match="invalid CSR"):
        npu.csr_spmm(ptr, col, v, mat)
    col[0] = 1
    ptr[1] = -1
    with pytest.raises(ValueError, match="invalid CSR"):
        npu.csr_spmm(ptr, col, v, mat)
    with pytest.raises((ValueError, RuntimeError)):
        npu.csr_spmm(ptr.int(), col, v, mat)


def test_current_stream_and_cpu_preserved():
    ptr, col, v, mat = sample(torch.float32)
    expected = golden(ptr, col, v, mat, "sum", outdtype=torch.float32)[0]
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        out = npu.csr_spmm(ptr.npu(), col.npu(), v.npu(), mat.npu())
        out = out + 2
    stream.synchronize()
    torch.testing.assert_close(out.cpu(), expected + 2)
    row = torch.repeat_interleave(torch.arange(4), ptr[1:] - ptr[:-1])
    torch.testing.assert_close(torch_sparse.spmm(torch.stack((row, col)), v, 4, 4, mat), expected)


def test_sparse_row_validation():
    mat = torch.ones(4, 3, device="npu")
    for row, message in [([-1, 2], "out of range"), ([2, 1], "sorted")]:
        sparse = torch_sparse.SparseTensor(
            row=torch.tensor(row, device="npu"),
            col=torch.tensor([0, 1], device="npu"),
            sparse_sizes=(4, 4),
            is_sorted=True,
            trust_data=True,
        )
        with pytest.raises(ValueError, match=message):
            sparse.matmul(mat)


def checked(actual, expected, dtype, name):
    assert actual.device.type == "npu"
    result = compare(actual.cpu(), expected, dtype)
    assert result["passed"], (name, result)


@pytest.mark.parametrize("edges,length", [(1, 2), (1, 0), (0, 1)])
def test_coo_rejects_original_weight_length(edges, length):
    index = torch.zeros(2, edges, dtype=torch.long, device="npu")
    value = torch.ones(length, device="npu")
    with pytest.raises(ValueError, match="length E"):
        torch_sparse.spmm(index, value, 1, 1, torch.ones(1, 1, device="npu"))


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.float64, torch.float16])
@pytest.mark.parametrize("frontend", ["helper", "coo", "sparse"])
def test_invalid_weight_dtype(dtype, frontend):
    ptr = torch.tensor([0, 1], device="npu")
    col = torch.tensor([0], device="npu")
    v = torch.ones(1, dtype=dtype, device="npu")
    b = torch.ones(1, 1, device="npu")
    with pytest.raises(ValueError, match="dtype"):
        if frontend == "helper":
            npu.csr_spmm(ptr, col, v, b)
        elif frontend == "coo":
            torch_sparse.spmm(torch.stack((col, col)), v, 1, 1, b)
        else:
            s = torch_sparse.SparseTensor(
                rowptr=ptr, col=col, value=v, sparse_sizes=(1, 1), is_sorted=True, trust_data=True
            )
            s.matmul(b)


def test_value_device_and_complex_rejected_before_indexing():
    index = torch.zeros(2, 1, dtype=torch.long, device="npu")
    b = torch.ones(1, 1, device="npu")
    with pytest.raises(ValueError, match="same NPU"):
        torch_sparse.spmm(index, torch.ones(1), 1, 1, b)
    with pytest.raises(ValueError, match="dtype"):
        torch_sparse.spmm(index, torch.ones(1, dtype=torch.complex64), 1, 1, b)


def test_native_value_backward_rejects_invalid_grad_before_launch():
    ptr = torch.tensor([0, 1, 1], device="npu", dtype=torch.int64)
    col = torch.tensor([0], device="npu", dtype=torch.int64)
    mat = torch.ones(1, 4, device="npu")
    invalid = (
        torch.ones(2, 4),
        torch.ones(8, device="npu"),
        torch.ones(3, 4, device="npu"),
        torch.ones(2, 3, device="npu"),
        torch.ones(2, 4, device="npu", dtype=torch.int32),
    )
    for grad in invalid:
        with pytest.raises(RuntimeError, match="grad must be"):
            native_spmm.value_backward(ptr, col, mat, grad, 0, 1)


@pytest.mark.parametrize("width", [0, 2, 5])
def test_native_value_backward_rejects_invalid_width(width):
    ptr = torch.tensor([0, 1], device="npu", dtype=torch.int64)
    col = torch.tensor([0], device="npu", dtype=torch.int64)
    mat = torch.ones(1, 4, device="npu")
    grad = torch.ones(1, 4, device="npu")
    with pytest.raises(RuntimeError, match="width"):
        native_spmm.value_backward(ptr, col, mat, grad, 0, width)


@pytest.mark.parametrize("m,n", [(True, 1), (1.5, 1), (-1, 1), (1, False), (1, 1.2)])
def test_coo_dimensions_are_integers(m, n):
    with pytest.raises(ValueError, match="nonnegative integers"):
        torch_sparse.spmm(
            torch.zeros(2, 1, dtype=torch.long, device="npu"), None, m, n, torch.ones(1, 1, device="npu")
        )


def metadata(mode):
    ptr = torch.tensor([0, 2, 2, 3], device="npu")
    col = torch.tensor([1, 0, 1], device="npu")
    row = torch.tensor([0, 0, 2], device="npu")
    counts = torch.tensor([2, 0, 1], device="npu")
    cp = torch.tensor([0, 1, 3], device="npu")
    perm = torch.tensor([1, 0, 2], device="npu")
    v = torch.ones(3, device="npu", requires_grad=mode in ("value", "both"))
    b = torch.ones(2, 4, device="npu", requires_grad=mode in ("mat", "both"))
    return [row, ptr, col, v, counts, cp, perm, b]


@pytest.mark.parametrize("mode", ["none", "value", "mat", "both", "no_grad"])
@pytest.mark.parametrize("field,message", [(0, "row"), (4, "rowcount"), (5, "colptr"), (6, "csr2csc")])
def test_bad_supplied_metadata_in_every_grad_mode(mode, field, message):
    args = metadata("both" if mode == "no_grad" else mode)
    args[field] = args[field].clone()
    args[field][0] = 99
    with torch.no_grad() if mode == "no_grad" else nullcontext():
        with pytest.raises(ValueError, match=message):
            torch.ops.torch_sparse.spmm_mean(*args)


@pytest.mark.parametrize("field,message", [(0, "row"), (5, "colptr"), (6, "csr2csc"), (4, "rowcount")])
def test_required_metadata_missing(field, message):
    args = metadata("both")
    args[field] = None
    with pytest.raises(ValueError, match=message):
        torch.ops.torch_sparse.spmm_mean(*args)


def test_unused_missing_metadata_allowed_and_cache_mutation_detected():
    args = metadata("none")
    expected = torch.tensor([[2.0] * 4, [0.0] * 4, [1.0] * 4])
    empty = list(args)
    for i in (0, 4, 5, 6):
        empty[i] = None
    checked(
        torch.ops.torch_sparse.spmm_mean(*empty),
        expected / torch.tensor([[2.0], [1.0], [1.0]]),
        torch.float32,
        "optional_missing",
    )
    torch.ops.torch_sparse.spmm_mean(*args)
    args[4][0] = 3
    with pytest.raises(ValueError, match="rowcount"):
        torch.ops.torch_sparse.spmm_mean(*args)


def test_metadata_cache_preserves_parameter_roles():
    p = torch.tensor([0, 1], device="npu")
    c = torch.tensor([0], device="npu")
    t = c.clone()
    b = torch.ones(1, 1, device="npu")
    torch.ops.torch_sparse.spmm_mean(t, p, c, None, None, None, None, b)
    with pytest.raises(ValueError, match="rowcount"):
        torch.ops.torch_sparse.spmm_mean(None, p, c, None, t, None, None, b)


@pytest.mark.parametrize(
    "field,message", [("_row", "row"), ("_rowcount", "rowcount"), ("_colptr", "colptr"), ("_csr2csc", "csr2csc")]
)
def test_sparse_tensor_bad_cached_metadata(field, message):
    row, p, c, v, counts, cp, perm, b = metadata("none")
    s = torch_sparse.SparseTensor(rowptr=p, col=c, value=v, sparse_sizes=(3, 2), is_sorted=True, trust_data=True)
    original = {"_row": row, "_rowcount": counts, "_colptr": cp, "_csr2csc": perm}[field].clone()
    original[0] = 99
    setattr(s.storage, field, original)
    with pytest.raises(ValueError, match=message):
        s.matmul(b)


@pytest.mark.parametrize("dtype,outdtype", COMBOS)
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("pattern", ["zeros", "small_signed", "large_signed", "cancellation"])
def test_controlled_numeric_values(dtype, outdtype, reduce, pattern):
    p = torch.tensor([0, 4, 4, 8])
    c = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
    scale = (
        1
        if dtype == torch.int8
        else {"zeros": 0.0, "small_signed": 2**-8, "large_signed": 16.0, "cancellation": 1.0}[pattern]
    )
    v = torch.tensor([1, -1, 1, -1, -1, 1, -1, 1], dtype=dtype) * scale
    b = torch.arange(32).reshape(2, 4, 4).remainder(5).sub(2).to(dtype) * scale
    if pattern == "zeros":
        v.zero_()
        b.zero_()
    v = v.to(dtype)
    b = b.to(dtype)
    expected, arg = golden(p, c, v, b, reduce, outdtype=torch.float64)
    out = npu.csr_spmm(p.npu(), c.npu(), v.npu(), b.npu(), reduce, output_dtype=outdtype)
    actual, aa = out if isinstance(out, tuple) else (out, None)
    checked(actual, expected, outdtype, f"numeric/{dtype}/{outdtype}/{reduce}/{pattern}")
    if aa is not None:
        assert aa.dtype == torch.int64
        torch.testing.assert_close(aa.cpu(), arg)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int8])
@pytest.mark.parametrize("frontend", ["coo", "sparse"])
@pytest.mark.parametrize(
    "empty,weighted,batch", [(False, True, 1), (False, False, 2), (True, True, 2), (True, False, 0)]
)
def test_public_frontend_dtype_batch_empty(dtype, frontend, empty, weighted, batch):
    p = torch.tensor([0, 0, 0, 0]) if empty else torch.tensor([0, 2, 2, 3])
    c = torch.empty(0, dtype=torch.long) if empty else torch.tensor([1, 0, 1])
    row = torch.repeat_interleave(torch.arange(3), p[1:] - p[:-1])
    b = torch.arange(batch * 2 * 5).reshape(batch, 2, 5).remainder(5).sub(2).to(dtype)
    v = torch.ones(c.numel(), dtype=dtype) if weighted else None
    pn, cn, rn, bn = p.npu(), c.npu(), row.npu(), b.npu()
    vn = None if v is None else v.npu()
    outdtype = torch.int32 if dtype == torch.int8 else dtype
    for reduce in ["sum"] if frontend == "coo" else ["sum", "add", "mean", "min", "max"]:
        if frontend == "coo":
            actual = torch_sparse.spmm(torch.stack((rn, cn)), vn, 3, 2, bn)
        else:
            s = torch_sparse.SparseTensor(
                rowptr=pn, col=cn, value=vn, sparse_sizes=(3, 2), is_sorted=True, trust_data=True
            )
            actual = s.matmul(bn, reduce)
        expected, _ = golden(p, c, v, b, reduce, outdtype=torch.float64)
        checked(actual, expected, outdtype, f"frontend/{frontend}/{dtype}/{reduce}/{empty}/{weighted}/{batch}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "frontend,reduce",
    [
        ("coo", "sum"),
        ("registered", "sum"),
        ("registered", "mean"),
        ("sparse", "sum"),
        ("sparse", "mean"),
        ("helper", "sum"),
        ("helper", "mean"),
    ],
)
def test_gradient_task_tolerance(dtype, frontend, reduce):
    p = torch.tensor([0, 3, 3, 5])
    c = torch.tensor([1, 0, 1, 0, 1])
    r = torch.tensor([0, 0, 0, 2, 2])
    torch.manual_seed(617)
    v = (torch.rand(5) * 2 - 1).to(dtype)
    b = (torch.rand(2, 2, 7) * 2 - 1).to(dtype)
    vn = v.npu().requires_grad_()
    bn = b.npu().requires_grad_()
    pn = p.npu()
    cn = c.npu()
    rn = r.npu()
    perm = cn.argsort(stable=True)
    cp = torch.tensor([0, 2, 5], device="npu")
    counts = (p[1:] - p[:-1]).npu()
    if frontend == "coo":
        out = torch_sparse.spmm(torch.stack((rn, cn)), vn, 3, 2, bn)
    elif frontend == "helper":
        out = npu.csr_spmm(pn, cn, vn, bn, reduce)
    elif frontend == "sparse":
        s = torch_sparse.SparseTensor(
            rowptr=pn, col=cn, value=vn, sparse_sizes=(3, 2), is_sorted=True, trust_data=True
        )
        out = s.matmul(bn, reduce)
    elif reduce == "sum":
        out = torch.ops.torch_sparse.spmm_sum(rn, pn, cn, vn, cp, perm, bn)
    else:
        out = torch.ops.torch_sparse.spmm_mean(rn, pn, cn, vn, counts, cp, perm, bn)
    loss_weight = (torch.rand(out.shape) * 2 - 1).to(dtype)
    out.backward(loss_weight.npu())
    rv = v.double().requires_grad_()
    rb = b.double().requires_grad_()
    ref = golden(p, c, rv, rb, reduce, outdtype=torch.float64)[0]
    ref.backward(loss_weight.double())
    checked(vn.grad, rv.grad, dtype, f"grad/value/{frontend}/{reduce}/{dtype}")
    checked(bn.grad, rb.grad, dtype, f"grad/mat/{frontend}/{reduce}/{dtype}")


@pytest.mark.parametrize("reduce", ["min", "max"])
def test_extrema_documented_special_policy(reduce):
    p = torch.tensor([0, 2, 2, 3], device="npu")
    c = torch.tensor([0, 1, 2], device="npu")
    b = torch.tensor(
        [[2.0, float("nan"), float("inf")], [2.0, 3.0, -float("inf")], [float("nan"), float("nan"), float("nan")]],
        device="npu",
    )
    out, arg = npu.csr_spmm(p, c, None, b, reduce)
    expected = torch.tensor(
        [[2.0, 3.0, -float("inf") if reduce == "min" else float("inf")], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    checked(out, expected.double(), torch.float32, f"special/{reduce}/documented_policy")
    torch.testing.assert_close(arg.cpu(), torch.tensor([[0, 1, 1 if reduce == "min" else 0], [3, 3, 3], [3, 3, 3]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "fmt,reduce", [("coo", "sum"), ("csr", "sum"), ("csr", "mean"), ("csr", "min"), ("csr", "max")]
)
@pytest.mark.parametrize("weighted", [False, True])
def test_upstream_cpu_same_public_api(dtype, fmt, reduce, weighted):
    # Independent upstream CPU comparison in addition to our FP64 golden.
    row = torch.tensor([0, 0, 2, 2])
    ptr = torch.tensor([0, 2, 2, 4])
    col = torch.tensor([1, 0, 0, 1])
    value = torch.tensor([0.5, -0.5, 1.0, -1.0], dtype=dtype) if weighted else None
    mat = torch.arange(24).reshape(2, 2, 6).remainder(9).sub(4).to(dtype) / 4
    case = {"format": fmt, "reduce": reduce, "m": 3, "n": 2}
    data = (row, ptr, col, value, mat)
    if fmt == "coo" and value is None:
        # Upstream COO requires value; the task's no-value path is exercised by CSR.
        value = torch.ones(4, dtype=dtype)
        data = (row, ptr, col, value, mat)
    expected = registered_call(case, data)()
    device_data = tuple(t.npu() if t is not None else None for t in data)
    actual = registered_call(case, device_data)()
    actual, arg = actual if isinstance(actual, tuple) else (actual, None)
    expected, expected_arg = expected if isinstance(expected, tuple) else (expected, None)
    checked(actual, expected.double(), dtype, f"upstream_cpu/{fmt}/{reduce}/{dtype}/{weighted}")
    if arg is not None:
        torch.testing.assert_close(arg.cpu(), expected_arg)


@pytest.mark.parametrize("dtype,outdtype", COMBOS)
@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("k", [4, 16, 17])
def test_grouped_features_and_scalar_tail(dtype, outdtype, reduce, k):
    p = torch.tensor([0, 2, 2, 5])
    c = torch.tensor([1, 0, 2, 1, 0])
    torch.manual_seed(872)
    if dtype == torch.int8:
        v = torch.randint(-4, 5, (5, k), dtype=dtype)
        b = torch.randint(-4, 5, (2, 3, k), dtype=dtype)
    else:
        v = (torch.rand(5, k) * 2 - 1).to(dtype)
        b = (torch.rand(2, 3, k) * 2 - 1).to(dtype)
    expected = golden(p, c, v, b, reduce, outdtype=torch.float64)[0]
    # Preserve a non-contiguous dense layout while moving to the NPU.
    bn = b.transpose(-2, -1).contiguous().npu().transpose(-2, -1)
    out = npu.csr_spmm(p.npu(), c.npu(), v.npu(), bn, reduce, output_dtype=outdtype)
    checked(out, expected, outdtype, f"grouped/{dtype}/{outdtype}/{reduce}/K{k}")


@pytest.mark.parametrize("dtype,outdtype", COMBOS)
@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("weighted", [False, True])
def test_large_grouped_feature_path(dtype, outdtype, reduce, weighted):
    m, k = 4096, 128
    p = torch.arange(m + 1, dtype=torch.long) * 2
    c = torch.arange(m * 2, dtype=torch.long) % 2
    torch.manual_seed(127)
    if dtype == torch.int8:
        v = torch.randint(-4, 5, (m * 2, k), dtype=dtype)
        b = torch.randint(-4, 5, (2, 2, k), dtype=dtype)
    else:
        v = (torch.rand(m * 2, k) * 2 - 1).to(dtype)
        b = (torch.rand(2, 2, k) * 2 - 1).to(dtype)
    if weighted:
        expected = (v.double().reshape(1, m, 2, k) * b.double().reshape(2, 1, 2, k)).sum(2)
    else:
        expected = b.double().sum(1, keepdim=True).expand(2, m, k).contiguous()
    if reduce == "mean":
        expected /= 2
    out = npu.csr_spmm(p.npu(), c.npu(), v.npu() if weighted else None, b.npu(), reduce, output_dtype=outdtype)
    checked(out, expected, outdtype, f"large_grouped/{dtype}/{outdtype}/{reduce}/{weighted}")


def coo_golden(index, value, mat, m=3):
    result = torch.zeros((m, mat.shape[1]), dtype=mat.dtype)
    return result.index_add(0, index[0], value[:, None] * mat[index[1]])


def coo_inputs():
    index = torch.tensor([[2, 0, 0, 1], [1, 2, 0, 1]], dtype=torch.int64).npu()
    value = torch.tensor([2.0, -1.0, 3.0, 4.0]).npu()
    mat = torch.arange(12, dtype=torch.float32).reshape(3, 4).npu()
    return index, value, mat


def check_coo(index, value, mat):
    actual = npu.coo_spmm(index, value, 3, 3, mat)
    torch.testing.assert_close(actual.cpu(), coo_golden(index.cpu(), value.cpu(), mat.cpu()))
    return actual


def test_cached_layout_observes_each_input_mutation():
    npu.clear_validation_cache()
    index, value, mat = coo_inputs()
    first = check_coo(index, value, mat)
    saved = first.clone()
    check_coo(index, value, mat)
    assert npu.validation_cache_info().coo_entries == 1
    value.add_(2)
    check_coo(index, value, mat)
    mat.mul_(2)
    check_coo(index, value, mat)
    index[1, 0] = 2
    check_coo(index, value, mat)
    index[0, 0] = 0
    check_coo(index, value, mat)
    torch.testing.assert_close(first, saved)


def test_cached_layout_still_rejects_bad_inputs():
    npu.clear_validation_cache()
    index, value, mat = coo_inputs()
    check_coo(index, value, mat)
    for bad in (torch.cat((value, value[:1])), value.to(torch.int16)):
        try:
            npu.coo_spmm(index, bad, 3, 3, mat)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid edge values accepted")
    index[0, 0] = 3
    try:
        npu.coo_spmm(index, value, 3, 3, mat)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("invalid row accepted after index mutation")


def test_layout_cache_releases_dead_indices():
    npu.clear_validation_cache()
    index, value, mat = coo_inputs()
    check_coo(index, value, mat)
    assert npu.validation_cache_info().coo_entries == 1
    del index
    gc.collect()
    assert npu.validation_cache_info().coo_entries == 0


def test_inference_indices_are_not_cached():
    npu.clear_validation_cache()
    with torch.inference_mode():
        index, value, mat = coo_inputs()
        check_coo(index, value, mat)
        index[0, 0] = 0
        check_coo(index, value, mat)
        assert npu.validation_cache_info().coo_entries == 0


def test_coo_gradients_survive_layout_reuse():
    npu.clear_validation_cache()
    index, value, mat = coo_inputs()
    check_coo(index, value, mat)
    value.requires_grad_()
    mat.requires_grad_()
    result = npu.coo_spmm(index, value, 3, 3, mat)
    result.square().sum().backward()
    cpu_value = value.detach().cpu().requires_grad_()
    cpu_mat = mat.detach().cpu().requires_grad_()
    reference = coo_golden(index.cpu(), cpu_value, cpu_mat)
    reference.square().sum().backward()
    torch.testing.assert_close(value.grad.cpu(), cpu_value.grad)
    torch.testing.assert_close(mat.grad.cpu(), cpu_mat.grad)


@pytest.mark.parametrize("dtype,outdtype", COMBOS)
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("features", [64, 128, 256, 512])
def test_prefetch_widths_and_eight_edge_boundary(dtype, outdtype, reduce, features):
    # Empty row, 7/8/9-edge rows, and a longer row exercise each loop boundary.
    ptr = torch.tensor([0, 0, 7, 15, 24, 41], dtype=torch.int64)
    col = torch.arange(41, dtype=torch.int64).remainder(5)
    values = torch.arange(41).remainder(7).sub(3).to(dtype)
    mat = torch.arange(5 * features).reshape(5, features).remainder(11).sub(5).to(dtype)
    expected, expected_arg = golden(ptr, col, values, mat, reduce, outdtype=torch.float64)
    if outdtype == torch.int32:
        expected = expected.trunc()
    result = npu.csr_spmm(ptr.npu(), col.npu(), values.npu(), mat.npu(), reduce, output_dtype=outdtype)
    actual, arg = result if isinstance(result, tuple) else (result, None)
    assert compare(actual.cpu(), expected, outdtype)["passed"]
    if arg is not None:
        torch.testing.assert_close(arg.cpu(), expected_arg)


@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("features", [64, 128, 256, 512])
def test_prefetch_extrema_special_values_and_first_tie(reduce, features):
    ptr = torch.tensor([0, 0, 9, 10], dtype=torch.int64)
    col = torch.tensor([0] * 8 + [1, 2], dtype=torch.int64)
    values = torch.ones(10)
    mat = torch.full((3, features), 2.0)
    mat[0, 1] = float("nan")
    mat[1, 1] = 3.0
    mat[0, 2] = float("inf")
    mat[1, 2] = -float("inf")
    mat[2] = float("nan")
    actual, arg = npu.csr_spmm(ptr.npu(), col.npu(), values.npu(), mat.npu(), reduce)
    expected = torch.zeros((3, features), dtype=torch.float64)
    expected[1] = 2.0
    expected[1, 1] = 3.0
    expected[1, 2] = -float("inf") if reduce == "min" else float("inf")
    expected_arg = torch.full((3, features), 10, dtype=torch.int64)
    expected_arg[1] = 0
    expected_arg[1, 1] = 8
    expected_arg[1, 2] = 8 if reduce == "min" else 0
    assert compare(actual.cpu(), expected, torch.float32)["passed"]
    torch.testing.assert_close(arg.cpu(), expected_arg)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int8])
@pytest.mark.parametrize("feature_weights", [False, True])
def test_native_coo_gather_reads_noncontiguous_values(dtype, feature_weights):
    index = torch.tensor([[2, 0, 0, 1], [1, 2, 0, 1]], dtype=torch.int64)
    base = torch.arange(32).remainder(5).sub(2).reshape(4, 8).to(dtype)
    values = base[:, ::2] if feature_weights else base[:, 0]
    mat = torch.arange(12).reshape(3, 4).remainder(5).sub(2).to(dtype)
    expected = torch.zeros((3, 4), dtype=torch.float64)
    product = mat.double()[index[1]] * (values.double() if feature_weights else values.double()[:, None])
    expected.index_add_(0, index[0], product)
    device_base = base.npu()
    device_values = device_base[:, ::2] if feature_weights else device_base[:, 0]
    assert not device_values.is_contiguous()
    actual = npu.coo_spmm(index.npu(), device_values, 3, 3, mat.npu())
    outdtype = torch.int32 if dtype == torch.int8 else dtype
    assert compare(actual.cpu(), expected, outdtype)["passed"]


@pytest.mark.parametrize("kind", ["gcn", "gin"])
def test_pyg_layer(kind, pyg_nn):
    gcn_conv = pyg_nn.GCNConv
    gin_conv = pyg_nn.GINConv
    sparse_tensor = torch_sparse.SparseTensor
    torch.manual_seed(123)
    rowptr = torch.tensor([0, 2, 4, 4, 6], dtype=torch.long)
    col = torch.tensor([0, 1, 0, 2, 1, 3], dtype=torch.long)
    value = torch.tensor([0.5, 0.5, 0.4, 0.6, 0.3, 0.7])
    # GCN graph normalization/self-loops are preprocessing, outside SpMM.
    cpu = (
        gcn_conv(8, 5, normalize=False, add_self_loops=False)
        if kind == "gcn"
        else gin_conv(
            torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.ReLU(), torch.nn.Linear(12, 5)), train_eps=True
        )
    )
    dut = copy.deepcopy(cpu).npu()
    x = torch.randn(4, 8, requires_grad=True)
    xn = x.detach().npu().requires_grad_()
    adjacency = sparse_tensor(rowptr=rowptr, col=col, value=value, sparse_sizes=(4, 4), is_sorted=True, trust_data=True)
    adjacency_npu = sparse_tensor(
        rowptr=rowptr.npu(), col=col.npu(), value=value.npu(), sparse_sizes=(4, 4), is_sorted=True, trust_data=True
    )
    expected, actual = cpu(x, adjacency), dut(xn, adjacency_npu)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(xn.grad.cpu(), x.grad, rtol=1e-4, atol=1e-5)
    for a, b in zip(dut.parameters(), cpu.parameters()):
        torch.testing.assert_close(a.grad.cpu(), b.grad, rtol=1e-4, atol=1e-5)

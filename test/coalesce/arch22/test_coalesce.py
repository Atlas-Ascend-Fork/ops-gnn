# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2/A3 acceptance for coalesce (PyG ``torch_geometric.utils.coalesce`` parity).

Requires an Ascend A2/A3 NPU. Every expected value comes from the independent
CPU reference in ``golden.py``; no device result is used to build expectations.
"""

import os

import pytest
import torch


import torch

INT64_MAX = 2**63 - 1


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return int(num_nodes)
    if edge_index.numel() == 0:
        return 0
    return int(edge_index.max()) + 1


def _validate_coalesce_reference(edge_index, reduce):
    if edge_index.device.type != "cpu":
        raise ValueError("reference requires a CPU edge_index")
    if edge_index.dtype != torch.int64 or edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("reference requires int64 [2, E] edge_index")
    if reduce == "add":
        reduce = "sum"
    if reduce not in ("sum", "mean", "min", "max"):
        raise ValueError("unsupported reduction: " + str(reduce))
    return reduce


def _reduce_groups(group, flat, unique, reduce):
    channels = flat.size(1)
    if reduce in ("sum", "mean"):
        out = torch.zeros((unique, channels), dtype=torch.float64)
        out.index_add_(0, group, flat)
        if reduce == "mean":
            counts = torch.zeros(unique, dtype=torch.float64)
            counts.index_add_(0, group, torch.ones(flat.size(0), dtype=torch.float64))
            out = out / counts.unsqueeze(1)
        return out
    limit = float("inf") if reduce == "min" else float("-inf")
    out = torch.full((unique, channels), limit, dtype=torch.float64)
    index = group.unsqueeze(1).expand(flat.size(0), channels)
    out.scatter_reduce_(0, index, flat, reduce="amin" if reduce == "min" else "amax")
    return out


def cpu_coalesce(edge_index, edge_attr=None, num_nodes=None, reduce="sum",
                 is_sorted=False, sort_by_row=True):
    """Reference coalesce; inputs are CPU tensors, output mirrors the operator."""
    reduce = _validate_coalesce_reference(edge_index, reduce)
    edges = edge_index.size(1)
    nodes = maybe_num_nodes(edge_index, num_nodes)
    if nodes * nodes > INT64_MAX:
        raise ValueError("num_nodes too large")

    row, col = (edge_index[0], edge_index[1]) if sort_by_row else (edge_index[1], edge_index[0])
    key = row * nodes + col

    if not is_sorted:
        perm = torch.argsort(key, stable=True)
        edge_index = edge_index[:, perm]
        key = key[perm]
        if edge_attr is not None:
            edge_attr = edge_attr[perm]

    mask = torch.ones(edges, dtype=torch.bool)
    if edges > 1:
        mask[1:] = key[1:] > key[:-1]
    unique_index = edge_index[:, mask]

    if edge_attr is None:
        return unique_index, None
    if bool(mask.all()):
        return unique_index, edge_attr.clone()

    group = torch.cumsum(mask.to(torch.int64), 0) - 1
    unique = int(group[-1]) + 1
    tail = edge_attr.shape[1:]
    flat = edge_attr.to(torch.float64).reshape(edges, -1)
    out = _reduce_groups(group, flat, unique, reduce)
    out = out.reshape((unique,) + tail).to(edge_attr.dtype)
    return unique_index, out


def _ulp_at_output_precision(expected, dtype):
    info = torch.finfo(dtype)
    magnitude = expected.to(dtype).double().abs()
    _, exponent = torch.frexp(magnitude)
    minimum = info.tiny * info.eps
    spacing = torch.ldexp(torch.full_like(magnitude, info.eps), exponent - 1)
    spacing = torch.where(magnitude == 0, minimum, spacing).clamp_min(minimum)
    return spacing


def precision_metrics(actual, expected):
    """Official mixed tolerance for floats; exact integer values."""
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    if actual.shape != expected.shape:
        raise AssertionError("actual and expected tensors must have matching shapes")
    if not actual.is_floating_point():
        if actual.dtype != expected.dtype:
            raise AssertionError("integer actual and expected dtypes must match")
        return {"exact": torch.equal(actual, expected)}, {"allowed_mismatches": 0}
    if not expected.is_floating_point():
        raise AssertionError("floating actual requires a floating reference")
    tolerance, floor = {
        torch.float16: (2**-10, 0.1),
        torch.bfloat16: (2**-7, 0.1),
        torch.float32: (2**-13, 0.01),
    }[actual.dtype]
    a, b = actual.double(), expected.double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    narrowed = expected.to(actual.dtype).double()
    special = (torch.isnan(a) & torch.isnan(b)) | (torch.isinf(a) & (a == narrowed))
    error = torch.where(finite, (a - b).abs(), 0.0)
    matches = (finite & (error <= tolerance * (1 + b.abs()))) | special
    ulp = _ulp_at_output_precision(torch.where(finite, b, 0.0), actual.dtype)
    flat = int(error.flatten().argmax()) if error.numel() else 0
    limit = max(floor, 32 * float(ulp.flatten()[flat])) if error.numel() else floor
    maximum = float(error.max()) if error.numel() else 0.0
    return {
        "matched_ratio": float(matches.double().mean()) if matches.numel() else 1.0,
        "max_abs_error": maximum,
        "hard_limit": limit,
        "hard_limit_pass": maximum <= limit and bool((finite | special).all()),
    }, {"rtol": tolerance, "atol": tolerance, "required_matched_ratio": 0.99}


def assert_close(actual, expected):
    metrics, limits = precision_metrics(actual, expected)
    passed = metrics.get(
        "exact", metrics.get("matched_ratio", 0) >= 0.99 and metrics.get("hard_limit_pass", False)
    )
    if not passed:
        raise AssertionError((metrics, limits))


DEVICE = os.environ.get("NPU_DEVICE_ID", "0")
DTYPES = (torch.float16, torch.float32)
REDUCES = ("sum", "mean", "min", "max")


@pytest.fixture(scope="module")
def npu_api():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(DEVICE))
    from ops_gnn import coalesce

    return coalesce


def _run(npu_api, edge_index, edge_attr=None, **kwargs):
    """Compare one call against the CPU reference and structural contracts."""
    ei_cpu = edge_index.clone()
    attr_cpu = None if edge_attr is None else edge_attr.clone()
    expected_ei, expected_attr = cpu_coalesce(ei_cpu, attr_cpu, **kwargs)

    npu_ei = ei_cpu.npu()
    npu_attr = None if attr_cpu is None else attr_cpu.npu()
    actual_ei, actual_attr = npu_api(npu_ei, npu_attr, **kwargs)

    assert actual_ei.device.type == "npu"
    assert actual_ei.dtype == torch.int64
    assert tuple(actual_ei.shape) == tuple(expected_ei.shape)
    assert_close(actual_ei, expected_ei)

    if expected_attr is None:
        assert actual_attr is None
    else:
        assert actual_attr is not None and actual_attr.device.type == "npu"
        assert actual_attr.dtype == attr_cpu.dtype
        assert tuple(actual_attr.shape) == tuple(expected_attr.shape)
        assert_close(actual_attr, expected_attr)

    # Inputs must never be modified.
    torch.testing.assert_close(npu_ei.cpu(), ei_cpu, rtol=0, atol=0)
    if npu_attr is not None:
        torch.testing.assert_close(npu_attr.cpu(), attr_cpu, rtol=0, atol=0)
    return actual_ei, actual_attr


def _random_graph(edges, nodes, duplicates=0.5, seed=0, dtype=torch.float32, channels=1):
    generator = torch.Generator().manual_seed(seed)
    unique = max(1, int(edges * (1.0 - duplicates)))
    base = torch.stack([
        torch.randint(nodes, (unique,), generator=generator),
        torch.randint(nodes, (unique,), generator=generator),
    ])
    # Repeat some edges to force duplicate groups.
    picks = torch.randint(unique, (edges,), generator=generator)
    edge_index = base[:, picks]
    attr = None
    if channels is not None:
        shape = (edges,) if channels == 1 else (edges, channels)
        attr = torch.randn(shape, generator=generator, dtype=dtype)
    return edge_index, attr


# ---------------------------------------------------------------------------
# L0 gates
# ---------------------------------------------------------------------------

def test_no_attr(npu_api):
    edge_index = torch.tensor([[0, 1, 0, 2, 0, 1], [1, 0, 1, 3, 1, 0]])
    _run(npu_api, edge_index)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sum(npu_api, dtype):
    edge_index, attr = _random_graph(64, 8, seed=1, dtype=dtype)
    _run(npu_api, edge_index, attr, reduce="sum")


@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_reductions(npu_api, reduce, dtype):
    edge_index, attr = _random_graph(96, 10, seed=2, dtype=dtype)
    _run(npu_api, edge_index, attr, reduce=reduce)


@pytest.mark.parametrize("dtype", DTYPES)
def test_add_alias(npu_api, dtype):
    edge_index, attr = _random_graph(96, 10, seed=2, dtype=dtype)
    _run(npu_api, edge_index, attr, reduce="add")


def test_sort_by_col(npu_api):
    edge_index, attr = _random_graph(80, 9, seed=3)
    _run(npu_api, edge_index, attr, reduce="sum", sort_by_row=False)


def test_is_sorted(npu_api):
    # Build a sorted-by-key input with adjacent duplicates.
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2], [1, 1, 2, 0, 0, 3, 3, 3]])
    attr = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    _run(npu_api, edge_index, attr, reduce="sum", is_sorted=True)


def test_explicit_num_nodes(npu_api):
    edge_index, attr = _random_graph(50, 6, seed=4)
    _run(npu_api, edge_index, attr, reduce="sum", num_nodes=100)


def test_empty_no_attr(npu_api):
    edge_index = torch.empty((2, 0), dtype=torch.int64)
    _run(npu_api, edge_index)


def test_empty_with_attr(npu_api):
    edge_index = torch.empty((2, 0), dtype=torch.int64)
    attr = torch.empty((0, 3), dtype=torch.float32)
    _run(npu_api, edge_index, attr, reduce="sum")


def test_single_edge(npu_api):
    edge_index = torch.tensor([[3], [7]])
    _run(npu_api, edge_index)
    _run(npu_api, edge_index, torch.tensor([[1.5]]), reduce="sum")


@pytest.mark.parametrize("reduce", REDUCES)
def test_all_duplicate(npu_api, reduce):
    edge_index = torch.tensor([[4] * 8, [2] * 8])
    attr = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    _run(npu_api, edge_index, attr, reduce=reduce)


def test_no_duplicate(npu_api):
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 3]])
    attr = torch.randn(4, 3, dtype=torch.float16)
    _run(npu_api, edge_index, attr, reduce="sum")


def test_structural_output(npu_api):
    edge_index, attr = _random_graph(32, 5, seed=5)
    actual_ei, actual_attr = _run(npu_api, edge_index, attr)
    assert actual_ei.dtype == torch.int64 and actual_ei.size(0) == 2
    assert actual_attr.dtype == attr.dtype


def test_non_contiguous(npu_api):
    base = torch.tensor([[0, 9, 1, 9, 0, 9, 2], [1, 9, 0, 9, 1, 9, 3]])
    edge_index = base[:, ::2]
    assert not edge_index.is_contiguous()
    attr = torch.randn(4, 2, dtype=torch.float32)
    _run(npu_api, edge_index, attr, reduce="sum")


@pytest.mark.parametrize("channels", [0, 1, 17, 32, 33, 129])
def test_multidim_attr(npu_api, channels):
    edge_index, attr = _random_graph(64, 8, seed=6, channels=channels)
    _run(npu_api, edge_index, attr, reduce="sum")


def test_rank3_attr(npu_api):
    edge_index, _ = _random_graph(40, 6, seed=7, channels=None)
    attr = torch.randn(40, 2, 3, dtype=torch.float16)
    _run(npu_api, edge_index, attr, reduce="mean")


# ---------------------------------------------------------------------------
# L1 behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("edges", [1, 2, 3, 16, 257, 1024, 4096, 8192])
def test_size_sweep(npu_api, edges):
    edge_index, attr = _random_graph(edges, max(2, edges // 2), seed=edges)
    _run(npu_api, edge_index, attr, reduce="sum")


def test_large(npu_api):
    edge_index, attr = _random_graph(200000, 5000, duplicates=0.5, seed=8)
    _run(npu_api, edge_index, attr, reduce="sum")


def test_long_group(npu_api):
    # A single duplicate group of 8192 edges.
    edge_index = torch.tensor([[1] * 8192, [2] * 8192])
    attr = torch.randn(8192, 1, dtype=torch.float16)
    _run(npu_api, edge_index, attr, reduce="sum")
    _run(npu_api, edge_index, attr, reduce="mean")


def test_num_nodes_variants(npu_api):
    edge_index, attr = _random_graph(48, 7, seed=9, dtype=torch.float16)
    _run(npu_api, edge_index, attr, reduce="max")
    _run(npu_api, edge_index, attr, reduce="max", num_nodes=int(edge_index.max()) + 1)
    _run(npu_api, edge_index, attr, reduce="max", num_nodes=int(edge_index.max()) + 100)


def test_idempotent(npu_api):
    edge_index, attr = _random_graph(64, 8, seed=10)
    ei1, attr1 = _run(npu_api, edge_index, attr)
    ei2, attr2 = _run(npu_api, ei1.cpu(), attr1.cpu(), is_sorted=True)
    assert_close(ei2, ei1)
    assert_close(attr2, attr1)


def test_zero_attr(npu_api):
    edge_index = torch.tensor([[0, 0, 1], [1, 1, 0]])
    attr = torch.zeros(3, 2, dtype=torch.float32)
    _run(npu_api, edge_index, attr, reduce="sum")


def test_signed_zero_and_inf(npu_api):
    edge_index = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]])
    attr = torch.tensor([0.0, -0.0, float("inf"), -float("inf")], dtype=torch.float32).reshape(4, 1)
    _run(npu_api, edge_index, attr, reduce="sum")
    _run(npu_api, edge_index, attr, reduce="min")
    _run(npu_api, edge_index, attr, reduce="max")


def test_transpose_attr_view(npu_api):
    edge_index, _ = _random_graph(24, 5, seed=11, channels=None)
    attr = torch.randn(3, 24, dtype=torch.float32).t()
    assert not attr.is_contiguous()
    _run(npu_api, edge_index, attr, reduce="sum")


# ---------------------------------------------------------------------------
# L2 argument validation
# ---------------------------------------------------------------------------

def _valid_edges():
    return torch.tensor([[0, 1, 0]], dtype=torch.int64, device="npu")


def test_rejects_bad_index_dtype(npu_api):
    for dtype in (torch.int32, torch.float32, torch.bool):
        with pytest.raises(RuntimeError):
            npu_api(torch.tensor([[0, 1]], dtype=dtype, device="npu"))


def test_rejects_bad_index_rank(npu_api):
    with pytest.raises(RuntimeError):
        npu_api(torch.tensor([0, 1], dtype=torch.int64, device="npu"))
    with pytest.raises(RuntimeError):
        npu_api(torch.zeros((2, 1, 3), dtype=torch.int64, device="npu"))
    with pytest.raises(RuntimeError):
        npu_api(torch.zeros((3, 2), dtype=torch.int64, device="npu"))
    with pytest.raises(RuntimeError):
        npu_api(torch.zeros((1, 2), dtype=torch.int64, device="npu"))


def test_rejects_cpu_index(npu_api):
    with pytest.raises(RuntimeError):
        npu_api(torch.tensor([[0, 1]], dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bfloat16, torch.int32])
def test_rejects_bad_attr_dtype(npu_api, dtype):
    attr = torch.zeros(3, 1, dtype=dtype, device="npu")
    with pytest.raises(RuntimeError):
        npu_api(_valid_edges(), attr)


def test_rejects_bad_attr_shape(npu_api):
    with pytest.raises(RuntimeError):
        npu_api(_valid_edges(), torch.zeros((), dtype=torch.float32, device="npu"))
    with pytest.raises(RuntimeError):
        npu_api(_valid_edges(), torch.zeros((4, 1), dtype=torch.float32, device="npu"))


def test_rejects_cpu_attr(npu_api):
    with pytest.raises(RuntimeError):
        npu_api(_valid_edges(), torch.zeros((3, 1), dtype=torch.float32))


@pytest.mark.parametrize("reduce", ["mul", "any", "SUM", ""])
def test_rejects_bad_reduce(npu_api, reduce):
    with pytest.raises(ValueError):
        npu_api(_valid_edges(), None, reduce=reduce)


def test_rejects_bad_num_nodes(npu_api):
    with pytest.raises(ValueError):
        npu_api(_valid_edges(), None, num_nodes=-1)
    with pytest.raises(ValueError):
        npu_api(_valid_edges(), None, num_nodes=2**32)
    with pytest.raises(TypeError):
        npu_api(_valid_edges(), None, num_nodes=1.5)

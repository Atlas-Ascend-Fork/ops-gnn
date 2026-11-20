# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2 acceptance for disjoint_union_coo (DGL DisjointUnionCoo parity).

Requires an Ascend A2 NPU. Every expected value comes from the independent CPU
reference below, which uses only PyTorch.
"""

import os

import pytest
import torch

INDEX_DTYPES = (torch.int32, torch.int64)
INT32_MAX = 2**31 - 1
DEVICE = os.environ.get("NPU_DEVICE_ID", "0")


def disjoint_union_reference(sources, dests, num_rows, num_cols, datas=None):
    """Concatenate COO matrices with cumulative row/column/nnz offsets.

    ``datas[i] is None`` contributes the global edge index of the matrix, which
    mirrors the implicit ``Range(0, nnz)`` DGL uses for COO matrices without an
    explicit data array. The returned data tensor is ``None`` when no matrix
    carries data.
    """
    sources = list(sources)
    dests = list(dests)
    num_rows = list(num_rows)
    num_cols = list(num_cols)
    datas = [None] * len(sources) if datas is None else list(datas)
    assert len(dests) == len(sources)
    assert len(num_rows) == len(sources)
    assert len(num_cols) == len(sources)
    assert len(datas) == len(sources)

    has_data = any(data is not None for data in datas)
    out_sources, out_dests, out_datas = [], [], []
    offset_rows = 0
    offset_cols = 0
    offset_edges = 0
    for index, source in enumerate(sources):
        out_sources.append(source + offset_rows)
        out_dests.append(dests[index] + offset_cols)
        if has_data:
            if datas[index] is not None:
                out_datas.append(datas[index] + offset_edges)
            else:
                out_datas.append(
                    torch.arange(offset_edges, offset_edges + source.numel(), dtype=source.dtype))
        offset_rows += num_rows[index]
        offset_cols += num_cols[index]
        offset_edges += source.numel()

    if not sources:
        return None, None, None, 0, 0
    data = torch.cat(out_datas) if has_data else None
    return torch.cat(out_sources), torch.cat(out_dests), data, offset_rows, offset_cols


def assert_close(actual, expected):
    """Exact integer equality; a mismatched dtype or shape is a failure."""
    assert actual.dtype == expected.dtype, f"dtype {actual.dtype} != {expected.dtype}"
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.fixture(scope="module")
def npu_api():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(DEVICE))
    from ops_gnn import disjoint_union_coo

    return disjoint_union_coo


def _cpu(sources, dests, num_rows, num_cols, datas=None):
    dtype = sources[0].dtype
    return disjoint_union_reference(
        [src.to(dtype) for src in sources],
        [dst.to(dtype) for dst in dests],
        num_rows,
        num_cols,
        datas,
    )


def _run(npu_api, sources, dests, num_rows, num_cols, datas=None):
    expected = _cpu(sources, dests, num_rows, num_cols, datas)
    npu_sources = [tensor.npu() for tensor in sources]
    npu_dests = [tensor.npu() for tensor in dests]
    kwargs = {}
    if datas is not None:
        kwargs["datas"] = [None if data is None else data.npu() for data in datas]
    actual = npu_api(npu_sources, npu_dests, num_rows, num_cols, **kwargs)
    assert len(actual) == 5
    source, dest, data, total_rows, total_cols = actual
    assert isinstance(total_rows, int) and isinstance(total_cols, int)
    assert total_rows == expected[3] and total_cols == expected[4]
    assert source.device.type == "npu" and dest.device.type == "npu"
    assert_close(source, expected[0])
    assert_close(dest, expected[1])
    if expected[2] is None:
        assert data is None
    else:
        assert data is not None and data.device.type == "npu"
        assert_close(data, expected[2])
    return actual


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_batch_three_graphs(npu_api, dtype):
    """DGL parity: the three-graph batch from DGL's own DisjointUnionCoo test."""
    sources = [torch.tensor([0, 1, 2, 0, 1], dtype=dtype),
               torch.tensor([0, 1], dtype=dtype),
               torch.tensor([0, 1, 2], dtype=dtype)]
    dests = [torch.tensor([1, 2, 0, 2, 0], dtype=dtype),
             torch.tensor([1, 0], dtype=dtype),
             torch.tensor([2, 0, 1], dtype=dtype)]
    # dgl.batch always carries edge ids (the COO data array), so pass them too.
    datas = [torch.arange(5, dtype=dtype), torch.arange(2, dtype=dtype),
             torch.arange(3, dtype=dtype)]
    source, dest, data, total_rows, total_cols = _run(npu_api, sources, dests, [3, 2, 3], [3, 2, 3], datas)
    assert (total_rows, total_cols) == (8, 8)
    torch.testing.assert_close(source.cpu(), torch.tensor([0, 1, 2, 0, 1, 3, 4, 5, 6, 7], dtype=dtype))
    torch.testing.assert_close(dest.cpu(), torch.tensor([1, 2, 0, 2, 0, 4, 3, 7, 5, 6], dtype=dtype))
    # Local edge ids are shifted by the accumulated nnz, i.e. a global 0..9 range.
    torch.testing.assert_close(data.cpu(), torch.arange(10, dtype=dtype))


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_shifted_data(npu_api, dtype):
    sources = [torch.tensor([0, 1], dtype=dtype), torch.tensor([0, 0, 1], dtype=dtype)]
    dests = [torch.tensor([1, 0], dtype=dtype), torch.tensor([1, 2, 0], dtype=dtype)]
    datas = [torch.tensor([10, 20], dtype=dtype), torch.tensor([30, 40, 50], dtype=dtype)]
    _, _, data, _, _ = _run(npu_api, sources, dests, [2, 2], [3, 3], datas)
    # Data is shifted by the accumulated nnz, exactly as DGL does for edge ids.
    torch.testing.assert_close(data.cpu(), torch.tensor([10, 20, 32, 42, 52], dtype=dtype))


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_mixed_data_and_identity(npu_api, dtype):
    sources = [torch.tensor([0, 1], dtype=dtype), torch.tensor([0], dtype=dtype),
               torch.tensor([0, 1], dtype=dtype)]
    dests = [torch.tensor([1, 0], dtype=dtype), torch.tensor([0], dtype=dtype),
             torch.tensor([1, 0], dtype=dtype)]
    datas = [torch.tensor([5, 6], dtype=dtype), None, torch.tensor([7, 8], dtype=dtype)]
    _, _, data, _, _ = _run(npu_api, sources, dests, [2, 1, 2], [2, 1, 2], datas)
    # Graph 1 has no data array: it contributes its global edge index (2).
    torch.testing.assert_close(data.cpu(), torch.tensor([5, 6, 2, 10, 11], dtype=dtype))


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_no_data_returns_none(npu_api, dtype):
    sources = [torch.tensor([0, 1], dtype=dtype), torch.tensor([0], dtype=dtype)]
    dests = [torch.tensor([1, 0], dtype=dtype), torch.tensor([0], dtype=dtype)]
    _, _, data, _, _ = _run(npu_api, sources, dests, [2, 1], [2, 1])
    assert data is None
    _run(npu_api, sources, dests, [2, 1], [2, 1], [None, None])


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_single_graph(npu_api, dtype):
    sources = [torch.tensor([0, 1, 2], dtype=dtype)]
    dests = [torch.tensor([1, 2, 0], dtype=dtype)]
    source, _, data, total_rows, total_cols = _run(npu_api, sources, dests, [3], [3])
    torch.testing.assert_close(source.cpu(), torch.tensor([0, 1, 2], dtype=dtype))
    assert (total_rows, total_cols) == (3, 3)


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_empty_graph_in_middle(npu_api, dtype):
    sources = [torch.tensor([0, 1], dtype=dtype), torch.tensor([], dtype=dtype),
               torch.tensor([0, 1], dtype=dtype)]
    dests = [torch.tensor([1, 0], dtype=dtype), torch.tensor([], dtype=dtype),
             torch.tensor([1, 0], dtype=dtype)]
    datas = [torch.tensor([0, 1], dtype=dtype), torch.tensor([], dtype=dtype),
             torch.tensor([0, 1], dtype=dtype)]
    source, dest, data, total_rows, total_cols = _run(npu_api, sources, dests, [4, 0, 4], [4, 0, 4],
                                                     datas)
    # The empty graph still contributes its row/col extent but no edges.
    assert (total_rows, total_cols) == (8, 8)
    torch.testing.assert_close(source.cpu(), torch.tensor([0, 1, 4, 5], dtype=dtype))
    torch.testing.assert_close(dest.cpu(), torch.tensor([1, 0, 5, 4], dtype=dtype))
    torch.testing.assert_close(data.cpu(), torch.tensor([0, 1, 2, 3], dtype=dtype))


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_all_graphs_empty(npu_api, dtype):
    sources = [torch.tensor([], dtype=dtype), torch.tensor([], dtype=dtype)]
    dests = [torch.tensor([], dtype=dtype), torch.tensor([], dtype=dtype)]
    source, dest, data, total_rows, total_cols = _run(npu_api, sources, dests, [2, 3], [4, 5])
    assert source.numel() == 0 and dest.numel() == 0
    assert source.dtype == dtype and dest.dtype == dtype
    assert data is None
    assert (total_rows, total_cols) == (5, 9)


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_zero_row_graph_with_data(npu_api, dtype):
    sources = [torch.tensor([], dtype=dtype), torch.tensor([0, 1], dtype=dtype)]
    dests = [torch.tensor([], dtype=dtype), torch.tensor([1, 0], dtype=dtype)]
    datas = [torch.tensor([], dtype=dtype), torch.tensor([3, 4], dtype=dtype)]
    _, _, data, _, _ = _run(npu_api, sources, dests, [0, 2], [0, 2], datas)
    torch.testing.assert_close(data.cpu(), torch.tensor([3, 4], dtype=dtype))


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_non_contiguous_inputs(npu_api, dtype):
    base = torch.tensor([0, 9, 1, 9, 2, 9, 0, 9, 1, 9], dtype=dtype)[::2]
    dest = torch.tensor([1, 9, 2, 9, 0, 9, 2, 9, 0, 9], dtype=dtype)[::2]
    assert not base.is_contiguous()
    sources = [base[:3], base[3:]]
    dests = [dest[:3], dest[3:]]
    _run(npu_api, sources, dests, [3, 2], [3, 2])


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_multiple_tiles_and_cores(npu_api, dtype):
    """Edges far beyond one UB tile and one core per matrix."""
    generator = torch.Generator().manual_seed(7)
    sources, dests, datas = [], [], []
    num_rows, num_cols = [], []
    for index in range(5):
        edges = 40_000 + index * 1_003
        rows = 5_000 + index
        sources.append(torch.randint(rows, (edges,), generator=generator, dtype=dtype))
        dests.append(torch.randint(rows, (edges,), generator=generator, dtype=dtype))
        datas.append(torch.arange(edges, dtype=dtype) + index)
        num_rows.append(rows)
        num_cols.append(rows + 1)
    _, _, data, total_rows, total_cols = _run(npu_api, sources, dests, num_rows, num_cols, datas)
    assert total_rows == sum(num_rows) and total_cols == sum(num_cols)


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
@pytest.mark.parametrize("edges", [1, 7, 8, 31, 32, 33, 1023, 1024, 1025, 4097])
def test_unaligned_sizes(npu_api, dtype, edges):
    generator = torch.Generator().manual_seed(edges)
    sources = [torch.randint(64, (edges,), generator=generator, dtype=dtype)]
    dests = [torch.randint(64, (edges,), generator=generator, dtype=dtype)]
    _run(npu_api, sources, dests, [64], [64])


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_identity_fill_across_tiles(npu_api, dtype):
    """A data-less matrix larger than one tile must keep a monotonic edge index."""
    edges = 20_000
    sources = [torch.zeros(1, dtype=dtype), torch.zeros(edges, dtype=dtype)]
    dests = [torch.zeros(1, dtype=dtype), torch.zeros(edges, dtype=dtype)]
    # Graph 0 carries data, so the implicit edge index is only materialised for
    # graph 1: it must continue at the accumulated nnz and cross every tile.
    datas = [torch.tensor([7], dtype=dtype), None]
    _, _, data, _, _ = _run(npu_api, sources, dests, [10, 1], [10, 1], datas)
    expected = torch.cat((torch.tensor([7], dtype=dtype),
                          torch.arange(1, edges + 1, dtype=dtype)))
    torch.testing.assert_close(data.cpu(), expected)


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_multi_tile_per_core(npu_api, dtype):
    """More edges per AIV than one UB tile must not drop or duplicate elements."""
    edges = 2_000_000
    generator = torch.Generator().manual_seed(11)
    sources = [torch.randint(1000, (edges,), generator=generator, dtype=dtype),
               torch.zeros(3, dtype=dtype)]
    dests = [torch.randint(1000, (edges,), generator=generator, dtype=dtype),
             torch.zeros(3, dtype=dtype)]
    datas = [torch.arange(edges, dtype=dtype), None]
    source, dest, data, total_rows, _ = _run(npu_api, sources, dests, [1000, 4], [1000, 4], datas)
    # The data-less matrix continues the global edge index at the accumulated nnz.
    torch.testing.assert_close(data[-3:].cpu(), torch.tensor([edges, edges + 1, edges + 2], dtype=dtype))
    torch.testing.assert_close(source[-3:].cpu(), torch.tensor([1000, 1000, 1000], dtype=dtype))
    assert source.numel() == edges + 3


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_large_offsets_stay_exact(npu_api, dtype):
    """Row offsets well beyond one core's element count stay exact."""
    sources = [torch.tensor([0, 1], dtype=dtype), torch.tensor([0, 1], dtype=dtype)]
    dests = [torch.tensor([1, 0], dtype=dtype), torch.tensor([1, 0], dtype=dtype)]
    num_rows = [1_000_000, 2_000_000]
    _, dest, _, total_rows, _ = _run(npu_api, sources, dests, num_rows, num_rows)
    assert total_rows == 3_000_000
    torch.testing.assert_close(dest.cpu(), torch.tensor([1, 0, 1_000_001, 1_000_000], dtype=dtype))


def test_int32_boundary_values_stay_exact(npu_api):
    """The largest legal extent keeps every shifted id exact on the int64 path."""
    biggest = 2**31 - 1
    sources = [torch.tensor([0], dtype=torch.int64), torch.tensor([1], dtype=torch.int64)]
    dests = [torch.tensor([0], dtype=torch.int64), torch.tensor([2], dtype=torch.int64)]
    datas = [torch.tensor([0], dtype=torch.int64), torch.tensor([1], dtype=torch.int64)]
    source, dest, data, total_rows, _ = _run(npu_api, sources, dests,
                                             [biggest - 2, 2], [biggest - 2, 2], datas)
    assert total_rows == biggest
    torch.testing.assert_close(source.cpu(), torch.tensor([0, biggest - 2 + 1], dtype=torch.int64))
    torch.testing.assert_close(dest.cpu(), torch.tensor([0, biggest], dtype=torch.int64))
    torch.testing.assert_close(data.cpu(), torch.tensor([0, 1 + 1], dtype=torch.int64))


def test_repeated_calls_are_deterministic(npu_api):
    sources = [torch.tensor([0, 1, 2], dtype=torch.int64), torch.tensor([2, 1], dtype=torch.int64)]
    dests = [torch.tensor([1, 2, 0], dtype=torch.int64), torch.tensor([0, 1], dtype=torch.int64)]
    npu_sources = [tensor.npu() for tensor in sources]
    npu_dests = [tensor.npu() for tensor in dests]
    first = npu_api(npu_sources, npu_dests, [3, 3], [3, 3])
    for _ in range(5):
        again = npu_api(npu_sources, npu_dests, [3, 3], [3, 3])
        for left, right in zip(first[:2], again[:2]):
            torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)


def test_inputs_are_not_modified(npu_api):
    sources = [torch.tensor([0, 1, 2], dtype=torch.int64), torch.tensor([2, 1], dtype=torch.int64)]
    dests = [torch.tensor([1, 2, 0], dtype=torch.int64), torch.tensor([0, 1], dtype=torch.int64)]
    datas = [torch.tensor([3, 4, 5], dtype=torch.int64), torch.tensor([6, 7], dtype=torch.int64)]
    npu_sources = [tensor.npu() for tensor in sources]
    npu_dests = [tensor.npu() for tensor in dests]
    npu_datas = [tensor.npu() for tensor in datas]
    npu_api(npu_sources, npu_dests, [3, 3], [3, 3], npu_datas)
    torch.npu.synchronize()
    for expected, actual in zip(sources + dests + datas, npu_sources + npu_dests + npu_datas):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _npu_vector(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype).npu()


def test_rejects_empty_graph_list(npu_api):
    with pytest.raises(RuntimeError, match="at least one COO matrix"):
        npu_api([], [], [], [])


def test_rejects_mismatched_lengths(npu_api):
    source = _npu_vector([0, 1])
    dest = _npu_vector([1, 0])
    with pytest.raises(RuntimeError, match="same length"):
        npu_api([source], [dest], [2, 2], [2])
    with pytest.raises(RuntimeError, match="same length"):
        npu_api([source, source], [dest], [2], [2])
    with pytest.raises(RuntimeError, match="one entry per COO matrix"):
        npu_api([source], [dest], [2], [2], [None, None])


def test_rejects_mixed_index_dtypes(npu_api):
    with pytest.raises(RuntimeError, match="same dtype"):
        npu_api([_npu_vector([0], torch.int32)], [_npu_vector([0], torch.int64)], [1], [1])


def test_rejects_cpu_tensors(npu_api):
    with pytest.raises(RuntimeError, match="NPU tensors"):
        npu_api([torch.tensor([0])], [torch.tensor([0])], [1], [1])


def test_rejects_rank_mismatch(npu_api):
    with pytest.raises(RuntimeError, match="must be 1-D"):
        npu_api([_npu_vector([[0]])], [_npu_vector([0])], [1], [1])


def test_rejects_bad_num_rows(npu_api):
    source = _npu_vector([0, 1])
    dest = _npu_vector([1, 0])
    with pytest.raises(RuntimeError, match="non-negative"):
        npu_api([source], [dest], [-1], [2])
    with pytest.raises(RuntimeError, match="must not exceed"):
        npu_api([source], [dest], [INT32_MAX + 1], [2])


def test_rejects_oversized_totals(npu_api):
    source = _npu_vector([0, 1])
    dest = _npu_vector([1, 0])
    with pytest.raises(RuntimeError, match="must not exceed"):
        npu_api([source, source], [dest, dest], [INT32_MAX, 2], [2, 2])


def test_rejects_bad_data(npu_api):
    source = _npu_vector([0, 1])
    dest = _npu_vector([1, 0])
    with pytest.raises(RuntimeError, match="data dtype must match"):
        npu_api([source], [dest], [2], [2], [_npu_vector([0, 1], torch.int32)])
    with pytest.raises(RuntimeError, match="one value per edge"):
        npu_api([source], [dest], [2], [2], [_npu_vector([0])])
    with pytest.raises(RuntimeError, match="data must be 1-D"):
        npu_api([source], [dest], [2], [2], [_npu_vector([[0, 1]])])
    with pytest.raises(RuntimeError, match="NPU"):
        npu_api([source], [dest], [2], [2], [torch.tensor([0, 1], dtype=torch.int64)])


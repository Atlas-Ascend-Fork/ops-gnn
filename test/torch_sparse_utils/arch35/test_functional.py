# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Functional contracts for public keyword calls, graph walks and CSR adapters."""

import os
from pathlib import Path
import runpy

import pytest
import torch
import torch_npu

import ops_gnn as utils
from ops_gnn.torch_sparse_utils import _diagonal_entries, _ind2ptr_npu, _ptr2ind_npu

_REFERENCE = runpy.run_path(Path(__file__).with_name("golden.py"))


@pytest.fixture(scope="module", name="device")
def _device():
    index = int(os.environ.get("NPU_DEVICE_ID", "0"))
    torch_npu.npu.set_device(index)
    return torch.device("npu", index)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("offset", [-1, 0, 2])
def test_public_set_diag_matches_torch_sparse_cpu(weighted, offset, device):
    torch_sparse = pytest.importorskip("torch_sparse")
    row = torch.tensor([6, 0, 2, 2, 0, 5])
    col = torch.tensor([8, 1, 4, 4, 1, 3])
    value = torch.arange(6, dtype=torch.float32) if weighted else None
    source = torch_sparse.SparseTensor(row=row, col=col, value=value, sparse_sizes=(7, 9))
    diagonal = min(7, 9 - offset) if offset >= 0 else min(7 + offset, 9)
    values = torch.arange(diagonal, dtype=torch.float32) + 2.5 if weighted else None
    expected = torch_sparse.set_diag(source, values=values, k=offset)
    actual = torch_sparse.set_diag(source.to(device),
                                   values=None if values is None else values.to(device), k=offset)
    for layout in ("coo", "csr"):
        for result, reference in zip(getattr(actual, layout)(), getattr(expected, layout)()):
            actual_cpu = None if result is None else result.cpu()
            torch.testing.assert_close(actual_cpu, reference, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (7, 0), (3, 7), (7, 3)])
@pytest.mark.parametrize("offset", [-4, 0, 4])
def test_public_mask_keyword_arguments(shape, offset, device):
    rows, columns = shape
    count = min(rows, columns)
    row = torch.arange(count, dtype=torch.int64)
    col = torch.arange(count, dtype=torch.int64)
    expected = _REFERENCE["non_diag_mask_reference"](row, col, rows, columns, offset)
    actual = utils.non_diag_mask(
        row=row.to(device), col=col.to(device), M=rows, N=columns, k=offset
    )
    assert actual.dtype == torch.bool and actual.device == device
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("length", [0, 1, 13])
@pytest.mark.parametrize("isolated", [False, True])
def test_walk_empty_starts_preserve_shape(length, isolated, device):
    rowptr = torch.tensor([0, 0, 0] if isolated else [0, 1, 2], device=device)
    col = (
        torch.empty(0, dtype=torch.int64, device=device)
        if isolated else torch.tensor([1, 0], device=device)
    )
    starts = torch.empty(0, dtype=torch.int64, device=device)
    result = utils.torch_sparse_random_walk(rowptr, col, starts, length)
    shared = utils.torch_sparse_random_walk_with_rand(
        rowptr, col, starts, length, torch.empty((0, length), device=device)
    )
    for actual in (result, shared):
        assert actual.shape == (0, length + 1)
        assert actual.dtype == torch.int64 and actual.device == device


@pytest.mark.parametrize("length", [0, 1, 11])
@pytest.mark.parametrize("isolated", [False, True])
def test_walk_single_neighbor_and_isolated_graph(length, isolated, device):
    rowptr = torch.zeros(6, dtype=torch.int64) if isolated else torch.arange(6)
    col = torch.empty(0, dtype=torch.int64) if isolated else torch.tensor([1, 2, 3, 4, 0])
    starts = torch.tensor([4, 0, 4, 2], dtype=torch.int64)
    rand = torch.full((len(starts), length), 0.99999994)
    expected = _REFERENCE["random_walk_reference"](rowptr, col, starts, length, rand)
    inputs = rowptr.to(device), col.to(device), starts.to(device)
    results = (
        utils.torch_sparse_random_walk(*inputs, length),
        utils.torch_sparse_random_walk_with_rand(*inputs, length, rand.to(device)),
    )
    for actual in results:
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("counts", [[], [0], [0, 0, 0], [0, 2, 0, 1, 0], [1, 1, 1, 1]])
def test_csr_conversion_roundtrip_with_empty_rows(counts, device):
    counts_tensor = torch.tensor(counts, dtype=torch.int64)
    index = torch.arange(len(counts)).repeat_interleave(counts_tensor)
    expected = torch.cat((torch.zeros(1, dtype=torch.int64), counts_tensor.cumsum(0)))
    pointer = _ind2ptr_npu(index.to(device), len(counts))
    actual = _ptr2ind_npu(pointer, len(index))
    assert torch.equal(pointer.cpu(), expected)
    assert torch.equal(actual.cpu(), index)
    assert pointer.device == device and actual.device == device


@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (7, 0), (3, 7), (7, 3)])
@pytest.mark.parametrize("offset", [-4, 0, 4])
def test_empty_coo_diagonal_insertion(shape, offset, device):
    row = torch.empty(0, dtype=torch.int64)
    col = torch.empty(0, dtype=torch.int64)
    value = torch.empty((0, 2), dtype=torch.float32)
    entries = row, col, value, shape
    supplied = torch.tensor([2.0, 5.0])
    expected = _REFERENCE["diagonal_entries_reference"](entries, supplied, offset)
    actual = _diagonal_entries(
        (row.to(device), col.to(device), value.to(device), shape),
        supplied.to(device), offset,
    )
    for result, reference in zip(actual, expected):
        assert result.device == device
        assert torch.equal(result.cpu(), reference)

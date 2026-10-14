# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Key value, interface and frontend contracts without an upstream-package dependency."""

import os
from pathlib import Path
import runpy
from typing import NamedTuple, Optional, Tuple

import pytest
import torch
import torch_npu

import ops_gnn as utils
from ops_gnn.torch_sparse_utils import _diagonal_entries, _validate_diag_values
_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
assert_valid_walk = _GOLDEN["assert_valid_walk"]
diagonal_entries_reference = _GOLDEN["diagonal_entries_reference"]
non_diag_mask_reference = _GOLDEN["non_diag_mask_reference"]
random_walk_reference = _GOLDEN["random_walk_reference"]


@pytest.fixture(scope="module", name="device")
def _npu_device():
    index = int(os.environ.get("NPU_DEVICE_ID", "0"))
    torch_npu.npu.set_device(index)
    return torch.device("npu", index)


@pytest.mark.parametrize(
    "shape", [(0, 0, 0), (4, 5, 0), (7, 7, 37), (3, 8, 97), (8, 3, 257)]
)
@pytest.mark.parametrize("offset", [0, 1, -1, 9, -(2**63), 2**63 - 1])
def test_mask_values_and_layout(shape, offset, device):
    rows, columns, count = shape
    row = torch.arange(count, dtype=torch.int64) % max(rows, 1)
    col = (torch.arange(count, dtype=torch.int64) * 5 + 1) % max(columns, 1)
    expected = non_diag_mask_reference(row, col, rows, columns, offset)
    # Materialize backing storage before making the device views noncontiguous.
    device_row = row.repeat_interleave(2).to(device)[::2]
    device_col = col.repeat_interleave(2).to(device)[::2]
    actual = utils.non_diag_mask(device_row, device_col, rows, columns, offset)
    assert actual.dtype == torch.bool and actual.device == device
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize(
    "case",
    ["dtype", "rank", "length", "negative_rows", "negative_columns", "cpu", "overflow"],
)
def test_mask_metadata_errors(case, device):
    row = torch.tensor([0, 1], dtype=torch.int64, device=device)
    col = torch.tensor([1, 0], dtype=torch.int64, device=device)
    rows, columns = 2, 2
    if case == "dtype":
        row = row.int()
    elif case == "rank":
        row = row.view(1, 2)
    elif case == "length":
        col = col[:1]
    elif case == "negative_rows":
        rows = -1
    elif case == "negative_columns":
        columns = -1
    elif case == "cpu":
        row, col = row.cpu(), col.cpu()
    else:
        rows = columns = 2**63 - 1
    with pytest.raises(RuntimeError):
        utils.non_diag_mask(row, col, rows, columns, 0)


@pytest.mark.parametrize("length", [0, 1, 9])
@pytest.mark.parametrize("starts", [[], [0], [0, 1, 2, 3]])
def test_walk_exact_and_dead_ends(length, starts, device):
    rowptr = torch.tensor([0, 1, 2, 2, 3], dtype=torch.int64)
    col = torch.tensor([1, 2, 3], dtype=torch.int64)
    source = torch.tensor(starts, dtype=torch.int64)
    rand = torch.zeros((len(starts), length), dtype=torch.float32)
    expected = random_walk_reference(rowptr, col, source, length, rand)
    actual = utils.torch_sparse_random_walk(
        rowptr.to(device), col.to(device), source.to(device), length
    )
    assert actual.dtype == torch.int64 and actual.device == device
    assert torch.equal(actual.cpu(), expected)


def test_walk_shared_random_and_native_reproducibility(device):
    rowptr = torch.tensor([0, 3, 4, 4, 5], dtype=torch.int64)
    col = torch.tensor([0, 1, 2, 3, 0], dtype=torch.int64)
    starts = torch.arange(4, dtype=torch.int64).repeat(8)
    rand = torch.rand((32, 13), generator=torch.Generator().manual_seed(73))
    rand[0, :3] = torch.tensor([0, 0.5, 0.99999994])
    expected = random_walk_reference(rowptr, col, starts, 13, rand)
    inputs = rowptr.to(device), col.to(device), starts.to(device)
    actual = utils.torch_sparse_random_walk_with_rand(*inputs, 13, rand.to(device))
    assert torch.equal(actual.cpu(), expected)
    torch.manual_seed(2026)
    first = utils.torch_sparse_random_walk(*inputs, 13).cpu()
    torch.manual_seed(2026)
    second = utils.torch_sparse_random_walk(*inputs, 13).cpu()
    assert torch.equal(first, second)
    assert_valid_walk(rowptr, col, starts, first)


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "rank",
        "empty_pointer",
        "negative_length",
        "overflow",
        "cpu",
        "rand_dtype",
        "rand_shape",
    ],
)
def test_walk_metadata_errors(case, device):
    rowptr = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    col = torch.tensor([1, 0], dtype=torch.int64, device=device)
    starts = torch.tensor([0, 1], dtype=torch.int64, device=device)
    length = 2
    rand = torch.zeros((2, 2), dtype=torch.float32, device=device)
    if case == "dtype":
        rowptr = rowptr.int()
    elif case == "rank":
        starts = starts.view(1, 2)
    elif case == "empty_pointer":
        rowptr = rowptr[:0]
    elif case == "negative_length":
        length = -1
    elif case == "overflow":
        length = 2**63 - 1
    elif case == "cpu":
        rowptr, col, starts = rowptr.cpu(), col.cpu(), starts.cpu()
    elif case == "rand_dtype":
        rand = rand.half()
    else:
        rand = rand[:, :1]
    with pytest.raises(RuntimeError):
        utils.torch_sparse_random_walk_with_rand(rowptr, col, starts, length, rand)


def _case_value(choices, key):
    try:
        return choices[key]
    except KeyError as error:
        raise AssertionError("unknown parameterized test case") from error


class _CooEntries(NamedTuple):
    row: torch.Tensor
    col: torch.Tensor
    value: Optional[torch.Tensor]
    shape: Tuple[int, int]


def _entries(dtype=torch.float32):
    row = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    col = torch.tensor([1, 1, 1, 2, 0, 2], dtype=torch.int64)
    value = torch.arange(12, dtype=dtype).view(6, 2)
    return _CooEntries(
        row,
        col,
        value,
        (3, 3),
    )


def _to_device(entries, device):
    row, col, value, shape = entries
    return _CooEntries(
        row.to(device),
        col.to(device),
        None if value is None else value.to(device),
        shape,
    )


@pytest.mark.parametrize(
    "kind",
    [
        "none",
        "scalar",
        "feature",
        "matrix",
        "scalar_tensor",
        "one_element",
        "one_by_one",
    ],
)
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_diagonal_values_and_stable_order(kind, offset, device):
    entries = _entries()
    diagonal = 3 - abs(offset)
    choices = {
        "none": None,
        "scalar": 2.5,
        "feature": torch.tensor([3.0, 4.0]),
        "matrix": torch.arange(diagonal * 2, dtype=torch.float32).view(diagonal, 2),
        "scalar_tensor": torch.tensor(7, dtype=torch.float16),
        "one_element": torch.tensor([8], dtype=torch.int64),
        "one_by_one": torch.tensor([[9]], dtype=torch.float16),
    }
    values = _case_value(choices, kind)
    expected = diagonal_entries_reference(entries, values, offset)
    supplied = values.to(device) if isinstance(values, torch.Tensor) else values
    actual = _diagonal_entries(_to_device(entries, device), supplied, offset)
    for result, reference in zip(actual, expected):
        assert torch.equal(result.cpu(), reference)


@pytest.mark.parametrize(
    "case", ["list", "string", "object", "dtype", "empty_dtype", "cpu", "shape"]
)
@pytest.mark.parametrize("offset", [0, 9])
def test_diagonal_values_errors(case, offset, device):
    entries = _to_device(_entries(), device)
    choices = {
        "list": [1, 2],
        "string": "invalid",
        "object": object(),
        "dtype": torch.ones((3, 2), dtype=torch.float16, device=device),
        "empty_dtype": torch.empty((0, 2), dtype=torch.float16, device=device),
        "cpu": torch.ones((3, 2), dtype=torch.float32),
        "shape": torch.ones((4, 2), dtype=torch.float32, device=device),
    }
    error = TypeError if case in ("list", "string", "object") else RuntimeError
    with pytest.raises(error):
        _diagonal_entries(entries, _case_value(choices, case), offset)


@pytest.mark.parametrize(
    "values", [None, 3, "ignored", torch.ones(2, dtype=torch.float64)]
)
def test_unweighted_diagonal_ignores_values(values, device):
    row, col, _, shape = _entries()
    entries = row, col, None, shape
    expected = diagonal_entries_reference(entries, values, 0)
    actual = _diagonal_entries(_to_device(entries, device), values, 0)
    assert actual[2] is None
    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])


@pytest.mark.parametrize("kind", ["none", "number", "empty", "scalar", "one", "one_by_one"])
def test_empty_diagonal_preserves_weighted_entries(kind, device):
    entries = _entries()
    choices = {
        "none": None,
        "number": 4,
        "empty": torch.empty((0, 2), dtype=torch.float32),
        "scalar": torch.tensor(5, dtype=torch.float16),
        "one": torch.tensor([6], dtype=torch.int64),
        "one_by_one": torch.tensor([[7]], dtype=torch.float16),
    }
    value = _case_value(choices, kind)
    expected = diagonal_entries_reference(entries, value, 9)
    supplied = value.to(device) if isinstance(value, torch.Tensor) else value
    actual = _diagonal_entries(_to_device(entries, device), supplied, 9)
    for result, reference in zip(actual, expected):
        assert result.device == device
        assert torch.equal(result.cpu(), reference)


@pytest.mark.parametrize("shape", [(), (1,), (1, 1), (2,), (0, 2)])
def test_scalar_dtype_rule_matches_deployed_cpu_assignment(shape):
    value = torch.zeros((3, 2), dtype=torch.float32)
    supplied = torch.ones(shape, dtype=torch.float16)
    selection = torch.ones(3, dtype=torch.bool)
    if supplied.numel() == 1:
        _validate_diag_values(value, supplied)
        expected = value.clone()
        expected[selection] = supplied
        actual = value.clone()
        actual[...] = supplied
        assert torch.equal(actual, expected)
    else:
        with pytest.raises(RuntimeError):
            value[selection] = supplied
        with pytest.raises(RuntimeError):
            _validate_diag_values(value, supplied)

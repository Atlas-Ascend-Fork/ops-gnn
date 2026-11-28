# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend A2/A3 (arch22) acceptance for ``spspmm``.

Every expected value comes from the independent CPU reference in ``golden.py``;
no device result is used to build an expectation. The cases cover the operator
contract (coalesced row-major output, structural zeros, merged duplicates), the
data dependent candidate buffers (hub rows, empty operands), the three-pass
prefix sum (a row count above one scan chunk) and the global-cursor spill pass
(a row above the UB cursor capacity).
"""

import runpy
from pathlib import Path

import pytest
import torch

_GOLDEN = runpy.run_path(str(Path(__file__).with_name("golden.py")))
spspmm_golden = _GOLDEN["spspmm_golden"]
coalesced_rows = _GOLDEN["coalesced_rows"]

DTYPES = [torch.float32, torch.float16]
# Same tolerance family the other sparse operators use: the kernel accumulates
# in float32 and only the final store is narrowed.
TOLERANCE = {torch.float32: (1e-5, 1e-5), torch.float16: (1e-2, 1e-2)}


def _coo(rows, cols, values, dtype):
    return (torch.tensor([rows, cols], dtype=torch.int64),
            torch.tensor(values, dtype=dtype))


def _empty(dtype):
    return (torch.empty((2, 0), dtype=torch.int64),
            torch.empty((0,), dtype=dtype))


def _random_operand(num_rows, num_cols, edges, seed, dtype):
    generator = torch.Generator().manual_seed(seed)
    if edges == 0:
        return _empty(dtype)
    index = torch.stack([
        torch.randint(num_rows, (edges,), generator=generator),
        torch.randint(num_cols, (edges,), generator=generator),
    ])
    value = torch.rand(edges, generator=generator, dtype=torch.float32).to(dtype)
    return index, value


def _check_layout(index):
    """The product must be row-major with strictly ascending columns per row."""
    rows = index[0].tolist()
    cols = index[1].tolist()
    for position in range(1, len(rows)):
        if rows[position] < rows[position - 1]:
            raise AssertionError(f"row ids are not ascending at {position}: {rows}")
        if rows[position] == rows[position - 1] and cols[position] <= cols[position - 1]:
            raise AssertionError(f"column ids are not strictly ascending at {position}: {cols}")


def _check(ops_gnn, index_a, value_a, index_b, value_b, m, k, n, coalesced=False):
    expected_index, expected_value = spspmm_golden(index_a, value_a, index_b, value_b, m, k, n)
    rtol, atol = TOLERANCE[value_a.dtype]
    index, value = ops_gnn.spspmm(
        index_a.npu(), value_a.npu(), index_b.npu(), value_b.npu(), m, k, n,
        coalesced=coalesced)
    if index.dtype != torch.int64 or index.device.type != "npu":
        raise AssertionError(
            f"index_c must be int64 on the NPU, got {index.dtype} on {index.device}")
    if value.dtype != value_a.dtype or value.device.type != "npu":
        raise AssertionError(f"value_c must be {value_a.dtype} on the NPU, got {value.dtype}")
    if tuple(index.shape) != (2, expected_index.size(1)):
        raise AssertionError(
            f"index_c shape {tuple(index.shape)} != (2, {expected_index.size(1)})")
    index_cpu = index.cpu()
    value_cpu = value.cpu()
    torch.testing.assert_close(index_cpu, expected_index, rtol=0, atol=0)
    torch.testing.assert_close(value_cpu.double(), expected_value, rtol=rtol, atol=atol,
                               check_dtype=False)
    _check_layout(index_cpu)
    return index_cpu, value_cpu


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [
    (1, 1, 1, 1),
    (8, 4, 6, 12),
    (16, 16, 16, 40),
    (64, 32, 16, 200),
    (128, 128, 128, 1000),
    (5, 1, 7, 9),
    (37, 64, 11, 300),
])
def test_random_products(npu_api, shape, dtype):
    m, k, n, edges = shape
    index_a, value_a = _random_operand(m, k, edges, seed=11, dtype=dtype)
    index_b, value_b = _random_operand(k, n, max(1, edges // 2), seed=29, dtype=dtype)
    _check(npu_api, index_a, value_a, index_b, value_b, m, k, n)


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_operands(npu_api, dtype):
    index_b, value_b = _random_operand(3, 4, 5, seed=3, dtype=dtype)
    index_a, value_a = _empty(dtype)
    _check(npu_api, index_a, value_a, index_b, value_b, 3, 3, 4)

    index_a, value_a = _random_operand(3, 4, 5, seed=3, dtype=dtype)
    index_b, value_b = _empty(dtype)
    _check(npu_api, index_a, value_a, index_b, value_b, 3, 4, 5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_output_column(npu_api, dtype):
    index_a, value_a = _empty(dtype)
    index_b, value_b = _empty(dtype)
    _check(npu_api, index_a, value_a, index_b, value_b, 4, 5, 0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_duplicate_coordinates_are_merged(npu_api, dtype):
    index_a, value_a = _coo([0, 0, 1, 1, 1], [0, 0, 0, 1, 1], [1.0, 2.0, 0.5, 3.0, 0.25], dtype)
    index_b, value_b = _coo([0, 0, 1, 1], [0, 1, 0, 1], [4.0, 5.0, 6.0, 7.0], dtype)
    _check(npu_api, index_a, value_a, index_b, value_b, 2, 2, 2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_sum_keeps_the_coordinate(npu_api, dtype):
    index_a, value_a = _coo([0, 0], [0, 1], [1.0, 1.0], dtype)
    index_b, value_b = _coo([0, 1], [0, 0], [1.0, -1.0], dtype)
    index, value = _check(npu_api, index_a, value_a, index_b, value_b, 1, 2, 1)
    if index.size(1) != 1 or value.numel() != 1:
        raise AssertionError(f"the cancelled coordinate must be kept, got {index.tolist()}")
    if float(value[0]) != 0.0:
        raise AssertionError(f"the cancelled coordinate must sum to zero, got {float(value[0])}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_self_loops(npu_api, dtype):
    index, value = _coo([0, 1, 2], [0, 1, 2], [1.0, 2.0, 3.0], dtype)
    index_cpu, value_cpu = _check(npu_api, index, value, index, value, 3, 3, 3)
    if index_cpu.tolist() != [[0, 1, 2], [0, 1, 2]]:
        raise AssertionError(f"unexpected product structure: {index_cpu.tolist()}")
    torch.testing.assert_close(value_cpu.double(), torch.tensor([1.0, 4.0, 9.0],
                                                                dtype=torch.float64),
                               rtol=TOLERANCE[dtype][0], atol=TOLERANCE[dtype][1],
                               check_dtype=False)


@pytest.mark.parametrize("dtype", DTYPES)
def test_hub_row(npu_api, dtype):
    # One output row touches every B row, so its candidate count is the total
    # length of all those B rows while the unique column count stays bounded.
    rows, cols, values = [], [], []
    for row in range(32):
        for offset in range(16):
            rows.append(row)
            cols.append((row * 3 + offset * 5) % 128)
            values.append(1.0 + offset)
    index_a, value_a = _coo([0] * 32, list(range(32)), [1.0] * 32, dtype)
    index_b, value_b = _coo(rows, cols, values, dtype)
    _check(npu_api, index_a, value_a, index_b, value_b, 1, 32, 128)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scan_chunking(npu_api, dtype):
    # More rows than one prefix-sum chunk, so both the row histogram and the
    # unique-count scan run their three-pass path.
    m = 9000
    index_a = torch.stack([torch.arange(m), torch.arange(m) % 7])
    value_a = torch.ones(m, dtype=dtype)
    index_b, value_b = _random_operand(7, 5, 20, seed=5, dtype=dtype)
    _check(npu_api, index_a, value_a, index_b, value_b, m, 7, 5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_long_row_spill(npu_api, dtype):
    # One row above the UB cursor capacity, so the merge falls back to the device
    # cursors of the spill pass; both half-columns must still sum the whole row.
    k = 9000
    index_a, value_a = _coo([0] * k, list(range(k)), [1.0] * k, dtype)
    index_b, value_b = _coo(list(range(k)), [inner % 2 for inner in range(k)], [1.0] * k, dtype)
    index_cpu, value_cpu = _check(npu_api, index_a, value_a, index_b, value_b, 1, k, 2)
    if index_cpu.tolist() != [[0, 0], [0, 1]]:
        raise AssertionError(f"unexpected product structure: {index_cpu.tolist()}")
    torch.testing.assert_close(value_cpu.double(), torch.tensor([4500.0, 4500.0],
                                                                dtype=torch.float64),
                               rtol=TOLERANCE[dtype][0], atol=TOLERANCE[dtype][1],
                               check_dtype=False)


@pytest.mark.parametrize("dtype", DTYPES)
def test_coalesced_flag_accepts_sorted_input(npu_api, dtype):
    index_a, value_a = _random_operand(24, 12, 60, seed=7, dtype=dtype)
    index_b, value_b = _random_operand(12, 9, 40, seed=8, dtype=dtype)
    index_a, value_a = coalesced_rows(index_a, value_a)
    index_b, value_b = coalesced_rows(index_b, value_b)
    _check(npu_api, index_a, value_a, index_b, value_b, 24, 12, 9, coalesced=True)


def test_coalesced_flag_rejects_unsorted_rows(npu_api):
    index_a = torch.tensor([[1, 0], [0, 0]], dtype=torch.int64)
    index_b = torch.tensor([[0], [0]], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="requires coalesced inputs"):
        npu_api.spspmm(index_a.npu(), torch.ones(2).npu(), index_b.npu(), torch.ones(1).npu(),
                       2, 1, 1, coalesced=True)


def test_coalesced_flag_rejects_duplicates(npu_api):
    index_a = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    index_b = torch.tensor([[0], [0]], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="requires coalesced inputs"):
        npu_api.spspmm(index_a.npu(), torch.ones(2).npu(), index_b.npu(), torch.ones(1).npu(),
                       2, 1, 1, coalesced=True)


@pytest.mark.parametrize("case", [
    "row_a_out_of_range", "col_a_out_of_range", "row_b_out_of_range",
    "col_b_out_of_range", "zero_k_with_edges",
])
def test_rejects_bad_coordinates(npu_api, case):
    index_a = torch.tensor([[0], [0]], dtype=torch.int64)
    index_b = torch.tensor([[0], [0]], dtype=torch.int64)
    m, k, n = 2, 2, 2
    if case == "row_a_out_of_range":
        index_a = torch.tensor([[4], [0]], dtype=torch.int64)
    elif case == "col_a_out_of_range":
        index_a = torch.tensor([[0], [5]], dtype=torch.int64)
    elif case == "row_b_out_of_range":
        index_b = torch.tensor([[3], [0]], dtype=torch.int64)
    elif case == "col_b_out_of_range":
        index_b = torch.tensor([[0], [7]], dtype=torch.int64)
    else:
        k = 0
    with pytest.raises(RuntimeError, match="requires coalesced inputs"):
        npu_api.spspmm(index_a.npu(), torch.ones(1).npu(), index_b.npu(), torch.ones(1).npu(),
                       m, k, n)


def test_rejects_bad_metadata(npu_api):
    index = torch.tensor([[0], [0]], dtype=torch.int64)
    value = torch.ones(1)
    with pytest.raises(TypeError, match="index_a dtype"):
        npu_api.spspmm(index.int().npu(), value.npu(), index.npu(), value.npu(), 1, 1, 1)
    with pytest.raises(ValueError, match="shape"):
        npu_api.spspmm(index.npu(), value.npu(), index.t().npu(), value.npu(), 1, 1, 1)
    with pytest.raises(ValueError, match="1-D"):
        npu_api.spspmm(index.npu(), value.unsqueeze(0).npu(), index.npu(), value.npu(), 1, 1, 1)
    with pytest.raises(ValueError, match="one weight per edge"):
        npu_api.spspmm(index.npu(), torch.ones(2).npu(), index.npu(), value.npu(), 1, 1, 1)
    with pytest.raises(TypeError, match="float32 or float16"):
        npu_api.spspmm(index.npu(), value.double().npu(), index.npu(), value.double().npu(),
                       1, 1, 1)
    with pytest.raises(RuntimeError, match="Ascend NPU"):
        npu_api.spspmm(index, value, index, value, 1, 1, 1)
    with pytest.raises(TypeError, match="share one dtype"):
        npu_api.spspmm(index.npu(), value.npu(), index.npu(), value.half().npu(), 1, 1, 1)
    with pytest.raises(RuntimeError, match="autograd"):
        npu_api.spspmm(index.npu(), value.npu().requires_grad_(True), index.npu(),
                       value.npu(), 1, 1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        npu_api.spspmm(index.npu(), value.npu(), index.npu(), value.npu(), -1, 1, 1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_non_contiguous_arguments(npu_api, dtype):
    index_a, value_a = _random_operand(12, 8, 30, seed=13, dtype=dtype)
    index_b, value_b = _random_operand(8, 10, 24, seed=17, dtype=dtype)
    index_a, value_a = index_a[:, ::2], value_a[::2]
    index_b, value_b = index_b[:, 1::2], value_b[1::2]
    _check(npu_api, index_a, value_a, index_b, value_b, 12, 8, 10)


@pytest.mark.parametrize("dtype", DTYPES)
def test_k_one_vector_product(npu_api, dtype):
    # k == 1 makes B a single row: the product is a rank-one outer term.
    index_a, value_a = _coo([0, 1, 2], [0, 0, 0], [2.0, -3.0, 0.5], dtype)
    index_b, value_b = _coo([0, 0, 0], [0, 1, 2], [1.5, 2.5, -4.0], dtype)
    _check(npu_api, index_a, value_a, index_b, value_b, 3, 1, 3)

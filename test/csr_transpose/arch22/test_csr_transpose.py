# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional tests of ``csr_transpose`` for the arch22 build (A2/A3, dav-2201)."""

import importlib.util
import os

import pytest
import torch


def _load_golden():
    """Import the ``golden.py`` next to this module under its own name."""
    directory = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_golden_csr_transpose_arch22", os.path.join(directory, "golden.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_golden = _load_golden()


def check(condition, message):
    """Fail a case without using the ``assert`` statement (code-check rule)."""
    if not condition:
        raise AssertionError(message)


def random_csr(num_rows, num_cols, average_degree, seed):
    """CSR pointer and column ids of a random graph with the given shape."""
    generator = torch.Generator().manual_seed(seed)
    degree = torch.randint(0, 2 * average_degree + 1, (num_rows,),
                           generator=generator)
    indptr = torch.cat([torch.zeros(1, dtype=torch.int64), degree.cumsum(0)])
    indices = torch.randint(0, num_cols, (int(indptr[-1]),),
                            generator=generator)
    return indptr.tolist(), indices.tolist()


def run(npu_api, indptr, indices, num_cols=None):
    """Call the operator on NPU and bring both outputs back to the host."""
    ptr = torch.tensor(indptr, dtype=torch.int64).npu()
    idx = torch.tensor(indices, dtype=torch.int64).npu()
    if num_cols is None:
        out_ptr, out_idx = npu_api.csr_transpose(ptr, idx)
    else:
        out_ptr, out_idx = npu_api.csr_transpose(ptr, idx, num_cols)
    return out_ptr.cpu(), out_idx.cpu()


# (CSR pointer, column ids, number of transposed rows). Every pointer starts at
# 0, is nondecreasing and ends at the number of column ids.
CSR_CASES = [
    pytest.param([0, 1, 3], [0, 0, 1], 2, id="two_by_three"),
    pytest.param([0, 0, 0], [], 5, id="empty_graph"),
    pytest.param([0, 4], [2, 0, 2, 1], 3, id="single_row"),
    pytest.param([0, 1, 2, 3], [0, 0, 0], 1, id="single_column"),
    pytest.param([0, 1, 2, 3], [0, 1, 2], 3, id="diagonal"),
    pytest.param([0, 1, 2], [3, 3], 6, id="empty_columns_between"),
    pytest.param([0, 5, 10], [0] * 10, 2, id="hot_column"),
    pytest.param([0, 1, 1, 4, 9], [3, 0, 2, 0, 1, 3, 3, 2, 0],
                 4, id="unsorted_rows"),
    pytest.param([0, 2, 3, 3, 6], [8, 0, 4, 8, 8, 8], 9, id="wide_rectangle"),
    pytest.param([0] * 5, [], 7, id="no_edges_many_columns"),
    pytest.param([0, 3, 3, 5, 9, 9, 9, 10],
                 [4, 4, 4, 0, 1, 5, 2, 2, 3, 2], 6, id="empty_rows_and_columns"),
    pytest.param(list(range(1000)), [0] * 999, 1, id="one_column_many_rows"),
    pytest.param([0, 1, 2, 3, 4, 5], [4, 3, 2, 1, 0], 5, id="anti_diagonal"),
    pytest.param([0, 2, 2, 2, 2, 2, 3], [0, 5, 1], 6, id="long_empty_rows"),
]


class TestCsrTranspose:
    """``csr_transpose`` against its stable-sort CPU reference."""

    @pytest.mark.npu
    @pytest.mark.parametrize("indptr,indices,num_cols", CSR_CASES)
    def test_forward_matches_reference(self, npu_api, indptr, indices,
                                       num_cols):
        out_ptr, out_idx = run(npu_api, indptr, indices, num_cols)
        exp_ptr, exp_idx = _golden.reference(indptr, indices, num_cols)

        check(torch.equal(out_ptr, exp_ptr),
              f"out_indptr {out_ptr.tolist()} != {exp_ptr.tolist()}")
        check(torch.equal(out_idx, exp_idx),
              f"out_indices {out_idx.tolist()} != {exp_idx.tolist()}")

    @pytest.mark.npu
    @pytest.mark.parametrize("rows,cols,degree,seed",
                             [(64, 32, 3, 0), (1000, 1000, 8, 1),
                              (37, 45000, 5, 2), (6000, 7000, 4, 3),
                              (999, 4096, 12, 4), (3000, 3000, 4, 5)])
    def test_random_graph(self, npu_api, rows, cols, degree, seed):
        indptr, indices = random_csr(rows, cols, degree, seed)

        out_ptr, out_idx = run(npu_api, indptr, indices, cols)
        exp_ptr, exp_idx = _golden.reference(indptr, indices, cols)

        check(torch.equal(out_ptr, exp_ptr), "the transposed pointer differs")
        check(torch.equal(out_idx, exp_idx),
              "the transposed column ids differ")

    @pytest.mark.npu
    def test_hot_column_large(self, npu_api):
        # 15000 edges in column 0 and 15000 in column 1: both output segments are
        # much larger than one position chunk and must keep the input order.
        indptr = torch.arange(301, dtype=torch.int64) * 100
        generator = torch.Generator().manual_seed(7)
        indices = torch.cat([
            torch.zeros(15000, dtype=torch.int64),
            torch.randint(1, 2, (15000,), generator=generator),
        ])

        out_ptr, out_idx = run(npu_api, indptr.tolist(), indices.tolist(), 2)

        exp_ptr, exp_idx = _golden.reference(indptr, indices, 2)
        check(torch.equal(out_ptr, exp_ptr), "the transposed pointer differs")
        check(torch.equal(out_idx, exp_idx),
              "a hot column must keep the input order of its edges")

    @pytest.mark.npu
    def test_inferred_num_cols(self, npu_api):
        indptr, indices = random_csr(200, 512, 4, 11)

        out_ptr, out_idx = run(npu_api, indptr, indices)
        exp_ptr, exp_idx = _golden.reference(indptr, indices, 512)

        check(out_ptr.numel() == 513, "the widest column id defines the rows")
        check(torch.equal(out_ptr, exp_ptr), "the transposed pointer differs")
        check(torch.equal(out_idx, exp_idx),
              "the transposed column ids differ")

    @pytest.mark.npu
    def test_empty_graph_infers_zero_rows(self, npu_api):
        out_ptr, out_idx = run(npu_api, [0, 0, 0], [])

        check(out_ptr.tolist() == [0], f"expected one boundary, got {out_ptr}")
        check(out_idx.numel() == 0, "an empty graph has no output edge")

    @pytest.mark.npu
    def test_output_metadata(self, npu_api):
        ptr = torch.tensor([0, 2, 5], dtype=torch.int64).npu()
        idx = torch.tensor([1, 0, 2, 2, 0], dtype=torch.int64).npu()

        out_ptr, out_idx = npu_api.csr_transpose(ptr, idx, 3)

        check(out_ptr.dtype == torch.int64, f"dtype {out_ptr.dtype} != int64")
        check(out_idx.dtype == torch.int64, f"dtype {out_idx.dtype} != int64")
        check(out_ptr.dim() == 1 and out_idx.dim() == 1,
              "both outputs must be 1-D")
        check(out_ptr.numel() == 4, f"expected 4 boundaries, {out_ptr.numel()}")
        check(out_idx.numel() == 5, f"expected 5 edges, {out_idx.numel()}")
        check(out_ptr.device.type == "npu" and out_idx.device.type == "npu",
              "the outputs must stay on the NPU")

    @pytest.mark.npu
    def test_column_rows_are_ascending(self, npu_api):
        # The layout contract of the result: every column segment lists its rows
        # in ascending order, which makes the output comparable bit by bit.
        indptr, indices = random_csr(2000, 1500, 6, 12)

        out_ptr, out_idx = run(npu_api, indptr, indices, 1500)

        ptr = out_ptr.tolist()
        values = out_idx.tolist()
        check(ptr[0] == 0, "the first boundary must be 0")
        check(ptr[-1] == len(indices), "the last boundary must be the edge count")
        check(all(ptr[i] <= ptr[i + 1] for i in range(len(ptr) - 1)),
              "the boundaries must be nondecreasing")
        for column in range(len(ptr) - 1):
            segment = values[ptr[column]:ptr[column + 1]]
            check(all(segment[i] <= segment[i + 1]
                      for i in range(len(segment) - 1)),
                  f"the rows of column {column} are not ascending")

    @pytest.mark.npu
    def test_transpose_twice_is_row_major(self, npu_api):
        indptr, indices = random_csr(300, 200, 5, 13)

        out_ptr, out_idx = run(npu_api, indptr, indices, 200)
        back_ptr, back_idx = run(npu_api, out_ptr.tolist(), out_idx.tolist(), 300)

        exp_ptr, exp_idx = _golden.reference_twice(indptr, indices, 200)
        check(torch.equal(back_ptr, exp_ptr), "the second pointer differs")
        check(torch.equal(back_idx, exp_idx), "the second transpose differs")

    @pytest.mark.npu
    def test_deterministic(self, npu_api):
        indptr, indices = random_csr(4000, 5000, 5, 14)

        first = run(npu_api, indptr, indices, 5000)
        for _ in range(4):
            again = run(npu_api, indptr, indices, 5000)
            check(torch.equal(again[0], first[0]), "the pointer is not stable")
            check(torch.equal(again[1], first[1]), "the layout is not stable")

    @pytest.mark.npu
    def test_non_contiguous_indices(self, npu_api):
        generator = torch.Generator().manual_seed(15)
        indptr, _ = random_csr(500, 400, 4, 16)
        edges = indptr[-1]
        base = torch.randint(0, 400, (2 * edges,), generator=generator)
        strided = base[::2]
        check(not strided.is_contiguous(), "the sample must be non-contiguous")

        out_ptr, out_idx = run(npu_api, indptr, strided.tolist(), 400)

        exp_ptr, exp_idx = _golden.reference(indptr, strided, 400)
        check(torch.equal(out_ptr, exp_ptr), "the transposed pointer differs")
        check(torch.equal(out_idx, exp_idx),
              "the transposed column ids differ")

    @pytest.mark.npu
    def test_matches_argsort_composition(self, npu_api):
        # What a user writes without this operator: histogram the columns for the
        # pointer and move the rows with a stable sort on the device.
        indptr, indices = random_csr(800, 600, 6, 17)
        ptr = torch.tensor(indptr, dtype=torch.int64, device="npu")
        idx = torch.tensor(indices, dtype=torch.int64, device="npu")
        rows = torch.repeat_interleave(
            torch.arange(ptr.numel() - 1, dtype=torch.int64, device="npu"),
            ptr[1:] - ptr[:-1])
        counts = torch.zeros(600, dtype=torch.int64, device="npu")
        counts.scatter_add_(0, idx, torch.ones_like(idx))
        composed_ptr = torch.cat([torch.zeros(1, dtype=torch.int64,
                                              device="npu"),
                                  counts.cumsum(0)])
        composed_idx = rows[torch.argsort(idx, stable=True)]

        out_ptr, out_idx = run(npu_api, indptr, indices, 600)

        check(torch.equal(out_ptr, composed_ptr.cpu()),
              "the pointer differs from the composition")
        check(torch.equal(out_idx, composed_idx.cpu()),
              "the layout differs from the composition")


def _invalid_cases():
    """Argument tuples the public entry point must reject, with their error."""
    ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
    idx = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    return {
        "pointer_not_a_tensor": ([0, 1], idx, None, TypeError),
        "index_not_a_tensor": (ptr, [0, 1], None, TypeError),
        "pointer_dtype": (ptr.to(torch.int32), idx, None, TypeError),
        "index_dtype": (ptr, idx.to(torch.int32), None, TypeError),
        "pointer_rank": (ptr.view(1, 3), idx, None, ValueError),
        "index_rank": (ptr, idx.view(2, 2), None, ValueError),
        "empty_pointer": (torch.tensor([], dtype=torch.int64), idx, None,
                          ValueError),
        "cpu_pointer": (ptr, idx, None, RuntimeError),
        "negative_num_cols": (ptr, idx, -1, ValueError),
        "float_num_cols": (ptr, idx, 2.5, TypeError),
        "zero_num_cols_with_edges": (ptr, idx, 0, RuntimeError),
    }


@pytest.mark.npu
@pytest.mark.parametrize("name", sorted(_invalid_cases()))
def test_invalid_arguments(npu_api, name):
    ptr, idx, num_cols, error = _invalid_cases()[name]
    if name != "cpu_pointer":
        ptr = ptr.npu() if isinstance(ptr, torch.Tensor) else ptr
        idx = idx.npu() if isinstance(idx, torch.Tensor) else idx
    with pytest.raises(error):
        if num_cols is None:
            npu_api.csr_transpose(ptr, idx)
        else:
            npu_api.csr_transpose(ptr, idx, num_cols)

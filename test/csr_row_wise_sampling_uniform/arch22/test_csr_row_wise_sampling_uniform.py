# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for csr_row_wise_sampling_uniform on Ascend A2/A3."""
import os
import random

import pytest
import torch


def expected_counts(indptr, indices, rows, num_samples, replace):
    """Exact per-row pick counts implied by the DGL semantics."""
    ptr = indptr.tolist()
    counts = []
    for rid in rows.tolist():
        if rid < 0 or rid >= len(ptr) - 1:
            counts.append(0)
            continue
        degree = ptr[rid + 1] - ptr[rid]
        if num_samples == -1:
            counts.append(degree)
        elif replace:
            counts.append(num_samples if degree > 0 else 0)
        else:
            counts.append(min(num_samples, degree))
    return counts


def reference_sample(indptr, indices, rows, num_samples, replace, data=None, seed=0):
    """Independent Python reference (own RNG; pick values are not bit-exact)."""
    rng = random.Random(seed)
    ptr = indptr.tolist()
    idx = indices.tolist()
    dat = None if data is None else data.tolist()
    sampled = []
    for rid in rows.tolist():
        row_picks = []
        if 0 <= rid < len(ptr) - 1:
            begin, end = ptr[rid], ptr[rid + 1]
            degree = end - begin
            count = expected_counts(indptr, indices, [rid], num_samples, replace)[0]
            if num_samples == -1 or (not replace and count == degree):
                local = list(range(degree))
            elif replace:
                local = [rng.randrange(degree) for _ in range(count)]
            else:
                local = rng.sample(range(degree), count)
            for pos in local:
                edge = begin + pos
                row_picks.append((rid, idx[edge], dat[edge] if dat is not None else edge))
        sampled.append(row_picks)
    return sampled


def check_result(out_rows, out_cols, out_data, out_ptr, indptr, indices, rows,
                 num_samples, replace, data=None):
    """Validate every semantic invariant that does not depend on the RNG."""
    row = out_rows.cpu().tolist()
    col = out_cols.cpu().tolist()
    dat = out_data.cpu().tolist()
    ptr = out_ptr.cpu().tolist()
    csr_ptr = indptr.cpu().tolist()
    csr_idx = indices.cpu().tolist()
    csr_dat = None if data is None else data.cpu().tolist()
    seed_rows = rows.cpu().tolist()
    counts = expected_counts(indptr, indices, rows, num_samples, replace)
    assert len(ptr) == len(seed_rows) + 1
    assert ptr[0] == 0 and ptr[-1] == len(row) == len(col) == len(dat)
    for i, rid in enumerate(seed_rows):
        expected = counts[i]
        assert ptr[i + 1] - ptr[i] == expected, (i, rid, ptr[i + 1] - ptr[i], expected)
        picked_cols = col[ptr[i]:ptr[i + 1]]
        if 0 <= rid < len(csr_ptr) - 1:
            allowed = csr_idx[csr_ptr[rid]:csr_ptr[rid + 1]]
            assert set(picked_cols).issubset(set(allowed)), (i, picked_cols, allowed)
            if not replace and csr_dat is None:
                picked_edges = dat[ptr[i]:ptr[i + 1]]
                assert len(set(picked_edges)) == len(picked_edges), (i, picked_edges)
        else:
            assert not picked_cols
        for pos in range(ptr[i], ptr[i + 1]):
            assert row[pos] == rid
            if csr_dat is None:
                edge = dat[pos]
                assert csr_ptr[rid] <= edge < csr_ptr[rid + 1]
                assert csr_idx[edge] == col[pos]
            else:
                assert dat[pos] in csr_dat[csr_ptr[rid]:csr_ptr[rid + 1]]
    return counts



@pytest.fixture
def op():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import csr_row_wise_sampling_uniform
    return csr_row_wise_sampling_uniform


def make_csr(dtype=torch.int32, degrees=(3, 0, 5, 1), seed=7):
    gen = torch.Generator().manual_seed(seed)
    ptr = [0]
    for degree in degrees:
        ptr.append(ptr[-1] + degree)
    idx = [int(torch.randint(0, 40, (1,), generator=gen)) for _ in range(ptr[-1])]
    values = [edge + 100 for edge in range(ptr[-1])]
    return (torch.tensor(ptr, dtype=dtype), torch.tensor(idx, dtype=dtype),
            torch.tensor(values, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_samples,replace", [(-1, False), (0, False), (1, False),
                                                  (3, False), (3, True), (8, False)])
def test_sampling_modes(op, dtype, num_samples, replace):
    ptr, idx, val = make_csr(dtype)
    rows = torch.tensor([0, 1, 2, 3, 2, 0], dtype=dtype)
    result = op(ptr.npu(), idx.npu(), rows.npu(), num_samples, replace, seed=11)
    check_result(*result, ptr, idx, rows, num_samples, replace)
    assert result[0].dtype == dtype


def test_with_data(op):
    ptr, idx, val = make_csr()
    rows = torch.tensor([0, 2, 3], dtype=ptr.dtype)
    result = op(ptr.npu(), idx.npu(), rows.npu(), 2, False, val.npu(), seed=1)
    check_result(*result, ptr, idx, rows, 2, False, val)


def test_invalid_rows_are_empty(op):
    ptr, idx, _ = make_csr()
    rows = torch.tensor([-1, 99, 1, 4], dtype=torch.int32)
    result = op(ptr.npu(), idx.npu(), rows.npu(), -1, False, seed=2)
    counts = check_result(*result, ptr, idx, rows, -1, False)
    assert counts == [0, 0, 0, 0]


def test_empty_rows_tensor(op):
    ptr, idx, _ = make_csr()
    rows = torch.tensor([], dtype=ptr.dtype)
    row, col, data, out_ptr = op(ptr.npu(), idx.npu(), rows.npu(), 4, True, seed=3)
    assert row.numel() == col.numel() == data.numel() == 0
    assert out_ptr.numel() == 1 and out_ptr.item() == 0


def test_max_output_and_large_degree(op):
    degrees = (4096, 4, 0, 512)
    ptr, idx, _ = make_csr(degrees=degrees, seed=9)
    rows = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    for num_samples, replace in [(-1, False), (100, False), (100, True)]:
        result = op(ptr.npu(), idx.npu(), rows.npu(), num_samples, replace, seed=4)
        check_result(*result, ptr, idx, rows, num_samples, replace)


def test_non_contiguous_inputs(op):
    ptr, idx, _ = make_csr()
    rows = torch.tensor([2, 0, 1, 3, 0], dtype=torch.int32)[::2]
    assert not rows.is_contiguous()
    result = op(ptr.npu(), idx.npu(), rows.npu(), 2, False, seed=5)
    check_result(*result, ptr, idx, rows, 2, False)


def test_seed_is_reproducible(op):
    ptr, idx, _ = make_csr(degrees=(20, 20, 20))
    rows = torch.tensor([0, 1, 2], dtype=torch.int32)
    first = op(ptr.npu(), idx.npu(), rows.npu(), 5, False, seed=123)
    second = op(ptr.npu(), idx.npu(), rows.npu(), 5, False, seed=123)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_seed_changes_samples(op):
    ptr, idx, _ = make_csr(degrees=(32, 32))
    rows = torch.tensor([0, 1], dtype=torch.int32)
    first = op(ptr.npu(), idx.npu(), rows.npu(), 4, False, seed=1)
    second = op(ptr.npu(), idx.npu(), rows.npu(), 4, False, seed=2)
    assert not torch.equal(first[1], second[1])


def test_sampling_covers_all_neighbors(op):
    ptr, idx, _ = make_csr(degrees=(12, 0, 7))
    rows = torch.tensor([0, 1, 2], dtype=torch.int32)
    row, col, data, out_ptr = op(ptr.npu(), idx.npu(), rows.npu(), -1, False, seed=17)
    assert out_ptr.tolist() == [0, 12, 12, 19]
    assert col[:12].tolist() == idx[:12].tolist()
    assert data[:12].tolist() == list(range(12))


@pytest.mark.parametrize("kwargs", [
    {"num_samples": -2},
    {"num_samples": True},
])
def test_bad_arguments(op, kwargs):
    ptr, idx, _ = make_csr()
    rows = torch.tensor([0])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        op(ptr.npu(), idx.npu(), rows.npu(), seed=1, **kwargs)


def test_dtype_mismatch_rejected(op):
    ptr, idx, _ = make_csr()
    rows = torch.tensor([0], dtype=torch.int64)
    with pytest.raises((ValueError, RuntimeError)):
        op(ptr.npu(), idx.npu(), rows.npu(), 2, False, seed=1)

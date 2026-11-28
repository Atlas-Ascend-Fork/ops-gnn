# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for csr_row_wise_per_etype_sampling_uniform on Ascend A2/A3."""
import bisect
import os

import pytest
import torch


def etype_of(eid, offsets):
    return bisect.bisect_right(offsets, eid) - 1


def expected_counts(row, rows, offsets, num_samples, replace):
    """Expected pick counts keyed by (etype, rid)."""
    row = row.tolist()
    counts = {}
    num_etypes = len(offsets) - 1
    for rid in rows.tolist():
        for e in range(num_etypes):
            begin, end = offsets[e], offsets[e + 1]
            degree = sum(1 for x in row[begin:end] if x == rid)
            if degree == 0 or num_samples[e] == 0:
                count = 0
            elif num_samples[e] == -1:
                count = degree
            elif replace:
                count = num_samples[e]
            else:
                count = min(num_samples[e], degree)
            counts[(e, rid)] = count
    return counts


def check_result(out_rows, out_cols, out_data, row, col, rows, offsets,
                 num_samples, replace, data=None):
    r = out_rows.cpu().tolist()
    c = out_cols.cpu().tolist()
    d = out_data.cpu().tolist()
    coo_row = row.cpu().tolist()
    coo_col = col.cpu().tolist()
    coo_data = None if data is None else data.cpu().tolist()
    seed_rows = set(rows.cpu().tolist())
    counts = expected_counts(row, rows, offsets, num_samples, replace)
    seen = {}
    for rr, cc, dd in zip(r, c, d):
        assert rr in seed_rows, (rr, seed_rows)
        if coo_data is None:
            assert 0 <= dd < len(coo_row), dd
            assert coo_row[dd] == rr and coo_col[dd] == cc, (rr, cc, dd, coo_row[dd], coo_col[dd])
            key = (etype_of(dd, offsets), rr)
            if not replace:
                assert dd not in seen.setdefault(key, set()), (key, dd)
                seen[key].add(dd)
            counts[key] -= 1
        else:
            found = any(coo_row[e] == rr and coo_col[e] == cc and coo_data[e] == dd
                        for e in range(len(coo_row)))
            assert found, (rr, cc, dd)
    if coo_data is None:
        for key, remaining in counts.items():
            assert remaining == 0, (key, remaining)
    else:
        # Without edge ids we cannot attribute edges to etypes; check the
        # per-row total across all edge types instead.
        totals = {}
        for (_, rid), count in counts.items():
            totals[rid] = totals.get(rid, 0) + count
        actual = {}
        for rr in r:
            actual[rr] = actual.get(rr, 0) + 1
        for rid in seed_rows:
            assert actual.get(rid, 0) == totals.get(rid, 0), (rid, actual.get(rid, 0), totals.get(rid, 0))
    return counts



@pytest.fixture
def op():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import csr_row_wise_per_etype_sampling_uniform
    return csr_row_wise_per_etype_sampling_uniform


def make_coo(dtype=torch.int32):
    etypes = [
        ([0, 0, 0, 1, 2, 2], [1, 2, 3, 3, 0, 4]),
        ([0, 1, 1], [2, 4, 5]),
        ([2, 2, 2, 3], [1, 2, 3, 4]),
    ]
    row = [x for src, _ in etypes for x in src]
    col = [x for _, dst in etypes for x in dst]
    offsets = [0]
    for src, _ in etypes:
        offsets.append(offsets[-1] + len(src))
    data = [100 + i for i in range(len(row))]
    return (torch.tensor(row, dtype=dtype), torch.tensor(col, dtype=dtype),
            torch.tensor(data, dtype=dtype), offsets)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("samples,replace", [([2, -1, 0], False), ([1, 2, 3], False),
                                              ([2, 2, 2], True), ([-1, -1, -1], False),
                                              ([0, 0, 0], False)])
def test_sampling_modes(op, dtype, samples, replace):
    row, col, _, offsets = make_coo(dtype)
    rows = torch.tensor([0, 1, 2, 3], dtype=dtype)
    out = op(row.npu(), col.npu(), rows.npu(), offsets, samples, replace, seed=11)
    check_result(*out, row, col, rows, offsets, samples, replace)
    assert out[0].dtype == dtype


def test_single_fanout_broadcast(op):
    row, col, _, offsets = make_coo()
    rows = torch.tensor([0, 1, 2], dtype=torch.int32)
    out = op(row.npu(), col.npu(), rows.npu(), offsets, 2, seed=1)
    check_result(*out, row, col, rows, offsets, [2, 2, 2], False)


def test_with_data(op):
    row, col, data, offsets = make_coo()
    rows = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    out = op(row.npu(), col.npu(), rows.npu(), offsets, [2, 1, -1], False,
             data.npu(), seed=2)
    check_result(*out, row, col, rows, offsets, [2, 1, -1], False, data)


def test_empty_etype_window(op):
    row, col, _, _ = make_coo()
    offsets = [0, 6, 6, 13]
    rows = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    out = op(row.npu(), col.npu(), rows.npu(), offsets, [2, 3, 1], False, seed=3)
    check_result(*out, row, col, rows, offsets, [2, 3, 1], False)


def test_invalid_rows_are_empty(op):
    row, col, _, offsets = make_coo()
    rows = torch.tensor([-1, 99, 0, 2], dtype=torch.int32)
    out = op(row.npu(), col.npu(), rows.npu(), offsets, [2, 2, 2], False, seed=4)
    check_result(*out, row, col, rows, offsets, [2, 2, 2], False)


def test_empty_rows_tensor(op):
    row, col, _, offsets = make_coo()
    rows = torch.tensor([], dtype=torch.int32)
    out = op(row.npu(), col.npu(), rows.npu(), offsets, [2, 2, 2], False, seed=5)
    assert all(t.numel() == 0 for t in out)


def test_empty_coo(op):
    row = torch.tensor([], dtype=torch.int32)
    col = torch.tensor([], dtype=torch.int32)
    rows = torch.tensor([0, 1], dtype=torch.int32)
    out = op(row.npu(), col.npu(), rows.npu(), [0, 0], [2], False, seed=6)
    assert all(t.numel() == 0 for t in out)


def test_seed_is_reproducible(op):
    row, col, _, offsets = make_coo()
    rows = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    first = op(row.npu(), col.npu(), rows.npu(), offsets, [2, 2, 2], False, seed=77)
    second = op(row.npu(), col.npu(), rows.npu(), offsets, [2, 2, 2], False, seed=77)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_bad_offsets(op):
    row, col, _, offsets = make_coo()
    rows = torch.tensor([0], dtype=torch.int32)
    with pytest.raises((ValueError, RuntimeError)):
        op(row.npu(), col.npu(), rows.npu(), [1] + offsets[1:], [2, 2, 2], seed=1)
    with pytest.raises((ValueError, RuntimeError)):
        op(row.npu(), col.npu(), rows.npu(), offsets[:-1] + [999], [2, 2, 2], seed=1)
    with pytest.raises((ValueError, RuntimeError)):
        op(row.npu(), col.npu(), rows.npu(), offsets, [2, 2], seed=1)


def test_dtype_mismatch_rejected(op):
    row, col, _, offsets = make_coo()
    rows = torch.tensor([0], dtype=torch.int64)
    with pytest.raises((ValueError, RuntimeError)):
        op(row.npu(), col.npu(), rows.npu(), offsets, [2, 2, 2], seed=1)

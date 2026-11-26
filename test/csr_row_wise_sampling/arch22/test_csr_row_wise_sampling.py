# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for csr_row_wise_sampling on Ascend A2/A3."""
import math
import os

import pytest
import torch


def expected_counts(indptr, rows, prob, num_samples, replace):
    ptr = indptr.tolist()
    weights = prob.tolist()
    counts = {}
    for rid in rows.tolist():
        if rid < 0 or rid >= len(ptr) - 1:
            counts[rid] = 0
            continue
        begin, end = ptr[rid], ptr[rid + 1]
        positive = sum(1 for e in range(begin, end)
                       if math.isfinite(weights[e]) and weights[e] > 0.0)
        if num_samples == -1:
            count = positive
        elif num_samples == 0:
            count = 0
        elif replace:
            count = num_samples if positive > 0 else 0
        else:
            count = min(num_samples, positive)
        counts[rid] = count
    return counts


def check_result(out_rows, out_cols, out_data, indptr, indices, rows, prob,
                 num_samples, replace, data=None):
    r = out_rows.cpu().tolist()
    c = out_cols.cpu().tolist()
    d = out_data.cpu().tolist()
    csr_ptr = indptr.cpu().tolist()
    csr_idx = indices.cpu().tolist()
    csr_dat = None if data is None else data.cpu().tolist()
    weights = prob.cpu().tolist()
    seed_rows = rows.cpu().tolist()
    counts = expected_counts(indptr, rows, prob, num_samples, replace)
    actual = {}
    for rr, _, _ in zip(r, c, d):
        actual[rr] = actual.get(rr, 0) + 1
    for rid, expected in counts.items():
        assert actual.get(rid, 0) == expected, (rid, actual.get(rid, 0), expected)
    for rr in actual:
        assert rr in counts, rr
    # rows are emitted in CSR order, so out_rows is nondecreasing
    assert all(a <= b for a, b in zip(r, r[1:])), r

    seen = {}
    for rr, cc, dd in zip(r, c, d):
        assert 0 <= rr < len(csr_ptr) - 1
        begin, end = csr_ptr[rr], csr_ptr[rr + 1]
        allowed_cols = csr_idx[begin:end]
        assert cc in allowed_cols, (rr, cc, allowed_cols)
        if csr_dat is None:
            assert begin <= dd < end, (rr, dd, begin, end)
            assert csr_idx[dd] == cc, (rr, cc, dd, csr_idx[dd])
            assert math.isfinite(weights[dd]) and weights[dd] > 0.0, (rr, dd)
            if not replace:
                key = rr
                assert dd not in seen.setdefault(key, set()), (rr, dd)
                seen[key].add(dd)
        else:
            positive_vals = [csr_dat[e] for e in range(begin, end)
                             if math.isfinite(weights[e]) and weights[e] > 0.0]
            assert dd in positive_vals, (rr, cc, dd, positive_vals)
    return counts



@pytest.fixture
def op():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import csr_row_wise_sampling
    return csr_row_wise_sampling


def make_csr(dtype=torch.int32, degrees=(4, 0, 6, 3), seed=7):
    gen = torch.Generator().manual_seed(seed)
    ptr = [0]
    for degree in degrees:
        ptr.append(ptr[-1] + degree)
    idx = [int(torch.randint(0, 40, (1,), generator=gen)) for _ in range(ptr[-1])]
    values = list(range(ptr[-1]))
    prob = torch.rand(ptr[-1], generator=gen)
    prob[::3] = 0.0  # some edges are unreachable
    return (torch.tensor(ptr, dtype=dtype), torch.tensor(idx, dtype=dtype),
            torch.tensor(values, dtype=dtype), prob)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_samples,replace", [(-1, False), (0, False), (1, False),
                                                  (3, False), (3, True), (8, False)])
def test_sampling_modes(op, dtype, num_samples, replace):
    ptr, idx, _, prob = make_csr(dtype)
    rows = torch.tensor([0, 1, 2, 3], dtype=dtype)
    out = op(ptr.npu(), idx.npu(), rows.npu(), num_samples, prob.npu(), replace, seed=11)
    check_result(*out, ptr, idx, rows, prob, num_samples, replace)
    assert out[0].dtype == dtype


def test_with_data(op):
    ptr, idx, val, prob = make_csr()
    rows = torch.tensor([0, 2, 3], dtype=ptr.dtype)
    out = op(ptr.npu(), idx.npu(), rows.npu(), 2, prob.npu(), False, val.npu(), seed=1)
    check_result(*out, ptr, idx, rows, prob, 2, False, val)


def test_invalid_rows_are_empty(op):
    ptr, idx, _, prob = make_csr()
    rows = torch.tensor([-1, 99, 1, 4], dtype=torch.int32)
    out = op(ptr.npu(), idx.npu(), rows.npu(), -1, prob.npu(), False, seed=2)
    counts = check_result(*out, ptr, idx, rows, prob, -1, False)
    assert counts == {-1: 0, 99: 0, 1: 0, 4: 0}


def test_empty_rows_tensor(op):
    ptr, idx, _, prob = make_csr()
    rows = torch.tensor([], dtype=ptr.dtype)
    out = op(ptr.npu(), idx.npu(), rows.npu(), 4, prob.npu(), True, seed=3)
    assert all(t.numel() == 0 for t in out)


def test_zero_prob_edges_never_picked(op):
    ptr, idx, _, _ = make_csr(degrees=(5, 0, 3))
    prob = torch.zeros(ptr[-1].item())
    prob[1] = 1.0  # only global edge 1 is reachable
    rows = torch.tensor([0, 1, 2], dtype=torch.int32)
    for num_samples, replace in [(-1, False), (4, False), (4, True)]:
        out = op(ptr.npu(), idx.npu(), rows.npu(), num_samples, prob.npu(),
                 replace, seed=4)
        check_result(*out, ptr, idx, rows, prob, num_samples, replace)
        assert set(out[2].cpu().tolist()) <= {1}


def test_mask_int8_and_uint8(op):
    ptr, idx, _, prob = make_csr()
    mask = (prob > 0.5).to(torch.int8)
    rows = torch.tensor([0, 2, 3], dtype=torch.int32)
    out = op(ptr.npu(), idx.npu(), rows.npu(), -1, mask.npu(), False, seed=5)
    check_result(*out, ptr, idx, rows, mask.float(), -1, False)
    umask = mask.to(torch.uint8)
    out = op(ptr.npu(), idx.npu(), rows.npu(), 2, umask.npu(), False, seed=5)
    check_result(*out, ptr, idx, rows, umask.float(), 2, False)


def test_float64_prob(op):
    ptr, idx, _, prob = make_csr()
    prob64 = prob.to(torch.float64)
    rows = torch.tensor([0, 2, 3], dtype=torch.int32)
    out = op(ptr.npu(), idx.npu(), rows.npu(), 2, prob64.npu(), False, seed=6)
    check_result(*out, ptr, idx, rows, prob, 2, False)


def test_large_degree(op):
    ptr, idx, _, prob = make_csr(degrees=(4096, 4, 0, 512), seed=9)
    rows = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    for num_samples, replace in [(-1, False), (100, False), (100, True)]:
        out = op(ptr.npu(), idx.npu(), rows.npu(), num_samples, prob.npu(),
                 replace, seed=7)
        check_result(*out, ptr, idx, rows, prob, num_samples, replace)


def test_seed_is_reproducible(op):
    ptr, idx, _, prob = make_csr(degrees=(20, 20, 20))
    rows = torch.tensor([0, 1, 2], dtype=torch.int32)
    first = op(ptr.npu(), idx.npu(), rows.npu(), 5, prob.npu(), True, seed=123)
    second = op(ptr.npu(), idx.npu(), rows.npu(), 5, prob.npu(), True, seed=123)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_bad_prob(op):
    ptr, idx, _, prob = make_csr()
    rows = torch.tensor([0], dtype=torch.int32)
    with pytest.raises((ValueError, RuntimeError)):
        op(ptr.npu(), idx.npu(), rows.npu(), 2, prob.to(torch.float16).npu(), seed=1)
    with pytest.raises((ValueError, RuntimeError)):
        op(ptr.npu(), idx.npu(), rows.npu(), 2, prob[:-1].npu(), seed=1)


def test_dtype_mismatch_rejected(op):
    ptr, idx, _, prob = make_csr()
    rows = torch.tensor([0], dtype=torch.int64)
    with pytest.raises((ValueError, RuntimeError)):
        op(ptr.npu(), idx.npu(), rows.npu(), 2, prob.npu(), seed=1)


@pytest.mark.parametrize("num_samples,replace", [(-1, False), (3, False), (3, True)])
def test_non_finite_probabilities_never_picked(op, num_samples, replace):
    ptr = torch.tensor([0, 4, 4], dtype=torch.int32)
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    prob = torch.tensor([float("nan"), float("inf"), -1.0, 2.0])
    rows = torch.tensor([0, 1], dtype=torch.int32)
    out = op(ptr.npu(), idx.npu(), rows.npu(), num_samples, prob.npu(),
             replace, seed=12)
    check_result(*out, ptr, idx, rows, prob, num_samples, replace)
    assert set(out[2].cpu().tolist()) <= {3}

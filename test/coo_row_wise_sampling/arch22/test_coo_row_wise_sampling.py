# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for coo_row_wise_sampling on Ascend A2/A3."""
import os

import pytest
import torch


@pytest.fixture
def ops():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import coo_row_wise_sampling, csr_row_wise_sampling
    return coo_row_wise_sampling, csr_row_wise_sampling


def make_coo(dtype=torch.int32, num_nodes=40, nnz=60, seed=7):
    gen = torch.Generator().manual_seed(seed)
    row = torch.randint(0, num_nodes, (nnz,), generator=gen).to(dtype)
    col = torch.randint(0, num_nodes, (nnz,), generator=gen).to(dtype)
    prob = torch.rand(nnz, generator=gen)
    prob[::4] = 0.0
    perm = torch.randperm(nnz, generator=gen)
    return row[perm], col[perm], prob[perm]


def build_csr(row, col, num_rows):
    perm = torch.argsort(row, stable=True)
    sr = row[perm]
    counts = torch.zeros(num_rows, dtype=torch.int64)
    counts.scatter_add_(0, sr.long(), torch.ones_like(sr, dtype=torch.int64))
    ptr = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to(row.dtype)
    return ptr, col[perm], perm.to(row.dtype)


def infer_num_rows(row, rows):
    n = 0
    if row.numel():
        n = max(n, int(row.max()) + 1)
    if rows.numel():
        n = max(n, int(rows.max()) + 1)
    return n


def as_sets(r, c, d):
    out = {}
    for rr, cc, dd in zip(r.tolist(), c.tolist(), d.tolist()):
        out.setdefault(rr, set()).add((cc, dd))
    return out


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_samples,replace", [(-1, False), (0, False), (1, False),
                                                  (3, False), (3, True), (10, False)])
def test_weighted_modes(ops, dtype, num_samples, replace):
    coo_op, csr_op = ops
    row, col, prob = make_coo(dtype)
    rows = torch.tensor([0, 5, 9, 39], dtype=dtype)
    num_rows = infer_num_rows(row, rows)
    ptr, indices, eid = build_csr(row, col, num_rows)
    out = coo_op(row.npu(), col.npu(), rows.npu(), num_samples, prob.npu(), replace, seed=11)
    ref = csr_op(ptr.npu(), indices.npu(), rows.npu(), num_samples, prob.npu(),
                 replace, eid.npu(), seed=11)
    assert as_sets(*out) == as_sets(ref[0], ref[1], ref[2])
    assert out[0].dtype == dtype


def test_mask_and_float64(ops):
    coo_op, csr_op = ops
    row, col, prob = make_coo()
    rows = torch.tensor([0, 5, 9], dtype=torch.int32)
    num_rows = infer_num_rows(row, rows)
    ptr, indices, eid = build_csr(row, col, num_rows)
    for p in [(prob > 0.5).to(torch.int8), prob.to(torch.float64)]:
        out = coo_op(row.npu(), col.npu(), rows.npu(), -1, p.npu(), False, seed=3)
        ref = csr_op(ptr.npu(), indices.npu(), rows.npu(), -1, p.npu(), False, eid.npu(), seed=3)
        assert as_sets(*out) == as_sets(ref[0], ref[1], ref[2])


def test_invalid_rows_are_empty(ops):
    coo_op, _ = ops
    row, col, prob = make_coo()
    rows = torch.tensor([-1, 999], dtype=torch.int32)
    out = coo_op(row.npu(), col.npu(), rows.npu(), -1, prob.npu(), seed=2)
    assert all(t.numel() == 0 for t in out)


def test_empty_inputs(ops):
    coo_op, _ = ops
    empty = torch.empty(0, dtype=torch.int32)
    rows = torch.tensor([0, 1], dtype=torch.int32)
    prob = torch.empty(0, dtype=torch.float32)
    out = coo_op(empty.npu(), empty.npu(), rows.npu(), 4, prob.npu(), seed=1)
    assert all(t.numel() == 0 for t in out)


def test_bad_prob_length(ops):
    coo_op, _ = ops
    row, col, prob = make_coo()
    rows = torch.tensor([0], dtype=torch.int32)
    with pytest.raises((ValueError, RuntimeError)):
        coo_op(row.npu(), col.npu(), rows.npu(), 2, prob[:-1].npu(), seed=1)

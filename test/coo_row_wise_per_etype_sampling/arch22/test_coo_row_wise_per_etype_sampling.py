# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for coo_row_wise_per_etype_sampling on Ascend A2/A3."""
import os

import pytest
import torch


@pytest.fixture
def ops():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    from ops_gnn import coo_row_wise_per_etype_sampling, csr_row_wise_sampling
    return coo_row_wise_per_etype_sampling, csr_row_wise_sampling


def make_coo(dtype=torch.int32, num_nodes=30, lengths=(20, 15, 25), seed=7):
    gen = torch.Generator().manual_seed(seed)
    rows, cols, offsets = [], [], [0]
    for length in lengths:
        rows.append(torch.randint(0, num_nodes, (length,), generator=gen).to(dtype))
        cols.append(torch.randint(0, num_nodes, (length,), generator=gen).to(dtype))
        offsets.append(offsets[-1] + length)
    row = torch.cat(rows)
    col = torch.cat(cols)
    return row, col, offsets


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


def reference(csr_op, row, col, rows, offsets, samples, probs, replace, seed, data):
    num_rows = infer_num_rows(row, rows)
    base = seed
    parts = []
    for e in range(len(offsets) - 1):
        off, length = offsets[e], offsets[e + 1] - offsets[e]
        if length == 0 or samples[e] == 0:
            continue
        ptr, indices, local_eid = build_csr(row[off:off + length], col[off:off + length], num_rows)
        weight = probs[e]
        if weight is None:
            weight = torch.ones(length, dtype=torch.float32)
        pr, pc, pl = csr_op(ptr.npu(), indices.npu(), rows.npu(), samples[e], weight.npu(),
                            replace, local_eid.npu(), (base + e) % (2 ** 32))
        if pr.numel() == 0:
            continue
        ge = pl.cpu() + off
        parts.append((pr.cpu(), pc.cpu(), ge if data is None else data[ge]))
    if not parts:
        empty = torch.empty(0, dtype=row.dtype)
        return empty, empty, empty
    return tuple(torch.cat(x) for x in zip(*parts))


def test_per_etype_modes(ops):
    coo_op, csr_op = ops
    row, col, offsets = make_coo()
    gen = torch.Generator().manual_seed(3)
    probs = [torch.rand(offsets[e + 1] - offsets[e], generator=gen) for e in range(3)]
    probs[1] = None
    rows = torch.tensor([0, 5, 9], dtype=torch.int32)
    for samples, replace in [([-1, -1, -1], False), ([2, 0, 3], False),
                             ([2, 3, 4], True), ([5, 5, 5], False)]:
        out = coo_op(row.npu(), col.npu(), rows.npu(), offsets, samples,
                     [None if p is None else p.npu() for p in probs], replace, seed=17)
        ref = reference(csr_op, row, col, rows, offsets, samples, probs, replace, 17, None)
        assert as_sets(*out) == as_sets(*ref)


def test_per_etype_with_data(ops):
    coo_op, csr_op = ops
    row, col, offsets = make_coo()
    gen = torch.Generator().manual_seed(4)
    probs = [torch.rand(offsets[e + 1] - offsets[e], generator=gen) for e in range(3)]
    values = torch.arange(row.numel(), dtype=row.dtype) * 2 + 1
    rows = torch.tensor([0, 5, 9], dtype=torch.int32)
    samples = [2, 3, 4]
    out = coo_op(row.npu(), col.npu(), rows.npu(), offsets, samples,
                 [p.npu() for p in probs], False, values.npu(), seed=21)
    ref = reference(csr_op, row, col, rows, offsets, samples, probs, False, 21, values)
    assert as_sets(*out) == as_sets(*ref)


def test_invalid_rows_are_empty(ops):
    coo_op, _ = ops
    row, col, offsets = make_coo()
    rows = torch.tensor([-1, 999], dtype=torch.int32)
    out = coo_op(row.npu(), col.npu(), rows.npu(), offsets, -1, seed=2)
    assert all(t.numel() == 0 for t in out)


def test_bad_offsets(ops):
    coo_op, _ = ops
    row, col, offsets = make_coo()
    rows = torch.tensor([0], dtype=torch.int32)
    with pytest.raises((ValueError, RuntimeError)):
        coo_op(row.npu(), col.npu(), rows.npu(), offsets[:-1] + [offsets[-1] - 1], 2)
    with pytest.raises((ValueError, RuntimeError)):
        coo_op(row.npu(), col.npu(), rows.npu(), offsets, [1, 2], seed=1)

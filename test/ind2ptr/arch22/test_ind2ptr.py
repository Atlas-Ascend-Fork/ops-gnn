# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional tests of the native arch22 ``ind2ptr`` kernel (A2/A3)."""

import os
from bisect import bisect_left

import pytest
import torch

import ops_gnn
from ops_gnn import _pybind


@pytest.fixture(autouse=True)
def npu_device():
    torch.npu.set_device(int(os.environ.get('NPU_DEVICE_ID', '0')))


def _cpu_golden(values, num_rows):
    ordered = values.tolist()
    return torch.tensor(
        [bisect_left(ordered, row) for row in range(num_rows + 1)],
        dtype=torch.int64,
    )


def _assert_exact(got, expected):
    if got.device.type != 'npu' or got.dtype != torch.int64:
        raise AssertionError(f'ind2ptr returned {got.device}/{got.dtype}')
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('num_rows,values', [
    (0, []),
    (1, [0]),
    (5, [0, 0, 0, 4, 4]),
    (8, [2, 2, 4, 5, 5, 6]),
    (64, list(range(64))),
    (1024, [0] * 2048),
])
def test_ind2ptr_matches_cpu_golden(num_rows, values):
    cpu = torch.tensor(values, dtype=torch.int64)
    expected = _cpu_golden(cpu, num_rows)
    _assert_exact(ops_gnn.ind2ptr(cpu.npu(), num_rows), expected)


@pytest.mark.parametrize('num_rows', [0, 1, 2, 8, 64, 1024, 2048, 2049])
def test_ind2ptr_empty_various_num_rows(num_rows):
    ind = torch.empty(0, dtype=torch.int64, device='npu')
    expected = torch.zeros(num_rows + 1, dtype=torch.int64)
    _assert_exact(ops_gnn.ind2ptr(ind, num_rows), expected)
    # The C++ host short-circuits empty input before launching the kernel, so
    # exercise that path directly as well.
    _assert_exact(_pybind.ind2ptr(ind, num_rows), expected)


@pytest.mark.parametrize('num_rows,divisor', [
    (2049, 1),
    (4097, 1),
    (65536, 4),
    (262144, 4),
    (262145, 8),
])
def test_ind2ptr_chunk_boundaries(num_rows, divisor):
    # These shapes force each AIV to own more than one 2048-row output chunk.
    values = torch.arange(num_rows, dtype=torch.int64) // divisor
    expected = _cpu_golden(values, num_rows)
    _assert_exact(ops_gnn.ind2ptr(values.npu(), num_rows), expected)


def test_ind2ptr_large_multi_core():
    num_rows = 100000
    generator = torch.Generator().manual_seed(7)
    counts = torch.randint(0, 4, (num_rows,), dtype=torch.int64,
                           generator=generator)
    values = torch.arange(num_rows, dtype=torch.int64).repeat_interleave(counts)
    expected = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    _assert_exact(ops_gnn.ind2ptr(values.npu(), num_rows), expected)


@pytest.mark.parametrize('num_rows,values', [
    (8, [2, 2, 4, 5, 5, 6]),
    (64, list(range(64))),
    (2048, [0] * 4096),
])
def test_ind2ptr_noncontiguous(num_rows, values):
    cpu = torch.tensor(values, dtype=torch.int64)
    expected = _cpu_golden(cpu, num_rows)
    ind = torch.stack((cpu, cpu), dim=-1).npu()[:, 0]
    if ind.is_contiguous():
        raise AssertionError('expected non-contiguous view')
    _assert_exact(ops_gnn.ind2ptr(ind, num_rows), expected)


def test_ind2ptr_strided_and_stream():
    cpu = torch.tensor([1, 1, 3, 3, 3, 7], dtype=torch.int64)
    expected = _cpu_golden(cpu, 8)
    ind = torch.stack((cpu, cpu), dim=-1).npu()[:, 0]
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        got = ops_gnn.ind2ptr(ind, 8)
    stream.synchronize()
    _assert_exact(got, expected)


def test_ind2ptr_stream_consumer_ordering():
    cpu = torch.tensor([1, 1, 3, 3, 3, 7], dtype=torch.int64)
    expected = _cpu_golden(cpu, 8)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        ind = cpu.npu()
        ptr = ops_gnn.ind2ptr(ind, 8)
        # Consume the result on the same stream without an explicit sync.
        counts = ptr[1:] - ptr[:-1]
    stream.synchronize()
    _assert_exact(ptr, expected)
    torch.testing.assert_close(counts.cpu(), torch.bincount(cpu, minlength=8),
                               rtol=0, atol=0)


def test_ind2ptr_empty_on_stream():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        ind = torch.empty(0, dtype=torch.int64, device='npu')
        ptr = ops_gnn.ind2ptr(ind, 6)
        touched = ptr + 1
    stream.synchronize()
    _assert_exact(ptr, torch.zeros(7, dtype=torch.int64))
    _assert_exact(touched, torch.ones(7, dtype=torch.int64))


@pytest.mark.parametrize('values,num_rows', [
    ([-1], 3),
    ([3], 3),
    ([1, 0], 3),
    ([], -1),
    ([0, 3], 3),
])
def test_ind2ptr_rejects_invalid(values, num_rows):
    ind = torch.tensor(values, dtype=torch.int64, device='npu')
    with pytest.raises(RuntimeError, match='ind2ptr'):
        ops_gnn.ind2ptr(ind, num_rows)


def test_pybind_rejects_row_equal_to_num_rows():
    ind = torch.tensor([3], dtype=torch.int64, device='npu')
    with pytest.raises(RuntimeError, match='ind2ptr'):
        _pybind.ind2ptr(ind, 3)

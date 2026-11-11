# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public arch22 COO/CSR index conversions against the existing CPU golden."""

import importlib
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
import ops_gnn

_GOLDEN = runpy.run_path(Path(__file__).resolve().parents[1] / 'arch35/golden.py')


@pytest.fixture(autouse=True)
def npu_device():
    torch.npu.set_device(int(os.environ.get('NPU_DEVICE_ID', '0')))


def strided_copy(cpu, strided):
    if strided:
        return torch.stack((cpu, cpu), dim=-1).npu()[:, 0]
    return cpu.npu()


@pytest.mark.parametrize('rows,indices', [
    (0, []), (5, []), (4, [0, 0, 2]), (5, [1, 1, 3]),
    (1, [0] * 70000), (4096, [0, 0, 2048, 4095]),
])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('inference', [False, True])
def test_ind2ptr(rows, indices, strided, inference):
    cpu = torch.tensor(indices, dtype=torch.int64)
    expected = _GOLDEN['ind2ptr_cpu'](cpu, rows)
    with torch.inference_mode(inference):
        ind = strided_copy(cpu, strided)
        got = ops_gnn.ind2ptr(ind, num_rows=rows)
        assert got.device == ind.device and got.dtype == torch.int64
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(ops_gnn.ptr2ind(got, cpu.numel()), ind, rtol=0, atol=0)


@pytest.mark.parametrize('offsets', [
    [0], [0, 0, 0], [0, 2, 2, 3], [0, 0, 3, 3, 4, 4],
    [0, 70000], [0, 1, 8, 8, 13, 14],
])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('inference', [False, True])
def test_ptr2ind(offsets, strided, inference):
    cpu = torch.tensor(offsets, dtype=torch.int64)
    expected = _GOLDEN['ptr2ind_cpu'](cpu, offsets[-1])
    with torch.inference_mode(inference):
        ptr = strided_copy(cpu, strided)
        got = ops_gnn.ptr2ind(ptr, num_edges=offsets[-1])
        assert got.device == ptr.device and got.dtype == torch.int64
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(ops_gnn.ind2ptr(got, len(offsets) - 1), ptr, rtol=0, atol=0)


@pytest.mark.parametrize('values,rows,dtype,device', [
    ([-1], 3, torch.int64, 'npu'), ([3], 3, torch.int64, 'npu'),
    ([1, 0], 3, torch.int64, 'npu'), ([], -1, torch.int64, 'npu'),
    ([0], 3, torch.int32, 'npu'), ([0.], 3, torch.float32, 'npu'),
    ([[0]], 3, torch.int64, 'npu'), (0, 3, torch.int64, 'npu'),
    ([0], 3, torch.int64, 'cpu'),
])
def test_ind2ptr_invalid(values, rows, dtype, device):
    ind = torch.tensor(values, dtype=dtype, device=device)
    with pytest.raises(RuntimeError, match='ind2ptr:'):
        ops_gnn.ind2ptr(ind, rows)


@pytest.mark.parametrize('offsets,edges,dtype,device', [
    ([], 0, torch.int64, 'npu'), ([0], -1, torch.int64, 'npu'),
    ([1, 2], 2, torch.int64, 'npu'), ([0, 3, 2], 2, torch.int64, 'npu'),
    ([0, 2], 3, torch.int64, 'npu'), ([0, -1], 0, torch.int64, 'npu'),
    ([0, 0], 0, torch.int32, 'npu'), ([0., 1.], 1, torch.float32, 'npu'),
    ([[0, 1]], 1, torch.int64, 'npu'), (0, 0, torch.int64, 'npu'),
    ([0, 1], 1, torch.int64, 'cpu'),
])
def test_ptr2ind_invalid(offsets, edges, dtype, device):
    ptr = torch.tensor(offsets, dtype=dtype, device=device)
    with pytest.raises(RuntimeError, match='ptr2ind:'):
        ops_gnn.ptr2ind(ptr, edges)


@pytest.mark.parametrize('name,values', [('ind2ptr', [0]), ('ptr2ind', [0, 0])])
def test_non_integer_size(name, values):
    with pytest.raises(TypeError):
        getattr(ops_gnn, name)(torch.tensor(values, device='npu'), 1.5)


def test_current_stream():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        ind = torch.tensor([1, 1, 3], device='npu')
        ptr = ops_gnn.ind2ptr(ind, 5)
        restored = ops_gnn.ptr2ind(ptr, 3)
    stream.synchronize()
    torch.testing.assert_close(ptr.cpu(), torch.tensor([0, 0, 2, 2, 3, 3]))
    torch.testing.assert_close(restored.cpu(), torch.tensor([1, 1, 3]))


def test_input_device_context():
    if torch.npu.device_count() < 2:
        pytest.skip('requires two NPU devices')
    current = torch.npu.current_device()
    other = (current + 1) % torch.npu.device_count()
    ind = torch.tensor([0, 2], device=f'npu:{other}')
    ptr = ops_gnn.ind2ptr(ind, 3)
    restored = ops_gnn.ptr2ind(ptr, 2)
    assert ptr.device == restored.device == ind.device
    assert torch.npu.current_device() == current
    torch.testing.assert_close(restored.cpu(), torch.tensor([0, 2]))


@pytest.mark.parametrize('name,values,size', [
    ('ind2ptr', [0, 0, 1, 1], 2), ('ptr2ind', [0, 0, 2, 2], 2),
])
def test_arch35_public_native_dispatch(monkeypatch, name, values, size):
    # Check delegation only; this is not an arch35 hardware execution test.
    module = importlib.import_module('ops_gnn.' + name)
    captured = []
    sentinel = object()

    def native(tensor, count):
        captured.append((tensor, count))
        return sentinel

    monkeypatch.setattr(module, '_pybind', SimpleNamespace(npu_arch='dav-3510', **{name: native}))
    tensor = torch.tensor(values)[::2]
    assert getattr(module, name)(tensor, size) is sentinel
    assert captured[0][0].is_contiguous() and captured[0][1] == size
    torch.testing.assert_close(captured[0][0], tensor)

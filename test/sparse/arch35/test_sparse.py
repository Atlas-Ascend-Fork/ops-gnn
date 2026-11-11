# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision tests for ops_gnn.ind2ptr / ops_gnn.ptr2ind.

These ops are integer index conversions (aligned with torch_sparse). Acceptance
criterion is exact equality against a CPU golden that mirrors torch_sparse semantics.
"""

from itertools import product
import os
from pathlib import Path
import runpy

import pytest
import torch
import ops_gnn

_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
ind2ptr_cpu = _GOLDEN["ind2ptr_cpu"]
make_sorted_row_indices = _GOLDEN["make_sorted_row_indices"]
ptr2ind_cpu = _GOLDEN["ptr2ind_cpu"]


def _set_device(device_id=None):
    if hasattr(torch, 'npu'):
        if device_id is None:
            device_id = int(os.environ.get('NPU_DEVICE_ID', '0'))
        torch.npu.set_device(device_id)


def assert_exact(npu_out: torch.Tensor, golden: torch.Tensor, msg: str = ''):
    """Integer ops: require bit-exact match."""
    if npu_out.device.type != 'npu':
        raise AssertionError(f'{msg}: output not on NPU')
    if npu_out.dtype != torch.long:
        raise AssertionError(f'{msg}: dtype mismatch')
    if npu_out.shape != golden.shape:
        raise AssertionError(f'{msg}: shape {npu_out.shape} vs {golden.shape}')
    if not torch.equal(npu_out.cpu(), golden):
        raise AssertionError(
            f'{msg}: NPU={npu_out.cpu().tolist()} golden={golden.tolist()}'
        )


def test_ind2ptr_torch_sparse_basic():
    _set_device()
    row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
    rowptr = ops_gnn.ind2ptr(row, 8)
    assert_exact(rowptr, torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long))


def test_ptr2ind_torch_sparse_basic():
    _set_device()
    rowptr = torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long, device='npu')
    row = ops_gnn.ptr2ind(rowptr, 6)
    assert_exact(row, torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long))


def test_ind2ptr_empty():
    _set_device()
    row = torch.tensor([], dtype=torch.long, device='npu')
    rowptr = ops_gnn.ind2ptr(row, 8)
    assert_exact(rowptr, torch.zeros(9, dtype=torch.long))


def test_ptr2ind_empty():
    _set_device()
    rowptr = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long, device='npu')
    row = ops_gnn.ptr2ind(rowptr, 0)
    assert_exact(row, torch.empty(0, dtype=torch.long))


def test_api_replace_style():
    _set_device()
    row = torch.tensor([1, 2, 2], dtype=torch.long, device='npu')
    rowptr = ops_gnn.ind2ptr(row, 4)
    assert_exact(rowptr, torch.tensor([0, 0, 1, 3, 3], dtype=torch.long))
    assert_exact(ops_gnn.ptr2ind(rowptr, 3), torch.tensor([1, 2, 2], dtype=torch.long))


@pytest.mark.parametrize('num_rows', [0, 1, 2, 8, 64, 1024])
def test_ind2ptr_empty_various_num_rows(num_rows):
    _set_device()
    row = torch.tensor([], dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, num_rows)
    assert_exact(
        out,
        torch.zeros(num_rows + 1, dtype=torch.long),
        msg=f'empty num_rows={num_rows}',
    )


def test_ind2ptr_single_edge_first_row():
    _set_device()
    row = torch.tensor([0], dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, 4)
    assert_exact(out, ind2ptr_cpu(row.cpu(), 4))


def test_ind2ptr_single_edge_last_row():
    _set_device()
    row = torch.tensor([7], dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, 8)
    assert_exact(out, ind2ptr_cpu(row.cpu(), 8))


def test_ind2ptr_all_same_row():
    _set_device()
    row = torch.tensor([3, 3, 3, 3, 3], dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, 6)
    assert_exact(out, ind2ptr_cpu(row.cpu(), 6))


def test_ind2ptr_leading_empty_rows():
    _set_device()
    row = torch.tensor([5, 5, 6], dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, 8)
    assert_exact(out, ind2ptr_cpu(row.cpu(), 8))


def test_ind2ptr_trailing_empty_rows():
    _set_device()
    row = torch.tensor([0, 0, 1], dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, 10)
    assert_exact(out, ind2ptr_cpu(row.cpu(), 10))


def test_ind2ptr_no_empty_rows():
    _set_device()
    row = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, 5)
    assert_exact(out, ind2ptr_cpu(row.cpu(), 5))


def test_ind2ptr_dense_one_per_row():
    _set_device()
    num_rows = 128
    for row in (
        torch.arange(num_rows, dtype=torch.long, device='npu'),
        torch.arange(num_rows, dtype=torch.long).to('npu'),
    ):
        out = ops_gnn.ind2ptr(row, num_rows)
        assert_exact(out, ind2ptr_cpu(row.cpu(), num_rows), msg=f'dense num_rows={num_rows}')


@pytest.mark.parametrize('num_edges,num_rows', [
    (32767, 257),
    (32768, 257),
    (65535, 257),
    (65536, 257),
    (65536, 960),
    (65536, 961),
    (65536, 962),
    (65536, 963),
    (65536, 1023),
    (65536, 1024),
    (1048575, 257),
    (1048576, 257),
    (1048576, 6240),
    (1048576, 6241),
])
@pytest.mark.parametrize('distribution', ['uniform', 'ends', 'single', 'first', 'last'])
def test_ind2ptr_dispatch_boundary(num_edges, num_rows, distribution):
    _set_device()
    ind_cpu = torch.arange(num_edges, dtype=torch.long) * num_rows // num_edges
    if distribution == 'ends':
        ind_cpu = (torch.arange(num_edges) >= num_edges // 2).long() * (num_rows - 1)
    elif distribution == 'single':
        ind_cpu.fill_(num_rows // 2)
    elif distribution == 'first':
        ind_cpu.zero_()
    elif distribution == 'last':
        ind_cpu.fill_(num_rows - 1)
    golden = ind2ptr_cpu(ind_cpu, num_rows)
    out = ops_gnn.ind2ptr(ind_cpu.to('npu'), num_rows)
    assert_exact(
        out,
        golden,
        msg=f'dispatch edges={num_edges} rows={num_rows} distribution={distribution}',
    )


def test_ptr2ind_all_in_one_row():
    _set_device()
    ptr = torch.tensor([0, 0, 0, 5, 5], dtype=torch.long, device='npu')
    out = ops_gnn.ptr2ind(ptr, 5)
    assert_exact(out, ptr2ind_cpu(ptr.cpu(), 5))


def test_ptr2ind_alternating_empty():
    _set_device()
    ptr = torch.tensor([0, 0, 2, 2, 3, 3], dtype=torch.long, device='npu')
    out = ops_gnn.ptr2ind(ptr, 3)
    assert_exact(out, ptr2ind_cpu(ptr.cpu(), 3))


def test_noncontiguous_ind_input():
    _set_device()
    base = torch.tensor([[2, 9], [2, 9], [4, 9], [5, 9], [5, 9], [6, 9]],
                        dtype=torch.long, device='npu')
    row = base[:, 0]
    if row.is_contiguous():
        raise AssertionError('expected non-contiguous row view')
    golden = ind2ptr_cpu(row.contiguous().cpu(), 8)
    out = ops_gnn.ind2ptr(row, 8)
    assert_exact(out, golden)


def test_noncontiguous_ptr_input():
    _set_device()
    base = torch.tensor(
        [[0, 9], [0, 9], [0, 9], [2, 9], [2, 9], [3, 9], [5, 9], [6, 9], [6, 9]],
        dtype=torch.long,
        device='npu',
    )
    ptr = base[:, 0]
    if ptr.is_contiguous():
        raise AssertionError('expected non-contiguous ptr view')
    golden = ptr2ind_cpu(ptr.contiguous().cpu(), 6)
    out = ops_gnn.ptr2ind(ptr, 6)
    assert_exact(out, golden)


IND2PTR_CASES = [
    ([], 0, 'empty_rows0'),
    ([], 1, 'empty_rows1'),
    ([0], 1, 'single_rows1'),
    ([0], 8, 'single_row0'),
    ([7], 8, 'single_row7'),
    ([0, 0], 3, 'dup_row0'),
    ([1, 1, 1], 3, 'dup_mid'),
    ([0, 1], 2, 'full_dense_2'),
    ([0, 1, 2], 3, 'full_dense_3'),
    ([0, 0, 1, 1], 2, 'balanced_2x2'),
    ([2, 2, 4, 5, 5, 6], 8, 'torch_sparse_ref'),
    ([0, 0, 1, 3, 3, 3, 7], 8, 'mixed_gaps'),
    ([0] * 16, 4, 'all_first_row'),
    ([3] * 16, 4, 'all_last_row'),
    (list(range(64)), 64, 'identity_64'),
    ([0, 2, 4, 6, 8, 10], 12, 'even_sparse'),
    ([1, 3, 5, 7, 9, 11], 12, 'odd_sparse'),
    ([0] * 100 + [50] * 100 + [99] * 100, 100, 'three_blocks'),
    (list(range(0, 256, 2)), 256, 'stride2_256'),
    ([0] + [255] * 31, 256, 'head_tail_256'),
]


@pytest.mark.parametrize(
    'ind_list,num_rows,case_id',
    IND2PTR_CASES,
    ids=[case[2] for case in IND2PTR_CASES],
)
def test_ind2ptr_precision_matrix(ind_list, num_rows, case_id):
    _set_device()
    ind_cpu = torch.tensor(ind_list, dtype=torch.long)
    golden = ind2ptr_cpu(ind_cpu, num_rows)
    out = ops_gnn.ind2ptr(ind_cpu.to('npu'), num_rows)
    assert_exact(out, golden, msg=f'ind2ptr/{case_id}')


PTR2IND_CASES = [
    ([0], 0, 'rows0_edges0'),
    ([0, 0], 0, 'rows1_edges0'),
    ([0, 1], 1, 'rows1_edges1'),
    ([0, 0, 0, 0], 0, 'all_empty'),
    ([0, 5], 5, 'single_row'),
    ([0, 0, 0, 5, 5], 5, 'mid_row_only'),
    ([0, 0, 0, 2, 2, 3, 5, 6, 6], 6, 'torch_sparse_ref'),
    ([0, 1, 2, 3, 4, 5], 5, 'one_each'),
    ([0, 2, 4, 6, 8], 8, 'even_bins'),
    ([0, 0, 2, 2, 3, 3], 3, 'alt_empty'),
    ([0, 10, 10, 20, 30], 30, 'uneven'),
    ([0] + list(range(1, 65)), 64, 'prefix_inc_64'),
    ([0] + [16] * 4 + [64], 64, 'blocky'),
    ([0, 0, 100, 100, 100], 100, 'leading_empty_block'),
    ([0, 50, 50, 50, 100], 100, 'middle_empty'),
]


@pytest.mark.parametrize(
    'ptr_list,num_edges,case_id',
    PTR2IND_CASES,
    ids=[case[2] for case in PTR2IND_CASES],
)
def test_ptr2ind_precision_matrix(ptr_list, num_edges, case_id):
    _set_device()
    ptr_cpu = torch.tensor(ptr_list, dtype=torch.long)
    golden = ptr2ind_cpu(ptr_cpu, num_edges)
    out = ops_gnn.ptr2ind(ptr_cpu.to('npu'), num_edges)
    assert_exact(out, golden, msg=f'ptr2ind/{case_id}')


RANDOM_SHAPES = list(product(
    [1, 7, 16, 64, 257, 1024, 4096],
    [1, 8, 64, 128, 512, 1025],
    [0, 1, 42],
))


@pytest.mark.parametrize(
    'num_edges,num_rows,seed',
    RANDOM_SHAPES[:36],
    ids=[
        f'edges{num_edges}_rows{num_rows}_s{seed}'
        for num_edges, num_rows, seed in RANDOM_SHAPES[:36]
    ],
)
def test_ind2ptr_random_vs_golden(num_edges, num_rows, seed):
    _set_device()
    ind_cpu = make_sorted_row_indices(num_edges, num_rows, seed=seed)
    if num_rows == 0:
        ind_cpu = torch.empty(0, dtype=torch.long)
        num_edges = 0
    golden = ind2ptr_cpu(ind_cpu, num_rows)
    out = ops_gnn.ind2ptr(ind_cpu.to('npu'), num_rows)
    assert_exact(
        out,
        golden,
        msg=f'random ind2ptr edges={num_edges} rows={num_rows} seed={seed}',
    )


@pytest.mark.parametrize(
    'num_edges,num_rows,seed',
    RANDOM_SHAPES[:36],
    ids=[
        f'edges{num_edges}_rows{num_rows}_s{seed}'
        for num_edges, num_rows, seed in RANDOM_SHAPES[:36]
    ],
)
def test_roundtrip_random(num_edges, num_rows, seed):
    _set_device()
    if num_rows == 0:
        ind_cpu = torch.empty(0, dtype=torch.long)
        num_edges = 0
    else:
        ind_cpu = make_sorted_row_indices(num_edges, num_rows, seed=seed)
    ind = ind_cpu.to('npu')
    ptr = ops_gnn.ind2ptr(ind, num_rows)
    restored = ops_gnn.ptr2ind(ptr, ind.numel())
    assert_exact(
        restored,
        ind_cpu,
        msg=f'roundtrip edges={num_edges} rows={num_rows} seed={seed}',
    )
    assert_exact(
        ptr,
        ind2ptr_cpu(ind_cpu, num_rows),
        msg=f'roundtrip-ptr edges={num_edges} rows={num_rows}',
    )


@pytest.mark.parametrize('num_edges,num_rows,seed', [
    (10000, 1000, 7),
    (50000, 4096, 11),
    (100000, 8192, 13),
], ids=['10k', '50k', '100k'])
def test_large_scale_precision(num_edges, num_rows, seed):
    _set_device()
    ind_cpu = make_sorted_row_indices(num_edges, num_rows, seed=seed)
    golden_ptr = ind2ptr_cpu(ind_cpu, num_rows)
    ptr = ops_gnn.ind2ptr(ind_cpu.to('npu'), num_rows)
    assert_exact(ptr, golden_ptr, msg=f'large ind2ptr edges={num_edges}')
    restored = ops_gnn.ptr2ind(ptr, num_edges)
    assert_exact(restored, ind_cpu, msg=f'large ptr2ind edges={num_edges}')


def test_output_device_and_dtype():
    _set_device()
    row = torch.tensor([0, 1, 1], dtype=torch.long, device='npu')
    ptr = ops_gnn.ind2ptr(row, 3)
    ind = ops_gnn.ptr2ind(ptr, 3)
    if ptr.device.type != 'npu' or ptr.dtype != torch.long:
        raise AssertionError('ptr device/dtype mismatch')
    if ind.device.type != 'npu' or ind.dtype != torch.long:
        raise AssertionError('ind device/dtype mismatch')
    if ptr.numel() != 4 or ind.numel() != 3:
        raise AssertionError('unexpected numel')


def test_ptr_monotonic_and_bounds():
    _set_device()
    row = torch.tensor([0, 0, 2, 5, 5], dtype=torch.long, device='npu')
    num_rows, num_edges = 6, 5
    ptr = ops_gnn.ind2ptr(row, num_rows).cpu()
    if ptr[0].item() != 0:
        raise AssertionError('ptr[0] must be 0')
    if ptr[-1].item() != num_edges:
        raise AssertionError('ptr[-1] must equal num_edges')
    if not torch.all(ptr[1:] >= ptr[:-1]):
        raise AssertionError('ptr must be non-decreasing')
    if not (torch.all(ptr >= 0) and torch.all(ptr <= num_edges)):
        raise AssertionError('ptr values out of bounds')


def _set_device(device_id=None):
    if hasattr(torch, 'npu'):
        if device_id is None:
            device_id = int(os.environ.get('NPU_DEVICE_ID', '0'))
        torch.npu.set_device(device_id)


def test_ind2ptr_async_default_stream():
    _set_device()
    row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, 8)
    torch.npu.current_stream().synchronize()
    assert_exact(out, torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long))


def test_ptr2ind_async_default_stream():
    _set_device()
    ptr = torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long, device='npu')
    out = ops_gnn.ptr2ind(ptr, 6)
    torch.npu.synchronize()
    assert_exact(out, torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long))


def test_ind2ptr_custom_stream():
    _set_device()
    row = torch.tensor([1, 2, 2], dtype=torch.long, device='npu')
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        out = ops_gnn.ind2ptr(row, 4)
    stream.synchronize()
    assert_exact(out, ind2ptr_cpu(row.cpu(), 4))


def test_ptr2ind_custom_stream():
    _set_device()
    ptr = torch.tensor([0, 0, 1, 3, 3], dtype=torch.long, device='npu')
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        out = ops_gnn.ptr2ind(ptr, 3)
    stream.synchronize()
    assert_exact(out, ptr2ind_cpu(ptr.cpu(), 3))


def test_ind2ptr_async_after_arange():
    """Preceding torch_npu op on same stream must be visible."""
    _set_device()
    num_rows = 128
    row = torch.arange(num_rows, dtype=torch.long, device='npu')
    out = ops_gnn.ind2ptr(row, num_rows)
    torch.npu.synchronize()
    assert_exact(out, ind2ptr_cpu(row.cpu(), num_rows), msg='arange async')


def test_roundtrip_same_stream():
    _set_device()
    row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
    ptr = ops_gnn.ind2ptr(row, 8)
    out = ops_gnn.ptr2ind(ptr, 6)
    torch.npu.synchronize()
    assert_exact(out, row.cpu())

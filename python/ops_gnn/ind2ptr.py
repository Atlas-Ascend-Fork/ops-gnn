# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public ind2ptr API."""

import operator

import torch

from . import _pybind
from .typing import Tensor


def _ind2ptr_npu(index: torch.Tensor, size: int) -> torch.Tensor:
    """Build CSR pointers on the index device; no index tensor visits CPU."""
    counts = index.new_zeros(size)
    counts.scatter_add_(0, index, torch.ones_like(index))
    return torch.cat((index.new_zeros(1), counts.cumsum(0)))


def ind2ptr(ind: Tensor, num_rows: int) -> Tensor:
    """Convert sorted row indices to a CSR row-pointer tensor.

    Drop-in replacement for ``torch.ops.torch_sparse.ind2ptr(ind, M)`` on Ascend NPU.
    The second argument is the row count (historically named ``M`` in torch_sparse).

    Ascend 950PR&950DT series products:
    Launches asynchronously on the current NPU stream (same as pytorch_sparse CUDA).
    Synchronize with ``torch.npu.synchronize()`` / ``torch.npu.current_stream().synchronize()``
    before reading results on host if needed.

    Calls with non-empty input on Atlas A3 series products and Atlas A2 series
    products include device synchronization.

    Args:
        ind (Tensor): 1-D ``torch.long`` tensor of non-decreasing row indices on NPU.
        num_rows (int): Number of rows. Output length is ``num_rows + 1``.

    Returns:
        Tensor: 1-D ``torch.long`` CSR pointer of shape ``[num_rows + 1]`` on NPU.

    Example:
        >>> import torch
        >>> import ops_gnn
        >>> row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
        >>> ops_gnn.ind2ptr(row, 8)
        tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], device='npu:0')
    """
    if _pybind.npu_arch != 'dav-2201':
        if not ind.is_contiguous():
            ind = ind.contiguous()
        return _pybind.ind2ptr(ind, int(num_rows))
    num_rows = operator.index(num_rows)
    if (ind.device.type != 'npu' or ind.layout != torch.strided
            or ind.dtype != torch.int64):
        raise RuntimeError('ind2ptr: ind must be a 1-D NPU int64 tensor')
    if ind.dim() != 1:
        raise RuntimeError('ind2ptr: ind must be a 1-D NPU int64 tensor')
    if num_rows < 0:
        raise RuntimeError('ind2ptr: num_rows must be non-negative')
    with torch.npu.device(ind.device):
        if ind.numel():
            invalid = ((ind < 0) | (ind >= num_rows)).any()
            invalid = invalid | (ind[1:] < ind[:-1]).any()
            if bool(invalid):
                raise RuntimeError('ind2ptr: ind must be sorted and in [0, num_rows)')
        return _ind2ptr_npu(ind, num_rows)

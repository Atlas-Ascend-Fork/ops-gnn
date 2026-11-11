# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public ptr2ind API."""

import operator

import torch

from . import _pybind
from .typing import Tensor


def _ptr2ind_npu(pointer: torch.Tensor, nnz: int) -> torch.Tensor:
    """Expand CSR row ids on NPU, with the output size already known."""
    rows = torch.arange(pointer.numel() - 1, dtype=pointer.dtype,
                        device=pointer.device)
    return torch.repeat_interleave(
        rows, pointer[1:] - pointer[:-1], output_size=nnz)


def ptr2ind(ptr: Tensor, num_edges: int) -> Tensor:
    """Convert a CSR row-pointer tensor to row indices.

    Drop-in replacement for ``torch.ops.torch_sparse.ptr2ind(ptr, E)`` on Ascend NPU.
    The second argument is the edge/nnz count (historically named ``E`` in torch_sparse).

    Ascend 950PR&950DT series products:
    Launches asynchronously on the current NPU stream (same as pytorch_sparse CUDA).
    Synchronize with ``torch.npu.synchronize()`` / ``torch.npu.current_stream().synchronize()``
    before reading results on host if needed.

    Calls on Atlas A3 series products and Atlas A2 series products include
    device synchronization.

    Args:
        ptr (Tensor): 1-D ``torch.long`` CSR pointer on NPU, length ``num_rows + 1``.
        num_edges (int): Number of edges / non-zeros. Output length is ``num_edges``.

    Returns:
        Tensor: 1-D ``torch.long`` row indices of shape ``[num_edges]`` on NPU.

    Example:
        >>> import torch
        >>> import ops_gnn
        >>> rowptr = torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long, device='npu')
        >>> ops_gnn.ptr2ind(rowptr, 6)
        tensor([2, 2, 4, 5, 5, 6], device='npu:0')
    """
    if _pybind.npu_arch != 'dav-2201':
        if not ptr.is_contiguous():
            ptr = ptr.contiguous()
        return _pybind.ptr2ind(ptr, int(num_edges))
    num_edges = operator.index(num_edges)
    if (ptr.device.type != 'npu' or ptr.layout != torch.strided
            or ptr.dtype != torch.int64):
        raise RuntimeError('ptr2ind: ptr must be a non-empty 1-D NPU int64 tensor')
    if ptr.dim() != 1 or ptr.numel() == 0:
        raise RuntimeError('ptr2ind: ptr must be a non-empty 1-D NPU int64 tensor')
    if num_edges < 0:
        raise RuntimeError('ptr2ind: num_edges must be non-negative')
    with torch.npu.device(ptr.device):
        invalid = (ptr[0] != 0) | (ptr[-1] != num_edges)
        invalid = invalid | (ptr[1:] < ptr[:-1]).any()
        if bool(invalid):
            raise RuntimeError('ptr2ind: ptr must be non-decreasing from 0 to num_edges')
        return _ptr2ind_npu(ptr, num_edges)

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public csr_transpose API."""

import operator
from typing import Optional, Tuple

import torch

from . import _pybind
from .typing import Tensor

_OP_NAME = "csr_transpose"

# The build whose device kernel this operator ships. On 950 the native int64
# argsort makes the primitive composition faster than the kernel, so that build
# runs the composition instead (see ``_transpose_composition``).
_KERNEL_ARCH = "dav-2201"


def _check_arguments(indptr: torch.Tensor, indices: torch.Tensor) -> None:
    """Reject an argument combination the operator cannot serve."""
    if not isinstance(indptr, torch.Tensor):
        raise TypeError(f"indptr must be a torch.Tensor, got {type(indptr).__name__}")
    if not isinstance(indices, torch.Tensor):
        raise TypeError(f"indices must be a torch.Tensor, got {type(indices).__name__}")
    if indptr.dim() != 1:
        raise ValueError(f"indptr must be 1-D, got {indptr.dim()}-D")
    if indices.dim() != 1:
        raise ValueError(f"indices must be 1-D, got {indices.dim()}-D")
    if indptr.dtype != torch.int64:
        raise TypeError(f"indptr dtype must be int64, got {indptr.dtype}")
    if indices.dtype != torch.int64:
        raise TypeError(f"indices dtype must be int64, got {indices.dtype}")
    if indptr.device != indices.device:
        raise ValueError(
            f"indptr and indices must be on the same device, got "
            f"{indptr.device} and {indices.device}")
    if indptr.device.type != "npu":
        raise RuntimeError(f"{_OP_NAME} requires Ascend NPU tensors")
    if indptr.numel() < 1:
        raise ValueError("indptr must hold at least one boundary")
    if indices.numel() != 0 and indptr.numel() < 2:
        raise RuntimeError(
            f"{_OP_NAME}: indptr must describe at least one row when the matrix has edges")


def _resolve_columns(indices: torch.Tensor, num_cols: Optional[int]) -> int:
    """Rows of the transposed matrix, inferred from ``indices`` when omitted."""
    if num_cols is not None:
        columns = operator.index(num_cols)
        if columns < 0:
            raise ValueError(f"num_cols must be non-negative, got {columns}")
    elif indices.numel() == 0:
        columns = 0
    else:
        columns = int(indices.max().item()) + 1
    if columns == 0 and indices.numel() != 0:
        raise RuntimeError(
            f"{_OP_NAME}: num_cols is 0 but the matrix has {indices.numel()} edges")
    return columns


def _transpose_composition(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    num_cols: int,
) -> Tuple[Tensor, Tensor]:
    """Transpose with the device primitives the 950 executes natively.

    The histogram of the column ids gives the output pointer, and a stable sort on
    the column ids gives the layout: a stable sort keeps the input order inside
    every column, and the rows are the expansion of the input pointer, which is
    row-major, so each output segment comes out ascending. Both primitives are
    native kernels on 950 (about 0.3 ms for 1M edges), which makes this
    composition roughly 40x faster there than the three-pass kernel.
    """
    device = indptr.device
    rows = torch.repeat_interleave(
        torch.arange(indptr.numel() - 1, dtype=torch.int64, device=device),
        indptr[1:] - indptr[:-1])
    counts = torch.zeros(num_cols, dtype=torch.int64, device=device)
    if indices.numel() != 0:
        counts.scatter_add_(0, indices, torch.ones_like(indices))
    out_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)])
    return out_indptr, rows[torch.argsort(indices, stable=True)]


def csr_transpose(
    indptr: Tensor,
    indices: Tensor,
    num_cols: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Transpose the structure of an (N, M) CSR matrix into an (M, N) CSR matrix.

    Every edge ``(row, column)`` of the input becomes ``(column, row)`` of the
    output: only the two index arrays are rewritten, values are not part of the
    call and duplicates are not merged. This is the layout change behind
    ``torch_sparse.transpose(..., coalesced=True)`` and behind the reverse-graph
    construction of message passing layers.

    Both implementations launch asynchronously on the current NPU stream, so
    ``torch.npu.synchronize()`` is only needed before reading the results on the
    host. The one exception is an omitted ``num_cols``: deriving it from
    ``indices.max() + 1`` reads a device scalar and therefore synchronizes.

    The row ids of each output segment come out ascending, because the input is
    scanned in row-major order and every edge is appended behind the previous edge
    of its column. The result is therefore determined by the input alone and can
    be compared element by element, independent of the number of AI cores.

    Args:
        indptr (Tensor): 1-D ``torch.long`` tensor on NPU with the ``N + 1`` row
            boundaries of the input. The values must be non-decreasing, start at
            ``0`` and end at ``indices.numel()``; they are not read on the host.
        indices (Tensor): 1-D ``torch.long`` tensor on NPU with the ``E`` column
            ids of the input, in ``[0, num_cols)``.
        num_cols (int, optional): Number of rows of the transposed matrix. When it
            is omitted the operator uses ``indices.max() + 1``, which reads one
            value from the device; pass it to keep the call asynchronous.

    Returns:
        Tuple[Tensor, Tensor]: ``(out_indptr, out_indices)``, the ``M + 1`` row
        boundaries and the ``E`` row ids of the transposed matrix, both 1-D
        ``torch.long`` on the same NPU as the input.

    Raises:
        TypeError: An argument is not a tensor, or has the wrong dtype or type.
        ValueError: An argument has the wrong rank, the two tensors live on
            different devices, ``indptr`` is empty, or ``num_cols`` is negative.
        RuntimeError: The inputs are not on an Ascend NPU, the matrix has edges
            but ``num_cols`` is zero, or ``indices`` is non-empty while ``indptr``
            holds a single boundary.

    Example:
        >>> import torch
        >>> import ops_gnn
        >>> indptr = torch.tensor([0, 1, 3], dtype=torch.long, device='npu')
        >>> indices = torch.tensor([0, 0, 1], dtype=torch.long, device='npu')
        >>> out_indptr, out_indices = ops_gnn.csr_transpose(indptr, indices, 2)
        >>> out_indptr
        tensor([0, 2, 3], device='npu:0')
        >>> out_indices
        tensor([0, 1, 1], device='npu:0')
    """
    _check_arguments(indptr, indices)
    columns = _resolve_columns(indices, num_cols)
    if _pybind.npu_arch == _KERNEL_ARCH:
        return _pybind.csr_transpose(indptr, indices, columns)
    return _transpose_composition(indptr, indices, columns)

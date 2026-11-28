# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public spspmm API (sparse x sparse matrix multiplication)."""

import operator
from typing import Tuple

import torch

from . import _pybind
from .typing import Tensor

_OP_NAME = "spspmm"

# Both device builds provide a native kernel: arch35 (SIMT) on 950 and arch22
# (classic AscendC kernels) on A2/A3.
_KERNEL_ARCHS = ("dav-3510", "dav-2201")

# Every integer below 2 ** 24 is exact in float32, so keys in that range sort
# identically whether they are compared as int64 or as float32.
_FLOAT32_EXACT_MAX = 1 << 24


def _check_side(index: torch.Tensor, value: torch.Tensor, side: str) -> None:
    """Reject a COO operand the operator cannot serve."""
    if not isinstance(index, torch.Tensor):
        raise TypeError(f"index_{side} must be a torch.Tensor, got {type(index).__name__}")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"value_{side} must be a torch.Tensor, got {type(value).__name__}")
    if index.dtype != torch.int64:
        raise TypeError(f"index_{side} dtype must be int64, got {index.dtype}")
    if index.dim() != 2 or index.size(0) != 2:
        raise ValueError(
            f"index_{side} must have shape [2, nnz], got {tuple(index.shape)}")
    if value.dim() != 1:
        raise ValueError(f"value_{side} must be 1-D, got {value.dim()}-D")
    if value.numel() != index.size(1):
        raise ValueError(
            f"value_{side} must hold one weight per edge, got {value.numel()} "
            f"for {index.size(1)} edges")
    if value.dtype not in (torch.float32, torch.float16):
        raise TypeError(f"value_{side} dtype must be float32 or float16, got {value.dtype}")
    if index.device != value.device:
        raise ValueError(
            f"index_{side} and value_{side} must be on the same device, got "
            f"{index.device} and {value.device}")
    if index.device.type != "npu":
        raise RuntimeError(f"{_OP_NAME} requires Ascend NPU tensors")


def _coalesce(
    index: torch.Tensor,
    value: torch.Tensor,
    num_rows: int,
    num_cols: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge duplicate coordinates and sort the edges into row-major order.

    Sorting the linear key ``row * num_cols + col`` and reducing every run of
    equal keys is exactly a coalesce of the operator's sum semantics, so the
    result is independent of the input order. Structural zeros survive: a run
    whose products cancel keeps its single entry.

    Every valid coordinate produces a key below ``num_rows * num_cols``. When
    that bound proves the keys are exact in float32 the sort runs on the AI
    cores, which is much faster than the int64 sort that the device falls back
    to the AI CPU for.
    """
    if index.size(1) == 0:
        return index, value
    row, col = index[0], index[1]
    key = row * num_cols + col
    bound = num_rows * num_cols
    if 0 <= bound <= _FLOAT32_EXACT_MAX:
        order = torch.argsort(key.to(torch.float32), stable=True)
    else:
        order = torch.argsort(key, stable=True)
    row, col, key = row[order], col[order], key[order]
    unique = torch.ones(key.numel(), dtype=torch.bool, device=key.device)
    unique[1:] = key[1:] != key[:-1]
    # ``segment`` numbers the runs, so one scatter-add reduces them in dtype.
    segment = torch.cumsum(unique.to(torch.int64), 0) - 1
    runs = int(unique.sum().item())
    merged = torch.zeros(runs, dtype=value.dtype, device=value.device)
    merged.scatter_add_(0, segment, value[order])
    return torch.stack([row[unique], col[unique]]), merged


def spspmm(
    index_a: Tensor,
    value_a: Tensor,
    index_b: Tensor,
    value_b: Tensor,
    m: int,
    k: int,
    n: int,
    coalesced: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Multiply an ``(m, k)`` sparse matrix by a ``(k, n)`` sparse matrix.

    The operator follows ``torch_sparse.spspmm``: both operands are COO tensors
    and the product is returned as coalesced COO, i.e. its row ids are ascending,
    the column ids ascend inside every row, and no coordinate repeats. Duplicate
    products of one coordinate are accumulated, and a coordinate whose weights
    cancel to zero is kept as a structural zero.

    The three-phase kernel runs on the current NPU stream: the row histogram and
    the candidate-product counts are prefix summed, an ordered per-row merge of
    the touched B rows emits the unique columns, and the per-row candidates are
    compacted into the result. The result length is data dependent, so the host
    reads a few scalars from the device (the validation flag, the candidate slot
    count and ``nnz(C)``) before allocating the output; every read drains the
    stream.

    Args:
        index_a (Tensor): 2-D ``torch.long`` tensor of shape ``[2, nnzA]`` with
            the ``(row, column)`` coordinates of the left operand.
        value_a (Tensor): 1-D ``torch.float32`` or ``torch.float16`` tensor with
            the ``nnzA`` weights of the left operand.
        index_b (Tensor): 2-D ``torch.long`` tensor of shape ``[2, nnzB]`` with
            the ``(row, column)`` coordinates of the right operand.
        value_b (Tensor): 1-D tensor with the ``nnzB`` weights of the right
            operand; it must share the dtype and the device of ``value_a``.
        m (int): Number of rows of the product.
        k (int): Shared inner dimension.
        n (int): Number of columns of the product.
        coalesced (bool): When ``True`` the two inputs are already coalesced and
            the merge pass is skipped; the kernel then rejects unsorted or
            duplicated coordinates. Defaults to ``False``, which merges the
            inputs first.

    Returns:
        Tuple[Tensor, Tensor]: ``(index_c, value_c)``, the ``[2, nnzC]`` int64
        coordinates and the ``nnzC`` weights of the product, on the same NPU as
        the inputs.

    Raises:
        TypeError: An argument is not a tensor, or a dtype is unsupported.
        ValueError: A shape is inconsistent, or the operands live on different
            devices.
        RuntimeError: The inputs are not on an Ascend NPU, an input is not
            coalesced although ``coalesced=True`` was passed, an autograd graph
            would be silently dropped, or the build targets another
            architecture.

    Example:
        >>> import torch
        >>> import ops_gnn
        >>> index_a = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device='npu')
        >>> value_a = torch.tensor([2.0, 3.0], dtype=torch.float32, device='npu')
        >>> index_c, value_c = ops_gnn.spspmm(index_a, value_a, index_a, value_a, 2, 2, 2)
        >>> index_c.cpu()
        tensor([[0, 1],
                [0, 1]])
        >>> value_c.cpu()
        tensor([4., 9.])
    """
    m = operator.index(m)
    k = operator.index(k)
    n = operator.index(n)
    if m < 0 or k < 0 or n < 0:
        raise ValueError(f"m, k and n must be non-negative, got {(m, k, n)}")
    _check_side(index_a, value_a, "a")
    _check_side(index_b, value_b, "b")
    if value_a.device != value_b.device:
        raise ValueError(
            f"value_a and value_b must be on the same device, got "
            f"{value_a.device} and {value_b.device}")
    if value_a.dtype != value_b.dtype:
        raise TypeError(
            f"value_a and value_b must share one dtype, got {value_a.dtype} and {value_b.dtype}")
    if value_a.requires_grad or value_b.requires_grad:
        raise RuntimeError(
            f"{_OP_NAME} has no autograd support yet; detach the inputs or build the "
            f"gradient by hand")
    if _pybind.npu_arch not in _KERNEL_ARCHS:
        raise RuntimeError(
            f"{_OP_NAME} is only available on Ascend 950 (arch35) and A2/A3 (arch22), "
            f"current arch is {_pybind.npu_arch}")

    if not coalesced:
        index_a, value_a = _coalesce(index_a, value_a, m, k)
        index_b, value_b = _coalesce(index_b, value_b, k, n)

    if index_a.size(1) == 0 or index_b.size(1) == 0:
        return (
            torch.empty((2, 0), dtype=torch.int64, device=value_a.device),
            torch.empty((0,), dtype=value_a.dtype, device=value_a.device),
        )

    return _pybind.spspmm(
        index_a.contiguous(), value_a.contiguous(),
        index_b.contiguous(), value_b.contiguous(), m, k, n)

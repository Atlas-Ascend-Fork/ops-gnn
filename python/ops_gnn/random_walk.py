# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyTorch-facing API for the Ascend NPU random-walk operator."""

__all__ = ["random_walk"]

import weakref
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from . import _pybind


# Random walk normally samples the same static graph many times. Keep the most
# recent coalesced CSR conversion; tensor identity plus PyTorch's mutation
# version counter makes reuse safe and invalidates the entry after in-place
# updates. Weak references avoid extending the input graph lifetime.
_COALESCED_CSR_CACHE = None
_MAX_EXACT_FLOAT32_INTEGER = 1 << 24


def _check_long_vector(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype torch.int64")
    if value.dim() != 1:
        raise ValueError(f"{name} must be one-dimensional")


def _resolve_num_nodes(row: Tensor, col: Tensor, start: Tensor, num_nodes: Optional[int]) -> int:
    if num_nodes is None:
        maxima = []
        for tensor in (row, col, start):
            if tensor.numel() > 0:
                maxima.append(int(tensor.max()))
        if not maxima:
            raise ValueError("num_nodes is required when row, col and start are empty")
        return max(maxima) + 1
    if not isinstance(num_nodes, int) or isinstance(num_nodes, bool):
        raise TypeError("num_nodes must be an int or None")
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    return num_nodes


def _validate_inputs(
    row: Tensor, col: Tensor, start: Tensor, walk_length: int, num_nodes: Optional[int]
) -> int:
    for name, tensor in (("row", row), ("col", col), ("start", start)):
        _check_long_vector(name, tensor)
    if row.numel() != col.numel():
        raise ValueError("row and col must contain the same number of edges")
    if row.device != col.device or row.device != start.device:
        raise ValueError("row, col and start must be on the same device")
    if not isinstance(walk_length, int) or isinstance(walk_length, bool):
        raise TypeError("walk_length must be an int")
    if walk_length < 0:
        raise ValueError("walk_length must be non-negative")

    resolved_nodes = _resolve_num_nodes(row, col, start, num_nodes)
    if row.numel() > 0:
        if int(row.min()) < 0 or int(col.min()) < 0:
            raise ValueError("row and col entries must be non-negative")
        if int(row.max()) >= resolved_nodes or int(col.max()) >= resolved_nodes:
            raise ValueError("row and col entries must be smaller than num_nodes")
    if start.numel() > 0 and (int(start.min()) < 0 or int(start.max()) >= resolved_nodes):
        raise ValueError("start entries must satisfy 0 <= start[i] < num_nodes")
    return resolved_nodes


def _sort_coo(row: Tensor, col: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor]:
    """Sort COO edges by ``(row, col)`` without the common int64 AICPU path."""
    if num_nodes <= _MAX_EXACT_FLOAT32_INTEGER:
        # Every node id is exactly representable as float32 in this range.
        # Two stable sorts are equivalent to sorting row * num_nodes + col,
        # while float32 ArgSort runs on AI Core on Ascend 950.
        permutation = torch.argsort(col.to(torch.float32), stable=True)
        row = row[permutation]
        col = col[permutation]
        permutation = torch.argsort(row.to(torch.float32), stable=True)
    else:
        # Preserve exact ordering for unusually large node identifiers.
        permutation = torch.argsort(row * num_nodes + col)
    return row[permutation], col[permutation]


def _tensor_version(tensor: Tensor) -> Optional[int]:
    """Return the mutation token, or disable caching for inference tensors."""
    try:
        return int(getattr(tensor, "_version", 0))
    except RuntimeError:
        return None


def _cache_key(row: Tensor, col: Tensor, num_nodes: int) -> Optional[tuple]:
    row_version, col_version = _tensor_version(row), _tensor_version(col)
    if row_version is None or col_version is None:
        return None
    return (
        row.data_ptr(),
        col.data_ptr(),
        (row_version, col_version, row.numel(), num_nodes, row.device),
    )


def _build_csr(row: Tensor, col: Tensor, num_nodes: int, coalesced: bool) -> Tuple[Tensor, Tensor]:
    row_work, col_work = row.contiguous(), col.contiguous()
    if coalesced and row_work.numel() > 0:
        row_work, col_work = _sort_coo(row_work, col_work, num_nodes)
    degree = row_work.new_zeros(num_nodes)
    if row_work.numel() > 0:
        edge_count = torch.ones(row_work.shape, dtype=row_work.dtype, device=row_work.device)
        degree.scatter_add_(0, row_work, edge_count)
    rowptr = row_work.new_zeros(num_nodes + 1)
    torch.cumsum(degree, 0, out=rowptr[1:])
    return rowptr, col_work


def _get_cached_csr(row: Tensor, col: Tensor, num_nodes: int) -> Optional[Tuple[Tensor, Tensor]]:
    cache = _COALESCED_CSR_CACHE
    if cache is None:
        return None
    row_ref, col_ref, saved_key, rowptr, saved_col = cache
    if row_ref() is row and col_ref() is col and saved_key == _cache_key(row, col, num_nodes):
        return rowptr, saved_col
    return None


def _prepare_csr(row: Tensor, col: Tensor, num_nodes: int, coalesced: bool) -> Tuple[Tensor, Tensor]:
    global _COALESCED_CSR_CACHE
    cacheable = coalesced and _cache_key(row, col, num_nodes) is not None
    if cacheable:
        cached = _get_cached_csr(row, col, num_nodes)
        if cached is not None:
            return cached
    rowptr, col_work = _build_csr(row, col, num_nodes, coalesced)
    if cacheable:
        if _COALESCED_CSR_CACHE is not None:
            torch.npu.synchronize(row.device)
        _COALESCED_CSR_CACHE = (
            weakref.ref(row), weakref.ref(col), _cache_key(row, col, num_nodes), rowptr, col_work,
        )
    return rowptr, col_work


def _parse_arguments(args: tuple, kwargs: dict) -> tuple:
    names = ("row", "col", "start", "walk_length", "p", "q", "coalesced", "num_nodes", "return_edge_indices")
    if len(args) > len(names):
        raise TypeError("random_walk() takes at most 9 positional arguments")
    values = dict(zip(names, args))
    for name, value in kwargs.items():
        if name not in names:
            raise TypeError(f"random_walk() got an unexpected keyword argument '{name}'")
        if name in values:
            raise TypeError(f"random_walk() got multiple values for argument '{name}'")
        values[name] = value
    missing = [name for name in names[:4] if name not in values]
    if missing:
        raise TypeError(f"random_walk() missing required arguments: {', '.join(missing)}")
    values.setdefault("p", 1.0)
    values.setdefault("q", 1.0)
    values.setdefault("coalesced", True)
    values.setdefault("num_nodes", None)
    values.setdefault("return_edge_indices", False)
    return tuple(values[name] for name in names)


def random_walk(*args, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Samples uniform or node2vec-biased walks from a COO graph on NPU.

    The public signature and COO-to-CSR preprocessing follow
    :func:`torch_cluster.random_walk`. When ``coalesced`` is false, edges must
    already be grouped by source node; target nodes do not need to be sorted.
    """
    row, col, start, walk_length, p, q, coalesced, num_nodes, return_edge_indices = _parse_arguments(args, kwargs)
    num_nodes = _validate_inputs(row, col, start, walk_length, num_nodes)
    rowptr, col_work = _prepare_csr(row, col, num_nodes, coalesced)
    start_work = start.contiguous()

    node_seq, edge_seq = _pybind.random_walk(
        rowptr, col_work, start_work, walk_length, float(p), float(q),
        bool(return_edge_indices), bool(coalesced)
    )
    if return_edge_indices:
        return node_seq, edge_seq
    return node_seq

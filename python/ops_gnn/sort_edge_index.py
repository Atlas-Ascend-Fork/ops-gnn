# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public sort_edge_index API."""

import operator
from typing import List, Optional, Tuple, Union

import torch

from .index_sort import (_FLOAT32_EXACT_MAX, _INT64_MAX, _stable_order,
                         _uses_float32)
from .typing import OptTensor, Tensor

_OP_NAME = 'sort_edge_index'

# Same sentinel as PyG: it tells an omitted attribute apart from ``None``.
MISSING = '???'


def _check_edge_index(edge_index: torch.Tensor) -> None:
    """Reject an argument combination the operator cannot serve."""
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f'{_OP_NAME}: edge_index must be a torch.Tensor, '
                        f'got {type(edge_index).__name__}')
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(f'{_OP_NAME}: edge_index must have shape [2, E], '
                         f'got {list(edge_index.shape)}')
    if edge_index.dtype != torch.int64:
        raise TypeError(f'{_OP_NAME}: edge_index dtype must be int64, '
                        f'got {edge_index.dtype}')
    if edge_index.device.type != 'npu':
        raise RuntimeError(f'{_OP_NAME} requires Ascend NPU tensors')


def _check_edge_attr(attribute: torch.Tensor, edges: int,
                     device: torch.device) -> None:
    """Reject a payload that cannot be moved by the permutation."""
    if not isinstance(attribute, torch.Tensor):
        raise TypeError(f'{_OP_NAME}: edge_attr entries must be torch.Tensor, '
                        f'got {type(attribute).__name__}')
    if attribute.dim() == 0 or attribute.shape[0] != edges:
        raise ValueError(f'{_OP_NAME}: edge_attr must hold one row per edge '
                         f'({edges}), got {list(attribute.shape)}')
    if attribute.device != device:
        raise ValueError(f'{_OP_NAME}: edge_attr must live on {device}, '
                         f'got {attribute.device}')


def _resolve_num_nodes(edge_index: torch.Tensor,
                       num_nodes: Optional[int]) -> int:
    """Node count that bounds both keys, read from the device when omitted."""
    if num_nodes is not None:
        nodes = operator.index(num_nodes)
        if nodes < 0:
            raise ValueError(
                f'{_OP_NAME}: num_nodes must be non-negative, got {nodes}')
        return nodes
    if edge_index.numel() == 0:
        return 0
    return int(edge_index.max().item()) + 1


def _edge_order(edge_index: torch.Tensor, num_nodes: int,
                sort_by_row: bool) -> torch.Tensor:
    """Stable permutation that orders the edges by their primary key first.

    One pass on the composite key ``primary * num_nodes + secondary`` is enough
    when that key stays in the exact range of the sort: int64 on 950, the 24-bit
    float32 range on A2/A3. Otherwise the edges are sorted by the secondary key
    and then stably by the primary key, which yields the same order without ever
    leaving the exact range.
    """
    primary = edge_index[0] if sort_by_row else edge_index[1]
    secondary = edge_index[1] if sort_by_row else edge_index[0]
    narrow = _uses_float32()
    limit = _FLOAT32_EXACT_MAX if narrow else _INT64_MAX
    if num_nodes * num_nodes <= limit:
        key = primary * num_nodes + secondary
        return _stable_order(key, num_nodes * num_nodes if narrow else None)
    first = _stable_order(secondary, num_nodes if narrow else None)
    rest = torch.index_select(primary, 0, first)
    second = _stable_order(rest, num_nodes if narrow else None)
    return torch.index_select(first, 0, second)


def sort_edge_index(
    edge_index: Tensor,
    edge_attr: Union[OptTensor, List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, OptTensor], Tuple[Tensor, List[Tensor]]]:
    """Row-wise or column-wise sort of a COO edge list.

    Counterpart of ``torch_geometric.utils.sort_edge_index``: the ``(2, E)``
    edge list is reordered so that the edges are grouped by their source node
    (``sort_by_row=True``) or by their destination node, with the other endpoint
    ascending inside every group. Edge payloads are moved by the same
    permutation, so a graph stored as ``edge_index`` plus ``edge_attr`` stays
    consistent.

    The sort is stable, which the upstream helper does not guarantee: duplicate
    edges keep their input order, so the result is determined by the input alone
    and can be compared element by element, independent of the number of AI
    cores. The operator is a composition of device primitives and launches
    asynchronously on the current NPU stream; an omitted ``num_nodes`` reads one
    scalar from the device and therefore synchronizes.

    Args:
        edge_index (Tensor): ``(2, E)`` ``torch.long`` COO edge list on NPU.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or features
            with one row per edge, or a list of such tensors. Like PyG, omitting
            it returns only the sorted edge list, ``None`` returns
            ``(edge_index, None)`` and a tensor returns it moved by the same
            permutation. A string is treated as an attribute name and ignored.
        num_nodes (int, optional): Number of nodes, i.e. one more than the
            largest id in ``edge_index``. It bounds both sort keys and must not
            be smaller than that. When it is omitted the operator uses
            ``edge_index.max() + 1``, which reads one value from the device.
        sort_by_row (bool, optional): Sort by source node when true, by
            destination node otherwise. (default: ``True``)

    Returns:
        Tensor or Tuple: The sorted edge list, or ``(edge_index, edge_attr)``
        when an attribute payload is passed. The sorted edge list is a plain
        ``torch.long`` tensor on the same NPU as the input; an ``EdgeIndex``
        input loses its subclass metadata, as the upstream helper keeps it.

    Raises:
        TypeError: An argument is not a tensor, has the wrong dtype, or a list
            entry is not a tensor.
        ValueError: ``edge_index`` is not ``(2, E)``, ``num_nodes`` is negative,
            or an attribute does not hold one row per edge on the same device.
        RuntimeError: The inputs are not on an Ascend NPU.

    Example:
        >>> import torch
        >>> import ops_gnn
        >>> edge_index = torch.tensor([[2, 1, 1, 0], [1, 2, 0, 1]],
        ...                           dtype=torch.long, device='npu')
        >>> ops_gnn.sort_edge_index(edge_index)
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]], device='npu:0')
    """
    _check_edge_index(edge_index)
    nodes = _resolve_num_nodes(edge_index, num_nodes)
    order = _edge_order(edge_index, nodes, sort_by_row)
    sorted_edges = torch.index_select(edge_index, 1, order)
    if edge_attr is MISSING:
        return sorted_edges
    if edge_attr is None:
        return sorted_edges, None
    if isinstance(edge_attr, torch.Tensor):
        _check_edge_attr(edge_attr, edge_index.shape[1], edge_index.device)
        return sorted_edges, torch.index_select(edge_attr, 0, order)
    if isinstance(edge_attr, (list, tuple)):
        moved = []
        for attribute in edge_attr:
            _check_edge_attr(attribute, edge_index.shape[1], edge_index.device)
            moved.append(torch.index_select(attribute, 0, order))
        return sorted_edges, moved
    return sorted_edges

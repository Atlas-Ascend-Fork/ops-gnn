# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyG-compatible COO coalescing API for Ascend NPU."""

from typing import Optional, Tuple

from . import _pybind
from .typing import OptTensor, Tensor

_REDUCE_CODES = {"sum": 0, "add": 0, "mean": 1, "min": 2, "max": 3}
_INT64_MAX = 2**63 - 1


def coalesce(
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    num_nodes: Optional[int] = None,
    reduce: str = "sum",
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Tuple[Tensor, OptTensor]:
    """Sort, deduplicate and reduce a COO edge list on Ascend NPU.

    Drop-in equivalent of :func:`torch_geometric.utils.coalesce`. Edges are
    sorted by ``(row, col)`` -- or ``(col, row)`` when ``sort_by_row`` is
    ``False`` -- duplicate edges are merged, and ``edge_attr`` is reduced per
    duplicate group with ``reduce``.

    The sort and duplicate detection run on the NPU; a dedicated Ascend C
    kernel folds each contiguous group of the sorted ``edge_attr`` into one
    output row. The operation is asynchronous on the current NPU stream;
    synchronise before reading the results on the host.

    Args:
        edge_index: ``[2, E]`` int64 NPU tensor holding the COO endpoints.
        edge_attr: Optional ``[E, ...]`` float16/float32 NPU tensor of per-edge
            features. ``None`` deduplicates the edges without a reduction.
        num_nodes: Key radix used to build the sort key. When ``None`` it is
            derived from the largest endpoint (PyG ``maybe_num_nodes``).
        reduce: Reduction for duplicate groups: ``"sum"`` (alias ``"add"``),
            ``"mean"``, ``"min"`` or ``"max"``.
        is_sorted: When ``True`` the input is assumed to be already sorted by
            ``(row, col)`` and the sort step is skipped; duplicates are still
            merged.
        sort_by_row: Sort by rows first (``True``) or by columns first
            (``False``).

    Returns:
        ``(edge_index, edge_attr)`` where ``edge_index`` is the ``[2, U]``
        deduplicated, sorted edge list and ``edge_attr`` is the ``[U, ...]``
        reduced feature tensor (or ``None`` when no ``edge_attr`` was given).

    Example:
        >>> import torch
        >>> import ops_gnn
        >>> edge_index = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 2]], device='npu')
        >>> edge_attr = torch.tensor([1.0, 2.0, 3.0, 4.0], device='npu')
        >>> edge_index, edge_attr = ops_gnn.coalesce(edge_index, edge_attr)
        >>> edge_index.cpu().tolist()
        [[0, 0, 1], [1, 2, 0]]
        >>> edge_attr.cpu().tolist()
        [4.0, 4.0, 2.0]
    """
    if reduce not in _REDUCE_CODES:
        raise ValueError(
            f"unsupported reduce '{reduce}'; expected one of {sorted(_REDUCE_CODES)}"
        )
    if num_nodes is not None:
        if not isinstance(num_nodes, int):
            raise TypeError("num_nodes must be an integer or None")
        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative")
        if num_nodes * num_nodes > _INT64_MAX:
            raise ValueError("num_nodes is too large and would overflow the edge key")

    return _pybind.coalesce(
        edge_index, edge_attr, num_nodes, _REDUCE_CODES[reduce], is_sorted, sort_by_row
    )

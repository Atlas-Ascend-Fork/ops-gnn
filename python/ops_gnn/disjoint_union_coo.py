# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public DisjointUnionCoo API."""

from typing import Optional, Sequence, Tuple

from . import _pybind
from .typing import OptTensor, Tensor


def disjoint_union_coo(
    sources: Sequence[Tensor],
    dests: Sequence[Tensor],
    num_rows: Sequence[int],
    num_cols: Sequence[int],
    datas: Optional[Sequence[OptTensor]] = None,
) -> Tuple[Tensor, Tensor, OptTensor, int, int]:
    """Concatenate COO matrices into one disjoint-union COO matrix.

    Drop-in equivalent of DGL's ``DisjointUnionCoo`` on Ascend NPU, which is
    what ``dgl.batch`` performs on the edge list of a graph batch.  Graph ``i``
    contributes

    * ``sources[i] + sum(num_rows[:i])`` as its destination offsets rows,
    * ``dests[i] + sum(num_cols[:i])`` as its shifted columns, and
    * ``datas[i] + sum(nnz[:i])`` as its shifted data, where ``nnz[j]`` is
      ``sources[j].numel()``.

    A graph whose ``datas`` entry is ``None`` contributes its global edge index
    ``sum(nnz[:i]) + arange(nnz[i])`` instead, matching the implicit
    ``Range(0, nnz)`` DGL materialises for COO matrices without a data array.
    When no graph carries data, the returned data tensor is ``None``.

    Launches asynchronously on the current NPU stream; synchronise before
    reading the results on the host.

    Args:
        sources: List of 1-D int32 or int64 NPU row-index tensors.
        dests: List of 1-D column-index tensors with the same dtype, device and
            length as ``sources``.
        num_rows: Number of rows of each input matrix (used for the row offset).
        num_cols: Number of columns of each input matrix (used for the column
            offset).
        datas: Optional list holding one per-edge data tensor or ``None`` for
            every graph.  Data values are shifted by the accumulated nnz, so an
            int32/int64 tensor of the same dtype as the indices is expected.
            ``None`` (or an empty sequence) means no graph carries data.

    Returns:
        ``(source, dest, data, total_rows, total_cols)`` where ``source``,
        ``dest`` and ``data`` are 1-D NPU tensors of length ``sum(nnz)`` and
        ``data`` is ``None`` when no graph carries one.

    Example:
        >>> import torch
        >>> import ops_gnn
        >>> s = [torch.tensor([0, 1], device='npu'), torch.tensor([0], device='npu')]
        >>> d = [torch.tensor([1, 0], device='npu'), torch.tensor([0], device='npu')]
        >>> e = [torch.tensor([0, 1], device='npu'), torch.tensor([0], device='npu')]
        >>> src, dst, data, rows, cols = ops_gnn.disjoint_union_coo(s, d, [3, 1], [3, 1], e)
        >>> src.cpu().tolist(), dst.cpu().tolist(), data.cpu().tolist()
        ([0, 1, 3], [1, 0, 3], [0, 1, 2])
        >>> ops_gnn.disjoint_union_coo(s, d, [3, 1], [3, 1])[2] is None
        True
    """
    graphs = list(sources)
    dest_list = list(dests)
    row_counts = [int(value) for value in num_rows]
    col_counts = [int(value) for value in num_cols]
    data_list = [] if datas is None else list(datas)
    return _pybind.disjoint_union_coo(graphs, dest_list, row_counts, col_counts, data_list)

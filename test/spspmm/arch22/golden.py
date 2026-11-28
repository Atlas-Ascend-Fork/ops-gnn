# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU reference for ``spspmm``.

Both operands are dense-free COO lists; the reference walks every product in
float64 and keys the accumulators by ``(row, column)``, so duplicate input
coordinates are summed and a coordinate whose products cancel keeps one entry
with value zero. The output is the row-major, column-ascending unique list the
operator must reproduce.
"""

import torch


def _rows(index, values):
    """Group one operand as ``row -> [(column, value), ...]``."""
    rows = {}
    for row, col, value in zip(index[0].tolist(), index[1].tolist(), values):
        rows.setdefault(row, []).append((col, value))
    return rows


def spspmm_golden(index_a, value_a, index_b, value_b, m, k, n):
    """Return ``(index_c, value_c)`` for ``A @ B`` as float64 on the CPU.

    Args:
        index_a, index_b (torch.Tensor): int64 ``[2, nnz]`` COO coordinates.
        value_a, value_b (torch.Tensor): 1-D weights of the same dtype.
        m, k, n (int): Product shape.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: int64 ``[2, nnzC]`` coordinates and
        float64 ``nnzC`` weights. An empty product is returned as ``[2, 0]``.
    """
    for index in (index_a, index_b):
        if index.dim() != 2 or index.size(0) != 2:
            raise ValueError(f"index must have shape [2, nnz], got {tuple(index.shape)}")
    if index_a.size(1) != value_a.numel() or index_b.size(1) != value_b.numel():
        raise ValueError("value length must match its index")
    for row, col in zip(index_a[0].tolist(), index_a[1].tolist()):
        if not 0 <= row < m or not 0 <= col < k:
            raise ValueError(f"A coordinate ({row}, {col}) is outside ({m}, {k})")
    for row, col in zip(index_b[0].tolist(), index_b[1].tolist()):
        if not 0 <= row < k or not 0 <= col < n:
            raise ValueError(f"B coordinate ({row}, {col}) is outside ({k}, {n})")

    left = _rows(index_a, value_a.double().tolist())
    right = _rows(index_b, value_b.double().tolist())

    acc = {}
    for row, entries in left.items():
        for inner, lhs in entries:
            for col, rhs in right.get(inner, ()):
                key = (row, col)
                acc[key] = acc.get(key, 0.0) + lhs * rhs

    keys = sorted(acc)
    index = torch.tensor([[key[0] for key in keys], [key[1] for key in keys]],
                         dtype=torch.int64).reshape(2, len(keys))
    values = torch.tensor([acc.get(key, 0.0) for key in keys], dtype=torch.float64)
    return index, values


def coalesced_rows(index, values):
    """Merge duplicate coordinates so the input satisfies ``coalesced=True``."""
    if index.size(1) == 0:
        return index, values
    merged = {}
    for row, col, value in zip(index[0].tolist(), index[1].tolist(),
                               values.double().tolist()):
        merged[(row, col)] = merged.get((row, col), 0.0) + value
    keys = sorted(merged)
    index = torch.tensor([[key[0] for key in keys], [key[1] for key in keys]],
                         dtype=torch.int64).reshape(2, len(keys))
    return index, torch.tensor([merged.get(key, 0.0) for key in keys], dtype=values.dtype)

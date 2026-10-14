# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent CPU references; no optional torch_sparse package is required."""

import torch


def non_diag_mask_reference(row, col, rows, columns, offset):
    diagonal = max(
        0, min(rows, columns - offset) if offset >= 0 else min(rows + offset, columns)
    )
    flags = [c - r != offset for r, c in zip(row.tolist(), col.tolist())]
    return torch.tensor(flags + [False] * diagonal, dtype=torch.bool)


def random_walk_reference(rowptr, col, starts, length, rand):
    """Shared float32 random samples determine exact CSR neighbor selection."""
    result = torch.full((starts.numel(), length + 1), -1, dtype=torch.int64)
    pointers, columns = rowptr.tolist(), col.tolist()
    for walk, start in enumerate(starts.tolist()):
        current = start
        result[walk, 0] = current
        for step in range(length):
            begin, end = pointers[current], pointers[current + 1]
            if begin == end:
                break
            product = rand[walk, step].float() * torch.tensor(
                end - begin, dtype=torch.float32
            )
            chosen = min(int(product.item()), end - begin - 1)
            current = columns[begin + chosen]
            result[walk, step + 1] = current
    return result


def assert_valid_walk(rowptr, col, starts, result):
    if not torch.equal(result[:, 0], starts):
        raise AssertionError("walk starting nodes differ")
    pointers, columns = rowptr.tolist(), col.tolist()
    for path in result.tolist():
        for current, following in zip(path, path[1:]):
            if current == -1 and following != -1:
                raise AssertionError("walk resumed after a dead end")
            if current == -1:
                continue
            begin, end = pointers[current], pointers[current + 1]
            neighbors = columns[begin:end]
            valid = following in neighbors if neighbors else following == -1
            if not valid:
                raise AssertionError("walk transition is outside the adjacency")


def diagonal_entries_reference(entries, values, offset):
    """Task adapter semantics, including the existing feature-valued None extension."""
    row, col, value, (rows, columns) = entries
    keep = [
        i for i, (r, c) in enumerate(zip(row.tolist(), col.tolist())) if c - r != offset
    ]
    diagonal = [(r, r + offset) for r in range(rows) if 0 <= r + offset < columns]
    pairs = [(int(row[i]), int(col[i])) for i in keep] + diagonal
    order = sorted(range(len(pairs)), key=pairs.__getitem__)
    out_row = torch.tensor([pairs[i][0] for i in order], dtype=torch.int64)
    out_col = torch.tensor([pairs[i][1] for i in order], dtype=torch.int64)
    out_value = None
    if value is not None:
        added = value.new_empty((len(diagonal),) + value.shape[1:])
        if values is None:
            added.fill_(1)
        else:
            added[torch.ones(len(diagonal), dtype=torch.bool)] = values
        out_value = torch.cat((value[keep], added))[order]
    return out_row, out_col, out_value

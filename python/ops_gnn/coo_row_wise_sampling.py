# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Probability/mask-weighted COO row-wise sampling (DGL COORowWiseSampling)."""
__all__ = ["coo_row_wise_sampling"]

from typing import Optional, Tuple

import torch

from .csr_row_wise_sampling import csr_row_wise_sampling

_ID_DTYPES = (torch.int32, torch.int64)


def _check_id(name: str, value: torch.Tensor, ndim: int = 1) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"coo_row_wise_sampling: {name} must be a Tensor")
    if value.device.type != "npu":
        raise ValueError(f"coo_row_wise_sampling: {name} must be an NPU tensor")
    if value.dim() != ndim:
        raise ValueError(f"coo_row_wise_sampling: {name} must be {ndim}-D")
    if value.dtype not in _ID_DTYPES:
        raise ValueError(f"coo_row_wise_sampling: {name} dtype must be int32 or int64")


def _infer_num_rows(row: torch.Tensor, rows: torch.Tensor) -> int:
    num_rows = 0
    if row.numel() > 0:
        num_rows = max(num_rows, int(row.max().item()) + 1)
    if rows.numel() > 0:
        num_rows = max(num_rows, int(rows.max().item()) + 1)
    return num_rows


def _coo_to_csr(row: torch.Tensor, col: torch.Tensor, num_rows: int):
    if row.numel() == 0:
        ptr = torch.zeros(num_rows + 1, dtype=row.dtype, device=row.device)
        return ptr, col.contiguous(), torch.empty(0, dtype=row.dtype, device=row.device)
    # num_rows is derived from row.max() + 1, so only negativity needs a check.
    if int(row.min().item()) < 0:
        raise ValueError("coo_row_wise_sampling: COO row index out of range")
    # Integer argsort runs on AiCpu (slow); row ids below 2^24 are exactly
    # representable in float32, so the stable float32 sort yields the same
    # permutation on AiCore.
    key = row.to(torch.float32) if num_rows < (1 << 24) else row
    perm = torch.argsort(key, stable=True)
    sorted_row = row[perm]
    counts = torch.zeros(num_rows, dtype=torch.int64, device=row.device)
    counts.scatter_add_(0, sorted_row.long(), torch.ones_like(sorted_row, dtype=torch.int64))
    ptr = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=row.device), counts.cumsum(0)
    ]).to(row.dtype)
    return ptr, col[perm].contiguous(), perm.to(row.dtype)


def coo_row_wise_sampling(
    row: torch.Tensor,
    col: torch.Tensor,
    rows: torch.Tensor,
    num_samples: int,
    prob: torch.Tensor,
    replace: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample COO neighbors with probability weights or a boolean mask.

    DGL ``COORowWiseSampling`` parity. The COO graph is converted to a CSR
    (stable order by row, original edge id carried as CSR data) and each
    requested row is sampled with the native weighted CSR sampler; ``prob`` is
    indexed by edge id and edges with ``prob <= 0`` are never picked. The
    returned ``data`` is the original COO edge id.

    ``prob`` supports float32, float64 (cast to float32) and int8/uint8
    (non-zero mask). ``num_samples == -1`` selects every positive edge, ``0``
    returns an empty result, anything else is the fanout. Out-of-range seed rows
    are treated as empty rows.

    Returns ``(row, col, data)`` on ``row``'s dtype and device.
    """
    _check_id("row", row)
    _check_id("col", col)
    _check_id("rows", rows)
    if not isinstance(prob, torch.Tensor):
        raise TypeError("coo_row_wise_sampling: prob must be a Tensor")
    if row.dtype != col.dtype or row.dtype != rows.dtype:
        raise ValueError("coo_row_wise_sampling: row/col/rows dtypes must match")
    if row.device != col.device or row.device != rows.device or row.device != prob.device:
        raise ValueError("coo_row_wise_sampling: inputs must share one NPU device")
    if row.numel() != col.numel():
        raise ValueError("coo_row_wise_sampling: row and col must have equal length")
    if prob.numel() < row.numel():
        raise ValueError("coo_row_wise_sampling: prob must have at least one entry per edge")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError("coo_row_wise_sampling: num_samples must be an int")
    if num_samples < -1:
        raise ValueError("coo_row_wise_sampling: num_samples must be >= -1")
    if not isinstance(replace, bool):
        raise TypeError("coo_row_wise_sampling: replace must be a bool")

    empty = torch.empty(0, dtype=row.dtype, device=row.device)
    if row.numel() == 0 or rows.numel() == 0 or num_samples == 0:
        return empty, empty, empty

    num_rows = _infer_num_rows(row, rows)
    ptr, indices, eid = _coo_to_csr(row, col, num_rows)
    out_row, out_col, out_data = csr_row_wise_sampling(
        ptr, indices, rows, num_samples, prob[:row.numel()], replace, eid, seed)
    return out_row, out_col, out_data

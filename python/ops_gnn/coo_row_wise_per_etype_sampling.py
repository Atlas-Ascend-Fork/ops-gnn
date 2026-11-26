# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Per-edge-type COO row-wise sampling (DGL COORowWisePerEtypeSampling)."""
__all__ = ["coo_row_wise_per_etype_sampling"]

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch

from .csr_row_wise_sampling import csr_row_wise_sampling

_ID_DTYPES = (torch.int32, torch.int64)


@dataclass(frozen=True)
class _SampleContext:
    row: torch.Tensor
    col: torch.Tensor
    rows: torch.Tensor
    offsets: Sequence[int]
    samples: Sequence[int]
    probs: Sequence[Optional[torch.Tensor]]
    replace: bool
    data: Optional[torch.Tensor]
    num_rows: int
    base_seed: int


def _check_id(name: str, value: torch.Tensor, ndim: int = 1) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"coo_row_wise_per_etype_sampling: {name} must be a Tensor")
    if value.device.type != "npu":
        raise ValueError(f"coo_row_wise_per_etype_sampling: {name} must be an NPU tensor")
    if value.dim() != ndim:
        raise ValueError(f"coo_row_wise_per_etype_sampling: {name} must be {ndim}-D")
    if value.dtype not in _ID_DTYPES:
        raise ValueError(f"coo_row_wise_per_etype_sampling: {name} dtype must be int32 or int64")


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
    if int(row.min().item()) < 0 or int(row.max().item()) >= num_rows:
        raise ValueError("coo_row_wise_per_etype_sampling: COO row index out of range")
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


def _resolve_seed(seed: Optional[int]) -> int:
    if seed is None:
        return int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("coo_row_wise_per_etype_sampling: seed must be an int or None")
    if not 0 <= seed < 2**32:
        raise ValueError("coo_row_wise_per_etype_sampling: seed must be in [0, 2**32)")
    return seed


def _parse_offsets(eid2etype_offset, nnz):
    offsets = [int(x) for x in eid2etype_offset]
    if len(offsets) < 2:
        raise ValueError(
            "coo_row_wise_per_etype_sampling: eid2etype_offset needs at least 2 entries")
    if offsets[0] != 0 or offsets[-1] != nnz:
        raise ValueError(
            "coo_row_wise_per_etype_sampling: eid2etype_offset must start at 0 and end at nnz")
    if any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError(
            "coo_row_wise_per_etype_sampling: eid2etype_offset must be nondecreasing")
    return offsets


def _parse_num_samples(num_samples, num_etypes):
    if isinstance(num_samples, bool):
        raise TypeError(
            "coo_row_wise_per_etype_sampling: num_samples must be int or a sequence")
    if isinstance(num_samples, int):
        samples = [num_samples] * num_etypes
    else:
        samples = [int(x) for x in num_samples]
        if len(samples) != num_etypes:
            raise ValueError(
                "coo_row_wise_per_etype_sampling: num_samples length must equal num_etypes")
    if any(x < -1 for x in samples):
        raise ValueError("coo_row_wise_per_etype_sampling: num_samples must be >= -1")
    return samples


def _parse_probs(prob_or_mask, num_etypes):
    if prob_or_mask is None:
        return [None] * num_etypes
    probs = list(prob_or_mask)
    if len(probs) != num_etypes:
        raise ValueError(
            "coo_row_wise_per_etype_sampling: prob_or_mask length must equal num_etypes")
    return probs


def _check_data(data, row):
    if data is None:
        return
    _check_id("data", data)
    if data.dtype != row.dtype or data.device != row.device:
        raise ValueError(
            "coo_row_wise_per_etype_sampling: data must match row dtype and device")
    if data.numel() != row.numel():
        raise ValueError(
            "coo_row_wise_per_etype_sampling: data length must match row")


def _resolve_weight(weight, length, device):
    if weight is None:
        return torch.ones(length, dtype=torch.float32, device=device)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            "coo_row_wise_per_etype_sampling: prob_or_mask entries must be Tensors or None")
    if weight.device != device:
        raise ValueError(
            "coo_row_wise_per_etype_sampling: prob_or_mask entries must share the NPU")
    if weight.numel() != length:
        raise ValueError(
            "coo_row_wise_per_etype_sampling: prob_or_mask entry length must equal its edge count")
    return weight


def _sample_etype(ctx: _SampleContext, e: int):
    off = ctx.offsets[e]
    length = ctx.offsets[e + 1] - off
    if length == 0 or ctx.samples[e] == 0:
        return None
    win_row = ctx.row[off:off + length]
    win_col = ctx.col[off:off + length]
    ptr, indices, local_eid = _coo_to_csr(win_row, win_col, ctx.num_rows)
    weight = _resolve_weight(ctx.probs[e], length, ctx.row.device)
    part_row, part_col, part_local = csr_row_wise_sampling(
        ptr, indices, ctx.rows, ctx.samples[e], weight, ctx.replace, local_eid,
        (ctx.base_seed + e) % (2 ** 32))
    if part_row.numel() == 0:
        return None
    global_eid = part_local + off
    part_data = global_eid if ctx.data is None else ctx.data.index_select(0, global_eid)
    return part_row, part_col, part_data


def coo_row_wise_per_etype_sampling(
    row: torch.Tensor,
    col: torch.Tensor,
    rows: torch.Tensor,
    eid2etype_offset: Sequence[int],
    num_samples: Union[int, Sequence[int]],
    prob_or_mask: Optional[Sequence[Optional[torch.Tensor]]] = None,
    replace: bool = False,
    data: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample COO neighbors per edge type with probability weights or masks.

    DGL ``COORowWisePerEtypeSampling`` parity. Edge ids are grouped by edge type
    through ``eid2etype_offset`` (``num_etypes + 1`` nondecreasing boundaries
    with ``offset[0] == 0`` and ``offset[-1] == nnz``). Each edge type's window
    is converted to a CSR slice and sampled independently with the native
    weighted CSR sampler using the edge type's own probability/mask array. A
    ``None`` entry (or ``prob_or_mask=None``) makes that edge type uniform.

    ``num_samples`` is one fanout for every edge type or one value per edge type:
    ``-1`` selects every edge, ``0`` skips the edge type, anything else is the
    fanout. Out-of-range seed rows are treated as empty rows.

    Returns ``(row, col, data)`` grouped by edge type (etype-major). ``data``
    holds the COO data value when supplied, otherwise the global edge id.
    """
    _check_id("row", row)
    _check_id("col", col)
    _check_id("rows", rows)
    if row.dtype != col.dtype or row.dtype != rows.dtype:
        raise ValueError("coo_row_wise_per_etype_sampling: row/col/rows dtypes must match")
    if row.device != col.device or row.device != rows.device:
        raise ValueError("coo_row_wise_per_etype_sampling: inputs must share one NPU device")
    if row.numel() != col.numel():
        raise ValueError("coo_row_wise_per_etype_sampling: row and col must have equal length")

    offsets = _parse_offsets(eid2etype_offset, row.numel())
    num_etypes = len(offsets) - 1
    samples = _parse_num_samples(num_samples, num_etypes)
    probs = _parse_probs(prob_or_mask, num_etypes)
    if not isinstance(replace, bool):
        raise TypeError("coo_row_wise_per_etype_sampling: replace must be a bool")
    _check_data(data, row)

    empty = torch.empty(0, dtype=row.dtype, device=row.device)
    if row.numel() == 0 or rows.numel() == 0:
        return empty, empty, empty

    num_rows = _infer_num_rows(row, rows)
    base_seed = _resolve_seed(seed)
    ctx = _SampleContext(row, col, rows, offsets, samples, probs, replace, data,
                         num_rows, base_seed)
    picked_rows, picked_cols, picked_data = [], [], []
    for e in range(num_etypes):
        part = _sample_etype(ctx, e)
        if part is None:
            continue
        picked_rows.append(part[0])
        picked_cols.append(part[1])
        picked_data.append(part[2])

    if not picked_rows:
        return empty, empty, empty
    return (torch.cat(picked_rows), torch.cat(picked_cols), torch.cat(picked_data))

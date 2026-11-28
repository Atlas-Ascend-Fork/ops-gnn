# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Per-edge-type uniform row-wise sampling on a COO graph."""
__all__ = ["csr_row_wise_per_etype_sampling_uniform"]

from typing import Optional, Sequence, Tuple, Union

import torch

from . import _pybind

_ID_DTYPES = (torch.int32, torch.int64)


def _check_id(name: str, value: torch.Tensor, ndim: int = 1) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"csr_row_wise_per_etype_sampling_uniform: {name} must be a Tensor")
    if value.device.type != "npu":
        raise ValueError(f"csr_row_wise_per_etype_sampling_uniform: {name} must be an NPU tensor")
    if value.dim() != ndim:
        raise ValueError(f"csr_row_wise_per_etype_sampling_uniform: {name} must be {ndim}-D")
    if value.dtype not in _ID_DTYPES:
        raise ValueError(f"csr_row_wise_per_etype_sampling_uniform: {name} dtype must be int32 or int64")


def _resolve_seed(seed: Optional[int]) -> int:
    if seed is None:
        return int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("csr_row_wise_per_etype_sampling_uniform: seed must be an int or None")
    if not 0 <= seed < 2**32:
        raise ValueError("csr_row_wise_per_etype_sampling_uniform: seed must be in [0, 2**32)")
    return seed


def _parse_offsets(eid2etype_offset, nnz):
    offsets = [int(x) for x in eid2etype_offset]
    if len(offsets) < 2:
        raise ValueError(
            "csr_row_wise_per_etype_sampling_uniform: eid2etype_offset needs at least 2 entries")
    if offsets[0] != 0 or offsets[-1] != nnz:
        raise ValueError(
            "csr_row_wise_per_etype_sampling_uniform: eid2etype_offset must start at 0 and end at nnz")
    if any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError(
            "csr_row_wise_per_etype_sampling_uniform: eid2etype_offset must be nondecreasing")
    return offsets


def _parse_samples(num_samples, num_etypes):
    if isinstance(num_samples, bool):
        raise TypeError(
            "csr_row_wise_per_etype_sampling_uniform: num_samples must be int or a sequence")
    if isinstance(num_samples, int):
        samples = [num_samples] * num_etypes
    else:
        samples = [int(x) for x in num_samples]
        if len(samples) != num_etypes:
            raise ValueError(
                "csr_row_wise_per_etype_sampling_uniform: num_samples length must equal num_etypes")
    if any(x < -1 for x in samples):
        raise ValueError("csr_row_wise_per_etype_sampling_uniform: num_samples must be >= -1")
    return samples


def _check_data(data, row):
    if data is None:
        return
    _check_id("data", data)
    if data.dtype != row.dtype or data.device != row.device:
        raise ValueError(
            "csr_row_wise_per_etype_sampling_uniform: data must match row dtype and device")
    if data.numel() != row.numel():
        raise ValueError("csr_row_wise_per_etype_sampling_uniform: data length must match row")


def csr_row_wise_per_etype_sampling_uniform(
    row: torch.Tensor,
    col: torch.Tensor,
    rows: torch.Tensor,
    eid2etype_offset: Sequence[int],
    num_samples: Union[int, Sequence[int]],
    replace: bool = False,
    data: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample neighbors uniformly per edge type from a COO graph.

    DGL ``CSRRowWisePerEtypeSamplingUniform`` parity. ``row``/``col`` describe a
    COO graph whose edge ids are ordered by edge type; ``eid2etype_offset`` holds
    ``num_etypes + 1`` nondecreasing boundaries with ``offset[0] == 0`` and
    ``offset[-1] == row.numel()``. ``rows`` holds the seed vertex ids.

    ``num_samples`` is either one fanout for every edge type or one value per
    edge type: ``-1`` selects every edge of the row, ``0`` skips the edge type,
    and any other value is the fanout. Without replacement the actual count is
    ``min(fanout, degree)``; with replacement a non-empty row returns ``fanout``
    edges. Out-of-range seed rows are treated as empty rows.

    Returns ``(row, col, data)`` for the sampled COO edges, grouped by edge type
    (etype-major). ``data`` holds the COO data value when supplied, otherwise the
    global edge id. All tensors use ``row``'s dtype and device.
    """
    _check_id("row", row)
    _check_id("col", col)
    _check_id("rows", rows)
    if row.dtype != col.dtype or row.dtype != rows.dtype:
        raise ValueError("csr_row_wise_per_etype_sampling_uniform: row/col/rows dtypes must match")
    if row.device != col.device or row.device != rows.device:
        raise ValueError("csr_row_wise_per_etype_sampling_uniform: inputs must share one NPU device")
    if row.numel() != col.numel():
        raise ValueError("csr_row_wise_per_etype_sampling_uniform: row and col must have equal length")
    offsets = _parse_offsets(eid2etype_offset, row.numel())
    num_etypes = len(offsets) - 1
    samples = _parse_samples(num_samples, num_etypes)
    if not isinstance(replace, bool):
        raise TypeError("csr_row_wise_per_etype_sampling_uniform: replace must be a bool")
    _check_data(data, row)

    return _pybind.csr_row_wise_per_etype_sampling_uniform(
        row.contiguous(), col.contiguous(), rows.contiguous(),
        offsets, samples, replace,
        None if data is None else data.contiguous(), _resolve_seed(seed))

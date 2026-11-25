# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Uniform row-wise sampling on a CSR graph (DGL CSRRowWiseSamplingUniform)."""
__all__ = ["csr_row_wise_sampling_uniform_fused"]

from typing import Optional, Tuple

import torch

from . import _pybind

_ID_DTYPES = (torch.int32, torch.int64)


def _check_id(name: str, value: torch.Tensor, ndim: int = 1) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"csr_row_wise_sampling_uniform_fused: {name} must be a Tensor")
    if value.device.type != "npu":
        raise ValueError(f"csr_row_wise_sampling_uniform_fused: {name} must be an NPU tensor")
    if value.dim() != ndim:
        raise ValueError(f"csr_row_wise_sampling_uniform_fused: {name} must be {ndim}-D")
    if value.dtype not in _ID_DTYPES:
        raise ValueError(f"csr_row_wise_sampling_uniform_fused: {name} dtype must be int32 or int64")


def _resolve_seed(seed: Optional[int]) -> int:
    if seed is None:
        return int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("csr_row_wise_sampling_uniform_fused: seed must be an int or None")
    if not 0 <= seed < 2**32:
        raise ValueError("csr_row_wise_sampling_uniform_fused: seed must be in [0, 2**32)")
    return seed


def csr_row_wise_sampling_uniform_fused(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    rows: torch.Tensor,
    num_samples: int,
    replace: bool = False,
    data: Optional[torch.Tensor] = None,
    seed_mapping: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Uniform CSR row-wise sampling fused with seed-node mapping.

    DGL ``CSRRowWiseSamplingUniformFused`` parity. ``indptr``/``indices`` describe
    a CSR matrix; ``rows`` holds the seed rows. ``num_samples == -1`` selects
    every edge of each row, ``0`` returns an empty result, and any other value is
    the fanout. Without replacement the actual count is ``min(fanout, degree)``;
    with replacement a non-empty row returns ``fanout`` edges. Invalid seed rows
    are treated as empty rows.

    Returns ``(block_indptr, picked_rows, picked_cols, picked_data,
    seed_mapping, new_seed_nodes)``:

    - ``block_indptr``: length ``rows.numel() + 1`` CSR pointer of the sampled
      result (the global prefix of per-row pick counts).
    - ``picked_rows``: **row position** in ``rows`` (not the vertex id).
    - ``picked_cols`` / ``picked_data``: sampled neighbors and CSR data values
      (global edge id when ``data`` is omitted).
    - ``seed_mapping``: when supplied, updated in place so that
      ``seed_mapping[rows[i]] = i`` for valid rows; otherwise ``None``.
    - ``new_seed_nodes``: ``rows.clone()`` when ``seed_mapping`` is supplied,
      otherwise ``None``.

    All tensors use ``indptr``'s dtype and device.
    """
    _check_id("indptr", indptr)
    _check_id("indices", indices)
    _check_id("rows", rows)
    if indptr.dtype != indices.dtype or indptr.dtype != rows.dtype:
        raise ValueError("csr_row_wise_sampling_uniform_fused: indptr/indices/rows dtypes must match")
    if indptr.device != indices.device or indptr.device != rows.device:
        raise ValueError("csr_row_wise_sampling_uniform_fused: inputs must share one NPU device")
    if indptr.numel() == 0:
        raise ValueError("csr_row_wise_sampling_uniform_fused: indptr must not be empty")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError("csr_row_wise_sampling_uniform_fused: num_samples must be an int")
    if num_samples < -1:
        raise ValueError("csr_row_wise_sampling_uniform_fused: num_samples must be >= -1")
    if not isinstance(replace, bool):
        raise TypeError("csr_row_wise_sampling_uniform_fused: replace must be a bool")
    if data is not None:
        _check_id("data", data)
        if data.dtype != indptr.dtype or data.numel() != indices.numel():
            raise ValueError(
                "csr_row_wise_sampling_uniform_fused: data must match indptr dtype and indices length")
    if seed_mapping is not None:
        _check_id("seed_mapping", seed_mapping)
        if seed_mapping.dtype != indptr.dtype or seed_mapping.device != indptr.device:
            raise ValueError(
                "csr_row_wise_sampling_uniform_fused: seed_mapping must match indptr dtype and device")
        if seed_mapping.numel() < indptr.numel() - 1:
            raise ValueError(
                "csr_row_wise_sampling_uniform_fused: seed_mapping is shorter than the graph rows")

    return _pybind.csr_row_wise_sampling_uniform_fused(
        indptr.contiguous(), indices.contiguous(), rows.contiguous(),
        num_samples, replace,
        None if data is None else data.contiguous(),
        seed_mapping, _resolve_seed(seed))

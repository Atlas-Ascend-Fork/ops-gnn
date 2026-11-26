# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Weighted (probability/mask) row-wise sampling on a CSR graph."""
__all__ = ["csr_row_wise_sampling"]

from typing import Optional, Tuple

import torch

from . import _pybind

_ID_DTYPES = (torch.int32, torch.int64)


def _check_id(name: str, value: torch.Tensor, ndim: int = 1) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"csr_row_wise_sampling: {name} must be a Tensor")
    if value.device.type != "npu":
        raise ValueError(f"csr_row_wise_sampling: {name} must be an NPU tensor")
    if value.dim() != ndim:
        raise ValueError(f"csr_row_wise_sampling: {name} must be {ndim}-D")
    if value.dtype not in _ID_DTYPES:
        raise ValueError(f"csr_row_wise_sampling: {name} dtype must be int32 or int64")


def _resolve_seed(seed: Optional[int]) -> int:
    if seed is None:
        return int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("csr_row_wise_sampling: seed must be an int or None")
    if not 0 <= seed < 2**32:
        raise ValueError("csr_row_wise_sampling: seed must be in [0, 2**32)")
    return seed


def _normalize_prob(prob: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if not isinstance(prob, torch.Tensor):
        raise TypeError("csr_row_wise_sampling: prob must be a Tensor")
    if prob.device != indices.device:
        raise ValueError("csr_row_wise_sampling: prob must be on the same NPU as indices")
    if prob.dim() != 1:
        raise ValueError("csr_row_wise_sampling: prob must be 1-D")
    if prob.numel() != indices.numel():
        raise ValueError("csr_row_wise_sampling: prob length must match indices")
    if prob.dtype == torch.float32:
        return prob.contiguous()
    if prob.dtype == torch.float64:
        return prob.to(torch.float32).contiguous()
    if prob.dtype in (torch.int8, torch.uint8):
        return (prob != 0).to(torch.float32).contiguous()
    raise ValueError(
        "csr_row_wise_sampling: prob dtype must be float32, float64, int8 or uint8")


def csr_row_wise_sampling(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    rows: torch.Tensor,
    num_samples: int,
    prob: torch.Tensor,
    replace: bool = False,
    data: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample CSR neighbors with probability weights or a boolean mask.

    DGL ``CSRRowWiseSampling`` parity. For every row in ``rows``, edges are
    drawn proportionally to ``prob``; edges with ``prob <= 0`` are never picked.
    ``int8``/``uint8`` inputs are treated as boolean masks (non-zero -> 1.0) and
    ``float64`` is evaluated in float32.

    ``num_samples == -1`` selects every edge with ``prob > 0`` in CSR order,
    ``0`` returns an empty result, and any other value is the fanout. Without
    replacement the actual count is ``min(fanout, #positive)``; with replacement
    a non-empty row returns ``fanout`` edges. Out-of-range seed rows are empty.

    Returns ``(row, col, data)``: sampled COO triples. ``data`` holds the CSR
    data value when supplied, otherwise the global edge id. All tensors use
    ``indptr``'s dtype and device.
    """
    _check_id("indptr", indptr)
    _check_id("indices", indices)
    _check_id("rows", rows)
    if indptr.dtype != indices.dtype or indptr.dtype != rows.dtype:
        raise ValueError("csr_row_wise_sampling: indptr/indices/rows dtypes must match")
    if indptr.device != indices.device or indptr.device != rows.device:
        raise ValueError("csr_row_wise_sampling: inputs must share one NPU device")
    if indptr.numel() == 0:
        raise ValueError("csr_row_wise_sampling: indptr must not be empty")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError("csr_row_wise_sampling: num_samples must be an int")
    if num_samples < -1:
        raise ValueError("csr_row_wise_sampling: num_samples must be >= -1")
    if not isinstance(replace, bool):
        raise TypeError("csr_row_wise_sampling: replace must be a bool")
    if data is not None:
        _check_id("data", data)
        if data.dtype != indptr.dtype or data.device != indptr.device:
            raise ValueError("csr_row_wise_sampling: data must match indptr dtype and device")
        if data.numel() != indices.numel():
            raise ValueError("csr_row_wise_sampling: data length must match indices")

    weights = _normalize_prob(prob, indices)
    return _pybind.csr_row_wise_sampling(
        indptr.contiguous(), indices.contiguous(), rows.contiguous(),
        num_samples, replace,
        None if data is None else data.contiguous(), weights, _resolve_seed(seed))

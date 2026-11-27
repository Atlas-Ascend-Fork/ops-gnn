# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Uniform row-wise sampling on a CSR graph (DGL CSRRowWiseSamplingUniform)."""
__all__ = ["csr_row_wise_sampling_uniform"]

from typing import Optional, Tuple

import torch

from . import _pybind

_ID_DTYPES = (torch.int32, torch.int64)


def _check_id(name: str, value: torch.Tensor, ndim: int = 1) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"csr_row_wise_sampling_uniform: {name} must be a Tensor")
    if value.device.type != "npu":
        raise ValueError(f"csr_row_wise_sampling_uniform: {name} must be an NPU tensor")
    if value.dim() != ndim:
        raise ValueError(f"csr_row_wise_sampling_uniform: {name} must be {ndim}-D")
    if value.dtype not in _ID_DTYPES:
        raise ValueError(f"csr_row_wise_sampling_uniform: {name} dtype must be int32 or int64")


def _resolve_seed(seed: Optional[int]) -> int:
    if seed is None:
        return int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("csr_row_wise_sampling_uniform: seed must be an int or None")
    if not 0 <= seed < 2**32:
        raise ValueError("csr_row_wise_sampling_uniform: seed must be in [0, 2**32)")
    return seed


def csr_row_wise_sampling_uniform(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    rows: torch.Tensor,
    num_samples: int,
    replace: bool = False,
    data: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample neighbors uniformly from the requested CSR rows.

    ``indptr``/``indices`` describe a CSR matrix; ``rows`` holds the seed rows.
    ``num_samples == -1`` selects every edge of each row, ``0`` returns an empty
    result, and any other value is the fanout. Without replacement the actual
    pick count is ``min(fanout, degree)``; with replacement it is ``fanout`` for
    every non-empty row. Invalid seed rows are treated as empty rows.

    Returns ``(row, col, data, out_ptr)``: sampled edge triples (``data`` is the
    CSR data value when supplied, otherwise the global edge id) and the CSR
    pointer of the sampled rows. All tensors use ``indptr``'s dtype and device.
    """
    _check_id("indptr", indptr)
    _check_id("indices", indices)
    _check_id("rows", rows)
    if indptr.dtype != indices.dtype or indptr.dtype != rows.dtype:
        raise ValueError("csr_row_wise_sampling_uniform: indptr/indices/rows dtypes must match")
    if indptr.device != indices.device or indptr.device != rows.device:
        raise ValueError("csr_row_wise_sampling_uniform: inputs must share one NPU device")
    if indptr.numel() == 0:
        raise ValueError("csr_row_wise_sampling_uniform: indptr must not be empty")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError("csr_row_wise_sampling_uniform: num_samples must be an int")
    if num_samples < -1:
        raise ValueError("csr_row_wise_sampling_uniform: num_samples must be >= -1")
    if not isinstance(replace, bool):
        raise TypeError("csr_row_wise_sampling_uniform: replace must be a bool")
    if data is not None:
        _check_id("data", data)
        if data.dtype != indptr.dtype or data.numel() != indices.numel():
            raise ValueError(
                "csr_row_wise_sampling_uniform: data must match indptr dtype and indices length")

    # Uniform sampling is the fused sampler without seed mapping. Reusing it
    # keeps one arch22 kernel instead of maintaining a duplicate implementation.
    rows_contig = rows.contiguous()
    out_ptr, positions, out_cols, out_data, _, _ = \
        _pybind.csr_row_wise_sampling_uniform_fused(
            indptr.contiguous(), indices.contiguous(), rows_contig,
            num_samples, replace,
            None if data is None else data.contiguous(), None, _resolve_seed(seed))
    row_ids = rows_contig.to(torch.long).index_select(
        0, positions.to(torch.long)).to(rows.dtype)
    return row_ids, out_cols, out_data, out_ptr

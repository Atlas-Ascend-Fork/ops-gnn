# Public interface signatures are adapted from torch_scatter 2.1.2 (MIT).
# Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
# The upstream permission notice is retained in the root LICENSE third-party notices.
# NPU bindings and architecture-specific semantics are CANN additions under the license below.
# Copyright (c) 2026.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Sorted COO reductions, forward only on NPU.

A2 follows CPU forward seed, strict-extremum and argument-state rules. Its
index can broadcast in leading dimensions and in the reduction dimension.
Byte mean explicitly returns zero when a nonempty count narrows to zero.
Ascend 950 retains its existing task semantics; see the architecture notes
in the API reference. No backward implementation is provided here.
"""

from typing import Optional, Tuple
import torch
from . import _pybind

# Arch22 has a Tensor-only binding so the value API need not allocate a Python
# tuple and discard its undefined arg. Arch35 keeps its existing dispatch.
_VALUE_FORWARD = getattr(_pybind, "segment_coo_value", None)
_REDUCE_CODES = {"sum": _pybind.SEGMENT_COO_SUM, "add": _pybind.SEGMENT_COO_SUM,
                 "mean": _pybind.SEGMENT_COO_MEAN, "min": _pybind.SEGMENT_COO_MIN,
                 "max": _pybind.SEGMENT_COO_MAX}


def segment_sum_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Sum segments; A2 accumulates supplied out, while 950 overwrites it."""
    if _VALUE_FORWARD is not None:
        return _VALUE_FORWARD(src, index, out, dim_size, _REDUCE_CODES["sum"])
    return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_SUM)[0]


def segment_add_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Alias for segment_sum_coo with the same output-buffer semantics."""
    return segment_sum_coo(src, index, out, dim_size)


def segment_mean_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute segment means; integer division truncates toward zero.

    A2 narrows the numerator and count to the source dtype before division;
    a nonempty byte count that narrows to zero produces zero explicitly.
    """
    if _VALUE_FORWARD is not None:
        return _VALUE_FORWARD(src, index, out, dim_size, _REDUCE_CODES["mean"])
    return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_MEAN)[0]


def segment_min_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return minima and int64 positions; A2 updates only on strict improvement.

    A2 argument state starts at zero per feature and carries across nonempty
    segments and batches; empty segments return the reduction-length sentinel.
    The existing 950 implementation keeps its later-tie convention.
    """
    return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_MIN)


def segment_max_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return maxima and int64 positions with the same rules as segment_min_coo."""
    return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_MAX)


def segment_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Reduce along index.dim() - 1; return values only for every reduction.

    Index must contain sorted, in-range int64 segment identifiers. See the
    API reference in docs/zh/api_reference.md for broadcasting and output buffers.
    On A2 all reductions use supplied initial values. The first nonempty segment
    of each batch uses out[0], even if its label is nonzero; later segments use
    their own seeds. Empty segments and empty inputs preserve supplied out.
    Without out, extrema use finite dtype identities and clear unchanged
    identities to zero. These are forward compatibility rules, not a backward
    guarantee for the returned argument state.
    """
    if _VALUE_FORWARD is not None:
        try:
            code = _REDUCE_CODES[reduce]
        except (KeyError, TypeError):
            raise ValueError("segment_coo supports sum/add/mean/min/max, not " + str(reduce)) from None
        return _VALUE_FORWARD(src, index, out, dim_size, code)
    if reduce in ("sum", "add"):
        return segment_sum_coo(src, index, out, dim_size)
    if reduce == "mean":
        return segment_mean_coo(src, index, out, dim_size)
    if reduce == "min":
        return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_MIN, False)[0]
    if reduce == "max":
        return _pybind.segment_coo_forward(src, index, out, dim_size, _pybind.SEGMENT_COO_MAX, False)[0]
    raise ValueError("segment_coo supports sum/add/mean/min/max, not " + str(reduce))

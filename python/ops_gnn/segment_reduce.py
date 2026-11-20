# Copyright (c) 2026 Starlink_. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""

DGL-compatible ``segment_reduce`` on Ascend NPU.

This is a thin wrapper over :func:`ops_gnn.segment_csr`. It exists for callers
that follow DGL's segment reduction contract,

    out[i] = reduce(src[offsets[i]:offsets[i + 1]])

with an offset (CSR pointer) vector. The only work performed here is promoting
an ``int32`` offset vector, which DGL uses as its ascending index type, to the
``int64`` pointers required by ``segment_csr``. Segment boundary values,
reduction semantics and empty-segment results are exactly those of
``segment_csr``.
"""

from typing import Optional

import torch

from .segment_csr import segment_csr


def segment_reduce(
    src: torch.Tensor,
    offsets: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Reduce segments delimited by CSR ``offsets`` and return values only.

    Parameters
    ----------
    src : torch.Tensor
        NPU tensor whose reduction axis is ``offsets.dim() - 1``.
    offsets : torch.Tensor
        Segment offsets on the same device; ``int32`` or ``int64``, nondecreasing
        along the last dimension, with ``offsets.size(-1) - 1`` segments.
    out : torch.Tensor, optional
        Optional same-device output buffer, forwarded to ``segment_csr``.
    reduce : str
        One of ``sum``, ``add``, ``mean``, ``min``, ``max``.

    Returns
    -------
    torch.Tensor
        Aggregated values. ``min``/``max`` indices are not returned; use
        ``segment_min_csr`` / ``segment_max_csr`` when they are needed.

    Notes
    -----
    ``src`` and ``offsets`` follow the same dtype and shape contracts as
    ``segment_csr``. Calling this function is equivalent to promoting ``offsets``
    to ``int64`` and calling ``segment_csr(src, offsets, out, reduce)``.
    """
    if offsets.dtype != torch.int64:
        offsets = offsets.to(torch.int64)
    return segment_csr(src, offsets, out, reduce)

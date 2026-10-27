# Copyright (c) 2026 Starlink_. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""

torch_scatter-compatible segment_csr family on Ascend NPU.
Signatures are aligned 1:1 with torch_scatter.segment_csr (>= 2.1.0);
there is no dim_size parameter (CSR pointers define the segment count).
"""

from typing import Optional, Tuple

import torch

from . import _pybind
if hasattr(_pybind, "segment_max_csr"):
    from .segment_max_csr import segment_max_csr as segment_max_csr_legacy

_REDUCE_OPS = ("sum", "add", "mean", "min", "max")


def _check_reduce(reduce: str) -> None:
    if reduce not in _REDUCE_OPS:
        raise ValueError(
            f"reduce value '{reduce}' is not valid "
            f"(use one of {_REDUCE_OPS})"
        )


def segment_sum_csr(
    src: torch.Tensor, indptr: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    _check_reduce("sum")
    return _pybind.segment_csr_value(src, indptr, out, "sum")


def segment_add_csr(
    src: torch.Tensor, indptr: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    _check_reduce("add")
    return _pybind.segment_csr_value(src, indptr, out, "add")


def segment_mean_csr(
    src: torch.Tensor, indptr: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    _check_reduce("mean")
    return _pybind.segment_csr_value(src, indptr, out, "mean")


def segment_min_csr(
    src: torch.Tensor, indptr: torch.Tensor, out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_reduce("min")
    result = _pybind.segment_csr(src, indptr, out, "min")
    return result[0], result[1]


def segment_max_csr(
    src: torch.Tensor, indptr: torch.Tensor, out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return values and first indices for int64 CSR pointers.

    Int32 pointers retain the repository's legacy value-only implementation,
    including its initial-output and empty-segment behavior. The historical
    ``optional_out`` keyword is available through ``segment_max_csr_legacy``.
    """
    _check_reduce("max")
    if indptr.dtype == torch.int32 and hasattr(_pybind, "segment_max_csr"):
        return segment_max_csr_legacy(src, indptr, out)
    result = _pybind.segment_csr(src, indptr, out, "max")
    return result[0], result[1]


def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Reduce CSR segments along ``dim = indptr.dim() - 1``.

    Return only values; min/max indices are provided by the sub-operators.
    """
    _check_reduce(reduce)
    return _pybind.segment_csr_value(src, indptr, out, reduce)

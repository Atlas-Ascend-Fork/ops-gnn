# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Numerically stable log-sum-exp reduction over CSR segments.

``segment_logsumexp`` follows the CSR convention of the ``segment_csr``
family: ``indptr`` is a nondecreasing pointer vector, the number of segments
is ``indptr.numel() - 1`` and ``src`` has ``indptr[-1]`` rows. For segment
``i`` the result is::

    out[i] = log(sum(exp(src[j])))  for j in [indptr[i], indptr[i + 1])

The operator has a dedicated AscendC kernel
(``csrc/npu/segment_logsumexp``) shared by arch22 (A2/A3) and arch35 (950). A
CSR segment is a contiguous row range, so the kernel reduces one
(segment, feature window) task per core step in three stages: the per-feature
maximum of the segment, then ``sum(exp(src - maximum))``, then
``maximum + log(sum)``. When there are fewer segments than AI cores, the rows
of a segment are additionally cut into chunks and reduced in two levels (chunk
maximum, reduced shift, chunk sum, result) so that every core contributes.

Every stage runs in float32, so a ``float16``/``bfloat16`` segment is not
limited by the source dtype range (``exp(src - max) <= 1`` would otherwise
overflow above 65504 elements). Empty segments evaluate to ``-inf``, and rows
before ``indptr[0]`` belong to no segment and are ignored.
"""

from typing import Optional

import torch

from . import _pybind

_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_SUPPORTED_PTR_DTYPES = (torch.int32, torch.int64)

# The device kernel reads the group boundaries as int32 scalars, so a pointer
# that does not fit is rejected instead of being silently truncated.
_MAX_POINTER = 2 ** 31 - 1


def _check_inputs(src: torch.Tensor, indptr: torch.Tensor) -> None:
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"src must be a torch.Tensor, got {type(src).__name__}")
    if not isinstance(indptr, torch.Tensor):
        raise TypeError(
            f"indptr must be a torch.Tensor, got {type(indptr).__name__}")
    if src.dim() < 1:
        raise ValueError("src must have at least one dimension")
    if indptr.dim() != 1:
        raise ValueError(f"indptr must be 1-D, got {indptr.dim()}-D")
    if src.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "src dtype must be one of float16, float32, bfloat16; "
            f"got {src.dtype}")
    if indptr.dtype not in _SUPPORTED_PTR_DTYPES:
        raise TypeError(
            f"indptr dtype must be int32 or int64, got {indptr.dtype}")
    if src.device != indptr.device:
        raise ValueError(
            f"src and indptr must be on the same device, got "
            f"{src.device} and {indptr.device}")
    if src.device.type != "npu":
        raise RuntimeError("segment_logsumexp requires Ascend NPU tensors")
    if indptr.numel() < 1:
        raise ValueError("indptr must contain at least one boundary")
    # The three range checks share one device pass and one copy back. A host
    # round trip costs a large fraction of a millisecond on some platforms, so
    # three of them would dominate the runtime of a small shape.
    flags = torch.stack((
        (indptr < 0).any().to(torch.int64),
        (indptr[1:] < indptr[:-1]).any().to(torch.int64),
        indptr[-1].to(torch.int64),
    )).cpu()
    if int(flags[0]) != 0:
        raise ValueError("indptr must be non-negative")
    if int(flags[1]) != 0:
        raise ValueError("indptr must be nondecreasing")
    last = int(flags[2])
    if last != src.size(0):
        raise ValueError(
            f"indptr[-1] ({last}) must equal src.size(0) ({src.size(0)})")
    if src.size(0) > _MAX_POINTER:
        raise ValueError(
            f"src.size(0) must fit the int32 pointer range, got "
            f"{src.size(0)}")


def _check_out(out: torch.Tensor, shape: tuple, src: torch.Tensor) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"out must be a torch.Tensor, got {type(out).__name__}")
    if tuple(out.shape) != shape:
        raise ValueError(
            f"out shape must be {shape}, got {tuple(out.shape)}")
    if out.dtype != src.dtype:
        raise TypeError(
            f"out dtype must match src ({src.dtype}), got {out.dtype}")
    if out.device != src.device:
        raise ValueError(
            f"out device must match src ({src.device}), got {out.device}")


def segment_logsumexp(
    src: torch.Tensor,
    indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``log(sum(exp(src)))`` over CSR segments on Ascend NPU.

    Args:
        src: Source tensor of shape ``(indptr[-1], ...)`` on NPU. Supported
            dtypes are ``float16``, ``float32`` and ``bfloat16``.
        indptr: 1-D ``int32``/``int64`` CSR pointer tensor on the same NPU.
            Must be non-negative, nondecreasing and end at ``src.size(0)``.
            Rows before ``indptr[0]`` belong to no segment and are ignored.
        out: Optional preallocated output of shape
            ``(indptr.numel() - 1, ...)`` with the same dtype/device as
            ``src``. When supplied, the result is written into ``out`` and
            ``out`` is returned.

    Returns:
        Tensor of shape ``(indptr.numel() - 1, ...)``. Empty segments are
        filled with ``-inf``.

    Raises:
        TypeError: ``src``/``indptr``/``out`` has an unsupported type or dtype.
        ValueError: Shape or CSR pointer metadata is invalid.
        RuntimeError: Inputs are not on an Ascend NPU.
    """
    _check_inputs(src, indptr)

    num_segments = indptr.numel() - 1
    shape = (num_segments,) + tuple(src.shape[1:])
    if out is not None:
        _check_out(out, shape, src)

    if num_segments == 0:
        # A single-element indptr describes no segment at all, so no kernel is
        # launched and the result stays empty.
        result = torch.empty(shape, dtype=src.dtype, device=src.device)
    else:
        # The kernel reads the boundaries as int32 scalars; the range check
        # above guarantees that every value fits.
        pointers = indptr.to(torch.int32).contiguous()
        result = _pybind.segment_logsumexp_forward(src.contiguous(), pointers)

    if out is not None:
        out.copy_(result)
        return out
    return result

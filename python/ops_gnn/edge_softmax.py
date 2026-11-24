# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DGL-compatible ``edge_softmax`` on Ascend NPU.

The CSR graph layout describes the groups, so the call surface follows
``dgl.nn.functional.edge_softmax``::

    edge_softmax(src, indptr)

``indptr`` is the row pointer of a CSR graph over the source nodes, ``src`` has
``indptr[-1]`` rows - one per edge - and the reducer normalizes the edges of
every destination node::

    out[e] = exp(src[e] - max_i) / sum_{j in [indptr[i], indptr[i+1])} exp(src[j] - max_i)

Every trailing feature of ``src`` is normalized independently, and the output
keeps the shape, dtype and device of ``src``.

The operator has a dedicated AscendC kernel (``csrc/npu/edge_softmax``) shared by
arch22 (A2/A3) and arch35 (950). It is the grouped softmax of the scatter family
driven by the CSR pointer instead of a materialised ``index``: a segment is a
contiguous row range, so the device cursor advances one boundary per segment
rather than reading one label per edge, and the three forward passes (segment
maximum, ``exp`` and its sum, normalization) plus the two backward passes are the
ones of ``scatter_softmax``. Every reduction runs in float32 so a long low
precision segment cannot collapse. The Python layer validates the arguments and
wires the analytic segment Jacobian::

    grad_src[e] = out[e] * (grad_out[e] - sum_{j in segment(e)} out[j] * grad_out[j])
"""

import torch

from . import _pybind

_OP_NAME = "edge_softmax"

SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
SUPPORTED_INDPTR_DTYPES = (torch.int32, torch.int64)

# The device kernel keeps one int32 label per row and one int32 cursor per
# segment, so both counts must fit that range.
_INT32_MAX = 2 ** 31 - 1


def _check_arguments(src: torch.Tensor, indptr: torch.Tensor) -> None:
    """Reject a call the device kernel cannot serve, before it launches."""
    if src.dim() < 1:
        raise ValueError("src must have at least one dimension")
    if src.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            "src dtype must be one of float16, float32, bfloat16; "
            f"got {src.dtype}")
    if indptr.dim() != 1:
        raise ValueError(f"indptr must be 1-D, got {indptr.dim()}-D")
    if indptr.dtype not in SUPPORTED_INDPTR_DTYPES:
        raise TypeError(
            f"indptr dtype must be int32 or int64, got {indptr.dtype}")
    if indptr.device != src.device:
        raise ValueError(
            f"src and indptr must be on the same device, got "
            f"{src.device} and {indptr.device}")
    if src.device.type != "npu":
        raise RuntimeError(f"{_OP_NAME} requires Ascend NPU tensors")
    if indptr.numel() < 2:
        raise ValueError(
            "indptr must contain at least two boundaries (one segment)")
    if src.size(0) > _INT32_MAX:
        raise ValueError(
            f"src.size(0) must fit the int32 range, got {src.size(0)}")
    if indptr.numel() - 1 > _INT32_MAX:
        raise ValueError(
            f"indptr describes more segments than the int32 range "
            f"({indptr.numel() - 1})")
    _check_indptr(indptr, src.size(0))


def _check_indptr(indptr: torch.Tensor, rows: int) -> None:
    """Check that ``indptr`` is the CSR pointer of a graph with ``rows`` edges."""
    invalid = torch.logical_or(indptr[0] != 0, indptr[-1] != rows)
    invalid = torch.logical_or(invalid, (indptr[1:] < indptr[:-1]).any())
    if bool(invalid.item()):
        _raise_indptr_error(indptr, rows)


def _raise_indptr_error(indptr: torch.Tensor, rows: int) -> None:
    """Report the first violation of the CSR contract of ``indptr``."""
    ptr = indptr.to("cpu")
    if int(ptr[0]) != 0:
        raise ValueError(f"indptr must start at 0, got indptr[0] = {int(ptr[0])}")
    if int(ptr[-1]) != rows:
        raise ValueError(
            f"indptr[-1] ({int(ptr[-1])}) must equal src.size(0) ({rows})")
    step = int((ptr[1:] < ptr[:-1]).nonzero()[0].item())
    raise ValueError(
        f"indptr must be nondecreasing, but indptr[{step}] = {int(ptr[step])} "
        f"> indptr[{step + 1}] = {int(ptr[step + 1])}")


def _edge_softmax_autograd() -> torch.autograd.Function:
    """Build the autograd node of the operator around the two device entries."""
    forward = _pybind.edge_softmax_forward
    backward = _pybind.edge_softmax_backward

    class _EdgeSoftmax(torch.autograd.Function):
        """CSR edge softmax with the analytic segment Jacobian."""

        @staticmethod
        def forward(ctx, src, indptr):
            # The device kernel reads the boundaries as int32 scalars; the range
            # check above already guarantees that every value fits.
            ptr = indptr.to(torch.int32).contiguous()
            if src.numel() == 0:
                out = torch.zeros_like(src)
            else:
                out = forward(src.contiguous(), ptr)
            ctx.save_for_backward(out, ptr)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            out, ptr = ctx.saved_tensors
            grad_in = grad_out.contiguous()
            if out.numel() == 0:
                grad_src = torch.zeros_like(grad_in)
            else:
                grad_src = backward(out, grad_in, ptr)
            return grad_src, None

    _EdgeSoftmax.__name__ = "_EdgeSoftmax"
    _EdgeSoftmax.__qualname__ = "_EdgeSoftmax"
    return _EdgeSoftmax


_EdgeSoftmax = _edge_softmax_autograd()


def edge_softmax(src: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
    """Normalize the edges of every CSR segment with a softmax on Ascend NPU.

    Args:
        src: Edge features on NPU with shape ``(E, ...)``. Supported dtypes are
            ``float16``, ``float32`` and ``bfloat16``.
        indptr: 1-D ``int32``/``int64`` row pointer on the same NPU, of length
            ``segments + 1``. It must start at ``0``, be nondecreasing and end
            at ``E``.

    Returns:
        Tensor with the same shape, dtype and device as ``src``. An empty
        segment owns no edge, so it produces no output row.

    Raises:
        TypeError: ``src``/``indptr`` has an unsupported type or dtype.
        ValueError: Shape, pointer value or segment count is invalid.
        RuntimeError: Inputs are not on an Ascend NPU.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"src must be a torch.Tensor, got {type(src).__name__}")
    if not isinstance(indptr, torch.Tensor):
        raise TypeError(
            f"indptr must be a torch.Tensor, got {type(indptr).__name__}")
    _check_arguments(src, indptr)
    return _EdgeSoftmax.apply(src, indptr)

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""torch_scatter-compatible ``scatter_softmax`` on Ascend NPU.

The public call surface matches ``torch_scatter.scatter_softmax``::

    scatter_softmax(src, index, dim=0, dim_size=None)

For every group defined by ``index`` the result is the numerically stable
softmax along ``dim``::

    out[i] = exp(src[i] - max_g) / sum_j exp(src[j] - max_g)

where ``max_g`` is the maximum of the group that ``i`` belongs to and ``j``
runs over the elements of that group. The output keeps the shape, dtype and
device of ``src``.

The operator has a dedicated AscendC kernel (``csrc/npu/scatter_softmax``)
shared by arch22 (A2/A3) and arch35 (950). Because ``index`` is not sorted, the
device implementation accumulates the per-(batch, group, feature) reductions
with the MTE3 atomic unit in three forward passes (maximum, ``exp`` and its sum,
normalization) and two backward passes, and it evaluates every reduction in
float32 so a long low-precision group cannot collapse. The Python layer only
validates the arguments and wires the analytic grouped-softmax Jacobian::

    grad_src = out * (grad_out - sum_j out[j] * grad_out[j])
"""

from typing import Optional

import torch

from . import _pybind

_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)


def _normalize_dim(dim: int, rank: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError(f"dim must be an int, got {type(dim).__name__}")
    if dim < -rank or dim >= rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{rank}, {rank - 1}], but got {dim})")
    return dim + rank if dim < 0 else dim


def _check_inputs(src: torch.Tensor, index: torch.Tensor, dim: int) -> None:
    if src.dim() < 1:
        raise ValueError("src must have at least one dimension")
    if index.dim() != 1:
        raise ValueError(f"index must be 1-D, got {index.dim()}-D")
    if src.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "src dtype must be one of float16, float32, bfloat16; "
            f"got {src.dtype}")
    if index.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise TypeError(
            f"index dtype must be int32 or int64, got {index.dtype}")
    if src.device != index.device:
        raise ValueError(
            f"src and index must be on the same device, got "
            f"{src.device} and {index.device}")
    if src.device.type != "npu":
        raise RuntimeError("scatter_softmax requires Ascend NPU tensors")
    if index.numel() != src.size(dim):
        raise ValueError(
            f"index length ({index.numel()}) must match src.size({dim}) "
            f"({src.size(dim)})")


def _check_index_range(index: torch.Tensor, dim_size: int) -> None:
    if index.numel() == 0:
        return
    if bool((index < 0).any()):
        raise ValueError("index must be non-negative")
    if int(index.max().item()) >= dim_size:
        raise ValueError(
            f"index values must be smaller than dim_size ({dim_size})")


def _resolve_dim_size(index: torch.Tensor, dim_size: Optional[int]) -> int:
    if dim_size is None:
        return int(index.max().item()) + 1 if index.numel() > 0 else 0
    if not isinstance(dim_size, int) or isinstance(dim_size, bool):
        raise TypeError(
            f"dim_size must be an int, got {type(dim_size).__name__}")
    if dim_size < 0:
        raise ValueError(f"dim_size must be non-negative, got {dim_size}")
    if dim_size > 2 ** 31 - 1:
        raise ValueError(
            f"dim_size must fit the int32 index range, got {dim_size}")
    return dim_size


class _ScatterSoftmax(torch.autograd.Function):
    """Grouped softmax with an analytic grouped-softmax Jacobian."""

    @staticmethod
    def forward(ctx, src, index, dim, num_groups):
        # The device kernel reads the group labels as int32 scalars, so narrow
        # the public int32/int64 index once here; the range check above already
        # guarantees that every value fits.
        idx = index.to(torch.int32).contiguous()
        ctx.dim = dim
        ctx.num_groups = num_groups

        if src.numel() == 0:
            out = torch.zeros_like(src)
        else:
            out = _pybind.scatter_softmax_forward(
                src.contiguous(), idx, dim, num_groups)

        ctx.save_for_backward(out, idx)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out, idx = ctx.saved_tensors
        grad_in = grad_out.contiguous()

        if out.numel() == 0:
            grad_src = torch.zeros_like(grad_in)
        else:
            grad_src = _pybind.scatter_softmax_backward(
                out, grad_in, idx, ctx.dim, ctx.num_groups)
        return grad_src, None, None, None


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute a grouped softmax along ``dim`` on Ascend NPU.

    Args:
        src: Source tensor on NPU. Supported dtypes are ``float16``,
            ``float32`` and ``bfloat16``.
        index: 1-D ``int32``/``int64`` group index on the same NPU. Its
            length must equal ``src.size(dim)``.
        dim: Dimension along which the softmax is computed. Negative
            dimensions are accepted.
        dim_size: Number of groups. Defaults to ``index.max() + 1`` and may
            be larger than that to leave empty groups.

    Returns:
        Tensor with the same shape, dtype and device as ``src``. Elements of
        empty groups do not exist, so no placeholder value is produced.

    Raises:
        TypeError: ``src``/``index`` has an unsupported type or dtype, or
            ``dim``/``dim_size`` has the wrong type.
        ValueError: Shape, index range or ``dim_size`` is invalid.
        RuntimeError: Inputs are not on an Ascend NPU.
        IndexError: ``dim`` is out of range.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"src must be a torch.Tensor, got {type(src).__name__}")
    if not isinstance(index, torch.Tensor):
        raise TypeError(
            f"index must be a torch.Tensor, got {type(index).__name__}")

    dim = _normalize_dim(dim, src.dim())
    _check_inputs(src, index, dim)

    num_groups = _resolve_dim_size(index, dim_size)
    _check_index_range(index, num_groups)

    return _ScatterSoftmax.apply(src, index, dim, num_groups)

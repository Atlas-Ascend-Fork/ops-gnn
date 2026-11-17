# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Argument handling shared by the grouped-softmax operators.

``scatter_softmax`` and ``scatter_log_softmax`` take the same arguments, accept
the same dtypes and reject the same inputs, so their validation lives here
instead of in two copies. Both operators reduce along ``dim`` over the groups
defined by a 1-D ``index`` vector; the public entry points only add the
documentation and the autograd wiring of their own domain.

The autograd wrapper is shared as well: the two operators have the same host
call surface (a forward and a backward entry point on ``_pybind``), so
``build_autograd`` binds one operator by name and the module then exposes a
single ``run`` that validates the arguments and applies it.
"""

from typing import Optional

import torch

from . import _pybind

SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)


def normalize_dim(dim: int, rank: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError(f"dim must be an int, got {type(dim).__name__}")
    if dim < -rank or dim >= rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{rank}, {rank - 1}], but got {dim})")
    return dim + rank if dim < 0 else dim


def check_inputs(op: str, src: torch.Tensor, index: torch.Tensor, dim: int) -> None:
    if src.dim() < 1:
        raise ValueError("src must have at least one dimension")
    if index.dim() != 1:
        raise ValueError(f"index must be 1-D, got {index.dim()}-D")
    if src.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            "src dtype must be one of float16, float32, bfloat16; "
            f"got {src.dtype}")
    if index.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError(
            f"index dtype must be int32 or int64, got {index.dtype}")
    if src.device != index.device:
        raise ValueError(
            f"src and index must be on the same device, got "
            f"{src.device} and {index.device}")
    if src.device.type != "npu":
        raise RuntimeError(f"{op} requires Ascend NPU tensors")
    if index.numel() != src.size(dim):
        raise ValueError(
            f"index length ({index.numel()}) must match src.size({dim}) "
            f"({src.size(dim)})")


def check_index_range(index: torch.Tensor, dim_size: int) -> None:
    if index.numel() == 0:
        return
    if bool((index < 0).any()):
        raise ValueError("index must be non-negative")
    if int(index.max().item()) >= dim_size:
        raise ValueError(
            f"index values must be smaller than dim_size ({dim_size})")


def resolve_dim_size(index: torch.Tensor, dim_size: Optional[int]) -> int:
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


def build_autograd(op_name: str):
    """Create the autograd function of one grouped-softmax operator.

    Both operators forward to ``<op_name>_forward`` / ``<op_name>_backward``
    with the same argument list and save the same tensors, so the wrapper is
    built once here and bound by name. The Jacobian itself stays in the device
    kernel selected by the operator.
    """
    forward = getattr(_pybind, f"{op_name}_forward")
    backward = getattr(_pybind, f"{op_name}_backward")

    class _GroupedSoftmax(torch.autograd.Function):
        """Grouped softmax family with the analytic grouped Jacobian."""

        @staticmethod
        def forward(ctx, src, index, dim, num_groups):
            # The device kernel reads the group labels as int32 scalars, so
            # narrow the public int32/int64 index once here; the range check
            # above already guarantees that every value fits.
            idx = index.to(torch.int32).contiguous()
            ctx.dim = dim
            ctx.num_groups = num_groups

            if src.numel() == 0:
                out = torch.zeros_like(src)
            else:
                out = forward(src.contiguous(), idx, dim, num_groups)

            ctx.save_for_backward(out, idx)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            out, idx = ctx.saved_tensors
            grad_in = grad_out.contiguous()

            if out.numel() == 0:
                grad_src = torch.zeros_like(grad_in)
            else:
                grad_src = backward(
                    out, grad_in, idx, ctx.dim, ctx.num_groups)
            return grad_src, None, None, None

    _GroupedSoftmax.__name__ = f"_{op_name}"
    _GroupedSoftmax.__qualname__ = f"_{op_name}"
    _GroupedSoftmax.__doc__ = (
        f"Grouped {op_name.replace('_', ' ')} with the analytic Jacobian.")
    return _GroupedSoftmax


def run(op: str, function, src: torch.Tensor, index: torch.Tensor,
        dim: int, dim_size: Optional[int]) -> torch.Tensor:
    """Validate the public arguments of a grouped-softmax call and apply it."""
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"src must be a torch.Tensor, got {type(src).__name__}")
    if not isinstance(index, torch.Tensor):
        raise TypeError(
            f"index must be a torch.Tensor, got {type(index).__name__}")

    dim = normalize_dim(dim, src.dim())
    check_inputs(op, src, index, dim)

    num_groups = resolve_dim_size(index, dim_size)
    check_index_range(index, num_groups)
    return function.apply(src, index, dim, num_groups)

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

import torch

from . import _pybind
from .typing import Tensor


class _MatrixExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return _pybind.matrix_exp(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        if a.numel() == 0:
            return grad_output

        # The upper-right block of exp([[A.T, G], [0, A.T]]) is the
        # Fréchet derivative of exp at A.T in direction G.
        adjoint = a.float().mT
        top = torch.cat((adjoint, grad_output.float()), dim=-1)
        bottom = torch.cat((torch.zeros_like(adjoint), adjoint), dim=-1)
        block = torch.cat((top, bottom), dim=-2)
        n = a.size(-1)
        return matrix_exp(block)[..., :n, n:].to(a.dtype)


def matrix_exp(a: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the matrix exponential of square matrices, aligned with
    torch.linalg.matrix_exp.

    The input is interpreted as a batch of matrices with shape ``(..., n, n)``,
    and each matrix is exponentiated independently.

    Args:
        a (Tensor): Square matrix or batch of square matrices with shape
            ``(..., n, n)``. Must be on NPU device. Supported dtypes:
            float32, float16, bfloat16. Maximum matrix order is 2048.
        out (Optional[Tensor]): Optional output tensor. If provided, must have
            the same shape, dtype, and device as ``a``. Autograd requires
            omitting ``out``.

    Returns:
        Tensor: The matrix exponential of ``A``, with the same shape and dtype.

    Differences from torch.linalg.matrix_exp:
        - float64 and complex dtypes are not supported (an error is raised).

    Example:
        >>> import torch
        >>> import ops_gnn
        >>> A = torch.zeros(2, 2, device='npu')  # exp(0) = I
        >>> A[0, 0] = 1.0
        >>> ops_gnn.matrix_exp(A)
        tensor([[2.7183, 0.0000],
                [0.0000, 1.0000]], device='npu:0')
    """
    if out is not None:
        if torch.is_grad_enabled() and (a.requires_grad or out.requires_grad):
            raise RuntimeError("matrix_exp: out= does not support automatic differentiation")
        return _pybind.matrix_exp(a, out)
    small_matrix = a.dim() >= 2 and a.size(-1) <= 128
    if torch.is_grad_enabled() and a.requires_grad and small_matrix:
        return _MatrixExp.apply(a)
    return _pybind.matrix_exp(a)

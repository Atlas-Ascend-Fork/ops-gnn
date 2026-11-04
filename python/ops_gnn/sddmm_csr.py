# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generalized sampled dense-dense matrix multiplication on a CSR graph."""
from typing import Optional, Tuple

import torch

from . import _pybind


_MESSAGE_OPS = {"dot", "add", "sub", "mul", "div", "copy_lhs", "copy_rhs"}
_TARGET_ALIASES = {
    "u": 0,
    "src": 0,
    "e": 1,
    "edge": 1,
    "v": 2,
    "dst": 2,
}


def _normalize_target(target, name: str) -> int:
    if isinstance(target, bool):
        raise ValueError(f"sddmm_csr: {name} must be 0, 1, 2, u, e or v")
    if isinstance(target, int):
        if target in (0, 1, 2):
            return target
        raise ValueError(f"sddmm_csr: {name} must be 0, 1, 2, u, e or v")
    if isinstance(target, str):
        alias = _TARGET_ALIASES.get(target.lower())
        if alias is not None:
            return alias
    raise ValueError(f"sddmm_csr: {name} must be 0, 1, 2, u, e or v")


def _operand_flags(op: str) -> Tuple[bool, bool]:
    if op not in _MESSAGE_OPS:
        raise ValueError(
            "sddmm_csr: op must be dot, add, sub, mul, div, copy_lhs or copy_rhs")
    if op == "copy_lhs":
        return True, False
    if op == "copy_rhs":
        return False, True
    return True, True


def _check_operand(name: str, value: Optional[torch.Tensor], required: bool) -> None:
    if not required:
        if value is not None:
            raise ValueError(f"sddmm_csr: {name} is not used by this op")
        return
    if value is None:
        raise ValueError(f"sddmm_csr: {name} is required by this op")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"sddmm_csr: {name} must be a Tensor")
    if value.dim() not in (1, 2):
        raise ValueError(
            f"sddmm_csr: {name} must be one- or two-dimensional")
    if value.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"sddmm_csr: {name} must be float16 or float32")


def _expand(value: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if value.dim() == 1:
        return value.unsqueeze(-1), True
    return value, False


def _check_inputs(indptr, indices) -> None:
    if not isinstance(indptr, torch.Tensor):
        raise TypeError("sddmm_csr: indptr must be a Tensor")
    if not isinstance(indices, torch.Tensor):
        raise TypeError("sddmm_csr: indices must be a Tensor")


def _prepare_operands(lhs, rhs, use_lhs, use_rhs):
    lhs_expanded = rhs_expanded = None
    lhs_was_1d = rhs_was_1d = False
    if use_lhs:
        lhs_expanded, lhs_was_1d = _expand(lhs)
    if use_rhs:
        rhs_expanded, rhs_was_1d = _expand(rhs)
    if use_lhs and use_rhs:
        if lhs_expanded.device != rhs_expanded.device:
            raise ValueError("sddmm_csr: lhs and rhs must be on the same device")
        if lhs_expanded.dtype != rhs_expanded.dtype:
            raise ValueError("sddmm_csr: lhs and rhs must have the same dtype")
        if lhs_expanded.size(1) != rhs_expanded.size(1):
            raise ValueError("sddmm_csr: lhs and rhs must have the same feature dim")
    return lhs_expanded, rhs_expanded, lhs_was_1d, rhs_was_1d


def _should_squeeze(is_dot, lhs_was_1d, rhs_was_1d, use_lhs, use_rhs):
    return (
        not is_dot
        and (lhs_was_1d or not use_lhs)
        and (rhs_was_1d or not use_rhs)
    )


def _compute_shapes(nnz, feature_dim, is_dot, squeeze):
    host_shape = (nnz,) if is_dot else (nnz, feature_dim)
    final_shape = (nnz,) if squeeze else host_shape
    return host_shape, final_shape


def _prepare_out(out, final_shape, squeeze):
    if out is None:
        return None
    if not isinstance(out, torch.Tensor):
        raise TypeError("sddmm_csr: out must be a Tensor")
    if out.shape != final_shape:
        raise ValueError(
            f"sddmm_csr: out must have shape {tuple(final_shape)}, got {tuple(out.shape)}")
    if squeeze:
        return out.unsqueeze(-1)
    return out


def sddmm_csr(indptr: torch.Tensor,
              indices: torch.Tensor,
              lhs: Optional[torch.Tensor] = None,
              rhs: Optional[torch.Tensor] = None,
              op: str = "dot",
              lhs_target=0,
              rhs_target=2,
              out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Run generalized SDDMM on a CSR graph.

    ``indptr`` and ``indices`` describe a CSR matrix whose row axis is the
    ``src`` (0) target; ``edge`` (1) is the CSR position and ``dst`` (2) is the
    CSR column. ``lhs``/``rhs`` are gathered by ``lhs_target``/``rhs_target``.
    ``dot`` reduces the feature axis and returns shape ``(nnz,)``; every other
    operation returns ``(nnz, feature_dim)``. One-dimensional operands are
    treated as ``(rows, 1)`` and the trailing axis is squeezed from the result.
    ``indptr``/``indices`` must be one-dimensional int32/int64 tensors on the
    same NPU with matching dtype; ``indptr`` must be nonempty, nondecreasing,
    start at 0 and end at ``nnz``. ``lhs``/``rhs`` must share the feature
    device, dtype and feature dimension. ``out`` must match the result shape and feature
    device/dtype, be contiguous, not require grad, and not alias any input.
    Only float16/float32 features are supported; autograd and heterogeneous
    graphs are not handled.
    """
    _check_inputs(indptr, indices)
    use_lhs, use_rhs = _operand_flags(op)
    _check_operand("lhs", lhs, use_lhs)
    _check_operand("rhs", rhs, use_rhs)

    lhs_expanded, rhs_expanded, lhs_was_1d, rhs_was_1d = _prepare_operands(
        lhs, rhs, use_lhs, use_rhs)

    lhs_target = _normalize_target(lhs_target, "lhs_target")
    rhs_target = _normalize_target(rhs_target, "rhs_target")

    feature = lhs_expanded if use_lhs else rhs_expanded
    nnz = indices.numel()
    is_dot = op == "dot"
    squeeze = _should_squeeze(is_dot, lhs_was_1d, rhs_was_1d, use_lhs, use_rhs)
    _, final_shape = _compute_shapes(nnz, feature.size(1), is_dot, squeeze)
    host_out = _prepare_out(out, final_shape, squeeze)

    result = _pybind.sddmm_csr(
        indptr, indices, lhs_expanded, rhs_expanded, op,
        lhs_target, rhs_target, host_out)
    if out is not None:
        return out
    return result.squeeze(-1) if squeeze else result

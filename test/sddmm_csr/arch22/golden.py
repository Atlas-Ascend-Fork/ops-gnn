# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent PyTorch reference for CSR SDDMM, usable on CPU without ops_gnn."""
import torch


_TARGET_ALIASES = {
    "u": 0, "src": 0,
    "e": 1, "edge": 1,
    "v": 2, "dst": 2,
}


def _target(target):
    if isinstance(target, str):
        return _TARGET_ALIASES[target.lower()]
    return int(target)


def _target_indices(indptr, indices, target):
    """Return the per-edge row, edge ordinal or column index."""
    target = _target(target)
    if target == 0:
        counts = (indptr[1:] - indptr[:-1]).long()
        rows = torch.arange(indptr.numel() - 1, device=indptr.device)
        return torch.repeat_interleave(rows, counts)
    if target == 1:
        return torch.arange(indices.numel(), device=indices.device)
    return indices.long()


def _should_squeeze(lhs_was_1d, rhs_was_1d, use_lhs, use_rhs):
    return (lhs_was_1d or not use_lhs) and (rhs_was_1d or not use_rhs)


def _prepare_operands(lhs, rhs, use_lhs, use_rhs):
    lhs_was_1d = rhs_was_1d = False
    if use_lhs and lhs.dim() == 1:
        lhs = lhs.unsqueeze(-1)
        lhs_was_1d = True
    if use_rhs and rhs.dim() == 1:
        rhs = rhs.unsqueeze(-1)
        rhs_was_1d = True
    return lhs, rhs, lhs_was_1d, rhs_was_1d


def _resolve_dtype(lhs, rhs, use_lhs):
    dtype = lhs.dtype if use_lhs else rhs.dtype
    return dtype, (torch.float32 if dtype == torch.float16 else dtype)


def _binary_op(op, left, right):
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "div":
        return left / right
    raise ValueError(f"unsupported op: {op}")


def _squeeze_result(result, squeeze):
    return result.squeeze(-1) if squeeze else result


def sddmm_reference(indptr, indices, lhs=None, rhs=None, op="dot",
                    lhs_target=0, rhs_target=2):
    use_lhs = op != "copy_rhs"
    use_rhs = op != "copy_lhs"

    lhs, rhs, lhs_was_1d, rhs_was_1d = _prepare_operands(lhs, rhs, use_lhs, use_rhs)
    dtype, compute_dtype = _resolve_dtype(lhs, rhs, use_lhs)
    squeeze = _should_squeeze(lhs_was_1d, rhs_was_1d, use_lhs, use_rhs)

    lhs_index = _target_indices(indptr, indices, lhs_target) if use_lhs else None
    rhs_index = _target_indices(indptr, indices, rhs_target) if use_rhs else None

    if op == "dot":
        values = lhs[lhs_index].to(compute_dtype) * rhs[rhs_index].to(compute_dtype)
        return values.sum(dim=-1).to(dtype)
    if op == "copy_lhs":
        return _squeeze_result(lhs[lhs_index], squeeze)
    if op == "copy_rhs":
        return _squeeze_result(rhs[rhs_index], squeeze)

    left = lhs[lhs_index].to(compute_dtype)
    right = rhs[rhs_index].to(compute_dtype)
    result = _binary_op(op, left, right).to(dtype)
    return _squeeze_result(result, squeeze)

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CSR SpMM aggregation implemented with ops-gnn tensor interfaces."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from . import _pybind


_MESSAGE_OPS = {"copy_lhs", "copy_rhs", "add", "sub", "mul", "div"}
_REDUCE_OPS = {"sum", "max", "min", "mean"}
_FP16_COPY_LHS_IMPL = {
    "sum": _pybind.spmm_sum_csr,
    "max": _pybind.spmm_max_csr,
    "min": _pybind.spmm_min_csr,
}


@dataclass(frozen=True)
class _SpmmRequest:
    indptr: torch.Tensor
    indices: torch.Tensor
    x: Optional[torch.Tensor]
    op: str
    reduce: str
    out: Optional[torch.Tensor]
    rhs: Optional[torch.Tensor]
    batched: bool


def _normalize_operands(x: Optional[torch.Tensor], rhs: Optional[torch.Tensor],
                        op: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if op == "copy_lhs":
        if x is None or rhs is not None:
            raise ValueError("copy_lhs requires x and does not accept rhs")
        return x, None
    if op == "copy_rhs":
        if rhs is not None:
            if x is not None:
                raise ValueError(
                "copy_rhs accepts edge features through either x or rhs, not both")
            return None, rhs
        if x is None:
            raise ValueError("copy_rhs requires edge features")
        return None, x
    if x is None or rhs is None:
        raise ValueError(f"{op} requires both x and rhs")
    return x, rhs


def _validate_tensor_types(request: _SpmmRequest,
                           features: List[torch.Tensor]) -> None:
    for name, value in (("indptr", request.indptr), ("indices", request.indices)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
    for value in features:
        if not isinstance(value, torch.Tensor):
            raise TypeError("x and rhs must be Tensors")
    if request.out is not None and not isinstance(request.out, torch.Tensor):
        raise TypeError("out must be a Tensor")
    if not features:
        raise ValueError("at least one feature tensor is required")


def _validate_devices(request: _SpmmRequest,
                      features: List[torch.Tensor]) -> None:
    device = features[0].device
    if device.type != "npu" or any(value.device != device for value in features):
        raise RuntimeError("spmm: feature tensors must be on the same NPU")
    if request.indptr.device != device or request.indices.device != device:
        raise RuntimeError("spmm: CSR tensors and features must be on the same NPU")


def _validate_metadata(request: _SpmmRequest, lhs: Optional[torch.Tensor],
                       rhs: Optional[torch.Tensor],
                       features: List[torch.Tensor]) -> None:
    indptr, indices = request.indptr, request.indices
    if indptr.dim() != 1 or indices.dim() != 1:
        raise RuntimeError("spmm: CSR tensors must be one-dimensional")
    if indptr.numel() == 0:
        raise RuntimeError(
            "spmm: indptr must be nonempty")
    valid_index_dtype = indptr.dtype in (torch.int32, torch.int64)
    if not valid_index_dtype or indices.dtype != indptr.dtype:
        raise RuntimeError("spmm: CSR tensors must have the same int32 or int64 dtype")
    if any(value.dtype not in (torch.float16, torch.float32) for value in features):
        raise RuntimeError("spmm: features must use float16 or float32")
    if lhs is not None and rhs is not None and lhs.dtype != rhs.dtype:
        raise RuntimeError("x and rhs must have the same dtype")
    ranks = [value.dim() for value in features]
    if any(rank < 1 for rank in ranks):
        raise RuntimeError("spmm feature tensors must have at least one dimension")
    if request.batched:
        if max(ranks) < 3:
            raise RuntimeError(
                "bspmm requires at least one feature tensor with rank 3 or higher")
    elif any(rank > 2 for rank in ranks):
        raise RuntimeError(
            "spmm supports scalar or two-dimensional features; use bspmm for higher-rank features")


def _validate_tensors(request: _SpmmRequest, lhs: Optional[torch.Tensor],
                      rhs: Optional[torch.Tensor]) -> None:
    features = [value for value in (lhs, rhs) if value is not None]
    _validate_tensor_types(request, features)
    _validate_devices(request, features)
    _validate_metadata(request, lhs, rhs, features)


def _validate_csr(indptr: torch.Tensor, indices: torch.Tensor,
                  lhs: Optional[torch.Tensor], rhs: Optional[torch.Tensor]) -> None:
    rows = indptr.numel() - 1
    edges = indices.numel()
    if int(indptr[0].item()) != 0 or int(indptr[rows].item()) != edges:
        raise RuntimeError("indptr must start at zero and end at indices.numel()")
    if rows and not bool((indptr[1:] >= indptr[:-1]).all().item()):
        raise RuntimeError("indptr must be non-decreasing")
    if lhs is not None and edges:
        if not bool(((indices >= 0) & (indices < lhs.size(0))).all().item()):
            raise RuntimeError("indices are out of range for x")
    if rhs is not None and rhs.size(0) != edges:
        raise RuntimeError("rhs.size(0) must equal indices.numel()")


def _align_feature_ranks(lhs: torch.Tensor,
                         rhs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad missing feature dimensions immediately after the item axis."""
    feature_rank = max(lhs.dim(), rhs.dim()) - 1

    def align(value: torch.Tensor) -> torch.Tensor:
        padding = feature_rank - (value.dim() - 1)
        if padding == 0:
            return value
        return value.reshape(
            (value.shape[0],) + (1,) * padding + tuple(value.shape[1:]))

    return align(lhs), align(rhs)


def _messages(indices: torch.Tensor, lhs: Optional[torch.Tensor],
              rhs: Optional[torch.Tensor], op: str) -> torch.Tensor:
    gathered = lhs.index_select(0, indices.to(torch.int64)) if lhs is not None else None
    if op == "copy_lhs":
        return gathered
    if op == "copy_rhs":
        return rhs
    gathered, rhs = _align_feature_ranks(gathered, rhs)
    if op == "add":
        return gathered + rhs
    if op == "sub":
        return gathered - rhs
    if op == "mul":
        return gathered * rhs
    return gathered / rhs


def _reduce_messages(indptr: torch.Tensor, messages: torch.Tensor,
                     reduce: str) -> torch.Tensor:
    row_count = indptr.numel() - 1
    counts = indptr[1:] - indptr[:-1]
    shape = (row_count,) + tuple(messages.shape[1:])
    if reduce in ("sum", "mean"):
        rows = torch.repeat_interleave(
            torch.arange(row_count, dtype=torch.int64, device=indptr.device),
            counts.to(torch.int64))
        host_half_mean = (
            reduce == "mean" and messages.dtype == torch.float16
            and messages.device.type == "cpu"
        )
        accumulator_dtype = torch.float64 if host_half_mean else torch.float32
        values = (
            messages.to(accumulator_dtype)
            if reduce == "mean" and messages.dtype == torch.float16
            else messages
        )
        result = values.new_zeros(shape).index_add(0, rows, values)
        if reduce == "mean":
            # Keep the count in FP32: FP16 cannot represent row degrees above
            # 65504, so converting a larger count can produce infinity.
            # CPU FP16 means keep wide sums through division: a long row of
            # maximum finite values can round to infinity after FP32 summation.
            mean_values = result if host_half_mean else result.to(torch.float32)
            divisor = counts.clamp(min=1).to(mean_values.dtype)
            divisor = divisor.reshape((row_count,) + (1,) * (messages.dim() - 1))
            result = (mean_values / divisor).to(messages.dtype)
        return result

    if row_count == 0:
        return messages.new_empty(shape)
    offsets = indptr.detach().cpu().tolist()
    rows = []
    for row in range(row_count):
        start, end = offsets[row], offsets[row + 1]
        if start == end:
            rows.append(messages.new_zeros(messages.shape[1:]))
        elif reduce == "max":
            rows.append(messages[start:end].max(dim=0).values)
        else:
            rows.append(messages[start:end].min(dim=0).values)
    return torch.stack(rows, dim=0)


def _write_out(result: torch.Tensor, out: Optional[torch.Tensor],
               grad_required: bool) -> torch.Tensor:
    if out is None:
        return result
    if grad_required:
        raise RuntimeError("out is not supported when feature tensors require gradients")
    if out.shape != result.shape or out.dtype != result.dtype or out.device != result.device:
        raise RuntimeError("out must match the result shape, dtype and device")
    if not out.is_contiguous() or out.requires_grad:
        raise RuntimeError("out must be contiguous and must not require gradients")
    out.copy_(result)
    return out


def _run_copy_lhs_kernel(request: _SpmmRequest, lhs: torch.Tensor,
                         grad_required: bool) -> Optional[torch.Tensor]:
    if grad_required or request.op != "copy_lhs":
        return None
    if request.batched:
        supported = lhs.dim() == 3 and lhs.dtype == torch.float16
        if supported and request.reduce in _FP16_COPY_LHS_IMPL:
            implementation = _FP16_COPY_LHS_IMPL[request.reduce]
            return implementation(request.indptr, request.indices, lhs, request.out)
        return None
    if lhs.dim() != 2 or request.reduce not in _FP16_COPY_LHS_IMPL:
        return None
    if lhs.dtype == torch.float16:
        implementation = _FP16_COPY_LHS_IMPL[request.reduce]
        return implementation(request.indptr, request.indices, lhs, request.out)
    if lhs.dtype == torch.float32:
        return _pybind.unified_spmm_csr(
            request.indptr, request.indices, lhs, request.op,
            request.reduce, request.out)
    return None


def _run_copy_rhs_kernel(request: _SpmmRequest, edge: torch.Tensor,
                         grad_required: bool) -> Optional[torch.Tensor]:
    if grad_required or request.batched or request.op != "copy_rhs":
        return None
    supported = edge.dim() == 2 and edge.dtype == torch.float32
    if supported and request.reduce != "mean":
        return _pybind.unified_spmm_csr(
            request.indptr, request.indices, edge, request.op,
            request.reduce, request.out)
    return None


def _spmm(request: _SpmmRequest) -> torch.Tensor:
    api = "bspmm" if request.batched else "spmm"
    for name, value in (("indptr", request.indptr), ("indices", request.indices),
                        ("x", request.x), ("rhs", request.rhs), ("out", request.out)):
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(f"{api}: {name} must be a Tensor")
    if not isinstance(request.op, str) or request.op not in _MESSAGE_OPS:
        raise RuntimeError(
            f"{api}: op must be copy_lhs, copy_rhs, add, sub, mul or div")
    if not isinstance(request.reduce, str) or request.reduce not in _REDUCE_OPS:
        raise RuntimeError(f"{api}: reduce must be sum, max, min or mean")
    lhs, edge = _normalize_operands(request.x, request.rhs, request.op)
    grad_required = any(
        value is not None and value.requires_grad for value in (lhs, edge))

    if lhs is not None:
        result = _run_copy_lhs_kernel(request, lhs, grad_required)
        if result is not None:
            return result
    if edge is not None:
        result = _run_copy_rhs_kernel(request, edge, grad_required)
        if result is not None:
            return result

    _validate_tensors(request, lhs, edge)
    if request.op == "copy_rhs" and edge.dtype != torch.float32:
        raise RuntimeError("copy_rhs supports only float32")
    _validate_csr(request.indptr, request.indices, lhs, edge)
    messages = _messages(request.indices, lhs, edge, request.op)
    result = _reduce_messages(request.indptr, messages, request.reduce)
    return _write_out(result, request.out, grad_required)


def _parse_call_options(api: str, args: tuple, kwargs: dict) -> tuple:
    names = ("op", "reduce", "out", "rhs")
    if len(args) > len(names):
        raise TypeError(f"{api} accepts at most seven positional arguments")
    options = {"op": "copy_lhs", "reduce": "sum", "out": None, "rhs": None}
    for name, value in zip(names, args):
        if name in kwargs:
            raise TypeError(f"{api} got multiple values for argument '{name}'")
        options[name] = value
    unexpected = set(kwargs) - set(names)
    if unexpected:
        name = sorted(unexpected)[0]
        raise TypeError(f"{api} got an unexpected keyword argument '{name}'")
    options.update(kwargs)
    return options["op"], options["reduce"], options["out"], options["rhs"]


def spmm(indptr: torch.Tensor, indices: torch.Tensor,
         x: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
    """Run generalized CSR aggregation with ops-gnn tensors.

    ``x`` stores source features for ``copy_lhs`` and the left operand for
    binary message operations. ``rhs`` stores CSR-ordered edge features.
    For compatibility, ``copy_rhs`` also accepts its edge features through
    ``x``. Supported message operations are ``copy_lhs``, ``copy_rhs``,
    ``add``, ``sub``, ``mul`` and ``div``; reductions are ``sum``, ``max``,
    ``min`` and ``mean``.
    """
    op, reduce, out, rhs = _parse_call_options("spmm", args, kwargs)
    request = _SpmmRequest(indptr, indices, x, op, reduce, out, rhs, False)
    return _spmm(request)

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""torch_scatter-compatible forward Scatter API for Ascend NPU.

The public call surface intentionally matches torch_scatter 2.1.2.  Index
broadcasting and validation happen here.  Contiguous L1 NPU tensors are sent
to the Ascend C extension; float64/int64 use the documented CPU fallback.
"""

from __future__ import annotations

import warnings
import weakref
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Broadcast ``src`` to ``other`` using torch_scatter's index rules."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


try:  # The extension is present after the CANN build.
    from . import _pybind as _scatter_npu
except ImportError:  # CPU-only development and documentation builds.
    _scatter_npu = None


_L1_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.uint8,
}
_L2_DTYPES = {torch.float64, torch.int64}
_NARROW_COUNT_DTYPES = {torch.uint8, torch.int8, torch.int16}
_REDUCE_CODE = {"sum": 0, "add": 0, "mul": 1, "mean": 2, "min": 3, "max": 4}
_warned_l2 = set()
_INDEX_BOUNDS_CACHE = {}
_INDEX_BOUNDS_CACHE_LIMIT = 256
_INDEX_HOT_TARGET_CACHE = {}


@dataclass(frozen=True)
class _ScatterRequest:
    dim: int
    out: Optional[torch.Tensor]
    dim_size: Optional[int]
    reduce: str


@dataclass
class _PreparedScatter:
    request: _ScatterRequest
    dim: int
    expanded_index: torch.Tensor
    original_out: Optional[torch.Tensor]
    work_out: Optional[torch.Tensor]
    output_shape: Tuple[int, ...]
    output_dim: int


@dataclass
class _ReferenceViews:
    src: torch.Tensor
    index: torch.Tensor
    out: torch.Tensor
    count: Optional[torch.Tensor]
    arg: Optional[torch.Tensor]
    dim_length: int


def native_available() -> bool:
    """Return whether the Ascend C extension was imported successfully."""
    return _scatter_npu is not None


def _normalize_dim(src: torch.Tensor, dim: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError("dim must be an int")
    if src.dim() < 1 or src.dim() > 8:
        raise ValueError("src must have between 1 and 8 dimensions")
    if dim < -src.dim() or dim >= src.dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{src.dim()}, {src.dim() - 1}], but got {dim})"
        )
    return dim + src.dim() if dim < 0 else dim


def _validate_dtype(src: torch.Tensor, index: torch.Tensor) -> None:
    if index.dtype != torch.int64:
        raise TypeError(f"index must have dtype torch.int64, got {index.dtype}")
    if src.dtype not in _L1_DTYPES | _L2_DTYPES:
        raise TypeError(
            "src dtype must be one of float16, bfloat16, float32, float64, "
            "int8, int16, int32, int64, or uint8"
        )


def _dtype_extreme(dtype: torch.dtype, *, minimum: bool):
    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    return info.min if minimum else info.max


def _shape_factors(src: torch.Tensor, dim: int) -> Tuple[int, int, int]:
    before = 1
    for size in src.shape[:dim]:
        before *= int(size)
    after = 1
    for size in src.shape[dim + 1:]:
        after *= int(size)
    return before, int(src.size(dim)), after


def _index_minmax(index: torch.Tensor) -> Tuple[int, int]:
    """Return exact bounds, caching synchronized NPU reductions by version."""
    if index.numel() == 0:
        return 0, -1

    cache_key = id(index)
    try:
        version_value = getattr(index, "_version", None)
        version = int(version_value) if version_value is not None else None
    except RuntimeError:
        # Inference tensors do not expose a version counter.  They remain
        # fully valid inputs, but cannot be cached safely across mutations.
        version = None
    cacheable = index.device.type == "npu" and version is not None
    if cacheable:
        cached = _INDEX_BOUNDS_CACHE.get(cache_key)
        if (
            cached is not None
            and cached[0]() is index
            and cached[1] == version
        ):
            return cached[2], cached[3]

    bounds_index = index
    if index.device.type == "npu" and index.numel() >= 65536:
        # Large int64 reductions may time out in the Ascend ConcatD path.
        # Bounds are checked once and cached, so perform this validation on CPU.
        bounds_index = index.cpu()
    minimum = int(bounds_index.min().item())
    maximum = int(bounds_index.max().item())
    if cacheable:
        # A bounded cache removes repeated .item() synchronization from warmup
        # and timed calls.  Tensor in-place updates increment _version, so a
        # mutated index is always revalidated before another kernel launch.
        if len(_INDEX_BOUNDS_CACHE) >= _INDEX_BOUNDS_CACHE_LIMIT:
            stale = [key for key, value in _INDEX_BOUNDS_CACHE.items()
                     if value[0]() is None]
            for key in stale:
                _INDEX_BOUNDS_CACHE.pop(key, None)
            if len(_INDEX_BOUNDS_CACHE) >= _INDEX_BOUNDS_CACHE_LIMIT:
                _INDEX_BOUNDS_CACHE.pop(next(iter(_INDEX_BOUNDS_CACHE)))
        _INDEX_BOUNDS_CACHE[cache_key] = (
            weakref.ref(index), version, minimum, maximum
        )
    return minimum, maximum


def _index_hot_target(index: torch.Tensor) -> int:
    """Return a majority target for compact NPU indices, or ``-1``.

    The one-time D2H inspection is performed during the first (normally
    warmup) call and is cached by tensor identity plus version.  Timed calls
    therefore do not synchronize.  A strict majority is intentionally used:
    it makes the per-AIV UB accumulator profitable while leaving random and
    moderately skewed indices on the existing fully pipelined vector path.
    """
    if index.device.type != "npu" or index.dim() != 1 or index.numel() < 4096:
        return -1
    try:
        version_value = getattr(index, "_version", None)
        version = int(version_value) if version_value is not None else None
    except RuntimeError:
        return -1
    if version is None:
        return -1
    cache_key = id(index)
    cached = _INDEX_HOT_TARGET_CACHE.get(cache_key)
    if cached is not None and cached[0]() is index and cached[1] == version:
        return cached[2]

    cpu_index = index.detach().cpu()
    counts = torch.bincount(cpu_index)
    hot_target = -1
    if counts.numel() > 0:
        hot_count, hot_index = counts.max(dim=0)
        if int(hot_count) * 2 > index.numel():
            hot_target = int(hot_index)

    if len(_INDEX_HOT_TARGET_CACHE) >= _INDEX_BOUNDS_CACHE_LIMIT:
        stale = [key for key, value in _INDEX_HOT_TARGET_CACHE.items()
                 if value[0]() is None]
        for key in stale:
            _INDEX_HOT_TARGET_CACHE.pop(key, None)
        if len(_INDEX_HOT_TARGET_CACHE) >= _INDEX_BOUNDS_CACHE_LIMIT:
            _INDEX_HOT_TARGET_CACHE.pop(next(iter(_INDEX_HOT_TARGET_CACHE)))
    _INDEX_HOT_TARGET_CACHE[cache_key] = (
        weakref.ref(index), version, hot_target
    )
    return hot_target


def _new_reference_result(
    src: torch.Tensor, request: _ScatterRequest, output_shape: Tuple[int, ...]
) -> torch.Tensor:
    if request.out is not None:
        return request.out
    if src.numel() == 0 or request.reduce in ("sum", "add", "mean"):
        return torch.zeros(output_shape, dtype=src.dtype, device=src.device)
    if request.reduce == "mul":
        return torch.ones(output_shape, dtype=src.dtype, device=src.device)
    minimum = request.reduce == "max"
    return torch.full(
        output_shape,
        _dtype_extreme(src.dtype, minimum=minimum),
        dtype=src.dtype,
        device=src.device,
    )


def _new_reference_views(
    src: torch.Tensor,
    index: torch.Tensor,
    result: torch.Tensor,
    dim: int,
    reduce: str,
) -> Tuple[_ReferenceViews, Optional[torch.Tensor]]:
    before, dim_length, after = _shape_factors(src, dim)
    output_dim = int(result.size(dim))
    arg_result = None
    if reduce in ("min", "max"):
        arg_result = torch.full(
            result.shape, dim_length, dtype=torch.int64, device=src.device
        )
    count = None
    if reduce == "mean":
        count = torch.zeros(
            (before, output_dim, after), dtype=torch.int64, device=src.device
        )
    views = _ReferenceViews(
        src.reshape(before, dim_length, after),
        index.reshape(before, dim_length, after),
        result.reshape(before, output_dim, after),
        count,
        arg_result.reshape(before, output_dim, after) if arg_result is not None else None,
        dim_length,
    )
    return views, arg_result


def _reference_update(current, value, reduce: str):
    if reduce in ("sum", "add", "mean"):
        return current + value, None
    if reduce == "mul":
        return current * value, None
    take = value <= current if reduce == "min" else value >= current
    return torch.where(take, value, current), take


def _run_reference_loop(views: _ReferenceViews, reduce: str) -> None:
    # One source position at a time preserves torch_scatter's ordered L2 path.
    for position in range(views.dim_length):
        target = views.index[:, position, :].unsqueeze(1)
        value = views.src[:, position, :]
        current = views.out.gather(1, target).squeeze(1)
        updated, take = _reference_update(current, value, reduce)
        views.out.scatter_(1, target, updated.unsqueeze(1))
        if views.count is not None:
            current_count = views.count.gather(1, target).squeeze(1)
            views.count.scatter_(1, target, (current_count + 1).unsqueeze(1))
        if views.arg is not None:
            current_arg = views.arg.gather(1, target).squeeze(1)
            new_arg = torch.full_like(current_arg, position)
            views.arg.scatter_(
                1, target, torch.where(take, new_arg, current_arg).unsqueeze(1)
            )


def _finalize_reference(
    views: _ReferenceViews, result: torch.Tensor, request: _ScatterRequest
) -> None:
    if views.count is not None:
        denominator = views.count
        if result.dtype in _NARROW_COUNT_DTYPES:
            denominator = denominator.to(result.dtype)
        denominator.clamp_min_(1)
        if result.is_floating_point():
            views.out.div_(denominator)
        else:
            quotient = torch.div(views.out, denominator, rounding_mode="floor")
            views.out.copy_(quotient)
    if views.arg is not None and request.out is None:
        views.out.masked_fill_(views.arg == views.dim_length, 0)


def _reference_forward(
    src: torch.Tensor,
    index: torch.Tensor,
    request: _ScatterRequest,
    output_shape: Tuple[int, ...],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Ordered reference used by CPU inputs and by the L2 CPU fallback."""
    result = _new_reference_result(src, request, output_shape)
    views, arg_result = _new_reference_views(
        src, index, result, request.dim, request.reduce
    )
    if src.numel() == 0 or result.numel() == 0:
        return result, arg_result
    _run_reference_loop(views, request.reduce)
    _finalize_reference(views, result, request)
    return result, arg_result


def _copy_back_if_needed(
    result: torch.Tensor, original_out: Optional[torch.Tensor]
) -> torch.Tensor:
    if original_out is None or result is original_out:
        return result
    original_out.copy_(result)
    return original_out


def _validate_request(
    src: torch.Tensor, index: torch.Tensor, request: _ScatterRequest
) -> int:
    if not isinstance(src, torch.Tensor) or not isinstance(index, torch.Tensor):
        raise TypeError("src and index must be torch.Tensor instances")
    dim = _normalize_dim(src, request.dim)
    _validate_dtype(src, index)
    if src.device != index.device:
        raise ValueError("src and index must be on the same device")
    if request.dim_size is not None:
        if not isinstance(request.dim_size, int) or isinstance(request.dim_size, bool):
            raise TypeError("dim_size must be an int or None")
        if request.dim_size < 0:
            raise ValueError("dim_size must be non-negative")
    return dim


def _prepare_output(
    src: torch.Tensor,
    request: _ScatterRequest,
    dim: int,
    max_index: int,
    expanded_index: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Tuple[int, ...], int]:
    out = request.out
    if out is None:
        output_shape = list(src.shape)
        output_dim = request.dim_size
        if output_dim is None:
            output_dim = 0 if expanded_index.numel() == 0 else max_index + 1
        output_shape[dim] = output_dim
        return None, tuple(output_shape), output_dim
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor or None")
    if out.dtype != src.dtype:
        raise TypeError("out dtype must match src dtype")
    if out.device != src.device:
        raise ValueError("out must be on the same device as src")
    if out.dim() != src.dim():
        raise ValueError("out and src must have the same rank")
    for axis in range(src.dim()):
        if axis != dim and out.size(axis) != src.size(axis):
            raise ValueError("out shape must match src outside the scatter dim")
    work_out = out if out.is_contiguous() else out.contiguous()
    return work_out, tuple(out.shape), int(out.size(dim))


def _prepare_scatter(
    src: torch.Tensor, index: torch.Tensor, request: _ScatterRequest
) -> _PreparedScatter:
    dim = _validate_request(src, index, request)

    try:
        expanded_index = broadcast(index, src, dim)
    except RuntimeError as exc:
        raise ValueError(
            f"index with shape {tuple(index.shape)} cannot broadcast to "
            f"src shape {tuple(src.shape)} at dim={dim}"
        ) from exc

    max_index = -1
    if expanded_index.numel() > 0:
        # Broadcasting does not introduce new values, so reduce the original
        # index rather than a potentially huge expanded view.
        min_index, max_index = _index_minmax(index)
        if min_index < 0:
            raise IndexError(f"index contains a negative value ({min_index})")

    work_out, output_shape, output_dim = _prepare_output(
        src, request, dim, max_index, expanded_index
    )
    if max_index >= output_dim:
        raise IndexError(
            f"index value {max_index} is out of bounds for output size "
            f"{output_dim} at dimension {dim}"
        )
    return _PreparedScatter(
        request, dim, expanded_index, request.out, work_out, output_shape, output_dim
    )


def _needs_original_mean_count(
    src: torch.Tensor, index: torch.Tensor, dim: int
) -> bool:
    if index.dim() <= dim:
        return True
    return index.size(dim) == 1 and src.size(dim) != 1


def _run_original_mean_count(
    src: torch.Tensor, index: torch.Tensor, plan: _PreparedScatter
) -> Tuple[torch.Tensor, None]:
    sum_request = _ScatterRequest(
        plan.dim, plan.original_out, plan.request.dim_size, "sum"
    )
    result, _ = _scatter_forward(src, index, sum_request)
    count_dim = plan.dim if index.dim() > plan.dim else index.dim() - 1
    # Accumulate in a wider tensor, then reproduce the count wrapping that
    # torch_scatter applies in the source dtype before clamping the divisor.
    count_src = torch.ones(index.size(), dtype=torch.int32, device=src.device)
    count_request = _ScatterRequest(count_dim, None, int(result.size(plan.dim)), "sum")
    count, _ = _scatter_forward(count_src, index, count_request)
    if src.dtype in _NARROW_COUNT_DTYPES:
        limits = torch.iinfo(src.dtype)
        count.remainder_(limits.max - limits.min + 1)
        count.masked_fill_(count > limits.max, 1)
    count = broadcast(count.clamp_min_(1), result, plan.dim)
    if result.is_floating_point():
        result.div_(count)
    else:
        result.copy_(torch.div(result, count, rounding_mode="floor"))
    return result, None


def _uses_compact_dim_index(
    src: torch.Tensor, index: torch.Tensor
) -> bool:
    return (
        src.device.type == "npu"
        and src.dtype not in _L2_DTYPES
        and index.dim() == 1
    )


def _avoid_output_aliases(src_work, index_work, work_out):
    if work_out is None:
        return src_work, index_work
    if src_work.data_ptr() == work_out.data_ptr():
        src_work = src_work.clone()
    if index_work.data_ptr() == work_out.data_ptr():
        index_work = index_work.clone()
    return src_work, index_work


def _run_l2_fallback(
    src: torch.Tensor,
    src_work: torch.Tensor,
    index_work: torch.Tensor,
    plan: _PreparedScatter,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if src.device.type == "npu" and src.dtype not in _warned_l2:
        warnings.warn(
            f"ops_gnn.scatter: {src.dtype} uses the synchronous CPU fallback "
            "because Ascend 950 has no native float64/int64 atomic reduction path",
            RuntimeWarning,
            stacklevel=3,
        )
        _warned_l2.add(src.dtype)
    cpu_out = plan.work_out.cpu() if plan.work_out is not None else None
    cpu_request = _ScatterRequest(plan.dim, cpu_out, plan.request.dim_size, plan.request.reduce)
    result, arg_result = _reference_forward(
        src_work.cpu(), index_work.cpu(), cpu_request, plan.output_shape
    )
    result = result.to(src.device)
    if arg_result is not None:
        arg_result = arg_result.to(src.device)
    return _copy_back_if_needed(result, plan.original_out), arg_result


def _should_find_hot_target(
    src: torch.Tensor,
    plan: _PreparedScatter,
    use_compact_dim_index: bool,
) -> bool:
    if not use_compact_dim_index or plan.original_out is not None:
        return False
    before, _, after = _shape_factors(src, plan.dim)
    if before != 1 or not 0 < after <= 4096:
        return False
    supported_dtype = src.dtype in (torch.float16, torch.float32)
    supported_reduce = plan.request.reduce in ("sum", "mean", "min", "max")
    return supported_dtype and supported_reduce


def _run_npu(
    src: torch.Tensor,
    index_work: torch.Tensor,
    src_work: torch.Tensor,
    plan: _PreparedScatter,
    use_compact_dim_index: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if _scatter_npu is None:
        raise RuntimeError(
            "Ascend C extension is not built. Run scripts/build.sh in a "
            "CANN environment before using L1 NPU tensors."
        )
    work_out = plan.work_out
    if work_out is None:
        work_out = torch.empty(plan.output_shape, dtype=src.dtype, device=src.device)
    hot_target = -1
    if _should_find_hot_target(src, plan, use_compact_dim_index):
        hot_target = _index_hot_target(index_work)
    result, arg_result = _scatter_npu.scatter_forward(
        src_work,
        index_work,
        plan.dim,
        work_out,
        _REDUCE_CODE[plan.request.reduce],
        plan.original_out is not None,
        hot_target,
    )
    is_minmax = plan.request.reduce in ("min", "max")
    if is_minmax and arg_result.dtype != torch.int64:
        arg_result = arg_result.to(dtype=torch.int64)
    result = _copy_back_if_needed(result, plan.original_out)
    return result, arg_result if is_minmax else None


def _scatter_forward(
    src: torch.Tensor, index: torch.Tensor, request: _ScatterRequest
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    plan = _prepare_scatter(src, index, request)

    # torch_scatter.scatter_mean builds its count tensor from the *original*
    # index.shape before broadcasting.  When index has no axis at ``dim``, the
    # upstream implementation reduces the last index axis instead.  Therefore
    # a singleton 1D index scattered at dim > 0 contributes one count even
    # though its value broadcasts over the complete src scatter dimension.
    # A present-but-collapsed scatter axis has the same one-count behavior.
    # Preserve the fused native mean path for equivalent shapes (including the
    # performance matrix's compact 1D dim-0 index), and compose the exact
    # upstream sum/count algorithm only where the broadcast can change counts.
    if request.reduce == "mean" and _needs_original_mean_count(src, index, plan.dim):
        return _run_original_mean_count(src, index, plan)

    src_work = src.contiguous()
    # Preserve torch_scatter's common 1D dim index as a compact vector.  The
    # NPU kernel applies the implicit zero-stride broadcast, avoiding an N*C
    # int64 materialization on the performance matrix.
    use_compact_dim_index = _uses_compact_dim_index(src, index)
    index_work = (
        index.contiguous()
        if use_compact_dim_index
        else plan.expanded_index.contiguous()
    )
    src_work, index_work = _avoid_output_aliases(
        src_work, index_work, plan.work_out
    )

    if src.dtype in _L2_DTYPES:
        return _run_l2_fallback(src, src_work, index_work, plan)

    if src.device.type == "npu":
        return _run_npu(src, index_work, src_work, plan, use_compact_dim_index)

    # CPU/CUDA development path.  It is deliberately ordered and doubles as
    # the golden implementation used by the pytest suite.
    reference_request = _ScatterRequest(
        plan.dim, plan.work_out, request.dim_size, request.reduce
    )
    result, arg_result = _reference_forward(
        src_work, index_work, reference_request, plan.output_shape
    )
    return _copy_back_if_needed(result, plan.original_out), arg_result


def scatter_sum(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    request = _ScatterRequest(dim, out, dim_size, "sum")
    return _scatter_forward(src, index, request)[0]


def scatter_add(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    request = _ScatterRequest(dim, out, dim_size, "mul")
    return _scatter_forward(src, index, request)[0]


def scatter_mean(src: torch.Tensor,
                 index: torch.Tensor,
                 dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    request = _ScatterRequest(dim, out, dim_size, "mean")
    return _scatter_forward(src, index, request)[0]


def scatter_min(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    request = _ScatterRequest(dim, out, dim_size, "min")
    result, arg_result = _scatter_forward(src, index, request)
    if arg_result is None:
        raise RuntimeError("scatter_min did not return arg_out")
    return result, arg_result


def scatter_max(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    request = _ScatterRequest(dim, out, dim_size, "max")
    result, arg_result = _scatter_forward(src, index, request)
    if arg_result is None:
        raise RuntimeError("scatter_max did not return arg_out")
    return result, arg_result


def scatter(src: torch.Tensor,
            index: torch.Tensor,
            dim: int = -1,
            out: Optional[torch.Tensor] = None,
            dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    """Match ``torch_scatter.scatter(src, index, dim, out, dim_size, reduce)``."""
    request = _ScatterRequest(dim, out, dim_size, reduce)
    if request.reduce not in _REDUCE_CODE:
        raise ValueError(
            "reduce must be one of 'sum', 'add', 'mul', 'mean', 'min', or 'max'"
        )
    return _scatter_forward(src, index, request)[0]

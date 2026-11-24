# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU reference for Scatter's public forward interfaces.

Validated against PyTorch 2.7.1+cpu and torch_scatter 2.1.2. Source semantics
were reviewed at pytorch_scatter commit 140d3ad677aae615767412873b90982cbf97d35d;
the installed binary is identified separately in the local validation record.
Only PyTorch and the standard library are used here. Reference self-checks
use exact finite values, matching NaNs and signed zeros, without NPU tolerance.
"""

from itertools import product
import math
from typing import NamedTuple, Optional, Tuple

import torch

DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)
REDUCES = ("sum", "add", "mul", "mean", "min", "max")
PRECISION_COMMIT = "a5e8e716743b20bfb2ffa16e0213c4aa73b823b3"
LEGACY_PRECISION_COMMIT = "adbe441e9254676174937047457419b6e645e384"
_CURRENT_THRESHOLDS = {
    torch.float16: (2**-10, 2**-10, 0.1),
    torch.bfloat16: (2**-7, 2**-7, 0.1),
    torch.float32: (2**-13, 2**-13, 0.01),
}
_LEGACY_THRESHOLDS = {
    torch.float16: (2**-9, 2**-9, 0.1),
    torch.bfloat16: (2**-6, 2**-6, 1.0),
    torch.float32: (2**-16, 2**-10, 0.01),
}


class _ReductionOptions(NamedTuple):
    """Geometry, initial output and operation for the ordered reference."""

    dim: int
    out: Optional[torch.Tensor]
    dim_size: Optional[int]
    reduce: str


def _broadcast(index, src, dim):
    axis = dim % src.ndim if src.ndim else dim
    shape = list(index.shape)
    if index.ndim == 1:
        shape = [1] * axis + shape
    shape += [1] * (src.ndim - len(shape))
    return index.reshape(shape).expand(src.shape)


def _allocate(src, expanded_index, dim, dim_size):
    shape = list(src.shape)
    inferred = int(expanded_index.max()) + 1 if expanded_index.numel() else 0
    shape[dim] = inferred if dim_size is None else dim_size
    return torch.zeros(shape, dtype=src.dtype, device=src.device)


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Use the same public PyTorch primitive, including its overlap checks."""
    expanded = _broadcast(index, src, dim)
    result = _allocate(src, expanded, dim, dim_size) if out is None else out
    return result.scatter_add_(dim, expanded, src)


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Alias of scatter_sum, including accumulation into the provided out."""
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Count original index elements with source-dtype wrapping and clamp."""
    result = scatter_sum(src, index, dim, out, dim_size)
    count_dim = min(dim % src.ndim, index.ndim - 1)
    count = scatter_sum(torch.ones_like(index, dtype=src.dtype), index, count_dim, dim_size=result.size(dim))
    count.masked_fill_(count < 1, 1)
    divisor = _broadcast(count, result, dim)
    if result.is_floating_point():
        return result.true_divide_(divisor)
    return result.div_(divisor, rounding_mode="floor")


def _initial_value(dtype, reduce):
    if reduce == "mul":
        return 1
    limits = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    return limits.max if reduce == "min" else limits.min


def _update(result, arg, target, sample, reduce):
    """Apply one source sample, retaining strict extrema comparisons."""
    value, position = sample
    if reduce == "mul":
        result[target] *= value
        return
    improved = value < result[target] if reduce == "min" else value > result[target]
    if bool(improved):
        result[target] = value
        arg[target] = position


def _ordered_reduce(src, index, options):
    """Evaluate ordered reductions in the CPU source traversal order."""
    out = options.out
    dirty_leaf = out is not None and out.requires_grad and out.is_leaf
    if dirty_leaf and torch.is_grad_enabled():
        # The CPU extension computes inside its autograd forward before the
        # public wrapper rejects a dirty leaf. The mutation remains visible.
        with torch.no_grad():
            _ordered_reduce(src, index, options)
        raise RuntimeError("a leaf Variable that requires grad was used in an in-place operation")
    if out is not None and not out.is_contiguous():
        raise RuntimeError("CPU public mul/min/max rejects noncontiguous out")
    axis = options.dim % src.ndim
    expanded = _broadcast(index, src, axis)
    source = src.contiguous()
    result = _allocate(src, expanded, axis, options.dim_size) if out is None else out
    arg = torch.full_like(result, src.size(axis), dtype=torch.int64)
    if not src.numel():
        return result, arg
    initial = _initial_value(src.dtype, options.reduce)
    if out is None:
        result.fill_(initial)
    # Row-major reads preserve CPU's observable ordering when src aliases out.
    for coordinate in product(*(range(size) for size in src.shape)):
        target = list(coordinate)
        target[axis] = int(expanded[coordinate])
        _update(result, arg, tuple(target), (source[coordinate], coordinate[axis]), options.reduce)
    if out is None and options.reduce != "mul":
        result.masked_fill_(result == initial, 0)
    return result, arg


def scatter_mul(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Multiply in CPU order, including the provided out initial value."""
    return _ordered_reduce(src, index, _ReductionOptions(dim, out, dim_size, "mul"))[0]


def scatter_min(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return minima with the first tied index, or a sentinel for unchanged out."""
    return _ordered_reduce(src, index, _ReductionOptions(dim, out, dim_size, "min"))


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return maxima with the first tied index, or a sentinel for unchanged out."""
    return _ordered_reduce(src, index, _ReductionOptions(dim, out, dim_size, "max"))


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Dispatch the public reduce name; min and max return values only."""
    functions = {
        "sum": scatter_sum,
        "add": scatter_add,
        "mul": scatter_mul,
        "mean": scatter_mean,
        "min": scatter_min,
        "max": scatter_max,
    }
    if reduce not in functions:
        raise ValueError("unsupported reduction")
    result = functions[reduce](src, index, dim, out, dim_size)
    return result[0] if isinstance(result, tuple) else result


def assert_exact(actual, expected):
    """Zero-tolerance comparison, matching NaN positions and signed zeros."""
    if type(actual) is not type(expected):
        raise AssertionError("return type differs")
    if isinstance(expected, tuple):
        if len(actual) != len(expected):
            raise AssertionError("tuple length differs")
        for actual_item, expected_item in zip(actual, expected):
            assert_exact(actual_item, expected_item)
        return
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    if actual.dtype != expected.dtype:
        raise AssertionError("dtype differs")
    if actual.shape != expected.shape:
        raise AssertionError("shape differs")
    if not expected.is_floating_point():
        if not torch.equal(actual, expected):
            raise AssertionError("integer values differ")
        return
    expected_nan = torch.isnan(expected)
    if not torch.equal(torch.isnan(actual), expected_nan):
        raise AssertionError("NaN locations differ")
    if not torch.equal(actual[~expected_nan], expected[~expected_nan]):
        raise AssertionError("values differ")
    zeros = expected == 0
    if not torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros])):
        raise AssertionError("zero signs differ")


def assert_bitwise(actual, expected):
    """Compare floating payloads and signed zeros as well as finite values."""
    assert_exact(actual, expected)
    if isinstance(expected, tuple):
        for actual_item, expected_item in zip(actual, expected):
            assert_bitwise(actual_item, expected_item)
        return
    actual_bytes = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    expected_bytes = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    if not torch.equal(actual_bytes, expected_bytes):
        raise AssertionError("tensor bytes differ")


def _ulp(value, dtype):
    """ULP spacing includes zero/subnormals and the largest finite binade."""
    limits = torch.finfo(dtype)
    magnitude = abs(float(value))
    if not math.isfinite(magnitude):
        raise ValueError("finite-error golden narrows to infinity: ULP is unspecified")
    if magnitude < limits.tiny:
        return limits.tiny * limits.eps
    _, exponent = math.frexp(magnitude)
    return math.ldexp(limits.eps, exponent - 1)


def _special_matches(actual, expected):
    narrowed = expected.to(actual.dtype)
    same_nan = torch.isnan(actual) & torch.isnan(expected)
    same_inf = torch.isinf(actual) & torch.isinf(narrowed) & (actual == narrowed)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    special_ok = bool((finite | same_nan | same_inf).all())
    both_zero = (actual == 0) & (expected == 0)
    zero_ok = torch.equal(torch.signbit(actual[both_zero]), torch.signbit(expected[both_zero]))
    return finite, same_nan | same_inf, special_ok and zero_ok


def _maximum_error(expected, difference, finite, dtype):
    """Return the largest finite error and its narrowed golden value."""
    finite_errors = difference[finite]
    if not finite_errors.numel():
        return 0.0, 0.0
    position = int(finite_errors.argmax())
    return float(finite_errors[position]), float(expected[finite][position].to(dtype))


def _precision_metrics(actual, expected, thresholds):
    atol, rtol = thresholds[:2]
    finite, special, special_ok = _special_matches(actual, expected)
    difference = (actual.double() - expected.double()).abs()
    matched = special | (finite & (difference <= atol + rtol * expected.double().abs()))
    max_error, anchor = _maximum_error(expected, difference, finite, actual.dtype)
    limit = max(thresholds[2], 32 * _ulp(anchor, actual.dtype))
    ratio = float(matched.sum()) / actual.numel() if actual.numel() else 1.0
    return {
        "passed": ratio >= 0.99 and max_error <= limit and special_ok,
        "matched_ratio": ratio,
        "max_abs_error": max_error,
        "max_abs_error_limit": limit,
        "golden_ulp_anchor": anchor,
        "atol": atol,
        "rtol": rtol,
        "required_matched_ratio": 0.99,
        "special_values_match": special_ok,
    }


def compare_precision(actual, expected):
    """Latest official mixed tolerance, with legacy thresholds as diagnostics.

    Source: cann/opbase docs/zh/ops_precision_standard/mixed_tolerance_standard.md
    at PRECISION_COMMIT. Legacy experimental_standard.md at
    LEGACY_PRECISION_COMMIT did not define its ULP anchor; its diagnostic uses
    the clarified current anchor. NaN locations, Inf signs and signed zeros
    must match independently of the 99% element ratio. Integers and FLOAT64
    require bitwise equality. CPU golden self-validation uses assert_exact.
    """
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.shape != expected.shape:
        raise AssertionError("shape differs")
    if actual.dtype not in _CURRENT_THRESHOLDS:
        try:
            assert_bitwise(actual, expected)
            passed = True
        except AssertionError:
            passed = False
        return {"passed": passed, "comparison": "bitwise", "current": {"passed": passed}, "legacy": {"passed": passed}}
    if not expected.is_floating_point():
        raise AssertionError("floating output requires floating golden")
    current = _precision_metrics(actual, expected, _CURRENT_THRESHOLDS[actual.dtype])
    legacy = _precision_metrics(actual, expected, _LEGACY_THRESHOLDS[actual.dtype])
    return {
        "passed": current["passed"],
        "current": current,
        "legacy": legacy,
        "standard_commit": PRECISION_COMMIT,
        "legacy_standard_commit": LEGACY_PRECISION_COMMIT,
        "legacy_diagnostic": "old thresholds with clarified current ULP anchor",
    }

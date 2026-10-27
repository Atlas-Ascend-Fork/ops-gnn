# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent PyTorch reference for CPU-compatible CSR forward reductions.

Numerical rules reviewed in pytorch_scatter/csrc/cpu/segment_csr_cpu.cpp
and reducer.h at f514c10f920b5aeed2eb162092f0ad20d3edee52. This module
uses only PyTorch; the native extension is reserved for separate validation.
Undefined native zero-divisor inputs are rejected by this reference.
Floating mean uses the full segment length on A2, including above INT32_MAX;
the CPU extension's signed-int count overflow is intentionally not reproduced.
"""

import math

import torch

DTYPES = (
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
)
REDUCES = ("sum", "add", "mean", "min", "max")


def checked_indptr(src, indptr, out=None):
    if src.device.type != "cpu" or indptr.device.type != "cpu":
        raise ValueError("reference requires CPU src and indptr")
    if out is not None and out.device.type != "cpu":
        raise ValueError("reference requires CPU out")
    if indptr.dtype != torch.int64 or not 1 <= indptr.ndim <= src.ndim:
        raise ValueError("invalid reference indptr metadata")
    if indptr.size(-1) < 1:
        raise ValueError("CSR pointer must contain at least one boundary")
    dim = indptr.ndim - 1
    expanded = indptr.expand(*src.shape[:dim], indptr.size(-1))
    if (
        bool((expanded < 0).any())
        or bool((expanded > src.size(dim)).any())
        or bool((expanded[..., 1:] < expanded[..., :-1]).any())
    ):
        raise ValueError("reference pointers must be nondecreasing and in range")
    if out is not None:
        shape = list(src.shape)
        shape[dim] = indptr.size(-1) - 1
        if out.dtype != src.dtype or list(out.shape) != shape:
            raise ValueError("invalid reference output")
    return expanded


def _check_defined(src, pointers, reduce):
    if reduce not in REDUCES or src.dtype not in DTYPES:
        raise ValueError("unsupported reference reduction or dtype")
    if src.numel() and pointers.size(-1) == 1:
        raise ValueError("CPU CSR divides by zero for nonempty src and zero segments")
    counts = pointers[..., 1:] - pointers[..., :-1]
    if reduce == "mean" and src.dtype in (torch.int8, torch.uint8):
        native_counts = counts.to(torch.int32).clamp_min(1)
        if bool((native_counts.remainder(256) == 0).any()):
            raise ValueError("CPU byte mean has an undefined zero divisor")
    return counts


def _initial(shape, dtype, reduce):
    work = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    value = 0
    if reduce in ("min", "max"):
        limits = torch.finfo(work) if work.is_floating_point else torch.iinfo(work)
        value = limits.max if reduce == "min" else limits.min
    return torch.full(shape, value, dtype=work)


def _fold_rows(source, offsets, count, reduce):
    """Fold equally long segments together, preserving source order per lane."""
    total = _initial((offsets.numel(), source.size(-1)), source.dtype, reduce)
    arguments = torch.full(total.shape, -1, dtype=torch.int64)
    compare = torch.lt if reduce == "min" else torch.gt
    for row in range(count):
        incoming = source[offsets + row].to(total.dtype)
        if reduce in ("sum", "add", "mean"):
            total.add_(incoming)
        else:
            improved = compare(incoming, total)
            total = torch.where(improved, incoming, total)
            arguments.masked_fill_(improved, row)
    return total.to(source.dtype), arguments


def _divide_mean(total, count):
    divisor = torch.tensor(max(count, 1), dtype=torch.int64)
    if total.is_floating_point():
        # A2 keeps the full count, then applies the source-type rounding rule.
        return total / divisor.float().to(total.dtype)
    # Integer reducers retain the native signed-int count and source-type casts.
    divisor = divisor.to(torch.int32).clamp_min(1).to(total.dtype)
    return total.long().div(divisor.long(), rounding_mode="trunc").to(total.dtype)


def _reduce_segments(src, pointers, counts, reduce):
    dim = pointers.ndim - 1
    length = src.size(dim)
    feature_dim = dim + 1
    features = math.prod(src.shape[feature_dim:])
    segments = pointers.size(-1) - 1
    bounds = pointers.reshape(-1, segments + 1)
    starts = bounds[:, :-1].reshape(-1)
    offsets = (bounds[:, :-1] + torch.arange(bounds.size(0))[:, None] * length).reshape(-1)
    lengths = counts.reshape(-1)
    source = src.reshape(-1, features)
    values = torch.zeros((lengths.numel(), features), dtype=src.dtype)
    args = torch.full(values.shape, -1, dtype=torch.int64)
    for count in lengths.unique().tolist():
        if count == 0:
            continue
        tasks = (lengths == count).nonzero().flatten()
        total, local_arg = _fold_rows(source, offsets[tasks], count, reduce)
        values[tasks] = _divide_mean(total, count) if reduce == "mean" else total
        args[tasks] = torch.where(local_arg >= 0, local_arg + starts[tasks, None], -1)
    return values, _carry_arguments(args, lengths, length)


def _carry_arguments(args, counts, length):
    # Native args starts at zero once and survives empty segments and batches.
    tasks = torch.arange(1, args.size(0) + 1)[:, None].expand_as(args)
    latest = torch.where(args >= 0, tasks, 0).cummax(dim=0).values
    candidates = torch.cat((torch.zeros((1, args.size(1)), dtype=torch.int64), args))
    carried = candidates.gather(0, latest)
    return torch.where(counts[:, None] > 0, carried, length)


def cpu_reference(src, indptr, out=None, reduce="sum", return_arg=False):
    """Return independent values/args, preserving the caller's output buffer."""
    pointers = checked_indptr(src, indptr, out)
    counts = _check_defined(src, pointers, reduce)
    shape = list(src.shape)
    shape[indptr.ndim - 1] = indptr.size(-1) - 1
    values = torch.zeros(shape, dtype=src.dtype) if out is None else out.clone()
    args = torch.full(shape, src.size(indptr.ndim - 1), dtype=torch.int64)
    if src.numel() and values.numel():
        values, args = _reduce_segments(src, pointers, counts, reduce)
        values, args = values.reshape(shape), args.reshape(shape)
    return (values, args) if return_arg and reduce in ("min", "max") else values


def zero_segment_task_reference(src, indptr):
    """Shape contract for a valid zero-segment request; CPU C++ is undefined."""
    checked_indptr(src, indptr)
    if indptr.size(-1) != 1:
        raise ValueError("zero-segment-only task reference")
    shape = list(src.shape)
    shape[indptr.ndim - 1] = 0
    return torch.empty(shape, dtype=src.dtype)


def precision_passed(metrics):
    return metrics.get(
        "exact", metrics.get("matched_ratio", 0) >= 0.99 and metrics.get("hard_limit_pass", False)
    )


def _ulp_at_output_precision(expected, dtype):
    """Spacing at the narrowed reference, including zero and subnormals."""
    info = torch.finfo(dtype)
    magnitude = expected.to(dtype).double().abs()
    _, exponent = torch.frexp(magnitude)
    minimum = info.tiny * info.eps
    spacing = torch.ldexp(torch.full_like(magnitude, info.eps), exponent - 1)
    spacing = torch.where(magnitude == 0, minimum, spacing).clamp_min(minimum)
    return spacing


def precision_metrics(actual, expected):
    """Official mixed tolerance for floats; exact integer values and arguments."""
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    if actual.shape != expected.shape:
        raise AssertionError("actual and expected tensors must have matching shapes")
    if not actual.is_floating_point():
        if actual.dtype != expected.dtype:
            raise AssertionError("integer actual and expected dtypes must match")
        return {"exact": torch.equal(actual, expected)}, {"allowed_mismatches": 0}
    if not expected.is_floating_point():
        raise AssertionError("floating actual requires a floating reference")
    tolerance, floor = {
        torch.float16: (2**-10, 0.1),
        torch.bfloat16: (2**-7, 0.1),
        torch.float32: (2**-13, 0.01),
    }[actual.dtype]
    a, b = actual.double(), expected.double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    narrowed = expected.to(actual.dtype).double()
    special = (torch.isnan(a) & torch.isnan(b)) | (torch.isinf(a) & (a == narrowed))
    error = torch.where(finite, (a - b).abs(), 0.0)
    matches = (finite & (error <= tolerance * (1 + b.abs()))) | special
    ulp = _ulp_at_output_precision(torch.where(finite, b, 0.0), actual.dtype)
    flat = int(error.flatten().argmax()) if error.numel() else 0
    limit = max(floor, 32 * float(ulp.flatten()[flat])) if error.numel() else floor
    maximum = float(error.max()) if error.numel() else 0.0
    return {
        "matched_ratio": float(matches.double().mean()) if matches.numel() else 1.0,
        "max_abs_error": maximum,
        "hard_limit": limit,
        "hard_limit_pass": maximum <= limit and bool((finite | special).all()),
    }, {"rtol": tolerance, "atol": tolerance, "required_matched_ratio": 0.99}


def assert_close(actual, expected):
    metrics, limits = precision_metrics(actual, expected)
    passed = metrics.get(
        "exact", metrics.get("matched_ratio", 0) >= 0.99 and metrics.get("hard_limit_pass", False)
    )
    if not passed:
        raise AssertionError((metrics, limits))

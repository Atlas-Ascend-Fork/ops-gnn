# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent PyTorch references for A2 CPU-compatible forward semantics.

CPU rules: pytorch_scatter/csrc/cpu/{segment_coo_cpu.cpp,reducer.h},
upstream commit f514c10f920b5aeed2eb162092f0ad20d3edee52 (version 2.1.2).
Source-order accumulation, out seeds, narrowed mean division and strict
extremum updates follow that source. Argument state survives segment/batch
boundaries; empty segments keep the source-length sentinel. A narrowed byte
mean divisor of zero explicitly produces zero without executing division.
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


def _check_cpu(src, index, out):
    if src.device.type != "cpu" or index.device.type != "cpu":
        raise ValueError("reference requires CPU src/index")
    if out is not None and out.device.type != "cpu":
        raise ValueError("reference requires CPU out")
    if index.dtype != torch.int64 or not 1 <= index.ndim <= src.ndim:
        raise ValueError("invalid reference index metadata")
    expanded = index.expand(src.shape[: index.ndim])
    if expanded.numel() and (
        bool((expanded < 0).any()) or bool((expanded[..., 1:] < expanded[..., :-1]).any())
    ):
        raise ValueError("reference indices must be nonnegative and sorted")
    return expanded


def cpu_reference(src, index, out=None, dim_size=None, reduce="sum"):
    """Compute COO values on CPU, accumulating each segment in source order."""
    return _coo_reference(src, index, out, dim_size, reduce)[0]


def cpu_reference_arg(src, index, out=None, dim_size=None, reduce="min"):
    """Return extrema and the first improving position, retaining argument state."""
    if reduce not in ("min", "max"):
        raise ValueError("CPU arg reference requires min or max")
    values, arguments = _coo_reference(src, index, out, dim_size, reduce)
    return values, _carry_arguments(src, index, arguments)


def _mean_in_source_type(total, count):
    divisor = torch.tensor(count, dtype=torch.int64).to(total.dtype)
    if divisor.item() == 0:
        return torch.zeros_like(total)
    if total.is_floating_point():
        return total / divisor
    return total.long().div(divisor.long(), rounding_mode="trunc").to(total.dtype)


def _fold_segment(rows, initial, positions, reduce):
    dtype = rows.dtype
    work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    accumulator = initial.to(work_dtype).clone()
    first = torch.full(initial.shape, -1, dtype=torch.int64)
    compare = torch.lt if reduce == "min" else torch.gt
    for offset in range(rows.size(0)):
        incoming = rows[offset].to(work_dtype)
        if reduce in ("sum", "mean"):
            accumulator.add_(incoming)
        else:
            improved = compare(incoming, accumulator)
            accumulator[improved] = incoming[improved]
            first[improved] = positions[offset]
    result = accumulator.to(dtype)
    if reduce == "mean":
        result = _mean_in_source_type(result, rows.size(0))
    return result, first


def _coo_buffers(src, index, out, dim_size):
    dim = index.ndim - 1
    labels = index.expand(src.shape[:index.ndim])
    segments = int(labels.max()) + 1 if labels.numel() else 0
    if dim_size is not None:
        segments = int(dim_size)
    if out is not None:
        segments = out.size(dim)
    shape = (*src.shape[:dim], segments, *src.shape[dim + 1:])
    if segments < 0 or (labels.numel() and int(labels.max()) >= segments):
        raise ValueError("reference index out of range")
    if out is not None and (out.shape != shape or out.dtype != src.dtype):
        raise ValueError("reference out must match the output shape and source dtype")
    result = src.new_zeros(shape) if out is None else out.clone().contiguous()
    arguments = torch.full(shape, src.size(dim), dtype=torch.int64)
    batches, channels = math.prod(src.shape[:dim]), math.prod(src.shape[dim + 1:])
    return (
        src.reshape(batches, src.size(dim), channels),
        labels.reshape(batches, src.size(dim)),
        result.reshape(batches, segments, channels),
        arguments.reshape(batches, segments, channels),
    )


def _reduce_coo(src, index, out, dim_size, reduce):
    source, labels, result, arguments = _coo_buffers(src, index, out, dim_size)
    limits = torch.finfo(src.dtype) if src.is_floating_point() else torch.iinfo(src.dtype)
    extreme = reduce in ("min", "max")
    identity = limits.max if reduce == "min" else limits.min
    for batch, row_labels in enumerate(labels):
        for label in torch.unique_consecutive(row_labels).tolist():
            positions = (row_labels == label).nonzero().flatten()
            initial = result[batch, label]
            if out is None and extreme:
                initial = torch.full_like(initial, identity)
            elif out is not None and positions[0].item() == 0:
                initial = result[batch, 0]
            value, first = _fold_segment(source[batch, positions], initial, positions, reduce)
            result[batch, label] = value
            arguments[batch, label] = torch.where(first >= 0, first, source.size(1))
    if out is None and extreme:
        result.masked_fill_(result == identity, 0)
    dim = index.ndim - 1
    shape = (*src.shape[:dim], result.size(1), *src.shape[dim + 1:])
    return result.reshape(shape), arguments.reshape(shape)


def _carry_arguments(src, index, arguments):
    """Carry the last improving coordinate across nonempty segments and batches.

    An initial value or NaN-only segment may leave the coordinate unchanged.
    State starts at zero for each feature; empty segments retain the length sentinel.
    """
    if not src.numel():
        return arguments
    dim = index.ndim - 1
    batches, length = math.prod(src.shape[:dim]), src.size(dim)
    channels, segments = math.prod(src.shape[dim + 1:]), arguments.size(dim)
    indices = index.expand(src.shape[:index.ndim]).reshape(batches, length)
    counts = torch.zeros((batches, segments), dtype=torch.int64)
    counts.scatter_add_(1, indices, torch.ones_like(indices))
    nonempty = counts.reshape(-1, 1) > 0
    flat = arguments.reshape(batches * segments, channels)
    positions = torch.arange(flat.size(0)).view(-1, 1).expand_as(flat)
    last_update = torch.where(nonempty & (flat != length), positions, -1).cummax(0).values
    carried = flat.gather(0, last_update.clamp_min(0))
    carried = torch.where(last_update >= 0, carried, 0)
    return torch.where(nonempty, carried, flat).reshape(arguments.shape)


def _restore_prefix(result, source_shape, dim):
    if isinstance(result, tuple):
        return tuple(_restore_prefix(value, source_shape, dim) for value in result)
    shape = (result.size(0), *source_shape[:dim], *source_shape[dim + 1:])
    return result.reshape(shape).movedim(0, dim).contiguous()


def _coo_reference(src, index, out, dim_size, reduce):
    _check_cpu(src, index, out)
    reduce = "sum" if reduce == "add" else reduce
    if reduce not in ("sum", "mean", "min", "max"):
        raise ValueError("unsupported reduction: " + str(reduce))
    dim = index.ndim - 1
    if not src.numel() or dim == 0 or index.numel() != src.size(dim):
        return _reduce_coo(src, index, out, dim_size, reduce)
    # Shared indices let prefix batches become independent feature columns.
    # Each column still visits its source rows in the original order.
    if out is not None and out.shape[:dim] + out.shape[dim + 1:] != src.shape[:dim] + src.shape[dim + 1:]:
        raise ValueError("output shape does not match source")
    values = src.movedim(dim, 0).reshape(src.size(dim), -1)
    initial = None if out is None else out.movedim(dim, 0).reshape(out.size(dim), -1)
    result = _reduce_coo(values, index.reshape(-1), initial, dim_size, reduce)
    return _restore_prefix(result, src.shape, dim)


def _csr_bounds(src, indptr):
    """Validate the one-dimensional CSR pointer and return its segment bounds."""
    if src.device.type != "cpu" or indptr.device.type != "cpu":
        raise ValueError("CSR reference requires CPU tensors")
    if indptr.ndim != 1 or indptr.dtype != torch.int64 or not indptr.numel():
        raise ValueError("CSR reference requires a nonempty one-dimensional int64 pointer")
    bounds = indptr.tolist()
    if bounds[0] < 0 or bounds[-1] > src.size(0) or bounds != sorted(bounds):
        raise ValueError("CSR pointers must be sorted and within source bounds")
    return bounds


def cpu_csr_reference(src, indptr, reduce="sum"):
    """Independent slice reductions for the one-dimensional CSR consistency cases."""
    bounds = _csr_bounds(src, indptr)
    if reduce not in REDUCES:
        raise ValueError("unsupported reduction: " + str(reduce))
    result = src.new_zeros((len(bounds) - 1, *src.shape[1:]))
    for segment, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        if begin == end:
            continue
        rows = src[begin:end]
        mode = "sum" if reduce == "add" else reduce
        work_dtype = torch.float32 if src.dtype in (torch.float16, torch.bfloat16) else src.dtype
        limits = torch.finfo(work_dtype) if src.is_floating_point() else torch.iinfo(work_dtype)
        seed = limits.max if mode == "min" else limits.min if mode == "max" else 0
        initial = torch.full(src.shape[1:], seed, dtype=work_dtype)
        result[segment] = _fold_segment(rows, initial, torch.arange(begin, end), mode)[0]
    return result


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

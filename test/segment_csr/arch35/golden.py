# Copyright (c) 2026 Starlink_. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""

CPU golden reference for segment_csr: a 1:1 replica of the reference
implementation shipped with the official test suite (the acceptance
baseline), plus a first-occurrence arg reference for min/max.
"""

import math

import torch


def _mean_from_typed_sum(total, counts):
    """Preserve reference count conversion and wrapped integer division."""
    if total.is_floating_point():
        # Keep the floating reference's count conversion semantics.
        mean = total.float() / counts.to(total.dtype).float()
    else:
        # Match the native mean contract: wrap count to source dtype.
        count = counts.to(total.dtype)
        valid_count = count != 0
        mean = torch.zeros_like(total)
        mean[valid_count] = torch.div(
            total[valid_count], count[valid_count], rounding_mode="trunc"
        )
    return mean


def _csr_views(src, indptr):
    sc = src.cpu()
    ipc = indptr.cpu().long()
    dim = ipc.dim() - 1
    pointer_shape = list(sc.shape[:dim]) + [ipc.size(-1)]
    ipc = ipc.expand(pointer_shape).contiguous()
    batch_count = math.prod(sc.shape[:dim])
    feature_count = math.prod(sc.shape[dim + 1:])
    segments = ipc.size(-1) - 1
    output_shape = list(sc.shape)
    output_shape[dim] = segments
    values = sc.contiguous().reshape(batch_count, sc.size(dim), feature_count)
    pointers = ipc.reshape(batch_count, segments + 1)
    return sc, dim, output_shape, values, pointers


def _reduce_segment(selected, reduce):
    if reduce in ("sum", "add", "mean"):
        total = selected.sum(dim=0).to(selected.dtype)
        if reduce == "mean":
            counts = torch.full_like(total, selected.size(0), dtype=torch.int64)
            total = _mean_from_typed_sum(total, counts).to(selected.dtype)
        return total
    if reduce == "min":
        return selected.min(dim=0).values
    if reduce == "max":
        return selected.max(dim=0).values
    return None


def segment_csr_cpu(src, indptr, reduce="sum"):
    sc, dim, output_shape, values, pointers = _csr_views(src, indptr)
    batch_count, length, feature_count = values.shape
    segments = pointers.size(1) - 1
    out = sc.new_zeros(output_shape)
    output = out.reshape(batch_count, segments, feature_count)
    for batch in range(batch_count):
        for segment in range(segments):
            start = int(pointers[batch, segment])
            end = min(int(pointers[batch, segment + 1]), length)
            if start >= end:
                continue
            result = _reduce_segment(values[batch, start:end], reduce)
            if result is not None:
                output[batch, segment] = result
    return out


def segment_argminmax_cpu(src, indptr, reduce):
    """Return first-occurrence indices, including broadcast batched pointers."""
    if reduce not in ("min", "max"):
        raise AssertionError("Validation condition failed")
    sc, dim, output_shape, values, pointers = _csr_views(src, indptr)
    batch_count, length, feature_count = values.shape
    segments = pointers.size(1) - 1
    arg = torch.full(output_shape, length, dtype=torch.long)
    output = arg.reshape(batch_count, segments, feature_count)
    for batch in range(batch_count):
        for segment in range(segments):
            start = int(pointers[batch, segment])
            end = min(int(pointers[batch, segment + 1]), length)
            if start >= end:
                continue
            selected = values[batch, start:end]
            reduction = selected.min(dim=0) if reduce == "min" else selected.max(dim=0)
            output[batch, segment] = start + reduction.indices
    return arg

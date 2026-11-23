# Copyright (c) 2026.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent CPU references requiring only PyTorch.

Source-dtype integer accumulation/division, float32 half accumulation, empty
segments and first extremum ties follow the common CPU semantic domain.
Task-specific later ties, output overwrite and safe byte-mean zero division
remain explicit tests in test_segment_coo.py.
"""

from dataclasses import dataclass
from functools import partial
from math import prod
from typing import Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class ReferenceOptions:
    reduce: str = "sum"
    return_arg: bool = False


@dataclass(frozen=True)
class ReferenceBuffers:
    result: torch.Tensor
    arguments: torch.Tensor
    values: torch.Tensor
    indices: torch.Tensor
    output: torch.Tensor
    output_args: torch.Tensor
    length: int
    has_out: bool


def _reduce_rows(rows, seed, positions, mode, dtype):
    accumulator_dtype = (
        torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    )
    value = seed.to(accumulator_dtype).clone()
    argument = torch.full(seed.shape, -1, dtype=torch.int64)
    for row, position in zip(rows, positions):
        incoming = row.to(accumulator_dtype)
        if mode in ("sum", "mean"):
            value.add_(incoming)
        else:
            better = incoming < value if mode == "min" else incoming > value
            value = torch.where(better, incoming, value)
            argument = torch.where(better, position, argument)
    if mode == "mean" and dtype.is_floating_point:
        return (value / len(positions)).to(dtype), argument
    value = value.to(dtype)
    if mode == "mean":
        divisor = torch.tensor(len(positions), dtype=torch.int64).to(dtype)
        if divisor.item() == 0:
            raise ValueError("byte mean has an undefined narrowed-count zero divisor")
        value = torch.div(
            value.to(torch.int64), divisor.to(torch.int64), rounding_mode="trunc"
        ).to(dtype)
    return value, argument


def _validate_sources(src, index, out):
    if src.device.type != "cpu" or index.device.type != "cpu":
        raise ValueError("golden inputs must be on CPU")
    if out is not None and out.device.type != "cpu":
        raise ValueError("golden output buffer must be on CPU")


def _reduction_mode(reduce):
    mode = "sum" if reduce == "add" else reduce
    if mode not in ("sum", "mean", "min", "max"):
        raise ValueError("unsupported reduction: " + str(reduce))
    return mode


def _segment_count(expanded, out, dim_size, dim):
    if out is not None:
        return out.size(dim)
    if dim_size is not None:
        return int(dim_size)
    return int(expanded.max()) + 1 if expanded.numel() else 0


def _reference_buffers(src, index, out, dim_size):
    if index.ndim < 1 or index.ndim > src.ndim or index.dtype != torch.int64:
        raise ValueError("indices must be an int64 tensor with valid rank")
    dim = index.ndim - 1
    expanded = index.expand(src.shape[:index.ndim])
    length = src.size(dim)
    batches = prod(src.shape[:dim])
    channels = prod(src.shape[dim + 1:])
    segments = _segment_count(expanded, out, dim_size, dim)
    shape = list(src.shape)
    shape[dim] = segments
    if out is not None and list(out.shape) != shape:
        raise ValueError("output shape does not match source")
    result = (
        torch.zeros(shape, dtype=src.dtype) if out is None else out.clone().contiguous()
    )
    arguments = torch.full(shape, length, dtype=torch.int64)
    values = src.contiguous().reshape(batches, length, channels)
    indices = expanded.contiguous().reshape(batches, length)
    output = result.reshape(batches, segments, channels)
    output_args = arguments.reshape(batches, segments, channels)
    return ReferenceBuffers(result, arguments, values, indices, output,
                            output_args, length, out is not None)


def _validate_row_indices(row_index, segments):
    if not row_index.numel():
        return
    if (bool((row_index < 0).any()) or bool((row_index >= segments).any())
            or bool((row_index[1:] < row_index[:-1]).any())):
        raise ValueError("indices must be sorted and within output bounds")


def _reduce_segment(buffers, batch, segment, mode):
    positions = (buffers.indices[batch] == segment).nonzero().flatten()
    seed = buffers.output[batch, segment]
    if not buffers.has_out and mode in ("min", "max"):
        seed = buffers.values[batch, positions[0]]
    elif buffers.has_out and positions[0].item() == 0:
        seed = buffers.output[batch, 0]
    value, arg = _reduce_rows(buffers.values[batch, positions], seed, positions,
                              mode, buffers.values.dtype)
    if not buffers.has_out and mode in ("min", "max"):
        arg = torch.where(arg >= 0, arg, positions[0])
    buffers.output[batch, segment] = value
    buffers.output_args[batch, segment] = torch.where(arg >= 0, arg, buffers.length)


def _reduce_buffers(buffers, mode):
    for batch in range(buffers.values.size(0)):
        row_index = buffers.indices[batch]
        _validate_row_indices(row_index, buffers.output.size(1))
        for segment in torch.unique_consecutive(row_index).tolist():
            _reduce_segment(buffers, batch, segment, mode)


def segment_coo_reference(
    src, index, out=None, dim_size=None, options=ReferenceOptions()
):
    """Reduce sorted, broadcastable CPU COO indices without optional packages."""
    _validate_sources(src, index, out)
    mode = _reduction_mode(options.reduce)
    buffers = _reference_buffers(src, index, out, dim_size)
    _reduce_buffers(buffers, mode)
    if options.return_arg and mode in ("min", "max"):
        return buffers.result, buffers.arguments
    return buffers.result


def cpu_reference(
    src: torch.Tensor, index: torch.Tensor, out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None, options: ReferenceOptions = ReferenceOptions()
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return segment_coo_reference(src, index, out, dim_size, options)


def reference_function(name):
    """Return the independent sum/mean/min/max subinterface reference."""
    mode = name.removeprefix("segment_").removesuffix("_coo")
    if mode not in ("sum", "add", "mean", "min", "max"):
        raise ValueError("unsupported reference function: " + name)
    options = ReferenceOptions(reduce=mode, return_arg=mode in ("min", "max"))
    return partial(segment_coo_reference, options=options)


segment_mean_reference = reference_function("segment_mean_coo")

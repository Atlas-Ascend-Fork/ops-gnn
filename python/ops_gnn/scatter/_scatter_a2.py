# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2 Scatter composition with the torch_scatter 2.1.2 CPU semantics."""

import ctypes
import importlib
import warnings
from dataclasses import dataclass, replace

import torch

from ._scatter_autograd import run_forward

_L1 = {torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.int16, torch.int32, torch.uint8}
_L2 = {torch.float64, torch.int64}
_CODES = {"sum": 0, "add": 0, "mul": 1, "mean": 2, "min": 3, "max": 4}
_WARNED = set()


@dataclass(frozen=True)
class _Plan:
    dim: int
    shape: tuple


def _broadcast(index, src, dim):
    if index.dim() == 1:
        for _ in range(dim):
            index = index.unsqueeze(0)
    while index.dim() < src.dim():
        index = index.unsqueeze(-1)
    return index.expand(src.shape)


def _validate(src, index, request):
    if not isinstance(src, torch.Tensor) or not isinstance(index, torch.Tensor):
        raise TypeError("src and index must be tensors")
    if src.device.type != "npu" or index.device != src.device:
        raise ValueError("src and index must be on the same NPU device")
    if src.dtype not in _L1 | _L2 or index.dtype != torch.int64:
        raise TypeError("unsupported src dtype or index dtype (expected INT64)")
    if src.dim() < 1 or src.dim() > 8:
        raise ValueError("src must have between 1 and 8 dimensions")
    if not isinstance(request.dim, int) or isinstance(request.dim, bool):
        raise TypeError("dim must be an integer")
    if not -src.dim() <= request.dim < src.dim():
        raise IndexError("dim is out of range")
    if request.reduce not in _CODES:
        raise ValueError("unsupported scatter reduction")
    return request.dim % src.dim()


def _output_shape(src, request, dim, maximum):
    if request.out is not None:
        out = request.out
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a tensor or None")
        if out.device != src.device or out.dtype != src.dtype:
            raise ValueError("out must match src device and dtype")
        if out.dim() != src.dim() or any(out.size(axis) != src.size(axis) for axis in range(src.dim()) if axis != dim):
            raise ValueError("out must match src shape outside dim")
        return tuple(out.shape)
    size = request.dim_size
    if size is not None and (not isinstance(size, int) or isinstance(size, bool)):
        raise TypeError("dim_size must be an integer or None")
    if size is not None and size < 0:
        raise ValueError("dim_size must be nonnegative")
    shape = list(src.shape)
    shape[dim] = maximum + 1 if size is None else size
    return tuple(shape)


def _prepare(src, index, request, native):
    dim = _validate(src, index, request)
    expanded = _broadcast(index, src, dim)
    minimum, maximum = 0, -1
    if expanded.numel():
        # Check the physical input every call. Independent storage aliases and
        # inference tensors make pointer/version caches insufficient here.
        minimum, maximum = native.scatter_index_stats(index)
    shape = _output_shape(src, request, dim, maximum)
    if minimum < 0 or maximum >= shape[dim]:
        raise IndexError("index is out of bounds for the output dimension")
    return _Plan(dim, shape)


def _native_sum(src, index, out, dim, native):
    dim %= src.dim()
    return run_forward(
        src,
        index,
        out,
        "sum",
        lambda: native.scatter_a2_forward(src, index, dim, out, None, 0),
    )[0]


def _can_fuse_mean(src, index, request, dim):
    features = 1
    feature_start = dim + 1
    for size in src.shape[feature_start:]:
        features *= size
    return (
        src.is_floating_point()
        and src.numel() > 0
        and request.out is None
        and index.dim() == 1
        and src.size(dim) <= 2**24
        and features >= 64
        and features * src.element_size() % 32 == 0
    )


def _mean(src, index, request, plan, native):
    output = request.out
    if output is None:
        output = torch.zeros(plan.shape, dtype=src.dtype, device=src.device)
    result = _native_sum(src, index, output, plan.dim, native)
    count_dim = plan.dim if index.dim() > plan.dim else index.dim() - 1
    ones = torch.ones(index.shape, dtype=src.dtype, device=src.device)
    count_shape = list(index.shape)
    count_shape[count_dim] = result.size(plan.dim)
    count = torch.zeros(count_shape, dtype=src.dtype, device=src.device)
    try:
        _native_sum(ones, index, count, count_dim, native)
    except IndexError as error:
        # PyTorch's scatter_add_ reports invalid count indices as RuntimeError.
        # Newer pybind translators preserve the native c10::IndexError type.
        raise RuntimeError(str(error)) from error
    if not src.is_floating_point():
        # Sum has already wrapped in src dtype. Widen only after that observable
        # rounding; CANN masked_fill does not implement INT16 or UINT8.
        count = count.to(torch.int32)
    count.masked_fill_(count < 1, 1)
    count = _broadcast(count, result, plan.dim)
    if src.is_floating_point():
        result.div_(count)
    else:
        result.div_(count, rounding_mode="floor")
    return result, None


def _storage_view(buffer, offset, length):
    if not length:
        return torch.empty(0, dtype=torch.uint8).untyped_storage()
    owner = (ctypes.c_ubyte * length).from_address(buffer.data_ptr() + offset)
    owner.backing_tensor = buffer
    return torch.frombuffer(owner, dtype=torch.uint8).untyped_storage()


def _copy_cpu_group(tensors):
    """Copy overlapping storage intervals together, retaining all view strides."""
    groups = []
    for tensor in tensors:
        storage = tensor.untyped_storage()
        start, end = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
        touching = [group for group in groups if start < group[1] and end > group[0]]
        members = [tensor]
        for group in touching:
            start, end = min(start, group[0]), max(end, group[1])
            members.extend(group[2])
            groups.remove(group)
        groups.append([start, end, members])
    result = {}
    for start, end, members in groups:
        buffer = torch.empty(end - start, dtype=torch.uint8)
        for tensor in members:
            storage = tensor.untyped_storage()
            raw = torch.empty(0, dtype=torch.uint8, device=tensor.device)
            raw.set_(storage, 0, (storage.nbytes(),), (1,))
            offset = storage.data_ptr() - start
            stop = offset + storage.nbytes()
            buffer[offset:stop].copy_(raw.cpu())
        storage_copies = {}
        for tensor in members:
            storage = tensor.untyped_storage()
            if storage not in storage_copies:
                offset = storage.data_ptr() - start
                storage_copies[storage] = _storage_view(buffer, offset, storage.nbytes())
            view = torch.empty(0, dtype=tensor.dtype)
            # A separate CPU storage wrapper for each original storage retains
            # PyTorch's overlap checks while the underlying bytes still alias.
            view.set_(storage_copies[storage], tensor.storage_offset(), tensor.shape, tensor.stride())
            result[id(tensor)] = view
    return [result[id(tensor)] for tensor in tensors]


def _cpu_backend():
    try:
        backend = importlib.import_module("torch_scatter")
    except ImportError as exc:
        raise ImportError(
            "A2 Scatter FLOAT64/INT64 requires the original torch_scatter==2.1.2 "
            "CPU extension built for PyTorch 2.7.1 or 2.12.0; L1 has no such dependency"
        ) from exc
    if backend.__name__ != "torch_scatter" or backend.__version__ != "2.1.2":
        raise RuntimeError("A2 L2 requires the original torch_scatter==2.1.2")
    if torch.__version__.split("+", maxsplit=1)[0] not in ("2.7.1", "2.12.0"):
        raise RuntimeError("A2 L2 CPU semantics are validated for PyTorch 2.7.1 or 2.12.0")
    return backend


def _l2(src, index, request):
    backend = _cpu_backend()
    if src.dtype not in _WARNED:
        warnings.warn(f"A2 Scatter {src.dtype} uses torch_scatter CPU with D2H/H2D", RuntimeWarning, stacklevel=4)
        _WARNED.add(src.dtype)
    inputs = [src, index] + ([] if request.out is None else [request.out])
    copied = _copy_cpu_group(inputs)
    cpu_out = None if request.out is None else copied[2]
    function = getattr(backend, "scatter_" + request.reduce)
    initial = None if cpu_out is None else cpu_out.clone()
    try:
        result = function(copied[0], copied[1], request.dim, cpu_out, request.dim_size)
    except (RuntimeError, IndexError):
        if cpu_out is not None and not torch.equal(
            cpu_out.contiguous().reshape(-1).view(torch.uint8),
            initial.contiguous().reshape(-1).view(torch.uint8),
        ):
            with torch.inference_mode():
                request.out.copy_(cpu_out)
        raise
    value, arg = result if isinstance(result, tuple) else (result, None)
    if request.out is None:
        output = value.to(device=src.device)
    else:
        with torch.inference_mode():
            request.out.copy_(value)
        output = request.out
    return output, None if arg is None else arg.to(device=src.device)


def _run_prepared_native(src, index, request, dim, native):
    if request.out is not None and not isinstance(request.out, torch.Tensor):
        raise TypeError("out must be a tensor or None")
    if request.out is None and request.dim_size is not None:
        if not isinstance(request.dim_size, int) or isinstance(request.dim_size, bool):
            raise TypeError("dim_size must be an integer or None")
        if request.dim_size < 0:
            raise ValueError("dim_size must be nonnegative")
    return run_forward(
        src,
        index,
        request.out,
        request.reduce,
        lambda: native.scatter_a2_forward(src, index, dim, request.out, request.dim_size, _CODES[request.reduce]),
    )


def _can_validate_native(src, index, request):
    """Exact built-ins avoid conversion/protocol errors before native checks."""
    # Subclasses must retain Python validation; isinstance would broaden this path.
    source_type, index_type = type(src), type(index)
    if source_type is not torch.Tensor or index_type is not torch.Tensor:
        return False
    if src.dtype not in _L1 or request.out is not None or src.requires_grad:
        return False
    dim_type = type(request.dim)
    if dim_type is not int or not -8 <= request.dim < 8:
        return False
    reduce_type = type(request.reduce)
    if reduce_type is not str or request.reduce not in ("sum", "add", "mul", "min", "max"):
        return False
    size = request.dim_size
    size_type = type(size)
    return size is None or (size_type is int and 0 <= size <= 2**63 - 1)


def forward(src, index, request, native):
    """Run L1 reduction on A2; only L2 imports the CPU extension."""
    if _can_validate_native(src, index, request):
        try:
            return _run_prepared_native(src, index, request, request.dim, native)
        except RuntimeError:
            # This binding reports native metadata errors as RuntimeError.
            # Recheck only after failure to preserve Python exception priority.
            _validate(src, index, request)
            raise
    dim = _validate(src, index, request)
    if src.dtype in _L1 and (request.reduce != "mean" or _can_fuse_mean(src, index, request, dim)):
        return _run_prepared_native(src, index, request, dim, native)
    plan = _prepare(src, index, request, native)
    if src.dtype in _L2:
        effective = request
        if request.reduce == "mean" and request.out is not None:
            if torch.is_inference(request.out) and not torch.is_inference_mode_enabled():
                effective = replace(request, reduce="sum")
        return run_forward(src, index, request.out, request.reduce, lambda: _l2(src, index, effective))
    return _mean(src, index, request, plan, native)

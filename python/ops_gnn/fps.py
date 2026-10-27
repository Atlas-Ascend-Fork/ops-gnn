# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Farthest point sampling for Ascend NPU tensors."""

from typing import List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor

from . import _pybind


@torch.jit._overload  # noqa: F811
def fps(src, batch, ratio, random_start, batch_size, ptr):
    # type: (Tensor, Optional[Tensor], Optional[float], bool, Optional[int], Optional[Tensor]) -> Tensor
    pass


@torch.jit._overload  # noqa: F811
def fps(src, batch, ratio, random_start, batch_size, ptr):
    # type: (Tensor, Optional[Tensor], Optional[Tensor], bool, Optional[int], Optional[Tensor]) -> Tensor
    pass


@torch.jit._overload  # noqa: F811
def fps(src, batch, ratio, random_start, batch_size, ptr):
    # type: (Tensor, Optional[Tensor], Optional[float], bool, Optional[int], Optional[List[int]]) -> Tensor
    pass


@torch.jit._overload  # noqa: F811
def fps(src, batch, ratio, random_start, batch_size, ptr):
    # type: (Tensor, Optional[Tensor], Optional[Tensor], bool, Optional[int], Optional[List[int]]) -> Tensor
    pass


def _validate_scalar_ratio(ratio: float) -> float:
    value = float(ratio)
    if not 0.0 < value <= 1.0:
        raise ValueError("fps: ratio must be in the interval (0, 1]")
    return value


def _ratio_tensor(src: Tensor, ratio: Union[Tensor, float]) -> Tensor:
    if isinstance(ratio, Tensor):
        # torch_cluster keeps Tensor ratios in their original dtype.
        result = ratio.to(device="cpu")
    else:
        # Its Python wrapper materializes scalar ratios in the source dtype.
        # Validate the caller's value before a valid tiny ratio may round to 0.
        value = _validate_scalar_ratio(float(ratio))
        result = torch.scalar_tensor(value, dtype=src.dtype, device="cpu")
    if result.dim() > 1:
        raise ValueError("fps: ratio tensor must be scalar or one-dimensional")
    if result.numel() == 0:
        raise ValueError("fps: ratio tensor must not be empty")
    if isinstance(ratio, Tensor) and not bool(((result > 0) & (result <= 1)).all().item()):
        raise ValueError("fps: every ratio must be in the interval (0, 1]")
    return result


def _sample_counts(degree: Tensor, ratio: Tensor) -> Tensor:
    # Match torch_cluster 1.6.3 CPU: deg.float() * stored ratio, then ceil.
    # Keep degree in float32 even when src/ratio are float16. A float64 Tensor
    # ratio retains its dtype through PyTorch type promotion, as upstream does.
    degree_cpu = degree.to(device="cpu", dtype=torch.long)
    product = degree_cpu.to(torch.float32) * ratio.to(device="cpu")
    counts = torch.ceil(product).to(torch.long)
    counts = torch.where(degree_cpu > 0, counts.clamp_min(1), torch.zeros_like(counts))
    return counts.to(device=degree.device)


def _single_sample_count(num_points: int, ratio: float, src_dtype: torch.dtype) -> int:
    # The low-overhead pybind entry still needs the same two rounding stages
    # as the batch entry: scalar -> src dtype, then float32 multiplication.
    value = _validate_scalar_ratio(ratio)
    stored_ratio = torch.scalar_tensor(value, dtype=src_dtype, device="cpu")
    degree = torch.scalar_tensor(num_points, dtype=torch.float32, device="cpu")
    return max(1, int(torch.ceil(degree * stored_ratio).item()))


def _random_start(degree: Tensor) -> Tensor:
    # torch_cluster 1.6.3 CPU uses libc rand() % degree.  The native helper
    # intentionally shares that process-global RNG so an explicit libc srand
    # reset reproduces the same starts in eager and TorchScript execution.
    return torch.ops.ops_gnn.fps_random_start(degree)


def _batch_ptr(src: Tensor, batch: Optional[Tensor],
               batch_size: Optional[int]) -> Tensor:
    if batch is None:
        return torch.tensor([0, src.size(0)], dtype=torch.long,
                            device=src.device)
    if batch.dim() != 1 or batch.numel() != src.size(0):
        raise ValueError("fps: batch must be one-dimensional with src.size(0) entries")
    if batch.dtype != torch.long:
        raise TypeError("fps: batch must have dtype int64")
    batch = batch.to(device=src.device).contiguous()
    if batch_size is None:
        batch_size = int(batch.max().item()) + 1
    if batch_size <= 0:
        raise ValueError("fps: batch_size must be positive")
    degree = torch.zeros(batch_size, dtype=torch.long, device=src.device)
    degree.scatter_add_(0, batch, torch.ones_like(batch))
    ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=src.device)
    torch.cumsum(degree, 0, out=ptr[1:])
    return ptr


class _EagerRequest(NamedTuple):
    batch: Optional[Tensor]
    ratio: Optional[Union[Tensor, float]]
    random_start: bool
    batch_size: Optional[int]
    ptr: Optional[Union[Tensor, List[int]]]


def _eager_ptr(src: Tensor, request: _EagerRequest) -> Tensor:
    if request.ptr is not None:
        if isinstance(request.ptr, list):
            ptr_tensor = torch.tensor(request.ptr, dtype=torch.long, device=src.device)
        else:
            ptr_tensor = request.ptr.to(device=src.device, dtype=torch.long)
        ptr_tensor = ptr_tensor.contiguous()
    else:
        ptr_tensor = _batch_ptr(src, request.batch, request.batch_size)
    return ptr_tensor


def _sample_metadata(src: Tensor, ptr_tensor: Tensor, ratio_tensor: Tensor,
                     random_start: bool) -> Tuple[Tensor, Tensor]:
    if ptr_tensor.dim() != 1 or ptr_tensor.numel() < 2:
        raise ValueError("fps: ptr must be one-dimensional with at least two entries")
    num_batches = ptr_tensor.numel() - 1
    if ratio_tensor.numel() != 1 and ratio_tensor.numel() != num_batches:
        raise ValueError("fps: ratio tensor must be scalar or have one value per batch")

    degree = ptr_tensor[1:] - ptr_tensor[:-1]
    counts = _sample_counts(degree, ratio_tensor)
    out_ptr = torch.zeros(num_batches + 1, dtype=torch.long, device=src.device)
    torch.cumsum(counts, 0, out=out_ptr[1:])
    if random_start:
        start = _random_start(degree)
    else:
        start = torch.zeros(num_batches, dtype=torch.long, device=src.device)
    return out_ptr, start


@torch.jit.unused
def _fps_eager(src: Tensor, request: _EagerRequest) -> Tensor:
    if src.dim() < 1:
        raise ValueError("fps: src must have at least one dimension")
    if src.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=src.device)
    if src.dtype not in (torch.float16, torch.float32):
        raise TypeError("fps: src dtype must be float16 or float32")
    if src.device.type != "npu":
        raise ValueError("fps: src must be an NPU tensor")
    if src.size(0) > 2147483647:
        raise ValueError("fps: src has too many points")

    src = src.reshape(src.size(0), -1).contiguous()
    normalized_ratio: Union[Tensor, float] = 0.5 if request.ratio is None else request.ratio

    # Official performance cases are a single batch, scalar ratio and a fixed
    # first point. Keep this path free from tiny H2D metadata copies and host
    # synchronization; the native entry receives only scalar layout values.
    if request.batch is None and request.ptr is None and not request.random_start:
        if not isinstance(normalized_ratio, Tensor):
            out_size = _single_sample_count(
                src.size(0), float(normalized_ratio), src.dtype
            )
            return _pybind.fps_single(src, out_size, 0)

    ratio_tensor = _ratio_tensor(src, normalized_ratio)
    ptr_tensor = _eager_ptr(src, request)
    out_ptr, start = _sample_metadata(src, ptr_tensor, ratio_tensor, request.random_start)
    return _pybind.fps_batch(src, ptr_tensor, out_ptr, start)


def _script_ratio(src: Tensor, ratio: Optional[Union[Tensor, float]]) -> Tensor:
    if ratio is None:
        r = torch.scalar_tensor(0.5, dtype=src.dtype, device="cpu")
    elif isinstance(ratio, float):
        _validate_scalar_ratio(ratio)
        r = torch.scalar_tensor(ratio, dtype=src.dtype, device="cpu")
    else:
        r = ratio.to(device="cpu")
    if r.dim() > 1 or r.numel() == 0:
        raise ValueError(
            "fps: ratio tensor must be scalar or a non-empty one-dimensional tensor"
        )
    if isinstance(ratio, Tensor) and not bool(((r > 0) & (r <= 1)).all().item()):
        raise ValueError("fps: every ratio must be in the interval (0, 1]")
    return r


def _script_ptr(src: Tensor, batch: Optional[Tensor], batch_size: Optional[int],
                ptr: Optional[Union[Tensor, List[int]]]) -> Tensor:
    if ptr is not None:
        if isinstance(ptr, list):
            ptr_tensor = torch.tensor(
                ptr, dtype=torch.long, device=src.device
            )
        else:
            ptr_tensor = ptr.to(device=src.device, dtype=torch.long)
        ptr_tensor = ptr_tensor.contiguous()
    elif batch is not None:
        if batch.dim() != 1 or batch.numel() != src.size(0):
            raise ValueError(
                "fps: batch must be one-dimensional with src.size(0) entries"
            )
        if batch_size is None:
            size = int(batch.max().item()) + 1
        else:
            size = batch_size
        degree_from_batch = torch.zeros(
            size, dtype=torch.long, device=src.device
        )
        degree_from_batch.scatter_add_(
            0, batch, torch.ones_like(batch)
        )
        ptr_tensor = torch.zeros(
            size + 1, dtype=torch.long, device=src.device
        )
        torch.cumsum(degree_from_batch, 0, out=ptr_tensor[1:])
    else:
        ptr_tensor = torch.tensor(
            [0, src.size(0)], dtype=torch.long, device=src.device
        )
    return ptr_tensor


def fps(
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
) -> torch.Tensor:
    r"""Samples points with iterative farthest point sampling.

    The public signature follows :func:`torch_cluster.fps`. Sample counts
    follow torch_cluster CPU's stored-ratio dtype and float32 degree product
    before ceil. Distance ties use the first (lowest) index,
    matching the CPU reference's argmax.  Random starts use the same libc
    ``rand() % degree`` stream as torch_cluster CPU; reproducible cross-device
    comparison therefore requires explicitly resetting libc ``srand``.
    """
    if torch.jit.is_scripting():
        if src.dim() < 1:
            raise ValueError("fps: src must have at least one dimension")
        if src.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=src.device)
        if src.dtype != torch.float16 and src.dtype != torch.float32:
            raise TypeError("fps: src dtype must be float16 or float32")
        if src.size(0) > 2147483647:
            raise ValueError("fps: src has too many points")
        src = src.reshape(src.size(0), -1).contiguous()

        r = _script_ratio(src, ratio)
        ptr_tensor = _script_ptr(src, batch, batch_size, ptr)
        out_ptr, start = _sample_metadata(src, ptr_tensor, r, random_start)
        return torch.ops.ops_gnn.fps_batch(
            src, ptr_tensor, out_ptr, start
        )
    else:
        return _fps_eager(
            src, _EagerRequest(batch, ratio, random_start, batch_size, ptr)
        )

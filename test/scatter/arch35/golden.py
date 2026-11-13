# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
import math

import torch


def reduce_selected(selected, reduce):
    values = torch.stack(selected)
    if reduce in ("sum", "add"):
        return values.sum()
    if reduce == "mul":
        return values.prod()
    if reduce == "mean":
        return values.mean()
    if reduce == "min":
        return values.min()
    return values.max()


@dataclass(frozen=True)
class CpuScatterRequest:
    dim: int = -1
    out: object = None
    dim_size: int = None
    reduce: str = "sum"


@dataclass
class CpuScatterViews:
    src: torch.Tensor
    index: torch.Tensor
    out: torch.Tensor
    count: object
    arg: object
    src_dim_size: int
    flat_after: int


def _new_cpu_output(src, request, out_size):
    if request.out is not None:
        return request.out.cpu().clone()
    if request.reduce == "mul":
        return src.new_ones(out_size)
    if request.reduce == "min":
        initial = (torch.finfo(src.dtype).max if src.dtype.is_floating_point
                   else torch.iinfo(src.dtype).max)
        return src.new_full(out_size, initial)
    if request.reduce == "max":
        initial = (torch.finfo(src.dtype).min if src.dtype.is_floating_point
                   else torch.iinfo(src.dtype).min)
        return src.new_full(out_size, initial)
    return src.new_zeros(out_size)


def _cpu_views(src, index, out, dim, request):
    flat_before = math.prod(src.shape[:dim])
    flat_after = math.prod(src.shape[dim + 1:])
    src_dim_size = src.size(dim)
    src_view = src.reshape(flat_before, src_dim_size, flat_after)
    index_view = index.reshape(flat_before, src_dim_size, flat_after)
    out_view = out.reshape(flat_before, request.dim_size, flat_after)
    count = torch.zeros(out.shape, dtype=torch.int64) if request.reduce == "mean" else None
    arg = out.new_full(out.shape, src_dim_size, dtype=torch.long)
    if request.reduce not in ("min", "max"):
        arg = None
    return CpuScatterViews(
        src_view,
        index_view,
        out_view,
        count.reshape(out_view.shape) if count is not None else None,
        arg.reshape(out_view.shape) if arg is not None else None,
        src_dim_size,
        flat_after,
    ), arg


def _apply_cpu_value(views, offset, reduce):
    row_size = views.src_dim_size * views.flat_after
    before = offset // row_size
    position = (offset % row_size) // views.flat_after
    inner = offset % views.flat_after
    target = int(views.index[before, position, inner].item())
    value = views.src[before, position, inner].item()
    key = (before, target, inner)
    if reduce in ("sum", "add", "mean"):
        views.out[key] += value
        if views.count is not None:
            views.count[key] += 1
    elif reduce == "mul":
        views.out[key] *= value
    elif reduce == "min" and value <= views.out[key]:
        views.out[key] = value
        views.arg[key] = position
    elif reduce == "max" and value >= views.out[key]:
        views.out[key] = value
        views.arg[key] = position


def _finalize_cpu_output(src, out, arg, views, request):
    if views.count is not None:
        nonzero = views.count > 0
        if src.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            denominator = views.count[nonzero]
            if src.dtype in (torch.uint8, torch.int8, torch.int16):
                denominator = denominator.to(src.dtype).clamp_min_(1)
            views.out[nonzero] = torch.div(
                views.out[nonzero], denominator, rounding_mode="floor"
            ).to(out.dtype)
        else:
            views.out[nonzero] /= views.count[nonzero]
    if arg is None:
        return out
    if request.out is None:
        empty = arg == src.size(request.dim)
        out[empty] = 0
    return out, arg


def scatter_cpu(src, index, **kwargs):
    request = CpuScatterRequest(**kwargs)
    sc = src.cpu()
    ic = index.cpu().long()
    dim = request.dim % sc.dim()
    dim_size = request.dim_size
    if dim_size is None:
        dim_size = int(ic.max().item()) + 1 if ic.numel() > 0 else 0
    request = CpuScatterRequest(dim, request.out, dim_size, request.reduce)
    out_size = list(sc.shape)
    out_size[dim] = dim_size
    oc = _new_cpu_output(sc, request, out_size)
    sbc = sc.expand_as(ic) if sc.size(dim) == 1 else sc
    views, arg = _cpu_views(sbc, ic, oc, dim, request)
    for offset in range(sbc.numel()):
        _apply_cpu_value(views, offset, request.reduce)
    return _finalize_cpu_output(sc, oc, arg, views, request)

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyTorch-facing API for the Ascend NPU radius / radius_graph operators."""

from typing import Optional, Tuple

import torch

from . import _pybind
from .typing import Tensor

_RADIUS_ARG_NAMES = ("x", "y", "r", "batch_x", "batch_y",
                     "max_num_neighbors", "num_workers",
                     "batch_size", "ignore_same_index")


def _parse_radius_arguments(args: Tuple, kwargs: dict) -> tuple:
    """Bind positional/keyword args to the torch_cluster.radius signature."""
    if len(args) > len(_RADIUS_ARG_NAMES):
        raise TypeError("radius() takes at most 9 positional arguments")
    values = dict(zip(_RADIUS_ARG_NAMES, args))
    for name, value in kwargs.items():
        if name not in _RADIUS_ARG_NAMES:
            raise TypeError(f"radius() got an unexpected keyword argument '{name}'")
        if name in values:
            raise TypeError(f"radius() got multiple values for argument '{name}'")
        values[name] = value
    missing = [name for name in _RADIUS_ARG_NAMES[:3] if name not in values]
    if missing:
        raise TypeError(f"radius() missing required arguments: {', '.join(missing)}")
    values.setdefault("batch_x", None)
    values.setdefault("batch_y", None)
    values.setdefault("max_num_neighbors", 32)
    values.setdefault("num_workers", 1)
    values.setdefault("batch_size", None)
    values.setdefault("ignore_same_index", False)
    return tuple(values[name] for name in _RADIUS_ARG_NAMES)


def _resolve_batch_size(x, y, batch_x, batch_y, batch_size):
    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            if x.size(0) != batch_x.numel():
                raise ValueError(
                    "radius: batch_x length must equal x.size(0)")
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            if y.size(0) != batch_y.numel():
                raise ValueError(
                    "radius: batch_y length must equal y.size(0)")
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    if batch_size <= 0:
        raise ValueError("radius: batch_size must be positive")
    return batch_size


def radius(*args, **kwargs) -> Tensor:
    r"""Finds all points in :obj:`x` within distance :obj:`r` of each point in
    :obj:`y` on NPU.

    Public signature follows :func:`torch_cluster.radius`:

    Args:
        x: source points, shape ``[n, num_features]``.
        y: query points, shape ``[m, num_features]``.
        r: search radius.
        batch_x: optional per-point batch ids for :obj:`x` ``[n]``.
        batch_y: optional per-point batch ids for :obj:`y` ``[m]``.
        max_num_neighbors: maximum number of neighbors to return per query;
            must be positive (torch_cluster's ``-1`` all-neighbor mode is not
            supported).
        num_workers: unused, kept for interface compatibility.
        batch_size: number of batches (inferred from batch ids when unset).
        ignore_same_index: exclude self-connections ``i == j``.
    """
    (x, y, r, batch_x, batch_y, max_num_neighbors, num_workers, batch_size,
     ignore_same_index) = _parse_radius_arguments(args, kwargs)

    if max_num_neighbors <= 0:
        raise ValueError(
            "radius(): max_num_neighbors must be positive; torch_cluster's -1 "
            "(unlimited) mode is not supported")

    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.reshape(-1, 1) if x.dim() == 1 else x
    y = y.reshape(-1, 1) if y.dim() == 1 else y
    x = x.contiguous()
    y = y.contiguous()

    batch_size = _resolve_batch_size(x, y, batch_x, batch_y, batch_size)

    ptr_x: Optional[Tensor] = None
    ptr_y: Optional[Tensor] = None
    if batch_size > 1:
        if batch_x is None or batch_y is None:
            raise ValueError(
                "radius: batch_x and batch_y are required when batch_size > 1")
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    # Pass the torch framework's current NPU stream so the kernel is enqueued
    # on the same stream as the prior H2D / zero-fill work (correct ordering
    # without a device-wide barrier) and the C++ side never Create/Destroy a
    # stream per call.
    stream_handle = int(torch.npu.current_stream().npu_stream)

    return _pybind.radius(x, y, ptr_x, ptr_y, r, max_num_neighbors,
                          num_workers, ignore_same_index, stream_handle)


_RADIUS_GRAPH_ARG_NAMES = ("x", "r", "batch", "loop", "max_num_neighbors",
                           "flow", "num_workers", "batch_size")


def _parse_radius_graph_arguments(args: Tuple, kwargs: dict) -> tuple:
    """Bind positional/keyword args to the torch_cluster.radius_graph signature."""
    if len(args) > len(_RADIUS_GRAPH_ARG_NAMES):
        raise TypeError("radius_graph() takes at most 8 positional arguments")
    values = dict(zip(_RADIUS_GRAPH_ARG_NAMES, args))
    for name, value in kwargs.items():
        if name not in _RADIUS_GRAPH_ARG_NAMES:
            raise TypeError(
                f"radius_graph() got an unexpected keyword argument '{name}'")
        if name in values:
            raise TypeError(
                f"radius_graph() got multiple values for argument '{name}'")
        values[name] = value
    missing = [name for name in _RADIUS_GRAPH_ARG_NAMES[:2] if name not in values]
    if missing:
        raise TypeError(
            f"radius_graph() missing required arguments: {', '.join(missing)}")
    values.setdefault("batch", None)
    values.setdefault("loop", False)
    values.setdefault("max_num_neighbors", 32)
    values.setdefault("flow", "source_to_target")
    values.setdefault("num_workers", 1)
    values.setdefault("batch_size", None)
    return tuple(values[name] for name in _RADIUS_GRAPH_ARG_NAMES)


def radius_graph(*args, **kwargs) -> Tensor:
    r"""Computes graph edges to all points within distance :obj:`r` on NPU.

    Public signature follows :func:`torch_cluster.radius_graph`:

    Args:
        x: node positions, shape ``[n, num_features]``.
        r: search radius.
        batch: optional per-node batch ids ``[n]``.
        loop: whether to include self-loops.
        max_num_neighbors: maximum number of neighbors to return per node.
        flow: ``'source_to_target'`` or ``'target_to_source'``.
        num_workers: unused, kept for interface compatibility.
        batch_size: number of batches (inferred from batch ids when unset).
    """
    (x, r, batch, loop, max_num_neighbors, flow, num_workers,
     batch_size) = _parse_radius_graph_arguments(args, kwargs)
    if flow not in ('source_to_target', 'target_to_source'):
        raise ValueError(
            "radius_graph: flow must be 'source_to_target' or "
            "'target_to_source'")
    edge_index = radius(x, x, r, batch, batch, max_num_neighbors,
                        num_workers, batch_size,
                        ignore_same_index=not loop)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]
    return torch.stack([row, col], dim=0)

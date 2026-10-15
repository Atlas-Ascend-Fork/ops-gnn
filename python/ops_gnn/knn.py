# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# API preparation and graph composition follow pytorch_cluster 1.6.3.
# Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
# These portions retain the MIT license in the repository-root LICENSE.

"""PyTorch-compatible Ascend NPU KNN APIs."""

from inspect import Signature, signature as inspect_signature
from typing import NamedTuple, Optional

import torch

from . import _pybind
from .typing import Tensor


class _KnnArguments(NamedTuple):
    x: Tensor
    y: Tensor
    k: int
    batch_x: Optional[Tensor] = None
    batch_y: Optional[Tensor] = None
    cosine: bool = False
    num_workers: int = 1
    batch_size: Optional[int] = None


class _GraphArguments(NamedTuple):
    x: Tensor
    k: int
    batch: Optional[Tensor] = None
    loop: bool = False
    flow: str = 'source_to_target'
    cosine: bool = False
    num_workers: int = 1
    batch_size: Optional[int] = None


def _public_signature(arguments_type) -> Signature:
    """Describe the compatibility API from its typed argument record."""
    return inspect_signature(arguments_type).replace(return_annotation=Tensor)


_KNN_SIGNATURE = _public_signature(_KnnArguments)
_GRAPH_SIGNATURE = _public_signature(_GraphArguments)


def _bind_arguments(arguments_type, signature, positional, keyword):
    """Use Python's binder to reject missing, duplicate or unknown arguments."""
    bound = signature.bind(*positional, **keyword)
    bound.apply_defaults()
    return arguments_type(**bound.arguments)


def knn(*args, **kwargs) -> Tensor:
    """Find the ``k`` nearest rows in ``x`` for every row in ``y``.

    Accepts the torch_cluster API: x, y, k, batch_x=None, batch_y=None,
    cosine=False, num_workers=1, batch_size=None, either positionally or
    by keyword. Parameters are bound before entering the NPU implementation.
    """
    query = _bind_arguments(_KnnArguments, _KNN_SIGNATURE, args, kwargs)
    return _knn_impl(query)


def _knn_impl(query: _KnnArguments) -> Tensor:
    x, y, k, batch_x, batch_y, cosine, num_workers, batch_size = query
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            if x.size(0) != batch_x.numel():
                raise ValueError("knn: batch_x length must equal x.size(0)")
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            if y.size(0) != batch_y.numel():
                raise ValueError("knn: batch_y length must equal y.size(0)")
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    if batch_size <= 0:
        raise ValueError("knn: batch_size must be positive")

    ptr_x: Optional[Tensor] = None
    ptr_y: Optional[Tensor] = None
    if batch_size > 1:
        if batch_x is None or batch_y is None:
            raise ValueError(
                "knn: batch_x and batch_y are required when batch_size > 1")
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    stream_handle = int(torch.npu.current_stream().npu_stream)
    return _pybind.knn(x, y, ptr_x, ptr_y, k, cosine, num_workers,
                       stream_handle)


def knn_graph(*args, **kwargs) -> Tensor:
    """Build a directed graph connecting each row to its nearest neighbors.

    Accepts x, k, batch=None, loop=False, flow='source_to_target',
    cosine=False, num_workers=1, batch_size=None, either positionally or
    by keyword, matching torch_cluster.knn_graph.
    """
    graph = _bind_arguments(_GraphArguments, _GRAPH_SIGNATURE, args, kwargs)
    return _knn_graph_impl(graph)


def _knn_graph_impl(graph: _GraphArguments) -> Tensor:
    if graph.flow not in ('source_to_target', 'target_to_source'):
        raise ValueError(
            "knn_graph: flow must be 'source_to_target' or "
            "'target_to_source'")
    query = _KnnArguments(
        x=graph.x, y=graph.x, k=graph.k + int(not graph.loop),
        batch_x=graph.batch, batch_y=graph.batch, cosine=graph.cosine,
        num_workers=graph.num_workers, batch_size=graph.batch_size)
    edges = _knn_impl(query)
    if not graph.loop:
        edges = edges[:, edges[0].ne(edges[1])]
    return edges.flip((0,)) if graph.flow == 'source_to_target' else edges


# Keep inspect/help metadata explicit for the variadic compatibility entrypoints,
# following the radius API's binding pattern without changing accepted calls.
knn.__signature__ = _KNN_SIGNATURE
knn.__annotations__ = {**_KnnArguments.__annotations__, 'return': Tensor}
knn_graph.__signature__ = _GRAPH_SIGNATURE
knn_graph.__annotations__ = {**_GraphArguments.__annotations__, 'return': Tensor}

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Tie-order compatibility adapts nanoflann's partition/traversal rules.
# Copyright 2011-16 Jose Luis Blanco (joseluisblancoc@gmail.com).
# See the repository-root LICENSE for the complete BSD license.

"""Independent CPU KNN reference; no extension or implementation under test.

Distances scan every candidate in bounded query blocks. The tree determines
only the documented L2 tie order; it never prunes distance calculations.
"""
__all__ = ['knn', 'knn_graph']

from inspect import signature
from typing import NamedTuple, Optional

import torch
from torch import Tensor

QUERY_BLOCK_SIZE = 32


# Declare the reference's API schema locally, without importing production records.
_KnnArguments = NamedTuple('_KnnArguments', [
    ('x', Tensor), ('y', Tensor), ('k', int),
    ('batch_x', Optional[Tensor]), ('batch_y', Optional[Tensor]),
    ('cosine', bool), ('num_workers', int), ('batch_size', Optional[int]),
])
_KnnArguments.__new__.__defaults__ = (None, None, False, 1, None)

_GraphArguments = NamedTuple('_GraphArguments', [
    ('x', Tensor), ('k', int), ('batch', Optional[Tensor]),
    ('loop', bool), ('flow', str), ('cosine', bool),
    ('num_workers', int), ('batch_size', Optional[int]),
])
_GraphArguments.__new__.__defaults__ = (None, False, 'source_to_target', False, 1, None)


class _TieOrder:
    """Describe each candidate's query-dependent path and leaf position."""

    def __init__(self, points):
        self.points = points
        self.paths = {}
        features = 1 if points.dtype == torch.float16 else points.shape[1]
        coordinates = points[:, :features]
        indices = torch.arange(len(points))
        self._build(indices, coordinates.amin(0), coordinates.amax(0), ())

    def key(self, index, query):
        path, position = self.paths[index]
        order = []
        for (axis, low, high), side in path:
            value = query[axis]
            # Round both subtractions and the addition in the input dtype.
            left_first = bool((value - low) + (value - high) < 0)
            order.append(int(side != (0 if left_first else 1)))
        return tuple(order) + (position,)

    def _partition(self, indices, axis, cut, inclusive):
        values = self.points[indices, axis]
        belongs_left = values <= cut if inclusive else values < cut
        boundary = int(belongs_left.sum())
        positions = torch.arange(len(indices))
        wrong_left = positions[(positions < boundary) & ~belongs_left]
        wrong_right = positions[(positions >= boundary) & belongs_left].flip(0)
        result = indices.clone()
        result[wrong_left], result[wrong_right] = indices[wrong_right], indices[wrong_left]
        return result, boundary

    def _build(self, indices, low, high, path):
        if len(indices) <= 10:
            for position, index in enumerate(indices.tolist()):
                self.paths[index] = (path, position)
            return

        axis = 0
        if self.points.dtype == torch.float32:
            spans = high - low
            eligible = (spans > (1 - 1e-5) * spans.max()).nonzero().flatten()
            if eligible.numel():
                selected = self.points[indices][:, eligible]
                spreads = selected.amax(0) - selected.amin(0)
                axis = int(eligible[int(spreads.argmax())])
        values = self.points[indices, axis]
        cut = ((low[axis] + high[axis]) / 2).clamp(values.min(), values.max())
        indices, below = self._partition(indices, axis, cut, False)
        indices, through = self._partition(indices, axis, cut, True)
        middle = max(below, min(len(indices) // 2, through))
        left, right = indices[:middle], indices[middle:]
        left_high = high.clone()
        left_high[axis] = cut
        right_low = low.clone()
        right_low[axis] = cut
        split = (axis, float(self.points[left, axis].max()),
                 float(self.points[right, axis].min()))
        self._build(left, low, left_high, path + ((split, 0),))
        self._build(right, right_low, high, path + ((split, 1),))


def _l2_distances(x, y):
    distances = torch.zeros((len(y), len(x)), dtype=x.dtype)
    grouped = x.shape[1] - x.shape[1] % 4
    for start in range(0, grouped, 4):
        delta = y[:, None, start:start + 4] - x[None, :, start:start + 4]
        square = delta * delta
        group = ((square[:, :, 0] + square[:, :, 1]) + square[:, :, 2]) + square[:, :, 3]
        distances = distances + group
    for feature in range(grouped, x.shape[1]):
        delta = y[:, None, feature] - x[None, :, feature]
        distances = distances + delta * delta
    return distances


def _select_l2(distances, queries, k, order):
    threshold = distances.topk(k, dim=1, largest=False).values[:, -1]
    result = []
    for distance, query, cutoff in zip(distances, queries, threshold):
        candidates = (distance <= cutoff).nonzero().flatten().tolist()
        selected = sorted(candidates, key=lambda i: (float(distance[i]), order.key(i, query)))[:k]
        result.append(torch.tensor(selected, dtype=torch.long))
    return torch.stack(result)


def _batch_labels(x, y, batch_x, batch_y, batch_size):
    labels = []
    for points, batch in ((x, batch_x), (y, batch_y)):
        if batch is None:
            labels.append(torch.zeros(len(points), dtype=torch.long))
        else:
            if batch.device.type != 'cpu' or batch.dtype != torch.long or batch.ndim != 1:
                raise ValueError('golden: batch vectors must be CPU int64 vectors')
            if len(batch) != len(points) or bool((batch[1:] < batch[:-1]).any()):
                raise ValueError('golden: batch vectors must have matching lengths and be sorted')
            if bool((batch < 0).any()):
                raise ValueError('golden: batch indices must be nonnegative')
            labels.append(batch)
    inferred = max(int(label.max()) for label in labels) + 1
    if batch_size is None:
        batch_size = inferred
    if batch_size <= 0 or batch_size < inferred:
        raise ValueError('golden: batch_size must cover every batch index')
    if batch_size > 1 and (batch_x is None or batch_y is None):
        raise ValueError('golden: both batch vectors are required for multiple batches')
    return labels


def _validated_points(x, y, k):
    if x.device.type != 'cpu' or y.device.type != 'cpu':
        raise ValueError('golden: inputs must be CPU tensors')
    if x.numel() == 0 or y.numel() == 0:
        return x, y
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError('golden: inputs must have matching feature dimensions')
    if x.dtype not in (torch.float16, torch.float32) or y.dtype != x.dtype:
        raise ValueError('golden: inputs must share FP16/FP32 dtype')
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError('golden: nonfinite inputs are outside the tested contract')
    if k < 1:
        raise ValueError('golden: k must be positive')
    return x, y


def _knn_impl(query):
    x, y = _validated_points(query.x, query.y, query.k)
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long)
    k, cosine = query.k, query.cosine
    bx, by = _batch_labels(x, y, query.batch_x, query.batch_y, query.batch_size)
    edges = []
    for batch in by.unique().tolist():
        source_ids = (bx == batch).nonzero().flatten()
        query_ids = (by == batch).nonzero().flatten()
        if k > len(source_ids):
            raise ValueError('golden: k exceeds the candidate count in a query batch')
        points = x[source_ids]
        order = None if cosine else _TieOrder(points)
        for first in range(0, len(query_ids), QUERY_BLOCK_SIZE):
            rows = query_ids[first:first + QUERY_BLOCK_SIZE]
            queries = y[rows]
            if cosine:
                a = torch.nn.functional.normalize(points.double(), dim=1)
                b = torch.nn.functional.normalize(queries.double(), dim=1)
                distances = 1 - b @ a.T
                columns = distances.argsort(dim=1, stable=True)[:, :k]
            else:
                columns = _select_l2(_l2_distances(points, queries), queries, k, order)
            edges.append(torch.stack((rows.repeat_interleave(k), source_ids[columns].flatten())))
    return torch.cat(edges, dim=1)


def knn(*args, **kwargs) -> Tensor:
    """Return query/source edges using exhaustive CPU distance evaluation.

    Accepts x, y, k, batch_x=None, batch_y=None, cosine=False,
    num_workers=1 and batch_size=None positionally or by keyword.
    """
    return _knn_impl(_KnnArguments(*args, **kwargs))


def knn_graph(*args, **kwargs) -> Tensor:
    """Compose the independent CPU reference with graph filtering and flow."""
    graph = _GraphArguments(*args, **kwargs)
    if graph.flow not in ('source_to_target', 'target_to_source'):
        raise ValueError('golden: invalid graph flow')
    query = _KnnArguments(graph.x, graph.x, graph.k + int(not graph.loop),
                          graph.batch, graph.batch, graph.cosine,
                          graph.num_workers, graph.batch_size)
    edges = _knn_impl(query)
    if not graph.loop:
        edges = edges[:, edges[0] != edges[1]]
    return edges.flip(0).contiguous() if graph.flow == 'source_to_target' else edges.contiguous()


# Retain inspect/help metadata while grouping the compatibility parameters.
knn.__signature__ = signature(_KnnArguments).replace(return_annotation=Tensor)
knn.__annotations__ = {**_KnnArguments.__annotations__, 'return': Tensor}
knn_graph.__signature__ = signature(_GraphArguments).replace(return_annotation=Tensor)
knn_graph.__annotations__ = {**_GraphArguments.__annotations__, 'return': Tensor}

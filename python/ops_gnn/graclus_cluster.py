# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from . import _pybind
from .typing import Tensor, OptTensor


_CSR_CACHE = {}
_MAX_CSR_CACHE_SIZE = 128
_FLOAT32_EXACT_INT_MAX = 16777216


@dataclass(frozen=True)
class _CsrCacheKey:
    row_id: int
    col_id: int
    weight_id: int
    row_ptr: int
    col_ptr: int
    weight_ptr: int
    row_version: int
    col_version: int
    weight_version: int
    edge_count: int
    node_count: int
    weight_dtype: str
    device: str


@dataclass
class _CpuGraph:
    rowptr_list: list
    col_list: list
    weight_list: Optional[list]
    cluster: list
    num_nodes: int


@dataclass
class _RunContext:
    row: Tensor
    col: Tensor
    weight: OptTensor
    node_count: int
    cache_key: Optional[_CsrCacheKey]


def _cache_put(key, value) -> None:
    if len(_CSR_CACHE) >= _MAX_CSR_CACHE_SIZE:
        _CSR_CACHE.pop(next(iter(_CSR_CACHE)))
    _CSR_CACHE[key] = value


def _get_node_perm(context: _RunContext) -> Tensor:
    node_perm = torch.randperm(context.node_count, device=torch.device("cpu"))
    return node_perm.to(context.row.device).contiguous()


def _tensor_version(tensor: Tensor) -> Optional[int]:
    try:
        return int(getattr(tensor, "_version", 0))
    except RuntimeError:
        # Inference tensors do not expose a version counter. Since their
        # mutations cannot be tracked, they must not participate in caching.
        return None


def _validate_inputs(row: Tensor, col: Tensor, weight: OptTensor, num_nodes: Optional[int]) -> None:
    if row.dim() != 1 or col.dim() != 1:
        raise ValueError("row and col must be 1D tensors")
    if row.numel() != col.numel():
        raise ValueError("row and col must have the same number of elements")
    if row.dtype != torch.long or col.dtype != torch.long:
        raise TypeError("row and col must be torch.long tensors")
    if row.device != col.device:
        raise ValueError("row and col must be on the same device")
    _validate_weight(row, weight)
    if num_nodes is not None and num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")


def _validate_weight(row: Tensor, weight: OptTensor) -> None:
    if weight is None:
        return
    if weight.dim() != 1 or weight.numel() != row.numel():
        raise ValueError("weight must be a 1D tensor with the same length as row and col")
    if weight.device != row.device:
        raise ValueError("weight must be on the same device as row and col")
    if not weight.is_floating_point():
        raise TypeError("weight must be a floating point tensor")


def _maybe_num_nodes(row: Tensor, col: Tensor, num_nodes: Optional[int]) -> int:
    if num_nodes is not None:
        return int(num_nodes)
    if row.numel() == 0 and col.numel() == 0:
        return 0
    return int(torch.maximum(row.max(), col.max()).item()) + 1


def _check_index_bounds(row: Tensor, col: Tensor, node_count: int) -> None:
    if row.numel() == 0:
        return
    max_index = int(torch.maximum(row.max(), col.max()).item())
    min_index = int(torch.minimum(row.min(), col.min()).item())
    if max_index >= node_count or min_index < 0:
        raise ValueError("row and col indices must be within [0, num_nodes)")


def _sort_perm_by_row(row: Tensor) -> Tensor:
    if row.device.type == "npu" and int(row.max().item()) <= _FLOAT32_EXACT_INT_MAX:
        # Avoid AiCPU fallback from int64 ArgSort while keeping row and col as int64.
        return torch.argsort(row.to(dtype=torch.float32), stable=True)
    return torch.argsort(row.cpu(), stable=True).to(row.device)


def _preprocess(row: Tensor, col: Tensor, weight: OptTensor, num_nodes: int) -> Tuple[Tensor, Tensor, OptTensor]:
    mask = row != col
    row = row[mask]
    col = col[mask]
    weight = weight[mask] if weight is not None else None
    if row.numel() == 0:
        rowptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=row.device)
        return rowptr, col, weight
    row, col, weight = _shuffle_unweighted_edges(row, col, weight)
    row, col, weight = _sort_edges_by_row(row, col, weight)
    arange = torch.arange(num_nodes + 1, dtype=torch.long, device=row.device)
    rowptr = torch.bucketize(arange, row)
    return rowptr, col, weight


def _shuffle_unweighted_edges(row: Tensor, col: Tensor, weight: OptTensor) -> Tuple[Tensor, Tensor, OptTensor]:
    if weight is not None:
        return row, col, weight
    edge_perm = torch.randperm(row.numel(), device=torch.device("cpu")).to(row.device)
    return row[edge_perm], col[edge_perm], weight


def _sort_edges_by_row(row: Tensor, col: Tensor, weight: OptTensor) -> Tuple[Tensor, Tensor, OptTensor]:
    is_sorted = row.numel() <= 1 or bool((row[1:] >= row[:-1]).all().item())
    if is_sorted:
        return row, col, weight
    perm = _sort_perm_by_row(row)
    weight = weight[perm] if weight is not None else None
    return row[perm], col[perm], weight


def _is_uniform_weight_candidate(weight: OptTensor) -> bool:
    if weight is None or weight.numel() == 0:
        return False
    return bool((weight[0] >= 0).item()) and bool((weight == weight[0]).all().item())


def _is_canonical_complete_graph(rowptr: Tensor, col: Tensor, num_nodes: int) -> bool:
    if num_nodes <= 1:
        return col.numel() == 0
    if col.numel() != num_nodes * (num_nodes - 1):
        return False
    expected = torch.arange(num_nodes, dtype=torch.long, device=col.device).repeat(num_nodes)
    nodes = torch.arange(num_nodes, dtype=torch.long, device=col.device).repeat_interleave(num_nodes)
    expected = expected[expected != nodes]
    return bool(torch.equal(col, expected))


def _weight_mode(rowptr: Tensor, col: Tensor, weight: OptTensor, num_nodes: int) -> int:
    if weight is None:
        return 0
    if _is_uniform_weight_candidate(weight):
        if _is_canonical_complete_graph(rowptr, col, num_nodes):
            return 3
        return 2
    return 1


def _available_neighbor(graph: _CpuGraph, neighbor: int) -> bool:
    return 0 <= neighbor < graph.num_nodes and graph.cluster[neighbor] < 0


def _assign_cluster_pair(graph: _CpuGraph, node: int, neighbor: int) -> None:
    cluster_value = min(node, neighbor)
    graph.cluster[node] = cluster_value
    graph.cluster[neighbor] = cluster_value


def _match_unweighted_cpu(graph: _CpuGraph, node: int) -> None:
    graph.cluster[node] = node
    start = graph.rowptr_list[node]
    end = graph.rowptr_list[node + 1]
    for edge_idx in range(start, end):
        neighbor = graph.col_list[edge_idx]
        if _available_neighbor(graph, neighbor):
            _assign_cluster_pair(graph, node, neighbor)
            return


def _best_weighted_neighbor(graph: _CpuGraph, node: int) -> int:
    best_neighbor = node
    best_weight = 0.0
    start = graph.rowptr_list[node]
    end = graph.rowptr_list[node + 1]
    for edge_idx in range(start, end):
        neighbor = graph.col_list[edge_idx]
        if not _available_neighbor(graph, neighbor):
            continue
        edge_weight = float(graph.weight_list[edge_idx])
        if edge_weight >= best_weight:
            best_weight = edge_weight
            best_neighbor = neighbor
    return best_neighbor


def _process_cpu_node(graph: _CpuGraph, node: int) -> None:
    if graph.cluster[node] >= 0:
        return
    if graph.weight_list is None:
        _match_unweighted_cpu(graph, node)
        return
    _assign_cluster_pair(graph, node, _best_weighted_neighbor(graph, node))


def _make_cpu_graph(rowptr: Tensor, col: Tensor, weight: OptTensor, num_nodes: int) -> _CpuGraph:
    weight_list = weight.cpu().tolist() if weight is not None else None
    return _CpuGraph(rowptr.cpu().tolist(), col.cpu().tolist(), weight_list, [-1] * num_nodes, num_nodes)


def _graclus_greedy_cpu(rowptr: Tensor, col: Tensor, weight: OptTensor, num_nodes: int, device: torch.device) -> Tensor:
    graph = _make_cpu_graph(rowptr, col, weight, num_nodes)
    node_perm = torch.randperm(num_nodes, device=torch.device("cpu")).tolist()
    for node in node_perm:
        _process_cpu_node(graph, node)
    return torch.tensor(graph.cluster, dtype=torch.long, device=device)


def _make_cache_key(row: Tensor, col: Tensor, weight: OptTensor, node_count: int):
    if weight is None:
        return None
    row_version = _tensor_version(row)
    col_version = _tensor_version(col)
    weight_version = _tensor_version(weight)
    if row_version is None or col_version is None or weight_version is None:
        return None
    return _CsrCacheKey(
        row_id=id(row),
        col_id=id(col),
        weight_id=id(weight),
        row_ptr=int(row.data_ptr()),
        col_ptr=int(col.data_ptr()),
        weight_ptr=int(weight.data_ptr()),
        row_version=row_version,
        col_version=col_version,
        weight_version=weight_version,
        edge_count=int(row.numel()),
        node_count=node_count,
        weight_dtype=str(weight.dtype),
        device=str(row.device),
    )


def _cached_preprocess(row: Tensor, col: Tensor, weight: OptTensor, node_count: int, cache_key):
    cached = _CSR_CACHE.get(cache_key) if cache_key is not None else None
    if cached is not None:
        return cached
    rowptr, csr_col, csr_weight = _preprocess(row, col, weight, node_count)
    weight_mode = _weight_mode(rowptr, csr_col, csr_weight, node_count)
    if cache_key is not None:
        _cache_put(cache_key, (rowptr, csr_col, csr_weight, weight_mode))
    return rowptr, csr_col, csr_weight, weight_mode


def _npu_weight(weight: OptTensor) -> OptTensor:
    if weight is None:
        return None
    return weight.to(dtype=torch.float32).contiguous()


def _run_npu(context: _RunContext) -> Tensor:
    cached = _CSR_CACHE.get(context.cache_key) if context.cache_key is not None else None
    if cached is None:
        cached = _cached_preprocess(
            context.row.contiguous(),
            context.col.contiguous(),
            _npu_weight(context.weight),
            context.node_count,
            context.cache_key,
        )
    rowptr, csr_col, csr_weight, weight_mode = cached
    node_perm = _get_node_perm(context)
    if csr_weight is None:
        csr_weight = torch.empty(0, dtype=torch.float32, device=context.row.device)
    return _pybind.graclus_cluster_npu(
        rowptr,
        csr_col,
        csr_weight,
        node_perm,
        context.node_count,
        context.weight is not None,
        weight_mode,
    )


def _run_cpu(context: _RunContext, work_device: torch.device) -> Tensor:
    weight_work = context.weight.contiguous() if context.weight is not None else None
    rowptr, csr_col, csr_weight, _ = _cached_preprocess(
        context.row.contiguous(),
        context.col.contiguous(),
        weight_work,
        context.node_count,
        context.cache_key,
    )
    return _graclus_greedy_cpu(rowptr, csr_col, csr_weight, context.node_count, work_device)


def graclus_cluster(row: Tensor, col: Tensor, weight: OptTensor = None, num_nodes: Optional[int] = None) -> Tensor:
    """
    Greedy graph clustering compatible with torch_cluster.graclus_cluster.
    """
    _validate_inputs(row, col, weight, num_nodes)
    node_count = _maybe_num_nodes(row, col, num_nodes)
    if node_count == 0:
        return torch.empty(0, dtype=torch.long, device=row.device)
    if weight is not None and weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("weight supports float16, bfloat16, float32, and float64")
    cache_key = _make_cache_key(row, col, weight, node_count)
    if cache_key is None or cache_key not in _CSR_CACHE:
        _check_index_bounds(row, col, node_count)
    context = _RunContext(row, col, weight, node_count, cache_key)
    use_npu_kernel = row.device.type == "npu" and (weight is None or weight.dtype != torch.float64)
    if use_npu_kernel:
        return _run_npu(context)
    return _run_cpu(context, row.device)

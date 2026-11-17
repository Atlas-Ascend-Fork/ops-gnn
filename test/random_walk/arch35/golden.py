# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Deterministic CPU golden for the Ascend 950 random_walk tests.

The production kernel and this reference share a counter-based Philox4x32-10
mapping. Each start node owns an independent counter stream, so scheduling,
block count and rejection-loop divergence cannot perturb another walk.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor


LOGGER = logging.getLogger(__name__)
_MASK32 = (1 << 32) - 1
_UINT32_RANGE = 1 << 32
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85


def _mul_hi_lo_u32(lhs: int, rhs: int) -> Tuple[int, int]:
    product = (lhs & _MASK32) * (rhs & _MASK32)
    return (product >> 32) & _MASK32, product & _MASK32


def philox4x32_10(
    counter: Tuple[int, int, int, int], key: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Returns one Philox4x32-10 block using Random123 constants."""
    c0, c1, c2, c3 = (value & _MASK32 for value in counter)
    k0, k1 = (value & _MASK32 for value in key)

    for _ in range(10):
        hi0, lo0 = _mul_hi_lo_u32(_PHILOX_M0, c0)
        hi1, lo1 = _mul_hi_lo_u32(_PHILOX_M1, c2)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & _MASK32,
            lo1,
            (hi0 ^ c3 ^ k1) & _MASK32,
            lo0,
        )
        k0 = (k0 + _PHILOX_W0) & _MASK32
        k1 = (k1 + _PHILOX_W1) & _MASK32

    return c0, c1, c2, c3


@dataclass
class PhiloxStream:
    """Per-walk random stream matching random_walk.cpp."""

    seed: int
    offset: int
    walk_index: int
    block_index: int = 0
    lane: int = 4
    values: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))

    def next_u32(self) -> int:
        if self.lane >= 4:
            self._refill()
        value = self.values[self.lane]
        self.lane += 1
        return value

    def _refill(self) -> None:
        seed_lo = self.seed & _MASK32
        seed_hi = (self.seed >> 32) & _MASK32
        block = self.offset + self.block_index
        key = (seed_lo, seed_hi)
        counter = (
            block & _MASK32,
            (block >> 32) & _MASK32,
            self.walk_index & _MASK32,
            (self.walk_index >> 32) & _MASK32,
        )
        self.values = philox4x32_10(counter, key)
        self.block_index += 1
        self.lane = 0


def _sample_index(random_u32: int, degree: int) -> int:
    if degree <= 0 or degree > _MASK32:
        raise ValueError("degree must be in [1, 2**32 - 1]")
    return ((random_u32 & _MASK32) * degree) >> 32


def _probability_threshold(probability: float) -> int:
    if probability <= 0:
        return 0
    if probability >= 1:
        return _UINT32_RANGE
    return max(1, int(math.floor(probability * _UINT32_RANGE)))


def _validate_vector(name: str, value: Tensor) -> None:
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor for the golden model")
    if value.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype torch.int64")
    if value.dim() != 1:
        raise ValueError(f"{name} must be one-dimensional")


def prepare_csr(
    row: Tensor,
    col: Tensor,
    start: Tensor,
    *,
    coalesced: bool = True,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Reproduces torch_cluster.rw.random_walk COO-to-CSR preprocessing."""
    _validate_vector("row", row)
    _validate_vector("col", col)
    _validate_vector("start", start)
    if row.numel() != col.numel():
        raise ValueError("row and col must contain the same number of edges")

    if num_nodes is None:
        maxima = []
        if row.numel() > 0:
            maxima.extend((int(row.max()), int(col.max())))
        if start.numel() > 0:
            maxima.append(int(start.max()))
        if not maxima:
            raise ValueError("num_nodes is required when row, col and start are empty")
        num_nodes = max(maxima) + 1

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if row.numel() > 0:
        if int(row.min()) < 0 or int(col.min()) < 0:
            raise ValueError("row and col must be non-negative")
        if int(row.max()) >= num_nodes or int(col.max()) >= num_nodes:
            raise ValueError("row and col entries must be smaller than num_nodes")
    if start.numel() > 0:
        if int(start.min()) < 0 or int(start.max()) >= num_nodes:
            raise ValueError("start entries must satisfy 0 <= start[i] < num_nodes")

    row_work = row.contiguous()
    col_work = col.contiguous()
    if coalesced and row.numel() > 0:
        permutation = torch.argsort(row_work * num_nodes + col_work)
        row_work = row_work[permutation]
        col_work = col_work[permutation]

    if row_work.numel() > 1 and bool((row_work[1:] < row_work[:-1]).any()):
        raise ValueError("coalesced=False requires edges grouped by source row")

    degree = row.new_zeros(num_nodes)
    if row_work.numel() > 0:
        degree.scatter_add_(0, row_work, torch.ones_like(row_work))
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(degree, 0, out=rowptr[1:])
    return rowptr, col_work


def is_neighbor(
    rowptr: Tensor, col: Tensor, node: int, target: int, neighbors_sorted: bool
) -> bool:
    begin = int(rowptr[node])
    end = int(rowptr[node + 1])
    if not neighbors_sorted or end - begin <= 16:
        for edge in range(begin, end):
            if int(col[edge]) == target:
                return True
        return False

    while begin < end:
        middle = begin + (end - begin) // 2
        value = int(col[middle])
        if value < target:
            begin = middle + 1
        else:
            end = middle
    return begin < int(rowptr[node + 1]) and int(col[begin]) == target


@dataclass(frozen=True)
class WalkThresholds:
    returning: int
    neighboring: int
    distant: int


@dataclass(frozen=True)
class WalkContext:
    rowptr: Tensor
    col: Tensor
    thresholds: WalkThresholds
    uniform: bool
    neighbors_sorted: bool


@dataclass(frozen=True)
class WalkConfig:
    walk_length: int
    seed: int
    offset: int


def _parse_named_arguments(args: tuple, kwargs: dict, names: tuple, defaults: dict, function_name: str) -> tuple:
    if len(args) > len(names):
        raise TypeError(f"{function_name}() takes at most {len(names)} positional arguments")
    values = dict(zip(names, args))
    for name, value in kwargs.items():
        if name not in names:
            raise TypeError(f"{function_name}() got an unexpected keyword argument '{name}'")
        if name in values:
            raise TypeError(f"{function_name}() got multiple values for argument '{name}'")
        values[name] = value
    required_count = len(names) - len(defaults)
    missing = [name for name in names[:required_count] if name not in values]
    if missing:
        raise TypeError(f"{function_name}() missing required arguments: {', '.join(missing)}")
    for name, value in defaults.items():
        values.setdefault(name, value)
    return tuple(values[name] for name in names)


def _validate_csr_inputs(
    rowptr: Tensor, col: Tensor, start: Tensor, walk_length: int, bias: Tuple[float, float]
) -> None:
    _validate_vector("rowptr", rowptr)
    _validate_vector("col", col)
    _validate_vector("start", start)
    if walk_length < 0:
        raise ValueError("walk_length must be non-negative")
    p, q = bias
    if not math.isfinite(p) or p <= 0:
        raise ValueError("p must be finite and greater than zero")
    if not math.isfinite(q) or q <= 0:
        raise ValueError("q must be finite and greater than zero")
    if rowptr.numel() == 0:
        raise ValueError("rowptr must contain at least one element")


def _compute_thresholds(p: float, q: float) -> WalkThresholds:
    log_weights = (-math.log(p), 0.0, -math.log(q))
    max_log_weight = max(log_weights)
    probabilities = tuple(math.exp(value - max_log_weight) for value in log_weights)
    return WalkThresholds(*(_probability_threshold(value) for value in probabilities))


def _accept_candidate(context: WalkContext, candidate: int, previous: int, draw: int) -> bool:
    if candidate == previous:
        return draw < context.thresholds.returning
    if is_neighbor(context.rowptr, context.col, candidate, previous, context.neighbors_sorted):
        return draw < context.thresholds.neighboring
    return draw < context.thresholds.distant


def _sample_biased_candidate(
    context: WalkContext, begin: int, degree: int, previous: int, random: PhiloxStream
) -> Tuple[int, int]:
    while True:
        edge = begin + _sample_index(random.next_u32(), degree)
        candidate = int(context.col[edge])
        if _accept_candidate(context, candidate, previous, random.next_u32()):
            return edge, candidate


def _sample_step(
    context: WalkContext, current: int, previous: int, step: int, random: PhiloxStream
) -> Tuple[int, int]:
    begin = int(context.rowptr[current])
    degree = int(context.rowptr[current + 1]) - begin
    if degree == 0:
        return -1, current
    if context.uniform or step == 0:
        edge = begin + _sample_index(random.next_u32(), degree)
        return edge, int(context.col[edge])
    if degree == 1:
        return begin, int(context.col[begin])
    return _sample_biased_candidate(context, begin, degree, previous, random)


def _walk_one(
    context: WalkContext, start_node: int, config: WalkConfig, walk_index: int
) -> Tuple[List[int], List[int]]:
    random = PhiloxStream(seed=config.seed, offset=config.offset, walk_index=walk_index)
    nodes = [start_node]
    edges = []
    previous = start_node
    current = start_node
    for step in range(config.walk_length):
        edge, candidate = _sample_step(context, current, previous, step, random)
        nodes.append(candidate)
        edges.append(edge)
        previous, current = current, candidate
    return nodes, edges


def random_walk_csr_golden(*args, **kwargs) -> Tuple[Tensor, Tensor]:
    """Runs the deterministic CSR random walk used by the NPU kernel."""
    names = ("rowptr", "col", "start", "walk_length", "p", "q", "seed", "offset", "neighbors_sorted")
    defaults = {"p": 1.0, "q": 1.0, "seed": 0, "offset": 0, "neighbors_sorted": True}
    values = _parse_named_arguments(args, kwargs, names, defaults, "random_walk_csr_golden")
    rowptr, col, start, walk_length, p, q, seed, offset, neighbors_sorted = values
    _validate_csr_inputs(rowptr, col, start, walk_length, (p, q))
    context = WalkContext(
        rowptr, col, _compute_thresholds(p, q),
        math.isclose(p, 1.0, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(q, 1.0, rel_tol=0.0, abs_tol=0.0),
        neighbors_sorted,
    )
    config = WalkConfig(walk_length, seed, offset)
    node_seq = torch.empty((start.numel(), walk_length + 1), dtype=torch.int64)
    edge_seq = torch.empty((start.numel(), walk_length), dtype=torch.int64)

    for walk_index, start_node in enumerate(start.tolist()):
        nodes, edges = _walk_one(context, start_node, config, walk_index)
        node_seq[walk_index] = torch.tensor(nodes, dtype=torch.int64)
        edge_seq[walk_index] = torch.tensor(edges, dtype=torch.int64)

    return node_seq, edge_seq


def random_walk_golden(*args, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    names = (
        "row", "col", "start", "walk_length", "p", "q", "coalesced",
        "num_nodes", "return_edge_indices", "seed", "offset",
    )
    defaults = {
        "p": 1.0, "q": 1.0, "coalesced": True, "num_nodes": None,
        "return_edge_indices": False, "seed": 0, "offset": 0,
    }
    values = _parse_named_arguments(args, kwargs, names, defaults, "random_walk_golden")
    row, col, start, walk_length, p, q, coalesced, num_nodes, return_edge_indices, seed, offset = values
    rowptr, sorted_col = prepare_csr(
        row, col, start, coalesced=coalesced, num_nodes=num_nodes
    )
    node_seq, edge_seq = random_walk_csr_golden(
        rowptr,
        sorted_col,
        start,
        walk_length,
        p,
        q,
        seed=seed,
        offset=offset,
        neighbors_sorted=coalesced,
    )
    if return_edge_indices:
        return node_seq, edge_seq
    return node_seq


def _validate_transition(rowptr: Tensor, col: Tensor, current: int, next_node: int, edge: int) -> None:
    if edge == -1:
        if int(rowptr[current]) != int(rowptr[current + 1]) or next_node != current:
            raise AssertionError("isolated-node transition is invalid")
        return
    if not int(rowptr[current]) <= edge < int(rowptr[current + 1]):
        raise AssertionError("edge index is outside the CSR row")
    if int(col[edge]) != next_node:
        raise AssertionError("edge target does not match the walk")


def assert_walk_consistent(
    rowptr: Tensor, col: Tensor, node_seq: Tensor, edge_seq: Tensor
) -> None:
    """Checks graph legality independently of the RNG implementation."""
    for walk in range(node_seq.size(0)):
        for step in range(edge_seq.size(1)):
            current = int(node_seq[walk, step])
            next_node = int(node_seq[walk, step + 1])
            edge = int(edge_seq[walk, step])
            _validate_transition(rowptr, col, current, next_node, edge)


def _self_check() -> dict:
    expected = (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    actual = philox4x32_10((0, 0, 0, 0), (0, 0))
    if actual != expected:
        raise AssertionError(f"Philox known-vector mismatch: {actual!r}")

    return {"philox_known_vector": "PASS"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="random_walk CPU golden self-check")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    result = _self_check()
    if args.json:
        LOGGER.info("%s", json.dumps(result, sort_keys=True))
    else:
        for name, status in result.items():
            LOGGER.info("%s: %s", name, status)


if __name__ == "__main__":
    main()

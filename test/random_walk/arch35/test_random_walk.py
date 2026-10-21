# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Offline correctness tests for the deterministic random_walk CPU golden."""

import math
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Tuple

import pytest
import torch

# Assigned only by the existing NPU fixture; Host tests do not import NPU bindings.
ops_gnn = None

_SPEC = spec_from_file_location("random_walk_golden", Path(__file__).with_name("golden.py"))
_GOLDEN = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GOLDEN
_SPEC.loader.exec_module(_GOLDEN)
PhiloxStream = _GOLDEN.PhiloxStream
is_neighbor = _GOLDEN.is_neighbor
assert_walk_consistent = _GOLDEN.assert_walk_consistent
philox4x32_10 = _GOLDEN.philox4x32_10
prepare_csr = _GOLDEN.prepare_csr
random_walk_golden = _GOLDEN.random_walk_golden


def test_philox_known_vector():
    assert philox4x32_10((0, 0, 0, 0), (0, 0)) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )


def test_philox_stream_is_repeatable_and_partitioned():
    first = PhiloxStream(seed=42, offset=8, walk_index=3)
    second = PhiloxStream(seed=42, offset=8, walk_index=3)
    other_walk = PhiloxStream(seed=42, offset=8, walk_index=4)
    values = [first.next_u32() for _ in range(12)]
    assert values == [second.next_u32() for _ in range(12)]
    assert values != [other_walk.next_u32() for _ in range(12)]


def test_uniform_degree_one_and_isolated_nodes_are_exact():
    row = torch.tensor([0, 1], dtype=torch.int64)
    col = torch.tensor([1, 0], dtype=torch.int64)
    start = torch.tensor([0, 1, 2], dtype=torch.int64)
    node, edge = random_walk_golden(
        row,
        col,
        start,
        4,
        num_nodes=3,
        return_edge_indices=True,
        seed=99,
    )
    assert node.tolist() == [
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [2, 2, 2, 2, 2],
    ]
    assert edge.tolist() == [[0, 1, 0, 1], [1, 0, 1, 0], [-1, -1, -1, -1]]


@pytest.mark.parametrize("p,q", [(1.0, 1.0), (0.25, 4.0), (4.0, 0.25)])
def test_paths_are_graph_consistent(p, q):
    row = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
    col = torch.tensor([1, 0, 2, 3, 1, 4, 1, 4, 2, 3], dtype=torch.int64)
    start = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    rowptr, sorted_col = prepare_csr(row, col, start)
    node, edge = random_walk_golden(
        row,
        col,
        start,
        32,
        p=p,
        q=q,
        return_edge_indices=True,
        seed=202608,
    )
    assert_walk_consistent(rowptr, sorted_col, node, edge)


def test_coalesced_sort_and_edge_index_semantics():
    row = torch.tensor([1, 0, 1], dtype=torch.int64)
    col = torch.tensor([2, 1, 0], dtype=torch.int64)
    start = torch.tensor([0], dtype=torch.int64)
    node, edge = random_walk_golden(
        row,
        col,
        start,
        1,
        coalesced=True,
        return_edge_indices=True,
        seed=0,
    )
    assert node.tolist() == [[0, 1]]
    assert edge.tolist() == [[0]]


def test_uncoalesced_large_unsorted_neighbor_lookup():
    targets = list(range(1, 21))
    targets[0], targets[-1] = targets[-1], targets[0]
    rowptr = torch.tensor([0, 20] + [20] * 20, dtype=torch.int64)
    col = torch.tensor(targets, dtype=torch.int64)
    assert is_neighbor(rowptr, col, 0, 20, neighbors_sorted=False)
    assert not is_neighbor(rowptr, col, 0, 99, neighbors_sorted=False)


def test_walk_length_zero():
    row = torch.tensor([0], dtype=torch.int64)
    col = torch.tensor([0], dtype=torch.int64)
    start = torch.tensor([0, 0], dtype=torch.int64)
    node, edge = random_walk_golden(
        row,
        col,
        start,
        0,
        p=2.0,
        q=0.5,
        return_edge_indices=True,
    )
    assert node.shape == (2, 1)
    assert edge.shape == (2, 0)
    assert node[:, 0].tolist() == [0, 0]


@pytest.mark.parametrize("p,q", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (math.inf, 1.0)])
def test_invalid_bias_is_rejected(p, q):
    value = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(ValueError):
        random_walk_golden(value, value, value, 1, p=p, q=q)


def test_extreme_finite_bias_does_not_overflow_normalization():
    value = torch.tensor([0], dtype=torch.int64)
    result = random_walk_golden(value, value, value, 2, p=5e-324, q=1e308)
    assert result.tolist() == [[0, 0, 0]]




@pytest.fixture(scope="module")
def _select_npu_device():
    """Run the module on the card selected by ``NPU_DEVICE_ID``."""
    global ops_gnn
    pytest.importorskip("torch_npu")
    ops_gnn = pytest.importorskip("ops_gnn")
    if not hasattr(ops_gnn, "random_walk"):
        pytest.skip("ops_gnn.random_walk is not built", allow_module_level=True)
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


TEST_CONFIGS = [
    (1024, 4, 8),
    (4096, 8, 16),
    (8192, 10, 32),
    (16384, 6, 64),
    (32768, 4, 128),
    (4096, 20, 8),
    (8192, 4, 256),
    (2048, 32, 4),
]

GENERAL_CONFIGS = [
    (4, 1, 1),
    (8, 0, 2),
    (16, 2, 4),
    (0, 1, 1),
    (65536, 8, 256),
    (16384, 2, 1024),
    (8192, 64, 16),
]


def _make_case(edge_count: int, walk_length: int, starts: int):
    """Build exactly the deterministic COO inputs described by the sample."""
    torch.manual_seed(42)
    num_nodes = max(200, edge_count // 2)
    row = torch.randint(0, num_nodes, (edge_count,), dtype=torch.int64)
    col = torch.randint(0, num_nodes, (edge_count,), dtype=torch.int64)
    start = torch.arange(min(starts, num_nodes), dtype=torch.int64)
    return row, col, start, walk_length, num_nodes


def _assert_result_contract(result: torch.Tensor, start: torch.Tensor, walk_length: int):
    assert isinstance(result, torch.Tensor)
    assert result.device.type == "npu"
    assert result.dtype == torch.int64
    assert tuple(result.shape) == (start.numel(), walk_length + 1)
    assert torch.equal(result[:, 0].cpu(), start)


def _assert_legal_walk(
    row: torch.Tensor,
    col: torch.Tensor,
    result: torch.Tensor,
    num_nodes: int,
):
    adjacency = [set() for _ in range(num_nodes)]
    for source, target in zip(row.tolist(), col.tolist()):
        adjacency[source].add(target)

    paths = result.cpu().tolist()
    for path in paths:
        for source, target in zip(path, path[1:]):
            assert 0 <= source < num_nodes
            assert 0 <= target < num_nodes
            if adjacency[source]:
                assert target in adjacency[source]
            else:
                assert target == source


def _run_and_check(edge_count: int, walk_length: int, starts: int):
    row, col, start, walk_length, num_nodes = _make_case(edge_count, walk_length, starts)
    kwargs = {"num_nodes": num_nodes} if edge_count == 0 else {}
    result = ops_gnn.random_walk(
        row.npu(), col.npu(), start.npu(), walk_length, **kwargs
    )
    _assert_result_contract(result, start, walk_length)
    _assert_legal_walk(row, col, result, num_nodes)


@pytest.mark.usefixtures("_select_npu_device")
class TestRandomWalk:
    @staticmethod
    @pytest.mark.parametrize("edge_count,walk_length,starts", TEST_CONFIGS)
    def test_configs(edge_count: int, walk_length: int, starts: int):
        _run_and_check(edge_count, walk_length, starts)

    @staticmethod
    @pytest.mark.parametrize("edge_count,walk_length,starts", GENERAL_CONFIGS)
    def test_general(edge_count: int, walk_length: int, starts: int):
        # The official sample catches RuntimeError/ValueError here.  Acceptance
        # must fail on either exception, including for the empty-edge case.
        _run_and_check(edge_count, walk_length, starts)

    @staticmethod
    def test_walk_length_zero():
        torch.manual_seed(42)
        num_nodes = 10
        row = torch.randint(0, num_nodes, (20,), dtype=torch.int64)
        col = torch.randint(0, num_nodes, (20,), dtype=torch.int64)
        start = torch.tensor([0, 1], dtype=torch.int64)
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 0)
        _assert_result_contract(result, start, 0)
        _assert_legal_walk(row, col, result, num_nodes)

    @staticmethod
    def test_walk_length_one():
        torch.manual_seed(42)
        num_nodes = 10
        row = torch.randint(0, num_nodes, (20,), dtype=torch.int64)
        col = torch.randint(0, num_nodes, (20,), dtype=torch.int64)
        start = torch.tensor([0], dtype=torch.int64)
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 1)
        _assert_result_contract(result, start, 1)
        _assert_legal_walk(row, col, result, num_nodes)

    @staticmethod
    def test_return_edge_indices():
        row = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
        col = torch.tensor([1, 0, 2, 1], dtype=torch.int64)
        start = torch.tensor([0], dtype=torch.int64)

        nodes, edges = ops_gnn.random_walk(
            row.npu(), col.npu(), start.npu(), 4,
            return_edge_indices=True,
        )
        _assert_result_contract(nodes, start, 4)
        _assert_legal_walk(row, col, nodes, 3)
        assert edges.device.type == "npu"
        assert edges.dtype == torch.int64
        assert tuple(edges.shape) == (1, 4)

        sorted_pairs: Tuple[Tuple[int, int], ...] = tuple(sorted(zip(row.tolist(), col.tolist())))
        node_values = nodes.cpu().tolist()[0]
        for step, edge_index in enumerate(edges.cpu().tolist()[0]):
            assert 0 <= edge_index < len(sorted_pairs)
            assert sorted_pairs[edge_index] == (node_values[step], node_values[step + 1])

        nodes_only = ops_gnn.random_walk(
            row.npu(), col.npu(), start.npu(), 4,
            return_edge_indices=False,
        )
        _assert_result_contract(nodes_only, start, 4)

    @staticmethod
    @pytest.mark.parametrize("p,q", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)])
    def test_pq_params(p: float, q: float):
        row, col, start, _, num_nodes = _make_case(100, 4, 1)
        result = ops_gnn.random_walk(
            row.npu(), col.npu(), start.npu(), 4, p=p, q=q
        )
        _assert_result_contract(result, start, 4)
        _assert_legal_walk(row, col, result, num_nodes)

    @staticmethod
    def test_reproducible():
        torch.manual_seed(42)
        row = torch.randint(0, 100, (200,), dtype=torch.int64)
        col = torch.randint(0, 100, (200,), dtype=torch.int64)
        start = torch.tensor([0, 5, 10], dtype=torch.int64)
        row_npu, col_npu, start_npu = row.npu(), col.npu(), start.npu()

        torch.manual_seed(42)
        torch.npu.manual_seed_all(42)
        first = ops_gnn.random_walk(row_npu, col_npu, start_npu, 4)
        torch.npu.synchronize()

        torch.manual_seed(42)
        torch.npu.manual_seed_all(42)
        second = ops_gnn.random_walk(row_npu, col_npu, start_npu, 4)
        torch.npu.synchronize()
        assert torch.equal(first, second)

    @staticmethod
    def test_coalesced_false_is_reproducible():
        row = torch.tensor([0, 0, 1, 1], dtype=torch.int64).npu()
        col = torch.tensor([1, 2, 0, 2], dtype=torch.int64).npu()
        start = torch.tensor([0, 1, 0, 1], dtype=torch.int64).npu()

        torch.manual_seed(202608)
        torch.npu.manual_seed_all(202608)
        first = ops_gnn.random_walk(
            row, col, start, 16, p=0.5, q=2.0, coalesced=False
        )
        torch.manual_seed(202608)
        torch.npu.manual_seed_all(202608)
        second = ops_gnn.random_walk(
            row, col, start, 16, p=0.5, q=2.0, coalesced=False
        )
        assert torch.equal(first, second)

    @staticmethod
    def test_isolated_node():
        row = torch.tensor([0, 1], dtype=torch.int64)
        col = torch.tensor([1, 2], dtype=torch.int64)
        start = torch.tensor([3], dtype=torch.int64)
        result = ops_gnn.random_walk(
            row.npu(), col.npu(), start.npu(), 4, num_nodes=4
        )
        _assert_result_contract(result, start, 4)
        assert result.cpu().tolist() == [[3, 3, 3, 3, 3]]

    @staticmethod
    def test_large():
        torch.manual_seed(42)
        num_nodes = 200
        row = torch.randint(0, num_nodes, (4096,), dtype=torch.int64)
        col = torch.randint(0, num_nodes, (4096,), dtype=torch.int64)
        start = torch.tensor([0, 10, 20, 30], dtype=torch.int64)
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 8)
        _assert_result_contract(result, start, 8)
        _assert_legal_walk(row, col, result, num_nodes)


@pytest.fixture
def csr_prepare(monkeypatch):
    # The Host-only cache path never calls the compiled random-walk binding.
    # Isolate its import dependency without changing the production import.
    from types import ModuleType

    path = Path(__file__).resolve().parents[3] / "python/ops_gnn/random_walk.py"
    package_name = "ops_gnn_host_cache_test"
    package = ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    bindings_name = package_name + "._pybind"
    bindings = ModuleType(bindings_name)
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, bindings_name, bindings)
    spec = spec_from_file_location(package_name + ".random_walk", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "_prepare_csr")


@pytest.mark.parametrize("has_edges", [False, True])
def test_inference_graphs_build_csr_and_observe_mutation(csr_prepare, has_edges):
    prepare = csr_prepare
    with torch.inference_mode():
        row = torch.tensor([1, 0] if has_edges else [], dtype=torch.int64)
        col = torch.tensor([0, 1] if has_edges else [], dtype=torch.int64)
        ptr, sorted_col = prepare(row, col, 2, True)
        assert torch.equal(ptr, torch.tensor([0, 1, 2] if has_edges else [0, 0, 0]))
        assert torch.equal(
            sorted_col, torch.tensor([1, 0] if has_edges else [], dtype=torch.int64)
        )
        if has_edges:
            row.fill_(0)
            ptr, sorted_col = prepare(row, col, 2, True)
            assert torch.equal(ptr, torch.tensor([0, 2, 2]))
            assert torch.equal(sorted_col, torch.tensor([0, 1]))


def test_regular_graphs_keep_reusable_csr(csr_prepare):
    row, col = torch.tensor([1, 0]), torch.tensor([0, 1])
    prepare = csr_prepare
    first = prepare(row, col, 2, True)
    second = prepare(row, col, 2, True)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])

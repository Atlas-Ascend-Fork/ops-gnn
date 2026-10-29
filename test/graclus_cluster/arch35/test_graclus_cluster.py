# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest
import torch
import ops_gnn

_SPEC = spec_from_file_location("graclus_cluster_golden", Path(__file__).with_name("golden.py"))
_GOLDEN = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GOLDEN
_SPEC.loader.exec_module(_GOLDEN)
make_random_edges = _GOLDEN.make_random_edges
reference_graclus_cluster = _GOLDEN.reference_graclus_cluster
_npu_or_skip = _GOLDEN.npu_or_skip
_run_reference_and_npu = _GOLDEN.run_reference_and_npu


def test_tc01_small_graph_without_weight():
    row = torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.long)
    col = torch.tensor([1, 0, 2, 1, 3, 2], dtype=torch.long)
    _run_reference_and_npu(row, col, seed=11)


def test_npu_weighted_calls_advance_node_permutation_rng():
    _npu_or_skip()
    row = torch.tensor([0, 0, 1, 2], dtype=torch.long, device="npu")
    col = torch.tensor([1, 2, 0, 0], dtype=torch.long, device="npu")
    weight = torch.tensor([1.0, 0.5, 1.0, 0.5], device="npu")

    torch.manual_seed(18027)
    before = torch.get_rng_state()
    ops_gnn.graclus_cluster(row, col, weight, num_nodes=3)
    after_first = torch.get_rng_state()
    ops_gnn.graclus_cluster(row, col, weight, num_nodes=3)
    after_second = torch.get_rng_state()

    assert not torch.equal(before, after_first)
    assert not torch.equal(after_first, after_second)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tc02_with_l1_weight(dtype):
    row = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 3, 0, 3, 1, 2], dtype=torch.long)
    weight = torch.tensor([0.1, 0.9, 0.1, 0.8, 0.9, 0.2, 0.8, 0.2], dtype=dtype)
    _run_reference_and_npu(row, col, weight, seed=17)


def test_tc03_num_nodes_with_isolated_nodes():
    row = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([1, 0], dtype=torch.long)
    _run_reference_and_npu(row, col, num_nodes=5, seed=23)


def test_tc04_self_loops_removed():
    row = torch.tensor([0, 0, 1, 2, 2, 3], dtype=torch.long)
    col = torch.tensor([0, 1, 0, 2, 3, 2], dtype=torch.long)
    _run_reference_and_npu(row, col, num_nodes=4, seed=29)


def test_tc05_float64_weight_cpu_fallback_semantics():
    row = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 0], dtype=torch.long)
    weight = torch.tensor([0.1, 0.7, 0.1, 0.7], dtype=torch.float64)
    torch.manual_seed(31)
    expected = reference_graclus_cluster(row, col, weight, num_nodes=3)
    torch.manual_seed(31)
    out = ops_gnn.graclus_cluster(row, col, weight, num_nodes=3)
    if not torch.equal(out, expected):
        pytest.fail("float64 CPU fallback result is not bit-wise equal to reference")


def test_float64_weight_cpu_fallback_with_inference_tensors():
    reference_row = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    reference_col = torch.tensor([1, 2, 0, 2], dtype=torch.long)
    reference_weight = torch.tensor([0.5, 0.2, 0.5, 0.3], dtype=torch.float64)
    torch.manual_seed(101)
    expected = reference_graclus_cluster(reference_row, reference_col, reference_weight, num_nodes=3)

    with torch.inference_mode():
        row = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        col = torch.tensor([1, 2, 0, 2], dtype=torch.long)
        weight = torch.tensor([0.5, 0.2, 0.5, 0.3], dtype=torch.float64)
        torch.manual_seed(101)
        actual = ops_gnn.graclus_cluster(row, col, weight, num_nodes=3)

    if not torch.equal(actual, expected):
        pytest.fail("inference tensor CPU fallback result is not bit-wise equal to reference")


def test_tc06_empty_edges():
    row = torch.empty(0, dtype=torch.long)
    col = torch.empty(0, dtype=torch.long)
    _run_reference_and_npu(row, col, num_nodes=4, seed=37)


@pytest.mark.parametrize("seed", [0, 1, 7, 2026])
def test_random_unweighted_graphs(seed):
    row, col = make_random_edges(num_nodes=12, num_edges=48, seed=seed)
    _run_reference_and_npu(row, col, num_nodes=12, seed=seed + 101)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seed", [3, 19])
def test_random_l1_weighted_graphs(dtype, seed):
    row, col = make_random_edges(num_nodes=10, num_edges=64, seed=seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1000)
    weight = torch.rand(row.numel(), dtype=torch.float32, generator=generator).to(dtype)
    _run_reference_and_npu(row, col, weight, num_nodes=10, seed=seed + 211)


@pytest.mark.parametrize("num_nodes,num_edges,seed", [(64, 256, 41), (128, 1024, 42), (257, 4096, 43)])
def test_large_random_unweighted_graphs(num_nodes, num_edges, seed):
    row, col = make_random_edges(num_nodes=num_nodes, num_edges=num_edges, seed=seed)
    _run_reference_and_npu(row, col, num_nodes=num_nodes, seed=seed + 3000)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_nodes,num_edges,seed", [(64, 384, 53), (128, 1536, 54)])
def test_large_random_l1_weighted_graphs(dtype, num_nodes, num_edges, seed):
    row, col = make_random_edges(num_nodes=num_nodes, num_edges=num_edges, seed=seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 4000)
    weight = torch.rand(row.numel(), dtype=torch.float32, generator=generator).to(dtype)
    _run_reference_and_npu(row, col, weight, num_nodes=num_nodes, seed=seed + 4100)


def test_dense_weighted_complete_graph_v32():
    row = []
    col = []
    for src in range(32):
        for dst in range(32):
            if src != dst:
                row.append(src)
                col.append(dst)
    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    weight = ((row * 17 + col * 31) % 97).to(torch.float32) / 97.0
    _run_reference_and_npu(row, col, weight, num_nodes=32, seed=61)


def test_duplicate_edges_and_tie_break_keep_first_sorted_edge():
    row = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3], dtype=torch.long)
    col = torch.tensor([1, 2, 3, 0, 2, 0, 1, 0], dtype=torch.long)
    weight = torch.tensor([0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.5], dtype=torch.float32)
    _run_reference_and_npu(row, col, weight, num_nodes=4, seed=67)


def test_negative_weighted_edges_follow_torch_cluster_zero_threshold():
    row = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long)
    weight = torch.tensor([-3.0, -1.0, -3.0, -2.0, -1.0, -2.0], dtype=torch.float32)
    _run_reference_and_npu(row, col, weight, num_nodes=3, seed=71)


def test_directed_asymmetric_graph_with_isolated_tail():
    row = torch.tensor([0, 0, 1, 4, 4, 5, 7, 8], dtype=torch.long)
    col = torch.tensor([1, 2, 3, 0, 5, 6, 8, 7], dtype=torch.long)
    _run_reference_and_npu(row, col, num_nodes=12, seed=73)


def test_non_contiguous_cpu_inputs_before_device_copy():
    row_base = torch.tensor([99, 0, 99, 0, 99, 1, 99, 2, 99, 3, 99, 4], dtype=torch.long)
    col_base = torch.tensor([99, 1, 99, 2, 99, 2, 99, 3, 99, 4, 99, 0], dtype=torch.long)
    weight_base = torch.tensor([9.0, 0.1, 9.0, 0.9, 9.0, 0.2, 9.0, 0.3, 9.0, 0.4, 9.0, 0.5], dtype=torch.float32)
    row = row_base[1::2]
    col = col_base[1::2]
    weight = weight_base[1::2]
    if row.is_contiguous() or col.is_contiguous() or weight.is_contiguous():
        pytest.fail("expected non-contiguous CPU inputs")
    _run_reference_and_npu(row, col, weight, num_nodes=5, seed=79)


def test_empty_graph_without_explicit_num_nodes_returns_empty_cpu():
    row = torch.empty(0, dtype=torch.long)
    col = torch.empty(0, dtype=torch.long)
    out = ops_gnn.graclus_cluster(row, col)
    if out.dtype != torch.long:
        pytest.fail(f"expected torch.long output, but got {out.dtype}")
    if out.numel() != 0:
        pytest.fail(f"expected empty output, but got {out.numel()} elements")


def test_single_node_without_edges_npu():
    row = torch.empty(0, dtype=torch.long)
    col = torch.empty(0, dtype=torch.long)
    _run_reference_and_npu(row, col, num_nodes=1, seed=83)


def test_cluster_labels_use_min_node_id_not_dense_ids():
    row = torch.tensor([4, 0, 2, 3], dtype=torch.long)
    col = torch.tensor([0, 4, 3, 2], dtype=torch.long)
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    torch.manual_seed(89)
    expected = reference_graclus_cluster(row, col, weight, num_nodes=6)
    if not set(expected.tolist()).issubset(set(range(6))):
        pytest.fail("reference cluster labels exceed node id range")
    _run_reference_and_npu(row, col, weight, num_nodes=6, seed=89)


def test_weighted_equal_tie_selects_last_max_edge_like_reference():
    row = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 0], dtype=torch.long)
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    _run_reference_and_npu(row, col, weight, num_nodes=3, seed=97)


def test_weight_cache_invalidates_after_col_inplace_update():
    row = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 0], dtype=torch.long)
    weight = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    torch.manual_seed(2)
    first = ops_gnn.graclus_cluster(row, col, weight, num_nodes=4)

    col[0] = 3
    torch.manual_seed(2)
    expected = reference_graclus_cluster(row, col, weight, num_nodes=4)
    torch.manual_seed(2)
    actual = ops_gnn.graclus_cluster(row, col, weight, num_nodes=4)

    if not torch.equal(actual, expected):
        pytest.fail("cache invalidation result is not bit-wise equal to reference")
    if torch.equal(first, actual):
        pytest.fail("cache did not reflect updated tensor")


def test_input_validation_errors():
    row = torch.tensor([0, 1], dtype=torch.int32)
    col = torch.tensor([1, 0], dtype=torch.long)
    with pytest.raises(TypeError):
        ops_gnn.graclus_cluster(row, col)

    row = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([1], dtype=torch.long)
    with pytest.raises(ValueError):
        ops_gnn.graclus_cluster(row, col)

    weight = torch.tensor([1, 2], dtype=torch.long)
    col = torch.tensor([1, 0], dtype=torch.long)
    with pytest.raises(TypeError):
        ops_gnn.graclus_cluster(row, col, weight)

    with pytest.raises(ValueError):
        ops_gnn.graclus_cluster(row, col, num_nodes=-1)


def test_non_contiguous_npu_inputs_direct():
    _npu_or_skip()
    row_base = torch.tensor([99, 0, 99, 0, 99, 1, 99, 2, 99, 3, 99, 4], dtype=torch.long)
    col_base = torch.tensor([99, 1, 99, 2, 99, 2, 99, 3, 99, 4, 99, 0], dtype=torch.long)
    weight_base = torch.tensor([9.0, 0.1, 9.0, 0.9, 9.0, 0.2, 9.0, 0.3, 9.0, 0.4, 9.0, 0.5], dtype=torch.float32)
    row = row_base[1::2]
    col = col_base[1::2]
    weight = weight_base[1::2]
    torch.manual_seed(101)
    expected = reference_graclus_cluster(row, col, weight, num_nodes=5)

    row_npu = row_base.to("npu")[1::2]
    col_npu = col_base.to("npu")[1::2]
    weight_npu = weight_base.to("npu")[1::2]
    if row_npu.is_contiguous() or col_npu.is_contiguous() or weight_npu.is_contiguous():
        pytest.fail("expected non-contiguous NPU inputs")

    torch.manual_seed(101)
    actual = ops_gnn.graclus_cluster(row_npu, col_npu, weight_npu, num_nodes=5)
    if not torch.equal(actual.cpu(), expected):
        pytest.fail("non-contiguous NPU result is not bit-wise equal to reference")


def test_all_self_loops_become_isolated_clusters():
    row = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    col = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    _run_reference_and_npu(row, col, num_nodes=4, seed=43)


def test_invalid_indices_raise():
    row = torch.tensor([0, 4], dtype=torch.long)
    col = torch.tensor([1, 0], dtype=torch.long)
    with pytest.raises(ValueError):
        ops_gnn.graclus_cluster(row, col, num_nodes=4)


def test_weight_cache_invalidates_after_inplace_update():
    row = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    col = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long)
    weight = torch.tensor([0.9, 0.1, 0.9, 0.2, 0.1, 0.2], dtype=torch.float32)

    torch.manual_seed(5)
    first = ops_gnn.graclus_cluster(row, col, weight, num_nodes=3)

    weight.copy_(torch.tensor([0.1, 0.9, 0.1, 0.2, 0.9, 0.2], dtype=torch.float32))
    torch.manual_seed(5)
    expected = reference_graclus_cluster(row, col, weight, num_nodes=3)
    torch.manual_seed(5)
    actual = ops_gnn.graclus_cluster(row, col, weight, num_nodes=3)

    if not torch.equal(actual, expected):
        pytest.fail("cache invalidation result is not bit-wise equal to reference")
    if torch.equal(first, actual):
        pytest.fail("cache did not reflect updated tensor")

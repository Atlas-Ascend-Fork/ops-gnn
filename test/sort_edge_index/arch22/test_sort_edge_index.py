# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional tests of ``sort_edge_index`` for the arch22 and arch35 builds.

The cases and the CPU composite-key reference are architecture independent, and
both builds must return the same order. What differs is the node count that
leaves the exact sort range: on A2/A3 (``dav-2201``) a square above ``2 ** 24`` no
longer fits float32, on 950 (``dav-3510``) only a square above ``int64`` does.
Both counts are exercised, so every build runs the two-pass order at least once,
and that order must equal the one-pass composite key.
"""

import importlib.util
import os

import pytest
import torch


def _load_golden():
    """Import the ``golden.py`` next to this module under its own name."""
    directory = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_golden_sort_edge_index_arch22", os.path.join(directory, "golden.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_golden = _load_golden()

# Node counts that make the composite key leave the exact sort range: above
# sqrt(2 ** 24) for the float32 sort of A2/A3, above sqrt(int64 maximum) for 950.
# Both are exercised, so every build runs at least one of them on the two-pass
# path and the other one on its own composite-key path, whichever it is.
TWO_PASS_NODES = (5000, 4_000_000_000)


def check(condition, message):
    """Fail a case without using the ``assert`` statement (code-check rule)."""
    if not condition:
        raise AssertionError(message)


def edges_of(rows, cols):
    return torch.tensor([rows, cols], dtype=torch.int64)


def run(npu_api, edge_index, num_nodes=None, by_row=True):
    """Call the operator on NPU and bring the sorted edge list to the host."""
    if num_nodes is None:
        result = npu_api.sort_edge_index(edge_index, sort_by_row=by_row)
    else:
        result = npu_api.sort_edge_index(edge_index, num_nodes=num_nodes,
                                         sort_by_row=by_row)
    return result.cpu()


def random_graph(nodes, edges, seed):
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, nodes, (edges,), generator=generator).tolist()
    cols = torch.randint(0, nodes, (edges,), generator=generator).tolist()
    return rows, cols


# (rows, columns, num_nodes, sort_by_row). Every id stays inside [0, num_nodes).
EDGE_CASES = [
    pytest.param([0], [0], 1, True, id="single_edge"),
    pytest.param([0, 1], [1, 0], 2, True, id="two_edges"),
    pytest.param([2, 1, 1, 0], [1, 2, 0, 1], 3, True, id="pyg_example"),
    pytest.param([2, 1, 1, 0], [1, 2, 0, 1], 3, False,
                 id="pyg_example_by_col"),
    pytest.param([1, 1, 1], [2, 0, 1], 3, True, id="single_row"),
    pytest.param([2, 0, 1], [0, 0, 0], 3, True, id="single_column"),
    pytest.param([5, 5, 5, 5], [4, 0, 2, 2], 6, True, id="repeated_tail"),
    pytest.param([0, 0, 3, 3], [9, 1, 1, 9], 10, True, id="wide_rectangle"),
    pytest.param([0, 1, 2, 3], [0, 1, 2, 3], 4, True, id="diagonal"),
    pytest.param([3, 2, 1, 0], [0, 1, 2, 3], 4, True, id="anti_diagonal"),
    pytest.param([], [], 0, True, id="empty_graph"),
    pytest.param([], [], 7, True, id="no_edges_many_nodes"),
    pytest.param([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, True, id="same_edge"),
]


class TestSortEdgeIndex:
    """``sort_edge_index`` against its composite-key CPU reference."""

    @pytest.mark.npu
    @pytest.mark.parametrize("rows,cols,num_nodes,by_row", EDGE_CASES)
    def test_matches_reference(self, npu_api, rows, cols, num_nodes, by_row):
        sorted_edges = run(npu_api, edges_of(rows, cols).npu(), num_nodes,
                           by_row)

        expected, order = _golden.reference(edges_of(rows, cols), num_nodes,
                                            by_row)
        check(torch.equal(sorted_edges, expected),
              f"sorted edges {sorted_edges.tolist()} != {expected.tolist()}")
        check(torch.equal(sorted_edges, edges_of(rows, cols)[:, order]),
              "the sorted edges differ from the reference permutation")

    @pytest.mark.npu
    def test_sort_by_column(self, npu_api):
        rows = [2, 1, 0, 3, 1]
        cols = [1, 2, 0, 2, 0]

        sorted_edges = run(npu_api, edges_of(rows, cols).npu(), 4, by_row=False)

        expected, _ = _golden.reference(edges_of(rows, cols), 4,
                                        sort_by_row=False)
        check(torch.equal(sorted_edges, expected),
              f"sorted edges {sorted_edges.tolist()} != {expected.tolist()}")
        check(sorted_edges[1].tolist() == sorted(sorted_edges[1].tolist()),
              "the destination ids must be nondecreasing")

    @pytest.mark.npu
    def test_medium_random_graph(self, npu_api):
        rows, cols = random_graph(1000, 20000, 1)

        sorted_edges = run(npu_api, edges_of(rows, cols).npu(), 1000)

        expected, _ = _golden.reference(edges_of(rows, cols), 1000)
        check(torch.equal(sorted_edges, expected), "the sorted edges differ")

    @pytest.mark.npu
    def test_two_pass_path(self, npu_api):
        # A square that leaves the exact sort range, so the operator sorts by
        # column and then stably by row instead of building a composite key over
        # the node count. Both counts are checked, which covers the float32 sort
        # of A2/A3 and the int64 sort of 950 on whichever build runs them.
        rows, cols = random_graph(4000, 60000, 2)
        edge_index = edges_of(rows, cols).npu()

        for nodes in TWO_PASS_NODES:
            sorted_edges = run(npu_api, edge_index, nodes)
            expected, _ = _golden.reference(edges_of(rows, cols), nodes)
            check(torch.equal(sorted_edges, expected),
                  f"the sorted edges differ at num_nodes={nodes}")

    @pytest.mark.npu
    def test_both_paths_agree(self, npu_api):
        # A larger node count only pads the keys, so the two-pass order must match
        # the composite-key order of the same graph, on both two-pass counts.
        rows, cols = random_graph(700, 30000, 3)
        edge_index = edges_of(rows, cols).npu()

        composite = run(npu_api, edge_index, 3000)
        for nodes in TWO_PASS_NODES:
            two_pass = run(npu_api, edge_index, nodes)
            check(torch.equal(composite, two_pass),
                  f"the composite and two-pass orders differ at num_nodes={nodes}")

    @pytest.mark.npu
    def test_duplicate_edges_keep_input_order(self, npu_api):
        rows = [1, 0, 1, 0, 1]
        cols = [1, 0, 1, 0, 1]

        sorted_edges = run(npu_api, edges_of(rows, cols).npu(), 2)

        expected, _ = _golden.reference(edges_of(rows, cols), 2)
        check(torch.equal(sorted_edges, expected),
              "duplicate edges must keep their input order")

    @pytest.mark.npu
    def test_inferred_num_nodes(self, npu_api):
        rows, cols = random_graph(800, 25000, 4)
        nodes = max(max(rows), max(cols)) + 1

        sorted_edges = run(npu_api, edges_of(rows, cols).npu())

        expected, _ = _golden.reference(edges_of(rows, cols), nodes)
        check(torch.equal(sorted_edges, expected),
              "the inferred node count must be max + 1")

    @pytest.mark.npu
    def test_edge_attr_tensor(self, npu_api):
        rows = [2, 1, 1, 0]
        cols = [1, 2, 0, 1]
        generator = torch.Generator().manual_seed(5)
        attr = torch.randint(0, 100, (4, 3), generator=generator)

        sorted_edges, moved = npu_api.sort_edge_index(
            edges_of(rows, cols).npu(), attr.npu(), num_nodes=3)

        expected, order = _golden.reference(edges_of(rows, cols), 3)
        check(torch.equal(sorted_edges.cpu(), expected), "the edges differ")
        check(torch.equal(moved.cpu(), attr[order]), "the attribute differs")

    @pytest.mark.npu
    def test_edge_attr_list(self, npu_api):
        rows = [2, 1, 1, 0]
        cols = [1, 2, 0, 1]
        first = torch.tensor([4.0, 3.0, 2.0, 1.0])
        second = torch.tensor([7.0, 5.0, 6.0, 8.0]).reshape(4, 1)

        sorted_edges, moved = npu_api.sort_edge_index(
            edges_of(rows, cols).npu(), [first.npu(), second.npu()],
            num_nodes=3)

        expected, order = _golden.reference(edges_of(rows, cols), 3)
        check(torch.equal(sorted_edges.cpu(), expected), "the edges differ")
        check(len(moved) == 2, "every attribute must be returned")
        check(torch.equal(moved[0].cpu(), first[order]), "the first attr differs")
        check(torch.equal(moved[1].cpu(), second[order]),
              "the second attr differs")

    @pytest.mark.npu
    def test_edge_attr_none(self, npu_api):
        edge_index = edges_of([1, 0], [0, 1]).npu()

        sorted_edges, moved = npu_api.sort_edge_index(edge_index, None,
                                                      num_nodes=2)

        check(moved is None, "an explicit None must stay None")
        check(sorted_edges[0].tolist() == [0, 1], "the edges are not sorted")

    @pytest.mark.npu
    def test_missing_attr_returns_edges(self, npu_api):
        edge_index = edges_of([1, 0], [0, 1]).npu()

        sorted_edges = npu_api.sort_edge_index(edge_index, num_nodes=2)

        check(isinstance(sorted_edges, torch.Tensor),
              "an omitted attribute must return the edge list only")
        check(sorted_edges.shape == (2, 2), "the edge list changed shape")

    @pytest.mark.npu
    def test_empty_graph(self, npu_api):
        sorted_edges = npu_api.sort_edge_index(
            torch.empty((2, 0), dtype=torch.int64).npu(), num_nodes=5)

        check(sorted_edges.shape == (2, 0), "an empty graph must stay empty")

    @pytest.mark.npu
    def test_output_metadata(self, npu_api):
        edge_index = edges_of([2, 0, 1], [0, 1, 1]).npu()

        sorted_edges = npu_api.sort_edge_index(edge_index, num_nodes=3)

        check(sorted_edges.dtype == torch.int64,
              f"dtype {sorted_edges.dtype} != int64")
        check(sorted_edges.shape == (2, 3), "the edge list changed shape")
        check(sorted_edges.device.type == "npu",
              "the output must stay on the NPU")

    @pytest.mark.npu
    def test_large_random_graph(self, npu_api):
        rows, cols = random_graph(50000, 200000, 6)

        sorted_edges = run(npu_api, edges_of(rows, cols).npu(), 50000)

        expected, _ = _golden.reference(edges_of(rows, cols), 50000)
        check(torch.equal(sorted_edges, expected), "the sorted edges differ")

    @pytest.mark.npu
    def test_non_contiguous(self, npu_api):
        rows, cols = random_graph(300, 4000, 7)
        source = edges_of(rows, cols)
        backing = torch.zeros((2, 2 * 4000), dtype=torch.int64)
        backing[:, ::2] = source
        view = backing[:, ::2]
        check(not view.is_contiguous(), "the sample must be non-contiguous")

        sorted_edges = run(npu_api, view.npu(), 300)

        expected, _ = _golden.reference(source, 300)
        check(torch.equal(sorted_edges, expected),
              "a non-contiguous edge list must sort the same")

    @pytest.mark.npu
    def test_deterministic(self, npu_api):
        rows, cols = random_graph(2000, 40000, 8)
        edge_index = edges_of(rows, cols).npu()

        first = run(npu_api, edge_index, 2000)
        for _ in range(4):
            again = run(npu_api, edge_index, 2000)
            check(torch.equal(again, first), "the sorted edges vary")


def _invalid_cases():
    """Argument tuples the public entry point must reject, with their error."""
    edge_index = edges_of([1, 0], [0, 1])
    return {
        "not_a_tensor": ([1, 0], None, 2, TypeError),
        "shape": (torch.zeros((3, 2), dtype=torch.int64), None, 2, ValueError),
        "rank": (torch.zeros((1, 2, 2), dtype=torch.int64), None, 2, ValueError),
        "integer_dtype": (edge_index.to(torch.int32), None, 2, TypeError),
        "cpu_edges": (edge_index, None, 2, RuntimeError),
        "negative_num_nodes": (edge_index, None, -1, ValueError),
        "float_num_nodes": (edge_index, None, 2.5, TypeError),
        "attr_rows": (edge_index, torch.ones(3), 2, ValueError),
        "attr_rank": (edge_index, torch.tensor(1.0), 2, ValueError),
        "attr_device": (edge_index, torch.ones(2), 2, ValueError),
        "attr_not_a_tensor": (edge_index, [1.0, 2.0], 2, TypeError),
    }


@pytest.mark.npu
@pytest.mark.parametrize("name", sorted(_invalid_cases()))
def test_invalid_arguments(npu_api, name):
    edge_index, attr, num_nodes, error = _invalid_cases()[name]
    if name != "cpu_edges":
        edge_index = edge_index.npu() if isinstance(edge_index, torch.Tensor) \
            else edge_index
    if isinstance(attr, torch.Tensor) and name != "attr_device":
        attr = attr.npu()
    if attr is None:
        with pytest.raises(error):
            npu_api.sort_edge_index(edge_index, num_nodes=num_nodes)
    else:
        with pytest.raises(error):
            npu_api.sort_edge_index(edge_index, attr, num_nodes=num_nodes)

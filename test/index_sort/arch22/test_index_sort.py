# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional tests of ``index_sort`` for the arch22 and arch35 builds.

Both builds sort stably and produce the same result for the same input; they only
pick a different device path. On A2/A3 (``dav-2201``) keys below ``2 ** 24`` are
sorted as float32, which is exact for integers, and larger keys fall back to
int64; on 950 (``dav-3510``) the int64 sort is a native fast kernel, so the keys
are always sorted in int64. The boundary cases around ``2 ** 24`` and the large
random case therefore cover the float32 path, the fallback and the plain int64
path, whichever build runs them, and the result never depends on the value range.
"""

import importlib.util
import os

import pytest
import torch


def _load_golden():
    """Import the ``golden.py`` next to this module under its own name."""
    directory = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_golden_index_sort_arch22", os.path.join(directory, "golden.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_golden = _load_golden()

FLOAT32_EXACT_MAX = 1 << 24


def check(condition, message):
    """Fail a case without using the ``assert`` statement (code-check rule)."""
    if not condition:
        raise AssertionError(message)


def run(npu_api, keys, max_value=None):
    """Call the operator on NPU and bring both outputs back to the host."""
    tensor = torch.tensor(keys, dtype=torch.int64).npu()
    if max_value is None:
        values, order = npu_api.index_sort(tensor)
    else:
        values, order = npu_api.index_sort(tensor, max_value)
    return values.cpu(), order.cpu()


def random_keys(count, bound, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, bound, (count,), generator=generator).tolist()


# (keys, max_value). Every key stays inside [0, max_value) when it is given.
KEYS_CASES = [
    pytest.param([], None, id="empty"),
    pytest.param([0], 1, id="single_zero"),
    pytest.param([9], 16, id="single_value"),
    pytest.param([3, 3, 3, 3], 4, id="all_equal"),
    pytest.param([0, 1, 2, 3, 4], 5, id="ascending"),
    pytest.param([4, 3, 2, 1, 0], 5, id="descending"),
    pytest.param([2, 0, 2, 1, 0], 3, id="duplicates"),
    pytest.param([0, 0, 0, 0], 1, id="single_bucket"),
    pytest.param([5, 0, 9, 3], 10, id="sparse_buckets"),
    pytest.param([1] * 64, 2, id="long_ties"),
    pytest.param([7, 7, 6, 6, 5, 5, 4, 4], 8, id="descending_pairs"),
    pytest.param(list(range(500))[::-1], 500, id="five_hundred"),
    pytest.param([0, 2 ** 24 - 1, 0, 2 ** 24 - 2], 2 ** 24,
                 id="float32_upper_edge"),
    pytest.param([2 ** 24, 0, 2 ** 24 + 3, 2 ** 24 - 1], 2 ** 24 + 4,
                 id="above_float32_range"),
]


class TestIndexSort:
    """``index_sort`` against its stable-sort CPU reference."""

    @pytest.mark.npu
    @pytest.mark.parametrize("keys,max_value", KEYS_CASES)
    def test_matches_reference(self, npu_api, keys, max_value):
        values, order = run(npu_api, keys, max_value)
        expected_values, expected_order = _golden.reference(keys)

        check(torch.equal(values, expected_values),
              f"sorted keys {values.tolist()} != {expected_values.tolist()}")
        check(torch.equal(order, expected_order),
              f"perm {order.tolist()} != {expected_order.tolist()}")

    @pytest.mark.npu
    def test_large_random(self, npu_api):
        keys = random_keys(200000, 100000, 1)

        values, order = run(npu_api, keys, 100000)
        expected_values, expected_order = _golden.reference(keys)

        check(torch.equal(order, expected_order), "the permutation differs")
        check(torch.equal(values, expected_values), "the sorted keys differ")

    @pytest.mark.npu
    def test_sorted_values_follow_the_permutation(self, npu_api):
        keys = torch.tensor(random_keys(5000, 300, 2), dtype=torch.int64)
        tensor = keys.npu()

        values, order = npu_api.index_sort(tensor, 300)

        check(torch.equal(values.cpu(), keys[order.cpu()]),
              "the sorted keys must be inputs[perm]")
        check(torch.equal(values.cpu(), torch.sort(keys).values),
              "the sorted keys are not sorted")

    @pytest.mark.npu
    def test_equal_keys_keep_input_order(self, npu_api):
        # The permutation of equal keys is the ascending input position, which is
        # what makes the result comparable bit by bit across runs and devices.
        keys = torch.tensor([4, 1, 4, 0, 4, 1, 4, 4], dtype=torch.int64)

        _, order = npu_api.index_sort(keys.npu(), 5)

        expected = torch.argsort(keys, stable=True)
        check(torch.equal(order.cpu(), expected),
              f"perm {order.tolist()} != {expected.tolist()}")
        check(torch.equal(order.cpu(), torch.tensor([3, 1, 5, 0, 2, 4, 6, 7])),
              "the equal keys must keep their input order")

    @pytest.mark.npu
    def test_inferred_max_value(self, npu_api):
        keys = random_keys(20000, 4096, 3)

        values, order = run(npu_api, keys)
        expected_values, expected_order = _golden.reference(keys)

        check(torch.equal(values, expected_values), "the sorted keys differ")
        check(torch.equal(order, expected_order), "the permutation differs")

    @pytest.mark.npu
    def test_inferred_max_value_of_empty_input(self, npu_api):
        values, order = run(npu_api, [])

        check(values.numel() == 0 and order.numel() == 0,
              "an empty vector must produce empty outputs")

    @pytest.mark.npu
    def test_bound_above_the_float32_range(self, npu_api):
        # A bound that does not prove the 24-bit range makes the operator read the
        # real maximum once; small keys must still take the exact float32 path and
        # produce the same order.
        keys = random_keys(50000, 2000, 4)

        values, order = run(npu_api, keys, 1000000 ** 2)
        expected_values, expected_order = _golden.reference(keys)

        check(torch.equal(values, expected_values), "the sorted keys differ")
        check(torch.equal(order, expected_order), "the permutation differs")

    @pytest.mark.npu
    def test_keys_above_the_float32_range(self, npu_api):
        # Above 2 ** 24 the values are no longer exact in float32, so the operator
        # must fall back to the int64 sort.
        keys = [2 ** 25 + 7, 2 ** 25 + 3, 0, 2 ** 25 + 7, 5]

        values, order = run(npu_api, keys, 2 ** 25 + 8)
        expected_values, expected_order = _golden.reference(keys)

        check(torch.equal(values, expected_values), "the sorted keys differ")
        check(torch.equal(order, expected_order), "the permutation differs")

    @pytest.mark.npu
    def test_exact_float32_boundary(self, npu_api):
        keys = [FLOAT32_EXACT_MAX - 1, 0, FLOAT32_EXACT_MAX - 2, 1]

        values, order = run(npu_api, keys, FLOAT32_EXACT_MAX)
        expected_values, expected_order = _golden.reference(keys)

        check(torch.equal(values, expected_values), "the sorted keys differ")
        check(torch.equal(order, expected_order), "the permutation differs")

    @pytest.mark.npu
    def test_deterministic(self, npu_api):
        keys = random_keys(50000, 7000, 5)
        tensor = torch.tensor(keys, dtype=torch.int64).npu()

        first_values, first_order = npu_api.index_sort(tensor, 7000)
        for _ in range(4):
            values, order = npu_api.index_sort(tensor, 7000)
            check(torch.equal(values, first_values), "the sorted keys vary")
            check(torch.equal(order, first_order), "the permutation varies")

    @pytest.mark.npu
    def test_output_metadata(self, npu_api):
        keys = torch.tensor([3, 0, 1], dtype=torch.int64).npu()

        values, order = npu_api.index_sort(keys, 4)

        check(values.dtype == torch.int64, f"dtype {values.dtype} != int64")
        check(order.dtype == torch.int64, f"dtype {order.dtype} != int64")
        check(values.dim() == 1 and order.dim() == 1, "both outputs must be 1-D")
        check(values.numel() == 3 and order.numel() == 3,
              "both outputs must hold one element per key")
        check(values.device.type == "npu" and order.device.type == "npu",
              "the outputs must stay on the NPU")

    @pytest.mark.npu
    def test_non_contiguous(self, npu_api):
        base = torch.tensor(random_keys(4000, 500, 6), dtype=torch.int64)
        strided = base[::2]
        check(not strided.is_contiguous(), "the sample must be non-contiguous")

        values, order = npu_api.index_sort(strided.npu(), 500)
        expected_values, expected_order = _golden.reference(strided)

        check(torch.equal(values.cpu(), expected_values), "the keys differ")
        check(torch.equal(order.cpu(), expected_order), "the permutation differs")


def _invalid_cases():
    """Argument tuples the public entry point must reject, with their error."""
    keys = torch.tensor([1, 0], dtype=torch.int64)
    return {
        "not_a_tensor": ([1, 0], None, TypeError),
        "integer_dtype": (keys.to(torch.int32), None, TypeError),
        "rank": (keys.view(1, 2), None, ValueError),
        "cpu_keys": (keys, None, RuntimeError),
        "negative_bound": (keys, -1, ValueError),
        "float_bound": (keys, 2.5, TypeError),
        "zero_bound": (keys, 0, RuntimeError),
    }


@pytest.mark.npu
@pytest.mark.parametrize("name", sorted(_invalid_cases()))
def test_invalid_arguments(npu_api, name):
    keys, max_value, error = _invalid_cases()[name]
    if name != "cpu_keys":
        keys = keys.npu() if isinstance(keys, torch.Tensor) else keys
    with pytest.raises(error):
        if max_value is None:
            npu_api.index_sort(keys)
        else:
            npu_api.index_sort(keys, max_value)

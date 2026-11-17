# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional suite shared by the grouped-softmax operators.

``scatter_softmax`` and ``scatter_log_softmax`` share the device kernel core and
accept the same arguments, so every functional case is written once here. An
operator test module binds the suite by subclassing ``GroupedSoftmaxSuite`` and
setting ``api_name`` and ``log``; the log-domain cases that have no softmax
counterpart stay in the log-softmax module.

The module keeps the ``test_`` name prefix even though it only defines a base
class: that prefix is what CodeCheck uses to treat a file as test code, and
pytest collects nothing from it because the class is not named ``Test*``.
"""

import importlib.util
import math
import os

import pytest
import torch

from common.grouped_softmax_cases import (CASES, DTYPES, INFINITY_VALUES,
                                          INVALID_CASES, LONG_GROUP_ROWS, SEED,
                                          invalid_inputs, reference_gradient,
                                          source, to_npu)
from common.grouped_softmax_golden import assert_close, grouped_reference


def check(condition, message) -> None:
    """Fail a case without using the ``assert`` statement (code-check rule)."""
    if not condition:
        raise AssertionError(message)


def load_sibling_golden(test_file):
    """Import the ``golden.py`` next to a test module under a unique name.

    Every operator directory keeps its own ``golden.py``; loading it by path
    keeps the four copies from colliding in ``sys.modules`` when a whole-tree
    run imports the arch22 modules of both operators.
    """
    directory = os.path.dirname(os.path.abspath(test_file))
    label = "_golden_" + "_".join(directory.split(os.sep)[-2:])
    spec = importlib.util.spec_from_file_location(
        label, os.path.join(directory, "golden.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GroupedSoftmaxSuite:
    """Cases shared by ``scatter_softmax`` and ``scatter_log_softmax``."""

    #: Name of the operator under test on the ``ops_gnn`` module.
    api_name = ""
    #: Whether the operator works in the log domain.
    log = False

    @pytest.fixture(scope="module")
    @classmethod
    def npu_api(cls):
        import torch_npu  # noqa: F401  # Register the NPU device.
        import ops_gnn

        torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))
        torch.manual_seed(SEED)
        return ops_gnn

    def reference(self, src, index, dim=0, dim_size=None):
        """float64 reference; operators override it with their ``golden.py``."""
        return grouped_reference(src, index, dim, dim_size,
                                 torch.log_softmax if self.log else torch.softmax)

    def call(self, api, src, index, dim=0, dim_size=None):
        op = getattr(api, self.api_name)
        return op(src, index, dim=dim, dim_size=dim_size)

    def expected(self, src, index, dim=0, dim_size=None):
        return self.reference(src, index, dim, dim_size)

    @pytest.mark.npu
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("index,shape,dim,dim_size", CASES)
    def test_matches_reference(self, npu_api, dtype, index, shape, dim,
                               dim_size):
        src = source(shape, dtype)

        result = self.call(npu_api, src.npu(),
                           torch.tensor(index, dtype=torch.int64).npu(),
                           dim=dim, dim_size=dim_size)

        check(result.shape == src.shape,
              f"shape {tuple(result.shape)} != {tuple(src.shape)}")
        check(result.dtype == dtype, f"dtype {result.dtype} != {dtype}")
        check(result.device.type == "npu", f"device {result.device}")
        assert_close(result, self.expected(src, index, dim, dim_size), dtype)

    @pytest.mark.npu
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_group_normalization(self, npu_api, dtype):
        """Every group must hold a normalized distribution."""
        src = source((6, 3), dtype)
        index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64).npu()

        result = self.call(npu_api, src.npu(), index, dim_size=3)
        if self.log:
            result = torch.exp(result)

        sums = torch.zeros((3, 3), dtype=dtype, device="npu")
        sums = sums.scatter_reduce(
            0, index.view(6, 1).expand(6, 3), result, reduce="sum",
            include_self=False)
        torch.testing.assert_close(
            sums.cpu().float(), torch.ones(3, 3), rtol=1e-2, atol=1e-2)

    @pytest.mark.npu
    def test_int32_index(self, npu_api):
        src = source((6, 4), torch.float32)
        index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int32)

        result = self.call(npu_api, src.npu(), index.npu(), dim_size=3)

        assert_close(result, self.expected(src, index, 0, 3), torch.float32)

    @pytest.mark.npu
    def test_dim_size_inferred(self, npu_api):
        src = source((5, 2), torch.float32)
        index = torch.tensor([0, 0, 1, 3, 3], dtype=torch.int64)

        result = self.call(npu_api, src.npu(), index.npu())

        check(result.shape == (5, 2), f"shape {tuple(result.shape)} != (5, 2)")
        assert_close(result, self.expected(src, index, 0, 4), torch.float32)

    @pytest.mark.npu
    def test_non_contiguous_source(self, npu_api):
        base = source((6, 6), torch.float32)
        src = base[:, ::2]
        check(not src.is_contiguous(), "the sample must be non-contiguous")
        index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)

        result = self.call(npu_api, src.npu(), index.npu(), dim_size=3)

        assert_close(result, self.expected(src, index, 0, 3), torch.float32)

    @pytest.mark.npu
    def test_large_magnitude(self, npu_api):
        src = torch.tensor(
            [[1000.0, -1000.0], [1000.0, -1000.0], [-1000.0, 1000.0],
             [1000.0, 1000.0]], dtype=torch.float32)
        index = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

        result = self.call(npu_api, src.npu(), index.npu(), dim_size=2)

        check(bool(torch.isfinite(result).all()), f"got {result.flatten().tolist()}")
        assert_close(result, self.expected(src, index, 0, 2), torch.float32)

    @pytest.mark.npu
    def test_negative_infinity(self, npu_api):
        src = torch.tensor(
            [[float("-inf"), 0.0], [0.0, 0.0], [float("-inf"), float("-inf")]],
            dtype=torch.float32)
        index = torch.tensor([0, 0, 0], dtype=torch.int64)

        result = self.call(npu_api, src.npu(), index.npu(), dim_size=1)

        assert_close(result, self.expected(src, index, 0, 1), torch.float32)

    @pytest.mark.npu
    def test_negative_infinity_group_with_finite_element(self, npu_api):
        """A group of ``-inf`` and one finite element keeps the finite value."""
        src = torch.tensor(
            [[float("-inf")], [0.0], [float("-inf")]], dtype=torch.float32)
        index = torch.zeros(3, dtype=torch.int64)

        out = self.call(npu_api, src.npu(), index.npu(), dim_size=1)

        assert_close(out, self.expected(src, index, 0, 1), torch.float32)
        # The single finite element is the whole group: probability 1, log 0.
        expected_value = 0.0 if self.log else 1.0
        check(out[1, 0].item() == pytest.approx(expected_value, abs=1e-6),
              f"got {out[1, 0].item()}")

    @pytest.mark.npu
    @pytest.mark.parametrize("values", INFINITY_VALUES)
    def test_positive_infinity_group_is_all_nan(self, npu_api, values):
        """A group containing ``+inf`` must be all NaN."""
        src = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        index = torch.zeros(len(values), dtype=torch.int64)

        out = self.call(npu_api, src.npu(), index.npu(), dim_size=1)

        check(bool(torch.isnan(out).all()), f"got {out.flatten().tolist()}")
        assert_close(out, self.expected(src, index, 0, 1), torch.float32)

    @pytest.mark.npu
    def test_empty_input(self, npu_api):
        src = torch.empty((0, 3), dtype=torch.float32)
        index = torch.empty((0,), dtype=torch.int64)

        result = self.call(npu_api, src.npu(), index.npu(), dim_size=4)

        check(result.shape == (0, 3), f"shape {tuple(result.shape)}")
        check(bool(torch.equal(result.cpu(), torch.empty((0, 3)))),
              "empty input must return an empty tensor")

    @pytest.mark.npu
    def test_gradient(self, npu_api):
        """Backward on a 1-D source must match the float64 reference."""
        src = source((6,), torch.float32)
        grad_out = source((6,), torch.float32)
        index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)

        npu_src = src.npu().detach().requires_grad_(True)
        result = self.call(npu_api, npu_src, index.npu(), dim_size=3)
        (result * grad_out.npu()).sum().backward()

        ref_src, ref_out = reference_gradient(src, index, 0, log=self.log)
        (ref_out * grad_out.double()).sum().backward()

        check(npu_src.grad is not None, "backward did not fill src.grad")
        torch.testing.assert_close(
            npu_src.grad.cpu().double(), ref_src.grad, equal_nan=True,
            rtol=1e-4, atol=1e-5)

    @pytest.mark.npu
    @pytest.mark.parametrize(
        "dim,shape", [(0, (6, 3)), (1, (2, 6)), (1, (2, 6, 4))])
    def test_gradient_matches_reference(self, npu_api, dim, shape):
        index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)
        src = source(shape, torch.float32)
        grad_out = source(shape, torch.float32)

        npu_src = src.npu().detach().requires_grad_(True)
        result = self.call(npu_api, npu_src, index.npu(), dim=dim, dim_size=3)
        (result * grad_out.npu()).sum().backward()

        ref_src, ref_out = reference_gradient(src, index, dim, log=self.log)
        (ref_out * grad_out.double()).sum().backward()

        check(npu_src.grad is not None, "backward did not fill src.grad")
        torch.testing.assert_close(
            npu_src.grad.cpu().double(), ref_src.grad, equal_nan=True,
            rtol=1e-4, atol=1e-5)

    @pytest.mark.npu
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gradient_matches_reference_low_precision(self, npu_api, dtype):
        """The analytic backward must match the reference in low precision."""
        index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)
        src = source((6, 3), dtype)
        grad_out = source((6, 3), dtype)

        npu_src = src.npu().detach().requires_grad_(True)
        result = self.call(npu_api, npu_src, index.npu(), dim_size=3)
        (result * grad_out.npu()).sum().backward()

        ref_src, ref_out = reference_gradient(src, index, 0, log=self.log)
        (ref_out * grad_out.double()).sum().backward()

        check(npu_src.grad is not None, "backward did not fill src.grad")
        torch.testing.assert_close(
            npu_src.grad.cpu().double(), ref_src.grad, equal_nan=True,
            rtol=4e-2 if dtype is torch.bfloat16 else 2e-2, atol=2e-2)

    @pytest.mark.npu
    def test_backward_rejects_grad_out_on_another_device(self, npu_api):
        """The backward must reject a grad_out that is not on the out device."""
        src = source((6,), torch.float32)
        index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int32).npu()
        out = self.call(npu_api, src.npu(), index, dim_size=3)

        # The extension binds the backward under a private attribute; the
        # negative case has to call it directly instead of going through
        # autograd, which would never pass a grad_out on another device.
        pybind = getattr(npu_api, "_pybind")
        backward = getattr(pybind, self.api_name + "_backward")
        with pytest.raises(RuntimeError, match="same device"):
            backward(out, torch.ones(6), index, 0, 3)

    @pytest.mark.npu
    def test_source_is_not_modified(self, npu_api):
        src = source((6, 2), torch.float32)
        original = src.clone()
        index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)

        self.call(npu_api, src.npu(), index.npu(), dim_size=3)

        check(bool(torch.equal(src, original)), "the forward modified src")

    @pytest.mark.npu
    @pytest.mark.parametrize("case,error", INVALID_CASES)
    def test_invalid_inputs(self, npu_api, case, error):
        src, index, dim, dim_size = invalid_inputs()[case]
        if case != "cpu":
            src = to_npu(src)
            index = to_npu(index)

        with pytest.raises(error):
            self.call(npu_api, src, index, dim=dim, dim_size=dim_size)

    @pytest.mark.npu
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("rows", LONG_GROUP_ROWS)
    def test_large_group_does_not_collapse(self, npu_api, dtype, rows):
        """A long low-precision group must keep its normalization.

        The group sum is accumulated in float32, so it neither overflows the
        fp16 integer range nor loses the normalization: ``sum(exp(out))`` stays
        1 and, in the log domain, ``out`` approaches ``-log(rows)``.
        """
        src = torch.zeros(rows, 1, dtype=dtype)
        index = torch.zeros(rows, dtype=torch.int64)

        out = self.call(npu_api, src.npu(), index.npu(), dim_size=1)

        values = torch.exp(out.float()) if self.log else out.float()
        total = values.sum().item()
        check(abs(total - 1.0) < 0.05, f"group of {rows} collapsed: sum={total}")
        if self.log:
            level = out.float().mean().item()
            check(abs(level + math.log(rows)) < 0.05,
                  f"group of {rows}: level={level}")
        assert_close(out, self.expected(src, index, 0, 1), dtype)

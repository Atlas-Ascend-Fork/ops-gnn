# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Log-domain cases of the grouped-softmax operators.

``scatter_log_softmax`` has two cases that have no softmax counterpart: the
agreement with ``log(scatter_softmax)`` and the all-``-inf`` group. They live
here rather than in the operator module, which only binds the operator name.

``scatter_logsumexp`` reduces its output, so it cannot take a case from the
softmax suite unchanged; its own suite sits in this module as well, because it
is the log-domain member of the same family and shares the case tables, the
device fixture and the argument / gradient cases of ``GroupedFamilySuite``.
"""

import math

import pytest
import torch

from common.grouped_softmax_cases import (CASES, DTYPES, INFINITY_VALUES,
                                          WIDE_FEATURE_ROWS, source)
from common.grouped_softmax_golden import (assert_close,
                                           grouped_logsumexp_reference,
                                           logsumexp_gradient)
from common.test_grouped_softmax_suite import (GroupedFamilySuite,
                                               GroupedSoftmaxSuite, check)


class LogSoftmaxSuite(GroupedSoftmaxSuite):
    """The shared cases plus the two log-domain-only ones."""

    log = True

    @pytest.mark.npu
    def test_matches_log_of_softmax(self, npu_api):
        """The log domain must agree with ``log(scatter_softmax)``."""
        src = source((32, 16), torch.float32)
        index = torch.randint(
            0, 5, (32,), generator=torch.Generator().manual_seed(7)).npu()

        log_out = self.call(npu_api, src.npu(), index, dim_size=5)
        softmax_out = npu_api.scatter_softmax(src.npu(), index, dim_size=5)

        torch.testing.assert_close(
            log_out.cpu(), torch.log(softmax_out).cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.npu
    def test_all_negative_infinity_group_is_nan(self, npu_api):
        """A group with no finite element is all NaN, like ``log_softmax``."""
        src = torch.full((4, 2), float("-inf"), dtype=torch.float32)
        index = torch.zeros(4, dtype=torch.int64)

        result = self.call(npu_api, src.npu(), index.npu(), dim_size=1)

        check(bool(torch.isnan(result).all()),
              f"got {result.flatten().tolist()}")
        assert_close(result, self.expected(src, index, 0, 1), torch.float32)


class LogSumExpSuite(GroupedFamilySuite):
    """Cases of ``scatter_logsumexp``; an operator module binds the API name."""

    @staticmethod
    def reference(src, index, dim=0, dim_size=None):
        """float64 reference; operator modules bind their ``golden.py``."""
        return grouped_logsumexp_reference(src, index, dim, dim_size)

    @staticmethod
    def gradient_reference(src, index, dim, dim_size):
        return logsumexp_gradient(src, index, dim, dim_size)

    @staticmethod
    def call_backward(backward, src, out, index):
        """The backward of this operator also reads ``src`` itself."""
        backward(src.npu(), out, torch.ones(out.numel()), index, 0, 3)

    @staticmethod
    def check_large_group(out, rows):
        """``logsumexp`` of ``rows`` zeros is ``log(rows)``."""
        level = float(out[0, 0].float())
        check(abs(level - math.log(rows)) < 0.05,
              f"group of {rows}: level={level}")

    @pytest.mark.npu
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("index,shape,dim,dim_size", CASES)
    def test_matches_reference(self, npu_api, dtype, index, shape, dim,
                               dim_size):
        src = source(shape, dtype)
        expected = self.expected(src, index, dim, dim_size)

        result = self.call(npu_api, src.npu(),
                           torch.tensor(index, dtype=torch.int64).npu(),
                           dim=dim, dim_size=dim_size)

        check(result.shape == expected.shape,
              f"shape {tuple(result.shape)} != {tuple(expected.shape)}")
        check(result.dtype == dtype, f"dtype {result.dtype} != {dtype}")
        check(result.device.type == "npu", f"device {result.device}")
        assert_close(result, expected, dtype)

    @pytest.mark.npu
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_wide_feature_row(self, npu_api, dtype):
        """A feature row wider than one tile must be reduced in full."""
        index = [0, 0, 1, 2, 2, 2]
        src = source((6, WIDE_FEATURE_ROWS), dtype)

        result = self.call(npu_api, src.npu(),
                           torch.tensor(index, dtype=torch.int64).npu(),
                           dim_size=3)

        assert_close(result, self.expected(src, index, 0, 3), dtype)

    @pytest.mark.npu
    def test_empty_group_is_negative_infinity(self, npu_api):
        """A group without members reduces to ``-inf``."""
        src = source((4, 2), torch.float32)
        index = torch.tensor([0, 0, 2, 2], dtype=torch.int64)

        out = self.call(npu_api, src.npu(), index.npu(), dim_size=4)

        for group in (1, 3):
            check(bool(torch.isinf(out[group]).all()),
                  f"group {group}: got {out[group].tolist()}")
            check(bool((out[group] < 0).all()),
                  f"group {group}: got {out[group].tolist()}")
        assert_close(out, self.expected(src, index, 0, 4), torch.float32)

    @pytest.mark.npu
    def test_no_source_fills_negative_infinity(self, npu_api):
        """A shape with no indexed member still produces one row per group."""
        src = torch.empty((0, 3), dtype=torch.float32)
        index = torch.empty((0,), dtype=torch.int64)

        out = self.call(npu_api, src.npu(), index.npu(), dim_size=4)

        check(out.shape == (4, 3), f"shape {tuple(out.shape)} != (4, 3)")
        check(bool(torch.isinf(out).all()) and bool((out < 0).all()),
              f"got {out.tolist()}")

    @pytest.mark.npu
    def test_single_element_group_keeps_the_value(self, npu_api):
        """The log-sum-exp of a one-element group is that element."""
        src = torch.tensor([[1.5], [-2.5]], dtype=torch.float32)
        index = torch.tensor([0, 1], dtype=torch.int64)

        out = self.call(npu_api, src.npu(), index.npu(), dim_size=2)

        torch.testing.assert_close(
            out.cpu().float(), src, rtol=1e-5, atol=1e-5)

    @pytest.mark.npu
    def test_negative_infinity_group(self, npu_api):
        """A group of ``-inf`` reduces to ``-inf``, not to NaN."""
        src = torch.tensor(
            [[float("-inf"), float("-inf")], [0.0, 1.0]], dtype=torch.float32)
        index = torch.tensor([0, 1], dtype=torch.int64)

        result = self.call(npu_api, src.npu(), index.npu(), dim_size=2)

        check(bool(torch.isinf(result[0]).all()) and
              bool((result[0] < 0).all()), f"got {result[0].tolist()}")
        assert_close(result, self.expected(src, index, 0, 2), torch.float32)

    @pytest.mark.npu
    @pytest.mark.parametrize("values", INFINITY_VALUES)
    def test_positive_infinity_group_is_positive_infinity(self, npu_api, values):
        """A group containing ``+inf`` reduces to ``+inf``."""
        src = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        index = torch.zeros(len(values), dtype=torch.int64)

        out = self.call(npu_api, src.npu(), index.npu(), dim_size=1)

        check(bool(torch.isinf(out).all()) and bool((out > 0).all()),
              f"got {out.flatten().tolist()}")

    @pytest.mark.npu
    def test_nan_group_is_nan(self, npu_api):
        src = torch.tensor(
            [[float("nan")], [0.0], [1.0]], dtype=torch.float32)
        index = torch.zeros(3, dtype=torch.int64)

        out = self.call(npu_api, src.npu(), index.npu(), dim_size=1)

        check(bool(torch.isnan(out).all()), f"got {out.flatten().tolist()}")

    @pytest.mark.npu
    def test_gradient_of_positive_infinity_group(self, npu_api):
        """The ``+inf`` member gets a NaN gradient, its siblings get zero."""
        src = torch.tensor([[float("inf")], [1.0], [2.0]],
                           dtype=torch.float32)
        index = torch.zeros(3, dtype=torch.int64)

        npu_src = src.npu().detach().requires_grad_(True)
        result = self.call(npu_api, npu_src, index.npu(), dim_size=1)
        check(bool(torch.isinf(result).all()), f"got {result.tolist()}")
        (result * torch.ones_like(result)).sum().backward()

        grad = npu_src.grad.cpu()
        check(bool(torch.isnan(grad[0, 0])), f"got {grad.tolist()}")
        check(float(grad[1, 0]) == 0.0 and float(grad[2, 0]) == 0.0,
              f"got {grad.tolist()}")

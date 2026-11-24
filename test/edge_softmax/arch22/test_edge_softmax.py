# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional tests of ``edge_softmax`` for the arch22 build (A2/A3, dav-2201)."""

import pytest
import torch

from common.grouped_softmax_cases import DTYPES, source
from common.grouped_softmax_golden import assert_close
from common.test_grouped_softmax_suite import load_sibling_golden

_golden = load_sibling_golden(__file__)

# A segment holding ``+inf`` normalizes to NaN: the stable softmax subtracts the
# maximum, so ``+inf - +inf`` is NaN and the segment sum inherits it. A segment
# holding NaN propagates it the same way `torch.softmax` does.
INFINITE_CASES = [
    pytest.param([float("inf"), 0.0, 1.0], id="one_inf"),
    pytest.param([float("inf"), float("inf"), 0.0], id="two_inf"),
    pytest.param([float("inf"), -float("inf")], id="inf_and_minus_inf"),
    pytest.param([float("nan"), 0.0, 1.0], id="nan"),
]

# (CSR pointer, source shape). Every pointer is consistent with axis 0 of the
# source: it starts at 0, is nondecreasing and ends at ``src.size(0)``.
CSR_CASES = [
    pytest.param([0, 6], (6,), id="single_segment_1d"),
    pytest.param([0, 3, 6], (6, 4), id="two_segments"),
    pytest.param([0, 2, 2, 5, 5, 6], (6, 3), id="empty_segments"),
    pytest.param([0, 1, 2, 3], (3, 2), id="singleton_segments"),
    pytest.param([0, 4, 40], (40, 1), id="narrow_features"),
    pytest.param([0, 2, 5], (5, 2, 3), id="three_dim"),
    pytest.param([0, 3, 9], (9, 4099), id="wide_features"),
    pytest.param([0, 0, 0], (0, 4), id="all_empty_segments"),
    pytest.param(list(range(0, 3001, 3)), (3000, 2), id="many_small_segments"),
    pytest.param([0, 50000], (50000, 1), id="long_segment"),
]


def check(condition, message):
    """Fail a case without using the ``assert`` statement (code-check rule)."""
    if not condition:
        raise AssertionError(message)


def _case(indptr, shape, dtype):
    """Build one deterministic source tensor and its int32 CSR pointer."""
    return source(shape, dtype), torch.tensor(indptr, dtype=torch.int32)


class TestEdgeSoftmax:
    """``edge_softmax`` against its float64 CSR softmax reference."""

    @pytest.mark.npu
    @pytest.mark.parametrize("indptr,shape", CSR_CASES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_matches_reference(self, npu_api, indptr, shape, dtype):
        src, ptr = _case(indptr, shape, dtype)

        out = npu_api.edge_softmax(src.npu(), ptr.npu())

        assert_close(out, _golden.reference(src, ptr), dtype)

    @pytest.mark.npu
    @pytest.mark.parametrize("indptr,shape", CSR_CASES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_backward_matches_reference(self, npu_api, indptr, shape, dtype):
        src, ptr = _case(indptr, shape, dtype)
        grad = source(shape, dtype)

        src_npu = src.npu().requires_grad_(True)
        npu_api.edge_softmax(src_npu, ptr.npu()).backward(grad.npu())

        if src.numel() == 0:
            # An empty source has no graph of its own either, so only the
            # absence of a grown gradient is checked.
            check(src_npu.grad is None or tuple(src_npu.grad.shape) == shape,
                  "an empty input must not grow a gradient")
            return

        ref_src, ref_out = _golden.gradient(src, ptr)
        ref_out.backward(grad.to(torch.float64))
        assert_close(src_npu.grad, ref_src.grad, dtype)

    @pytest.mark.npu
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_output_metadata(self, npu_api, dtype):
        src, ptr = _case([0, 3, 6], (6, 4), dtype)

        out = npu_api.edge_softmax(src.npu(), ptr.npu())

        check(out.dtype == dtype, f"dtype {out.dtype} != {dtype}")
        check(out.shape == src.shape,
              f"shape {tuple(out.shape)} != {tuple(src.shape)}")
        check(out.device.type == "npu", "the output must stay on the NPU")

    @pytest.mark.npu
    def test_int64_indptr(self, npu_api):
        src, ptr = _case([0, 2, 5], (5, 4), torch.float32)

        out = npu_api.edge_softmax(src.npu(), ptr.to(torch.int64).npu())

        assert_close(out, _golden.reference(src, ptr), torch.float32)

    @pytest.mark.npu
    def test_non_contiguous_source(self, npu_api):
        base = source((6, 6), torch.float32)
        src = base[:, ::2]
        check(not src.is_contiguous(), "the sample must be non-contiguous")
        ptr = torch.tensor([0, 2, 6], dtype=torch.int32)

        out = npu_api.edge_softmax(src.npu(), ptr.npu())

        assert_close(out, _golden.reference(src, ptr), torch.float32)

    @pytest.mark.npu
    def test_singleton_segment_is_one(self, npu_api):
        src = source((4, 3), torch.float32)
        ptr = torch.arange(0, 5, dtype=torch.int32)

        out = npu_api.edge_softmax(src.npu(), ptr.npu()).cpu()

        torch.testing.assert_close(out, torch.ones_like(src), rtol=0.0,
                                   atol=1e-6)

    @pytest.mark.npu
    def test_uniform_segment_sums_to_one(self, npu_api):
        rows = 1000
        src = torch.ones((rows, 1), dtype=torch.float16)
        ptr = torch.tensor([0, 400, rows], dtype=torch.int32)

        out = npu_api.edge_softmax(src.npu(), ptr.npu()).cpu().float()

        for begin, end in ((0, 400), (400, rows)):
            torch.testing.assert_close(out[begin:end].sum(), torch.tensor(1.0),
                                       rtol=5e-3, atol=5e-3)

    @pytest.mark.npu
    def test_long_segment_does_not_collapse(self, npu_api):
        # A segment far longer than the fp16 integer limit: the segment sum is
        # kept in fp32, so the probabilities stay representable instead of
        # underflowing to zero.
        rows = 70000
        src = torch.ones((rows, 1), dtype=torch.float16)
        ptr = torch.tensor([0, rows], dtype=torch.int32)

        out = npu_api.edge_softmax(src.npu(), ptr.npu()).cpu().float()

        check(bool(torch.isfinite(out).all()), "the result must stay finite")
        torch.testing.assert_close(out.sum(), torch.tensor(1.0), rtol=5e-3,
                                   atol=5e-3)
        torch.testing.assert_close(out, torch.full_like(out, 1.0 / rows),
                                   rtol=2e-2, atol=1e-7)

    @pytest.mark.npu
    @pytest.mark.parametrize("values", INFINITE_CASES)
    def test_non_finite_segment_is_nan(self, npu_api, values):
        src = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
        ptr = torch.tensor([0, src.size(0)], dtype=torch.int32)

        out = npu_api.edge_softmax(src.npu(), ptr.npu()).cpu()

        check(bool(torch.isnan(out).all()),
              f"a segment holding a non-finite value must be NaN, got {out}")

    @pytest.mark.npu
    def test_matches_scatter_softmax_composition(self, npu_api):
        # The COO form of the same normalization, i.e. what a user writes when
        # only ``scatter_softmax`` is available: expand the pointer into one row
        # index per edge and reduce by group.
        src, ptr = _case([0, 3, 7, 10], (10, 4), torch.float32)
        # `ptr2ind` reads an int64 pointer on A2/A3, an int32 one on 950.
        index = npu_api.ptr2ind(ptr.to(torch.int64).npu(), src.size(0))

        composed = npu_api.scatter_softmax(src.npu(), index, dim=0, dim_size=3)
        out = npu_api.edge_softmax(src.npu(), ptr.npu())

        assert_close(out, composed.cpu(), torch.float32)


def _invalid_cases():
    """Argument tuples the public entry point must reject, with their error."""
    src = source((4, 2), torch.float32)
    ptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    return {
        "bad_src_dtype": (src.to(torch.int32), ptr, TypeError),
        "bad_indptr_dtype": (src, ptr.to(torch.float32), TypeError),
        "bad_indptr_rank": (src, ptr.view(3, 1), ValueError),
        "no_boundary": (src, torch.tensor([], dtype=torch.int32), ValueError),
        "single_boundary": (
            src, torch.tensor([0], dtype=torch.int32), ValueError),
        "not_starting_at_zero": (
            src, torch.tensor([1, 2, 4], dtype=torch.int32), ValueError),
        "not_nondecreasing": (
            src, torch.tensor([0, 3, 2, 4], dtype=torch.int32), ValueError),
        "last_boundary_mismatch": (
            src, torch.tensor([0, 2, 3], dtype=torch.int32), ValueError),
        "cpu": (src, ptr, RuntimeError),
        "bad_src_type": ([[0.0, 1.0]], ptr, TypeError),
        "bad_indptr_type": (src, [0, 2, 4], TypeError),
    }


@pytest.mark.npu
@pytest.mark.parametrize("name", sorted(_invalid_cases()))
def test_invalid_arguments(npu_api, name):
    src, ptr, error = _invalid_cases()[name]

    if name != "cpu":
        src = src.npu() if isinstance(src, torch.Tensor) else src
        ptr = ptr.npu() if isinstance(ptr, torch.Tensor) else ptr
    with pytest.raises(error):
        npu_api.edge_softmax(src, ptr)

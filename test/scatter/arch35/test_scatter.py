# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import inspect
import itertools
import os
from pathlib import Path
import runpy
import warnings

import pytest
import torch

import ops_gnn
import ops_gnn as torch_scatter
from ops_gnn import (
    scatter,
    scatter_add,
    scatter_max,
    scatter_mean,
    scatter_min,
    scatter_mul,
    scatter_sum,
)
from ops_gnn.scatter import _warned_l2

_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
reduce_selected = _GOLDEN["reduce_selected"]
scatter_cpu = _GOLDEN["scatter_cpu"]


DEVICE_ID = int(os.environ.get("NPU_DEVICE_ID", "0"))
REDUCTIONS = ("sum", "add", "mul", "mean", "min", "max")
DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


@pytest.fixture(scope="module", autouse=True)
def _select_npu_device():
    """Select the requested NPU without making CPU-only tests require one."""
    if hasattr(torch, "npu"):
        torch.npu.set_device(DEVICE_ID)


def _assert_equal(actual, expected):
    if actual.dtype.is_floating_point:
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    else:
        assert torch.equal(actual, expected)


def test_public_api_and_signatures():
    expected = {
        "src": inspect.Parameter.empty,
        "index": inspect.Parameter.empty,
        "dim": -1,
        "out": None,
        "dim_size": None,
    }
    for fn in (
        scatter_sum,
        scatter_add,
        scatter_mul,
        scatter_mean,
        scatter_min,
        scatter_max,
    ):
        assert {name: p.default for name, p in inspect.signature(fn).parameters.items()} == expected

    scatter_defaults = dict(expected)
    scatter_defaults["reduce"] = "sum"
    assert {
        name: p.default for name, p in inspect.signature(scatter).parameters.items()
    } == scatter_defaults
    assert set(torch_scatter.__all__) >= {
        "scatter", "scatter_sum", "scatter_add", "scatter_mul",
        "scatter_mean", "scatter_min", "scatter_max",
    }


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc01_one_dimensional_all_reductions(reduce):
    src = torch.tensor([1, 3, 2, 4, 5, 6], dtype=torch.float32)
    index = torch.tensor([0, 1, 0, 1, 1, 3], dtype=torch.int64)
    values = {
        "sum": [3, 12, 0, 6],
        "add": [3, 12, 0, 6],
        "mul": [2, 60, 1, 6],
        "mean": [1.5, 4, 0, 6],
        "min": [1, 3, 0, 6],
        "max": [2, 5, 0, 6],
    }
    actual = scatter(src, index, reduce=reduce)
    _assert_equal(actual, torch.tensor(values.get(reduce), dtype=src.dtype))

    if reduce == "min":
        out, arg = scatter_min(src, index)
        _assert_equal(out, actual)
        assert torch.equal(arg, torch.tensor([0, 1, 6, 5]))
    if reduce == "max":
        out, arg = scatter_max(src, index)
        _assert_equal(out, actual)
        assert torch.equal(arg, torch.tensor([2, 4, 6, 5]))


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc02_two_dimensional_index_broadcast_dim_zero(reduce):
    src = torch.tensor(
        [[1, 2], [5, 6], [3, 4], [7, 8], [9, 10], [11, 12]],
        dtype=torch.float32,
    )
    index = torch.tensor([0, 1, 0, 1, 1, 3])
    expected = {
        "sum": [[4, 6], [21, 24], [0, 0], [11, 12]],
        "add": [[4, 6], [21, 24], [0, 0], [11, 12]],
        "mul": [[3, 8], [315, 480], [1, 1], [11, 12]],
        "mean": [[2, 3], [7, 8], [0, 0], [11, 12]],
        "min": [[1, 2], [5, 6], [0, 0], [11, 12]],
        "max": [[3, 4], [9, 10], [0, 0], [11, 12]],
    }
    _assert_equal(
        scatter(src, index, dim=0, reduce=reduce),
        torch.tensor(expected.get(reduce), dtype=src.dtype),
    )


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc03_two_dimensional_dim_one(reduce):
    src = torch.tensor(
        [[1, 5, 3, 7, 9, 11], [2, 4, 8, 6, 10, 12]], dtype=torch.int32
    )
    index = torch.tensor([[0, 1, 0, 1, 1, 3], [0, 0, 1, 0, 1, 2]])
    expected = {
        "sum": [[4, 21, 0, 11], [12, 18, 12, 0]],
        "add": [[4, 21, 0, 11], [12, 18, 12, 0]],
        "mul": [[3, 315, 1, 11], [48, 80, 12, 1]],
        "mean": [[2, 7, 0, 11], [4, 9, 12, 0]],
        "min": [[1, 5, 0, 11], [2, 8, 12, 0]],
        "max": [[3, 9, 0, 11], [6, 10, 12, 0]],
    }
    _assert_equal(
        scatter(src, index, dim=1, reduce=reduce),
        torch.tensor(expected.get(reduce), dtype=src.dtype),
    )


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc04_three_dimensional_src_two_dimensional_index(reduce):
    src = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ],
        dtype=torch.float32,
    )
    index = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int64)
    expected = {
        "sum": [[[6, 8], [3, 4], [0, 0]], [[9, 10], [18, 20], [0, 0]]],
        "add": [[[6, 8], [3, 4], [0, 0]], [[9, 10], [18, 20], [0, 0]]],
        "mul": [[[5, 12], [3, 4], [1, 1]], [[9, 10], [77, 96], [1, 1]]],
        "mean": [[[3, 4], [3, 4], [0, 0]], [[9, 10], [9, 10], [0, 0]]],
        "min": [[[1, 2], [3, 4], [0, 0]], [[9, 10], [7, 8], [0, 0]]],
        "max": [[[5, 6], [3, 4], [0, 0]], [[9, 10], [11, 12], [0, 0]]],
    }
    _assert_equal(
        scatter(src, index, dim=1, dim_size=3, reduce=reduce),
        torch.tensor(expected.get(reduce), dtype=src.dtype),
    )


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc05_index_collapsed_along_scatter_dimension(reduce):
    src = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ],
        dtype=torch.float32,
    )
    index = torch.tensor([[[0, 1]], [[1, 0]]], dtype=torch.int64)
    expected = {
        "sum": [[[9, 0], [0, 12]], [[0, 30], [27, 0]]],
        "add": [[[9, 0], [0, 12]], [[0, 30], [27, 0]]],
        "mul": [[[15, 1], [1, 48]], [[1, 960], [693, 1]]],
        # torch_scatter 2.1.2 builds mean counts from index.shape before
        # broadcasting, so an explicitly collapsed scatter dimension counts
        # once and this upstream corner case equals sum.
        "mean": [[[9, 0], [0, 12]], [[0, 30], [27, 0]]],
        "min": [[[1, 0], [0, 2]], [[0, 8], [7, 0]]],
        "max": [[[5, 0], [0, 6]], [[0, 12], [11, 0]]],
    }
    _assert_equal(
        scatter(src, index, dim=1, dim_size=2, reduce=reduce),
        torch.tensor(expected.get(reduce), dtype=src.dtype),
    )


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc06_one_dimensional_index_broadcast_to_four_dimensions(reduce):
    torch.manual_seed(7)
    src = torch.randn(4, 3, 8, 8)
    index = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    actual = scatter(src, index, dim=2, dim_size=8, reduce=reduce)
    assert actual.shape == (4, 3, 8, 8)

    for b, c, w in itertools.product(range(4), range(3), range(8)):
        selected = [src[b, c, h, w] for h in range(8) if index[h].item() == 0]
        expected = reduce_selected(selected, reduce)
        torch.testing.assert_close(actual[b, c, 0, w], expected)

    if reduce == "mean":
        # Upstream scatter_mean counts from the original index shape.  A 1D
        # singleton index at dim > 0 therefore has count=1 even though it
        # broadcasts across the complete scatter dimension.
        singleton_src = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        singleton_index = torch.tensor([0], dtype=torch.int64)
        singleton_expected = torch.tensor([[[6.0], [15.0]]])
        torch.testing.assert_close(
            scatter_mean(singleton_src, singleton_index, dim=2),
            singleton_expected,
        )
        torch.testing.assert_close(
            scatter(singleton_src, singleton_index, dim=2, reduce="mean"),
            singleton_expected,
        )


def test_tc07_provided_out_in_place_semantics():
    src = torch.tensor([2.0, 3.0, 4.0])
    index = torch.tensor([0, 0, 1])

    out = torch.tensor([10.0, 20.0, 30.0])
    assert scatter_sum(src, index, out=out) is out
    assert torch.equal(out, torch.tensor([15.0, 24.0, 30.0]))

    out = torch.tensor([10.0, 20.0, 30.0])
    assert scatter_mul(src, index, out=out) is out
    assert torch.equal(out, torch.tensor([60.0, 80.0, 30.0]))

    out = torch.tensor([10.0, 20.0, 30.0])
    assert scatter_mean(src, index, out=out) is out
    assert torch.equal(out, torch.tensor([7.5, 24.0, 30.0]))

    out = torch.tensor([10.0, 20.0, 30.0])
    result, arg = scatter_min(src, index, out=out)
    assert result is out
    assert torch.equal(out, torch.tensor([2.0, 4.0, 30.0]))
    assert torch.equal(arg, torch.tensor([0, 2, 3]))

    out = torch.tensor([1.0, 20.0, 30.0])
    result, arg = scatter_max(src, index, out=out)
    assert result is out
    assert torch.equal(out, torch.tensor([3.0, 20.0, 30.0]))
    assert torch.equal(arg, torch.tensor([1, 3, 3]))


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc08_non_contiguous_src_index_and_out(reduce):
    src = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).t()
    index = torch.tensor([[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]).t()
    assert not src.is_contiguous()
    assert not index.is_contiguous()

    expected = scatter(src.contiguous(), index.contiguous(), dim=0, reduce=reduce)
    base = torch.empty(expected.shape[::-1], dtype=src.dtype)
    out = base.t()
    assert not out.is_contiguous()
    # A provided out participates in reduction.  Exercise non-contiguous out
    # directly for sum/add and isolate src/index contiguity for other modes.
    if reduce in ("sum", "add"):
        out.zero_()
        actual = scatter(src, index, dim=0, out=out, reduce=reduce)
    else:
        actual = scatter(src, index, dim=0, reduce=reduce)
        out.copy_(actual)
        actual = out
    _assert_equal(actual, expected)


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc09_empty_tensor_with_explicit_dim_size(reduce):
    src = torch.empty((0, 2), dtype=torch.float32)
    index = torch.empty((0,), dtype=torch.int64)
    actual = scatter(src, index, dim=0, dim_size=3, reduce=reduce)
    assert actual.shape == (3, 2)
    assert torch.equal(actual, torch.zeros(3, 2))


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc10_dim_size_larger_than_index_max(reduce):
    src = torch.tensor([1, 2, 3], dtype=torch.int32)
    index = torch.tensor([0, 1, 0])
    actual = scatter(src, index, dim_size=5, reduce=reduce)
    assert actual.shape == (5,)


def test_tc11_high_conflict_sum_and_mean():
    src = torch.arange(1, 10001, dtype=torch.float32)
    index = torch.zeros(src.numel(), dtype=torch.int64)
    torch.testing.assert_close(
        scatter_sum(src, index), src.sum().reshape(1), rtol=1e-4, atol=1e-3
    )
    torch.testing.assert_close(
        scatter_mean(src, index), src.mean().reshape(1), rtol=1e-4, atol=1e-4
    )


def test_tc12_min_max_arg_out_and_later_writer_tie():
    src = torch.tensor([4.0, 2.0, 2.0, 5.0, 5.0])
    index = torch.tensor([0, 0, 0, 1, 1])
    min_out, arg_min = scatter_min(src, index)
    max_out, arg_max = scatter_max(src, index)
    assert torch.equal(min_out, torch.tensor([2.0, 5.0]))
    assert torch.equal(max_out, torch.tensor([4.0, 5.0]))
    assert torch.equal(arg_min, torch.tensor([2, 4]))
    assert torch.equal(arg_max, torch.tensor([0, 4]))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc13_all_documented_dtypes(dtype, reduce):
    src = torch.tensor([1, 2, 3, 4], dtype=dtype)
    index = torch.tensor([0, 0, 1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = scatter(src, index, dim=0, dim_size=3, reduce=reduce)
    assert actual.dtype == dtype
    assert actual.shape == (3,)


def test_tc14_negative_dims_and_integer_floor_mean():
    src = torch.tensor([[-3, -2, 4], [-2, -1, 5]], dtype=torch.int32)
    index = torch.tensor([0, 0, 1])
    result = scatter_mean(src, index, dim=-1)
    assert torch.equal(result, torch.tensor([[-3, 4], [-2, 5]], dtype=torch.int32))


@pytest.mark.parametrize("rank", range(1, 9))
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_rank_1_through_8_with_compact_dim_index(rank, reduce):
    dim = rank // 2
    shape = [1] * rank
    shape[dim] = 3
    src = torch.tensor([1.0, 2.0, 3.0]).reshape(shape)
    index = torch.tensor([0, 1, 0], dtype=torch.int64)
    actual = scatter(src, index, dim=dim, dim_size=2, reduce=reduce)
    expected_values = {
        "sum": [4.0, 2.0],
        "add": [4.0, 2.0],
        "mul": [3.0, 2.0],
        "mean": [2.0, 2.0],
        "min": [1.0, 2.0],
        "max": [3.0, 2.0],
    }
    expected_shape = list(shape)
    expected_shape[dim] = 2
    expected = torch.tensor(expected_values.get(reduce)).reshape(expected_shape)
    _assert_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_tc15_tc16_l2_cpu_fallback(dtype, reduce):
    src = torch.tensor([2, 3, 4, 5], dtype=dtype)
    index = torch.tensor([0, 0, 1, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = scatter(src, index, reduce=reduce)
    assert result.device.type == "cpu"
    assert result.dtype == dtype
    assert not any("CPU fallback" in str(item.message) for item in caught)


def test_invalid_inputs_are_rejected_before_kernel_launch():
    src = torch.tensor([1.0, 2.0])
    with pytest.raises(TypeError, match="int64"):
        scatter(src, torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(IndexError, match="negative"):
        scatter(src, torch.tensor([0, -1]))
    with pytest.raises(IndexError, match="out of bounds"):
        scatter(src, torch.tensor([0, 2]), dim_size=2)
    with pytest.raises(IndexError, match="Dimension out of range"):
        scatter(src, torch.tensor([0, 1]), dim=1)
    with pytest.raises(ValueError, match="reduce"):
        scatter(src, torch.tensor([0, 1]), reduce="median")
    with pytest.raises(TypeError, match="src dtype"):
        scatter(torch.tensor([True]), torch.tensor([0]))

DEVICE_ID = int(os.environ.get("NPU_DEVICE_ID", "0"))
pytestmark = pytest.mark.npu

# Ecology operator precision standard (mixed tolerance):
# https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md
FLOAT_TOLERANCES = {
    torch.float16: {"rtol": 2 ** -9, "atol": 2 ** -9, "max_abs": 1e-1},
    torch.bfloat16: {"rtol": 2 ** -6, "atol": 2 ** -6, "max_abs": 1.0},
    torch.float32: {"rtol": 2 ** -10, "atol": 2 ** -16, "max_abs": 1e-2},
}
REQUIRED_MATCHED_RATIO = 0.99

SUPPORTED_DTYPES = [torch.float16, torch.bfloat16, torch.float32,
                    torch.int8, torch.int16, torch.int32, torch.uint8]

REDUCES = ["sum", "add", "mul", "mean", "min", "max"]

TEST_SHAPES = [(128,), (1024,), (4096,), (8192,),
               (32, 512), (64, 768), (8, 16, 64), (4, 128, 256)]

GENERAL_SHAPES = [(1,), (2,), (4,), (2, 2), (1, 128),
                  (512, 768), (512, 1024), (1024, 768), (1024, 1024)]


def _scatter_fn(reduce):
    return getattr(ops_gnn, f'scatter_{reduce}', None)


def _call_scatter(src, index, reduce, dim=-1):
    fn = _scatter_fn(reduce)
    if fn is None:
        return ops_gnn.scatter(src, index, dim=dim, reduce=reduce)
    return fn(src, index, dim=dim)


def _is_float_dtype(dtype):
    return dtype in (torch.float16, torch.bfloat16, torch.float32)


def _assert_match(result, expected, dtype):
    if _is_float_dtype(dtype):
        actual = result.cpu().float()
        golden = expected.cpu().float()
        tolerance = FLOAT_TOLERANCES[dtype]
        abs_error = (actual - golden).abs()
        allowed = tolerance["atol"] + tolerance["rtol"] * golden.abs()
        matched_ratio = (abs_error <= allowed).float().mean().item()
        max_abs_error = abs_error.max().item() if abs_error.numel() else 0.0
        assert matched_ratio >= REQUIRED_MATCHED_RATIO, (
            f"matched_ratio={matched_ratio:.8f} < {REQUIRED_MATCHED_RATIO:.8f}; "
            f"dtype={dtype}; max_abs_error={max_abs_error:.8g}"
        )
        assert max_abs_error <= tolerance["max_abs"], (
            f"max_abs_error={max_abs_error:.8g} > {tolerance['max_abs']:.8g}; "
            f"dtype={dtype}; matched_ratio={matched_ratio:.8f}"
        )
    else:
        assert torch.equal(result.cpu(), expected)


def _value_only(result):
    return result[0] if isinstance(result, tuple) else result


def _assert_npu_case(result, expected, dtype):
    actual = _value_only(result)
    golden = _value_only(expected)
    assert actual.device.type == "npu"
    _assert_match(actual, golden, dtype)


class TestScatter:
    @staticmethod
    def test_large():
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(512, 768, device='cpu')
        index = torch.randint(0, 1024, (512, 768), device='cpu')
        result = ops_gnn.scatter_sum(src.npu(), index.npu(), dim=1, dim_size=1024)
        expected = scatter_cpu(src, index, dim=1, dim_size=1024, reduce="sum")
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    @staticmethod
    def test_sum_add_equivalent():
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.randint(0, 32, (128,), device='cpu')
        r1 = ops_gnn.scatter_sum(src.npu(), index.npu())
        r2 = ops_gnn.scatter_add(src.npu(), index.npu())
        assert torch.allclose(r1.cpu(), r2.cpu())

    @staticmethod
    def test_dim_size_one():
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.zeros(128, dtype=torch.long)
        result = ops_gnn.scatter_sum(src.npu(), index.npu(), dim_size=1)
        expected = scatter_cpu(src, index, dim_size=1, reduce="sum")
        assert result.shape[0] == 1
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    @staticmethod
    def test_dim_size_large():
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.randint(0, 32, (128,), device='cpu')
        result = ops_gnn.scatter_sum(src.npu(), index.npu(), dim_size=10000)
        expected = scatter_cpu(src, index, dim_size=10000, reduce="sum")
        assert result.shape[0] == 10000
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    @staticmethod
    def test_empty_index():
        torch.npu.set_device(DEVICE_ID)
        src = torch.randn(0, device='cpu')
        index = torch.empty(0, dtype=torch.long)
        result = ops_gnn.scatter_sum(src.npu(), index.npu())
        assert result.numel() == 0

    @staticmethod
    def test_mean_int_floor():
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randint(0, 10, (64,), dtype=torch.int32)
        index = torch.randint(0, 8, (64,))
        result = ops_gnn.scatter_mean(src.npu(), index.npu())
        expected = scatter_cpu(src, index, reduce="mean")
        assert torch.equal(result.cpu(), expected)

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_shapes(self, shape, reduce, dtype):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        dim = -1
        if len(shape) >= 2:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, max(1, shape[dim] // 2), shape)
        else:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, 32, shape)
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=dim)
        if isinstance(result, tuple):
            result = result[0]
        expected = scatter_cpu(src, index, dim=dim, reduce=reduce)
        if isinstance(expected, tuple):
            expected = expected[0]
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)

    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_general_shapes(self, shape, reduce, dtype):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        if len(shape) >= 2:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, max(1, shape[-1] // 2), shape)
        else:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, max(1, shape[0] // 4), shape)
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=-1)
        if isinstance(result, tuple):
            result = result[0]
        expected = scatter_cpu(src, index, dim=-1, reduce=reduce)
        if isinstance(expected, tuple):
            expected = expected[0]
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_dim0(self, reduce):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, 32, device='cpu')
        index = torch.randint(0, 64, (128, 32), device='cpu')
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=0)
        expected = scatter_cpu(src, index, dim=0, reduce=reduce)
        _assert_npu_case(result, expected, torch.float32)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_dim1(self, reduce):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, 32, device='cpu')
        index = torch.randint(0, 64, (128, 32), device='cpu')
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=1)
        expected = scatter_cpu(src, index, dim=1, reduce=reduce)
        _assert_npu_case(result, expected, torch.float32)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_with_out(self, reduce):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(32, 128, device='cpu')
        index = torch.randint(0, 256, (32, 128), device='cpu')
        out_cpu = torch.ones(32, 256, device='cpu')
        expected = scatter_cpu(
            src, index, dim=1, out=out_cpu, dim_size=256, reduce=reduce
        )
        out = out_cpu.npu()
        fn = _scatter_fn(reduce) or ops_gnn.scatter
        kwargs = {'dim': 1, 'out': out}
        if fn == ops_gnn.scatter:
            kwargs['reduce'] = reduce
        result = fn(src.npu(), index.npu(), **kwargs)
        if isinstance(result, tuple):
            assert result[0] is out
            _assert_match(result[0], expected[0], torch.float32)
            assert torch.equal(result[1].cpu(), expected[1])
        else:
            assert result is out
            _assert_match(result, expected, torch.float32)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_high_conflict(self, reduce):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(1024, 64, device='cpu')
        index = torch.zeros(1024, 64, dtype=torch.long)
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=1)
        expected = scatter_cpu(src, index, dim=1, reduce=reduce)
        _assert_npu_case(result, expected, torch.float32)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_sparse_index(self, reduce):
        torch.npu.set_device(DEVICE_ID)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.randint(0, 10000, (128,), device='cpu')
        result = _call_scatter(src.npu(), index.npu(), reduce)
        expected = scatter_cpu(src, index, reduce=reduce)
        _assert_npu_case(result, expected, torch.float32)


@pytest.mark.parametrize(
    "dtype,length",
    [
        (torch.int8, 128),
        (torch.int16, 40000),
        (torch.uint8, 300),
    ],
)
def test_integer_mean_uses_wide_count(dtype, length):
    """Counts must not wrap in the input dtype before integer floor division."""
    torch.npu.set_device(DEVICE_ID)
    src = torch.ones(length, dtype=dtype)
    index = torch.zeros(length, dtype=torch.long)
    actual = ops_gnn.scatter_mean(src.npu(), index.npu())
    expected = scatter_cpu(src, index, reduce="mean")
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("reduce", ["min", "max"])
def test_integer_minmax_value_and_later_writer_arg(dtype, reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([4, 2, 2, 5, 5], dtype=dtype)
    index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    result = _call_scatter(src.npu(), index.npu(), reduce)
    expected = scatter_cpu(src, index, reduce=reduce)
    assert torch.equal(result[0].cpu(), expected[0])
    assert torch.equal(result[1].cpu(), expected[1])


@pytest.mark.parametrize("rank", range(1, 9))
@pytest.mark.parametrize("reduce", REDUCES)
def test_compact_index_rank_1_through_8(rank, reduce):
    """Keep compact 1D index broadcasting on the NPU path for every rank."""
    torch.npu.set_device(DEVICE_ID)
    dim = rank // 2
    shape = [1] * rank
    shape[dim] = 3
    src = torch.tensor([1.0, 2.0, 3.0]).reshape(shape)
    index = torch.tensor([0, 1, 0], dtype=torch.long)
    expected_values = (
        [4.0, 2.0], [4.0, 2.0], [3.0, 2.0],
        [2.0, 2.0], [1.0, 2.0], [3.0, 2.0],
    )[REDUCES.index(reduce)]
    expected_shape = list(shape)
    expected_shape[dim] = 2
    expected = torch.tensor(expected_values).reshape(expected_shape)
    actual = _call_scatter(src.npu(), index.npu(), reduce, dim=dim)
    _assert_match(_value_only(actual), expected, torch.float32)


@pytest.mark.parametrize("reduce", REDUCES)
def test_compact_index_broadcast_to_four_dimensions(reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.arange(1, 2 * 3 * 8 * 4 + 1, dtype=torch.float32).reshape(
        2, 3, 8, 4
    )
    index = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    expanded_index = index.reshape(1, 1, 8, 1).expand_as(src)
    expected = scatter_cpu(
        src, expanded_index, dim=2, dim_size=6, reduce=reduce
    )
    actual = ops_gnn.scatter(
        src.npu(), index.npu(), dim=2, dim_size=6, reduce=reduce
    )
    _assert_npu_case(actual, expected, torch.float32)


def test_compact_singleton_mean_uses_original_index_count():
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32
    )
    index = torch.tensor([0], dtype=torch.long)
    expected = torch.tensor([[[6.0], [15.0]]], dtype=torch.float32)
    actual = ops_gnn.scatter_mean(src.npu(), index.npu(), dim=2)
    _assert_match(actual, expected, torch.float32)


@pytest.mark.parametrize("reduce", REDUCES)
def test_non_contiguous_npu_inputs_and_out(reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).t()
    index = torch.tensor(
        [[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.long
    ).t()
    out_shape = (3, src.size(1))
    initial = {
        "sum": 0.0,
        "add": 0.0,
        "mul": 1.0,
        "mean": 0.0,
        "min": torch.finfo(src.dtype).max,
        "max": torch.finfo(src.dtype).min,
    }[reduce]
    out_cpu = torch.full(tuple(reversed(out_shape)), initial).t()
    expected = scatter_cpu(
        src, index, dim=0, out=out_cpu, dim_size=3, reduce=reduce
    )

    src_npu = src.t().contiguous().npu().t()
    index_npu = index.t().contiguous().npu().t()
    out = torch.full(
        tuple(reversed(out_shape)), initial, dtype=src.dtype, device="npu"
    ).t()
    assert not src_npu.is_contiguous()
    assert not index_npu.is_contiguous()
    assert not out.is_contiguous()
    fn = _scatter_fn(reduce) or ops_gnn.scatter
    kwargs = {"dim": 0, "out": out}
    if fn == ops_gnn.scatter:
        kwargs["reduce"] = reduce
    actual = fn(src_npu, index_npu, **kwargs)
    assert _value_only(actual) is out
    _assert_match(_value_only(actual), _value_only(expected), torch.float32)
    if isinstance(actual, tuple):
        assert torch.equal(actual[1].cpu(), expected[1])


@pytest.mark.parametrize("reduce", REDUCES)
def test_empty_npu_source_with_explicit_dim_size(reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.empty((0, 2), dtype=torch.float32, device="npu")
    index = torch.empty((0,), dtype=torch.long, device="npu")
    actual = ops_gnn.scatter(src, index, dim=0, dim_size=3, reduce=reduce)
    assert actual.shape == (3, 2)
    assert torch.equal(actual.cpu(), torch.zeros(3, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_vector_sum_mean_fast_path(dtype, reduce):
    torch.npu.set_device(DEVICE_ID)
    src = ((torch.arange(128 * 32).reshape(128, 32) % 7) + 1).to(dtype)
    index = torch.arange(128, dtype=torch.long) % 8
    expanded_index = index.reshape(-1, 1).expand_as(src)
    expected = scatter_cpu(
        src, expanded_index, dim=0, dim_size=12, reduce=reduce
    )
    actual = ops_gnn.scatter(
        src.npu(), index.npu(), dim=0, dim_size=12, reduce=reduce
    )
    _assert_match(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", ["min", "max"])
def test_vector_minmax_fast_path_value_and_arg(dtype, reduce):
    torch.npu.set_device(DEVICE_ID)
    src = ((torch.arange(128 * 32).reshape(128, 32) % 17) - 8).to(dtype)
    index = torch.arange(128, dtype=torch.long) % 8
    expanded_index = index.reshape(-1, 1).expand_as(src)
    expected = scatter_cpu(
        src, expanded_index, dim=0, dim_size=12, reduce=reduce
    )
    fn = _scatter_fn(reduce)
    actual = fn(src.npu(), index.npu(), dim=0, dim_size=12)
    _assert_match(actual[0], expected[0], dtype)
    assert torch.equal(actual[1].cpu(), expected[1])


@pytest.mark.parametrize("rows", [512, 513, 769])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", ["min", "max"])
def test_vector_minmax_finalize_aligned_block_boundaries(rows, dtype, reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.arange(1, rows + 1, dtype=torch.float32).reshape(rows, 1)
    src = src.expand(rows, 16).to(dtype).contiguous().npu()
    index = torch.zeros(rows, dtype=torch.int64, device="npu")
    expanded_index = index.reshape(-1, 1).expand_as(src)
    expected = scatter_cpu(
        src.cpu(), expanded_index.cpu(), dim=0, dim_size=1, reduce=reduce
    )
    actual = _scatter_fn(reduce)(src, index, dim=0, dim_size=1)
    _assert_match(actual[0], expected[0], dtype)
    assert torch.equal(actual[1].cpu(), expected[1])


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
@pytest.mark.parametrize("reduce", REDUCES)
def test_l2_npu_fallback_out_identity_value_arg_and_warning(dtype, reduce):
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([2, 3, 4, 5], dtype=dtype)
    index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    initial = torch.tensor([10, 20, 30], dtype=dtype)
    expected = scatter_cpu(
        src, index, out=initial, dim_size=3, reduce=reduce
    )

    _warned_l2.discard(dtype)
    out = initial.npu()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        fn = _scatter_fn(reduce) or ops_gnn.scatter
        kwargs = {"out": out}
        if fn == ops_gnn.scatter:
            kwargs["reduce"] = reduce
        actual = fn(src.npu(), index.npu(), **kwargs)
    fallback_warnings = [
        item for item in caught if "CPU fallback" in str(item.message)
    ]
    assert len(fallback_warnings) == 1
    assert _value_only(actual) is out
    assert torch.equal(_value_only(actual).cpu(), _value_only(expected))
    if isinstance(actual, tuple):
        assert torch.equal(actual[1].cpu(), expected[1])


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_l2_broadcast_mean_npu_fallback(dtype):
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([[2, 4, 6], [3, 6, 9]], dtype=dtype)
    index = torch.tensor([[0], [1]], dtype=torch.long)
    expected = torch.tensor([[12, 0], [0, 18]], dtype=dtype)
    _warned_l2.discard(dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        actual = ops_gnn.scatter_mean(
            src.npu(), index.npu(), dim=1, dim_size=2
        )
    assert len([
        item for item in caught if "CPU fallback" in str(item.message)
    ]) == 1
    assert torch.equal(actual.cpu(), expected)


def test_npu_invalid_metadata_before_kernel_launch():
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([1.0, 2.0], device="npu")
    index = torch.tensor([0, 1], dtype=torch.long, device="npu")
    with pytest.raises(TypeError, match="int64"):
        ops_gnn.scatter(src, index.to(torch.int32))
    with pytest.raises(IndexError, match="Dimension out of range"):
        ops_gnn.scatter(src, index, dim=1)
    with pytest.raises(ValueError, match="non-negative"):
        ops_gnn.scatter(src, index, dim_size=-1)
    with pytest.raises(ValueError, match="reduce"):
        ops_gnn.scatter(src, index, reduce="median")
    with pytest.raises(TypeError, match="out dtype"):
        ops_gnn.scatter(
            src, index, out=torch.empty(2, dtype=torch.float16, device="npu")
        )
    with pytest.raises(ValueError, match="out shape"):
        ops_gnn.scatter(
            src.reshape(1, 2), index, dim=1,
            out=torch.empty((2, 2), device="npu")
        )
    with pytest.raises(IndexError, match="negative"):
        ops_gnn.scatter(
            src, torch.tensor([0, -1], dtype=torch.long, device="npu"),
            dim_size=2
        )
    with pytest.raises(IndexError, match="out of bounds"):
        ops_gnn.scatter(
            src, torch.tensor([0, 2], dtype=torch.long, device="npu"),
            dim_size=2
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        ops_gnn.scatter(src, index, unsupported=True)
    with pytest.raises(TypeError, match="multiple values"):
        ops_gnn.scatter(src, index, -1, None, None, "sum", reduce="sum")
    with pytest.raises(TypeError, match="positional arguments"):
        ops_gnn.scatter(src, index, -1, None, None, "sum", "extra")


def test_public_scatter_positional_option_binding():
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([1.0, 2.0], device="npu")
    index = torch.tensor([0, 1], dtype=torch.long, device="npu")
    actual = ops_gnn.scatter(src, index, -1, None, 3, "sum")
    expected = torch.tensor([1.0, 2.0, 0.0])
    _assert_match(actual, expected, torch.float32)


def test_npu_index_bounds_cache_invalidates_after_inplace_update():
    torch.npu.set_device(DEVICE_ID)
    src = torch.tensor([1.0, 2.0], device="npu")
    index = torch.tensor([0, 1], dtype=torch.long, device="npu")
    ops_gnn.scatter(src, index, dim_size=2)
    index[1] = -1
    with pytest.raises(IndexError, match="negative"):
        ops_gnn.scatter(src, index, dim_size=2)

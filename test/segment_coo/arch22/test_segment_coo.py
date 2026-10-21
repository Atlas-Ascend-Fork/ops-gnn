# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2 COO acceptance: TC-01 through TC-18, plus the reviewed stream regression.

Requires an Ascend A2 NPU. CPU references use only PyTorch.
"""

import os
from pathlib import Path
import runpy
import sys
from typing import NamedTuple, Optional

import pytest
import torch

_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
cpu_reference = _GOLDEN["cpu_reference"]
cpu_reference_arg = _GOLDEN["cpu_reference_arg"]
cpu_csr_reference = _GOLDEN["cpu_csr_reference"]
_metrics = _GOLDEN["precision_metrics"]
assert_close = _GOLDEN["assert_close"]
DTYPES = _GOLDEN["DTYPES"]
REDUCES = _GOLDEN["REDUCES"]
FLOATS = (torch.float16, torch.float32, torch.bfloat16)
INTEGERS = (torch.int8, torch.uint8, torch.int32, torch.int64)
TEST_SHAPES = ((128,), (1024,), (4096,), (8192,), (32, 512), (64, 768), (8, 16, 64), (4, 128, 256))
GENERAL_SHAPES = ((1,), (2,), (4,), (2, 2), (512, 768), (512, 1024), (1024, 768), (1024, 1024))
SEED = 42


class RunOptions(NamedTuple):
    dim_size: Optional[int] = None
    out: Optional[torch.Tensor] = None
    api_name: str = "segment_coo"
    layout: str = "contiguous"
    expected: Optional[torch.Tensor] = None
    expected_arg: Optional[torch.Tensor] = None
    device_inputs: Optional[tuple] = None


def _run(npu_api, src, index, reduce="sum", options=RunOptions()):
    dim_size, out, api_name, layout, expected, expected_arg, device_inputs = options
    if expected is None:
        expected = cpu_reference(src, index, out=out, dim_size=dim_size, reduce=reduce)
    want_arg = api_name in ("segment_min_coo", "segment_max_coo")
    if want_arg and expected_arg is None:
        _, expected_arg = cpu_reference_arg(src, index, out=out, dim_size=dim_size, reduce=reduce)
    source, indices, output = device_inputs or (
        _device_view(src, layout),
        _device_view(index, layout),
        None if out is None else _device_view(out, layout),
    )
    kwargs = {"out": output, "dim_size": dim_size}
    if api_name == "segment_coo":
        kwargs["reduce"] = reduce
    result = getattr(npu_api, api_name)(source, indices, **kwargs)
    assert isinstance(result, tuple) == want_arg
    actual, arguments = result if want_arg else (result, None)
    assert actual.device == source.device and actual.device.type == "npu"
    assert actual.dtype == src.dtype
    if output is not None:
        assert actual is output
    assert_close(actual, expected)
    if want_arg:
        assert arguments.device == source.device
        assert arguments.dtype == torch.int64
        assert_close(arguments, expected_arg)
    assert torch.equal(indices.cpu(), index)
    torch.testing.assert_close(source.cpu(), src, rtol=0, atol=0, equal_nan=True)
    return actual


def _run_supplied_out(npu_api, src, index, initial, reduce):
    options = RunOptions(out=initial, api_name=f"segment_{reduce}_coo", layout="strided")
    return _run(npu_api, src, index, reduce, options)


@pytest.fixture(scope="module")
def npu_api():
    import torch_npu  # noqa: F401  # Register the NPU device.
    import ops_gnn

    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))
    assert torch.npu.get_device_name().startswith(
        "Ascend910B"
    ), "A2 acceptance requires an Ascend A2 NPU"
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    return ops_gnn


def _source(shape, dtype, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=generator).to(dtype)
    high = 2 if dtype in (torch.int8, torch.uint8) else 128
    return torch.randint(0, high, shape, generator=generator).to(dtype)


def _regular_index(length, segments):
    index = torch.arange(segments).repeat_interleave(length // segments)
    if index.numel() < length:
        index = torch.cat((index, torch.full((length - index.numel(),), segments - 1)))
    return index


def _last_index(index, src):
    return index.view(*[1] * (src.ndim - 1), index.numel())


def _device_view(value, layout="contiguous"):
    if layout == "strided":
        shape = list(value.shape)
        shape[-1] = 2 * shape[-1] + 1
        storage = torch.empty(shape, dtype=value.dtype, device="npu")
        view = storage[..., 1::2]
        view.copy_(value.npu())
        if value.numel() > 1:
            assert not view.is_contiguous()
        assert view.storage_offset() > 0
        return view
    if layout == "offset":
        storage = torch.empty(value.numel() + 1, dtype=value.dtype, device="npu")
        view = storage[1:].view(value.shape)
        view.copy_(value.npu())
        return view
    return value.npu()


def _run_view_case(npu_api, dtype, reduce, source_layout="contiguous", index_layout="contiguous"):
    src = _source((37, 33), dtype)
    index = torch.arange(37) // 7
    _run(
        npu_api,
        src,
        index,
        reduce=reduce,
        options=RunOptions(
            dim_size=8,
            api_name=f"segment_{reduce}_coo",
            device_inputs=(_device_view(src, source_layout), _device_view(index, index_layout), None),
        ),
    )


class TestCpuReference:
    """Small executable evidence of CPU forward semantics; no NPU needed."""

    @staticmethod
    def test_without_optional_extension(monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "torch_scatter", None)
        standalone = tmp_path / "golden.py"
        standalone.write_text(Path(__file__).with_name("golden.py").read_text())
        golden = runpy.run_path(str(standalone))
        source = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        expected = torch.tensor([[8.0, 12.0]])
        assert torch.equal(golden["cpu_reference"](source, torch.tensor([0, 0])), expected)
        assert torch.equal(golden["cpu_csr_reference"](source, torch.tensor([0, 2])), expected)

    @staticmethod
    def test_byte_mean_wraps_before_division():
        result = cpu_reference(
            torch.tensor([250, 10], dtype=torch.uint8), torch.tensor([0, 0]), reduce="mean"
        )
        assert result.item() == 2

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_out_initial_and_first_tie(self, dtype):
        source = torch.tensor([2, 2, 3, 3], dtype=dtype)
        index = torch.tensor([0, 0, 2, 2])
        initial = torch.tensor([10, 11, 12, 13], dtype=dtype)
        assert torch.equal(
            cpu_reference(source, index, out=initial), torch.tensor([14, 11, 18, 13], dtype=dtype)
        )
        _, arg = cpu_reference_arg(source, index, reduce="min", dim_size=4)
        assert torch.equal(arg, torch.tensor([0, 4, 2, 4]))

    @pytest.mark.parametrize("dtype", (torch.int8, torch.int32, torch.int64))
    def test_integer_mean_truncates(self, dtype):
        result = cpu_reference(
            torch.tensor([-3, 0], dtype=dtype), torch.tensor([0, 0]), reduce="mean"
        )
        assert result.item() == -1

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "reduce,expected",
        (
            ("sum", [[[8, 12], [24, 28]], [[12, 16], [28, 32]]]),
            ("add", [[[8, 12], [24, 28]], [[12, 16], [28, 32]]]),
            ("mean", [[[4, 6], [12, 14]], [[6, 8], [14, 16]]]),
            ("min", [[[2, 4], [10, 12]], [[4, 6], [12, 14]]]),
            ("max", [[[6, 8], [14, 16]], [[8, 10], [16, 18]]]),
        ),
    )
    def test_broadcast_and_noncontiguous_reference(self, dtype, reduce, expected):
        base = torch.tensor([[2, 4], [6, 8], [10, 12], [14, 16]], dtype=dtype)
        source = torch.stack((base, base + 2), dim=1).transpose(0, 1)
        index = torch.tensor([[0, 0, 2, 2]])
        result = cpu_reference(source, index, dim_size=4, reduce=reduce)
        assert not source.is_contiguous()
        assert result.shape == (2, 4, 2)
        assert torch.equal(result[:, ::2], torch.tensor(expected, dtype=dtype))
        assert torch.equal(result[:, 1::2], torch.zeros((2, 2, 2), dtype=dtype))

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_empty_reference_preserves_initial_out(self, reduce):
        source = torch.empty(0)
        index = torch.empty(0, dtype=torch.int64)
        initial = torch.tensor([11.0, 22.0])
        result = cpu_reference(source, index, out=initial, reduce=reduce)
        assert torch.equal(result, initial)
        assert result.data_ptr() != initial.data_ptr()

    @pytest.mark.parametrize("index", ([1, 0], [-1, 0], [0, 2]))
    def test_invalid_reference_index(self, index):
        with pytest.raises(ValueError):
            cpu_reference(torch.ones(2), torch.tensor(index), dim_size=2)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_unmatched_out_arguments(self, dtype):
        source = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
        index = torch.tensor([0, 2])
        initial = torch.tensor([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=dtype)
        value, arg = cpu_reference_arg(source, index, out=initial, reduce="max")
        expected_arg = torch.tensor([[0, 0], [2, 2], [0, 0], [2, 2]])
        assert torch.equal(value, initial)
        assert torch.equal(arg, expected_arg)

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize(
        "reduce,expected", (("min", [0, 0, 2, 3, 5, 7]), ("max", [0, 1, 1, 3, 5, 7]))
    )
    def test_special_value_argument_carry(self, dtype, reduce, expected):
        source = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 1, float("nan"), -0.0, 0.0], dtype=dtype
        )
        index = torch.tensor([0, 1, 2, 3, 3, 4, 4])
        _, arg = cpu_reference_arg(source, index, dim_size=6, reduce=reduce)
        assert torch.equal(arg, torch.tensor(expected))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_argument_carry_between_prefix_batches(self, dtype):
        source = torch.tensor([[[2], [1]], [[0], [0]]], dtype=dtype)
        initial = torch.tensor([[[0], [0]], [[10], [10]]], dtype=dtype)
        _, arg = cpu_reference_arg(source, torch.tensor([[0, 1]]), out=initial, reduce="max")
        assert torch.equal(arg, torch.tensor([[[0], [1]], [[1], [1]]]))

    @pytest.mark.parametrize("dtype", (torch.int8, torch.uint8))
    def test_narrowed_zero_divisor_is_explicit_zero(self, dtype):
        src = torch.full((256, 3), 3, dtype=dtype)
        result = cpu_reference(src, torch.zeros(256, dtype=torch.int64), reduce="mean")
        assert torch.equal(result, torch.zeros((1, 3), dtype=dtype))

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("reduce", ("min", "max"))
    def test_csr_nan_uses_strict_update(self, dtype, reduce):
        source = torch.tensor([[float("nan")], [1.0]], dtype=dtype)
        result = cpu_csr_reference(source, torch.tensor([0, 2, 2]), reduce)
        assert torch.equal(result, torch.tensor([[1.0], [0.0]], dtype=dtype))


def _check_shape(npu_api, shape, reduce, dtype):
    src = _source(shape, dtype)
    length = shape[-1]
    divisor = 16 if shape in TEST_SHAPES else 32
    index = _last_index(_regular_index(length, max(2, length // divisor)), src)
    _run(npu_api, src, index, reduce=reduce, options=RunOptions(api_name=f"segment_{reduce}_coo"))


class TestPrecisionStandard:
    """Comparator checks for opbase mixed_tolerance_standard.md sections 2.1-2.3."""

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("kind", ("zero", "subnormal", "normal"))
    def test_ulp_spacing(self, dtype, kind):
        info = torch.finfo(dtype)
        value = {"zero": 0.0, "subnormal": info.tiny / 2, "normal": 1.0}[kind]
        expected = info.eps if kind == "normal" else info.tiny * info.eps
        spacing = _GOLDEN["_ulp_at_output_precision"](torch.tensor([value]).double(), dtype)
        assert spacing.item() == expected

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("above", (False, True))
    def test_element_tolerance_boundary(self, dtype, above):
        tolerance = {torch.float16: 2**-10, torch.bfloat16: 2**-7, torch.float32: 2**-13}[dtype]
        actual = torch.tensor([tolerance], dtype=dtype)
        if above:
            actual = torch.nextafter(actual, torch.full_like(actual, float("inf")))
        metrics, _ = _metrics(actual, torch.zeros_like(actual))
        assert metrics["matched_ratio"] == (0.0 if above else 1.0)

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("mismatches", (1, 2))
    def test_matched_ratio_boundary(self, dtype, mismatches):
        actual = torch.zeros(100, dtype=dtype)
        actual[:mismatches] = (
            2 * {torch.float16: 2**-10, torch.bfloat16: 2**-7, torch.float32: 2**-13}[dtype]
        )
        if mismatches == 1:
            assert_close(actual, torch.zeros_like(actual))
        else:
            with pytest.raises(AssertionError):
                assert_close(actual, torch.zeros_like(actual))

    @pytest.mark.parametrize("other_value", (0.0, 1024.0))
    def test_zero_worst_point_hard_limit(self, other_value):
        expected = torch.zeros(100, dtype=torch.bfloat16)
        expected[-1] = other_value
        actual = expected.clone()
        actual[0] = 0.1201171875
        metrics, _ = _metrics(actual, expected)
        assert metrics["matched_ratio"] == 0.99
        assert metrics["hard_limit"] == 0.1
        assert not metrics["hard_limit_pass"]
        with pytest.raises(AssertionError):
            assert_close(actual, expected)

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("above", (False, True))
    def test_ulp_hard_limit_boundary(self, dtype, above):
        expected = torch.full((100,), 4096.0, dtype=dtype)
        actual = expected.clone()
        actual[0] += 32 * (4096 * torch.finfo(dtype).eps)
        if above:
            actual[0] = torch.nextafter(actual[0], torch.tensor(float("inf"), dtype=dtype))
        metrics, _ = _metrics(actual, expected)
        assert metrics["matched_ratio"] >= 0.99
        assert metrics["hard_limit_pass"] != above

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("sign", (-1, 1))
    def test_higher_precision_reference_overflow(self, dtype, sign):
        expected = torch.tensor([sign * torch.finfo(dtype).max * 2], dtype=torch.float64)
        actual = expected.to(dtype)
        assert_close(actual, expected)
        with pytest.raises(AssertionError):
            assert_close(-actual, expected)

    @pytest.mark.parametrize("dtype", FLOATS)
    def test_higher_precision_reference_keeps_finite_error(self, dtype):
        expected = torch.tensor([1.001], dtype=torch.float64)
        actual = expected.to(dtype)
        metrics, _ = _metrics(actual, expected)
        assert metrics["max_abs_error"] == abs(actual.double().item() - 1.001)
        assert_close(actual, expected)

    @pytest.mark.parametrize("dtype", FLOATS)
    def test_special_values_and_empty(self, dtype):
        expected = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=dtype)
        assert_close(expected.clone(), expected)
        with pytest.raises(AssertionError):
            assert_close(torch.zeros_like(expected), expected)
        assert_close(torch.empty(0, dtype=dtype), torch.empty(0, dtype=dtype))


class TestTC01To06Shapes:
    """Standard shapes, dtypes and reductions."""

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_original_shapes(self, npu_api, shape, reduce, dtype):
        _check_shape(npu_api, shape, reduce, dtype)


class TestTC07Out:
    """Supplied out: initial values, layout, aliasing and write-back."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_out_initial_and_strides(self, npu_api, dtype, reduce):
        src = torch.tensor([1, 2, 3, 4], dtype=dtype).view(4, 1).expand(4, 17).contiguous()
        index = torch.tensor([0, 0, 2, 2])
        initial = torch.tensor([10, 20, 30, 40], dtype=dtype).view(4, 1).expand(4, 17).contiguous()
        _run_supplied_out(npu_api, src, index, initial, reduce)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_nonzero_first_segment_with_out(self, npu_api, dtype, reduce):
        src = torch.tensor([2, 4], dtype=dtype).view(2, 1).expand(2, 17).contiguous()
        index = torch.tensor([2, 2])
        initial = torch.tensor([11, 22, 33, 44], dtype=dtype).view(4, 1).expand(4, 17).contiguous()
        expected = cpu_reference(src, index, out=initial, reduce=reduce)
        _run(
            npu_api,
            src,
            index,
            reduce=reduce,
            options=RunOptions(out=initial, expected=expected, api_name=f"segment_{reduce}_coo"),
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_source_out_partial_alias(self, npu_api, dtype, reduce):
        src = torch.tensor([1, 2, 3, 4, 3, 6]).view(6, 1).expand(6, 32).to(dtype).contiguous()
        index = torch.tensor([0, 0, 1, 1, 2, 2])
        initial = src[1:4].clone()
        expected = cpu_reference(src, index, out=initial, reduce=reduce)
        source = src.npu()
        output = source[1:4]
        result = getattr(npu_api, f"segment_{reduce}_coo")(source, index.npu(), out=output)
        actual, arg = result if reduce in ("min", "max") else (result, None)
        assert actual is output and actual.data_ptr() == output.data_ptr()
        metrics, thresholds = _metrics(actual, expected)
        passed = metrics.get(
            "exact",
            metrics.get("matched_ratio", 0) >= 0.99 and metrics.get("hard_limit_pass", False),
        )
        if arg is not None:
            _, expected_arg = cpu_reference_arg(src, index, out=initial, reduce=reduce)
            assert_close(arg, expected_arg)
        assert passed, metrics
        expected_storage = src.clone()
        expected_storage[1:4] = expected
        torch.testing.assert_close(source.cpu(), expected_storage, rtol=0, atol=0, equal_nan=True)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_overlapping_out_storage_rejected(self, npu_api, dtype):
        source = torch.ones((8, 17), dtype=dtype, device="npu")
        index = torch.arange(8, device="npu", dtype=torch.int64) // 2
        output = torch.zeros((4, 1), dtype=dtype, device="npu").expand(4, 17)
        with pytest.raises((RuntimeError, ValueError), match="overlap"):
            npu_api.segment_coo(source, index, out=output)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_supplied_out_controls_size(self, npu_api, dtype, reduce):
        src = torch.tensor([2, 3], dtype=dtype)
        index = torch.tensor([0, 1], dtype=torch.int64)
        initial = torch.tensor([10, 20, 30], dtype=dtype)
        for dim_size in (-1, 0, 1, 4):
            expected = cpu_reference(src, index, out=initial, dim_size=dim_size, reduce=reduce)
            _run(
                npu_api,
                src,
                index,
                reduce=reduce,
                options=RunOptions(
                    out=initial, dim_size=dim_size, expected=expected, layout="strided"
                ),
            )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_out_view_preserves_other_storage(self, npu_api, dtype, reduce):
        src = _source((4, 17), dtype)
        index = torch.tensor([2, 2, 4, 4])
        initial = torch.arange(6 * 17).reshape(6, 17).remainder(13).to(dtype)
        storage = torch.full((6, 34), 17, dtype=dtype, device="npu")
        output = storage[:, 1::2]
        output.copy_(initial.npu())
        result = _run(
            npu_api,
            src,
            index,
            reduce,
            RunOptions(
                out=initial,
                api_name=f"segment_{reduce}_coo",
                device_inputs=(src.npu(), index.npu(), output),
            ),
        )
        assert result is output
        assert torch.equal(storage[:, ::2].cpu(), torch.full((6, 17), 17, dtype=dtype))


class TestTC08NoncontiguousSource:
    """Noncontiguous source and offset views."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("layout", ("strided", "offset"))
    def test_noncontiguous_views(self, npu_api, dtype, reduce, layout):
        _run_view_case(npu_api, dtype, reduce, source_layout=layout)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_transposed_source_reducing_dim_zero(self, npu_api, dtype, reduce):
        base = _source((64, 37), dtype)
        src = base.t()
        index = torch.arange(37) // 7
        source = base.npu().t()
        assert not source.is_contiguous()
        _run(
            npu_api,
            src,
            index,
            reduce=reduce,
            options=RunOptions(
                api_name=f"segment_{reduce}_coo", device_inputs=(source, index.npu(), None)
            ),
        )


class TestTC09Empty:
    """Empty tensors, inferred outputs and supplied out."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize(
        "metadata",
        (((0,), (0,)), ((2, 0, 17), (1, 0)), ((0, 3, 7), (0, 3)), ((2, 7, 0), (1, 7))),
    )
    def test_empty_inputs(self, npu_api, dtype, reduce, metadata):
        shape, index_shape = metadata
        src = torch.empty(shape, dtype=dtype)
        index = torch.zeros(index_shape, dtype=torch.int64)
        _run(
            npu_api,
            src,
            index,
            reduce=reduce,
            options=RunOptions(dim_size=4, api_name=f"segment_{reduce}_coo"),
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_empty_inferred(self, npu_api, dtype, reduce):
        _run(npu_api, torch.empty(0, dtype=dtype), torch.empty(0, dtype=torch.long), reduce=reduce)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_empty_source_supplied_out(self, npu_api, dtype, reduce):
        src = torch.empty((0, 17), dtype=dtype)
        index = torch.empty(0, dtype=torch.long)
        initial = torch.full((3, 17), 7, dtype=dtype)
        _run_supplied_out(npu_api, src, index, initial, reduce)


class TestTC10IndexConstraint:
    """Invalid indices are rejected by Host before kernel launch."""

    @pytest.mark.parametrize(
        "indices,dim_size", (([0, 2, 1, 3], 4), ([-1, 0, 0, 1], 4), ([0, 0, 1, 4], 4))
    )
    def test_host_rejects_invalid_index_before_launch(self, npu_api, indices, dim_size):
        src = torch.ones(4, 3, device="npu")
        index = torch.tensor(indices, device="npu", dtype=torch.int64)
        with pytest.raises((RuntimeError, ValueError)):
            npu_api.segment_coo(src, index, dim_size=dim_size)


class TestTC11Dtypes:
    """Supported types, APIs, numerical boundaries and explicit byte-mean zero divisor."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("specialized", (False, True))
    def test_all_public_interfaces(self, npu_api, dtype, reduce, specialized):
        src = torch.arange(1, 38).reshape(37, 1).expand(37, 17).to(dtype)
        index = torch.arange(37) // 4 * 2
        api_name = f"segment_{reduce}_coo" if specialized else "segment_coo"
        _run(npu_api, src, index, reduce=reduce, options=RunOptions(dim_size=22, api_name=api_name))

    @pytest.mark.parametrize("dtype", INTEGERS)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("channels", (1, 17, 32, 129))
    def test_integer_boundaries(self, npu_api, dtype, reduce, channels):
        limit = torch.iinfo(dtype)
        values = [
            limit.max,
            1,
            limit.min,
            -1 if dtype != torch.uint8 else 255,
            limit.max,
            limit.max,
            limit.min,
            limit.min,
            3,
            0,
            0,
            1,
        ]
        if dtype != torch.uint8:
            values[-4:] = [-3, 0, -1, 0]
        if dtype == torch.int64:
            values += [2**24 + 1, 2**24 + 2, 2**53 + 1, 2**53 + 2, -(2**53 + 1), -(2**53 + 2)]
        src = torch.tensor(values, dtype=dtype)[:, None].expand(-1, channels).contiguous()
        index = torch.arange(len(values)) // 2
        _run(
            npu_api, src, index, reduce=reduce, options=RunOptions(api_name=f"segment_{reduce}_coo")
        )

    @pytest.mark.parametrize("dtype", (torch.int8, torch.uint8))
    @pytest.mark.parametrize("count", (1, 2, 3, 7, 127, 128, 129, 255, 257, 511, 513))
    def test_byte_mean_count_narrowing(self, npu_api, dtype, count):
        src = torch.full((count, 33), 3, dtype=dtype)
        if dtype == torch.int8:
            src[:, 1::2] *= -1
        _run(npu_api, src, torch.zeros(count, dtype=torch.int64), reduce="mean")

    @pytest.mark.parametrize("dtype", (torch.int8, torch.uint8))
    @pytest.mark.parametrize("count", (256, 512, 768, 1024))
    @pytest.mark.parametrize("layout", ("contiguous", "strided"))
    def test_byte_mean_zero_divisor(self, npu_api, dtype, count, layout):
        src = torch.full((count, 33), 3, dtype=dtype)
        if dtype == torch.int8:
            src[:, 1::2] = -3
        index = torch.zeros(count, dtype=torch.int64)
        expected = torch.zeros((2, 33), dtype=dtype)
        _run(
            npu_api,
            src,
            index,
            reduce="mean",
            options=RunOptions(dim_size=2, expected=expected, layout=layout),
        )

    @pytest.mark.parametrize("dtype", (torch.int8, torch.uint8))
    @pytest.mark.parametrize("has_out", (False, True))
    @pytest.mark.parametrize("api_name", ("segment_coo", "segment_mean_coo"))
    def test_byte_mean_zero_divisor_with_out(self, npu_api, dtype, has_out, api_name):
        count = 256
        src = torch.zeros((count, 3), dtype=dtype)
        src[:, 0] = 3
        src[0, 0], src[1, 0] = (68, 67)
        src[0, 1] = -3 if dtype == torch.int8 else 3
        src[:, 2] = torch.iinfo(dtype).max
        index = torch.zeros(count, dtype=torch.int64)
        initial = (
            torch.tensor([[torch.iinfo(dtype).max, 0, 0], [7, 8, 9]], dtype=dtype)
            if has_out
            else None
        )
        expected = initial.clone() if has_out else torch.zeros((2, 3), dtype=dtype)
        expected[0] = 0
        _run(
            npu_api,
            src,
            index,
            reduce="mean",
            options=RunOptions(
                dim_size=2, out=initial, api_name=api_name, layout="strided", expected=expected
            ),
        )

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("reduce", ("sum", "mean", "min", "max"))
    def test_special_values(self, npu_api, dtype, reduce):
        src = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 1.0, float("nan"), -0.0, 0.0], dtype=dtype
        )
        index = torch.tensor([0, 1, 2, 3, 3, 4, 4])
        _run(
            npu_api,
            src,
            index,
            reduce=reduce,
            options=RunOptions(dim_size=6, api_name=f"segment_{reduce}_coo"),
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", ("min", "max"))
    def test_nonempty_extreme_values_cpu_sentinel(self, npu_api, dtype, reduce):
        limits = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
        extreme = limits.max if reduce == "min" else limits.min
        src = torch.tensor([[extreme, 9], [extreme, 9], [5, 7], [5, 7]], dtype=dtype)
        index = torch.tensor([0, 0, 2, 2])
        expected = torch.tensor([[0, 9], [0, 0], [5, 7], [0, 0]], dtype=dtype)
        expected_arg = torch.tensor([[0, 0], [4, 4], [2, 2], [4, 4]], dtype=torch.int64)
        _run(
            npu_api,
            src,
            index,
            reduce=reduce,
            options=RunOptions(
                dim_size=4,
                api_name=f"segment_{reduce}_coo",
                expected=expected,
                expected_arg=expected_arg,
            ),
        )

    @pytest.mark.parametrize("dtype", FLOATS)
    def test_mean_narrows_numerator_before_division(self, npu_api, dtype):
        source = torch.full((2, 3), torch.finfo(dtype).max, dtype=dtype)
        _run(npu_api, source, torch.tensor([0, 0]), "mean")


class TestTC12CsrConsistency:
    """COO and independent CPU CSR value consistency."""

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_consistency_with_cpu_csr(self, npu_api, reduce):
        src = _source((128, 17), torch.float32)
        counts = torch.tensor([16, 24, 40, 48])
        index = torch.arange(4).repeat_interleave(counts)
        indptr = torch.cat((torch.tensor([0]), counts.cumsum(0)))
        expected = cpu_csr_reference(src, indptr, reduce=reduce)
        _run(npu_api, src, index, reduce=reduce, options=RunOptions(expected=expected))


class TestTC13GeneralShapes:
    """General shapes, broadcasts, long segments and input distributions."""

    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_general_shapes(self, npu_api, shape, reduce, dtype):
        _check_shape(npu_api, shape, reduce, dtype)

    @pytest.mark.parametrize("rank", range(1, 9))
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_rank_one_to_eight(self, npu_api, rank, reduce):
        shape = (2,) * (rank - 1) + (17,)
        src = _source(shape, torch.float32)
        index = _last_index(torch.arange(17) // 4, src)
        _run(npu_api, src, index, reduce=reduce, options=RunOptions(dim_size=7))

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_partial_prefix_broadcast(self, npu_api, dtype, reduce):
        src = _source((2, 3, 11, 7), dtype)
        index = torch.tensor(
            [
                [
                    [0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4],
                    [0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 4],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
                ]
            ]
        )
        _run(npu_api, src, index, reduce=reduce, options=RunOptions(dim_size=7))

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("channels", (1, 31, 32, 33, 127, 129))
    def test_unequal_lengths_and_tail_blocks(self, npu_api, dtype, reduce, channels):
        counts = torch.tensor([0, 1, 3, 0, 17, 31, 65, 129, 257, 0])
        index = torch.arange(len(counts)).repeat_interleave(counts)
        src = _source((index.numel(), channels), dtype)
        _run(npu_api, src, index, reduce=reduce, options=RunOptions(dim_size=len(counts)))

    @pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.int32, torch.int64))
    @pytest.mark.parametrize("reduce", ("sum", "mean", "min", "max"))
    @pytest.mark.parametrize("shape", ((2, 32), (16, 32), (32, 128)))
    def test_uniform_batches(self, npu_api, dtype, reduce, shape):
        rows, channels = shape
        segments = 2053  # Enough independent segments to exercise the batched paths.
        src = _source((segments * rows, channels), dtype)
        if dtype.is_floating_point:
            src.clamp_(-5, 5)
        index = torch.arange(segments).repeat_interleave(rows)
        _run(npu_api, src, index, reduce=reduce, options=RunOptions(dim_size=segments))

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("reduce", ("sum", "mean", "min", "max"))
    def test_float_long_and_extreme(self, npu_api, dtype, reduce):
        src = _source((8193, 17), dtype)
        src *= 2 if dtype == torch.float16 else 128
        index = torch.zeros(8193, dtype=torch.long)
        _run(npu_api, src, index, reduce=reduce, options=RunOptions(dim_size=2))

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("distribution", ("uniform", "normal"))
    def test_precision_input_distributions(self, npu_api, dtype, reduce, distribution):
        generator = torch.Generator().manual_seed(SEED)
        if distribution == "uniform":
            src = 10 * torch.rand(511, 31, generator=generator) - 5
        else:
            mu = 10 * torch.rand((), generator=generator) - 5
            sigma = 0.1 + 1.9 * torch.rand((), generator=generator)
            src = mu + sigma * torch.randn(511, 31, generator=generator)
        src = src.to(dtype)
        index = torch.arange(511) // 17
        _run(npu_api, src, index, reduce=reduce)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("layout", ("shared", "partial", "strided"))
    def test_index_reduction_dimension_broadcast(self, npu_api, dtype, reduce, layout):
        src = _source((2, 3, 11, 3), dtype)
        index = torch.tensor([[[2]]]) if layout == "shared" else torch.tensor([[[0], [2], [3]]])
        indices = _device_view(index, "strided") if layout == "strided" else index.npu()
        _run(
            npu_api,
            src,
            index,
            reduce,
            RunOptions(
                dim_size=5,
                api_name=f"segment_{reduce}_coo",
                device_inputs=(src.npu(), indices, None),
            ),
        )

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_broadcast_cache_tracks_source_length(self, npu_api, reduce):
        index = torch.tensor([0], dtype=torch.int64)
        indices = index.npu()
        for length in (3, 17, 1, 0, 9):
            src = torch.arange(length * 3, dtype=torch.float32).reshape(length, 3)
            _run(
                npu_api,
                src,
                index,
                reduce,
                RunOptions(dim_size=1, device_inputs=(src.npu(), indices, None)),
            )


class TestTC14SumAdd:
    """Sum and add equivalence."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_sum_add_equivalent(self, npu_api, dtype):
        src = _source((128, 33), dtype)
        index = torch.arange(8).repeat_interleave(16)
        summed = _run(npu_api, src, index, options=RunOptions(api_name="segment_sum_coo"))
        added = _run(
            npu_api, src, index, reduce="add", options=RunOptions(api_name="segment_add_coo")
        )
        assert torch.equal(summed.cpu(), added.cpu())

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_sum_add_with_initial_out(self, npu_api, dtype):
        source = _source((7, 3), dtype)
        index = torch.tensor([1, 1, 1, 3, 3, 3, 3])
        initial = torch.full((5, 3), 7, dtype=dtype)
        results = [
            _run(
                npu_api,
                source,
                index,
                reduce,
                RunOptions(out=initial, api_name=f"segment_{reduce}_coo"),
            )
            for reduce in ("sum", "add")
        ]
        assert torch.equal(results[0].cpu(), results[1].cpu())


class TestTC15UnsupportedReduce:
    """The unsupported mul reduction raises ValueError."""

    @staticmethod
    def test_reduce_mul_error(npu_api):
        with pytest.raises(ValueError):
            npu_api.segment_coo(
                torch.ones(4, device="npu"),
                torch.zeros(4, dtype=torch.int64, device="npu"),
                reduce="mul",
            )


class TestTC16SingleGroup:
    """Single segment, including cancellation-sensitive inputs."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_single_group(self, npu_api, dtype, reduce):
        src = _source((17, 3), dtype)
        index = torch.zeros(17, dtype=torch.int64)
        _run(
            npu_api,
            src,
            index,
            reduce=reduce,
            options=RunOptions(dim_size=1, api_name=f"segment_{reduce}_coo"),
        )

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("reduce", ("sum", "mean"))
    def test_cancellation_preserves_cpu_order(self, npu_api, dtype, reduce):
        src = torch.tensor([32768.0, 2**-9, -32768.0] + [0.0] * 13, dtype=dtype)
        _run(
            npu_api,
            src,
            torch.zeros(16, dtype=torch.int64),
            reduce=reduce,
            options=RunOptions(dim_size=1),
        )


class TestTC17HolesAndTies:
    """Empty segments, holes and the CPU first-tie convention."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", ("min", "max"))
    def test_first_ties_and_empty_args(self, npu_api, dtype, reduce):
        src = torch.tensor([[3, 2], [1, 5], [1, 5], [4, 4], [4, 4]], dtype=dtype)
        index = torch.tensor([0, 0, 0, 2, 2])
        _run(
            npu_api,
            src,
            index,
            reduce=reduce,
            options=RunOptions(dim_size=4, api_name=f"segment_{reduce}_coo"),
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", ("min", "max"))
    def test_arg_state_across_segments_and_batches(self, npu_api, dtype, reduce):
        src = torch.tensor(
            [[[5, 8], [7, 6], [10, 10], [10, 10]], [[1, 1], [2, 2], [3, 3], [4, 4]]], dtype=dtype
        )
        initial = torch.tensor(
            [[[0, 0], [9, 9], [10, 10], [20, 20]], [[20, 20], [20, 20], [20, 20], [20, 20]]],
            dtype=dtype,
        )
        if reduce == "min":
            src, initial = 20 - src, 20 - initial
        index = torch.tensor([[0, 0, 2, 2]])
        _run(npu_api, src, index, reduce, RunOptions(out=initial, api_name=f"segment_{reduce}_coo"))

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("reduce", ("min", "max"))
    @pytest.mark.parametrize("has_out", (False, True))
    def test_all_nan_segment_carries_argument(self, npu_api, dtype, reduce, has_out):
        src = torch.tensor([2, 7, float("nan"), float("nan")], dtype=dtype)
        index = torch.tensor([0, 0, 1, 1])
        initial = torch.tensor([float("nan"), 10, 20], dtype=dtype) if has_out else None
        _run(
            npu_api,
            src,
            index,
            reduce,
            RunOptions(out=initial, dim_size=3, api_name=f"segment_{reduce}_coo"),
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", ("min", "max"))
    @pytest.mark.parametrize("has_out", (False, True))
    def test_first_tie_across_tiles(self, npu_api, dtype, reduce, has_out):
        src = torch.ones((1031, 65), dtype=dtype)
        src[17] = 7
        src[1024] = 7
        initial = torch.full((2, 65), 7, dtype=dtype) if has_out else None
        if reduce == "min":
            src = 10 - src
            initial = 10 - initial if initial is not None else None
        _run(
            npu_api,
            src,
            torch.zeros(1031, dtype=torch.int64),
            reduce,
            RunOptions(out=initial, dim_size=2, api_name=f"segment_{reduce}_coo"),
        )


class TestTC18NoncontiguousIndex:
    """Strided index tensors."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_noncontiguous_index(self, npu_api, dtype, reduce):
        _run_view_case(npu_api, dtype, reduce, index_layout="strided")


class TestStreamRegression:
    """Review regression: producer/operator/consumer on a nondefault stream."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    def test_nondefault_stream(self, npu_api, dtype, reduce):
        src = _source((73, 33), dtype)
        index = torch.arange(73) // 5
        source, indices = (src.npu(), index.npu())
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            source = source + 1
            actual = npu_api.segment_coo(source, indices, dim_size=17, reduce=reduce)
            consumed = actual.clone()
        torch.npu.current_stream().wait_stream(stream)
        assert consumed.dtype == src.dtype
        expected = cpu_reference(src + 1, index, dim_size=17, reduce=reduce)
        metrics, thresholds = _metrics(consumed, expected)
        passed = metrics.get(
            "exact",
            metrics.get("matched_ratio", 0) >= 0.99 and metrics.get("hard_limit_pass", False),
        )
        assert passed, metrics

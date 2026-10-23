# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""COO API, numerical, boundary and stream contracts, grouped in one functional suite."""

import os
from pathlib import Path
import runpy

import pytest
import torch

import ops_gnn

_REFERENCE = runpy.run_path(Path(__file__).with_name("golden.py"))
segment_coo_reference = _REFERENCE["segment_coo_reference"]
ReferenceOptions = _REFERENCE["ReferenceOptions"]
segment_mean_reference = _REFERENCE["segment_mean_reference"]
reference_function = _REFERENCE["reference_function"]



# Segment coo

SEGMENT_COO_SUPPORTED_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.int32,
    torch.int64,
    torch.uint8,
]


CPU_COMPARISON_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
]


SEGMENT_COO_REDUCES = ["sum", "add", "mean", "min", "max"]


SEGMENT_COO_TEST_SHAPES = [
    (128,),
    (1024,),
    (4096,),
    (8192,),
    (32, 512),
    (64, 768),
    (8, 16, 64),
    (4, 128, 256),
]


SEGMENT_COO_GENERAL_SHAPES = [
    (1,),
    (2,),
    (4,),
    (2, 2),
    (512, 768),
    (512, 1024),
    (1024, 768),
    (1024, 1024),
]


def _index_for_last_dim(index, src):
    """Broadcast a 1-D segment index over all leading source dimensions."""
    if src.dim() == 1:
        return index
    return index.view(*([1] * (src.dim() - 1)), index.numel())


def _assert_result_close(result, expected):
    if result.dtype == torch.float16:
        torch.testing.assert_close(result.cpu(), expected, rtol=2e-2, atol=2e-2)
    elif result.dtype == torch.bfloat16:
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-1, atol=1e-1)
    elif result.is_floating_point():
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
    else:
        assert torch.equal(result.cpu(), expected)


def _integer_input(shape, dtype):
    high = 2 if dtype in (torch.int8, torch.uint8) else 128
    return torch.randint(0, high, shape).to(dtype)


def segment_coo_cpu(src, index, reduce="sum"):
    sc = src.cpu()
    ic = index.cpu().long()
    dim = ic.dim() - 1
    dim_size = int(ic.max().item()) + 1 if ic.numel() > 0 else 0
    out_size = list(sc.shape)
    out_size[dim] = dim_size
    oc = sc.new_zeros(out_size)
    values = sc.movedim(dim, 0)
    output = oc.movedim(dim, 0)
    flat_index = ic.reshape(-1)
    for segment in range(dim_size):
        selected = values[flat_index == segment]
        if selected.numel() == 0:
            continue
        if reduce in ("sum", "add"):
            output[segment] = selected.sum(dim=0)
        elif reduce == "mean":
            output[segment] = selected.float().mean(dim=0).to(sc.dtype)
        elif reduce == "min":
            output[segment] = selected.min(dim=0).values
        elif reduce == "max":
            output[segment] = selected.max(dim=0).values
    return output.movedim(0, dim)


def _check_shape(shape, reduce, dtype, divisor):
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    length = shape[-1] if len(shape) >= 2 else shape[0]
    nseg = max(2, length // divisor)
    if dtype in (torch.int8, torch.int32, torch.int64, torch.uint8):
        src = _integer_input(shape, dtype)
    else:
        src = torch.randn(shape).to(dtype)
    idx = torch.arange(nseg).repeat_interleave(length // nseg)
    if idx.size(0) < length:
        idx = torch.cat([idx, torch.full((length - idx.size(0),), nseg - 1)])
    idx = _index_for_last_dim(idx, src)
    fn = getattr(ops_gnn, f"segment_{reduce}_coo", ops_gnn.segment_coo)
    kwargs = {"reduce": reduce} if fn == ops_gnn.segment_coo else {}
    result = fn(src.npu(), idx.npu(), **kwargs)
    if isinstance(result, tuple):
        result = result[0]
    expected = segment_coo_cpu(
        src,
        idx,
        reduce=reduce if fn == ops_gnn.segment_coo else fn.__name__.split("_")[1],
    )
    assert result.device.type == "npu"
    _assert_result_close(result, expected)


INTEGER_MEAN_COUNTS = (
    1,
    0,
    2,
    3,
    4,
    7,
    8,
    16,
    31,
    32,
    63,
    64,
    127,
    128,
    255,
    256,
    257,
    511,
    512,
    513,
)


def _fill_packed_segment(source, index, mode, s, outputs):
    expected, args = outputs
    dtype = source.dtype
    positions = (index == s).nonzero().flatten()
    if positions.numel() == 0:
        return
    values = source[positions]
    if mode in ("sum", "mean"):
        acc = values.to(torch.float64 if dtype.is_floating_point else torch.int64).sum(
            0
        )
        if mode == "mean":
            acc = (
                acc / len(positions)
                if dtype.is_floating_point
                else torch.div(acc, len(positions), rounding_mode="trunc")
            )
        expected[s] = acc.to(dtype)
    else:
        extreme = getattr(values, mode)(0).values
        expected[s] = extreme
        args[s] = torch.where(values == extreme, positions[:, None], -1).max(0).values


def _packed_batch_reference(x, idx, dtype, mode, channels):
    expected = torch.zeros(2, 6, channels, dtype=dtype)
    args = torch.full((2, 6, channels), 37, dtype=torch.int64)
    for b in range(2):
        for s in range(6):
            _fill_packed_segment(x[b], idx[b], mode, s, (expected[b], args[b]))
    return expected, args


def _mutating_reference(src, cpu_index, mode):
    dtype = src.dtype
    channels = src.size(1)
    expected = torch.zeros((10, channels), dtype=dtype)
    expected_arg = torch.full((10, channels), 512, dtype=torch.int64)
    for segment in range(10):
        positions = (cpu_index == segment).nonzero().flatten()
        if not positions.numel():
            continue
        values = src[positions]
        if mode in ("sum", "mean"):
            result = values.sum(0)
            if mode == "mean":
                result = (
                    result / len(positions)
                    if dtype.is_floating_point
                    else torch.div(result, len(positions), rounding_mode="trunc")
                )
        else:
            result = getattr(values, mode)(0).values
            expected_arg[segment] = (
                torch.where(values == result, positions[:, None], -1).max(0).values
            )
        expected[segment] = result
    return expected, expected_arg


def _random_batched_source(shape, dtype, generator):
    if dtype.is_floating_point:
        return torch.randn(shape, generator=generator).to(dtype)
    source = torch.randint(0, 11, shape, generator=generator, dtype=dtype)
    if dtype != torch.uint8:
        source -= 5
    if dtype == torch.int64:
        source += ((torch.arange(shape[1]) % 3) - 1).view(1, -1, 1) * 2**45
    return source


def _assert_reference(actual, reference):
    if actual.dtype.is_floating_point:
        tolerance = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 1e-1}[
            actual.dtype
        ]
        torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    else:
        assert torch.equal(actual, reference)


def _offset_reference(cpu, index_cpu, reduce, seed):
    rows, channels = cpu.shape
    segments = 6
    dtype = cpu.dtype
    expected = torch.full((segments, channels), seed, dtype=dtype)
    expected_arg = torch.full((segments, channels), rows, dtype=torch.int64)
    for segment in range(segments):
        positions = (index_cpu == segment).nonzero().flatten()
        if positions.numel() == 0:
            continue
        values = cpu[positions]
        if reduce == "sum":
            expected[segment] = values.sum(0)
        elif reduce == "mean":
            total = values.sum(0)
            expected[segment] = (
                total / len(positions)
                if dtype.is_floating_point
                else torch.div(total, len(positions), rounding_mode="trunc")
            )
        else:
            expected[segment] = getattr(values, reduce)(0).values
            expected_arg[segment] = (
                torch.where(values == expected[segment], positions[:, None], -1)
                .max(0)
                .values
            )
    return expected, expected_arg


def _check_offset_view(dtype, reduce, offset):
    """Packed loads/stores must handle contiguous views with unaligned starts."""
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
    rows, channels, segments = 37, 32, 6
    cpu = (
        torch.arange(rows * channels, dtype=torch.int64)
        .reshape(rows, channels)
        .to(dtype)
    )
    index_cpu = torch.arange(rows, dtype=torch.int64) // 9
    source_storage = torch.cat((torch.zeros(offset, dtype=dtype), cpu.flatten())).npu()
    src = source_storage[offset:].view(rows, channels)
    limits = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    seed = limits.max if reduce == "min" else limits.min if reduce == "max" else 0
    backing = torch.full((segments * channels + offset,), seed, dtype=dtype).npu()
    out = backing[offset:].view(segments, channels)
    assert src.is_contiguous() and out.is_contiguous()
    assert src.storage_offset() == out.storage_offset() == offset
    assert src.data_ptr() % 32 == out.data_ptr() % 32
    before_ptr = out.data_ptr()
    expected, expected_arg = _offset_reference(cpu, index_cpu, reduce, seed)
    result = getattr(ops_gnn, "segment_" + reduce + "_coo")(
        src, index_cpu.npu(), out=out
    )
    if reduce in ("min", "max"):
        result, indices = result
        torch.testing.assert_close(indices.cpu(), expected_arg, rtol=0, atol=0)
    assert result.data_ptr() == out.data_ptr() == before_ptr
    if dtype.is_floating_point:
        torch.testing.assert_close(
            out.cpu(),
            expected,
            rtol=2e-2 if dtype == torch.float16 else 1e-4,
            atol=1e-3,
        )
    else:
        assert torch.equal(out.cpu(), expected)
    assert torch.all(backing[:offset].cpu() == seed)


class TestSegmentCoo:
    @staticmethod
    @pytest.mark.parametrize("shape", SEGMENT_COO_TEST_SHAPES)
    @pytest.mark.parametrize("reduce", SEGMENT_COO_REDUCES)
    @pytest.mark.parametrize("dtype", SEGMENT_COO_SUPPORTED_DTYPES)
    def test_shapes(shape, reduce, dtype):
        _check_shape(shape, reduce, dtype, 16)

    @staticmethod
    @pytest.mark.parametrize("shape", SEGMENT_COO_GENERAL_SHAPES)
    @pytest.mark.parametrize("reduce", SEGMENT_COO_REDUCES)
    @pytest.mark.parametrize("dtype", SEGMENT_COO_SUPPORTED_DTYPES)
    def test_general_shapes(shape, reduce, dtype):
        _check_shape(shape, reduce, dtype, 32)

    @staticmethod
    def test_sum_add_equivalent():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        src = torch.randn(128, device="cpu")
        idx = torch.arange(8).repeat_interleave(16)
        r1 = ops_gnn.segment_sum_coo(src.npu(), idx.npu())
        r2 = ops_gnn.segment_add_coo(src.npu(), idx.npu())
        expected = segment_coo_cpu(src, idx, reduce="sum")
        assert torch.allclose(r1.cpu(), r2.cpu())
        assert torch.allclose(r1.cpu(), expected, rtol=1e-3, atol=1e-3)

    @staticmethod
    def test_reduce_mul_error():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        src = torch.randn(64, device="cpu")
        idx = torch.arange(4).repeat_interleave(16)
        with pytest.raises(ValueError):
            ops_gnn.segment_coo(src.npu(), idx.npu(), reduce="mul")

    @staticmethod
    def test_empty_index():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        src = torch.randn(0, device="cpu")
        idx = torch.empty(0, dtype=torch.long)
        result = ops_gnn.segment_sum_coo(src.npu(), idx.npu())
        assert result.numel() == 0

    @staticmethod
    def test_dim_size_one():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        src = torch.randn(128, device="cpu")
        idx = torch.zeros(128, dtype=torch.long)
        result = ops_gnn.segment_sum_coo(src.npu(), idx.npu())
        assert result.shape[0] == 1

    @staticmethod
    def test_empty_segment():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        src = torch.randn(8, device="cpu")
        idx = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
        result = ops_gnn.segment_sum_coo(src.npu(), idx.npu())
        assert result.device.type == "npu"

    @staticmethod
    @pytest.mark.parametrize("reduce", SEGMENT_COO_REDUCES)
    def test_with_out(reduce):
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        src = torch.randn(32, 512, device="cpu")
        idx = _index_for_last_dim(torch.arange(4).repeat_interleave(128), src)
        out = torch.zeros(32, 4, device="cpu")
        fn = getattr(ops_gnn, f"segment_{reduce}_coo", ops_gnn.segment_coo)
        kwargs = {"out": out.npu()}
        if fn == ops_gnn.segment_coo:
            kwargs["reduce"] = reduce
        result = fn(src.npu(), idx.npu(), **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        assert result.device.type == "npu"

    @staticmethod
    @pytest.mark.parametrize("reduce", SEGMENT_COO_REDUCES)
    @pytest.mark.parametrize("dtype", SEGMENT_COO_SUPPORTED_DTYPES)
    @pytest.mark.filterwarnings(
        "error:Cannot create tensor with internal format:UserWarning"
    )
    def test_non_contiguous_src(reduce, dtype):
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        base = (
            _integer_input((256,), dtype)
            if dtype in (torch.int8, torch.int32, torch.int64, torch.uint8)
            else torch.randn(256)
        )
        src = base.to(dtype)[::2]
        assert not src.is_contiguous()
        src_npu = base.to(dtype).npu()[::2]
        assert not src_npu.is_contiguous()
        idx = torch.arange(8).repeat_interleave(16)
        fn = getattr(ops_gnn, f"segment_{reduce}_coo", ops_gnn.segment_coo)
        kwargs = {"reduce": reduce} if fn == ops_gnn.segment_coo else {}
        result = fn(src_npu, idx.npu(), **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        expected = segment_coo_cpu(src, idx, reduce=reduce)
        assert result.device.type == "npu"
        _assert_result_close(result, expected)

    @staticmethod
    @pytest.mark.parametrize("reduce", SEGMENT_COO_REDUCES)
    @pytest.mark.parametrize("dtype", SEGMENT_COO_SUPPORTED_DTYPES)
    @pytest.mark.filterwarnings(
        "error:Cannot create tensor with internal format:UserWarning"
    )
    def test_non_contiguous_src_2d(reduce, dtype):
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        base = (
            _integer_input((128, 64), dtype)
            if dtype in (torch.int8, torch.int32, torch.int64, torch.uint8)
            else torch.randn(128, 64)
        )
        src = base.to(dtype).t()
        assert not src.is_contiguous()
        src_npu = base.to(dtype).npu().t()
        assert not src_npu.is_contiguous()
        idx = torch.arange(8).repeat_interleave(8)
        fn = getattr(ops_gnn, f"segment_{reduce}_coo", ops_gnn.segment_coo)
        kwargs = {"reduce": reduce} if fn == ops_gnn.segment_coo else {}
        result = fn(src_npu, idx.npu(), **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        expected = segment_coo_cpu(src, idx, reduce=reduce)
        assert result.device.type == "npu"
        _assert_result_close(result, expected)

    @staticmethod
    def test_non_contiguous_index():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        src = torch.randn(128, device="cpu")
        full = torch.arange(32).repeat_interleave(8)
        idx = full[::2]
        assert not idx.is_contiguous()
        result = ops_gnn.segment_sum_coo(src.npu(), idx.npu())
        expected = segment_coo_cpu(src, idx, reduce="sum")
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)


# Byte mean


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("channels", [1, 8, 32])
@pytest.mark.parametrize("kind", ["random", "positive", "boundary"])
def test_byte_mean_cpu_narrowing(dtype, channels, kind):
    counts = torch.tensor([0, 1, 2, 7, 17, 31, 33, 127, 129, 257, 0])
    index = torch.arange(len(counts)).repeat_interleave(counts)
    if kind == "boundary":
        source = torch.full(
            (len(index), channels), torch.iinfo(dtype).max // 4, dtype=dtype
        )
    else:
        low = -5 if dtype == torch.int8 and kind == "random" else 0
        source = torch.randint(
            low,
            6,
            (len(index), channels),
            dtype=dtype,
            generator=torch.Generator().manual_seed(923),
        )
    expected = segment_mean_reference(source, index, dim_size=len(counts))
    actual = ops_gnn.segment_mean_coo(source.npu(), index.npu(), dim_size=len(counts))
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("channels", [1, 8, 32])
def test_byte_mean_zero_narrowed_count_is_safe(dtype, channels):
    # CPU Reducer division is undefined here: do not call the crashing oracle.
    # Our extension defines a safe mathematical result instead of dividing by 0.
    source = torch.full((256, channels), 3, dtype=dtype, device="npu")
    index = torch.zeros(256, dtype=torch.int64, device="npu")
    actual = ops_gnn.segment_mean_coo(source, index, dim_size=1)
    assert torch.equal(actual.cpu(), torch.full((1, channels), 3, dtype=dtype))


# Cpu reference


@pytest.mark.parametrize(
    "dtype",
    CPU_COMPARISON_DTYPES,
)
@pytest.mark.parametrize("mode", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("channels", [1, 8, 32, 128])
def test_official_cpu(dtype, mode, channels):
    x = torch.arange(37).view(37, 1).expand(37, channels).contiguous()
    if dtype != torch.uint8:
        x = x - 18
    if dtype == torch.int64:
        x = x + 2**45
    x = x.to(dtype)
    idx = (torch.arange(37) // 4) * 2
    s = 22
    f = "segment_" + mode + "_coo"
    ref = reference_function(f)(x, idx, dim_size=s)
    actual = getattr(ops_gnn, f)(x.npu(), idx.npu(), dim_size=s)
    if mode in ("min", "max"):
        # Values are unique within every segment, so arg comparison is exact.
        assert torch.equal(actual[1].cpu(), ref[1])
        actual, ref = actual[0], ref[0]
    _assert_reference(actual.cpu(), ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("mode", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("layout", ["rank8", "partial_broadcast"])
def test_high_rank_and_partial_broadcast(dtype, mode, layout):
    if layout == "rank8":
        x = (torch.arange(384) - 100).reshape(2, 1, 2, 3, 2, 1, 2, 8).to(dtype)
        idx = torch.tensor([0, 2, 2]).view(1, 1, 1, 3)
    else:
        x = (torch.arange(2 * 3 * 17 * 8) - 100).reshape(2, 3, 17, 8).to(dtype)
        idx = torch.stack([torch.arange(17) // 4, (torch.arange(17) + 1) // 5]).view(
            2, 1, 17
        )
    f = "segment_" + mode + "_coo"
    ref = reference_function(f)(x, idx, dim_size=6)
    y = ops_gnn.segment_coo(x.npu(), idx.npu(), dim_size=6, reduce=mode)
    if mode in ("min", "max"):
        value, arg = getattr(ops_gnn, f)(x.npu(), idx.npu(), dim_size=6)
        assert torch.equal(arg.cpu(), ref[1])
        torch.testing.assert_close(value.cpu(), ref[0], rtol=1e-4, atol=1e-4)
        ref = ref[0]
    torch.testing.assert_close(y.cpu(), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.int32, torch.int64]
)
@pytest.mark.parametrize("mode", ["sum", "mean"])
@pytest.mark.parametrize("channels", [32, 64, 128, 256])
def test_long_unequal_segments(dtype, mode, channels):
    """Exercise cooperative row partitions, remainders, holes and INT64 precision."""
    counts = torch.tensor([0, 1, 7, 8, 9, 31, 33, 511, 513, 0])
    idx = torch.arange(len(counts)).repeat_interleave(counts)
    generator = torch.Generator().manual_seed(913)
    if dtype.is_floating_point:
        x = torch.randn(len(idx), channels, generator=generator).to(dtype)
    else:
        x = torch.randint(
            -100, 100, (len(idx), channels), generator=generator, dtype=dtype
        )
        if dtype == torch.int64:
            x += ((torch.arange(len(idx)) % 3) - 1).view(-1, 1) * 2**45
    reference = reference_function("segment_" + mode + "_coo")(
        x, idx, dim_size=len(counts)
    )
    actual = ops_gnn.segment_coo(
        x.npu(), idx.npu(), dim_size=len(counts), reduce=mode
    ).cpu()
    if dtype.is_floating_point:
        tolerance = 2e-2 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    else:
        assert torch.equal(actual, reference)


@pytest.mark.parametrize(
    "dtype",
    CPU_COMPARISON_DTYPES,
)
@pytest.mark.parametrize("mode", ["sum", "mean"])
@pytest.mark.parametrize("channels", [32, 96])
@pytest.mark.parametrize("segments", [16, 19])
def test_short_segment_addressing(dtype, mode, channels, segments):
    """Power-of-two coordinate fast path and exact non-power-of-two fallback."""
    generator = torch.Generator().manual_seed(915)
    idx = torch.randint(0, segments, (3, 71), generator=generator).sort(dim=-1).values
    x = _random_batched_source((3, 71, channels), dtype, generator)

    reference = reference_function("segment_" + mode + "_coo")(
        x, idx, dim_size=segments
    )
    actual = ops_gnn.segment_coo(
        x.npu(), idx.npu(), dim_size=segments, reduce=mode
    ).cpu()
    _assert_reference(actual, reference)


# Direct boundaries

DIRECT_BOUNDARIES_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
]


@pytest.mark.parametrize("dtype", DIRECT_BOUNDARIES_DTYPES)
@pytest.mark.parametrize("mode", ["sum", "mean", "min", "max"])
def test_pointer_nonzero_endpoints_large_holes_batched(dtype, mode):
    # Both non-power-of-two actual ranges and distinct per-batch index arrays.
    # No count reaches the CPU byte-mean undefined zero-denominator domain.
    index = torch.stack(
        (
            torch.tensor([13] * 17 + [100] * 31 + [4096] * 79),
            torch.tensor([23] * 63 + [4000] * 1 + [4095] * 63),
        )
    )
    values = (torch.arange(2 * 127 * 8).reshape(2, 127, 8) % 7).to(dtype)
    if dtype == torch.int64:
        values += 2**40
    expected = segment_coo_reference(values, index, dim_size=4099,
                                     options=ReferenceOptions(reduce=mode))
    actual = ops_gnn.segment_coo(
        values.npu(), index.npu(), dim_size=4099, reduce=mode
    ).cpu()
    if dtype.is_floating_point:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    else:
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", DIRECT_BOUNDARIES_DTYPES)
@pytest.mark.parametrize("mode", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("channels", [3, 8, 32])
def test_empty_input_nonempty_output(dtype, mode, channels):
    src = torch.empty((0, channels), dtype=dtype, device="npu")
    index = torch.empty(0, dtype=torch.int64, device="npu")
    for seeded in (False, True):
        seed = (
            torch.full((5, channels), 7, dtype=dtype, device="npu") if seeded else None
        )
        value = ops_gnn.segment_coo(src, index, out=seed, dim_size=5, reduce=mode)
        expected = 7 if seeded and mode != "sum" else 0
        assert torch.equal(
            value.cpu(), torch.full((5, channels), expected, dtype=dtype)
        )
        if mode in ("min", "max"):
            value, arg = getattr(ops_gnn, "segment_" + mode + "_coo")(
                src, index, out=seed, dim_size=5
            )
            assert torch.equal(arg.cpu(), torch.zeros((5, channels), dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("mode", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("channels", [8, 32])
def test_same_shape_mutating_skewed_index(dtype, mode, channels):
    # Same allocation, shape and endpoints, with entirely different inner bounds.
    # Large native integers also rule out using float arithmetic for the result.
    count_sets = (
        [1, 0, 0, 0, 510, 0, 0, 1],
        [255, 1, 0, 0, 0, 0, 1, 255],
        [64, 64, 64, 64, 64, 64, 64, 64],
        [0, 0, 0, 512, 0, 0, 0, 0],
    )
    src = (torch.arange(512 * channels).reshape(512, channels) % 11 - 5).to(dtype)
    if dtype == torch.int64:
        src += 2**45
    device_src = src.npu()
    index = torch.empty(512, dtype=torch.int64, device="npu")
    for counts in count_sets:
        cpu_index = torch.arange(8).repeat_interleave(torch.tensor(counts))
        index.copy_(cpu_index.npu())
        expected, expected_arg = _mutating_reference(src, cpu_index, mode)
        actual = ops_gnn.segment_coo(device_src, index, dim_size=10, reduce=mode).cpu()
        if dtype.is_floating_point:
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        else:
            assert torch.equal(actual, expected)
        if mode in ("min", "max"):
            _, arg = getattr(ops_gnn, "segment_" + mode + "_coo")(
                device_src, index, dim_size=10
            )
            assert torch.equal(arg.cpu(), expected_arg)


# Extreme coordinates


@pytest.mark.parametrize(
    "dtype",
    CPU_COMPARISON_DTYPES,
)
@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("channels", [32, 96, 128])
@pytest.mark.parametrize("segments", [16, 19])
def test_extreme_coordinate_paths(dtype, mode, channels, segments):
    length = 97
    generator = torch.Generator().manual_seed(931)
    index = torch.randint(0, segments // 2, (2, length), generator=generator)
    index = (index * 2).sort(dim=-1).values  # Uneven segments and holes.
    source = torch.randint(0, 17, (2, length, channels), generator=generator)
    if dtype != torch.uint8:
        source -= 8
    if dtype == torch.int64:
        source += 2**45
    source = source.to(dtype)
    function_name = "segment_" + mode + "_coo"
    reference = reference_function(function_name)(source, index, dim_size=segments)[
        0
    ]
    expected_arg = torch.full((2, segments, channels), length, dtype=torch.int64)
    for batch in range(2):
        for segment in range(segments):
            positions = (index[batch] == segment).nonzero().flatten()
            if positions.numel():
                # CPU torch_scatter uses first ties; the task explicitly asks
                # for later ties, so derive that arg from actual matching rows.
                matches = source[batch, positions] == reference[batch, segment]
                expected_arg[batch, segment] = (
                    torch.where(matches, positions[:, None], -1).max(dim=0).values
                )
    actual, arg = getattr(ops_gnn, function_name)(
        source.npu(), index.npu(), dim_size=segments
    )
    assert torch.equal(actual.cpu(), reference)
    assert torch.equal(arg.cpu(), expected_arg)
    value_only = ops_gnn.segment_coo(
        source.npu(), index.npu(), dim_size=segments, reduce=mode
    )
    assert torch.equal(value_only.cpu(), reference)


# Integer mean extremes


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("channels", [1, 8, 32, 96, 128])
@pytest.mark.parametrize("with_out", [False, True])
def test_integer_mean_extremes(dtype, channels, with_out):
    # CPU's first accumulator seeds out[0] even if the first index is not 0.
    # Start at segment0 for this direct CPU comparison, keeping an interior hole.
    # The separate empty/nonzero-endpoint tests still cover their task semantics.
    counts = INTEGER_MEAN_COUNTS
    index = torch.arange(len(counts)).repeat_interleave(torch.tensor(counts))
    source = torch.zeros(len(index), channels, dtype=dtype)
    initial = torch.zeros(len(counts), channels, dtype=dtype)
    expected = torch.zeros_like(initial)
    limits = torch.iinfo(dtype)
    begin = 0
    for segment, count in enumerate(counts):
        values = [
            limits.min,
            limits.max,
            -count - 1,
            count + 1,
            -count + 1,
            count - 1,
            -1,
            1,
        ]
        for channel in range(channels):
            value = values[(channel + segment) % len(values)]
            # For optional out, use an opposite-sign seed to avoid sum overflow.
            seed = (3 if value < 0 else -3) if with_out else 0
            initial[segment, channel] = seed
            if count:
                source[begin, channel] = value
                numerator = value + seed
                quotient = abs(numerator) // count
                expected[segment, channel] = -quotient if numerator < 0 else quotient
            else:
                expected[segment, channel] = seed
        begin += count

    reference = segment_mean_reference(
        source, index, out=initial.clone() if with_out else None, dim_size=len(counts)
    )
    assert torch.equal(reference, expected)
    actual = ops_gnn.segment_mean_coo(
        source.npu(),
        index.npu(),
        out=initial.npu() if with_out else None,
        dim_size=len(counts),
    ).cpu()
    assert torch.equal(actual, expected)


# Long coordinates


@pytest.mark.parametrize(
    "dtype",
    CPU_COMPARISON_DTYPES,
)
@pytest.mark.parametrize("mode", ["sum", "mean"])
@pytest.mark.parametrize("channels", [32, 96, 128])
@pytest.mark.parametrize("segments", [16, 19])
@pytest.mark.parametrize("shared_index", [False, True])
def test_long_coordinate_paths(dtype, mode, channels, segments, shared_index):
    counts = torch.tensor(
        [0, 1, 64, 3, 31, 32, 17, 16, 127, 4, 0, 48, 11, 2, 70, 8, 5, 6, 7]
    )[:segments]
    assert int(counts.sum()) >= segments * 16  # Exercise cooperative dispatch.
    batches = 1 if shared_index else 3
    index = torch.stack(
        [
            torch.arange(segments).repeat_interleave(counts.roll(batch))
            for batch in range(batches)
        ]
    )
    generator = torch.Generator().manual_seed(926)
    shape = (3, index.shape[-1], channels)
    source = _random_batched_source(shape, dtype, generator)

    expected = reference_function("segment_" + mode + "_coo")(
        source, index, dim_size=segments
    )
    actual = ops_gnn.segment_coo(
        source.npu(), index.npu(), dim_size=segments, reduce=mode
    ).cpu()
    _assert_reference(actual, expected)


# Mean dispatch


@pytest.mark.parametrize("channels", [8, 32, 64, 128])
@pytest.mark.parametrize("average", [15, 16, 17])
@pytest.mark.parametrize("skewed", [False, True])
def test_int64_mean_dispatch(channels, average, skewed):
    segments = 16
    length = segments * average
    generator = torch.Generator().manual_seed(930)
    if skewed:
        index = torch.randint(0, segments // 2, (length,), generator=generator)
        index = (index * 2).sort().values
    else:
        index = torch.arange(segments).repeat_interleave(average)
    source = torch.randint(-100, 100, (length, channels), generator=generator)
    source += ((torch.arange(length) % 3) - 1).view(-1, 1) * 2**45
    reference = segment_mean_reference(source, index, dim_size=segments)
    actual = ops_gnn.segment_mean_coo(
        source.npu(), index.npu(), dim_size=segments
    ).cpu()
    assert torch.equal(actual, reference)


# Semantics


@pytest.mark.parametrize(
    "dtype",
    CPU_COMPARISON_DTYPES,
)
@pytest.mark.parametrize("mode", ["sum", "mean", "min", "max"])
def test_unequal_segments_and_args(dtype, mode):
    x = torch.tensor(
        [[2, 5], [2, 1], [3, 6], [4, 6], [1, 2]], dtype=dtype, device="npu"
    )
    idx = torch.tensor([0, 0, 2, 2, 2], device="npu")
    out = ops_gnn.segment_coo(x, idx, dim_size=4, reduce=mode).cpu()
    ref = {
        "sum": [[4, 6], [0, 0], [8, 14], [0, 0]],
        "mean": [[2, 3], [0, 0], [8 / 3, 14 / 3], [0, 0]],
        "min": [[2, 1], [0, 0], [1, 2], [0, 0]],
        "max": [[2, 5], [0, 0], [4, 6], [0, 0]],
    }[mode]
    torch.testing.assert_close(out, torch.tensor(ref).to(dtype), rtol=1e-2, atol=1e-2)
    if mode in ("min", "max"):
        v, arg = getattr(ops_gnn, "segment_" + mode + "_coo")(x, idx, dim_size=4)
        expected = {
            "min": [[1, 1], [5, 5], [4, 4], [5, 5]],
            "max": [[1, 0], [5, 5], [3, 3], [5, 5]],
        }[mode]
        assert torch.equal(arg.cpu(), torch.tensor(expected))


def test_int64_exact():
    x = torch.tensor([2**55, 2**55 + 7, -(2**55), -(2**55) + 3], device="npu")
    idx = torch.tensor([0, 0, 1, 1], device="npu")
    assert ops_gnn.segment_sum_coo(x, idx).cpu().tolist() == [2**56 + 7, -(2**56) + 3]
    assert ops_gnn.segment_mean_coo(x, idx).cpu().tolist() == [2**55 + 3, -(2**55) + 2]


def test_multiple_index_batches():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    idx = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    y = ops_gnn.segment_sum_coo(x.npu(), idx.npu()).cpu()
    expected = torch.stack(
        [
            torch.stack([x[0, :2].sum(0), x[0, 2:].sum(0)]),
            torch.stack([x[1, 0], x[1, 1:].sum(0)]),
        ]
    )
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize(
    "dtype",
    CPU_COMPARISON_DTYPES,
)
@pytest.mark.parametrize("mode", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("channels", [4, 32, 128])
def test_packed_uneven_batches(dtype, mode, channels):
    x = (torch.arange(2 * 37 * channels).reshape(2, 37, channels) % 7).to(dtype)
    if dtype != torch.uint8:
        x = x - 3
    if dtype == torch.int64:
        x = x + 2**50
    idx = torch.stack(
        [
            torch.tensor([0] * 2 + [2] * 5 + [3] * 30),
            torch.tensor([1] * 11 + [2] * 25 + [4]),
        ]
    )
    expected, args = _packed_batch_reference(x, idx, dtype, mode, channels)
    if mode in ("min", "max"):
        y, arg = getattr(ops_gnn, "segment_" + mode + "_coo")(
            x.npu(), idx.npu(), dim_size=6
        )
        assert torch.equal(arg.cpu(), args)
    else:
        y = ops_gnn.segment_coo(x.npu(), idx.npu(), dim_size=6, reduce=mode)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("mode", ["sum", "add", "mean", "min", "max"])
@pytest.mark.parametrize("channels", [1, 8, 32])
def test_nonzero_noncontiguous_out(dtype, mode, channels):
    x = torch.arange(1, 5, dtype=dtype).view(4, 1).expand(4, channels).npu()
    idx = torch.tensor([0, 0, 2, 2], device="npu")
    storage = torch.full((4, channels * 2), -77, dtype=dtype, device="npu")
    out = storage[:, ::2]
    out.copy_(torch.tensor([10, 20, 30, 40], dtype=dtype).view(4, 1).npu())
    result = ops_gnn.segment_coo(x, idx, out=out, reduce=mode)
    expected = {
        "sum": [3, 0, 7, 0],
        "add": [3, 0, 7, 0],
        "mean": [6.5, 20, 18.5, 40],
        "min": [1, 20, 3, 40],
        "max": [10, 20, 30, 40],
    }[mode]
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(
        result.cpu(), torch.tensor(expected).to(dtype).view(4, 1).expand(4, channels)
    )
    assert torch.all(storage[:, 1::2].cpu() == -77)


# Stream order


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_pending_work_nondefault_stream(dtype, strided):
    source = (torch.arange(71 * 64).reshape(71, 64) % 17 - 8).to(dtype)
    index = (torch.arange(71) // 5).long()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        device_source = source.npu() + 2
        device_index = index.npu()
        if strided:
            device_source = device_source[:, ::2]
        for mode in ("sum", "mean", "min", "max"):
            expected_source = (source + 2)[:, ::2] if strided else source + 2
            expected = segment_coo_reference(
                expected_source, index, dim_size=17, options=ReferenceOptions(reduce=mode)
            )
            result = ops_gnn.segment_coo(
                device_source, device_index, dim_size=17, reduce=mode
            )
            consumed = result + 3
            stream.synchronize()
            torch.testing.assert_close(consumed.cpu(), expected + 3, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.int32, torch.int64]
)
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_contiguous_offset_views_keep_values_and_output_storage(dtype, reduce):
    for offset in (1, 2, 4):
        _check_offset_view(dtype, reduce, offset)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("mode", ["sum", "mean"])
@pytest.mark.parametrize("shared_index", [False, True])
def test_plan_path_transitions(dtype, mode, shared_index):
    """Alternate direct-index, bounds and scalar shapes without reusing a plan."""
    index = torch.stack((torch.arange(37) // 5, (torch.arange(37) + 3) // 6))
    if shared_index:
        index = index[:1]
    for channels in (8, 7, 16, 9, 32, 1):
        source = (torch.arange(2 * 37 * channels).reshape(2, 37, channels) % 17 - 8).to(dtype)
        if dtype == torch.int64:
            source += 2**45
        expected = reference_function("segment_" + mode + "_coo")(
            source, index, dim_size=10
        )
        actual = ops_gnn.segment_coo(source.npu(), index.npu(), dim_size=10, reduce=mode)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

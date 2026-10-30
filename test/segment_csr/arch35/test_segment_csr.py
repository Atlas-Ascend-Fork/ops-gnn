# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright (c) 2026 Starlink_. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CSR API, cross-call semantics and vector-route regression coverage."""

import os
from pathlib import Path
import runpy

import pytest
import torch

import ops_gnn
_GOLDEN = runpy.run_path(str(Path(__file__).with_name("golden.py")))
segment_csr_cpu = _GOLDEN["segment_csr_cpu"]
segment_argminmax_cpu = _GOLDEN["segment_argminmax_cpu"]

# test_segment_csr

SUPPORTED_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.int32,
    torch.int64,
    torch.uint8,
]


REDUCES = ["sum", "add", "mean", "min", "max"]


TEST_SHAPES = [
    (128,),
    (1024,),
    (4096,),
    (8192,),
    (32, 512),
    (64, 768),
    (8, 16, 64),
    (4, 128, 256),
]


GENERAL_SHAPES = [
    (1,),
    (4,),
    (2, 2),
    (1, 128),
    (512, 768),
    (512, 1024),
    (1024, 768),
    (1024, 1024),
]


def _indptr_for_last_dim(indptr, src):
    """Broadcast a 1-D CSR pointer over all leading source dimensions."""
    if src.dim() == 1:
        return indptr
    return indptr.view(*([1] * (src.dim() - 1)), indptr.numel())


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


def _make_indptr(n, nseg):
    base = n // nseg
    sizes = torch.full((nseg,), base, dtype=torch.long)
    sizes[-1] += n - base * nseg
    return torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])


def _coo_reduce_cpu(src, index, num_segments, reduce):
    """Independent CPU COO reference for the task-book CSR/COO consistency case."""
    result = torch.zeros(num_segments, dtype=src.dtype)
    for segment in range(num_segments):
        values = src[index == segment]
        if values.numel() == 0:
            continue
        if reduce in ("sum", "add"):
            result[segment] = values.sum()
        elif reduce == "mean":
            result[segment] = values.mean()
        else:
            result[segment] = getattr(values, reduce)()
    return result


def _check_shape(shape, reduce, dtype, divisor):
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    n = shape[-1] if len(shape) >= 2 else shape[0]
    nseg = max(2, n // divisor)
    if dtype in (torch.int8, torch.int32, torch.int64, torch.uint8):
        src = _integer_input(shape, dtype)
    else:
        src = torch.randn(shape).to(dtype)
    indptr = _indptr_for_last_dim(_make_indptr(n, nseg), src)
    fn = getattr(ops_gnn, f"segment_{reduce}_csr", ops_gnn.segment_csr)
    kwargs = {"reduce": reduce} if fn == ops_gnn.segment_csr else {}
    result = fn(src.npu(), indptr.npu(), **kwargs)
    if isinstance(result, tuple):
        result = result[0]
    expected = segment_csr_cpu(
        src,
        indptr,
        reduce=reduce if fn == ops_gnn.segment_csr else fn.__name__.split("_")[1],
    )
    assert result.device.type == "npu"
    _assert_result_close(result, expected)


class TestSegmentCsr:
    @staticmethod
    def test_segment_csr_generic():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        src = torch.randn(64, device="cpu")
        indptr = torch.tensor([0, 16, 40, 64])
        r1 = ops_gnn.segment_sum_csr(src.npu(), indptr.npu())
        r2 = ops_gnn.segment_csr(src.npu(), indptr.npu(), reduce="sum")
        expected = segment_csr_cpu(src, indptr, reduce="sum")
        assert torch.allclose(r1.cpu(), r2.cpu())
        assert torch.allclose(r1.cpu(), expected, rtol=1e-3, atol=1e-3)

    @staticmethod
    def test_reduce_mul_error():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        src = torch.randn(64, device="cpu")
        indptr = torch.tensor([0, 16, 40, 64])
        with pytest.raises(ValueError):
            ops_gnn.segment_csr(src.npu(), indptr.npu(), reduce="mul")

    @staticmethod
    def test_empty_src():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        src = torch.empty(0, device="cpu")
        indptr = torch.tensor([0], dtype=torch.long)
        result = ops_gnn.segment_sum_csr(src.npu(), indptr.npu())
        assert result.numel() == 0

    @staticmethod
    def test_empty_segment():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        src = torch.randn(8, device="cpu")
        indptr = torch.tensor([0, 0, 4, 4, 8])
        result = ops_gnn.segment_sum_csr(src.npu(), indptr.npu())
        assert result.device.type == "npu"

    @staticmethod
    def test_all_empty_segments():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        src = torch.randn(8, device="cpu")
        indptr = torch.tensor([0, 0, 0, 0, 8])
        result = ops_gnn.segment_sum_csr(src.npu(), indptr.npu())
        assert result.device.type == "npu"

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_shapes(self, shape, reduce, dtype):
        _check_shape(shape, reduce, dtype, 32)

    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_general_shapes(self, shape, reduce, dtype):
        _check_shape(shape, reduce, dtype, 16)

    @pytest.mark.parametrize("reduce", [r for r in REDUCES if r != "max"])
    def test_with_out(self, reduce):
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        src = torch.randn(32, 512, device="cpu")
        indptr = torch.tensor([[0, 128, 256, 384, 512]])
        out = torch.zeros(32, 4, device="cpu")
        fn = getattr(ops_gnn, f"segment_{reduce}_csr", ops_gnn.segment_csr)
        kwargs = {"out": out.npu()}
        if fn == ops_gnn.segment_csr:
            kwargs["reduce"] = reduce
        result = fn(src.npu(), indptr.npu(), **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        assert result.device.type == "npu"

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_large(self, dtype):
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        if dtype in (torch.int8, torch.int32, torch.int64, torch.uint8):
            src = _integer_input((512, 768), dtype)
        else:
            src = torch.randn(512, 768).to(dtype)
        indptr = _indptr_for_last_dim(_make_indptr(768, 6), src)
        result = ops_gnn.segment_sum_csr(src.npu(), indptr.npu())
        assert result.device.type == "npu"

    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.filterwarnings(
        "error:Cannot create tensor with internal format:UserWarning"
    )
    def test_non_contiguous_src(self, reduce, dtype):
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
        indptr = torch.tensor([0, 32, 96, 128])
        fn = getattr(ops_gnn, f"segment_{reduce}_csr", ops_gnn.segment_csr)
        kwargs = {"reduce": reduce} if fn == ops_gnn.segment_csr else {}
        result = fn(src_npu, indptr.npu(), **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        expected = segment_csr_cpu(src, indptr, reduce=reduce)
        assert result.device.type == "npu"
        _assert_result_close(result, expected)

    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.filterwarnings(
        "error:Cannot create tensor with internal format:UserWarning"
    )
    def test_non_contiguous_src_2d(self, reduce, dtype):
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
        # Transfer contiguous storage first, then create the strided NPU view.
        # This avoids an implicit internal-format allocation in Tensor.to().
        src_npu = base.to(dtype).npu().t()
        assert not src_npu.is_contiguous()
        indptr = torch.tensor([0, 8, 32, 64])
        fn = getattr(ops_gnn, f"segment_{reduce}_csr", ops_gnn.segment_csr)
        kwargs = {"reduce": reduce} if fn == ops_gnn.segment_csr else {}
        result = fn(src_npu, indptr.npu(), **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        expected = segment_csr_cpu(src, indptr, reduce=reduce)
        assert result.device.type == "npu"
        _assert_result_close(result, expected)

    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("lengths", [(16, 24, 40, 48), (16, 0, 64, 48)])
    def test_consistency_with_coo(self, reduce, lengths):
        """TC-11: compare NPU CSR against an independent CPU COO reduction."""
        torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
        torch.manual_seed(42)
        src = torch.randn(sum(lengths), dtype=torch.float32)
        counts = torch.tensor(lengths, dtype=torch.int64)
        indptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
        index = torch.arange(len(lengths), dtype=torch.int64).repeat_interleave(counts)
        expected = _coo_reduce_cpu(src, index, len(lengths), reduce)
        actual = ops_gnn.segment_csr(src.npu(), indptr.npu(), reduce=reduce)
        assert actual.device.type == "npu"
        _assert_result_close(actual, expected)


@pytest.mark.parametrize("reduce", ["sum", "add", "mean", "min", "max"])
def test_noncontiguous_provided_out_is_updated_and_returned(reduce):
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
    src = (torch.arange(16 * 32, dtype=torch.float32).view(16, 32) % 13) - 6
    indptr = torch.tensor([0, 4, 4, 16], dtype=torch.int64)
    backing = torch.full((3, 64), 777.0).npu()
    out = backing[:, ::2]
    assert not out.is_contiguous()

    expected = torch.zeros((3, 32), dtype=src.dtype)
    expected_arg = torch.full((3, 32), src.size(0), dtype=torch.int64)
    for segment, (begin, end) in enumerate(
        zip(indptr[:-1].tolist(), indptr[1:].tolist())
    ):
        if begin == end:
            continue
        values = src[begin:end]
        if reduce in ("sum", "add"):
            expected[segment] = values.sum(dim=0)
        elif reduce == "mean":
            expected[segment] = values.mean(dim=0)
        else:
            expected[segment], relative_arg = getattr(values, reduce)(dim=0)
            expected_arg[segment] = relative_arg + begin

    result = getattr(ops_gnn, f"segment_{reduce}_csr")(src.npu(), indptr.npu(), out)
    if isinstance(result, tuple):
        result, arg = result
        torch.testing.assert_close(arg.cpu(), expected_arg, rtol=0, atol=0)
    assert result.data_ptr() == out.data_ptr()
    assert result.stride() == out.stride()
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        backing[:, 1::2].cpu(), torch.full((3, 32), 777.0), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("reduce", ["min", "max"])
@pytest.mark.parametrize("provide_out", [False, True])
@pytest.mark.parametrize(
    "layout",
    [
        ((4, 3), [0], (0, 3)),
        ((0, 4, 3), [[0, 2, 4]], (0, 2, 3)),
        ((4, 0), [0, 2, 4], (2, 0)),
    ],
    ids=["zero_segments", "zero_batches", "zero_channels"],
)
def test_empty_extrema_return_values_and_indices(dtype, reduce, provide_out, layout):
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
    source_shape, pointers, output_shape = layout
    src = torch.empty(source_shape, dtype=dtype).npu()
    indptr = torch.tensor(pointers, dtype=torch.int64).npu()
    out = torch.empty(output_shape, dtype=dtype).npu() if provide_out else None

    result = getattr(ops_gnn, f"segment_{reduce}_csr")(src, indptr, out)

    assert isinstance(result, tuple) and len(result) == 2
    values, indices = result
    assert tuple(values.shape) == tuple(indices.shape) == output_shape
    assert values.numel() == indices.numel() == 0
    assert values.dtype == dtype
    assert indices.dtype == torch.int64
    assert values.device == indices.device == src.device
    if out is not None:
        assert values is out

    generic = ops_gnn.segment_csr(src, indptr, out, reduce=reduce)
    assert isinstance(generic, torch.Tensor)
    assert tuple(generic.shape) == output_shape
    assert generic.numel() == 0 and generic.dtype == dtype


# test_segment_csr_ext

EXT_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
]


EXT_REDUCES = ["sum", "mean", "min", "max"]


def _rand(shape, dtype):
    if dtype in (torch.int8, torch.uint8):
        return torch.randint(0, 8, shape).to(dtype)
    if dtype in (torch.int32, torch.int64):
        return torch.randint(-50, 50, shape).to(dtype)
    return torch.randn(shape).to(dtype)


def _close(result, expected, dtype):
    if dtype == torch.float16:
        torch.testing.assert_close(result.cpu(), expected, rtol=2e-2, atol=2e-2)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-1, atol=1e-1)
    elif result.is_floating_point():
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
    else:
        assert torch.equal(result.cpu(), expected)


def _fn(reduce):
    return getattr(ops_gnn, f"segment_{reduce}_csr")


def test_determinism_bitwise():
    torch.manual_seed(7)
    src = torch.randn(100000, 64)
    indptr = torch.zeros(257, dtype=torch.long)
    indptr[1:] = torch.cumsum(torch.randint(0, 1200, (256,)), 0)
    indptr = torch.clamp(indptr, max=100000)
    indptr[-1] = 100000
    for dtype in (torch.float32, torch.float16, torch.int32, torch.int64):
        s = src.to(dtype).npu()
        for reduce in EXT_REDUCES:

            def _first(r):
                return r[0] if isinstance(r, tuple) else r

            outs = [_first(_fn(reduce)(s, indptr.npu())) for _ in range(5)]
            for o in outs[1:]:
                assert torch.equal(outs[0].cpu(), o.cpu()), (dtype, reduce)


def test_arg_first_occurrence():
    src = torch.tensor([[5.0, 1.0, 1.0, 5.0, 2.0], [3.0, 3.0, 3.0, 3.0, 3.0]])
    indptr = torch.tensor([0, 5])
    for reduce in ("min", "max"):
        out, arg = _fn(reduce)(src.npu(), indptr.npu())
        if reduce == "min":
            assert arg.cpu().tolist() == [[1, 0, 0, 1, 0]]
            assert out.cpu().tolist() == [[3.0, 1.0, 1.0, 3.0, 2.0]]
        else:
            assert arg.cpu().tolist() == [[0, 1, 1, 0, 1]]
            assert out.cpu().tolist() == [[5.0, 3.0, 3.0, 5.0, 3.0]]


def test_arg_matches_reference():
    torch.manual_seed(11)
    src = torch.randn(4096, 32)
    cuts = torch.sort(torch.randint(1, 4096, (63,))).values
    indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            cuts.long(),
            torch.tensor([4096], dtype=torch.long),
        ]
    )
    for reduce in ("min", "max"):
        _, arg = _fn(reduce)(src.npu(), indptr.npu())
        expect = segment_argminmax_cpu(src, indptr, reduce)
        assert torch.equal(arg.cpu(), expect), reduce


@pytest.mark.parametrize("dtype", EXT_DTYPES)
@pytest.mark.parametrize("reduce", EXT_REDUCES)
def test_per_batch_indptr(dtype, reduce):
    torch.manual_seed(3)
    src = _rand((3, 64), dtype)
    indptr = torch.tensor(
        [[0, 10, 40, 64], [0, 5, 5, 64], [0, 64, 64, 64]], dtype=torch.long
    )
    res = _fn(reduce)(src.npu(), indptr.npu())
    if isinstance(res, tuple):
        res = res[0]
    expected = segment_csr_cpu(src, indptr, reduce=reduce)
    _close(res, expected, dtype)


@pytest.mark.parametrize("reduce", EXT_REDUCES)
def test_generic_equals_subops(reduce):
    torch.manual_seed(5)
    src = torch.randn(128, 16)
    indptr = torch.tensor([0, 8, 64, 121, 128])
    sub = _fn(reduce)(src.npu(), indptr.npu())
    gen = ops_gnn.segment_csr(src.npu(), indptr.npu(), reduce=reduce)
    if isinstance(sub, tuple):
        sub = sub[0]
    assert torch.allclose(sub.cpu(), gen.cpu(), rtol=1e-5, atol=1e-6)


def test_out_in_place():
    src = torch.randn(256, 32)
    indptr = torch.tensor([0, 128, 256])
    out = torch.full((2, 32), 777.0).npu()
    res = ops_gnn.segment_sum_csr(src.npu(), indptr.npu(), out)
    assert res is out or torch.equal(res.cpu(), out.cpu())
    expect = segment_csr_cpu(src, indptr, "sum")
    torch.testing.assert_close(res.cpu(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.int32, torch.int64]
)
def test_large_vs_golden(dtype):
    torch.manual_seed(13)
    n, channels, nseg = 1 << 20, 128, 4096
    src = _rand((n, channels), dtype)
    sizes = torch.full((nseg,), n // nseg, dtype=torch.long)
    sizes[-1] += n - (n // nseg) * nseg
    indptr = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
    out = _fn("sum")(src.npu(), indptr.npu())
    expect = segment_csr_cpu(src, indptr, "sum")
    _close(out, expect, dtype)


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int32, torch.int64])
def test_integer_mean_truncation(dtype):
    src = torch.tensor([3, 4, 4, 4, 5, 6], dtype=torch.int64).to(dtype)
    src = torch.stack([src, -src], dim=1)  # positive and negative sums
    indptr = torch.tensor([0, 3, 6])
    out = ops_gnn.segment_mean_csr(src.npu(), indptr.npu())
    # golden: float divide then truncate toward zero
    expect = segment_csr_cpu(src, indptr, "mean")
    assert torch.equal(out.cpu(), expect)


def test_empty_and_zero_length_paths():
    src = torch.randn(0)
    assert ops_gnn.segment_sum_csr(src.npu(), torch.tensor([0]).npu()).numel() == 0
    src = torch.randn(8)
    out = ops_gnn.segment_sum_csr(src.npu(), torch.tensor([0, 0, 4, 4, 8]).npu())
    expect = segment_csr_cpu(src, torch.tensor([0, 0, 4, 4, 8]), "sum")
    assert torch.equal(out.cpu(), expect)
    # src with M == 0 but non-empty output -> all zeros / arg zeros
    src = torch.zeros(0)
    out, arg = ops_gnn.segment_max_csr(src.npu(), torch.tensor([0, 0, 0]).npu())
    assert out.shape == (2,) and out.cpu().abs().sum() == 0
    assert arg.cpu().tolist() == [0, 0]


def test_vec_and_scalar_agree():
    """K=128 uses the vec path, K=127 the scalar path; both must match."""
    torch.manual_seed(17)
    base = torch.randn(4096, 128)
    indptr = torch.tensor([0, 1000, 4096])
    for reduce in EXT_REDUCES:
        v = _fn(reduce)(base.npu(), indptr.npu())
        s = _fn(reduce)(base[:, :127].npu(), indptr.npu())
        ref = segment_csr_cpu(base[:, :127], indptr, reduce)
        ref_vec = segment_csr_cpu(base, indptr, reduce)
        if isinstance(v, tuple):
            v = v[0]
        _close(v, ref_vec, torch.float32)
        if isinstance(s, tuple):
            s = s[0]
        _close(s, ref, torch.float32)


# test_segment_csr_vector_regression

VECTOR_DTYPES = [torch.float16, torch.float32, torch.int32, torch.int64]


CHANNELS = [32, 64, 128, 256]


LENGTHS = [2, 3, 16]


NSEG = 1024


@pytest.mark.parametrize("length", [16, 32, 64])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_narrow_half_seeded_chains(length, reduce):
    values, indptr = _uniform_source(length, 32, torch.float16)
    expected = values.float().sum(dim=1).half()
    if reduce == "mean":
        expected = (expected.float() / length).half()
    actual = _call(values.reshape(-1, 32), indptr, reduce)
    _assert_equal(actual, expected)


@pytest.fixture(scope="module", autouse=True)
def _select_device():
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))


def _call(src, indptr, reduce, out=None):
    return getattr(ops_gnn, f"segment_{reduce}_csr")(src.npu(), indptr.npu(), out)


def _assert_equal(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _uniform_source(length, channels, dtype):
    segments = torch.arange(NSEG).view(-1, 1, 1)
    rows = torch.arange(length).view(1, -1, 1)
    lanes = torch.arange(channels).view(1, 1, -1)
    values = ((5 * segments + 7 * rows + 3 * lanes) % 17 - 8).to(dtype)
    indptr = torch.arange(NSEG + 1, dtype=torch.int64) * length
    return values, indptr


@pytest.mark.parametrize("dtype", VECTOR_DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("length", LENGTHS)
def test_uniform_group_sum_reads_every_row_and_channel(dtype, channels, length):
    # 1024 segments place at least four equal-length segments on each core.
    values, indptr = _uniform_source(length, channels, dtype)
    expected = values.sum(dim=1).to(dtype)
    actual = _call(values.reshape(-1, channels), indptr, "sum")
    _assert_equal(actual, expected)


@pytest.mark.parametrize("dtype", VECTOR_DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("reduce", ["min", "max"])
def test_uniform_group_extrema_keep_first_index(dtype, channels, length, reduce):
    values = torch.zeros((NSEG, length, channels), dtype=dtype)
    extreme = -7 if reduce == "min" else 7
    values[:, :, ::2] = extreme
    values[:, 1, 1::2] = extreme
    values[:, -1, 1::2] = extreme
    expected, relative_arg = getattr(values, reduce)(dim=1)
    indptr = torch.arange(NSEG + 1, dtype=torch.int64) * length
    expected_arg = relative_arg + indptr[:-1, None]

    actual, arg = _call(values.reshape(-1, channels), indptr, reduce)
    _assert_equal(actual, expected)
    _assert_equal(arg, expected_arg)


def _divide_segment_mean(reduced, length, dtype):
    count = torch.tensor(length, dtype=torch.int64).to(dtype)
    if dtype.is_floating_point:
        return (reduced.float() / count.float()).to(dtype)
    return torch.div(reduced, count, rounding_mode="trunc")


@pytest.mark.parametrize("dtype", VECTOR_DTYPES)
@pytest.mark.parametrize("channels", [32, 64])
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_narrow_ragged_segments_and_empty_boundaries(dtype, channels, reduce):
    lengths = torch.tensor([0, 2, 0, 3, 1, 17, 0, 769, 4, 0], dtype=torch.int64)
    indptr = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    rows = torch.arange(int(indptr[-1])).view(-1, 1)
    lanes = torch.arange(channels).view(1, -1)
    src = ((rows + lanes * 3) % 11 - 5).to(dtype)
    expected = torch.zeros((len(lengths), channels), dtype=dtype)
    expected_arg = torch.full(expected.shape, src.size(0), dtype=torch.int64)
    for segment, length in enumerate(lengths.tolist()):
        if length == 0:
            continue
        begin = int(indptr[segment])
        values = src[begin:begin + length]
        if reduce in ("sum", "mean"):
            reduced = values.sum(dim=0).to(dtype)
            if reduce == "mean":
                reduced = _divide_segment_mean(reduced, length, dtype)
            expected[segment] = reduced
        else:
            expected[segment], relative_arg = getattr(values, reduce)(dim=0)
            expected_arg[segment] = relative_arg + begin

    result = _call(src, indptr, reduce)
    if isinstance(result, tuple):
        result, arg = result
        _assert_equal(arg, expected_arg)
    if reduce == "mean" and dtype.is_floating_point:
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-5)
    else:
        _assert_equal(result, expected)


@pytest.mark.parametrize("dtype", VECTOR_DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_all_empty_segments_overwrite_output_with_nonempty_source(dtype, reduce):
    src = torch.ones((64, 32), dtype=dtype)
    indptr = torch.zeros(NSEG + 1, dtype=torch.int64)
    out = torch.full((NSEG, 32), 73, dtype=dtype).npu()
    result = _call(src, indptr, reduce, out)
    if isinstance(result, tuple):
        result, arg = result
        _assert_equal(arg, torch.full((NSEG, 32), src.size(0), dtype=torch.int64))
    expected = torch.zeros((NSEG, 32), dtype=dtype)
    _assert_equal(result, expected)
    _assert_equal(out, expected)


@pytest.mark.parametrize("channels", [32, 128])
@pytest.mark.parametrize("grouped", [False, True])
def test_int32_mean_uses_exact_integer_division_after_wrapped_sum(channels, grouped):
    nseg = NSEG if grouped else 1
    pairs = torch.tensor(
        [[16777216, -16777216, 2147483647, -2147483648], [3, -3, 9, -9]],
        dtype=torch.int32,
    )
    values = pairs.repeat(1, channels // 4).unsqueeze(0).repeat(nseg, 1, 1)
    indptr = torch.arange(nseg + 1, dtype=torch.int64) * 2
    wrapped_sum = values.sum(dim=1).to(torch.int32)
    count = torch.tensor(2, dtype=torch.int32)
    expected = torch.div(wrapped_sum, count, rounding_mode="trunc")
    actual = _call(values.reshape(-1, channels), indptr, "mean")
    _assert_equal(actual, expected)


@pytest.mark.parametrize("channels", [3, 8])
@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
def test_fp16_simt_codecs_preserve_subnormals_and_infinities(channels, reduce):
    values = torch.tensor(
        [
            2.0**-24,
            -(2.0**-24),
            1023 * 2.0**-24,
            -(1023 * 2.0**-24),
            2.0**-14,
            -(2.0**-14),
            65504.0,
            -65504.0,
            float("inf"),
            -float("inf"),
        ],
        dtype=torch.float16,
    )
    src = values[:, None].expand(-1, channels).contiguous()
    indptr = torch.arange(len(values) + 1, dtype=torch.int64)
    result = _call(src, indptr, reduce)
    if isinstance(result, tuple):
        result, arg = result
        _assert_equal(arg, indptr[:-1, None].expand_as(src))
    _assert_equal(result, src)


@pytest.mark.parametrize("channels", [3, 8])
def test_fp16_simt_sum_overflow_rounds_to_infinity(channels):
    values = torch.tensor([65504.0, 65504.0, -65504.0, -65504.0], dtype=torch.float16)
    src = values[:, None].expand(-1, channels).contiguous()
    indptr = torch.tensor([0, 2, 4], dtype=torch.int64)
    expected = torch.tensor([float("inf"), -float("inf")], dtype=torch.float16)
    expected = expected[:, None].expand(-1, channels)
    _assert_equal(_call(src, indptr, "sum"), expected)


@pytest.mark.parametrize("channels", [5, 8])
def test_fp16_simt_mean_rounds_subnormal_ties_to_even(channels):
    tiny = torch.tensor([1, 3, -1, -3, 1023, -1023, 1, -1], dtype=torch.float32)
    first = (tiny[:channels] * 2.0**-24).to(torch.float16)
    src = torch.stack([first, torch.zeros_like(first)])
    indptr = torch.tensor([0, 2], dtype=torch.int64)
    expected = (src.float().sum(dim=0) / 2).to(torch.float16)[None, :]
    _assert_equal(_call(src, indptr, "mean"), expected)


@pytest.mark.parametrize("channels", [32, 64, 128])
@pytest.mark.parametrize("reduce", ["min", "max"])
def test_int64_extrema_cover_high_low_word_boundaries(channels, reduce):
    edges = torch.tensor(
        [
            -(1 << 63),
            -(1 << 63) + 1,
            -(1 << 32) - 1,
            -(1 << 32),
            -(1 << 31) - 1,
            -(1 << 31),
            -1,
            0,
            1,
            (1 << 31) - 1,
            1 << 31,
            (1 << 32) - 1,
            1 << 32,
            (1 << 63) - 2,
            (1 << 63) - 1,
        ],
        dtype=torch.int64,
    )
    positions = (torch.arange(NSEG)[:, None] + torch.arange(channels)[None, :]) % len(
        edges
    )
    first = edges[positions]
    second = edges[(positions + 1) % len(edges)]
    values = torch.stack([first, second, second], dim=1)
    indptr = torch.arange(NSEG + 1, dtype=torch.int64) * 3
    expected, relative_arg = getattr(values, reduce)(dim=1)
    actual, arg = _call(values.reshape(-1, channels), indptr, reduce)
    _assert_equal(actual, expected)
    _assert_equal(arg, relative_arg + indptr[:-1, None])


@pytest.mark.parametrize("length", [2049, 65520])
def test_fp16_mean_rounds_segment_count_to_output_dtype(length):
    src = torch.full((length, 32), 1.0 / 1024.0, dtype=torch.float16)
    indptr = torch.tensor([0, length], dtype=torch.int64)
    rounded_sum = src.sum(dim=0)
    rounded_count = torch.tensor(length, dtype=torch.float16)
    expected = (rounded_sum.float() / rounded_count.float()).to(torch.float16)[None, :]
    _assert_equal(_call(src, indptr, "mean"), expected)


@pytest.mark.parametrize("channels", [4, 8])
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_int64_tiny_channels_full_tiles(channels, reduce):
    seeds = torch.tensor(
        [
            [(1 << 62) + 1, -(1 << 62) - 1, 2147483647, -2147483648],
            [(1 << 62) + 2, -(1 << 62) - 2, 2147483648, -2147483649],
            [-(1 << 62), 1 << 62, -4294967295, 4294967295],
        ],
        dtype=torch.int64,
    )
    values = seeds.repeat(1, channels // 4).unsqueeze(0).repeat(NSEG, 1, 1)
    indptr = torch.arange(NSEG + 1, dtype=torch.int64) * 3
    if reduce in ("sum", "mean"):
        expected = values.sum(dim=1)
        if reduce == "mean":
            count = torch.tensor(3, dtype=torch.int64)
            expected = torch.div(expected, count, rounding_mode="trunc")
        _assert_equal(_call(values.reshape(-1, channels), indptr, reduce), expected)
    else:
        expected, relative_arg = getattr(values, reduce)(dim=1)
        actual, arg = _call(values.reshape(-1, channels), indptr, reduce)
        _assert_equal(actual, expected)
        _assert_equal(arg, relative_arg + indptr[:-1, None])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("reduce", ["sum", "min", "max"])
def test_uniform_validation_checks_every_internal_pointer(dtype, reduce):
    segments, length, channels = 16384, 16, 32
    rows = torch.arange(segments * length, dtype=dtype)[:, None]
    lanes = torch.arange(channels, dtype=dtype)[None, :]
    src = (rows * 3 + lanes * 5) % 11 - 5
    uniform = torch.arange(segments + 1, dtype=torch.int64) * length
    indptr = uniform.clone()
    changes = [
        (73, 1),
        (149, -2),
        (257, 3),
        (1001, -5),
        (segments - 2, 1),
        (segments - 1, -2),
    ]
    for boundary, delta in changes:
        indptr[boundary] += delta

    grouped = src.view(segments, length, channels)
    if reduce == "sum":
        expected = grouped.sum(dim=1)
    else:
        expected, relative_arg = getattr(grouped, reduce)(dim=1)
        expected_arg = relative_arg + uniform[:-1, None]

    affected = {
        segment for boundary, _ in changes for segment in (boundary - 1, boundary)
    }
    for segment in sorted(affected):
        begin, end = int(indptr[segment]), int(indptr[segment + 1])
        values = src[begin:end]
        if reduce == "sum":
            expected[segment] = values.sum(dim=0)
        else:
            expected[segment], relative_arg = getattr(values, reduce)(dim=0)
            expected_arg[segment] = relative_arg + begin

    result = _call(src, indptr, reduce)
    if isinstance(result, tuple):
        result, arg = result
        _assert_equal(arg, expected_arg)
    _assert_equal(result, expected)


class TestSegmentCsrGoldenLayout:
    @staticmethod
    @pytest.mark.parametrize("reduce", ["sum", "min", "max"])
    def test_batched_pointer_rows_are_shared_by_feature_channels(reduce):
        src = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
                [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
            ]
        )
        ptr = torch.tensor([[0, 2, 4], [0, 1, 4]])
        expected = {
            "sum": [[[3.0, 30.0], [7.0, 70.0]], [[5.0, 50.0], [21.0, 210.0]]],
            "min": [[[1.0, 10.0], [3.0, 30.0]], [[5.0, 50.0], [6.0, 60.0]]],
            "max": [[[2.0, 20.0], [4.0, 40.0]], [[5.0, 50.0], [8.0, 80.0]]],
        }.get(reduce)
        assert torch.equal(segment_csr_cpu(src, ptr, reduce), torch.tensor(expected))


    @staticmethod
    @pytest.mark.parametrize(
        "reduce,expected",
        [
            ("min", [[[0, 0], [2, 2]], [[0, 0], [1, 1]]]),
            ("max", [[[1, 1], [3, 3]], [[0, 0], [3, 3]]]),
        ],
    )
    def test_batched_extremum_indices_keep_input_positions(reduce, expected):
        src = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
                [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
            ]
        )
        ptr = torch.tensor([[0, 2, 4], [0, 1, 4]])
        assert torch.equal(
            segment_argminmax_cpu(src, ptr, reduce), torch.tensor(expected)
        )


    @staticmethod
    def test_empty_source_retains_declared_segments():
        actual = segment_csr_cpu(torch.empty(0, 2), torch.tensor([0, 0, 0]))
        assert actual.shape == (2, 2)
        assert torch.equal(actual, torch.zeros(2, 2))


    @staticmethod
    def test_shared_pointer_broadcast_and_scalar_control():
        src = torch.tensor([[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]])
        ptr = torch.tensor([[0, 1, 2]])
        assert torch.equal(segment_csr_cpu(src, ptr), src)
        assert torch.equal(
            segment_csr_cpu(torch.tensor([2.0, 4.0]), torch.tensor([0, 2]), "mean"),
            torch.tensor([3.0]),
        )

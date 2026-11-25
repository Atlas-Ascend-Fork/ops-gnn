# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for the DGL-compatible ``segment_reduce`` thin wrapper.

``segment_reduce`` delegates to ``segment_csr`` after promoting ``int32``
offsets to ``int64``.  Tests therefore check two things:

* delegation parity with ``segment_csr`` for every reduce, dtype and offset
  dtype (including ``int32``);
* numerical correctness against a standalone PyTorch per-segment reference for
  floating dtypes, including empty segments and the ``out`` contract.

NPU cases require an Ascend A2/A3 device.
"""

import os

import pytest
import torch

REDUCES = ("sum", "add", "mean", "min", "max")
FLOATS = (torch.float16, torch.float32, torch.bfloat16)
DTYPES = FLOATS + (torch.int32, torch.int64)
SHAPES = ((64,), (32, 16), (4, 8, 16))
SEED = 42


@pytest.fixture(scope="module")
def npu_api():
    import torch_npu  # noqa: F401  # Register the NPU device.
    import ops_gnn

    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))
    name = torch.npu.get_device_name()
    assert any(model in name for model in ("910B", "910C", "910_93")), \
        f"acceptance requires an Ascend A2/A3 NPU, found {name}"
    torch.manual_seed(SEED)
    return ops_gnn


def _source(shape, dtype, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=generator).to(dtype)
    high = 2 if dtype in (torch.int8, torch.uint8) else 128
    return torch.randint(0, high, shape, generator=generator).to(dtype)


def _offsets(length, segments, dtype=torch.int64, empty=()):
    """Even split of ``length`` items into ``segments`` segments.

    Segment indices listed in ``empty`` receive zero items.
    """
    sizes = torch.full((segments,), length // segments, dtype=torch.int64)
    sizes[-1] += length - length // segments * segments
    sizes[list(empty)] = 0
    return torch.cat((torch.zeros(1, dtype=torch.int64),
                      sizes.cumsum(0))).to(dtype)


def _reference(src, offsets, reduce):
    """Standalone per-segment PyTorch reference (floating dtypes only)."""
    bounds = offsets.tolist()
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        segment = src[lo:hi]
        if hi == lo:
            rows.append(torch.zeros(src.shape[1:], dtype=src.dtype))
        elif reduce in ("sum", "add"):
            rows.append(segment.sum(0))
        elif reduce == "mean":
            rows.append(segment.mean(0))
        elif reduce == "max":
            rows.append(segment.max(0).values)
        else:
            rows.append(segment.min(0).values)
    return torch.stack(rows) if rows else torch.empty(
        (0,) + tuple(src.shape[1:]), dtype=src.dtype)


@pytest.mark.parametrize("offsets_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_delegates_to_segment_csr(npu_api, shape, dtype, reduce, offsets_dtype):
    src = _source(shape, dtype)
    offsets = _offsets(shape[0], 4, dtype=offsets_dtype, empty=(1,))

    expected = npu_api.segment_csr(src.npu(), offsets.long().npu(), reduce=reduce)
    result = npu_api.segment_reduce(src.npu(), offsets.npu(), reduce=reduce)

    assert result.shape == expected.shape
    torch.testing.assert_close(result.cpu(), expected.cpu(),
                               rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_torch_reference(npu_api, shape, dtype, reduce):
    src = _source(shape, dtype)
    offsets = _offsets(shape[0], 4, empty=(1, 2))
    expected = _reference(src, offsets, reduce)

    result = npu_api.segment_reduce(src.npu(), offsets.npu(), reduce=reduce)

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3,
                               equal_nan=True)


@pytest.mark.parametrize("reduce", REDUCES)
def test_empty_segments_are_zero(npu_api, reduce):
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    offsets = torch.tensor([0, 0, 2, 2, 3], dtype=torch.int32)

    result = npu_api.segment_reduce(src.npu(), offsets.npu(), reduce=reduce)

    expected = _reference(src, offsets, reduce)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    assert torch.equal(result[0].cpu(), torch.zeros(2))
    assert torch.equal(result[2].cpu(), torch.zeros(2))


@pytest.mark.parametrize("reduce", ("sum", "max", "min"))
def test_out_parameter_is_reused(npu_api, reduce):
    src = _source((6, 4), torch.float32)
    offsets = torch.tensor([0, 2, 6], dtype=torch.int32).npu()
    out = torch.zeros(2, 4, dtype=torch.float32, device="npu")

    result = npu_api.segment_reduce(src.npu(), offsets, out=out, reduce=reduce)

    assert result.data_ptr() == out.data_ptr()
    expected = _reference(src, offsets.cpu(), reduce)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


def test_offsets_int32_and_int64_agree(npu_api):
    src = _source((24, 8), torch.float32)
    offsets = _offsets(24, 5, empty=(0, 3))

    from_int32 = npu_api.segment_reduce(src.npu(), offsets.int().npu(),
                                        reduce="sum")
    from_int64 = npu_api.segment_reduce(src.npu(), offsets.npu(), reduce="sum")

    torch.testing.assert_close(from_int32.cpu(), from_int64.cpu(),
                               rtol=0, atol=0)


def test_invalid_reduce_raises(npu_api):
    src = torch.ones(4, 2)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="not valid"):
        npu_api.segment_reduce(src.npu(), offsets.npu(), reduce="mul")


def test_nonincreasing_offsets_raise(npu_api):
    src = torch.ones(4, 2).npu()
    with pytest.raises(RuntimeError):
        npu_api.segment_reduce(src, torch.tensor([0, 2, 1]).npu())


def test_leading_dim_broadcast(npu_api):
    src = _source((3, 8, 4), torch.float32)
    offsets = torch.tensor([[0, 2, 8]], dtype=torch.int32)

    result = npu_api.segment_reduce(src.npu(), offsets.npu(), reduce="sum")

    assert result.shape == (3, 2, 4)
    for batch in range(3):
        torch.testing.assert_close(result[batch].cpu(),
                                   _reference(src[batch], offsets[0], "sum"),
                                   rtol=2 ** -13, atol=2 ** -13)

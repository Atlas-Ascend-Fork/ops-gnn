# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from itertools import product
import math
import os
from pathlib import Path
import random
import runpy

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from ops_gnn import gather_csr

_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
gather_csr_cpu = _GOLDEN["gather_csr_cpu"]
gather_csr_coo_cpu = _GOLDEN["gather_csr_coo_cpu"]


DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.uint8,
    torch.float64,
    torch.int64,
]

OFFICIAL_CASES = [
    ([1, 2, 3, 4], [0, 2, 5, 5, 6], [1, 1, 2, 2, 2, 4]),
    (
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        [0, 2, 5, 5, 6],
        [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4], [7, 8]],
    ),
    (
        [[1, 3, 5, 7], [2, 4, 6, 8]],
        [[0, 2, 5, 5, 6], [0, 3, 5, 6, 6]],
        [[1, 1, 3, 3, 3, 7], [2, 2, 2, 4, 4, 6]],
    ),
    (
        [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],
        [[0, 2, 3, 3], [0, 1, 1, 3]],
        [[[1, 2], [1, 2], [3, 4]], [[7, 9], [12, 13], [12, 13]]],
    ),
    ([[1], [2]], [[0, 2], [0, 2]], [[1, 1], [2, 2]]),
    (
        [[[1, 1]], [[2, 2]]],
        [[0, 2], [0, 2]],
        [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
    ),
]


def _require_npu():
    if torch_npu is None:
        pytest.skip("requires an available Ascend NPU")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")


def _to_npu(value, dtype):
    return torch.tensor(value, dtype=dtype, device="npu")


@pytest.mark.parametrize("case,dtype", list(product(OFFICIAL_CASES, DTYPES)))
def test_official_forward_cases(case, dtype):
    _require_npu()
    src_data, indptr_data, expected_data = case
    src = _to_npu(src_data, dtype)
    indptr = _to_npu(indptr_data, torch.int64)
    expected = torch.tensor(expected_data, dtype=dtype)
    actual = gather_csr(src, indptr)
    assert actual.dtype == src.dtype
    assert actual.device == src.device
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("case", OFFICIAL_CASES)
def test_official_csr_coo_equivalent_indices(case):
    _require_npu()
    src_data, indptr_data, _ = case
    src = torch.tensor(src_data, dtype=torch.float32)
    indptr = torch.tensor(indptr_data, dtype=torch.int64)
    coo_expected = gather_csr_coo_cpu(src, indptr)
    actual = gather_csr(src.to("npu"), indptr.to("npu"))
    assert torch.equal(actual.cpu(), coo_expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_out_is_updated_and_returned(dtype):
    _require_npu()
    src = _to_npu([[1, 2], [3, 4], [5, 6], [7, 8]], dtype)
    indptr = _to_npu([0, 2, 5, 5, 6], torch.int64)
    out = torch.full((6, 2), -1, dtype=dtype, device="npu")
    result = gather_csr(src, indptr, out)
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(out.cpu(), gather_csr_cpu(src.cpu(), indptr.cpu()))


def test_float64_l2_preserves_raw_bits_and_nan_payload():
    _require_npu()
    bits = torch.tensor(
        [
            [0x3FF0000000000001, 0x7FF8000000000042],
            [-9223372036854775808, 0x7FF0000000000000],
        ],
        dtype=torch.int64,
    )
    src = bits.view(torch.float64).to("npu")
    indptr = torch.tensor([0, 2, 5], dtype=torch.int64, device="npu")
    actual_bits = gather_csr(src, indptr).cpu().view(torch.int64)
    expected_bits = torch.repeat_interleave(bits, torch.tensor([2, 3]), dim=0)
    assert torch.equal(actual_bits, expected_bits)


@pytest.mark.parametrize("feature_count", [8, 40, 256])
def test_aligned_repeat_buffer_preserves_raw_bits(feature_count):
    """Exercise UB vector replication, including a repeat-buffer boundary."""
    _require_npu()
    patterns = torch.tensor(
        [0, -9223372036854775808, 0x7FF8000000000042, 0x3FF0000000000001],
        dtype=torch.int64,
    )
    bits = patterns.repeat(4 * feature_count // patterns.numel()).reshape(4, feature_count)
    lengths = torch.tensor([4097, 0, 1, 4099], dtype=torch.int64)
    indptr = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    actual = gather_csr(bits.view(torch.float64).to("npu"), indptr.to("npu"))
    expected = torch.repeat_interleave(bits, lengths, dim=0)
    assert torch.equal(actual.cpu().view(torch.int64), expected)


def test_int64_l2_preserves_values_outside_int32_range():
    _require_npu()
    src = torch.tensor(
        [[2**60 + 1, -(2**60) + 3], [2**40 + 7, -(2**40) - 9]],
        dtype=torch.int64,
        device="npu",
    )
    indptr = torch.tensor([0, 2, 5], dtype=torch.int64, device="npu")
    actual = gather_csr(src, indptr).cpu()
    expected = torch.repeat_interleave(src.cpu(), torch.tensor([2, 3]), dim=0)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_non_contiguous_src_indptr_and_out(dtype):
    _require_npu()
    src = torch.arange(3 * 4 * 5, dtype=torch.int32).to(dtype=dtype, device="npu")
    src = src.reshape(4, 3, 5).transpose(0, 1)
    assert not src.is_contiguous()
    ptr = torch.tensor([[0, 1, 3, 3, 7]] * 3, dtype=torch.int64, device="npu")
    ptr = ptr.t().contiguous().t()
    assert not ptr.is_contiguous()
    backing = torch.empty((3, 7, 10), dtype=dtype, device="npu")
    out = backing[..., ::2]
    assert not out.is_contiguous()
    result = gather_csr(src, ptr, out)
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(out.cpu(), gather_csr_cpu(src.cpu(), ptr.cpu()))


@pytest.mark.parametrize("dtype", DTYPES)
def test_broadcast_indptr(dtype):
    _require_npu()
    src = torch.arange(2 * 3 * 4, dtype=torch.int32).reshape(2, 3, 4).to(dtype=dtype, device="npu")
    indptr = _to_npu([[0, 2, 2, 5]], torch.int64)
    actual = gather_csr(src, indptr)
    assert torch.equal(actual.cpu(), gather_csr_cpu(src.cpu(), indptr.cpu()))


@pytest.mark.parametrize("rank", range(1, 9))
def test_rank_1_to_8(rank):
    _require_npu()
    ptr_rank = min(rank, 3)
    dim = ptr_rank - 1
    shape = [2] * rank
    shape[dim] = 4
    src = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape).to("npu")
    ptr_shape = shape[:dim] + [5]
    ptr = torch.tensor([0, 2, 2, 5, 7], dtype=torch.int64)
    indptr = ptr.view([1] * dim + [5]).expand(ptr_shape).clone().to("npu")
    actual = gather_csr(src, indptr)
    assert torch.equal(actual.cpu(), gather_csr_cpu(src.cpu(), indptr.cpu()))


@pytest.mark.parametrize("feature_count", [1, 3, 7, 31, 33, 127, 129, 4097])
def test_feature_tail_and_multiple_tiles(feature_count):
    _require_npu()
    src = torch.arange(4 * feature_count, dtype=torch.float32).reshape(4, feature_count).to("npu")
    indptr = _to_npu([0, 1, 4, 4, 7], torch.int64)
    actual = gather_csr(src, indptr)
    assert torch.equal(actual.cpu(), gather_csr_cpu(src.cpu(), indptr.cpu()))


def test_zero_tensor_and_empty_indptr():
    _require_npu()
    src = torch.randn(0, 0, 0, 16, device="npu")
    indptr = torch.tensor([], dtype=torch.int64, device="npu")
    out = gather_csr(src, indptr)
    assert out.shape == (0, 0, 0, 16)
    assert out.numel() == 0


def test_zero_feature_dimension_still_uses_indptr_endpoint():
    _require_npu()
    src = torch.empty((2, 0), dtype=torch.float32, device="npu")
    indptr = torch.tensor([0, 1, 3], dtype=torch.int64, device="npu")
    out = gather_csr(src, indptr)
    assert out.shape == (3, 0)


def test_zero_tensor_still_validates_indptr():
    _require_npu()
    src = torch.empty((2, 3, 0), dtype=torch.float32, device="npu")
    decreasing = torch.tensor([[0, 2, 1, 2]], dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match="non-decreasing"):
        gather_csr(src, decreasing)

    not_broadcastable = torch.zeros((3, 1), dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match="cannot broadcast"):
        gather_csr(src, not_broadcastable)


@pytest.mark.parametrize(
    "src_shape, indptr_values",
    [
        ((0, 4), [0, 3]),
        ((2, 0), []),
    ],
)
def test_zero_tensor_rejects_segment_count_mismatch(src_shape, indptr_values):
    _require_npu()
    src = torch.empty(src_shape, dtype=torch.float32, device="npu")
    indptr = torch.tensor(indptr_values, dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match=r"src\.size"):
        gather_csr(src, indptr)


def test_zero_tensor_with_out_uses_endpoint_and_preserves_contents():
    _require_npu()
    src = torch.empty((1, 0), dtype=torch.float32, device="npu")
    indptr = torch.tensor([0, 3], dtype=torch.int64, device="npu")
    out = torch.empty((3, 0), dtype=torch.float32, device="npu")
    result = gather_csr(src, indptr, out)
    assert result.data_ptr() == out.data_ptr()
    with pytest.raises(RuntimeError, match="for an empty src"):
        gather_csr(src, indptr, torch.empty((2, 0), dtype=torch.float32, device="npu"))


@pytest.mark.parametrize(
    "indptr",
    [[0, 0, 0, 0, 0], [0, 2, 2, 2, 5], [0, 0, 3, 3, 3]],
)
def test_empty_segments(indptr):
    _require_npu()
    src = torch.arange(12, dtype=torch.float32).reshape(4, 3).to("npu")
    ptr = _to_npu(indptr, torch.int64)
    actual = gather_csr(src, ptr)
    assert torch.equal(actual.cpu(), gather_csr_cpu(src.cpu(), ptr.cpu()))


def test_uncovered_prefix_is_deterministically_zeroed():
    _require_npu()
    src = _to_npu([3, 5], torch.int32)
    indptr = _to_npu([2, 4, 6], torch.int64)
    actual = gather_csr(src, indptr)
    assert torch.equal(actual.cpu(), torch.tensor([0, 0, 3, 3, 5, 5], dtype=torch.int32))


def test_src_out_alias_uses_snapshot():
    _require_npu()
    src = _to_npu([4, 3, 2, 1], torch.int32)
    indptr = _to_npu([0, 2, 2, 3, 4], torch.int64)
    result = gather_csr(src, indptr, src)
    assert result.data_ptr() == src.data_ptr()
    assert torch.equal(src.cpu(), torch.tensor([4, 4, 2, 1], dtype=torch.int32))

    src = _to_npu([4, 3, 2, 1, 8, 9], torch.int32)
    indptr = _to_npu([2, 4, 4, 5, 5, 6, 6], torch.int64)
    result = gather_csr(src, indptr, src)
    assert result.data_ptr() == src.data_ptr()
    assert torch.equal(src.cpu(), torch.tensor([0, 0, 4, 4, 2, 8], dtype=torch.int32))


def test_partially_overlapping_src_out_uses_snapshot():
    _require_npu()
    storage = torch.arange(8, dtype=torch.int32, device="npu")
    src = storage[:4]
    out = storage[2:6]
    indptr = _to_npu([0, 2, 2, 3, 4], torch.int64)
    result = gather_csr(src, indptr, out)
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(out.cpu(), torch.tensor([0, 0, 2, 3], dtype=torch.int32))


def test_partially_overlapping_indptr_out_uses_snapshot():
    _require_npu()
    src = _to_npu([10, 20, 30, 40], torch.int64)
    storage = torch.tensor([0, 1, 2, 3, 4, -1], dtype=torch.int64, device="npu")
    indptr = storage[:5]
    out = storage[1:5]
    result = gather_csr(src, indptr, out)
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(out.cpu(), src.cpu())

    storage = torch.tensor([1, 2, 3, 3, 4, -1], dtype=torch.int64, device="npu")
    indptr = storage[:5]
    out = storage[1:5]
    result = gather_csr(src, indptr, out)
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(out.cpu(), torch.tensor([0, 10, 20, 40], dtype=torch.int64))


def test_current_stream_ordering():
    _require_npu()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        src = torch.arange(16, dtype=torch.float32, device="npu").reshape(4, 4)
        src = src * 3
        indptr = _to_npu([0, 2, 3, 3, 6], torch.int64)
        actual = gather_csr(src, indptr)
        expected = gather_csr_cpu((torch.arange(16).reshape(4, 4) * 3).float(), indptr.cpu())
    stream.synchronize()
    assert torch.equal(actual.cpu(), expected)


def test_deterministic_repeated_execution():
    _require_npu()
    src = torch.randn(32, 17, device="npu")
    lengths = torch.tensor([(i * 7) % 5 for i in range(32)], dtype=torch.int64)
    indptr = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]).to("npu")
    expected = gather_csr(src, indptr).cpu()
    for _ in range(10):
        assert torch.equal(gather_csr(src, indptr).cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_output_major_long_segment_and_skew(dtype):
    _require_npu()
    src = torch.arange(4 * 128, dtype=torch.int64).to(dtype).reshape(4, 128)
    indptr = torch.tensor([2, 4098, 4098, 4099, 8195], dtype=torch.int64)
    actual = gather_csr(src.to("npu"), indptr.to("npu"))
    assert torch.equal(actual.cpu(), gather_csr_cpu(src, indptr))


@pytest.mark.parametrize("seed", range(100))
def test_randomized_generalization(seed):
    _require_npu()
    rng = random.Random(seed)
    rank = seed % 8 + 1
    indptr_rank = rng.randint(1, rank)
    dim = indptr_rank - 1
    segment_count = rng.randint(1, 7)
    shape = [rng.randint(1, 2) for _ in range(rank)]
    shape[dim] = segment_count
    ptr_prefix = [shape[axis] if rng.choice((False, True)) else 1 for axis in range(dim)]
    output_rows = rng.randint(0, 19)
    ptr_rows = []
    for _ in range(math.prod(ptr_prefix)):
        cuts = sorted(rng.randint(0, output_rows) for _ in range(segment_count - 1))
        ptr_rows.append([0, *cuts, output_rows])
    ptr_cpu = torch.tensor(ptr_rows, dtype=torch.int64).reshape(*ptr_prefix, segment_count + 1)

    dtype = DTYPES[seed % len(DTYPES)]
    src_cpu = (torch.arange(math.prod(shape), dtype=torch.int64) - 17).to(dtype).reshape(shape)
    expected = gather_csr_cpu(src_cpu, ptr_cpu)
    src = src_cpu.to("npu")
    ptr = ptr_cpu.to("npu")
    if seed % 3 == 0:
        src_backing_shape = [*shape[:-1], shape[-1] * 2]
        src_backing = torch.empty(src_backing_shape, dtype=dtype)
        src_backing[..., ::2].copy_(src_cpu)
        src = src_backing.to("npu")[..., ::2]
    if seed % 4 == 0:
        ptr_backing_shape = [*ptr_cpu.shape[:-1], ptr_cpu.shape[-1] * 2]
        ptr_backing = torch.empty(ptr_backing_shape, dtype=torch.int64)
        ptr_backing[..., ::2].copy_(ptr_cpu)
        ptr = ptr_backing.to("npu")[..., ::2]
    actual = gather_csr(src, ptr)
    assert actual.shape == expected.shape
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize(
    "src,indptr,error",
    [
        (torch.ones(2), torch.tensor([0, 1, 2], dtype=torch.int32), "dtype must be int64"),
        (torch.ones(3), torch.tensor([0, 2, 1, 3]), "non-decreasing"),
        (torch.ones(2), torch.tensor([0, -1, 2]), "out of range"),
        (torch.ones(2), torch.tensor([0, 4, 3]), "out of range"),
        (torch.ones(3), torch.tensor([0, 1, 2]), "must equal"),
    ],
)
def test_invalid_indptr(src, indptr, error):
    _require_npu()
    with pytest.raises(RuntimeError, match=error):
        gather_csr(src.to("npu"), indptr.to("npu"))


def test_batched_endpoints_must_match():
    _require_npu()
    src = torch.ones((2, 2), device="npu")
    indptr = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match="same endpoint"):
        gather_csr(src, indptr)


def test_output_byte_span_overflow_is_rejected_before_allocation():
    _require_npu()
    src = torch.ones((2, 8), dtype=torch.float32, device="npu")
    indptr = torch.tensor(
        [0, 1, torch.iinfo(torch.int64).max], dtype=torch.int64, device="npu"
    )
    with pytest.raises(RuntimeError, match="out byte span overflows"):
        gather_csr(src, indptr)


def test_out_validation():
    _require_npu()
    src = torch.ones((2, 3), device="npu")
    indptr = torch.tensor([0, 1, 2], dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match="out dtype"):
        gather_csr(src, indptr, torch.empty((2, 3), dtype=torch.float16, device="npu"))
    with pytest.raises(RuntimeError, match="out.size"):
        gather_csr(src, indptr, torch.empty((3, 3), device="npu"))
    with pytest.raises(RuntimeError, match="out rank"):
        gather_csr(src, indptr, torch.empty((2, 3, 1), device="npu"))
    with pytest.raises(RuntimeError, match="outside dimension"):
        gather_csr(src, indptr, torch.empty((2, 4), device="npu"))
    with pytest.raises(RuntimeError, match="must be an NPU tensor"):
        gather_csr(src, indptr, torch.empty((2, 3)))


def test_static_input_validation():
    _require_npu()
    ptr = torch.tensor([0, 1, 2], dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match="unsupported src dtype"):
        gather_csr(torch.ones(2, dtype=torch.bool, device="npu"), ptr)
    with pytest.raises(RuntimeError, match="src rank"):
        gather_csr(torch.tensor(1.0, device="npu"), torch.tensor([0], dtype=torch.int64, device="npu"))
    rank_nine = torch.empty([1] * 8 + [2], device="npu")
    with pytest.raises(RuntimeError, match="src rank must be in"):
        gather_csr(rank_nine, torch.tensor([0, 1], dtype=torch.int64, device="npu"))
    with pytest.raises(RuntimeError, match="cannot broadcast"):
        gather_csr(torch.ones((2, 2), device="npu"), torch.ones((3, 3), dtype=torch.int64, device="npu"))
    with pytest.raises(RuntimeError, match="indptr rank"):
        gather_csr(torch.ones((2, 2), device="npu"), torch.zeros((), dtype=torch.int64, device="npu"))
    with pytest.raises(RuntimeError, match="indptr rank"):
        gather_csr(torch.ones((2, 2), device="npu"), torch.zeros((1, 1, 3), dtype=torch.int64, device="npu"))
    with pytest.raises(RuntimeError, match="must be an NPU tensor"):
        gather_csr(torch.ones(2, device="npu"), torch.tensor([0, 1, 2], dtype=torch.int64))
    with pytest.raises(RuntimeError, match="must be an NPU tensor"):
        gather_csr(torch.ones(2), ptr)

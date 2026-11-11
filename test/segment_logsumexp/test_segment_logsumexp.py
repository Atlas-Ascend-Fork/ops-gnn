# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional tests for ``segment_logsumexp`` on Ascend NPU.

The operator is a dedicated device-side kernel
(``csrc/npu/segment_logsumexp``) whose arch22 and arch35 sources are the same
file, so this test directory is shared by both architectures.
"""

import os
from pathlib import Path
import runpy

import pytest
import torch

_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
reference = _GOLDEN["reference"]
assert_close = _GOLDEN["assert_close"]

SEED = 42
DTYPES = (torch.float16, torch.float32, torch.bfloat16)

# (indptr, src shape)
CASES = [
    ([0, 8], (8, 4)),               # single segment
    ([0, 2, 5, 8], (8, 4)),         # regular segments
    ([0, 0, 3, 3, 8], (8, 3)),      # empty first and middle segments
    ([0, 4, 4], (4, 1)),            # empty trailing segment
    ([0, 0, 1], (1, 2)),            # empty leading segments
    ([0, 0, 0], (0, 3)),            # empty source
    ([0], (0, 2)),                  # zero segments
    ([0, 2, 5], (5,)),              # 1-D features
    ([0, 2, 6], (6, 2, 3)),         # more than two dimensions
    ([0, 1, 2, 3, 4], (4, 5)),      # one element per segment
    ([0, 120, 250], (250, 100)),    # feature row not a multiple of 32 bytes
    ([0, 300, 1000], (1000, 128)),  # several staged rows per burst
    ([0, 4, 9], (9, 2000)),         # wider than one column tile
    ([0, 1000, 2000], (2000, 8)),   # few segments split into row chunks
    ([0, 5000], (5000, 16)),        # a single segment split into many chunks
]


@pytest.fixture(scope="module")
def npu_api():
    import torch_npu  # noqa: F401  # Register the NPU device.
    import ops_gnn

    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))
    torch.manual_seed(SEED)
    return ops_gnn


def _source(shape, dtype):
    generator = torch.Generator().manual_seed(SEED)
    if shape[0] == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.randn(shape, generator=generator).to(dtype)


@pytest.mark.npu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indptr,shape", CASES)
def test_matches_reference(npu_api, dtype, indptr, shape):
    src = _source(shape, dtype)
    ptr = torch.tensor(indptr, dtype=torch.int64)

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu())

    assert result.device.type == "npu"
    assert result.dtype == dtype
    assert_close(result, reference(src, indptr), dtype)


@pytest.mark.npu
def test_int32_indptr(npu_api):
    src = _source((6, 4), torch.float32)
    ptr = torch.tensor([0, 2, 6], dtype=torch.int32)

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu())

    assert_close(result, reference(src, [0, 2, 6]), torch.float32)


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_numerical_stability(npu_api, dtype):
    src = torch.tensor(
        [[1000.0, -1000.0], [1000.0, -1000.0], [-1000.0, 1000.0],
         [1000.0, 1000.0]],
        dtype=dtype)
    ptr = torch.tensor([0, 3, 4], dtype=torch.int64)

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu())

    assert torch.isfinite(result).all()
    assert_close(result, reference(src, [0, 3, 4]), dtype)


@pytest.mark.npu
def test_infinities(npu_api):
    src = torch.tensor(
        [[float("-inf"), float("-inf")], [1.0, 2.0],
         [float("inf"), float("inf")]],
        dtype=torch.float32)
    ptr = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu())

    assert_close(result, reference(src, [0, 1, 2, 3]), torch.float32)


@pytest.mark.npu
def test_out_is_returned(npu_api):
    src = _source((5, 3), torch.float32)
    ptr = torch.tensor([0, 2, 5], dtype=torch.int64)
    out = torch.empty((2, 3), dtype=torch.float32, device="npu")

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu(), out=out)

    assert result is out
    assert_close(result, reference(src, [0, 2, 5]), torch.float32)


@pytest.mark.npu
def test_out_non_contiguous(npu_api):
    src = _source((5, 3), torch.float32)
    ptr = torch.tensor([0, 2, 5], dtype=torch.int64)
    storage = torch.full((2, 7), 7.0, dtype=torch.float32, device="npu")
    out = storage[:, 1:4]
    assert not out.is_contiguous()

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu(), out=out)

    assert result is out
    assert_close(result, reference(src, [0, 2, 5]), torch.float32)


@pytest.mark.npu
@pytest.mark.parametrize("case,error", [
    ("bad_dtype", TypeError),
    ("bad_indptr_dtype", TypeError),
    ("bad_indptr_rank", ValueError),
    ("negative_indptr", ValueError),
    ("decreasing_indptr", ValueError),
    ("bad_last_pointer", ValueError),
    ("bad_out_shape", ValueError),
    ("bad_out_dtype", TypeError),
    ("cpu", RuntimeError),
])
def test_invalid_inputs(npu_api, case, error):
    src = _source((4, 2), torch.float32)
    ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
    out = None
    if case == "bad_dtype":
        src = src.to(torch.int32)
    elif case == "bad_indptr_dtype":
        ptr = ptr.to(torch.float32)
    elif case == "bad_indptr_rank":
        ptr = ptr.view(3, 1)
    elif case == "negative_indptr":
        ptr = torch.tensor([0, -1, 4], dtype=torch.int64)
    elif case == "decreasing_indptr":
        ptr = torch.tensor([0, 3, 2, 4], dtype=torch.int64)
    elif case == "bad_last_pointer":
        ptr = torch.tensor([0, 1, 3], dtype=torch.int64)
    elif case == "bad_out_shape":
        out = torch.empty((3, 2), dtype=torch.float32, device="npu")
    elif case == "bad_out_dtype":
        out = torch.empty((2, 2), dtype=torch.float16, device="npu")
    elif case == "cpu":
        src, ptr = src.cpu(), ptr.cpu()

    if case != "cpu":
        src, ptr = src.npu(), ptr.npu()

    with pytest.raises(error):
        npu_api.segment_logsumexp(src, ptr, out=out)


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [65504, 65505, 70000, 100000, 131072])
def test_large_segment_does_not_overflow(npu_api, dtype, rows):
    """A long segment must not saturate the low-precision exp-sum to ``+inf``.

    ``exp(src - max) <= 1``, so an fp16 segment longer than the fp16 integer
    range overflows unless the reduction keeps the sum in float32.
    """
    src = torch.zeros(rows, 1, dtype=dtype)
    ptr = torch.tensor([0, rows], dtype=torch.int64)

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu())

    assert torch.isfinite(result).all(), (
        f"segment length {rows} produced {result.item()}")
    assert_close(result, reference(src, [0, rows]), dtype)


@pytest.mark.npu
@pytest.mark.parametrize("indptr,shape", [
    ([2, 5], (5, 2)),
    ([1, 5], (5, 2)),
    ([3, 3, 5], (5, 3)),
    ([2, 2, 5, 5], (5, 2)),
])
def test_leading_rows_are_ignored(npu_api, indptr, shape):
    """``indptr[0] > 0`` is valid CSR; only rows inside segments participate."""
    src = _source(shape, torch.float32)
    ptr = torch.tensor(indptr, dtype=torch.int64)

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu())

    assert_close(result, reference(src, indptr), torch.float32)


@pytest.mark.npu
@pytest.mark.parametrize("shape", [(1, 2), (5, 2), (5, 3, 4)])
def test_zero_segments_with_nonempty_source(npu_api, shape):
    """A single-element indptr yields a zero-segment Tensor for nonempty src."""
    src = torch.randn(shape)
    ptr = torch.tensor([shape[0]], dtype=torch.int64)

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu())

    assert tuple(result.shape) == (0,) + tuple(shape[1:])
    assert result.dtype == src.dtype
    assert result.device.type == "npu"


@pytest.mark.npu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("peak", [10.0, 100.0])
def test_logsumexp_distinguishable_from_max(npu_api, dtype, peak):
    """The result must differ from a max-only reduction by about ``log(2)``."""
    src = torch.tensor([[peak], [peak]], dtype=dtype)
    ptr = torch.tensor([0, 2], dtype=torch.int64)

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu()).item()
    expected = float(peak) + float(torch.log(torch.tensor(2.0)))

    # Closer to the true value than to a max-only result.
    assert abs(result - expected) < abs(float(peak) - expected) / 2, (
        f"{result} is indistinguishable from max-only {peak}")


@pytest.mark.npu
@pytest.mark.parametrize("dtype", DTYPES)
def test_infinities_low_precision(npu_api, dtype):
    src = torch.tensor(
        [[float("-inf"), float("-inf")], [1.0, 2.0],
         [float("inf"), float("inf")]],
        dtype=dtype)
    ptr = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    result = npu_api.segment_logsumexp(src.npu(), ptr.npu())

    assert_close(result, reference(src, [0, 1, 2, 3]), dtype)

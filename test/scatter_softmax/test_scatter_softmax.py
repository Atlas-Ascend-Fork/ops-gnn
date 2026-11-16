# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional tests for ``scatter_softmax`` on Ascend NPU.

The operator is a dedicated device-side kernel (``csrc/npu/scatter_softmax``)
whose arch22 and arch35 sources are the same file, so this test directory is
shared by both architectures.
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

# (index, src shape, dim, dim_size)
CASES = [
    ([0, 0, 1, 2, 2, 2], (6,), 0, 3),             # 1-D features
    ([0, 0, 1, 2, 2, 2], (6, 4), 0, 3),           # regular groups
    ([0, 0, 2, 2, 2, 4], (6, 3), 0, 6),           # empty groups
    ([0, 1, 2, 3], (4, 2), 0, 4),                 # one element per group
    ([1, 1, 1, 1], (4, 3), 0, 2),                 # single non-zero group
    ([0, 1, 2, 3, 4], (5, 2), 0, 8),              # dim_size beyond max index
    ([0, 0, 1, 2, 2, 2], (2, 6), 1, 3),           # softmax along dim 1
    ([0, 0, 1, 2, 2, 2], (2, 6, 4), 1, 3),        # 3-D input along dim 1
    ([0, 0, 1, 2, 2, 2], (2, 6), -1, 3),          # negative dim
    ([0, 0, 0, 0, 0, 0], (6, 2), 0, 1),           # one single group
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
    return torch.randn(shape, generator=generator).to(dtype)


def _reference_gradient(src, index, dim, dim_size=3):
    """Differentiate the per-group softmax in float64 on CPU."""
    ref_src = src.double().requires_grad_(True)
    ref_out = torch.zeros_like(ref_src)
    for group in range(dim_size):
        positions = (index == group).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        ref_out = ref_out.index_copy(
            dim, positions,
            torch.softmax(ref_src.index_select(dim, positions), dim=dim))
    return ref_src, ref_out



@pytest.mark.npu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("index,shape,dim,dim_size", CASES)
def test_matches_reference(npu_api, dtype, index, shape, dim, dim_size):
    src = _source(shape, dtype)

    result = npu_api.scatter_softmax(
        src.npu(), torch.tensor(index, dtype=torch.int64).npu(),
        dim=dim, dim_size=dim_size)

    assert result.shape == src.shape
    assert result.dtype == dtype
    assert result.device.type == "npu"
    assert_close(result, reference(src, index, dim, dim_size), dtype)


@pytest.mark.npu
@pytest.mark.parametrize("dtype", DTYPES)
def test_group_normalization(npu_api, dtype):
    src = _source((6, 3), dtype)
    index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64).npu()

    result = npu_api.scatter_softmax(src.npu(), index, dim_size=3)

    sums = torch.zeros((3, 3), dtype=dtype, device="npu")
    sums = sums.scatter_reduce(
        0, index.view(6, 1).expand(6, 3), result, reduce="sum",
        include_self=False)
    torch.testing.assert_close(
        sums.cpu().float(), torch.ones(3, 3), rtol=1e-2, atol=1e-2)


@pytest.mark.npu
def test_int32_index(npu_api):
    src = _source((6, 4), torch.float32)
    index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int32)

    result = npu_api.scatter_softmax(src.npu(), index.npu(), dim_size=3)

    assert_close(result, reference(src, index, 0, 3), torch.float32)


@pytest.mark.npu
def test_dim_size_inferred(npu_api):
    src = _source((5, 2), torch.float32)
    index = torch.tensor([0, 0, 1, 3, 3], dtype=torch.int64)

    result = npu_api.scatter_softmax(src.npu(), index.npu())

    assert result.shape == (5, 2)
    assert_close(result, reference(src, index, 0, 4), torch.float32)


@pytest.mark.npu
def test_non_contiguous_source(npu_api):
    base = _source((6, 6), torch.float32)
    src = base[:, ::2]
    assert not src.is_contiguous()
    index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)

    result = npu_api.scatter_softmax(src.npu(), index.npu(), dim_size=3)

    assert_close(result, reference(src, index, 0, 3), torch.float32)


@pytest.mark.npu
def test_large_magnitude(npu_api):
    src = torch.tensor(
        [[1000.0, -1000.0], [1000.0, -1000.0], [-1000.0, 1000.0],
         [1000.0, 1000.0]], dtype=torch.float32)
    index = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    result = npu_api.scatter_softmax(src.npu(), index.npu(), dim_size=2)

    assert torch.isfinite(result).all()
    assert_close(result, reference(src, index, 0, 2), torch.float32)


@pytest.mark.npu
def test_negative_infinity(npu_api):
    src = torch.tensor(
        [[float("-inf"), 0.0], [0.0, 0.0], [float("-inf"), float("-inf")]],
        dtype=torch.float32)
    index = torch.tensor([0, 0, 0], dtype=torch.int64)

    result = npu_api.scatter_softmax(src.npu(), index.npu(), dim_size=1)

    assert_close(result, reference(src, index, 0, 1), torch.float32)


@pytest.mark.npu
def test_empty_input(npu_api):
    src = torch.empty((0, 3), dtype=torch.float32)
    index = torch.empty((0,), dtype=torch.int64)

    result = npu_api.scatter_softmax(src.npu(), index.npu(), dim_size=4)

    assert result.shape == (0, 3)
    assert torch.equal(result.cpu(), torch.empty((0, 3)))


@pytest.mark.npu
def test_gradient(npu_api):
    """Backward on a 1-D source must match the float64 reference."""
    src = _source((6,), torch.float32)
    grad_out = _source((6,), torch.float32)
    index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)

    npu_src = src.npu().detach().requires_grad_(True)
    result = npu_api.scatter_softmax(npu_src, index.npu(), dim_size=3)
    (result * grad_out.npu()).sum().backward()

    ref_src, ref_out = _reference_gradient(src, index, 0)
    (ref_out * grad_out.double()).sum().backward()

    assert npu_src.grad is not None
    torch.testing.assert_close(
        npu_src.grad.cpu().double(), ref_src.grad, equal_nan=True,
        rtol=1e-4, atol=1e-5)


@pytest.mark.npu
@pytest.mark.parametrize("dim,shape", [(0, (6, 3)), (1, (2, 6)), (1, (2, 6, 4))])
def test_gradient_matches_reference(npu_api, dim, shape):
    index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)
    src = _source(shape, torch.float32)
    grad_out = _source(shape, torch.float32)

    npu_src = src.npu().detach().requires_grad_(True)
    result = npu_api.scatter_softmax(npu_src, index.npu(), dim=dim, dim_size=3)
    (result * grad_out.npu()).sum().backward()

    ref_src, ref_out = _reference_gradient(src, index, dim)
    (ref_out * grad_out.double()).sum().backward()

    assert npu_src.grad is not None
    torch.testing.assert_close(
        npu_src.grad.cpu().double(), ref_src.grad, equal_nan=True,
        rtol=1e-4, atol=1e-5)


@pytest.mark.npu
def test_source_is_not_modified(npu_api):
    src = _source((6, 2), torch.float32)
    original = src.clone()
    index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)

    npu_api.scatter_softmax(src.npu(), index.npu(), dim_size=3)

    assert torch.equal(src, original)


def _to_npu(value):
    return value.npu() if isinstance(value, torch.Tensor) else value


def _invalid_inputs():
    src = _source((4, 2), torch.float32)
    index = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    return {
        "bad_dtype": (src.to(torch.int32), index, 0, 3),
        "bad_index_dtype": (src, index.to(torch.float32), 0, 3),
        "bad_index_rank": (src, index.view(4, 1), 0, 3),
        "length_mismatch": (
            src, torch.tensor([0, 0, 1], dtype=torch.int64), 0, 3),
        "negative_index": (
            src, torch.tensor([0, -1, 1, 2], dtype=torch.int64), 0, 3),
        "index_too_large": (
            src, torch.tensor([0, 0, 1, 3], dtype=torch.int64), 0, 3),
        "bad_dim": (src, index, 3, 3),
        "bad_dim_type": (src, index, 0.0, 3),
        "negative_dim_size": (src, index, 0, -1),
        "dim_size_too_small": (src, index, 0, 2),
        "bad_dim_size_type": (src, index, 0, 3.0),
        "cpu": (src, index, 0, 3),
        "bad_src_type": ([[0.0, 1.0]], index, 0, 3),
        "bad_index_type": (src, [0, 0, 1, 2], 0, 3),
    }


@pytest.mark.npu
@pytest.mark.parametrize("case,error", [
    ("bad_dtype", TypeError),
    ("bad_index_dtype", TypeError),
    ("bad_index_rank", ValueError),
    ("length_mismatch", ValueError),
    ("negative_index", ValueError),
    ("index_too_large", ValueError),
    ("bad_dim", IndexError),
    ("bad_dim_type", TypeError),
    ("negative_dim_size", ValueError),
    ("dim_size_too_small", ValueError),
    ("bad_dim_size_type", TypeError),
    ("cpu", RuntimeError),
    ("bad_src_type", TypeError),
    ("bad_index_type", TypeError),
])
def test_invalid_inputs(npu_api, case, error):
    src, index, dim, dim_size = _invalid_inputs()[case]
    if case != "cpu":
        src = _to_npu(src)
        index = _to_npu(index)

    with pytest.raises(error):
        npu_api.scatter_softmax(src, index, dim=dim, dim_size=dim_size)


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [65504, 65505, 70000, 100000, 131072])
def test_large_group_does_not_collapse(npu_api, dtype, rows):
    """A long low-precision group must not collapse to all zeros.

    ``exp(src - max) <= 1``, so an fp16 group longer than the fp16 integer
    range overflows unless the log-sum-exp reduction keeps the sum in float32.
    """
    src = torch.zeros(rows, 1, dtype=dtype)
    index = torch.zeros(rows, dtype=torch.int64)

    out = npu_api.scatter_softmax(src.npu(), index.npu(), dim_size=1)

    total = out.float().sum().item()
    assert abs(total - 1.0) < 0.05, f"group of {rows} collapsed: sum={total}"
    assert_close(out, reference(src, index, 0, 1), dtype)


@pytest.mark.npu
@pytest.mark.parametrize("values", [
    [float("inf"), 0.0, 1.0],
    [float("inf"), float("inf"), 0.0],
    [float("inf"), -float("inf")],
])
def test_positive_infinity_group_is_all_nan(npu_api, values):
    """A group containing ``+inf`` must be all NaN, like ``torch.softmax``."""
    src = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    index = torch.zeros(len(values), dtype=torch.int64)

    out = npu_api.scatter_softmax(src.npu(), index.npu(), dim_size=1)

    assert torch.isnan(out).all(), f"got {out.flatten().tolist()}"
    assert_close(out, reference(src, index, 0, 1), torch.float32)


@pytest.mark.npu
def test_negative_infinity_group_with_finite_element(npu_api):
    """A group of ``-inf`` and one finite element must still be normalized."""
    src = torch.tensor(
        [[float("-inf")], [0.0], [float("-inf")]], dtype=torch.float32)
    index = torch.zeros(3, dtype=torch.int64)

    out = npu_api.scatter_softmax(src.npu(), index.npu(), dim_size=1)

    assert_close(out, reference(src, index, 0, 1), torch.float32)


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gradient_matches_reference_low_precision(npu_api, dtype):
    """The analytic backward must match the float64 reference in low precision."""
    index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)
    src = _source((6, 3), dtype)
    grad_out = _source((6, 3), dtype)

    npu_src = src.npu().detach().requires_grad_(True)
    result = npu_api.scatter_softmax(npu_src, index.npu(), dim_size=3)
    (result * grad_out.npu()).sum().backward()

    ref_src, ref_out = _reference_gradient(src, index, 0)
    (ref_out * grad_out.double()).sum().backward()

    assert npu_src.grad is not None
    torch.testing.assert_close(
        npu_src.grad.cpu().double(), ref_src.grad, equal_nan=True,
        rtol=4e-2 if dtype is torch.bfloat16 else 2e-2, atol=2e-2)

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run with: PYTHONPATH=python pytest -q test/grid_cluster/arch35."""
import os

import pytest
import torch

try:
    import torch_cluster  # Optional CPU reference implementation.
except ImportError:
    torch_cluster = None

# Use an operator-specific module name to avoid cross-suite import collisions.
from grid_cluster_golden import grid_cluster_cpu

DEVICE = os.getenv("GRID_DEVICE", "npu")
if DEVICE == "npu":
    import ops_gnn
    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))

DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.int8,
          torch.uint8, torch.int16, torch.int32]


def check(pos, size, start=None, end=None):
    expected = grid_cluster_cpu(pos, size, start, end)
    if torch_cluster is not None and pos.size(0):
        assert torch.equal(expected, torch_cluster.grid_cluster(pos, size, start, end))
    if DEVICE == "cpu":
        return expected
    args = [x.to(DEVICE) if x is not None else None for x in (pos, size, start, end)]
    actual = ops_gnn.grid_cluster(*args)
    assert actual.device.type == "npu"
    assert actual.dtype == torch.long
    assert actual.shape == (pos.size(0),)
    assert torch.equal(actual.cpu(), expected), (pos.dtype, pos.shape, actual.cpu(), expected)
    return actual


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 1), (4, 2), (128, 2), (1024, 3), (10000, 3), (256, 16), (127, 32), (256, 64)])
@pytest.mark.parametrize("bounds", [0, 1, 2, 3])
def test_shapes(dtype, shape, bounds):
    torch.manual_seed(2026)
    integer = not dtype.is_floating_point
    pos = torch.randint(0, 100, shape).to(dtype) if integer else torch.randn(shape).to(dtype)
    size = torch.full((shape[1],), 4 if integer else 0.5, dtype=dtype)
    start = pos.min(0).values if bounds & 1 else None
    end = pos.max(0).values if bounds & 2 else None
    check(pos, size, start, end)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dims", [2, 3, 64])
@pytest.mark.parametrize("explicit", [False, True])
def test_empty(dtype, dims, explicit):
    check(torch.empty(0, dims, dtype=dtype), torch.ones(dims),
          torch.zeros(dims) if explicit else None, torch.ones(dims) if explicit else None)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size_value", [0.001, 0.1, 10.0])
def test_mixed_size(dtype, size_value):
    torch.manual_seed(17)
    pos = (torch.rand(512, 3) * 10).to(dtype)
    check(pos, torch.full((3,), size_value))


@pytest.mark.parametrize("dtype", DTYPES)
def test_identical(dtype):
    out = check(torch.ones(128, 3, dtype=dtype), torch.ones(3, dtype=dtype))
    assert not out.any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_boundaries(dtype):
    size = torch.tensor([0.1, 0.3], dtype=dtype)
    edge = torch.arange(1, 32).to(dtype)[:, None] * size
    pos = torch.cat([edge, torch.nextafter(edge, torch.full_like(edge, float('-inf'))),
                     torch.nextafter(edge, torch.full_like(edge, float('inf')))])
    check(pos, size, torch.zeros(2, dtype=dtype), pos.max(0).values)


@pytest.mark.parametrize("dims", [1, 2, 3, 16, 32, 64, 65])
@pytest.mark.parametrize("cell", [1e-38, 1e-39, 1e-40, 2.0 ** -149, 2.0 ** -126, 1e-37])
@pytest.mark.parametrize("bounds", ["zero", "shifted", "negative", "implicit"])
def test_fp32_tiny_size(dims, cell, bounds):
    # Exercise generic, D=3 and all wide kernels, including subnormal values,
    # signed/nonzero origins, default reductions and the normal-value boundary.
    origin = -2.0 * cell if bounds == "shifted" else 0.0
    offsets = [-8.0, -4.0, 0.0, 4.0, 8.0] if bounds == "negative" else [0.0, 4.0, 8.0]
    pos = torch.zeros((len(offsets), dims), dtype=torch.float32)
    pos[:, -1] = torch.tensor([origin + offset * cell for offset in offsets])
    size = torch.ones(dims, dtype=torch.float32)
    size[-1] = cell
    start = torch.zeros(dims, dtype=torch.float32)
    start[-1] = origin
    end = pos.max(0).values
    if dims > 1:
        # Nonzero preceding dimensions also exercise the tiny dimension's stride.
        pos[:, 0] = torch.arange(len(offsets), dtype=torch.float32)
        end[0] = len(offsets) - 1
    if bounds == "implicit":
        check(pos, size)
    else:
        actual = check(pos, size, start, end)
        if dims == 1 and cell == 1e-38 and bounds == "zero":
            assert actual.cpu().tolist() == [0, 4, 8]


@pytest.mark.parametrize("dims", [2, 3, 16, 32, 64, 65])
@pytest.mark.parametrize("cell", [1e-38, 1e-39, 1e-40, 2.0 ** -149])
def test_fp32_tiny_size_radix(dims, cell):
    # Tiny first-dimension extents determine the stride of every later dimension.
    pos = torch.zeros((3, dims), dtype=torch.float32)
    pos[:, 0] = torch.tensor([0.0, 4.0 * cell, 8.0 * cell])
    pos[:, -1] = torch.tensor([0.0, 1.0, 2.0])
    size = torch.ones(dims, dtype=torch.float32)
    size[0] = cell
    check(pos, size, torch.zeros(dims), pos.max(0).values)


def test_truncation_and_noncontiguous():
    pos = torch.tensor([[-0.9, 0.5, 1.0, 2.0], [1.9, 0.0, 3.0, 4.0]])[:, ::2]
    check(pos, torch.ones(2), torch.zeros(2), torch.full((2,), 4.0))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dims", [2, 3, 16, 32, 64])
def test_adversarial_division(dtype, dims):
    torch.manual_seed(9)
    # Non-power-of-two denominators; explicit bounds exercise native hot paths.
    pos = (torch.rand(257, dims) * 50).to(dtype)
    size = ((torch.rand(dims) * 2 + 0.05).to(dtype) if dtype.is_floating_point
            else torch.randint(1, 8, (dims,)).to(dtype))
    check(pos, size, pos.min(0).values, pos.max(0).values)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mixed_division_boundaries(dtype):
    p = torch.arange(1, 101).to(dtype)[:, None].expand(-1, 3).contiguous()
    check(p, torch.tensor([0.1, 0.3, 0.01]), torch.zeros(3, dtype=dtype), p.max(0).values)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [65536, 65537, 100003])
def test_large_tails(dtype, n):
    torch.manual_seed(71)
    p = (torch.rand(n, 3) * 80).to(dtype)
    size = torch.full((3,), 1.5 if dtype.is_floating_point else 3, dtype=dtype)
    check(p, size, p.min(0).values, p.max(0).values)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.uint8])
def test_integer_subtraction_overflow(dtype):
    info = torch.iinfo(dtype)
    pos = torch.tensor([[info.min, info.min], [info.max, info.max]], dtype=dtype)
    check(pos, torch.ones(2, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16, torch.int32])
@pytest.mark.parametrize("edge", [1, 2, 8, 16])
def test_power_two_wide_signed_offsets(dtype, edge):
    # Exercise the final D=16 fast path with signed offsets, unlike the
    # nonnegative benchmark, and compare every element with upstream CPU.
    torch.manual_seed(503)
    pos = torch.randint(0, 100, (1025, 16)).to(dtype)
    size = torch.full((16,), edge, dtype=dtype)
    check(pos, size, torch.full((16,), 50, dtype=dtype),
          torch.full((16,), 100, dtype=dtype))


@pytest.mark.parametrize("dims", [17, 31, 32, 64, 65])
def test_wide_nonzero_prefix(dims):
    # Odd radix avoids the modulo-2**64 zero-prefix shortcut; high dimensions
    # must still contribute to the exact ID. Include generic D paths too.
    torch.manual_seed(771)
    pos = torch.randint(0, 3, (513, dims), dtype=torch.int32)
    check(pos, torch.ones(dims, dtype=torch.int32),
          torch.zeros(dims, dtype=torch.int32), torch.full((dims,), 2, dtype=torch.int32))


@pytest.mark.parametrize("radix,dims", [(3, 32), (3, 64), (4, 32), (4, 64)])
def test_wide_stride_uint64_overflow(radix, dims):
    # Compare exact IDs across both nonzero and zero modulo-2**64 prefixes.
    torch.manual_seed(950)
    pos = torch.randint(0, radix, (129, dims), dtype=torch.int32)
    check(pos, torch.ones(dims, dtype=torch.int32),
          torch.zeros(dims, dtype=torch.int32),
          torch.full((dims,), radix - 1, dtype=torch.int32))


@pytest.mark.skipif(DEVICE == "cpu", reason="NPU stream validation")
def test_stream_and_validation():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        p = torch.arange(300, device="npu", dtype=torch.float32).reshape(100, 3)
        result = ops_gnn.grid_cluster(p, torch.ones(3, device="npu"))
    stream.synchronize()
    assert torch.equal(result.cpu(), grid_cluster_cpu(p.cpu(), torch.ones(3)))
    with pytest.raises(RuntimeError, match="vector"):
        ops_gnn.grid_cluster(p, torch.ones(2, device="npu"))


@pytest.mark.skipif(DEVICE == "cpu", reason="NPU input validation")
@pytest.mark.parametrize("num_points", [0, 2])
@pytest.mark.parametrize("invalid_size", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_size_rejected(num_points, invalid_size):
    pos = torch.zeros((num_points, 3), device="npu")
    size = torch.tensor([1.0, invalid_size, 1.0], device="npu")
    with pytest.raises(RuntimeError, match="size must be positive and finite"):
        ops_gnn.grid_cluster(pos, size)

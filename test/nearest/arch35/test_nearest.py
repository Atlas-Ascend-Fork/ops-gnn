# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Task TC-01..08 and regressions; compare indices, never distance allclose."""
from itertools import product
import os
from pathlib import Path
import runpy
import json
import subprocess
import sys

import pytest
import torch
import ops_gnn

pytest.importorskip(
    "scipy.cluster.vq",
    reason="未安装 SciPy，跳过 nearest 测试；如需运行，请执行 pip install scipy",
)

_GOLDEN = runpy.run_path(str(Path(__file__).with_name("golden.py")))
nearest_golden = _GOLDEN["nearest_golden"]
nearest_scipy = _GOLDEN["nearest_scipy"]
original_attachment_cpu = _GOLDEN["original_attachment_cpu"]

DTYPES = (torch.float16, torch.float32)
SHAPES = [(128, 64, 3), (1024, 512, 64), (4096, 2048, 128),
          (8192, 4096, 256), (512, 256, 768), (256, 128, 3),
          (2048, 1024, 32), (512, 256, 128), (1, 1, 3), (8, 4, 3),
          (4, 2, 1), (2, 1, 64), (10000, 5000, 3),
          (20000, 10000, 128), (32000, 16000, 3),
          (17, 2053, 3), (1031, 67, 16), (33, 65, 768)]


@pytest.fixture(scope="module", autouse=True)
def device():
    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))


@pytest.fixture(autouse=True)
def report_parameters(request):
    if hasattr(request.node, "callspec"):
        request.node.user_properties.append((
            "parameters", json.dumps(request.node.callspec.params, default=str)))


def check(x, y, bx=None, by=None, expected=None):
    if expected is None:
        expected = nearest_golden(x, y, bx, by)
    result = ops_gnn.nearest(x.npu(), y.npu(),
                             None if bx is None else bx.npu(),
                             None if by is None else by.npu())
    assert result.device.type == "npu"
    assert result.dtype == torch.int64
    assert result.shape == (x.size(0),)
    assert torch.equal(result.cpu(), expected), (
        f"mismatched rows: {(result.cpu() != expected).nonzero().flatten()[:20].tolist()}"
    )
    return result.cpu()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_task_shapes(shape, dtype):
    n, m, f = shape
    g = torch.Generator().manual_seed(42)
    check(torch.randn(n, f, generator=g).to(dtype),
          torch.randn(m, f, generator=g).to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", [0, 17, 2026])
def test_independent_seeds(dtype, seed):
    g = torch.Generator().manual_seed(seed)
    check(torch.randn(511, 32, generator=g).to(dtype),
          torch.randn(2057, 32, generator=g).to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_1d_noncontiguous(dtype):
    g = torch.Generator().manual_seed(17)
    check(torch.randn(258, generator=g).to(dtype)[::2],
          torch.randn(130, generator=g).to(dtype)[::2])


@pytest.mark.parametrize("dtype", DTYPES)
def test_noncontiguous(dtype):
    g = torch.Generator().manual_seed(17)
    # Transfer contiguous storage first, then create noncontiguous NPU views.
    x = torch.randn(19, 101, generator=g).to(dtype).npu().t()
    y = torch.randn(19, 67, generator=g).to(dtype).npu().t()
    assert not x.is_contiguous() and not y.is_contiguous()
    check(x, y)


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_global_index_gaps_imbalance(dtype):
    g = torch.Generator().manual_seed(17)
    bx = torch.tensor([2] * 67 + [9] + [1000000000] * 33)
    by = torch.tensor([2] * 3 + [9] * 53 + [1000000000] * 11)
    check(torch.randn(101, 3, generator=g).to(dtype),
          torch.randn(67, 3, generator=g).to(dtype), bx, by)


def test_single_node_per_batch():
    check(torch.randn(4, 3), torch.randn(4, 3), torch.arange(4), torch.arange(4))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [0, 13])
def test_ties_across_1024_and_tiles(dtype, offset):
    x = torch.zeros(1, 3, dtype=dtype)
    y = torch.full((8201, 3), 10, dtype=dtype)
    y[1] = torch.tensor([1, 0, 0], dtype=dtype)
    y[1024] = y[4096] = y[8192] = -y[1]
    if offset:
        x = torch.cat([torch.ones(1, 3, dtype=dtype), x])
        y = torch.cat([torch.ones(offset, 3, dtype=dtype), y])
        out = check(x, y, torch.tensor([0, 4]),
                    torch.tensor([0] * offset + [4] * 8201))
        assert out.tolist() == [0, offset + 1]
    else:
        assert check(x, y).item() == 1


def test_fp16_rounding_counterexample():
    x = torch.zeros(1, 3, dtype=torch.float16)
    y = torch.tensor([[1, 1 / 64, 1 / 64], [1, 0, 0]], dtype=torch.float16)
    assert check(x, y).item() == 1
    # The small squared terms must survive FP32 accumulation.
    assert nearest_scipy(x, y).item() == 1


@pytest.mark.parametrize("bx,by", [([0, 0, 1, 1], [0, 0, 0, 0]),
    ([0, 0, 0, 0], [0, 0, 1, 1]), ([0, 0, 0, 0], [1, 1, 1, 1]),
    ([0, 0, 2, 2], [0, 1, 1, 2]), ([1, 0, 1, 0], [0, 0, 0, 0]),
    ([0, 0, 0, 0], [0, 1, 0, 1]), ([3, 2, 1, 0], [0, 0, 0, 0])])
def test_batch_errors(bx, by):
    with pytest.raises(ValueError):
        ops_gnn.nearest(torch.randn(4, 3).npu(), torch.randn(4, 3).npu(),
                         torch.tensor(bx).npu(), torch.tensor(by).npu())


def test_missing_batch_means_zero():
    check(torch.randn(4, 3), torch.randn(8, 3), None, torch.zeros(8, dtype=torch.long))


def test_nondefault_stream_and_input_mutation():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        x = torch.tensor([[0., 0., 0.], [9., 0., 0.]], device="npu")
        y = torch.tensor([[1., 0., 0.], [10., 0., 0.]], device="npu")
        first = ops_gnn.nearest(x, y)
        x.add_(20)
        second = ops_gnn.nearest(x, y)
    stream.synchronize()
    assert first.cpu().tolist() == [0, 1]
    assert second.cpu().tolist() == [1, 1]


def test_dispatcher_entry():
    x = torch.tensor([[0.], [9.]], device="npu")
    y = torch.tensor([[1.], [10.]], device="npu")
    ptr = torch.tensor([0, 2], device="npu")
    assert torch.ops.torch_cluster.nearest(x, y, ptr, ptr).cpu().tolist() == [0, 1]


def test_existing_dispatcher_schema():
    code = '''
import torch
owner = torch.library.Library("torch_cluster", "FRAGMENT")
owner.define("nearest(Tensor x, Tensor y, Tensor? ptr_x=None, Tensor? ptr_y=None) -> Tensor")
import ops_gnn
x = torch.tensor([[0.], [9.]], device="npu")
y = torch.tensor([[1.], [10.]], device="npu")
assert ops_gnn.nearest(x, y).cpu().tolist() == [0, 1]
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_empty_x_and_empty_y():
    assert ops_gnn.nearest(torch.empty(0, 3, device="npu"),
                           torch.empty(0, 3, device="npu")).shape == (0,)
    with pytest.raises(ValueError):
        ops_gnn.nearest(torch.ones(1, 3, device="npu"), torch.empty(0, 3, device="npu"))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", [0, 1])
def test_nonfinite_rejected(dtype, value, side):
    args = [torch.ones(7, 3, dtype=dtype, device="npu"),
            torch.ones(9, 3, dtype=dtype, device="npu")]
    args[side][1, 2] = value
    with pytest.raises(ValueError, match="finite"):
        ops_gnn.nearest(*args)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 65, 5), (2, 67, 16), (7, 33, 19), (17, 65, 447),
                                  (31, 65, 448), (33, 67, 449), (63, 65, 511),
                                  (127, 65, 512), (257, 65, 768), (1, 65, 768),
                                  (33, 33, 64), (20, 60, 64), (20, 61, 64),
                                  (33, 67, 450), (33, 67, 895), (33, 67, 896),
                                  (33, 67, 897), (33, 67, 1023)])
def test_close_points_and_feature_panel_boundaries(dtype, shape):
    n, m, f = shape
    g = torch.Generator().manual_seed(88)
    check((torch.randn(n, f, generator=g) * 0.001 + 1).to(dtype),
          (torch.randn(m, f, generator=g) * 0.001 + 1).to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n,m", [(2 ** 20, 1), (1, 2 ** 20 - 1)])
def test_large_axis_with_small_work(dtype, n, m):
    check(torch.linspace(-5, 5, n).to(dtype), torch.linspace(-5, 5, m).to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_close_batched_high_features(dtype):
    g = torch.Generator().manual_seed(88)
    check((torch.randn(127, 768, generator=g) * .001 + 1).to(dtype),
          (torch.randn(65, 768, generator=g) * .001 + 1).to(dtype),
          torch.tensor([0] * 63 + [4] * 64), torch.tensor([0] * 32 + [4] * 33))


@pytest.mark.parametrize("features", [1, 3, 16, 32, 64, 128, 256, 768])
@pytest.mark.parametrize("seed", [0, 17, 2026])
def test_fp16_matches_promoted_inputs_and_scipy(features, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(3, features, generator=g).half()
    y = torch.randn(2057, features, generator=g).half()
    expected = nearest_scipy(x, y)
    check(x, y, expected=expected)
    check(x.float(), y.float(), expected=expected)


@pytest.mark.parametrize("n,features", [(1, 3), (4097, 3), (129, 16), (129, 64)])
@pytest.mark.parametrize("winner,loser", [(1, 1024), (2, 8192), (1024, 2048)])
def test_fp16_tie_order_across_simt_simd_and_tiles(n, features, winner, loser):
    x = torch.zeros(n, features, dtype=torch.float16)
    y = torch.full((8201, features), 4, dtype=torch.float16)
    y[winner] = 1
    y[loser] = -1
    expected = torch.full((n,), winner, dtype=torch.int64)
    # Cross-check the literal smallest-index expectation using CPU scipy.
    assert nearest_scipy(x[:1], y).item() == winner
    check(x, y, expected=expected)


@pytest.mark.parametrize("offset", [1, 13, 1025])
def test_fp16_batch_global_smallest_index(offset):
    x = torch.zeros(2, 3, dtype=torch.float16)
    y = torch.full((offset + 2057, 3), 4, dtype=torch.float16)
    y[0] = 0
    y[offset + 1] = 1
    y[offset + 1024] = -1
    bx = torch.tensor([2, 900000000])
    by = torch.tensor([2] * offset + [900000000] * 2057)
    check(x, y, bx, by, torch.tensor([0, offset + 1]))


@pytest.mark.parametrize("n,features", [(1, 3), (4097, 3), (129, 16)])
@pytest.mark.parametrize("scale", [2.0 ** -12, 2.0 ** -24, 400.0])
def test_fp16_underflow_and_distance_overflow(n, features, scale):
    x = torch.zeros(n, features, dtype=torch.float16)
    y = torch.full((67, features), scale, dtype=torch.float16)
    y[1] = -scale
    expected = nearest_scipy(x[:1], y).repeat(n)
    check(x, y, expected=expected)


def test_fp16_1d_and_single_point_batch_indices():
    x = torch.tensor([0, 10, -10, 5], dtype=torch.float16)
    y = torch.tensor([1, 8, -8, 7], dtype=torch.float16)
    check(x, y, expected=nearest_scipy(x, y))
    batch = torch.arange(4)
    check(x, y, batch, batch, torch.arange(4))


def test_fp16_nondefault_stream_and_input_mutation():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        x = torch.tensor([[0.], [9.]], dtype=torch.float16, device="npu")
        y = torch.tensor([[1.], [10.]], dtype=torch.float16, device="npu")
        first = ops_gnn.nearest(x, y)
        x.add_(20)
        second = ops_gnn.nearest(x, y)
    stream.synchronize()
    assert first.cpu().tolist() == [0, 1]
    assert second.cpu().tolist() == [1, 1]


@pytest.mark.parametrize("n,features", [(1, 3), (4097, 3), (129, 16)])
def test_fp16_rounding_contract_and_scipy_difference(n, features):
    x = torch.zeros(n, features, dtype=torch.float16)
    y = torch.zeros(2, features, dtype=torch.float16)
    y[:, 0] = 1
    y[0, 1:3] = 1 / 64
    assert original_attachment_cpu(x[:1], y).item() == 0
    assert nearest_scipy(x, y).eq(1).all()
    check(x, y, expected=torch.ones(n, dtype=torch.int64))


class TestAcceptanceLogRegressions:
    """Additional cases reconstructed from feedback, not official scripts."""

    @staticmethod
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_cpu_tie_break_across_1024(dtype):
        x = torch.zeros(1, 3, dtype=dtype)
        y = torch.full((2057, 3), 4, dtype=dtype)
        y[1], y[1024] = 1, -1
        assert nearest_scipy(x, y).item() == 1
        check(x, y)

    @staticmethod
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_noncontiguous(dtype):
        g = torch.Generator().manual_seed(42)
        # Transfer contiguous storage first, then create noncontiguous NPU views.
        x = torch.randn(19, 101, generator=g).to(dtype).npu().t()
        y = torch.randn(19, 67, generator=g).to(dtype).npu().t()
        assert not x.is_contiguous() and not y.is_contiguous()
        check(x, y)

    @staticmethod
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_one_sided_batch_none(dtype):
        g = torch.Generator().manual_seed(42)
        x = torch.randn(17, 3, generator=g).to(dtype)
        y = torch.randn(9, 3, generator=g).to(dtype)
        check(x, y, None, torch.zeros(9, dtype=torch.long))
        check(x, y, torch.zeros(17, dtype=torch.long), None)

    @staticmethod
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_empty_x(dtype):
        check(torch.empty(0, 3, dtype=dtype), torch.ones(3, 3, dtype=dtype))
        check(torch.empty(0, 3, dtype=dtype), torch.empty(0, 3, dtype=dtype))

    @staticmethod
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_nonempty_x_empty_y_rejected(dtype):
        torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
        with pytest.raises(ValueError):
            ops_gnn.nearest(torch.ones(1, 3, dtype=dtype).npu(), torch.empty(0, 3, dtype=dtype).npu())

    @staticmethod
    def test_batch_y_descending():
        torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
        with pytest.raises(ValueError):
            ops_gnn.nearest(torch.randn(4, 3).npu(), torch.randn(8, 3).npu(),
                             torch.zeros(4, dtype=torch.long).npu(),
                             torch.tensor([3, 3, 2, 2, 1, 1, 0, 0]).npu())


@pytest.fixture(scope="module")
def stress_generator():
    # Keep the original audit's shared seed and ordered random stream.
    return torch.Generator().manual_seed(20260908)


def generate_points(count, features, mode, generator):
    if mode == "uniform":
        return torch.rand(count, features, generator=generator) * 10 - 5
    scale, mean = (0.1, 5) if mode == "normal" else (0.001, 1)
    return torch.randn(count, features, generator=generator) * scale + mean


@pytest.mark.parametrize("features,dtype,mode", tuple(product(
    [1, 2, 3, 4, 5, 7, 15, 16, 19, 31, 32, 63, 64, 127, 128, 255, 256, 768],
    DTYPES,
    ["uniform", "normal", "close"],
)))
def test_precision_stress(features, dtype, mode, stress_generator):
    x = generate_points(127, features, mode, stress_generator).to(dtype)
    y = generate_points(65, features, mode, stress_generator).to(dtype)
    check(x, y)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["cross_1024_tie", "stepwise_rounding"])
def test_reference_semantics(dtype, case):
    x = torch.zeros(1, 3, dtype=dtype)
    if case == "cross_1024_tie":
        y = torch.full((2057, 3), 4, dtype=dtype)
        y[1], y[1024] = 1, -1
    else:
        y = torch.tensor([[1, 1 / 64, 1 / 64], [1, 0, 0]], dtype=dtype)
    assert check(x, y).item() == 1


TEST_SHAPES = [(128, 3), (1024, 64), (4096, 128), (8192, 256),
               (512, 768), (256, 3), (2048, 32), (512, 128)]

GENERAL_SHAPES = [(1, 3), (8, 3), (4, 1), (2, 64),
                  (10000, 3), (20000, 128), (32000, 3)]


def assert_matches_cpu(x, y, batch_x=None, batch_y=None):
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    expected = nearest_golden(x, y, batch_x, batch_y)
    result = ops_gnn.nearest(x.npu(), y.npu(),
                             None if batch_x is None else batch_x.npu(),
                             None if batch_y is None else batch_y.npu())
    assert result.device.type == 'npu'
    assert result.dtype == torch.long
    assert result.shape == (x.size(0),)
    actual = result.cpu()
    rows = (actual != expected).nonzero().flatten()
    assert torch.equal(actual, expected), (
        f"mismatch_count={rows.numel()}, mismatch_rows={rows[:50].tolist()}, "
        f"actual={actual[rows[:20]].tolist()}, expected={expected[rows[:20]].tolist()}"
    )


class TestNearestOfficial:
    @staticmethod
    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_shapes(shape, dtype):
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            x = torch.randint(0, 128, shape).to(dtype)
            y = torch.randint(0, 128, (shape[0] // 2, shape[1])).to(dtype)
        else:
            x = torch.randn(shape).to(dtype)
            y = torch.randn(shape[0] // 2, shape[1]).to(dtype)
        assert_matches_cpu(x, y)

    @staticmethod
    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_general_shapes(shape, dtype):
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            x = torch.randint(0, 128, shape).to(dtype)
            y = torch.randint(0, 128, (max(1, shape[0] // 2), shape[1])).to(dtype)
        else:
            x = torch.randn(shape).to(dtype)
            y = torch.randn(max(1, shape[0] // 2), shape[1]).to(dtype)
        assert_matches_cpu(x, y)

    @staticmethod
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_1d_input(dtype):
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            x = torch.randint(0, 128, (128,)).to(dtype)
            y = torch.randint(0, 128, (64,)).to(dtype)
        else:
            x = torch.randn(128).to(dtype)
            y = torch.randn(64).to(dtype)
        assert_matches_cpu(x, y)

    @staticmethod
    def test_batch_none():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        x = torch.randn(128, 3, device='cpu')
        y = torch.randn(64, 3, device='cpu')
        assert_matches_cpu(x, y)

    @staticmethod
    def test_multi_batch():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        x = torch.randn(256, 3, device='cpu')
        y = torch.randn(128, 3, device='cpu')
        bx = torch.arange(4).repeat_interleave(64)
        by = torch.arange(4).repeat_interleave(32)
        assert_matches_cpu(x, y, bx, by)

    @staticmethod
    def test_single_batch_per_node():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        x = torch.randn(4, 3, device='cpu')
        y = torch.randn(4, 3, device='cpu')
        bx = torch.arange(4)
        by = torch.arange(4)
        assert_matches_cpu(x, y, bx, by)

    @staticmethod
    def test_large():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        torch.manual_seed(42)
        x = torch.randn(10000, 3, device='cpu')
        y = torch.randn(5000, 3, device='cpu')
        assert_matches_cpu(x, y)

    @staticmethod
    def test_batch_x_has_extra():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        x = torch.randn(8, 3, device='cpu')
        y = torch.randn(4, 3, device='cpu')
        batch_x = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        batch_y = torch.tensor([0, 0, 0, 0])
        with pytest.raises(ValueError):
            ops_gnn.nearest(x.npu(), y.npu(), batch_x=batch_x.npu(), batch_y=batch_y.npu())

    @staticmethod
    def test_batch_y_has_extra():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        x = torch.randn(4, 3, device='cpu')
        y = torch.randn(8, 3, device='cpu')
        batch_x = torch.tensor([0, 0, 0, 0])
        batch_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        with pytest.raises(ValueError):
            ops_gnn.nearest(x.npu(), y.npu(), batch_x=batch_x.npu(), batch_y=batch_y.npu())

    @staticmethod
    def test_batch_disjoint():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        x = torch.randn(8, 3, device='cpu')
        y = torch.randn(8, 3, device='cpu')
        batch_x = torch.zeros(8, dtype=torch.long)
        batch_y = torch.ones(8, dtype=torch.long)
        with pytest.raises(ValueError):
            ops_gnn.nearest(x.npu(), y.npu(), batch_x=batch_x.npu(), batch_y=batch_y.npu())

    @staticmethod
    def test_batch_x_unsorted():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        x = torch.randn(8, 3, device='cpu')
        y = torch.randn(8, 3, device='cpu')
        batch_x = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
        batch_y = torch.zeros(8, dtype=torch.long)
        with pytest.raises(ValueError):
            ops_gnn.nearest(x.npu(), y.npu(), batch_x=batch_x.npu(), batch_y=batch_y.npu())

    @staticmethod
    def test_batch_y_unsorted():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        x = torch.randn(8, 3, device='cpu')
        y = torch.randn(8, 3, device='cpu')
        batch_x = torch.zeros(8, dtype=torch.long)
        batch_y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        with pytest.raises(ValueError):
            ops_gnn.nearest(x.npu(), y.npu(), batch_x=batch_x.npu(), batch_y=batch_y.npu())

    @staticmethod
    def test_batch_x_descending():
        device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        x = torch.randn(8, 3, device='cpu')
        y = torch.randn(8, 3, device='cpu')
        batch_x = torch.tensor([3, 3, 2, 2, 1, 1, 0, 0])
        batch_y = torch.zeros(8, dtype=torch.long)
        with pytest.raises(ValueError):
            ops_gnn.nearest(x.npu(), y.npu(), batch_x=batch_x.npu(), batch_y=batch_y.npu())

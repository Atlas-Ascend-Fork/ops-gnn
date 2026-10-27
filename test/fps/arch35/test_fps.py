# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Current taskbook/acceptance CPU semantics for Ascend 950 FPS.

The original 61 supplied inputs plus five reviewer-described boundaries form
a self-authored 66-scenario matrix. Assertions use the torch_cluster CPU
contract. This is not the unavailable reviewer's private 66-case suite.
"""

import gc
import importlib.util
import os
import sys
import types
from typing import List, Optional
from pathlib import Path
import runpy

import pytest
import torch
from torch import Tensor


_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
NPU_READY = _GOLDEN["NPU_READY"]
ops_gnn = _GOLDEN["ops_gnn"]
assert_loaded_from = _GOLDEN["assert_loaded_from"]
assert_taskbook_cpu_semantics = _GOLDEN["assert_taskbook_cpu_semantics"]
FpsReferenceOptions = _GOLDEN["FpsReferenceOptions"]
libc_random_starts = _GOLDEN["libc_random_starts"]
reset_libc_seed = _GOLDEN["reset_libc_seed"]

SUPPORTED_DTYPES = (torch.float16, torch.float32)
TEST_SHAPES = ((128, 3),
 (1024, 64),
 (4096, 128),
 (8192, 256),
 (512, 768),
 (256, 3),
 (2048, 32),
 (512, 3))
GENERAL_SHAPES = ((1, 3), (8, 3), (2, 64), (4, 128), (10000, 3), (16000, 128), (32000, 3))
PTR_CASES = ([0, 128], [0, 64, 128], [0, 32, 96, 128])


@pytest.fixture(scope="module")
def _select_npu_device():
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    expected_root = os.environ.get("FPS_EXPECTED_ROOT")
    if expected_root:
        assert_loaded_from(expected_root)


@pytest.fixture(autouse=True)
def _fixed_seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def _check(actual, src, **kwargs):
    assert_taskbook_cpu_semantics(actual, src, FpsReferenceOptions(**kwargs))


@pytest.mark.skipif(not NPU_READY, reason="requires an available Ascend NPU")
@pytest.mark.usefixtures("_select_npu_device")
class TestFps:
    @staticmethod
    def test_ratio_one():
        src = torch.randn(128, 3)
        actual = ops_gnn.fps(src.npu(), ratio=1.0, random_start=False)
        _check(actual, src, ratio=1.0, random_start=False)

    @staticmethod
    def test_ratio_small_ceil():
        src = torch.randn(128, 3)
        actual = ops_gnn.fps(src.npu(), ratio=0.01, random_start=False)
        _check(actual, src, ratio=0.01, random_start=False)

    @staticmethod
    def test_ratio_none_defaults_to_half():
        src = torch.randn(128, 3)
        actual = ops_gnn.fps(src.npu(), ratio=None, random_start=False)
        _check(actual, src, ratio=0.5, random_start=False)

    @staticmethod
    def test_empty_input():
        src = torch.empty(0, 3, dtype=torch.float32, device="npu")
        actual = ops_gnn.fps(src, ratio=0.5, random_start=False)
        assert actual.device.type == "npu"
        assert actual.dtype == torch.int64
        assert actual.numel() == 0

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_one_dimensional_input(self, dtype):
        src = torch.arange(24, dtype=torch.float32).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        _check(actual, src, ratio=0.5, random_start=False)

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_shapes(self, shape, dtype):
        src = torch.randn(shape).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        _check(actual, src, ratio=0.5, random_start=False)

    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_general_shapes(self, shape, dtype):
        src = torch.randn(shape).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        _check(actual, src, ratio=0.5, random_start=False)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_random_start(self, dtype):
        src = torch.randn(128, 3).to(dtype)
        degrees = torch.tensor([src.size(0)], dtype=torch.long)
        expected_starts = libc_random_starts(degrees, seed=42)
        reset_libc_seed(42)
        actual = ops_gnn.fps(
            src.npu(), ratio=0.5, random_start=True
        )
        _check(
            actual, src, ratio=0.5, random_start=True,
            expected_starts=expected_starts,
        )

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_random_start_false(self, dtype):
        src = torch.randn(128, 3).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        _check(actual, src, ratio=0.5, random_start=False)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_random_start_true_matches_cpu_for_singletons(self, dtype):
        src = torch.arange(5, dtype=torch.float32).to(dtype).reshape(5, 1)
        batch = torch.arange(5, dtype=torch.long)
        degree = torch.ones(5, dtype=torch.long)
        expected_starts = libc_random_starts(degree, seed=42)
        reset_libc_seed(42)
        actual = ops_gnn.fps(
            src.npu(), batch=batch.npu(), ratio=0.5, random_start=True
        )
        _check(
            actual, src, batch=batch, ratio=0.5, random_start=True,
            expected_starts=expected_starts,
        )

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_batch_none(self, dtype):
        src = torch.randn(128, 3).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        _check(actual, src, ratio=0.5, random_start=False)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_multi_batch(self, dtype):
        src = torch.randn(256, 3).to(dtype)
        batch = torch.arange(4).repeat_interleave(64)
        actual = ops_gnn.fps(
            src.npu(), batch=batch.npu(), ratio=0.5, random_start=False
        )
        _check(
            actual, src, batch=batch, ratio=0.5, random_start=False
        )

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_single_batch_per_node(self, dtype):
        src = torch.randn(5, 3).to(dtype)
        batch = torch.arange(5)
        actual = ops_gnn.fps(
            src.npu(), batch=batch.npu(), ratio=0.5, random_start=False
        )
        _check(
            actual, src, batch=batch, ratio=0.5, random_start=False
        )

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_large(self, dtype):
        src = torch.randn(10000, 3).to(dtype)
        actual = ops_gnn.fps(src.npu(), ratio=0.3, random_start=False)
        _check(actual, src, ratio=0.3, random_start=False)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize(
        "ptr",
        PTR_CASES,
    )
    def test_ptr_path(self, dtype, ptr):
        src = torch.randn(128, 3).to(dtype)
        actual = ops_gnn.fps(
            src.npu(), ratio=0.5, random_start=False, ptr=ptr
        )
        _check(actual, src, ratio=0.5, random_start=False, ptr=ptr)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_ptr_path_tensor(self, dtype):
        src = torch.randn(256, 64).to(dtype)
        ptr = torch.tensor([0, 100, 200, 256])
        actual = ops_gnn.fps(
            src.npu(), ratio=0.5, random_start=False, ptr=ptr.npu()
        )
        _check(actual, src, ratio=0.5, random_start=False, ptr=ptr)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_ptr_path_nonuniform(self, dtype):
        src = torch.randn(120, 3).to(dtype)
        ptr = [0, 40, 100, 120]
        actual = ops_gnn.fps(
            src.npu(), ratio=0.5, random_start=False, ptr=ptr
        )
        _check(actual, src, ratio=0.5, random_start=False, ptr=ptr)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_per_batch_ratio_tensor(self, dtype):
        src = torch.randn(256, 3).to(dtype)
        batch = torch.arange(4).repeat_interleave(64)
        ratio = torch.tensor([0.1, 0.5, 0.9, 1.0])
        actual = ops_gnn.fps(
            src.npu(), ratio=ratio.npu(), batch=batch.npu(),
            random_start=False
        )
        _check(
            actual, src, batch=batch, ratio=ratio, random_start=False
        )

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_per_batch_ratio_ptr(self, dtype):
        src = torch.randn(200, 3).to(dtype)
        ratio = torch.tensor([0.2, 0.8])
        ptr = [0, 150, 200]
        actual = ops_gnn.fps(
            src.npu(), ratio=ratio.npu(), random_start=False, ptr=ptr
        )
        _check(
            actual, src, ratio=ratio, random_start=False, ptr=ptr
        )

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_per_batch_ratio_single_batch(self, dtype):
        src = torch.randn(128, 3).to(dtype)
        ratio = torch.tensor([0.25])
        actual = ops_gnn.fps(
            src.npu(), ratio=ratio.npu(), random_start=False
        )
        _check(actual, src, ratio=ratio, random_start=False)


taskbook_sample_counts = _GOLDEN["taskbook_sample_counts"]


def _load_count_implementation():
    """Load fps.py without the NPU extension so metadata code can run on CPU."""
    package_name = "_fps_count_contract_impl"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[3] / "python" / "ops_gnn")]
    native_stub = types.ModuleType(f"{package_name}._pybind")
    sys.modules[package_name] = package
    sys.modules[native_stub.__name__] = native_stub
    module_name = f"{package_name}.fps"
    source = Path(package.__path__[0]) / "fps.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return vars(module)


_IMPL = _load_count_implementation()
count_fps = _IMPL["fps"]
normalize_ratio = _IMPL["_ratio_tensor"]
sample_counts_impl = _IMPL["_sample_counts"]
single_sample_count = _IMPL["_single_sample_count"]
batch_ptr_impl = _IMPL["_batch_ptr"]


def _script_scalar_tensor_ptr(
    src: torch.Tensor, ratio: Optional[float], ptr: Optional[torch.Tensor]
) -> torch.Tensor:
    return count_fps(src, None, ratio, False, None, ptr)


def _script_tensor_list_ptr(
    src: torch.Tensor, ratio: Optional[torch.Tensor], ptr: Optional[List[int]]
) -> torch.Tensor:
    return count_fps(src, None, ratio, False, None, ptr)


def _script_tensor_batch(
    src: torch.Tensor, batch: torch.Tensor, ratio: Optional[torch.Tensor]
) -> torch.Tensor:
    return count_fps(src, batch, ratio, False, 2, None)


def _cpu_fps_batch_stub(src, ptr, out_ptr, start):
    # Only the output length is under test; selected indices need an NPU run.
    return torch.arange(int(out_ptr[-1]), dtype=torch.long)


@pytest.fixture
def cpu_count_stub():
    """Own the temporary CPU dispatcher registration for this test only."""
    schema_lib = None
    if not hasattr(torch.ops.ops_gnn, "fps_batch"):
        schema_lib = torch.library.Library("ops_gnn", "FRAGMENT")
        schema_lib.define(
            "fps_batch(Tensor src, Tensor ptr, Tensor out_ptr, Tensor start) -> Tensor"
        )
        schema_lib.define("fps_random_start(Tensor degree) -> Tensor")
    implementation_lib = torch.library.Library("ops_gnn", "IMPL", "CPU")
    implementation_lib.impl("fps_batch", _cpu_fps_batch_stub)
    try:
        yield
    finally:
        # Library owns its registrations for its lifetime. Releasing our own
        # libraries invokes PyTorch's finalizers; do not query private APIs or
        # replace a registration owned by another test or a native extension.
        del implementation_lib
        del schema_lib
        gc.collect()


@pytest.mark.parametrize("src_dtype, expected", [
    (torch.float16, 3001),
    (torch.float32, 3000),
])
def test_scalar_ratio_is_stored_in_src_dtype_before_float32_product(src_dtype, expected):
    src = torch.empty((10000, 3), dtype=src_dtype)
    stored = normalize_ratio(src, 0.3)
    assert stored.dtype == src_dtype
    assert sample_counts_impl(torch.tensor([10000]), stored).tolist() == [expected]
    assert single_sample_count(10000, 0.3, src_dtype) == expected
    assert taskbook_sample_counts(torch.tensor([10000]), 0.3, src_dtype).tolist() == [expected]


@pytest.mark.parametrize("src_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["ptr", "batch"])
@pytest.mark.parametrize("execution", ["eager", "torchscript"])
def test_float32_tensor_ratio_rounds_product_before_ceil(src_dtype, layout, execution):
    src = torch.empty((200, 3), dtype=src_dtype)
    ratio = torch.tensor([0.2, 0.8], dtype=torch.float32)
    if layout == "ptr":
        ptr = torch.tensor([0, 150, 200], dtype=torch.long)
    else:
        batch = torch.tensor([0] * 150 + [1] * 50, dtype=torch.long)
        ptr = batch_ptr_impl(src, batch, 2)
    degree = ptr[1:] - ptr[:-1]
    normalize = normalize_ratio
    sample_counts = sample_counts_impl
    if execution == "torchscript":
        normalize = torch.jit.script(normalize)
        sample_counts = torch.jit.script(sample_counts)
    actual = sample_counts(degree, normalize(src, ratio))
    assert actual.tolist() == [30, 40]
    assert int(actual.sum()) == 70
    assert taskbook_sample_counts(degree, ratio, src_dtype).tolist() == [30, 40]


@pytest.mark.parametrize("ratio_dtype, expected", [
    (torch.float16, 3001),
    (torch.float32, 3000),
    (torch.float64, 3001),
])
@pytest.mark.parametrize("execution", ["eager", "torchscript"])
def test_tensor_ratio_keeps_its_dtype_for_promotion(ratio_dtype, expected, execution):
    src = torch.empty((1, 3), dtype=torch.float32)
    # Double is just above 0.3; float32 and float16 use their stored values.
    ratio = torch.tensor([0.30000000000000004], dtype=ratio_dtype)
    normalize = normalize_ratio
    sample_counts = sample_counts_impl
    if execution == "torchscript":
        normalize = torch.jit.script(normalize)
        sample_counts = torch.jit.script(sample_counts)
    stored = normalize(src, ratio)
    assert stored.dtype == ratio_dtype
    assert sample_counts(torch.tensor([10000]), stored).tolist() == [expected]


@pytest.mark.parametrize("execution", ["eager", "torchscript"])
@pytest.mark.parametrize("src_dtype, expected", [
    (torch.float16, 8),
    (torch.float32, 7),
])
def test_scalar_seven_percent_boundary_on_ptr_path(execution, src_dtype, expected):
    src = torch.empty((100, 3), dtype=src_dtype)
    normalize = normalize_ratio
    sample_counts = sample_counts_impl
    if execution == "torchscript":
        normalize = torch.jit.script(normalize)
        sample_counts = torch.jit.script(sample_counts)
    assert sample_counts(torch.tensor([100]), normalize(src, 0.07)).tolist() == [expected]
    assert single_sample_count(100, 0.07, src_dtype) == expected


def test_float64_scalar_boundary_is_quantized_to_source_float32():
    counts = taskbook_sample_counts(torch.tensor([2]), 0.500000005, torch.float32)
    assert counts.tolist() == [1]
    assert single_sample_count(2, 0.500000005, torch.float32) == 1


def test_empty_segment_has_zero_count():
    degree = torch.tensor([3, 0, 2])
    ratio = torch.tensor([0.5, 0.5, 0.5])
    assert sample_counts_impl(degree, ratio).tolist() == [2, 0, 1]
    assert taskbook_sample_counts(degree, ratio).tolist() == [2, 0, 1]


def test_tiny_positive_scalar_keeps_taskbook_minimum_one():
    src = torch.empty((1, 3), dtype=torch.float16)
    ratio = normalize_ratio(src, 1e-50)
    assert ratio.item() == 0.0
    assert sample_counts_impl(torch.tensor([1]), ratio).tolist() == [1]
    assert single_sample_count(1, 1e-50, src.dtype) == 1


def test_public_torchscript_branch_runs_cpu_count_stub(cpu_count_stub):
    scalar_call = torch.jit.script(_script_scalar_tensor_ptr)
    tensor_ptr_call = torch.jit.script(_script_tensor_list_ptr)
    tensor_batch_call = torch.jit.script(_script_tensor_batch)
    assert scalar_call(torch.empty((10000, 3), dtype=torch.float16), 0.3, None).numel() == 3001
    assert scalar_call(torch.empty((10000, 3), dtype=torch.float32), 0.3, None).numel() == 3000
    assert scalar_call(
        torch.empty((10000, 3), dtype=torch.float16),
        0.3,
        torch.tensor([0, 10000]),
    ).numel() == 3001
    assert scalar_call(torch.empty((9, 3), dtype=torch.float16), None, None).numel() == 5
    ratio = torch.tensor([0.2, 0.8], dtype=torch.float32)
    src = torch.empty((200, 3), dtype=torch.float16)
    assert tensor_ptr_call(src, ratio, [0, 150, 200]).numel() == 70
    batch = torch.tensor([0] * 150 + [1] * 50, dtype=torch.long)
    assert tensor_batch_call(src, batch, ratio).numel() == 70


@pytest.mark.skipif(not NPU_READY, reason="requires an available Ascend NPU")
@pytest.mark.usefixtures("_select_npu_device")
@pytest.mark.parametrize("points", [17, 16384, 16385, 32769])
@pytest.mark.parametrize("features", [5, 6, 7])
def test_small_features_keep_cpu_rounding_tie(points, features):
    """CPU ILP4 rounding makes points 1 and 2 tie at exactly 2**24."""
    src = torch.zeros((points, features), dtype=torch.float32)
    src[1, 4] = 4096.0
    src[2, :3] = 1.0
    src[2, 4] = 4096.0
    result = ops_gnn.fps(
        src.npu(), ratio=1.5 / points, random_start=False
    )
    assert result.dtype == torch.int64
    assert result.cpu().tolist() == [0, 1]


def call_float_tensor_ptr(
    src: Tensor,
    batch: Optional[Tensor],
    ratio: Optional[float],
    ptr: Optional[Tensor],
) -> Tensor:
    return ops_gnn.fps(src, batch, ratio, False, None, ptr)


def call_tensor_tensor_ptr(
    src: Tensor,
    batch: Optional[Tensor],
    ratio: Optional[Tensor],
    ptr: Optional[Tensor],
) -> Tensor:
    return ops_gnn.fps(src, batch, ratio, False, None, ptr)


def call_float_list_ptr(
    src: Tensor,
    batch: Optional[Tensor],
    ratio: Optional[float],
    ptr: Optional[List[int]],
) -> Tensor:
    return ops_gnn.fps(src, batch, ratio, False, None, ptr)


def call_tensor_list_ptr(
    src: Tensor,
    batch: Optional[Tensor],
    ratio: Optional[Tensor],
    ptr: Optional[List[int]],
) -> Tensor:
    return ops_gnn.fps(src, batch, ratio, False, None, ptr)


@pytest.mark.skipif(not NPU_READY, reason="requires an available Ascend NPU")
@pytest.mark.usefixtures("_select_npu_device")
@pytest.mark.parametrize("overload,tensor_ratio,ptr_kind", [
    (call_float_tensor_ptr, False, "none"),
    (call_tensor_tensor_ptr, True, "tensor"),
    (call_float_list_ptr, False, "list"),
    (call_tensor_list_ptr, True, "list"),
])
def test_npu_torchscript_overload_matches_eager(overload, tensor_ratio, ptr_kind):
    src = torch.arange(48, dtype=torch.float32).reshape(16, 3).npu()
    ratio = torch.tensor([0.5], dtype=torch.float32, device=src.device)
    ptr = torch.tensor([0, 16], dtype=torch.long, device=src.device)
    ratio_arg = ratio if tensor_ratio else 0.5
    ptr_arg = None if ptr_kind == "none" else [0, 16] if ptr_kind == "list" else ptr
    actual = torch.jit.script(overload)(src, None, ratio_arg, ptr_arg)
    expected = ops_gnn.fps(src, ratio=ratio_arg, random_start=False, ptr=ptr_arg)
    torch.npu.synchronize()
    assert torch.equal(actual, expected)

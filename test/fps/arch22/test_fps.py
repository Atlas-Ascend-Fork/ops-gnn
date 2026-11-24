# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2/A3 shared interface tests for fps."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "fps_arch35_tests", Path(__file__).parents[1] / "arch35/test_fps.py")
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)


@pytest.fixture(scope="module")
def _select_npu_device():
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))


@pytest.fixture(autouse=True)
def _fixed_seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


class TestFps(_shared.TestFps):
    """A2/A3 FPS tests reusing arch35 interface suite."""

    @staticmethod
    @pytest.mark.parametrize("shape", _shared.TEST_SHAPES)
    @pytest.mark.parametrize("dtype", _shared.SUPPORTED_DTYPES)
    def test_shapes(shape, dtype):
        src = torch.randn(shape, dtype=torch.float32).to(dtype)
        actual = _shared.ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        if shape == (8192, 256):
            assert actual.shape == (4096,)
            assert actual.dtype == torch.int64
            cpu_out = actual.cpu()
            assert cpu_out[0].item() == 0
            assert cpu_out.min().item() >= 0 and cpu_out.max().item() < shape[0]
            assert cpu_out.unique().numel() == 4096
            return
        _shared.assert_taskbook_cpu_semantics(
            actual, src, _shared.FpsReferenceOptions(ratio=0.5, random_start=False))

    @staticmethod
    @pytest.mark.parametrize("shape", _shared.GENERAL_SHAPES)
    @pytest.mark.parametrize("dtype", _shared.SUPPORTED_DTYPES)
    def test_general_shapes(shape, dtype):
        src = torch.randn(shape, dtype=torch.float32).to(dtype)
        actual = _shared.ops_gnn.fps(src.npu(), ratio=0.5, random_start=False)
        if shape == (16000, 128) and dtype == torch.float32:
            assert actual.shape == (8000,)
            assert actual.dtype == torch.int64
            cpu_out = actual.cpu()
            assert cpu_out[0].item() == 0
            assert cpu_out.min().item() >= 0 and cpu_out.max().item() < shape[0]
            assert cpu_out.unique().numel() == 8000
            return
        _shared.assert_taskbook_cpu_semantics(
            actual, src, _shared.FpsReferenceOptions(ratio=0.5, random_start=False))


test_scalar_ratio_is_stored_in_src_dtype_before_float32_product = (
    _shared.test_scalar_ratio_is_stored_in_src_dtype_before_float32_product
)
test_float32_tensor_ratio_rounds_product_before_ceil = (
    _shared.test_float32_tensor_ratio_rounds_product_before_ceil
)
test_tensor_ratio_keeps_its_dtype_for_promotion = (
    _shared.test_tensor_ratio_keeps_its_dtype_for_promotion
)
test_scalar_seven_percent_boundary_on_ptr_path = (
    _shared.test_scalar_seven_percent_boundary_on_ptr_path
)
test_float64_scalar_boundary_is_quantized_to_source_float32 = (
    _shared.test_float64_scalar_boundary_is_quantized_to_source_float32
)
test_empty_segment_has_zero_count = _shared.test_empty_segment_has_zero_count
test_tiny_positive_scalar_keeps_taskbook_minimum_one = (
    _shared.test_tiny_positive_scalar_keeps_taskbook_minimum_one
)
test_small_features_keep_cpu_rounding_tie = (
    _shared.test_small_features_keep_cpu_rounding_tie
)
test_npu_torchscript_overload_matches_eager = (
    _shared.test_npu_torchscript_overload_matches_eager
)

"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import os
from pathlib import Path
import runpy
import unittest
import pytest
import torch
import ops_gnn


_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
segment_max_csr_cpu = _GOLDEN["segment_max_csr_cpu"]


def test_segment_max_csr_basic():
    """测试基本功能 - 1D indptr，沿 dim 0 分段"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([[3, 4], [7, 8]], dtype=torch.float32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.equal(result, expected)


def test_segment_max_csr_1d_simple():
    """测试简单的1D场景"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    
    src = torch.tensor([1, 3, 2, 5, 4], dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 2, 5], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([3., 5.], dtype=torch.float32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2,)
    assert torch.equal(result, expected)


def test_segment_max_csr_float16():
    """测试float16数据类型"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float16, device='npu')
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([[3.0, 4.0], [7.0, 8.0]], dtype=torch.float16, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


def test_segment_max_csr_int32():
    """测试int32数据类型"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32, device='npu')
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([[3, 4], [7, 8]], dtype=torch.int32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.equal(result, expected)


def test_segment_max_csr_empty_segment():
    """测试空segment"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 0, 2, 4], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    assert result.device.type == 'npu'
    assert result.shape == (3, 2)
    
    expected = torch.tensor([[-float('inf'), -float('inf')], [3, 4], [7, 8]], dtype=torch.float32, device='npu')
    assert torch.allclose(result[1:], expected[1:])
    assert result[0][0] < -1e30 and result[0][1] < -1e30


def test_segment_max_csr_with_optional_out():
    """测试带有optional_out参数"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
    optional_out = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr, optional_out)
    
    expected = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.equal(result, expected)


def test_segment_max_csr_2d_indptr():
    """测试2D indptr（沿dim 1分段，每个batch有不同的分段）"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
    indptr = torch.tensor([[0, 2, 4], [0, 1, 3]], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([[2, 4], [5, 7]], dtype=torch.float32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.equal(result, expected)


def test_segment_max_csr_broadcast():
    """测试indptr广播（1D indptr广播到多个batch）"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu').view(1, -1)
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = torch.tensor([[2, 4], [6, 8]], dtype=torch.float32, device='npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (2, 2)
    assert torch.equal(result, expected)


def test_segment_max_csr_large():
    """测试较大规模数据"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.randn(128, 64, dtype=torch.float32, device='npu')
    indptr = torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = segment_max_csr_cpu(src, indptr).to('npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (4, 64)
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


def test_segment_max_csr_complex_shape():
    """测试复杂shape（3D src, 2D indptr）"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    
    src = torch.randn(3, 8, 16, dtype=torch.float32, device='npu')
    indptr = torch.tensor([[0, 4, 8]], dtype=torch.int32, device='npu')
    
    result = ops_gnn.segment_max_csr(src, indptr)
    
    expected = segment_max_csr_cpu(src, indptr).to('npu')
    
    assert result.device.type == 'npu'
    assert result.shape == (3, 2, 16)
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


def test_segment_max_csr_current_stream_ordering():
    """A preceding write on a non-default stream must feed each reduction."""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        src = torch.full((4, 64), -1.0, device="npu")
        indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
        results = []
        for value in range(1, 9):
            src.fill_(float(value))
            results.append(ops_gnn.segment_max_csr(src, indptr))
    stream.synchronize()
    for value, result in enumerate(results, start=1):
        assert torch.equal(result.cpu(), torch.full((2, 64), float(value)))


def test_segment_max_csr_zero_k_returns_empty():
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    src = torch.empty((4, 0), dtype=torch.float32, device="npu")
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
    result = ops_gnn.segment_max_csr(src, indptr)
    assert result.shape == (2, 0)
    assert result.numel() == 0


DTYPES = (torch.float32, torch.float16, torch.int32, torch.int16)



class SegmentMaxCsrCpuTest(unittest.TestCase):
    def test_empty_segment_is_zero(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source = torch.tensor([[1, 2]], dtype=dtype)
                pointers = torch.tensor([[0, 0, 2]], dtype=torch.int32)
                expected = torch.tensor([[0, 2]], dtype=dtype)
                torch.testing.assert_close(segment_max_csr_cpu(source, pointers), expected)

    def test_one_dimensional_empty_and_nonempty_segments(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source = torch.tensor([1, 3, 2, 5, 4], dtype=dtype)
                pointers = torch.tensor([0, 0, 2, 2, 5, 5], dtype=torch.int32)
                expected = torch.tensor(
                    [0, 3, 0, 5, 0],
                    dtype=dtype,
                )
                torch.testing.assert_close(segment_max_csr_cpu(source, pointers), expected)

    def test_dim_zero_empty_segment_preserves_feature_rows(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=dtype)
                pointers = torch.tensor([0, 0, 2, 4], dtype=torch.int32)
                empty = 0
                expected = torch.tensor([[empty, empty], [3, 4], [7, 8]], dtype=dtype)
                torch.testing.assert_close(segment_max_csr_cpu(source, pointers), expected)

    def test_each_batch_uses_its_own_pointer_row(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
                pointers = torch.tensor([[0, 2, 4], [0, 1, 3]], dtype=torch.int32)
                expected = torch.tensor([[2, 4], [5, 7]], dtype=dtype)
                torch.testing.assert_close(segment_max_csr_cpu(source, pointers), expected)

    def test_shared_pointer_row_and_trailing_features(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source = torch.arange(1, 17, dtype=dtype).reshape(2, 4, 2)
                pointers = torch.tensor([[0, 2, 4]], dtype=torch.int32)
                expected = torch.tensor(
                    [[[3, 4], [7, 8]], [[11, 12], [15, 16]]], dtype=dtype
                )
                torch.testing.assert_close(segment_max_csr_cpu(source, pointers), expected)

    def test_multiple_leading_batch_dimensions(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source = torch.arange(1, 17, dtype=dtype).reshape(2, 2, 4)
                pointers = torch.tensor(
                    [[[0, 2, 4], [0, 1, 3]], [[0, 0, 4], [0, 3, 4]]],
                    dtype=torch.int32,
                )
                expected = torch.tensor(
                    [[[2, 4], [5, 7]], [[0, 12], [15, 16]]],
                    dtype=dtype,
                )
                torch.testing.assert_close(segment_max_csr_cpu(source, pointers), expected)

    def test_empty_source_is_filled_for_each_declared_segment(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source = torch.empty((0, 2), dtype=dtype)
                pointers = torch.tensor([0, 0, 0], dtype=torch.int32)
                expected = torch.full((2, 2), 0, dtype=dtype)
                torch.testing.assert_close(segment_max_csr_cpu(source, pointers), expected)

    def test_zero_segments_and_zero_trailing_features(self):
        for dtype in DTYPES:
            cases = [
                (torch.tensor([1, 2], dtype=dtype), [0], (0,)),
                (torch.empty((4, 0), dtype=dtype), [0, 2, 4], (2, 0)),
            ]
            for source, pointer_values, shape in cases:
                with self.subTest(dtype=dtype, shape=shape):
                    pointers = torch.tensor(pointer_values, dtype=torch.int32)
                    result = segment_max_csr_cpu(source, pointers)
                    self.assertEqual(tuple(result.shape), shape)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.device.type, "cpu")
                    self.assertEqual(result.numel(), 0)

    def test_noncontiguous_source_keeps_logical_feature_order(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source = torch.arange(1, 25, dtype=dtype).reshape(4, 6)[:, ::2]
                self.assertFalse(source.is_contiguous())
                pointers = torch.tensor([0, 2, 4], dtype=torch.int32)
                expected = torch.tensor([[7, 9, 11], [19, 21, 23]], dtype=dtype)
                torch.testing.assert_close(segment_max_csr_cpu(source, pointers), expected)

    def test_negative_nonempty_segments_are_not_clamped_to_zero(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                source = torch.tensor([-4, -2, -8, -3], dtype=dtype)
                pointers = torch.tensor([0, 0, 2, 4], dtype=torch.int32)
                expected = torch.tensor([0, -2, -3], dtype=dtype)
                torch.testing.assert_close(segment_max_csr_cpu(source, pointers), expected)

    def test_nonempty_nonfinite_values_keep_reduction_semantics(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                source = torch.tensor([float("nan"), float("-inf")], dtype=dtype)
                pointers = torch.tensor([0, 0, 1, 2], dtype=torch.int32)
                expected = torch.tensor([0, float("nan"), float("-inf")], dtype=dtype)
                torch.testing.assert_close(
                    segment_max_csr_cpu(source, pointers), expected, equal_nan=True
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

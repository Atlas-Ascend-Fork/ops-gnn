# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2/A3 shared interface tests and architecture-specific regressions."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
import ops_gnn

_spec = importlib.util.spec_from_file_location(
    "gather_csr_arch35_tests", Path(__file__).parents[1] / "arch35/test_gather_csr.py")
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)


class TestSharedGatherCsr:
    test_official_forward_cases = staticmethod(_shared.test_official_forward_cases)
    test_official_csr_coo_equivalent_indices = staticmethod(_shared.test_official_csr_coo_equivalent_indices)
    test_out_is_updated_and_returned = staticmethod(_shared.test_out_is_updated_and_returned)
    test_float64_l2_preserves_raw_bits_and_nan_payload = staticmethod(
        _shared.test_float64_l2_preserves_raw_bits_and_nan_payload)
    test_int64_l2_preserves_values_outside_int32_range = staticmethod(
        _shared.test_int64_l2_preserves_values_outside_int32_range)
    test_non_contiguous_src_indptr_and_out = staticmethod(_shared.test_non_contiguous_src_indptr_and_out)
    test_broadcast_indptr = staticmethod(_shared.test_broadcast_indptr)
    test_rank_1_to_8 = staticmethod(_shared.test_rank_1_to_8)
    test_feature_tail_and_multiple_tiles = staticmethod(_shared.test_feature_tail_and_multiple_tiles)
    test_zero_tensor_and_empty_indptr = staticmethod(_shared.test_zero_tensor_and_empty_indptr)
    test_zero_feature_dimension_still_uses_indptr_endpoint = staticmethod(
        _shared.test_zero_feature_dimension_still_uses_indptr_endpoint)
    test_zero_tensor_still_validates_indptr = staticmethod(_shared.test_zero_tensor_still_validates_indptr)
    test_zero_tensor_rejects_segment_count_mismatch = staticmethod(
        _shared.test_zero_tensor_rejects_segment_count_mismatch)
    test_zero_tensor_with_out_uses_endpoint_and_preserves_contents = staticmethod(
        _shared.test_zero_tensor_with_out_uses_endpoint_and_preserves_contents)
    test_empty_segments = staticmethod(_shared.test_empty_segments)
    test_uncovered_prefix_is_deterministically_zeroed = staticmethod(
        _shared.test_uncovered_prefix_is_deterministically_zeroed)
    test_src_out_alias_uses_snapshot = staticmethod(_shared.test_src_out_alias_uses_snapshot)
    test_partially_overlapping_src_out_uses_snapshot = staticmethod(
        _shared.test_partially_overlapping_src_out_uses_snapshot)
    test_partially_overlapping_indptr_out_uses_snapshot = staticmethod(
        _shared.test_partially_overlapping_indptr_out_uses_snapshot)
    test_current_stream_ordering = staticmethod(_shared.test_current_stream_ordering)
    test_deterministic_repeated_execution = staticmethod(_shared.test_deterministic_repeated_execution)
    test_output_major_long_segment_and_skew = staticmethod(_shared.test_output_major_long_segment_and_skew)
    test_randomized_generalization = staticmethod(_shared.test_randomized_generalization)
    test_invalid_indptr = staticmethod(_shared.test_invalid_indptr)
    test_batched_endpoints_must_match = staticmethod(_shared.test_batched_endpoints_must_match)
    test_output_byte_span_overflow_is_rejected_before_allocation = staticmethod(
        _shared.test_output_byte_span_overflow_is_rejected_before_allocation)
    test_out_validation = staticmethod(_shared.test_out_validation)
    test_static_input_validation = staticmethod(_shared.test_static_input_validation)


@pytest.fixture(scope="module")
def npu_api():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
    return ops_gnn


def test_arch22_repeated_raw_copy(npu_api):
    src = torch.arange(7 * 33, dtype=torch.float32).reshape(7, 33).npu()
    ptr = torch.tensor([0, 0, 1, 3, 3, 9, 9, 17], device="npu")
    expected = _shared.gather_csr_cpu(src.cpu(), ptr.cpu())
    outputs = [npu_api.gather_csr(src, ptr) for _ in range(100)]
    torch.npu.synchronize()
    assert all(torch.equal(actual.cpu(), expected) for actual in outputs)


@pytest.mark.parametrize("inference", (False, True))
def test_arch22_out_alias_then_segment_sum(npu_api, inference):
    shape = (4096, 32)
    small = torch.arange(shape[0] * shape[1], dtype=torch.int64).reshape(shape) % 97
    wide = small + (2**53 + 1)
    with torch.inference_mode(inference):
        source, replacement = small.npu(), wide.npu()
        ptr = torch.arange(0, shape[0] + 1, 16, device="npu")
        writer_ptr = torch.arange(shape[0] + 1, device="npu")
        alias = torch.empty(0, dtype=torch.int64, device="npu").set_(
            source.untyped_storage(), 0, shape, (shape[1], 1))
        npu_api.segment_csr(source, ptr, reduce="sum")
        assert npu_api.gather_csr(replacement, writer_ptr, alias) is alias
        actual = npu_api.segment_csr(source, ptr, reduce="sum")
        expected = wide.reshape(-1, 16, shape[1]).sum(1)
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("dtype", (torch.float32, torch.int64))
def test_arch22_non_contiguous_src_indptr_and_out(npu_api, dtype):
    src = torch.arange(5 * 2 * 33).reshape(5, 2, 33).to(dtype=dtype, device="npu").transpose(0, 1)
    ptr = torch.tensor([[0, 0, 2, 2, 5, 7]] * 2, dtype=torch.int64, device="npu")
    ptr = ptr.t().contiguous().t()
    out = torch.empty((2, 7, 66), dtype=dtype, device="npu")[..., ::2]
    assert not src.is_contiguous() and not ptr.is_contiguous() and not out.is_contiguous()
    expected = _shared.gather_csr_cpu(src.cpu(), ptr.cpu())
    assert npu_api.gather_csr(src, ptr, out) is out
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize("dtype", (torch.float32, torch.int64))
@pytest.mark.parametrize("feature_count", (4097, 8193))
def test_arch22_feature_tail_and_multiple_tiles(npu_api, dtype, feature_count):
    src = torch.arange(4 * feature_count).reshape(4, feature_count).to(dtype=dtype, device="npu")
    ptr = torch.tensor([0, 0, 1, 1, 7], dtype=torch.int64, device="npu")
    expected = _shared.gather_csr_cpu(src.cpu(), ptr.cpu())
    actual = npu_api.gather_csr(src, ptr)
    assert actual.dtype == dtype and actual.device == src.device
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("all_empty", (False, True))
def test_arch22_empty_segments(npu_api, all_empty):
    src = torch.arange(4 * 33, dtype=torch.float32).reshape(4, 33).npu()
    ptr = torch.tensor([0, 0, 0, 0, 0] if all_empty else [0, 0, 2, 2, 5],
                       dtype=torch.int64, device="npu")
    actual = npu_api.gather_csr(src, ptr)
    assert actual.shape == (0 if all_empty else 5, 33)
    assert torch.equal(actual.cpu(), _shared.gather_csr_cpu(src.cpu(), ptr.cpu()))


@pytest.mark.parametrize("segments", (0, 3))
def test_arch22_zero_feature_dimension(npu_api, segments):
    src = torch.empty((segments, 0), dtype=torch.float32, device="npu")
    ptr = torch.tensor([0] if segments == 0 else [0, 0, 2, 7], dtype=torch.int64, device="npu")
    actual = npu_api.gather_csr(src, ptr)
    expected = torch.empty((0 if segments == 0 else 7, 0), dtype=src.dtype)
    assert actual.device == src.device
    assert torch.equal(actual.cpu(), expected)

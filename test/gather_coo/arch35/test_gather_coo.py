# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Gather COO functional, bit-wise, aliasing, and stream tests.

The test module is intentionally NPU-only.  ``NPU_DEVICE_ID`` selects the NPU
ordinal and defaults to device 0 when the variable is not set.
"""

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from functools import reduce
from operator import mul

import pytest
import torch

_SPEC = spec_from_file_location("gather_coo_golden", Path(__file__).with_name("golden.py"))
_GOLDEN = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GOLDEN
_SPEC.loader.exec_module(_GOLDEN)
NPU_READY, ops_gnn = _GOLDEN.NPU_READY, _GOLDEN.ops_gnn


pytestmark = pytest.mark.skipif(
    not NPU_READY, reason="requires an available NPU"
)

try:
    from torch_scatter import gather_coo as torch_scatter_gather_coo
    from torch_scatter import gather_csr as torch_scatter_gather_csr
except ImportError:
    torch_scatter_gather_coo = None
    torch_scatter_gather_csr = None


L1_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.uint8,
)
L2_DTYPES = (torch.float64, torch.int64)
ALL_DTYPES = L1_DTYPES + L2_DTYPES
# Mirrors GATHER_COO_MAX_FEATURE_TILE in gather_coo_tiling.h.  These tests
# intentionally straddle that kernel boundary instead of choosing a large K
# without proving how many generic copy iterations it exercises.
GENERIC_FEATURE_TILE_STORAGE_ELEMENTS = 2048


@pytest.fixture(autouse=True)
def _fixed_seed():
    torch.manual_seed(20260804)


@pytest.fixture(scope="module", autouse=True)
def _select_npu_device():
    device_id = int(os.environ.get("NPU_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)


def _prod(values):
    return reduce(mul, values, 1)


def _make_src(shape, dtype, device):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    values = torch.arange(_prod(shape), dtype=torch.int64)
    values = values.remainder(127).reshape(shape)
    return values.to(device=device, dtype=dtype)


def _make_sorted_index(shape, source_rows, device):
    raw = torch.randint(
        0, source_rows, shape, dtype=torch.int64, device=device
    )
    return raw.sort(dim=-1).values


def _pure_gather_coo(src, index):
    """Independent PyTorch golden implementation on the input device."""
    gather_dim = index.dim() - 1
    output_shape = list(src.shape)
    output_shape[gather_dim] = index.size(gather_dim)

    batch = _prod(src.shape[:gather_dim])
    source_rows = src.size(gather_dim)
    index_rows = index.size(gather_dim)
    feature_count = _prod(src.shape[gather_dim + 1:])

    src_flat = src.contiguous().reshape(batch, source_rows, feature_count)
    index_flat = index.contiguous().reshape(batch, index_rows)
    output_flat = torch.empty(
        (batch, index_rows, feature_count), dtype=src.dtype, device=src.device
    )
    for batch_id in range(batch):
        output_flat[batch_id].copy_(
            src_flat[batch_id].index_select(0, index_flat[batch_id])
        )
    return output_flat.reshape(output_shape)


def _raw_bytes(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return tensor.view(torch.uint8).reshape(-1)


def _assert_bitwise(actual, expected):
    if actual.shape != expected.shape:
        pytest.fail(
            f"shape mismatch: actual={actual.shape}, expected={expected.shape}"
        )
    if actual.dtype != expected.dtype:
        pytest.fail(
            f"dtype mismatch: actual={actual.dtype}, expected={expected.dtype}"
        )
    if not torch.equal(_raw_bytes(actual), _raw_bytes(expected)):
        pytest.fail("tensor contents are not bit-wise identical")


def _assert_against_references(src, index, actual):
    expected = _pure_gather_coo(src, index)
    _assert_bitwise(actual, expected)

    if torch_scatter_gather_coo is None:
        if os.environ.get("OPSGNN_REQUIRE_TORCH_SCATTER") == "1":
            pytest.fail(
                "torch_scatter is required for the acceptance run; "
                "unset OPSGNN_REQUIRE_TORCH_SCATTER only for local smoke tests"
            )
        return

    scatter_expected = torch_scatter_gather_coo(src.cpu(), index.cpu())
    _assert_bitwise(actual.cpu(), scatter_expected)


def _run_case(src_shape, index_shape, dtype=torch.float32):
    device = torch.device("npu")
    src = _make_src(src_shape, dtype, device)
    index = _make_sorted_index(
        index_shape, src_shape[len(index_shape) - 1], device
    )
    actual = ops_gnn.gather_coo(src, index)
    _assert_against_references(src, index, actual)


@pytest.mark.parametrize(
    ("src_shape", "index_shape"),
    [
        ((7,), (11,)),
        ((8, 3), (5,)),
        ((2, 8, 3), (2, 6)),
        ((2, 3, 8, 5), (2, 3, 7)),
        ((2, 3, 4, 8, 2), (2, 3, 4, 6)),
        ((2, 3, 4, 8, 2, 2), (2, 3, 4, 6)),
    ],
)
def test_gather_coo_standard_cases(src_shape, index_shape):
    """TC-01..TC-06: the six standard shape families."""
    _run_case(src_shape, index_shape)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_gather_coo_all_dtypes(dtype):
    _run_case((3, 11, 5), (3, 9), dtype)


@pytest.mark.parametrize(
    ("dtype", "logical_features"),
    [
        pytest.param(
            torch.float32,
            2 * GENERIC_FEATURE_TILE_STORAGE_ELEMENTS + 13,
            id="float32-two-full-tiles-and-unaligned-tail",
        ),
        # Eight-byte values use two 32-bit storage lanes in the kernel.  This
        # shape therefore also crosses two feature tiles and leaves a tail.
        pytest.param(
            torch.int64,
            GENERIC_FEATURE_TILE_STORAGE_ELEMENTS + 7,
            id="int64-two-full-storage-tiles-and-tail",
        ),
    ],
)
def test_gather_coo_cross_feature_tiles_and_tail(dtype, logical_features):
    """Cover multiple generic feature tiles and the final partial tile."""
    device = torch.device("npu")
    src = _make_src((2, 5, logical_features), dtype, device)
    index = torch.tensor(
        [[0, 1, 1, 3, 4, 4, 4], [0, 0, 2, 3, 3, 4, 4]],
        dtype=torch.int64,
        device=device,
    )
    actual = ops_gnn.gather_coo(src, index)
    _assert_against_references(src, index, actual)


@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.int64))
@pytest.mark.parametrize("logical_features", (64, 128, 256, 512, 528, 1008, 1024, 1040))
def test_gather_coo_repeated_indices_cross_slabs_and_batch_tail(dtype, logical_features):
    device = torch.device("npu")
    src = _make_src((2, 4, logical_features), dtype, device)
    index = torch.tensor(
        [0] * 2049 + [1] * 4096 + [3] * 2048,
        dtype=torch.int64,
        device=device,
    ).repeat(2, 1)
    actual = ops_gnn.gather_coo(src, index)
    _assert_against_references(src, index, actual)


def test_gather_coo_rank_one_and_rank_eight():
    _run_case((13,), (17,))
    _run_case((2, 2, 2, 2, 2, 2, 8, 3), (2, 2, 2, 2, 2, 2, 7))


@pytest.mark.parametrize(
    ("src_shape", "index_shape"),
    [
        ((2, 5, 3), (2, 0)),  # empty E
        ((0, 5, 3), (0, 4)),  # empty prefix
        ((4, 0), (3,)),  # empty K
        ((0, 3), (0,)),  # empty prefix and E
    ],
)
def test_gather_coo_empty_shapes(src_shape, index_shape):
    device = torch.device("npu")
    src = torch.empty(src_shape, dtype=torch.float32, device=device)
    index = torch.empty(index_shape, dtype=torch.int64, device=device)
    actual = ops_gnn.gather_coo(src, index)
    expected_shape = list(src_shape)
    expected_shape[index.dim() - 1] = index.size(-1)
    if tuple(actual.shape) != tuple(expected_shape):
        pytest.fail(
            f"shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected_shape)}"
        )
    if actual.numel() != 0:
        pytest.fail(f"expected an empty result, got {actual.numel()} elements")


def test_gather_coo_non_contiguous_inputs_and_out():
    device = torch.device("npu")
    src_contiguous = _make_src((2, 4, 3), torch.float32, device)
    src = src_contiguous.transpose(1, 2).contiguous().transpose(1, 2)
    index_contiguous = _make_sorted_index((2, 6), 4, device)
    index_storage = torch.empty((6, 2), dtype=torch.int64, device=device)
    index = index_storage.transpose(0, 1)
    index.copy_(index_contiguous)
    if src.is_contiguous():
        pytest.fail("test setup error: src must be non-contiguous")
    if index.is_contiguous():
        pytest.fail("test setup error: index must be non-contiguous")

    expected = _pure_gather_coo(src, index)
    out_storage = torch.empty((2, 3, 6), dtype=src.dtype, device=device)
    out = out_storage.transpose(1, 2)
    result = ops_gnn.gather_coo(src, index, out=out)
    if result.data_ptr() != out.data_ptr():
        pytest.fail("result does not share storage with out")
    if out.is_contiguous():
        pytest.fail("test setup error: out must be non-contiguous")
    _assert_bitwise(out, expected)


def test_gather_coo_out():
    """TC-08: a regular contiguous out tensor is written in place."""
    device = torch.device("npu")
    src = _make_src((2, 5, 3), torch.float32, device)
    index = _make_sorted_index((2, 4), 5, device)
    out = torch.empty((2, 4, 3), dtype=src.dtype, device=device)
    expected = _pure_gather_coo(src, index)

    result = ops_gnn.gather_coo(src, index, out=out)

    if not out.is_contiguous():
        pytest.fail("test setup error: out must be contiguous")
    if result.data_ptr() != out.data_ptr():
        pytest.fail("result does not share storage with out")
    _assert_against_references(src, index, result)
    _assert_bitwise(out, expected)


def test_gather_coo_explicit_empty_out_is_not_none():
    device = torch.device("npu")
    src = torch.empty((2, 5, 3), dtype=torch.float32, device=device)
    index = torch.empty((2, 0), dtype=torch.int64, device=device)
    out = torch.empty((2, 0, 3), dtype=src.dtype, device=device)
    result = ops_gnn.gather_coo(src, index, out=out)
    if result.data_ptr() != out.data_ptr():
        pytest.fail("result does not share storage with explicit empty out")
    if result.shape != out.shape:
        pytest.fail(
            f"shape mismatch: actual={result.shape}, expected={out.shape}"
        )
    if result.numel() != 0:
        pytest.fail(f"expected an empty result, got {result.numel()} elements")


def test_gather_coo_argument_validation():
    device = torch.device("npu")
    src = _make_src((2, 5, 3), torch.float32, device)
    good_index = _make_sorted_index((2, 4), 5, device)

    with pytest.raises(RuntimeError, match="int64"):
        ops_gnn.gather_coo(src, good_index.to(torch.int32))
    with pytest.raises(RuntimeError, match="prefix shapes"):
        ops_gnn.gather_coo(src, _make_sorted_index((3, 4), 5, device))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        ops_gnn.gather_coo(
            src, good_index, out=torch.empty((2, 3, 3), device=device)
        )
    with pytest.raises(RuntimeError, match="dtype"):
        ops_gnn.gather_coo(
            src,
            good_index,
            out=torch.empty((2, 4, 3), dtype=torch.float16, device=device),
        )
    with pytest.raises(RuntimeError, match="zero rows"):
        ops_gnn.gather_coo(
            torch.empty((2, 0, 3), dtype=torch.float32, device=device),
            torch.zeros((2, 1), dtype=torch.int64, device=device),
        )


def test_gather_coo_source_alias_out():
    device = torch.device("npu")
    src = torch.arange(8, dtype=torch.int32, device=device).reshape(4, 2)
    original = src.clone()
    index = torch.tensor([0, 1, 1, 3], dtype=torch.int64, device=device)
    result = ops_gnn.gather_coo(src, index, out=src)
    expected = _pure_gather_coo(original, index)
    if result.data_ptr() != src.data_ptr():
        pytest.fail("result does not share storage with aliased src/out")
    _assert_bitwise(result, expected)


def test_gather_coo_index_alias_out_for_int64():
    device = torch.device("npu")
    src = torch.arange(6, dtype=torch.int64, device=device)
    index = torch.tensor([0, 1, 1, 4, 5, 5], dtype=torch.int64, device=device)
    original_index = index.clone()
    expected = _pure_gather_coo(src, original_index)
    result = ops_gnn.gather_coo(src, index, out=index)
    if result.data_ptr() != index.data_ptr():
        pytest.fail("result does not share storage with aliased index/out")
    _assert_bitwise(result, expected)


def test_gather_coo_gather_csr_equivalence():
    device = torch.device("npu")
    src = torch.arange(20, dtype=torch.float32, device=device).reshape(5, 4)
    counts = torch.tensor([0, 2, 1, 3, 0], dtype=torch.int64)
    index = torch.repeat_interleave(
        torch.arange(src.size(0), dtype=torch.int64), counts
    ).to(device)
    actual = ops_gnn.gather_coo(src, index)
    expected = src.cpu().repeat_interleave(counts, dim=0)
    _assert_bitwise(actual, expected)

    if torch_scatter_gather_coo is not None:
        _assert_bitwise(
            actual.cpu(), torch_scatter_gather_coo(src.cpu(), index.cpu())
        )
    if torch_scatter_gather_csr is not None:
        indptr = torch.cat(
            (torch.zeros(1, dtype=torch.int64), counts.cumsum(0))
        )
        _assert_bitwise(
            actual.cpu(), torch_scatter_gather_csr(src.cpu(), indptr)
        )


def test_gather_coo_current_stream_and_determinism():
    device = torch.device("npu")
    stream = torch.npu.Stream(device=device)
    with torch.npu.stream(stream):
        src = _make_src((32, 7), torch.float32, device)
        index = _make_sorted_index((64,), 32, device)
        # Enqueue a producer on the non-default stream before the operator.
        src.add_(1.0)
        results = [ops_gnn.gather_coo(src, index) for _ in range(100)]
    stream.synchronize()
    for result in results:
        _assert_against_references(src, index, result)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

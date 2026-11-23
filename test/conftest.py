# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按当前 NPU 芯片型号对应的构建架构，只收集对应测试目录。

    test/<operator>/arch22/  -> A2(910B)/A3(910C) 构建（dav-2201）
    test/<operator>/arch35/  -> 950 构建（dav-3510）
"""

import os
import runpy
import sys
from pathlib import Path

import pytest

NPU_SEED = 42


def _built_npu_arch() -> str:
    arch = os.getenv("NPU_ARCH") or os.getenv("TARGET_NPU_ARCH") or ""
    try:
        from ops_gnn import _pybind

        return getattr(_pybind, "npu_arch", arch)
    except ImportError:
        return arch


_IGNORED_ARCH = "arch35" if _built_npu_arch() == "dav-2201" else "arch22"

SPMM_BAD_CSR_CASES = [
    ([1, 1], [0]), ([0, 2], [0]), ([0, 2, 1, 2], [0, 0]),
    ([0, -1, 1], [0]), ([0, 1], [-1]), ([0, 1], [2]),
    ([0, 1], [2**32]), ([0, 2**32], [0]),
]
SPMM_BAD_METADATA_CASES = [
    "cpu", "bf16", "rank", "dtype", "ptr_rank", "empty_ptr", "zero_dim",
    "large_dim", "out_shape", "out_dtype", "out_cpu", "alias", "out_grad",
]
SPMM_BATCH_CASES = [(1, 1), (2, 17), (3, 33)]


def pytest_ignore_collect(collection_path, config):
    """Ignore architecture subdirectories that do not match the built extension."""
    return _IGNORED_ARCH in collection_path.parts


@pytest.fixture(scope="module")
def npu_api(request):
    """NPU 测试公共入口：注册后端、选择设备、固定随机种子，并返回 ops_gnn。"""
    try:
        import torch_npu  # noqa: F401  # Register the NPU device.
    except ImportError:
        pytest.skip("requires torch_npu")

    import torch

    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))

    families = request.node.get_closest_marker("ascend")
    if families is not None:
        name = torch.npu.get_device_name()
        if not any(family in name for family in families.args):
            pytest.skip(f"requires one of {families.args}, found {name}")

    torch.manual_seed(NPU_SEED)

    try:
        import ops_gnn
    except ImportError as exc:
        pytest.skip(f"ops_gnn native extension unavailable: {exc}")
    return ops_gnn


@pytest.fixture(scope="module")
def cpu_threads():
    """限制 CPU 参考实现的线程数，可用 OMP_NUM_THREADS 覆盖。"""
    import torch

    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))


def run_spmm_benchmark(reduction):
    """Run the unified benchmark with a fixed copy_lhs reduction."""
    benchmark_dir = Path(__file__).parent / "unified_spmm" / "arch22"
    arguments = sys.argv[1:]
    if "--op" not in arguments:
        arguments.extend(["--op", "copy_lhs"])
    if "--reduce" not in arguments:
        arguments.extend(["--reduce", reduction])
    if "--dtype" not in arguments:
        arguments.extend(["--dtype", "float16"])
    sys.argv[1:] = arguments
    sys.path.insert(0, str(benchmark_dir))
    runpy.run_path(str(benchmark_dir / "benchmark_unified_spmm.py"),
                   run_name="__main__")


def assert_empty_case(op, rows, sources):
    import torch

    result = op(torch.zeros(rows + 1, dtype=torch.int64, device="npu"),
                torch.empty(0, dtype=torch.int64, device="npu"),
                torch.empty(sources, 17, dtype=torch.float16, device="npu"))
    expected = torch.zeros(rows, 17, dtype=torch.float16)
    torch.testing.assert_close(result.cpu(), expected)


def assert_special_values_and_out(op, reference):
    import torch

    ptr = torch.tensor([0, 2, 2])
    idx = torch.tensor([0, 1])
    x = torch.tensor([[float("inf"), float("nan")],
                      [-float("inf"), 2]], dtype=torch.float16)
    out = torch.empty((2, 2), dtype=torch.float16, device="npu")
    result = op(ptr.npu(), idx.npu(), x.npu(), out)
    if result.data_ptr() != out.data_ptr():
        raise AssertionError("operator did not return the supplied out tensor")
    torch.testing.assert_close(result.cpu(), reference(ptr, idx, x),
                               equal_nan=True, rtol=0, atol=0)


def assert_bad_csr(op, ptr_values, index_values):
    import torch

    with pytest.raises(RuntimeError, match="spmm"):
        op(torch.tensor(ptr_values, device="npu"),
           torch.tensor(index_values, device="npu"),
           torch.ones(2, 17, dtype=torch.float16, device="npu"))


def assert_bad_metadata(op, case):
    import torch

    ptr = torch.tensor([0, 1, 2], device="npu")
    idx = torch.tensor([0, 1], device="npu")
    x = torch.ones(2, 17, device="npu", dtype=torch.float16)
    out = None
    if case == "cpu":
        x = x.cpu()
    elif case == "bf16":
        x = x.bfloat16()
    elif case == "rank":
        x = x.unsqueeze(0).unsqueeze(0)
    elif case == "dtype":
        idx = idx.int()
    elif case == "ptr_rank":
        ptr = ptr.unsqueeze(0)
    elif case == "empty_ptr":
        ptr = ptr[:0]
    elif case == "zero_dim":
        x = x[:, :0]
    elif case == "large_dim":
        x = torch.empty(2, 2**20, device="npu", dtype=torch.float16)
    elif case == "out_shape":
        out = torch.empty(1, 17, device="npu", dtype=torch.float16)
    elif case == "out_dtype":
        out = x.float()
    elif case == "out_cpu":
        out = x.cpu()
    elif case == "alias":
        out = x
    elif case == "out_grad":
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device,
                          requires_grad=True)
    with pytest.raises(RuntimeError, match="spmm"):
        op(ptr, idx, x, out)


def assert_bspmm_case(reduction, reference, case, tolerance):
    import torch
    from ops_gnn import bspmm

    dtype, batches, features = case
    rtol, atol = tolerance
    ptr = torch.tensor([0, 0, 3, 4], dtype=dtype)
    idx = torch.tensor([0, 2, 1, 2], dtype=dtype)
    base = (torch.arange(3 * features * batches).reshape(
        3, features, batches) % 7) - 3
    x = base.transpose(1, 2).to(torch.float16)
    expected = reference(ptr, idx, x)
    out = torch.empty((3, batches, features), dtype=torch.float16,
                      device="npu")
    result = bspmm(ptr.npu(), idx.npu(),
                   base.to(torch.float16).npu().transpose(1, 2),
                   reduce=reduction, out=out)
    if result.data_ptr() != out.data_ptr():
        raise AssertionError("operator did not return the supplied out tensor")
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol,
                               equal_nan=True)


def assert_bspmm_special_values(reduction, reference, rtol, atol):
    import torch
    from ops_gnn import bspmm

    ptr = torch.tensor([0, 2, 2])
    idx = torch.tensor([0, 1])
    x = torch.tensor([[[-float("inf"), float("nan")], [1, 2]],
                      [[float("inf"), 4], [float("nan"), -3]]],
                     dtype=torch.float16)
    expected = reference(ptr, idx, x)
    result = bspmm(ptr.npu(), idx.npu(), x.npu(), reduce=reduction)
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol,
                               equal_nan=True)


def assert_stream(op):
    import torch

    stream = torch.npu.Stream()
    ptr = torch.tensor([0, 1, 2], device="npu")
    idx = torch.tensor([1, 0], device="npu")
    x = torch.ones(2, 17, device="npu", dtype=torch.float16)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        x = x * -3
        result = op(ptr, idx, x) + 2
    torch.npu.current_stream().wait_stream(stream)
    expected = torch.full((2, 17), -1, dtype=torch.float16)
    torch.testing.assert_close(result.cpu(), expected)


def assert_ub_boundary(op):
    import re
    import torch

    ptr = torch.tensor([0, 1], device="npu")
    idx = torch.tensor([0], device="npu")
    with pytest.raises(RuntimeError, match="UB limit") as error:
        op(ptr, idx, torch.empty(1, 2**20,
                                 dtype=torch.float16, device="npu"))
    limit = int(re.search(r"UB limit (\d+)", str(error.value)).group(1))
    for size in (limit - 1, limit):
        x = torch.randn(1, size, dtype=torch.float16, device="npu")
        torch.testing.assert_close(op(ptr, idx, x).cpu(), x.cpu(),
                                   rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="UB limit"):
        op(ptr, idx, torch.empty(1, limit + 1,
                                 dtype=torch.float16, device="npu"))


def assert_large_random_graph(op, reference, dtype, rtol, atol):
    import torch

    generator = torch.Generator().manual_seed(42)
    destinations = torch.randint(1000, (5000,), generator=generator)
    ptr = torch.cat([
        torch.zeros(1, dtype=torch.long),
        torch.bincount(destinations, minlength=1000).cumsum(0),
    ]).to(dtype)
    idx = torch.randint(1000, (5000,), generator=generator).to(dtype)
    x = torch.rand(1000, 64, generator=generator).half()
    expected = reference(ptr, idx, x)
    result = op(ptr.npu(), idx.npu(), x.npu())
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)


@pytest.fixture
def segment_out_bounds_check(npu_api):
    """Check segment output mutation and scatter bounds through public APIs."""
    import torch
    from ops_gnn.scatter import scatter_sum

    def check(kind, strided, mode, *, coo_accumulates):
        with torch.inference_mode(mode == "inference"):
            backing = torch.tensor([0, 9, 1, 9] if strided else [0, 1],
                                   dtype=torch.int64, device="npu")
            out = backing[::2] if strided else backing
            values = torch.ones(2, dtype=torch.int64, device="npu")
            warmed = scatter_sum(values, out, dim=0)
            torch.testing.assert_close(warmed.cpu(), torch.ones(2, dtype=torch.int64))
            probe = None
            if not out.is_inference():
                token = torch.ones(2, device="npu", requires_grad=True)
                probe = (token * out).sum()
            source = [1, 1] if kind == "coo" and coo_accumulates else [1, 2]
            src = torch.tensor(source, dtype=torch.int64, device="npu")
            metadata = torch.tensor([0, 1, 2] if kind == "csr" else [0, 1],
                                    dtype=torch.int64, device="npu")
            with torch.inference_mode(mode != "normal"):
                result = getattr(npu_api, f"segment_sum_{kind}")(src, metadata, out=out)
            torch.npu.synchronize()
            if result.data_ptr() != out.data_ptr():
                pytest.fail("segment must return the supplied output storage")
            torch.testing.assert_close(out.cpu(), torch.tensor([1, 2]))
            if strided:
                torch.testing.assert_close(backing[1::2].cpu(), torch.tensor([9, 9]))
            if probe is not None:
                with pytest.raises(RuntimeError, match="modified by an inplace operation"):
                    probe.backward()
            inferred = scatter_sum(values, out, dim=0)
            torch.testing.assert_close(inferred.cpu(), torch.tensor([0, 1, 1]))
            with pytest.raises(IndexError, match="out of bounds"):
                scatter_sum(values, out, dim=0, dim_size=2)

    return check

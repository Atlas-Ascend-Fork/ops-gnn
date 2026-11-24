# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""A2 public Scatter functional, precision and explicit large-shape checks.

L1 tests do not require torch_scatter. Full dtype coverage additionally needs
the original torch_scatter 2.1.2 CPU extension built for the installed PyTorch;
the validated L2 versions are PyTorch 2.7.1 and 2.12.0. In either matching
environment, build the CPU extension from source with:

    FORCE_ONLY_CPU=1 python -m pip install --no-deps --no-build-isolation --no-binary=torch-scatter torch-scatter==2.1.2

Only L2 cases skip when the package is absent; broken installed packages fail.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import gc
import importlib.abc
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import weakref

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location("scatter_a2_golden", Path(__file__).with_name("golden.py"))
GOLDEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GOLDEN)
DTYPES = GOLDEN.DTYPES
REDUCES = GOLDEN.REDUCES
L1 = tuple(dtype for dtype in DTYPES if dtype not in (torch.float64, torch.int64))
CASES = (
    ((7,), -1, (7,), 5),
    ((7, 3), 0, (7,), 5),
    ((3, 7), 1, (3, 7), 5),
    ((2, 7, 3), 1, (2, 7), 5),
    ((2, 7, 3), 1, (2, 1, 3), 5),
    ((2, 7, 3, 2), -3, (7,), 5),
    ((2, 3, 2, 2, 2), -4, (3,), 4),
    ((2, 3, 2, 2, 2, 2), 1, (3,), 4),
    ((2, 3, 2, 2, 2, 2, 2), 1, (3,), 4),
    ((2, 3, 2, 2, 2, 2, 2, 2), 1, (3,), 4),
    ((0, 3), 0, (0,), 5),
    ((3, 0), -1, (0,), 0),
)
_COMPARISONS = ContextVar("scatter_comparisons", default=None)


@pytest.fixture(autouse=True)
def l2_cpu_dependency(request):
    """Skip missing L2 dependencies without masking broken installed packages."""
    callspec = getattr(request.node, "callspec", None)
    if callspec is None or callspec.params.get("dtype") not in (torch.float64, torch.int64):
        return
    params = callspec.params
    early_grad = request.node.originalname == "test_out_autograd_state" and params["context"] == "grad"
    if (
        early_grad
        and params["kind"] in ("leaf", "view_of_leaf", "inference_grad")
        and params["reduce"] in ("sum", "add", "mean")
    ):
        return  # These forward errors occur before the CPU fallback is called.
    if request.node.originalname == "test_inference_out_dispatch_boundary" and not (
        params["src_inference"] and params["index_inference"]
    ):
        return
    if importlib.util.find_spec("torch_scatter") is None:
        pytest.skip("未安装 torch_scatter，相关用例不可运行")
    importlib.import_module("torch_scatter")


def _comparison_counts(events):
    """Count assertion attempts, not unique tensors or successful pytest cases."""
    result = {}
    for kind in ("value", "arg", "out"):
        selected = [event for event in events if event["kind"] == kind]
        result[kind] = {
            "attempted": len(selected),
            "passed": sum(event["outcome"] == "passed" for event in selected),
            "failed": sum(event["outcome"] == "failed" for event in selected),
            "incomplete": sum(event["outcome"] == "incomplete" for event in selected),
            "elements": sum(event["elements"] for event in selected),
        }
    return result


def _precision_summary(events, edition):
    """Summarize only comparisons that actually produced precision metrics."""
    metrics = [event["precision"][edition] for event in events if event.get("precision") is not None]
    return {
        "comparisons": len(metrics),
        "passed": sum(item["passed"] for item in metrics),
        "failed": sum(not item["passed"] for item in metrics),
        "min_matched_ratio": min((item["matched_ratio"] for item in metrics), default=None),
        "max_abs_error": max((item["max_abs_error"] for item in metrics), default=None),
        "max_error_over_limit": max(
            (item["max_abs_error"] / item["max_abs_error_limit"] for item in metrics), default=None
        ),
        "special_value_failures": sum(not item["special_values_match"] for item in metrics),
    }


@pytest.fixture(autouse=True)
def comparison_evidence(request, record_property):
    """Attach NPU comparison evidence; pytest remains authoritative for outcome."""
    events = []
    token = _COMPARISONS.set(events)
    try:
        yield
    finally:
        _COMPARISONS.reset(token)
        record_property("scatter_nodeid", request.node.nodeid)
        record_property("scatter_evidence_scope", "NPU assertion attempts; pytest outcome is independent")
        for kind, counts in _comparison_counts(events).items():
            for metric, value in counts.items():
                record_property(f"npu_{kind}_{metric}", value)
        summary = {
            "current_commit": GOLDEN.PRECISION_COMMIT,
            "legacy_commit": GOLDEN.LEGACY_PRECISION_COMMIT,
            "current": _precision_summary(events, "current"),
            "legacy": _precision_summary(events, "legacy"),
            "legacy_diagnostic": "old thresholds with clarified current ULP anchor",
        }
        record_property("scatter_precision_summary", json.dumps(summary, sort_keys=True))
        record_property("scatter_comparison_events", json.dumps(events, sort_keys=True))


def _precision_diagnostic(event, actual, expected):
    """Add metrics to strict checks without changing their original verdict."""
    if event["method"] == "precision" or actual.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return
    try:
        event["precision"] = GOLDEN.compare_precision(actual, expected)
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        event["precision_error"] = f"{type(error).__name__}: {error}"


def _compare_tensor(actual, expected, kind, method):
    """Preserve the selected assertion and count only actual NPU tensors."""
    events = _COMPARISONS.get()
    tracked = events is not None and isinstance(actual, torch.Tensor) and actual.device.type == "npu"
    event = {
        "kind": kind,
        "method": method,
        "execution": "pytest",
        "outcome": "incomplete",
        "precision": None,
    }
    if tracked:
        event.update(elements=actual.numel(), shape=list(actual.shape), dtype=str(actual.dtype))
    try:
        if method == "precision":
            metrics = GOLDEN.compare_precision(actual.cpu(), expected)
            event["precision"] = metrics
            assert metrics["passed"], metrics
        else:
            getattr(GOLDEN, "assert_" + method)(actual, expected)
        event["outcome"] = "passed"
    except Exception as error:
        event["outcome"] = "failed"
        event["error_type"] = type(error).__name__
        raise
    finally:
        if tracked:
            _precision_diagnostic(event, actual, expected)
            events.append(event)


def _assert_tree(actual, expected, kind, method):
    """Retain exact return-type checks and split value/arg tensor assertions."""
    assert actual.__class__ is expected.__class__, "return type differs"
    if isinstance(expected, tuple):
        assert len(actual) == len(expected), "tuple length differs"
        for position, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_tree(actual_item, expected_item, "arg" if position == 1 else kind, method)
        return
    _compare_tensor(actual, expected, kind, method)


def _assert_exact(actual, expected, kind="value"):
    """Count an unchanged zero-tolerance reference assertion."""
    _assert_tree(actual, expected, kind, "exact")


def _assert_bitwise(actual, expected, kind="value"):
    """Count an unchanged bitwise reference assertion."""
    _assert_tree(actual, expected, kind, "bitwise")


@pytest.fixture(name="api", scope="module")
def api_fixture():
    """Load the A2 public APIs and select the configured test device."""
    importlib.import_module("torch_npu")
    api_module = importlib.import_module("ops_gnn")
    native = importlib.import_module("ops_gnn._pybind")
    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "3")))
    assert native.npu_arch == "dav-2201"
    assert api_module.native_available()
    return api_module


def _invoke(api_module, src, index, settings):
    arguments = (src, index, settings.dim, settings.out, settings.dim_size)
    if settings.generic:
        return api_module.scatter(*arguments, reduce=settings.reduce)
    return getattr(api_module, "scatter_" + settings.reduce)(*arguments)


def _assert_result(actual, expected):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == 2
        assert actual[1].device.type == "npu"
        _assert_result(actual[0], expected[0])
        _assert_bitwise(actual[1], expected[1], "arg")
        return
    assert actual.device.type == "npu"
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if expected.dtype in (torch.float16, torch.bfloat16, torch.float32):
        _compare_tensor(actual, expected, "value", "precision")
    else:
        _assert_bitwise(actual, expected)


def _sample(case, dtype):
    shape, dim, index_shape, buckets = case
    count = 1
    for length in shape:
        count *= length
    src = (torch.arange(count).reshape(shape) % 5 - 2).to(dtype)
    generator = torch.Generator().manual_seed(43017)
    index = torch.randint(max(buckets - 1, 1), index_shape, generator=generator)
    return src, index, dim, buckets


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("generic", (False, True))
def test_public_matrix(api, case, dtype, reduce, generic):
    """Compare generic and dedicated APIs across shapes, dtypes and reductions."""
    src, index, dim, buckets = _sample(case, dtype)
    settings = SimpleNamespace(dim=dim, dim_size=buckets, out=None, reduce=reduce, generic=generic)
    expected = _invoke(GOLDEN, src, index, settings)
    actual = _invoke(api, src.npu(), index.npu(), settings)
    _assert_result(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("layout", ("contiguous", "strided", "zero_stride"))
def test_layout_out(api, dtype, reduce, layout):
    """Check accumulated out values and identity across source and index layouts."""
    src = torch.arange(21).reshape(7, 3).to(dtype) % 5
    index = torch.tensor([2, 0, 2, 0, 1, 0, 2])
    src_npu, index_npu = src.npu(), index.npu()
    if layout == "strided":
        src, src_npu = src.t(), src_npu.t()
        index = torch.stack((index, index), dim=1)[:, 0]
        index_npu = torch.stack((index_npu, index_npu), dim=1)[:, 0]
    elif layout == "zero_stride":
        src, src_npu = src[:1].expand(7, 3), src_npu[:1].expand(7, 3)
    dim = 1 if layout == "strided" else 0
    shape = list(src.shape)
    shape[dim] = 4
    out = torch.full(shape, 2, dtype=dtype)
    npu_out = out.npu()
    settings = SimpleNamespace(dim=dim, dim_size=99, out=out, reduce=reduce, generic=False)
    expected = _invoke(GOLDEN, src, index, settings)
    settings.out = npu_out
    actual = _invoke(api, src_npu, index_npu, settings)
    assert (actual[0] if isinstance(actual, tuple) else actual) is npu_out
    _assert_result(actual, expected)
    _assert_exact(npu_out, out, "out")


@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("context", ("grad", "no_grad", "inference"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.int64))
def test_noncontiguous_out(api, reduce, context, dtype):
    """Match CPU success or errors for noncontiguous out in each execution context."""
    manager = {"grad": torch.enable_grad, "no_grad": torch.no_grad, "inference": torch.inference_mode}[context]
    with manager():
        backing = torch.full((4, 3), 2, dtype=dtype)
        npu_backing = backing.npu()
        out, npu_out = backing.t(), npu_backing.t()
        src = torch.arange(21).reshape(3, 7).to(dtype)
        index = torch.tensor([2, 0, 2, 0, 1, 0, 2])
        settings = SimpleNamespace(dim=1, dim_size=None, out=npu_out, reduce=reduce, generic=False)
        if reduce in ("mul", "min", "max"):
            with pytest.raises(RuntimeError):
                _invoke(api, src.npu(), index.npu(), settings)
            _assert_exact(npu_backing, backing, "out")
            return
        actual = _invoke(api, src.npu(), index.npu(), settings)
        settings.out = out
        expected = _invoke(GOLDEN, src, index, settings)
        assert actual is npu_out
        _assert_result(actual, expected)
        _assert_exact(npu_backing, backing, "out")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
def test_source_out_alias(api, dtype, reduce):
    """Preserve CPU results or overlap errors when source and out are the same tensor."""
    out = torch.tensor([2, 3, 4], dtype=dtype)
    npu_out = out.npu()
    index = torch.tensor([1, 2, 0])
    settings = SimpleNamespace(dim=0, dim_size=None, out=npu_out, reduce=reduce, generic=False)
    if reduce in ("sum", "add", "mean"):
        with pytest.raises(RuntimeError):
            _invoke(api, npu_out, index.npu(), settings)
        _assert_exact(npu_out, out, "out")
        return
    actual = _invoke(api, npu_out, index.npu(), settings)
    settings.out = out
    expected = _invoke(GOLDEN, out, index, settings)
    _assert_result(actual, expected)


def _index_out_alias(device, stride, independent):
    values = [2, 2] if stride == 1 else [4, 0, 4, 0]
    backing = torch.tensor(values, dtype=torch.int64, device=device)
    storage = backing.untyped_storage()
    if independent:
        storage = storage[:]
    out = torch.empty(0, dtype=torch.float32, device=device).set_(storage, 0, (len(values) * 2,), (1,))
    assert out.data_ptr() == backing.data_ptr()
    return backing[::stride], out


@pytest.mark.parametrize("stride", (1, 2))
@pytest.mark.parametrize("independent", (False, True))
@pytest.mark.parametrize("reduce", REDUCES)
def test_index_out_alias(api, stride, independent, reduce):
    """Check dynamic index reads and overlap errors for shared output storage."""
    index, out = _index_out_alias("cpu", stride, independent)
    npu_index, npu_out = _index_out_alias("npu", stride, independent)
    src = torch.zeros(2)
    settings = SimpleNamespace(dim=0, dim_size=None, out=out, reduce=reduce, generic=False)
    if stride == 1 and not independent and reduce in ("sum", "add", "mean"):
        with pytest.raises(RuntimeError):
            _invoke(GOLDEN, src, index, settings)
        settings.out = npu_out
        with pytest.raises(RuntimeError):
            _invoke(api, src.npu(), npu_index, settings)
    else:
        expected = _invoke(GOLDEN, src, index, settings)
        settings.out = npu_out
        actual = _invoke(api, src.npu(), npu_index, settings)
        _assert_result(actual, expected)
    _assert_bitwise(npu_index, index, "out")
    _assert_exact(npu_out, out, "out")


def _make_autograd_out(kind, dtype):
    """Construct out states on NPU without inspecting private view metadata."""
    values = [1.0, 9.0] if kind.startswith("view_") else [1.0]
    if kind in ("inference", "inference_grad"):
        with torch.inference_mode():
            backing = torch.tensor(values, dtype=dtype, device="npu", requires_grad=kind == "inference_grad")
        return backing, backing
    with torch.enable_grad():
        backing = torch.tensor(values, dtype=dtype, device="npu", requires_grad=True)
        if kind in ("nonleaf", "view_of_nonleaf"):
            backing = backing.clone()
        out = backing[:1] if kind.startswith("view_") else backing
    return out, backing


def _autograd_expectation(kind, context, reduce):
    """CPU forward observations, including the PyTorch 2.8 dirty-view check.

    The independent CPU record covers values, errors after mutation, identity,
    and requires_grad. Daily NPU tests use these facts without importing it.
    """
    sum_like = reduce in ("sum", "add", "mean")
    early_error = context == "grad" and kind in ("leaf", "view_of_leaf", "inference_grad") and sum_like
    inference_error = kind in ("inference", "inference_grad") and context != "inference"
    custom_error = context == "grad" and (kind == "leaf" or (kind.startswith("view_") and reduce in ("min", "max")))
    if context == "grad" and kind == "view_of_leaf" and reduce == "mul":
        # PyTorch 2.8 added the missing leaf-view check after custom forward.
        version = tuple(int(part) for part in torch.__version__.split(".")[:2])
        custom_error = version >= (2, 8)
    value = {"sum": 6.0, "add": 6.0, "mul": 6.0, "mean": 3.0, "min": 1.0, "max": 3.0}[reduce]
    if early_error:
        value = 1.0
    elif inference_error and reduce == "mean":
        value = 6.0  # CPU raises after the sum, before division.
    original_object = kind == "inference" or context == "grad" or sum_like
    return SimpleNamespace(
        raises=early_error or inference_error or custom_error,
        value=value,
        original_object=original_object,
        requires_grad=kind != "inference" and (context == "grad" or sum_like),
    )


def _assert_autograd_return(actual, out, expected, reduce):
    """Check successful public returns, including shared-storage wrappers."""
    value = actual[0] if isinstance(actual, tuple) else actual
    assert value.device == out.device and value.dtype == out.dtype
    assert value.shape == out.shape
    assert (value is out) == expected.original_object
    assert value.data_ptr() == out.data_ptr()
    assert value.requires_grad == expected.requires_grad
    _assert_bitwise(value, torch.tensor([expected.value], dtype=out.dtype))
    if reduce in ("min", "max"):
        assert isinstance(actual, tuple) and len(actual) == 2
        assert actual[1].device == out.device and actual[1].dtype == torch.int64
        assert not actual[1].requires_grad
        _assert_bitwise(actual[1], torch.tensor([2 if reduce == "min" else 1]), "arg")


@pytest.mark.usefixtures("api")
@pytest.mark.parametrize("kind", ("leaf", "view_of_leaf", "nonleaf", "view_of_nonleaf"))
def test_out_checker_preserves_saved_autograd_graph(kind):
    """Check mutation legality without changing data or an existing PyTorch graph."""
    native = importlib.import_module("ops_gnn._pybind")
    with torch.enable_grad():
        leaf = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)
        nonleaf = kind in ("nonleaf", "view_of_nonleaf")
        scale = 2 if nonleaf else 1
        backing = leaf * scale if nonleaf else leaf
        out = backing[::2] if kind.startswith("view_") else backing
        initial = out.detach().clone()
        saved = (out * out).sum()
        if nonleaf:
            native.scatter_check_out(out)
        else:
            with pytest.raises(RuntimeError, match="leaf Variable"):
                native.scatter_check_out(out)
        GOLDEN.assert_bitwise(out, initial)
        saved.backward()  # Only ordinary PyTorch operations participate in this graph.
        expected_grad = 2 * leaf.detach() * scale**2
        if kind.startswith("view_"):
            expected_grad[1::2] = 0
        GOLDEN.assert_bitwise(leaf.grad, expected_grad)


@pytest.mark.parametrize("kind", ("leaf", "view_of_leaf", "nonleaf", "view_of_nonleaf", "inference", "inference_grad"))
@pytest.mark.parametrize("context", ("grad", "no_grad", "inference"))
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_out_autograd_state(api, kind, context, reduce, dtype):
    """Match CPU forward errors and side effects without exercising backward."""
    out, backing = _make_autograd_out(kind, dtype)
    expected = _autograd_expectation(kind, context, reduce)
    src = torch.tensor([2.0, 3.0], dtype=dtype, device="npu")
    index = torch.tensor([0, 0], dtype=torch.int64, device="npu")
    manager = {"grad": torch.enable_grad, "no_grad": torch.no_grad, "inference": torch.inference_mode}[context]
    with manager():
        if expected.raises:
            with pytest.raises(RuntimeError) as raised:
                getattr(api, "scatter_" + reduce)(src, index, 0, out)
            assert raised.type is RuntimeError
        else:
            actual = getattr(api, "scatter_" + reduce)(src, index, 0, out)
            _assert_autograd_return(actual, out, expected, reduce)
    _assert_bitwise(out, torch.tensor([expected.value], dtype=dtype), "out")
    expected_backing = [expected.value, 9.0] if kind.startswith("view_") else [expected.value]
    _assert_bitwise(backing, torch.tensor(expected_backing, dtype=dtype), "out")


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("reduce", ("sum", "add", "mean"))
@pytest.mark.parametrize("src_inference", (False, True))
@pytest.mark.parametrize("index_inference", (False, True))
def test_inference_out_dispatch_boundary(api, dtype, reduce, src_inference, index_inference):
    """Match CPU dispatch before or after writing a gradient-enabled inference out."""
    out, _ = _make_autograd_out("inference_grad", dtype)
    with torch.inference_mode(src_inference):
        src = torch.tensor([2.0, 3.0], dtype=dtype, device="npu")
    with torch.inference_mode(index_inference):
        index = torch.tensor([0, 0], dtype=torch.int64, device="npu")
    with torch.enable_grad(), pytest.raises(RuntimeError):
        getattr(api, "scatter_" + reduce)(src, index, 0, out)
    expected = 6.0 if src_inference and index_inference else 1.0
    _assert_bitwise(out, torch.tensor([expected], dtype=dtype), "out")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", ("min", "max"))
@pytest.mark.parametrize("supplied", (False, True))
def test_extremes_ties(api, dtype, reduce, supplied):
    """Preserve initialization identities, first ties and unchanged-out sentinels."""
    limits = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    identity = limits.max if reduce == "min" else limits.min
    src = torch.tensor([identity, 2, 2], dtype=dtype)
    index = torch.tensor([0, 1, 1])
    out = torch.tensor([identity, 2, 3], dtype=dtype) if supplied else None
    settings = SimpleNamespace(dim=0, dim_size=3, out=out, reduce=reduce, generic=False)
    expected = _invoke(GOLDEN, src, index, settings)
    settings.out = torch.tensor([identity, 2, 3], dtype=dtype).npu() if supplied else None
    actual = _invoke(api, src.npu(), index.npu(), settings)
    _assert_exact(actual, expected)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("reduce", REDUCES)
def test_special_floats(api, dtype, reduce):
    """Compare NaN, infinity and signed-zero behavior for every floating dtype."""
    src = torch.tensor([float("nan"), -0.0, 0.0, float("inf"), -float("inf"), float("nan"), 2.0, 2.0], dtype=dtype)
    index = torch.tensor([0, 1, 1, 2, 3, 4, 4, 4])
    settings = SimpleNamespace(dim=0, dim_size=6, out=None, reduce=reduce, generic=False)
    _assert_result(_invoke(api, src.npu(), index.npu(), settings), _invoke(GOLDEN, src, index, settings))


@pytest.mark.parametrize(
    "dtype,count,expected",
    (
        (torch.int8, 128, -128),
        (torch.uint8, 300, 1),
        (torch.int16, 32768, -32768),
    ),
)
def test_mean_count_wrap(api, dtype, count, expected):
    """Preserve source-dtype count overflow before integer mean division."""
    src = torch.ones(count, dtype=dtype, device="npu")
    index = torch.zeros(count, dtype=torch.int64, device="npu")
    _assert_exact(api.scatter_mean(src, index), torch.tensor([expected], dtype=dtype))


@pytest.mark.parametrize("dtype,large", ((torch.float16, 2048), (torch.bfloat16, 256)))
@pytest.mark.parametrize("reduce", ("sum", "add", "mean"))
def test_low_precision_cancellation(api, dtype, large, reduce):
    """Retain small residuals in CPU-compatible low-precision sums."""
    src = torch.tensor([large, 1, -large], dtype=dtype)
    index = torch.zeros(3, dtype=torch.long)
    expected = getattr(GOLDEN, "scatter_" + reduce)(src, index)
    actual = getattr(api, "scatter_" + reduce)(src.npu(), index.npu())
    _assert_exact(actual, expected)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("reduce", ("sum", "add", "mean"))
def test_multibucket_ordered_cancellation(api, dtype, reduce):
    """Check cancellation groups that cross source transfer tile boundaries."""
    large, small = (65504.0, 3 * 2**-11) if dtype == torch.float16 else (2**25, 1.0)
    generator = torch.Generator().manual_seed(20261104)
    index = torch.randperm(512, generator=generator).repeat_interleave(3)
    src = torch.tensor([large, small, -large], dtype=dtype).repeat(512)
    src = src[:, None].expand(-1, 64).contiguous()
    expected = GOLDEN.scatter(src, index, 0, dim_size=512, reduce=reduce)
    actual = api.scatter(src.npu(), index.npu(), 0, dim_size=512, reduce=reduce)
    _assert_result(actual, expected)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("reduce", ("sum", "add", "mean"))
@pytest.mark.parametrize("shape", ((4097, 64), (65536, 8)))
def test_high_conflict_ordered_cancellation(api, dtype, reduce, shape):
    """Check one-bucket cancellation across long rows and wide features."""
    generator = torch.Generator().manual_seed(20261101)
    src = torch.randn(shape, generator=generator)
    paired = shape[0] // 2
    paired_end = 2 * paired
    src[paired:paired_end, 1::2] = -src[:paired, 1::2]
    if shape[0] % 2:
        src[-1, 1::2] = 0
    src = src.to(dtype)
    index = torch.zeros(shape[0], dtype=torch.int64)
    expected = GOLDEN.scatter(src, index, 0, dim_size=1, reduce=reduce)
    actual = api.scatter(src.npu(), index.npu(), 0, dim_size=1, reduce=reduce)
    _assert_result(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_original_index_count(api, dtype):
    """Count the original index before broadcasting it across source elements."""
    src = torch.tensor([[2, 4, 6]], dtype=dtype, device="npu")
    index = torch.tensor([0], device="npu")
    _assert_exact(api.scatter_mean(src, index, 1), torch.tensor([[12]], dtype=dtype))


@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_large_integer_mean(api, dtype):
    """Keep integer precision and floor division for large signed values."""
    src = torch.tensor([2**30 + 3, -1, -(2**30) - 3], dtype=dtype)
    index = torch.zeros(3, dtype=torch.long)
    _assert_exact(api.scatter_mean(src.npu(), index.npu()), GOLDEN.scatter_mean(src, index))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.int64))
def test_mean_exception_preserves_prior_out_write(api, dtype):
    """Retain the sum update when later count broadcasting raises an error."""
    src = torch.arange(24).reshape(2, 3, 4).to(dtype)
    index = torch.tensor([[0], [1]])
    out = torch.full((2, 3, 2), 7, dtype=dtype)
    npu_out = out.npu()
    with pytest.raises(RuntimeError):
        GOLDEN.scatter_mean(src, index, 2, out)
    with pytest.raises(RuntimeError):
        api.scatter_mean(src.npu(), index.npu(), 2, npu_out)
    _assert_bitwise(npu_out, out, "out")


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.int64))
def test_scalar_index_mean_exception(api, dtype):
    """Preserve the out update before the scalar-index count error."""
    src = torch.tensor([[2, 4, 6]], dtype=dtype, device="npu")
    index = torch.tensor(0, device="npu")
    out = torch.tensor([[5]], dtype=dtype, device="npu")
    with pytest.raises(IndexError):
        api.scatter_mean(src, index, 1, out)
    _assert_bitwise(out, torch.tensor([[17]], dtype=dtype), "out")


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("reduce", ("sum", "mean", "min", "max", "mul"))
@pytest.mark.parametrize("width", (64, 128, 2048))
@pytest.mark.parametrize("supplied", (False, True))
def test_wide_feature_paths(api, dtype, reduce, width, supplied):
    """Compare wide feature paths with special values and optional initial out."""
    generator = torch.Generator().manual_seed(32457)
    src = torch.randn((17, width), generator=generator).to(dtype)
    index = torch.randint(5, (17,), generator=generator)
    src[0, :5] = torch.tensor([float("nan"), 0.0, -0.0, float("inf"), -float("inf")], dtype=dtype)
    index[1] = index[0]
    src[1, 1:3] = torch.tensor([-0.0, 0.0], dtype=dtype)
    initial = torch.zeros((7, width), dtype=dtype) if supplied else None
    settings = SimpleNamespace(
        dim=0, dim_size=7, out=None if initial is None else initial.clone(), reduce=reduce, generic=False
    )
    expected = _invoke(GOLDEN, src, index, settings)
    settings.out = None if initial is None else initial.npu()
    actual = _invoke(api, src.npu(), index.npu(), settings)
    _assert_result(actual, expected)
    if supplied:
        assert (actual[0] if isinstance(actual, tuple) else actual) is settings.out
        expected_value = expected[0] if isinstance(expected, tuple) else expected
        _compare_tensor(
            settings.out, expected_value, "out", "precision" if settings.out.is_floating_point() else "bitwise"
        )


@pytest.mark.parametrize("inference", (False, True))
def test_scatter_out_invalidates_csr_source_proof(api, inference):
    """Invalidate CSR source proofs after Scatter writes through an alias."""
    with torch.inference_mode(inference):
        cpu = torch.ones((2053 * 16, 64), dtype=torch.float16)
        source = cpu.npu()
        pointers = (torch.arange(2054, dtype=torch.long) * 16).npu()
        api.segment_sum_csr(source, pointers)
        alias = torch.empty(0, dtype=source.dtype, device=source.device).set_(
            source.untyped_storage(), 0, source.shape, source.stride()
        )
        pattern = torch.tensor([32768.0, 2**-9, -32768.0] + [0.0] * 13, dtype=source.dtype).repeat(2053).view(-1, 1)
        replacement = pattern.expand_as(cpu).clone()
        multiplier = replacement.npu()
        index = torch.arange(source.size(0), device="npu")
        api.scatter_mul(multiplier, index, dim=0, out=alias)
        # CSR's ordered low-precision accumulation rounds the cancellation
        # group to zero; stale bounded-input proof would retain the tiny term.
        expected = torch.zeros((2053, 64), dtype=torch.float16)
        _assert_exact(api.segment_sum_csr(source, pointers), expected)


@pytest.mark.parametrize("inference", (False, True))
def test_scatter_out_invalidates_csr_pointer_proof(api, inference):
    """Invalidate CSR pointer proofs after Scatter changes aliased pointer bytes."""
    with torch.inference_mode(inference):
        source = torch.ones((4096, 8), dtype=torch.float16, device="npu")
        pointers = torch.arange(4097, dtype=torch.int64, device="npu")
        api.segment_sum_csr(source, pointers)
        alias = torch.empty(0, dtype=torch.int32, device="npu").set_(pointers.untyped_storage()[:], 0, (8194,), (1,))
        assert alias.data_ptr() == pointers.data_ptr()
        api.scatter_add(torch.ones(1, dtype=torch.int32, device="npu"), torch.tensor([2], device="npu"), out=alias)
        expected = torch.ones((4096, 8), dtype=torch.float16)
        expected[0], expected[1] = 2, 0
        _assert_exact(api.segment_sum_csr(source, pointers), expected)


@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
@pytest.mark.parametrize("writer", ("public", "a2_native", "legacy_native"))
def test_scatter_out_invalidates_spmm_source_proof(api, dtype, writer, record_property):
    """Recompute cached SpMM NaN proofs after a native Scatter alias write."""
    source = torch.ones((3, 8), dtype=dtype, device="npu")
    pointers = torch.arange(4, dtype=torch.int64, device="npu")
    indices = torch.arange(3, dtype=torch.int64, device="npu")
    replacement = torch.full_like(source, float("nan"))
    alias = torch.empty(0, dtype=dtype, device=source.device).set_(
        source.untyped_storage()[:], 0, source.shape, source.stride()
    )
    native = importlib.import_module("ops_gnn._pybind")
    for _ in range(3):
        api.spmm(pointers, indices, source, reduce="max")
    if writer == "public":
        written = api.scatter_sum(replacement, indices, dim=0, out=alias)
    elif writer == "a2_native":
        written = native.scatter_a2_forward(replacement, indices, 0, alias, 3, 0)[0]
    else:
        written = native.scatter_forward(replacement, indices, 0, alias, 0, True)[0]
    result = api.spmm(pointers, indices, source, reduce="max")
    expected = torch.full((3, 8), float("nan"), dtype=dtype)
    _assert_result(written, expected)
    _assert_exact(alias, expected, "out")
    GOLDEN.assert_exact(result, expected)
    record_property("spmm_alias_result_checks", 1)


@pytest.mark.parametrize("dtype", (torch.float64, torch.int64))
@pytest.mark.parametrize("reduce", REDUCES)
def test_empty_storage_view(api, dtype, reduce):
    """Handle empty views that retain nonempty storage in both L2 dtypes."""
    src = torch.arange(8, dtype=dtype)[2:2]
    backing = torch.arange(8, dtype=dtype, device="npu")
    index = torch.empty(0, dtype=torch.long)
    actual = api.scatter(backing[2:2], index.npu(), dim_size=3, reduce=reduce)
    expected = GOLDEN.scatter(src, index, dim_size=3, reduce=reduce)
    _assert_result(actual, expected)


@pytest.mark.parametrize("layout", (((0, 3), 1), ((3, 0), 0)))
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.int32, torch.float64, torch.int64))
@pytest.mark.parametrize("dim_size", (None, 2))
@pytest.mark.parametrize("reduce", REDUCES)
def test_empty_nonreduced_axis(api, layout, dtype, dim_size, reduce):
    """Match CPU empty-output shapes and mean count errors on nonreduced axes."""
    shape, dim = layout
    src = torch.empty(shape, dtype=dtype)
    index = torch.tensor([0, 1, 0])
    if reduce == "mean" and dim_size is None:
        with pytest.raises(RuntimeError):
            api.scatter(src.npu(), index.npu(), dim, dim_size=dim_size, reduce=reduce)
    else:
        expected = GOLDEN.scatter(src, index, dim, dim_size=dim_size, reduce=reduce)
        actual = api.scatter(src.npu(), index.npu(), dim, dim_size=dim_size, reduce=reduce)
        _assert_result(actual, expected)


@pytest.mark.parametrize("reduce", REDUCES)
def test_nondefault_stream(api, reduce):
    """Follow explicit producer and consumer stream dependencies."""
    src = torch.arange(21).reshape(7, 3).float().npu()
    index = torch.tensor([0, 2, 1, 0, 2, 0, 1], device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        changed = src * 2
        actual = api.scatter(changed, index, 0, dim_size=4, reduce=reduce)
    torch.npu.current_stream().wait_stream(stream)
    expected = GOLDEN.scatter(src.cpu() * 2, index.cpu(), 0, dim_size=4, reduce=reduce)
    _assert_result(actual, expected)


def _compact_inputs(dtype, device="npu"):
    """Use 64 features with a four-column period to keep the independent oracle small."""
    rows = torch.arange(512).reshape(-1, 1)
    features = (rows * 7 + torch.arange(4) * 3) % 29
    features = (features if dtype == torch.uint8 else features - 14).to(dtype)
    generator = torch.Generator().manual_seed(72419)
    index = torch.randperm(512, generator=generator) % 128
    return SimpleNamespace(
        reference=features,
        source=features.repeat(1, 16).to(device),
        original=index,
        index=index.clone().to(device),
        warm_reference=index,
    )


def _compact_reference(state, index, reduce, supplied=False):
    """Evaluate independent columns, then repeat the chosen input's feature period."""
    out = torch.full((128, 4), 14 if reduce == "min" else 2, dtype=state.reference.dtype) if supplied else None
    result = getattr(GOLDEN, "scatter_" + reduce)(state.reference, index, 0, out, 128)
    if isinstance(result, tuple):
        return tuple(value.repeat(1, 16) for value in result)
    return result.repeat(1, 16)


def _compact_call(api_module, state, reduce, out=None):
    return getattr(api_module, "scatter_" + reduce)(state.source, state.index, 0, out, 128)


def _warm_compact(api_module, state, reduce, index=None):
    """Do not run CPU assertions between warm calls and the mutation being tested."""
    state.warm_reference = state.original if index is None else index
    return tuple(_compact_call(api_module, state, reduce) for _ in range(3))


def _check_compact_warm(state, reduce, warm):
    expected = _compact_reference(state, state.warm_reference, reduce)
    for actual in warm:
        _assert_result(actual, expected)


def _verify_compact(api_module, state, index, reduce, warm):
    """Capture the first post-write result before assertions can trigger ATen calls."""
    actual = _compact_call(api_module, state, reduce)
    expected = _compact_reference(state, index, reduce)
    _check_compact_warm(state, reduce, warm)
    _assert_result(actual, expected)
    out = torch.full((128, 64), 14 if reduce == "min" else 2, dtype=state.source.dtype, device=state.source.device)
    expected = _compact_reference(state, index, reduce, supplied=True)
    actual = _compact_call(api_module, state, reduce, out)
    assert (actual[0] if isinstance(actual, tuple) else actual) is out
    _assert_result(actual, expected)
    expected_value = expected[0] if isinstance(expected, tuple) else expected
    _compare_tensor(out, expected_value, "out", "precision" if out.is_floating_point() else "bitwise")


def _independent_index_slice(index, first=0, step=1):
    """Create an independent Storage wrapper using only public slicing and set_."""
    byte_offset = (index.storage_offset() + first) * index.element_size()
    storage = index.untyped_storage()[byte_offset:]
    length = (index.numel() - first + step - 1) // step
    alias = torch.empty(0, dtype=index.dtype, device=index.device).set_(storage, 0, (length,), (step,))
    assert alias.data_ptr() == index.data_ptr() + first * index.element_size()
    return alias


@pytest.mark.parametrize("dtype", L1)
@pytest.mark.parametrize("reduce", ("sum", "mul", "mean", "min", "max"))
def test_compact_bucket_rows_preserve_source_order(api, dtype, reduce):
    """Keep original coordinates when neighboring buckets interleave source rows."""
    rows = torch.tensor([2, 3, 5, 7, 11, 13], dtype=torch.float32).unsqueeze(1)
    src = (rows + (torch.arange(64) % 3).unsqueeze(0) / 4).to(dtype)
    index = torch.tensor([0, 2, 0, 2, 0, 2], dtype=torch.int64)
    settings = SimpleNamespace(dim=0, dim_size=4, out=None, reduce=reduce, generic=False)
    expected = _invoke(GOLDEN, src, index, settings)
    device_src, device_index = src.npu(), index.npu()
    for _ in range(2):
        _assert_result(_invoke(api, device_src, device_index, settings), expected)


@pytest.mark.parametrize(
    "rows",
    (
        pytest.param((2**24, 1.0, -(2**24)), id="cancellation"),
        pytest.param((1.0, float("nan"), 2.0), id="nan"),
        pytest.param((float("inf"), 1.0, 2.0), id="positive-inf"),
        pytest.param((float("-inf"), 1.0, 2.0), id="negative-inf"),
        pytest.param((float("inf"), float("-inf"), 1.0), id="inf-cancellation"),
        pytest.param((-0.0, -0.0, -0.0), id="negative-zero"),
    ),
)
@pytest.mark.parametrize("api_name", ("scatter", "scatter_sum", "scatter_add"))
def test_compact_sum_order_and_special_values_after_mutation(api, rows, api_name):
    """Retain source-order values and special signs after repeated index writes."""
    source = torch.tensor(rows, dtype=torch.float32).unsqueeze(1).expand(3, 64).clone()
    variants = tuple(torch.tensor(values) for values in ((0, 0, 0), (1, 1, 1), (0, 1, 0), (0, 0, 0)))
    device_source, device_index = source.npu(), variants[0].npu()
    operation = getattr(api, api_name)
    kwargs = {"reduce": "sum"} if api_name == "scatter" else {}
    retained = []
    for variant in variants:
        expected = GOLDEN.scatter_sum(source, variant, dim=0, dim_size=None)
        device_index.copy_(variant.to(device_index.device))
        for _ in range(3):
            actual = operation(device_source, device_index, dim=0, dim_size=None, **kwargs)
            retained.append((actual, expected))
    for actual, expected in retained:
        _assert_exact(actual, expected)


@pytest.mark.parametrize("dtype", L1)
@pytest.mark.parametrize("reduce", ("sum", "mean", "min"))
@pytest.mark.parametrize("inference", (False, True))
@pytest.mark.parametrize("mutation", ("inplace", "shared_view", "independent_slice"))
def test_compact_index_mutation_after_warm(api, dtype, reduce, inference, mutation):
    """Revalidate compact indices after ordinary or aliased writes."""
    with torch.inference_mode(inference):
        state = _compact_inputs(dtype)
        changed = (state.original * 3 + 17) % 128
        target = state.index if mutation == "inplace" else state.index.view(-1)
        if mutation == "independent_slice":
            target = _independent_index_slice(state.index, 1, 2)
            changed[::2] = state.original[::2]
        updates = (changed, state.original)
        for previous, updated in zip((state.original, changed), updates):
            payload = (updated[1::2] if mutation == "independent_slice" else updated).to(state.source.device)
            warm = _warm_compact(api, state, reduce, previous)
            target.copy_(payload)
            _verify_compact(api, state, updated, reduce, warm)
            _assert_bitwise(state.index, updated, "out")


def _inferred_shape_inputs(dtype, index_elements):
    """Exercise both broadcast and rowwise indices with growing and shrinking maxima."""
    source = (torch.arange(257 * 64).reshape(257, 64) % 17 - 8).to(dtype)
    original = torch.tensor([2]) if index_elements == 1 else torch.arange(index_elements) % 3
    return source, (original, original + 4, original % 2)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("index_elements", (1, 257))
@pytest.mark.parametrize("inference", (False, True))
def test_compact_index_mutation_updates_inferred_shape(api, dtype, index_elements, inference):
    """Recompute inferred shapes after same-address writes and preserve old results."""
    with torch.inference_mode(inference):
        source, variants = _inferred_shape_inputs(dtype, index_elements)
        expected = [GOLDEN.scatter_sum(source, variant, 0, out=None, dim_size=None) for variant in variants]
        source = source.npu()
        variants = tuple(variant.npu() for variant in variants)
        index = variants[0].clone()
        address = index.data_ptr()
        retained = []
        for phase, payload in enumerate(variants):
            if phase:
                index.copy_(payload)
            assert index.data_ptr() == address
            retained.append(tuple(api.scatter_sum(source, index, 0, out=None, dim_size=None) for _ in range(3)))
        # Preserve every old output; assertions must not intervene before the next mutation.
        for values, reference in zip(retained, expected):
            for value in values:
                _assert_result(value, reference)


@pytest.mark.parametrize("dtype", L1)
@pytest.mark.parametrize("reduce", ("sum", "mean", "min"))
@pytest.mark.parametrize("inference", (False, True))
@pytest.mark.parametrize("writer", ("csr", "coo"))
def test_native_out_updates_compact_index(api, dtype, reduce, inference, writer):
    """Observe index mutations written by native CSR and COO outputs."""
    with torch.inference_mode(inference):
        state = _compact_inputs(dtype)
        changed = (state.original * 3 + 17) % 128
        alias = _independent_index_slice(state.index)
        values = changed if writer == "csr" else changed - state.original
        values = values.to(state.source.device)
        pointers = torch.arange(513 if writer == "csr" else 512, dtype=torch.int64, device=state.source.device)
        warm = _warm_compact(api, state, reduce)
        if writer == "csr":
            written = api.segment_sum_csr(values, pointers, out=alias)
        else:
            # A2 COO sums include out; the precomputed difference yields changed.
            written = api.segment_sum_coo(values, pointers, out=alias, dim_size=512)
        _verify_compact(api, state, changed, reduce, warm)
        assert written is alias
        _assert_bitwise(written, changed, "out")
        _assert_bitwise(state.index, changed, "out")


@pytest.mark.parametrize("dtype", L1)
@pytest.mark.parametrize("reduce", ("sum", "mean", "min"))
@pytest.mark.parametrize("inference", (False, True))
def test_ordered_stream_index_mutation(api, dtype, reduce, inference):
    """Revalidate indices after a synchronized write on another stream."""
    with torch.inference_mode(inference):
        state = _compact_inputs(dtype)
        changed = (state.original * 3 + 17) % 128
        payload = changed.to(state.source.device)
        alias = _independent_index_slice(state.index)
        origin = torch.npu.current_stream()
        producer, consumer = torch.npu.Stream(), torch.npu.Stream()
        warm = _warm_compact(api, state, reduce)
        producer.wait_stream(origin)
        with torch.npu.stream(producer):
            alias.copy_(payload)
        consumer.wait_stream(producer)
        with torch.npu.stream(consumer):
            _verify_compact(api, state, changed, reduce, warm)
        origin.wait_stream(consumer)
        _assert_bitwise(state.index, changed, "out")


def _concurrent_group_inputs(rows, reduce, stream, dtype=torch.float32, geometry=(128, 128)):
    """Use four independent feature values to build a compact CPU oracle."""
    buckets, feature_repeats = geometry
    source = (torch.arange(rows * 4).reshape(rows, 4) % 19 - 9).to(dtype)
    generator = torch.Generator().manual_seed(9147 + rows)
    index = torch.randint(buckets, (rows,), generator=generator)
    variants = (index, (index + buckets // 2) % buckets)
    expected = []
    for variant in variants:
        result = getattr(GOLDEN, "scatter_" + reduce)(source, variant, dim=0, dim_size=buckets)
        if isinstance(result, tuple):
            result = tuple(item.repeat_interleave(feature_repeats, dim=1) for item in result)
        else:
            result = result.repeat_interleave(feature_repeats, dim=1)
        expected.append(result)
    return SimpleNamespace(
        source=source.repeat_interleave(feature_repeats, dim=1).npu(),
        index=index.npu(),
        variants=tuple(variant.npu() for variant in variants),
        expected=expected,
        reduce=reduce,
        stream=stream,
        buckets=buckets,
        output_shape=(buckets, 4 * feature_repeats),
    )


def _concurrent_group_worker(api_module, state, barrier):
    """Keep returned tensors alive while two Host threads enqueue reductions."""
    torch.npu.set_device(state.source.device)
    results = []
    with torch.no_grad(), torch.npu.stream(state.stream):
        barrier.wait(timeout=30)
        for iteration in range(12):
            variant = iteration // 4 % 2
            if iteration % 4 == 0:
                state.index.copy_(state.variants[variant])
            result = getattr(api_module, "scatter_" + state.reduce)(
                state.source, state.index, dim=0, dim_size=state.buckets
            )
            results.append((variant, result))
    return results


def _concurrent_empty_cache(device, barrier):
    """Exercise public cache release while native calls can release the GIL."""
    torch.npu.set_device(device)
    barrier.wait(timeout=30)
    for _ in range(8):
        torch.npu.empty_cache()


@pytest.mark.parametrize("reduce", ("sum", "mean", "min"))
@pytest.mark.parametrize("shared_stream", (False, True))
@pytest.mark.parametrize("release_cache", (False, True))
def test_concurrent_scatter_streams_and_cache_release(api, reduce, shared_stream, release_cache):
    """Retain correct results across concurrent calls, index changes and release."""
    streams = [torch.npu.Stream()]
    streams.append(streams[0] if shared_stream else torch.npu.Stream())
    states = [_concurrent_group_inputs(rows, reduce, stream) for rows, stream in zip((4095, 4097), streams)]
    torch.npu.synchronize()
    barrier = threading.Barrier(3 if release_cache else 2)
    with ThreadPoolExecutor(max_workers=barrier.parties) as executor:
        futures = [executor.submit(_concurrent_group_worker, api, state, barrier) for state in states]
        cache = executor.submit(_concurrent_empty_cache, states[0].source.device, barrier) if release_cache else None
        results = [future.result(timeout=60) for future in futures]
        if cache is not None:
            cache.result(timeout=60)
    torch.npu.synchronize()
    for state, retained in zip(states, results):
        for variant, result in retained:
            _assert_result(result, state.expected[variant])


def _retire_group_output(result, release):
    """Retire only this call's output without reading its pending values."""
    if release == "resize":
        result.untyped_storage().resize_(0)
        result.resize_(0)
    elif release == "set":
        empty = torch.empty(0, dtype=result.dtype, device=result.device).untyped_storage()
        result.set_(empty, 0, (0,), (1,))


def _run_lifetime_group_calls(api_module, state, barrier, values, sentinels):
    """Retire each temporary output before the cache-release barrier."""
    for iteration in range(12):
        result = api_module.scatter_sum(state.source, state.index, dim=0, dim_size=state.buckets)
        if iteration % 4 == 3:
            values.append(result)
        else:
            _retire_group_output(result, state.release)
        del result
        # Cache release starts only after both producers retired their
        # outputs, then races with the next independent allocation.
        barrier.wait(timeout=30)
        marker = 20 + iteration
        sentinels.append(
            (
                marker,
                torch.full(state.output_shape, marker, dtype=state.source.dtype, device=state.source.device),
            )
        )
        barrier.wait(timeout=30)


def _lifetime_group_worker(api_module, state, barrier):
    """Release private outputs while retaining results and allocation sentinels."""
    torch.npu.set_device(state.source.device)
    values, sentinels = [], []
    try:
        with torch.no_grad(), torch.npu.stream(state.stream):
            _run_lifetime_group_calls(api_module, state, barrier, values, sentinels)
    except BaseException:
        barrier.abort()
        raise
    return SimpleNamespace(values=values, sentinels=sentinels, retired=12 - len(values))


def _lifetime_empty_cache(device, barrier):
    """Release cached blocks after each pair of output retirements."""
    torch.npu.set_device(device)
    try:
        for _ in range(12):
            barrier.wait(timeout=30)
            torch.npu.empty_cache()
            barrier.wait(timeout=30)
    except BaseException:
        barrier.abort()
        raise


def _verify_lifetime_results(states, results):
    """Count retained API values separately from storage-integrity sentinels."""
    sentinel_checks = 0
    for state, retained in zip(states, results):
        assert len(retained.values) == 3 and retained.retired == 9
        for value in retained.values:
            _assert_result(value, state.expected[0])
        for marker, sentinel in retained.sentinels:
            GOLDEN.assert_bitwise(sentinel, torch.full(sentinel.shape, marker, dtype=sentinel.dtype))
            sentinel_checks += 1
    return sentinel_checks


def _run_output_release(api, states, release, record_property):
    """Run producers and cache release with the same synchronization and checks."""
    for state in states:
        state.release = release
    torch.npu.synchronize()
    barrier = threading.Barrier(3)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_lifetime_group_worker, api, state, barrier) for state in states]
        cache = executor.submit(_lifetime_empty_cache, states[0].source.device, barrier)
        results = [future.result(timeout=90) for future in futures]
        cache.result(timeout=90)
    torch.npu.synchronize()
    record_property("sentinel_checks", _verify_lifetime_results(states, results))
    record_property("retired_private_outputs", sum(result.retired for result in results))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("shared_stream", (False, True))
@pytest.mark.parametrize("release", ("drop", "resize", "set"))
def test_concurrent_scatter_output_release(api, dtype, shared_stream, release, record_property):
    """Keep public Sum safe when returned storage is released before stream completion."""
    streams = [torch.npu.Stream()]
    streams.append(streams[0] if shared_stream else torch.npu.Stream())
    states = [_concurrent_group_inputs(rows, "sum", stream, dtype) for rows, stream in zip((4095, 4097), streams)]
    _run_output_release(api, states, release, record_property)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("shared_stream", (False, True))
@pytest.mark.parametrize("release", ("drop", "resize", "set"))
def test_concurrent_scatter_output_release_many_buckets(api, dtype, shared_stream, release, record_property):
    """Keep Sum output storage safe with many buckets and 64 features."""
    streams = [torch.npu.Stream()]
    streams.append(streams[0] if shared_stream else torch.npu.Stream())
    states = [
        _concurrent_group_inputs(rows, "sum", stream, dtype, geometry=(1024, 16))
        for rows, stream in zip((4095, 4097), streams)
    ]
    _run_output_release(api, states, release, record_property)


@pytest.mark.parametrize("dtype", L1)
@pytest.mark.parametrize("reduce", ("sum", "mean", "min"))
@pytest.mark.parametrize("inference", (False, True))
@pytest.mark.parametrize("rebind", ("tensor", "storage"))
def test_compact_index_rebinding_at_same_address(api, dtype, reduce, inference, rebind):
    """Reject stale groups after tensor or storage rebinding at the same address."""
    with torch.inference_mode(inference):
        state = _compact_inputs(dtype)
        changed = (state.original * 3 + 17) % 128
        payload = changed.to(state.source.device)
        original = state.index
        alias = _independent_index_slice(original)
        warm = _warm_compact(api, state, reduce)
        if rebind == "tensor":
            state.index = _independent_index_slice(original)
            assert state.index is not original
        else:
            state.index.set_(alias.untyped_storage(), 0, original.shape, original.stride())
            assert state.index is original
        assert state.index.data_ptr() == original.data_ptr()
        alias.copy_(payload)
        _verify_compact(api, state, changed, reduce, warm)
        _assert_bitwise(original, changed, "out")


def _find_reused_index(address, device):
    """Hold nonmatches so repeated allocations explore available blocks safely."""
    held = []
    for attempt in range(64):
        candidate = torch.empty(512, dtype=torch.int64, device=device)
        if candidate.data_ptr() == address:
            return candidate, attempt + 1
        held.append(candidate)
    return None, 64


@pytest.mark.parametrize("reduce", ("sum", "mean", "min"))
@pytest.mark.parametrize("inference", (False, True))
def test_compact_index_allocator_reuse_probe(api, reduce, inference, record_property):
    """Check allocator address reuse when it occurs within a bounded search."""
    with torch.inference_mode(inference):
        state = _compact_inputs(torch.float32)
        changed = (state.original * 3 + 17) % 128
        payload = changed.to(state.source.device)
        warm = _warm_compact(api, state, reduce)
        address, original = state.index.data_ptr(), weakref.ref(state.index)
        del state.index
        torch.npu.synchronize()
        gc.collect()
        candidate, attempts = _find_reused_index(address, state.source.device)
        record_property("allocator_reuse_observed", candidate is not None)
        record_property("allocator_reuse_attempts", attempts)
        record_property("allocator_original_tensor_alive", original() is not None)
        if candidate is None:
            _check_compact_warm(state, reduce, warm)
            pytest.skip("allocator address reuse was not observed in 64 bounded allocations")
        state.index = candidate
        candidate.copy_(payload)
        _verify_compact(api, state, changed, reduce, warm)
        _assert_bitwise(candidate, changed, "out")


@pytest.mark.parametrize("kind", ("negative_index", "large_index", "bad_dim", "bad_dtype", "bad_out", "bad_reduce"))
def test_invalid_parameters(api, kind):
    """Reject invalid public metadata and index values without unsafe accesses."""
    src = torch.ones(3, device="npu")
    index = torch.tensor([0, 1, 0], device="npu")
    options = {"dim_size": 2}
    if kind == "negative_index":
        index[0] = -1
    elif kind == "large_index":
        index[0] = 2
    elif kind == "bad_dim":
        options["dim"] = 1
    elif kind == "bad_dtype":
        index = index.int()
    elif kind == "bad_out":
        options["out"] = torch.ones(2, dtype=torch.int32, device="npu")
    else:
        options["reduce"] = "unsupported"
    with pytest.raises((ValueError, TypeError, IndexError, RuntimeError)):
        api.scatter(src, index, **options)


@pytest.mark.parametrize("invalid", (2**32, -(2**32), 2**63 - 1, -(2**63)))
def test_index_validation_observes_all_int64_bits(api, invalid):
    """Reject invalid mutations in every part of the INT64 index representation."""
    src = torch.ones((512, 64), device="npu")
    index = torch.arange(512, device="npu") % 128
    api.scatter_sum(src, index, dim=0, dim_size=128)
    index[0] = invalid
    with pytest.raises((IndexError, RuntimeError)):
        api.scatter_sum(src, index, dim=0, dim_size=128)
    out = torch.full((128, 64), 3.0, device="npu")
    with pytest.raises((IndexError, RuntimeError)):
        api.scatter_sum(src, index, dim=0, out=out)
    _assert_exact(out, torch.full((128, 64), 3.0), "out")


@pytest.mark.parametrize("before,after", ((32766, 32767), (32768, 32769), (98302, 98303)))
def test_index_validation_preserves_int16_limb_differences(api, before, after):
    """Detect low-bit changes across signed 16-bit limb boundaries."""
    src = torch.ones(1, device="npu")
    index = torch.tensor([before], device="npu")
    api.scatter_sum(src, index, dim_size=after + 1)
    index[0] = after
    actual = api.scatter_sum(src, index, dim_size=after + 1)
    expected = torch.zeros(after + 1)
    expected[after] = 1
    _assert_exact(actual, expected)


@pytest.mark.usefixtures("api")
def test_l1_without_cpu_extension():
    """Exercise L1 APIs while rejecting imports of the original CPU extension."""
    command = [sys.executable, str(Path(__file__).resolve()), "--reject-torch-scatter"]
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=os.environ.copy(), timeout=180)
    _merge_probe_evidence(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"l1_calls": 42' in result.stderr


def _merge_probe_evidence(stderr):
    """Retain completed child comparisons even when its later call failed."""
    events = _COMPARISONS.get()
    if events is None:
        return
    for line in stderr.splitlines():
        if '"scatter_subprocess_evidence"' in line:
            payload_start = line.index("{")
            payload = json.loads(line[payload_start:])
            events.extend(payload["scatter_subprocess_evidence"])


class _RejectCpuExtension(importlib.abc.MetaPathFinder):
    """Block the CPU package without intercepting unrelated imports."""

    def find_spec(self, fullname, path=None, target=None):
        """Keep finder protocol keywords while rejecting only the CPU extension."""
        if fullname == "torch_scatter" or fullname.startswith("torch_scatter."):
            raise ImportError("torch_scatter intentionally unavailable")
        # path and target are required protocol parameters; unrelated imports fall through.


def _dependency_probe(api_module):
    """Expose actual subprocess assertion evidence to the owning pytest node."""
    events = []
    token = _COMPARISONS.set(events)
    try:
        _run_dependency_probe(api_module)
    finally:
        _COMPARISONS.reset(token)
        for event in events:
            event["execution"] = "subprocess"
        logging.info(json.dumps({"scatter_subprocess_evidence": events}))


def _run_dependency_probe(api_module):
    assert not any(name == "torch_scatter" or name.startswith("torch_scatter.") for name in sys.modules)
    count = 0
    for dtype in L1:
        src = torch.tensor([2, 1, -1], dtype=dtype, device="npu")
        index = torch.tensor([0, 1, 0], device="npu")
        for reduce in REDUCES:
            actual = api_module.scatter(src, index, reduce=reduce)
            expected = GOLDEN.scatter(src.cpu(), index.cpu(), reduce=reduce)
            _assert_result(actual, expected)
            count += 1
    imported = any(name == "torch_scatter" or name.startswith("torch_scatter.") for name in sys.modules)
    assert not imported
    with pytest.raises(ImportError, match="torch_scatter==2.1.2"):
        api_module.scatter(
            torch.ones(2, dtype=torch.int64, device="npu"), torch.zeros(2, dtype=torch.int64, device="npu")
        )
    logging.info(json.dumps({"l1_calls": count, "torch_scatter_imported": imported}))


def _large_probe(api_module, args):
    rows, columns, buckets = 65536, args.columns, 16384
    src = torch.ones((rows, columns), device="npu", dtype=getattr(torch, args.dtype))
    index = (torch.arange(rows, device="npu") * 13) % buckets
    result = api_module.scatter(src, index, dim=0, dim_size=buckets, reduce=args.reduce)
    expected_value = rows // buckets if args.reduce in ("sum", "add") else 1
    expected = torch.full((buckets, columns), expected_value, dtype=src.dtype)
    _assert_exact(result, expected)
    logging.info(
        json.dumps({"shape": [rows, columns], "physical_index": rows, "logical_index": rows * columns, "passed": True})
    )


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--reject-torch-scatter", action="store_true")
    choice.add_argument("--large", action="store_true")
    parser.add_argument("--columns", type=int, choices=(512, 1024, 2048), default=2048)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--reduce", choices=REDUCES, default="sum")
    args = parser.parse_args()
    if args.reject_torch_scatter:
        sys.meta_path.insert(0, _RejectCpuExtension())
    importlib.import_module("torch_npu")
    api_module = importlib.import_module("ops_gnn")
    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "3")))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.reject_torch_scatter:
        _dependency_probe(api_module)
    else:
        _large_probe(api_module, args)


if __name__ == "__main__":
    _main()

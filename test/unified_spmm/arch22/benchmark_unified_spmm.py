# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Unified spmm/bspmm benchmark; timings include validation.

Two modes share this file.

``single`` (the default) times one configured case and logs its JSON report.
Run it from the repository root with PYTHONPATH=python::

    python test/unified_spmm/arch22/benchmark_unified_spmm.py
    # Higher-rank binary broadcast: [N,3,1] and [E,2,3,4] reduce to [M,2,3,4]
    python test/unified_spmm/arch22/benchmark_unified_spmm.py \\
        --op mul --reduce mean --lhs-shape 3,1 --rhs-shape 2,3,4

``cache-ab`` (issue #96) replays the host-cache matrix: 16 cases over rows
128/4096, the four paths below, and static/changing inputs. Every sample goes
into one JSON evidence file, so run it once per build label against the same
device and environment; the script neither builds nor switches code::

    PYTHONPATH=python python test/unified_spmm/arch22/benchmark_unified_spmm.py \\
        --mode cache-ab --label repaired --output /tmp/spmm-repaired.json

For non-repeated inputs, select ``--input-modes fresh fresh-e2e``. Each call
uses a new CSR and feature tensor with different values, never a warmed input.
``fresh`` measures the public operator on pre-created device inputs; allocation,
H2D and golden checks stay outside timing. ``fresh-e2e`` also includes H2D and
device allocation. Both synchronize at each timing boundary and check EVERY
output against the CPU golden. Ten separate inputs warm the runtime, not the
measured tensors. Run both builds in alternating order in separate processes.

``--label`` or ``--output`` alone also selects ``cache-ab``. Both are required
there, ``--repeat`` must be at least 2, and the output file is created
exclusively. Importing this module never imports torch or torch_npu, so
``--help`` and the CPU-only helper checks work without an NPU.
Run those helper checks explicitly (no compiler or NPU required)::

    python -m pytest -q test/unified_spmm/arch22/benchmark_unified_spmm.py
"""
import argparse
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import runpy
import statistics
import subprocess
import sys
import time
import unittest

import pytest

_LOGGER = logging.getLogger(__name__)
PATHS = ("fp16", "fp32", "batched_fp16", "copy_rhs")
MODES = ("single", "cache-ab")
INPUT_MODES = ("static", "changing", "fresh", "fresh-e2e")


def _parse_feature_shape(value, fallback):
    if value is None:
        return (fallback,)
    try:
        dimensions = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("feature shapes must be comma-separated integers") from error
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise ValueError("feature shape dimensions must be positive")
    return dimensions


def _feature_shapes(args):
    lhs_shape = _parse_feature_shape(args.lhs_shape, args.features)
    rhs_shape = _parse_feature_shape(args.rhs_shape, args.features)
    if args.batches and args.lhs_shape is None:
        lhs_shape = (args.batches, args.features)
    return lhs_shape, rhs_shape


def _resolve_mode(args):
    if args.mode is not None:
        return args.mode
    if args.label is not None or args.output is not None:
        return "cache-ab"
    return "single"


def _validate_single_arguments(parser, args):
    invalid_size = min(args.rows, args.sources, args.features, args.repeat) <= 0
    if invalid_size or args.edges < 0 or args.batches < 0:
        parser.error(
            "rows, sources, features, repeat must be positive; "
            "edges and batches nonnegative")
    if args.op == "copy_rhs" and args.dtype != "float32":
        parser.error("copy_rhs supports only float32")


def _validate_cache_arguments(parser, args):
    if args.label is None:
        parser.error("label is required in cache-ab mode")
    if args.output is None:
        parser.error("output is required in cache-ab mode")
    if args.repeat < 2:
        parser.error("repeat must be >= 2")
    if args.output.exists():
        parser.error("output already exists; choose a new evidence file")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", choices=MODES, default=None,
        help="single case run (default) or the issue #96 cache A/B matrix; "
             "--label/--output alone also select cache-ab")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--sources", type=int, default=4096)
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--features", type=int, default=33)
    parser.add_argument("--batches", type=int, default=0,
                        help="positive value uses [sources, batches, features]")
    parser.add_argument(
        "--lhs-shape", help="source feature shape without the source axis, e.g. 3,1")
    parser.add_argument(
        "--rhs-shape", help="edge feature shape without the edge axis, e.g. 2,3,4")
    parser.add_argument("--op",
                        choices=["copy_lhs", "copy_rhs", "add", "sub", "mul", "div"],
                        default="copy_lhs")
    parser.add_argument("--reduce", choices=["sum", "min", "max", "mean"], default="sum")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--distribution", choices=["uniform", "powerlaw", "hub", "empty"], default="uniform")
    parser.add_argument("--index-dtype", choices=["int32", "int64"], default="int64")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--label", help="cache-ab only; build label recorded in the evidence file")
    parser.add_argument(
        "--output", type=Path,
        help="cache-ab only; new JSON evidence file, created exclusively")
    parser.add_argument(
        "--paths", nargs="+", choices=PATHS, default=list(PATHS),
        help="cache-ab only; cache paths to sample")
    parser.add_argument(
        "--input-modes", nargs="+", choices=INPUT_MODES,
        default=["static", "changing"],
        help="cache-ab only; fresh uses unique pre-created inputs; fresh-e2e includes H2D")
    args = parser.parse_args(argv)
    args.mode = _resolve_mode(args)
    if args.mode == "cache-ab":
        _validate_cache_arguments(parser, args)
    else:
        _validate_single_arguments(parser, args)
    try:
        _feature_shapes(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def iter_cases(paths, input_modes=("static", "changing")):
    for rows, degree, features in ((128, 8, 32), (4096, 16, 64)):
        for path in paths:
            for input_mode in input_modes:
                yield {"rows": rows, "degree": degree, "features": features,
                       "path": path, "changing": input_mode == "changing",
                       "input_mode": input_mode}


def make_inputs(torch, case):
    rows, degree, features = (case[key] for key in ("rows", "degree", "features"))
    path = case["path"]
    ptr_cpu = torch.arange(rows + 1, dtype=torch.int64) * degree
    idx_cpu = torch.arange(rows * degree, dtype=torch.int64) % rows
    dtype = torch.float16 if "fp16" in path else torch.float32
    source_rows = rows * degree if path == "copy_rhs" else rows
    batch = 2 if path == "batched_fp16" else 1
    shape = (source_rows, 2, features) if batch == 2 else (source_rows, features)
    x_cpu = (torch.arange(source_rows * batch * features).reshape(shape) % 7 - 3).to(dtype)
    return ptr_cpu, idx_cpu, x_cpu


def timed_call(torch, operation, tensors, message, changing):
    ptr, idx, x = tensors
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    if changing:
        x.fill_(2)  # Include real input-write and revalidation cost.
    result = operation(ptr, idx, x, op=message, reduce="sum")
    torch.npu.synchronize()
    return (time.perf_counter_ns() - start) / 1e6, result


def benchmark_case(torch, ops_gnn, reference, case, repeat):
    if case["input_mode"].startswith("fresh"):
        return benchmark_fresh_case(torch, ops_gnn, reference, case, repeat)
    path, changing = case["path"], case["changing"]
    _LOGGER.info("starting path=%s rows=%s changing=%s", path, case["rows"], changing)
    ptr_cpu, idx_cpu, x_cpu = make_inputs(torch, case)
    tensors = (ptr_cpu.npu(), idx_cpu.npu(), x_cpu.npu())
    operation = ops_gnn.bspmm if path == "batched_fp16" else ops_gnn.spmm
    message = "copy_rhs" if path == "copy_rhs" else "copy_lhs"
    cold, got = timed_call(torch, operation, tensors, message, changing)
    _LOGGER.info("cold completed: %.4fms", cold)
    expected = reference(ptr_cpu, idx_cpu, tensors[2].cpu(), "sum", message)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    for _ in range(10):
        timed_call(torch, operation, tensors, message, changing)
    samples = []
    for _ in range(repeat):
        elapsed, got = timed_call(torch, operation, tensors, message, changing)
        samples.append(elapsed)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    result = {
        "path": path, "mode": "changing" if changing else "static",
        "rows": case["rows"], "edges": case["rows"] * case["degree"],
        "features": case["features"], "cold_ms": cold,
        "median_ms": statistics.median(samples), "min_ms": min(samples),
        "max_ms": max(samples), "samples_ms": samples, "correct": True,
    }
    _LOGGER.info("%s", json.dumps(result))
    return result


def make_fresh_inputs(torch, case, sample):
    """Deterministic, distinct valid graphs/features, independent of build label."""
    rows, degree, features = (case[key] for key in ("rows", "degree", "features"))
    edges = rows * degree
    generator = torch.Generator().manual_seed(4096 + sample)
    # Different row lengths and source permutations, not clones of one graph.
    destinations = torch.randint(rows, (edges,), generator=generator)
    counts = torch.bincount(destinations, minlength=rows)
    ptr = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    idx = torch.randint(rows, (edges,), generator=generator)
    path = case["path"]
    source_rows = edges if path == "copy_rhs" else rows
    shape = (source_rows, 2, features) if path == "batched_fp16" else (source_rows, features)
    dtype = torch.float16 if "fp16" in path else torch.float32
    x = torch.randint(-3, 4, shape, generator=generator).to(dtype)
    return ptr, idx, x


def fresh_timed_call(torch, operation, cpu_inputs, device_inputs, message):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    if device_inputs is None:
        device_inputs = tuple(value.npu() for value in cpu_inputs)
    result = operation(*device_inputs, op=message, reduce="sum")
    torch.npu.synchronize()
    return (time.perf_counter_ns() - start) / 1e6, result


def benchmark_fresh_case(torch, ops_gnn, reference, case, repeat):
    path, input_mode = case["path"], case["input_mode"]
    operation = ops_gnn.bspmm if path == "batched_fp16" else ops_gnn.spmm
    message = "copy_rhs" if path == "copy_rhs" else "copy_lhs"
    _LOGGER.info("starting path=%s rows=%s mode=%s", path, case["rows"], input_mode)
    # Keep every CPU triple alive. In fresh mode also keep every device triple
    # alive: no TensorImpl/storage address recycling can create accidental hits.
    cpu_inputs = [make_fresh_inputs(torch, case, sample) for sample in range(repeat + 11)]
    device_inputs = ([tuple(value.npu() for value in values) for values in cpu_inputs]
                     if input_mode == "fresh" else [None] * len(cpu_inputs))
    samples = []
    cold = None
    for sample, values in enumerate(cpu_inputs):
        elapsed, got = fresh_timed_call(torch, operation, values, device_inputs[sample], message)
        expected = reference(*values, "sum", message)
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        if sample == 0:
            cold = elapsed
        elif sample > 10:
            samples.append(elapsed)
    result = {
        "path": path, "mode": input_mode, "rows": case["rows"],
        "edges": case["rows"] * case["degree"], "features": case["features"],
        "cold_ms": cold, "median_ms": statistics.median(samples),
        "min_ms": min(samples), "max_ms": max(samples),
        "p95_ms": sorted(samples)[min(len(samples) - 1, math.ceil(len(samples) * .95) - 1)],
        "samples_ms": samples, "correct": True, "checked_outputs": len(cpu_inputs),
        "unique_input_triples": len(cpu_inputs), "warmup": 10,
        "includes_h2d": input_mode == "fresh-e2e", "seed_start": 4096,
    }
    _LOGGER.info("%s", json.dumps(result))
    return result


def native_hashes():
    # Resolve the actual loaded extension using the standard import API, not a
    # protected package attribute or a guessed device-specific installation.
    native = importlib.import_module("ops_gnn._pybind")
    return {
        str(library): hashlib.sha256(library.read_bytes()).hexdigest()
        for library in Path(native.__file__).parent.glob("*.so")
    }


def build_report(args, torch, torch_npu, ops_gnn):
    return {
        "label": args.label, "python": platform.python_version(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(), "ops_gnn": ops_gnn.__file__,
        "repeat": args.repeat, "paths": args.paths, "input_modes": args.input_modes, "cases": [],
        "native_sha256": native_hashes(),
    }


def save_report(output_path, report):
    # Exclusive creation also protects a file created after argument validation.
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    _LOGGER.info("%s", json.dumps({key: value for key, value in report.items()
                                   if key != "cases"}, indent=2))
    for case in report["cases"]:
        _LOGGER.info("%-14s %-8s rows=%4d median=%.4fms correct=%s",
                     case["path"], case["mode"], case["rows"],
                     case["median_ms"], case["correct"])


def _npu_runtime():
    """Import the NPU runtime after parsing, so ``--help`` works on a CPU host."""
    import torch
    import torch_npu
    import ops_gnn
    return torch, torch_npu, ops_gnn


def _make_inputs(torch, args, lhs_shape, rhs_shape):
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    torch.manual_seed(42)
    weights = torch.ones(args.rows)
    if args.distribution == "powerlaw":
        weights = torch.arange(1, args.rows + 1).float().pow(-1.5)
    if args.distribution == "hub":
        weights[0] = args.rows * 100
    if args.distribution == "empty":
        weights[1:] = 0
    if args.edges:
        rows = torch.multinomial(weights, args.edges, replacement=True)
    else:
        rows = torch.empty(0, dtype=torch.long)
    index_dtype = getattr(torch, args.index_dtype)
    counts = torch.bincount(rows, minlength=args.rows).cumsum(0)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts]).to(index_dtype)
    idx = torch.randint(args.sources, (args.edges,), dtype=index_dtype)
    dtype = getattr(torch, args.dtype)
    node_features = torch.randn((args.sources,) + lhs_shape, dtype=dtype)
    edge_features = torch.randn((args.edges,) + rhs_shape, dtype=dtype)
    if args.op == "div":
        edge_features = edge_features.abs() + 0.5
    x = edge_features if args.op == "copy_rhs" else node_features
    rhs = edge_features if args.op in ("add", "sub", "mul", "div") else None
    return ptr, idx, x, rhs


def _run_once(torch, spmm, bspmm, args, ptr, idx, x, rhs):
    torch.npu.synchronize()
    start = time.perf_counter()
    feature_rank = max(x.dim(), rhs.dim() if rhs is not None else 0)
    aggregate = bspmm if feature_rank >= 3 else spmm
    result = aggregate(ptr, idx, x, op=args.op, reduce=args.reduce, rhs=rhs)
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000, result


def _tolerances(args):
    if args.dtype == "float16" and args.reduce in ("sum", "mean"):
        return 1e-2, 5e-2
    if args.reduce in ("sum", "mean"):
        return 1e-5, 1e-6
    return 0, 0


def _report(torch, torch_npu, args, cold, times):
    return {
        "parameters": vars(args),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(),
        "NPU_ARCH": os.getenv("NPU_ARCH", os.getenv("TARGET_NPU_ARCH")),
        "cold_ms": cold,
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.pstdev(times),
    }


def run_single(args):
    torch, torch_npu, ops_gnn = _npu_runtime()
    reference = runpy.run_path(str(Path(__file__).with_name("golden.py")))[
        "unified_spmm_reference"]
    spmm, bspmm = ops_gnn.spmm, ops_gnn.bspmm
    lhs_shape, rhs_shape = _feature_shapes(args)
    ptr, idx, x, rhs = _make_inputs(torch, args, lhs_shape, rhs_shape)
    expected = reference(ptr, idx, x, args.reduce, args.op, rhs=rhs)
    ptr, idx, x = ptr.npu(), idx.npu(), x.npu()
    rhs = rhs.npu() if rhs is not None else None
    cold, got = _run_once(torch, spmm, bspmm, args, ptr, idx, x, rhs)
    rtol, atol = _tolerances(args)
    torch.testing.assert_close(got.cpu(), expected, rtol=rtol, atol=atol,
                               equal_nan=True)
    for _ in range(10):
        _run_once(torch, spmm, bspmm, args, ptr, idx, x, rhs)
    times = [_run_once(torch, spmm, bspmm, args, ptr, idx, x, rhs)[0]
             for _ in range(args.repeat)]
    _LOGGER.info(json.dumps(_report(torch, torch_npu, args, cold, times), indent=2))


def run_cache_ab(args):
    torch, torch_npu, ops_gnn = _npu_runtime()
    reference = runpy.run_path(str(Path(__file__).with_name("golden.py")))[
        "unified_spmm_reference"]
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    report = build_report(args, torch, torch_npu, ops_gnn)
    for case in iter_cases(args.paths, args.input_modes):
        report["cases"].append(
            benchmark_case(torch, ops_gnn, reference, case, args.repeat))
    save_report(args.output, report)


def test_benchmark_case_matrix():
    check = unittest.TestCase()
    cases = list(iter_cases(PATHS))
    check.assertEqual(len(cases), 16)
    keys = {(case["rows"], case["path"], case["changing"]) for case in cases}
    check.assertEqual(len(keys), 16)
    check.assertEqual({case["rows"] for case in cases}, {128, 4096})
    check.assertEqual([case["changing"] for case in cases], [False, True] * 8)


def test_fresh_case_matrix():
    check = unittest.TestCase()
    cases = list(iter_cases(PATHS, ("fresh", "fresh-e2e")))
    check.assertEqual(len(cases), 16)
    check.assertEqual(len({(case["rows"], case["path"], case["input_mode"])
                           for case in cases}), 16)
    check.assertTrue(all(not case["changing"] for case in cases))


@pytest.mark.parametrize("path", PATHS)
def test_fresh_inputs_are_distinct_and_valid(path):
    check = unittest.TestCase()
    torch = pytest.importorskip("torch")
    case = next(iter_cases([path], ("fresh",)))
    first, second = (make_fresh_inputs(torch, case, sample) for sample in (0, 1))
    duplicate = make_fresh_inputs(torch, case, 0)
    for left, right, repeated in zip(first, second, duplicate):
        check.assertNotEqual(left.data_ptr(), right.data_ptr())
        check.assertFalse(torch.equal(left, right))
        check.assertTrue(torch.equal(left, repeated))
    ptr, idx, x = first
    check.assertEqual((ptr[0].item(), ptr[-1].item()), (0, idx.numel()))
    check.assertTrue(bool((ptr[1:] >= ptr[:-1]).all()))
    check.assertTrue(bool(((idx >= 0) & (idx < case["rows"])).all()))
    check.assertEqual(x.size(0), idx.numel() if path == "copy_rhs" else case["rows"])


@pytest.mark.parametrize("path", PATHS)
def test_benchmark_cpu_inputs(path):
    check = unittest.TestCase()
    torch = pytest.importorskip("torch")
    case = next(iter_cases([path]))
    ptr, idx, x = make_inputs(torch, case)
    edges = case["rows"] * case["degree"]
    check.assertEqual(ptr.shape, (case["rows"] + 1,))
    check.assertEqual((ptr[0].item(), ptr[-1].item()), (0, edges))
    check.assertEqual(idx.numel(), edges)
    check.assertEqual(x.size(0), edges if path == "copy_rhs" else case["rows"])
    check.assertEqual(x.dim(), 3 if path == "batched_fp16" else 2)
    check.assertEqual(x.dtype, torch.float16 if "fp16" in path else torch.float32)
    check.assertEqual((x.min().item(), x.max().item()), (-3, 3))


def test_benchmark_report_does_not_overwrite(tmp_path):
    check = unittest.TestCase()
    output = tmp_path / "evidence.json"
    report = {"label": "harness-only", "cases": []}
    save_report(output, report)
    check.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_report(output, {"cases": []})
    check.assertEqual(output.read_bytes(), original)


def test_benchmark_arguments(tmp_path):
    check = unittest.TestCase()
    output = tmp_path / "fresh.json"
    args = parse_arguments([
        "--label", "check", "--output", str(output), "--paths", "fp32"])
    check.assertEqual((args.repeat, args.paths), (100, ["fp32"]))
    command = [sys.executable, __file__, "--label", "check", "--output", str(output)]
    invalid_repeat = subprocess.run(
        command + ["--repeat", "1"], text=True, capture_output=True, timeout=15, check=False)
    check.assertEqual(invalid_repeat.returncode, 2)
    check.assertIn("repeat must be >= 2", invalid_repeat.stderr)
    save_report(output, {"cases": []})
    existing_output = subprocess.run(
        command, text=True, capture_output=True, timeout=15, check=False)
    check.assertEqual(existing_output.returncode, 2)
    check.assertIn("output already exists", existing_output.stderr)


def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.mode == "cache-ab":
        run_cache_ab(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()

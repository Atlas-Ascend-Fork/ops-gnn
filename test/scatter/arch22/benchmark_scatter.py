# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Explicitly selected Scatter A2 benchmarks of public Python APIs.

Example: --device npu:0 --dtype float32 --reduce sum --shape 128 32 --json run.json
The large published matrix requires --official-shapes. Published cases use the
official self-test configuration. Other L1 baselines remain explicitly missing.
"""

import argparse
import hashlib
import importlib
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
DTYPES = ("float16", "bfloat16", "float32", "int8", "int16", "int32", "uint8")
REDUCES = ("sum", "add", "mul", "mean", "min", "max")
SHAPES = ((16384, 512), (32768, 512), (65536, 512), (65536, 1024), (65536, 2048))
REFERENCE_SHA = "ce603dc73335bbb47b3a77d0e4a63fa16f66d136"
REFERENCE_ROOT = "https://gitcode.com/cann/cann-ops-competitions/blob/" + REFERENCE_SHA
REFERENCE_PATH = "/04_tasks/01_community-task-2026/docs/202608/"
A100_MS = {
    "sum": {"float32": (0.062, 0.124, 0.254, 0.496, 1.053), "float16": (0.077, 0.181, 0.331, 0.707, 1.455)},
    "mean": {"float32": (0.096, 0.169, 0.319, 0.633, 1.252), "float16": (0.112, 0.226, 0.440, 0.820, 1.643)},
    "min": {"float32": (0.281, 0.534, 0.982, 1.875, 3.755), "float16": (0.304, 0.579, 0.955, 1.940, 3.885)},
    "max": {"float32": (0.281, 0.534, 0.927, 1.873, 3.753), "float16": (0.305, 0.483, 0.955, 1.940, 3.886)},
}


@dataclass(frozen=True)
class Runtime:
    """Imported modules and the explicitly selected target device."""

    torch: object
    npu: object
    ops: object
    device: object


@dataclass(frozen=True)
class Case:
    """One public-interface workload and its physical index extent."""

    dtype: str
    reduce: str
    rows: int
    channels: int
    buckets: int


def parse_args():
    """Validate the requested matrix and preserve existing result files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="Explicit idle card, e.g. npu:0")
    parser.add_argument("--dtype", nargs="+", required=True, choices=(*DTYPES, "all"))
    parser.add_argument("--reduce", nargs="+", required=True, choices=(*REDUCES, "all"))
    shapes = parser.add_mutually_exclusive_group(required=True)
    shapes.add_argument("--shape", nargs=2, type=int, action="append", metavar=("N", "C"))
    shapes.add_argument("--official-shapes", action="store_true")
    parser.add_argument("--buckets", type=int, help="Default max(1, N//4)")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--api", choices=("generic", "dedicated"), default="generic")
    parser.add_argument("--out", choices=("none", "zeros", "ones", "random"), default="none")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iter < 1 or args.rounds < 1:
        parser.error("warmup must be nonnegative; iter and rounds must be positive")
    if args.buckets is not None and args.buckets < 1:
        parser.error("buckets must be positive")
    if any(n < 1 or c < 1 for n, c in args.shape or ()):
        parser.error("shape dimensions must be positive")
    if args.json.exists():
        parser.error("choose a new JSON path to preserve previous rounds")
    return args


def load_runtime(args):
    """Load this worktree and select the explicitly named NPU."""
    torch = importlib.import_module("torch")
    npu = importlib.import_module("torch_npu")
    device = torch.device(args.device)
    if device.type != "npu" or device.index is None:
        raise ValueError("device must name an explicit NPU index")
    torch.npu.set_device(device)
    ops = importlib.import_module("ops_gnn")
    package = Path(ops.__file__).resolve()
    if ROOT not in package.parents:
        raise RuntimeError(f"ops_gnn must load from this worktree: {package}")
    native = importlib.import_module("ops_gnn._pybind")
    if native.npu_arch != "dav-2201" or not hasattr(native, "scatter_forward"):
        raise RuntimeError("the loaded native module does not provide Scatter A2")
    return Runtime(torch, npu, ops, device)


def selected_cases(args):
    """Expand dtype, reduction and shape selections without duplicates."""
    dtypes = DTYPES if "all" in args.dtype else tuple(dict.fromkeys(args.dtype))
    reductions = REDUCES if "all" in args.reduce else tuple(dict.fromkeys(args.reduce))
    shapes = SHAPES if args.official_shapes else tuple(map(tuple, args.shape))
    for dtype, reduce, (rows, channels) in itertools.product(dtypes, reductions, shapes):
        yield Case(dtype, reduce, rows, channels, args.buckets or max(1, rows // 4))


def source_distribution(dtype):
    """Describe the generator used for this source dtype."""
    if dtype == "uint8":
        return "uniform_integer_[0,4)"
    if dtype.startswith("int"):
        return "uniform_integer_[-3,4)"
    return "standard_normal"


def random_values(runtime, shape, dtype_name):
    """Generate source values directly on the selected device."""
    torch = runtime.torch
    dtype = getattr(torch, dtype_name)
    if dtype_name == "uint8":
        return torch.randint(0, 4, shape, dtype=dtype, device=runtime.device)
    if dtype_name.startswith("int"):
        return torch.randint(-3, 4, shape, dtype=dtype, device=runtime.device)
    return torch.randn(shape, dtype=dtype, device=runtime.device)


def make_inputs(runtime, args, case):
    """Create seeded inputs and an independent reusable out initial value."""
    torch = runtime.torch
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    src = random_values(runtime, (case.rows, case.channels), case.dtype)
    index = torch.randint(0, case.buckets, (case.rows,), dtype=torch.int64, device=runtime.device)
    if args.out == "none":
        return src, index, None, None
    if args.out == "random":
        initial = random_values(runtime, (case.buckets, case.channels), case.dtype)
    else:
        initial = torch.full(
            (case.buckets, case.channels), int(args.out == "ones"), dtype=src.dtype, device=runtime.device
        )
    return src, index, initial.clone(), initial


def make_call(runtime, args, case, inputs):
    """Include any required out restoration inside each timed public call."""
    src, index, out, initial = inputs
    kwargs = {"dim": 0, "out": out, "dim_size": case.buckets}
    if args.api == "generic":
        function = runtime.ops.scatter
        kwargs["reduce"] = case.reduce
    else:
        function = getattr(runtime.ops, "scatter_" + case.reduce)

    def invoke():
        if initial is not None:
            out.copy_(initial)
        return function(src, index, **kwargs)

    return invoke


def measure(runtime, args, invoke):
    """Warm the complete public interface and retain every timing round."""
    torch = runtime.torch
    samples = []
    with torch.no_grad():
        for _ in range(args.warmup):
            invoke()
        torch.npu.synchronize(runtime.device)
        for round_index in range(args.rounds):
            sample = measure_round(runtime, args, invoke, round_index)
            samples.append(sample)
            if sample["status"] == "error":
                break
    return samples


def measure_round(runtime, args, invoke, round_index):
    """Measure a batch including final device synchronization."""
    start = time.perf_counter_ns()
    try:
        for _ in range(args.iter):
            invoke()
        runtime.torch.npu.synchronize(runtime.device)
        elapsed = time.perf_counter_ns() - start
        return {
            "round": round_index,
            "status": "measured",
            "total_ns": elapsed,
            "npu_ms": elapsed / args.iter / 1_000_000,
        }
    except (RuntimeError, ValueError, TypeError, AttributeError) as error:
        return {
            "round": round_index,
            "status": "error",
            "npu_ms": None,
            "elapsed_before_error_ns": time.perf_counter_ns() - start,
            "error": f"{type(error).__name__}: {error}",
        }


def published_baseline(case):
    """Return only the exact published dtype, reduction and shape entry."""
    values = A100_MS.get(case.reduce, {}).get(case.dtype)
    shape = (case.rows, case.channels)
    if values is None or shape not in SHAPES:
        return None
    return values[SHAPES.index(shape)]


def case_record(args, case):
    """Record workload details before execution and baseline comparison."""
    baseline = published_baseline(case)
    configured = (
        case.buckets == case.rows // 4
        and args.out == "none"
        and args.api == "generic"
        and args.warmup == 20
        and args.iter == 100
    )
    return {
        "case_id": f"{case.reduce}_{case.dtype}_{case.rows}x{case.channels}_{case.buckets}",
        "reduce": case.reduce,
        "dtype": case.dtype,
        "shape": [case.rows, case.channels],
        "buckets": case.buckets,
        "dim": 0,
        "dim_size": case.buckets,
        "index_shape": [case.rows],
        "index_dtype": "int64",
        "index_physical_elements": case.rows,
        "index_logical_elements": case.rows * case.channels,
        "index_distribution": "uniform_integer_[0,buckets)",
        "src_distribution": source_distribution(case.dtype),
        "seed": args.seed,
        "api": args.api,
        "returns_arg": args.api == "dedicated" and case.reduce in ("min", "max"),
        "out": args.out,
        "warmup": args.warmup,
        "iterations": args.iter,
        "out_reset": "none" if args.out == "none" else "copy_initial_inside_timed_loop",
        "published_a100_ms": baseline,
        "published_parameters_matched": configured if baseline is not None else None,
        "baseline_status": baseline_status(baseline, configured),
        "status": "pending",
        "performance_pass": None,
        "samples": [],
        "mean_npu_ms": None,
        "mean_ratio": None,
        "worst_ratio": None,
    }


def baseline_status(baseline, configured):
    """Distinguish missing references from comparable published workloads."""
    if baseline is None:
        return "missing_baseline"
    return "published_comparable" if configured else "development_configuration"


def run_case(runtime, args, case):
    """Record first-call cost separately, then run all requested warm rounds."""
    row = case_record(args, case)
    try:
        inputs = make_inputs(runtime, args, case)
        invoke = make_call(runtime, args, case, inputs)
        runtime.torch.npu.synchronize(runtime.device)
        start = time.perf_counter_ns()
        with runtime.torch.no_grad():
            invoke()
        runtime.torch.npu.synchronize(runtime.device)
        row["first_call_ms"] = (time.perf_counter_ns() - start) / 1_000_000
        samples = measure(runtime, args, invoke)
        row["samples"] = samples
        summarize_measurements(row)
    except (RuntimeError, ValueError, TypeError, AttributeError) as error:
        row["status"] = "development_error"
        row["error"] = f"{type(error).__name__}: {error}"
        LOGGER.exception("Benchmark failed for %s", row["case_id"])
    LOGGER.info("%s", json.dumps(row, ensure_ascii=False))
    return row


def summarize_measurements(row):
    """Require every recorded round to meet the per-case threshold."""
    baseline = row["published_a100_ms"]
    samples = row["samples"]
    for sample in samples:
        measured = sample["status"] == "measured"
        sample["ratio"] = baseline / sample["npu_ms"] if baseline is not None and measured else None
    if any(sample["status"] == "error" for sample in samples):
        row["status"] = "development_error"
        return
    row["mean_npu_ms"] = sum(sample["npu_ms"] for sample in samples) / len(samples)
    if baseline is not None:
        row["mean_ratio"] = baseline / row["mean_npu_ms"]
        row["worst_ratio"] = min(sample["ratio"] for sample in samples)
    if row["baseline_status"] == "published_comparable":
        for sample in samples:
            sample["performance_pass"] = sample["ratio"] >= 0.6
        row["performance_pass"] = all(sample["performance_pass"] for sample in samples)
        row["status"] = "performance_pass" if row["performance_pass"] else "performance_fail"
    else:
        row["status"] = row["baseline_status"]


def capture(command):
    """Capture a bounded environment query without failing the benchmark."""
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=15, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def file_record(path):
    """Hash a source file or a loaded shared library in bounded chunks."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def native_records():
    """Record the binding and loaded operator libraries from this process."""
    paths = {Path(importlib.import_module("ops_gnn._pybind").__file__).resolve()}
    maps = Path("/proc/self/maps")
    if maps.exists():
        for line in maps.read_text(encoding="utf-8").splitlines():
            path = line.split()[-1]
            if not path.startswith("/") or ".so" not in path:
                continue
            if "ops_gnn" in path or "opsgnn" in path:
                paths.add(Path(path).resolve())
    return [file_record(path) for path in sorted(paths) if path.is_file()]


def source_records():
    """Hash computation sources and build configuration for traceability."""
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp", ".asc"}
    source_paths = {path for path in (ROOT / "csrc").rglob("*") if path.suffix in suffixes}
    source_paths.update((ROOT / "python/ops_gnn").rglob("*.py"))
    source_paths.update(ROOT / name for name in ("CMakeLists.txt", "setup.py", "setup.cfg", "pyproject.toml"))
    source_paths.add(Path(__file__))
    return [file_record(path) for path in sorted(source_paths) if path.is_file()]


def metadata(runtime, args):
    """Record versions, source identity, device state and timing controls."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "python_executable": sys.executable,
        "worktree": str(ROOT),
        "git_head": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "source_files": source_records(),
        "ops_gnn": file_record(runtime.ops.__file__),
        "native_libraries": native_records(),
        "torch": runtime.torch.__version__,
        "torch_npu": runtime.npu.__version__,
        "device": str(runtime.device),
        "device_name": runtime.torch.npu.get_device_name(runtime.device),
        "cann_target": os.environ.get("CANN_TARGET"),
        "npu_arch": os.environ.get("NPU_ARCH"),
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        "group_cache_environment": os.environ.get("OPSGNN_SCATTER_GROUP_CACHE", "default"),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch_cpu_threads": runtime.torch.get_num_threads(),
        "npu_smi_before": capture(["npu-smi", "info"]),
        "seed": args.seed,
    }


def write_results(path, payload):
    """Replace the result atomically after each completed workload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize_run(payload):
    """Count passes, failures and missing baselines as separate outcomes."""
    rows = payload["rows"]
    payload["status"] = "completed_selected_cases"
    payload["measurement_errors"] = sum(row["status"] == "development_error" for row in rows)
    payload["performance_passed"] = sum(row["performance_pass"] is True for row in rows)
    payload["performance_failed"] = sum(row["performance_pass"] is False for row in rows)
    payload["missing_baselines"] = sum(row["baseline_status"] == "missing_baseline" for row in rows)
    payload["development_cases"] = sum(row["baseline_status"] == "development_configuration" for row in rows)
    return int(payload["measurement_errors"] > 0 or payload["performance_failed"] > 0)


def main():
    """Run the selected public workloads and persist all measured rounds."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    runtime = load_runtime(args)
    payload = {
        "schema_version": 1,
        "status": "running",
        "timing": "public_api_host_batch_mean_with_final_synchronize",
        "output_lifetime": "released_after_each_public_call_as_in_official_benchmark",
        "input_generation": "seeded_on_target_device_before_timing",
        "first_call_timing": "separate synchronized public call before warmup; not a published A100 comparison",
        "threshold": 0.6,
        "a100_source": REFERENCE_ROOT + REFERENCE_PATH + "scatter_task_doc.md",
        "configuration_source": REFERENCE_ROOT + REFERENCE_PATH + "self_test_case/ops-gnn/benchmark/run_benchmark.py",
        "a100_provenance": "published_external_reference_not_measured_in_this_run",
        "metadata": metadata(runtime, args),
        "rows": [],
    }
    for case in selected_cases(args):
        payload["rows"].append(run_case(runtime, args, case))
        write_results(args.json, payload)
    payload["metadata"]["npu_smi_after"] = capture(["npu-smi", "info"])
    exit_code = summarize_run(payload)
    write_results(args.json, payload)
    LOGGER.info("Saved %d cases to %s", len(payload["rows"]), args.json)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The task's 36 timing points with explicit dtype reference contracts."""
import argparse
import csv
import hashlib
import json
import logging
import os
import runpy
from pathlib import Path
import subprocess
import shutil
import time

import torch
import ops_gnn

nearest_golden = runpy.run_path(str(Path(__file__).with_name("golden.py")))["nearest_golden"]

# The order and reference milliseconds are copied from task section 7.
CASES = [
    (1024, 1024, 3, .084, .090), (4096, 4096, 3, .125, .176),
    (8192, 8192, 3, .223, .363), (16384, 16384, 3, .583, .996),
    (32768, 32768, 3, 2.093, 3.300),
    (1024, 1024, 16, .117, .106), (1024, 1024, 32, .222, .159),
    (1024, 1024, 64, .363, .364),
    (8192, 8192, 16, 2.374, 1.391), (8192, 8192, 32, 9.111, 4.699),
    (8192, 8192, 64, 18.096, 17.758),
    (32768, 32768, 16, 35.776, 19.424), (32768, 32768, 32, 142.734, 71.623),
    (32768, 32768, 64, 286.879, 278.309),
    (4096, 1024, 3, .113, .142), (8192, 4096, 3, .175, .276),
    (16384, 8192, 3, .371, .650), (32768, 16384, 3, 1.090, 1.916),
]


def code_hashes():
    repo = Path(__file__).resolve().parents[3]
    files = list((repo / "csrc/npu/nearest").rglob("*.cpp"))
    files += list((repo / "csrc/npu/nearest").rglob("*.h"))
    files += [repo / "python/ops_gnn/nearest.py", repo / "CMakeLists.txt"]
    files += list((repo / "test/nearest").rglob("*.py"))
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


@torch.no_grad()
def bench(x, y, warmup, iterations):
    for _ in range(warmup):
        ops_gnn.nearest(x, y)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        ops_gnn.nearest(x, y)
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def run_case(shape, dtype, reference, seed, options):
    n, m, f = shape
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, f, generator=generator).to(dtype)
    y = torch.randn(m, f, generator=generator).to(dtype)
    expected = nearest_golden(x, y)
    xn, yn = x.npu(), y.npu()
    actual = ops_gnn.nearest(xn, yn).cpu()
    mismatches = int((actual != expected).sum())
    latency = bench(xn, yn, options.warmup, options.iter)
    ratio = reference / latency
    result = {"N": n, "M": m, "F": f, "dtype": str(dtype).split(".")[-1],
              "precision_reference": "scipy_fp32",
              "seed": seed, "mismatches": mismatches,
              "precision_pass": mismatches == 0, "npu_ms": latency,
              "reference_ms": reference, "ratio": ratio, "performance_pass": ratio >= .45,
              "pass": mismatches == 0 and ratio >= .45}
    return result


def main():
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git executable is required to record source provenance")
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", default="artifacts/nearest/performance.json")
    parser.add_argument("--case", type=int, help="Diagnostic: run one 0-based shape row")
    args = parser.parse_args()
    if args.iter < 1 or args.warmup < 0:
        parser.error("iter must be positive and warmup nonnegative")
    if args.case is not None and not 0 <= args.case < len(CASES):
        parser.error("case must be in [0, 17]")
    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))
    torch.set_num_threads(4)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"git_commit": subprocess.check_output([git_executable, "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": code_hashes(), "seed": args.seed, "warmup": args.warmup,
              "iterations": args.iter, "device": torch.npu.get_device_name(),
              "timing": "whole Python API, perf_counter; synchronization outside timed loop",
              "precision_contract": {"float32": "task-text scipy FP32 vq",
                                     "float16": "task-text scipy FP32 vq after exact half promotion"},
              "cases": []}
    logging.info("%s", f"nearest | {report['device']} | warmup={args.warmup} iter={args.iter} seed={args.seed}")
    logging.info("%s", "      N       M    F   dtype  precision     npu_ms     ref_ms     ratio  gate")
    for row, (n, m, f, ref32, ref16) in enumerate(CASES):
        if args.case is not None and row != args.case:
            continue
        for dtype, reference in [(torch.float32, ref32), (torch.float16, ref16)]:
            result = run_case((n, m, f), dtype, reference, args.seed + row, args)
            report["cases"].append(result)
            logging.info("%s", f"{n:7d} {m:7d} {f:4d} {result['dtype']:>7s}  "
                  f"{('PASS' if result['precision_pass'] else 'FAIL'):>9s} {result['npu_ms']:10.6f} "
                  f"{reference:10.6f} {result['ratio']:9.3f}  {'PASS' if result['pass'] else 'FAIL'}")
            path.write_text(json.dumps(report, indent=2) + "\n")
    report["all_pass"] = all(c["pass"] for c in report["cases"])
    report["full_matrix"] = len(report["cases"]) == 36
    path.write_text(json.dumps(report, indent=2) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(report["cases"][0]))
        writer.writeheader()
        writer.writerows(report["cases"])
    passed = sum(c["pass"] for c in report["cases"])
    logging.info("PASS %d/%d; min ratio=%.3f", passed, len(report['cases']),
                 min(case['ratio'] for case in report['cases']))
    raise SystemExit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""32-shape KNN performance benchmark for Ascend 950."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch
import torch_npu  # noqa: F401

from ops_gnn import knn

LOGGER = logging.getLogger(__name__)

REFERENCE_ROWS = [
    (1024, 32, 8, 5.671, 3.092),
    (4096, 32, 8, 21.201, 11.453),
    (8192, 32, 8, 41.118, 22.579),
    (16384, 32, 8, 81.042, 44.858),
    (1024, 3, 8, 0.896, 0.948),
    (1024, 16, 8, 1.916, 1.472),
    (1024, 64, 8, 10.251, 10.057),
    (4096, 3, 8, 2.594, 2.766),
    (4096, 16, 8, 6.659, 5.037),
    (4096, 64, 8, 39.374, 38.894),
    (8192, 3, 8, 4.887, 5.159),
    (8192, 16, 8, 12.940, 9.912),
    (8192, 64, 8, 77.854, 77.159),
    (16384, 3, 8, 9.467, 9.962),
    (16384, 16, 8, 25.472, 19.587),
    (16384, 64, 8, 154.470, 153.362),
    (1024, 32, 1, 4.603, 2.591),
    (1024, 32, 4, 5.060, 2.767),
    (1024, 32, 16, 6.977, 4.273),
    (1024, 32, 32, 12.606, 8.998),
    (4096, 32, 1, 18.171, 10.097),
    (4096, 32, 4, 19.484, 10.601),
    (4096, 32, 16, 24.457, 13.800),
    (4096, 32, 32, 34.496, 23.199),
    (8192, 32, 1, 36.356, 20.183),
    (8192, 32, 4, 38.614, 21.005),
    (8192, 32, 16, 46.718, 26.196),
    (8192, 32, 32, 60.642, 41.877),
    (16384, 32, 1, 72.807, 40.270),
    (16384, 32, 4, 76.747, 41.795),
    (16384, 32, 16, 90.138, 51.096),
    (16384, 32, 32, 111.646, 73.052),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", "--iter", dest="iters", type=int, default=100)
    parser.add_argument("--device", default=f"npu:{os.getenv('NPU_DEVICE_ID', '0')}")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


@torch.no_grad()
def measure(x, k, warmup, iters):
    for _ in range(warmup):
        knn(x, x, k)
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    for _ in range(iters):
        knn(x, x, k)
    torch.npu.synchronize()
    return (time.perf_counter_ns() - started) / iters / 1e6


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("warmup must be >= 0 and iters must be > 0")
    torch.npu.set_device(args.device)
    rows = REFERENCE_ROWS
    if args.quick:
        rows = [REFERENCE_ROWS[0], REFERENCE_ROWS[3],
                REFERENCE_ROWS[7], REFERENCE_ROWS[-1]]

    LOGGER.info("device=%s warmup=%s iters=%s", args.device, args.warmup, args.iters)
    LOGGER.info("      N    F   k    dtype measured_ms reference_ms budget_ms ratio verdict")
    results = []
    for n, f, k, ref32, ref16 in rows:
        for dtype_name, dtype, reference in (
            ("float32", torch.float32, ref32),
            ("float16", torch.float16, ref16),
        ):
            torch.manual_seed(42)
            x = torch.randn(n, f, dtype=dtype, device=args.device)
            measured = measure(x, k, args.warmup, args.iters)
            budget = reference / 0.45
            ratio = reference / measured
            passed = measured <= budget
            record = {
                "n": n, "f": f, "k": k, "dtype": dtype_name,
                "warmup": args.warmup, "iters": args.iters,
                "reference_ms": reference, "budget_ms": budget,
                "measured_ms": measured, "ratio": ratio, "pass": passed,
            }
            results.append(record)
            LOGGER.info("%7d %4d %3d %8s %11.3f %12.3f %9.3f %5.2f %s",
                        n, f, k, dtype_name, measured, reference, budget, ratio,
                        'PASS' if passed else 'FAIL')

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2) + "\n")
    failures = sum(not row["pass"] for row in results)
    LOGGER.info("RESULT: %d/%d rows passed", len(results) - failures, len(results))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

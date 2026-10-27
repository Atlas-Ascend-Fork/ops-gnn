# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""End-to-end FPS NPU latency gate for the 42 official task points."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch
import torch_npu  # noqa: F401

import ops_gnn


class _StdoutProtocolHandler(logging.Handler):
    """Keep protocol writes and their failures independent of log settings."""

    def emit(self, record: logging.LogRecord) -> None:
        stream = sys.stdout
        if stream is None:
            return
        stream.write(self.format(record))
        stream.write(record.end)
        if record.flush:
            stream.flush()


_STDOUT_HANDLER = _StdoutProtocolHandler()
_STDOUT_HANDLER.setFormatter(logging.Formatter("%(message)s"))


def emit_stdout(message: str, end: str = "\n", flush: bool = False) -> None:
    """Emit one string with the caller's original terminator and flush policy."""
    record = logging.LogRecord(
        __name__, logging.INFO, __file__, 0, message, (), None
    )
    record.end = end
    record.flush = flush
    _STDOUT_HANDLER.handle(record)


MINIMUM_PERFORMANCE_RATIO = 0.45

# (N, F, ratio, official float32 ms, official float16 ms)
CASES = (
    (1024, 4, 0.25, 0.540, 0.630),
    (1024, 4, 0.50, 0.911, 1.088),
    (1024, 4, 1.00, 1.647, 2.012),
    (4096, 4, 0.25, 3.743, 5.237),
    (4096, 4, 0.50, 7.323, 10.343),
    (4096, 4, 1.00, 14.488, 20.497),
    (8192, 4, 0.25, 12.834, 18.676),
    (8192, 4, 0.50, 25.471, 37.179),
    (8192, 4, 1.00, 50.685, 74.123),
    (16384, 4, 0.25, 111.219, 70.868),
    (16384, 4, 0.50, 222.085, 141.659),
    (16384, 4, 1.00, 444.082, 283.090),
    (1024, 16, 0.50, 2.976, 2.020),
    (1024, 32, 0.50, 9.732, 5.582),
    (1024, 64, 0.50, 18.705, 19.755),
    (4096, 16, 0.50, 42.563, 25.288),
    (4096, 32, 0.50, 147.654, 82.835),
    (4096, 64, 0.50, 290.735, 304.279),
    (8192, 16, 0.50, 163.339, 123.059),
    (8192, 32, 0.50, 585.604, 324.389),
    (8192, 64, 0.50, 1159.109, 1211.818),
)


def _dtype_from_name(name):
    names = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return names[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "dtype must be float16/fp16 or float32/fp32"
        ) from exc


@torch.no_grad()
def _measure(call, warmup, repeats):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def _run(warmup, repeats, dtypes, case_ids):
    records = []
    torch.manual_seed(42)
    for case_id in case_ids:
        num_points, features, ratio, float32_ms, float16_ms = CASES[
            case_id - 1
        ]
        for dtype in dtypes:
            src = torch.randn(
                num_points, features, dtype=dtype, device="npu"
            )
            elapsed_ms = _measure(
                lambda: ops_gnn.fps(
                    src, None, ratio, False
                ),
                warmup,
                repeats,
            )
            reference_ms = (
                float32_ms if dtype == torch.float32 else float16_ms
            )
            threshold_ms = reference_ms / MINIMUM_PERFORMANCE_RATIO
            performance_ratio = reference_ms / elapsed_ms
            record = {
                "case": case_id,
                "n": num_points,
                "features": features,
                "ratio": ratio,
                "dtype": str(dtype).replace("torch.", ""),
                "elapsed_ms": elapsed_ms,
                "reference_ms": reference_ms,
                "threshold_ms": threshold_ms,
                "performance_ratio": performance_ratio,
                "pass": elapsed_ms <= threshold_ms,
            }
            records.append(record)
            status = "PASS" if record["pass"] else "FAIL"
            emit_stdout(
                f"{status} case={case_id:02d} N={num_points:5d} "
                f"F={features:3d} r={ratio:.2f} {record['dtype']:7s} "
                f"elapsed={elapsed_ms:9.3f}ms "
                f"limit={threshold_ms:9.3f}ms "
                f"ratio={performance_ratio:.3f}x",
                flush=True,
            )
    return records


def _write_json(path, records, warmup, repeats):
    payload = {
        "warmup": warmup,
        "iterations": repeats,
        "minimum_performance_ratio": MINIMUM_PERFORMANCE_RATIO,
        "passed": sum(record["pass"] for record in records),
        "total": len(records),
        "records": records,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path, records):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iter", dest="repeats", type=int, default=100)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", type=_dtype_from_name, default=None)
    parser.add_argument("--case", type=int, action="append", dest="case_ids")
    parser.add_argument("--json-output")
    parser.add_argument("--csv-output")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.device != "npu":
        raise ValueError("the official FPS performance gate requires --device npu")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be >= 0 and iter must be > 0")
    case_ids = args.case_ids or list(range(1, len(CASES) + 1))
    if any(case_id < 1 or case_id > len(CASES) for case_id in case_ids):
        raise ValueError(f"case must be in [1, {len(CASES)}]")

    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    if not torch.npu.is_available():
        raise RuntimeError("an available Ascend NPU is required")
    dtypes = (
        (args.dtype,)
        if args.dtype is not None
        else (torch.float32, torch.float16)
    )

    emit_stdout(
        f"device=npu warmup={args.warmup} iterations={args.repeats} "
        f"required_ratio={MINIMUM_PERFORMANCE_RATIO:.2f}"
    )
    records = _run(args.warmup, args.repeats, dtypes, case_ids)
    passed = sum(record["pass"] for record in records)
    minimum_ratio = min(record["performance_ratio"] for record in records)
    emit_stdout(
        f"SUMMARY passed={passed}/{len(records)} "
        f"minimum_ratio={minimum_ratio:.3f}x"
    )

    if args.json_output:
        _write_json(args.json_output, records, args.warmup, args.repeats)
    if args.csv_output:
        _write_csv(args.csv_output, records)
    if passed != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

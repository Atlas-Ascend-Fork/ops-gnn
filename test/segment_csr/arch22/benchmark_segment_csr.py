# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Segment CSR A2 public-interface latency benchmark (milliseconds).

Usage: python3 benchmark_segment_csr.py [--reduce sum] [--iter 100] [--warmup 20]
"""

import argparse
import csv
from pathlib import Path
import logging
import os
import time
from typing import NamedTuple

import torch
import ops_gnn

_BASE_CONFIGS = ((256, 16), (512, 32), (1024, 32), (2048, 64), (4096, 64))
_MAX_ELEMENTS = 4096 * 128


def benchmark_shapes():
    shapes = []
    for n_k, seg_k in _BASE_CONFIGS:
        for channels in (32, 64, 128, 256):
            if n_k * channels <= _MAX_ELEMENTS:
                label = f" N={n_k:>4}K C={channels:>3} seg={seg_k:>3}K"
                shapes.append((label, n_k * 1024, channels, seg_k * 1024))
    for n_k in (256, 1024, 4096):
        for seg_k in (8, 16, 64, 128):
            if (n_k, seg_k) not in ((256, 16), (4096, 64)):
                label = f" N={n_k:>4}K C=128 seg={seg_k:>3}K"
                shapes.append((label, n_k * 1024, 128, seg_k * 1024))
    return shapes


SHAPES = benchmark_shapes()
DTYPES = (torch.float32, torch.float16, torch.int32, torch.int64)


def setup_device(name):
    device = torch.device(name)
    if device.type == "npu":
        index = device.index if device.index is not None else int(os.getenv("NPU_DEVICE_ID", "0"))
        torch.npu.set_device(index)
        device = torch.device(f"npu:{index}")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


class InputShape(NamedTuple):
    rows: int
    channels: int
    segments: int


def make_inputs(shape, reduce, dtype, device):
    n, channels, segments = shape
    if dtype in (torch.int32, torch.int64):
        src = torch.randint(-100, 100, (n, channels), dtype=dtype, device=device)
    else:
        src = torch.randn(n, channels, dtype=dtype, device=device)
    indptr = torch.arange(segments + 1, dtype=torch.int64, device=device) * (n // segments)
    indptr[-1] = n
    return [src, indptr, None, reduce]


@torch.no_grad()
def bench(tensor_args, warmup, iters, device):
    for _ in range(warmup):
        ops_gnn.segment_csr(*tensor_args)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        ops_gnn.segment_csr(*tensor_args)
    synchronize(device)
    return (time.perf_counter() - start) / iters * 1000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--reduce", default="sum", choices=("all", "sum", "mean", "min", "max"))
    parser.add_argument(
        "--dtype", default="all", choices=("all", "float32", "float16", "int32", "int64")
    )
    parser.add_argument(
        "--shape-index",
        type=int,
        choices=range(len(SHAPES)),
        help="run one shape instead of all 29",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="optional CSV with unrounded latency_ms")
    args = parser.parse_args()
    if args.iter <= 0 or args.warmup < 0:
        parser.error("--iter must be positive and --warmup must be nonnegative")
    return args


def measure_matrix(args, device, reductions, dtypes):
    rows = []
    logging.info("device=%s warmup=%d iters=%d seed=%d", device, args.warmup, args.iter, args.seed)
    for reduction in reductions:
        logging.info("segment_csr %s (ms)", reduction)
        logging.info(
            "%35s%s",
            "shape",
            "".join(f" {str(dtype).removeprefix('torch.'):>12s}" for dtype in dtypes),
        )
        for shape_index, (label, n, channels, segments) in enumerate(SHAPES):
            if args.shape_index is not None and shape_index != args.shape_index:
                continue
            values = []
            for dtype in dtypes:
                tensor_args = make_inputs(
                    InputShape(n, channels, segments), reduction, dtype, device
                )
                latency = bench(tensor_args, args.warmup, args.iter, device)
                values.append(latency)
                rows.append(
                    {
                        "reduce": reduction,
                        "dtype": str(dtype).removeprefix("torch."),
                        "shape_index": shape_index,
                        "N": n,
                        "C": channels,
                        "segments": segments,
                        "latency_ms": latency,
                        "warmup": args.warmup,
                        "iterations": args.iter,
                        "seed": args.seed,
                    }
                )
                del tensor_args
            logging.info("%35s%s", label, "".join(f" {value:12.6f}" for value in values))
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = setup_device(args.device)
    if device.type != "npu" or not torch.npu.get_device_name().startswith("Ascend910B"):
        parser.error("arch22 benchmark requires an Ascend A2 NPU")
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    reductions = ("sum", "mean", "min", "max") if args.reduce == "all" else (args.reduce,)
    dtypes = DTYPES if args.dtype == "all" else (getattr(torch, args.dtype),)
    rows = measure_matrix(args, device, reductions, dtypes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logging.info("Saved %d measurements to %s", len(rows), args.output)


if __name__ == "__main__":
    main()

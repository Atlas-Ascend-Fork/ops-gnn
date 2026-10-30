# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run from the repository root with PYTHONPATH=python; timings include validation."""
import argparse
import json
import logging
import os
from pathlib import Path
import runpy
import platform
import statistics
import time

import torch
import torch_npu
from ops_gnn import bspmm, spmm
unified_spmm_reference = runpy.run_path(str(Path(__file__).with_name("golden.py")))["unified_spmm_reference"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOGGER = logging.getLogger(__name__)


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


def _parse_args():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    invalid_size = min(args.rows, args.sources, args.features, args.repeat) <= 0
    if invalid_size or args.edges < 0 or args.batches < 0:
        parser.error(
            "rows, sources, features, repeat must be positive; "
            "edges and batches nonnegative")
    if args.op == "copy_rhs" and args.dtype != "float32":
        parser.error("copy_rhs supports only float32")
    try:
        lhs_shape = _parse_feature_shape(args.lhs_shape, args.features)
        rhs_shape = _parse_feature_shape(args.rhs_shape, args.features)
        if args.batches and args.lhs_shape is None:
            lhs_shape = (args.batches, args.features)
    except ValueError as error:
        parser.error(str(error))
    return args, lhs_shape, rhs_shape


def _make_inputs(args, lhs_shape, rhs_shape):
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


def _run_once(args, ptr, idx, x, rhs):
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


def _report(args, cold, times):
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


def main():
    args, lhs_shape, rhs_shape = _parse_args()
    ptr, idx, x, rhs = _make_inputs(args, lhs_shape, rhs_shape)
    expected = unified_spmm_reference(
        ptr, idx, x, args.reduce, args.op, rhs=rhs)
    ptr, idx, x = ptr.npu(), idx.npu(), x.npu()
    rhs = rhs.npu() if rhs is not None else None
    cold, got = _run_once(args, ptr, idx, x, rhs)
    rtol, atol = _tolerances(args)
    torch.testing.assert_close(got.cpu(), expected, rtol=rtol, atol=atol,
                               equal_nan=True)
    for _ in range(10):
        _run_once(args, ptr, idx, x, rhs)
    times = [_run_once(args, ptr, idx, x, rhs)[0] for _ in range(args.repeat)]
    _LOGGER.info(json.dumps(_report(args, cold, times), indent=2))


if __name__ == "__main__":
    main()

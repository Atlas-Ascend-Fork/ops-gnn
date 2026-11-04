# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark CSR SDDMM. Run with PYTHONPATH=python from the repository root."""
import argparse
import logging
import os
import statistics
import sys
import time

import torch

from ops_gnn import sddmm_csr


LOGGER = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--op", choices=["dot", "add", "sub", "mul", "div", "copy_lhs", "copy_rhs"],
                        default="dot")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--index-dtype", choices=["int32", "int64"], default="int64")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.features, args.repeat) <= 0 or args.edges < 0:
        parser.error("rows, columns, features, repeat must be positive; edges nonnegative")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = _parse_args()
    torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
    torch.manual_seed(42)

    index_dtype = getattr(torch, args.index_dtype)
    dtype = getattr(torch, args.dtype)
    src = torch.randint(args.rows, (args.edges,))
    dst = torch.randint(args.columns, (args.edges,))
    order = torch.argsort(src, stable=True)
    counts = torch.bincount(src[order], minlength=args.rows)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).to(index_dtype)
    idx = dst[order].to(index_dtype)

    lhs = torch.randn(args.rows, args.features, dtype=dtype)
    rhs = torch.randn(args.columns, args.features, dtype=dtype)
    if args.op == "div":
        rhs = rhs.abs() + 0.5
    ptr = ptr.npu()
    idx = idx.npu()
    lhs = lhs.npu()
    rhs = rhs.npu()

    call = {
        "lhs": lhs,
        "rhs": rhs,
        "op": args.op,
    }
    if args.op == "copy_lhs":
        call["rhs"] = None
    elif args.op == "copy_rhs":
        call["lhs"] = None

    for _ in range(args.warmup):
        sddmm_csr(ptr, idx, **call)
    torch.npu.synchronize()

    samples = []
    for _ in range(args.repeat):
        torch.npu.synchronize()
        start = time.perf_counter()
        sddmm_csr(ptr, idx, **call)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)

    LOGGER.info("op=%s dtype=%s rows=%d cols=%d edges=%d features=%d",
                args.op, args.dtype, args.rows, args.columns,
                args.edges, args.features)
    LOGGER.info("mean=%.1f us  p50=%.1f us  min=%.1f us",
                statistics.mean(samples), statistics.median(samples),
                min(samples))


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Eternal_ZCL.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate SpMM inputs and report synchronized warm-call NPU latency."""

import argparse
from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
import runpy
import time

import torch

COMBOS = runpy.run_path(str(Path(__file__).with_name("golden.py")))["COMBOS"]


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "int8": torch.int8}
SHAPES = [(256, 256, 16, 2048), (1024, 1024, 64, 8192), (4096, 4096, 128, 32768), (4096, 4096, 256, 65536)]


@dataclass(frozen=True)
class BenchmarkCase:
    shape: tuple
    dtype: torch.dtype
    output_dtype: torch.dtype
    reduce: str
    device: torch.device


def make_inputs(shape, dtype, device):
    m, n, k, edges = shape
    generator = torch.Generator().manual_seed(41)
    row = torch.randint(m, (edges,), generator=generator).sort().values
    col = torch.randint(n, (edges,), generator=generator)
    ptr = torch.cat((torch.zeros(1, dtype=torch.int64), torch.bincount(row, minlength=m).cumsum(0)))
    if dtype == torch.int8:
        value = torch.randint(-4, 5, (edges,), generator=generator, dtype=dtype)
        mat = torch.randint(-4, 5, (n, k), generator=generator, dtype=dtype)
    else:
        value = (torch.rand(edges, generator=generator) * 2 - 1).to(dtype)
        mat = (torch.rand((n, k), generator=generator) * 2 - 1).to(dtype)
    return tuple(t.to(device) for t in (row, ptr, col, value, mat))


@torch.no_grad()
def benchmark(call, warmup, iterations, device):
    for _ in range(warmup):
        call()
    torch.npu.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.npu.synchronize(device)
    return (time.perf_counter() - start) / iterations * 1000


def run_case(adapter, args, case):
    row, ptr, col, value, mat = make_inputs(case.shape, case.dtype, case.device)
    if args.format == "coo":
        index = torch.stack((row, col))
        call = partial(adapter.coo_spmm, index, value, case.shape[0], case.shape[1], mat)
    else:
        call = partial(adapter.csr_spmm, ptr, col, value, mat, case.reduce, output_dtype=case.output_dtype)
    latency = benchmark(call, args.warmup, args.iterations, case.device)
    logging.info(
        "%s %s %s -> %s M/N/K/E=%s mean_call_ms=%.6f",
        args.format, case.reduce, case.dtype, case.output_dtype, case.shape, latency
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter", dest="iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--format", choices=("csr", "coo"), default="csr")
    parser.add_argument("--reduce", choices=("sum", "mean", "min", "max", "all"), default="sum")
    parser.add_argument("--dtype", choices=(*DTYPES, "all"), default="all")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("--iter and --warmup must be positive")
    if args.format == "coo" and args.reduce != "sum":
        parser.error("COO supports --reduce sum only")
    import torch_npu  # Register the NPU device before constructing it.
    from ops_gnn import torch_sparse_spmm

    device = torch.device(args.device)
    if device.type != "npu" or not torch_npu.npu.is_available():
        parser.error("requires an available Ascend NPU")
    torch.npu.set_device(device)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reductions = ("sum", "mean", "min", "max") if args.reduce == "all" else (args.reduce,)
    for dtype, outdtype in COMBOS:
        if args.format == "coo" and outdtype != (torch.int32 if dtype == torch.int8 else dtype):
            continue
        if args.dtype != "all" and dtype != DTYPES[args.dtype]:
            continue
        for shape in SHAPES:
            for reduce in reductions:
                case = BenchmarkCase(shape, dtype, outdtype, reduce, device)
                run_case(torch_sparse_spmm, args, case)


if __name__ == "__main__":
    main()

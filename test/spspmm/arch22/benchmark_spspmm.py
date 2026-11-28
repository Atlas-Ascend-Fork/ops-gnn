# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark ``spspmm`` for the arch22 build (A2/A3, dav-2201).

Two timings are reported per shape: ``kernel`` measures the native path with
already coalesced inputs (``coalesced=True``), ``default`` measures the public
default path, which first merges the raw COO operands on the device. Each row is
the median of three rounds, synchronising outside the timed loop.
"""

import argparse
import importlib
import logging
import os
import statistics
import time

import torch

import ops_gnn

# (m, k, n, average out-degree of A, average out-degree of B)
CASES = [
    (1000, 1000, 1000, 8, 8),
    (5000, 5000, 5000, 8, 8),
    (20000, 20000, 20000, 4, 4),
    (2000, 2000, 2000, 32, 32),
]
DTYPES = [torch.float32, torch.float16]


def _raw_operand(num_rows, num_cols, edges, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    index = torch.stack([
        torch.randint(num_rows, (edges,), generator=generator),
        torch.randint(num_cols, (edges,), generator=generator),
    ])
    value = torch.rand(edges, generator=generator, dtype=torch.float32).to(dtype)
    return index, value


def _to_coalesced(index, value):
    merged = {}
    for row, col, weight in zip(index[0].tolist(), index[1].tolist(), value.float().tolist()):
        key = (row, col)
        merged[key] = merged.get(key, 0.0) + weight
    keys = sorted(merged)
    index = torch.tensor([[key[0] for key in keys], [key[1] for key in keys]],
                         dtype=torch.int64).reshape(2, len(keys))
    return index, torch.tensor([merged.get(key, 0.0) for key in keys], dtype=value.dtype)


def _time_once(index_a, value_a, index_b, value_b, m, k, n, coalesced, iterations):
    for _ in range(3):
        ops_gnn.spspmm(index_a, value_a, index_b, value_b, m, k, n, coalesced=coalesced)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        ops_gnn.spspmm(index_a, value_a, index_b, value_b, m, k, n, coalesced=coalesced)
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def _median(index_a, value_a, index_b, value_b, m, k, n, coalesced, iterations):
    return statistics.median([
        _time_once(index_a, value_a, index_b, value_b, m, k, n, coalesced, iterations)
        for _ in range(3)
    ])


def _row(shape, degrees, name, mode, milliseconds):
    m, k, n = shape
    degree_a, degree_b = degrees
    logging.info("| %d x %d x %d | %d/%d | %s | %s | %.3f |",
                 m, k, n, degree_a, degree_b, name, mode, milliseconds)


def main(cases=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--skip-default", action="store_true")
    args = parser.parse_args()
    if args.iter < 1:
        parser.error("iter must be positive")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    shapes = CASES if cases is None else cases

    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))
    native = importlib.import_module("ops_gnn._pybind")
    logging.info("device: %s | arch: %s", torch.npu.get_device_name(), native.npu_arch)
    logging.info("")
    logging.info("| 形状 (m x k x n) | 平均度 A/B | dtype | 模式 | 中位耗时 (ms) |")
    logging.info("| --- | --- | --- | --- | --- |")
    for m, k, n, degree_a, degree_b in shapes:
        shape, degrees = (m, k, n), (degree_a, degree_b)
        for dtype in DTYPES:
            name = str(dtype).split(".")[-1]
            raw_a = _raw_operand(m, k, m * degree_a, dtype, args.seed)
            raw_b = _raw_operand(k, n, k * degree_b, dtype, args.seed + 1)
            col_a = _to_coalesced(*raw_a)
            col_b = _to_coalesced(*raw_b)
            kernel_ms = _median(col_a[0].npu(), col_a[1].npu(), col_b[0].npu(),
                                col_b[1].npu(), m, k, n, True, args.iter)
            _row(shape, degrees, name, "kernel", kernel_ms)
            if args.skip_default:
                continue
            default_ms = _median(raw_a[0].npu(), raw_a[1].npu(), raw_b[0].npu(),
                                 raw_b[1].npu(), m, k, n, False, args.iter)
            _row(shape, degrees, name, "default", default_ms)


if __name__ == "__main__":
    main()

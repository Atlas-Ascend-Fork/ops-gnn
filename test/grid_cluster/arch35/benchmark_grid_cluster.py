# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reproducible task benchmark: explicit bounds, complete public Python call."""
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import ops_gnn
from grid_cluster_golden import grid_cluster_cpu

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--iter', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', default='grid-benchmark.csv')
    parser.add_argument('--dims', type=int, nargs='+')
    parser.add_argument('--n-m', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.iter <= 0 or args.warmup < 0:
        parser.error('iter must be positive and warmup must be nonnegative')
    return args, parser


BASELINE_CASES = (
    (1, 3, 0.005, 'float32', 0.014),
    (1, 3, 0.005, 'float16', 0.015),
    (4, 3, 0.005, 'float32', 0.062),
    (4, 3, 0.005, 'float16', 0.064),
    (8, 3, 0.005, 'float32', 0.119),
    (8, 3, 0.005, 'float16', 0.124),
    (16, 3, 0.005, 'float32', 0.233),
    (16, 3, 0.005, 'float16', 0.243),
    (32, 3, 0.005, 'float32', 0.461),
    (32, 3, 0.005, 'float16', 0.482),
    (1, 16, 0.01, 'float32', 0.056),
    (1, 16, 0.01, 'float16', 0.048),
    (1, 32, 0.01, 'float32', 0.166),
    (1, 32, 0.01, 'float16', 0.099),
    (1, 64, 0.01, 'float32', 0.321),
    (1, 64, 0.01, 'float16', 0.322),
    (4, 16, 0.01, 'float32', 0.197),
    (4, 16, 0.01, 'float16', 0.172),
    (4, 32, 0.01, 'float32', 0.636),
    (4, 32, 0.01, 'float16', 0.366),
    (4, 64, 0.01, 'float32', 1.251),
    (4, 64, 0.01, 'float16', 1.257),
    (8, 16, 0.01, 'float32', 0.384),
    (8, 16, 0.01, 'float16', 0.336),
    (8, 32, 0.01, 'float32', 1.249),
    (8, 32, 0.01, 'float16', 0.717),
    (8, 64, 0.01, 'float32', 2.459),
    (8, 64, 0.01, 'float16', 2.471),
    (32, 16, 0.01, 'float32', 1.503),
    (32, 16, 0.01, 'float16', 1.315),
    (32, 32, 0.01, 'float32', 4.926),
    (32, 32, 0.01, 'float16', 2.827),
    (32, 64, 0.01, 'float32', 9.71),
    (32, 64, 0.01, 'float16', 9.755),
    (1, 3, 0.001, 'float32', 0.014),
    (1, 3, 0.001, 'float16', 0.015),
    (1, 3, 0.002, 'float32', 0.014),
    (1, 3, 0.002, 'float16', 0.015),
    (1, 3, 0.01, 'float32', 0.014),
    (1, 3, 0.01, 'float16', 0.015),
    (8, 3, 0.001, 'float32', 0.119),
    (8, 3, 0.001, 'float16', 0.124),
    (8, 3, 0.002, 'float32', 0.119),
    (8, 3, 0.002, 'float16', 0.124),
    (8, 3, 0.01, 'float32', 0.119),
    (8, 3, 0.01, 'float16', 0.124),
    (32, 3, 0.001, 'float32', 0.46),
    (32, 3, 0.001, 'float16', 0.483),
    (32, 3, 0.002, 'float32', 0.46),
    (32, 3, 0.002, 'float16', 0.483),
    (32, 3, 0.01, 'float32', 0.46),
    (32, 3, 0.01, 'float16', 0.483),
    (1, 3, 4, 'int16', 0.014),
    (1, 3, 4, 'int32', 0.014),
    (4, 3, 4, 'int16', 0.053),
    (4, 3, 4, 'int32', 0.05),
    (8, 3, 4, 'int16', 0.101),
    (8, 3, 4, 'int32', 0.095),
    (16, 3, 4, 'int16', 0.198),
    (16, 3, 4, 'int32', 0.184),
    (32, 3, 4, 'int16', 0.392),
    (32, 3, 4, 'int32', 0.364),
    (1, 16, 8, 'int16', 0.047),
    (1, 16, 8, 'int32', 0.049),
    (1, 32, 8, 'int16', 0.091),
    (1, 32, 8, 'int32', 0.163),
    (1, 64, 8, 'int16', 0.313),
    (1, 64, 8, 'int32', 0.321),
    (4, 16, 8, 'int16', 0.174),
    (4, 16, 8, 'int32', 0.174),
    (4, 32, 8, 'int16', 0.343),
    (4, 32, 8, 'int32', 0.619),
    (4, 64, 8, 'int16', 1.221),
    (4, 64, 8, 'int32', 1.236),
    (8, 16, 8, 'int16', 0.34),
    (8, 16, 8, 'int32', 0.336),
    (8, 32, 8, 'int16', 0.672),
    (8, 32, 8, 'int32', 1.211),
    (8, 64, 8, 'int16', 2.398),
    (8, 64, 8, 'int32', 2.425),
    (32, 16, 8, 'int16', 1.33),
    (32, 16, 8, 'int32', 1.308),
    (32, 32, 8, 'int16', 2.644),
    (32, 32, 8, 'int32', 4.768),
    (32, 64, 8, 'int16', 9.457),
    (32, 64, 8, 'int32', 9.557),
    (1, 3, 1, 'int16', 0.014),
    (1, 3, 1, 'int32', 0.014),
    (1, 3, 16, 'int16', 0.014),
    (1, 3, 16, 'int32', 0.014),
    (8, 3, 1, 'int16', 0.101),
    (8, 3, 1, 'int32', 0.094),
    (8, 3, 16, 'int16', 0.101),
    (8, 3, 16, 'int32', 0.094),
    (32, 3, 1, 'int16', 0.391),
    (32, 3, 1, 'int32', 0.363),
    (32, 3, 16, 'int16', 0.391),
    (32, 3, 16, 'int32', 0.363),
)


def select_cases(args, parser):
    fields = ('n_m', 'd', 's', 'dtype', 'baseline_ms')
    cases = [dict(zip(fields, values)) for values in BASELINE_CASES]
    if not args.full:
        cases = [c for c in cases if c['n_m'] == 1 and c['s'] in (0.005, 0.01, 4, 8)]
    if args.dims:
        cases = [c for c in cases if c['d'] in args.dims]
    if args.n_m:
        cases = [c for c in cases if c['n_m'] in args.n_m]
    if not cases:
        parser.error('No matching benchmark cases')
    return cases


def make_inputs(case):
    dtype = getattr(torch, case['dtype'])
    shape = (case['n_m'] * 1024 * 1024, case['d'])
    pos = (torch.randn(shape, dtype=dtype, device='npu') if dtype.is_floating_point else
           torch.randint(0, 128, shape, dtype=dtype, device='npu'))
    size = torch.full((case['d'],), case['s'], dtype=dtype, device='npu')
    reduction = pos.int() if dtype == torch.int16 else pos
    start, end = reduction.min(0).values.to(dtype), reduction.max(0).values.to(dtype)
    return pos, size, start, end


def sampled_exact(call, pos, size, start, end):
    result = call()
    count = min(4096, pos.size(0))
    # float32 linspace rounds 32M-1 up to 32M (an invalid row).
    rows = (torch.arange(count, dtype=torch.long) * (pos.size(0) - 1) // max(1, count - 1)).npu()
    expected = grid_cluster_cpu(pos[rows].cpu(), size.cpu(), start.cpu(), end.cpu())
    return torch.equal(result[rows].cpu(), expected)


def measure(call, warmup, iterations):
    for _ in range(warmup):
        call()
    wall, device = [], []
    for _ in range(3):
        begin, finish = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        before = time.perf_counter()
        begin.record()
        for _ in range(iterations):
            call()
        finish.record()
        torch.npu.synchronize()
        wall.append((time.perf_counter() - before) * 1000 / iterations)
        device.append(begin.elapsed_time(finish) / iterations)
    return statistics.median(wall), statistics.median(device)


def run_case(case, args, writer, output):
    pos, size, start, end = make_inputs(case)

    def call():
        return ops_gnn.grid_cluster(pos, size, start, end)

    exact = sampled_exact(call, pos, size, start, end)
    latency, event_ms = measure(call, args.warmup, args.iter)
    speedup = case['baseline_ms'] / latency
    row = dict(case, wall_ms=latency, event_ms=event_ms, speedup=speedup,
               **{'pass': exact and speedup >= 0.45, 'exact': exact})
    writer.writerow(row)
    output.flush()
    LOGGER.info(json.dumps(row))


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    args, parser = parse_args()
    torch.npu.set_device(int(os.getenv('NPU_DEVICE_ID', '0')))
    cases = select_cases(args, parser)
    torch.manual_seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=[*cases[0], 'wall_ms', 'event_ms', 'speedup', 'pass', 'exact'])
        writer.writeheader()
        for case in cases:
            run_case(case, args, writer, output)
    LOGGER.info('Saved %s', args.output)
    with open(args.output, newline='') as result_file:
        results = list(csv.DictReader(result_file))
    failed = [r for r in results if r['pass'] != 'True']
    LOGGER.info('Summary: %s/%s passed; seed=%s', len(results) - len(failed), len(results), args.seed)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

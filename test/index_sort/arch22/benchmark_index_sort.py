# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark ``index_sort`` for the arch22 and arch35 builds.

The baseline is the composition a user writes without this operator: a stable
``argsort`` on int64 keys plus the gather that moves the keys. That composition
is what PyG does without ``pyg_lib``. On A2/A3 (``dav-2201``) the int64 sort runs
on the AiCpu kernel, so the comparison also shows what the float32 path is worth;
on 950 (``dav-3510``) it is a native fast kernel, so the same comparison shows the
overhead of the public wrapper instead. Every backend is checked against the CPU
golden before it is timed, and both platforms report their own numbers.
"""

import argparse
import logging
import time

import torch


def _arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark index_sort on Ascend NPU.")
    parser.add_argument("--keys", type=int, default=1 << 20)
    parser.add_argument("--values", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--cpu", action="store_true",
                        help="also time the CPU stable sort")
    return parser.parse_args()


def _timed(step, warmup, iters, device):
    """Milliseconds per call of ``step``, timed where ``device`` runs it."""
    for _ in range(warmup):
        step()
    if device == "cpu":
        begin = time.perf_counter()
        for _ in range(iters):
            step()
        return (time.perf_counter() - begin) * 1e3 / iters
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def _keys(count, values, device):
    generator = torch.Generator().manual_seed(42)
    return torch.randint(0, values, (count,), generator=generator).to(device)


def _compose(keys):
    """Sorted keys and permutation built from device primitives."""
    order = torch.argsort(keys, stable=True)
    return torch.index_select(keys, 0, order), order


def _ratio(reference, measured):
    return measured / reference if reference > 0.0 else float("nan")


def main():
    args = _arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch_npu  # noqa: F401  # Register the NPU device.
    import ops_gnn

    device = "npu"
    keys = _keys(args.keys, args.values, device)

    def operator():
        return ops_gnn.index_sort(keys, args.values)

    def composition():
        return _compose(keys)

    expected = _compose(keys.cpu())
    got = operator()
    for name, value, reference in (("sorted keys", got[0], expected[0]),
                                   ("perm", got[1], expected[1])):
        if not torch.equal(value.cpu(), reference):
            raise RuntimeError(f"{name} differs from the CPU composition")

    operator_ms = _timed(operator, args.warmup, args.iters, device)
    composition_ms = _timed(composition, args.warmup, args.iters, device)

    logging.info("index_sort keys=%d values=%d", args.keys, args.values)
    logging.info("operator: %.3f ms", operator_ms)
    logging.info("argsort composition (NPU): %.3f ms (%.2fx)",
                 composition_ms, _ratio(operator_ms, composition_ms))

    if args.cpu:
        host_keys = keys.cpu()
        cpu_ms = _timed(lambda: _compose(host_keys),
                        max(1, args.warmup // 5), max(1, args.iters // 5),
                        "cpu")
        logging.info("argsort composition (CPU): %.3f ms (%.2fx)",
                     cpu_ms, _ratio(operator_ms, cpu_ms))


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark ``csr_transpose`` for the arch22 build (A2/A3, dav-2201).

The baseline is the composition a user writes without this operator: histogram
the column ids for the pointer and move the rows with a stable ``argsort`` on the
device. A CPU composition and, when it is installed, ``torch_sparse.transpose``
are measured on the same graph to put the device numbers in context. Every
backend is checked against the CPU golden before it is timed.
"""

import argparse
import logging
import time

import torch


def _arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark csr_transpose on Ascend NPU.")
    parser.add_argument("--nodes", type=int, default=1 << 14)
    parser.add_argument("--cols", type=int, default=1 << 14)
    parser.add_argument("--degree", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--cpu", action="store_true",
                        help="also time the CPU composition")
    return parser.parse_args()


def _timed_npu(step, warmup, iters):
    """Milliseconds per call of ``step``, measured with NPU events."""
    for _ in range(warmup):
        step()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def _timed_cpu(step, warmup, iters):
    """Milliseconds per call of ``step`` on the host."""
    for _ in range(warmup):
        step()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    return (time.perf_counter() - start) * 1e3 / iters


def _graph(nodes, cols, degree, device):
    """Random CSR structure with about ``nodes * degree`` edges."""
    generator = torch.Generator().manual_seed(42)
    counts = torch.full((nodes,), degree, dtype=torch.int64)
    indptr = torch.cat([torch.zeros(1, dtype=torch.int64),
                        counts.cumsum(0)])
    edges = int(indptr[-1])
    indices = torch.randint(0, cols, (edges,), generator=generator)
    return indptr.to(device), indices.to(device)


def _rows(indptr):
    """Row id of every edge, i.e. the inverse of the CSR pointer."""
    return torch.repeat_interleave(
        torch.arange(indptr.numel() - 1, dtype=torch.int64,
                     device=indptr.device),
        indptr[1:] - indptr[:-1])


def _compose(indptr, indices, cols):
    """Pointer and layout of the transpose built from device primitives."""
    counts = torch.zeros(cols, dtype=torch.int64, device=indices.device)
    counts.scatter_add_(0, indices, torch.ones_like(indices))
    out_ptr = torch.cat([torch.zeros(1, dtype=torch.int64,
                                     device=indices.device),
                         counts.cumsum(0)])
    out_idx = _rows(indptr)[torch.argsort(indices, stable=True)]
    return out_ptr, out_idx


def _ratio(reference, measured):
    return measured / reference if reference > 0.0 else float("nan")


def main():
    args = _arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch_npu  # noqa: F401  # Register the NPU device.
    import ops_gnn

    device = "npu"
    indptr, indices = _graph(args.nodes, args.cols, args.degree, device)
    edges = indices.numel()

    def operator():
        return ops_gnn.csr_transpose(indptr, indices, args.cols)

    def composition():
        return _compose(indptr, indices, args.cols)

    expected = _compose(indptr.cpu(), indices.cpu(), args.cols)
    got = operator()
    for name, value, reference in (("out_indptr", got[0], expected[0]),
                                   ("out_indices", got[1], expected[1])):
        if not torch.equal(value.cpu(), reference):
            raise RuntimeError(f"{name} differs from the CPU composition")

    operator_ms = _timed_npu(operator, args.warmup, args.iters)
    composition_ms = _timed_npu(composition, args.warmup, args.iters)

    logging.info("csr_transpose rows=%d cols=%d edges=%d",
                 args.nodes, args.cols, edges)
    logging.info("operator: %.3f ms", operator_ms)
    logging.info("argsort composition (NPU): %.3f ms (%.2fx)",
                 composition_ms, _ratio(operator_ms, composition_ms))

    if args.cpu:
        host_ptr, host_idx = indptr.cpu(), indices.cpu()
        cpu_ms = _timed_cpu(lambda: _compose(host_ptr, host_idx, args.cols),
                            max(1, args.warmup // 5), max(1, args.iters // 5))
        logging.info("argsort composition (CPU): %.3f ms (%.2fx)",
                     cpu_ms, _ratio(operator_ms, cpu_ms))

    try:
        import torch_sparse  # noqa: F401
    except ImportError:
        logging.info("torch_sparse: not installed")
        return

    sparse_input = (indices.cpu(),
                    torch.ones(edges, dtype=torch.float32),
                    torch.Size((args.nodes, args.cols)))

    def third_party():
        return torch_sparse.transpose(sparse_input, coalesced=True)

    third_party()
    third_party_ms = _timed_cpu(third_party, max(1, args.warmup // 5),
                                max(1, args.iters // 5))
    logging.info("torch_sparse.transpose (CPU): %.3f ms (%.2fx)",
                 third_party_ms, _ratio(operator_ms, third_party_ms))


if __name__ == "__main__":
    main()

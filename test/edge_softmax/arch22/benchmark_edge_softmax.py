# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark ``edge_softmax`` for the arch22 build (A2/A3, dav-2201).

The baseline is the COO composition a user writes without this operator: expand
the CSR pointer into one row index per edge with ``ptr2ind`` and reduce the
``scatter_softmax`` groups. A static graph lets the user cache that index, so
both forms are measured: the per-call softmax alone, and the full composition
that rebuilds the index on every call. ``edge_softmax`` does neither - it reads
the segment of a row from the pointer - so the ratio against the two baselines
bounds its benefit. All three run on the same NPU, source and graph.
"""

import argparse
import logging
import os

import torch

_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark edge_softmax on Ascend NPU.")
    parser.add_argument("--nodes", type=int, default=1 << 14)
    parser.add_argument("--degree", type=int, default=64)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def _timed(step, warmup, iters):
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


def _ratio(operator, composed):
    return composed / operator if operator > 0.0 else float("nan")


def _expanded(ops_gnn, indptr, rows):
    """Row index of every edge, as ``ptr2ind`` reports it (int64 pointer)."""
    return ops_gnn.ptr2ind(indptr.to(torch.int64), rows)


def _measure(ops_gnn, graph, args):
    """Time the operator, the cached composition and the rebuilding one."""
    src, indptr, index, grad, nodes = graph
    # Gradients accumulate into the leaf otherwise, so every step goes through
    # `torch.autograd.grad`, which frees the graph after each call.
    op_src = src.clone().requires_grad_(True)
    coo_src = src.clone().requires_grad_(True)
    rebuild_src = src.clone().requires_grad_(True)

    def op_forward():
        return ops_gnn.edge_softmax(src, indptr)

    def coo_forward():
        return ops_gnn.scatter_softmax(src, index, dim_size=nodes)

    def rebuild_forward():
        return ops_gnn.scatter_softmax(
            src, _expanded(ops_gnn, indptr, src.size(0)), dim_size=nodes)

    def op_step():
        out = ops_gnn.edge_softmax(op_src, indptr)
        return torch.autograd.grad(out, op_src, grad)

    def coo_step():
        out = ops_gnn.scatter_softmax(coo_src, index, dim_size=nodes)
        return torch.autograd.grad(out, coo_src, grad)

    def rebuild_step():
        out = ops_gnn.scatter_softmax(
            rebuild_src, _expanded(ops_gnn, indptr, src.size(0)),
            dim_size=nodes)
        return torch.autograd.grad(out, rebuild_src, grad)

    forward = [op_forward, coo_forward, rebuild_forward]
    step = [op_step, coo_step, rebuild_step]
    return {
        "forward": [_timed(f, args.warmup, args.iters) for f in forward],
        "step": [_timed(s, args.warmup, args.iters) for s in step],
    }


def _report(args, edges, result):
    logging.info(
        "edge_softmax nodes=%d degree=%d edges=%d features=%d dtype=%s",
        args.nodes, args.degree, edges, args.features, args.dtype)
    for label, key in (("forward", "forward"),
                       ("forward+backward", "step")):
        operator, cached, rebuilt = result[key]
        logging.info(
            "%s: operator %.3f ms | cached composition %.3f ms (%.2fx) | "
            "rebuilding composition %.3f ms (%.2fx)",
            label, operator, cached, _ratio(operator, cached), rebuilt,
            _ratio(operator, rebuilt))


def main():
    args = _arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch_npu  # noqa: F401  # Register the NPU device.
    import ops_gnn

    device = f"npu:{int(os.getenv('NPU_DEVICE_ID', '0'))}"
    torch.npu.set_device(device)
    dtype = _DTYPES[args.dtype]

    edges = args.nodes * args.degree
    src = torch.randn(edges, args.features, dtype=dtype).to(device)
    indptr = (torch.arange(0, args.nodes + 1, dtype=torch.int32,
                           device=device) * args.degree).to(torch.int32)
    grad = torch.randn(edges, args.features, dtype=dtype).to(device)
    index = _expanded(ops_gnn, indptr, edges)

    result = _measure(ops_gnn, (src, indptr, index, grad, args.nodes), args)
    _report(args, edges, result)


if __name__ == "__main__":
    main()

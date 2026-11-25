# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark ``sort_edge_index`` for the arch22 and arch35 builds.

The baseline is the composition a user writes without this operator, and the one
PyG uses without ``pyg_lib``: build the composite key ``row * num_nodes + col``,
sort it stably and move the edge list with the permutation. On A2/A3
(``dav-2201``) that sort runs on the AiCpu kernel, which is what the two-pass
float32 order avoids; on 950 (``dav-3510``) the operator takes the same single
pass over int64 keys, so the comparison shows the overhead of the public wrapper
rather than a different algorithm. Both platforms report their own numbers.
"""

import argparse
import logging

import torch


def _arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark sort_edge_index on Ascend NPU.")
    parser.add_argument("--nodes", type=int, default=1 << 20)
    parser.add_argument("--edges", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
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


def _graph(nodes, edges, device):
    """Random COO edge list with about ``edges`` edges over ``nodes`` nodes."""
    generator = torch.Generator().manual_seed(42)
    rows = torch.randint(0, nodes, (edges,), generator=generator)
    cols = torch.randint(0, nodes, (edges,), generator=generator)
    return torch.stack([rows, cols]).to(device)


def _compose(edge_index, num_nodes):
    """Sorted edge list built from device primitives, as PyG does without pyg_lib."""
    key = edge_index[0] * num_nodes + edge_index[1]
    order = torch.argsort(key, stable=True)
    return torch.index_select(edge_index, 1, order)


def _ratio(reference, measured):
    return measured / reference if reference > 0.0 else float("nan")


def main():
    args = _arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch_npu  # noqa: F401  # Register the NPU device.
    import ops_gnn

    device = "npu"
    edge_index = _graph(args.nodes, args.edges, device)

    def operator():
        return ops_gnn.sort_edge_index(edge_index, num_nodes=args.nodes)

    def composition():
        return _compose(edge_index, args.nodes)

    expected = _compose(edge_index.cpu(), args.nodes)
    if not torch.equal(operator().cpu(), expected):
        raise RuntimeError("sorted edges differ from the CPU composition")

    operator_ms = _timed_npu(operator, args.warmup, args.iters)
    composition_ms = _timed_npu(composition, args.warmup, args.iters)

    logging.info("sort_edge_index nodes=%d edges=%d", args.nodes, args.edges)
    logging.info("operator: %.3f ms", operator_ms)
    logging.info("composite argsort composition (NPU): %.3f ms (%.2fx)",
                 composition_ms, _ratio(operator_ms, composition_ms))


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark ``scatter_softmax`` on Ascend NPU.

Examples:
    NPU_DEVICE_ID=0 python3 test/scatter_softmax/benchmark_scatter_softmax.py

    # Long-group shape: 131072 edges in one group, above the fp16 sum limit.
    NPU_DEVICE_ID=0 python3 test/scatter_softmax/benchmark_scatter_softmax.py \
        --edges 131072 --features 64 --groups 1 --dtype float16
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


def _build(edges, features, groups, dtype, device):
    src = torch.randn(edges, features, dtype=dtype)
    index = torch.randint(0, groups, (edges,), dtype=torch.int64)
    return src.to(device), index.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, default=1 << 20)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--groups", type=int, default=1 << 14)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch_npu  # noqa: F401
    import ops_gnn

    device = f"npu:{int(os.getenv('NPU_DEVICE_ID', '0'))}"
    torch.npu.set_device(device)
    src, index = _build(args.edges, args.features, args.groups,
                        _DTYPES[args.dtype], device)

    for _ in range(args.warmup):
        ops_gnn.scatter_softmax(src, index, dim_size=args.groups)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        ops_gnn.scatter_softmax(src, index, dim_size=args.groups)
    end.record()
    torch.npu.synchronize()

    elapsed = start.elapsed_time(end) / args.iters
    logging.info(
        "scatter_softmax edges=%d features=%d groups=%d dtype=%s: "
        "%.3f ms/iter",
        args.edges, args.features, args.groups, args.dtype, elapsed)


if __name__ == "__main__":
    main()

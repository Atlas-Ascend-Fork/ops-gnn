# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Measure eager matrix exponential latency."""

import argparse
import json
import logging
import sys
from statistics import median
from time import perf_counter


def measure(prepare, synchronize, warmup, iterations):
    samples = []
    for index in range(warmup + iterations):
        run = prepare()
        synchronize()
        start = perf_counter()
        result = run()
        synchronize()
        elapsed_ms = (perf_counter() - start) * 1000
        if index >= warmup:
            samples.append(elapsed_ms)
        del result, run
    return median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--norm", type=float, default=4.0, help="Target matrix 1-norm")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.n, args.batch, args.iters) < 1 or args.warmup < 0 or not 0 <= args.norm < float("inf"):
        parser.error("Require positive n, batch and iters, and finite nonnegative norm and warmup")

    import torch
    import torch_npu
    from ops_gnn import matrix_exp

    accelerator = torch.npu
    accelerator.set_device(0)
    device = "npu:0"
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (args.batch, args.n, args.n)
    a = torch.randn(shape, generator=rng, dtype=torch.float32)
    norm = a.abs().sum(dim=-2).amax(dim=-1)
    a *= args.norm / norm[:, None, None]
    x = a.to(device)
    grad = torch.randn(shape, generator=rng, dtype=torch.float32).to(device)
    x_grad = x.detach().requires_grad_()

    @torch.no_grad()
    def forward():
        return matrix_exp(x)

    def forward_backward():
        return torch.autograd.grad(matrix_exp(x_grad), x_grad, grad)

    def prepare_backward():
        y = matrix_exp(x_grad)
        return lambda: torch.autograd.grad(y, x_grad, grad)

    results = {}
    for label, prepare in (
        ("forward_ms", lambda: forward),
        ("backward_ms", prepare_backward),
        ("forward_backward_ms", lambda: forward_backward),
    ):
        results[label] = measure(prepare, accelerator.synchronize, args.warmup, args.iters)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.info("%s", json.dumps({
        "backend": "npu",
        **vars(args),
        "device": accelerator.get_device_name(0),
        "dtype": "float32",
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "backend_version": torch_npu.__version__,
        **results,
    }))


if __name__ == "__main__":
    main()

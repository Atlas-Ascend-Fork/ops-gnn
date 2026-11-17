# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shapes, dtypes and inputs shared by the grouped-softmax test suites.

``scatter_softmax`` and ``scatter_log_softmax`` accept the same arguments and
reject the same inputs, so the case tables and the sample builders live here
instead of in two copies.
"""

import torch

SEED = 42
DTYPES = (torch.float16, torch.float32, torch.bfloat16)

# (index, src shape, dim, dim_size)
CASES = [
    ([0, 0, 1, 2, 2, 2], (6,), 0, 3),             # 1-D features
    ([0, 0, 1, 2, 2, 2], (6, 4), 0, 3),           # regular groups
    ([0, 0, 2, 2, 2, 4], (6, 3), 0, 6),           # empty groups
    ([0, 1, 2, 3], (4, 2), 0, 4),                 # one element per group
    ([1, 1, 1, 1], (4, 3), 0, 2),                 # single non-zero group
    ([0, 1, 2, 3, 4], (5, 2), 0, 8),              # dim_size beyond max index
    ([0, 0, 1, 2, 2, 2], (2, 6), 1, 3),           # reduction along dim 1
    ([0, 0, 1, 2, 2, 2], (2, 6, 4), 1, 3),        # 3-D input along dim 1
    ([0, 0, 1, 2, 2, 2], (2, 6), -1, 3),          # negative dim
    ([0, 0, 0, 0, 0, 0], (6, 2), 0, 1),           # one single group
]

# (case, expected error)
INVALID_CASES = [
    ("bad_dtype", TypeError),
    ("bad_index_dtype", TypeError),
    ("bad_index_rank", ValueError),
    ("length_mismatch", ValueError),
    ("negative_index", ValueError),
    ("index_too_large", ValueError),
    ("bad_dim", IndexError),
    ("bad_dim_type", TypeError),
    ("negative_dim_size", ValueError),
    ("dim_size_too_small", ValueError),
    ("bad_dim_size_type", TypeError),
    ("cpu", RuntimeError),
    ("bad_src_type", TypeError),
    ("bad_index_type", TypeError),
]

# A group longer than the fp16 integer limit must not collapse.
LONG_GROUP_ROWS = (65504, 65505, 70000, 100000, 131072)

# Groups holding at least one +inf become all NaN.
INFINITY_VALUES = [
    [float("inf"), 0.0, 1.0],
    [float("inf"), float("inf"), 0.0],
    [float("inf"), -float("inf")],
]


def source(shape, dtype):
    """Deterministic sample tensor on CPU."""
    generator = torch.Generator().manual_seed(SEED)
    return torch.randn(shape, generator=generator).to(dtype)


def reference_gradient(src, index, dim, dim_size=3, log=False):
    """Differentiate the per-group reduction in float64 on CPU."""
    ref_src = src.double().requires_grad_(True)
    ref_out = torch.zeros_like(ref_src)
    for group in range(dim_size):
        positions = (index == group).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        subset = ref_src.index_select(dim, positions)
        if log:
            reduced = torch.log_softmax(subset, dim=dim)
        else:
            reduced = torch.softmax(subset, dim=dim)
        ref_out = ref_out.index_copy(dim, positions, reduced)
    return ref_src, ref_out


def invalid_inputs():
    """Argument tuples the public entry points must reject."""
    src = source((4, 2), torch.float32)
    index = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    return {
        "bad_dtype": (src.to(torch.int32), index, 0, 3),
        "bad_index_dtype": (src, index.to(torch.float32), 0, 3),
        "bad_index_rank": (src, index.view(4, 1), 0, 3),
        "length_mismatch": (
            src, torch.tensor([0, 0, 1], dtype=torch.int64), 0, 3),
        "negative_index": (
            src, torch.tensor([0, -1, 1, 2], dtype=torch.int64), 0, 3),
        "index_too_large": (
            src, torch.tensor([0, 0, 1, 3], dtype=torch.int64), 0, 3),
        "bad_dim": (src, index, 3, 3),
        "bad_dim_type": (src, index, 0.0, 3),
        "negative_dim_size": (src, index, 0, -1),
        "dim_size_too_small": (src, index, 0, 2),
        "bad_dim_size_type": (src, index, 0, 3.0),
        "cpu": (src, index, 0, 3),
        "bad_src_type": ([[0.0, 1.0]], index, 0, 3),
        "bad_index_type": (src, [0, 0, 1, 2], 0, 3),
    }


def to_npu(value):
    return value.npu() if isinstance(value, torch.Tensor) else value

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent PyTorch reference for ``scatter_softmax``.

The reference runs on CPU in float64, so it is not affected by the source
dtype rounding. It deliberately does not import ``ops_gnn`` or
``torch_scatter``, so functional tests can validate the operator on their own.
"""

import torch

TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-5},
    torch.float16: {"rtol": 2e-2, "atol": 2e-2},
    torch.bfloat16: {"rtol": 4e-2, "atol": 4e-2},
}


def reference(src: torch.Tensor, index, dim: int = 0, dim_size=None):
    """Compute a grouped softmax on CPU in float64."""
    if not isinstance(index, torch.Tensor):
        index = torch.tensor(index, dtype=torch.int64)
    idx = index.to(torch.int64).reshape(-1)
    values = src.to(torch.float64)
    if dim < 0:
        dim += values.dim()
    if dim_size is None:
        groups = int(idx.max()) + 1 if idx.numel() else 0
    else:
        groups = int(dim_size)

    result = torch.zeros_like(values)
    for group in range(groups):
        positions = (idx == group).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        subset = values.index_select(dim, positions)
        result.index_copy_(dim, positions, torch.softmax(subset, dim=dim))
    return result


def assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype) -> None:
    """Compare NPU output with the float64 reference under dtype tolerance."""
    tolerance = TOLERANCES[dtype]
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), equal_nan=True, **tolerance)

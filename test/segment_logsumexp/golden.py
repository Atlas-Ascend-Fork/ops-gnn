# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent PyTorch reference for ``segment_logsumexp``.

The reference accumulates in float64 so it is not affected by the source
dtype rounding. It deliberately does not import ``ops_gnn`` or
``torch_scatter``, so functional tests can be validated on their own.
"""

import torch

TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-5},
    torch.float16: {"rtol": 2e-2, "atol": 2e-2},
    torch.bfloat16: {"rtol": 4e-2, "atol": 4e-2},
}


def reference(src: torch.Tensor, indptr) -> torch.Tensor:
    """Compute log-sum-exp per CSR segment on CPU in float64."""
    if not isinstance(indptr, torch.Tensor):
        indptr = torch.tensor(indptr, dtype=torch.int64)
    if indptr.dtype != torch.int64:
        indptr = indptr.to(torch.int64)
    ptr = indptr.tolist()
    segments = []
    for start, end in zip(ptr[:-1], ptr[1:]):
        segment = src[start:end].to(torch.float64)
        if segment.numel() == 0:
            segments.append(
                torch.full(segment.shape[1:], float("-inf"),
                           dtype=torch.float64))
        else:
            segments.append(torch.logsumexp(segment, dim=0))
    if not segments:
        return torch.empty((0,) + tuple(src.shape[1:]), dtype=torch.float64)
    return torch.stack(segments)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype) -> None:
    """Compare NPU output with the float64 reference under dtype tolerance."""
    tolerance = TOLERANCES[dtype]
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), equal_nan=True, **tolerance)

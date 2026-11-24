# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Float64 golden of ``edge_softmax`` for the arch22 build (A2/A3, dav-2201).

The reference expands the CSR pointer into one segment label per edge and then
reduces every segment with ``torch.softmax`` in float64, so it shares no code
with the device kernel and does not import ``ops_gnn``.
"""

import torch

from common.grouped_softmax_golden import grouped_reference


def labels(indptr):
    """Segment label of every edge described by the CSR pointer ``indptr``."""
    ptr = indptr.to(torch.int64).cpu()
    return torch.repeat_interleave(
        torch.arange(ptr.numel() - 1, dtype=torch.int64), ptr[1:] - ptr[:-1])


def reference(src, indptr):
    """Normalize every CSR segment of ``src`` along axis 0 in float64."""
    return grouped_reference(src, labels(indptr), 0, indptr.numel() - 1)


def gradient(src, indptr):
    """Return the float64 forward of ``reference`` with an autograd graph."""
    ref_src = src.to(torch.float64).requires_grad_(True)
    return ref_src, reference(ref_src, indptr)

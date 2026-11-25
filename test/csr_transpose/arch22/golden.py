# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden of ``csr_transpose`` for the arch22 build (A2/A3, dav-2201).

The reference sorts the edge list by column with a stable ``argsort`` and rebuilds
the pointers from the column histogram, so it shares no code with the device
kernel and does not import ``ops_gnn``.
"""

import torch


def rows_of(indptr):
    """Row id of every edge described by the CSR pointer ``indptr``."""
    ptr = torch.as_tensor(indptr, dtype=torch.int64)
    return torch.repeat_interleave(
        torch.arange(ptr.numel() - 1, dtype=torch.int64), ptr[1:] - ptr[:-1])


def reference(indptr, indices, num_cols):
    """Return ``(out_indptr, out_indices)`` of the transposed CSR structure."""
    ptr = torch.as_tensor(indptr, dtype=torch.int64)
    idx = torch.as_tensor(indices, dtype=torch.int64)
    counts = torch.zeros(num_cols, dtype=torch.int64)
    if idx.numel() != 0:
        counts.scatter_add_(0, idx, torch.ones_like(idx))
    out_ptr = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    order = torch.argsort(idx, stable=True) if idx.numel() != 0 else \
        torch.zeros(0, dtype=torch.int64)
    return out_ptr, rows_of(ptr)[order]


def reference_twice(indptr, indices, num_cols):
    """Golden of the transposition applied twice, i.e. the row-major input."""
    out_ptr, out_idx = reference(indptr, indices, num_cols)
    rows = torch.as_tensor(indptr, dtype=torch.int64).numel() - 1
    return reference(out_ptr, out_idx, rows)

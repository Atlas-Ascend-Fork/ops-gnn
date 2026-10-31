# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent PyTorch reference, usable on CPU without ops_gnn."""
import torch


def spmm_sum_reference(indptr, indices, x, fp32=False):
    """Compute 2D SpMM or 3D BSpMM copy_lhs + sum."""
    m, feature_shape = indptr.numel() - 1, tuple(x.shape[1:])
    flat = x.reshape(x.size(0), x.shape[1:].numel())
    degrees = (indptr[1:] - indptr[:-1]).long()
    rows = torch.repeat_interleave(torch.arange(m, device=x.device), degrees)
    values = flat.float() if fp32 else flat
    result = torch.zeros((m, flat.size(1)), dtype=values.dtype, device=x.device)
    result.index_add_(0, rows, values[indices.long()])
    return result.reshape((m,) + feature_shape).to(x.dtype)

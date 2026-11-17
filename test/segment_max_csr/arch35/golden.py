# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU reference implementation for segment_max_csr."""

from math import prod

import torch


def segment_max_csr_cpu(src, indptr):
    """Compute the segment_max_csr reference result on CPU."""
    segment_dim = indptr.dim() - 1
    n_segments = indptr.shape[-1] - 1
    result_shape = list(src.shape)
    result_shape[segment_dim] = n_segments
    result = torch.zeros(result_shape, dtype=src.dtype, device="cpu")
    outer = prod(src.shape[:segment_dim])
    feature_dim = segment_dim + 1
    inner = prod(src.shape[feature_dim:])
    source_rows = src.cpu().reshape(outer, src.shape[segment_dim], inner)
    result_rows = result.reshape(outer, n_segments, inner)
    expanded_shape = list(indptr.shape)
    for axis in range(segment_dim):
        if expanded_shape[axis] == 1:
            expanded_shape[axis] = src.shape[axis]
    pointer_rows = indptr.cpu().expand(expanded_shape).reshape(outer, n_segments + 1)
    for batch_idx in range(outer):
        for seg in range(n_segments):
            start = int(pointer_rows[batch_idx, seg])
            end = int(pointer_rows[batch_idx, seg + 1])
            if start < end:
                result_rows[batch_idx, seg] = (
                    source_rows[batch_idx, start:end].max(dim=0).values
                )
    return result

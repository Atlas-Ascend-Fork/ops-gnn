# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Batched CSR SpMM aggregation on Ascend NPU."""
from typing import Optional

import torch

from .spmm import _parse_call_options, _SpmmRequest, _spmm


def bspmm(indptr: torch.Tensor, indices: torch.Tensor,
          x: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
    """Run generalized batched CSR aggregation.

    At least one node or edge feature tensor has rank three or higher. Trailing
    feature dimensions follow item-axis broadcasting rules, with missing dimensions
    inserted immediately after the item axis. The supported message
    operations and reductions are the same as :func:`ops_gnn.spmm`.
    """
    op, reduce, out, rhs = _parse_call_options("bspmm", args, kwargs)
    request = _SpmmRequest(indptr, indices, x, op, reduce, out, rhs, True)
    return _spmm(request)

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""torch_cluster-compatible grid clustering on Ascend NPU."""
from typing import Optional

import torch

from . import _pybind


def grid_cluster(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return linear int64 voxel IDs; dimension zero has stride one.

    ``pos`` is [N, D]; size/start/end are [D] on the same NPU.
    Size must be positive and finite. Omitted bounds are computed on the NPU.
    Arithmetic follows torch_cluster CPU (division then truncation to long).
    """
    return _pybind.grid_cluster(pos, size, start, end)

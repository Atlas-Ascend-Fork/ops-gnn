# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden of ``index_sort`` for the arch22 and arch35 builds.

The reference is the host ``torch.argsort`` with ``stable=True``, so it shares no
code with the operator and does not import ``ops_gnn``. It is architecture
independent, which is why the arch35 module re-exports this one.
"""

import torch


def reference(keys):
    """Return ``(sorted_keys, perm)`` of a stable ascending sort."""
    values = torch.as_tensor(keys, dtype=torch.int64)
    if values.numel() == 0:
        return values.clone(), torch.zeros(0, dtype=torch.int64)
    order = torch.argsort(values, stable=True)
    return values[order], order


def reference_perm(keys):
    """Permutation of a stable ascending sort, for callers that only need it."""
    return reference(keys)[1]

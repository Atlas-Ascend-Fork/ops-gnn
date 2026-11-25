# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden of ``sort_edge_index`` for the arch22 and arch35 builds.

The reference builds the composite key ``primary * num_nodes + secondary`` and
sorts it stably on the host, which is the single-pass order the operator targets.
It is architecture independent, which is why the arch35 module re-exports this
one.
"""

import torch


def reference(edge_index, num_nodes, sort_by_row=True):
    """Return ``(sorted edge_index, perm)`` for the given edge list."""
    edges = torch.as_tensor(edge_index, dtype=torch.int64)
    primary = edges[0] if sort_by_row else edges[1]
    secondary = edges[1] if sort_by_row else edges[0]
    if edges.numel() == 0:
        return edges.clone(), torch.zeros(0, dtype=torch.int64)
    order = torch.argsort(primary * num_nodes + secondary, stable=True)
    return edges[:, order], order

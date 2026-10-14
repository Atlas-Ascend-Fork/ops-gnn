# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Operator-specific CPU oracle from pytorch_cluster/csrc/cpu/grid_cpu.cpp.

Preserve dtype at every ATen operation. Empty input is the task's extension
to upstream behavior. Never relabel IDs or compare only cluster membership.
"""
import torch


def grid_cluster_cpu(pos, size, start=None, end=None):
    if pos.device.type != "cpu":
        raise ValueError("grid_cluster_cpu expects CPU tensors")
    if not pos.size(0):
        return torch.empty(0, dtype=torch.long)
    start = pos.min(0).values if start is None else start
    end = pos.max(0).values if end is None else end
    count = torch.true_divide(end - start, size).long() + 1
    prefix = torch.cat((torch.ones(1, dtype=torch.long), count.cumprod(0)))[:-1]
    return (torch.true_divide(pos - start, size).long() * prefix).sum(1)

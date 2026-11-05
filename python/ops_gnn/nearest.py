# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exact batched point-set 1-NN on Ascend 950PR."""

from typing import Optional

import torch

from . import _pybind  # Load the PrivateUse1 implementation.


# Coexist with an installed torch_cluster, which may already own the schema.
_nearest_library = torch.library.Library("torch_cluster", "FRAGMENT")
if not hasattr(torch.ops.torch_cluster, "nearest"):
    _nearest_library.define(
        "nearest(Tensor x, Tensor y, Tensor? ptr_x=None, Tensor? ptr_y=None) -> Tensor"
    )


def _batch_ptr(batch: Optional[torch.Tensor], count: int, device: torch.device,
               name: str):
    if batch is None:
        return ([0] if count else []), [0, count]
    if batch.dtype != torch.int64 or batch.ndim != 1 or batch.numel() != count:
        raise ValueError(f"{name} must be int64 [{count}]")
    if batch.device != device:
        raise ValueError(f"{name} must be on the same NPU as x and y")
    # Only batch metadata crosses to the host. Point coordinates never do.
    values = batch.detach().cpu().tolist()
    if any(a > b for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} is not sorted")
    if values and values[0] < 0:
        raise ValueError(f"{name} must contain non-negative batch IDs")
    labels, ptr = [], [0]
    for i, value in enumerate(values):
        if not labels or value != labels[-1]:
            if labels:
                ptr.append(i)
            labels.append(value)
    ptr.append(count)
    return labels, ptr


def nearest(x: torch.Tensor, y: torch.Tensor,
            batch_x: Optional[torch.Tensor] = None,
            batch_y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Assign each x point its nearest y point, returning global int64 indices.

    Inputs are NPU float16/float32 [N, F] and [M, F], or 1-D [N]/[M].
    Batch IDs must be sorted int64 and have identical nonempty sets. Gaps in
    IDs are allowed. Both dtypes are evaluated in FP32 against the pinned
    scipy/OpenBLAS reference. Equal distances choose the smallest global
    y index, including across search tiles and batch boundaries.
    """
    if x.device.type != "npu" or x.device != y.device:
        raise ValueError("x and y must be on the same NPU")
    if x.dtype not in (torch.float16, torch.float32) or x.dtype != y.dtype:
        raise ValueError("x and y must have matching float16 or float32 dtype")
    if x.ndim not in (1, 2) or y.ndim not in (1, 2):
        raise ValueError("x and y must be 1-D or 2-D")
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    if x.size(1) != y.size(1) or x.size(1) == 0:
        raise ValueError("x and y must have the same positive feature dimension")
    if x.size(0) and not y.size(0):
        raise ValueError("y must be nonempty when x is nonempty")
    if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
        raise ValueError("x and y must contain only finite coordinates")
    ptr_x = ptr_y = None
    if batch_x is not None or batch_y is not None:
        labels_x, px = _batch_ptr(batch_x, x.size(0), x.device, "batch_x")
        labels_y, py = _batch_ptr(batch_y, y.size(0), y.device, "batch_y")
        if labels_x != labels_y:
            raise ValueError("batch_x and batch_y have different nonempty batch sets")
        # 单个非空批次覆盖全部点，可复用无分区的 SIMD 路径。
        # 保留上面的完整校验；多个批次仍按压缩后的编号构造指针。
        if len(labels_x) > 1:
            ptr_x = torch.tensor(px, dtype=torch.int64, device=x.device)
            ptr_y = torch.tensor(py, dtype=torch.int64, device=x.device)
    return torch.ops.torch_cluster.nearest(x, y, ptr_x, ptr_y)

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""torch_scatter-compatible ``scatter_logsumexp`` on Ascend NPU.

The public call surface matches the grouped reduction spelled out with
``torch_scatter.composite.logsumexp``::

    scatter_logsumexp(src, index, dim=0, dim_size=None)

For every group defined by ``index`` the result is the numerically stable
log-sum-exp along ``dim``::

    out[g] = ln(sum_j exp(src[j]))

where ``j`` runs over the elements of group ``g``. Unlike the grouped softmax
family the output is reduced: it has one row per ``(batch, group)`` pair, so
``src.size(dim)`` is replaced by ``dim_size``.

The reduction reuses the grouped-softmax kernel core of ``scatter_softmax``
(``csrc/npu/scatter_softmax/op_kernel/common``) with its log-sum-exp mode
enabled: the maximum and the exponential sum are accumulated exactly as in the
softmax, a finalize pass writes ``max + ln(sum)`` once per group, and the
backward is a single elementwise pass::

    grad_src = grad_out[group] * exp(src - out[group])

``exp(src - out)`` is the softmax probability of the element inside its group, so
the saved reduced output is the only forward result the backward needs - besides
the source, which the analytic gradient reads directly.

Edge cases follow ``torch.logsumexp``: an empty group, and a group made only of
``-inf``, reduce to ``-inf``; a group that holds ``+inf`` reduces to ``+inf``; a
group that holds ``NaN`` reduces to ``NaN``. The backward of an all-``-inf``
group and the gradient of a ``+inf`` member are ``NaN``, and the gradient of the
finite siblings of a ``+inf`` member is zero.
"""

from typing import Optional

import torch

from . import _pybind
from ._grouped_softmax import check_index_range, check_inputs, normalize_dim, resolve_dim_size

_OP_NAME = "scatter_logsumexp"

_forward = _pybind.scatter_logsumexp_forward
_backward = _pybind.scatter_logsumexp_backward


class _ScatterLogSumExp(torch.autograd.Function):
    """Grouped log-sum-exp with the analytic grouped gradient.

    The source is saved as well as the reduced output: the gradient weighs the
    group gradient by ``exp(src - out)``, which is read from the source and the
    saved output instead of being stored by the forward pass.
    """

    @staticmethod
    def forward(ctx, src, index, dim, num_groups):
        # The device kernel reads the group labels as int32 scalars, so narrow
        # the public int32/int64 index once here; the range check in ``run``
        # already guarantees that every value fits.
        idx = index.to(torch.int32).contiguous()
        ctx.dim = dim
        ctx.num_groups = num_groups
        source = src.contiguous()
        out = _forward(source, idx, dim, num_groups)
        ctx.save_for_backward(source, out, idx)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        source, out, idx = ctx.saved_tensors
        grad_src = _backward(
            source, out, grad_out.contiguous(), idx, ctx.dim, ctx.num_groups)
        return grad_src, None, None, None


def scatter_logsumexp(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute a grouped log-sum-exp along ``dim`` on Ascend NPU.

    Args:
        src: Source tensor on NPU. Supported dtypes are ``float16``,
            ``float32`` and ``bfloat16``.
        index: 1-D ``int32``/``int64`` group index on the same NPU. Its
            length must equal ``src.size(dim)``.
        dim: Dimension along which the reduction is computed. Negative
            dimensions are accepted.
        dim_size: Number of groups. Defaults to ``index.max() + 1`` and may
            be larger than that to leave empty groups.

    Returns:
        Tensor with the shape of ``src`` but ``dim_size`` in place of
        ``src.size(dim)``, holding the log-sum-exp of every group. A group
        without members reduces to ``-inf``.

    Raises:
        TypeError: ``src``/``index`` has an unsupported type or dtype, or
            ``dim``/``dim_size`` has the wrong type.
        ValueError: Shape, index range or ``dim_size`` is invalid.
        RuntimeError: Inputs are not on an Ascend NPU.
        IndexError: ``dim`` is out of range.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"src must be a torch.Tensor, got {type(src).__name__}")
    if not isinstance(index, torch.Tensor):
        raise TypeError(
            f"index must be a torch.Tensor, got {type(index).__name__}")

    dim = normalize_dim(dim, src.dim())
    check_inputs(_OP_NAME, src, index, dim)

    num_groups = resolve_dim_size(index, dim_size)
    check_index_range(index, num_groups)
    return _ScatterLogSumExp.apply(src, index, dim, num_groups)

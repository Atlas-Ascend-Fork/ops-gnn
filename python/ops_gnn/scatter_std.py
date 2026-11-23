# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""torch_scatter-compatible ``scatter_std`` on Ascend NPU.

The public call surface matches the grouped reduction spelled out with
``torch_scatter.composite.std``::

    scatter_std(src, index, dim=0, dim_size=None, unbiased=True)

For every group defined by ``index`` the result is the standard deviation along
``dim``::

    out[g] = sqrt(sum_j (src[j] - mean[g]) ** 2 / (n - 1))

where ``j`` runs over the elements of group ``g`` and ``n`` is the number of
those elements. ``unbiased=False`` divides by ``n`` instead. Like the grouped
log-sum-exp - and unlike the grouped softmax family - the output is reduced: it
has one row per ``(batch, group)`` pair, so ``src.size(dim)`` is replaced by
``dim_size``.

The reduction reuses the grouped-softmax kernel core of ``scatter_softmax``
(``csrc/npu/scatter_softmax/op_kernel/common``) with its standard-deviation mode
enabled: a first pass accumulates the group sums and the member counts, a second
pass accumulates the squared deviations from the group mean, and a finalize pass
writes ``sqrt(deviation / (n - correction))`` once per group while keeping the
mean for the backward::

    grad_src = grad_out[group] * (src - mean[group]) / ((n - correction) * std)

The deviation is accumulated around the actual mean rather than expanded into
``sum(x^2) - n * mean^2``, so a group whose values share a large offset keeps its
significant digits.

Edge cases follow ``torch.std``: an empty group is ``NaN``, the single member of
an unbiased deviation is ``NaN`` while the biased one is ``0``, a group that
holds ``NaN`` or ``+/-inf`` is ``NaN``, and the backward of a group without
spread is ``0``: ``torch.std`` zeroes the variance gradient before it
propagates.
"""

from typing import Optional

import torch

from . import _pybind
from ._grouped_softmax import check_index_range, check_inputs, normalize_dim, resolve_dim_size

_OP_NAME = "scatter_std"

_forward = _pybind.scatter_std_forward
_backward = _pybind.scatter_std_backward


class _ScatterStd(torch.autograd.Function):
    """Grouped standard deviation with the analytic grouped gradient.

    The forward keeps the group means and the member counts next to the reduced
    output, so the backward only has to rescale the centered source with its
    group gradient: no reduction runs again.
    """

    @staticmethod
    def forward(ctx, src, index, dim, num_groups, unbiased):
        # The device kernel reads the group labels as int32 scalars, so narrow
        # the public int32/int64 index once here; the range check in ``run``
        # already guarantees that every value fits.
        idx = index.to(torch.int32).contiguous()
        ctx.dim = dim
        ctx.num_groups = num_groups
        ctx.unbiased = unbiased
        source = src.contiguous()
        out, means, counts = _forward(
            source, idx, dim, num_groups, unbiased)
        ctx.save_for_backward(source, out, means, counts, idx)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        source, out, means, counts, idx = ctx.saved_tensors
        grad_src = _backward(
            source, out, grad_out.contiguous(), means, counts, idx,
            ctx.dim, ctx.num_groups, ctx.unbiased)
        return grad_src, None, None, None, None


def scatter_std(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    unbiased: bool = True,
) -> torch.Tensor:
    """Compute a grouped standard deviation along ``dim`` on Ascend NPU.

    Args:
        src: Source tensor on NPU. Supported dtypes are ``float16``,
            ``float32`` and ``bfloat16``.
        index: 1-D ``int32``/``int64`` group index on the same NPU. Its
            length must equal ``src.size(dim)``.
        dim: Dimension along which the reduction is computed. Negative
            dimensions are accepted.
        dim_size: Number of groups. Defaults to ``index.max() + 1`` and may
            be larger than that to leave empty groups.
        unbiased: Divide by ``n - 1`` (sample standard deviation) when true
            and by ``n`` (population standard deviation) when false.

    Returns:
        Tensor with the shape of ``src`` but ``dim_size`` in place of
        ``src.size(dim)``, holding the standard deviation of every group. A
        group without members, and the single member of an unbiased
        deviation, are ``NaN``.

    Raises:
        TypeError: ``src``/``index`` has an unsupported type or dtype, or
            ``dim``/``dim_size``/``unbiased`` has the wrong type.
        ValueError: Shape, index range or ``dim_size`` is invalid.
        RuntimeError: Inputs are not on an Ascend NPU.
        IndexError: ``dim`` is out of range.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"src must be a torch.Tensor, got {type(src).__name__}")
    if not isinstance(index, torch.Tensor):
        raise TypeError(
            f"index must be a torch.Tensor, got {type(index).__name__}")
    if not isinstance(unbiased, bool):
        raise TypeError(
            f"unbiased must be a bool, got {type(unbiased).__name__}")

    dim = normalize_dim(dim, src.dim())
    check_inputs(_OP_NAME, src, index, dim)

    num_groups = resolve_dim_size(index, dim_size)
    check_index_range(index, num_groups)
    return _ScatterStd.apply(src, index, dim, num_groups, unbiased)

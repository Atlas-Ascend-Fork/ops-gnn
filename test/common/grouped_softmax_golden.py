# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent PyTorch reference of the grouped-softmax operators.

The reference runs on CPU in float64, so it is not affected by the source
dtype rounding. It deliberately does not import ``ops_gnn`` or
``torch_scatter``, so functional tests can validate the operators on their own.

``scatter_softmax`` and ``scatter_log_softmax`` differ in the per-group
reduction only, so the caller passes it in (``torch.softmax`` or
``torch.log_softmax``) and each operator's ``golden.py`` binds the matching one.
``scatter_logsumexp`` additionally reduces the output, so it keeps one row per
``(batch, group)`` pair and has its own reference here.
"""

import torch

TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-5},
    torch.float16: {"rtol": 2e-2, "atol": 2e-2},
    torch.bfloat16: {"rtol": 4e-2, "atol": 4e-2},
}


def normalized_groups(src, index, dim, dim_size):
    """Normalize the arguments every reference of this family shares.

    Returns the flat int64 index, the float64 source, the non-negative ``dim``
    and the number of groups, so the references differ only in the reduction.
    """
    if not isinstance(index, torch.Tensor):
        index = torch.tensor(index, dtype=torch.int64)
    idx = index.to(torch.int64).reshape(-1)
    values = src.to(torch.float64)
    if dim < 0:
        dim += values.dim()
    if dim_size is None:
        groups = int(idx.max()) + 1 if idx.numel() else 0
    else:
        groups = int(dim_size)
    return idx, values, dim, groups


def grouped_reference(src: torch.Tensor, index, dim: int = 0, dim_size=None,
                      reduce=None):
    """Apply ``reduce`` to every group of ``src`` in float64 on CPU."""
    reduce = torch.softmax if reduce is None else reduce
    idx, values, dim, groups = normalized_groups(src, index, dim, dim_size)

    result = torch.zeros_like(values)
    for group in range(groups):
        positions = (idx == group).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        subset = values.index_select(dim, positions)
        result.index_copy_(dim, positions, reduce(subset, dim=dim))
    return result


def grouped_logsumexp_reference(src: torch.Tensor, index, dim: int = 0,
                                dim_size=None) -> torch.Tensor:
    """Reduce every group of ``src`` with ``torch.logsumexp`` in float64.

    The output is reduced, not ``src``-shaped: it has one row per
    ``(batch, group)`` pair, so ``src.size(dim)`` is replaced by the number of
    groups. A group without members reduces to ``-inf``, which is also what
    ``torch.logsumexp`` reports for an empty reduction.
    """
    idx, values, dim, groups = normalized_groups(src, index, dim, dim_size)

    shape = list(values.shape)
    shape[dim] = groups
    result = torch.full(shape, float("-inf"), dtype=torch.float64)
    for group in range(groups):
        positions = (idx == group).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        subset = values.index_select(dim, positions)
        result.select(dim, group).copy_(torch.logsumexp(subset, dim=dim))
    return result


def logsumexp_gradient(src: torch.Tensor, index, dim: int = 0, dim_size=None):
    """Return the float64 forward of ``grouped_logsumexp_reference`` with a graph.

    The forward reference builds the output, so the empty-group ``-inf`` rule
    and the group assembly live in one place; ``requires_grad`` only adds the
    autograd graph the gradient tests need. The backward of an empty group is
    zero, which is what ``torch.logsumexp`` reports for an empty reduction.
    """
    ref_src = src.to(torch.float64).requires_grad_(True)
    return ref_src, grouped_logsumexp_reference(ref_src, index, dim, dim_size)


def make_reference(reduce):
    """Return the float64 grouped reference that applies ``reduce``.

    ``reduce`` is ``torch.softmax`` or ``torch.log_softmax``; each operator's
    ``golden.py`` binds its own, so the three-line body of that file carries no
    logic of its own.
    """

    def reference(src: torch.Tensor, index, dim: int = 0, dim_size=None):
        return grouped_reference(src, index, dim, dim_size, reduce)

    return reference


def assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype) -> None:
    """Compare NPU output with the float64 reference under dtype tolerance."""
    tolerance = TOLERANCES[dtype]
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), equal_nan=True, **tolerance)

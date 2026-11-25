# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public index_sort API."""

import operator
from typing import Optional, Tuple

import torch

from . import _pybind
from .typing import Tensor

_OP_NAME = 'index_sort'

# The build whose int64 sort only has the slow SortAiCpu implementation. Its
# float32 sort runs the SortAiCore kernel instead, and every integer below
# ``2 ** 24`` is exact in float32, so that build sorts such keys as float32 and
# casts the values back without loss.
_NARROW_ARCH = 'dav-2201'
_FLOAT32_EXACT_MAX = 1 << 24
_INT64_MAX = (1 << 63) - 1


def _uses_float32() -> bool:
    """Whether this build profits from sorting exact-integer keys as float32."""
    return _pybind.npu_arch == _NARROW_ARCH


def _check_inputs(inputs: torch.Tensor) -> None:
    """Reject an argument combination the operator cannot serve."""
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(f'{_OP_NAME}: inputs must be a torch.Tensor, '
                        f'got {type(inputs).__name__}')
    if inputs.dim() != 1:
        raise ValueError(f'{_OP_NAME}: inputs must be 1-D, got {inputs.dim()}-D')
    if inputs.dtype != torch.int64:
        raise TypeError(f'{_OP_NAME}: inputs dtype must be int64, '
                        f'got {inputs.dtype}')
    if inputs.device.type != 'npu':
        raise RuntimeError(f'{_OP_NAME} requires Ascend NPU tensors')


def _resolve_bound(inputs: torch.Tensor,
                   max_value: Optional[int]) -> Optional[int]:
    """Upper bound of the keys used to pick the sort path.

    ``max_value`` is checked for internal consistency on every build, but it only
    decides the sort path where the build sorts through float32: keys below
    ``_FLOAT32_EXACT_MAX`` are exact there. A caller supplied bound is trusted
    when it already proves that range, so the common call does not read the
    device. The bound is never checked against the keys themselves, so the
    operator behaves the same way on every build.
    """
    bound = None
    if max_value is not None:
        bound = operator.index(max_value)
        if bound < 0:
            raise ValueError(
                f'{_OP_NAME}: max_value must be non-negative, got {bound}')
        if bound == 0 and inputs.numel() != 0:
            raise RuntimeError(
                f'{_OP_NAME}: max_value is 0 but inputs holds '
                f'{inputs.numel()} keys')
    if not _uses_float32():
        return None
    if bound is not None and bound <= _FLOAT32_EXACT_MAX:
        return bound
    if inputs.numel() == 0:
        return 0
    return int(inputs.max().item()) + 1


def _stable_order(keys: torch.Tensor, bound: Optional[int]) -> torch.Tensor:
    """Stable ascending permutation of ``keys`` within ``[0, bound)``."""
    if bound is not None and bound <= _FLOAT32_EXACT_MAX:
        return torch.argsort(keys.to(torch.float32), stable=True)
    return torch.argsort(keys, stable=True)


def _stable_sort(keys: torch.Tensor,
                 bound: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stable ascending sort of ``keys``; returns the values and the permutation."""
    if bound is not None and bound <= _FLOAT32_EXACT_MAX:
        values, order = torch.sort(keys.to(torch.float32), stable=True)
        return values.to(torch.int64), order
    return torch.sort(keys, stable=True)


def index_sort(
    inputs: Tensor,
    max_value: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Stably sort a vector of non-negative integer keys.

    Counterpart of ``torch_geometric.utils.index_sort``: the keys are sorted in
    ascending order and the permutation is returned next to the sorted keys, so
    that any payload can be moved with ``perm``. The sort is stable, which the
    upstream helper does not promise: equal keys keep their input order, so the
    result is determined by the input alone and can be compared element by
    element, independent of the number of AI cores.

    The operator is a composition of device primitives, so it launches
    asynchronously on the current NPU stream. The one exception is an omitted
    ``max_value``, or one that does not prove the keys fit in 24 bits: both read
    one scalar from the device and therefore synchronize. On Atlas A2 series
    products and Atlas A3 series products the keys are sorted as float32 when
    every key is below ``2 ** 24`` (16777216), which is exact for integers and
    much faster than the int64 sort there; larger keys fall back to int64.

    ``max_value`` is a precondition, not a checked property: when it already
    proves the 24-bit range the operator does not read the keys, so a bound
    smaller than the real maximum is undefined behaviour, exactly as in the
    upstream helper. The keys themselves must be non-negative.

    Args:
        inputs (Tensor): 1-D ``torch.long`` tensor of non-negative keys on NPU.
        max_value (int, optional): Upper bound of the keys, i.e. every key must be
            inside ``[0, max_value)``. When it is omitted the operator uses
            ``inputs.max() + 1``, which reads one value from the device. A value
            above ``2 ** 24`` is accepted as well; the operator then reads the
            real maximum to stay on the float32 path when the keys allow it.

    Returns:
        Tuple[Tensor, Tensor]: ``(sorted_inputs, perm)``, the sorted keys and the
        ``E`` positions that produced them, both 1-D ``torch.long`` on the same
        NPU as the input.

    Raises:
        TypeError: An argument is not a tensor, has the wrong dtype, or
            ``max_value`` is not an integer.
        ValueError: ``inputs`` is not 1-D, or ``max_value`` is negative.
        RuntimeError: The inputs are not on an Ascend NPU, or ``max_value`` is
            zero while the vector holds keys.

    Example:
        >>> import torch
        >>> import ops_gnn
        >>> keys = torch.tensor([2, 0, 2, 1], dtype=torch.long, device='npu')
        >>> sorted_keys, perm = ops_gnn.index_sort(keys, 3)
        >>> sorted_keys
        tensor([0, 1, 2, 2], device='npu:0')
        >>> perm
        tensor([1, 3, 0, 2], device='npu:0')
    """
    _check_inputs(inputs)
    bound = _resolve_bound(inputs, max_value)
    if inputs.numel() == 0:
        return inputs.clone(), torch.empty(0, dtype=torch.int64,
                                          device=inputs.device)
    return _stable_sort(inputs, bound)

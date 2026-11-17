# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""torch_scatter-compatible ``scatter_log_softmax`` on Ascend NPU.

The public call surface matches ``torch_scatter.scatter_log_softmax``::

    scatter_log_softmax(src, index, dim=0, dim_size=None)

For every group defined by ``index`` the result is the numerically stable
log-softmax along ``dim``::

    out[i] = (src[i] - max_g) - log(sum_j exp(src[j] - max_g))

where ``max_g`` is the maximum of the group that ``i`` belongs to and ``j``
runs over the elements of that group. The output keeps the shape, dtype and
device of ``src``.

The operator reuses the grouped-softmax kernel core of ``scatter_softmax``
(``csrc/npu/scatter_softmax/op_kernel/common``) with its compile-time log mode
enabled: the maximum and the exponential sum are accumulated exactly as in the
softmax, the normalization writes ``(src - max) - log(sum)`` instead of a
quotient, and the backward reduces the plain group sum of ``grad_out`` and
weights it by ``exp(out)``, because the log-domain Jacobian is
``d out_j / d src_i = delta_ij - p_i``. Taking the logarithm of the sum keeps a
group whose exponentials underflow from collapsing, and it lets the exponential
pass skip its store. The Python layer only validates the arguments and wires the
analytic Jacobian::

    grad_src = grad_out - exp(out) * sum_j grad_out[j]
"""

from typing import Optional

import torch

from ._grouped_softmax import build_autograd, run

_OP_NAME = "scatter_log_softmax"

_ScatterLogSoftmax = build_autograd(_OP_NAME)


def scatter_log_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute a grouped log-softmax along ``dim`` on Ascend NPU.

    Args:
        src: Source tensor on NPU. Supported dtypes are ``float16``,
            ``float32`` and ``bfloat16``.
        index: 1-D ``int32``/``int64`` group index on the same NPU. Its
            length must equal ``src.size(dim)``.
        dim: Dimension along which the log-softmax is computed. Negative
            dimensions are accepted.
        dim_size: Number of groups. Defaults to ``index.max() + 1`` and may
            be larger than that to leave empty groups.

    Returns:
        Tensor with the same shape, dtype and device as ``src``, holding the
        log probability of every element. Elements of empty groups do not
        exist, so no placeholder value is produced.

    Raises:
        TypeError: ``src``/``index`` has an unsupported type or dtype, or
            ``dim``/``dim_size`` has the wrong type.
        ValueError: Shape, index range or ``dim_size`` is invalid.
        RuntimeError: Inputs are not on an Ascend NPU.
        IndexError: ``dim`` is out of range.
    """
    return run(_OP_NAME, _ScatterLogSoftmax, src, index, dim, dim_size)

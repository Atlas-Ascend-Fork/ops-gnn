# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Preserve public forward mutation checks without implementing backward."""

import torch

from .._pybind import scatter_check_out


class _ScatterForward(torch.autograd.Function):
    """Let PyTorch apply the CPU extension's dirty-output checks after writing."""

    @staticmethod
    def forward(ctx, _src, _index, out, operation, returns_arg):
        value, arg = operation()
        if out is not None:
            ctx.mark_dirty(out)
        if returns_arg:
            ctx.mark_non_differentiable(arg)
            return value, arg
        return value

    @staticmethod
    def backward(_ctx, *_grad_outputs):
        """Reject reverse-mode differentiation explicitly; forward is supported."""
        raise NotImplementedError("Scatter A2 supports forward execution only; backward is not implemented")


def run_forward(src, index, out, reduce, operation):
    """Run one reduction and preserve its observable forward error timing.

    ``operation`` takes no arguments and returns ``(value, arg_or_none)``.
    It must use the supplied tensors and return the original ``out`` when
    present. It runs with autograd recording disabled, as the CPU extension's
    custom forward does. Mean must preserve the sum-stage error boundary. L1
    can bridge its native sum before division; an L2 mean callback must execute
    only sum when an inference out is used outside inference mode. Otherwise
    wrapping a completed mean would report that error after the wrong update.

    No backward implementation is provided. Native launch code remains
    responsible for current-stream execution and asynchronous storage lifetime.
    """
    if reduce not in ("sum", "add", "mul", "mean", "min", "max"):
        raise ValueError("forward mutation checks require an individual reduction")
    if out is None and not src.requires_grad:
        value, arg = operation()
        return value, arg if reduce in ("min", "max") else None
    if reduce in ("sum", "add", "mean") and out is not None:
        if torch.is_grad_enabled() and out.requires_grad:
            # All-inference inputs bypass PyTorch's Autograd dispatch. Preserve
            # its write-then-error boundary; other inputs need a read-only check.
            if any(not torch.is_inference(tensor) for tensor in (src, index, out)):
                scatter_check_out(out)
    returns_arg = reduce in ("min", "max")
    result = _ScatterForward.apply(src, index, out, operation, returns_arg)
    if out is not None and reduce in ("sum", "add", "mean"):
        # The public sum primitive keeps the supplied object even in no_grad.
        # Custom mul/min/max instead retain Function.apply's returned wrapper.
        return out, None
    return result if returns_arg else (result, None)

# Copyright (c) 2026 Eternal_ZCL.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU references and comparison rules for torch_sparse SpMM."""

import torch


def registered_call(case, tensors):
    """Same torch_sparse public API boundary on NPU and GPU."""
    import torch_sparse

    row, ptr, col, value, mat = tensors
    if case["format"] == "coo":
        if case["reduce"] not in ("sum", "add"):
            raise ValueError("public COO supports sum only")
        index = torch.stack((row, col))
        return lambda: torch_sparse.spmm(index, value, int(case["m"]), int(case["n"]), mat)
    reduce = case["reduce"]
    if reduce in ("sum", "add"):
        return lambda: torch.ops.torch_sparse.spmm_sum(row, ptr, col, value, None, None, mat)
    if reduce == "mean":
        counts = ptr[1:] - ptr[:-1]
        return lambda: torch.ops.torch_sparse.spmm_mean(row, ptr, col, value, counts, None, None, mat)
    return lambda: getattr(torch.ops.torch_sparse, "spmm_" + reduce)(ptr, col, value, mat)


def compare(actual, expected, dtype):
    if actual.shape != expected.shape:
        return {"passed": False, "reason": "shape mismatch"}
    if actual.dtype != dtype:
        return {"passed": False, "reason": f"dtype mismatch: {actual.dtype} != {dtype}"}
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    if dtype == torch.int32:
        return {"passed": torch.equal(actual, expected.to(dtype)), "rule": "EXACT"}
    rules = {
        torch.float16: (2**-9, 2**-9, 0.1),
        torch.bfloat16: (2**-6, 2**-6, 1.0),
        torch.float32: (2**-10, 2**-16, 0.01),
    }
    rule = rules.get(dtype)
    if rule is None:
        return {"passed": False, "reason": f"unsupported comparison dtype: {dtype}"}
    rt, at, limit = rule
    a, g = actual.double(), expected.double()
    special_ok = (torch.isnan(a) == torch.isnan(g)) & (torch.isinf(a) == torch.isinf(g))
    special_ok &= ~torch.isinf(g) | (torch.signbit(a) == torch.signbit(g))
    if not bool(special_ok.all()):
        return {"passed": False, "reason": "NaN/Inf position or sign mismatch"}
    finite = torch.isfinite(a) & torch.isfinite(g)
    if not bool(finite.any()):
        return {"passed": True, "matched_ratio": 1.0, "max_abs_error": 0.0, "finite_elements": 0}
    a, g = a[finite], g[finite]
    error = (a - g).abs()
    matched = float((error <= at + rt * g.abs()).double().mean())
    rounded = g.to(dtype)
    ulp = (torch.nextafter(rounded, torch.full_like(rounded, float("inf"))) - rounded).double().abs()
    maximum = torch.maximum(torch.full_like(ulp, limit), 32 * ulp)
    return {
        "passed": matched >= 0.99 and bool((error <= maximum).all()),
        "matched_ratio": matched,
        "max_abs_error": float(error.max()),
        "finite_elements": int(finite.sum()),
        "rule": "CPU_FP64 mixed tolerance + max bound + special values",
    }


COMBOS = [
    (torch.float32, torch.float32),
    (torch.float16, torch.float16),
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float32),
    (torch.int8, torch.int32),
    (torch.int8, torch.float32),
]


def golden(ptr, col, v, mat, reduce, *, outdtype):
    m, k = ptr.numel() - 1, mat.size(-1)
    out = torch.zeros((*mat.shape[:-2], m, k), dtype=torch.float64)
    arg = torch.full(out.shape, col.numel(), dtype=torch.int64)
    for r in range(m):
        a, z = int(ptr[r]), int(ptr[r + 1])
        if a == z:
            continue
        product = mat.double()[..., col[a:z], :]
        if v is not None:
            weight = v.double()[a:z]
            if weight.dim() == 1:
                weight = weight.unsqueeze(-1)
            product = product * weight
        if reduce in ("sum", "add"):
            out[..., r, :] = product.sum(-2)
        elif reduce == "mean":
            out[..., r, :] = product.mean(-2)
        else:
            values, indices = getattr(product, reduce)(-2)
            out[..., r, :] = values
            arg[..., r, :] = a + indices
    if outdtype == torch.int32:
        out = out.trunc()
    return out.to(outdtype), arg


def sample(dtype, batch=False, feature_values=False):
    torch.manual_seed(41)
    ptr = torch.tensor([0, 3, 3, 4, 7], dtype=torch.long)
    col = torch.tensor([2, 0, 2, 1, 3, 0, 1], dtype=torch.long)
    shape = (2, 3, 4, 17) if batch else (4, 17)
    vs = (7, 17) if feature_values else (7,)
    if dtype == torch.int8:
        mat = torch.randint(-4, 5, shape, dtype=dtype)
        v = torch.randint(-4, 5, vs, dtype=dtype)
    else:
        mat = torch.randn(shape).to(dtype)
        v = torch.randn(vs).to(dtype)
    return ptr, col, v, mat

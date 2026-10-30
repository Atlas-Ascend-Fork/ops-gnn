# Copyright (c) 2026 Eternal_ZCL.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Ascend 950 torch_sparse adapters. Import before importing frontend aliases."""

__all__ = ["csr_spmm", "coo_spmm", "coo_to_csr", "clear_validation_cache", "validation_cache_info"]

from collections import OrderedDict
import importlib
from numbers import Integral
from typing import NamedTuple, Optional, Tuple, Union
import weakref

import torch
import torch_npu  # Explicit device/backend initialization.
import torch_sparse
from . import _pybind

if _pybind.npu_arch != "dav-3510":
    raise ImportError("torch_sparse SpMM requires an ops_gnn build for Ascend 950")
_C = _pybind.torch_sparse_spmm

_REDUCE = {"sum": 0, "add": 0, "mean": 1, "min": 2, "max": 3}
_DTYPE = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2, torch.int8: 3, torch.int32: 4}
_VALIDATED = OrderedDict()
_METADATA_VALIDATED = OrderedDict()
_COO_LAYOUTS = OrderedDict()
_INPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.int8)


def _check_mat(mat):
    if mat.device.type != "npu" or mat.dim() < 2 or mat.dtype not in _INPUT_DTYPES:
        raise ValueError("mat must be an NPU fp32/fp16/bf16/int8 tensor [...,N,K]")


def _check_value(value, edges, mat):
    if value is None:
        return
    if value.dtype not in _INPUT_DTYPES or value.dtype != mat.dtype:
        raise ValueError("value dtype must be supported and match mat dtype")
    if value.device != mat.device or value.dim() < 1 or value.size(0) != edges:
        raise ValueError("value must be on the same NPU and have leading length E")
    width = 1
    for size in value.shape[1:]:
        width *= size
    if width not in (1, mat.size(-1)):
        raise ValueError("value must contain 1 or K weights per edge")


def _version(t):
    return _C.tensor_version(t)


def _validate(ptr, col, n):
    """Check contents on device; only the small validation status reaches host.

    Cache up to 128 immutable tensor identities + versions, not raw addresses.
    Contents modified through PyTorch invalidate the entry. Bypass writes through
    external pointers are unsupported. Cold validation is included in API cost.
    """
    key = (id(ptr), id(col), _version(ptr), _version(col), n)
    cached = _VALIDATED.get(key)
    if cached is not None and cached[0]() is ptr and cached[1]() is col:
        _VALIDATED.move_to_end(key)
        return
    status = _C.validate(ptr, col, n)
    if int(status.max().item()):
        raise ValueError("invalid CSR rowptr or column index out of range")
    if key[2] is not None and key[3] is not None:
        _VALIDATED[key] = (weakref.ref(ptr), weakref.ref(col))
        if len(_VALIDATED) > 128:
            _VALIDATED.popitem(last=False)


def clear_validation_cache() -> None:
    _VALIDATED.clear()
    _METADATA_VALIDATED.clear()
    _COO_LAYOUTS.clear()


class _CacheInfo(NamedTuple):
    csr_entries: int
    metadata_entries: int
    coo_entries: int


def validation_cache_info() -> _CacheInfo:
    """Return an immutable snapshot of cache entry counts; expose no tensors."""
    return _CacheInfo(len(_VALIDATED), len(_METADATA_VALIDATED), len(_COO_LAYOUTS))


def _meta_tensor(t, length, name, device):
    if t is None:
        raise ValueError(f"{name} must be int64 [{length}] on {device}")
    if t.dtype != torch.int64 or t.device != device or t.shape != (length,):
        raise ValueError(f"{name} must be int64 [{length}] on {device}")


def _grad_meta(ptr, col, value, mat, reduce, *, row, rowcount, colptr, perm):
    """Validate supplied metadata even when no backward is requested.

    Omitted, unused metadata is allowed. Required gradient metadata must exist.
    Content checks are cached only for unchanged tensor identities/versions.
    """
    gradients = torch.is_grad_enabled() and reduce < 2
    need_v = gradients and value is not None and value.requires_grad
    need_m = gradients and mat.requires_grad
    e, m, n = col.numel(), ptr.numel() - 1, mat.size(-2)
    if row is not None or need_v or need_m:
        _meta_tensor(row, e, "row", col.device)
    if colptr is not None or need_m:
        _meta_tensor(colptr, n + 1, "colptr", col.device)
    if perm is not None or need_m:
        _meta_tensor(perm, e, "csr2csc", col.device)
    if rowcount is not None or (need_m and reduce == 1):
        _meta_tensor(rowcount, m, "rowcount", col.device)
    slots = (ptr, col, row, rowcount, colptr, perm)
    supplied = [t for t in slots if t is not None]
    if len(supplied) == 2:
        return
    key = (n, tuple(None if t is None else (id(t), _version(t)) for t in slots))
    cached = _METADATA_VALIDATED.get(key)
    if cached is not None and all(ref() is t for ref, t in zip(cached, supplied)):
        _METADATA_VALIDATED.move_to_end(key)
        return
    if row is not None and not torch.equal(row, _C.convert(ptr, e, False)):
        raise ValueError("row disagrees with rowptr")
    if rowcount is not None and not torch.equal(rowcount, ptr[1:] - ptr[:-1]):
        raise ValueError("rowcount disagrees with rowptr")
    sorted_col = None
    if perm is not None:
        if not torch.equal(perm.sort().values, torch.arange(e, device=col.device)):
            raise ValueError("csr2csc must be a permutation of [0,E)")
        sorted_col = col[perm]
        if e > 1 and bool((sorted_col[1:] < sorted_col[:-1]).any().item()):
            raise ValueError("csr2csc must order edges by column")
    if colptr is not None:
        if sorted_col is None:
            sorted_col = col.sort().values
        if not torch.equal(colptr, _C.convert(sorted_col, n + 1, True)):
            raise ValueError("colptr disagrees with col and csr2csc")
    if all(item is None or item[1] is not None for item in key[1]):
        _METADATA_VALIDATED[key] = tuple(weakref.ref(t) for t in supplied)
        if len(_METADATA_VALIDATED) > 128:
            _METADATA_VALIDATED.popitem(last=False)


class _SpmmMetadata(NamedTuple):
    ptr: torch.Tensor
    col: torch.Tensor
    row: Optional[torch.Tensor]
    colptr: Optional[torch.Tensor]
    perm: Optional[torch.Tensor]
    reduce: int
    output_type: int


class _Spmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, mat, metadata):
        ptr, col, row, colptr, perm, reduce, output_type = metadata
        out, arg = _C.forward(ptr, col, value, mat, reduce, output_type)
        ctx.save_for_backward(
            ptr,
            col,
            value if value is not None else col,
            mat,
            row if row is not None else col,
            colptr if colptr is not None else col,
            perm if perm is not None else col,
            arg,
        )
        ctx.has_value, ctx.reduce = value is not None, reduce
        ctx.mark_non_differentiable(arg)
        return out, arg

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad, unused_arg):
        ptr, col, saved_value, mat, row, colptr, perm, arg = ctx.saved_tensors
        value = saved_value if ctx.has_value else None
        dv = dm = None
        if ctx.reduce >= 2:
            # Min/max backward is not required by the task; reject explicitly.
            raise NotImplementedError("min/max backward is outside this extension's supported contract")
        if ctx.needs_input_grad[0]:
            width = 1
            for size in value.shape[1:]:
                width *= size
            dv = _C.value_backward(ptr, col, mat, grad, ctx.reduce, width).reshape_as(value).to(value.dtype)
        if ctx.needs_input_grad[1]:
            g = grad.float()
            if ctx.reduce == 1:
                degree = (ptr[1:] - ptr[:-1]).clamp_min(1).float()
                g = g / degree.unsqueeze(-1)
            tv = None if value is None else value[perm].float()
            dm = _C.forward(colptr, row[perm], tv, g, 0, 0)[0].to(mat.dtype)
        return dv, dm, None


def _dispatch(ptr, col, value, mat, reduce, *, row=None, rowcount=None, colptr=None, perm=None, output_dtype=None):
    _check_mat(mat)
    _check_value(value, col.numel(), mat)
    if ptr.device != mat.device or col.device != mat.device:
        raise ValueError("indices and mat must be on the same NPU")
    _validate(ptr, col, mat.size(-2))
    if reduce < 2:
        _grad_meta(ptr, col, value, mat, reduce, row=row, rowcount=rowcount, colptr=colptr, perm=perm)
    dtype = output_dtype or (torch.int32 if mat.dtype == torch.int8 else mat.dtype)
    if dtype not in _DTYPE:
        raise ValueError("unsupported output dtype")
    if not torch.is_grad_enabled() or not (mat.requires_grad or (value is not None and value.requires_grad)):
        return _C.forward(ptr, col, value, mat, reduce, _DTYPE[dtype])
    metadata = _SpmmMetadata(ptr, col, row, colptr, perm, reduce, _DTYPE[dtype])
    return _Spmm.apply(value, mat, metadata)


def csr_spmm(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    value: Optional[torch.Tensor],
    mat: torch.Tensor,
    reduce: str = "sum",
    *,
    output_dtype: Optional[torch.dtype] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Convenience CSR API; output_dtype is an extension-only precision choice.

    Missing backward metadata is constructed on the NPU. Public torch_sparse
    operator schemas retain their original signatures and validate supplied metadata.
    Min/max return (out,arg); sum/mean return out. Empty rows are zero with arg=E.
    """
    _check_mat(mat)
    _check_value(value, col.numel(), mat)
    if rowptr.device != mat.device or col.device != mat.device:
        raise ValueError("indices and mat must be on the same NPU")
    if not isinstance(reduce, str) or reduce not in _REDUCE:
        raise ValueError("reduce must be sum/add/mean/min/max")
    r = _REDUCE[reduce]
    row = rowcount = cp = perm = None
    _validate(rowptr, col, mat.size(-2))
    if r < 2 and torch.is_grad_enabled():
        if mat.requires_grad or (value is not None and value.requires_grad):
            row = _C.convert(rowptr, col.numel(), False)
            if mat.requires_grad:
                perm = torch.argsort(col, stable=True)
                cp = _C.convert(col[perm], mat.size(-2) + 1, True)
                rowcount = rowptr[1:] - rowptr[:-1]
    result = _dispatch(
        rowptr, col, value, mat, r, row=row, rowcount=rowcount, colptr=cp, perm=perm, output_dtype=output_dtype
    )
    return result if r >= 2 else result[0]


_original_coo = torch_sparse.spmm


def _build_coo_layout(index, m):
    if bool(((index[0] < 0) | (index[0] >= m)).any().item()):
        raise ValueError("COO row out of range")
    perm = torch.argsort(index[0], stable=True)
    row, col = index[0][perm], index[1][perm]
    return _C.convert(row, m + 1, True), col, perm


def coo_to_csr(
    index: torch.Tensor, value: Optional[torch.Tensor], m: int
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert COO rows to CSR without caching, for conversion profiling.

    Inputs follow coo_spmm: int64 [2,E] indices, optional edge values and a
    nonnegative row count. Columns are stably ordered by row. This helper does
    not validate dense-input compatibility or populate the SpMM caches.
    """
    ptr, col, perm = _build_coo_layout(index, m)
    return ptr, col, None if value is None else value[perm]


def _cached_coo_layout(index, m):
    """Reuse only topology. Edge values and dense inputs are read on every call."""
    version = _version(index)
    if version is None:
        return _build_coo_layout(index, m)
    key = (id(index), version, m)
    cached = _COO_LAYOUTS.get(key)
    if cached is not None and cached[0]() is index:
        _COO_LAYOUTS.move_to_end(key)
        return cached[1:]
    layout = _build_coo_layout(index, m)

    def release(ref):
        entry = _COO_LAYOUTS.get(key)
        if entry is not None and entry[0] is ref:
            _COO_LAYOUTS.pop(key, None)

    _COO_LAYOUTS[key] = (weakref.ref(index, release), *layout)
    if len(_COO_LAYOUTS) > 32:
        _COO_LAYOUTS.popitem(last=False)
    return layout


def coo_spmm(index: torch.Tensor, value: Optional[torch.Tensor], m: int, n: int, matrix: torch.Tensor) -> torch.Tensor:
    if matrix.device.type != "npu":
        return _original_coo(index, value, m, n, matrix)
    _check_mat(matrix)
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in (m, n)):
        raise ValueError("COO m and n must be nonnegative integers")
    if index.dtype != torch.int64 or index.dim() != 2 or index.size(0) != 2:
        raise ValueError("COO index must be int64 [2,E] and dimensions nonnegative")
    if index.device != matrix.device or matrix.dim() < 2 or matrix.size(-2) != n:
        raise ValueError("COO device or dense shape mismatch")
    # Validate the original value before value[perm] can discard extra entries.
    _check_value(value, index.size(1), matrix)
    ptr, col, perm = _cached_coo_layout(index, m)
    if value is None:
        weights = None
    elif torch.is_grad_enabled() and value.requires_grad:
        weights = value[perm]
    else:
        weights = _C.permute_values(value, perm)
    return csr_spmm(ptr, col, weights, matrix)


_mm_module = importlib.import_module("torch_sparse.matmul")
_old_matmul = _mm_module.matmul
_old_spmm = _mm_module.spmm


def _sparse_rowptr(src, other):
    storage = src.storage
    ptr = storage.rowptr() if storage.has_rowptr() else None
    if ptr is None:
        row = storage.row() if storage.has_row() else None
        if row is None or row.dtype != torch.int64 or row.device != other.device:
            raise ValueError("SparseTensor row must be int64 on the same NPU")
        if bool(((row < 0) | (row >= src.sparse_size(0))).any().item()):
            raise ValueError("SparseTensor row out of range")
        if row.numel() > 1 and bool((row[1:] < row[:-1]).any().item()):
            raise ValueError("SparseTensor row must be sorted")
        # The registered ind2ptr implementation constructs and caches this pointer.
        ptr = storage.rowptr()
    return ptr


def _sparse_spmm(src, other, reduce="sum"):
    if other.device.type != "npu":
        return _old_spmm(src, other, reduce)
    _check_mat(other)
    if not isinstance(reduce, str) or reduce not in _REDUCE:
        raise ValueError("reduce must be sum/add/mean/min/max")
    if other.size(-2) != src.sparse_size(1):
        raise ValueError("sparse columns must match dense N")
    storage = src.storage
    ptr = _sparse_rowptr(src, other)
    value, col = storage.value(), storage.col()
    _check_value(value, col.numel(), other)
    _validate(ptr, col, other.size(-2))
    # Check existing caches; csr_spmm constructs any missing backward metadata.
    with torch.no_grad():
        _grad_meta(
            ptr,
            col,
            value,
            other,
            _REDUCE[reduce],
            row=storage.row() if storage.has_row() else None,
            rowcount=storage.rowcount() if storage.has_rowcount() else None,
            colptr=storage.colptr() if storage.has_colptr() else None,
            perm=storage.csr2csc() if storage.has_csr2csc() else None,
        )
    out = csr_spmm(ptr, col, value, other, reduce)
    return out[0] if reduce in ("min", "max") else out


def _matmul(src, other, reduce="sum"):
    if isinstance(other, torch.Tensor):
        return _sparse_spmm(src, other, reduce)
    return _old_matmul(src, other, reduce)


def _tensor_matmul(self, other):
    return _matmul(self, other, "sum")


# Keep Library objects alive; registration is specific to NPU dispatch keys.
_LIBRARIES = []
for _key in ("PrivateUse1", "AutogradPrivateUse1"):
    _lib = torch.library.Library("torch_sparse", "IMPL", _key)
    _lib.impl(
        "spmm_sum",
        lambda row, ptr, col, value, cp, perm, mat: _dispatch(ptr, col, value, mat, 0, row=row, colptr=cp, perm=perm)[
            0
        ],
    )
    _lib.impl(
        "spmm_mean",
        lambda row, ptr, col, value, counts, cp, perm, mat: _dispatch(
            ptr, col, value, mat, 1, row=row, rowcount=counts, colptr=cp, perm=perm
        )[0],
    )
    _lib.impl("spmm_min", lambda ptr, col, value, mat: _dispatch(ptr, col, value, mat, 2))
    _lib.impl("spmm_max", lambda ptr, col, value, mat: _dispatch(ptr, col, value, mat, 3))
    # torch_sparse_utils may already have registered NPU conversion kernels.
    for _name, _impl in (
        ("ind2ptr", lambda ind, m: _C.convert(ind, m + 1, True)),
        ("ptr2ind", lambda ptr, e: _C.convert(ptr, e, False)),
    ):
        if not _pybind.has_private_use1_kernel("torch_sparse::" + _name):
            _lib.impl(_name, _impl)
    _LIBRARIES.append(_lib)

torch_sparse.spmm = coo_spmm
importlib.import_module("torch_sparse.spmm").spmm = coo_spmm
torch_sparse.matmul = _matmul
_mm_module.matmul = _matmul
_mm_module.spmm = _sparse_spmm
torch_sparse.SparseTensor.spmm = _sparse_spmm
torch_sparse.SparseTensor.matmul = _matmul
torch_sparse.SparseTensor.__matmul__ = _tensor_matmul

# PyG 2.8's optimization checks `not src.is_cuda()`, which also matches NPU
# tensors and sends them into an unsupported native sparse-CSR mm path.
# Adapt the utility and the GCN/GIN aliases captured during PyG module import.
# CPU and non-SparseTensor inputs retain the original PyG implementation.
if importlib.util.find_spec("torch_geometric") is not None:
    _pyg_utils = importlib.import_module("torch_geometric.utils")
    _pyg_module = importlib.import_module("torch_geometric.utils._spmm")
    _old_pyg_spmm = _pyg_module.spmm

    def _pyg_spmm(src, other, reduce="sum"):
        if isinstance(src, torch_sparse.SparseTensor) and other.device.type == "npu":
            return _sparse_spmm(src, other, reduce)
        return _old_pyg_spmm(src, other, reduce)

    _pyg_utils.spmm = _pyg_spmm
    _pyg_module.spmm = _pyg_spmm
    for _module_name in ("gcn_conv", "gin_conv"):
        importlib.import_module("torch_geometric.nn.conv." + _module_name).spmm = _pyg_spmm

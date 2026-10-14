# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""torch_sparse utility operators on Ascend NPU.

``import ops_gnn`` is the single activation step: it loads the shared
``ops_gnn._pybind`` extension which registers NPU (PrivateUse1) implementations for

* ``torch.ops.torch_sparse.non_diag_mask``
* ``torch.ops.torch_sparse.random_walk``

The import also installs NPU-only SparseTensor diagonal adapters and index
conversions. The adapters maintain sorted COO and consistent CSR storage while
preserving the task-book mask returned by the public ``non_diag_mask`` op.
CPU and CUDA frontends continue to use their original implementations.
"""

__version__ = "0.1.0"

__all__ = [
    "non_diag_mask",
    "torch_sparse_random_walk",
    "torch_sparse_random_walk_with_rand",
    "is_available",
]

from functools import wraps
from numbers import Number

import torch

# Optional upstream schemas activate the registered PrivateUse1 kernels.
# Without torch_sparse, the direct ops_gnn interfaces remain usable.
try:
    import torch_sparse
except ImportError:
    torch_sparse = None

from ops_gnn import _pybind as _C  # registers the TORCH_LIBRARY blocks

if _C.npu_arch != "dav-3510":
    raise ImportError("torch_sparse utility operators require an Ascend 950 (dav-3510) build")

# importlib.reload() reuses this module dictionary. Keep the Library handle and
# installed wrapper identities here so registration and wrapping stay
# idempotent without attaching private attributes to the upstream module.
if "_frontend_state" not in globals():
    _frontend_state = {}


# The native binding already exposes this signature, including the public M/N
# keyword names. Export it directly instead of adding a forwarding layer.
non_diag_mask = _C.non_diag_mask


def torch_sparse_random_walk(rowptr, col, start, walk_length):
    """NPU uniform random walk with the kernel-internal Philox RNG."""
    return _C.torch_sparse_random_walk(rowptr, col, start, walk_length)


def torch_sparse_random_walk_with_rand(rowptr, col, start, walk_length, rand):
    """NPU random walk consuming a caller supplied [S, L] float32 rand tensor.

    A shared rand tensor verifies task-book walk semantics independently of the
    backend RNG. Isolated nodes write -1; float32 rounding is clamped so a
    selected neighbor cannot cross into the next CSR row. Public CPU/NPU RNG
    streams are not aligned by this helper.
    """
    return _C.torch_sparse_random_walk_with_rand(rowptr, col, start, walk_length, rand)


def is_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _ind2ptr_npu(index: torch.Tensor, size: int) -> torch.Tensor:
    """Build CSR pointers on the index device; no index tensor visits CPU."""
    counts = index.new_zeros(size)
    counts.scatter_add_(0, index, torch.ones_like(index))
    return torch.cat((index.new_zeros(1), counts.cumsum(0)))


def _ptr2ind_npu(pointer: torch.Tensor, nnz: int) -> torch.Tensor:
    """Expand CSR row ids on NPU, with the output size already known."""
    rows = torch.arange(pointer.numel() - 1, dtype=pointer.dtype,
                        device=pointer.device)
    return torch.repeat_interleave(
        rows, pointer[1:] - pointer[:-1], output_size=nnz)


def _validate_diag_values(value, values):
    """Weighted replacement values follow Tensor assignment and same-device rules."""
    if value is None or values is None:
        return
    if isinstance(values, torch.Tensor):
        if values.device != value.device:
            raise RuntimeError("values and stored value must be on the same device")
        # Indexed assignment treats a one-element Tensor as a scalar. Retain
        # that conversion while rejecting silent casts of other Tensor inputs.
        if values.numel() != 1 and values.dtype != value.dtype:
            raise RuntimeError("values and stored value must have the same dtype")
    elif not isinstance(values, Number):
        raise TypeError("values must be a Tensor, numeric scalar, or None")


def _diagonal_entries(entries, values=None, k=0):
    """Replace/sort real COO tensors; independent of the optional SparseStorage class.

    entries is (row, col, value, (rows, columns)). Unweighted inputs ignore values.
    """
    row, col, value, (rows, cols) = entries
    _validate_diag_values(value, values)
    num_diag = max(0, min(rows, cols - k) if k >= 0 else min(rows + k, cols))
    # The public mask deliberately appends the new diagonal slots. Reordering
    # belongs here in the SparseTensor frontend, not in the task-book operator.
    mask = non_diag_mask(row, col, rows, cols, k)
    keep = mask[:row.numel()]
    if num_diag == 0:
        # -INT64_MIN is outside int64 even when arange would be empty.
        diagonal = row.new_empty(0)
        diagonal_col = col.new_empty(0)
    else:
        start = max(0, -k)
        diagonal = torch.arange(start, start + num_diag, dtype=row.dtype,
                                device=row.device)
        diagonal_col = diagonal + k
    new_row = torch.cat((row[keep], diagonal))
    new_col = torch.cat((col[keep], diagonal_col))
    new_value = None
    if value is not None:
        diagonal_value = value.new_empty((num_diag,) + value.shape[1:])
        # Match upstream set_diag's assignment and broadcasting rules. In an
        # unweighted SparseTensor, supplied values are ignored just as upstream.
        diagonal_value[...] = 1 if values is None else values
        new_value = torch.cat((value[keep], diagonal_value))

    # Lexicographic sorting must not multiply int64 indices by a dimension.
    # Stable passes also preserve the value order of existing duplicate edges.
    order = torch.argsort(new_col, stable=True)
    order = order[torch.argsort(new_row[order], stable=True)]
    new_row, new_col = new_row[order], new_col[order]
    if new_value is not None:
        new_value = new_value[order]
    return new_row, new_col, new_value


def _set_diag_npu(src, values=None, k: int = 0):
    """Wrap the tensor implementation and rebuild optional upstream storage caches."""
    entries = (*src.coo(), src.sparse_sizes())
    new_row, new_col, new_value = _diagonal_entries(entries, values, k)
    from torch_sparse.storage import SparseStorage

    rows, cols = src.sparse_sizes()
    rowptr = _ind2ptr_npu(new_row, rows)
    rowcount = rowptr[1:] - rowptr[:-1]
    colcount = new_col.new_zeros(cols)
    colcount.scatter_add_(0, new_col, torch.ones_like(new_col))
    storage = SparseStorage(
        row=new_row, rowptr=rowptr, col=new_col, value=new_value,
        sparse_sizes=(rows, cols), rowcount=rowcount, colptr=None,
        colcount=colcount, csr2csc=None, csc2csr=None,
        is_sorted=True, trust_data=True,
    )
    return src.from_storage(storage)


def _install_torch_sparse_frontends() -> None:
    if torch_sparse is None:
        return

    # Keep the Library alive in this compatibility module across reloads.
    # Existing explicit NPU implementations are not replaced.
    if "conversion_library" not in _frontend_state:
        library = torch.library.Library("torch_sparse", "IMPL", "PrivateUse1")
        for name, implementation in (("ind2ptr", _ind2ptr_npu),
                                     ("ptr2ind", _ptr2ind_npu)):
            if not _C.has_private_use1_kernel("torch_sparse::" + name):
                library.impl(name, implementation)
        _frontend_state["conversion_library"] = library

    import torch_sparse.diag as diag

    installed_set_diag = _frontend_state.get("set_diag")
    installed_fill_diag = _frontend_state.get("fill_diag")
    if (installed_set_diag is not None
            and diag.set_diag is installed_set_diag
            and diag.fill_diag is installed_fill_diag):
        # Reload with the same torch_sparse modules: re-expose the already
        # installed wrappers and do not stack another layer.
        torch_sparse.set_diag = diag.set_diag
        torch_sparse.fill_diag = diag.fill_diag
        torch_sparse.SparseTensor.set_diag = diag.set_diag
        torch_sparse.SparseTensor.fill_diag = diag.fill_diag
        return

    original_set_diag = diag.set_diag
    original_fill_diag = diag.fill_diag

    @wraps(original_set_diag)
    def set_diag(src, values=None, k=0):
        if src.device().type == "npu":
            return _set_diag_npu(src, values, k)
        return original_set_diag(src, values, k)

    @wraps(original_fill_diag)
    def fill_diag(src, fill_value, k=0):
        if src.device().type != "npu":
            return original_fill_diag(src, fill_value, k)
        value = src.storage.value()
        rows, cols = src.sparse_sizes()
        num_diag = max(0, min(rows, cols - k) if k >= 0 else min(rows + k, cols))
        values = None if value is None else value.new_full(
            (num_diag,) + value.shape[1:], fill_value)
        return _set_diag_npu(src, values, k)

    diag.set_diag = torch_sparse.set_diag = set_diag
    diag.fill_diag = torch_sparse.fill_diag = fill_diag
    torch_sparse.SparseTensor.set_diag = set_diag
    torch_sparse.SparseTensor.fill_diag = fill_diag
    _frontend_state["set_diag"] = set_diag
    _frontend_state["fill_diag"] = fill_diag


_install_torch_sparse_frontends()

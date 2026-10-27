# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU checks for the current written FPS acceptance contract.

This reference deliberately separates metadata counting from greedy point
selection. Counts follow the taskbook's torch_cluster CPU oracle: scalar
ratios are stored in src.dtype, then deg.float() * ratio is rounded and ceiled.
The selected NPU sequence is then checked inductively with real PyTorch CPU
distance, minimum and first-index argmax operations. No ratio is changed
before a CPU call, and this is not the unavailable reviewer 66-case suite.
"""

import ctypes
import os
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


Ratio = Union[Tensor, float]
Ptr = Union[Tensor, List[int]]


def _npu_ready():
    try:
        import torch_npu  # noqa: F401
        import ops_gnn as module
    except ImportError:
        pass
    else:
        try:
            torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", "0")))
            return bool(torch.npu.is_available()), module
        except RuntimeError:
            pass
    return False, None


NPU_READY, ops_gnn = _npu_ready()


def _ptr_and_global_indices(
    src: Tensor,
    batch: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
    ptr: Optional[Ptr] = None,
) -> Tuple[Tensor, List[Tensor]]:
    """Return CPU CSR boundaries and each segment original global indices."""
    num_points = src.size(0)
    if ptr is not None:
        ptr_cpu = (
            ptr.detach().cpu().to(torch.long)
            if isinstance(ptr, Tensor)
            else torch.tensor(ptr, dtype=torch.long)
        )
        if ptr_cpu.dim() != 1 or ptr_cpu.numel() < 2:
            raise AssertionError("ptr must be one-dimensional with B+1 entries")
        if int(ptr_cpu[0]) != 0 or int(ptr_cpu[-1]) != num_points:
            raise AssertionError("ptr endpoints must be 0 and src.size(0)")
        if bool((ptr_cpu[1:] < ptr_cpu[:-1]).any()):
            raise AssertionError("ptr must be non-decreasing")
        global_indices = [
            torch.arange(int(ptr_cpu[i]), int(ptr_cpu[i + 1]), dtype=torch.long)
            for i in range(ptr_cpu.numel() - 1)
        ]
        return ptr_cpu, global_indices

    if batch is None:
        ptr_cpu = torch.tensor([0, num_points], dtype=torch.long)
        return ptr_cpu, [torch.arange(num_points, dtype=torch.long)]

    batch_cpu = batch.detach().cpu().to(torch.long)
    if batch_cpu.dim() != 1 or batch_cpu.numel() != num_points:
        raise AssertionError("batch must have src.size(0) entries")
    if batch_size is None:
        batch_size = int(batch_cpu.max()) + 1 if batch_cpu.numel() else 0
    global_indices = [
        (batch_cpu == batch_id).nonzero(as_tuple=False).flatten().long()
        for batch_id in range(batch_size)
    ]
    degrees = torch.tensor([indices.numel() for indices in global_indices])
    ptr_cpu = torch.zeros(batch_size + 1, dtype=torch.long)
    torch.cumsum(degrees, 0, out=ptr_cpu[1:])
    return ptr_cpu, global_indices


def taskbook_sample_counts(
    degrees: Tensor, ratio: Ratio, src_dtype: torch.dtype = torch.float32
) -> Tensor:
    """Compute counts with torch_cluster 1.6.3 CPU's dtype transitions."""
    degree_cpu = degrees.detach().cpu().to(torch.long).flatten()
    if isinstance(ratio, Tensor):
        ratio_cpu = ratio.detach().cpu().flatten()
    else:
        ratio_cpu = torch.tensor([float(ratio)], dtype=src_dtype)
    if ratio_cpu.numel() not in (1, degree_cpu.numel()):
        raise AssertionError("ratio must be scalar or contain one value per batch")
    if isinstance(ratio, Tensor) and bool(((ratio_cpu <= 0) | (ratio_cpu > 1)).any()):
        raise AssertionError("every ratio must be in (0, 1]")
    if not isinstance(ratio, Tensor) and not 0 < float(ratio) <= 1:
        raise AssertionError("ratio must be in (0, 1]")
    product = degree_cpu.to(torch.float32) * ratio_cpu
    counts = torch.ceil(product).to(torch.long)
    return torch.where(degree_cpu > 0, counts.clamp_min(1), torch.zeros_like(counts))


def libc_random_starts(degrees: Tensor, seed: int) -> Tensor:
    """Generate torch_cluster 1.6.3 CPU starts after srand(seed)."""
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.srand.restype = None
    libc.rand.argtypes = []
    libc.rand.restype = ctypes.c_int
    libc.srand(seed)
    starts = []
    for degree in degrees.detach().cpu().to(torch.long).tolist():
        starts.append(0 if degree == 0 else libc.rand() % degree)
    return torch.tensor(starts, dtype=torch.long)


def reset_libc_seed(seed: int) -> None:
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.srand.restype = None
    libc.srand(seed)


def _point_distances(src: Tensor, point_index: int) -> Tensor:
    """Run the exact torch_cluster CPU distance expression on the real dtype."""
    delta = src - src[point_index]
    return delta.pow_(2).sum(1)


class FpsReferenceOptions(NamedTuple):
    """Related reference metadata, with the previous helper's exact defaults."""
    batch: Optional[Tensor] = None
    ratio: Ratio = 0.5
    random_start: bool = True
    batch_size: Optional[int] = None
    ptr: Optional[Ptr] = None
    expected_starts: Optional[Sequence[int]] = None


class _SequenceCheck(NamedTuple):
    actual_cpu: Tensor
    src_cpu: Tensor
    global_indices: List[Tensor]
    counts: Tensor
    starts: List[int]


def fps_reference(
    src: Tensor, options: FpsReferenceOptions = FpsReferenceOptions(),
) -> Tensor:
    """Generate an independent CPU greedy FPS result without native extensions.

    ``expected_starts`` supplies the per-segment starts for random sampling;
    deterministic sampling starts at zero. Equal distances select the first
    index, including repeated indices for identical points, as the CPU oracle
    does. Scalar-ratio storage and count promotion follow the public contract.
    """
    if src.device.type != "cpu":
        raise ValueError("the reference requires CPU input")
    if src.dtype not in (torch.float16, torch.float32):
        raise ValueError("the reference supports float16 and float32")
    if src.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    points = src.reshape(src.size(0), -1).contiguous()
    boundaries, segments = _ptr_and_global_indices(
        points, options.batch, options.batch_size, options.ptr
    )
    ratio = 0.5 if options.ratio is None else options.ratio
    counts = taskbook_sample_counts(boundaries[1:] - boundaries[:-1], ratio, src.dtype)
    if options.random_start:
        if options.expected_starts is None:
            raise ValueError("random sampling requires explicit per-segment starts")
        starts = [int(value) for value in options.expected_starts]
    else:
        starts = [0] * len(segments)
    if len(starts) != len(segments):
        raise ValueError("one start index is required per segment")
    result = []
    for indices, count, start in zip(segments, counts.tolist(), starts):
        if count == 0:
            continue
        if not 0 <= start < indices.numel():
            raise ValueError("the starting index must belong to its segment")
        segment = points.index_select(0, indices)
        selected = start
        distance = _point_distances(segment, selected)
        result.append(int(indices[selected]))
        for _ in range(1, count):
            selected = int(distance.argmax())
            result.append(int(indices[selected]))
            distance = torch.minimum(distance, _point_distances(segment, selected))
    return torch.tensor(result, dtype=torch.long)


def assert_taskbook_cpu_semantics(
    actual: Tensor, src: Tensor,
    options: FpsReferenceOptions = FpsReferenceOptions(),
) -> None:
    """Inductively verify NPU output with actual PyTorch CPU operations."""
    if not (actual.device.type == "npu"):
        raise AssertionError()
    if not (actual.dtype == torch.int64):
        raise AssertionError()
    actual_cpu = actual.cpu()
    src_cpu = src.detach().cpu().reshape(src.size(0), -1).contiguous()
    ptr_cpu, global_indices = _ptr_and_global_indices(
        src_cpu, batch=options.batch, batch_size=options.batch_size, ptr=options.ptr
    )
    degrees = ptr_cpu[1:] - ptr_cpu[:-1]
    counts = taskbook_sample_counts(degrees, options.ratio, src_cpu.dtype)
    if not (actual_cpu.numel() == int(counts.sum())):
        raise AssertionError(f"count mismatch: actual={actual_cpu.numel()} expected={int(counts.sum())} "
        f"per_segment={counts.tolist()}")

    if options.random_start:
        if options.expected_starts is None:
            raise AssertionError("random_start=True requires precomputed libc starts")
        starts = [int(value) for value in options.expected_starts]
    else:
        starts = [0] * len(global_indices)
    if len(starts) != len(global_indices):
        raise AssertionError("one start index is required per segment")

    _assert_selected_sequence(_SequenceCheck(
        actual_cpu, src_cpu, global_indices, counts, starts
    ))


def _assert_selected_sequence(state: _SequenceCheck) -> None:
    """Check the same ordered greedy sequence independently of the product."""
    actual_cpu, src_cpu, global_indices, counts, starts = state
    offset = 0
    for batch_id, (indices, count) in enumerate(zip(global_indices, counts.tolist())):
        if count == 0:
            continue
        chosen_global = actual_cpu[offset:offset + count]
        offset += count
        positions = {
            int(global_index): local
            for local, global_index in enumerate(indices)
        }
        try:
            chosen_local = [positions[int(value)] for value in chosen_global]
        except KeyError as error:
            raise AssertionError(
                f"batch {batch_id}: selected index {error.args[0]} is outside its segment"
            ) from error

        if not (chosen_local[0] == starts[batch_id]):
            raise AssertionError(f"batch {batch_id}: first index {chosen_local[0]} != {starts[batch_id]}")
        segment = src_cpu.index_select(0, indices)
        distance = _point_distances(segment, chosen_local[0])
        for step in range(1, count):
            expected = int(distance.argmax())
            if not (chosen_local[step] == expected):
                raise AssertionError(f"batch {batch_id} step {step}: selected local index "
                f"{chosen_local[step]} != CPU argmax {expected}")
            current = _point_distances(segment, chosen_local[step])
            distance = torch.minimum(distance, current)
    if not (offset == actual_cpu.numel()):
        raise AssertionError()


def assert_loaded_from(expected_root: Union[str, Path]) -> None:
    """Fail if Python or native libraries were loaded outside this candidate."""
    if not NPU_READY:
        raise AssertionError("an Ascend NPU is required")
    import ops_gnn._pybind as native

    root = Path(expected_root).resolve()
    module_path = Path(ops_gnn.__file__).resolve()
    native_path = Path(native.__file__).resolve()
    if not module_path.is_relative_to(root) or not native_path.is_relative_to(root):
        raise AssertionError("ops_gnn or _pybind loaded outside expected root")
    mapped = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libopsgnn_npu_kernel.so" in line or str(native_path) in line
    }
    if not mapped or any(not path.is_relative_to(root) for path in mapped):
        raise AssertionError("native mappings are missing or outside expected root")

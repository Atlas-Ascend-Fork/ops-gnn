# ops-gnn API Reference

This document provides detailed API interface documentation for the ops-gnn library, including operator signatures, parameter descriptions, return values, and usage examples.

## Type Definitions

### Tensor

```python
import torch
Tensor = torch.Tensor
```

Tensor on NPU device, memory managed by PyTorch.

### OptTensor

```python
from typing import Optional
OptTensor = Optional[torch.Tensor]
```

Optional Tensor type for parameters that may have default values. `None` is represented explicitly at the pybind boundary; a valid empty Tensor remains a real `out` argument.

---

## Core Operator APIs

### nearest — Batched Nearest Neighbor Indices

```python
def nearest(x: Tensor, y: Tensor,
            batch_x: Optional[Tensor] = None,
            batch_y: Optional[Tensor] = None) -> Tensor:
```

On Ascend 950 (arch35), return the global int64 index of the nearest y point in the same batch for each x point. The output has shape [N]. Inputs share an NPU and float16/float32 dtype, with shapes [N,F]/[M,F]; 1-D inputs use F=1. Both dtypes use FP32 distance arithmetic and choose the smallest global y index on ties.

Optional batches are nonnegative sorted 1-D int64 labels of length N/M on the input device, with matching nonempty label sets; an omitted batch means zero. Invalid batches raise ValueError. Noncontiguous inputs are supported. Empty x without batches returns an empty output; nonempty x with empty y is rejected. NaN/Inf coordinates and F=0 are rejected.

```python
x = torch.tensor([[0., 0., 0.], [9., 0., 0.]], device="npu")
y = torch.tensor([[1., 0., 0.], [10., 0., 0.]], device="npu")
indices = ops_gnn.nearest(x, y)  # tensor([0, 1], device='npu:0')
```

Computation uses an Ascend C kernel registered at torch.ops.torch_cluster.nearest. No CPU/scipy fallback or gradients are provided. N, M and F must not exceed INT32_MAX and are limited by available memory. Public API validation includes synchronization; the kernel uses the current NPU stream. If torch_cluster is installed, import it first to provide the schema.

The fixed precision reference is SciPy 1.18.0 / OpenBLAS 0.3.31.dev SkylakeX (this SciPy release requires Python ≥3.12). A different CPU, BLAS or matrix partition may change integer reference indices for nearly tied candidates. Accuracy is not guaranteed for extreme FP32 coordinates that overflow intermediate distances.

Tests in `test/nearest/arch35/` have two entry points and share the reference helper `golden.py`:

- Functional tests: `test_nearest.py` contains 338 cases: the 42 task-attachment cases and 296 supplemental cases covering functionality, FP16, reconstructed acceptance logs, 108 numerical stress cases and four reference-semantics cases. The reference calculation and assertions remain unchanged. Task shapes and cross-1024 ties cover the original nine failure scenarios.
- Performance tests: `benchmark_nearest.py` covers 18 shapes and two dtypes, with 20 warmups and 100 timed calls per point. It measures the full Python API, requires reference/NPU latency of at least 0.45 and compares all output indices exactly.

After building the repository and activating the CANN and Python environments, run from the repository root (installing the reference dependency `scipy==1.18.0` requires Python ≥3.12):

```bash
python -m pip install scipy==1.18.0 pytest
export NPU_DEVICE_ID=0
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest test/nearest/arch35/test_nearest.py -v
python test/nearest/arch35/benchmark_nearest.py --warmup 20 --iter 100
```

Tests compare int64 indices with `torch.equal`. Performance results are written locally to `artifacts/nearest/` and are not committed. The CPU reference keeps each batch intact; its largest temporary matrix is about 4 GiB, so at least 8 GiB of free memory is recommended. Run NPU tests sequentially.

---

### segment_max_csr — CSR Segmented Max

This section describes the existing `int32` pointer API. The new `int64`
pointer family is documented under [`segment_csr`](#segment_csr--int64-csr-segment-reductions).
The two paths differ in empty-segment and return-value semantics.

**Function Signature:**

```python
def segment_max_csr(
    src: Tensor,
    indptr: Tensor,
    optional_out: Optional[Tensor] = None,
) -> Tensor:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| src | Tensor | Input | Input data Tensor, must be on NPU device |
| indptr | Tensor | Input | CSR format index pointer Tensor, dtype must be `torch.int32`, must be on NPU device. Last dimension length - 1 determines the number of segments |
| optional_out | Optional[Tensor] | Input | Optional output Tensor, on NPU device. If provided, the first element of each segment is compared with the corresponding value in `optional_out` via max (i.e., `max(src[start:end], optional_out)`), rather than taking max only from src |

**Return Value:**

Returns a new Tensor, same dtype as `src`. Shape is the same as `src` but the dimension corresponding to the last dim of `indptr` is reduced to `nSegments = indptr_last_dim - 1`.

**Description:**

Performs segmented max reduction on `src` along the dimension specified by CSR format `indptr`. For each segment `[indptr[seg], indptr[seg+1])`, computes the maximum of all elements within that segment. Supports multi-dimensional broadcasting: when `indptr` has fewer dimensions than `src`, indptr is broadcast along corresponding dimensions.

**Implementation Architecture:**

- **Kernel Mode**: Kernel class mode (TPipe + Buffer + Event), 4-file implementation
- **Data Movement**: DataCopy + double-buffer pipeline (MTE2 ↔ VECCALC ↔ MTE3)
- **Tiling**: `SegmentMaxCsrTilingData` struct, includes block parameters (coreDataNum, KloopTime, ALIGN_NUM, etc.)
- **Multi-AIV Parallelism**: Evenly distributes `E_1` (broadcast dimension) across AIV cores, each core processes independently

**Call Chain:**

```text
ops_gnn.segment_max_csr(src, indptr, optional_out)
    → _pybind.segment_max_csr()           # PyTorch binding
        → segment_max_csr()                # Host: dimension parsing, Tiling fill, dtype dispatch
            → LaunchSegmentMaxCsrKernel<T>()  # Kernel Launch: get AIV cores, <<<>>> launch
                → segment_max_csr_kernel<T>   # Device: Init → Process → Compute pipeline
```

**Supported dtypes:**

| dtype | Fill value (empty segment) |
|-------|---------------------------|
| `torch.float32` | `-3.4028235e+38` (-FLT_MAX) |
| `torch.float16` | `-65504` (-HALF_MAX) |
| `torch.int32` | `-2147483648` (INT32_MIN) |
| `torch.int16` | `-32768` (INT16_MIN) |

**Notes:**

- `indptr` dtype must be `torch.int32`
- `indptr` values must be non-decreasing (i.e., `indptr[seg] <= indptr[seg+1]`)
- Empty segment (`indptr[seg] == indptr[seg+1]`): if no `optional_out`, fills with the corresponding dtype minimum; if `optional_out` exists, directly copies the value from `optional_out`
- Supports broadcasting: 1D indptr can broadcast to multiple batches (`indptr.view(1, -1)`)
- Inputs must be on NPU device (`device='npu'`)

**Usage Examples:**

#### Basic Usage — 1D indptr

```python
import torch
import os
import ops_gnn

device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
torch.npu.set_device(device_id)
torch.manual_seed(42)

# Create input data
# src shape: (4, 2), 4 "rows", 2 elements each
src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
# indptr: [0, 2, 4] means segment 0 takes src[0:2], segment 1 takes src[2:4]
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[3, 4], [7, 8]]  — shape: (2, 2)
#           seg0: max(src[0:2])  seg1: max(src[2:4])
```

#### Different dtypes

```python
# float16
src = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16, device='npu')
indptr = torch.tensor([0, 2], dtype=torch.int32, device='npu')
result = ops_gnn.segment_max_csr(src, indptr)

# int32
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device='npu')
indptr = torch.tensor([0, 2], dtype=torch.int32, device='npu')
result = ops_gnn.segment_max_csr(src, indptr)
```

#### Empty Segment

```python
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='npu')
# indptr: [0, 0, 2] — segment 0 is empty, segment 1 takes src[0:2]
indptr = torch.tensor([0, 0, 2], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result[0] = [-3.4028235e+38, -3.4028235e+38]  — empty segment filled with finite float32 minimum (-FLT_MAX)
# result[1] = [3, 4]        — max of src[0:2]
```

#### With optional_out

```python
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='npu')
indptr = torch.tensor([0, 2], dtype=torch.int32, device='npu')
optional_out = torch.tensor([[10, 20]], dtype=torch.float32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr, optional_out)
# result = [[10, 20]]  — max(src[0:2], optional_out) = max([[1,2],[3,4]], [[10,20]])
```

#### 2D indptr (Different Segments per Batch)

```python
src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
# 2D indptr: batch 0 uses [0, 2, 4], batch 1 uses [0, 1, 3]
indptr = torch.tensor([[0, 2, 4], [0, 1, 3]], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[2, 4],    — batch 0: seg0=src[0:2], seg1=src[2:4]
#           [5, 7]]    — batch 1: seg0=src[0:1], seg1=src[1:3]]
```

#### indptr Broadcasting

```python
src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
# 1D indptr, viewed as (1, -1) to broadcast across two batches
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu').view(1, -1)

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[2, 4],    — batch 0: seg0=src[0:2], seg1=src[2:4]
#           [6, 8]]    — batch 1: same segments
```

#### Complex Shapes (3D src + 2D indptr)

```python
src = torch.randn(3, 8, 16, dtype=torch.float32, device='npu')
indptr = torch.tensor([[0, 4, 8]], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result shape: (3, 2, 16) — indptr last dim reduced from 3→2, other dims unchanged
```

---

## segment_csr — int64 CSR Segment Reductions

On Ascend 950 / arch35, reduce CSR segments deterministically with `sum`,
`add`, `mean`, `min`, or `max`. The reduction axis is `indptr.dim() - 1`, and
its output size is `indptr.size(-1) - 1`. Each output has one writer; no global
atomic accumulation is used.

```python
segment_csr(src, indptr, out=None, reduce="sum")  # values Tensor only
segment_sum_csr(src, indptr, out=None)            # Tensor
segment_add_csr(src, indptr, out=None)            # Tensor
segment_mean_csr(src, indptr, out=None)           # Tensor
segment_min_csr(src, indptr, out=None)            # (values, indices)
segment_max_csr(src, indptr, out=None)            # int64 indptr: (values, indices)
```

| Argument | Contract |
| --- | --- |
| `src` | NPU Tensor; float16, float32, bfloat16, int8, uint8, int32, or int64 |
| `indptr` | Same-device `int64` Tensor; nondecreasing last dimension with in-range boundaries; leading dimensions may broadcast to `src` |
| `out` | Optional same-device Tensor with matching dtype and shape; noncontiguous views are supported |
| `reduce` | Generic API only; `mul` and `dim_size` are unsupported |

For `min/max`, `indices` contains the first extremum's `int64` position along
the input reduction axis. Empty segments yield zero values and an index equal
to the input reduction-axis length. Empty `src` preserves a supplied `out`.
Integer `mean` casts the sum and segment length to the source dtype before
division toward zero; floating `mean` preserves the sum and count rounding
order. An int8/uint8 count that casts to zero yields zero. Invalid pointers
and NaN extrema propagation are outside this interface's extended contract.

```python
import torch
import torch_npu
import ops_gnn

src = torch.tensor([[1., 4.], [2., 3.], [5., 6.]], device="npu")
indptr = torch.tensor([0, 2, 3], dtype=torch.int64, device="npu")
summed = ops_gnn.segment_csr(src, indptr, reduce="sum")
values, indices = ops_gnn.segment_max_csr(src, indptr)
```

For an `int32` pointer, `segment_max_csr` retains the existing value-only
path. The historical `optional_out=` keyword remains available through
`segment_max_csr_legacy`.

---

### radius / radius_graph — Radius Neighbor Search

NPU implementation interface-compatible with `torch_cluster.radius` /
`radius_graph` (>= 1.6.0), Ascend 950PR. For each query point in `y`, finds all
neighbors in `x` with Euclidean distance `dist² <= r²` (within the same batch);
when the count exceeds `max_num_neighbors`, keeps the first K in scan order
(deterministic). Outputs `edge_index [2, E]` (int64).

```python
def radius(
    x, y, r,
    batch_x=None, batch_y=None,
    max_num_neighbors=32, num_workers=1,
    batch_size=None, ignore_same_index=False,
) -> torch.Tensor          # [2, E] int64

def radius_graph(
    x, r,
    batch=None, loop=False,
    max_num_neighbors=32,
    flow='source_to_target', num_workers=1, batch_size=None,
) -> torch.Tensor          # [2, E] int64
```

| Parameter | Type | In/Out | Description |
|-----------|------|--------|-------------|
| `x` | Tensor [N,F] | in | Neighbor candidate set (float16/bf16/float32; float64 via CPU fallback) |
| `y` | Tensor [M,F] | in | Query point set |
| `r` | float | in | Search radius (>0) |
| `batch_x` / `batch_y` | Tensor | in | Batch membership (must be sorted) |
| `max_num_neighbors` | int | in | Max neighbors per query (default 32) |
| `ignore_same_index` | bool | in | Skip `i == q` self-loops |
| `loop` / `flow` | bool/str | in | radius_graph self-loop and direction semantics |

```python
x = torch.randn(1000, 3, device='npu')
edge = ops_gnn.radius(x, x, 0.8)          # neighbors within radius 0.8
edge_g = ops_gnn.radius_graph(x, 0.8)     # build a radius graph (loop=False by default)
```

**Implementation architecture:**

- **Kernel mode**: Ascend C SIMT (`__simt_vf__` + `VF_CALL`), spatial-grid pruning + sorted-array top-K + early-break
- **Grid build**: executed on device (min/max, cell, sort, index_select, offsets), avoiding CPU sort bottleneck
- **Output compaction**: device-side `repeat_interleave` + `masked_select` into `[2, E]`
- **Use case**: 3D point-cloud / GNN neighborhood construction (PointNet++, DGCNN, etc.)

**Notes:**

- `x` / `y` must be tensors on the same NPU device, with matching feature dim `F`; non-contiguous inputs are made contiguous internally
- `batch_x` / `batch_y` must be sorted; search is confined within each batch
- Exceeding `max_num_neighbors` keeps first K in scan order (deterministic, consistent with CPU reference)
- `radius_graph` `loop` / `flow` semantics match `torch_cluster`
- L1: float16 / bfloat16 / float32 (NPU); float64 via CPU fallback (bit-wise, not performance-tested)
- Empty input returns `[2, 0]` LongTensor without entering the kernel

### graclus_cluster — Greedy Graph Clustering

**Function Signature:**

```python
def graclus_cluster(
    row: Tensor,
    col: Tensor,
    weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| row | Tensor | Input | COO source node indices, dtype must be `torch.long` |
| col | Tensor | Input | COO target node indices, dtype must be `torch.long` |
| weight | Optional[Tensor] | Input | Optional edge weights. float16, bfloat16, and float32 use the NPU path; float64 uses CPU fallback semantics |
| num_nodes | Optional[int] | Input | Number of nodes. If omitted, it is inferred from `row` and `col` |

**Return Value:**

Returns a `torch.long` Tensor with shape `[num_nodes]`. Each element is the cluster ID of the corresponding node.

**Description:**

Implements the greedy graph clustering semantics of `torch_cluster.graclus_cluster`. The Python layer infers `num_nodes`, removes self-loops, shuffles edges when `weight` is absent, sorts edges by row, and converts COO to CSR. The NPU path launches an Ascend C kernel to visit nodes in random order and greedily match each unmarked node with an unmarked neighbor. When `weight` is provided, the neighbor with maximum weight is selected; otherwise the first unmarked neighbor is selected.

**Notes:**

- `row` and `col` must be 1D `torch.long` tensors
- `row`, `col`, and `weight` must be on the same device
- The algorithm contains randomness; set `torch.manual_seed` before calling the operator when reproducible output is required
- Self-loops are removed before clustering

### gather_coo — COO Row Expansion

**Function signature:**

```python
def gather_coo(
    src: Tensor,
    index: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
```

Let `dim = index.dim() - 1`. The prefix shape of `index` must match the first `dim` dimensions of `src`. The output has the same shape as `src`, except `src.size(dim)` is replaced by `index.size(-1)`, and:

```text
out[..., e, ...] = src[..., index[..., e], ...]
```

`index` must be an NPU `torch.int64` tensor and satisfy the task's sorted, in-range preconditions. The operator is forward-only and performs a raw bit copy; it supports ranks 1–8, non-contiguous inputs/outputs, empty tensors, and FP16/BF16/FP32, INT8/16/32, UINT8, FP64, and INT64.

When `out` is supplied, it must have the exact inferred shape, matching dtype, and matching device. The returned tensor shares storage with `out`. `None` is distinct from an explicitly supplied empty `out` tensor.

**Example:**

```python
import torch
import ops_gnn

device = torch.device("npu")  # use the process's current NPU; no fixed ordinal
src = torch.arange(20, dtype=torch.float32, device=device).reshape(5, 4)
index = torch.tensor([0, 1, 1, 4], dtype=torch.int64, device=device)
out = ops_gnn.gather_coo(src, index)
# out.shape == (4, 4); repeated index=1 copies the same source row

provided = torch.empty_like(out)
returned = ops_gnn.gather_coo(src, index, out=provided)
assert returned.data_ptr() == provided.data_ptr()
```

**Implementation and performance notes:**

- The Host flattens inputs to `B × N × K`, `B × E`, and `B × E × K`, using 64-bit lengths and offsets. The kernel reuses PyTorch's current NPU stream and never creates or synchronizes a private ACL stream.
- Non-contiguous `src/index` tensors are made contiguous on the current stream. Explicit `out` uses a contiguous temporary result before copy-back, covering non-contiguous and aliasing cases.
- Consecutive equal sorted indices reuse a source row within one UB tile. Dispatch does not depend on test-case IDs or fixed-shape whitelists.

---

## ind2ptr — Sorted Row Indices to CSR Row Pointer

**Function signature:**

```python
def ind2ptr(
    ind: Tensor,
    num_rows: int,
) -> Tensor:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| ind | Tensor | Input | 1-D `torch.long` non-decreasing row indices on NPU |
| num_rows | int | Input | Number of rows (historically `M` in torch_sparse); output length is `num_rows + 1` |

**Returns:**

A 1-D `torch.long` CSR row-pointer tensor of shape `[num_rows + 1]` on the same device as `ind`.

**Description:**

Converts sorted row indices to a CSR row pointer. Drop-in replacement for `torch.ops.torch_sparse.ind2ptr(ind, M)`. Runs asynchronously on the current NPU stream; synchronize with `torch.npu.synchronize()` before reading results on host if needed.

**Notes:**

- `ind` must be non-decreasing; an empty input yields an all-zero pointer of length `num_rows + 1`
- dtype must be `torch.long` (int64)
- Non-contiguous inputs are made contiguous before the call

**Example:**

```python
import torch
import ops_gnn

row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
rowptr = ops_gnn.ind2ptr(row, 8)
# tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], device='npu:0')
```

---

## ptr2ind — CSR Row Pointer to Row Indices

**Function signature:**

```python
def ptr2ind(
    ptr: Tensor,
    num_edges: int,
) -> Tensor:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| ptr | Tensor | Input | 1-D `torch.long` CSR row pointer of length `num_rows + 1` on NPU |
| num_edges | int | Input | Number of edges / non-zeros (historically `E` in torch_sparse); output length is `num_edges` |

**Returns:**

A 1-D `torch.long` row-index tensor of shape `[num_edges]` on the same device as `ptr`.

**Description:**

Converts a CSR row pointer to row indices. Drop-in replacement for `torch.ops.torch_sparse.ptr2ind(ptr, E)`. Runs asynchronously on the current NPU stream; synchronize with `torch.npu.synchronize()` before reading results on host if needed.

**Notes:**

- dtype must be `torch.long` (int64)
- `num_edges == 0` returns an empty index tensor
- Non-contiguous inputs are made contiguous before the call

**Example:**

```python
import torch
import ops_gnn

rowptr = torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long, device='npu')
row = ops_gnn.ptr2ind(rowptr, 6)
# tensor([2, 2, 4, 5, 5, 6], device='npu:0')
```

---

## random_walk — NPU Random Walk

**Function Signature:**

```python
def random_walk(
    row: Tensor,
    col: Tensor,
    start: Tensor,
    walk_length: int,
    p: float = 1,
    q: float = 1,
    coalesced: bool = True,
    num_nodes: Optional[int] = None,
    return_edge_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
```

**Parameters:**

| Parameter | Type | I/O | Description |
|-----------|------|-----|-------------|
| row | Tensor | Input | COO source nodes, a 1D `torch.int64` NPU Tensor |
| col | Tensor | Input | COO destination nodes, a 1D `torch.int64` NPU Tensor with the same length as `row` |
| start | Tensor | Input | Walk starting nodes, a 1D `torch.int64` NPU Tensor |
| walk_length | int | Input | Number of walk steps; must be non-negative |
| p | float | Input | node2vec return parameter; must be finite and greater than zero |
| q | float | Input | node2vec BFS/DFS parameter; must be finite and greater than zero |
| coalesced | bool | Input | Sorts edges by `(row, col)` when True; when False, edges must already be grouped by source node |
| num_nodes | Optional[int] | Input | Number of nodes; inferred from the maxima of `row`, `col`, and `start` when None |
| return_edge_indices | bool | Input | Whether to return the edge position selected at each step |

**Return Value:**

Returns `node_seq`, a `torch.int64` Tensor with shape `[S, walk_length + 1]`.
When `return_edge_indices=True`, returns `(node_seq, edge_seq)`, where `edge_seq`
is a `torch.int64` Tensor with shape `[S, walk_length]`. An isolated node remains
unchanged and its corresponding `edge_seq` value is `-1`.

**Description:**

Runs uniform or node2vec-biased random walks from multiple starting nodes on a
COO graph. The Python layer sorts the COO edges and builds CSR data on NPU. An
Ascend 950 SIMT kernel performs uniform sampling when `p=1, q=1`; otherwise it
uses rejection sampling with node2vec transition probabilities.

**Implementation Architecture:**

- **Python preprocessing**: argument validation, COO sorting, degree counting, and CSR `rowptr` construction
- **Host layer**: obtains the current PyTorch NPU stream and Philox seed/offset, then calculates integer node2vec thresholds
- **Kernel mode**: Ascend C SIMT, with one thread processing one walk
- **Output path**: skips edge-sequence writes when `return_edge_indices=False`

**Notes:**

- `row`, `col`, and `start` must be on the same NPU device, with `0 <= index < num_nodes`
- `coalesced=False` does not reorder edges; input edges must already be grouped by source node
- With sorting enabled, edge indices refer to positions in the sorted COO/CSR representation
- The total edge count and the degree of each node must not exceed `2^32-1`
- Only forward computation is supported; use `torch.manual_seed(seed)` to seed the default NPU Generator

**Usage Example:**

```python
import torch
import ops_gnn

device = "npu:0"
row = torch.tensor([0, 1, 1, 2], dtype=torch.int64, device=device)
col = torch.tensor([1, 0, 2, 1], dtype=torch.int64, device=device)
start = torch.tensor([0, 2], dtype=torch.int64, device=device)

torch.manual_seed(202608)
nodes, edges = ops_gnn.random_walk(
    row, col, start, walk_length=8, p=0.5, q=2.0,
    return_edge_indices=True,
)
assert nodes.shape == (2, 9)
assert edges.shape == (2, 8)
```

---

### gather_csr - CSR Segment Expansion

**Signature:**

```python
def gather_csr(
    src: Tensor,
    indptr: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
```

`gather_csr` expands segment features and is the inverse operation of
`segment_csr`. Let `dim = indptr.dim() - 1`. For every segment `i`:

```text
out[..., indptr[i]:indptr[i + 1], ...] = src[..., i, ...]
```

The interface contains only `src`, `indptr`, and optional `out`; it has no
`reduce`, `dim`, or `dim_size` argument.

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | Tensor | NPU tensor with a supported L1 or L2 dtype |
| `indptr` | Tensor | Non-decreasing int64 CSR pointers on the same NPU |
| `out` | Optional[Tensor] | Optional output matching the inferred dtype, device, rank, and shape |

The operator supports prefix broadcasting, empty segments and tensors, and
non-contiguous `src`, `indptr`, and `out`. When provided, the same `out` object
is updated and returned. Only forward execution is supported.

**Supported inputs:**

| Item | Support |
|------|---------|
| Hardware | Ascend 950PR (`dav-3510`) |
| `src/out` dtype | float16, bfloat16, float32, int8, int16, int32, uint8, float64, int64 |
| `indptr` dtype | int64 |
| Rank | `1 <= indptr.dim() <= src.dim()` |
| Layout | Contiguous and non-contiguous tensors |
| Special cases | Batch broadcasting, empty segments/tensors, optional `out` |
| Direction | Forward only |

Every dtype is copied as raw bytes without arithmetic or type conversion.
Float64 and int64 are L2 functional paths and are excluded from performance
acceptance.

**Constraints:**

- `src`, `indptr`, and optional `out` must be on the same NPU.
- `indptr.dim() <= src.dim()`, and its prefix dimensions must broadcast to `src`.
- If `indptr.size(-1) > 0`, `src.size(dim) == indptr.size(-1) - 1`. If the last dimension of
  `indptr` is empty, `src.size(dim)` must be `0`.
- Every `indptr` row must be non-decreasing, stay in `[0, endpoint]`, and have
  the same endpoint after broadcasting.
- The output length at `dim` is the endpoint, or zero for an empty `indptr`.
- Optional `out` must match the inferred dtype, device, rank, and shape.

Empty `src` inputs still use the endpoint to infer the output Shape and validate
broadcasting, monotonicity, and range. An empty `indptr` uses an endpoint of zero.

**Implementation:**

The Host layer validates arguments, materializes broadcast pointers, handles
non-contiguous tensors, allocates output, and prepares tiling. The Ascend C
Kernel selects `SegmentMajor` or `OutputMajor`. The former assigns
`(batch, segment)` jobs to AIV cores; the latter partitions output rows when
there are too few segments or one segment is heavily skewed. Aligned features
use a 64 KB UB repeat buffer for batched writes, while other features use 16 KB
tiles. The Kernel runs on the current PyTorch NPU stream.

**Example:**

```python
import os
import torch
from ops_gnn import gather_csr

device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
torch.npu.set_device(device_id)
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device="npu")
indptr = torch.tensor([0, 2, 5], dtype=torch.int64, device="npu")
out = gather_csr(src, indptr)
# [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
```

**Build and test:**

```bash
# Install torch_scatter >= 2.1.0 for the CPU reference.
cmake -S . -B build/cmake_release \
  -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake_release -j4
export PYTHONPATH=$PWD/python
export NPU_DEVICE_ID=<device_id>
pytest -q test/gather_csr/arch35/test_gather_csr.py
python test/gather_csr/arch35/golden.py
python test/gather_csr/arch35/benchmark_gather_csr.py --ascendoptest \
  --ascendoptest-root /path/to/AscendOpTest
python test/gather_csr/arch35/benchmark_gather_csr.py --warmup 20 --iterations 101
```

---

## scatter — Scatter Reductions

The Scatter family follows the forward semantics of `torch_scatter` 2.1.2:

```python
ops_gnn.scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum")
ops_gnn.scatter_sum(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_add(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_mul(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_mean(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_min(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_max(src, index, dim=-1, out=None, dim_size=None)
```

`scatter_min` and `scatter_max` return `(out, arg_out)`; the other functions
return `out`. `scatter_add` and `scatter_sum` have identical reduction
semantics.

### Arguments

- `src`: input Tensor with rank 1 through 8.
- `index`: INT64 Tensor broadcastable to `src` under `torch_scatter` rules.
- `dim`: reduction dimension; negative dimensions are supported. Default: `-1`.
- `out`: optional output Tensor, updated in place while preserving identity.
- `dim_size`: optional output size on `dim`; inferred from `index` when omitted.
- `reduce`: `sum`, `add`, `mul`, `mean`, `min`, or `max`.

#### Dtypes and execution paths

| Level | `src/out` dtype | Execution path | Reductions |
|---|---|---|---|
| L1 | float16, bfloat16, float32, int8, int16, int32, uint8 | Ascend C NPU kernel | all six |
| L2 | float64, int64 | synchronous CPU fallback copied back to the original device | all six |

Integer `mean` uses floor division. For equal extrema, `min/max` selects the
later writer. Unwritten output positions contain zero and their `arg_out`
value is `src.size(dim)`. The APIs support explicit `dim_size`, empty and
non-contiguous Tensors, unordered or duplicate indices, and high-contention
indices. Only forward computation is provided.

#### Example

```python
import os
import torch
import ops_gnn

torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
src = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
index = torch.tensor([0, 1, 0, 1], dtype=torch.long, device="npu")

out = ops_gnn.scatter_sum(src, index)
# tensor([4., 6.], device='npu:0')

values, arg = ops_gnn.scatter_max(src, index)
# values: tensor([3., 4.], device='npu:0')
# arg:    tensor([2, 3], device='npu:0')
```

## spmm — General CSR Aggregation

```python
ops_gnn.spmm(
    indptr, indices, x=None, op="copy_lhs", reduce="sum",
    out=None, rhs=None
) -> Tensor
```

Runs message generation and destination-row aggregation through ops-gnn CSR
tensors without depending on `DGLGraph`, `SparseMatrix`, or the DGL FFI.

- `indptr`: one-dimensional NPU `int32` or `int64` destination-row pointer.
- `indices`: source indices in CSR edge order, with the same dtype and device.
- `x`: source features. As a shorthand, `copy_rhs` may receive CSR-ordered edge features here.
- `rhs`: CSR-ordered edge features used by binary message operations.
- `op`: `copy_lhs`, `copy_rhs`, `add`, `sub`, `mul`, or `div`.
- `reduce`: `sum`, `max`, `min`, or `mean`.
- `out`: optional output; it cannot be supplied when a feature requires gradients.

`copy_lhs` uses `x[indices]`, while `copy_rhs` uses edge features directly.
Binary operations compute `op(x[indices], rhs)` with PyTorch broadcasting.
Features may be one-dimensional scalars or two-dimensional tensors. Empty CSR
rows produce zero.

```python
from ops_gnn import spmm

y = spmm(ptr, idx, node_features, reduce="sum")
y = spmm(ptr, idx, node_features, rhs=edge_features,
         op="mul", reduce="mean")
```

No-gradient `copy_lhs/copy_rhs + sum/max/min` calls whose shapes and dtypes
match a kernel path use the migrated NPU kernels. Binary messages, `mean`,
higher-rank features, and calls requiring gradients use differentiable NPU
tensor operations. `copy_rhs` supports float32; the other paths support float16
and float32.

### bspmm — General Batched CSR Aggregation

```python
ops_gnn.bspmm(
    indptr, indices, x=None, op="copy_lhs", reduce="sum",
    out=None, rhs=None
) -> Tensor
```

`bspmm` supports the same message operations, reductions, broadcasting, and
autograd as `spmm`. At least one node or edge feature tensor must have rank
three or higher. For binary operations, missing feature dimensions are inserted
after the item axis before PyTorch broadcasting, and the output retains every
broadcast trailing dimension. Rank-3 FP16 `copy_lhs + sum/max/min` calls use the
migrated BSpMM kernels; higher-rank and other combinations use NPU tensor
operations.

```python
from ops_gnn import bspmm

# node_features: [K, 3, 1], edge_features: [nnz, 2, 3, 4]
y = bspmm(ptr, idx, node_features, rhs=edge_features,
          op="mul", reduce="mean")
# y: [M, 2, 3, 4]
```

## 2.11 grid_cluster — Regular Grid Clustering

Map `[N,D]` point coordinates to linear regular-grid IDs. Points in the same cell receive the same ID. The interface is compatible with `torch_cluster.grid_cluster`.

**Signature:**

```python
def grid_cluster(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...
```

**Arguments:**

| Argument | Shape | Description |
| --- | --- | --- |
| `pos` | `[N,D]` | Point coordinates on an NPU; `N >= 0`, `D > 0`. Supports float16, bfloat16, float32, int8, uint8, int16, and int32. |
| `size` | `[D]` | Cell size in each dimension. All values must be positive and finite. |
| `start` | `[D]` or `None` | Optional grid origin; defaults to the per-dimension minimum of `pos`. |
| `end` | `[D]` or `None` | Optional grid bound; defaults to the per-dimension maximum of `pos`. |

All inputs must be on the same NPU, and supplied vectors must have length `D`. Noncontiguous inputs are accepted. Mixed dtypes follow PyTorch type promotion.

**Returns:** A `torch.int64` tensor of shape `[N]` on the same NPU. Empty `pos` returns an empty tensor.

**Example:**

```python
import torch
import ops_gnn

pos = torch.tensor([[0., 0.], [0.4, 0.4], [1.2, 0.]], device="npu")
size = torch.tensor([1., 1.], device="npu")
cluster = ops_gnn.grid_cluster(pos, size)
# tensor([0, 0, 1], device='npu:0')
```

**Semantics and limits:** Each grid coordinate is computed by truncating `(pos - start) / size` toward zero, then linearized with stride one in dimension 0. Explicit `start/end` values do not clip points outside the grid. Final IDs use `int64` and follow the CPU reference's overflow behavior. Autograd is not provided.

---


## torch_sparse_spmm

NPU sparse-dense multiplication for torch_sparse, integrated into the ops_gnn
build and package. COO supports sum; CSR supports sum/add/mean/min/max, optional
weights, batching and floating-point sum/mean first-order gradients.

See the [development guide](development_guide.md#torch_sparse_spmm) for building/testing and the [design document](../experiment/design/torch_sparse_spmm/torch_sparse_spmm设计文档_en.md) for implementation details.

### Usage

Import the adapter before importing torch_sparse/PyG frontend aliases:

```python
import torch
import torch_npu
from ops_gnn import torch_sparse_spmm
from torch_sparse import spmm

torch.npu.set_device(0)
index = torch.tensor([[0, 0, 2], [1, 2, 0]], device="npu", dtype=torch.int64)
value = torch.tensor([0.5, 1.0, -1.0], device="npu", requires_grad=True)
dense = torch.randn(3, 16, device="npu", requires_grad=True)
out = spmm(index, value, 3, 3, dense)
out.square().sum().backward()
```

SparseTensor.spmm/matmul dense-input paths and torch.ops.torch_sparse retain
upstream schemas. The csr_spmm helper accepts output_dtype for mixed outputs:

```python
rowptr = torch.tensor([0, 2, 2, 3], device="npu", dtype=torch.int64)
col = index[1].contiguous()
out = torch_sparse_spmm.csr_spmm(rowptr, col, value, dense, reduce="mean")
with torch.no_grad():
    out, arg = torch_sparse_spmm.csr_spmm(rowptr, col, value, dense, reduce="max")
```

#### Contract

| Input | Output | Accumulation |
| --- | --- | --- |
| FP32 | FP32 | FP32 |
| FP16 | FP16 or FP32 | FP32 |
| BF16 | BF16 or FP32 | FP32 |
| INT8 | INT32 or FP32 | INT32 or FP32 respectively |

- Indices are int64 on the same NPU. Dense shape is [..., N, K]; output is [..., M, K].
- Weights are optional or have E leading entries with 1 or K values per edge,
  matching the dense input dtype and device.
- COO supports sum; CSR supports all four reductions, with add equivalent to sum.
  Empty rows return zero with arg=E. Min/max ignore NaN and retain the first tied
  edge; all-NaN rows return zero/E. Integer mean truncates toward zero.
- FP16/BF16 use FP32 computation. Floating outputs default to the input dtype;
  INT8 defaults to INT32.
- Floating sum/mean support first-order value/mat gradients. Min/max gradients,
  double backward and sparse-sparse multiplication are outside scope.
- Python/host checks and device CSR validation protect inputs. NPU computation
  never falls back to CPU. Topology caches do not cache numerical inputs.
- PyTorch mutations invalidate caches. Inference tensors are not cached. Clear
  caches after external raw-pointer writes. clear_validation_cache clears entries;
  validation_cache_info returns an immutable snapshot. coo_to_csr profiles a fresh
  conversion on inputs satisfying its documented preconditions.

### knn / knn_graph — Exact Neighbor Search

This operator performs exact candidate scanning and Top-K selection with
Ascend C SIMT. It exposes `ops_gnn.knn` and the composed Python API
`ops_gnn.knn_graph`.

#### Interfaces

```python
knn(x, y, k, batch_x=None, batch_y=None, cosine=False,
    num_workers=1, batch_size=None)
knn_graph(x, k, batch=None, loop=False, flow='source_to_target',
          cosine=False, num_workers=1, batch_size=None)
```

- `x[N,F]` and `y[M,F]` must share the NPU device, feature dimension and
  FP16/FP32 dtype. One-dimensional inputs become `[N,1]`. Noncontiguous
  inputs are made contiguous within the call.
- The result is an NPU int64 tensor `[2,M*k]`: query indices in row 0 and
  source indices in row 1. An input with `numel()==0` produces `[2,0]`.
- Batch vectors are sorted int64. `bucketize` constructs pointers for
  multiple batches; candidates never cross the query's batch boundary.
- Cosine distance executes on NPU. A zero-norm pair has distance 1.
- `knn_graph(loop=False)` requests `k+1` neighbors and removes self-loops.
  Both flow directions are supported. `num_workers` is accepted but ignored.

#### Boundaries and numerical semantics

Require `1 <= k <= N` within each query's batch. Fixed-capacity local
state handles `k<=64`; larger K uses dynamic Top-K state. The reference
GPU's `k<=100` limit is not imposed. Regression cases cover
65/100/101/128/129/257, `k=N`, and graph calls across dispatch boundaries.
Without self-loops, graph calls require `k+1<=N`. Batches must be sorted.
NaN/Inf ordering is not part of the validated contract.

L2 follows CPU torch_cluster's nanoflann L2_Adaptor: squared differences
are added left-associatively in groups of four features, then added to the
distance accumulator. Remaining features are processed sequentially.
FP16 explicitly rounds subtraction, multiplication and each addition to
FP16; it does not accumulate in FP32. FP32 preserves the same grouping
in source and uses the repository's default compiler settings.
L2 ties follow CPU tree-query order using metadata built on NPU. Cosine
uses FP32 dot products and norms, breaking ties by smaller source index.

Numerical compatibility targets genuine CPU torch_cluster 1.6.3. Other
references with sequential FP16 accumulation can select different neighbor
sets and are not substitutes for the grouped-four arithmetic above.

#### Implementation

| File | Responsibility |
| --- | --- |
| `csrc/npu/knn/op_host/knn.cpp` | Argument and batch validation, output allocation, current-stream launch |
| `csrc/npu/knn/op_kernel/arch35/knn_kernel.cpp` | Query partitioning, distances, ordered Top-K |
| `csrc/npu/knn/op_kernel/arch35/knn_tree.h` | NPU partitioning, leaf order and tie traversal; see [third-party sources](#knn-third-party-sources) |
| `python/ops_gnn/knn.py` (repository root) | Input preparation, bucketize, graph composition |

Each AIV runs 256 SIMT threads. A thread scans all candidates for each
assigned query. Small K dispatches capacities 1/4/8/16/32/64; F=3/16/32
with capacity at most 32 caches query coordinates. Large K allocates
M-by-K FP32 distance state and stores indices directly in output columns.
Queries own disjoint state slices. No extra M-by-N distance table is
materialized. Tree metadata costs O(N) for FP16 and O(N*F) for FP32 and
is included in full API timing.

One block builds each batch's tree. All 256 threads cooperate on the
first 7 FIFO nodes, increasing to 31 for batchN>=8192 and 63 for
batchN>=16384 when effective tree dimensions are at most 3. FP16 uses
only the first coordinate for tree construction. Remaining disjoint
subtrees are processed in parallel frontiers. An 8 KiB UB holds block
min/max reductions, counts and prefix offsets. FP32 reuses the selected
axis range; temporary swap positions reuse unallocated node storage and
are discarded before that storage is reused.

For `k=N`, Top-K state naturally reaches the full-table order of size.
There is no cross-call answer cache, CPU distance computation or
approximate search.

#### KNN Third-Party Sources

The repository-root [LICENSE](../../LICENSE) retains the original copyrights
and complete MIT and BSD licenses once, without duplicating license text in
translated documentation. The [CANN license](../../LICENSE) for new local
portions is unchanged; it does not replace the third-party terms.

##### Python interfaces

Argument preparation and graph composition in `python/ops_gnn/knn.py`
follow [`torch_cluster/knn.py` from pytorch_cluster 1.6.3](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/torch_cluster/knn.py)
by Matthias Fey (MIT). The adapted scope comprises both public APIs'
parameter names/defaults and the empty-input, shape/contiguity, batch-size
and bucketize preparation in `_knn_impl`. `_knn_graph_impl` retains k+1,
self-loop removal and flow semantics. Argument records, `Signature.bind`
and NPU PyBind dispatch are local adaptations.

No torch_cluster CPU/CUDA search source or binaries are bundled. Tests use
the independent PyTorch CPU reference in `test/knn/arch35/golden.py`;
neither testing nor the NPU runtime requires the torch_cluster package.

##### nanoflann rules

The tree partition, traversal-order and distance-arithmetic rules adapt
the [nanoflann header bundled in pytorch_cluster](https://github.com/rusty1s/pytorch_cluster/blob/e9a855c284b45edcbf0282cf70ac09bee0ce4e49/csrc/cpu/utils/nanoflann.hpp)
at commit `e9a855c284b45edcbf0282cf70ac09bee0ce4e49`
(`csrc/cpu/utils/nanoflann.hpp`, BSD, Jose Luis Blanco).

| Local file and functions | Upstream rules |
| --- | --- |
| `csrc/npu/knn/op_kernel/arch35/knn_tree.h`: `SelectSplit`, `SerialPartition`, `CooperativePartition` | `middleSplit_` and `planeSplit`: axis, thresholds and two-pointer permutation |
| `csrc/npu/knn/op_kernel/arch35/knn_tree.h`: tree construction and `Earlier` | `divideTree` and `searchLevel`: bounds, leaf order and traversal |
| `csrc/npu/knn/op_kernel/arch35/knn_kernel.cpp`: `L2Distance`, `SquaredDifference` | `L2_Adaptor::evalMetric`: grouped-four arithmetic |
| `test/knn/arch35/golden.py`: `_TieOrder` | Independently implemented CPU partition/traversal rules for equal-distance ordering |

These are NPU adaptations, not a copy of the complete nanoflann header.
NPU layouts, cooperative reductions/scans, kernel dispatch and streaming
Top-K are implemented here. Three production files contain adapted
interfaces/rules: `knn.py`, `knn_tree.h` and `knn_kernel.cpp`; this does not
attribute the whole project to a third party. SCA similarity percentages
do not define the adaptation scope. MIT and BSD cover different sources
and are not interchangeable. Attribution remains in each affected source.

#### Build and test

Validated: Ascend950PR (`dav-3510`), CANN 9.1.0, PyTorch 2.7.1,
torch_npu 2.7.1.post8 and Python 3.12.13. Other platforms are not claimed
as validated. From the repository root in the configured environment:

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S . -B build/cmake_python -DWITH_PYTHON=ON \
      -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH=dav-3510
cmake --build build/cmake_python -j 4
export PYTHONPATH="$PWD/python"
NPU_DEVICE_ID=0 python -m pytest test/knn/arch35 -v --maxfail=1
python test/knn/arch35/benchmark_knn.py --warmup 20 --iters 100 \
       --json-output /path/to/new-run/benchmark.json
```

The test directory contains functional tests, API contracts, the independent
CPU reference, CPU-only reference self-tests and a performance script.
`golden.py` exhaustively evaluates every candidate in blocks of at most 32
queries, preserving grouped-four FP16/FP32 arithmetic and L2 tree tie order;
tree metadata only determines equal-distance ordering, never search pruning.
It imports neither `ops_gnn`, `torch_npu` nor `torch_cluster`. Run the reference
self-tests are included in the single `test_knn.py` test entry point. Without
an NPU or its extensions, select only these tests and bypass the shared
device-architecture conftest:

```bash
python -m pytest test/knn/arch35/test_knn.py \
       --confcutdir=test/knn/arch35 -k test_golden_ -q
```

Functional tests still compare complete neighbor-edge multisets; cosine uses
a CPU double reference. The benchmark covers 32 shapes in both dtypes,
requiring `reference_ms / measured_ms >= 0.45` for each case. `--quick`
is diagnostic only. Store reports outside the source tree and attach
validation results to the PR rather than committing them as source files.

#### Example

```python
import torch
import torch_npu
from ops_gnn import knn, knn_graph

torch.npu.set_device(0)
x = torch.randn(128, 3, device="npu", dtype=torch.float32)
y = torch.randn(32, 3, device="npu", dtype=torch.float32)
edge = knn(x, y, k=8)           # [2, 256]: y indices, x indices
graph = knn_graph(x, k=8)       # No self-loops, source_to_target
torch.npu.synchronize()
```

## Testing Guide

### Running Tests

```sh
# Run all tests for the local chip model (950→arch35, A2/A3→arch22; `pytest test/` only collects the matching directory automatically)
pytest test/ -v

# Run single operator test (arch35 on 950, arch22 on A2/A3)
pytest test/spmm_max/arch22/test_spmm_max.py -v
pytest test/spmm_min/arch22/test_spmm_min.py -v
pytest test/spmm_sum/arch22/test_spmm_sum.py -v
pytest test/unified_spmm/arch22/test_unified_spmm.py -v
pytest test/gather_csr/arch35/test_gather_csr.py -v
pytest test/segment_max_csr/arch35/test_segment_max_csr.py -v
pytest test/graclus_cluster/arch35/test_graclus_cluster.py -v
python -m pytest test/gather_coo/arch35/test_gather_coo.py -v
pytest test/random_walk/arch35/test_random_walk.py -v
pytest test/radius/arch35/test_radius.py -v
pytest test/sparse/arch35/test_sparse.py -v

# Run the random_walk performance benchmark
NPU_DEVICE_ID=<device_id> python test/random_walk/arch35/benchmark_random_walk.py

# Run SpMM performance benchmarks
NPU_DEVICE_ID=<device_id> python test/spmm_max/arch22/benchmark_spmm_max.py
NPU_DEVICE_ID=<device_id> python test/spmm_min/arch22/benchmark_spmm_min.py
NPU_DEVICE_ID=<device_id> python test/spmm_sum/arch22/benchmark_spmm_sum.py
# 3D BSpMM example; the max/min scripts also accept --batches
NPU_DEVICE_ID=<device_id> python test/spmm_sum/arch22/benchmark_spmm_sum.py --batches 4
NPU_DEVICE_ID=<device_id> python test/unified_spmm/arch22/benchmark_unified_spmm.py
# Higher-rank binary broadcast: [N,3,1] and [E,2,3,4] reduce to [M,2,3,4]
NPU_DEVICE_ID=<device_id> python test/unified_spmm/arch22/benchmark_unified_spmm.py \
  --op mul --reduce mean --lhs-shape 3,1 --rhs-shape 2,3,4

# Run the radius official-baseline performance benchmark
python test/radius/arch35/benchmark_radius.py

# Run single test case
pytest test/segment_max_csr/arch35/test_segment_max_csr.py::test_segment_max_csr_basic -v
```

## Test Writing Template

```python
import pytest
import torch
import ops_gnn

def test_my_operator():
    """Test basic functionality"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)          # 1. Set NPU device
    torch.manual_seed(42)            # 2. Set random seed

    # 3. Create NPU Tensor
    src = torch.tensor([...], dtype=torch.float32, device=device)

    # 4. Call operator
    result = ops_gnn.my_op(src)

    # 5. Verify: device, shape, values
    assert result.device.type == 'npu'
    assert result.shape == expected_shape
    assert torch.allclose(result, expected)
```

---

### segment_coo — Sorted COO segment reductions

`segment_coo(src, index, out=None, dim_size=None, reduce="sum")` returns values.
The sum/add/mean-specific functions use the same arguments without reduce;
`segment_min_coo` and `segment_max_coo` return `(values, indices)`.

The int64 index must be sorted, nonnegative and in range. Reduction uses
`index.dim()-1`, with broadcasting across leading dimensions. Source rank is
1–8; supported dtypes are float16/float32/bfloat16/int8/uint8/int32/int64.
Strided inputs and outputs are supported. Alignment temporaries are copied
back to the caller's original out. An out buffer determines the segment count;
otherwise dim_size or the maximum index plus one is used. mul is unsupported.

A2 forward semantics: all reductions use supplied initial values. Sum/add
accumulate them, mean divides the accumulated numerator, and min/max update
only on strict improvement. Each batch's first nonempty segment uses out[0],
even when its label is nonzero; subsequent segments use their own seeds.
Empty segments and empty inputs preserve supplied out. Ordinary extrema ties
retain the first position. Each feature's argument state starts at zero and
carries across nonempty segments and batches; empty segments emit the reduction
length sentinel without changing that state. Without out, extrema start at
finite dtype identities and results still equal to the identity become zero.
Source NaNs do not update extrema; all-NaN nonempty segments retain prior arg
state. INT8/UINT8 mean explicitly returns zero when a nonempty count narrows to
zero, without executing device division by zero. These are forward rules only;
no backward implementation or gradient guarantee for special arg states is provided.

A2 also broadcasts a length-one index along the reduction dimension. Noncontiguous
out values/args are checked against a contiguous CPU copy with the same seed;
NPU writeback preserves the original out object and storage outside the view.
This does not assert identical behavior to the original noncontiguous CPU out API.

Ascend 950 retains its existing overwrite sum/add, last-tie and mathematical
byte-mean zero-divisor conventions. This A2 change does not alter that backend.

## Back to Main

- **[Back to README](../../README_en.md)**
- **[Development Guide](development_guide.md)**

## torch_sparse utility interfaces

### non_diag_mask

```text
non_diag_mask(row, col, M, N, k) -> Tensor
```

`row` and `col` are equal-length one-dimensional int64 COO indices on the same NPU. M and N are nonnegative dimensions; k is a signed diagonal offset. Noncontiguous inputs are accepted and copied to contiguous storage on their original device.

Every entry must satisfy `0 <= row[i] < M` and `0 <= col[i] < N`. If either dimension is zero, the input must have no entries. Indices need not be sorted and duplicates are permitted. The Host checks metadata immediately; index contents are checked on the NPU, so an out-of-range entry raises an asynchronous device error at a synchronization point such as `torch.npu.synchronize()`. A returned Tensor before synchronization is not evidence that invalid input succeeded. Do not consume results or reuse the failing context for measurements.

The bool output has length `E + num_diag`. For `k >= 0`, `num_diag = max(0, min(M, N-k))`; otherwise it is `max(0, min(M+k, N))`. The first E entries are `col-row != k`; the remaining `num_diag` entries are False.

This public layout follows the task contract. The NPU adapters for `SparseTensor.set_diag/fill_diag` separately sort COO entries and rebuild CSR caches; they do not use this layout as upstream sorted-insertion scatter indices. CPU/CUDA frontend implementations remain unchanged.

### Optional SparseTensor replacement values

For a weighted input, `values` may be None, a Python numeric scalar, or a Tensor broadcastable to `[num_diag, *value.shape[1:]]`. Tensor values must share the stored value's exact device, including its device index. Multi-element and empty Tensors must also have the same dtype; a one-element Tensor retains PyTorch scalar conversion, including zero-dimensional and `[1,1]` shapes. Unsupported conversions and incompatible broadcasting raise framework errors. Type, device and dtype checks run before the native mask call, including an empty diagonal.

For an unweighted input, supplied values are ignored and the output remains unweighted. None fills weighted diagonal features with ones; this is the existing NPU feature-valued behavior and is not a claim of equivalence to every upstream broadcasting corner case. The tensor computation is independently testable without installing the optional SparseTensor package.


### torch_sparse_random_walk

```text
torch_sparse_random_walk(rowptr, col, start, walk_length) -> Tensor
torch_sparse_random_walk_with_rand(rowptr, col, start, walk_length, rand) -> Tensor
```

The first three arguments are one-dimensional int64 tensors on the same NPU: CSR pointers, column indices and starting nodes. `walk_length` is nonnegative. The int64 output has shape `[S, walk_length+1]`, with starting nodes in its first column. Each step selects a neighbor uniformly. A valid zero-degree node terminates the walk and leaves the remaining entries at -1. Invalid CSR content, indices or starting nodes raise errors rather than being treated as valid dead ends.

The public operation obtains state from the NPU generator and runs Philox on the device. `torch_sparse_random_walk_with_rand` takes a same-device float32 `[S, walk_length]` tensor for comparisons with shared random inputs. Values must be finite and in `[0, 1)`. If float32 multiplication rounds a sampled index up to degree, the index is limited to degree-1.

Equal seeds do not imply equal random sequences across backends. Single-neighbor graphs and paths using shared rand tensors support elementwise comparisons. Cross-backend equality for the public multi-neighbor operation requires matching both the RNG algorithm and its consumption order.


### Implementation and errors

- Host code validates dtype, dimensions, devices, sizes and output-length overflow. Kernels incorporate content bounds checks. Outputs and contiguous copies stay on the NPU.
- Kernels use the caller's current NPU stream. Metadata errors are immediate; device content errors surface at synchronization. Run negative cases in separate processes with explicit synchronization and do not reuse their device context for timing.
- Kernels reside in `csrc/npu/non_diag_mask` and `csrc/npu/torch_sparse_random_walk`, avoiding collisions with the existing random_walk implementation and build objects.
- Public Host/Kernel C++ interfaces use `opsgnn`. The utility walk parameters are named `TorchSparseRandomWalkLaunchParams`, distinct from the existing node2vec operator's parameter type. Dispatcher namespaces remain independent of C++ namespaces.
- The compatibility frontend resides in `python/ops_gnn/torch_sparse_utils.py`. Dispatcher registration retains the `torch_sparse` and auxiliary `ops_gnn` interfaces without CPU compute fallback.
- Kernels allocate no algorithm workspace. Random inputs, outputs, contiguous copies and SparseTensor frontend temporaries still consume memory; the complete call is not a zero-extra-memory operation.

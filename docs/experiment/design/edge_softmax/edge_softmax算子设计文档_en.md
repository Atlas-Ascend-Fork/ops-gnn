# edge_softmax Design Document (A2/A3 arch22, 950 arch35)

> Status: implemented. Functional, numerical and performance measurements live in
> the test report of the corresponding PR; this document covers the interface,
> the semantics, the reuse boundary and the blocking scheme.
> Dependencies: `scatter_softmax`, `scatter_log_softmax`, `scatter_logsumexp` and
> `scatter_std` are merged, and this operator reuses their host helpers and their
> device pipeline; the CSR segment semantics follow `segment_logsumexp`.

## 1. Requirements and acceptance criteria

Implement the NPU version of `dgl.nn.functional.edge_softmax`:

```python
edge_softmax(src, indptr)  # Tensor
```

Every row of the CSR pointer describes the incoming edges of one destination
node, and `src` has `indptr[-1]` rows along axis 0. For every edge `e` of segment
`i`:

```
out[e] = exp(src[e] - max_i) / sum_{j in segment i} exp(src[j] - max_i)
```

Every trailing feature (`src.shape[1:]`) is normalized independently.

| Item | Requirement |
| --- | --- |
| Interface | Aligned with `dgl.nn.functional.edge_softmax`: the CSR pointer describes the groups; no `dim` / `dim_size` / `out=` |
| Output | Shape, dtype and device of `src` (like DGL, not a reduced result) |
| dtypes | `float16` / `float32` / `bfloat16`; maximum, exponential and sum run in fp32 |
| indptr | 1-D `int32` / `int64` on the same device, length `segments + 1`; it must start at 0, be nondecreasing and end at `src.size(0)` |
| Autograd | `grad_src[e] = out[e] * (grad_out[e] - sum_{j in segment(e)} out[j] * grad_out[j])` |
| Platform | arch22 (A2/A3) and arch35 (950) share one kernel source; the two arch directories only hold thin compilation units |
| Performance | Net gain over `ptr2ind + scatter_softmax`: one `E`-long index materialization and read-back less |

## 2. Host dimensions and output contract

The batch / group / feature decomposition of `scatter_softmax` is reused with
`dim` fixed to 0 and the group count taken from the pointer length:

- `batch  = 1` (a CSR graph only segments axis 0)
- `dimLength = src.size(0) = indptr[-1]`
- `groups = indptr.numel() - 1`
- `after  = prod(src.shape[1:])`, aligned to 8 elements as `afterPadded`
- input rows `rows = dimLength`, reduced output rows `outRows = groups`

The tiling is therefore `grouped_softmax::BuildTiling(src, 0, segments)`: the
workspace stays `workspaceElems = groups * afterPadded` fp32 values, the same
order as the `batch * groups * afterPadded` workspace of `scatter_softmax`.

The host adds `csrc/npu/edge_softmax/common/edge_softmax_shared.h` with two
entry points:

```cpp
torch::Tensor edge_softmax_forward(src, indptr);
torch::Tensor edge_softmax_backward(out, grad_out, indptr);
```

- `ValidateCsr` only checks what the device really needs: `src` contiguous, at
  least 1-D and of a supported dtype; `indptr` contiguous, 1-D, int32, on the
  same device and with at least two boundaries (that is, at least one segment);
  and that the segment count and the row count fit int32 (one int32 label per
  row, one int32 cursor per segment). Fewer than two boundaries describe no
  segment at all, and the empty branch below returns an uninitialised tensor of
  `src`'s shape, so it is rejected here: a direct C++ caller gets a clear error
  instead of the garbage that tensor would hold.
- The pointer values (not starting at 0, decreasing, or ending away from `E`) are
  validated on the Python side with a single reducing synchronisation; on failure
  the pointer is copied back once to name the first offending boundary in a
  `ValueError`. The check is kept out of the C++ host entry so that a C++ caller
  does not pay a device synchronisation per call.
- The kernel is memory safe for any int32 pointer: the cursor stops at
  `groupCount - 1`, so an inconsistent pointer only assigns trailing rows to the
  last segment and never reads or writes outside the pointer array or the
  workspace.
- `workspaceElems == 0`, `rows == 0` or `after == 0` returns the `empty` result
  without launching a kernel; given at least one segment, each of them means the
  result itself is empty. `maximum` is pre-filled with
  `SCATTER_SOFTMAX_NEUTRAL_MAXIMUM` and `sum` with zeros.

## 3. Device algorithm

No new kernel is written: the shared pipeline of `scatter_softmax` gains the
compile-time mode `kModeEdgeCsr`, and the operator macro
`SCATTER_GROUPED_SOFTMAX_OPERATOR(EdgeSoftmax, EdgeLaunch, kModeEdgeCsr)` expands
the five kernels and the two exported entry points.

### 3.1 Forward: three full passes

| Pass | Reads | Writes | Notes |
| --- | --- | --- | --- |
| Maximum | `src` | `maximum` | segment maximum, fp32 atomic max (or a per-core UB accumulator) |
| Exp | `src`, `maximum` | `out`, `sum` | `exp(src - max)` stored in the source dtype and accumulated per segment |
| Normalize | `out`, `sum` | `out` | divide by the segment sum |

### 3.2 Backward: two full passes

| Pass | Reads | Writes | Notes |
| --- | --- | --- | --- |
| GradSum | `out`, `grad_out` | `sum` | segment sum of `out * grad_out` |
| Grad | `out`, `grad_out`, `sum` | `grad_src` | `out * (grad_out - sum)` |

### 3.3 Where the group label comes from: the CSR cursor

`scatter_softmax` loads `rowBatch` group labels with one `DataCopyPad` per row
batch. In the CSR mode the label is derived from the pointer, so
`LoadGroups<kModeEdgeCsr>` instead:

- walks the rows in order, where the segment index never decreases, so one cursor
  plus the next boundary `csrNext` advances the label: crossing a segment costs
  one `indptr` read, not one label read per row;
- locates the first row batch of a core with a binary search over `indptr`
  (`O(log segments)` scalar reads) and then keeps advancing the cursor;
- steps over empty segments (`indptr[i] == indptr[i + 1])`, so every row lands in
  the segment that contains it.

The cursor state (`csrSegment` / `csrNext` / `csrColumn` / `csrStarted`) lives on
`PhaseTensors` and is reset per kernel launch, so it is never shared across
passes or cores.

### 3.4 Reuse boundary

The shared headers change in three places, all compile-time switches that leave
the merged operators untouched:

| File | Change |
| --- | --- |
| `scatter_softmax_kernel_common.h` | adds `kModeEdgeCsr`; turns `IsLogDomain` from "not softmax" into an explicit log-domain predicate and adds `IsSoftmaxDomain` (softmax / edge CSR) |
| `scatter_softmax_kernel_phase.h` / `_io.h` | `kStore`, the log normalization and `LoadWorkspace` use the two predicates; `LoadGroups` gains the CSR branch |
| `scatter_softmax_kernel_entry.h` | binds `indexGm` with `groupCount + 1` elements in `kModeEdgeCsr` (`dimLength` otherwise) |

`scatter_std` instantiates `kModeStd` and `scatter_logsumexp` instantiates
`kModeLogSumExp`, so neither compiles the `kModeEdgeCsr` branch and their
behaviour is unchanged.

## 4. Blocking, alignment and pipeline

Row batching, tiling and the pipeline follow `scatter_softmax`: a core owns a
contiguous row chunk, buffers up to 64 rows at a time (sized from the UB budget)
and walks the feature axis in 32-byte aligned `colTile` tiles, synchronizing the
three pipeline stages once per batch. Compared with the label mode the group
label is a scalar loop instead of one `DataCopyPad`:

- for wide features (`after >= colTile`) the vector work dominates and the label
  cost is invisible;
- for narrow features (e.g. `after = 8` in GAT) a batch costs 64 scalar writes,
  the same order as the per-row vector instructions, so a separate copy is not
  worth it;
- if this ever needs to shrink further, the row-to-segment mapping can become a
  vector compare plus `Select`, but that trades an O(rows) scalar loop for an
  O(rows) vector instruction and only pays off for small `after`.

## 5. Numerical semantics

- An empty segment owns no edge and produces no output row; `indptr` all zeros
  (`E = 0`) yields an empty tensor.
- A one-edge segment produces `1` (`exp(0) / 1`).
- A segment holding `+inf`: the maximum is `+inf`, `exp(+inf - +inf)` is `NaN`,
  and that feature column of every element of the segment is `NaN`. A segment
  holding `NaN` is `NaN` as well. Both match `torch.softmax` on the same input.
- A segment holding only `-inf`: the maximum is `-inf`,
  `exp(-inf - -inf)` is `NaN`, matching `torch.softmax`.
- The maximum and the segment sum accumulate in fp32, so the length of an
  fp16/bf16 segment is not limited by 65504.
- The backward reads the saved forward output (source dtype) as the probability
  and rounds `grad_src` to the source dtype.

## 6. Differences from DGL and from the composed form

| Item | Note |
| --- | --- |
| `dim` | DGL's `edge_softmax` only segments axis 0 of a CSR graph; this operator is the same and exposes no `dim` |
| Pointer dtype | Both DGL backends accept int64; this operator accepts int32/int64 and narrows to int32 in Python (an explicit error when `E` or the segment count exceeds the int32 range) |
| Pointer validation | DGL does not validate; this operator checks the 0 start, the monotonicity and the last boundary, and raises `ValueError` |
| Composed form | `ptr2ind(indptr, E) + scatter_softmax` is semantically equivalent and is used as the cross-check in the tests; the operator saves the `E`-long int32 index write and read-back |
| Empty segments | Same as DGL: no output |

## 7. Testing and performance

### 7.1 Case layout

The `test/<operator>/arch22|arch35` layout is kept, with only `golden.py`,
`test_edge_softmax.py` and `benchmark_edge_softmax.py` per directory:

- `golden.py` expands `indptr` into one segment label per edge and reduces it with
  a float64 `torch.softmax` reference; it does not import `ops_gnn`;
- `test_edge_softmax.py` covers the shape/dtype matrix (1-D, ND, narrow and wide
  features, empty segments, one-edge segments, a long segment, thousands of small
  segments), forward and backward comparison, int64 pointers, non-contiguous
  input, `±inf`/`NaN`, agreement with the `ptr2ind + scatter_softmax` composition
  and ten invalid-argument cases;
- `benchmark_edge_softmax.py` compares the forward and the forward+backward step
  against `ptr2ind + scatter_softmax` on one uniform-degree graph
  (`nodes x degree`).

Both architectures run the same cases and the same reference (arch35 reuses the
arch22 implementation by path), so only the measured numbers are per platform.
Nothing is added to `test/common`.

### 7.2 Performance protocol

- Target shape: `nodes = 16384`, `degree = 64` (about 1M edges),
  `features = 128`, in fp16 and fp32; plus a degree sweep (8 / 32 / 64 / 128) to
  cover long and short segments;
- Baseline: `ptr2ind + scatter_softmax` on the same NPU, the same `src` and the
  same graph;
- Timing: `torch.npu.Event`, 10 warm-up iterations and 50 timed iterations,
  averaged; every measurement is repeated to confirm the spread;
- The report separates A2 and A5 and never mixes their statistics.

## 8. Steps and risks

### 8.1 Steps

1. Add `kModeEdgeCsr` and the CSR label branch to the shared headers without
   changing the merged operators;
2. Add `csrc/npu/edge_softmax/{common,op_host,op_kernel/{arch22,arch35,common}}`
   and bind `edge_softmax_forward/backward`;
3. Add `python/ops_gnn/edge_softmax.py`: argument checks, CSR pointer checks and
   autograd;
4. Add the tests and the documentation (`README`, `docs/zh|en/api_reference.md`,
   this document);
5. Build, test and benchmark on A5 (950) first, then repeat on A2 (arch22), and
   report the two platforms separately;
6. Open the PR against the issue and wait for CI and review.

### 8.2 Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| The shared change affects merged operators | The new mode only takes `if constexpr` branches; `IsLogDomain` is made explicit and every instantiation of `scatter_softmax` / `scatter_log_softmax` / `scatter_logsumexp` / `scatter_std` is re-checked against the full test suite |
| An invalid pointer reads out of bounds | The cursor is capped at the last segment, so the kernel never depends on pointer validity; the Python layer validates the semantics and names the offending boundary |
| A very long segment (hundreds of thousands of edges) | The reduction runs in fp32 and does not tile by segment length, so a long segment cannot collapse; a 70000-edge segment is covered |
| Many segments with a small `after` | The workspace is `segments x afterPadded`, the same order as `scatter_softmax`; row batching still follows the UB budget, and 64 scalar labels per batch are acceptable for narrow features |
| Mixing the two platforms | The report is split per platform and the benchmark script has no platform default |

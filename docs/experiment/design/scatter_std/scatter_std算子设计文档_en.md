# scatter_std Design Document (A2/A3 arch22, 950 arch35)

> Status: implemented. The functional, accuracy and performance measurements live
> in the test report of the matching PR; this file documents the interface, the
> semantics, the reuse boundary and the tiling.
> Dependencies: `scatter_softmax` (#165), `scatter_log_softmax` (#186) and
> `scatter_logsumexp` (#252) are merged; this operator reuses their host
> scaffolding and device-side pipeline.

## 1. Requirement and acceptance

Provide the NPU version of `torch_scatter.scatter_std`:

```python
scatter_std(src, index, dim=0, dim_size=None, unbiased=True)  # Tensor
```

For every group `g`:

```
mean_g = (1 / n) * sum_{i : index[i] = g} src[..., i, ...]
out[..., g, ...] = sqrt( sum_{i : index[i] = g} (src[..., i, ...] - mean_g)^2 / (n - 1) )
```

`unbiased=False` replaces the denominator by `n`. Acceptance:

| Item | Requirement |
| --- | --- |
| Interface | Matches `torch_scatter.scatter_std`; `dim` defaults to 0 like the in-tree `scatter_softmax`; no `out=` argument; `unbiased` defaults to `True` |
| Output | `src.shape` with `dim` replaced by `dim_size`; dtype and device follow `src` |
| dtype | `float16` / `float32` / `bfloat16`; the reduction always runs in fp32 |
| index | 1-D `int32` / `int64` on the same device, length = `src.size(dim)`, values in `[0, dim_size)` |
| Autograd | `grad_src[i] = grad_out[index[i]] * (src[i] - mean[index[i]]) / ((n_g - c) * out[index[i]])` with `c = unbiased ? 1 : 0` |
| Platforms | arch22 (A2/A3) and arch35 (950) share one kernel source; the architecture directories only hold thin translation units |
| Performance | Net gain over the composed `argsort + segment_sum_csr × 3 + sqrt` baseline; correctness first, tuning from measurements |

## 2. Host dimensions and output contract

The batch/group/feature decomposition of `scatter_softmax` is kept:

- `batch  = prod(src.shape[:dim])`
- `groups = dim_size`
- `after  = prod(src.shape[dim+1:])`, padded to 8 elements as `afterPadded`
- input rows `rows = batch * dimLength`, output rows `outRows = batch * groups`

The output shape comes from `grouped_softmax::ReducedSizes` (the same helper
`scatter_logsumexp` uses). The workspace is still `workspaceElems = batch *
groups * afterPadded` float32 values, plus the member-count workspace
`countElems = align_up(groups, 8)` (one fp32 counter per group, padded to whole
32-byte blocks, see §3.1).

The host side adds `csrc/npu/scatter_std/common/grouped_std_shared.h` with two
entry points:

```cpp
std::vector<torch::Tensor> scatter_std_forward(src, index, dim, dim_size, unbiased);
torch::Tensor scatter_std_backward(src, out, grad_out, mean, count, index, dim, dim_size, unbiased);
```

- Forward reuses `ValidateInputs` / `BuildTiling` / `CheckLaunchable` /
  `ResolveDType` from
  `csrc/npu/scatter_softmax/common/grouped_softmax_shared.h`. `BuildTiling` gets a
  `standardDeviation` flag that reserves the count workspace and stores the
  output row count in the tiling. Only the output allocation differs - it
  replaces `dim` by `dim_size`.
- Forward returns `{out, mean, count}`. `mean` and `count` are saved by autograd
  and read directly by the backward, so neither the mean nor the member counts
  are recomputed. The mean reuses the group-sum workspace (the finalize pass
  overwrites it in place), so the forward allocates only one extra full
  workspace (`deviation`) plus the counts.
- A zero-element output (`after == 0` or `groups == 0`) returns immediately. When
  `rows == 0` but the output is not empty (groups without members) the output is
  filled with `NaN` and no kernel is launched.
- Backward restores `src.sizes()`; `grad_out` has the forward output shape. It
  reuses `ValidateInputs` (now `src.size(dim) == dim_size`, so the index length
  check still holds) and then checks contiguous / shape / dtype / same-device
  for `out`, `grad_out`, `mean` and `count`, and that the workspace lengths match
  the re-derived tiling, so a workspace from another shape cannot be reused.
- `dim_size == 0` makes the backward return `zeros_like(src)` directly: the
  output holds no element, so the gradient is identically zero.

## 3. Device-side algorithm

### 3.1 Forward: two full passes plus one finalize pass

| Pass | Reads | Writes | Notes |
| --- | --- | --- | --- |
| P1 group sum | `src` (N×C) | `mean` workspace (G×C fp32), `count` (G×8 fp32) | `mean += src` (a sum for now) and counts the members; fp32 accumulation |
| P2 squared deviation | `src` + `mean` (now `sum / n`) | `deviation` workspace (G×C fp32) | `deviation += (src - mean)^2` |
| P3 finalize | `mean` + `deviation` + `count` | `mean` (becomes the mean in place), `out` (G×C) | `out = sqrt(deviation / (n - correction))`, only `dim_size` rows |

Key points:

- **Member counts.** P1 adds 1 to the count of the row's group once per row,
  only when `col == 0` and `base < dimLength`, so the feature axis does not
  count twice. The workspace holds one fp32 counter per group (dense, padded to
  whole 32-byte blocks). While it fits the UB budget (64 KB, i.e. 16384 groups)
  a core accumulates in UB and flushes with one atomic per group; otherwise it
  falls back to one 32-byte atomic per row that targets the block holding the
  group, with the group's lane set to one and the rest zero. The finalize pass
  reads the member count as `n = countGm.GetValue(g)`.
- **One source read per pass.** P1 and P2 both walk the full input; P2 reads the
  mean P1 wrote back. The finalize pass walks only `batch * groups` rows, does
  not read `index`, and is therefore independent of `batch * dimLength`.
- **Centered deviation.** The variance is accumulated as "mean first, then
  `(x - mean)^2`" rather than expanded into `sum(x^2) - n * mean^2`. The
  expansion subtracts two float32 numbers close to a million to recover a
  result of order `1e-4` on data like `1000 ± 0.01` and cancels every
  significant digit; the centered form costs one extra read of the source and
  keeps the relative error in the range `torch.std` reports.
- **Saved mean.** The `mean` workspace first holds the group sums. P3 derives the
  mean from the sum and `n` and writes it back to the same slot, so the backward
  needs neither a fourth workspace nor a second mean computation. An empty group
  has no mean, so P3 writes `NaN` into its whole row - the same rule `out`
  follows; the backward only reads the groups `index` actually selects and never
  touches an empty row.
- P3 row count, per-core row chunk and row batch come from `outRows` /
  `outRowChunk` / `outBlockNum` / `outRowBatch` in the tiling - the same fields
  `scatter_logsumexp` uses.

### 3.2 Backward: one elementwise pass

```
grad_src[i] = grad_out[index[i]] * (src[i] - mean[index[i]]) / ((n_g - c) * out[index[i]])
```

- One full read/write: read `src`, the group rows of `grad_out` / `out` / `mean`
  broadcast through `index`, and the per-group count; write `grad_src`.
- Three group tensors must be read, so the two fp32 buffers of `GroupsFitUb` do
  not hold them all: the std backward reads every group row from GM and does
  **not** enable the "keep small group tensors in UB" shortcut. This is a known
  optimisation point, recorded in §8.2.
- The normalisation factor `(n_g - c) * out[g]` is computed once per row; the
  element work is one subtract, one divide and one multiply.
- **A group without spread has a gradient of 0.** `torch.std` zeroes the
  variance gradient where `out == 0` before propagating it, so a group whose
  members are all equal reports 0 instead of the `0 / 0` the quotient alone
  would give. The backward applies the same rule with one `CompareScalar`
  (`out == 0`) and one `Select` (zero where the mask holds, the quotient
  elsewhere) per tile. The comparison is `== 0` rather than `!= 0` on purpose:
  the `NaN` of an empty group, of the single member of an unbiased deviation or
  of a group holding `±inf` compares unordered with 0 and must keep the `NaN`,
  which only the `== 0` form guarantees. The mask reuses the source buffer of
  the already-widened group output row, and the row buffer ends with one extra
  256-byte repeat for the rounded-up count of the mask instruction.
- `out[g] == NaN` (empty group, single member of an unbiased deviation, group
  holding `±inf`) leaves the mask false and keeps the `NaN` the division
  produced, matching `torch.std`.
- The backward reduces nothing: no atomics, no second row schedule.

### 3.3 Reuse boundary

| Reuse | Content |
| --- | --- |
| Direct | `ValidateInputs` / `BuildTiling` / `CheckLaunchable` / `ResolveDType` / `ReducedSizes`; `LoadGroups`, `CopyRowsIn/Out`, `CopyGroupRowsIn`, `CopyWorkspaceIn/Out`, `AccumulateRows`, `FlushAccumulator`; the `Sum` / `Finalize` / `GradElement` driver and the three-stage pipeline |
| Compile-time extension | The mode enum in `scatter_softmax_kernel_common.h` gains `Std`; `scatter_softmax_kernel_phase.h` gains the `CountRows` / `DeviationBatch` / `FlushCounts` batches and the std branches of `Finalize` / `GradElement`; the reduction body is not copied |
| Shared tiling | `scatter_softmax_tiling.h` gains `countElems` and `correction`, and `BuildTiling` gains the `standardDeviation` argument. With the flag false both fields stay zero, so the tiling byte layout of the existing operators is unchanged |
| New | The std branches of `Finalize` / `GradElement`, the member-count path, the `scatter_std` launch / tiling instantiation and the two architecture translation units |
| Not reused | The softmax normalisation batch (writes the `src` shape) and the max/exponential passes; std needs no maximum |

The row schedule, the UB/workspace budget and the reduction bookkeeping therefore
exist once for four operators, and CodeCheck does not see the same reduction
twice.

## 4. Tiling, alignment and pipeline

The `scatter_softmax` tiling constants and budgets are kept:

- `SCATTER_SOFTMAX_WORKSPACE_ALIGN = 8` (32 bytes), `ROW_ALIGN = 16` (32 bytes for
  fp16/bf16, 64 for fp32);
- `rowBatch` is derived from `ROW_BUDGET_BYTES = 80 KB` and `colTile × 20` bytes
  per row, capped at 64 rows (a larger batch means fewer synchronization
  round-trips: 8 -> 64 took the scalar wait of P1 from 1.78 ms to 0.13 ms);
- the resident UB workspace is capped at `ACC_BUDGET_BYTES = 16 KB`; beyond that
  every row performs one atomic write;
- **Two count paths.** When `countElems * 4 <= SCATTER_SOFTMAX_COUNT_BUDGET_BYTES`
  (64 KB, i.e. `groups <= 16384`) every core accumulates its own members in UB
  and flushes once. Beyond that the counts fall back to one 32-byte atomic per
  row, targeting the block that holds the group with that lane set to one;
- each core owns a run of consecutive rows; MTE2 / V / MTE3 are pipelined inside
  a batch, and P3 and the backward reuse the same driver.

P3 walks `batch * groups` rows, usually far fewer than the input rows; a wide
`after` is tiled along the feature axis by `colTile` exactly like P1/P2.

## 5. Numeric semantics

| Input | `out` | Reason |
| --- | --- | --- |
| Empty group (`dim_size` beyond the real groups) | `NaN` | empty `torch.std` reduction; `n - c <= 0` writes `NaN` directly |
| Single-member group, `unbiased=True` | `NaN` | denominator `n - 1 = 0` |
| Single-member group, `unbiased=False` | `0` | zero deviation, denominator 1 |
| Group holding `±inf` | `NaN` | the mean is polluted, `x - mean` hits `inf - inf` |
| Group holding `NaN` | `NaN` | propagation |
| All members equal | `0` | zero deviation |
| Small group with a large offset (`1000 ± 0.01`) | accurate | centered two-pass deviation, no million-scale cancellation |
| Long low-precision group | no collapse | fp32 reduction; the member count is an integer, free of fp32 rounding |

Backward:

| `out[g]` | Element | `grad_src` |
| --- | --- | --- |
| `NaN` (empty / single / `±inf`) | any | `NaN` |
| `0` (all equal) | equal to the mean | `0` (`torch.std` zeroes the variance gradient first) |
| finite | any | `grad_out[g] * (x - mean) / ((n - c) * out[g])` |

Every row above is pinned against `torch.std` on CPU in fp32/fp64 and compared
element by element by the tests.

## 6. Differences from `torch_scatter`

| Case | `torch_scatter.scatter_std` | This implementation |
| --- | --- | --- |
| Empty group | `NaN` | `NaN` (same) |
| Single member, `unbiased=True` | `NaN` | `NaN` (same) |
| Group with `±inf` | decided by the composed implementation, usually `NaN` | `NaN`, matching `torch.std` |
| `dim` default | `-1` | `0`, like the in-tree `scatter_softmax` |
| `out=` argument | supported | not supported, like the in-tree `scatter_softmax` |

The last two are in-tree interface consistency choices; the numeric semantics
follow `torch.std` and are documented explicitly in the API reference.

## 7. Tests and performance

### 7.1 Case layout (the directory structure of #186)

| Location | Content |
| --- | --- |
| `test/common/grouped_softmax_golden.py` | Shared float64 references; adds `grouped_std_reference` (`dim` / `dim_size` / `unbiased` / empty groups) and `std_gradient` built on it |
| `test/common/grouped_softmax_cases.py` | Shared shapes, dtypes and invalid-input table (unchanged) |
| `test/common/test_grouped_softmax_suite.py` | Adds `StdSuite`: inherits the argument, gradient and invalid-input cases of `GroupedFamilySuite` and adds both normalisations, empty groups, single-member groups, the large-offset accuracy case and the backward boundaries |
| `test/scatter_std/arch22/` and `arch35/` | Only `golden.py`, `test_scatter_std.py`, `benchmark_scatter_std.py`; binding only |

No new file is added to `test/common`; the reference, the suite and the benchmark
join the existing family modules, so the file count of `test/common` stays what
it was when `scatter_logsumexp` merged.

Coverage:

- shape × dtype matrix (1-D features, regular groups, empty groups, one member per
  group, a single group, oversized `dim_size`, `dim=1`, 3-D, negative dim, wide
  feature rows split by the tile);
- both normalisations (`unbiased` true/false) over the whole shape matrix;
- boundary values: `±inf`, `NaN`, `±1000`, single-member groups, empty groups
  (`NaN` under both normalisations);
- large-offset small group accuracy (`1000 ± 0.01`, which separates the centered
  form from the expanded one);
- long groups do not collapse (65504 / 131072 rows, fp16/bf16); non-contiguous
  `src`; `int32`/`int64` index;
- gradient comparison (fp64 reference, fp16/bf16 low precision, the
  `unbiased=False` Jacobian, a constant group giving 0); a `grad_out` on
  another device raises; the forward does not modify `src`;
- invalid inputs: dtype, `index` dtype/rank/length/range, `dim` out of range,
  invalid `dim_size`, CPU tensors, non-tensors, non-bool `unbiased`.

### 7.2 Performance protocol

- Three measurements: `kernel` (`_pybind` plus one `synchronize()`), `operator`
  (the public Python entry point) and `composed` (`argsort` + `segment_sum_csr`
  for the sum + divide by `n` + subtract the mean + square + `segment_sum_csr`
  for the squared sum + `sqrt`, all in-tree NPU operators; the A2 branch has no
  `scatter_*`);
- A2: 7 shapes × 2 dtypes, 14 rows, plus a 1–65536 group sweep;
- A5: the same shapes, a rate probe and 3 repetitions (the #186 script and
  criteria);
- The expected gain has two sources: the composed baseline needs at least three
  full passes and materialises `N×C` intermediates, while this implementation
  needs two full passes plus one pass that only writes `G×C`. The actual factor
  is measured.

## 8. Steps and risks

### 8.1 Steps

1. Branch `feat/scatter-std` off the latest master (the shared phase changes are
   the same ones #252 merged);
2. One commit with host + kernel + Python + pybind + docs + tests, keeping one
   issue to one PR;
3. Full A2 rebuild, functional / gradient / independent checks, the 14-row
   benchmark and the family regression; A5 goes through CI compilation and the
   team's 950 measurements;
4. `/compile` triggers CI, CodeCheck must report zero findings (nothing
   suppressed), then squash into a single commit;
5. Test report v1.0 with a timestamp, A2 and A5 labelled separately.

### 8.2 Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Editing `scatter_softmax_kernel_phase.h` affects the three merged operators | The `Std` mode is only compiled when instantiated; the full suites and benchmarks of the existing operators are re-run |
| The count workspace grows linearly with the group count | 4 bytes per group (padded to 32-byte blocks); above 16384 groups the counts fall back to one atomic per row - same semantics - and the benchmark keeps a `dim_size = 65536` row to observe it |
| The backward reads three group tensors, so small group counts miss the cache | Recorded as an optimisation: merge `mean` and `out` into one resident fp32 buffer, or reuse the row buffer while tiling `after`. Correctness comes first |
| The centered form reads the source one more time than the expansion | The expansion is unusable on large-offset data; the row buffer is reused inside an `after` tile to cut bandwidth, and the measured factor is in the report |
| `torch_scatter` is not installed, so a direct third-party baseline is missing | The baseline uses a per-group `torch.std` loop (float64/float32) and the in-tree composed implementation; if `torch_scatter` becomes installable, add the direct comparison |

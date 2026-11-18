# scatter_logsumexp Design (A2/A3 arch22, 950 arch35)

> Status: implemented. The measured functional, numerical and performance data
> live in the test report of the pull request; this document describes the
> interface, the semantics, the reuse boundary and the tiling.
> Dependency: `scatter_softmax` and `scatter_log_softmax` are already merged
> (#165, #186); this operator reuses their host scaffolding and device pipeline.

## 1. Scope and acceptance

An NPU implementation of `torch_scatter.scatter_logsumexp`:

```python
scatter_logsumexp(src, index, dim=0, dim_size=None)  # Tensor
```

For every group `g`:

```
out[..., g, ...] = log( sum_{i : index[i] = g} exp(src[..., i, ...]) )
```

The numerically stable form matches `torch_scatter`: subtract the group maximum
first:

```
m_g = max_{i : index[i] = g} src[...]
out[..., g, ...] = m_g + log( sum_i exp(src[..., i, ...] - m_g) )
```

Acceptance:

| Item | Requirement |
| --- | --- |
| Interface | aligned with `torch_scatter.scatter_logsumexp`; `dim` defaults to 0, as in the in-repo `scatter_softmax`; no `out=` argument |
| Output | `src.shape` with `dim` replaced by `dim_size`; dtype and device follow `src` |
| dtype | `float16` / `float32` / `bfloat16`; the reduction runs in fp32 throughout |
| index | same-device 1-D `int32` / `int64`, length = `src.size(dim)`, values in `[0, dim_size)` |
| Autograd | supported; `grad_src[i] = grad_out[index[i]] * exp(src[i] - out[index[i]])` |
| Platform | arch22 (A2/A3) and arch35 (950) share one kernel source; each architecture directory only holds a thin compilation unit |
| Performance | net gain over the `scatter_max + exp + scatter_sum + log` composition; correctness first, then measured tuning |

## 2. Host shape and output contract

The batch/group/feature decomposition of `scatter_softmax` is reused:

- `batch  = prod(src.shape[:dim])`
- `groups = dim_size`
- `after  = prod(src.shape[dim+1:])`, padded to 8 elements as `afterPadded`
- input rows `rows = batch * dimLength`, output rows `outRows = batch * groups`

The input and the workspace layout stay unchanged
(`workspaceElems = batch * groups * afterPadded`); only the output is new:

```
out[b, g, f] = maximum[b, g, f] + log(total[b, g, f])
```

The output allocation therefore cannot reuse the
`torch::empty(src.sizes())` of `RunForward`. The host side adds
`csrc/npu/scatter_logsumexp/common/grouped_logsumexp_shared.h` with two
entry points:

```cpp
torch::Tensor scatter_logsumexp_forward(src, index, dim, dim_size);
torch::Tensor scatter_logsumexp_backward(src, out, grad_out, index, dim, dim_size);
```

- Forward: `ValidateInputs` / `BuildTiling` / `CheckLaunchable` /
  `ResolveDType` all come from
  `csrc/npu/scatter_softmax/common/grouped_softmax_shared.h`; only the output
  allocation changes, "replace `dim` with `dim_size`".
- An output with zero elements (`after == 0` or `groups == 0`) returns
  immediately; `rows == 0` with a non-empty output (groups without members)
  fills the output with `-inf` and returns without launching a kernel.
- Backward: the shape returns to `src.sizes()` and `grad_out` has the forward
  output shape. The checks start from `ValidateInputs` (here
  `src.size(dim) == dim_size`, so the index-length check still holds) and add
  the contiguous, shape, dtype and **same-device** checks for `out` /
  `grad_out`. The source tensor is an explicit argument because the gradient
  reads `src` itself.
- `dim_size == 0` makes the backward return `zeros_like(src)` directly (the
  output has no elements, so the gradient is identically zero).

## 3. Device algorithm

### 3.1 Forward: two full passes plus one output pass

| Pass | Read | Write | Notes |
| --- | --- | --- | --- |
| P1 maximum | `src` (N×C) | `maximum` workspace (G×C fp32) | reuses the max batch of `scatter_softmax`; empty groups keep the neutral value |
| P2 exponential sum | `src` + `maximum` | `total` workspace (G×C fp32) | `total += exp(src - maximum)`, fp32 accumulation |
| P3 finalize | `maximum` + `total` | `out` (G×C) | `out = maximum + log(total)`, only `dim_size` rows |

Key points:

- **P3 only walks `batch * groups` rows**, independent of the input size
  `batch * dimLength`. That is one of the main gains over the composition,
  which writes the whole N×C intermediate back to device memory. The row
  count, per-core row chunk and row batch of P3 come from the new tiling
  fields `outRows` / `outRowChunk` / `outBlockNum` / `outRowBatch`, copied
  over a tiling copy at launch time; the pass does not read `index`, because
  the dense workspace rows are the output rows.
- The reduction bookkeeping of P1 / P2 (a private accumulator resident in UB,
  or one atomic write per row) reuses the existing implementation; there is no
  second row scheduler.
- **`+inf` correction**: `exp(+inf - +inf) = NaN`, while `torch.logsumexp`
  reports `+inf` for that column. P2 clamps the subtracted maximum to the
  largest finite value `FLT_MAX` (one `Mins` per row): a `+inf` member
  contributes `exp(+inf) = +inf`, `total = +inf`, and P3 reports
  `+inf + log(+inf) = +inf`, which is the expected value; the finite members
  of that group contribute `exp(-inf) = 0`. Finite groups are unaffected
  (their maximum is already below `FLT_MAX`). `-inf` and empty groups need no
  correction: the neutral value is `-FLT_MAX`, so
  `-inf - (-FLT_MAX) = -inf` and `exp` yields 0, and P3 reports
  `-FLT_MAX + log(0) = -inf`, again the expected value.

### 3.2 Backward: one elementwise pass

```
grad_src[i] = grad_out[index[i]] * exp(src[i] - out[index[i]])
```

- one full read/write: read `src`, broadcast-read the group rows of `grad_out`
  and `out` through `index`, write `grad_src`;
- no reduction workspace and no intermediate tensor, so the pass only
  allocates three dtype buffers (`src`, the `out` row, the `grad_out` row) and
  three fp32 tiles, still inside the 20 bytes per element budget of the
  shared tiling;
- **the two group-shaped tensors are kept in UB for a small group count**:
  every source row reads the same row of `out` and of `grad_out`, and with a
  small `groups` all those reads hit the same cache line without overlapping.
  Measured on `1048576 × 128`, the backward of `groups = 1` degrades to about
  40 ms (`groups >= 16` takes about 6 ms). When the whole group-shaped tensor
  fits the on-core budget (an fp32 copy of `workspaceElems` <= 16 KB, i.e. a
  small `groups`), the backward widens `out` and `grad_out` into a resident
  fp32 copy once before the row loop and reads them with `Adds` inside it.
  The two copies are exactly the space the tiling already reserves for "two
  fp32 workspaces", so the row batch does not shrink and the UB peak stays
  below 96 KB. When the copy does not apply (`after > colTile` or the
  group-shaped tensor exceeds the budget) the pass falls back to the per-row
  GM read with unchanged semantics;
- with `out[g] = ±inf` the formula reproduces PyTorch (measured against
  `torch.logsumexp`): an all-`-inf` group gives `-inf - (-inf) = NaN`, a `+inf`
  member gives `+inf - (+inf) = NaN`, and the finite members of that group give
  `x - (+inf) = -inf`, whose `exp` is 0.

### 3.3 Reuse boundary

| Reuse | Content |
| --- | --- |
| Reused directly | `ValidateInputs` / `BuildTiling` / `CheckLaunchable` / `ResolveDType`; `LoadGroups`, `CopyRowsIn/Out`, `CopyGroupRowsIn`, `CopyWorkspace*`, `AccumulateRows`, `FlushAccumulator`; the maximum and exponential-sum batch bodies; and the same "widen the group-shaped tensor into a resident fp32 copy" trick for a small group count (the softmax family caches the workspace, the backward caches `out` and `grad_out`) |
| Compile-time extension | the phase enum of `scatter_softmax_kernel_common.h` becomes the three modes `Softmax` / `LogSoftmax` / `LogSumExp`; the exponential pass gains the `+inf` clamp of `LogSumExp` and does not write the `src`-shaped output |
| File split | the transfer and resident-workspace helpers live in `scatter_softmax_kernel_io.h` (`phase.h` no longer exceeds the 500-line single-header limit of the static check); all three operators include it |
| New | the P3 finalize batch, the elementwise backward batch, the launch / tiling instantiation of `scatter_logsumexp`; the `DispatchDType` of both operators collapses into one copy in the entry header |
| Not reused | the normalize batch of softmax (it writes the `src`-shaped result per row, while log-sum-exp only writes the group rows) |

This keeps one copy of the row scheduling, the UB/workspace budget and the
reduction bookkeeping for all three operators, so the static check does not
report "the same reduction code twice".

## 4. Tiling, alignment and pipelining

The tiling constants and budgets of `scatter_softmax` are reused as-is:

- `SCATTER_SOFTMAX_WORKSPACE_ALIGN = 8` (32 bytes), `ROW_ALIGN = 16`
  (32 bytes for fp16/bf16, 64 bytes for fp32);
- the row batch `rowBatch` is derived from `ROW_BUDGET_BYTES = 80 KB` and the
  `colTile × 20` bytes of one row;
- the UB-resident workspace limit is `ACC_BUDGET_BYTES = 16 KB`; beyond it the
  pass writes one atomic per row;
- a core owns a contiguous range of rows; every batch runs the MTE2 / V / MTE3
  pipeline, and P3 and the backward reuse the same driver.

P3 walks `batch * groups` rows, usually far fewer than the input rows; when
`after` is large it tiles the feature axis by `colTile`, exactly like P1/P2.

## 5. Numerics

| Input | `out` | Basis |
| --- | --- | --- |
| empty group (`dim_size` above the actual group count) | `-inf` | empty reduction of `torch.logsumexp`; `-FLT_MAX + log(0)` |
| all `-inf` in a group | `-inf` | `torch.logsumexp([-inf, -inf]) = -inf` (measured) |
| `+inf` in a group | `+inf` | `torch.logsumexp([inf, x]) = inf` (measured); guaranteed by the P2 clamp |
| `NaN` in a group | `NaN` | `exp(NaN)` enters `total` and propagates through P3 |
| an element equal to `-FLT_MAX` | `-FLT_MAX` | the offset is 0 and `total = 1` |
| long groups in low precision | no collapse | fp32 reduction; `out = m + log(sum)`, the sum cannot underflow the whole group to `-inf` unless the group really is all `-inf` |

Backward:

| `out[g]` | Element | `grad_src` |
| --- | --- | --- |
| `-inf` | `-inf` | `NaN` (`-inf - (-inf)`) |
| `+inf` | `+inf` | `NaN` |
| `+inf` | finite | `0` |
| finite | any | `grad_out[g] * exp(x - out[g])` |

Every row above was measured on the CPU implementation of `torch.logsumexp`
(fp32) and is checked case by case after implementation.

## 6. Differences from `torch_scatter`

| Case | `torch_scatter.scatter_logsumexp` | This implementation |
| --- | --- | --- |
| empty group | `-inf` | `-inf` (identical) |
| all `-inf` group | the composed implementation evaluates `m + log(sum(exp(x-m)))`, and `-inf - -inf` produces `NaN` | `-inf`, as `torch.logsumexp` reports |
| group holding `+inf` | same composition, gives `NaN` | `+inf`, as `torch.logsumexp` reports |
| `dim` default | `-1` | `0`, as the in-repo `scatter_softmax` |
| `out=` argument | supported | not supported, as the in-repo `scatter_softmax` |

The last two are interface-consistency choices inside the repository; the
first two take the mathematically correct result that also matches the native
`torch.logsumexp`, and the API documentation states them explicitly.

## 7. Tests and performance

### 7.1 Case organisation (the layout requested in #186)

| Location | Content |
| --- | --- |
| `test/common/grouped_softmax_golden.py` | family float64 reference: grouped softmax / log-softmax plus `grouped_logsumexp_reference` (`dim` / `dim_size` / empty groups) and `logsumexp_gradient`, which reuses it to build the graph |
| `test/common/grouped_softmax_cases.py` | family shape, dtype and invalid-input case tables |
| `test/common/test_grouped_softmax_suite.py` | family base `GroupedFamilySuite` (argument, gradient and boundary cases shared by the three operators) plus the softmax / log-softmax suites |
| `test/common/test_grouped_log_softmax_suite.py` | the log-domain suites, `LogSoftmaxSuite` and `LogSumExpSuite` (`test_` prefix, class name not `Test*`) |
| `test/scatter_logsumexp/arch22/` and `arch35/` | each holds only `golden.py`, `test_scatter_logsumexp.py`, `benchmark_scatter_logsumexp.py` and only binds them |

The operator adds no `test/common` file: the reference, the case tables, the
suite and the benchmark are merged into the existing family modules, so the
number of files under `test/common` stays the same as when
`scatter_log_softmax` was merged.

Coverage:

- shape × dtype matrix (1-D features, regular groups, empty groups,
  single-element groups, a single group, an oversized `dim_size`, `dim=1`,
  3-D, negative `dim`, wide feature rows spanning tiles);
- extreme values: all `-inf`, `+inf` mixtures, `NaN`, `±1000` magnitudes,
  single-element groups;
- long groups do not collapse (65504 / 131072 rows, fp16/bf16);
  non-contiguous `src`; `int32`/`int64` index;
- the `-inf` fill path when there are no members but the output is not empty;
- gradient comparison (fp64 reference, fp16/bf16 low precision, `+inf`
  element is NaN and its finite siblings are 0); an error when `grad_out`
  lives on another device; the forward does not modify `src`;
- invalid input: dtype, `index` dtype/rank/length/range, out-of-range `dim`,
  invalid `dim_size`, CPU tensor, non-Tensor.

### 7.2 Performance methodology

- three measurements: `kernel` (`_pybind` plus one `synchronize()`),
  `operator` (the public Python entry) and `composed` (`argsort` +
  `segment_max_csr` + subtract the maximum + `exp` + `segment_sum_csr` +
  `log`, all in-repo NPU operators; the A2 branch has no `scatter_max`);
- A2: 7 shapes × 2 dtypes, 14 rows, plus a group sweep from 1 to 65536;
- A5: same-shape comparison, a rate probe and three repetitions (the scripts
  and criteria of #186);
- the expected gain comes from two places: the full read/write drops from the
  three passes of the composition (max read, subtract-and-`exp` read/write,
  sum read) to two (max read, `exp` read), and no `N×C` intermediates of `exp`
  and `x-m` are materialised; P3 only writes `G×C`. The exact ratios are the
  measured ones.

## 8. Steps and risks

### 8.1 Steps

1. Branch `feat/scatter-logsumexp` from the latest master (the shared phase
   header change is the same one as #186);
2. one commit for host + kernel + Python + pybind + docs + tests, keeping
   "one issue, one pull request";
3. full A2 rebuild + functional/gradient/standalone checks + the 14-row
   benchmark; A5 is covered by the CI compilation and measured by the team;
4. `/compile` triggers CI and the static check must report 0 findings (never
   suppressed); squash into one commit;
5. test report v1.0 with a timestamp, with A2 and A5 labelled separately.

### 8.2 Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| changing `scatter_softmax_kernel_phase.h` affects the two merged operators | the `LogSumExp` mode is only compiled when instantiated; the full cases and the benchmark of both operators are re-run |
| the `+inf` clamp adds a vector instruction | one `Mins` per row in the exponential pass only; the 14-row benchmark is compared item by item with #186 to confirm no regression |
| P3 cost when `dim_size` is far above the actual group count | P3 walks `batch * groups` rows, independent of the input rows; the group sweep includes `dim_size = 2 × max+1` and `65536` |
| the composition and `torch_scatter` differ on `±inf` | the difference is listed explicitly (section 6) and the tests take `torch.logsumexp` as the reference |
| `torch_scatter` is not installed, so a direct third-party baseline is missing | the baseline is a per-group loop over `torch.logsumexp` (float64/float32) plus the in-repo composition; a direct `torch_scatter` measurement is added if the environment provides it |

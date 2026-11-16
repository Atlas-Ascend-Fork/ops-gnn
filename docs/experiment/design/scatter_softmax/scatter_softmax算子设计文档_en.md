# scatter_softmax Design (A2/A3 arch22, 950 arch35)

## 1. Scope and acceptance

An NPU implementation of `torch_scatter.scatter_softmax`:

```python
scatter_softmax(src, index, dim=0, dim_size=None)  # Tensor
```

- For every group `g`, `out[i] = exp(src[i] - max_g) / sum_j exp(src[j] - max_g)`
  along `dim`;
- the output keeps the shape, dtype and device of `src`; float16, float32 and
  bfloat16 are supported;
- the backward pass evaluates the grouped-softmax Jacobian analytically, so
  autograd is available;
- the operator ships its own device implementation under
  `csrc/npu/scatter_softmax/` and no longer composes `segment_max_csr`,
  `segment_sum_csr` and `index_add_`;
- arch22 (A2/A3) and arch35 (950) share one AscendC source
  (`op_kernel/common/scatter_softmax_kernel.h`); each architecture only adds a
  thin compilation unit.

Acceptance: all 74 cases in `test/scatter_softmax/` pass (precision, edge cases,
`±inf`, long groups, low-precision gradients); the full A2 regression reports
4804 passed.

## 2. Host shape contract

```text
before     = prod(src.shape[:dim])           # batch axes
dimLength  = src.shape[dim]                  # reduced axis indexed by `index`
after      = prod(src.shape[dim + 1:])       # trailing feature axis
rows       = before * dimLength
address of src[b, j, a] = (b * dimLength + j) * after + a
```

- `index` must be 1-D with `numel() == src.size(dim)`; the broadcast form with
  `index.numel() == src.numel()` is outside this operator's contract, exactly as
  in the previous Python version.
- `src` is made contiguous first; the output is a fresh contiguous tensor.
- The reduction workspace is fp32 with row stride
  `afterPadded = round_up(after, 8)`, so every (batch, group) workspace row is a
  whole number of 32-byte blocks.
- `src.numel() == 0` returns a zero tensor without launching a kernel.
- The kernels run on the current NPU stream of `src`; no private stream and no
  host synchronisation.

## 3. Device algorithm

`index` need not be sorted, so the per-group reduction is accumulated by the
MTE3 atomic unit. Five kernels are launched:

| # | kernel | computation |
| --- | --- | --- |
| 1 | `ScatterSoftmaxMaxKernel` | `atomicMax(maximum[b][g][a], src[b][j][a])`, `g = index[j]` |
| 2 | `ScatterSoftmaxExpKernel` | `out = exp(src - maximum)`, plus `atomicAdd(sum[b][g][a], out)` |
| 3 | `ScatterSoftmaxNormalizeKernel` | `out /= sum` |
| 4 | `ScatterSoftmaxGradSumKernel` | `atomicAdd(sum[b][g][a], out * grad_out)` |
| 5 | `ScatterSoftmaxGradKernel` | `grad_src = out * (grad_out - sum)` |

The three forward kernels and the two backward kernels are issued back to back
on one stream, so the kernel boundary is the synchronisation point and no
cross-core barrier is needed.

## 4. Tiling, alignment and pipelining

- **Rows**: each AIV core owns a contiguous run of rows,
  `rowChunk = ceil(rows / blockNum)` with `blockNum = min(AIV core count, rows)`.
  A core buffers `rowBatch` rows at a time (sized from the UB budget, at most 8)
  and each row of the batch keeps its own group label and source row base.
- **Columns**: `colTile = min(2048, round_up(after, 16))`, and the trailing axis
  is tiled by `colTile`. `colTile` is also the stride between the per-row UB
  slices of a batch, so a multiple of 16 keeps 32 bytes of fp16/bf16 and 64 bytes
  of fp32 slice starts on a 32-byte boundary, which is what DataCopyPad and the
  vector unit require.
- **Atomic length**: `width = min(colTile, afterPadded - col)`, always a multiple
  of 8, so a tile never writes past its own row into the next group's padding.
- **Unaligned rows**: source and output rows at `row * after + col` are not
  guaranteed to be aligned, so they use `DataCopyPad` with exactly
  `valid * sizeof(T)` bytes; the workspace uses aligned `DataCopy` (a multiple of
  8 elements at a row base that is a multiple of `afterPadded`).
- **Pipeline events**: each kernel allocates four events (`MTE2_V`, `V_MTE2`,
  `V_MTE3`, `MTE3_V`) and synchronises load -> compute -> store -> buffer reuse
  once per batch instead of once per row.
- **Reduction bookkeeping**: when `batch x groups x afterPadded` fits the
  16 KB workspace budget, the workspace stays in UB for the whole pass: the
  reduction passes accumulate in UB and each core writes the workspace back once
  with an atomic store, and a pass that reads the workspace reads it once instead
  of once per row. Otherwise the reduction falls back to one atomic write per row
  and the reads fall back to one per row. Both paths depend only on the group
  label, so `index` need not be sorted.
- **Cost of keeping it resident**: the exp pass holds the cached maximum
  workspace and the sum it accumulates into at the same time, so the row budget
  reserves two copies. Both are capped at 16 KB, and `rowBatch` stays at 8 for
  `after = 128`.
- **Padding lanes**: every vector instruction writes at a 32-byte aligned buffer
  base. Only when the tile still has padding lanes (`valid < width`) is the
  neutral value duplicated over the whole tile first (neutral maximum for pass 1,
  zero for the sums) before the meaningful `valid` lanes are overwritten; a full
  tile skips that pass.
- **UB budget**: 80 KB of row buffers (20 bytes per (row, feature) covering the
  two dtype buffers and three fp32 tiles) plus at most two 16 KB workspace copies
  (the accumulator and the read cache), so
  at most about 112 KB of the 192 KB UB is used.

## 5. Numerics

- The per-group maximum, the `exp` sum, the normalization and the whole backward
  pass are evaluated in float32; `exp` and the output round in the source dtype
  (`CAST_RINT`). A long fp16/bf16 group is therefore not limited by the 65504 sum
  ceiling.
- The workspace maximum is initialised by the host to the same constant the
  kernel uses for padding, `-FLT_MAX`
  (`SCATTER_SOFTMAX_NEUTRAL_MAXIMUM`), so a padded lane can never become the
  group maximum and host/kernel semantics cannot diverge.
- Empty groups hold no elements and produce no output.
- An all-`-inf` group keeps the neutral maximum, so `exp(-inf - neutral) = 0`,
  the sum is 0 and `0 / 0 = NaN`, matching `torch.softmax`.
- A group containing `+inf` gets `+inf` as its maximum, `exp(+inf - +inf) = NaN`,
  a NaN sum and an all-NaN column.
- The zero-sum case is produced by `Div` itself (`0/0 = NaN`); there is no
  special case in the kernel.

## 6. Differences from `torch_scatter`

- Only a 1-D group index of length `src.size(dim)` is supported; broadcasting
  `index` to the shape of `src` is not.
- `dim_size` may exceed `index.max() + 1` to keep empty groups, and must fit in
  int32 because the device side reads the group label as an int32 scalar.
- The accumulation order inside a group is decided by the atomic unit, so the
  floating-point result may differ in the last bits and repeated runs are not
  guaranteed to be bit-identical. The deviation is far below the source-dtype
  tolerance.

## 7. Tests and performance

- Function: `test/scatter_softmax/test_scatter_softmax.py`, 74 cases covering
  1-D/2-D/3-D inputs, empty groups, single-element groups, inferred
  `dim`/`dim_size`, int32 index, non-contiguous source, large magnitudes,
  `+inf`/`-inf`, long low-precision groups, fp32/fp16/bf16 gradients and invalid
  arguments.
- Golden: `test/scatter_softmax/golden.py`, an independent float64 CPU
  implementation.
- Performance (Ascend 910B4, `benchmark_scatter_softmax.py`, median of 5 x 20):

| shape (edges x features, groups) | dtype | previous Python composition | kernel before | kernel now |
| --- | --- | --- | --- | --- |
| 65536 x 1024, 64 | float32 | 23.33 ms | 3.65 ms | 2.44 ms |
| 65536 x 1024, 64 | float16 | 21.13 ms | 3.06 ms | 1.66 ms |
| 131072 x 1, 1 | float32 | 72.79 ms | 16.66 ms | 10.30 ms |
| 131072 x 1, 1 | float16 | 70.48 ms | 14.81 ms | 10.16 ms |
| 262144 x 256, 4096 | float32 | 77.61 ms | 10.36 ms | 3.54 ms |
| 1048576 x 128, 16384 | float32 | 305.56 ms | 38.87 ms | 12.31 ms |
| 1048576 x 128, 1024 | float32 | 250.79 ms | 38.69 ms | 12.06 ms |
| 1048576 x 128, 4 | float32 | 249.19 ms | 45.26 ms | 23.84 ms |
| 1048576 x 128, 1 | float32 | 430.90 ms | 147.11 ms | 82.82 ms |

- Row batching plus in-UB accumulation gives a 1.46-3.21x speed-up. The largest
  gains are the many-group, wide-`after` shapes that used to make many cores
  contend on the same workspace row; `after = 1` rows are still limited by the
  per-row DMA size, but they now synchronise once per batch.

## 8. Known trade-offs and follow-up work

- The workspace is `2 x groups x afterPadded x 4` bytes (maximum and sum), so
  memory grows linearly with the number of groups; a graph with very many tiny
  groups is space-inefficient.
- The forward pass reads `src` in two passes (and `out` in the third); a later
  version can fuse the maximum with `exp`, or fall back to a deterministic
  owner-row path when the groups are already contiguous.
- The backward pass reuses the forward output, so one tensor the size of `src`
  stays alive for training.
- Row batching trades UB for parallelism: a larger `rowBatch` removes
  synchronisations but occupies more UB, and the current cap of 8 keeps
  `colTile` at the full 2048 within the 192 KB UB budget.
- A resident workspace replaces one atomic write per row and one workspace read
  per row with one reduction per core, one atomic write-back and one workspace
  read per pass, at the cost of up to two 16 KB copies per core. A workspace
  larger than that falls back to the per-row atomic and per-row read path, so
  shapes with a very large group count keep the previous behaviour. The gain is
  largest for one or a few groups: a million rows that all reduce into one group
  re-read the same 512-byte line per row, and those reads do not overlap - 46 ms
  per pass on 910B4 versus 4 ms once the line stays resident.

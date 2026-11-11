# segment_logsumexp Operator Design (A2/A3 arch22, 950 arch35)

## 1. Scope and acceptance

Port `torch_scatter.segment_logsumexp` to Ascend NPU:

```python
segment_logsumexp(src, indptr, out=None)  # Tensor
```

- The reduction follows the CSR pointer convention: segment `i` covers the row
  range `[indptr[i], indptr[i + 1])` and yields
  `out[i] = log(sum_j exp(src[j]))`;
- The output has shape `(indptr.numel() - 1,) + src.shape[1:]` and the dtype and
  device of `src`; float16, float32 and bfloat16 are supported;
- An empty segment yields `-inf`; a segment of only `-inf` yields `-inf` and a
  segment containing `+inf` yields `+inf`, matching `torch.logsumexp`;
- The operator ships its own device implementation in
  `csrc/npu/segment_logsumexp/` instead of composing `segment_max_csr` and
  `segment_sum_csr`;
- arch22 (A2/A3) and arch35 (950) share one AscendC source
  (`op_kernel/common/segment_logsumexp_kernel.h`); each architecture directory
  only holds a thin compilation unit.

Acceptance: all 86 cases in `test/segment_logsumexp/` pass (precision, empty
segments, `±inf`, non-aligned feature widths, features wider than one column
tile, long segments, row chunking); the A2 full regression reports 4816 passed;
arch35 compiles, and the team re-measures it on an A5 device.

## 2. Host-side contract

```text
rows     = src.size(0)                  # must equal indptr[-1]
features = prod(src.shape[1:])          # flattened trailing axes
segments = indptr.numel() - 1
src[row, a] address = row * features + a
out[s, a]   address = s * features + a
```

- Python validates `indptr` (non-negative, nondecreasing,
  `indptr[-1] == src.size(0)`) and narrows it to int32; the kernel reads int32
  scalars, hence `src.size(0) <= 2**31 - 1`.
- With `out=` the result is computed into a fresh contiguous tensor and copied
  back with `out.copy_(result)`, so non-contiguous views work.
- `indptr.numel() == 1` (zero segments) returns an empty tensor without
  launching a kernel.
- Rows before `indptr[0]` belong to no segment. The kernel only walks the row
  ranges of the segments, so those rows are ignored and no pointer rebasing is
  needed (the former composition had to rebase them).
- The platform core count is queried once per call and every block count is
  derived from it, which keeps the fixed cost of a call low.

## 3. Device algorithm: two-level reduction, two schedules

A CSR segment is a **contiguous row range**, so nothing is reordered and no
atomic unit is involved. One core per segment wastes the machine when the cores
outnumber the segments (A5 has 64 vector cores and GNN segment counts are often
just the batch size), so the schedule depends on the number of segments.

### 3.1 Enough segments to fill the machine: one fused kernel

One core takes one (segment, feature window) task and runs the whole reduction
inside a single kernel: per-feature maximum, clamp, `sum(exp(src - shift))`,
`shift + ln(sum)`. Everything stays in the unified buffer, so there is no
workspace traffic and no extra launch.

### 3.2 Too few segments: row chunks and a two-level reduction

The rows of a segment are cut into `rowChunks` pieces and four kernels run back
to back on the same stream:

| phase | task grid | computation | exchange |
| --- | --- | --- | --- |
| 1 | (segment, window, chunk) | per-feature maximum of the chunk | → `splitMax` |
| 2 | (segment, window) | maximum over the chunks, clamped, → shift | written back into the first chunk slot |
| 3 | (segment, window, chunk) | `sum(exp(src - shift))` | → `splitSum` |
| 4 | (segment, window) | sum over the chunks → `shift + ln(sum)` → output | — |

- The kernel boundary is the cross-core synchronisation point; no barrier is
  needed.
- Phase 2 writes its result into the first chunk slot of that window: only the
  owning core touches those slots, so the reduction is safely in place.
- The workspace holds `2 × tasks × colTile` fp32 values; every entry is written
  before it is read, so it needs no initialisation.

### 3.3 Chunk granularity

```text
wanted      = ceil(cores / baseTasks)         # baseTasks = segments × colTiles
affordable  = rows / (segments × 16)          # at least 16 rows per chunk
rowChunks   = clamp(min(wanted, affordable), 1, 64)
```

A segment of length `len` is cut into `per = ceil(len / rowChunks)` rows per
piece, and the last piece stops at the segment end; a segment shorter than
`rowChunks` leaves trailing empty pieces, which contribute neutral values
(`-inf` / `0`). Very short inputs are not split, because four launches would
cost more than the parallel gain.

## 4. Tiling, alignment and pipelining

- **Task walking**: `task = blockIdx; task += blockCount`, where `blockCount` is
  `min(platform core count, tasks of that phase)`.
- **Columns**: `colTile = min(1024, round_up(features, 8))`; every tile starts on
  a multiple of 8 lanes, so all fp32 vector destinations stay 32-byte aligned.
- **Row batching**: one row is only a few hundred bytes, so a burst per row would
  pay the MTE2 completion latency on every iteration. When the window spans the
  whole row (`colTiles == 1`) a single `DataCopyPad` moves `rowBatch` rows at
  once (`rowBatch = min(64, 16KB / row bytes)`), which cuts the number of bursts
  by an order of magnitude. Narrower windows fall back to one row per transfer:
  their row gap would have to be expressed as a source stride whose unit differs
  between the architectures, and that is not guessed here.
- **Partial batches**: `NextBatch` never reads past the last row of the task.
- **Unaligned access**: source and output rows are not 32-byte aligned, so every
  transfer moves exactly `valid * sizeof(T)` bytes with `DataCopyPad`; the
  unified buffer is padded to 32 bytes and the padding is never written back.
- **Workspace access**: whole `colTile`-row fp32 entries, 32-byte aligned, moved
  with `DataCopyPad` by byte count.
- **Pipeline events**: four events per task (`MTE2_V`, `V_MTE2`, `V_MTE3`,
  `MTE3_V`), each `SetFlag` immediately followed by `WaitFlag`, covering
  "load → reduce → store → reuse".
- **Unified-buffer usage**: a 16KB staging buffer plus four `colTile`-sized
  buffers (one for the source dtype, three fp32); about 32KB when `colTile` is
  1024 and the source is fp32.

## 5. Numerical semantics

- The per-segment maximum, `sum(exp(src - shift))` and `ln` all run in float32,
  while `exp` and the output round in the source dtype; a float16/bfloat16
  segment is therefore not limited by the 65504 ceiling, because
  `exp(src - shift) <= 1` and only the accumulator can grow, which is fp32.
- Subtracting the maximum before exponentiating keeps `exp` of a large positive
  value out of range.
- `shift` is clamped into `[-FLT_MAX, +FLT_MAX]` with `Mins/Maxs`. An empty or
  all-`-inf` segment then gets `shift = -FLT_MAX`, `exp(-inf + FLT_MAX) = 0`,
  `ln(0) = -inf`, so the result is `-inf`; a segment containing `+inf` gets
  `shift = FLT_MAX`, `src - shift = +inf`, a `+inf` total and therefore `+inf`.
  This avoids `inf - inf` and matches `torch.logsumexp`.
- An empty segment (`indptr[i] == indptr[i + 1]`) has no rows, so its total is
  `0` and `ln(0) = -inf`; combined with the clamped shift that is `-inf` without
  a dedicated branch.
- The summation order inside a segment is fixed (ascending row order, then
  ascending chunk order), so repeated runs are bit-identical **within one
  schedule**. The fused and split schedules add in a different order (the fused
  one accumulates row by row into a single fp32 accumulator, the split one adds
  per-chunk partial sums), so their results agree within fp32 tolerance but are
  not bit-identical to each other.

## 6. Differences from `torch_scatter`

- `indptr[-1]` must equal `src.size(0)`; trailing rows outside the segments are
  rejected instead of silently ignored.
- `indptr` is narrowed to int32 internally, so `src.size(0)` must not exceed
  `2**31 - 1`.
- A non-contiguous `out=` tensor is written through its strides and the same
  object is returned.

## 7. Tests and performance

- Functional tests: `test/segment_logsumexp/test_segment_logsumexp.py`, 86 cases
  covering 1-D features, multi-dimensional trailing axes, empty segments
  (leading/middle/trailing/all), zero segments, single-element segments, `±inf`,
  a non-aligned feature width (100), features wider than one column tile (2000),
  segments spanning several bursts, row chunking (2000 rows in 2 segments, 5000
  rows in 1 segment), `out=` and non-contiguous `out`, invalid arguments and
  long low-precision segments.
- Golden: `test/segment_logsumexp/golden.py`, a CPU float64 implementation that
  imports neither `ops_gnn` nor `torch_scatter`.
- Performance (Ascend 910B4 A2, `benchmark_segment_logsumexp.py`, iters=50):

| shape (rows × features, segments) | dtype | former Python composition | this implementation | speed-up |
| --- | --- | --- | --- | --- |
| 1048576 × 128, 16384 | float32 | 47.649 ms | 3.735 ms | 12.8× |
| 1048576 × 128, 16384 | float16 | 46.473 ms | 3.404 ms | 13.7× |
| 262144 × 256, 4096 | float32 | 13.848 ms | 1.399 ms | 9.9× |
| 262144 × 256, 4096 | float16 | 13.214 ms | 1.246 ms | 10.6× |
| 65536 × 128, 64 | float32 | 1.529 ms | 0.583 ms | 2.6× |
| 65536 × 128, 64 | float16 | 1.554 ms | 0.566 ms | 2.7× |
| 262144 × 128, 4 (long segments) | float32 | 19.226 ms | 1.210 ms | 15.9× |
| 262144 × 128, 4 (long segments) | float16 | 18.570 ms | 1.070 ms | 17.4× |

Segment-count scan (`1048576 × 128`, float32, iters=20), showing that row
chunking flattens the curve:

| segments | this implementation | former Python composition | speed-up |
| --- | --- | --- | --- |
| 1 | 3.803 ms | 284.487 ms | 74.8× |
| 4 | 3.783 ms | 75.403 ms | 19.9× |
| 16 | 6.049 ms | 23.161 ms | 3.8× |
| 64 | 4.589 ms | 14.600 ms | 3.2× |
| 256 | 3.919 ms | 13.380 ms | 3.4× |
| 1024 | 3.657 ms | 14.604 ms | 4.0× |
| 4096 | 3.645 ms | 20.986 ms | 5.8× |
| 16384 | 3.695 ms | 47.708 ms | 12.9× |
| 65536 | 4.352 ms | 156.377 ms | 35.9× |

"Former Python composition" is the `segment_max_csr` + `segment_sum_csr` + `log`
implementation on the same branch, measured with the same script on the same
machine.

### 7.1 A5 (950PR) measurement

The first version only had the "one core per segment" schedule, which was a
**net loss** on A5 (64 vector cores): 0.18× at 4 segments and 1.06× at 16384
segments, because the composition parallelises over rows and fills 64 cores
while a single-level schedule uses at most `segments` cores. Row chunking is the
response.

Measured on an Ascend 950PR (64 vector cores, iters=20, median of 5 repeats, idle machine):

| shape (rows × features, segments) | dtype | kernel | composition | kernel / composition |
| --- | --- | --- | --- | --- |
| 1048576 × 128, 16384 | fp32 | 2.179 ms | 3.597 ms | 1.65× |
| 1048576 × 128, 16384 | fp16 | 1.986 ms | 2.906 ms | 1.46× |
| 262144 × 256, 4096 | fp32 | 0.700 ms | 2.262 ms | 3.23× |
| 262144 × 256, 4096 | fp16 | 0.603 ms | 1.665 ms | 2.76× |
| 65536 × 128, 64 | fp32 | 0.156 ms | 0.852 ms | 5.45× |
| 65536 × 128, 64 | fp16 | 0.151 ms | 0.931 ms | 6.18× |
| 262144 × 128, 4 (long segments) | fp32 | 0.527 ms | 1.872 ms | 3.56× |
| 262144 × 128, 4 (long segments) | fp16 | 0.503 ms | 1.582 ms | 3.15× |

Segment-count scan (`1048576 × 128` fp32) — the curve is flat at 2.1–2.6 ms and no
longer collapses with the segment count:

| segments | kernel | operator | composition | kernel / composition | operator / composition |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.316 ms | 2.893 ms | 14.532 ms | 6.27× | 5.02× |
| 4 | 2.291 ms | 2.887 ms | 6.478 ms | 2.83× | 2.24× |
| 16 | 2.291 ms | 2.928 ms | 3.728 ms | 1.63× | 1.27× |
| 64 | 2.265 ms | 2.956 ms | 3.433 ms | 1.52× | 1.16× |
| 256 | 2.180 ms | 2.864 ms | 3.493 ms | 1.60× | 1.22× |
| 1024 | 2.110 ms | 2.828 ms | 3.442 ms | 1.63× | 1.22× |
| 4096 | 2.118 ms | 2.780 ms | 3.417 ms | 1.61× | 1.23× |
| 16384 | 2.167 ms | 2.669 ms | 3.597 ms | 1.66× | 1.35× |
| 65536 | 2.606 ms | 3.223 ms | 4.113 ms | 1.58× | 1.28× |

Compared with the first single-level version: 1 segment 0.10× → 5.02×,
4 segments 0.18× → 2.24× at operator level (6.27× / 2.83× at kernel level);
16/64/256 segments move from 0.90–0.97× to 1.16–1.27×; many-segment shapes do not
regress.

**Fixed cost of a small shape**: `65536 × 128, 64` is the noisiest point on the
machine (its composed baseline swings between 0.39 and 0.93 ms). Three independent
methods — paired A/B, a three-level comparison (`now` / `batched` / `no check`) and
isolated processes — agree on 1.18–1.83×, with the kernel steady at 5.5–6.4×. The
remaining gap is Python-side: for that shape `no check` takes 0.130 ms against
0.625 ms for `now`. The three host↔device round trips of `_check_inputs` cost
0.75–1.29 ms on A5; they are now **one round trip** (the three predicates are
stacked into one vector and copied back once), giving
`op now / op batched` = 1.03× / 1.00× / 1.03× with unchanged rejection behaviour.
What is left is that single copy plus the Python attribute checks themselves.

**Narrow features still lose**: in the fixed-8.4M-element feature scan,
feat 16–64 lands at 0.30–0.71× — a row is only 64–256 bytes, so the vector
instructions are mostly empty while the composition parallelises over rows and
fills 64 cores. That is the trade-off listed in section 8 (one row per transfer
and per-row reduction when the window is narrower than a row), not a limitation of
the two-level schedule.

## 8. Trade-offs and follow-up work

- The split schedule costs three extra kernel launches and two workspace passes,
  so it is only used when `baseTasks < core count`; shapes with many segments
  keep the fused single-kernel path with no workspace at all.
- Row chunking requires at least 16 rows per chunk
  (`SEGMENT_LOGSUMEXP_MIN_ROWS_PER_CHUNK`); very short inputs stay unsplit and
  remain bounded by the segment count.
- Windows narrower than a whole row (`features > 1024`) still move one row per
  transfer and do not benefit from row batching; that can be enabled once the
  source-stride unit is unified across architectures.
- The forward pass reads `src` twice (maximum, then sum). A single pass would
  need the online recurrence (`m_new = max(m, x)`,
  `s = s * exp(m - m_new) + exp(x - m_new)`), which adds two exponentials per
  element; the current implementation prefers "correct first, faster later".
- Narrow rows (`features` 16–64) still lose on a 64-core platform (0.30–0.71× on
  A5): a row is only 64–256 bytes, so most vector lanes are empty while the
  composition parallelises over rows and fills all 64 cores. Improving this needs
  interleaving several rows into 1024 lanes before reducing; that is a later
  stage.
- On many-segment shapes the runtime is close to the platform floor set by
  per-core transfer bandwidth (about 3.6 ms for two passes over 1M × 128 on A2).
  Going further needs fewer small vector instructions per row, for example
  interleaving 8 rows into 1024 lanes before reducing; that is a later stage.

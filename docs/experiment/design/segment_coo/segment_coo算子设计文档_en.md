# Segment COO design: Ascend 950 and A2

> The original sections below describe the Ascend 950 integration and its
> historical validation. The separate A2 extension section records the new
> backend and current validation scope. Historical results are not new regressions.

This document describes the product interface, data flow and boundaries. The
community design review is [PR #1485](https://gitcode.com/cann/cann-ops-competitions/pull/1485).
The integration originates from acceptance-repository commit
`0cdb291cea227d6c5be03ed8b3cf85c4e9d4e9f9`. Validation results for the product
revision are provided in the product PR's corresponding evidence archive.

## Interface and input contract

The API provides `segment_coo`, `segment_sum_coo`, `segment_add_coo`,
`segment_mean_coo`, `segment_min_coo` and `segment_max_coo`.
`index` contains sorted, nonnegative, in-range int64 segment IDs along its final
dimension. The reduction dimension is `index.dim() - 1`; source rank is 1–8.
Leading index dimensions may be 1 or match the source dimensions. Trailing
feature dimensions share the index.

Supported types are float16, float32, bfloat16, int8, uint8, int32 and int64.
`dim_size` optionally specifies the number of output segments. When `out` is
provided, its corresponding dimension determines that number. Dedicated min/max
functions return `(values, indices)`; the generic interface returns values only.
Unsupported reductions, including mul, are rejected in Python.

Sum/add overwrite a supplied output. Mean includes its initial value before
dividing by the segment count. Min/max compare against the initial output.
Empty segments without `out` are zero; mean/min/max preserve supplied values in
empty segments. Extremum ties choose the later source position. These task
semantics differ from upstream CPU nonzero-output accumulation and first-tie
behavior; separate tests explicitly verify those differences.

## Host data flow

1. Validate device, dtype, rank, broadcasting and output metadata; select a dtype ID.
2. Broadcast and materialize the index, and make the source contiguous. Contiguous
   views can have storage offsets, so check actual source/output addresses for
   32-byte alignment and clone unaligned buffers.
3. Compute batches, reduction length, trailing channels, segments and index
   batches; populate `SegmentCooPlan`.
4. Allocate segment-boundary storage when the selected path needs it and use
   the current PyTorch NPU stream.
5. Enqueue through `OpCommand::RunOpApi`. The lambda captures source, index,
   output, argument and boundary tensors to retain temporary storage until launch.
6. Invoke `SegmentCoo`. Copy temporary output back into a supplied view and return
   the original output, preserving its storage identity.

## Kernel organization

`op_kernel/arch35/segment_coo.h` declares the plan and entry point;
`segment_coo.cpp` contains boundary searches, SIMT reductions and launch selection.
Host declarations and definitions are in `op_host/segment_coo.h/.cpp`.
The entry-point naming follows neighboring `SegmentMaxCsr` and `GatherCoo` operators.

Every call reads the actual index and searches its boundaries. No path assumes
uniform segments or caches previous indices/results. Dtype, channels and address
range select scalar or packed accesses and cooperative sum/mean for long segments.
The aligned INT64 mean path can search indices directly without intermediate
boundary construction. Power-of-two coordinates use equivalent unsigned shifts
and masks; all other shapes retain general coordinate calculations.

FP16/BF16 accumulate in wider floating point; integer reductions retain integer
arithmetic. INT8/UINT8 mean follows upstream narrowing. When a nonempty count
narrows to zero, the implementation defines a mathematical mean and avoids the
CPU reference's division by zero. That boundary is not claimed as CPU equivalence.

## Tests and performance

`test/segment_coo/arch35/` contains three Python files:

- `test_segment_coo.py`: parameterized interface, numerical, boundary, asynchronous
  stream and alignment coverage. Original parameter matrices and assertions are
  retained. Two strided-input cases transfer contiguous backing storage before
  constructing and checking NPU views; internal-format warnings fail those tests.
- `golden.py`: the actual CPU `torch_scatter` reference and its comparable domain.
- `benchmark_segment_coo.py`: a logging version preserving the supplied shapes,
  input construction and timing. Formal validation uses the byte-identical
  `benchmark_supplied.py` retained separately in the evidence archive.

The complete performance matrix has four reductions × 29 shapes × four dtypes,
or 464 cases, with 20 warmups and 100 iterations. Each case must reach the task's
GPU-baseline/NPU-latency ratio of 0.45. The supplied script prints 0.001 ms precision;
the external checker uses `printed_ms + 0.0005` to compute a conservative ratio.
Raw logs, task baselines and the checker are included in validation evidence;
diagnostic scripts, self-test reports and historical binaries stay outside the
product source tree.

CPU-reference coverage and task-specific behavior are documented separately.
Host validation checks metadata without traversing NPU indices; callers must
satisfy sortedness and range prerequisites. This revision targets Ascend 950 and
does not export these operators on other architectures.

## AI assistance

The original implementation and evidence preparation used ZCode/GLM-5.3 and
Codex. GPT-6 (Codex) assisted with this integration, test consolidation and
alignment protection. The contributor is responsible for reviewing the code
and device results.

## Ascend A2 backend design

### Interface and input contract

The arch22 backend provides `segment_coo`, `segment_sum_coo`, `segment_add_coo`,
`segment_mean_coo`, `segment_min_coo` and `segment_max_coo`. The generic interface
returns values; dedicated min/max interfaces return `(values, arg)`.

Source rank is 1–8, with float16, float32, bfloat16, int8, uint8, int32 and int64
support. The int64 index is nonnegative and nondecreasing along its last dimension;
segment labels may have gaps. Index rank satisfies `1 <= index.dim() <= src.dim()`.
Reduction uses `index.dim() - 1`. Each index dimension
may match the source or equal one, including the reduction dimension. Trailing
feature dimensions share the same indices.

The internal layout is `[B, N, C]`: B is the product of preceding dimensions, N
is the reduction length and C is the product of trailing dimensions. Output has
layout `[B, S, C]` and retains the original public shape. A supplied out determines
S; otherwise dim_size is used, or the maximum label plus one is inferred. Empty
indices infer zero segments. Noncontiguous src/index/out are supported, and the
interface rejects unsupported reductions.

| Reduction | Initial values and results |
| --- | --- |
| sum/add | Equivalent operations that accumulate the supplied out seed |
| mean | Accumulate inputs and the seed, then divide by the input count; integer division truncates toward zero |
| min/max | Update values and positions only on strict improvement; ordinary ties retain the first extremum |
| Empty segments/inputs | Zero without out; preserve supplied values with out; empty arg is N |

Each batch's first nonempty segment uses out[0], even if its label is nonzero.
Later segments use their corresponding seeds. Argument state starts at zero per
feature and carries across nonempty segments and batches. Empty segments emit
the N sentinel without changing that state.

### Host data flow

1. Validate device, dtype, rank, reduction and output metadata; resolve the
   reduction dimension and broadcasting relationships.
2. Expand and materialize indices before validating order, nonnegativity and
   bounds. Prove uniform segmentation using the expanded logical shape so that
   strides, segment lengths and scheduling facts agree.
3. Make source/output buffers contiguous and check actual 32-byte alignment.
   Use temporary outputs when needed and preserve source/index copies when they
   alias the output.
4. Construct `SegmentCooPlan` from B/N/C/S, dtype, reduction, initial-value and
   arg requirements, core count and UB capacity. Snapshot extrema seeds on NPU
   for paths that need the original out values.
5. Submit through `OpCommand::RunOpApi` on the current NPU stream. Capture input,
   output and temporary tensors and record cross-stream storage use so their
   lifetimes cover asynchronous execution.
6. Copy temporary results into the caller's original out and return that object.
   Preserve storage outside the view and update the tensor version after writes.

Caches retain validated input properties and tiling plans. Changes in tensor
identity, version, shape or stride require revalidation; inputs whose mutations
cannot be tracked are checked again. Cached entries do not contain reduction results.

### Kernel organization and work partitioning

| File | Responsibility |
| --- | --- |
| `op_host/segment_coo.cpp` | Input normalization, validation, path selection, tiling and output management |
| `op_kernel/arch22/segment_coo_tiling.h` | Shared shapes, strides, tile sizes and execution flags |
| `op_kernel/arch22/segment_coo_reduce.cpp` | Floating-point/INT32 vector reductions, compensation and batching |
| `op_kernel/arch22/segment_coo.cpp` | Byte/INT64 specializations, extrema state scanning and launch dispatch |

Value-only reductions partition work by output segment and feature tile. The
regular path finds `[begin, end)` in sorted indices; proven uniform segments
use direct boundary calculation. Source rows are staged into UB, reduced and
merged across tiles before writeback. Each aligned output region has one owner.

`tileK` controls feature width, `tileRows` controls rows per transfer and
`outputGroup` controls jointly processed segments. The Host budgets input,
conversion, accumulation, compensation, argument and alignment buffers against
UB capacity. Tail transfers cover valid elements, and adjacent unaligned outputs
share core ownership to avoid overlapping writes.

Uniform short segments use batched paths when layout and capacity conditions
hold, reducing repeated transfers and vector submissions. Longer segments are
split into row tiles or feature columns that accommodate a whole segment.
Other inputs use the regular boundary search and reduction paths.

### Reduction and numerical handling

| Type/scenario | Computation |
| --- | --- |
| FLOAT16/BFLOAT16 | FP32 accumulation; mean narrows the numerator and count to the source dtype before division and result conversion |
| FLOAT32 | Input-range and error assessment select source-order or compensated accumulation, preserving cancellation-sensitive accuracy |
| INT8/UINT8 | Integer accumulation and source-width numerator/count conversion; nonempty mean counts narrowing to zero explicitly produce zero |
| INT32 | Integer sums, truncating division and extrema comparison; value-only min/max use integer tree reduction |
| INT64 | Exact digit decomposition and carries for sums, high/low-word extrema comparisons, and exact INT32 narrowing when a range proof permits it |

Compensated floating reductions maintain high/low parts at tree nodes and tile
merges. Input properties select the tile geometry and arithmetic path, with
deterministic segment boundaries and output ownership.

Extrema calls requiring seeds or arguments partition work by feature and scan
segments and batches in source order. Each segment initializes its value
accumulator; only strict improvement updates arg, which carries into the next
nonempty segment. Seed snapshots, comparisons and int64 argument generation
remain on NPU.

Without out, min/max start from finite source-dtype identities and clear results
still equal to the identity to zero. Source NaNs do not update extrema; all-NaN
nonempty segments return zero and retain argument state. Supplied out preserves
its initial-value rules, including NaN seeds. Empty segments are handled separately
from mean division and extrema state updates.

### Tests and performance

The three scripts under `test/segment_coo/arch22/` have distinct responsibilities:

- `golden.py`: an independent PyTorch CPU reference and precision comparator,
  runnable as a standalone file.
- `test_segment_coo.py`: TC01–TC18 coverage for interfaces, types, broadcasting,
  layouts, out, empty segments, special values and argument state, plus
  nondefault-stream dependency checks.
- `benchmark_segment_coo.py`: the established shape matrix and public-interface timing.

Floating-point checks follow the [pinned official mixed-tolerance standard](https://gitcode.com/cann/opbase/blob/d9a3cc9f52c641c25f8b234d37d2c099e6454517/docs/zh/ops_precision_standard/mixed_tolerance_standard.md):
elementwise tolerances, a 99% matched ratio and the maximum-error ULP limit.
Integers and arguments require exact equality. Separate comparisons with the real
CPU extension validate reference seeds, conversions, extrema state and broadcasting.

Performance covers 29 shapes × four data types (FLOAT32, FLOAT16, INT32, INT64) ×
sum/mean/min/max, totaling 464 cases.
All use out=None and value-only min/max. Each case has 20 warmups and 100 timed
iterations, with synchronization around timing and input generation outside it.
The task's external-baseline/NPU-latency threshold is 0.45.

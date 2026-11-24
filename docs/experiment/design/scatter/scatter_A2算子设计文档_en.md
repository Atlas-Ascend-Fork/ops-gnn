<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Scatter Forward Design for Ascend A2

## 1. Scope and reference

This implementation provides `scatter`, `scatter_sum`, `scatter_add`, `scatter_mul`,
`scatter_mean`, `scatter_min`, and `scatter_max` on Ascend A2 (910B, `dav-2201`).
The public signatures are documented in the [English API reference](../../../en/api_reference.md#scatter--scatter-reductions).
The design covers the A2 forward path, retains dispatch to the existing arch35 implementation,
and adds no backward interface. Implementation descriptions and successful compilation are
not substitutes for measurements recorded in a device validation report.

The semantic reference is the original `torch_scatter==2.1.2` CPU public API with PyTorch 2.7.1.
Source review uses pytorch_scatter commit `140d3ad677aae615767412873b90982cbf97d35d`.
The installed extension is identified separately by its import path, shared-library SHA256,
and environment; a package version alone does not establish the binary's source revision.

| Level | src / out dtype | Execution path | Reductions |
| --- | --- | --- | --- |
| L1 | FLOAT16, BFLOAT16, FLOAT32, INT8, INT16, INT32, UINT8 | A2 Ascend C; eligible fused mean or native sum/count composition | sum/add/mul/mean/min/max |
| L2 | FLOAT64, INT64 | Original torch_scatter CPU extension with D2H / H2D | sum/add/mul/mean/min/max |

Indices and dedicated min/max arguments are INT64. Generic `scatter` returns only values;
dedicated min/max return `(value, arg)`. The current native entry computes both value and arg
for extrema; the generic entry discards arg. L1 does not import torch_scatter.
For sum/add/mul/mean, the grouped native entry returns an undefined Tensor instead
of allocating an empty arg Tensor. PyTorch converts that field to Python None;
dedicated min/max still return the device INT64 arg.
L2 accepts PyTorch 2.7.1 and 2.12.0 and requires the original `torch_scatter==2.1.2`
CPU extension compiled for the installed PyTorch. A missing dependency raises an
explanatory ImportError, and an incompatible version raises RuntimeError. PyTorch 2.7.1
remains the original numerical reference; 2.12.0 has passed independent CPU numerical
and L2 equivalence checks. Section 2 describes the known forward autograd difference.
This version list excludes other 2.12.x releases and other versions. Each L2 dtype
emits a fallback warning on first use. FLOAT64 and INT64 reductions are never approximated
by narrowing their values to FLOAT32 or INT32.

## 2. Public semantics

Inputs support ranks 1–8, negative dim, unordered and repeated indices, zero-stride
broadcasting, and non-contiguous src/index. A one-dimensional index aligns with the
reduction axis; other index tensors append singleton dimensions before expansion.
With out, its reduction-axis length determines the output length. Otherwise dim_size
selects that length or it is inferred from the effective index; empty expanded inputs
infer zero. Supplied out must match src in dtype, device, rank, and other dimensions.

| Case | A2 behavior |
| --- | --- |
| out | Initial value participates in reduction and is updated in place; returned identity depends on the grad state described below |
| sum / add | Accumulate into out; FLOAT16/BFLOAT16 sums accumulate in FLOAT32 before narrowing |
| mul | Multiply into out; narrow floating products to the source dtype after each update |
| mean | Follow the CPU public sum/count composition; integer division rounds toward negative infinity |
| min / max | Update value and arg only on strict improvement; ordinary ties retain the first effective axis position |
| arg sentinel | Start at src.size(dim); retain it when no source strictly improves supplied out |
| extrema without out | Clear values still equal to the CPU finite initialization extreme to zero, without replacing their arg |
| Empty bucket in nonempty src | Zero for sum/mean/min/max and one for mul, subject to supplied out and mean's divisor rules |
| Entirely empty src without out | Zero initialization for every reduction, including mul |
| Non-contiguous out | sum/add/mean use a device work buffer and copy back; mul/min/max raise RuntimeError |
| Overlapping tensors | Follow overlap errors detectable by the CPU primitive; permitted independent storage wrappers and aliases with strides preserve observable reads and writes |

Mean counts the **original index before broadcasting**. Both ones and count have src dtype;
overflow or low-precision rounding occurs before replacing `count < 1` with one.
This intentionally differs from an unbounded mathematical average: 128 INT8 ones in one
bucket produce -128, 300 UINT8 ones produce 1, and
`src=[[2,4,6]], index=[0], dim=1` produces `[[12]]`.
NaN, Inf, all-NaN buckets, signed zeros, identity extrema, negative integer means,
and integer wrapping are distinct semantic boundary cases.

### Forward autograd states and errors

The implementation preserves observable forward exceptions and mutation timing without
implementing differentiation. For sum/add/mean, `python/ops_gnn/scatter/_scatter_autograd.py`
calls the A2 native `scatter_check_out` entry, which uses PyTorch `check_inplace` without writing data or
incrementing the Tensor version counter. `torch.autograd.Function.mark_dirty` then leaves
post-write state checks to PyTorch, without Python inspecting private Tensor base/version
fields. Mul's Function returns one Tensor and min/max return two, preserving the runtime's
leaf, view, and inference-state behavior.

The numerical reference remains the original CPU extension with PyTorch 2.7.1; forward
autograd errors follow the runtime. The [PyTorch 2.8 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.8.0)
record the new custom Function leaf-view in-place check, so the table distinguishes that
version boundary.

| Out state and call context | sum/add/mean | mul | min/max |
| --- | --- | --- | --- |
| Requires-grad leaf, grad enabled | RuntimeError before writing | RuntimeError after writing | RuntimeError after writing |
| View of a requires-grad leaf, grad enabled | RuntimeError before writing | Succeeds in PyTorch 2.7.1; RuntimeError after writing from 2.8 onward | RuntimeError after writing |
| Requires-grad nonleaf, grad enabled | Succeeds | Succeeds | Succeeds |
| View of a requires-grad nonleaf, grad enabled | Succeeds | Succeeds | RuntimeError after writing because a view has multiple outputs |
| Ordinary inference Tensor outside inference_mode | Error after sum; mean does not divide | Error after writing | Error after writing |

If inference out is also a requires-grad leaf and grad is enabled, sum/add/mean error
timing additionally depends on input Autograd dispatch. If src or index is an ordinary
Tensor, the leaf precheck raises before writing. When src/index/out are all inference
Tensors, they follow the CPU behavior that bypasses this precheck: the inference-mutation
error follows the sum write, and mean does not divide.

Ordinary out tensors are returned as the original object on success. When a requires-grad
out is used under no_grad/inference_mode, sum/add/mean still return that object with its
requires_grad flag, while mul/min/max return a different Tensor wrapper sharing storage
and having requires_grad=False. The latter wrapper must not be replaced unconditionally
with the original out. L2 copies back under inference_mode before the outer bridge applies
post-write errors. L2 mean with an inference out outside its context executes only CPU sum
before reporting the error. Backward explicitly raises NotImplementedError.

## 3. Python, Host, and build integration

`python/ops_gnn/scatter/` contains the public entry and private adapter modules.
`__init__.py` preserves public signatures and checks whether the native module exports
`scatter_forward`. It dispatches `dav-2201` to `_scatter_a2.py` in the same directory
while retaining existing dispatch for other architectures. The A2 module and native entry
validate metadata and index bounds, and compose mean and L2 execution;
`_scatter_autograd.py` handles forward states and errors. Relative imports access the
root package's `ops_gnn._pybind`; the seven public APIs remain exported from `ops_gnn`.
One-dimensional indices stay compact; other index layouts are expanded and made contiguous.

The seven public Python APIs do not expose `hot_target`. It belongs only to the
shared native `scatter_forward` binding as an arch35 performance hint. A2 retains
the parameter for binding compatibility and checks that it is -1 or nonnegative,
but does not use it for routing or tiling. The dedicated A2 native entry
`scatter_a2_forward` does not expose it.

The A2 C++ implementation adds eight files under `csrc/npu/scatter/`, organized
by responsibility:

| File | Responsibility |
| --- | --- |
| `op_host/scatter_a2.cpp` | Main entry, metadata validation, planning, and grouped forward dispatch |
| `op_host/scatter_groups.cpp` | Index-group cache, validation resources, and asynchronous lifetime management |
| `op_host/scatter_groups.h` | Internal grouped Host types and interface declarations |
| `op_kernel/arch22/scatter.cpp` | Kernel entries, launches, data type dispatch, and exact index-equality validation (`IndexEqualShared`) |
| `op_kernel/arch22/scatter.h` | Both tiling structures and launch declarations |
| `op_kernel/arch22/scatter_common.h` | Numeric utilities, scalar/count paths, count initialization, shared tile transfers, and `SourceTile` description |
| `op_kernel/arch22/scatter_vector.h` | Vector reductions and atomic paths |
| `op_kernel/arch22/scatter_grouped.h` | Stable grouped reduction (`GroupedScatter`) |

Some L1 calls other than mean enter native validation directly. Both src and index must
be ordinary Tensor objects, out must be None, and src must not require gradients. Dim
must be a built-in int in [-8, 8), reduce a supported built-in str, and dim_size either
None or a nonnegative built-in int within INT64 range. Booleans, Tensor subclasses, and
other parameter forms retain the original Python validation order. Native validation
still checks device, dtype, rank, dim, shape, overflow, and index bounds. On a native
RuntimeError, the adapter reruns the original Python `_validate` metadata checks:
a validation failure restores the public exception type and precedence, while a successful
recheck rethrows the original native exception. Successful calls avoid that duplicate Python
check. Supplied-out behavior, forward autograd checks, mean, and L2 keep their existing
composition. This entry optimization adds no backward and does not change reduction routing.

The native entry prefers stable grouping for nonempty src with compact one-dimensional
index when out shares no bytes with src or index. `scatter_groups.cpp` handles index
metadata only: clone a private INT64 device snapshot, copy that snapshot to CPU, check
bounds, construct `rowptr`/rowlist in original source order, and upload the metadata.
Ascend C computes all L1 source-value reductions and outputs on NPU. The CPU never
reduces L1 values. Cold construction, upload, and synchronization are public-call work.

Caching is enabled by default; only `OPSGNN_SCATTER_GROUP_CACHE=0` disables reuse for
cold-cost diagnostics. The cache holds at most eight entries and 64 MiB of snapshot,
`rowptr`, and rowlist. Its key includes an address hint, index sizes/strides/offset,
axis length, output-size hint, device, and current stream. Every hit still compares
all current INT64 index values against the private snapshot. It never relies on a hash,
version counter, Python identity, or an earlier bounds verdict. Each INT64 is reinterpreted
as two INT32 components and compared with native integer Compare EQ, preserving all
64 bits without a floating conversion. The equality kernel uses two 4096-byte raw
input buffers, a 128-byte mask, and a 32-byte flag buffer: 8352 UB bytes in total.
Each core writes its own 32-byte flag block. When supported, EQ also writes the same
block to a registered Host mapping; otherwise the small-result D2H remains. Necessary
transfers and synchronization stay inside interface timing. Host does not poll mapped
memory: it first obtains a completion proof for the current event.

Before any output read or write, an ordinary grouped owner transfers all GPU flags to
UB in one DMA, waits for completion, and checks every core. Any mismatch returns without
writes. Host queues the owner before waiting for validation; a mismatch rebuilds and
reruns within the same public call. Non-contiguous out is copied back only after successful
validation. Cold construction allocates no validation buffer. Eligible no-out FLOAT32
Sum/add cache hits, defined in section 4, instead combine EQ with ordered reduction.
Each core compares its own index partition and computes a private output from the
immutable old plan, without reading other cores' pending flags. Host waits for the
whole fused kernel and any D2H, checks every flag, and publishes the result only when all
index flags report equality.
Mismatch discards the private output and rebuilds, including inferred output dimensions
that grow or shrink with dim_size=None. Cold construction, rebuilds, and supplied-out
calls keep the ordinary route.

Ordinary in-place mutation, independent-storage aliases, inference tensors, address reuse,
and native out writes from other operators must all pass the current value comparison.
Version counters alone are insufficient. Callers still establish cross-stream event
dependencies; caching does not make data races valid. Private snapshots and group metadata
are immutable and fully uploaded before publication.

GPU flags, pinned Host memory, an optional mapping, and a reusable event form an exclusive
lease. Each device retains at most eight idle resources. Reuse matches device, actual
ACL stream, current allocator identity, and sufficient capacity; larger resources use
only the current blocks' valid prefix. Different streams and active tickets do not share
leases or flags. Mapped allocation retains its original base, adds alignment slack, and
selects a 4 KiB-aligned range fully inside that allocation. Registration length rounds up
to whole pages covering the actual core count times 32 bytes. D2H fallback is allowed only
for known unsupported capabilities, or recoverable setup failures with proven safe cleanup
and clear Runtime state. Even a failed registration requires successful unregister before
assuming no mapping remains. Unknown Runtime errors are neither cleared nor treated as
ordinary index mismatches.

Events use aclrtCreateEventExWithFlag with ACL_EVENT_SYNC. Mapped Finish always successfully
calls aclrtSynchronizeEvent before reading Host flags. D2H Finish queries once: a confirmed
COMPLETE skips waiting; otherwise it successfully synchronizes the event before scanning.
The ordinary event covers EQ and any D2H and precedes the owner; the fused event covers
the entire private-output producer and any D2H. Mapped visibility does not use the D2H
Query completion shortcut.

On the ordinary route, the last ticket owner returns a lease after successful Owner
submission to the matching actual stream, validation completion, and Host flag scanning.
The next EQ on that stream follows the old Owner, so old flag reads precede overwrite;
the entire old reduction need not have completed. Ordinary Owner records remain for
source, out, defined arg, `rowptr`, rows, and flags. Pool overflow and exceptional resource
destruction therefore still protect actual consumers through the allocator. The ticket
also retains index, snapshot, and flags Storage until validation completes. Mutable index
still records the stream: after an error, caller Storage resize can replace its allocation,
so retaining StorageImpl alone is insufficient. The immutable snapshot is not an ordinary
Owner argument; `rowptr`/rows record their actual launch stream.

Before any possible fused submission, the ticket retains eight Storage slots for index,
snapshot, flags, source, output, empty arg, `rowptr`, and rows; seven hold Sum allocations.
It sets submitted and hostPending before launch. Index, source, `rowptr`, rows, and flags
keep stream records. Private output and snapshot cannot be rebound by a caller before
full producer completion, so ownership protects them. Only this private output, published
after producer completion, may omit its producer stream record. Successful return to the
pool clears owners; output values are never cached.

A lease without a successful Owner submission mark is cleaned up after proving current
operation completion, rather than pooled. Exception cleanup returns a marked lease only
after event or stream completion is proved. Without a successfully recorded current event,
the actual stream must exclude PERSISTENT, DEVICE_USE_ONLY, and CPU_SCHEDULE flags and
successfully synchronize; an old event state cannot prove current completion. Cleanup on
the original device first establishes no pending access, successfully unregisters a mapping,
destroys the event, frees the original pinned base, and only then releases Tensor/Storage
members. Unproved completion, unregister failure, or another state preventing safe cleanup
retains the whole resource and rejects further hot-validation leases on that device.
No potentially accessed Host address or retained owner is freed separately.

Other routes scan the physical INT64 index with `scatter_index_stats` on NPU: each core
processes 512-element blocks and sends a 32-byte bounds summary to Host. This statistics
route does not cache value proofs. A thread-local resource retains pinned Host memory,
a reusable SYNC event, and the current index/stats Storage. Every summary transfer is
followed by event recording and completion before Host reads. Device switches, capacity
growth, and thread exit clean up on the original device. Partial-submission failures use
the same completion-proof rule above; unproved completion quarantines the whole resource
and rejects further statistics calls on that device. Actual index/stats consumers retain
stream records. L2 separately transfers inputs for its actual CPU
reduction. Empty expanded inputs may skip reduction bounds checks, while mean still
checks nonempty original count indices in CPU order and can raise after its sum stage.

`op_host/scatter_a2.cpp` is enabled by `OPSGNN_DAV_2201`; the original Host file retains
the other architecture. The Host checks shape products, byte-size overflow and overlap,
prepares contiguous tensors, allocates arg/workspaces, and chooses a launch plan.
Tiling uses 64-bit address, element, bucket, and scheduling fields; these do not alter
the source-dtype count semantics of mean. Both A2 Scatter write bindings,
`scatter_a2_forward` and `scatter_forward`, use the CSR `ExternalCsrProofBoundary`
and the SpMM `ExternalSpmmProofBoundary` (`SpmmExternalWriteScope`) so that proof data
potentially modified through independent out storage aliases is invalidated. Both
boundaries cover normal and exceptional exits. Read-only `scatter_index_stats` does
not enter these external-write boundaries.

Launches use the current stream of src's device. The grouped route flushes queued
copies before direct launch; other routes use `OpCommand::RunOpApi`. Storage ownership
and allocator stream recording protect asynchronous input, output, and workspace
lifetimes; private fused results follow the completion-before-publication rule above.
No-out grouped calls write private allocations only, so internal
RecordFunction observation can be disabled locally while retaining the public binding's
CSR and SpMM external-write boundaries. Tensors are normally made contiguous, but permitted
ordered aliases use their original strides. Low-precision overlapping sums use a source
snapshot consistent with CPU accumulation behavior. Necessary copies and permitted out
copy-back are part of the public-interface workload.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CANN_TARGET=a2 NPU_ARCH=dav-2201 python3 -m pip install --no-build-isolation --no-deps -e .
```

Replace the toolkit path when necessary. The current build stores the A2/A3 extension
at `python/ops_gnn/A2-A3/_pybind.so` and uses `libopsgnn_npu_kernel-A2-A3.so`;
950 has separate artifacts under `950/`. Set `CANN_TARGET=a2` to choose that package
explicitly at runtime. Record the actual loaded library path and SHA256 to avoid
mistaking an older worktree's artifacts for the current build.

## 4. Kernel paths and resources

`arch22/scatter.cpp` dispatches seven L1 data types. Add shares Sum; mean preserves the
CPU source-dtype numerator/count rounding and division order. Routing depends on actual
dtype, layout, and platform capacity, without benchmark-shape whitelists or CPU value
reductions for L1 data types lacking a particular atomic instruction.

| Path | Selection and use | Computation and writes |
| --- | --- | --- |
| StableGrouped | Nonempty src, compact 1D index, no byte overlap between out and src/index | CPU builds index metadata only; each NPU bucket/feature owner reduces in original source-row order |
| PrivateFusedSum | Eligible FLOAT32 Sum/add no-out hot hits within stable grouping | Combined EQ and private ordered reduction, published after completion and validation |
| OutputOwner | L1 inputs not assigned another route | Tiles of 64 output elements scan matching indices in source order with a unique writer |
| FeatureOwner | Compact floating feature path retained by the low-level scatter_forward entry | Each core owns a feature tile across all buckets and updates in source order, including sum and mul |
| Atomic extrema | Eligible min/max through the low-level scatter_forward entry | Atomic extrema followed by first-effective-arg selection; never unordered floating sum |
| Integer BucketTable | Low-level scatter_forward with compact 1D integer Sum, flattened dimensions before/after dim both one, length at least 256 | UB bucket tables merge modular integer sums |
| AliasedOrdered | CPU-permitted byte aliases requiring observable update order | One core processes source order, using original strides when needed |

### Stable grouped reduction

Rows within each bucket's rowlist retain original source order. The owner updates its
accumulator one row at a time: source tiles are transfer units, not partial sums to merge.
This avoids floating addition reordering across tiles or cores. FLOAT16/BFLOAT16 sums
accumulate in FLOAT32 before narrowing; low-precision mul narrows after every update.
Min/max use strict improvement and report the original axis row, never the position in
a grouped list. An unimproved supplied out retains its arg sentinel. Integer arithmetic
keeps modular addition/multiplication and floor-division semantics.

An output group within one batch first prefetches adjacent bucket boundaries from `rowptr`.
When the combined row count fits tileRows, it loads row IDs and source once, then reduces
each bucket in its own original order. Larger groups retain the per-bucket loop while
reusing prefetched boundaries; groups crossing a batch boundary use the original loop.
Concatenated bucket rowlists need not be monotonic, so contiguous DMA requires checking
both endpoints and every coordinate. Low-precision floating input is converted to a
complete FLOAT32 work tile before processing any bucket, keeping later inputs intact
when mul/mean reuse source scratch for narrowing. Empty buckets read no source values
and retain the initial-value, count, and arg-sentinel rules.

The current public fused mean route requires floating src, no out, compact 1D index,
nonempty input, axis length at most 2^24, at least 64 trailing features, and row bytes
aligned to 32. It narrows the numerator before division by source-dtype count. A single
physical index contributes count one, regardless of expanded rowlist length. Other mean
calls compose native sum, count sum over the original index, and device division,
including the CPU-visible order of exceptions.

Base feature tiles are multiples of 64, capped at 2048. Source tiles start at 32 rows
and shrink to fit UB; feature width shrinks if necessary. Base output groups have at least
four rows and respect 32-byte DMA blocks. For unaligned row bytes, one owner handles every
feature tile in the group. An unaligned output base uses one core, preventing two cores
from writing the same physical DMA block.

Private fused Sum first qualifies the base plan: FLOAT32 Sum/add, no out, a cache hit,
nonempty output, one tile spanning all features, and at most four buckets per output group.
The actual per-core index partition must fit one 512-element EQ chunk; the cached maximum
bucket length must fit base tileRows; all simultaneously live Owner and EQ buffers must
fit UB. The sum of per-core src and out element counts, each rounded up, must not exceed
eight times the number of FLOAT32 elements fitting actual UB. This is a scheduling limit
on waiting for the complete producer. Other inputs use ordinary grouping; it neither
restricts the legal domain nor changes precision criteria or enumerates benchmark shapes.

After base qualification, tiling tries eight buckets and 64 source rows, reducing source
rows and then feature width when UB requires it. If the minimum eight-bucket plan cannot
fit, or the final actual blocks, EQ coverage, maximum bucket length, complete-feature
coverage, or work budget fails qualification, it retains the already qualified base plan.
Replanning also retains the original single-core restriction from unaligned output.
The fused kernel shares one TPipe, with EQ and Owner buffers simultaneously live.

Only when a same-batch group's combined source rows fit tileRows does private FLOAT32
Sum use waves. Disjoint bucket accumulators occupy the existing output buffer; each wave
adds one source row per bucket before a shared vector barrier. Every bucket starts at
positive zero and adds FLOAT32 values in original source-row order. Waves batch only
synchronization between independent buckets; there are no tree partial sums or floating
atomic sums. Groups exceeding tileRows or crossing a batch retain ordered per-bucket loops.

The Host budgets every simultaneously live UB buffer. Below, W=tileWidth, R=tileRows,
G=outputGroup, V=validationBlocks, and s is the source dtype size. Each item rounds up
to a 32-byte boundary.

| Buffer | Bytes before alignment | Allocation condition |
| --- | --- | --- |
| Source / output | R×W×s / G×W×s | All routes |
| Row IDs / accumulator | 8×R / 4×W | All routes |
| groupPointers | 8×(G+1) | All routes |
| validation | 32×V | Ordinary cache hit, V>0; absent from private fusion |
| Fused EQ | 8352 (4096×2+128+32) | Private fused Sum, sharing the Owner TPipe |
| FLOAT32 work | 4×R×W | Non-scalar FLOAT16/BFLOAT16 |
| INT64 arg group | 8×G×W | Min/max |
| Floating arg/row temporaries and mask | Two 4×W items, plus ceil(W/8) | Non-scalar min/max |
| Mean divisor | 4×W | Non-scalar mean |
| Reserve | 8192 | All routes |

Ordinary validation remains allocated after checking, so its budget includes overlap
with groupPointers and all later computation buffers. Private fusion adds the complete
8352-byte EQ budget to the Owner budget, without consuming the 8192-byte reserve. Waves
use the existing output buffer for disjoint bucket accumulators. Grouped reduction needs
no output-sized FLOAT32 accumulation workspace. Floating arg vectors are used only up
to axis length 2^24; longer extrema use scalar INT64 arg. Core count and UB capacity
come from platform queries.

### Remaining routes

Public composed mean also sends its sums through scatter_a2_forward, so eligible
numerator and count sums use grouping. FeatureOwner, Integer BucketTable, and atomic
extrema remain in the low-level scatter_forward entry; grouping has replaced them for
public compact, nonaliased inputs. These retained routes must not be presented as the
current public performance implementation. They retain initialization, update, and
finalization stages. Atomic arg candidates must equal the final extreme
and strictly improve the initial value; choosing the smallest original row preserves
the first effective arg. Finalization restores source bit patterns or unimproved out,
including NaN and signed-zero behavior. The Host continues to budget their workspace,
alignment, tails, and asynchronous lifetimes. Unused floating Atomic sum/mean branches
have been removed from Kernel dispatch: LaunchAtomic has a compile-time min/max
restriction, and BucketTable dispatch accepts integer Sum only. Floating sum uses
neither unordered atomic accumulation nor floating local-bucket partial-sum merges.

Only index metadata is cached, never src, value, arg, or benchmark-specific results.
Full equality validation, cold construction, necessary copies, and synchronization
belong to the public call. The `T_A100/T_A2 >= 0.6` threshold requires actual per-case
measurements; route names do not establish a performance pass.

## 5. Precision criteria

The normative document is the [clarified opbase standard](https://gitcode.com/cann/opbase/blob/a5e8e716743b20bfb2ffa16e0213c4aa73b823b3/docs/zh/ops_precision_standard/mixed_tolerance_standard.md),
commit `a5e8e716743b20bfb2ffa16e0213c4aa73b823b3`, document SHA256
`bdefa81d0e7b1052c7b8e5727fe88e048cd91a61bbef4fa67be4ae5a54bef023`.

| Dtype | atol | `rtol` | Absolute-error floor |
| --- | --- | --- | --- |
| FLOAT16 | 2^-10 | 2^-10 | 0.1 |
| BFLOAT16 | 2^-7 | 2^-7 | 0.1 |
| FLOAT32 | 2^-13 | 2^-13 | 0.01 |

An element satisfies `abs(actual-golden) <= atol + rtol*abs(golden)`. Each case requires
at least 0.99 matched elements, plus a maximum finite absolute error bounded by
`max(floor, 32*ULP(g_low))`. The anchor g_low is the golden value at the largest finite
error, narrowed to the output dtype. When actual is Inf and golden is finite, RNE
narrowing of golden to a same-sign Inf makes that comparison match.
The independent comparator additionally checks NaN locations, Inf signs, and signed-zero
rules strictly; the 0.99 proportion does not waive those checks. If finite high-precision
golden narrows to Inf while actual stays finite, the unspecified ULP anchor raises an
explicit error instead of yielding an infinite tolerance. The usual same-dtype CPU
reference does not encounter that extended comparison input.

The earlier `experimental_standard.md`, commit `adbe441e9254676174937047457419b6e645e384`,
is reported only as a parallel diagnostic. Since it did not specify the ULP anchor,
that diagnostic explicitly uses the current clarified anchor. Integers, INT64 arg,
and L2 require bitwise comparison. CPU golden self-checks use strict comparison,
not the tolerances intended for NPU arithmetic.

## 6. Validation and performance evidence

The committed test suite uses `golden.py`, `test_scatter.py`, and `benchmark_scatter.py`.
Golden depends only on PyTorch and the standard library, never ops_gnn, torch_scatter,
or arch35 golden. Independent processes compare it with the original CPU extension;
auxiliary tools and raw inputs/results remain in local output. L1 is also validated
with torch_scatter imports rejected; L2 explicitly validates its dependency behavior.

The complete functional and precision suite includes L2 cases and requires the original
`torch_scatter==2.1.2` CPU extension compiled for the installed PyTorch 2.7.1 or 2.12.0.
The numerical reference remains 2.7.1; 2.12.0 has verified CPU equivalence. See the [API build and validation instructions](../../../en/api_reference.md#a2-build-and-validation)
for aarch64 source installation and a CPU extension execution check. A missing dependency
skips only L2 functional cases requiring that extension. L1 cases, L1 checks with
imports rejected, and the L2 missing-dependency error contract still execute. Skips
are counted separately, never as passes; complete acceptance requires installing
the dependency and running every case. CPU equivalence and autograd version-difference
checks do not establish complete device validation for a PyTorch/CANN combination;
actual device results belong in its validation record. The L2 version check accepts
exactly PyTorch 2.7.1 and 2.12.0.

Each unique pytest test identifier records value, arg, and out assertion attempts, passes, failures,
and element counts through `record_property`, together with current/legacy precision
summaries and individual comparison events. Only assertions on actual NPU tensors are
counted; generating CPU references is not a comparison. Rechecking a tensor constitutes
another assertion, not another unique test case. Dependency-isolation subprocess events
are labelled and attributed to their owning test identifier. Pytest outcomes remain authoritative
for failures, skips, and setup errors; zero comparisons cannot establish a pass.
Exact and bitwise verdicts stay unchanged; their mixed-tolerance metrics are diagnostic.
Use `--junitxml=<path> -o junit_family=xunit1` to save these properties.

Coverage includes TC-01–TC-16, nine data types, six reductions, every public entry, broadcasting,
tails, empty tensors, non-contiguous-out errors, identity extrema, cancellation, wrapping,
aliases, and non-default streams. Hot-hit checks also cover inferred output-size changes,
release or storage rebinding of returned outputs, and private-wave special values and order.
Shared-proof regression also warms the SpMM source NaN proof, writes NaN through an
independent storage alias using public Scatter or either native write binding, and
calls SpMM again to verify invalidation and recomputation. This covers FLOAT16 and
FLOAT32; executed and passed counts come from the actual test record for each run.
Large experiments require explicit selection in the test file and are not automatic
ordinary pytest allocations. Shared COO/CSR/SpMM checks
and available 950 build/dispatch checks are recorded separately; absent 950 device
execution must not be described as a successful 950 device test.

Performance uses the 40 published A100 values in the [original task](https://gitcode.com/cann/cann-ops-competitions/blob/ce603dc73335bbb47b3a77d0e4a63fa16f66d136/04_tasks/01_community-task-2026/docs/202608/scatter_task_doc.md).
Its linked self-test sets five bucket counts to 4096, 8192, 16384, 16384, and 16384;
FLOAT16/FLOAT32; sum/mean/min/max; standard-normal src; uniform random INT64 1D index;
dim=0; out=None; and the generic value-only API. It warms up 20 calls, measures 100 public
calls with synchronization before and after the batch, and uses host perf_counter mean time.
All validation, conversions, reductions, and required copies remain timed. This run fixes
and records a seed; obtaining historical A100 logs is not a new acceptance prerequisite.
Before official warmup, the script separately synchronizes one call to record
`first_call_ms`; it then follows the original warmup/iteration protocol. It records
actual cache settings, visible devices, CPU affinity, and thread counts. Metadata is
never preloaded outside the interface. Default-cache and explicit
`OPSGNN_SCATTER_GROUP_CACHE=0` cold runs retain separate results.

Every case and measured round requires `R=T_A100/T_A2 >= 0.6`. Raw times and failed rounds
are retained. This round first compares the 40 published FLOAT32/FLOAT16 cases. Mul,
add, and the remaining L1 data types lack published numeric baselines; their entries remain
available as `missing_baseline`, pending later acceptance when references become available.
Deferred performance work does not reduce functional or precision coverage across data types,
permit substitution from another dtype, or permanently exempt those entries. L2 has no
performance threshold. Changed buckets, out, API, or
iteration settings are development configurations. Development out cases restore their
initial values on every call and explicitly include that copy in timing. The five large
shapes require `--official-shapes`.

```bash
CANN_TARGET=a2 NPU_DEVICE_ID=0 python3 -m pytest test/scatter/arch22/test_scatter.py -q
CANN_TARGET=a2 python3 test/scatter/arch22/benchmark_scatter.py \
  --device npu:0 --dtype float16 float32 --reduce sum mean min max \
  --official-shapes --json output/scatter-a2-performance.json
```

Choose an available idle card, replace the device index in the commands, and complete
correctness checks before separate timing. Record the devices and actual results of
any concurrent multi-device validation separately. Source review of device keys,
DeviceGuard, and recordStream does not establish that test. Available 950 CPU/dispatch
checks are not 950 device passes. Supplementary small-case
accuracy validation also ran through official ATK commit
`45cc4061bc019e73f63a11e96496e97df0d6810c`, using a BaseApi plugin to call the original
CPU API and the frozen public A2 API. The custom comparator uses the approved criteria
above, retains legacy diagnostics, and strictly compares arg, integers, and L2 outputs.
Supplied-out cases separately compare their actual mutations. ATK framework source was
unchanged. Complete CLI/environment records, logs, individual comparisons, tool reports,
source and both native-library hashes are retained in the validation record. These
results do not replace the formal 40-case performance comparison; this design does not
claim that performance has passed.

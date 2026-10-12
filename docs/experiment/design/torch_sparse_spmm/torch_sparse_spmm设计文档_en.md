# torch_sparse SpMM on Ascend 950PR — design

This document describes the ops_gnn implementation. The community-task design
review is recorded in cann/cann-ops-competitions PR #1409.

## Requirement background

The BUPT CANN task asks for sparse-dense graph aggregation on Ascend 950PR in
`ops-gnn/csrc/npu/torch_sparse_spmm`. COO inputs are converted to CSR on NPU;
the resulting CSR graph supports sum/add, mean, min and max. Sparse-sparse
multiplication is outside scope. Public Python indices are int64.

The implementation reuses the SIMT CSR traversal/direct-launch approach from
`cann/ops-sparse` commit `9a514470c2cc99e469c34a615048eb8b30a66694`,
`sparse/spmm/arch35/spmm_kernel.cpp`. That reference repository is unchanged.
The implementation follows the target ops-gnn architecture-specific build layout.

## Requirement decomposition and API

For each batch b, row r and feature k, form messages
`message[e,b,k] = value[e,k] * mat[b,col[e],k]` for `e in [ptr[r],ptr[r+1])`.
Absent value means a multiplier of one without allocating an all-ones tensor.
The output is the chosen reduction of messages. Mean divides by the number of
edges in the row, including duplicate edges.

Public interfaces, registered by importing `ops_gnn.torch_sparse_spmm`:

```text
torch_sparse.spmm(index, value, m, n, matrix) -> out       # COO, sum
torch_sparse::spmm_sum(row, rowptr, col, value, colptr, csr2csc, mat)
torch_sparse::spmm_mean(row, rowptr, col, value, rowcount, colptr, csr2csc, mat)
torch_sparse::spmm_min/max(rowptr, col, value, mat) -> (out, arg)
SparseTensor.spmm/matmul(dense, reduce) -> out
```

`index` is `[2,E]`; row/col are `[E]`; rowptr is `[M+1]`; mat is
`[...,N,K]`; out/arg are `[...,M,K]`. Arg is int64 and uses E for no selected
edge. Value has E leading entries, with 1 or K values per edge; trailing
dimensions are flattened. Other broadcasting is not claimed.

Supported input/output/accumulator combinations:

| Input | Output | Accumulator |
| --- | --- | --- |
| fp32 | fp32 | fp32 |
| fp16 | fp16 / fp32 | fp32 |
| bf16 | bf16 / fp32 | fp32 |
| int8 | int32 / fp32 | int32 / fp32 |

Public defaults preserve floating dtype and use int32 for int8. The helper
`csr_spmm(..., output_dtype=...)` provides explicit mixed output selection.
Whether that helper is accepted for the task's ambiguous output selection is an
open task-owner question; the public ABI has no precision-label argument.

## Detailed implementation

### Host launch and dispatch

The C++ extension validates structural shape/device/dtype constraints, makes
noncontiguous inputs contiguous, allocates out and (only for min/max) arg, and
launches the appropriate typed kernel. It supplies `SpmmShape` containing
M/N/K/E, flattened batch count, value width and reduce id. This is the direct
torch launch mode requested by the task; it does not generate or invoke an
ACLNN/aclsparseSpMM wrapper.

Block count comes from the actual AIV core count. The scalar path uses 512 SIMT
threads; the four-feature path uses 256. A grid-stride loop covers output elements
or four-element groups. Integer offset arithmetic
uses int32 only if flattened input/output/value offsets plus the final stride
fit; otherwise int64 is used. Externally stored indices remain int64.

The stream is obtained after tensor copies/allocations have been queued, because
`getCurrentNPUStream(...).stream()` drains torch_npu's host dispatch queue. This
prevents a direct launch from overtaking pending contiguous copies.

### Kernel and parallel strategy

The general path owns one output element per thread and walks that CSR row's
edges, fusing multiply with reduction. Adjacent threads process adjacent features,
allowing feature reads to be coalesced. Each thread writes its output once;
there are no output atomics and no forward-kernel workspace. Sum/mean use the
specified accumulator, with final casting after reduction. Integer mean uses
truncation toward zero.

For scalar weighted, unbatched inputs with safe int32 offsets and
K=64/128/256/512, the kernel prefetches eight column indices and dense values
before accumulating them in their original edge order. K is a template parameter.
Up to 1024 threads per block are selected in multiples of 32 to balance the
output across the device's AIV cores. Floating mean computes the row reciprocal
before traversal; integer mean retains division with truncation toward zero.
Min/max retains the first edge on ties, ignores NaN products and uses arg=E for
empty or all-NaN rows. Other shapes use the existing general paths below.

For sum/mean with safe int32 offsets, K divisible by four and at least 1,000,000
output elements, each thread instead owns four adjacent features and shares CSR
index traversal/scalar edge weights across four accumulators. Smaller workloads
use the scalar path because the first all-size four-feature experiment regressed
small-case latency. Feature sizes that are powers of two use shifts/masks for
output indexing; a single batch avoids redundant row/batch divisions. Other
shapes retain general indexing, including non-power-of-two tails.

This strategy avoids an E×K materialized message tensor, but long hub rows have
serial dependency chains. Row-splitting/multi-stage reduction is a future
optimization, not part of the current implementation. No unimplemented tiling
strategy is presented as complete.

### COO conversion

The Python adapter first checks original index/value attributes and value's
original E length, then checks row bounds on NPU, stably sorts by row, reorders
col/value, and creates rowptr through an Ascend C binary-search conversion
kernel. Col bounds and CSR validity are checked before SpMM. Torch NPU's sort
may select AiCPU on the device for int64; there is no host CPU tensor fallback.

An LRU cache retains at most 32 (rowptr, sorted col, permutation) layouts. Its key
includes the source index tensor's identity, version counter and M; a weak reference
checks identity and releases the layout when that index dies. Each call gathers
the current edge values and reads the current dense matrix. Inference tensors are
not cached. Clearing validation caches also clears these layouts. Warm timing
includes the edge-value gather and SpMM; cold timing also includes conversion.
For values that do not require gradients, the gather is a direct Ascend C
`PermuteValuesKernel` launch with no extra workspace. This avoids a queued
general-purpose Index operator before the direct SpMM launch. If edge values
require gradients, PyTorch indexing retains the original autograd connection.

### Validation and cached metadata

The device validation kernel checks ptr bounds, monotonicity, first/last entries
and col bounds. Only a small status scalar is returned to host. Public NPU
frontends reject unsupported/bool/complex value types and dtype mismatches before
conversion, including SparseTensor; CPU behavior remains upstream.

Supplied row, rowcount, colptr and csr2csc are checked even in no_grad mode.
Metadata needed for requested gradients is mandatory on the registered API;
the convenience helper builds missing metadata on NPU. Existing SparseTensor
caches are checked before use. Unused, omitted optional metadata is allowed.

Bounded weak-reference caches use parameter positions, tensor identities,
version counters and N. PyTorch mutations invalidate entries. Inference tensors
without version counters are never cached. External writes that bypass version
counters require `clear_validation_cache()`; this is a documented limitation.

### First-order autograd

For sum, `dvalue[e,k] = sum_b grad[b,row[e],k]*mat[b,col[e],k]`; scalar value
also sums over k. Mean additionally divides by the original row's degree.
An Ascend C value-gradient kernel accumulates in fp32 before casting to the
input dtype. Dense gradient is computed as SpMM on the transposed graph, using
CSC colptr and csr2csc; mean scales the upstream gradient by row degrees first.
The helper constructs CSC metadata on NPU. Shared edge weights accumulate over
all leading batch dimensions. Min/max backward and second derivatives are
explicitly unsupported and outside the task's required backward scope.

## Validation strategy

Tests in `test/torch_sparse_spmm/arch35` cover public interfaces, seven precision
combinations, four CSR reductions, COO conversion, optional metadata, batching,
empty graphs, malformed inputs, cache mutation and floating first-order gradients.
CPU FP64 is the floating reference; integer INT32 comparisons are exact.
The benchmark generates inputs locally and reports synchronized warm-call latency.
PyG GCN/GIN forward/backward checks are in the functional test file; optional PyG
dependencies are scoped to those cases. The test directory contains exactly
benchmark_torch_sparse_spmm.py, golden.py and test_torch_sparse_spmm.py.
Task acceptance cases, ATK and profiler tools are retained in a separate PR
validation attachment. Measurements and reports are outside this source tree.

## Integration and compatibility

The historical acceptance build used Ascend950PR_9579/dav-3510, CANN 9.1.0,
Python 3.12.13, torch 2.7.1 and torch_npu 2.7.1.post8. A layout or binding
revision requires a fresh extension build and NPU regression. Build/install
commands are in the [development guide](../../../en/development_guide.md#torch_sparse_spmm). The extension is scoped to Ascend 950.

PyG 2.8's non-CUDA optimization otherwise routes NPU SparseTensor to unsupported
native sparse CSR mm. The adapter patches the SpMM utility and captured GCN/GIN
aliases only for NPU SparseTensor. Tests compare preprocessed-adjacency single
layers and gradients; GCN default normalization/self-loop insertion requires
other sparse utility operators. Import the extension before capturing frontend
aliases.

## Semantics and limitations

Unchanged public schemas preserve upstream output dtype selection; csr_spmm
provides explicit mixed-output selection. Empty min/max rows return zero/E.
Min/max ignores NaN and retains the first equal non-NaN edge, including Inf;
all-NaN rows return zero/E. Integer mean truncates toward zero. These behaviors
are covered by functional tests and described in the [API reference](../../../en/api_reference.md#torch_sparse_spmm).

CPU differential checks verify numerical consistency. Native compilation and
fresh NPU execution remain separate validation steps.

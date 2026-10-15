# KNN operator design

## Function and interfaces

`ops_gnn.knn` and `ops_gnn.knn_graph` provide nearest-neighbor search and
graph construction on Ascend 950, using Ascend C SIMT, PyBind and `opsgnn`.

- Same-device FP16/FP32 inputs `x[N,F]` and `y[M,F]`.
- One-dimensional inputs become columns; noncontiguous inputs are made contiguous.
- Sorted int64 batches become pointers through bucketize; candidates stay in batch.
- NPU int64 output `[2,M*k]` contains query/source indices; empty input returns `[2,0]`.
- Euclidean and cosine distances are supported. `num_workers` preserves the CPU
  torch_cluster API but does not schedule NPU work, which uses AIV cores and the current stream.
- Graph composition supports both flow directions and requests `k+1` neighbors before removing self-loops.

Require `1 <= k <= N` within each batch, or `k+1 <= N` without self-loops.
Large K uses dynamic state within available memory, without an extra k<=100
restriction. NaN/Inf ordering and unsorted batches are not guaranteed.

## Numerical semantics

L2 follows the nanoflann `L2_Adaptor` bundled with CPU torch_cluster 1.6.3.
Squared differences are added left-associatively in groups of four before
adding each group to the accumulator; remaining features are accumulated individually.
FP16 rounds every subtraction, multiplication and addition to FP16 instead
of substituting FP32 accumulation.

```text
s = half(0)
R(v) = round_half_rn(v)
For each four-feature group:
    d[i] = R(y[i] - x[i]); p[i] = R(d[i] * d[i])
    group = R(R(R(p[0] + p[1]) + p[2]) + p[3])
    s = R(s + group)
Remaining features: s = R(s + R(R(y[i] - x[i]) ** 2))
```

FP32 retains the same four-feature grouping in source and uses the repository's
default compiler settings; KNN no longer disables contraction separately.
On Ascend 950, both compiler variants passed 300 functional cases and two
64-case performance suites, with no observed neighbor-set or threshold failure.
This does not prove bitwise equivalence for every near-tie FP32 input.
Sequential FP16 accumulation may select different neighbors and is not an
interchangeable reference for grouped-four arithmetic.

L2 ties follow CPU tree traversal order. Cosine uses FP32 dot products and
norms, distance 1 for zero-norm pairs, and smaller source indices for ties.

## Host and device implementation

Python handles shape normalization, contiguous inputs, batch inference and
the current stream. C++ validates device, dtype, dimensions, K and batch
pointers, allocates output and calls `opsgnn::Knn` under a device guard.
Only batch pointers return to CPU for validation; coordinates, tree
construction, distances and selection remain on NPU.

Each AIV runs 256 SIMT threads, consistent between `LAUNCH_BOUND` and
`VF_CALL`. Block count uses the platform AIV count, capped by M. Each thread
processes independent queries and scans every candidate in the batch,
maintaining ordered Top-K without an M-by-N distance matrix.
Small K dispatches capacities 1/4/8/16/32/64. F=3/16/32 with capacity at
most 32 caches query coordinates; other dimensions use the general loop.
K>64 uses M-by-K FP32 GM distance state with indices in the output. Queries
own disjoint slices. Allocation and launch follow the current-stream
PyTorch allocator lifetime, without cross-call answer caching.

L2 tree metadata reproduces tie order. Root bounds initialize before tree
construction on the same stream. Partition/traversal rules adapt BSD-licensed
nanoflann; see [source scope](../../../en/api_reference.md#knn-third-party-sources) and the
root [LICENSE](../../../../LICENSE). Leaves hold at most ten
points. Two partition phases use `<cut` and `<=cut`; ascending misplaced left
positions pair with descending misplaced right positions, preserving the
original two-pointer permutation. Same-leaf ties use permutation positions;
cross-leaf ties use the lowest common ancestor and query-to-plane comparisons.

Each batch builds in one 256-thread block. The first 7 FIFO nodes are
cooperative, increasing to 31 for batchN>=8192 and 63 for batchN>=16384
when effective dimensions are at most 3. Disjoint remaining subtrees execute
as parallel frontiers. FP16 reference rounding keeps split axis 0, requiring
only first-coordinate bounds; FP32 retains all dimensions.
An 8 KiB UB holds min/max, integer counts and prefixes. Ties retain earlier
positions; barriers protect reductions, scans and frontier publication.
Temporary swap indices reuse unallocated node storage and are no longer
read before later nodes overwrite it. There is no inter-block spinning.

## Complexity and storage

Worst-case work is `O(M*N*(F+k))`, summed per batch. Output uses `O(M*k)`;
small-K thread state is `O(KMax+FeatureCache)` and large K adds `O(M*k)` GM
distances. Trees use at most 2N nodes of eight int64 fields, plus 3N int64
permutation/inverse/leaf entries. Bounds occupy `2N*(2*bbox_features+2)`
floats, where bbox_features is 1 for FP16 and F for FP32. Allocation, tree
construction and search are all included in full API timing.

## Testing and compatibility

Tests live under `test/knn/arch35/`; commands are in the [API reference](../../../en/api_reference.md#knn--knn_graph--exact-neighbor-search).
Functional tests compare complete edge multisets against an independent
PyTorch CPU golden, with a CPU double reference for cosine. Coverage
includes dtypes, batches, K boundaries, ties, graph directions, nondefault
streams and large trees. The golden imports neither torch_cluster nor the
implementation under test. Large CPU queries split into blocks of at most
32 queries, each exhaustively retaining all x candidates; tree metadata only
determines L2 tie order. The NPU
call remains full-shape without answer caching. Separate contracts verify
argument binding, defaults, annotations and graph postprocessing. CPU-only
golden self-tests cover rounding, ties, batch isolation, query blocks,
cosine, graph composition and dependency isolation.

The benchmark covers 32 shapes in both dtypes, with 20 warmups, 100 complete
API calls and synchronization at timing boundaries by default. Each case
requires `reference_ms/measured_ms>=0.45`, without combining best results
from different rounds. Logs and self-test reports accompany PR validation
outside the source tree. The implementation targets Ascend950PR/arch35
(`dav-3510`), without A2/A3 exports or builds. Dependencies are listed in
the [API reference](../../../en/api_reference.md#knn--knn_graph--exact-neighbor-search); cross-device multi-NPU calls are not claimed validated.

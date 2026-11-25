# index_sort / sort_edge_index Operator Design

## Function and Interface

The library exposes two public entry points whose semantics match PyG, both used to group
a COO edge list by node:

```python
index_sort(inputs, max_value=None) -> (sorted_inputs, perm)
sort_edge_index(edge_index, edge_attr=MISSING, num_nodes=None,
                sort_by_row=True) -> Tensor or (Tensor, attribute)
```

- `index_sort` sorts a one-dimensional vector of non-negative integer keys in ascending
  order and returns the permutation `perm`, so any payload can be moved with it.
- `sort_edge_index` takes a `(2, E)` edge list. With `sort_by_row=True` the edges are
  grouped by source node and ascending by destination; with `False` the other way round.
  Edge attributes move with the same permutation, and the return convention matches PyG:
  omitting the attribute returns the edge list only, `None` returns `(edge_index, None)`,
  and a tensor or a list is moved and returned next to the edges.
- Preconditions: keys are non-negative, `max_value` is an exclusive upper bound of the
  keys, and `num_nodes` is at least `edge_index.max() + 1`. All three are caller
  guarantees, as upstream, so the operator does not read the device to verify them and
  does not add a synchronization for it.

## Numerical Semantics

Both entry points guarantee a **stable** sort: equal keys (duplicate edges) keep their input
order, so the permutation is determined by the input alone, independent of the number of AI
cores and of the tiling, and can be compared element by element. Upstream `index_sort` does
not promise stability, so the guarantee here is stronger.

The target order of `sort_edge_index` is equivalent to a stable sort of the composite key
`primary * num_nodes + secondary`. Two formulations are equivalent:

1. One pass: build the composite key and sort it stably, which requires the key to stay in
   the exact range of the sort.
2. Two passes: sort stably by the secondary key, then stably by the primary key. Stability
   keeps the secondary order inside every primary group, so the result is bit-identical to
   the one-pass order while no key ever needs more than `2 ** 24` (A2) or `int64` (950).

The float32 path on A2/A3 relies on one fact: every integer in `[0, 2 ** 24)` is exactly
representable in float32, so sorting and converting back loses no bit. Above `2 ** 24` the
property does not hold, and the implementation falls back to int64.

Differences from PyG: the sort is stable; a composite key that overflows int64 is handled
by two passes instead of `lexsort`; the permutation is not returned (use `index_sort`); an
`EdgeIndex` subclass is treated as a plain tensor.

## Host and Device Implementation

The operator is a composition of native device primitives with no new AscendC kernel, and
dispatches on `_pybind.npu_arch`:

- **arch35 (950, `dav-3510`)**: `argsort(int64, stable=True)` and `sort(int64, stable=True)`
  are native fast kernels there (about 0.29 ms for 1M keys), so the composite key is built
  in int64 and one pass suffices. A `num_nodes ** 2` that overflows int64 switches to two
  passes.
- **arch22 (A2/A3, `dav-2201`)**: CANN's `aclnnSort` only has the `SortAiCore` implementation
  for float32/float16; int32/int64 fall back to `SortAiCpu`. The profiler recorded
  `aclnnSort_SortAiCore_Sort` at 1.48 ms and `aclnnSort_SortAiCpu_Sort` at 214 ms for 1M
  keys. Keys bounded by `2 ** 24` are therefore cast to float32, sorted stably, and the
  sorted values are cast back to int64 losslessly; larger keys fall back to int64. A
  `num_nodes ** 2` within `2 ** 24` takes the one-pass composite key, otherwise two passes.

Data is always moved with `index_select`: advanced indexing `keys[perm]` costs 4.27 ms per
1M elements on A2 while `index_select` costs 0.57 ms, because advanced indexing takes the
generic gather path.

There is a single synchronization point: when `max_value` / `num_nodes` is omitted, or (on
A2) when the supplied bound cannot prove the keys fit in `2 ** 24`, one device scalar is
read to pick the path. Every other call launches asynchronously on the current NPU stream.
An empty input returns empty outputs without launching a sort.

`index_sort` uses `torch.sort` to obtain the sorted values directly, which saves one gather;
`sort_edge_index` only needs the permutation and uses `torch.argsort`, avoiding a
materialized sorted key.

## Complexity and Storage

The sort dominates, so the time is `O(E log E)`, with a constant factor of two for the
two-pass order. The extra storage is the output keys/edge list and the permutation, both
`O(E)`. No bucket or counting table of size `max_value` is allocated, so memory does not
depend on the key range and never reaches the `num_nodes ** 2` scale — which is why the
implementation does not copy the composite-key counting sort.

The float32 path on A2 additionally holds one `E × 4` byte float32 key vector, about 4 MB
for 1M edges, released with the call.

## Testing and Compatibility

Tests live in `test/index_sort/{arch22,arch35}/` and
`test/sort_edge_index/{arch22,arch35}/`, and every directory only holds `golden.py`,
`test_<op>.py` and `benchmark_<op>.py`. The golden is an independent CPU implementation
(stable `argsort`, composite key) that does not import `ops_gnn`. Neither platform has
`torch_geometric` installed, so the semantics are aligned with its source rather than
tested against a PyG runtime.

The three files under `arch35` only re-export and subclass their `arch22` counterparts:
the cases, the reference and the measurement loop are architecture independent, and the
single platform-dependent input — the `num_nodes` that forces the two-pass order, `5000`
versus `4e9` — is derived at run time from `npu_arch`. Both platforms therefore run the
same suite instead of two near-identical copies that drift apart.

Functional tests cover empty input, a single element, all-equal keys, monotonic ascending
and descending vectors, many duplicates, wide tables with few edges, stability of duplicate
edges, both `sort_by_row` directions, inferred and explicit `num_nodes`, agreement of the
one-pass and two-pass orders, the four attribute return conventions, non-contiguous input,
output metadata, determinism, and invalid arguments (dtype, rank, device, negative bound,
attribute row count and device). The two sides of `2 ** 24` have dedicated cases that check
both paths against the golden bit for bit.

The benchmark baseline is the PyG-equivalent composition: a stable int64 `argsort` plus
`index_select`, and the one-pass composite key. On A2, `index_sort` measures 1.50 ms for 1M
keys (baseline 219 ms) and `sort_edge_index` 5.2 ms for 1M edges (baseline 221 ms). On 950
both are in the same range as the baseline, where the value is the API rather than a
different algorithm. The two platforms are measured and reported separately.

Not covered: backward (integer index inputs have no gradient), preservation of `EdgeIndex`
subclass metadata, and active validation of `num_nodes` / `max_value`.

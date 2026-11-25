# csr_transpose Design (A2/A3 arch22 three-pass kernel, 950 native composition)

> Status: implemented. The measured functionality, accuracy and performance live in
> the test report of the corresponding PR; this file covers the interface, the
> semantics, and the implementation and tiling of each platform.
> Dependencies: none. The operator only rewrites two int64 index arrays, adds no
> shared implementation and changes no merged operator.

## 1. Requirement and acceptance

Implement the CSR structural transpose (CSR -> CSC, still expressed as CSR):

```python
out_indptr, out_indices = csr_transpose(indptr, indices, num_cols=None)
```

The input is the two index arrays of an (N, M) CSR matrix and the output the two
index arrays of the (M, N) CSR matrix: every edge `(row, column)` becomes
`(column, row)` and the edge count is preserved.

| Item | Requirement |
| --- | --- |
| Interface | `csr_transpose(indptr, indices, num_cols=None) -> (out_indptr, out_indices)`, the structure half of `torch_sparse.transpose(..., coalesced=True)` |
| dtype | Inputs are 1-D `torch.long` (int64); both outputs are int64 |
| Output | `out_indptr` of length `M + 1`, `out_indices` of length `E`, on the input device |
| Order | The row ids of every output segment ascend (the input is scanned in row-major order), so the result does not depend on the AIV count and is comparable element by element |
| Platforms | Each platform uses the implementation its hardware favours: A2/A3 (arch22 / `dav-2201`) run the three-pass AscendC kernel, 950 (arch35 / `dav-3510`) runs a composition of native primitives (2.3); the public interface and semantics are identical |
| Performance | Net gain over the equivalent "histogram + `torch.argsort`" composition on A2 (4.36x-8.03x measured); on 950 the operator is at the same order as that composition (0.85 ms) and two orders of magnitude faster than the CPU one |

With `num_cols` omitted the row count is inferred as `indices.max() + 1`, which
costs one device scalar read; passing it keeps the call asynchronous.

## 2. Host contract

### 2.1 Arguments and validation

`CheckInputs` only performs what the device needs: both tensors are 1-D, int64, on
the same NPU, and `indptr.numel() >= 1`. The pointer values (not starting at 0,
decreasing, last value not equal to `E`) are **not** validated on the host, in line
with the merged `ptr2ind` (arch35) and `ind2ptr`: a device synchronisation on every
call costs more than it is worth, and the kernels are memory safe for any pointer
value (see 3.1).

`num_cols == 0` together with `E > 0` is contradictory and rejected (the kernel
build raises `TORCH_CHECK` on the host, the composition raises `RuntimeError` in
Python - same error type); for `E == 0` the pointer is zeroed and returned without
any computation.

The Python layer `python/ops_gnn/csr_transpose.py` checks types, rank, device,
dtype and the value of `num_cols` (`TypeError` / `ValueError` / `RuntimeError`),
with the same error semantics as `edge_softmax`.

### 2.2 Core count and workspace

```
blocks    = min(ceil(E / 4096), available AIVs, max(1, M / 4))
workspace = counts[M] + blockTotals[blocks * 4]
```

- one AIV per 4096 edges, so a small matrix stays on few AIVs;
- the `M / 4` cap comes from the alignment rule: column range ends are 32-byte
  (4 int64) aligned, and with more AIVs than `M / 4` a range would go empty - while
  the last range of the prefix-sum pass must be non-empty to emit `out_indptr[M]`;
- `blockTotals` gives every AIV its own 32-byte block
  (`kTransposeBlockStride = 4`) because one scalar store rewrites the whole block it
  lands in, so neighbouring totals must not share one; scalar stores also lose
  values in bulk on A2, so a range total leaves as an 8-byte `DataCopyPad` from a
  block-aligned UB slot like every other write.

### 2.3 Platform split: why 950 does not share the kernel

The write-back contract of the kernel gives every AIV a set of whole 32-byte output
blocks (4), so every AIV has to scan all edges itself to find the order inside its
columns: the work scales with the AIV count. That price pays off on A2, where
`torch.argsort` runs on the AiCPU (about 0.3 s for 1M edges), so 40 AIVs of
amplified scanning still win by 4.5x.

On 950 both premises change: the AIV count is 64, so the amplification grows, while
native `torch.argsort(int64, stable)` is a fast operator of about 0.29 ms for 1M
edges and `scatter_add` about 0.36 ms. On the same graph the kernel measures about
37 ms against about 0.9 ms for the native composition - not a matter of tuning, but
a write-back contract with no chance on 64 cores.

950 therefore implements the operator with those native primitives, producing a
bit-identical contract:

| Step | Implementation on 950 |
| --- | --- |
| row ids | `repeat_interleave(arange(N), indptr[1:] - indptr[:-1])` gives the row of each edge |
| histogram and pointer | `counts.scatter_add_(0, indices, ones)` then `cumsum`, prefixed with 0, gives `out_indptr` |
| order inside a column | a **stable** `argsort` of `indices` selects the row ids in that order |

The stable sort keeps the input order inside every column, and the input is scanned
in row-major order, so every output segment ascends naturally; the result is
independent of the core count and comparable element by element - exactly the
invariant of the kernel.

The dispatch happens in Python: `python/ops_gnn/csr_transpose.py` reads
`_pybind.npu_arch`, calls the device kernel for `dav-2201` and the composition
otherwise. The 950 build neither compiles nor references the kernel:

- with `op_kernel/arch35/` gone, `collect_npu_arch_sources` in `CMakeLists.txt`
  collects nothing for this operator under `arch35`, and its `op_host` sources drop
  out of the 950 build too (that function only adds host sources of operators that
  have a kernel directory for the requested architecture);
- the `csr_transpose` binding in `csrc/pybind.cpp` sits inside
  `#ifdef OPSGNN_DAV_2201`.

This follows the merged `ind2ptr` / `ptr2ind` idea: on a platform whose native
`scatter_add` and stable sort are fast enough, an index-only operator is cheaper as
a composition than as a dedicated kernel.

## 3. Device algorithm (arch22 / `dav-2201`)

Three kernels launch in order on one stream, all `KERNEL_TYPE_AIV_ONLY` (they are
compiled for arch22 only; 950 uses the composition of 2.3):

| Pass | Reads | Writes | Purpose |
| --- | --- | --- | --- |
| count | `indices` | `counts`, `blockTotals` | per-column histogram, one contiguous column range per AIV |
| indptr | `counts`, `blockTotals` | `out_indptr` | exclusive prefix sum per range |
| write | `indices`, `indptr`, `out_indptr` | `out_indices` | append each row id at the cursor of its column |

### 3.1 Per-column histogram (count)

Each AIV owns one contiguous column range `[begin, end)`, 4-column aligned, writes
the counts of its columns and publishes the total of its range into its own
`blockTotals` slot with an 8-byte `DataCopyPad`. A range with more columns than UB
can keep is walked in windows.

### 3.2 Prefix sum (indptr)

Each AIV starts from the sum of the `blockTotals` of the AIVs before it and then
prefix-sums the `counts` of its range. No AIV reads a pointer another AIV is
writing: the cross-range dependency only travels through `blockTotals` written by
the previous kernel.

The range that ends at `M` also emits `out_indptr[M]`; because range starts are
4-column aligned and neighbouring ranges are at least 4 columns apart (see the AIV
cap in 2.2), the last range is never empty.

### 3.3 Write-back (write)

This pass carries the correctness of the operator and has three parts:

1. **column range**: the range end of an AIV is a column whose *output position*
   starts a 32-byte block (`BlockAlignedColumn`). The pointer array is
   non-decreasing, so the next such column is found by jumping to the next column
   with a different pointer value - long runs of empty columns stay cheap.
2. **position chunk**: the output positions of a range are contiguous
   (`out_indptr[begin]` .. `out_indptr[end]`); they are cut into chunks of the rows
   UB can hold. A chunk starts on a multiple of 4, so every store starts on a
   32-byte boundary; the length may be unaligned.
3. **counting sort in UB**: `cursor[column]` keeps the next output position of each
   column of the range in UB, the edges are scanned once, every hit writes its row
   id into `cache[position - chunkStart]` with `SetValue`, and the chunk is stored
   with one `DataCopyPad`.

A range whose positions do not fit one chunk is rescanned once per chunk, so **a
column of several hundred thousand edges runs through the same code as an ordinary
one**, with no special case.

The row id of an edge comes from a cursor over `indptr` that advances while the
edges are replayed: the edges are visited in order, so the row the cursor sits on
is the row of the edge, and no `E`-long row array has to be materialised. `indptr`
is read once per row crossing, with one comparison per edge.

## 4. Alignment, pipeline and platform constraints

Three hardware constraints shape the code; each has a measured justification:

| Constraint | Handling |
| --- | --- |
| One scalar global store rewrites the whole 32-byte block it lands in | Output ranges are block aligned so two AIVs never share a block, and every store goes through `DataCopyPad` |
| A `DataCopyPad` global address must be 32-byte aligned (the length may be unaligned) | Range ends are output-position block starts; position chunks are 4-aligned; the `out_indptr` / `counts` writes start at 4-element boundaries too |
| An MTE source in UB must sit at the start of the buffer (a small copy from a high offset raises an MTE alignment fault) | The row `cache` occupies the base of the UB block and the cursors plus the edge tile follow it |

Every transition uses explicit `SetFlag`/`WaitFlag` events: the visibility between
the scalar pipe and MTE2/MTE3 is not guaranteed by a `PipeBarrier`, which was a
measured trap on A2.

UB budget (A2: 196608 B):

```
reserved        = 32768
edge tile       = 4096 * 8 = 32768
cache + cursor  = 131072
cache (rows)    = 12288 int64
cursor (columns)= 4096 int64
```

## 5. Complexity and known cost

With `E` edges, `M` columns, `C` AIVs and a row cache of `capPos`:

- every pass of every AIV scans **all** edges, so the wall time of the count and
  write passes grows linearly with `E`;
- the write pass additionally rescans per position chunk, with about
  `ceil(E / (C * capPos))` chunks, so the total work is about
  `E * ceil(E / capPos)` - nearly independent of `C` (the AIV count only sets the
  parallelism);
- measured (A2, 40 AIVs, `E = 1M`, `M = 16384`): `82.0 ms`, against `366.5 ms` for the
  equivalent composition (device histogram + `int64 argsort`, 4.47x) and `103.3 ms` for
  the CPU composition. With `M = 200k` at the same edge count the operator takes
  `89.6 ms` against `390.4 ms` (4.36x); for `100k x 1000` with 300k edges it takes
  `13.3 ms` against `107.0 ms` (8.03x);
- 950 does not take this path (2.3): on the same `16384^2` / 1M-edge graph the
  composition measures `0.85 ms` against `104.3 ms` for the CPU composition. The
  amplification therefore only stays hidden where the native sort falls back to the
  AiCPU as it does on A2; on 950 the native composition carries the operator.

That is the price of avoiding a `cores * columns` workspace: keeping a per-AIV
column histogram in global memory would remove the amplification but needs `C * M`
extra memory. The index arrays of graph operators are usually small, so arch22
picks the former and records the per-core histogram as the follow-up optimisation;
950 is unaffected by that amplification because of the split in 2.3.

## 6. Semantic boundaries

| Case | Behaviour |
| --- | --- |
| `E == 0` | All-zero pointer of length `M + 1` and an empty index; `M = 0` when `num_cols` is omitted |
| `E > 0` but `indptr` holds a single element | No row to attribute the edges to; the host raises `RuntimeError` |
| Empty column | Two equal `out_indptr` entries, no output element |
| Duplicate edge `(r, c)` twice | `r` appears twice in column `c`; duplicates are not merged (the `coalesced=False` structure) |
| Shuffled column order inside a row | The output groups by column and ascends by row, independent of the input order |
| `num_cols` above the widest column id | Trailing equal pointers, a legal empty row |
| `num_cols` below the widest column id | Out-of-range columns are never written - a caller contract violation, not checked on the host (see 2.1) |
| Non-monotone pointer | Undefined result: the arch22 kernel stays memory safe but the output is meaningless, while the 950 composition may raise on a negative repeat count (see 2.1) |

## 7. Tests and performance

### 7.1 Test layout

Following `test/<operator>/arch22|arch35`, each directory keeps only `golden.py`,
`test_csr_transpose.py` and `benchmark_csr_transpose.py`:

- `golden.py` rebuilds both index arrays on the CPU with a stable `argsort` and a
  column histogram, without importing `ops_gnn`;
- `test_csr_transpose.py` covers shapes (square, rectangular, very wide, single row,
  single column, diagonal, empty graph, all columns empty, empty rows *and*
  columns, long empty rows, hot column), random graphs (including 45000 columns and
  999x4096), inferred `num_cols`, output metadata, the ascending-rows invariant,
  transpose-twice back to row-major, determinism, non-contiguous input, agreement
  with the `argsort` composition and 11 invalid-argument cases;
- `benchmark_csr_transpose.py` compares the operator with the device `argsort`
  composition on one graph, plus the optional CPU composition and
  `torch_sparse.transpose` when it is installed.

`test/common` gains no files; both architecture directories share the same cases
and reference, and only the measured numbers differ per platform.

### 7.2 Performance protocol

- main case `16384 x 16384` with `degree = 64` (about 1M edges); also
  `200k x 200k` (1M edges, average degree 1) and `100k x 1000` (300k edges) to see
  the effect of column count and degree skew;
- the baseline is the equivalent composition (histogram + stable `argsort`) on the
  same NPU and graph;
- timing with `torch.npu.Event`, 10 warm-up calls and the mean of 20-50 timed calls;
- the report separates A2 and A5 and never mixes them.

## 8. Steps and risks

### 8.1 Steps

1. add `csrc/npu/csr_transpose/{op_host,op_kernel/{common,arch22}}` with the three
   kernels;
2. register `csr_transpose` inside the `OPSGNN_DAV_2201` section of
   `csrc/pybind.cpp` (950 binds no kernel);
3. Python layer `python/ops_gnn/csr_transpose.py` plus the `__init__.py` export,
   dispatching on `npu_arch` to the kernel (`dav-2201`) or the composition (950);
4. the three test files and the documentation (`README`,
   `docs/zh|en/api_reference.md`, this design);
5. build/test/benchmark on A2 (arch22) first, then repeat on A5 (950) once the user
   confirms the VPN switch, reporting each platform separately;
6. open the PR against the issue and wait for CI and review.

### 8.2 Risks

| Risk | Mitigation |
| --- | --- |
| Scalar global stores lose values in bulk (measured on A2) | Write-back goes entirely through `DataCopyPad`, range totals included (8 bytes each, one 32-byte block per AIV) |
| Two AIVs writing one 32-byte block | Range ends are output-position block starts, so no block is shared; tests cover `M` around `4C` |
| A very wide column overflowing UB | Positions are chunked and rescanned; a 200000-edge single column is covered by a case |
| Far more columns than edges (many empties) | `BlockAlignedColumn` jumps to the next column with a different pointer, so empty columns cost no scalar reads |
| Rescan amplification for large `E` | Documented `O(E^2 / capPos)` work plus the per-core histogram follow-up; at the current scale it is still faster than the composition |
| Two implementations drifting apart in semantics | Both architecture directories run the same 40 cases against one `golden.py` (which does not import `ops_gnn`), covering the ascending-segment invariant, transpose-twice, determinism and invalid arguments; the benchmark compares against the same composition on both platforms |
| Mixing platform numbers | The report separates the platforms; the benchmark has no platform default and follows the runtime |

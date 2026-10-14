# grid_cluster Operator Design

## Revision history

| Version | Date | Author | Change |
| --- | --- | --- | --- |
| V0.1 | 2026-09-07 | flying_whj | Initial draft based on the task specification and design template |
| V1.0 | 2026-09-07 | flying_whj | Completed the implementation and documented precision semantics and specialized SIMT paths |
| V1.1 | 2026-09-08 | flying_whj | Removed stale references and refined the implementation design |

# Requirements Background (required)

## Source of the requirement

This design implements a `torch_cluster.grid_cluster`-compatible NPU operator in Ascend C for the Ascend 950 series.

Reference semantics and target locations:

- Python API: `torch_cluster/grid.py`.
- CPU reference: `torch_cluster/csrc/cpu/grid_cpu.cpp`.
- CUDA reference: `torch_cluster/csrc/cuda/grid_cuda.cu`.
- Target repository: `cann/ops-gnn`.
- Operator directory: `csrc/npu/grid_cluster/`.
- Target hardware: Ascend 950 series, `arch35 / dav-3510`.

## Background

`grid_cluster` maps N points in D dimensions onto a regular grid. It computes a grid coordinate in each dimension, then converts those coordinates to a mixed-radix linear `int64` cluster ID. Points in the same cell receive the same ID. Typical uses include point-cloud voxelization, spatial bucketing before graph construction, and neighborhood-search preprocessing.

Points are independent. The dimensions of one point must be processed in order while preserving the CPU reference's subtraction, true division, truncation toward zero, and `int64` overflow semantics. The implementation therefore focuses on:

1. Reproducing intermediate rounding and overflow for each dtype.
2. Specialized parallel paths for the task's common dimensions, D=3/16/32/64.
3. Avoiding Host synchronization and unnecessary intermediate tensors.
4. Reaching `baseline time / measured time >= 0.45` for every prescribed performance case.

Call chain:

```text
ops_gnn.grid_cluster
        |
        v
_pybind.grid_cluster
        |
        v
PyBind / NPU Host implementation
        |
        v
Ascend C arch35 SIMT Kernel
```

# Requirements Analysis (required)

## Operator interface

### Python signature

```python
def grid_cluster(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ...
```

### PyBind and C++ entry point

Python calls `_pybind.grid_cluster(pos, size, start, end)`, which PyBind binds to `opsgnn::grid_cluster_npu`.

### Parameters

| Parameter | Direction | Shape | Dtype | Constraints and meaning |
| --- | --- | --- | --- | --- |
| `pos` | Input | `[N,D]` | float16, bfloat16, float32, int8, uint8, int16, int32 | NPU tensor, `N>=0`, `D>0` |
| `size` | Input | `[D]` | Same dtype as `pos` on the fast path; mixed dtypes use the compatible ATen path | Same device as `pos`; positive finite elements |
| `start` | Optional input | `[D]` | As above | Defaults to `pos.min(0)` on the NPU |
| `end` | Optional input | `[D]` | As above | Defaults to `pos.max(0)` on the NPU |
| `cluster` | Output | `[N]` | int64 | Bitwise equal to the CPU reference |

## Functional requirements

1. Accept seven input dtypes.
2. Accept both bounds omitted, only one bound supplied, or both supplied.
3. Return an empty `torch.long` tensor for empty input.
4. Accept noncontiguous `pos`, made contiguous on the Host side.
5. Execute asynchronously on the caller's current NPU stream.
6. Use one fused Ascend C Kernel for equal-dtype inputs; use ATen type promotion for mixed dtypes.
7. Produce deterministic integer results matching the CPU `torch_cluster` reference elementwise.
8. Cover D=1 through 64 in the task specification; the generic path has no fixed D limit of its own.
9. Meet the 0.45 performance threshold for all prescribed float16, float32, int16, and int32 cases.

# Detailed Design (required)

## Operator analysis

### Mathematical definition

For point `i` and dimension `d`:

\[
c_{i,d}=\operatorname{trunc}\left(\frac{pos_{i,d}-start_d}{size_d}\right)
\]

\[
n_d=\operatorname{trunc}\left(\frac{end_d-start_d}{size_d}\right)+1
\]

Here, `trunc` means truncation toward zero. The linearization stride and ID are:

\[
stride_0=1,\qquad stride_d=\prod_{k=0}^{d-1}n_k
\]

\[
cluster_i=\sum_{d=0}^{D-1}c_{i,d}\times stride_d
\]

CPU-equivalent pseudocode:

```cpp
for (int64_t i = 0; i < N; ++i) {
    uint64_t result = 0;
    uint64_t stride = 1;
    for (int64_t d = 0; d < D; ++d) {
        T delta = dtype_sub(pos[i, d], start[d]);
        T extent = dtype_sub(end[d], start[d]);
        int64_t coord = trunc_to_long(dtype_div(delta, size[d]));
        int64_t count = trunc_to_long(dtype_div(extent, size[d])) + 1;
        result += uint64_t(coord) * stride;
        stride *= uint64_t(count);
    }
    output[i] = int64_t(result);
}
```

### Precision semantics

- `long()` truncates toward zero rather than flooring; negative offsets follow the same rule.
- Subtraction is materialized in the input dtype. For int32, unsigned subtraction followed by conversion to int32 supplies defined modulo-`2^32` behavior.
- Float16 and bfloat16 quotients are written back to the input dtype before conversion to `int64`, matching low-precision intermediate rounding.
- Division computes a float32 quotient and applies one FMA remainder correction, reducing errors by one voxel near integer boundaries.
- Strides and accumulation use `uint64_t` modulo `2^64`, then convert the bit pattern to `int64_t`.

### Dtype mapping

| PyTorch dtype | Kernel type | `Round` | Processing |
| --- | --- | ---: | --- |
| uint8 | `uint8_t` | false | Subtract in dtype, then divide using float32 |
| int8 | `int8_t` | false | Subtract in dtype, then divide using float32 |
| int16 | `int16_t` | false | Subtract in dtype, then divide using float32 |
| int32 | `int32_t` | false | Emulate int32 wrap with unsigned subtraction, then convert to float32 |
| float16 | `half` | true | Float32 helper arithmetic; round the quotient back to half before conversion to int64 |
| float32 | `float` | false | Compute in float32, then truncate to int64 |
| bfloat16 | `bfloat16_t` | true | Float32 helper arithmetic; round the quotient back to bf16 before conversion to int64 |

## Implementation choice

- [x] Ascend C Kernel.
- [ ] CATLASS template library.
- [ ] ACLNN wrapper as the main implementation.

An Ascend C SIMT Kernel fits this operation: each output point is independent, and per-point work consists of multidimensional scalar indexing and integer linearization. It is neither matrix multiplication nor an inter-point reduction. The Host uses ATen/ACLNN only for omitted bounds and mixed-dtype compatibility.

## Host design

Entry point:

```cpp
at::Tensor grid_cluster_npu(
    at::Tensor pos,
    at::Tensor size,
    std::optional<at::Tensor> start,
    std::optional<at::Tensor> end);
```

Execution sequence:

1. Check `pos.dim()==2` and `D>0`.
2. Check that `pos` has one of the seven supported dtypes.
3. Check that `size/start/end` are on the same device as `pos` and are one-dimensional vectors of length D.
4. On the current NPU, check that every `size[d]` is positive and finite; reject invalid sizes even when `N==0`.
5. Allocate an `[N]` int64 output; return immediately when `N==0`.
6. Make the inputs contiguous.
7. Compute per-dimension minimum or maximum when `start` or `end` is omitted.
8. For int8/uint8/int16, widen to int32 without loss for `min/max`, then cast the bounds back. This works around the lack of narrow-integer support in the current 950 `aclnnMinDim/aclnnMaxDim` environment.
9. If all four input dtypes match, obtain the current NPU stream and call `GridCluster`.
10. If metadata dtypes differ, use ATen `true_divide/cumprod/sum` to preserve PyTorch promotion semantics.

Positive `size` is part of the public input contract. The host reads the scalar validation result with `.all().item<bool>()`, which adds NPU-to-Host synchronization; performance must be remeasured.

## Kernel design

### Generic path

`GridSimt<T,Round>` uses 256 threads. A logical thread handles multiple points in a grid-stride loop, iterating over D at runtime:

```text
tid  = blockIdx * threadNum + threadIdx
step = blockNum * threadNum
for i = tid; i < N; i += step:
    compute coord, count, stride, and result dimension by dimension
    out[i] = result
```

This path handles dimensions other than D=3/16/32/64 and is the safe fallback for a very large address range.

### Specialized D=3 path

`Grid3Simt` uses 512 threads, one thread per point. It loads the three-dimensional `start/size` values, computes reciprocals and linearization strides outside the point loop, and unrolls all three dimensions inside it:

```text
cluster = c0 + n0*c1 + n0*n1*c2
```

When `N*D <= UINT32_MAX`, `tid/step/i` use uint32 to reduce address-arithmetic cost in the large D=3 cases. Otherwise, dispatch falls back to the generic int64-index path. Cluster IDs and strides remain 64-bit.

### Wide D=16/32/64 paths

A lane group cooperates on one point:

| D | Group width W | Dimensions per lane | Threads |
| ---: | ---: | ---: | ---: |
| 16 | 2 | 8 | 512 |
| 32, float16/bfloat16 | 4 | 8 | 512 |
| 32, other dtypes | 4 | 8 | 256 |
| 64 | 4 | 16 | 256 |

Each lane caches its assigned `origin/size/reciprocal/stride` outside the point loop, computes a local `uint64` sum inside the loop, and uses `WarpShflDownSync` for reduction. Lane 0 writes the result.

For integer D=16 inputs with positive power-of-two `size[d]` in every dimension, an exact reciprocal-multiplication fast path is used. It still performs CPU-compatible subtraction in the input dtype. This reduces FMA correction work for the task's `size=8` cases.

For D>16, once the prefix product of the first 16 radices is zero modulo `2^64`, all later strides remain zero and later dimensions cannot affect the output. The path then evaluates only the first 16 dimensions. It neither hashes nor renumbers IDs and preserves the `int64` bit pattern.

### Dispatch

```text
N*D > UINT32_MAX ? generic int64-index path
D == 3           ? Grid3Simt<512>
D == 16          ? GridWide512Simt<16,W=2>
D == 32          ? Round ? GridWide512Simt<32,W=4>
                         : GridWideSimt<32,W=4>
D == 64          ? GridWideSimt<64,W=4>
otherwise        ? GridSimt<256>
```

## Tiling strategy

The operator has no separate `TilingData` and does not use conventional GM-to-UB-to-GM tiles. Each dimension contributes one scalar coordinate to index computation; the wide paths keep metadata in registers. Direct GM scalar access and SIMT parallelism suit this implementation.

### Block partitioning

Host calculation:

```text
needed_blocks = ceil(N / 256)
block_num = ResolveAivCoreNum(needed_blocks)
```

Each path uses a grid-stride loop. Wide paths map W lanes to one logical point. Each point is processed once; the loop condition naturally handles the tail without a separate tail-kernel parameter.

### In-kernel partitioning

| Path | Logical point mapping | Tail handling |
| --- | --- | --- |
| D=3 | 1 thread per point | `i < N` |
| D=16 | 2 lanes per point | `i < N`; lane 0 writes |
| D=32/64 | 4 lanes per point | `i < N`; lane 0 writes |
| Generic | 1 thread per point | `i < N` |

### UB tiling

Not applicable. The final path does not instantiate `TPipe/TQue/TBuf`; UB usage is zero. The design template's 32-byte UB alignment, double buffering, and UB capacity formulas do not apply.

## Data flow and memory management

```text
GM(pos, size, start, end)
        | SIMT scalar loads
        v
registers: delta / quotient / reciprocal / stride / partial sum
        | lane shuffle on wide paths
        v
GM(cluster)
```

| Region | Size | Description |
| --- | ---: | --- |
| Input GM | External input | `pos/size/start/end` |
| Output GM | `N * 8` bytes | `int64` cluster ID |
| UB queue/buffer | 0 | No UB staging |
| Custom workspace | 0 | Kernel launch passes `nullptr` for workspace |

## Workspace requirements

There is no custom workspace. The Host allocates the final output. ATen temporary tensors from omitted-bound and mixed-dtype paths are managed by the PyTorch runtime and are not custom Kernel workspace.

## Ascend C API and mechanism mapping

| Step | API/mechanism | Purpose |
| --- | --- | --- |
| SIMT invocation | `AscendC::Simt::VF_CALL` | Launch a vector function from an AIV Kernel |
| Thread indexing | `GetBlockIdx/GetBlockNum/GetThreadIdx/GetThreadNum` | Grid-stride work assignment |
| Division correction | `AscendC::Simt::Fma` | Remainder correction near integer boundaries |
| Lane reduction | `WarpShflDownSync` | Reduction of wide-path partial sums |
| Asynchronous launch | `<<<cores, nullptr, stream>>>` | Submit work on the current NPU stream |

## Determinism and concurrency safety

- One thread or lane group writes each output element.
- No atomics, inter-block dependencies, or inter-kernel synchronization.
- Lane reduction order is fixed.
- Output is an integer bit pattern and repeats exactly for identical inputs.

# Performance Optimization

## Key optimizations

1. Compile-time D=3/16/32/64 specializations remove dynamic dimension loops.
2. D=3 and wide paths use uint32 indices within a safe address range.
3. `1/size[d]` is precomputed outside the point loop and reused.
4. D=16 splits each point across two lanes; D=32/64 across four lanes, reducing per-thread work and register pressure.
5. Integer D=16 with power-of-two voxel sizes uses exact reciprocal multiplication.
6. D>16 skips subsequent zero-contribution dimensions when the prefix is zero modulo `2^64`.
7. The normal Host fast path does not read device values back or synchronize.

## Performance characteristics

| Item | Assessment |
| --- | --- |
| Computation | Scalar index computation; arithmetic and register demand increase with D |
| Memory access | Sequential `pos` reads, repeated reuse of small `size/start/end` vectors, sequential output writes |
| Parallelism | High across points; lane groups cooperate on wide dimensions |
| Main risks | Small-case launch overhead, register/stack pressure at large D, and shared-device performance variability |

# Supported Hardware

| Device | Architecture | Status |
| --- | --- | --- |
| Ascend 950PR | `arch35 / dav-3510` | Built and validated for precision and performance on a real device |
| Other devices | — | Not validated |

Validation environment: CANN 9.1.0, Python 3.12, PyTorch 2.7.1, and torch_npu 2.7.1.post10.

# Operator Constraints and Limitations

1. `pos` must have shape `[N,D]`, with `D>0`.
2. `size/start/end` must be `[D]` tensors on the same device as `pos`.
3. Every `size[d]` must be positive and finite. Zero, negative, NaN, and infinite values raise an error.
4. `start/end` define the grid origin and extent; they do not clip `pos`.
5. The custom Kernel fast path requires all four inputs to have the same dtype; mixed dtypes use the ATen path.
6. If a high-dimensional grid exceeds int64, the CPU reference wraps modulo `2^64`. This implementation preserves its bitwise result, so nonnegative IDs and CPU bitwise equality cannot both be guaranteed.
7. The task tests D=1 through 64. The generic Kernel can accept a larger D, but that is outside the current acceptance commitment.
8. Large benchmark cases check 4096 sampled rows, while small and medium functional tests perform complete elementwise comparisons.

# Observability and Testability

## Precision criterion

The `int64` cluster IDs must pass elementwise `torch.equal` against the CPU `torch_cluster` reference: bitwise equality. Coverage includes:

- Seven supported NPU dtypes.
- D=1/2/3/16/32/64/65.
- All four optional-bound combinations.
- Empty input, identical points, and noncontiguous input.
- `nextafter` boundaries and negative-offset truncation toward zero.
- Integer subtraction overflow and high-dimensional stride overflow.
- D=16 power-of-two voxel sizes and nonzero high-dimensional prefixes.
- Current-stream execution, determinism, and parameter validation.

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Different FP16/BF16 boundary rounding | Incorrect ID | Round the quotient back to the input dtype before conversion to long; add `nextafter` tests |
| int32 subtraction overflow | Undefined signed C++ overflow | Subtract as uint32 and convert back to int32 |
| int64 stride overflow | Undefined behavior or mismatched output | Compute modulo `2^64` using uint64 |
| Narrow-integer min/max unsupported | Default bounds fail | Reduce after lossless int32 widening, then cast back |
| Large-D per-thread pressure | Performance regression | Lane-group cooperation for D=16/32/64 |
| External load on a shared server | Performance variance | Check `npu-smi` before and after tests; retain original CSV and logs |

# Code Layout

```text
ops-gnn/
├── csrc/npu/grid_cluster/
│   ├── op_host/
│   │   ├── grid_cluster.cpp
│   │   └── grid_cluster.h
│   └── op_kernel/arch35/
│       ├── grid_cluster.cpp
│       └── grid_cluster.h
├── python/ops_gnn/
│   ├── __init__.py
│   └── grid_cluster.py
├── docs/experiment/design/grid_cluster/
│   └── grid_cluster算子设计文档.md
└── test/grid_cluster/arch35/
    ├── grid_cluster_golden.py
    ├── test_grid_cluster.py
    └── benchmark_grid_cluster.py
```

# References

- `torch_cluster` CPU, CUDA, and Python implementations.
- Community task document `grid_cluster_task_doc.md`.
- Community operator-design template.
- Ascend C index-oriented and general tiling guidelines.

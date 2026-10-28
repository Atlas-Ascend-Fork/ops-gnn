# matrix_exp Operator Design Document (Ascend 950PR / A5 / arch35)

> Status: Review draft (Iteration 1 verified on silicon: 27 test cases green across all
> norm buckets, 2026-10; algorithm parameters frozen, see 5.4)
> Reference: `.agents/skills/ascendc-docs-gen` detailed-design template; op_api / op_graph
> modules in the template are trimmed to this repository's actual structure
> (op_host + op_kernel + direct pybind registration).

## 1. Overview

### 1.1 Basic Information

| Item | Content |
|------|---------|
| Operator name | matrix_exp |
| Category | BLAS / matrix functions |
| Supported dtypes | float32, float16, bfloat16 (all internal computation in float32) |
| Target chip | Ascend950PR (A5) |
| Target architecture | arch35 (bisheng `--npu-arch dav-3510`) |
| Programming model | MIX_AIC_1_2 cooperation: AIC Cube (CANN SDK tensor_api Mmad, `AscendC::Te`) + AIV Vector, **no SIMT** |
| Interface baseline | `torch.linalg.matrix_exp` |

### 1.2 Operator Functionality

Computes the matrix exponential exp(A) of square matrices. The input is a batch of
square matrices `A: (..., n, n)`, and the output is `Y: (..., n, n)`, where
`Y[..., :, :] = expm(A[..., :, :])`. Semantics align with
`torch.linalg.matrix_exp`: matrices in a batch are computed independently, and the
per-matrix numerical results agree with the torch reference implementation
(scaling-and-squaring Padé [HIM05]) within tolerance, not bit-exactly.

float64 / complex64 / complex128 are not supported (the 950 Cube has no fp64, and
there is no precedent for complex in this repository). This is an **explicit
divergence** from torch: the interface layer raises an error directly. All other
semantics (batching, shapes, the out= argument, empty batch) align with torch.

### 1.3 Mathematical Formulation

```text
exp(A) = sum_{k=0}^{inf} A^k / k!
```

The implementation uses **scaling-and-squaring + Taylor expansion (Horner evaluation)**:

```text
1. Take s = max(0, ceil(log2( ||A||_F / theta ))), let A' = A / 2^s, so ||A'||_F <= theta
   (Frobenius norm; the Taylor remainder bound holds for any submultiplicative norm,
   no need to convert to the 1-norm, see 3.1)
2. exp(A') = sum_{k=0}^{K} A'^k / k!, evaluated by Horner nested multiplication (K-1 Matmuls):
       E = I + (A'/K)·I = I + A'/K      # uses (A'/K)·I = A'/K: Muls+Adds suffice, saves 1 Matmul
       for k = K-1 .. 1:   E = I + (A' / k) * E        # one n×n Matmul per step
3. exp(A) = exp(A')^(2^s): square E s consecutive times (one Matmul per step)
```

### 1.4 Interface Definition

```cpp
// op_host public C++ interface (pybind registration name matrix_exp)
torch::Tensor matrix_exp(torch::Tensor A, torch::Tensor out);   // empty tensor for out means "not provided"

// Python wrapper (aligned with the torch.linalg.matrix_exp signature)
ops_gnn.matrix_exp(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor
```

**Input / output specification**:

| Item | Constraint |
|------|-----------|
| A shape | `(..., n, n)`, at least 2 dims, last two dims must be equal (square), otherwise error |
| A dtype | float32 / float16 / bfloat16, anything else raises an error |
| A contiguity | non-contiguous inputs are converted on host to a contiguous tensor on the current device (an actual copy) |
| out | when provided, dtype / device / shape are validated against the expected output; a non-contiguous out is computed via a contiguous buffer then copied back into the view |
| Output | dtype / shape same as A |
| Edge cases | batch dim 0: return empty directly; n = 1: `exp([[a]]) = [[e^a]]`, takes the pure Vector element-wise Exp fast path (batch dim parallelizes perfectly, bypassing the minimum-shape risk of Matmul), no Cube involved |
| n upper bound (Iteration 1) | n ≤ 128 (`MATRIX_EXP_N_MAX_UB`, the all-resident path limit); n > 128 raises a TORCH_CHECK error, the tiled path is planned for Iteration 3 (3.5) |

**Divergence statement vs. torch.linalg.matrix_exp**:

| Divergence | torch | This operator | Handling |
|--------|-------|--------|---------|
| float64 | supported | not supported | TORCH_CHECK error |
| complex64/128 | supported | not supported | TORCH_CHECK error |
| Numerical results | Padé-13 scaling-and-squaring | Taylor scaling-and-squaring | agree within tolerance (see the accuracy contract in 3.1) |
| inf/nan input | produces inf/nan in some cases | undefined (after the s cap, computation continues numerically) | not an acceptance item, declared in docs |
| autograd | supports backward | not supported (inference only, consistent with other operators in this repo) | declared in docs; backward left to upper layers |
| n > 128 large matrices | supported | Iteration 1 errors out | TORCH_CHECK error; the tiled path comes in Iteration 3 (3.5) |

---

## 2. Architecture Design

### 2.1 Logical View

Follows the unified structure of this repository (no separate op_api / op_graph layers):

| Module | Responsibility | Core files |
|------|------|---------|
| **op_host** | input validation, shape derivation, tiling computation, kernel launch | `op_host/matrix_exp.h`, `op_host/matrix_exp.cpp` |
| **op_kernel** | NPU computation logic (Cube + Vector) | `op_kernel/arch35/matrix_exp.cpp` (launch), `matrix_exp_kernel.h` (core classes), `matrix_exp_tiling.h` (TilingData) |
| **pybind** | Python binding registration | `csrc/pybind.cpp` (registered inside the `OPSGNN_HAS_MATRIX_EXP` macro) |
| **python** | user-facing wrapper with a torch-aligned signature | `python/ops_gnn/matrix_exp.py` |

**Module dependencies**:

```
python/ops_gnn/matrix_exp.py
    └── _pybind.matrix_exp(A, out)          # registered in csrc/pybind.cpp
          └── op_host matrix_exp()          # validation + tiling + launch
                └── MatrixExp(..., dtypeTag, stream)   # launch wrapper (dtypeTag three-way branch)
                      └── matrix_exp_kernel<T><<<coreNum, stream>>>
                            └── MatrixExpAic / MatrixExpAiv   # MIX: Cube + Vector compute
```

### 2.2 Development View (Deliverable File List)

```
csrc/npu/matrix_exp/
├── op_host/
│   ├── matrix_exp.h                 # torch interface declaration (op_host same-name header,
│   │                                 #   quoted include takes priority)
│   └── matrix_exp.cpp               # validation / tiling / launch
└── op_kernel/
    └── arch35/
        ├── matrix_exp.cpp           # launch wrapper MatrixExp (dtypeTag dispatch float/half/bf16)
        ├── matrix_exp_kernel.h      # MatrixExpAic / MatrixExpAiv core classes (MIX: Cube + Vector)
        └── matrix_exp_tiling.h      # MatrixExpTilingData struct

python/ops_gnn/matrix_exp.py        # torch.linalg.matrix_exp-aligned signature
csrc/pybind.cpp                     # OPSGNN_HAS_MATRIX_EXP conditional registration added
test/matrix_exp/arch35/
└── test_matrix_exp.py          # functional / accuracy / edge cases (built-in fp64 reference, see 5.3)
```

Zero build-system changes: CMakeLists.txt glob-collects `op_kernel/arch35`, and the
`OPSGNN_HAS_MATRIX_EXP` macro is auto-generated from "the operator has an arch35
source directory" (per-operator conditional-compilation mechanism, see the
development guide section 4.2).

### 2.3 Runtime View

**Data flow**:

```
GM (A, batch × n×n)
  │ AIV: DataCopyPad (MTE2) → Cast (fp16/bf16 → fp32)
  ▼
UB (A, float32, nPad square)                                 [AIV]
  │ AIV: norm statistics → s/σ; write s into the meta slot; A staging + E0 written to GM ws
  ▼
GM ws per core [A nPad² | E nPad² | tmp nPad² | meta 8]      [AIV MTE3 writes / AIC reads]
  │ AIC: CopyGM2L1 (ND→NZ/ZN) → L1 → L0A/L0B
  ▼
L0C = A@E_j (Mmad × 8 Horner rounds) + E@E (× s squaring rounds)   [AIC]
  │ AIC: fixpipe writes back to the ws tmp region; between Horner rounds the AIV reads tmp
  │      → Muls(σ/k) + E←I + Add → writes the ws E region
  ▼
GM ws (E / tmp regions)
  │ AIV: read tmp (when s>0) → Cast (fp16/bf16 output)
  ▼
GM (out)
```

**Execution flow**:

```
ops_gnn.matrix_exp(A)
  → op_host: TORCH_CHECK square/dtype → contiguous → out allocation/validation
  → ComputeTiling: coreNum(AIC), matrices per core, n, dtype written into MatrixExpTilingData
  → kernel launched on the current NPU stream (no temp stream created, no host-side sync)
  → device, per core: loop over assigned matrices (load in → determine s → Horner → square → write back)
```

---

## 3. Implementation Plan

### 3.1 Algorithm Selection and Accuracy Contract

**Selection comparison**:

| Option | Matmuls per matrix | Extra work | Conclusion |
|------|-----------------|----------|------|
| Padé-13 scaling-and-squaring (torch reference) | 6 | needs to solve (V−U)R = (V+U): LU decomposition + triangular back-substitution, serial per matrix, irregular memory access, costly under the Cube/Vector model | Not adopted (listed as a long-term optimization) |
| **Taylor scaling-and-squaring (Horner)** | K−1+s | only Matmul + element-wise Muls/Adds, all regular coarse-grained computation | **Adopted** |
| Paterson–Stockmeyer piecewise evaluation | saves ≤1 when K=9 | needs resident extra n² buffers for A³/A⁶ etc. (≥5 blocks of 64KB = 320KB > UB 248KB) | Not adopted (does not fit in UB; may be revisited when K grows) |

**Parameters and error bounds** (theta = 0.5, K = 9, i.e. truncated at the A^9 term):

```text
Truncation error  <= e^theta * theta^(K+1)/(K+1)! = e^0.5 * 0.5^10/10!  ≈ 4.4e-10  << fp32 eps (1.19e-7)
Rounding error    <= (K-1+s) floating-point roundings, relative error ~ c·eps per step,
                    typically ~ 1e-6 in total
```

s is determined by the norm: the Taylor remainder bound
`e^{||A||}·||A||^{K+1}/(K+1)!` holds for any **submultiplicative** norm, and the
Frobenius norm is submultiplicative and can be reduced in a single continuous
pass, hence it is **used directly**:
`s = clamp(ceil(log2( ||A||_F / 0.5 )), 0, S_MAX)`, `S_MAX = 30` (defensive cap
against inf/nan inputs). Compared with the "convert the F norm × √n to the
1-norm" formulation (which spends ⌈log2(√n)⌉ extra squarings), the two error
bounds are identical; using the F norm directly at n=128 saves 4 Matmuls — a
zero-cost optimization.

**2^s error amplification (an inherent limitation of fp32 scaling-and-squaring)**:
each squaring roughly doubles the relative error, `err_final ≈ 2^s · err₀`
(err₀ is the Taylor-stage rounding, a few fp32 eps ≈ 5e-7). Beyond s > 4
(`||A||_F ≳ 8`), 1e-5-level tolerances are mathematically unreachable — the
Padé implementation in torch suffers from the same limitation.

**Accuracy acceptance contract (bucketed by `||A||_F`)**:

| Input dtype | `\|\|A\|\|_F` range | Reference | Acceptance tolerance (order of magnitude) |
|-----------|---------------|--------|------------------------|
| float32 | ≤ 8 (s ≤ 4) | fp64 reference (torch Padé, double) | ≤ 1e-5 |
| float32 | (8, 64] | same | ≤ 1e-4 |
| float32 | (64, 1e3] | same | ≤ 1e-3 |
| float32 | (1e3, 1e4] | same | ≤ 1e-2 |
| float16 | all buckets | same (results compared after promotion to the output dtype) | max(bucket tol, 2e-3) |
| bfloat16 | all buckets | same | max(bucket tol, 2e-2) |

The criterion is **hybrid**: `|out − ref| ≤ tol·(|ref| + max|ref|)` — for large
norms the dynamic range of exp is extremely wide (cancellation-dominated small
elements coexist with e^+90-scale large elements), a purely element-wise
relative criterion is not achievable in fp32, and the absolute margin scales
with max|ref| (measured: every bucket passes with an 18~40× margin).
fp16/bf16 are dominated by output quantization (half-ulp ~5e-4 / ~2e-3), hence
max(bucket tol, 2e-3/2e-2).

### 3.2 Template Partition Overview

**Template parameter definitions**:

| Parameter | Type | Value range | Description |
|-----|------|---------|------|
| D_T | DataType | {0: float, 1: half, 2: bfloat16} | input/output dtype; computation always in float32 (launch dispatches on dtypeTag, three branches) |
| N_SMALL | BOOL | {false, true} | reserved field (constantly 0 in Iteration 1, enabled when the three-template small-n path lands) |

**N_MAX_UB derivation** (DAV_3510 vector-available UB = 248 KB; AIV-resident
buffers: three fp32 blocks A, E, TMP + one raw-width ioBuf for fp16/bf16):

```text
fp32:  3 * nPad² * 4B <= 248 KiB                =>  nPad <= 143  =>  N_MAX_UB = 128
n = 128: fp32 3 * 128² * 4B = 192 KiB ✅; fp16/bf16 add ioBuf 32 KiB = 224 KiB ✅
(each block allocated as max(nPad², 8192) fp32 elements, plus a 32B metaBuf)
```

**Template partition table**:

| Template | Trigger condition | Template parameters | Applicable scenario |
|-----|---------|---------|---------|
| Template 1 | fp32 and n ≤ 128 | `D_T=float` (nSmall reserved 0) | Main path: UB fp32 compute state + GM ws staging (AIC/AIV cooperation, see 3.4) |
| Template 2 | fp16/bf16 and n ≤ 128 | `D_T=half/bf16` (same) | Cast to fp32 after load-in, computation identical to Template 1 (delivered in Iteration 1 with the launch three-way branch) |
| Template 3 | n > 128 (any dtype) | `N_SMALL=false` (reserved) | Not implemented in Iteration 1: host TORCH_CHECK error; the tiled path is planned for Iteration 3 (3.5) |

### 3.3 TilingData Structure

**File location**: `op_kernel/arch35/matrix_exp_tiling.h`

```cpp
// Algorithm constants are compile-time defined in matrix_exp_kernel.h:
// MATRIX_EXP_THETA=0.5, MATRIX_EXP_HORNER_ROUNDS=8 (K=9 order, E0 saves 1 Matmul),
// MATRIX_EXP_S_MAX=30.

struct MatrixExpTilingData {
    uint64_t batchSize;       // number of batched matrices = prod(A.shape[:-2])
    uint32_t n;               // matrix order
    uint32_t coreNum;         // participating blocks (= AIC core count under MIX)
    uint32_t matricesPerCore; // matrices per core (floor)
    uint32_t coresWithExtra;  // the first coresWithExtra cores each take 1 extra matrix
    uint32_t dtSize;          // input dtype width in bytes (2/4), for address stepping
    uint32_t nSmall;          // reserved field (unused in Iteration 1, constantly 0)
};
// Core i gets matricesPerCore + (i < coresWithExtra ? 1 : 0) matrices.
// Balanced splitting avoids the underflow (uint32 wraparound) of the last core and
// the idle-core problem of the ceil + last-core-absorbs mode at batchSize=5/coreNum=4.
```

### 3.4 Template 1 / Template 2: n ≤ 128 All-Resident Path

**Implementation revision (Iteration 1, verified against the Blaze skill and ops-tensor sources)**:
This section was originally written for "Cube+Vector hybrid on the AIC with the
matmul::Matmul high-level API"; silicon verification overturned both premises,
and the implementation was revised per the following facts:

1. **The dav-3510 AIC has no V pipe (no-op)**: Vector APIs such as
   Muls/Adds/Cast/Exp/WholeReduceSum can only execute on the AIV; Cube (Mmad)
   and Vector must cooperate via CrossCore. The kernel uses
   `KERNEL_TYPE_MIX_AIC_1_2` (per block 1 AIC + 2 AIV; only the subBlock-0 AIV
   participates in the computation, the subBlock-1 AIV exits immediately).
2. **No matmul::Matmul high-level API on the 950** (A2/A3 only): the Cube path
   uses the CANN SDK tensor_api (`AscendC::Te`, **CamelCase**, not Blaze's
   `asc::te`): `Te::Copy` + `Te::MakeCopy(Te::CopyGM2L1{})` (ND→NZ/ZN
   reordering), `Te::CopyL12L0A/L0B`, `Te::MakeMmad(Te::MmadOperation{})` +
   `Te::MmadParams`, `Te::MakeCopy(Te::CopyL0C2GM{}).with(Te::FixpipeParams{3})`
   (fixpipe write-back); the L0C layout is explicitly BLOCK_CUBE=16 (for fp32,
   LayoutTraitDefault would compute C0=8); Mmad/Fixpipe unitFlag are both 3
   (FINAL_ACCUMULATION; single-shot accumulation and write-back must both be 3).
3. **No fp32 UB→L1 direct path exists** (`CopyPaddedUBToL1` is fp16/bf16 only):
   the AIV-updated E round-trips through the GM workspace back to the AIC
   (layout below), which reorders it via `Te::Copy(CopyGM2L1{})` for Mmad;
   the Mmad result is written back to the GM workspace tmp region via
   fixpipe. The fp32 full-chain compute state is unchanged.
4. **Data and synchronization protocol** (GM workspace per core
   `[A nPad² | E nPad² | tmp nPad² | meta 8]` fp32, nPad=ceil(n/16)·16; all three
   regions are nPad-square layouts, allocated by coreNum on host; the A region
   holds the AIV-staged aligned A, see note 5; the meta slot is written by the
   AIV and scalar-read by the AIC): lockstep via CrossCore MODE4 flags
   (AIV_READY/AIC_DONE, 2 slots each, rotating) — the AIC Set-DONE is bound to
   PIPE_FIX and Wait to PIPE_MTE2; the AIV Set-READY is bound to PIPE_MTE3 and
   Wait to PIPE_MTE2. Flags are strictly paired per matrix: READY = 8+(s>0)
   times, DONE = 8+(s>0) times; in the final round with s==0 the AIC no longer
   waits and the AIV holds the final E and outputs it directly; idle cores and
   the n==1 path make no flag interactions on either side (no deadlock, no
   residue). The AIC has 8 independent in-core event IDs:
   MTE2_MTE1/MTE1_M/M_FIX/FIX_MTE2/MTE2_S for cross-pipe handoff +
   M_MTE1/FIX_M/MTE1_MTE2 single-buffer reverse-dependency (pre-issued in the
   first round); the AIV has 7: MTE2_V/V_MTE3/V_S/V_MTE2/S_MTE2/MTE2_MTE3/
   MTE3_V completion pairs.
   **meta cross-core visibility (proven on silicon)**: the AIC Scalar unit's
   reads from GM hit the in-core DCache, which is not kept coherent with the
   other core's (AIV) MTE3 writes — after the first per-core read misses and
   fetches the new value, all subsequent matrices forever hit the stale line
   and reuse the core's first matrix's s (systematically misaligned squaring
   counts). Before every read one must
   `DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(metaGm_)`
   to invalidate the line and force a GM refetch; volatile only prevents
   compiler CSE and does not fix the hardware DCache (measured ineffective
   alone), kept as a second line of defense.
5. **n not a multiple of 16**: E0 is initialized with a host-generated
   nPad16-square identity matrix (an eye constant, zero tail rows/row tails);
   all three ws regions A/E/tmp use the **nPad-square layout** (not compact
   n×n) — the AIC-side CopyGM2L1(ND→NZ) d_value and the fixpipe row width both
   require ≥ 32B (8 elements for fp32); the compact layout triggers device
   error 507015 when n < 8, and the AIC, having no V pipe, cannot reorder the
   user's compact GM tensor itself, so A is staged by the AIV into the aligned
   A region (mathematically the nPad-order zero-padded matrix); the Mmad m/n/k
   are passed **nPad** (zero-padded fractals do not change the n×n effective
   block result; verified n=2..128 all pass). The eye constant is held in a
   process-level cache keyed by (device, nPad16) (op_host GetEyeConst): a
   call-local tensor races with an asynchronously enqueued kernel over block
   reuse (measured sporadic zero reads); written once via a synchronous H2D
   copy and reused for the process lifetime.

#### 3.4.1 Trigger Condition

`n <= N_MAX_UB(128)`; the dtype determines D_T.

#### 3.4.2 Host-Side Tiling

**File**: `op_host/matrix_exp.cpp` (same pattern as segment_max_csr: pure
ComputeTiling function + LaunchKernel dispatch)

```cpp
// aicCanUseNum is fetched once by the caller and passed in (GetCoreNumAic(), 28 measured on 950PR)
MatrixExpTilingData ComputeTiling(uint64_t batchSize, uint32_t n, uint32_t dtSize, uint32_t aicCanUseNum)
{
    // Balanced split: floor + remainder spread, avoiding the underflow of ceil+last-core-absorbs
    uint32_t coreNum = (aicCanUseNum < batchSize) ? aicCanUseNum : static_cast<uint32_t>(batchSize);
    coreNum = coreNum >= 1 ? coreNum : 1;

    MatrixExpTilingData tiling;
    tiling.batchSize = batchSize;
    tiling.n = n;
    tiling.coreNum = coreNum;
    tiling.matricesPerCore = static_cast<uint32_t>(batchSize / coreNum);
    tiling.coresWithExtra = static_cast<uint32_t>(batchSize % coreNum);
    tiling.dtSize = dtSize;
    tiling.nSmall = 0;  // reserved field (unused in Iteration 1)
    return tiling;
}
```

#### 3.4.3 Kernel-Side Core Flow (MatrixExpAic / MatrixExpAiv cooperation, per matrix)

```text
0. n == 1 fast path: exp([[a]]) = [[e^a]], batch-dim element-wise Exp written back directly
   (pure Vector, perfectly parallel, bypasses the Matmul minimum-shape risk), skips all steps below
1. DataCopyPad: GM(A_b) → UB(aBuf), n×n fp32 (fp16/bf16 loaded at raw width first, then Cast)
2. Norm (AIV): tmp = A*A; tree halving Add reduction (the 3510 lacks the (dst,src,ws,count)
                WholeReduceSum form; the halving src2 offset must be 32B-aligned) + a tail
                segment of ≤16 elements accumulated as scalars on the S pipe → ||A||_F².
                No Sqrt: s follows directly from the equivalence comparison between norm²
                and (0.5·2^s)² (bound initial value 0.25, ×4 per round); σ = 2^-s (an exact
                power of two)
3. Scale folding: no separate scaling pass; A keeps its original values, 2^-s is folded into
                the Horner per-step scalars; the AIV stages A (nPad square) into the ws A
                region (alignment requirements in 3.4 note 5)
4. Horner (K-1 = 8 Matmuls, polynomial to degree K=9; AIC produces products, AIV updates E,
   lockstep via the flags of note 4):
                E = I + (σ/K)·A          # AIV: eye constant + Muls + Add, saves 1 Matmul
                for k = K-1 .. 1:
                    Mmad(A@E_j) → ws tmp  # AIC: A resident in L0A, E reloaded into L0B each round
                    E = I + (σ/k)·(A@E_j) # AIV: read ws tmp → Muls(σ/k) + E←I + Add → write ws E
5. Squaring (pure AIC): for i = 0 .. s-1:  Mmad(E@E) → ws tmp. i=0 reads the ws E region,
                i>0 reads ws tmp (the FIX_MTE2 event guarantees the previous round's fixpipe
                write completes before the read); each round writes ws tmp (not ping-pong;
                single-point write-back + event ordering suffices for ordering)
6. Write-back:   when s>0 the AIV waits for the AIC DONE then reads back from ws tmp;
                Cast (when D_T ≠ float) → row-by-row DataCopyPad UB → GM(out_b)
```

Buffers are reused across the matrix loop (Iteration 1 is single-buffered serial;
the AIC-side L0A/L0B single buffer is reused serially across matrices via the
M_MTE1/FIX_M/MTE1_MTE2 reverse-dependency event pairs; matrix-level
double-buffering is an Iteration-2 optimization, see 4.2).

#### 3.4.4 API Mapping

| Compute step | Ascend C API | Parameter signature (key points) | Platform verified | Constraint notes | Alternative |
|---------|-------------|-----------------|---------|---------|---------|
| Data load-in | DataCopyPad\<T\> | GM→UB row-wise (blockCount=n) + dstStride controls row pitch nPad | ✅ silicon-verified | 3510 rightPadding supports only ≤1 aligned tail block; UB→GM non-aligned srcStride triggers device error 507015 (measured crash at n=2..7/17) → padding=0 + dstStride / row-wise copy-out | DataCopy (usable when everything is 32B-aligned; used for ws region transfers) |
| dtype promotion | Cast\<fp32, T\> | in CAST_NONE / out CAST_RINT | ✅ silicon-verified | fp16/bf16 → fp32 | - |
| Element-wise scalar mul/add | Muls\<float\> / Adds\<float\> / Add\<float\> | (dst, src, scalar/value, count) | ✅ silicon-verified | count = bufElems (=max(nPad², 8192), zero-padded tail); Horner scalar = σ/k (scale folding); E←I via Adds(eBuf, eye resident, 0.0) copy | - |
| Norm reduction | tree halving Add\<float\> (the 3510 lacks the (dst,src,ws,count) WholeReduceSum form) | halving step aligned to 8 elements (src2 offset must be 32B-aligned, else error code 340); tail ≤16 elements accumulated as scalars | ✅ silicon-verified | no sharedTmpBuffer needed; norm² compared directly against (0.5·2^s)², no Sqrt | block-wise accumulation after Abs |
| Scalarizing s | LocalTensor GetValue (S-pipe scalar accumulation) + while(normSq > bound) equivalence comparison | - | ✅ silicon-verified | s is a per-matrix scalar controlling the squaring loop count; written to the ws meta slot via Duplicate + DataCopy (the AIC must invalidate DCache before reading, see 3.4 note 4) | - |
| n=1 exponential | Exp\<float\> | (dst, src, count), chunked by 8192 elements | ✅ silicon-verified | n=1 fast path only (pure AIV, no flag interaction) | Taylor path fallback |
| Identity diagonal write | host-generated eye constant (nPad16 square) -> DataCopyPad full-block copy | - | ✅ adopted (Iteration 1) | tail rows/row tails zero double as pad clearing; Muls+Add realize E0 = I + (sigma/9)A | Dup strided diagonal write (revisit after silicon verification) |
| Matrix multiply | Mmad (CANN SDK tensor_api `AscendC::Te`; the 950 has no matmul::Matmul high-level API) | `Te::MmadParams{nPad, nPad, nPad, unitFlag=3, cmatrixInitVal=true}`; `Te::Copy(CopyGM2L1{})` ND→NZ/ZN | ✅ silicon-verified (n=2..128, accuracy green across buckets) | L0C layout explicitly BLOCK_CUBE=16 (for fp32, LayoutTraitDefault computes C0=8); m/n/k passed nPad; Mmad/Fixpipe unitFlag both 3 | Vector dot-product for small n (slow, fallback) |
| Data copy-out | Cast\<T, fp32\> (CAST_RINT) + DataCopyPad\<T\> | same as load-in, direction UB→GM (row-wise) | ✅ silicon-verified | - | - |
| Synchronization | SetFlag/WaitFlag (in-core event pairs, AIC 8 / AIV 7 independent eventIDs) + CrossCore MODE4 flags (AIC↔AIV lockstep) | HardEvent enums | ✅ silicon-verified | the AIV-side GetBlockIdx() must be divided by GetTaskRation() to get the logical block (rawIdx contains the subblock id; without conversion, odd blocks go unprocessed/hang; SDK production kernels do the same) | - |


#### 3.4.5 Data-Flow Design

```
Per matrix (AIC/AIV lockstep via the flag protocol of 3.4 note 4):
1. AIV: GM(A_i) → UB(fp32); norm → s/σ; stage A → ws A region; E0 → ws E region; s → ws meta slot
2. AIC: ws A/E → L1(NZ/ZN) → L0A/L0B → Mmad → L0C → fixpipe → ws tmp region
   AIV: ws tmp → UB → E update → ws E region (each Horner round, 8 rounds total)
3. s squaring rounds (pure AIC): ws E/tmp region → Mmad(E@E) → ws tmp region
4. AIV: ws tmp (s>0) or UB eF32 (s==0) → Cast → GM(out_i)
```

#### 3.4.6 Memory Management

| Memory region | Size formula (n ≤ 128) | Notes |
|---------|--------------------|------|
| **UB aBuf** | max(nPad², 8192)×4B (64 KB at n=128) | input matrix fp32 compute state / eye resident |
| **UB eBuf** | same as aBuf | E accumulation result (UB compute state, resident before writing ws) |
| **UB tmpBuf** | same as aBuf | AIC product read-back staging / norm-square |
| **UB ioBuf** | max(nPad², 8192)×sizeof(T) (32 KB for fp16/bf16; absent for fp32) | raw-dtype load-in / output Cast staging |
| **UB metaBuf** | 8×4B = 32 B | s scalar transfer (Duplicate → DataCopy writes the ws meta slot) |
| **L1** | 2×nPad²×4B (A NZ region at offset 0 + E B-side NZ region at offset nPad²·4B) | CopyGM2L1 reorder target, explicitly managed offsets |
| **L0A/L0B/L0C** | nPad²×4B×2 + nPad²×4B (L0C layout BLOCK_CUBE=16) | explicit tensor_api layouts; Mmad unitFlag=3 |
| **Workspace** | coreNum × (3·nPad² + 8) × 4B fp32 | per core [A \| E \| tmp \| meta] (see 3.4 note 4) |

**UB total verification**: fp32 three blocks 192 KB (n=128) + metaBuf 32 B ≈
192 KB ≤ 248 KB (DAV_3510) ✅; fp16/bf16 add ioBuf 32 KB = 224 KB ≤ 248 KB ✅.

### 3.5 Template 3: Large-Matrix Tiled Path (n > 128)

#### 3.5.1 Trigger Condition

`n > N_MAX_UB` (any D_T).

#### 3.5.2 Plan Outline

- The Horner / squaring Matmuls are tiled automatically by the Matmul API per
  L1/L0 capacity, each multiplication streamed as n≤256 tiles; after scale
  folding, **the original A is read directly from GM** (no A' copy needed);
  the E intermediate is staged in place in `out[b]` (for fp32 output) or in
  an fp32 workspace (for fp16/bf16 output; half precision cannot hold the
  intermediate); the workspace shrinks to `coreNum × n² × 4B` (one Matmul C
  write slot T per core, ping-ponging with the E slot) + an fp16/bf16-only
  `batchSize × n² × 4B` E staging area;
- Norm, scaling, and Cast are all done in row blocks (each block several
  rows × n columns, block height determined by UB);
- Expected performance is significantly below the all-resident path; provided
  for functional coverage only, no performance gate;
- Iteration-1 status: **implemented as "host TORCH_CHECK error when n > 128"**
  (op_host validates against `MATRIX_EXP_N_MAX_UB=128`); the tiled path is
  left for Iteration 3 (the review's "error out first" option, landed).

---

## 4. Performance Optimization

### 4.1 Parallelism Strategy

- The batch dim is evenly distributed over all AICs (`GetCoreNumAic()`); a
  single matrix never spans cores (the intra-matrix Horner→squaring dependency
  chain is serial; cross-core splitting yields no gain);
- When batch < coreNum the actual core count shrinks to batch (see
  ComputeTiling).

### 4.2 Pipeline Design

- Iteration 1 is **single-buffered serial across matrices** (the AIC-side
  L0A/L0B single buffer is reused across matrices via the M_MTE1/FIX_M/
  MTE1_MTE2 reverse-dependency event pairs); matrix-level double buffering
  (overlapping the next matrix's A load-in with the current Horner's first
  Mmad) is an **Iteration-2 optimization item** — under the MIX data flow the
  UB requirement must be re-estimated for the four buffers
  aBuf/eBuf/tmpBuf/ioBuf (fp32 n=128: 4×64KB = 256KB > 248KB exceeds the limit);
- In-core event pairs guarantee serial correctness across pipe segments (AIC:
  MTE1/MTE2/M/FIX/S; AIV: MTE2/V/MTE3/S); Cube(AIC)↔Vector(AIV) lockstep via
  CrossCore MODE4 flags (3.4 note 4); the buffer-reuse style matches the
  event-pairing style of segment_max_csr;
- The scale factor σ = 2^-s is an exact power of two, no rounding; in the
  Horner scalar σ/k, 1/k (when k is not a power of 2) carries ~eps scalar
  rounding, already budgeted in the 3.1 rounding estimate; does not affect
  the contract.

---

## 5. Risk Assessment

### 5.1 API Risks

| Risk | Impact | Mitigation |
|------|------|------|
| 950 Cube fp32 enables hf32/TF32 by default (low-precision mantissa) | Accuracy misses the target | **Not reproduced in Iteration-1 measurements**: fp32 Mmad accuracy meets the ≤8-bucket 1e-5 contract (27 cases green, ~30× margin); if accuracy regresses later, investigate the switch bit / a two-Mmad fp32 high-low compensation |
| Matmul support gap for fp32 in/out or odd n | Path unusable | **Resolved**: tensor_api Mmad fp32 measured usable; odd n handled via the nPad-square layout (padded to a multiple of 16, not merely even; 3.4 note 5) |
| WholeReduceSum large-repeat segmentation limits | Wrong norm computation | **Avoided**: the 3510 lacks this API form; switched to tree halving Add reduction (32B-aligned steps) + scalar tail accumulation (3.4.4); covered by n=2..128 measurements |

### 5.2 Accuracy Risks

| Risk | Description | Mitigation |
|------|------|------|
| Taylor vs. torch Padé result differences | Theoretically comparable in a backward-stable sense, but element-wise differences exist | Acceptance compares against an fp64 reference under the 3.1 tolerances, not bit-exact against torch |
| Rounding accumulates with s and K | Each squaring ~×2 relative error, err ≈ 2^s·err₀; at norm 1e4, s≈15 (amplification ~3e4×) | Truncation error 4.4e-10 leaves ample margin; the contract buckets by `\|\|A\|\|_F` with the hybrid criterion (3.1, absolute margin scales with max\|ref\|); this is an inherent limitation of fp32 scaling-and-squaring (torch Padé has it too); measured: every bucket passes with an 18~40× margin |
| fp16/bf16 output double rounding | torch casts after internal accscalar(float), same behavior | Same "compute in fp32, cast at the end", aligned with torch |

### 5.3 Reference Implementation (Delivery Form)

The fp64 reference is built directly into `test/matrix_exp/arch35/test_matrix_exp.py`
(`_golden`): `torch.linalg.matrix_exp(A.detach().cpu().double())` (CPU, fp64
Padé) — the fp64 reference is independent of the algorithm under test (Taylor
vs. Padé under fp64 differ far below tolerance), so no separate golden.py is
needed; the criterion is the 3.1 bucketed hybrid tolerance. Interface-level
behavior checks (shape/dtype/errors/out=/non-contiguous/empty batch/repeated
launch determinism) are covered in the same file.

### 5.4 Mitigation Summary

Iteration-1 accuracy review point (2026-10, **passed**): fp32 meets the 1e-5
contract in the `||A||_F ≤ 8` bucket (measured ~30× margin under the hybrid
criterion, 27 cases green across buckets); the algorithm parameters θ=0.5 /
K=9 / S_MAX=30 are **frozen**; if accuracy regresses later, investigate hf32
first, then raise K to 13 (+4 Matmuls).

---

## 6. Deliverables Checklist

**Required** (this repository's structure):

| Category | File |
|------|------|
| op_host | `csrc/npu/matrix_exp/op_host/matrix_exp.h` / `matrix_exp.cpp` |
| op_kernel | `csrc/npu/matrix_exp/op_kernel/arch35/matrix_exp.cpp` / `matrix_exp_kernel.h` / `matrix_exp_tiling.h` |
| pybind | `csrc/pybind.cpp` registration (automatically gains `OPSGNN_HAS_MATRIX_EXP` conditional compilation) |
| python | `python/ops_gnn/matrix_exp.py` + `__init__.py` export |
| tests | `test/matrix_exp/arch35/test_matrix_exp.py` (built-in fp64 reference, see 5.3) |
| docs | `docs/zh\|en/api_reference.md`, `docs/zh\|en/development_guide.md`, README additions |

**Optional**: English version of the design document
(`matrix_exp算子设计文档_en.md`, delivered with this PR).

---

## 7. Iteration Plan

| Iteration | Goal | Code development | Verification |
|------|------|---------|------|
| Iteration 1 | All-dtype all-resident path working | Template 1/2 (fp32 + fp16/bf16, n ≤ 128) + n=1 pure-Vector fast path + TilingData + launch + pybind + python wrapper; all items of the silicon checklist closed | **Completed** (2026-10): n∈{1,2,3,8,15,16,17,32,128} × norm buckets (0.1~1e4) × three dtypes, 27 cases green |
| Iteration 2 | Double-buffer pipelining | Matrix-level double buffering + event pipelining (fp16/bf16 and edge cases already delivered with Iteration 1: launch three-dtype branches, empty batch/n=1/non-contiguous/out= coverage) | Performance data recording and tuning |
| Iteration 3 | Large matrices + wrap-up | Template 3 tiled path (n>128 currently errors out, see 3.5) | Large-n functional coverage; performance data recording (batch×n matrices); docs (api_reference / README / dev guide) wrap-up |

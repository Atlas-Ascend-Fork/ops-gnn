# matrix_exp 算子设计文档 (Ascend 950PR / A5 / arch35)

> 状态：评审稿（迭代一真机验证通过：27 用例分桶全绿，2026-10；算法参数已冻结，见 5.4）
> 参考：`.agents/skills/ascendc-docs-gen` 详细设计模板；模板中 op_api / op_graph 等模块按本仓库
> 实际结构（op_host + op_kernel + pybind 直连）裁剪。

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | matrix_exp |
| 算子类别 | BLAS / 矩阵函数 |
| 支持数据类型 | float32、float16、bfloat16（内部一律以 float32 计算） |
| 目标芯片 | Ascend950PR（A5） |
| 目标架构 | arch35（bisheng `--npu-arch dav-3510`） |
| 编程模型 | MIX_AIC_1_2 协同：AIC Cube（CANN SDK tensor_api Mmad，`AscendC::Te`）+ AIV Vector，**不使用 SIMT** |
| 接口基准 | `torch.linalg.matrix_exp` |

### 1.2 算子功能

计算方阵的矩阵指数 exp(A)。输入为批量方阵 `A: (..., n, n)`，输出
`Y: (..., n, n)`，其中 `Y[..., :, :] = expm(A[..., :, :])`。语义对齐
`torch.linalg.matrix_exp`：批量独立计算、逐矩阵数值结果与 torch 参考实现
（Padé 缩放平方法 [HIM05]）在容差内一致，不要求逐位一致。

不支持 float64 / complex64 / complex128（950 Cube 无 fp64，complex 本仓
无先例），属于与 torch 的**显式差异**，接口层直接报错。其余语义（批量、
形状、out= 参数、空 batch）与 torch 对齐。

### 1.3 数学公式

```text
exp(A) = sum_{k=0}^{inf} A^k / k!
```

实现采用**缩放平方 + Taylor 展开（Horner 求值）**：

```text
1. 取 s = max(0, ceil(log2( ||A||_F / theta )))，令 A' = A / 2^s，保证 ||A'||_F <= theta
   （Frobenius 范数；Taylor 余项界对任意次乘性范数成立，无需折算 1-范数，见 3.1）
2. exp(A') = sum_{k=0}^{K} A'^k / k!，按 Horner 嵌套乘法求值（共 K-1 次 Matmul）：
       E = I + (A'/K)·I = I + A'/K      # 利用 (A'/K)·I = A'/K：Muls+Adds 即可，免 1 次 Matmul
       for k = K-1 .. 1:   E = I + (A' / k) * E        # 每步一次 n×n Matmul
3. exp(A) = exp(A')^(2^s)：将 E 连续平方 s 次（每步一次 Matmul）
```

### 1.4 接口定义

```cpp
// op_host 对外 C++ 接口（pybind 注册名 matrix_exp）
torch::Tensor matrix_exp(torch::Tensor A, torch::Tensor out);   // out 传空张量表示未提供

// Python 包装（对齐 torch.linalg.matrix_exp 签名）
ops_gnn.matrix_exp(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor
```

**输入 / 输出规格**：

| 项 | 约束 |
|-----|------|
| A 形状 | `(..., n, n)`，至少 2 维，最后两维必须相等（方阵），否则报错 |
| A dtype | float32 / float16 / bfloat16，其余报错 |
| A 连续性 | 非连续输入在 host 转为当前设备连续张量（产生实际拷贝） |
| out | 提供时校验 dtype / device / shape 与期望输出一致；非连续 out 通过连续缓冲计算后 copy 回视图 |
| 输出 | dtype / shape 与 A 相同 |
| 边界 | batch 维为 0：直接返回 empty；n = 1：`exp([[a]]) = [[e^a]]`，走纯 Vector 逐元素 Exp 快速路径（batch 维完美并行，绕开 Matmul 最小 shape 风险），不走 Cube |
| n 上限（迭代一） | n ≤ 128（`MATRIX_EXP_N_MAX_UB`，全驻留路径上限）；n > 128 时 TORCH_CHECK 报错，分块路径为迭代三规划（3.5） |

**与 torch.linalg.matrix_exp 的差异声明**：

| 差异点 | torch | 本算子 | 处理方式 |
|--------|-------|--------|---------|
| float64 | 支持 | 不支持 | TORCH_CHECK 报错 |
| complex64/128 | 支持 | 不支持 | TORCH_CHECK 报错 |
| 数值结果 | Padé-13 缩放平方 | Taylor 缩放平方 | 容差内一致（见 3.1 精度契约） |
| inf/nan 输入 | 部分场景产生 inf/nan | 未定义（s 上限封顶后按数值继续计算） | 不作为验收项，文档声明 |
| autograd | 支持反向传播 | 不支持（推理 only，与本仓其他算子一致） | 文档声明，反向由上层实现 |
| n > 128 大矩阵 | 支持 | 迭代一报错 | TORCH_CHECK 报错；迭代三补分块路径（3.5） |

---

## 2. 架构设计

### 2.1 逻辑视图

沿用本仓库统一结构（无独立 op_api / op_graph 层）：

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **op_host** | 输入校验、shape 推导、Tiling 计算、kernel 启动 | `op_host/matrix_exp.h`、`op_host/matrix_exp.cpp` |
| **op_kernel** | NPU 计算逻辑（Cube + Vector） | `op_kernel/arch35/matrix_exp.cpp`（launch）、`matrix_exp_kernel.h`（核心类）、`matrix_exp_tiling.h`（TilingData） |
| **pybind** | Python 绑定注册 | `csrc/pybind.cpp`（`OPSGNN_HAS_MATRIX_EXP` 宏内注册） |
| **python** | 用户接口包装、与 torch 对齐的签名 | `python/ops_gnn/matrix_exp.py` |

**模块依赖**：

```
python/ops_gnn/matrix_exp.py
    └── _pybind.matrix_exp(A, out)          # csrc/pybind.cpp 注册
          └── op_host matrix_exp()          # 校验 + Tiling + 启动
                └── MatrixExp(..., dtypeTag, stream)   # launch 封装（dtypeTag 三分支）
                      └── matrix_exp_kernel<T><<<coreNum, stream>>>
                            └── MatrixExpAic / MatrixExpAiv   # MIX：Cube + Vector 计算
```

### 2.2 开发视图（交付文件清单）

```
csrc/npu/matrix_exp/
├── op_host/
│   ├── matrix_exp.h                 # torch 接口声明（op_host 同名头文件，带引号 include 优先命中）
│   └── matrix_exp.cpp               # 校验 / Tiling / Launch
└── op_kernel/
    └── arch35/
        ├── matrix_exp.cpp           # launch 封装 MatrixExp（dtypeTag 分发 float/half/bf16）
        ├── matrix_exp_kernel.h      # MatrixExpAic / MatrixExpAiv 核心类（MIX：Cube + Vector）
        └── matrix_exp_tiling.h      # MatrixExpTilingData 结构体

python/ops_gnn/matrix_exp.py        # 对齐 torch.linalg.matrix_exp 签名
csrc/pybind.cpp                     # 增加 OPSGNN_HAS_MATRIX_EXP 条件注册
test/matrix_exp/arch35/
└── test_matrix_exp.py          # 功能 / 精度 / 边界用例（fp64 参考内建，见 5.3）
```

构建系统零改动：CMakeLists.txt 按 `op_kernel/arch35` 通配收集，`OPSGNN_HAS_MATRIX_EXP`
宏由"算子存在 arch35 源码目录"自动生成（逐算子条件编译机制，见开发指南 4.2）。

### 2.3 运行视图

**数据流**：

```
GM (A, batch × n×n)
  │ AIV: DataCopyPad (MTE2) → Cast (fp16/bf16 → fp32)
  ▼
UB (A, float32, nPad 方阵)                                  [AIV]
  │ AIV: 范数统计 → s/σ；meta 槽写 s；A 中转 + E0 写 GM ws
  ▼
GM ws 每核 [A nPad² | E nPad² | tmp nPad² | meta 8]          [AIV MTE3 写 / AIC 读]
  │ AIC: CopyGM2L1 (ND→NZ/ZN) → L1 → L0A/L0B
  ▼
L0C = A@E_j（Mmad × 8 轮 Horner）+ E@E（× s 轮平方）           [AIC]
  │ AIC: fixpipe 写回 ws tmp 区；Horner 轮间 AIV 读 tmp → Muls(σ/k) + E←I + Add → 写 ws E 区
  ▼
GM ws (E / tmp 区)
  │ AIV: 读 tmp（s>0 时）→ Cast（fp16/bf16 输出）
  ▼
GM (out)
```

**执行流程**：

```
ops_gnn.matrix_exp(A)
  → op_host: TORCH_CHECK 方阵/dtype → contiguous → out 分配/校验
  → ComputeTiling: coreNum(AIC)、每核矩阵数、n、dtype 写入 MatrixExpTilingData
  → 当前 NPU 流启动 kernel（不创建临时流、不做 host 同步）
  → device 每核：循环处理分到的矩阵（搬入 → 定 s → Horner → 平方 → 写回）
```

---

## 3. 实现方案

### 3.1 算法选型与精度契约

**选型对比**：

| 方案 | Matmul 次数/矩阵 | 额外操作 | 结论 |
|------|-----------------|----------|------|
| Padé-13 缩放平方（torch 参考） | 6 | 需解 (V−U)R = (V+U)：LU 分解 + 三角回代，逐矩阵串行、访存不规则，Cube/Vector 模型代价高 | 不采用（列为远期优化） |
| **Taylor 缩放平方（Horner）** | K−1+s | 仅 Matmul + 逐元素 Muls/Adds，全部为规则大粒度计算 | **采用** |
| Paterson–Stockmeyer 分段求值 | K=9 时仅省 ≤1 次 | 需常驻 A³/A⁶ 等额外 n² 缓冲（≥5 块 64KB = 320KB > UB 248KB） | 不采用（UB 放不下；K 增大后可重评） |

**参数与误差界**（theta = 0.5，K = 9，即截断到 A^9 项）：

```text
截断误差  <= e^theta * theta^(K+1)/(K+1)! = e^0.5 * 0.5^10/10!  ≈ 4.4e-10  << fp32 eps(1.19e-7)
舍入误差  <= (K-1+s) 次浮点舍入，每步相对误差 ~ c·eps，典型合计 ~ 1e-6 量级
```

s 由范数决定：Taylor 余项界 `e^{||A||}·||A||^{K+1}/(K+1)!` 对任意**次乘性**
（submultiplicative）范数成立，而 Frobenius 范数满足次乘性且可单遍连续
归约，故**直接使用**：`s = clamp(ceil(log2( ||A||_F / 0.5 )), 0, S_MAX)`，
`S_MAX = 30`（防御 inf/nan 输入）。相比"F 范数 × √n 折算 1-范数"的写法
（多花 ⌈log2(√n)⌉ 次平方），两者误差界完全相同；n=128 时直接用 F 范数
省 4 次 Matmul，是零代价的优化。

**2^s 误差放大（fp32 缩放平方法的固有限制）**：每次平方使相对误差约
翻倍，`err_final ≈ 2^s · err₀`（err₀ 为 Taylor 阶段舍入，约数个 fp32
eps ≈ 5e-7）。s > 4（`||A||_F ≳ 8`）后 1e-5 级容差在数学上不可达——
torch 的 Padé 实现同样受此限制。

**精度验收契约（按 `||A||_F` 分桶）**：

| 输入 dtype | `\|\|A\|\|_F` 区间 | 参考值 | 验收容差（分量级） |
|-----------|---------------|--------|------------------------|
| float32 | ≤ 8（s ≤ 4） | fp64 参考（torch Padé，double） | ≤ 1e-5 |
| float32 | (8, 64] | 同上 | ≤ 1e-4 |
| float32 | (64, 1e3] | 同上 | ≤ 1e-3 |
| float32 | (1e3, 1e4] | 同上 | ≤ 1e-2 |
| float16 | 各桶 | 同上（结果按输出 dtype 提升后比较） | max(桶容差, 2e-3) |
| bfloat16 | 各桶 | 同上 | max(桶容差, 2e-2) |

判据为**混合式**：`|out − ref| ≤ tol·(|ref| + max|ref|)`——大范数下 exp 结果
动态范围极宽（对消主导的小元素与 e^+90 量级大元素共存），纯逐元素相对判据在
fp32 下不可实现，绝对余量随 max|ref| 缩放（实测各桶以 18~40× 余量通过）。
fp16/bf16 由输出量化主导（half-ulp ~5e-4 / ~2e-3），取 max(桶容差, 2e-3/2e-2）。

### 3.2 模板划分总览

**模板参数定义**：

| 参数 | 类型 | 取值范围 | 说明 |
|-----|------|---------|------|
| D_T | DataType | {0: float, 1: half, 2: bfloat16} | 输入/输出 dtype；计算一律 float32（launch 按 dtypeTag 三分支） |
| N_SMALL | BOOL | {false, true} | 预留字段（迭代一恒 0，模板三分块路径落地时启用） |

**N_MAX_UB 推导**（DAV_3510 向量可用 UB = 248 KB，AIV 驻留缓冲：A、E、TMP 三块
fp32 + fp16/bf16 时 ioBuf 一块原始宽度）：

```text
fp32:  3 * nPad² * 4B <= 248 KiB                =>  nPad <= 143  =>  取 N_MAX_UB = 128
n = 128: fp32 3 * 128² * 4B = 192 KiB ✅；fp16/bf16 另加 ioBuf 32 KiB = 224 KiB ✅
（每块按 max(nPad², 8192) fp32 分配，另 metaBuf 32B）
```

**模板划分表**：

| 模板 | 触发条件 | 模板参数 | 适用场景 |
|-----|---------|---------|---------|
| 模板一 | fp32 且 n ≤ 128 | `D_T=float`（nSmall 预留 0） | 主力路径：UB fp32 计算态 + GM ws 中转（AIC/AIV 协作，见 3.4） |
| 模板二 | fp16/bf16 且 n ≤ 128 | `D_T=half/bf16`（同上） | 搬入后 Cast fp32，计算同模板一（迭代一已随 launch 三分支交付） |
| 模板三 | n > 128（任意 dtype） | `N_SMALL=false`（预留） | 迭代一未实现：host TORCH_CHECK 报错；分块路径为迭代三规划（3.5） |

### 3.3 TilingData 结构体

**文件位置**：`op_kernel/arch35/matrix_exp_tiling.h`

```cpp
// 算法常量在 matrix_exp_kernel.h 内编译期定义：MATRIX_EXP_THETA=0.5、
// MATRIX_EXP_HORNER_ROUNDS=8（K=9 阶，E0 免 1 次 Matmul）、MATRIX_EXP_S_MAX=30。

struct MatrixExpTilingData {
    uint64_t batchSize;       // 批量矩阵个数 = prod(A.shape[:-2])
    uint32_t n;               // 方阵阶数
    uint32_t coreNum;         // 参与计算的 block 数（MIX 下 = AIC 核数）
    uint32_t matricesPerCore; // 每核矩阵数（floor）
    uint32_t coresWithExtra;  // 前 coresWithExtra 个核各多分 1 个矩阵
    uint32_t dtSize;          // 输入 dtype 字宽（2/4），用于地址推进
    uint32_t nSmall;          // 预留字段（迭代一未使用，恒 0）
};
// 核 i 分得 matricesPerCore + (i < coresWithExtra ? 1 : 0) 个矩阵。
// 均衡切分，避免 ceil + 末核吸收模式在 batchSize=5/coreNum=4 时的
// lastCore 下溢（uint32 回绕）与整核空闲问题。
```

### 3.4 模板一 / 模板二：n ≤ 128 全驻留路径

**实现修订（迭代一，依据 Blaze skill 与 ops-tensor 源码核验）**：
本节最初按"AIC 上 Cube+Vector 混编、matmul::Matmul 高阶 API"编写，真机核验结论推翻
了两项前提，实现按以下事实修订：

1. **dav-3510 的 AIC 无 V pipe（no-op）**：Muls/Adds/Cast/Exp/WholeReduceSum 等 Vector
   API 只能在 AIV 执行，Cube(Mmad) 与 Vector 必须 CrossCore 协作。kernel 采用
   `KERNEL_TYPE_MIX_AIC_1_2`（每 block 1 AIC + 2 AIV，仅 subBlock 0 的 AIV 参与计算，
   subBlock 1 空转退出）。
2. **950 上无 matmul::Matmul 高阶 API**（仅 A2/A3），Cube 路径使用 CANN SDK
   tensor_api（`AscendC::Te`，**CamelCase**，非 Blaze 的 `asc::te`）：
   `Te::Copy` + `Te::MakeCopy(Te::CopyGM2L1{})`（ND→NZ/ZN 重排）、
   `Te::CopyL12L0A/L0B`、`Te::MakeMmad(Te::MmadOperation{})` + `Te::MmadParams`、
   `Te::MakeCopy(Te::CopyL0C2GM{}).with(Te::FixpipeParams{3})`（fixpipe 写回）；
   L0C 布局显式 BLOCK_CUBE=16（fp32 下 LayoutTraitDefault 会算成 C0=8）；
   Mmad/Fixpipe unitFlag 均为 3（FINAL_ACCUMULATION，单发累加与写回须同为 3）。
3. **fp32 的 UB→L1 直通不存在**（`CopyPaddedUBToL1` 仅 fp16/bf16）：AIV 更新后的 E
   经 GM workspace 中转回 AIC（布局见下），由 AIC 用 `Te::Copy(CopyGM2L1{})` 重排后供 Mmad；
   Mmad 结果经 fixpipe 写回 GM workspace tmp 区。fp32 全链路计算态不变。
4. **数据与同步协议**（GM workspace 每核 `[A nPad² | E nPad² | tmp nPad² | meta 8]`
   fp32，nPad=ceil(n/16)·16，三区均 nPad 方阵布局，host 按 coreNum 分配；A 区为 AIV
   中转的对齐 A（见注记 5），meta 槽由 AIV 写 s、AIC 标量读）：CrossCore MODE4 flag
   （AIV_READY/AIC_DONE 各 2 slot 轮换）锁步——AIC Set DONE 绑 PIPE_FIX、Wait 绑
   PIPE_MTE2；AIV Set READY 绑 PIPE_MTE3、Wait 绑 PIPE_MTE2。每矩阵 flag 严格配对：
   READY = 8+(s>0) 次、DONE = 8+(s>0) 次，尾轮 s==0 时 AIC 不再等待、AIV 持有最终 E
   直接输出；空闲核与 n==1 路径两侧均不做任何 flag 交互（无死锁、无残留）。
   AIC 核内事件独立 eventID 共 8 个：MTE2_MTE1/MTE1_M/M_FIX/FIX_MTE2/MTE2_S 跨 pipe
   交接 + M_MTE1/FIX_M/MTE1_MTE2 单缓冲反向依赖（首轮预发）；AIV 核内 7 个：
   MTE2_V/V_MTE3/V_S/V_MTE2/S_MTE2/MTE2_MTE3/MTE3_V 完成对。
   **meta 跨核可见性（真机实证）**：AIC Scalar 单元读 GM 命中核内 DCache、与 AIV
   （他核）MTE3 写入不保持一致——每核首读 miss 取到新值后，后续矩阵永远命中陈旧行、
   沿用核首矩阵的 s（squaring 次数系统性错位）。读前必须
   `DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(metaGm_)`
   失效行强制重取 GM；volatile 仅防编译器 CSE、不解决硬件 DCache（实测单独无效），
   保留作双保险。
5. **n 非 16 倍数**：E0 用 host 生成的 nPad16 方阵单位矩阵（eye 常量，尾行/行尾零）
   初始化；ws 的 A/E/tmp 三区均采用 **nPad 方阵布局**（非紧凑 n×n）——AIC 侧
   CopyGM2L1(ND→NZ) 的 d_value 与 fixpipe 行宽均要求 ≥ 32B（fp32 即 8 元素），紧凑
   布局 n < 8 时触发 device error 507015，且 AIC 无 V pipe 无法自行重排用户 GM
   紧凑张量，A 由 AIV 中转至对齐 A 区（数学上等价 nPad 阶零填充矩阵）；Mmad 的
   m/n/k 传 **nPad**（零填充分形不改变 n×n 有效块结果，实测 n=2..128 全通过）。
   eye 常量按 (device, nPad16) 进程级缓存持有（op_host GetEyeConst）：调用局部
   张量与异步入队 kernel 存在分配复用竞态（实测偶发读零），同步 H2D 写入一次终身复用。

#### 3.4.1 触发条件

`n <= N_MAX_UB(128)`；dtype 决定 D_T。

#### 3.4.2 Host 侧 Tiling

**文件**：`op_host/matrix_exp.cpp`（模式与 segment_max_csr 一致：纯函数 ComputeTiling + LaunchKernel 分发）

```cpp
// aicCanUseNum 由调用方一次获取传入（GetCoreNumAic()，950PR 实测 28）
MatrixExpTilingData ComputeTiling(uint64_t batchSize, uint32_t n, uint32_t dtSize, uint32_t aicCanUseNum)
{
    // 均衡切分：floor + 余数均摊，避免 ceil+末核吸收的下溢与整核空闲
    uint32_t coreNum = (aicCanUseNum < batchSize) ? aicCanUseNum : static_cast<uint32_t>(batchSize);
    coreNum = coreNum >= 1 ? coreNum : 1;

    MatrixExpTilingData tiling;
    tiling.batchSize = batchSize;
    tiling.n = n;
    tiling.coreNum = coreNum;
    tiling.matricesPerCore = static_cast<uint32_t>(batchSize / coreNum);
    tiling.coresWithExtra = static_cast<uint32_t>(batchSize % coreNum);
    tiling.dtSize = dtSize;
    tiling.nSmall = 0;  // 预留字段（迭代一未使用）
    return tiling;
}
```

#### 3.4.3 Kernel 侧核心流程（MatrixExpAic / MatrixExpAiv 协作，每矩阵）

```text
0. n == 1 快速路径: exp([[a]]) = [[e^a]]，batch 维逐元素 Exp 直接写回
   （纯 Vector，完美并行，绕开 Matmul 最小 shape 风险），跳过以下全部步骤
1. DataCopyPad: GM(A_b) → UB(aBuf)，n×n fp32（fp16/bf16 先搬入原始宽度再 Cast）
2. 范数（AIV）: tmp = A*A；树形折半 Add 归约（3510 无 (dst,src,ws,count) 形式的
                WholeReduceSum，折半 src2 偏移须 32B 对齐）+ 尾段 ≤16 元素 S pipe 标量
                累加 → ||A||_F²。不开 Sqrt：s 由 norm² 与 (0.5·2^s)² 的等价比较直接
                得出（bound 初值 0.25，每轮 ×4）；σ = 2^-s（精确二次幂）
3. 缩放折叠:    不做独立的缩放趟；A 保持原始值，2^-s 折叠进 Horner 各步标量；
                AIV 将 A（nPad 方阵）中转至 ws A 区（对齐要求见 3.4 注记 5）
4. Horner（K-1 = 8 次 Matmul，多项式达 K=9 阶；AIC 出积、AIV 更新 E，按注记 4 flag 锁步）:
                E = I + (σ/K)·A          # AIV：eye 常量 + Muls + Add，免 1 次 Matmul
                for k = K-1 .. 1:
                    Mmad(A@E_j) → ws tmp  # AIC：A 常驻 L0A，E 每轮重装 L0B
                    E = I + (σ/k)·(A@E_j) # AIV：读 ws tmp → Muls(σ/k) + E←I + Add → 写 ws E
5. 平方（纯 AIC）: for i = 0 .. s-1:  Mmad(E@E) → ws tmp。i=0 读 ws E 区、i>0 读
                ws tmp（FIX_MTE2 事件保证上一轮 fixpipe 写完成后再读）；每轮结果均写
                ws tmp（非乒乓，单点写回 + 事件排序即可保证顺序）
6. 写回:        s>0 时 AIV 等 AIC DONE 后从 ws tmp 读回；Cast（D_T ≠ float 时）→
                逐行 DataCopyPad UB → GM(out_b)
```

矩阵间循环复用上述缓冲（迭代一为单缓冲串行；AIC 侧 L0A/L0B 单缓冲经
M_MTE1/FIX_M/MTE1_MTE2 反向依赖事件对跨矩阵串行复用，矩阵级双缓冲为
迭代二优化项，见 4.2）。

#### 3.4.4 API 映射

| 计算步骤 | Ascend C API | 参数签名（要点） | 平台验证 | 约束说明 | 替代方案 |
|---------|-------------|-----------------|---------|---------|---------|
| 数据搬入 | DataCopyPad\<T\> | GM→UB 逐行（blockCount=n）+ dstStride 控制行距 nPad | ✅ 已真机核验 | 3510 rightPadding 仅支持 ≤1 个 32B 对齐块尾填充、UB→GM srcStride 非对齐均触发 device error 507015（实测 n=2..7/17 崩）→ padding=0 + dstStride / 逐行搬出 | DataCopy（全 32B 对齐时可用，ws 区间搬运采用） |
| dtype 提升 | Cast\<fp32, T\> | 入 CAST_NONE / 出 CAST_RINT | ✅ 已真机核验 | fp16/bf16 → fp32 | - |
| 逐元素标量乘/加 | Muls\<float\> / Adds\<float\> / Add\<float\> | (dst, src, scalar/value, count) | ✅ 已真机核验 | count = bufElems(=max(nPad², 8192)，尾区零填充)；Horner 各步 scalar = σ/k（缩放折叠）；E←I 经 Adds(eBuf, eye 常驻, 0.0) 拷贝 | - |
| 范数归约 | 树形折半 Add\<float\>（3510 无 (dst,src,ws,count) 形式 WholeReduceSum） | 折半步长 8 元素对齐（src2 偏移须 32B 对齐，否则错误码 340）；尾段 ≤16 元素转标量累加 | ✅ 已真机核验 | 无需 sharedTmpBuffer；norm² 直接与 (0.5·2^s)² 比较，无 Sqrt | Abs 后逐块累加 |
| 标量化定 s | LocalTensor GetValue（S pipe 标量累加）+ while(normSq > bound) 等价比较 | - | ✅ 已真机核验 | s 为逐矩阵标量，控制平方循环次数；经 Duplicate + DataCopy 写 ws meta 槽（AIC 读前须 DCache 失效，见 3.4 注记 4） | - |
| n=1 指数 | Exp\<float\> | (dst, src, count)，分块 8192 元素 | ✅ 已真机核验 | n=1 快速路径专用（纯 AIV，不交互 flag） | Taylor 路径兜底 |
| 单位阵对角写 | host 生成 eye 常量（nPad16 方阵）→ DataCopyPad 全块拷贝 | - | ✅ 已采用（迭代一） | 尾行/行尾零兼作 pad 区清零；Muls+Add 实现 E0 = I + (σ/9)A | Dup 跨步写对角（真机核验后再评估） |
| 矩阵乘 | Mmad（CANN SDK tensor_api `AscendC::Te`，950 无 matmul::Matmul 高阶 API） | `Te::MmadParams{nPad, nPad, nPad, unitFlag=3, cmatrixInitVal=true}`；`Te::Copy(CopyGM2L1{})` ND→NZ/ZN | ✅ 已真机核验（n=2..128 分桶精度全绿） | L0C 布局显式 BLOCK_CUBE=16（fp32 下 LayoutTraitDefault 会算成 C0=8）；m/n/k 传 nPad；Mmad/Fixpipe unitFlag 同为 3 | n 较小时用 Vector 点积实现（性能差，保底） |
| 数据搬出 | Cast\<T, fp32\>（CAST_RINT）+ DataCopyPad\<T\> | 同搬入，方向 UB→GM（逐行） | ✅ 已真机核验 | - | - |
| 同步 | SetFlag/WaitFlag（核内事件对，AIC 8 / AIV 7 个独立 eventID）+ CrossCore MODE4 flag（AIC↔AIV 锁步） | HardEvent 枚举 | ✅ 已真机核验 | AIV 侧 GetBlockIdx() 须除以 GetTaskRation() 换算逻辑 block（rawIdx 含 subblockid，不换算会导致奇数 block 无人处理/挂死，SDK 生产 kernel 同法） | - |


#### 3.4.5 数据流设计

```
逐矩阵（AIC/AIV 按 3.4 注记 4 flag 协议锁步）：
1. AIV: GM(A_i) → UB(fp32)；范数 → s/σ；A 中转 → ws A 区；E0 → ws E 区；s → ws meta 槽
2. AIC: ws A/E → L1(NZ/ZN) → L0A/L0B → Mmad → L0C → fixpipe → ws tmp 区
   AIV: ws tmp → UB → E 更新 → ws E 区（Horner 每轮，共 8 轮）
3. 平方 s 轮（纯 AIC）：ws E/tmp 区 → Mmad(E@E) → ws tmp 区
4. AIV: ws tmp（s>0）或 UB eF32（s==0）→ Cast → GM(out_i)
```

#### 3.4.6 内存管理

| 内存区域 | 大小计算（n ≤ 128） | 说明 |
|---------|--------------------|------|
| **UB aBuf** | max(nPad², 8192)×4B（n=128 时 64 KB） | 输入矩阵 fp32 计算态 / eye 常驻 |
| **UB eBuf** | 同 aBuf | E 累加结果（UB 计算态，写 ws 前驻留） |
| **UB tmpBuf** | 同 aBuf | AIC 积中转读回 / 范数平方 |
| **UB ioBuf** | max(nPad², 8192)×sizeof(T)（fp16/bf16 32 KB；fp32 无此块） | 原始 dtype 搬入 / 输出 Cast 暂存 |
| **UB metaBuf** | 8×4B = 32 B | s 标量中转（Duplicate → DataCopy 写 ws meta 槽） |
| **L1** | 2×nPad²×4B（A NZ 区偏移 0 + E B 侧 NZ 区偏移 nPad²·4B） | CopyGM2L1 重排目标，显式偏移管理 |
| **L0A/L0B/L0C** | nPad²×4B×2 + nPad²×4B（L0C 布局 BLOCK_CUBE=16） | tensor_api 显式布局；Mmad unitFlag=3 |
| **Workspace** | coreNum × (3·nPad² + 8) × 4B fp32 | 每核 [A \| E \| tmp \| meta]（见 3.4 注记 4） |

**UB 总和验证**：fp32 三块 192 KB（n=128）+ metaBuf 32 B ≈ 192 KB ≤ 248 KB
（DAV_3510）✅；fp16/bf16 另加 ioBuf 32 KB = 224 KB ≤ 248 KB ✅。

### 3.5 模板三：大矩阵分块路径（n > 128）

#### 3.5.1 触发条件

`n > N_MAX_UB`（D_T 任意）。

#### 3.5.2 方案要点

- Horner / 平方的 Matmul 由 Matmul API 按 L1/L0 容量自动分块，每次乘法以
  n≤256 的 tile 流水搬运；缩放折叠后**原始 A 直接从 GM 读**（无需 A' 副本）；
  E 中间态就地暂存 `out[b]`（fp32 输出时）或 fp32 workspace（fp16/bf16 输出时，
  半精度不足以承载中间态）；workspace 缩减为 `coreNum × n² × 4B`（每核一个
  Matmul C 写位 T，与 E 槽位乒乓）+ fp16/bf16 时的 `batchSize × n² × 4B` E 暂存；
- 范数、缩放、Cast 均按行分块做（每块多行 × n 列，块高由 UB 决定）；
- 性能预期显著低于全驻留路径，仅作功能覆盖，不设性能门槛；
- 迭代一现状：**已按"`n > 128` 时 host TORCH_CHECK 报错"实施**（op_host 以
  `MATRIX_EXP_N_MAX_UB=128` 校验），分块路径留待迭代三（评审"先报错"选项落地）。

---

## 4. 性能优化

### 4.1 并行策略

- batch 维均分到全部 AIC（`GetCoreNumAic()`），单矩阵不跨核（矩阵内
  数据依赖链 Horner→平方 串行，跨核拆分无收益）；
- batch < coreNum 时实际核数收缩为 batch（见 ComputeTiling）。

### 4.2 流水线设计

- 迭代一现状为**矩阵间单缓冲串行**（AIC 侧 L0A/L0B 单缓冲经 M_MTE1/FIX_M/
  MTE1_MTE2 反向依赖事件对跨矩阵复用）；矩阵级双缓冲（下一矩阵 A 搬入与当前
  Horner 首个 Mmad 重叠）为**迭代二优化项**——MIX 数据流下 UB 需求须按
  aBuf/eBuf/tmpBuf/ioBuf 四块重估（fp32 n=128 时 4×64KB = 256KB > 248KB 超限）；
- 核内事件对保证各 pipe 段串行正确（AIC：MTE1/MTE2/M/FIX/S；AIV：MTE2/V/MTE3/S），
  Cube(AIC)↔Vector(AIV) 经 CrossCore MODE4 flag 锁步（3.4 注记 4），缓冲反复用
  与 segment_max_csr 的 event 配对风格一致；
- 缩放因子 σ = 2^-s 为精确二次幂、无舍入；Horner 标量 σ/k 中 1/k（k 非
  2 的幂时）有 ~eps 的标量舍入，已计入 3.1 舍入预算，不影响契约。

---

## 5. 风险评估

### 5.1 API 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| 950 Cube fp32 默认启用 hf32/TF32（低精度尾数） | 精度不达标 | **迭代一实测未复现**：fp32 Mmad 精度达 ≤8 桶 1e-5 契约（27 用例全绿，余量 ~30×）；若后续精度回退再排查开关位 / fp32 高低位双 Mmad 补偿 |
| Matmul 对 fp32 in/out 或奇数 n 的支持缺口 | 路径不可用 | **已解决**：tensor_api Mmad fp32 实测可用；奇数 n 以 nPad 方阵布局规避（pad 到 16 倍数而非偶数，3.4 注记 5） |
| WholeReduceSum 大 repeat 分段限制 | 范数计算错误 | **已规避**：3510 无该 API 形式，改树形折半 Add 归约（32B 对齐步进）+ 标量尾段累加（3.4.4）；n=2..128 实测覆盖 |

### 5.2 精度风险

| 风险 | 说明 | 缓解 |
|------|------|------|
| Taylor 与 torch Padé 结果差异 | 理论均在后向稳定意义下精度相当，但逐元素存在差异 | 验收按 3.1 容差对 fp64 参考对拍，而非与 torch 逐位对拍 |
| 舍入随 s、K 累积 | 每次平方相对误差约 ×2，err ≈ 2^s·err₀；norm 1e4 时 s≈15（放大 ~3e4 倍） | 截断误差 4.4e-10 留足余量；契约按 `\|\|A\|\|_F` 分桶 + 混合判据（3.1，绝对余量随 max\|ref\| 缩放）；此为 fp32 缩放平方的固有限制（torch Padé 同样存在），实测各桶以 18~40× 余量通过 |
| fp16/bf16 输出二次舍入 | torch 在内部 accscalar(float) 后 cast，行为一致 | 同样"fp32 计算末态 cast"，与 torch 对齐 |

### 5.3 参考实现（交付形态）

fp64 参考直接内建于 `test/matrix_exp/arch35/test_matrix_exp.py`（`_golden`）：
`torch.linalg.matrix_exp(A.detach().cpu().double())`（CPU，fp64 Padé）——
fp64 参考与被测算法无关（Taylor 与 Padé 在 fp64 下差异 << 容差），无需独立
golden.py；判据为 3.1 分桶混合容差。接口级行为对拍（shape/dtype/报错/out=/
非连续/空 batch/重复 launch 确定性）同文件覆盖。

### 5.4 应对措施汇总

迭代一精度评审点（2026-10，**已通过**）：fp32 在 `||A||_F ≤ 8` 桶达 1e-5 契约
（混合判据下实测余量 ~30×，27 用例分桶全绿），算法参数 θ=0.5 / K=9 /
S_MAX=30 **冻结**；后续若精度回退，优先排查 hf32、其次提高 K 至 13（Mmad 次数 +4）。

---

## 6. 交付件清单

**必需**（本仓库结构）：

| 类别 | 文件 |
|------|------|
| op_host | `csrc/npu/matrix_exp/op_host/matrix_exp.h` / `matrix_exp.cpp` |
| op_kernel | `csrc/npu/matrix_exp/op_kernel/arch35/matrix_exp.cpp` / `matrix_exp_kernel.h` / `matrix_exp_tiling.h` |
| pybind | `csrc/pybind.cpp` 注册（自动获得 `OPSGNN_HAS_MATRIX_EXP` 条件编译） |
| python | `python/ops_gnn/matrix_exp.py` + `__init__.py` 导出 |
| 测试 | `test/matrix_exp/arch35/test_matrix_exp.py`（fp64 参考内建，见 5.3） |
| 文档 | `docs/zh\|en/api_reference.md`、`docs/zh\|en/development_guide.md`、README 增补 |

**可选**：英文版设计文档（`matrix_exp算子设计文档_en.md`，已随本 PR 交付）。

---

## 7. 迭代规划

| 迭代 | 目标 | 代码开发 | 验证 |
|------|------|---------|------|
| 迭代一 | 全 dtype 全驻留路径跑通 | 模板一/二（fp32 + fp16/bf16, n ≤ 128）+ n=1 纯 Vector 快速路径 + TilingData + launch + pybind + python 包装；真机核验清单全部关闭 | **已完成**（2026-10）：n∈{1,2,3,8,15,16,17,32,128} × norm 分桶（0.1~1e4）× 三 dtype，27 用例全绿 |
| 迭代二 | 双缓冲流水优化 | 矩阵级双缓冲 + 事件流水（fp16/bf16 与边界用例已随迭代一交付：launch 三 dtype 分支、空 batch/n=1/非连续/out= 覆盖） | 性能数据记录与调优 |
| 迭代三 | 大矩阵 + 收尾 | 模板三分块路径（n>128 现为报错，见 3.5） | 大 n 功能覆盖；性能数据记录（batch×n 矩阵）；文档（api_reference / README / dev guide）收口 |

# scatter_logsumexp 算子设计文档（A2/A3 arch22，950 arch35）

> 状态：已实现。功能、精度与性能实测数据见对应 PR 的测试报告；本文件说明接口、
> 语义、复用边界与分块方案。
> 依赖：`scatter_softmax`、`scatter_log_softmax` 已合入（#165、#186），本算子复用
> 它们的 Host 共享件与设备侧流水。

## 1. 需求与验收口径

实现 `torch_scatter.scatter_logsumexp` 的 NPU 版本：

```python
scatter_logsumexp(src, index, dim=0, dim_size=None)  # Tensor
```

对每个分组 `g`：

```
out[..., g, ...] = log( sum_{i : index[i] = g} exp(src[..., i, ...]) )
```

数值稳定写法与 `torch_scatter` 相同，先减分组最大值：

```
m_g = max_{i : index[i] = g} src[...]
out[..., g, ...] = m_g + log( sum_i exp(src[..., i, ...] - m_g) )
```

验收口径：

| 项 | 要求 |
| --- | --- |
| 接口 | 与 `torch_scatter.scatter_logsumexp` 对齐；`dim` 默认 0，与仓内 `scatter_softmax` 一致；不提供 `out=` 参数 |
| 输出 | 形状为 `src.shape` 把 `dim` 换成 `dim_size`；dtype / device 与 `src` 相同 |
| dtype | `float16` / `float32` / `bfloat16`；归约全程 fp32 |
| index | 同设备 1-D `int32` / `int64`，长度 = `src.size(dim)`，取值 `[0, dim_size)` |
| 求导 | 支持 autograd，`grad_src[i] = grad_out[index[i]] * exp(src[i] - out[index[i]])` |
| 平台 | arch22（A2/A3）与 arch35（950）共用同一份 kernel 源码，两个架构目录只有薄编译单元 |
| 性能 | 对 `scatter_max + exp + scatter_sum + log` 组合实现有净收益；先保证正确，再按实测优化 |

## 2. Host 维度与输出契约

沿用 `scatter_softmax` 的批/组/特征三分解：

- `batch  = prod(src.shape[:dim])`
- `groups = dim_size`
- `after  = prod(src.shape[dim+1:])`，按 8 个元素对齐得到 `afterPadded`
- 输入行数 `rows = batch * dimLength`，输出行数 `outRows = batch * groups`

输入与工作区布局不变（`workspaceElems = batch * groups * afterPadded`），
新增的只有输出：

```
out[b, g, f] = maximum[b, g, f] + log(total[b, g, f])
```

因此输出分配不能复用 `RunForward` 的 `torch::empty(src.sizes())`，Host 侧新增
`csrc/npu/scatter_logsumexp/common/grouped_logsumexp_shared.h`，给出两个入口：

```cpp
torch::Tensor scatter_logsumexp_forward(src, index, dim, dim_size);
torch::Tensor scatter_logsumexp_backward(src, out, grad_out, index, dim, dim_size);
```

- 前向：`ValidateInputs` / `BuildTiling` / `CheckLaunchable` / `ResolveDType` 全部复用
  `csrc/npu/scatter_softmax/common/grouped_softmax_shared.h`；只有输出分配改成
  “把 `dim` 换成 `dim_size`”。
- 输出为 0 元素（`after == 0` 或 `groups == 0`）时直接返回；`rows == 0` 但输出非空
  （有分组、无成员）时用 `-inf` 填满输出后返回，不启动 kernel。
- 反向：形状回到 `src.sizes()`，`grad_out` 形状为前向输出形状。校验先复用
  `ValidateInputs`（此时 `src.size(dim) == dim_size`，索引长度检查仍然成立），再补
  `out` / `grad_out` 的 contiguous、shape、dtype、**同设备**检查。源张量作为显式
  入参传入，因为解析梯度要读 `src` 本身。
- `dim_size == 0` 时反向直接返回 `zeros_like(src)`（输出无元素，梯度恒为 0）。

## 3. 设备侧算法

### 3.1 前向：2 遍全量 + 1 遍输出

| 遍 | 读 | 写 | 说明 |
| --- | --- | --- | --- |
| P1 最大值 | `src`（N×C） | `maximum` 工作区（G×C fp32） | 复用 `scatter_softmax` 的 max 批处理；空组保持 neutral |
| P2 指数和 | `src` + `maximum` | `total` 工作区（G×C fp32） | `total += exp(src - maximum)`，fp32 累加 |
| P3 收尾 | `maximum` + `total` | `out`（G×C） | `out = maximum + log(total)`，只写 `dim_size` 行 |

关键点：

- **P3 只处理 `batch * groups` 行**，与输入规模 `batch * dimLength` 无关，
  这是相对组合实现的主要收益之一（组合实现要把 N×C 的中间结果整体写回显存）。
  P3 的行数、每核行块与行批由 tiling 里新增的 `outRows` / `outRowChunk` /
  `outBlockNum` / `outRowBatch` 单独给出，启动时覆盖一份 tiling 副本；该遍不读
  `index`，稠密工作区行就是输出行。
- P1 / P2 的归约簿记（UB 常驻私有累加器或每行一次原子写）直接复用现有实现，
  不新增第二套行调度。
- **`+inf` 修正**：`exp(+inf - +inf) = NaN`，但 `torch.logsumexp` 对该列给 `+inf`。
  P2 把参与减法的最大值截断为最大有限值 `FLT_MAX`（每行一次 `Mins`）：
  含 `+inf` 的元素得到 `exp(+inf) = +inf`，`total = +inf`，P3 的
  `+inf + log(+inf) = +inf` 正是期望值；同组有限元素得到 `exp(-inf) = 0`。
  有限分组不受影响（其最大值本来就小于 `FLT_MAX`）。`-inf` 与空组不需要修正：
  neutral 取 `-FLT_MAX`，`-inf - (-FLT_MAX) = -inf`、`exp` 得 0，P3 的
  `-FLT_MAX + log(0) = -inf` 正是期望值。

### 3.2 反向：1 遍 elementwise

```
grad_src[i] = grad_out[index[i]] * exp(src[i] - out[index[i]])
```

- 一次全量读写：读 `src`、按 `index` 广播读 `grad_out` 与 `out` 的分组行，写 `grad_src`；
- 不需要归约工作区，不需要中间张量，因此该遍只分配三个 dtype 缓冲（`src`、`out` 行、
  `grad_out` 行）与三个 fp32 tile，仍在共享 tiling 的每行 20 字节预算内；
- **小组数时把两个分组张量常驻 UB**：每个源行都要读 `out` 与 `grad_out` 的同一行，
  `groups` 很小时这些读全部落在同一条缓存行上且不重叠，实测 `1048576 × 128` 的
  `groups = 1` 反向会退化到约 40 ms（`groups ≥ 16` 约 6 ms）。当整个分组张量能放进
  核内预算（`workspaceElems` 的 fp32 副本 ≤ 16 KB，即 `groups` 较小）时，反向在行循环
  之前一次性把 `out` 与 `grad_out` 展成 fp32 常驻 UB，行循环里改为 `Adds` 取值；
  这两份缓存的字节数正好等于 tiling 已为「两份 fp32 工作区」预留的额度，因此行批
  不需要再缩小，UB 峰值仍在 96 KB 以内。缓存不适用时（`after > colTile` 或分组张量
  超出预算）自动退回逐行 GM 读，语义不变；
- `out[g] = ±inf` 时公式自然给出与 PyTorch 一致的结果（实测 `torch.logsumexp`）：
  全 `-inf` 组 `-inf - (-inf) = NaN`；`+inf` 元素 `+inf - (+inf) = NaN`，
  同组有限元素 `x - (+inf) = -inf`、`exp` 得 0。

### 3.3 复用边界

| 复用 | 内容 |
| --- | --- |
| 直接复用 | `ValidateInputs` / `BuildTiling` / `CheckLaunchable` / `ResolveDType`；`LoadGroups`、`CopyRowsIn/Out`、`CopyGroupRowsIn`、`CopyWorkspace*`、`AccumulateRows`、`FlushAccumulator`；最大值与指数和两遍的批处理本体；以及「小组数把分组张量展成 fp32 常驻 UB」的同一手法（softmax 家族用它缓存工作区，反向用它缓存 `out` 与 `grad_out`） |
| 编译期扩展 | `scatter_softmax_kernel_common.h` 的阶段枚举改为 `Softmax` / `LogSoftmax` / `LogSumExp` 三模式，指数遍增加 `LogSumExp` 的 `+inf` 截断且不写 `src` 形状输出 |
| 文件拆分 | 搬运与工作区常驻助手独立到 `scatter_softmax_kernel_io.h`（`phase.h` 不再超过 CodeCheck 的单头文件 500 行上限），三个算子都包含它 |
| 新增 | P3 收尾批处理、反向 elementwise 批处理、`scatter_logsumexp` 的 launch / tiling 实例化；两个算子的 `DispatchDType` 收敛到入口头的一份 |
| 不复用 | softmax 的 normalize 批处理（它按行写 `src` 形状，logsumexp 只写分组行） |

这样三个算子的行调度、UB/工作区预算、归约簿记只保留一份，CodeCheck 不会出现
“同一段归约代码两份”的告警。

## 4. 分块、对齐与流水

完全沿用 `scatter_softmax` 的 tiling 常量与预算：

- `SCATTER_SOFTMAX_WORKSPACE_ALIGN = 8`（32 字节）、`ROW_ALIGN = 16`（fp16/bf16 32 字节、
  fp32 64 字节）；
- 行批 `rowBatch` 由 `ROW_BUDGET_BYTES = 80 KB` 与每行 `colTile × 20` 字节推导；
- UB 常驻工作区上限 `ACC_BUDGET_BYTES = 16 KB`，超出即走每行一次原子写；
- 每核负责一段连续行，批内 MTE2 / V / MTE3 三段流水，P3 与反向沿用同一驱动。

P3 的行数是 `batch * groups`，通常远小于输入行数；`after` 很大时沿特征轴按
`colTile` 分块，与 P1/P2 相同。

## 5. 数值语义

| 输入 | `out` | 依据 |
| --- | --- | --- |
| 空组（`dim_size` 大于实际分组数） | `-inf` | `torch.logsumexp` 空约简；`-FLT_MAX + log(0)` |
| 组内全 `-inf` | `-inf` | `torch.logsumexp([-inf, -inf]) = -inf`（实测） |
| 组内含 `+inf` | `+inf` | `torch.logsumexp([inf, x]) = inf`（实测）；P2 截断保证 |
| 组内含 `NaN` | `NaN` | `exp(NaN)` 进 `total`，P3 传播 |
| 组内元素恰为 `-FLT_MAX` | `-FLT_MAX` | 偏移为 0，`total = 1` |
| 长组低精度 | 不塌缩 | fp32 归约；`out = m + log(sum)`，`sum` 不会因 `exp` 下溢而整组变 `-inf`，除非确实全 `-inf` |

反向：

| `out[g]` | 元素 | `grad_src` |
| --- | --- | --- |
| `-inf` | `-inf` | `NaN`（`-inf - (-inf)`） |
| `+inf` | `+inf` | `NaN` |
| `+inf` | 有限值 | `0` |
| 有限 | 任意 | `grad_out[g] * exp(x - out[g])` |

上述行为均以 `torch.logsumexp` 的 CPU 实测为准（fp32），实现后逐条对拍。

## 6. 与 `torch_scatter` 的行为差异

| 场景 | `torch_scatter.scatter_logsumexp` | 本实现 |
| --- | --- | --- |
| 空组 | `-inf` | `-inf`（一致） |
| 全 `-inf` 组 | 组合实现按 `m + log(sum(exp(x-m)))` 计算，`-inf - -inf` 会产生 `NaN` | `-inf`，与 `torch.logsumexp` 一致 |
| 含 `+inf` 组 | 同上，组合实现给 `NaN` | `+inf`，与 `torch.logsumexp` 一致 |
| `dim` 默认值 | `-1` | `0`，与仓内 `scatter_softmax` 一致 |
| `out=` 参数 | 支持 | 不支持，与仓内 `scatter_softmax` 一致 |

后两条是仓内接口一致性取舍；前两条取数学上正确、且与 PyTorch 原生
`torch.logsumexp` 一致的结果，已在 API 文档中显式说明。

## 7. 测试与性能

### 7.1 用例组织（沿用 #186 的目录结构）

| 位置 | 内容 |
| --- | --- |
| `test/common/grouped_softmax_golden.py` | 家族共用 float64 参考：分组 softmax / log-softmax，以及本算子的 `grouped_logsumexp_reference`（支持 `dim` / `dim_size` / 空组）与复用它建图的 `logsumexp_gradient` |
| `test/common/grouped_softmax_cases.py` | 家族共用形状、dtype、非法输入用例表 |
| `test/common/test_grouped_softmax_suite.py` | 家族基类 `GroupedFamilySuite`（三个算子共有的参数、梯度与边界用例）以及 softmax / log-softmax 套件 |
| `test/common/test_grouped_log_softmax_suite.py` | 对数域套件 `LogSoftmaxSuite` 与 `LogSumExpSuite`（`test_` 前缀，类名非 `Test*`） |
| `test/scatter_logsumexp/arch22/` 与 `arch35/` | 各只留 `golden.py`、`test_scatter_logsumexp.py`、`benchmark_scatter_logsumexp.py`，只做绑定 |

本算子不新增 `test/common` 文件：参考实现、用例表、套件与基准全部并入既有家族模块，
`test/common` 的文件数与合入 `scatter_log_softmax` 时保持一致。

用例覆盖：

- 形状 × dtype 矩阵（1-D 特征、常规分组、空组、每组单元素、单组、`dim_size` 过大、
  `dim=1`、3-D、负维度、宽特征行分块）；
- 极值语义：全 `-inf`、`+inf` 混合、`NaN`、`±1000` 量级、单元素组；
- 长组不塌缩（65504 / 131072 行，fp16/bf16）；非连续 `src`；`int32`/`int64` 索引；
- 无成员但输出非空的 `-inf` 填充路径；
- 梯度对拍（fp64 参考、fp16/bf16 低精度、`+inf` 元素为 NaN 且同组有限元素为 0）；
  `grad_out` 在另一设备时报错；前向不修改 `src`；
- 非法输入：dtype、`index` dtype/维度/长度/越界、`dim` 越界、`dim_size` 非法、
  CPU 张量、非 Tensor。

### 7.2 性能口径

- 三口径：`kernel`（`_pybind` + 一次 `synchronize()`）、`operator`（公开 Python 入口）、
  `composed`（`argsort` + `segment_max_csr` + 减最大值 + `exp` + `segment_sum_csr` + `log`，
  全为仓内 NPU 算子；A2 分支没有 `scatter_max`）；
- A2：7 形状 × 2 dtype 共 14 行，附 1–65536 分组扫描；
- A5：同形对比 + 速率探针 + 3 次重复（沿用 #186 的脚本与判据）；
- 预期收益来自两处：全量读写从组合实现的 3 遍（max 读、减最大值 + `exp` 读写、
  求和读）降到 2 遍（max 读、`exp` 读），且不产生 `N×C` 的 `exp` 与 `x-m` 中间张量；
  P3 只写 `G×C`。具体倍数以实测为准。

## 8. 实施步骤与风险

### 8.1 步骤

1. 从最新 master 开 `feat/scatter-logsumexp` 分支（共享的 phase 头改动与 #186 同一份）；
2. Host + kernel + Python + pybind + 文档 + 测试一次提交，保持“一个 issue 一个 PR”；
3. A2 全量重编 + 功能/梯度/独立校验 + 14 行基准；A5 走 CI 编译与团队实测；
4. `/compile` 触发 CI，CodeCheck 必须 0 条（不屏蔽）；squash 成单提交；
5. 测试报告 v1.0（时间戳，A2/A5 分别标注）。

### 8.2 风险与对策

| 风险 | 对策 |
| --- | --- |
| 改 `scatter_softmax_kernel_phase.h` 影响已合入的两个算子 | `LogSumExp` 模式只在实例化时编译；两个既有算子的全量用例与基准复跑 |
| `+inf` 截断带来额外向量指令 | 只在指数遍每行加 1 次 `Mins`；14 行基准与 #186 数据逐项对比确认无回退 |
| `dim_size` 远大于实际分组数时 P3 开销 | P3 行数为 `batch * groups`，与输入行数无关；分组扫描里加 `dim_size = 2 × max+1` 与 `65536` 两档 |
| 组合实现路径与 `torch_scatter` 在 `±inf` 上语义不同 | 文档显式列表（本文件 §6），测试以 `torch.logsumexp` 为基准 |
| `torch_scatter` 未安装，三方对比缺少直接基线 | 对比基线用 `torch.logsumexp` 逐组循环（float64/float32）与仓内组合实现；若环境可安装再补 `torch_scatter` 直测 |

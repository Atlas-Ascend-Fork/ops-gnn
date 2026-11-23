# scatter_std 算子设计文档（A2/A3 arch22，950 arch35）

> 状态：已实现。功能、精度与性能实测数据见对应 PR 的测试报告；本文件说明接口、
> 语义、复用边界与分块方案。
> 依赖：`scatter_softmax`（#165）、`scatter_log_softmax`（#186）、
> `scatter_logsumexp`（#252）已合入，本算子复用它们的 Host 共享件与设备侧流水。

## 1. 需求与验收口径

实现 `torch_scatter.scatter_std` 的 NPU 版本：

```python
scatter_std(src, index, dim=0, dim_size=None, unbiased=True)  # Tensor
```

对每个分组 `g`：

```
mean_g = (1 / n) * sum_{i : index[i] = g} src[..., i, ...]
out[..., g, ...] = sqrt( sum_{i : index[i] = g} (src[..., i, ...] - mean_g)^2 / (n - 1) )
```

`unbiased=False` 时分母换成 `n`。验收口径：

| 项 | 要求 |
| --- | --- |
| 接口 | 与 `torch_scatter.scatter_std` 对齐；`dim` 默认 0，与仓内 `scatter_softmax` 一致；不提供 `out=` 参数；`unbiased` 默认 `True` |
| 输出 | 形状为 `src.shape` 把 `dim` 换成 `dim_size`；dtype / device 与 `src` 相同 |
| dtype | `float16` / `float32` / `bfloat16`；归约全程 fp32 |
| index | 同设备 1-D `int32` / `int64`，长度 = `src.size(dim)`，取值 `[0, dim_size)` |
| 求导 | 支持 autograd，`grad_src[i] = grad_out[index[i]] * (src[i] - mean[index[i]]) / ((n_g - c) * out[index[i]])`，`c = unbiased ? 1 : 0` |
| 平台 | arch22（A2/A3）与 arch35（950）共用同一份 kernel 源码，两个架构目录只有薄编译单元 |
| 性能 | 对 `argsort + segment_sum_csr × 3 + sqrt` 组合实现有净收益；先保证正确，再按实测优化 |

## 2. Host 维度与输出契约

沿用 `scatter_softmax` 的批/组/特征三分解：

- `batch  = prod(src.shape[:dim])`
- `groups = dim_size`
- `after  = prod(src.shape[dim+1:])`，按 8 个元素对齐得到 `afterPadded`
- 输入行数 `rows = batch * dimLength`，输出行数 `outRows = batch * groups`

输出维度由 `grouped_softmax::ReducedSizes` 给出（与 `scatter_logsumexp` 同一份），
工作区仍是 `workspaceElems = batch * groups * afterPadded` 的 fp32，另加成员计数
工作区 `countElems = align_up(groups, 8)`（每组一个 fp32 计数器，按 32 字节对齐补齐，
见 §3.1）。

Host 侧新增 `csrc/npu/scatter_std/common/grouped_std_shared.h`，给出两个入口：

```cpp
std::vector<torch::Tensor> scatter_std_forward(src, index, dim, dim_size, unbiased);
torch::Tensor scatter_std_backward(src, out, grad_out, mean, count, index, dim, dim_size, unbiased);
```

- 前向：`ValidateInputs` / `BuildTiling` / `CheckLaunchable` / `ResolveDType` 全部复用
  `csrc/npu/scatter_softmax/common/grouped_softmax_shared.h`；`BuildTiling` 的
  `standardDeviation` 参数为真时预留计数工作区并把输出行数写进 tiling；只有输出分配
  改成“把 `dim` 换成 `dim_size`”。
- 前向返回 `{out, mean, count}` 三个张量：`mean` 与 `count` 由 autograd 保存，
  反向直接读取，避免重算均值与成员数。均值复用“分组和”工作区（收尾遍原地覆盖），
  因此前向只多分配 `deviation` 与 `count` 两块。
- 输出为 0 元素（`after == 0` 或 `groups == 0`）时直接返回；`rows == 0` 但输出非空
  （有分组、无成员）时用 `NaN` 填满输出后返回，不启动 kernel。
- 反向：形状回到 `src.sizes()`，`grad_out` 形状为前向输出形状。校验先复用
  `ValidateInputs`（此时 `src.size(dim) == dim_size`，索引长度检查仍然成立），再补
  `out` / `grad_out` / `mean` / `count` 的 contiguous、shape、dtype、**同设备**检查，
  并把工作区长度与重新推导的 tiling 对齐（防止跨形状误用）。
- `dim_size == 0` 时反向直接返回 `zeros_like(src)`（输出无元素，梯度恒为 0）。

## 3. 设备侧算法

### 3.1 前向：2 遍全量 + 1 遍收尾

| 遍 | 读 | 写 | 说明 |
| --- | --- | --- | --- |
| P1 分组和 | `src`（N×C） | `mean` 工作区（G×C fp32）、`count`（G×8 fp32） | `mean += src`（先当和用），同时统计成员数；fp32 累加 |
| P2 平方偏差 | `src` + `mean`（此时是 `sum / n`） | `deviation` 工作区（G×C fp32） | `deviation += (src - mean)^2` |
| P3 收尾 | `mean` + `deviation` + `count` | `mean`（原地变成均值）、`out`（G×C） | `out = sqrt(deviation / (n - correction))`，只写 `dim_size` 行 |

关键点：

- **成员计数**：P1 每处理一行就把 1 累加进该行的分组计数，只在 `col == 0` 且
  `base < dimLength` 时做一次，避免沿特征轴重复计数。计数工作区是每个分组一个
  fp32 计数器（稠密，按 32 字节对齐补齐），整块能放进 UB 时（≤ 64 KB，即
  ≤ 16384 个分组）每核在 UB 里累加、收尾按分组做一次原子加；放不下时退化为每行
  一次 32 字节原子加，原子加的目标是“含该分组的那一个 32 字节块”，块内对应通道
  置 1、其余置 0，因此不会踩到相邻分组。收尾遍按 `n = countGm.GetValue(g)` 读成员数。
- **两遍都只读一次源**：P1 与 P2 都是全量读写，P2 读的是 P1 收尾后写回的均值；
  收尾遍（P3）只处理 `batch * groups` 行、不读 `index`，与输入规模
  `batch * dimLength` 无关。
- **中心化偏差**：方差按“先求均值、再累加 `(x - mean)^2`”计算，而不是展开成
  `sum(x^2) - n * mean^2`。后者在 `1000 ± 0.01` 这类数据上把两个接近一百万的
  float32 相减来还原 `1e-4` 量级的结果，有效位全部抵消；两遍中心化只多读一遍
  源，换来与 `torch.std` 同量级的相对误差。
- **均值的保存**：`mean` 工作区先存分组和，P3 用它和 `n` 求出均值后**写回同一个
  位置**，反向直接复用，不需要再分配第四块工作区，也不需要反向重算均值。空组没有
  均值，P3 把整行写成 `NaN`（与 `out` 的口径一致）；反向只读取 `index` 实际选中的
  组，不会读到空组行。
- P3 的行数、每核行块与行批由 tiling 里的 `outRows` / `outRowChunk` /
  `outBlockNum` / `outRowBatch` 给出，与 `scatter_logsumexp` 复用同一套字段。

### 3.2 反向：1 遍 elementwise

```
grad_src[i] = grad_out[index[i]] * (src[i] - mean[index[i]]) / ((n_g - c) * out[index[i]])
```

- 一次全量读写：读 `src`、按 `index` 广播读 `grad_out` / `out` / `mean` 的分组行、
  按分组读计数，写 `grad_src`；
- 三个分组张量（`out`、`grad_out`、`mean`）都要读，`GroupsFitUb` 的两份 fp32 缓存
  装不下第三份，因此 std 的反向**全部走 GM 逐行读**，不启用「小组数常驻 UB」的
  捷径；这一点是已知的可优化项，登记在 §8.2；
- 每行的归一化因子 `(n_g - c) * out[g]` 每行算一次，元素级只做一次减、一次除、
  一次乘；
- **偏差为 0 的组梯度为 0**：`torch.std` 在反向先把方差梯度按 `out == 0` 置 0，
  因此元素完全相同的组得到 0，而不是 `0 / 0`；实现按同一规则在每块 tile 上做一次
  `CompareScalar`（`out == 0`）与一次 `Select`（真值取 0、假值取商）。比较取
  “等于 0”而不是“不等于 0”，因为空组、单元素无偏组与含 `±inf` 的组 `out` 为
  `NaN`，与 0 的无序比较必须返回假，让 `NaN` 原样保留；掩码复用已展宽的分组输出
  行缓冲，不新增 UB 分配，行缓冲末尾多留一个 256 字节 repeat 供掩码指令向上取整
  后读取；
- `out[g]` 为 `NaN`（空组、单元素的无偏组、含 `±inf` 的组）时掩码为假、保留除法
  得到的 `NaN`，与 `torch.std` 一致；
- 反向不做任何归约，不需要锁，也不需要第二套行调度。

### 3.3 复用边界

| 复用 | 内容 |
| --- | --- |
| 直接复用 | `ValidateInputs` / `BuildTiling` / `CheckLaunchable` / `ResolveDType` / `ReducedSizes`；`LoadGroups`、`CopyRowsIn/Out`、`CopyGroupRowsIn`、`CopyWorkspaceIn/Out`、`AccumulateRows`、`FlushAccumulator`；`Sum` / `Finalize` / `GradElement` 三段驱动与三遍流水 |
| 编译期扩展 | `scatter_softmax_kernel_common.h` 的模式枚举增加 `Std`；`scatter_softmax_kernel_phase.h` 增加 `CountRows` / `DeviationBatch` / `FlushCounts` 三个批处理与 `Finalize` / `GradElement` 的 std 分支，归约主体不复制 |
| 共享 tiling | `scatter_softmax_tiling.h` 新增 `countElems` 与 `correction`，`BuildTiling` 新增 `standardDeviation` 参数；std 为假时两个字段保持 0，既有算子的 tiling 字节布局不变 |
| 新增 | `Finalize` / `GradElement` 的 std 分支、成员计数路径、`scatter_std` 的 launch / tiling 实例化与两个架构的编译单元 |
| 不复用 | softmax 的 normalize 批处理（按行写 `src` 形状）与 max/exponential 两遍；std 不需要最大值 |

这样四个算子的行调度、UB/工作区预算、归约簿记只保留一份，CodeCheck 不会出现
“同一段归约代码两份”的告警。

## 4. 分块、对齐与流水

完全沿用 `scatter_softmax` 的 tiling 常量与预算：

- `SCATTER_SOFTMAX_WORKSPACE_ALIGN = 8`（32 字节）、`ROW_ALIGN = 16`（fp16/bf16 32 字节、
  fp32 64 字节）；
- 行批 `rowBatch` 由 `ROW_BUDGET_BYTES = 80 KB` 与每行 `colTile × 20` 字节推导，
  上限 64 行（批越大、每批的同步往返越少；实测 8 → 64 后 P1 的 scalar 等待从
  1.78 ms 降到 0.13 ms）；
- UB 常驻工作区上限 `ACC_BUDGET_BYTES = 16 KB`，超出即走每行一次原子写；
- **计数双路径**：`countElems * 4 <= SCATTER_SOFTMAX_COUNT_BUDGET_BYTES`（64 KB，
  即 `groups <= 16384`）时，每核在 UB 里累加自己处理的成员数、每核只 flush 一次；
  否则逐行一次 32 字节原子加，目标是“含该分组的那块”，块内对应通道为 1、其余为 0；
- 每核负责一段连续行，批内 MTE2 / V / MTE3 三段流水，P3 与反向沿用同一驱动。

P3 的行数是 `batch * groups`，通常远小于输入行数；`after` 很大时沿特征轴按
`colTile` 分块，与 P1/P2 相同。

## 5. 数值语义

| 输入 | `out` | 依据 |
| --- | --- | --- |
| 空组（`dim_size` 大于实际分组数） | `NaN` | `torch.std` 空约简；`n - c <= 0` 时直接写 `NaN` |
| 单元素组，`unbiased=True` | `NaN` | 分母 `n - 1 = 0` |
| 单元素组，`unbiased=False` | `0` | 偏差为 0、分母 1 |
| 组内含 `±inf` | `NaN` | 均值被 `±inf` 污染，`x - mean` 出现 `inf - inf` |
| 组内含 `NaN` | `NaN` | 传播 |
| 组内元素完全相同 | `0` | 偏差恒为 0 |
| 大偏移小组（`1000 ± 0.01`） | 保精度 | 两遍中心化，不出现百万量级相减 |
| 长组低精度 | 不塌缩 | fp32 归约；成员数用整数计数，不受 fp32 精度限制 |

反向：

| `out[g]` | 元素 | `grad_src` |
| --- | --- | --- |
| `NaN`（空组/单元素组/含 `±inf`） | 任意 | `NaN` |
| `0`（组内完全相同） | 相等于均值 | `0`（`torch.std` 先把方差梯度置 0） |
| 有限 | 任意 | `grad_out[g] * (x - mean) / ((n - c) * out[g])` |

上述行为均以 `torch.std` 的 CPU 实测为准（fp32/fp64），实现后逐条对拍。

## 6. 与 `torch_scatter` 的行为差异

| 场景 | `torch_scatter.scatter_std` | 本实现 |
| --- | --- | --- |
| 空组 | `NaN` | `NaN`（一致） |
| 单元素组 `unbiased=True` | `NaN` | `NaN`（一致） |
| 含 `±inf` 组 | 由组合实现决定，通常 `NaN` | `NaN`，与 `torch.std` 一致 |
| `dim` 默认值 | `-1` | `0`，与仓内 `scatter_softmax` 一致 |
| `out=` 参数 | 支持 | 不支持，与仓内 `scatter_softmax` 一致 |

后两条是仓内接口一致性取舍；数值语义与 `torch.std` 对齐，已在 API 文档中显式说明。

## 7. 测试与性能

### 7.1 用例组织（沿用 #186 的目录结构）

| 位置 | 内容 |
| --- | --- |
| `test/common/grouped_softmax_golden.py` | 家族共用 float64 参考，新增 `grouped_std_reference`（支持 `dim` / `dim_size` / `unbiased` / 空组）与复用它建图的 `std_gradient` |
| `test/common/grouped_softmax_cases.py` | 家族共用形状、dtype、非法输入用例表（不改动） |
| `test/common/test_grouped_softmax_suite.py` | 新增 `StdSuite`：继承 `GroupedFamilySuite` 的参数、梯度与非法输入用例，补齐两种归一化、空组、单元素组、大偏移精度与反向边界 |
| `test/scatter_std/arch22/` 与 `arch35/` | 各只留 `golden.py`、`test_scatter_std.py`、`benchmark_scatter_std.py`，只做绑定 |

本算子不新增 `test/common` 文件：参考实现、套件与基准全部并入既有家族模块，
`test/common` 的文件数与合入 `scatter_logsumexp` 时保持一致。

用例覆盖：

- 形状 × dtype 矩阵（1-D 特征、常规分组、空组、每组单元素、单组、`dim_size` 过大、
  `dim=1`、3-D、负维度、宽特征行分块）；
- 两种归一化（`unbiased` 真/假）在全部形状上的对拍；
- 极值语义：`±inf`、`NaN`、`±1000` 量级、单元素组、空组（两种归一化都是 `NaN`）；
- 大偏移小组保精度（`1000 ± 0.01`，与展开式实现区分开）；
- 长组不塌缩（65504 / 131072 行，fp16/bf16）；非连续 `src`；`int32`/`int64` 索引；
- 梯度对拍（fp64 参考、fp16/bf16 低精度、`unbiased=False` 的 Jacobian、常量组为
  0）；`grad_out` 在另一设备时报错；前向不修改 `src`；
- 非法输入：dtype、`index` dtype/维度/长度/越界、`dim` 越界、`dim_size` 非法、
  CPU 张量、非 Tensor、`unbiased` 非 bool。

### 7.2 性能口径

- 三口径：`kernel`（`_pybind` + 一次 `synchronize()`）、`operator`（公开 Python 入口）、
  `composed`（`argsort` + `segment_sum_csr`（和）+ 除以 `n` + 减均值 + 平方 +
  `segment_sum_csr`（平方和）+ `sqrt`，全为仓内 NPU 算子；A2 分支没有 `scatter_*`）；
- A2：7 形状 × 2 dtype 共 14 行，附 1–65536 分组扫描；
- A5：同形对比 + 速率探针 + 3 次重复（沿用 #186 的脚本与判据）；
- 预期收益来自两处：组合实现至少三遍全量读写并产生 `N×C` 的中间张量，本实现
  两遍全量 + 一遍只写 `G×C`；具体倍数以实测为准。

## 8. 实施步骤与风险

### 8.1 步骤

1. 从最新 master 开 `feat/scatter-std` 分支（共享的 phase 头改动与 #252 同一份）；
2. Host + kernel + Python + pybind + 文档 + 测试一次提交，保持“一个 issue 一个 PR”；
3. A2 全量重编 + 功能/梯度/独立校验 + 14 行基准 + 同族回归；A5 走 CI 编译与团队实测；
4. `/compile` 触发 CI，CodeCheck 必须 0 条（不屏蔽）；squash 成单提交；
5. 测试报告 v1.0（时间戳，A2/A5 分别标注）。

### 8.2 风险与对策

| 风险 | 对策 |
| --- | --- |
| 改 `scatter_softmax_kernel_phase.h` 影响已合入的三个算子 | `Std` 模式只在实例化时编译；既有算子的全量用例与基准复跑 |
| 计数工作区随分组数线性增长 | 每分组 4 字节（按 32 字节对齐补齐）；`groups > 16384` 时退化为逐行原子加，功能不变，基准里加 `dim_size = 65536` 一档观察 |
| 反向读三份分组张量，小组数时 GM 命中率低 | 已登记为优化项：可把 `mean` 与 `out` 合并成一份 fp32 常驻 UB 缓存，或按 `after` 分块时复用行缓冲；当前以正确性优先 |
| 两遍中心化比展开式多读一遍源 | 展开式在大偏移数据上不可用；仅在 `after` 分块内复用行缓冲以降低带宽，实测倍数见报告 |
| `torch_scatter` 未安装，三方对比缺少直接基线 | 对比基线用 `torch.std` 逐组循环（float64/float32）与仓内组合实现；若环境可安装再补 `torch_scatter` 直测 |

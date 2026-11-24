# edge_softmax 算子设计文档（A2/A3 arch22，950 arch35）

> 状态：已实现。功能、精度与性能实测数据见对应 PR 的测试报告；本文件说明接口、
> 语义、复用边界与分块方案。
> 依赖：`scatter_softmax`、`scatter_log_softmax`、`scatter_logsumexp`、`scatter_std`
> 已合入，本算子复用它们的 Host 共享件与设备侧流水；CSR 的段语义与
> `segment_logsumexp` 保持一致。

## 1. 需求与验收口径

实现 `dgl.nn.functional.edge_softmax` 的 NPU 版本：

```python
edge_softmax(src, indptr)  # Tensor
```

CSR 的每一行对应一个目标节点的入边集合，`indptr` 给出段边界，`src` 的第 0 维
长度为 `indptr[-1]`。对段 `i` 的每条边 `e ∈ [indptr[i], indptr[i+1])`：

```
out[e] = exp(src[e] - max_i) / sum_{j ∈ 段 i} exp(src[j] - max_i)
```

尾部特征（`src.shape[1:]`）逐元素独立归一化。验收口径：

| 项 | 要求 |
| --- | --- |
| 接口 | 与 `dgl.nn.functional.edge_softmax` 对齐：CSR 指针描述分组，不提供 `dim` / `dim_size` / `out=` |
| 输出 | 形状、dtype、device 均与 `src` 相同（与 DGL 一致，不是约简结果） |
| dtype | `float16` / `float32` / `bfloat16`；最大值、指数与求和全程 fp32 |
| indptr | 同设备 1-D `int32` / `int64`，长度 = `段数 + 1`；必须以 0 开头、单调不减、末位等于 `src.size(0)` |
| 求导 | 支持 autograd，`grad_src[e] = out[e] * (grad_out[e] - sum_{j ∈ 段(e)} out[j] * grad_out[j])` |
| 平台 | arch22（A2/A3）与 arch35（950）共用同一份 kernel 源码，两个架构目录只有薄编译单元 |
| 性能 | 对 `ptr2ind + scatter_softmax` 组合有净收益：少一次 E 长索引的物化与回读 |

## 2. Host 维度与输出契约

复用 `scatter_softmax` 的批/组/特征三分解，只是 `dim` 固定为 0，组数由指针长度
给出：

- `batch  = 1`（CSR 只在第 0 维分段）
- `dimLength = src.size(0) = indptr[-1]`
- `groups = indptr.numel() - 1`
- `after  = prod(src.shape[1:])`，按 8 个元素对齐得到 `afterPadded`
- 输入行数 `rows = dimLength`，约简输出行数 `outRows = groups`

因此 tiling 直接复用 `grouped_softmax::BuildTiling(src, 0, segments)`：工作区仍是
`workspaceElems = groups * afterPadded` 的 fp32，与 `scatter_softmax` 的
`batch * groups * afterPadded` 同阶。

Host 侧新增 `csrc/npu/edge_softmax/common/edge_softmax_shared.h`，给出两个入口：

```cpp
torch::Tensor edge_softmax_forward(src, indptr);
torch::Tensor edge_softmax_backward(out, grad_out, indptr);
```

- `ValidateCsr` 只做设备侧真正需要的检查：`src` contiguous / 至少 1 维 / dtype 合法，
  `indptr` contiguous / 1-D / int32 / 同设备 / 至少两个边界（即至少一个段），段数与行数
  落在 int32 范围内（内核每组一个 int32 标签、每段一个 int32 游标）。少于两个边界时
  没有任何段，而下面的空分支返回的是未初始化的 `src` 形状张量，所以在这里直接拒绝，
  直接调 C++ 入口的使用者也能拿到明确报错而不是脏数据。
- 指针取值（非 0 开头、递减、末位不等于 `E`）在 Python 层用一次归约同步判定，失败
  时再拷回 CPU 定位第一个越界下标，抛出带下标的 `ValueError`。这条检查不放在
  Host C++ 里，避免 C++ 接口每次调用都要同步一次设备。
- 内核本身对任意 int32 指针都是内存安全的：游标在 `groupCount - 1` 处停止，越界
  指针只会让尾部若干行落入最后一段，不会读写到指针数组或工作区之外。
- `workspaceElems == 0`、`rows == 0` 或 `after == 0` 时直接返回 `empty` 结果，不启动
  kernel；这三条在"至少一个段"的前提下都意味着输出本身就是空张量。工作区
  `maximum` 预填 `SCATTER_SOFTMAX_NEUTRAL_MAXIMUM`，`sum` 置零。

## 3. 设备侧算法

设备侧不新增内核，而是给 `scatter_softmax` 的共享流水增加一个编译期模式
`kModeEdgeCsr`：算子宏 `SCATTER_GROUPED_SOFTMAX_OPERATOR(EdgeSoftmax, EdgeLaunch,
kModeEdgeCsr)` 直接展开出五个 kernel 与两个导出入口。

### 3.1 前向：3 遍全量

| 遍 | 读 | 写 | 说明 |
| --- | --- | --- | --- |
| Maximum | `src` | `maximum` | 段内最大值，fp32 原子 max（或 UB 常驻累加） |
| Exp | `src`、`maximum` | `out`、`sum` | `exp(src - max)` 按源 dtype 写回，同时段内累加 |
| Normalize | `out`、`sum` | `out` | 逐元素除以段内和 |

### 3.2 反向：2 遍全量

| 遍 | 读 | 写 | 说明 |
| --- | --- | --- | --- |
| GradSum | `out`、`grad_out` | `sum` | 段内累加 `out * grad_out` |
| Grad | `out`、`grad_out`、`sum` | `grad_src` | `out * (grad_out - sum)` |

### 3.3 组标记的来源：CSR 游标

`scatter_softmax` 的每个行批一次 `DataCopyPad` 读入 `rowBatch` 个组标签；CSR 模式
下组的标签由指针推出，`LoadGroups<kModeEdgeCsr>` 改为：

- 行按顺序遍历，段号单调不减，因此用一个段游标 + 下一个边界 `csrNext` 推进：
  每跨一个段边界只读一次 `indptr`，而不是每行读一次标签；
- 每核的第一个行批用二分查找定位起点（`O(log 段数)` 次标量读），此后沿用游标；
- 空段（`indptr[i] == indptr[i+1]`）在推进时被跳过，因此行总能落到包含它的段。

这套游标的状态（`csrSegment` / `csrNext` / `csrColumn` / `csrStarted`）挂在
`PhaseTensors` 上，随每次 kernel 启动重置，因此不跨遍、不跨核共享。

### 3.4 复用边界

共享件的改动只有三处，且都是编译期开关，不影响已合入算子的行为：

| 位置 | 改动 |
| --- | --- |
| `scatter_softmax_kernel_common.h` | 新增 `kModeEdgeCsr`；把 `IsLogDomain` 从“非 softmax”收敛成显式 log 模式判定，并新增 `IsSoftmaxDomain`（softmax / edge CSR） |
| `scatter_softmax_kernel_phase.h` / `_io.h` | `kStore`、log 归一化、`LoadWorkspace` 等处改用上述两个判定；`LoadGroups` 增加 CSR 分支 |
| `scatter_softmax_kernel_entry.h` | `kModeEdgeCsr` 下 `indexGm` 按 `groupCount + 1` 绑定（标签模式下仍是 `dimLength`） |

`scatter_std` 走 `kModeStd`，`scatter_logsumexp` 走 `kModeLogSumExp`，两者都不实例化
`kModeEdgeCsr` 的分支，因此已合入算子的编译产物与行为不变。

## 4. 分块、对齐与流水

行批、tile 与流水沿用 `scatter_softmax`：每核负责一段连续的行，按
`rowBatch`（最多 64 行，按 UB 预算自适应）装载，特征方向按 32 字节对齐的
`colTile` 分块；三个阶段每批各同步一次。与标签模式相比，CSR 模式的组标记是标量
循环而非一次 `DataCopyPad`：

- 宽特征（`after >= colTile`）时组标记的开销被向量计算摊薄，几乎不可见；
- 窄特征（如 GAT 的 `after = 8`）时每批只有 64 次标量写，与逐行的向量指令同阶，
  因此不值得为它引入一次额外的搬运；
- 若将来需要进一步压缩，可把“每批行号 → 段号”改成向量比较 + `Select`，但那会把
  一个 O(行数) 的标量循环换成 O(行数) 的向量指令，只在 `after` 很小时才有意义。

## 5. 数值语义

- 空段没有边，不产生输出元素；`indptr` 全为 0（`E = 0`）时结果是空张量；
- 单边段输出 `1`（`exp(0) / 1`）；
- 段内含 `+inf`：最大值为 `+inf`，`exp(+inf - +inf) = NaN`，该段每个元素的该特征
  列为 `NaN`；含 `NaN` 的段同样输出 `NaN`，与 `torch.softmax` 对相同输入的行为一致；
- 段内含 `-inf` 且全部为 `-inf`：最大值为 `-inf`，`exp(-inf - -inf) = NaN`，
  与 `torch.softmax` 一致；
- 段内和与最大值都在 fp32 中累加，fp16/bf16 的段长不受 65504 上限限制；
- 反向使用前向保存的 `out`（源 dtype）作为概率，`grad_src` 按源 dtype 舍入。

## 6. 与 DGL / 组合实现的行为差异

| 项 | 说明 |
| --- | --- |
| `dim` | DGL 的 `edge_softmax` 只在 CSR 的第 0 维分段；本算子相同，不提供 `dim` 参数 |
| 指针 dtype | DGL 两个架构都接受 int64；本算子接受 int32/int64，Python 层窄化为 int32（`E` 与段数超过 int32 范围时显式报错） |
| 指针校验 | DGL 不做校验；本算子校验 0 起点、单调不减与末位，非法输入抛 `ValueError` |
| 与组合实现 | `ptr2ind(indptr, E) + scatter_softmax` 语义等价，少一条 E 长的 int32 索引写入与回读；测试里以该组合作为交叉验证 |
| 空段 | 与 DGL 一致：空段不产生输出 |

## 7. 测试与性能

### 7.1 用例组织

沿用 `test/<算子>/arch22|arch35` 结构，每个目录只保留 `golden.py`、
`test_edge_softmax.py`、`benchmark_edge_softmax.py`：

- `golden.py`：把 `indptr` 展开为逐边段标签后交给 float64 的 `torch.softmax`
  参考实现，不导入 `ops_gnn`；
- `test_edge_softmax.py`：形状/dtype 矩阵（含 1-D、ND、窄/宽特征、空段、单边段、
  长段、上千个小段）、前向/反向对拍、int64 指针、非连续输入、`±inf`/`NaN`、
  与 `ptr2ind + scatter_softmax` 组合的一致性、非法参数（10 例）；
- `benchmark_edge_softmax.py`：同一张均匀度图（`nodes × degree`）上，前向与
  “前向+反向”分别对比 `ptr2ind + scatter_softmax` 组合。

arch22 与 arch35 的用例集与参考实现相同（arch35 通过路径复用 arch22 的实现），
平台差异只体现在实测数据上；`test/common` 不新增文件。

### 7.2 性能口径

- 目标场景：`nodes = 16384`、`degree = 64`（约 1M 边）、`features = 128`，dtype
  覆盖 fp16/fp32；另扫一组不同度（如 8 / 32 / 64 / 128）观察长段与短段的差异；
- 对比基线：`ptr2ind + scatter_softmax`（同一 NPU、同一 `src`、同一张图）；
- 计时：`torch.npu.Event`，预热 10 次、计时 50 次取均值；每次运行至少重复一遍以
  确认波动；
- 报告按 A2 / A5 两个平台分别标注，不混合统计。

## 8. 实施步骤与风险

### 8.1 步骤

1. 共享件增加 `kModeEdgeCsr` 与 CSR 组标记分支，保持已合入算子行为不变；
2. 新增 `csrc/npu/edge_softmax/{common,op_host,op_kernel/{arch22,arch35,common}}`，
   绑定 `edge_softmax_forward/backward`；
3. Python 层 `python/ops_gnn/edge_softmax.py`：参数校验 + CSR 指针校验 + autograd；
4. 测试与文档（`README`、`docs/zh|en/api_reference.md`、本设计文档）；
5. 先 A5（950）构建/功能/性能，再 A2（arch22）复测，按平台分别出数据；
6. 提 PR 关联 issue，等 CI 与检视。

### 8.2 风险与对策

| 风险 | 对策 |
| --- | --- |
| 改共享件影响已合入算子 | 新模式只走 `if constexpr` 分支；把 `IsLogDomain` 的语义收敛为显式枚举后，逐一核对 `scatter_softmax` / `scatter_log_softmax` / `scatter_logsumexp` / `scatter_std` 的实例化路径，回归全量用例 |
| 指针非法导致越界 | 游标封顶在最后一段，内核不依赖指针合法性；Python 层做语义校验并给出带下标的错误 |
| 段内极长（十万级边） | 归约全程 fp32，`sum` 与 `maximum` 不按段长分块，长段不塌缩；用例覆盖 70000 边的段 |
| 段数很多而 `after` 很小 | 工作区为 `段数 × afterPadded`，与 `scatter_softmax` 同阶；行批仍按 UB 预算自适应，窄特征下每批最多 64 行的标量标签可接受 |
| 两平台数据混合 | 报告按平台分段呈现，benchmark 脚本不带平台默认值，由运行环境决定 |

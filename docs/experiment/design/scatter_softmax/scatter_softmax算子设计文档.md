# scatter_softmax 算子设计文档 (A2/A3 arch22, 950 arch35)

## 1. 需求与验收口径

实现 `torch_scatter.scatter_softmax` 的 NPU 版本：

```python
scatter_softmax(src, index, dim=0, dim_size=None)  # Tensor
```

- 对每个分组 `g`，沿 `dim` 计算 `out[i] = exp(src[i] - max_g) / sum_j exp(src[j] - max_g)`；
- 输出形状、dtype、设备与 `src` 相同，支持 float16/float32/bfloat16；
- 反向按分组 softmax 的 Jacobian 解析计算，支持 autograd；
- 算子必须自带设备侧实现：`csrc/npu/scatter_softmax/`，不再用
  `segment_max_csr` / `segment_sum_csr` / `index_add_` 拼接；
- arch22（A2/A3）与 arch35（950）使用同一份 AscendC 源码
  （`op_kernel/common/scatter_softmax_kernel.h`），各架构只有一层薄编译单元。

验收：`test/scatter_softmax/` 全部 74 例通过（含精度、边界、`±inf`、长分组、
低精度梯度）；A2 全量回归 4804 passed。

## 2. Host 维度与输出契约

```text
before     = prod(src.shape[:dim])           # 批维度
dimLength  = src.shape[dim]                  # 被 index 索引的归约维
after      = prod(src.shape[dim + 1:])       # 尾部特征维
rows       = before * dimLength
src[b, j, a] 地址 = (b * dimLength + j) * after + a
```

- `index` 必须是长度等于 `src.size(dim)` 的一维张量；`index.numel() == src.numel()`
  的广播形式不在本算子支持范围内（与既有 Python 版契约一致）。
- `src` 先做连续化；输出为同形状的新连续张量。
- 归约工作区是 fp32，行步长 `afterPadded = round_up(after, 8)`，因此每个
  (batch, group) 工作区行都是 32 字节的整数倍。
- `src.numel() == 0` 时直接返回零张量，不启动 kernel。
- 使用 `src` 所在设备的当前 NPU 流，不创建临时流、不做 host 同步。

## 3. 设备侧算法

`index` 不要求有序，逐组归约由 MTE3 原子单元完成，共 5 个 kernel 发射：

| # | kernel | 计算 |
| --- | --- | --- |
| 1 | `ScatterSoftmaxMaxKernel` | `atomicMax(maximum[b][g][a], src[b][j][a])`，`g = index[j]` |
| 2 | `ScatterSoftmaxExpKernel` | `out = exp(src - maximum)`，同时 `atomicAdd(sum[b][g][a], out)` |
| 3 | `ScatterSoftmaxNormalizeKernel` | `out /= sum` |
| 4 | `ScatterSoftmaxGradSumKernel` | `atomicAdd(sum[b][g][a], out * grad_out)` |
| 5 | `ScatterSoftmaxGradKernel` | `grad_src = out * (grad_out - sum)` |

前向 3 个 kernel 与反向 2 个 kernel 依次在同一流上发射，kernel 边界即为同步点，
不需要额外的跨核同步。

## 4. 分块、对齐与流水

- **行**：每个 AIV 核负责一段连续的行，
  `rowChunk = ceil(rows / blockNum)`，`blockNum = min(AIV 核数, rows)`。
  核内一次缓冲 `rowBatch` 行（按 UB 预算自适应，上限 8），
  批内每行记录自己的组号与源行基址。
- **列**：`colTile = min(2048, round_up(after, 16))`，尾部维度按 `colTile`
  分块。`colTile` 同时是批内每行在 UB 中的步长：取 16 的倍数后，
  fp16/bf16 的 32 字节与 fp32 的 64 字节都是 32 字节的整数倍，
  每个切片基址都能满足 UB/MTE 的 32 字节对齐要求。
- **原子写长度**：`width = min(colTile, afterPadded - col)`，恒为 8 的倍数，
  不会越过本行的补齐区写坏相邻分组。
- **行访问**：源/输出行偏移 `(row * after + col)` 不保证对齐，统一用
  `DataCopyPad` 只搬运 `valid * sizeof(T)` 字节；工作区读写用对齐的
  `DataCopy`（`width` 为 8 的倍数、行基址为 `afterPadded` 的倍数，
  两端因此都是 32 字节对齐）。
- **流水事件**：每个 kernel 申请 4 个事件
  （`MTE2_V`、`V_MTE2`、`V_MTE3`、`MTE3_V`），每批只做一次
  “载入 → 计算 → 写回 → 复用缓冲”的同步，而不是每行一次；
  批内各行的装载与上一批的写回排在同一事件链上。
- **归约记账**：`batch × 分组数 × afterPadded` 能放进 16 KB 工作区预算时，
  整个 pass 期间工作区常驻 UB：归约阶段在 UB 内累加，末尾每核只向工作区
  原子回写一次；读工作区的阶段每趟只读一次，不再每行读一次。
  放不进时，归约退回逐行原子写、读回退到逐行读。
  两条路径都只依赖组号，`index` 无需有序。
- **常驻的代价**：`exp` 阶段同时持有“缓存的 maximum 工作区”和“累加 sum
  的工作区”，因此行预算按两份预留；两份各自上限 16 KB，
  `after = 128` 时 `rowBatch` 由行预算推到 32。
- **尾部补位**：所有向量指令都写在 32 字节对齐的 buffer 基址上。
  只有当本块还存在补位车道（`valid < width`）时，才先用 `Duplicate`
  覆盖整块（最大值用 neutral、和用 0），再用有效长度覆盖前 `valid` 个元素，
  从而避免在非对齐偏移上做向量写；整块填满时跳过这次覆盖。
- **批大小与同步**：`rowBatch` 上限 64 行。每批只做一轮 MTE2/V/MTE3 同步，
  批太小则同步往返成为主要开销：1M × 32 的归约遍把上限从 8 提到 64 后，
  scalar 等待从 1.78 ms 降到 0.13 ms。
- **UB 预算**：行缓冲 80 KB（每 (行, 特征) 按 20 字节估算两条 dtype 缓冲
  加三个 fp32 分块）、工作区副本最多两份各 16 KB（累加器与读缓存），
  两者同时占用；分配失败不会发生，因为 `rowBatch` 就是按总预算反推的。

## 5. 数值语义

- 逐组最大值、`exp` 和、归一化、反向全部在 float32 中完成；`exp` 与输出按源
  dtype 舍入（`CAST_RINT`）。fp16/bf16 的单组长度不受 65504 上限限制。
- 工作区最大值初值 = kernel 补位值 = `-FLT_MAX`（`SCATTER_SOFTMAX_NEUTRAL_MAXIMUM`），
  保证补齐车道永远不会成为分组最大值，也不会和 host 初始化产生语义差异。
- 空分组没有元素，不产生输出。
- 整组为 `-inf`：分组最大值保持 neutral，`exp(-inf - neutral) = 0`，
  和为 0，`0 / 0 = NaN`，与 `torch.softmax` 一致。
- 分组内含 `+inf`：最大值为 `+inf`，`exp(+inf - +inf) = NaN`，和为 NaN，
  整列输出 NaN。
- 和为零的语义完全由 `Div` 产生（`0/0 = NaN`），不做特判。

## 6. 与 `torch_scatter` 的行为差异

- 只支持 `dim` 维长度为 `index.numel()` 的一维分组索引；不支持把 `index`
  广播成与 `src` 同形状。
- `dim_size` 允许大于 `index.max() + 1` 以保留空分组；`dim_size` 必须能放进
  int32（设备侧按 int32 读取组号）。
- 同一分组内的累加顺序由原子单元决定，浮点结果可能有末位差异，量级远小于源
  dtype 容差；同一输入多次运行的结果不保证逐位相同。

## 7. 测试与性能

- 功能：`test/scatter_softmax/test_scatter_softmax.py`，74 例，覆盖 1-D/2-D/3-D、
  空分组、单元素分组、`dim`/`dim_size` 推导、int32 索引、非连续源、大数、
  `+inf`/`-inf`、长低精度分组、fp32/fp16/bf16 梯度、非法参数。
- golden：`test/scatter_softmax/golden.py`，CPU float64 独立实现。
- 性能（Ascend 910B4，`benchmark_scatter_softmax.py` 口径，iters=20 × 5 次取中位数）：

| 形状 (edges × features, groups) | dtype | 旧 Python 组合 | 修正前内核 | 当前内核 |
| --- | --- | --- | --- | --- |
| 65536 × 1024, 64 | float32 | 23.33 ms | 3.65 ms | 2.44 ms |
| 65536 × 1024, 64 | float16 | 21.13 ms | 3.06 ms | 1.66 ms |
| 131072 × 1, 1 | float32 | 72.79 ms | 16.66 ms | 10.30 ms |
| 131072 × 1, 1 | float16 | 70.48 ms | 14.81 ms | 10.16 ms |
| 262144 × 256, 4096 | float32 | 77.61 ms | 10.36 ms | 3.54 ms |
| 1048576 × 128, 16384 | float32 | 305.56 ms | 38.87 ms | 12.31 ms |
| 1048576 × 128, 1024 | float32 | 250.79 ms | 38.69 ms | 12.06 ms |
| 1048576 × 128, 4 | float32 | 249.19 ms | 45.26 ms | 23.84 ms |
| 1048576 × 128, 1 | float32 | 430.90 ms | 147.11 ms | 82.82 ms |

- 行批处理 + 核内累加带来 1.46–3.21倍的提速，收益最大的是分组数多、
  `after` 宽的形状（原先多核抢同一行工作区）；`after = 1` 的窄行仍受
  单行 DMA 边长限制，但已从每行一次同步降到每批一次。

## 8. 已知取舍与后续优化

- 工作区大小为 `2 × 分组数 × afterPadded × 4` 字节（最大值、和），显存占用随
  分组数线性增长；分组数极大而每分组元素很少时空间效率偏低。
- 三遍前向各自读一遍 `src`（第三遍读 `out`），后续可把最大值与 `exp` 合并为
  “一次读入、两阶段归约”，或在分组连续（已排序）时退化为 owner-row 确定路径。
- 反向复用前向输出 `out`，因此需要保存一份与 `src` 同大的张量。
- 行批处理以 UB 为代价换取并行度：`rowBatch` 越大，每批同步次数越少，
  但每个核占用的 UB 也越多；当前上限 64 行，行预算仍按 80 KB 反推，
  窄行（`after <= 128`）能取到 64，宽行按预算自动降到 1–8 行。
- 工作区常驻 UB 把“每行一次原子写 + 每行一次工作区读”换成
  “每核一次元素归约 + 一次原子回写 + 每趟一次工作区读”，
  代价是每个核最多两块 16 KB 的工作区副本；工作区超过 16 KB
  时自动退回逐行原子写与逐行读，因此分组数极大的形状仍保持原有行为。
  单组/少数组的收益最大：一百万行归到同一组时，每行读的是同一块 512 B，
  这类逐行读互不重叠，A2 上单趟 46 ms，常驻后 4 ms。

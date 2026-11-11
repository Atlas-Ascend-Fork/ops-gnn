# segment_logsumexp 算子设计文档 (A2/A3 arch22, 950 arch35)

## 1. 需求与验收口径

实现 `torch_scatter.segment_logsumexp` 的 NPU 版本：

```python
segment_logsumexp(src, indptr, out=None)  # Tensor
```

- 按 CSR 行指针归约，段 `i` 覆盖行区间 `[indptr[i], indptr[i + 1])`，
  输出 `out[i] = log(sum_j exp(src[j]))`；
- 输出形状为 `(indptr.numel() - 1,) + src.shape[1:]`，dtype/设备与 `src` 相同，
  支持 float16/float32/bfloat16；
- 空段输出 `-inf`；整段只有 `-inf` 输出 `-inf`，只有 `+inf` 输出 `+inf`
  （与 `torch.logsumexp` 一致）；
- 算子必须自带设备侧实现：`csrc/npu/segment_logsumexp/`，不再用
  `segment_max_csr` / `segment_sum_csr` 拼接；
- arch22（A2/A3）与 arch35（950）使用同一份 AscendC 源码
  （`op_kernel/common/segment_logsumexp_kernel.h`），各架构只有一层薄编译单元。

验收：`test/segment_logsumexp/` 全部 86 例通过（含精度、空段、`±inf`、非对齐
特征宽、超过一个列分块的宽特征、长段、行切块）；A2 全量回归 4816 passed；
950 编译验证通过，A5 实机复测由团队完成。

## 2. Host 维度与输出契约

```text
rows       = src.size(0)                     # 必须等于 indptr[-1]
features   = prod(src.shape[1:])              # 尾部特征维展平
segments   = indptr.numel() - 1
src[row, a] 地址 = row * features + a
out[s, a]   地址 = s * features + a
```

- `indptr` 在 Python 侧校验（非负、非降序、`indptr[-1] == src.size(0)`）并转为
  int32；kernel 只读取 int32 标量边界，因此要求 `src.size(0) <= 2**31 - 1`。
- `out=` 时先算到新的连续张量再 `out.copy_(result)`，因此支持非连续视图。
- `indptr.numel() == 1`（零段）在 Python 侧直接返回空张量，不发射 kernel。
- `indptr[0] > 0` 的前导行不属于任何段；kernel 只遍历各段的行区间，天然忽略
  这些行，无需像旧实现那样先 rebase 指针。
- 平台核数（`ResolveAivCoreNum`）每次调用只查一次，其余并行度都由它派生，
  避免重复查询平台的固定开销。

## 3. 设备侧算法：两级归约，两种调度

CSR 段是**连续行区间**，不需要重排，也不需要原子单元。**一个核只负责一段**
在核数远多于段数时会闲置（A5 有 64 个 vector core，GNN 的段数常等于 batch
规模），因此按段数选择两种调度：

### 3.1 分段已能填满机器：单 kernel 融合（`rowChunks == 1`）

一个核领取一个「段 × 特征窗口」任务，在同一个 kernel 内完成
`段内逐特征最大值 → 夹取 → sum(exp(src - shift)) → shift + ln(sum)`，
中间结果全部留在 UB，不落 workspace，也不产生额外 kernel 发射。

### 3.2 段数不足：行切块 + 两级归约（`rowChunks > 1`）

把每段的行再切成 `rowChunks` 块，四个 kernel 依次在同一 stream 上发射：

| 阶段 | 任务网格 | 计算 | 交换 |
| --- | --- | --- | --- |
| 1 | (段, 窗口, 块) | 块内逐特征最大值 | → `splitMax` |
| 2 | (段, 窗口) | 对块取最大值并夹到 `±FLT_MAX` → shift | `splitMax` 首块原地写回 |
| 3 | (段, 窗口, 块) | `sum(exp(src - shift))` | → `splitSum` |
| 4 | (段, 窗口) | 对块求和 → `shift + ln(sum)` → 输出 | — |

- kernel 边界就是跨核同步点，不需要额外屏障。
- 阶段 2 把结果写回该窗口第一个块的位置：这些 slot 只由当前核访问，原地归约安全。
- workspace 为 `2 × tasks × colTile` 个 fp32，每个条目先写后读，无需初始化。

### 3.3 切块粒度

```text
wanted      = ceil(核数 / baseTasks)          # baseTasks = segments × colTiles
affordable  = rows / (segments × 16)          # 每块至少 16 行
rowChunks   = clamp(min(wanted, affordable), 1, 64)
```

段内按 `per = ceil(段长 / rowChunks)` 均匀切分，最后一块按段尾截断；段长小于
`rowChunks` 时后面若干块为空块（贡献中性值 `-inf` / `0`）。行数太少时不切块，
避免为几次搬运付四次发射。

## 4. 分块、对齐与流水

- **任务领取**：`task = blockIdx; task += blockCount`，`blockCount` 取
  `min(平台核数, 该阶段任务数)`。
- **列**：`colTile = min(1024, round_up(features, 8))`，尾部特征维按 `colTile`
  分块；tile 起始列恒为 8 的倍数，所以 fp32 buffer 的向量基址恒为 32 字节对齐。
- **行批量搬运**：单行只有几百字节，逐行搬运会让每行都付一次 MTE2 完成时延。
  整行窗口（`colTiles == 1`）时一次 `DataCopyPad` 搬运 `rowBatch` 行
  （`rowBatch = min(64, 16KB / 行字节数)`），burst 数下降一个量级；窗口窄于整行时
  回退为逐行搬运，因为此时行间距需要源 stride，而其单位在两个架构上不一致，
  不做跨架构猜测。
- **末尾不足一个 burst**：`NextBatch` 只搬剩余行数，不会越界读。
- **非对齐访问**：源行/输出行偏移不保证 32 字节对齐，统一用 `DataCopyPad` 只搬运
  `valid * sizeof(T)` 字节；UB 侧按 32 字节补齐，补齐区不写回 GM。
- **workspace 访问**：整行 `colTile` 个 fp32，32 字节对齐，同样用 `DataCopyPad`
  按字节数搬运。
- **流水事件**：申请 4 个事件（`MTE2_V`、`V_MTE2`、`V_MTE3`、`MTE3_V`），
  每次 `SetFlag` 后立即 `WaitFlag`，覆盖「载入 → 计算 → 写回 → 复用缓冲」。
- **UB 用量**：`16KB` 批量缓冲 + 4 个 `colTile` 大小的 buffer（T 一个、fp32 三个），
  `colTile = 1024` 的 fp32 情形约 32KB。

## 5. 数值语义

- 段内最大值、`sum(exp(src - shift))` 与 `ln` 全部在 float32 中完成，`exp` 与输出
  按源 dtype 舍入；fp16/bf16 的单段长度不受 65504 上限限制
  （`exp(src - shift) <= 1`，只有累加值可能超过 65504，而累加在 fp32）。
- 先取最大值再移位，因此不会出现 `exp(大正数)` 溢出。
- `shift` 用 `Mins/Maxs` 夹到 `[-FLT_MAX, +FLT_MAX]`：空段/全 `-inf` 段的
  `shift = -FLT_MAX`，`exp(-inf + FLT_MAX) = 0`，`ln(0) = -inf`，结果 `-inf`；
  含 `+inf` 的段 `shift = FLT_MAX`，`src - shift = +inf`，和为 `+inf`，结果
  `+inf`。这样既避免 `inf - inf`，也与 `torch.logsumexp` 一致。
- 空段（`indptr[i] == indptr[i + 1]`）没有行，`sum = 0`、`ln(0) = -inf`，
  与上面的夹取配合直接得到 `-inf`，不需要单独分支。
- 段内求和顺序固定（按行号升序，块内再按块号升序），**同一调度下**同一输入多次运行
  结果逐位一致；`fused` 与 `split` 两条调度的浮点加法顺序不同（前者逐行累加到单个
  fp32 累加器，后者先算块内部分和再相加），跨调度结果在 fp32 容差内相等但不逐位相等。

## 6. 与 `torch_scatter` 的行为差异

- 要求 `indptr[-1] == src.size(0)`，不接受段外的尾部行（比 `segment_csr` 严格）。
- `indptr` 在内部收窄为 int32，因此 `src.size(0)` 不能超过 `2**31 - 1`。
- `out=` 传入非连续张量时按 stride 拷回，返回的仍是同一个对象。

## 7. 测试与性能

- 功能：`test/segment_logsumexp/test_segment_logsumexp.py`，86 例，覆盖 1-D 特征、
  多维尾部特征、空段（首/中/尾/全空）、零段、单元素段、`±inf`、非对齐特征宽
  （100）、超过一个列分块的宽特征（2000）、多 burst 段、行切块（2000 行 2 段、
  5000 行 1 段）、`out=` 与非连续 `out`、非法参数、长低精度段。
- golden：`test/segment_logsumexp/golden.py`，CPU float64 独立实现（不 import
  `ops_gnn` / `torch_scatter`）。
- 性能（Ascend 910B4 A2，`benchmark_segment_logsumexp.py`，iters=50）：

| 形状 (rows × features, segments) | dtype | 旧 Python 组合 | 本实现 | 加速比 |
| --- | --- | --- | --- | --- |
| 1048576 × 128, 16384 | float32 | 47.649 ms | 3.735 ms | 12.8× |
| 1048576 × 128, 16384 | float16 | 46.473 ms | 3.404 ms | 13.7× |
| 262144 × 256, 4096 | float32 | 13.848 ms | 1.399 ms | 9.9× |
| 262144 × 256, 4096 | float16 | 13.214 ms | 1.246 ms | 10.6× |
| 65536 × 128, 64 | float32 | 1.529 ms | 0.583 ms | 2.6× |
| 65536 × 128, 64 | float16 | 1.554 ms | 0.566 ms | 2.7× |
| 262144 × 128, 4（长段） | float32 | 19.226 ms | 1.210 ms | 15.9× |
| 262144 × 128, 4（长段） | float16 | 18.570 ms | 1.070 ms | 17.4× |

段数扫描（`1048576 × 128`，float32，iters=20），验证行切块把耗时拉平：

| segments | 本实现 | 旧 Python 组合 | 加速比 |
| --- | --- | --- | --- |
| 1 | 3.803 ms | 284.487 ms | 74.8× |
| 4 | 3.783 ms | 75.403 ms | 19.9× |
| 16 | 6.049 ms | 23.161 ms | 3.8× |
| 64 | 4.589 ms | 14.600 ms | 3.2× |
| 256 | 3.919 ms | 13.380 ms | 3.4× |
| 1024 | 3.657 ms | 14.604 ms | 4.0× |
| 4096 | 3.645 ms | 20.986 ms | 5.8× |
| 16384 | 3.695 ms | 47.708 ms | 12.9× |
| 65536 | 4.352 ms | 156.377 ms | 35.9× |

「旧 Python 组合」= 同一分支上 `segment_max_csr` + `segment_sum_csr` + `log` 的
实现，在同一台机器、同一脚本内测得。

### 7.1 A5（950PR）实测

第一版只有「一个核负责一段」这一种调度，在 A5（64 个 vector core）上是**负收益**：
段数 4 时 0.18×、16384 段时 1.06×——因为组合实现按行并行、能吃满 64 核，而单级调度
最多只能用到 `segments` 个核。行切块两级归约就是针对这一点引入的。

A5 实机（Ascend 950PR，64 vector core，iters=20、5 次取中位数，机器空闲）复测：

| 形状（rows × features，segments） | dtype | 内核级 | 组合实现 | 内核/组合 |
| --- | --- | --- | --- | --- |
| 1048576 × 128，16384 | fp32 | 2.179 ms | 3.597 ms | 1.65× |
| 1048576 × 128，16384 | fp16 | 1.986 ms | 2.906 ms | 1.46× |
| 262144 × 256，4096 | fp32 | 0.700 ms | 2.262 ms | 3.23× |
| 262144 × 256，4096 | fp16 | 0.603 ms | 1.665 ms | 2.76× |
| 65536 × 128，64 | fp32 | 0.156 ms | 0.852 ms | 5.45× |
| 65536 × 128，64 | fp16 | 0.151 ms | 0.931 ms | 6.18× |
| 262144 × 128，4（长段） | fp32 | 0.527 ms | 1.872 ms | 3.56× |
| 262144 × 128，4（长段） | fp16 | 0.503 ms | 1.582 ms | 3.15× |

段数扫描（`1048576 × 128` fp32）——曲线被拉平到 2.1–2.6 ms，不再随段数塌陷：

| segments | 内核级 | 算子级 | 组合实现 | 内核/组合 | 算子/组合 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.316 ms | 2.893 ms | 14.532 ms | 6.27× | 5.02× |
| 4 | 2.291 ms | 2.887 ms | 6.478 ms | 2.83× | 2.24× |
| 16 | 2.291 ms | 2.928 ms | 3.728 ms | 1.63× | 1.27× |
| 64 | 2.265 ms | 2.956 ms | 3.433 ms | 1.52× | 1.16× |
| 256 | 2.180 ms | 2.864 ms | 3.493 ms | 1.60× | 1.22× |
| 1024 | 2.110 ms | 2.828 ms | 3.442 ms | 1.63× | 1.22× |
| 4096 | 2.118 ms | 2.780 ms | 3.417 ms | 1.61× | 1.23× |
| 16384 | 2.167 ms | 2.669 ms | 3.597 ms | 1.66× | 1.35× |
| 65536 | 2.606 ms | 3.223 ms | 4.113 ms | 1.58× | 1.28× |

与第一版（单级调度）相比：1 段 0.10× → 5.02×、4 段 0.18× → 2.24×（算子级），
内核级 6.27× / 2.83×；16/64/256 段从 0.90–0.97× 抬到 1.16–1.27×；段数多的形状没有
退化。

**小形状的固定开销**：`65536 × 128，64` 的算子级在本机噪声最大（组合基线在
0.39–0.93 ms 间摆动），用配对法、三档对照（`now` / `batched` / `no check`）和隔离进程
三种方法交叉后为 1.18–1.83×，内核级稳定 5.5–6.4×。剩余差距来自 Python 层
（该形状 `no check` 0.130 ms vs `now` 0.625 ms）：`_check_inputs` 的三次 host↔device
往返在 A5 上是 0.75–1.29 ms，现已合并为**一次往返**（三个判断堆成一个向量一次拷回），
`op now / op batched` = 1.03× / 1.00× / 1.03×，拒绝行为不变；剩下的是返回一次拷贝与
Python 属性校验本身的开销。

**窄特征仍不占优**：固定 8388608 元素的特征宽扫描中，feat 16–64 为 0.30–0.71×——
每行只有 64–256 字节，向量指令利用率低，而组合实现按行并行能吃满 64 核。这是第 8 节
列出的取舍（窗口窄于整行时逐行搬运、按行归约），不是两级调度的问题。

## 8. 已知取舍与后续优化

- 切块模式比融合模式多 3 次 kernel 发射与两遍 workspace 读写；因此只在
  `baseTasks < 核数` 时启用，段数已经足够多时仍是单 kernel、零 workspace。
- 行切块要求每块至少 16 行（`SEGMENT_LOGSUMEXP_MIN_ROWS_PER_CHUNK`），极少行数
  的形状不切块，此时并行度仍受段数限制。
- 窗口窄于整行（`features > 1024`）时仍是逐行搬运，未享受行批量；
  跨架构统一源 stride 单位后可打开。
- 前向读 `src` 两遍（最大值、求和）。若要一遍完成，需要在线 log-sum-exp
  递推（`m_new = max(m, x)`、`s = s * exp(m - m_new) + exp(x - m_new)`），
  每个元素多两条指数指令，当前实现按「先正确、后优化」取舍。
- 窄行（`features` 16–64）在 64 核平台上仍不占优（A5 上 0.30–0.71×）：每行只有
  64–256 字节，向量指令大半是空的，而组合实现按行并行能吃满 64 核。要改善需要把
  多行交错成 1024 lanes 再归约，属下一阶段优化。
- 多段形状下本实现的耗时已接近单核搬运带宽决定的平台下限（A2 约 3.6 ms 处理
  1M × 128 两遍读取）；再要提升需要减少每行的小向量指令数（例如把 8 行交错成
  1024 lanes 再归约），属下一阶段优化。

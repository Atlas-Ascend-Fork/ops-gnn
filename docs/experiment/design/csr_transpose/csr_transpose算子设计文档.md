# csr_transpose 算子设计文档（A2/A3 arch22 三趟内核，950 原生组合）

> 状态：已实现。功能、精度与性能实测数据见对应 PR 的测试报告；本文件说明接口、
> 语义，以及两个平台各自的实现与分块方案。
> 依赖：无。算子只重写两个 int64 索引数组，不引入共享件，也不改动已合入算子。

## 1. 需求与验收口径

实现 CSR 结构转置（CSR → CSC，仍以 CSR 表达）：

```python
out_indptr, out_indices = csr_transpose(indptr, indices, num_cols=None)
```

输入是 (N, M) CSR 的两个索引数组，输出是 (M, N) CSR 的两个索引数组：每条边
`(row, column)` 变成 `(column, row)`，边数不变。验收口径：

| 项 | 要求 |
| --- | --- |
| 接口 | `csr_transpose(indptr, indices, num_cols=None) -> (out_indptr, out_indices)`，对齐 `torch_sparse.transpose(..., coalesced=True)` 的结构部分 |
| dtype | 输入必须是 1-D `torch.long`（int64），两输出同为 int64 |
| 输出 | `out_indptr` 长度 `M + 1`，`out_indices` 长度 `E`；与输入同 device |
| 顺序 | 每个输出分段内的行号升序（输入按行主序扫描天然产生），结果与 AIV 核数无关，可逐位比较 |
| 平台 | 两个平台按各自硬件特征选择实现：A2/A3（arch22 / `dav-2201`）走三趟 AscendC 内核；950（arch35 / `dav-3510`）走设备原生算子组合（§2.3），对外接口与语义完全一致 |
| 性能 | A2 对「直方图 + `torch.argsort`」等价组合有净收益（实测 4.36×–8.03×）；950 与同一组合同量级（0.85 ms），相对 CPU 组合快两个数量级 |

`num_cols` 省略时用 `indices.max() + 1` 推断，需要一次设备标量读取；显式传入可以让
调用全程异步。

## 2. Host 契约

### 2.1 参数与校验

`CheckInputs` 只做设备侧真正需要的检查：两个张量都是 1-D、dtype 为 int64、位于同一
NPU、`indptr.numel() >= 1`。指针取值（非 0 起点、递减、末位不等于 `E`）**不在 Host
校验**，与已合入的 `ptr2ind`（arch35）和 `ind2ptr` 保持一致：每次调用都同步一次设备
的代价高于收益，而且内核本身对任意指针都是内存安全的（见 §3.1）。

`num_cols == 0` 且 `E > 0` 是矛盾输入，直接拒绝（内核实现在 Host 用 `TORCH_CHECK`，
组合实现在 Python 层抛 `RuntimeError`，错误类型一致）；`E == 0` 时 `out_indptr` 置零后
立即返回，不做任何计算。

Python 层 `python/ops_gnn/csr_transpose.py` 负责参数类型/秩/设备/dtype 与 `num_cols`
的取值检查（`TypeError` / `ValueError` / `RuntimeError`），并保持与 `edge_softmax`
一致的错误语义。

### 2.2 核数与工作区

```
blocks      = min(ceil(E / 4096), 可用 AIV 数, max(1, M / 4))
workspace   = counts[M] + blockTotals[blocks * 4]
```

- 每核约 4096 条边起算，小图只用少量 AIV；
- 列数上限 `M / 4` 来自对齐约束：列区间边界按 32 字节（4 个 int64）对齐，核数超过
  `M / 4` 时区间会退化成空区间，而前缀和那一趟的最后一个区间必须非空才能写出
  `out_indptr[M]`；
- `blockTotals` 每核独占一个 32 字节块（`kTransposeBlockStride = 4`）：一次标量全局写会
  重写它所在的整个块，相邻核共块就会丢更新；而且标量全局写在 A2 上实测会整批丢值，所以
  区间总和也和其它数据一样，从 UB 的 32 字节对齐槽位出发、用 8 字节 `DataCopyPad` 发布。

### 2.3 平台分流：为什么 950 不用同一份内核

三趟内核的写回契约要求每个 AIV 独占输出侧一段完整的 32 字节块（§4），因此每个 AIV
必须自己扫遍全部边才能确定列内行序：工作量随 AIV 数线性放大。这个代价在 A2 上划得来，
因为 A2 的 `torch.argsort` 走 AiCPU（1M 边约 0.3 s），40 个 AIV 的扫描放大后仍快 4.5×。

950 上两个前提同时变了：AIV 数变成 64，放大因子随之变大；而 950 的原生
`torch.argsort(int64, stable)` 是 0.29 ms 量级的快算子，`scatter_add`（1M 边）0.36 ms。
同一张图上内核实测 37 ms 量级、原生组合 0.9 ms 量级——不是参数没调好，而是这套
「每核各扫全部边」的写回契约在 64 核并行下没有胜算。

因此 950 的实现直接就是这组原生算子，输出契约与内核逐位一致：

| 步骤 | 950 上的实现 |
| --- | --- |
| 行号展开 | `repeat_interleave(arange(N), indptr[1:] - indptr[:-1])`，得到每条边的行号 |
| 列直方图 + 指针 | `counts.scatter_add_(0, indices, ones)` 后 `cumsum`，前置 0 得到 `out_indptr` |
| 列内行序 | 对 `indices` 做**稳定** `argsort`，按该顺序取行号 |

稳定排序保持同一列内边的输入顺序，而输入按行主序扫描，所以每个输出分段内行号天然
升序、结果与核数无关、可逐位比较——与内核的不变量完全相同。

分派在 Python 层完成：`python/ops_gnn/csr_transpose.py` 读 `_pybind.npu_arch`，
`dav-2201` 调用设备内核，其余（950）走上述组合。950 构建不编译本算子内核，也不引用
内核符号：

- 删除 `op_kernel/arch35/` 后，`CMakeLists.txt` 的 `collect_npu_arch_sources` 在
  `arch35` 下不再收集到该算子，`op_host` 也随之不参与 950 构建（该函数只为存在对应
  架构内核目录的算子收集 Host 源文件）；
- `csrc/pybind.cpp` 中 `csr_transpose` 的绑定收在 `#ifdef OPSGNN_DAV_2201` 内。

这与已合入的 `ind2ptr` / `ptr2ind` 同一思路：索引类算子在原生 `scatter_add` + 稳定
排序足够快的平台上用组合实现，比自研内核更省。

## 3. 设备侧算法（arch22 / `dav-2201`）

三趟内核依次在同一 stream 上启动，每趟都是 `KERNEL_TYPE_AIV_ONLY`（只在 arch22 构建中
编译；950 走 §2.3 的组合）：

| 趟 | 读 | 写 | 说明 |
| --- | --- | --- | --- |
| count | `indices` | `counts`、`blockTotals` | 按列直方图，每核一段连续列区间 |
| indptr | `counts`、`blockTotals` | `out_indptr` | 区间独占前缀和 |
| write | `indices`、`indptr`、`out_indptr` | `out_indices` | 按列回填行号 |

### 3.1 按列直方图（count）

每核负责一段连续列区间 `[begin, end)`（按 4 列对齐），把区间内的列计数写到
`counts`，区间总和用 8 字节 `DataCopyPad` 发布到自己的 `blockTotals` 槽位。区间内的
列数超过 UB 能常驻的游标数时，按窗口分多次扫描。

### 3.2 前缀和（indptr）

每核先把自己之前各核的 `blockTotals` 求和作为起点，再对区间内的 `counts` 做独占
前缀和。这样没有任何核读别人正在写的指针值：区间之间的依赖只通过上一趟内核写好的
`blockTotals` 表达。

区间末尾正好是 `M` 的那一核额外写出 `out_indptr[M]`；由于列区间起点按 4 列对齐且
相邻区间至少相差 4 列（§2.2 的核数上限），最后一个区间一定非空。

### 3.3 按列回填（write）

这一趟是算子正确性的核心，分三步：

1. **列区间**：每核的区间端点取「输出位置落在 32 字节块起点」的列
   （`BlockAlignedColumn`）。指针数组非降序，所以定位下一个这样的列只需要跳到
   「指针值发生变化」的列，空列再长也不会退化成线性扫描。
2. **位置分块**：区间内的输出位置是连续的（`out_indptr[begin]` 到 `out_indptr[end]`），
   按 UB 能容纳的行数切成若干块，块起点是 4 的倍数，因此每次回写的 GM 起点都是
   32 字节对齐；块长度可以不对齐。
3. **UB 内计数排序**：块内用 `cursor[列]` 常驻 UB 记录该列下一个输出位置，扫描
   全部边，把命中的行号用 `SetValue` 写进 `cache[位置 - 块起点]`，扫描结束后整块
   `DataCopyPad` 回写。

每行的行号由一个随扫描推进的 `indptr` 游标给出：边按输入顺序访问，所以「游标所在
行」就是该边的行，不需要预先展开一个长度 `E` 的行号数组。跨行时才读一次 `indptr`，
每条边一次比较。

区间内位置装不下一个块时，对每个块重扫一遍边，因此**超长列（几十万条边）和普通列
走同一段代码**，没有特殊分支。

## 4. 对齐、流水与平台约束

三条硬件约束决定了上面的写法，每条都有对应的实测依据：

| 约束 | 处理 |
| --- | --- |
| 一次标量全局写会重写它所在的 32 字节块 | 每核的输出位置区间按块对齐，核与核之间不共块；写回全部改走 `DataCopyPad` |
| `DataCopyPad` 的 GM 起点必须 32 字节对齐（长度可以不对齐） | 列区间端点取「输出位置块起点」；位置块长度取 4 的倍数；`out_indptr` / `counts` 的写起点同样按 4 元素对齐 |
| UB 上的 MTE 源地址必须落在缓冲区起点（高偏移处做单元素/小块搬运会报 MTE 对齐异常） | 回写的 `cache` 放在 UB 块起始位置，游标与边 tile 排在它之后 |

同步一律用 `SetFlag`/`WaitFlag` 显式事件：标量 pipe 与 MTE2/MTE3 之间的可见性只靠
`PipeBarrier` 不可靠，这是 A2 实测踩过的坑。

UB 预算（A2：196608 B）：

```
reserved            = 32768
边 tile              = 4096 * 8 = 32768
cache + cursor      = 131072
cache（行）          = 12288 个 int64
cursor（列）         = 4096 个 int64
```

## 5. 复杂度与已知代价

设边数 `E`、列数 `M`、核数 `C`、行缓存 `capPos`：

- 每核每趟都扫描**全部**边，因此 count 趟与 write 趟的墙钟时间随 `E` 线性增长；
- write 趟还要按位置块重扫，块数约为 `ceil(E / (C * capPos))`，总工作量约为
  `E * ceil(E / capPos)`，与核数近似无关（核数只决定并行度）；
- 实测（A2，40 个 AIV，`E = 1M`、`M = 16384`）：`82.0 ms`；等价组合（NPU 直方图 +
  `int64 argsort`）`366.5 ms`（4.47×）；CPU 组合 `103.3 ms`。列数拉到 200k、边数不变时
  算子 `89.6 ms`，等价组合 `390.4 ms`（4.36×）；`100k × 1000`、300k 边时算子 `13.3 ms`，
  等价组合 `107.0 ms`（8.03×）。
- 950 不走这条路径（§2.3）：同一张 `16384²` / 1M 边的图上组合实现 `0.85 ms`，同平台
  CPU 组合 `104.3 ms`。也就是说阴影里的放大代价只在 A2 这种「原生排序走 AiCPU」的
  平台上被内核的正收益盖过，950 上由原生算子组合直接承担。

这是「不用 `cores × columns` 工作区」换来的代价：另一种做法是每核在 GM 里维护一份
按列直方图，把放大因子降到 1，但需要 `C * M` 的额外工作区。指数型算子的索引通常
不大，arch22 先选前者，并在文档里标明后续可用 per-core 直方图优化；950 由 §2.3 的
组合实现，不受这条放大代价影响。

## 6. 语义边界

| 情形 | 行为 |
| --- | --- |
| `E == 0` | 返回长度 `M + 1` 的全 0 指针与空索引；省略 `num_cols` 时 `M = 0` |
| `E > 0` 但 `indptr` 只有 1 个元素 | 边的行号无从归属，Host 抛 `RuntimeError` |
| 空列 | `out_indptr` 相邻两项相等，不产生输出元素 |
| 重复边 `(r, c)` 出现两次 | 输出里 `c` 列出现两次 `r`，不合并（与 `coalesced=False` 的结构一致） |
| 行内列序打乱 | 输出按列分组、段内按行升序，与输入行内顺序无关 |
| `num_cols` 大于实际最大列号 | 尾部多出全等指针，是合法的空行 |
| `num_cols` 小于实际最大列号 | 越界列不会被回填，属于调用方违约；Host 不校验，与 §2.1 的取舍一致 |
| 指针不单调 | 结果未定义：arch22 内核内存安全但输出无意义；950 组合可能出现负的重复次数并直接报错（同 §2.1 的取舍） |

## 7. 测试与性能

### 7.1 用例组织

沿用 `test/<算子>/arch22|arch35` 结构，每个目录只保留 `golden.py`、
`test_csr_transpose.py`、`benchmark_csr_transpose.py`：

- `golden.py`：CPU 上用稳定 `argsort` + 列直方图重建两个索引数组，不导入 `ops_gnn`；
- `test_csr_transpose.py`：形状矩阵（方阵/矩形/超宽/单行/单列/对角/空图/全空列/
  空行空列混合/长空行/热列）、随机图（含 45000 列与 999×4096）、推断 `num_cols`、
  输出元信息、分段内升序不变量、转置两次回到行主序、确定性、非连续输入、
  与「`argsort` 组合」的一致性、非法参数（11 例）；
- `benchmark_csr_transpose.py`：同一张图上对比算子、NPU 上的 `argsort` 组合、
  可选 CPU 组合与 `torch_sparse.transpose`（未安装时跳过）。

`test/common` 不新增文件；两个架构目录的用例与参考实现相同，平台差异只体现在实测
数据上。

### 7.2 性能口径

- 主场景 `16384 × 16384`、`degree = 64`（约 1M 边）；另测 `200k × 200k`（1M 边、
  平均度 1）与 `100k × 1000`（300k 边）观察列数与度数倾斜的影响；
- 对比基线是等价组合（直方图 + 稳定 `argsort`），同一 NPU、同一张图；
- 计时用 `torch.npu.Event`，预热 10 次、计时 20–50 次取均值；
- 报告按 A2 / A5 两个平台分别标注，不混合统计。

## 8. 实施步骤与风险

### 8.1 步骤

1. 新增 `csrc/npu/csr_transpose/{op_host,op_kernel/{common,arch22}}`，三个内核；
2. `csrc/pybind.cpp` 在 `OPSGNN_DAV_2201` 段内注册 `csr_transpose`（950 不绑定内核）；
3. Python 层 `python/ops_gnn/csr_transpose.py` + `__init__.py` 导出，按 `npu_arch` 分派
   内核（`dav-2201`）或原生组合（950）；
4. 测试三件套与文档（`README`、`docs/zh|en/api_reference.md`、本设计文档）；
5. 先 A2（arch22）构建/功能/性能，再按用户通知在 A5（950）复测，按平台分别出数据；
6. 提 PR 关联 issue，等 CI 与检视。

### 8.2 风险与对策

| 风险 | 对策 |
| --- | --- |
| 标量全局写在大批量下丢值（A2 实测） | 写回全部走 `DataCopyPad`；区间总和也改走 8 字节 `DataCopyPad`，每核独占 32 字节块 |
| 两核写同一个 32 字节块 | 列区间端点取输出位置块起点，核与核之间不共块；单测覆盖 `M` 在 `4C` 附近的情形 |
| 超长列撑爆 UB | 位置按块切分、逐块重扫，与普通列同一段代码；用例覆盖 200000 条边的单列 |
| 列数远大于边数（大量空列） | 列区间按 4 列对齐后仍可能很长，`BlockAlignedColumn` 用二分跳到下一个指针变化的列，空列不产生标量读 |
| `E` 很大时的重扫放大 | 文档标明 `O(E² / capPos)` 的工作量与后续 per-core 直方图方案；当前规模实测仍快于等价组合 |
| 两平台实现不同导致语义漂移 | 两个架构目录跑完全相同的 40 个用例与同一份 `golden.py`（不导入 `ops_gnn`），覆盖段内升序、转置两次、确定性与非法参数；`benchmark` 在两平台上都对照同一等价组合 |
| 两平台数据混合 | 报告按平台分段呈现，benchmark 脚本不带平台默认值，由运行环境决定 |

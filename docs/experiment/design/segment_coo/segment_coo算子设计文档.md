# Segment COO 算子设计：Ascend 950 与 A2

> 下文原有章节保留 Ascend 950 产品接入的设计与历史验证口径；
> 新增 A2 后端的实现、当前精度标准及本次验证范围见文末独立章节。
> 950 历史结论不能视为本次 A2 或 950 回归结果。

本文件说明产品代码的接口、数据流与边界。社区任务的设计评审记录见
[设计 PR #1485](https://gitcode.com/cann/cann-ops-competitions/pull/1485)。
本次产品接入基于验收仓提交 `0cdb291cea227d6c5be03ed8b3cf85c4e9d4e9f9`；
本版本的测试结论以产品 PR 所附的同版日志为准。

## 接口与输入合同

提供 `segment_coo` 与 `segment_sum_coo`、`segment_add_coo`、
`segment_mean_coo`、`segment_min_coo`、`segment_max_coo`。
`index` 是沿最后一维排序、非负且不越界的 int64 segment ID。
归约维为 `index.dim() - 1`，src rank 为 1–8；index 前部维度可为 1
或对应 src 维度，尾部 feature 维共享 index。

支持 float16、float32、bfloat16、int8、uint8、int32、int64。
可选 `dim_size` 指定输出段数，提供 `out` 时以 out 的对应维度确定段数。
专用 min/max 返回 `(values, indices)`，通用接口只返回 values。
不支持的 reduce（包括 mul）由 Python 接口拒绝。

sum/add 覆盖给定 out；mean 使用 out 初值参与归约后除以段内计数；
min/max 与 out 初值比较。无 out 的空段值为零，有 out 时 mean/min/max
保留空段初值；极值平局选择较后的源坐标。上述 task-specific 行为与
CPU 参考库的非零 out 累加和首次极值规则不完全相同，测试分别检查。

## Host 数据流

1. 校验设备、dtype、rank、广播及 out 元数据，选择输入 dtype 分发编号。
2. 广播并连续化 index，连续化 src。连续视图可能带非零 storage offset，
   因而继续检查实际 src/out 地址的 32 字节对齐；不满足时克隆到对齐暂存。
3. 由 src/index 形状计算 batch、归约长度、尾部 channels、segments 和
   index batch 数，写入 `SegmentCooPlan`。
4. 按路径需要分配段边界指针暂存，复用 PyTorch 当前 NPU stream。
5. 用 `OpCommand::RunOpApi` 排入框架任务队列。lambda 持有 src/index/out/
   arg/ptr Tensor，确保连续化和对齐暂存活到启动处理完成。
6. 调用 Kernel 入口 `SegmentCoo`。若提供的 out 使用了暂存，则将结果
   写回原视图，并返回原 out，保留调用者的存储身份。

## Kernel 组织

`op_kernel/arch35/segment_coo.h` 定义 plan 与入口；
`segment_coo.cpp` 包含实际 index 边界查找、SIMT 归约和启动逻辑。
Host 对应文件位于 `op_host/segment_coo.h/.cpp`。Kernel 入口遵循同仓
`SegmentMaxCsr`、`GatherCoo` 等算子名命名方式。

每次调用都读取真实 index 并查找边界；没有依赖均匀分段输入，也不缓存
之前调用的 index 或输出。实现按 dtype、channels 和坐标范围选择标量或
packed 访存，以及长段 sum/mean 的协作归约。INT64 mean 的特定对齐路径
可直接读取 index，减少中间指针生成。2 的幂坐标分解仅替代可证明等价的
除余运算，其余形状保留精确的通用计算。

FP16/BF16 使用较宽浮点累加；整数归约保持整数路径。INT8/UINT8 mean
保留参考实现的窄化行为；当非空段计数窄化为零时，使用明确的数学均值，
不调用可能发生除零的 CPU 原生参考，也不将该边界标为参考库一致。

## 测试与性能口径

`test/segment_coo/arch35/` 只保留三个 Python 文件：

- `test_segment_coo.py`：接口、数值、边界、异步 stream 与地址对齐用例，
  以参数化区分 shape/dtype/reduce；原用例的断言与参数矩阵保留。两处
  非连续输入先传输连续底层数据，再在 NPU 上创建视图并断言非连续性，
  避免 CPU 转置视图直接转换引发内部格式告警；该告警按测试失败处理。
- `golden.py`：真实 CPU `torch_scatter` 参考入口与可比较语义范围。
- `benchmark_segment_coo.py`：保持原题形状、输入与计时逻辑的日志版本。
  正式验证另用附件中保留原字节的 `benchmark_supplied.py`。

完整性能分母为四种 reduce × 29 组 shape × 四种 dtype，即 464 项。
warmup=20、iteration=100，逐项使用任务书 GPU 基准/NPU 延迟 ≥ 0.45。
原脚本输出到 0.001 ms，外部报告检查使用耗时上端点 `printed_ms + 0.0005`
计算保守比值。原始日志、任务书基线与检查脚本随验证附件交付，不把
诊断脚本、自测报告或历史二进制放入产品代码树。

CPU 参考库的支持范围和 task-specific 语义由各自用例说明；未把二者
混为完全等价。Host 校验元数据，不遍历 NPU index 内容，调用者需满足
index 排序和范围前提。Ascend 950 为本版本目标，其他架构不导出本算子。

## AI 参与说明

原实现及证据整理使用过 ZCode/GLM-5.3 与 Codex。本次产品接入、测试合并
及地址对齐保护使用 GPT-6（Codex）辅助，提交人负责检查代码和真机结果。

## Ascend A2 后端设计

### 接口与输入约定

A2 后端通过 `arch22` 实现 Segment COO，提供 `segment_coo` 以及
`segment_sum_coo`、`segment_add_coo`、`segment_mean_coo`、`segment_min_coo`、
`segment_max_coo`。通用接口返回归约值，专用 min/max 返回 `(values, arg)`。

`src` 支持 1～8 维，数据类型为 float16、float32、bfloat16、int8、uint8、
int32、int64。`index` 为非负、沿最后一维非降序的 int64 张量，允许标签空洞。
index 满足 `1 <= index.dim() <= src.dim()`，归约维为 `index.dim() - 1`；
index 的各维可等于对应 src 维度或为 1，
包含归约维上的广播。尾部 feature 维共享同一组索引。

内部将输入规范化为 `[B, N, C]`：B 为归约维之前各维的乘积，N 为归约长度，
C 为其后各维的乘积；输出对应 `[B, S, C]`，返回时保留原始多维形状。
提供 `out` 时由其对应维度确定 S；否则使用 `dim_size`，未指定时由最大标签加一
推断，空 index 推断为 0。支持非连续 src、index 和 out；不支持的 reduce 由接口拒绝。

| 归约方式 | 初值与结果规则 |
| --- | --- |
| sum/add | 两者等价，在提供的 out 初值上累加 |
| mean | 先累加输入和初值，再按该段输入个数计算均值；整数除法向零截断 |
| min/max | 与初值比较，仅在严格改善时更新值和位置；普通平局保留首次极值位置 |
| 空段、空输入 | 无 out 时输出零；有 out 时保留初值；空段 arg 为 N |

每个 batch 的首个非空段使用 `out[0]` 作为初值，即使其标签不为 0；后续非空段
使用对应段的初值。极值位置的状态按 feature 保存，初始为 0，跨非空段和 batch
延续；空段写入 N 哨兵且不改变该状态。

### Host 数据流

1. 校验设备、dtype、rank、reduce 和输出元数据，确定归约维及输入广播关系。
2. 展开并连续化 index，再校验排序、非负和范围约束；根据展开后的逻辑形状判断
   是否满足等长分段条件，使索引步长、段长和调度依据保持一致。
3. 连续化 src/out，检查实际地址的 32 字节对齐。非连续或未对齐的输出使用暂存，
   src/index 与 out 存在存储别名时保留输入副本。
4. 根据 B、N、C、S、dtype、reduce、是否提供初值、是否需要 arg，以及设备核数和
   UB 容量生成 `SegmentCooPlan`。需要读取原始极值初值的路径在 NPU 上保存 out 快照。
5. 通过 `OpCommand::RunOpApi` 提交到当前 NPU stream；启动闭包持有输入、输出和
   暂存张量，并记录跨 stream 使用的存储，保证异步执行期间的数据有效。
6. Kernel 完成后，将暂存结果写回调用方原 out 并返回原对象；视图外存储保持不变，
   原地写入同步更新张量版本。

Host 缓存已验证的输入属性和分块计划。Tensor 身份、版本、形状或步长变化后重新
校验；无法可靠跟踪修改的输入重新检查。缓存内容不包含归约结果。

### Kernel 组织与任务划分

| 文件 | 职责 |
| --- | --- |
| `op_host/segment_coo.cpp` | 输入规范化、属性验证、路径选择、tiling 和输出管理 |
| `op_kernel/arch22/segment_coo_tiling.h` | Host 与设备共享的形状、步长、分块参数和执行标志 |
| `op_kernel/arch22/segment_coo_reduce.cpp` | 浮点及 INT32 通用向量归约、补偿归约和批量处理 |
| `op_kernel/arch22/segment_coo.cpp` | byte/INT64 专用归约、极值状态扫描及统一启动入口 |

只返回 value 的归约以输出段和特征块分配任务。通用路径在有序 index 中查找每段
的 `[begin, end)`；已验证为等长的连续分组直接计算边界。段内输入分批搬入 UB，
完成局部归约及跨块合并后写回。每个对齐输出范围由唯一核负责，避免写回竞争。

`tileK` 控制特征块宽度，`tileRows` 控制一次搬运的行数，`outputGroup` 控制共同
处理的输出段数。Host 将输入、类型转换、累加、补偿、索引和对齐缓冲计入 UB 预算，
根据实际容量选择分块。尾块只搬运和写回有效元素；相邻非对齐输出按同一核管理。

等长短段在满足布局和容量条件时使用批量路径，一次搬运多个段，减少重复传输和
向量指令提交。较长的段按行分块，或划分特征列以容纳完整段。条件不满足时使用
通用边界查找和归约路径。

### 归约与数值处理

| 类型/场景 | 计算方式 |
| --- | --- |
| FLOAT16/BFLOAT16 | 使用 FP32 累加；mean 在除法前将分子和段内计数收窄到源 dtype，再按源类型返回结果 |
| FLOAT32 | 根据输入数值范围和累加误差评估选择按源行序累加或补偿归约，保留抵消敏感输入的精度 |
| INT8/UINT8 | 使用整数累加，按源位宽转换分子和计数；mean 的非空计数收窄为零时显式输出零 |
| INT32 | 整数求和、截断除法及极值比较；只返回 value 的 min/max 使用整数树归约 |
| INT64 | 求和采用精确数位拆分和进位，极值采用高低字比较；数值范围证明成立时选择精确的 INT32 窄化路径 |

浮点补偿路径在树节点和块间合并时维护高低部分，减少局部求和误差。分块选择和
运算路径由输入属性决定，段划分及输出归属保持确定性。

极值调用需要初值或 arg 时，以 feature 块分配任务，按源顺序遍历段和 batch。
每段重新建立值累加器，仅在输入严格改善当前值时更新 arg；位置状态保留到下一
非空段。初值快照、值比较和 arg 生成均在 NPU 上完成，位置使用 int64 保存。

无 out 的 min/max 以源 dtype 的有限极值初始化，最终仍等于初始化值的结果置零。
源 NaN 不更新极值；全 NaN 的非空段输出零并沿用已有 arg 状态。有 out 时保留
相应初值规则，包括 NaN 初值。空段独立处理，避免参与 mean 除法或改变极值状态。

### 测试与性能口径

`test/segment_coo/arch22/` 保持三个脚本的分工：

- `golden.py`：本工程独立 PyTorch CPU 参考与精度比较器，可以单文件运行。
- `test_segment_coo.py`：按 TC01–TC18 覆盖接口、dtype、广播、输入布局、out、
  空段、特殊值及 arg 状态，并验证非默认 stream 的依赖关系。
- `benchmark_segment_coo.py`：生成既有形状矩阵，测量完整 PyTorch 公开接口时延。

浮点结果按[官方混合容差标准固定版本](https://gitcode.com/cann/opbase/blob/d9a3cc9f52c641c25f8b234d37d2c099e6454517/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
检查元素容差、99% 匹配率和最大误差 ULP 上限；整数及 arg 精确比较。独立对拍
使用真实 CPU 扩展核对参考实现，覆盖初值、类型转换、极值状态和广播等边界。

性能矩阵为 29 组形状 × 4 种 dtype（FLOAT32、FLOAT16、INT32、INT64）×
sum/mean/min/max，共 464 项；全部使用 out=None，min/max 只返回 value。每项预热 20 次、计时 100 次，计时前后同步，
输入生成放在计时外；按任务书外部基线与 NPU 时延之比核验 0.45 门槛。

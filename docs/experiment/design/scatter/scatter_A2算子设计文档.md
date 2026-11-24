<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Scatter Ascend A2 前向设计

## 1. 范围与参考

实现 `scatter`、`scatter_sum`、`scatter_add`、`scatter_mul`、`scatter_mean`、
`scatter_min`、`scatter_max`，目标为 Ascend A2（910B，`dav-2201`）。
公共签名见[中文 API](../../../zh/api_reference.md#scatter--scatter-系列归约)。
本设计描述 A2 实现和验证口径，不改变已有 arch35 实现的分派，也不增加反向接口。
实现与验证分开记录；编译成功或本文描述不能替代设备实测报告。

语义参考固定为原版 `torch_scatter==2.1.2` 的 CPU 公开 API 与 PyTorch 2.7.1。
源码审查使用 pytorch_scatter 的 `140d3ad677aae615767412873b90982cbf97d35d`；
已安装扩展的来源另以导入路径、共享库 SHA256 和环境版本记录，不能由包版本推断二进制提交。

| 级别 | src / out dtype | 执行路径 | 归约范围 |
| --- | --- | --- | --- |
| L1 | FLOAT16、BFLOAT16、FLOAT32、INT8、INT16、INT32、UINT8 | A2 Ascend C；mean 使用符合条件的融合路径或原生 sum/count 组成 | sum/add/mul/mean/min/max |
| L2 | FLOAT64、INT64 | 原版 torch_scatter CPU 扩展，D2H / H2D | sum/add/mul/mean/min/max |

`index` 和专用 min/max 的 `arg_out` 固定为 INT64。通用 `scatter` 的 min/max 返回 value，
专用 min/max 返回 `(value, arg)`；当前原生入口共用 value/arg 计算，通用入口丢弃 arg。
分组原生入口的 sum/add/mul/mean 不分配空 arg Tensor，返回 undefined Tensor，
经 PyTorch 绑定转为 Python None；专用 min/max 仍返回设备上的 INT64 arg。
L1 不导入 torch_scatter；L2 接受 PyTorch 2.7.1 和 2.12.0，显式依赖
针对当前 PyTorch 编译的原版 `torch_scatter==2.1.2` CPU 扩展。
缺失依赖抛出说明安装要求的 ImportError，版本不匹配抛出 RuntimeError。
原数值语义参考仍是 PyTorch 2.7.1；2.12.0 已通过独立 CPU 数值及 L2 等价核验，
前向 autograd 的已知版本差异见第 2 节。此版本列表不包含其他 2.12.x 或其他版本。
L2 首次遇到每种 dtype 时提示 CPU 回退，不进行 FLOAT64→FLOAT32 或 INT64→INT32 降精度归约。

## 2. 公开语义

支持 rank 1～8、负 dim、无序及重复 index、广播零 stride、非连续 src/index。
一维 index 对齐归约轴，其他 index 在尾部补维后广播至 src。
有 out 时以 out 的归约轴长度为输出长度；无 out 且 dim_size=None 时由有效 index 推导。
空广播输入推导长度为零。提供的 out 须与 src 的 dtype、device 及非归约维相符。

| 情形 | A2 处理规则 |
| --- | --- |
| out | 参与归约的初值；成功时原地修改；返回对象与 grad 状态见下文 |
| sum / add | 加到已有 out；FP16/BF16 在 FLOAT32 中累加后收窄 |
| mul | 与已有 out 连乘；低精度浮点每步按源 dtype 舍入 |
| mean | 先执行 CPU 公开接口相同的 sum 组成，再除计数；整数用 floor 除法 |
| min / max | 仅严格改善时替换 value 和 arg；普通平局保留沿 dim 的首次有效位置 |
| 极值 arg 哨兵 | 初始化为 src.size(dim)；已有 out 未被严格改善时保留哨兵 |
| 极值无 out | 最终仍等于 CPU 有限初始化极值的 value 清零，arg 不因此改写 |
| 非空输入的空桶 | sum/mean/min/max 为零，mul 为一；有 out 时遵循初值和 mean 除数规则 |
| 整体空 src，无 out | 包括 mul 在内返回零初始化输出 |
| 非连续 out | sum/add/mean 允许设备工作缓冲和原位写回；mul/min/max 抛 RuntimeError |
| 重叠输入 | 沿用 CPU 原语可检测的重叠拒绝规则；CPU 允许的独立 storage 包装或步长别名保留其可观察读写行为 |

mean 的计数来自**广播前的 index**，ones 与 count 使用 src dtype。
count 的溢出或低精度舍入发生在 `count < 1` 置一之前，不能改成数学上的宽整数平均值。
例如 128 个 INT8 的一归入一桶得到 -128；300 个 UINT8 的一归入一桶得到 1；
`src=[[2,4,6]], index=[0], dim=1` 得到 `[[12]]`。
NaN、Inf、全 NaN、正负零、初始化极值、负整数均值、整数回绕分别作为语义边界验证。

### 前向 autograd 状态与异常

保留前向调用可观察的异常和写入时序，不实现反向计算。
`python/ops_gnn/scatter/_scatter_autograd.py` 对 sum/add/mean 使用 A2 原生 `scatter_check_out` 调用
PyTorch `check_inplace`；该前置检查不写入数据、不增加 Tensor 版本计数。
后续通过 `torch.autograd.Function.mark_dirty` 交由 PyTorch 执行写后状态检查，
不在 Python 中读取 Tensor 私有 base/version 字段。mul 的 Function 返回一个 Tensor，
min/max 返回两个，保留运行时的叶子、view 和 inference 状态处理。

数值参考仍固定为 PyTorch 2.7.1 的原版 CPU 扩展；前向 autograd 异常跟随运行时。
[PyTorch 2.8 发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.8.0)
记录了 custom Function 新增叶子 view 原位检查，因此下表区分该版本边界。

| out 状态与调用上下文 | sum/add/mean | mul | min/max |
| --- | --- | --- | --- |
| requires_grad 叶子，grad 开启 | 写入前报 RuntimeError | 写入后报 RuntimeError | 写入后报 RuntimeError |
| requires_grad 叶子的 view，grad 开启 | 写入前报 RuntimeError | PyTorch 2.7.1 成功；2.8 起写入后报 RuntimeError | 写入后报 RuntimeError |
| requires_grad 非叶子，grad 开启 | 成功 | 成功 | 成功 |
| requires_grad 非叶子的 view，grad 开启 | 成功 | 成功 | 写入后因 view 多输出报 RuntimeError |
| 普通 inference Tensor，在 inference_mode 外 | sum 更新后报错；mean 未进入除法 | 写入后报错 | 写入后报错 |

若 inference out 同时为 requires_grad 叶子且 grad 开启，sum/add/mean 的异常时序
还取决于输入的 Autograd 分派：src 或 index 为普通 Tensor 时，前置叶子检查在写入前报错；
src/index/out 全部为 inference Tensor 时，遵循 CPU 绕过该前置检查的行为，
在 sum 写入后报 inference 修改错误，mean 不进入除法。

普通 out 成功时返回原对象。对于 requires_grad out，在 no_grad/inference_mode 下，
sum/add/mean 仍返回原对象并保留 requires_grad；mul/min/max 返回共享存储的另一个 Tensor wrapper，
其 requires_grad 为 False。不能把所有成功路径都强制替换回原 out 对象。
L2 回写使用 inference_mode 允许复制发生，再由外层桥接触发相应写后异常；
inference out 在 context 外的 L2 mean 只执行 CPU sum 阶段后报错。
调用 backward 时明确抛 NotImplementedError。

## 3. Python、Host 与构建边界

`python/ops_gnn/scatter/` 集中保存公开入口与私有适配模块。
`__init__.py` 保留公共签名，并检查原生模块是否真正导出 `scatter_forward`；
`dav-2201` 分派到同目录的 `_scatter_a2.py`，其他架构保留原分派。
`_scatter_a2.py` 与原生入口共同校验元信息和 index 范围，并执行 mean 与 L2 组成；
`_scatter_autograd.py` 处理前向状态与异常。包内通过相对导入访问根包的 `ops_gnn._pybind`，
七个公开接口仍从 `ops_gnn` 导出。一维 index 保持紧凑存储，其他 index 广播后连续化。

七个公开 Python API 都不包含 `hot_target`。该参数只属于共用原生绑定
`scatter_forward`，为 arch35 提供性能提示；A2 为兼容绑定签名保留参数，
仍检查其值为 -1 或非负整数，但不参与 A2 的路径选择或 tiling。
专用 A2 原生入口 `scatter_a2_forward` 不包含该参数。

A2 新增 C++ 实现按职责组织为以下八个文件，均位于 `csrc/npu/scatter/`：

| 文件 | 职责 |
| --- | --- |
| `op_host/scatter_a2.cpp` | 主入口、元信息检查、路径规划及分组前向调度 |
| `op_host/scatter_groups.cpp` | 索引分组缓存、校验资源与异步生命周期管理 |
| `op_host/scatter_groups.h` | 分组 Host 内部类型与接口声明 |
| `op_kernel/arch22/scatter.cpp` | 各 kernel 入口、launch、数据类型分派及索引精确相等校验（`IndexEqualShared`） |
| `op_kernel/arch22/scatter.h` | 两类 tiling 结构与 launch 声明 |
| `op_kernel/arch22/scatter_common.h` | 数值工具、标量与计数路径、计数初始化、通用分块搬运及 `SourceTile` 描述 |
| `op_kernel/arch22/scatter_vector.h` | 向量归约与原子路径 |
| `op_kernel/arch22/scatter_grouped.h` | 稳定分组归约（`GroupedScatter`） |

部分 L1 非 mean 调用直接进入原生校验：src/index 必须为普通 Tensor，out=None，
src 不需要梯度；dim 为 [-8, 8) 内的内置 int，reduce 为合法的内置 str，
dim_size 为 None 或 INT64 范围内的非负内置 int。bool、Tensor 子类以及其他参数形式
保留原 Python 校验顺序。原生入口仍检查 device、dtype、rank、dim、shape、溢出和索引范围。
若原生调用抛 RuntimeError，适配层重新执行原 Python `_validate` 元信息检查：
该检查发现错误时恢复公开异常类型和优先级，检查通过则原样重新抛出该原生异常。
成功调用不重复执行这次 Python 检查。已有 out、前向 autograd 检查、mean 和 L2
仍走各自原有组成；这里的入口优化不增加 backward，也不改变归约路径的选择条件。

紧凑一维 index、非空 src 且 out 不与 src/index 共享字节时，原生入口优先使用稳定分组。
`scatter_groups.cpp` 只处理索引元数据：在设备上创建私有 INT64 快照，将该快照复制到 CPU，
检查范围并按原始行序生成 `rowptr`/rowlist，再上传到 NPU。src 数值、归约中间值和 L1 输出
始终由 Ascend C 在设备上计算；CPU 不执行 L1 数值归约。首次构建、上传和同步都在公开调用内。

缓存默认启用；仅设置 `OPSGNN_SCATTER_GROUP_CACHE=0` 关闭复用，逐次记录冷构建成本。
缓存最多保存 8 项、共 64 MiB 的 snapshot/`rowptr`/rowlist。查询键包含数据地址、原 index
的形状/步长/offset、归约长度、输出长度提示、device 和当前 stream；地址仅用于查找。
每次命中仍对当前 index 与私有快照执行全量 INT64 精确相等比较，不依赖 hash、版本号、
Python id 或旧范围结论。每个 INT64 重解释为两个 INT32 分量，使用原生 Compare EQ
检查全部 64 位，不做浮点数值转换。比较 kernel 的 UB 为两个 4096 字节原始输入块、
128 字节 mask 和 32 字节 flag，共 8352 字节。
每核输出独占的 32 字节 flag 块。设备支持时，EQ 除写入原 GPU flags，
还将相同块写入已注册的 Host 映射区；否则保留小结果 D2H。必要传输和同步都计入接口时间。
Host 不轮询映射内存，必须先取得本轮 event 的完成证明。

普通分组 owner 在任何输出读写之前，把全部 GPU flags 一次 DMA 到 UB，等待搬运完成
再逐核检查；任一不相等即零写入返回。Host 排入 owner 后等待校验，失败时在同一次
公开调用内重建并重跑。非连续 out 仅在校验成功后写回。冷构建不分配校验缓冲。
符合第 4 节条件的无 out FLOAT32 Sum/add 热命中则合并 EQ 与有序归约：每核先比较
自己负责的索引区间，再按不可变旧计划计算私有输出，不在核间读取尚未就绪的 flags。
Host 等待整个合并 kernel 及必要 D2H 完成，再检查所有 flags；仅全部相等才发布输出。
不匹配时丢弃私有结果并重新构建，包括 dim_size=None 时推导输出维度增大或缩小。
冷构建、重建及用户提供 out 的调用仍使用普通路径。

普通原位写、独立 storage alias、inference tensor、地址复用和其他算子的原生 out 写入
均须通过本次数值比较，不能只靠版本计数。调用方仍须对跨 stream 的写入建立正确的
事件依赖；数据竞争不属于缓存保证。快照与分组元数据不可变，发布前等待上传完成。

GPU flags、Host pinned 缓冲、可选映射和可重用 event 组成独占租赁资源，每 device
最多保留 8 项空闲资源。复用必须同时匹配设备、实际 ACL stream、当前分配器身份及
足够容量；较大资源仅使用本次 blocks 对应的有效前缀。不同 stream 不共享租赁资源，
活动 ticket 之间也不共享 flags。映射分配保留原始 base，额外申请对齐余量，选择完整
位于分配内的 4 KiB 对齐区域；注册长度向上取整到整页，覆盖实际核数乘 32 字节。
只有已知不支持，或可恢复设置失败且清理已证明安全、Runtime 状态清楚时，才回退 D2H。
即使注册返回失败，也须注销成功才能认定该区域没有遗留映射；未知 Runtime 错误不被清除，
也不当作普通索引不匹配。

event 通过 aclrtCreateEventExWithFlag 以 ACL_EVENT_SYNC 创建。映射分支的 Finish
始终成功执行 aclrtSynchronizeEvent 后才读取 Host flags；D2H 分支先查询一次，
仅在确认 COMPLETE 时省去等待，否则成功同步 event 后再扫描。
普通路径的 event 覆盖 EQ 与必要 D2H，排在 owner 之前；合并路径的 event 则覆盖整个
私有输出生产者与必要 D2H。映射分支不借用 D2H 的 Query 完成快捷路径。

普通路径在 owner 已成功提交到匹配的实际 stream、校验完成且 Host flags 扫描结束后，
ticket 的最后一个持有者才归还资源。下一次 EQ 在同一 stream 排在旧 owner 之后，
因此旧 flags 读取完成后才被覆盖；无需等待旧 owner 的整个归约完成。普通 owner 的
source、out、已定义 arg、`rowptr`、rows 和 flags 均保留 recordStream，池满释放或异常
丢弃资源时仍由分配器保护实际消费者。ticket 另强持 index、snapshot 和 flags Storage
至校验完成；可变 index 仍记录 stream，防止异常后调用方 resize Storage 替换分配
而提前释放原地址。不可变 snapshot 不传给普通 owner，`rowptr`/rows 在实际发射时记录。

私有合并路径在任何可能的提交前，强持 index、snapshot、flags、source、output、
空 arg、`rowptr`、rows 共八槽 Storage，其中 Sum 实际七项有分配，并先设置 submitted
及 hostPending。index、source、`rowptr`、rows 和 flags 仍记录 stream；私有 output
与 snapshot 在完整生产者完成前不能被调用方重绑，故由强持有保护。仅这种完成后才
公开的私有 output 可省去生产者记录。正常完成后归池会清除 owners，不缓存输出值。

没有成功提交 owner 标志的租赁不回池，证明本轮操作完成后直接清理。异常析构仅在
证明 event 或 stream 完成后才可归还有提交标志的资源；没有成功记录本轮 event 时，
须确认实际 stream 不含 PERSISTENT、DEVICE_USE_ONLY 或 CPU_SCHEDULE 标志且
stream 同步成功，不能把旧 event 状态当作本轮完成。清理在原设备上依次执行：确认
没有待完成访问、成功注销映射、销毁 event、释放原 pinned base，最后释放 Tensor/Storage。
无法证明完成、注销失败或其他无法安全清理的状态保留整个资源包并禁止该设备后续
热校验租赁；不会单独释放设备可能仍在访问的 Host 地址或任何保留的 owners。

未进入分组的路径使用 `scatter_index_stats` 在 NPU 扫描物理 INT64 index，
每核按 512 元素块计算 min/max，仅取回每核 32 字节摘要。该统计路径不缓存数值结论。
线程局部资源保留 pinned Host 缓冲、可重用 SYNC event 和本次 index/stats Storage，
每次在摘要拷贝后记录 event 并等待完成才读取。跨设备切换、容量增长和线程退出均在
原设备上清理；部分提交失败使用上述完成证明规则，无法证明时隔离整包并拒绝该设备
后续统计调用。index/stats 的实际消费者 stream 记录保留。
L2 另保留真实 CPU 归约所需的输入复制。广播后为空时可跳过归约值域检查；
mean 对非空原始 index 的计数仍按 CPU 顺序校验，必要时在 sum 后抛错。

`op_host/scatter_a2.cpp` 在 `OPSGNN_DAV_2201` 下编译；原 `scatter.cpp` 保留其他架构实现。
Host 做形状乘积和字节数溢出检查、overlap 校验、连续化、arg/workspace 分配及路径规划。
Tiling 的地址、元素数、桶数与任务数使用 64 位字段；这与 mean 的源 dtype 计数语义分开。
共享 pybind 的两个 A2 Scatter 写入口 `scatter_a2_forward` 和 `scatter_forward` 同时使用
CSR 的 `ExternalCsrProofBoundary` 与 SpMM 的 `ExternalSpmmProofBoundary`
（`SpmmExternalWriteScope`），使可能被 out 独立存储别名改写的数据证明失效。
两套保护都覆盖正常返回与异常退出；只读的 `scatter_index_stats` 不进入外部写入边界。

当前流从 src 所在设备取得。分组路径在冲刷已排入的拷贝后直接发射 kernel，
其余路径使用 `OpCommand::RunOpApi` 排队。输入、输出和工作缓冲通过 storage 所有权
及实际异步消费者的流记录保护存活时间；私有融合结果遵循前述完成后发布规则。
无 out 的分组路径只写私有分配，
可局部关闭内部 RecordFunction 观察；公共绑定的 CSR 与 SpMM 外部写入边界仍然保留。
通常将非连续 src 和允许写回的 out 连续化；合法的有序步长别名路径直接使用原 strides。
低精度 sum 的重叠情形使用与 CPU 累加行为相符的源快照。必要复制都属于接口工作量。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CANN_TARGET=a2 NPU_ARCH=dav-2201 python3 -m pip install --no-build-isolation --no-deps -e .
```

自定义 CANN 安装路径按实际环境替换。新版构建把 A2/A3 扩展放在
`python/ops_gnn/A2-A3/_pybind.so`，kernel 为 `libopsgnn_npu_kernel-A2-A3.so`；
950 使用单独的 `950/` 产物。运行时设置 `CANN_TARGET=a2` 可明确选择 A2 包。
报告须核实实际加载库的绝对路径与 SHA256，不以其他工作树的旧库代替本次产物。

## 4. A2 Kernel 路径与资源

`arch22/scatter.cpp` 分发七种 L1 dtype。add 共用 Sum；mean 保留原版 CPU 的
源 dtype 分子/计数舍入和除法顺序。路径按实际 dtype、布局和平台容量选择，
不通过性能表 shape 白名单分派，不把不支持某类原子指令的 L1 dtype 交给 CPU。

| 路径 | 选择条件与用途 | 计算与写入 |
| --- | --- | --- |
| StableGrouped | 非空 src、紧凑一维 index，out 与 src/index 无共享字节 | CPU 仅生成稳定索引元数据；NPU 每个桶/特征有唯一 owner，沿原始 source 行序归约 |
| PrivateFusedSum | 稳定分组中满足下述资格的 FLOAT32 Sum/add 无 out 热命中 | 合并 EQ 与私有有序归约，完成并校验后发布结果 |
| OutputOwner | 未进入优化路径的 L1 输入 | 每块 64 个输出元素，按原行序扫描匹配 index，唯一写入者 |
| FeatureOwner | 底层 scatter_forward 保留的紧凑浮点特征路径 | 每核独占一个特征块的全部桶，按行更新；支持顺序 sum 和 mul |
| Atomic extrema | 底层 scatter_forward 中满足特征几何及大小条件的 min/max | 原子求值后再选首次有效 arg；不用于浮点 sum |
| Integer BucketTable | 底层 scatter_forward 的紧凑一维整数 Sum，前后展平维均为一，长度至少 256 | UB 桶表按整数模加合并 |
| AliasedOrdered | CPU 允许且必须保留读写顺序的字节别名 | 单核按 source 顺序执行，必要时直接使用原 strides |

### 稳定分组归约

rowlist 内同桶的行号严格保持原 source 次序。分组 owner 逐行更新 accumulator，
source tile 只用于搬运，不先算局部和再合并，因此不引入跨 tile 或跨核的浮点加法重排。
FP16/BF16 sum 在 FLOAT32 中逐行累加后收窄；低精度 mul 每次更新后收窄。
min/max 仅严格改善时更新，arg 保存原归约轴行号；未改善 out 时保留哨兵，
不会将分组内位置当作 arg。整数运算保留整数域的模加、模乘和 floor mean。

同一 batch 内的一个 output group 先批量读取相邻桶的 `rowptr` 边界。
若这些桶的总行数不超过当前 tileRows，一次搬入 row IDs 和 source tile，再逐桶按各自
原始行序归约；行数超限时保留逐桶循环并复用已预取边界。跨 batch 的输出组走原逐桶路径。
联合 rowlist 跨桶时可能不单调，因此只有端点及每个行号都确认连续时才使用连续 DMA。
低精度浮点先将整块 source 转为 FLOAT32 work，避免前一桶的 mul/mean 舍入 scratch
覆盖后续桶输入。空桶不读取 source，仍保留初值、count 和 arg 哨兵规则。

当前公开 mean 仅在浮点、无 out、紧凑一维 index、非空、归约长度不超过 2^24、
尾部特征至少 64 且行字节数为 32 的倍数时进入融合分组路径。
分子先收窄到源 dtype，再使用同源 dtype 的 count 相除；物理 index 仅一个元素时
count 为一，不能以展开 rowlist 的长度替代原始 index 计数。
其他 mean 继续组成原生 sum、原始 index 的 count sum 及设备除法，保留异常发生顺序。

分组的基础特征 tile 是 64 的倍数，最多 2048；source tile 从 32 行开始按 UB 容量减小，
必要时再减小特征宽度。基础输出组至少四行，并按 32 字节 DMA 块选择组大小。
非对齐输出行由同一 owner 处理全部特征 tile；输出起始地址不对齐时使用单核，
避免两个核同时写入同一物理 DMA 块。

私有合并 Sum 先检查基础方案：FLOAT32 Sum/add、无 out、缓存热命中、输出非空，
一个 tile 覆盖完整特征、输出组不超过四桶；实际每核索引量不超过一个 512 元素 EQ 块，
缓存计划的最大桶长度不超过基础 tileRows，Owner 与 EQ 全部同时存活 UB 可容纳。
每核向上取整的 src 与 out 元素数之和还须不超过实际 UB 容量所能容纳的 FLOAT32
元素数的八倍。这只是控制等待完整生产者成本的调度条件，未满足时走普通分组，
不缩小合法域、不改变精度标准，也不使用性能表 shape 白名单。

基础资格成立后，尝试每组 8 桶、source tile 64 行；实际 UB 不足时先缩小 source tile，
必要时缩小特征宽度。若最小 8 桶方案放不下，或重算后的实际 blocks、EQ 覆盖、最大桶长、
完整特征和工作量预算不满足条件，则保留已通过的基础方案。重新规划也保留原输出地址
不对齐导致的单核限制。合并 kernel 共用一个 TPipe，EQ 与 Owner 缓冲同时存活。

同一 batch 内一组桶的总 source 行数能装入 tileRows 时，私有 FLOAT32 Sum 才使用 wave：
不同桶直接使用输出缓冲中互不重叠的累加器，在一轮内各加一行，再统一执行向量同步。
每桶从正零开始，严格按原 source 行序逐项 FLOAT32 相加；wave 仅合并独立桶之间的同步，
不形成树形部分和或浮点原子和。总行数超限、跨 batch 等情况保留原有逐桶有序循环。

Host 对每个同时存活的 UB 缓冲逐项预算。下表记 W=tileWidth、R=tileRows、
G=outputGroup、V=validationBlocks、s=源 dtype 字节数；每项均向上对齐至 32 字节。

| 缓冲 | 字节预算 | 分配条件 |
| --- | --- | --- |
| source / output | R×W×s / G×W×s | 全部路径 |
| row IDs / accumulator | 8×R / 4×W | 全部路径 |
| groupPointers | 8×(G+1) | 全部路径 |
| validation | 32×V | 普通缓存命中，V>0；私有合并路径不分配 |
| 合并 EQ | 8352（4096×2+128+32） | 私有合并 Sum，与 Owner 共用 TPipe |
| FLOAT32 work | 4×R×W | 非标量 FP16/BF16 |
| INT64 arg group | 8×G×W | min/max |
| arg/row 浮点临时值与 mask | 两项 4×W，及 ceil(W/8) | 非标量 min/max |
| mean divisor | 4×W | 非标量 mean |
| 预留 | 8192 | 全部路径 |

普通路径的 validation 缓冲在校验后仍占用 UB，已与后续 groupPointers 和计算缓冲
同时计入预算。私有合并路径的 8352 字节 EQ 缓冲完整叠加到 Owner 预算，8192 字节
预留不挪作 EQ 缓冲；wave 使用已有 output 缓冲作为独立桶的累加器。
分组路径无需按整个输出大小分配 FLOAT32 累加 workspace。
浮点 arg 向量路径只用于长度不超过 2^24；更长的 min/max 使用标量 INT64 arg。
核数与 UB 容量从平台查询。

### 其余路径

公开的 mean 组成也通过 scatter_a2_forward 执行 sum；满足条件的分子和 count 同样使用分组。
旧 FeatureOwner、整数 BucketTable 和原子极值保留在底层 scatter_forward 入口，
公开紧凑无别名路径已由分组接管，不能把这些旧路径写成当前公开接口的性能实现。
这些底层路径保留各自的初始化、更新与结束阶段。
原子极值的候选必须等于最终极值且严格改善初值，再取最小原行号；结束时恢复选中
source 的原始位模式或未改善的 out，以保持 NaN、正负零和初值语义。
这些路径的 workspace、对齐、尾部长度及生命周期仍由 Host 预算和校验。
无消费者的浮点 Atomic sum/mean 分支已从 Kernel 分派移除；LaunchAtomic 用编译期断言
限定 min/max，BucketTable 分派也限定整数 Sum。浮点 sum 不采用无序原子累加或浮点局部桶表合并。

缓存仅保存 index 元数据，不能缓存 src、value、arg 或性能表专用结果。
全量精确校验、冷构建、必要拷贝和同步均属于公开接口成本。
是否达到 `T_A100/T_A2 >= 0.6` 由正式实测逐项判定，不能从路径名称推断。

## 5. 精度判定

本轮采用 [opbase 最新澄清版](https://gitcode.com/cann/opbase/blob/a5e8e716743b20bfb2ffa16e0213c4aa73b823b3/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)，
固定提交 `a5e8e716743b20bfb2ffa16e0213c4aa73b823b3`，正文 SHA256 为
`bdefa81d0e7b1052c7b8e5727fe88e048cd91a61bbef4fa67be4ae5a54bef023`。

| dtype | atol | `rtol` | 硬上限兜底 |
| --- | --- | --- | --- |
| FLOAT16 | 2^-10 | 2^-10 | 0.1 |
| BFLOAT16 | 2^-7 | 2^-7 | 0.1 |
| FLOAT32 | 2^-13 | 2^-13 | 0.01 |

逐元素采用 `abs(actual-golden) <= atol + rtol*abs(golden)`，每用例匹配比例至少 0.99，
且最大有限绝对误差不超过 `max(兜底值, 32*ULP(g_low))`。
g_low 是最大有限误差点的 golden 收窄到实际输出 dtype 后的值。
actual 为 Inf 而 golden 有限时按 RNE 收窄 golden，同号 Inf 视为匹配。
独立比较器另外要求 NaN 位置、Inf 符号及正负零规则一致；这些严格检查不由 0.99 比例豁免。
高精度参考有限但收窄为 Inf、actual 却有限的未定义 ULP 锚点会显式报错，
不以无限误差上限放行；常规同 dtype CPU 参考不涉及这个扩展输入。

旧 `experimental_standard.md` 提交 `adbe441e9254676174937047457419b6e645e384`
作为并列诊断输出。旧版未明确 ULP 锚点，诊断使用新澄清锚点并明确标记，
不冒充对旧版歧义的唯一解释。整数、INT64 arg 和 L2 按位比较；
CPU golden 自检使用严格比较，不能用 NPU 容差掩盖参考自身差异。

## 6. 测试与性能记录

测试保持三个正式文件：`golden.py`、`test_scatter.py`、`benchmark_scatter.py`。
golden 仅依赖 PyTorch 和标准库，不导入 ops_gnn、torch_scatter 或 arch35 golden。
原版 CPU 对拍在独立进程执行，辅助脚本和输入/输出原始记录保存在本地 output。
日常 L1 测试还应在拒绝导入 torch_scatter 的进程验证，L2 则明确验证其依赖行为。

完整功能与精度套件含 L2 用例，须安装与当前 PyTorch 2.7.1 或 2.12.0 编译匹配的
原版 `torch_scatter==2.1.2` CPU 扩展。数值参考仍为 2.7.1，2.12.0 已核验 CPU 等价。
aarch64 源码安装命令及 CPU 扩展执行检查见
[API 构建与测试说明](../../../zh/api_reference.md#a2-构建与测试)。
缺少依赖时只跳过需要该扩展的 L2 功能用例；L1 用例、拒绝导入 torch_scatter 的
L1 验证及 L2 缺依赖应报错的契约测试照常执行。跳过必须单独统计，不能计入通过数；
完整验收须安装依赖后执行全部用例。CPU 等价及 autograd 版本差异核验不等于
对应 PyTorch/CANN 组合已通过完整设备验证；设备结果以实际验收记录为准。
L2 版本检查精确限定为 PyTorch 2.7.1 和 2.12.0。

测试以唯一 pytest 测试标识 记录证据，通过 `record_property` 分别输出 value、arg、out 的
比较尝试数、通过数、失败数和元素数，并保留新旧精度指标摘要与逐次比较记录。
只有实际 NPU Tensor 与参考的断言计入，CPU 参考生成不计；重复检查同一输出仍是两次断言，
不能把比较次数当成唯一样例数。依赖隔离子进程的比较带 subprocess 标记并归入所属 测试标识。
失败、跳过和 setup error 以 pytest 结果为准；没有执行比较不代表通过。
严格 exact/bitwise 断言保持原判定，附加混合容差指标仅用于诊断。
保存这些属性时使用 `--junitxml=<path> -o junit_family=xunit1`。

功能覆盖 TC-01～TC-16、九 dtype、六 reduce、全部公共入口，以及广播、尾块、空输入、
非连续 out 的异常、极值身份值、低精度抵消、整数回绕、别名和非默认流。
热命中还覆盖推导输出维度变化、已返回输出的释放或存储重绑，以及私有 wave 的特殊值与顺序。
共享证明回归还覆盖：预热 SpMM 源值 NaN 证明后，通过独立存储别名调用公开 Scatter
或上述两个原生写入口写入 NaN，再调用 SpMM 验证证明已失效并重新计算。
该场景覆盖 FLOAT16/FLOAT32；执行与通过数量以本轮实际测试记录为准。
大规模输入使用测试文件的显式手动入口，不让普通 pytest 自动分配巨型张量。
COO/CSR/SpMM 共享边界回归和 950 可用的构建/分派检查单独记录；
没有 950 设备结果时不得写为 950 设备通过。

性能以[原任务书](https://gitcode.com/cann/cann-ops-competitions/blob/ce603dc73335bbb47b3a77d0e4a63fa16f66d136/04_tasks/01_community-task-2026/docs/202608/scatter_task_doc.md)
40 项公布 A100 数值为外部标杆。关联自测工程明确五个 shape 的桶数依次为
4096、8192、16384、16384、16384，FLOAT16/FLOAT32，sum/mean/min/max，
随机标准正态 src、均匀随机 INT64 一维 index、dim=0、out=None、统一 value-only API。
预热 20 次，计时 100 次公开调用，循环前后同步，以 host perf_counter 批量均值计时。
全部校验、转换、归约和必要复制都计入；本轮固定随机种子，不要求维护方补交旧 A100 日志。
脚本在正式预热前单独同步测量 `first_call_ms`，随后按原口径预热与计时；
记录实际缓存环境、设备可见性、CPU affinity 和线程数，不在计时外预载分组元数据。
默认缓存与 `OPSGNN_SCATTER_GROUP_CACHE=0` 的冷模式分别保留数据。

每个用例、每轮要求 `R=T_A100/T_A2 >= 0.6`，保存未舍入时间及失败轮次。
本轮先比较 FLOAT32/FLOAT16 已公布的 40 项；mul、add 以及其余 L1 dtype 的原数值
未公布，保留测试入口并标 `missing_baseline`，待后续资料补齐后验收。
其他 dtype 的性能暂缓不影响全 dtype 的功能和精度范围，不能套用其他 dtype 的性能数值
或将暂缓写成永久免验；L2 不考核性能。
更改桶数、out、API 或迭代口径的测试标为开发测试。
out 开发测试每次恢复初值，恢复复制明确包含在计时内。
只有显式 `--official-shapes` 才批量选择五个大 shape。

```bash
CANN_TARGET=a2 NPU_DEVICE_ID=0 python3 -m pytest test/scatter/arch22/test_scatter.py -q
CANN_TARGET=a2 python3 test/scatter/arch22/benchmark_scatter.py \
  --device npu:0 --dtype float16 float32 --reduce sum mean min max \
  --official-shapes --json output/scatter-a2-performance.json
```

选择可用的空闲设备并替换命令中的设备编号，先完成正确性验证，再单独计时。
多设备并行验证应单独记录实际使用的设备和结果；device key、DeviceGuard 和
recordStream 的源码核查不能替代该实测。950 只记录实际执行的 CPU/分派等检查，
不能写成 950 设备通过。
本轮另使用官方 ATK 固定提交 `45cc4061bc019e73f63a11e96496e97df0d6810c`，
通过 BaseApi 插件直接调用原版 CPU 与冻结 A2 公开接口，完成小规模功能精度补充验证。
自定义比较器沿用本文已批准的精度规则，保留旧版诊断，并严格比较 arg、整数和 L2；
有 out 的用例还单独比较实际写回值。ATK 框架源码未修改，完整 CLI、环境、日志、
逐项比较与工具报告、源码及双库哈希保存在本轮验收记录中。
该结果不替代正式 40 项性能判定；本文不声明性能已通过。

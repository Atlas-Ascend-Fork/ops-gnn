# KNN 算子设计

## 功能与接口

为 Ascend 950 提供 `ops_gnn.knn` 和 `ops_gnn.knn_graph`，用于点云近邻
查询和图构建。复用仓库的 Ascend C SIMT、PyBind 和 `opsgnn` 命名空间。

- 输入 `x[N,F]`、`y[M,F]` 位于同一 NPU，支持 FP16/FP32。
- 一维输入转为列向量；非连续输入在调用内转为连续布局。
- sorted int64 batch 通过 bucketize 转为指针，只搜索查询所在 batch。
- 输出为 int64 `[2,M*k]`，两行分别为查询索引和候选索引；空输入返回 `[2,0]`。
- 支持欧氏和余弦距离。`num_workers` 为兼容 CPU torch_cluster 接口而保留，
  不参与 NPU 调度；NPU 使用平台 AIV 数和当前 stream。
- 图接口支持去自环和两个 flow 方向。`loop=False` 请求 `k+1` 个邻居后去自环。

要求 `1 <= k <= N`，N 为对应 batch 的候选数；无自环图要求 `k+1 <= N`。
大 k 使用动态状态，不额外施加 k<=100 限制，受实际显存容量约束。
NaN/Inf 排序和非 sorted batch 不属于接口保证。

## 数值语义

欧氏距离兼容 CPU torch_cluster 1.6.3 捆绑的 nanoflann `L2_Adaptor`。
平方差以四个特征为一组左结合相加，然后加入距离累加器，尾部逐项累加。
FP16 的每次减法、乘法和加法显式舍入至 FP16；不使用 FP32 累加替代。

```text
s = half(0)
R(v) = round_half_rn(v)
每四个特征一组:
    d[i] = R(y[i] - x[i]); p[i] = R(d[i] * d[i])
    group = R(R(R(p[0] + p[1]) + p[2]) + p[3])
    s = R(s + group)
剩余特征: s = R(s + R(R(y[i] - x[i]) ** 2))
```

FP32 在源码中保持相同的四项分组表达式，编译沿用仓库默认参数，不再为
KNN 单独禁用乘加融合。两种编译配置在 Ascend 950 上各通过 300 项
功能测试和两套各 64 项性能测试，未观察到邻居集合差异或门槛失败；
这不是对所有近乎等距 FP32 输入的位级等价证明。
逐特征 FP16 累加与上述分组运算可能产生不同集合，不能互换参考实现。

L2 等距时按照 CPU 树遍历次序取舍。余弦距离使用 FP32 点积和范数，
零范数点对的距离为 1，等距时优先较小的候选索引。

## Host 与设备实现

Python 处理输入形状、连续化、batch 推断和当前 NPU stream。C++ 校验
device、dtype、维度、k 和 batch 指针，分配输出并在设备 guard 下调用
`opsgnn::Knn`。只有 batch 指针复制到 CPU 做合法性检查；点坐标、建树、
距离计算和近邻选择均留在 NPU。

每 AIV 启动 256 个 SIMT 线程，`LAUNCH_BOUND` 与 `VF_CALL` 保持一致。
核数取平台 AIV 数且不超过 M。每个线程独立处理查询并扫描对应 batch
的全部候选，维护有序 Top-K，不物化 M×N 距离表。
小 k 按 1/4/8/16/32/64 模板容量分派；F=3/16/32 且容量不超过 32
时缓存查询坐标，其余维度走通用循环。k>64 使用 M×k FP32 GM 距离状态，
索引直接存入输出，每个查询独占其切片。当前 stream 的分配和启动遵循
PyTorch allocator 生命周期，不缓存跨调用答案。

树元数据用于复现 L2 等距次序，在同一 stream 上先初始化根包围盒再建树。
分区与遍历规则改编自 BSD 许可的 nanoflann，来源范围见
[API 文档](../../../zh/api_reference.md#knn-第三方来源)，许可全文见根目录 [LICENSE](../../../../LICENSE)。
叶子容量为 10，两阶段分区分别使用 `<cut` 和 `<=cut`。协作分区将左侧
错误位置升序与右侧错误位置降序配对交换，保留原 two-pointer 置换顺序。
同叶内比较置换位置，跨叶通过最近公共祖先和查询到切面的比较决定先后。

每 batch 使用一个 256 线程块。通常协作处理前 7 个 FIFO 节点；有效
建树维度不超过 3 时，batchN>=8192 使用 31 个，batchN>=16384 使用
63 个，其余不相交子树按前沿并行。FP16 切分轴遵循参考舍入规则保持为 0，
只需第一维包围盒；FP32 保留所有维度。
8 KiB UB 用于 min/max、整数计数和前缀计算；等值保留较早位置，
同步屏障保护归约、前缀和前沿发布。临时交换索引复用尚未分配的节点区，
后续节点覆盖前已结束对临时列表的使用，不使用核间自旋。

## 复杂度与存储

最坏计算量为 `O(M*N*(F+k))`，多 batch 时按各 batch 求和。
输出为 `O(M*k)`；小 k 每线程状态为 `O(KMax+FeatureCache)`，大 k
额外使用 `O(M*k)` GM 距离状态。
树最多 2N 个节点，每节点 8 个 int64，另有 3N 个 int64 置换、逆置换和
叶子索引；包围盒为 `2N*(2*bbox_features+2)` 个 float，FP16 时
bbox_features=1，FP32 时为 F。分配、建树和查询全部计入完整 API 耗时。

## 测试与兼容性

测试位于 `test/knn/arch35/`，运行命令见[API 文档](../../../zh/api_reference.md#knn--knn_graph--精确近邻搜索)。功能测试使用
独立 PyTorch CPU golden 比较完整近邻边多重集合，余弦使用 CPU double
标杆；覆盖 dtype、batch、k 边界、等距、图方向、非默认 stream 和大树。
golden 不依赖 torch_cluster、不调用待测实现；大形状 CPU 查询分成
至多 32 条查询的片段，每片保留完整 x 并穷举距离，树仅决定等距顺序；
NPU 仍为一次完整形状调用，不缓存答案。
接口契约单独检查参数绑定、默认值、类型注解和图后处理；CPU 标杆自测
覆盖距离舍入、等距、batch 隔离、分块、余弦、图组合和依赖隔离。

性能脚本包含 32 个形状、两种 dtype；默认 20 次预热、100 次完整 API
调用并在计时边界同步。每项单独比较 `reference_ms/measured_ms>=0.45`，
不能拼接不同轮次的最好成绩。日志和自测报告作为 PR 验证材料保留在源码树外。
本实现针对 Ascend950PR/arch35 (`dav-3510`)，不为 A2/A3 导出或编译。
构建与测试依赖见[API 文档](../../../zh/api_reference.md#knn--knn_graph--精确近邻搜索)；多 NPU 交叉调用不作为已验证功能。

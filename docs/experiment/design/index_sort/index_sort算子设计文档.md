# index_sort / sort_edge_index 算子设计

## 功能与接口

为 `ops_gnn` 提供两个公开入口，语义对齐 PyG，用于把 COO 边表按节点分组：

```python
index_sort(inputs, max_value=None) -> (sorted_inputs, perm)
sort_edge_index(edge_index, edge_attr=MISSING, num_nodes=None,
                sort_by_row=True) -> Tensor 或 (Tensor, 属性)
```

- `index_sort` 对一维非负整数键做升序排序，同时返回置换 `perm`，供调用方搬移任意载荷。
- `sort_edge_index` 输入 `(2, E)` 边表；`sort_by_row=True` 时先按源节点、再按目标节点升序，
  `False` 时先按目标节点、再按源节点。边属性用同一置换搬移，返回约定与 PyG 一致：
  不传属性只返回边表，传 `None` 返回 `(edge_index, None)`，传 `Tensor` 或列表逐个搬移。
- 前置条件：键非负；`max_value` 是键的上界（不含）；`num_nodes` 不小于 `edge_index.max() + 1`。
  三者都是调用方承诺，与上游一致，算子不读取设备来校验，避免为校验引入同步。

## 数值语义

两个入口都承诺**稳定**排序：相等键（重复边）保持输入先后顺序，因此置换只由输入决定，
与 AIV 核数、分块方式无关，可以逐位比较。上游 `index_sort` 默认不承诺稳定性，这里更强。

`sort_edge_index` 的目标顺序等价于对复合键 `primary * num_nodes + secondary` 做稳定排序。
实现有两种等价写法：

1. 单趟：直接构造复合键并稳定排序；要求复合键落在排序的精确范围内。
2. 两趟：先按次键稳定排序，再按主键稳定排序。稳定排序保证第二次排序不会打乱同主键内
   已按次键排好的顺序，因此结果与单趟逐位相同，且键始终不超出 `2 ** 24`（A2）或 `int64`
   （950）的范围。

A2/A3 的 float32 路径依赖一个事实：`[0, 2 ** 24)` 内的整数在 float32 中精确可表示，排序
与转换都不丢位。`2 ** 24` 以上不满足该性质，必须回退 int64。

与 PyG 的差异：排序稳定；复合键超出 int64 时用两趟代替 `lexsort`；不返回置换（由
`index_sort` 提供）；`EdgeIndex` 子类按普通张量处理。

## Host 与设备实现

实现完全由设备侧原生算子组合而成，不新增 AscendC 内核，按 `_pybind.npu_arch` 分流：

- **arch35（950，`dav-3510`）**：`argsort(int64, stable=True)` 与 `sort(int64, stable=True)`
  是该平台的原生快算子（1M 键约 0.29 ms），复合键用 int64 构造，单趟完成。
  `num_nodes ** 2` 超出 int64 时改用两趟。
- **arch22（A2/A3，`dav-2201`）**：CANN 的 `aclnnSort` 只有 float32/float16 的
  `SortAiCore` 实现，int32/int64 落到 `SortAiCpu`；profiler 在 1M 键上分别记录到
  `aclnnSort_SortAiCore_Sort`（1.48 ms）与 `aclnnSort_SortAiCpu_Sort`（214 ms）。
  因此键的上界不超过 `2 ** 24` 时先转 float32 稳定排序，再把有序值无损转回 int64；
  否则回退 int64。`num_nodes ** 2` 不超过 `2 ** 24` 时复合键单趟，否则两趟。

搬移数据统一用 `index_select`：A2 上高级索引 `keys[perm]` 实测 4.27 ms/1M，而
`index_select` 为 0.57 ms，差异来自高级索引的通用 gather 路径。

同步点只有一处：`max_value` / `num_nodes` 省略，或（A2 上）给定的上界不能证明键在
`2 ** 24` 以内时，读取一次设备标量来选路径；其余调用在当前 NPU stream 上异步执行。
空输入直接返回空输出，不启动排序。

`index_sort` 用 `torch.sort` 直接拿到有序值，省掉一次 gather；`sort_edge_index` 只需要
置换，用 `torch.argsort`，避免物化有序键。

## 复杂度与存储

排序主导，时间为 `O(E log E)`；两趟为常数倍。额外存储为输出键/边表与置换，均为 `O(E)`。
实现不为 `max_value` 分配桶或计数表，因此栈/显存占用与键的取值范围无关，不会出现
`num_nodes ** 2` 级别的分配——这也是选两趟而不是照搬复合键计数排序的原因。

A2 的 float32 路径额外需要一份 `E × 4` 字节的 float32 键；1M 边约 4 MB，随调用释放。

## 测试与兼容性

测试位于 `test/index_sort/{arch22,arch35}/` 与 `test/sort_edge_index/{arch22,arch35}/`，
每个目录只有 `golden.py`、`test_<op>.py`、`benchmark_<op>.py`。golden 是独立的 CPU 实现
（稳定 `argsort`、复合键），不导入 `ops_gnn`；两个平台都没有安装 `torch_geometric`，
因此语义以源码对照为准，不与 PyG 运行时对拍。

`arch35` 下的三个文件只是对 `arch22` 同名模块的再导出与继承：用例、参考实现和测量循环
都与架构无关，唯一按平台推导的是两趟排序所需的 `num_nodes`（`5000` 与 `4e9`），由用例
在运行时按 `npu_arch` 取值。这样两个平台跑同一份用例，不会出现两份几乎相同的代码随
平台改动而漂移。

功能测试覆盖：空输入、单元素、全同键、单调增减、大量重复、宽表少边、重复边稳定性、
`sort_by_row` 两个方向、`num_nodes` 省略与显式、单趟与两趟结果一致、属性搬移的四种返回约定、
非连续输入、输出元信息、确定性、非法参数（dtype/rank/device/负上界/属性行数与 device）。
`2 ** 24` 两侧的边界单独成例，确认两条路径都与 golden 逐位一致。

性能脚本的 baseline 是 PyG 等价组合：`int64` 稳定 `argsort` 加 `index_select`，以及
复合键单趟。A2 上 1M 键 `index_sort` 实测 1.50 ms（baseline 219 ms），1M 边
`sort_edge_index` 5.2 ms（baseline 221 ms）；950 上两者与 baseline 同量级，收益来自
API 封装而非算法差异。两平台数据分别采集、分别记录，不混合统计。

未覆盖：反向（整数索引入参没有梯度）、`EdgeIndex` 子类元数据保留、`num_nodes`/`max_value`
的主动校验。

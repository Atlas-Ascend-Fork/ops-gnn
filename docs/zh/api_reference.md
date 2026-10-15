# ops-gnn API文档和使用示例

本文档提供 ops-gnn 库的详细 API 接口说明，包括算子签名、参数说明、返回值和使用示例。

## 一、类型定义

### Tensor

```python
import torch
Tensor = torch.Tensor
```

NPU 设备上的 Tensor，由 PyTorch 管理内存。

### OptTensor

```python
from typing import Optional
OptTensor = Optional[torch.Tensor]
```

可选 Tensor 类型，用于可能有默认值的可选参数。`None` 在 pybind 层显式表示“未提供”；合法的空 Tensor 仍按真实 `out` 参数处理。

---

## 二、核心算子API

### nearest — 同 batch 最近邻索引

```python
def nearest(x: Tensor, y: Tensor,
            batch_x: Optional[Tensor] = None,
            batch_y: Optional[Tensor] = None) -> Tensor:
```

Ascend 950（arch35）上为每个x点返回同batch最近y点的全局int64索引，输出形状[N]。x/y为同一NPU、相同float16或float32 dtype的[N,F]/[M,F]；一维输入按F=1处理。两种dtype提升到FP32计算，等距选最小全局y下标。

可选batch为同设备int64的一维非负非降序标签，长度分别为N/M；两侧非空标签集合须一致，省略的一侧视为batch 0。不合法batch抛ValueError。支持非连续输入；未传batch的空x返回空输出，非空x配空y报错。NaN/Inf坐标及F=0不支持。

```python
x = torch.tensor([[0., 0., 0.], [9., 0., 0.]], device="npu")
y = torch.tensor([[1., 0., 0.], [10., 0., 0.]], device="npu")
indices = ops_gnn.nearest(x, y)  # tensor([0, 1], device='npu:0')
```

NPU使用独立Ascend C内核，通过torch.ops.torch_cluster.nearest分派；无CPU/scipy回退或梯度。N、M、F各不超过INT32_MAX，实际规模受内存限制。公共API校验包含同步，内核使用当前NPU stream；已安装torch_cluster时应先导入该包，由其提供schema。

精度参考固定为SciPy 1.18.0 / OpenBLAS 0.3.31.dev SkylakeX。不同CPU、BLAS或矩阵切分可能改变极近候选的整数标杆；极大FP32坐标导致中间距离溢出的精度不作保证。

测试位于`test/nearest/arch35/`，分为两项，共用参考辅助文件`golden.py`：

- 功能测试：`test_nearest.py`，共338项，包含任务书附件的42项用例，以及功能、FP16、验收日志重建、108项数值压力和4项参考语义回归等296项补充用例。沿用本PR现有的参考计算和断言；最初9项失败场景由任务形状及跨1024平局用例覆盖。
- 性能测试：`benchmark_nearest.py`，18组形状×2 dtype，每点预热20次、计时100次，覆盖完整Python接口，要求标杆耗时/NPU耗时不低于0.45，并严格比较全部输出下标。

完成仓库构建并激活CANN和Python环境后，在仓库根目录执行：

```bash
python -m pip install scipy==1.18.0 pytest
export NPU_DEVICE_ID=0
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest test/nearest/arch35/test_nearest.py -v
python test/nearest/arch35/benchmark_nearest.py --warmup 20 --iter 100
```

测试使用`torch.equal`比较int64下标。性能结果默认写入本地`artifacts/nearest/`，不提交仓库。CPU参考不切分batch，最大性能形状临时矩阵约4 GiB，建议至少8 GiB可用内存；请顺序执行NPU测试。

---

### segment_max_csr — CSR 分段最大值

本节描述原有的 `int32` 指针接口。新增的 `int64` 指针算子族见下方
[`segment_csr`](#segment_csr--int64-csr-分段归约算子族)；两条路径的空段和返回值约定不同。

**函数签名：**

```python
def segment_max_csr(
    src: Tensor,
    indptr: Tensor,
    optional_out: Optional[Tensor] = None,
) -> Tensor:
```

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| src | Tensor | 输入 | 输入数据 Tensor，必须位于 NPU 设备 |
| indptr | Tensor | 输入 | CSR 格式的索引指针 Tensor，dtype 必须为 `torch.int32`，必须位于 NPU 设备。最后一维的长度 - 1 决定分段数量 |
| optional_out | Optional[Tensor] | 输入 | 可选输出 Tensor，位于 NPU 设备。如果提供，每个 segment 的第一个元素会与 `optional_out` 的对应值做 max 比较（即 `max(src[start:end], optional_out)`），而非仅从 src 中取最大值 |

**返回值说明：**

返回一个新的 Tensor，dtype 与 `src` 相同。形状与 `src` 相同，但 `indptr` 最后一维对应的维度被缩减为 `nSegments = indptr_last_dim - 1`。

**功能说明：**

沿 CSR 格式的 `indptr` 指定的维度对 `src` 进行分段最大值规约。对于每个分段 `[indptr[seg], indptr[seg+1])`，计算该分段内所有元素的最大值。支持多维广播：当 `indptr` 的维度低于 `src` 时，沿对应的维度广播 indptr。

**实现架构：**

- **Kernel 模式**：Kernel 类模式（TPipe + Buffer + Event），4 文件实现
- **数据搬移**：DataCopy + 双缓冲流水线（MTE2 ↔ VECCALC ↔ MTE3）
- **Tiling**：`SegmentMaxCsrTilingData` 结构体，包含分块参数（coreDataNum、KloopTime、ALIGN_NUM 等）
- **多 AIV 核并行**：按 `E_1`（广播维度）均匀分配到各 AIV 核，每核独立处理

**调用链：**

```test
ops_gnn.segment_max_csr(src, indptr, optional_out)
    → _pybind.segment_max_csr()           # PyTorch 绑定
        → segment_max_csr()                # Host 端：维度解析、Tiling 填充、dtype 分发
            → LaunchSegmentMaxCsrKernel<T>()  # Kernel Launch：获取 AIV 核数、<<<>>> 启动
                → segment_max_csr_kernel<T>   # Device 端：Init → Process → Compute 流水线
```

**支持的 dtype：**

| dtype | 对应 fill 值（空 segment） |
|-------|--------------------------|
| `torch.float32` | `-3.4028235e+38`（-FLT_MAX） |
| `torch.float16` | `-65504`（-HALF_MAX） |
| `torch.int32` | `-2147483648`（INT32_MIN） |
| `torch.int16` | `-32768`（INT16_MIN） |

**注意事项：**

- `indptr` 的 dtype 必须为 `torch.int32`
- `indptr` 的值必须是非递减的（即 `indptr[seg] <= indptr[seg+1]`）
- 空 segment（`indptr[seg] == indptr[seg+1]`）：若无 `optional_out`，填充对应 dtype 的最小值；若有 `optional_out`，直接拷贝 `optional_out` 对应位置的值
- 支持广播：1D indptr 可广播到多个 batch（`indptr.view(1, -1)`）
- 输入必须位于 NPU 设备（`device='npu'`）

**使用示例：**

#### 基本用法 — 1D indptr

```python
import torch
import os
import ops_gnn

device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
torch.npu.set_device(device_id)
torch.manual_seed(42)

# 创建输入数据
# src shape: (4, 2)，4 个"行"，每行 2 个元素
src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
# indptr: [0, 2, 4] 表示第 0 个分段取 src[0:2]，第 1 个取 src[2:4]
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[3, 4], [7, 8]]  — shape: (2, 2)
#           分段0: max(src[0:2])  分段1: max(src[2:4])
```

#### 不同 dtype

```python
# float16
src = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16, device='npu')
indptr = torch.tensor([0, 2], dtype=torch.int32, device='npu')
result = ops_gnn.segment_max_csr(src, indptr)

# int32
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device='npu')
indptr = torch.tensor([0, 2], dtype=torch.int32, device='npu')
result = ops_gnn.segment_max_csr(src, indptr)
```

#### 空 segment

```python
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='npu')
# indptr: [0, 0, 2] — 第 0 个分段是空的，第 1 个取 src[0:2]
indptr = torch.tensor([0, 0, 2], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result[0] = [-3.4028235e+38, -3.4028235e+38]  — 空分段填充 float32 最小有限值（-FLT_MAX）
# result[1] = [3, 4]        — src[0:2] 的最大值
```

#### 带 optional_out

```python
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='npu')
indptr = torch.tensor([0, 2], dtype=torch.int32, device='npu')
optional_out = torch.tensor([[10, 20]], dtype=torch.float32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr, optional_out)
# result = [[10, 20]]  — max(src[0:2], optional_out) = max([[1,2],[3,4]], [[10,20]])
```

#### 2D indptr（每 batch 不同分段）

```python
src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
# 2D indptr: batch 0 用 [0, 2, 4]，batch 1 用 [0, 1, 3]
indptr = torch.tensor([[0, 2, 4], [0, 1, 3]], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[2, 4],    — batch 0: seg0=src[0:2], seg1=src[2:4]
#           [5, 7]]    — batch 1: seg0=src[0:1], seg1=src[1:3]]
```

#### indptr 广播

```python
src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, device='npu')
# 1D indptr，view 成 (1, -1) 后广播到两个 batch
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu').view(1, -1)

result = ops_gnn.segment_max_csr(src, indptr)
# result = [[2, 4],    — batch 0: seg0=src[0:2], seg1=src[2:4]
#           [6, 8]]    — batch 1: 同上分段
```

#### 复杂形状（3D src + 2D indptr）

```python
src = torch.randn(3, 8, 16, dtype=torch.float32, device='npu')
indptr = torch.tensor([[0, 4, 8]], dtype=torch.int32, device='npu')

result = ops_gnn.segment_max_csr(src, indptr)
# result shape: (3, 2, 16) — indptr 的最后一维从 3→2，其他维度不变
```

---

### segment_csr — int64 CSR 分段归约算子族

Ascend 950 / arch35 上按 CSR 行指针做确定性分段归约，支持 `sum`、`add`、
`mean`、`min`、`max`。归约维度为 `indptr.dim() - 1`，输出该维长度为
`indptr.size(-1) - 1`。每个输出由一个处理单元写回，不使用全局原子累加。

```python
segment_csr(src, indptr, out=None, reduce="sum")  # 仅返回值 Tensor
segment_sum_csr(src, indptr, out=None)            # Tensor
segment_add_csr(src, indptr, out=None)            # Tensor
segment_mean_csr(src, indptr, out=None)           # Tensor
segment_min_csr(src, indptr, out=None)            # (values, indices)
segment_max_csr(src, indptr, out=None)            # int64 indptr: (values, indices)
```

| 参数 | 约束 |
| --- | --- |
| `src` | NPU 张量；支持 float16、float32、bfloat16、int8、uint8、int32、int64 |
| `indptr` | 同设备 `int64` 张量；最后一维非降序，边界在归约维范围内；前导维可向 `src` 广播 |
| `out` | 可选，同设备、同 dtype、匹配输出形状；支持非连续视图 |
| `reduce` | 仅通用接口使用；不支持 `mul`，无 `dim_size` 参数 |

`min/max` 的 `indices` 为首次极值在输入归约维中的 `int64` 位置；空段值为零，
索引为输入归约维长度。空 `src` 且提供 `out` 时保留已有输出。
整数 `mean` 先按源 dtype 转换和及段长，再向零截断整数除法；浮点
`mean` 保留和及段长的舍入顺序。`int8/uint8` 段长转换为零时返回零。
非法指针和 NaN 极值传播不在此接口的扩展承诺内。

```python
import torch
import torch_npu
import ops_gnn

src = torch.tensor([[1., 4.], [2., 3.], [5., 6.]], device="npu")
indptr = torch.tensor([0, 2, 3], dtype=torch.int64, device="npu")
summed = ops_gnn.segment_csr(src, indptr, reduce="sum")
values, indices = ops_gnn.segment_max_csr(src, indptr)
```

`segment_max_csr` 遇到 `int32` 指针时仍走原有的仅返回值路径；历史
`optional_out=` 关键字可通过 `segment_max_csr_legacy` 使用。

---

### radius / radius_graph — 半径内邻居搜索

与 `torch_cluster.radius` / `radius_graph`（>= 1.6.0）接口完全一致的 NPU 实现，
Ascend 950PR。对 `y` 中每个查询点，在 `x` 中查找欧氏距离 `dist² <= r²` 的所有
邻居（同 batch 内），超过 `max_num_neighbors` 时保留扫描顺序前 K 个（确定性）。
输出 `edge_index [2, E]`（int64）。

```python
def radius(
    x, y, r,
    batch_x=None, batch_y=None,
    max_num_neighbors=32, num_workers=1,
    batch_size=None, ignore_same_index=False,
) -> torch.Tensor          # [2, E] int64

def radius_graph(
    x, r,
    batch=None, loop=False,
    max_num_neighbors=32,
    flow='source_to_target', num_workers=1, batch_size=None,
) -> torch.Tensor          # [2, E] int64
```

| 参数 | 类型 | 输入/输出 | 描述 |
|------|------|-----------|------|
| `x` | Tensor [N,F] | 输入 | 邻居候选点集（float16/bf16/float32；float64 走 CPU 回退） |
| `y` | Tensor [M,F] | 输入 | 查询点集 |
| `r` | float | 输入 | 搜索半径（>0） |
| `batch_x` / `batch_y` | Tensor | 输入 | batch 归属（需已排序） |
| `max_num_neighbors` | int | 输入 | 每查询保留的邻居上限（默认 32） |
| `ignore_same_index` | bool | 输入 | 跳过 `i == q` 自环 |
| `loop` / `flow` | bool/str | 输入 | radius_graph 自环与方向语义 |

```python
x = torch.randn(1000, 3, device='npu')
edge = ops_gnn.radius(x, x, 0.8)          # 找半径 0.8 内的邻居
edge_g = ops_gnn.radius_graph(x, 0.8)     # 构建半径邻居图（默认不含自环）
```

**实现架构：**

- **Kernel 模式**：Ascend C SIMT（`__simt_vf__` + `VF_CALL`），空间网格剪枝 + 排序数组 top-K + early-break
- **网格构建**：device 端完成（min/max、cell、sort、index_select、offsets），免 CPU sort 瓶颈
- **输出压缩**：device 端 `repeat_interleave` + `masked_select` 压缩为 `[2, E]`
- **适用场景**：3D 点云/GNN 邻域构图（PointNet++、DGCNN 等）

**注意事项：**

- `x` / `y` 须为同一 NPU 设备上的 Tensor（`device='npu'`），`F` 一致；非连续输入会在 host 入口自动连续化
- `batch_x` / `batch_y` 须 sorted；搜索限定在同一 batch 内
- 邻居数超过 `max_num_neighbors` 时保留扫描顺序前 K 个（确定性截断，与 CPU 参考一致）
- `radius_graph` 的 `loop` / `flow` 语义与 `torch_cluster` 一致
- L1 支持 float16 / bfloat16 / float32（NPU 路径）；float64 走 CPU 回退（bit-wise，不参与性能考核）
- 空输入返回 `[2, 0]` LongTensor，不进入 kernel

### graclus_cluster — 图贪心聚类

**函数签名：**

```python
def graclus_cluster(
    row: Tensor,
    col: Tensor,
    weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
```

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|-----------|------|
| row | Tensor | 输入 | COO 源节点索引，dtype 为 `torch.long` |
| col | Tensor | 输入 | COO 目标节点索引，dtype 为 `torch.long` |
| weight | Optional[Tensor] | 输入 | 可选边权，float16/bfloat16/float32 走 NPU 路径，float64 走 CPU 回退语义 |
| num_nodes | Optional[int] | 输入 | 节点数量；默认从 row/col 推断 |

**返回值说明：**

返回 `torch.long` Tensor，shape 为 `[num_nodes]`，每个元素为对应节点的 cluster ID。

**功能说明：**

Python 层复现 `torch_cluster.graclus_cluster` 预处理：推断 `num_nodes`、去除自环、无 weight 时随机打乱边顺序、按 row 排序并转 CSR。NPU L1 路径随后调用 Ascend C kernel 按随机节点顺序执行图贪心匹配：有 weight 时选择未标记邻居中权重最大的节点配对，无 weight 时选择第一个未标记邻居。float64 weight 按任务书 L2 口径使用 CPU 回退语义。

**注意事项：**

- `row` / `col` 必须为 1D `torch.long` Tensor。
- `row` / `col` / `weight` 必须位于同一设备。
- 算法包含随机性；固定 `torch.manual_seed` 后结果可复现。
- 自环会在 Python 层去除。

### gather_coo — COO 行扩展

**函数签名：**

```python
def gather_coo(
    src: Tensor,
    index: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
```

令 `dim = index.dim() - 1`，`index` 的前缀形状必须与 `src` 的前 `dim` 个维度一致。输出形状与 `src` 相同，仅将 `dim` 维替换为 `index.size(-1)`，并满足：

```text
out[..., e, ...] = src[..., index[..., e], ...]
```

`index` 必须是 NPU 上的 `torch.int64`，并满足任务书规定的非降序和值域前置条件。算子只实现前向原始位拷贝，不提供反向实现；支持 rank 1～8、非连续输入/输出、空 Tensor，以及 FP16/BF16/FP32、INT8/16/32、UINT8、FP64/INT64。

当提供 `out` 时，必须是与推导结果完全相同 shape、dtype 和 device 的 Tensor；返回值与 `out` 共享存储。`None` 与显式传入的空 `out` 不等价。

**示例：**

```python
import torch
import ops_gnn

device = torch.device("npu")  # 使用调用进程的当前 NPU，不硬编码设备序号
src = torch.arange(20, dtype=torch.float32, device=device).reshape(5, 4)
index = torch.tensor([0, 1, 1, 4], dtype=torch.int64, device=device)
out = ops_gnn.gather_coo(src, index)
# out.shape == (4, 4)，连续重复的 index=1 会复制同一行

provided = torch.empty_like(out)
returned = ops_gnn.gather_coo(src, index, out=provided)
assert returned.data_ptr() == provided.data_ptr()
```

**实现与性能说明：**

- Host 将输入展平为 `B × N × K`、`B × E` 和 `B × E × K`，使用 64-bit 长度/偏移；kernel 复用当前 PyTorch NPU stream，不创建或同步私有 ACL stream。
- 非连续 `src/index` 在当前 stream 上连续化；显式 `out` 通过连续临时结果回写，以覆盖非连续和别名场景。
- 有序的连续重复索引在单 UB tile 内复用源行；不按测试 case 或固定 shape 白名单路由。

---

### ind2ptr — 行索引转 CSR 行指针

**函数签名：**

```python
def ind2ptr(
    ind: Tensor,
    num_rows: int,
) -> Tensor:
```

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| ind | Tensor | 输入 | 1-D `torch.long` 非降序行索引，必须位于 NPU |
| num_rows | int | 输入 | 行数（对应 torch_sparse 中的 `M`），输出长度为 `num_rows + 1` |

**返回值说明：**

返回 1-D `torch.long` CSR 行指针，形状为 `[num_rows + 1]`，与输入同 device。

**功能说明：**

将有序行索引转换为 CSR 行指针，API 对齐 `torch.ops.torch_sparse.ind2ptr(ind, M)`。在当前 NPU stream 上异步执行；如需在 Host 读取结果，请先 `torch.npu.synchronize()`。

**注意事项：**

- `ind` 须为非降序；空输入返回全 0 的 `[num_rows + 1]`
- dtype 为 `torch.long`（int64）
- 非连续输入会在调用前转为连续张量

**使用示例：**

```python
import torch
import ops_gnn

row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
rowptr = ops_gnn.ind2ptr(row, 8)
# tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], device='npu:0')
```

---

### ptr2ind — CSR 行指针转行索引

**函数签名：**

```python
def ptr2ind(
    ptr: Tensor,
    num_edges: int,
) -> Tensor:
```

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|------|------|
| ptr | Tensor | 输入 | 1-D `torch.long` CSR 行指针，长度 `num_rows + 1`，必须位于 NPU |
| num_edges | int | 输入 | 边数 / 非零元个数（对应 torch_sparse 中的 `E`），输出长度为 `num_edges` |

**返回值说明：**

返回 1-D `torch.long` 行索引，形状为 `[num_edges]`，与输入同 device。

**功能说明：**

将 CSR 行指针转换为行索引，API 对齐 `torch.ops.torch_sparse.ptr2ind(ptr, E)`。在当前 NPU stream 上异步执行；如需在 Host 读取结果，请先 `torch.npu.synchronize()`。

**注意事项：**

- dtype 为 `torch.long`（int64）
- `num_edges == 0` 时返回空索引张量
- 非连续输入会在调用前转为连续张量

**使用示例：**

```python
import torch
import ops_gnn

rowptr = torch.tensor([0, 0, 0, 2, 2, 3, 5, 6, 6], dtype=torch.long, device='npu')
row = ops_gnn.ptr2ind(rowptr, 6)
# tensor([2, 2, 4, 5, 5, 6], device='npu:0')
```

---

### random_walk — NPU 随机游走

**函数签名：**

```python
def random_walk(
    row: Tensor,
    col: Tensor,
    start: Tensor,
    walk_length: int,
    p: float = 1,
    q: float = 1,
    coalesced: bool = True,
    num_nodes: Optional[int] = None,
    return_edge_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
```

**参数说明：**

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|-----------|------|
| row | Tensor | 输入 | COO 源节点，一维 `torch.int64` NPU Tensor |
| col | Tensor | 输入 | COO 目标节点，一维 `torch.int64` NPU Tensor，与 `row` 等长 |
| start | Tensor | 输入 | 游走起点，一维 `torch.int64` NPU Tensor |
| walk_length | int | 输入 | 游走步数，必须大于等于 0 |
| p | float | 输入 | node2vec 返回参数，必须有限且大于 0 |
| q | float | 输入 | node2vec BFS/DFS 参数，必须有限且大于 0 |
| coalesced | bool | 输入 | 为 True 时按 `(row, col)` 排序；为 False 时输入边须已按源节点分组 |
| num_nodes | Optional[int] | 输入 | 节点数；为 None 时由 `row`、`col`、`start` 的最大值推导 |
| return_edge_indices | bool | 输入 | 是否同时返回每一步的边位置 |

**返回值说明：**

返回 `node_seq`，其 dtype 为 `torch.int64`，shape 为 `[S, walk_length + 1]`。
当 `return_edge_indices=True` 时返回 `(node_seq, edge_seq)`，其中 `edge_seq` 的
dtype 为 `torch.int64`，shape 为 `[S, walk_length]`。孤立节点保持当前节点，
对应的 `edge_seq` 值为 `-1`。

**功能说明：**

从 COO 图上的多个起点执行均匀或 node2vec 偏置随机游走。Python 层在 NPU 上完成
COO 排序和 CSR 构造；Ascend 950 SIMT kernel 负责采样。当 `p=1, q=1` 时使用均匀
采样路径，否则按照 node2vec 转移概率执行拒绝采样。

**实现架构：**

- **Python 预处理**：参数校验、COO 排序、degree 统计及 CSR `rowptr` 构造
- **Host 层**：获取当前 PyTorch NPU stream 和 Philox seed/offset，计算 node2vec 整数门限
- **Kernel 模式**：Ascend C SIMT，每个线程负责一条 walk
- **输出路径**：`return_edge_indices=False` 时不写边序列

**注意事项：**

- `row`、`col`、`start` 必须位于同一 NPU 设备，节点编号满足 `0 <= index < num_nodes`
- `coalesced=False` 不会重新排列输入，边必须已经按源节点分组
- 排序开启时，edge index 对应排序后的 COO/CSR 位置
- 总边数及单节点出度不超过 `2^32-1`
- 仅支持前向计算；可使用 `torch.manual_seed(seed)` 固定 NPU 默认 Generator

**使用示例：**

```python
import torch
import ops_gnn

device = "npu:0"
row = torch.tensor([0, 1, 1, 2], dtype=torch.int64, device=device)
col = torch.tensor([1, 0, 2, 1], dtype=torch.int64, device=device)
start = torch.tensor([0, 2], dtype=torch.int64, device=device)

torch.manual_seed(202608)
nodes, edges = ops_gnn.random_walk(
    row, col, start, walk_length=8, p=0.5, q=2.0,
    return_edge_indices=True,
)
assert nodes.shape == (2, 9)
assert edges.shape == (2, 8)
```

---

### gather_csr - CSR 分段展开

**函数签名：**

```python
def gather_csr(
    src: Tensor,
    indptr: Tensor,
    out: Optional[Tensor] = None,
) -> Tensor:
```

`gather_csr` 是 `segment_csr` 的逆向展开操作。令 `dim = indptr.dim() - 1`，
对每个 segment `i`，将 `src[..., i, ...]` 复制到：

```text
out[..., indptr[i]:indptr[i + 1], ...] = src[..., i, ...]
```

接口仅包含 `src`、`indptr` 和可选 `out`，不包含 `reduce`、`dim` 或
`dim_size` 参数。

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | Tensor | NPU Tensor；支持 float16、bfloat16、float32、int8、int16、int32、uint8、float64、int64 |
| `indptr` | Tensor | 同一 NPU 上的 int64 CSR 指针，最后一维非降序 |
| `out` | Optional[Tensor] | 可选输出；dtype、device、rank 和推导 Shape 必须匹配 `src` |

支持前缀广播、空段、空 Tensor，以及非连续的 `src`、`indptr` 和 `out`。
提供 `out` 时函数写入并返回原 Tensor；只提供前向计算。

**支持范围：**

| 项目 | 支持范围 |
|------|----------|
| 硬件 | Ascend 950PR（`dav-3510`） |
| `src/out` dtype | float16、bfloat16、float32、int8、int16、int32、uint8、float64、int64 |
| `indptr` dtype | int64 |
| rank | `1 <= indptr.dim() <= src.dim()` |
| 布局 | 连续及非连续 Tensor |
| 特殊场景 | batch 广播、空段、空 Tensor、可选 `out` |
| 计算范围 | 仅前向 |

所有 dtype 均按原始字节复制，不进行数值计算或类型转换，因此结果与 CPU
参考实现逐位一致。float64 和 int64 属于 L2 功能路径，不参与性能验收。

**参数约束：**

- `src`、`indptr` 和可选 `out` 必须位于同一 NPU；
- `indptr.dim() <= src.dim()`，其前缀维可广播到 `src`；
- 当 `indptr.size(-1) > 0` 时，满足 `src.size(dim) == indptr.size(-1) - 1`；若 `indptr`
  的末维为空，则 `src.size(dim)` 必须为 `0`；
- 每行 `indptr` 非降序、值位于 `[0, endpoint]`，且 batch endpoint 相同；
- 未提供 `out` 时，输出的 `dim` 维长度为公共 endpoint；
- 提供 `out` 时，其 dtype、device、rank 和推导 Shape 必须匹配。

空 `src` 仍按 `indptr` endpoint 推导输出段维，并校验广播、单调性和范围；
`indptr` 为空时 endpoint 按 0 处理。

**实现说明：**

Host 层负责参数校验、`indptr` 广播、非连续 Tensor 规整、输出申请和 Tiling。
Ascend C Kernel 根据数据分布选择 `SegmentMajor` 或 `OutputMajor` 调度。
前者按 `(batch, segment)` 分核，后者在 segment 数过少或段长严重倾斜时按输出行
分核。完整且 32 字节对齐的 feature 使用 64 KB UB 重复缓冲区批量写出，其他
feature 按 16 KB tile 搬运。Kernel 使用当前 PyTorch NPU stream。

**使用示例：**

```python
import os
import torch
from ops_gnn import gather_csr

device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
torch.npu.set_device(device_id)
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device="npu")
indptr = torch.tensor([0, 2, 5], dtype=torch.int64, device="npu")
out = gather_csr(src, indptr)
# [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
```

**构建与测试：**

```bash
# CPU 参考实现要求 torch_scatter >= 2.1.0。
cmake -S . -B build/cmake_release \
  -DNPU_ARCH=dav-3510 -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake_release -j4
export PYTHONPATH=$PWD/python
export NPU_DEVICE_ID=<device_id>
pytest -q test/gather_csr/arch35/test_gather_csr.py
python test/gather_csr/arch35/golden.py
python test/gather_csr/arch35/benchmark_gather_csr.py --ascendoptest \
  --ascendoptest-root /path/to/AscendOpTest
python test/gather_csr/arch35/benchmark_gather_csr.py --warmup 20 --iterations 101
```

---

### scatter — Scatter 系列归约

`scatter` 系列接口与 `torch_scatter` 2.1.2 的前向语义保持一致：

```python
ops_gnn.scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum")
ops_gnn.scatter_sum(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_add(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_mul(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_mean(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_min(src, index, dim=-1, out=None, dim_size=None)
ops_gnn.scatter_max(src, index, dim=-1, out=None, dim_size=None)
```

`scatter_min` 和 `scatter_max` 返回 `(out, arg_out)`，其他接口返回
`out`；`scatter_add` 与 `scatter_sum` 使用相同的归约语义。

#### 参数

- `src`：输入 Tensor，支持 1 至 8 维。
- `index`：INT64 Tensor，可按 `torch_scatter` 规则广播到 `src`。
- `dim`：归约维度，支持负维度，默认为 `-1`。
- `out`：可选输出 Tensor；传入时在原对象上更新并保持对象身份。
- `dim_size`：可选输出归约维长度；省略时由 `index` 最大值推导。
- `reduce`：`sum`、`add`、`mul`、`mean`、`min` 或 `max`。

#### 数据类型与执行路径

| 级别 | `src/out` 数据类型 | 执行路径 | 支持的归约 |
|---|---|---|---|
| L1 | float16、bfloat16、float32、int8、int16、int32、uint8 | Ascend C NPU Kernel | 全部六种 |
| L2 | float64、int64 | 同步 CPU 回退并复制回原设备 | 全部六种 |

整数 `mean` 使用 floor 除法。`min/max` 出现相同最值时由后写入者胜出；
未写入位置的值为零，`arg_out` 为 `src.size(dim)`。接口支持显式
`dim_size`、空 Tensor、非连续 Tensor、无序或重复索引以及高冲突索引。
本接口仅提供前向计算。

#### 示例

```python
import os
import torch
import ops_gnn

torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))
src = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
index = torch.tensor([0, 1, 0, 1], dtype=torch.long, device="npu")

out = ops_gnn.scatter_sum(src, index)
# tensor([4., 6.], device='npu:0')

values, arg = ops_gnn.scatter_max(src, index)
# values: tensor([3., 4.], device='npu:0')
# arg:    tensor([2, 3], device='npu:0')
```

### spmm — CSR 通用聚合

```python
ops_gnn.spmm(
    indptr, indices, x=None, op="copy_lhs", reduce="sum",
    out=None, rhs=None
) -> Tensor
```

使用 ops-gnn 自有 CSR Tensor 接口执行消息生成和目标行聚合，不依赖
`DGLGraph`、`SparseMatrix` 或 DGL FFI。

- `indptr`：目标行 CSR 指针，一维 NPU `int32` 或 `int64` Tensor。
- `indices`：CSR 边对应的源节点索引，dtype 和设备与 `indptr` 相同。
- `x`：源节点特征。`copy_rhs` 为兼容简写，也可以通过这里传入 CSR 顺序的边特征。
- `rhs`：二元消息操作使用的 CSR 顺序边特征。
- `op`：`copy_lhs`、`copy_rhs`、`add`、`sub`、`mul` 或 `div`。
- `reduce`：`sum`、`max`、`min` 或 `mean`。
- `out`：可选输出；输入需要梯度时不允许指定 `out`。

`copy_lhs` 使用 `x[indices]`，`copy_rhs` 直接使用边特征；二元操作计算
`op(x[indices], rhs)`，支持 PyTorch 广播。输入可以是一维标量特征或二维特征。
空 CSR 行输出零。

```python
from ops_gnn import spmm

y = spmm(ptr, idx, node_features, reduce="sum")
y = spmm(ptr, idx, node_features, rhs=edge_features,
         op="mul", reduce="mean")
```

不需要梯度且形状、dtype 满足 Kernel 条件的 `copy_lhs/copy_rhs + sum/max/min`
优先使用迁移的 NPU Kernel；二元消息、`mean`、高维特征和需要梯度的调用使用
可求导的 NPU Tensor 组合实现。`copy_rhs` 仅支持 float32；其他路径支持 float16
和 float32。

### bspmm — 批量 CSR 通用聚合

```python
ops_gnn.bspmm(
    indptr, indices, x=None, op="copy_lhs", reduce="sum",
    out=None, rhs=None
) -> Tensor
```

`bspmm` 与 `spmm` 支持相同的消息操作、归约、广播和 autograd。节点或边特征中
至少一个必须为三维或更高维。二元操作在数量维之后插入值为 1 的维度以对齐
特征 rank，再执行 PyTorch 广播；输出保留广播后的全部尾部特征维。三维 FP16
`copy_lhs + sum/max/min` 使用迁移的 BSpMM Kernel，更高维及其他组合由 NPU
Tensor 运算完成。

```python
from ops_gnn import bspmm

# node_features: [K, 3, 1]，edge_features: [nnz, 2, 3, 4]
y = bspmm(ptr, idx, node_features, rhs=edge_features,
          op="mul", reduce="mean")
# y: [M, 2, 3, 4]
```


### 2.11 grid_cluster — 规则网格聚类

将 `[N,D]` 点坐标映射为规则网格的线性 ID；落在同一网格单元的点得到相同 ID。接口兼容 `torch_cluster.grid_cluster`。

**函数签名：**

```python
def grid_cluster(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...
```

**参数：**

| 参数 | 形状 | 说明 |
| --- | --- | --- |
| `pos` | `[N,D]` | NPU 上的点坐标；`N >= 0`、`D > 0`，支持 float16、bfloat16、float32、int8、uint8、int16、int32。 |
| `size` | `[D]` | 各维网格大小，所有元素必须为正有限值。 |
| `start` | `[D]` 或 `None` | 可选网格起点，默认取 `pos` 逐维最小值。 |
| `end` | `[D]` 或 `None` | 可选网格终点，默认取 `pos` 逐维最大值。 |

所有输入须在同一 NPU；提供的向量长度须等于 `D`。允许非连续输入；混合 dtype 按 PyTorch 类型提升规则计算。

**返回值：**同一 NPU 上形状为 `[N]` 的 `torch.int64` Tensor。空 `pos` 返回空 Tensor。

**示例：**

```python
import torch
import ops_gnn

pos = torch.tensor([[0., 0.], [0.4, 0.4], [1.2, 0.]], device="npu")
size = torch.tensor([1., 1.], device="npu")
cluster = ops_gnn.grid_cluster(pos, size)
# tensor([0, 0, 1], device='npu:0')
```

**语义与限制：**各维坐标由 `(pos - start) / size` 向零截断得到，再按第 0 维步长为 1 的顺序线性化。显式 `start/end` 不裁剪范围外的点。最终 ID 使用 `int64`，遵循 CPU 参考实现的溢出语义；不提供梯度计算。

---

### torch_sparse_spmm

为 torch_sparse COO/CSR 稀疏矩阵乘稠密矩阵提供 NPU 实现，支持 CSR
sum/add/mean/min/max、批量输入、可选边权，以及浮点 sum/mean 一阶梯度。
实现随 `ops_gnn` 统一编译和分发。

构建与测试见 [开发指南](development_guide.md#torch_sparse_spmm)，实现设计见 [设计文档](../experiment/design/torch_sparse_spmm/torch_sparse_spmm设计文档.md)。

#### 使用

在导入 torch_sparse/PyG 的前端别名前导入适配模块：

```python
import torch
import torch_npu
from ops_gnn import torch_sparse_spmm
from torch_sparse import spmm

torch.npu.set_device(0)
index = torch.tensor([[0, 0, 2], [1, 2, 0]], device="npu", dtype=torch.int64)
value = torch.tensor([0.5, 1.0, -1.0], device="npu", requires_grad=True)
dense = torch.randn(3, 16, device="npu", requires_grad=True)
out = spmm(index, value, 3, 3, dense)
out.square().sum().backward()
```

`SparseTensor.spmm/matmul` 的稠密输入路径和 `torch.ops.torch_sparse`
注册接口保持上游签名。`csr_spmm` 辅助接口可通过 `output_dtype` 选择混合输出：

```python
rowptr = torch.tensor([0, 2, 2, 3], device="npu", dtype=torch.int64)
col = index[1].contiguous()
out = torch_sparse_spmm.csr_spmm(rowptr, col, value, dense, reduce="mean")
with torch.no_grad():
    out, arg = torch_sparse_spmm.csr_spmm(rowptr, col, value, dense, reduce="max")
```

#### 接口约束

| 输入类型 | 输出类型 | 累加类型 |
| --- | --- | --- |
| FP32 | FP32 | FP32 |
| FP16 | FP16 或 FP32 | FP32 |
| BF16 | BF16 或 FP32 | FP32 |
| INT8 | INT32 或 FP32 | 对应 INT32 或 FP32 |

- 索引为同一 NPU 上的 int64，稠密输入为 `[..., N, K]`，输出为 `[..., M, K]`。
- 边权可省略，或为以 E 为首维、每条边 1 或 K 个权重的 Tensor；与稠密输入的类型、设备一致。
- COO 仅 sum，CSR 支持四种归约，add 等同 sum。空行输出 0，最值 arg 为 E。
- 最值忽略 NaN，平局保留第一条边；全 NaN 行返回 0/E。整数 mean 向零截断。
- FP16/BF16 使用 FP32 计算。浮点默认输出同输入，INT8 默认输出 INT32。
- sum/mean 支持浮点 value/mat 一阶梯度；min/max 梯度、二阶梯度及稀疏乘稀疏不支持。
- 输入合法性由 Python/Host 检查和设备端 CSR 验证共同保证；NPU 输入不回退 CPU 计算。
- 缓存仅复用索引拓扑，输入数值每次读取；原地 PyTorch 修改使缓存失效。
  无版本号的 inference Tensor 不缓存；外部裸指针写入后须清除缓存。
- `clear_validation_cache()` 清理缓存，`validation_cache_info()` 返回只读统计。
  `coo_to_csr` 用于独立转换计时，输入必须符合其文档约束。

### knn / knn_graph — 精确近邻搜索

本算子在 Ascend C SIMT 中执行精确候选点扫描与 Top-K，提供
`ops_gnn.knn` 和 Python 组合接口 `ops_gnn.knn_graph`。

#### 接口

```python
knn(x, y, k, batch_x=None, batch_y=None, cosine=False,
    num_workers=1, batch_size=None)
knn_graph(x, k, batch=None, loop=False, flow='source_to_target',
          cosine=False, num_workers=1, batch_size=None)
```

- `x[N,F]`、`y[M,F]`：同一 NPU、相同特征维和 dtype，支持 FP16/FP32；
  一维输入转为 `[N,1]`，非连续输入在调用中转为连续布局。
- 输出为 NPU int64 `[2,M*k]`，第一行是查询 y 的索引，第二行是候选 x
  的索引；若任一输入 `numel()==0`，输出 `[2,0]`。
- batch 为排序后的 int64 向量；多 batch 经 `bucketize` 生成指针，
  kernel 只扫描查询所在 batch，不允许跨 batch。
- `cosine=True` 在 NPU 上执行；零范数点对距离定义为 1。
- `knn_graph(loop=False)` 调用 `k+1` 后删除自环；支持两个 flow 方向。
  `num_workers` 保留但 NPU 路径忽略。

#### 当前边界与数值口径

要求 `1 <= k <= N`（查询所在 batch 的候选数）。k≤64 使用固定容量
局部状态，k>64 使用动态 Top-K 状态，不再施加参考 GPU 的 k≤100 上限。
当前 CPU 标杆定向回归覆盖 k=65/100/101/128/129/257、k=N 及图接口跨分派边界。
`loop=False` 的图接口须满足 `k+1 <= N`。
batch 必须 sorted；NaN/Inf 的排序行为不作为已验证接口承诺。

欧氏距离与 CPU torch_cluster 的 nanoflann L2_Adaptor 对齐：每四个
特征的平方差先按左结合求和，再加入距离累加器，尾部逐特征处理。FP16
在减法、平方及每个加法后显式舍入回 FP16，不是 FP32 累加。FP32
保留相同分组顺序，使用仓库默认编译参数。L2 等距点按
CPU 树查询顺序取舍，树元数据在 NPU 构建；cosine 使用 FP32 点积和
范数，等距按较小索引优先。

数值兼容目标为真实 CPU torch_cluster 1.6.3。按特征逐次 FP16 累加的
其他参考实现可能产生不同近邻集合，不能替代上述四项分组运算。

#### 实现组织

| 文件 | 职责 |
|---|---|
| `csrc/npu/knn/op_host/knn.cpp` | 参数检查、batch 指针校验、输出分配、当前 stream 启动 |
| `csrc/npu/knn/op_kernel/arch35/knn_kernel.cpp` | 查询分核、距离计算、稳定有序 Top-K |
| `csrc/npu/knn/op_kernel/arch35/knn_tree.h` | NPU 树分区、叶子次序和等距查询顺序；BSD 来源见[第三方来源](#knn-第三方来源) |
| `python/ops_gnn/knn.py`（仓库根目录相对路径） | 一维/连续化、bucketize、图接口组合 |

每 AIV 启动 256 个 SIMT 线程，每线程独立处理查询，串行扫描该 batch
全部候选。小 k 按 1/4/8/16/32/64 容量模板分派，F=3/16/32 且
容量不超过 32 时缓存查询值。大 k 分配 M×k FP32 距离状态，索引状态
直接存于输出 col；每个查询的存储切片互不重叠。不额外物化 M×N 候选
距离表；另分配 FP16 O(N)、FP32 O(N*F) 树元数据，计入完整 API 耗时。
每 batch 一个线程块。通常前 7 个 FIFO 节点由 256 线程协作；有效
建树维度不超过 3 时，batchN>=8192 用 31 节点，>=16384 用 63 节点。
FP16 建树只用第一维。其余不相交子树按前沿并行。8 KiB UB 用于块内
并行 min/max、整数归约和前缀计数；FP32 复用选中切分维已计算的范围，
临时交换索引复用尚未分配的节点空间，之后不再使用该临时列表。
k=N 时 Top-K 状态自然与全表同阶。没有跨调用答案缓存、CPU
距离计算或近似检索。

#### KNN 第三方来源

原作者版权及 MIT、BSD 许可全文统一保留在根目录 [LICENSE](../../LICENSE)，
中英文文档不重复嵌入许可全文。本仓新增部分的 [CANN 许可](../../LICENSE)
保持不变，不替代第三方许可条款。

##### Python 接口

`python/ops_gnn/knn.py` 的参数准备与图接口组合遵循 Matthias Fey 的
[pytorch_cluster 1.6.3 `torch_cluster/knn.py`](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/torch_cluster/knn.py)
（MIT）。改编范围为两个公共接口的参数名与默认值，以及 `_knn_impl`
中的空输入、一维/连续化、batch_size 推断和 bucketize 指针准备；
`_knn_graph_impl` 保留 k+1、去自环及 flow 语义。参数记录、
`Signature.bind` 和 NPU PyBind 派发是本仓适配。

未捆绑 torch_cluster 的 CPU/CUDA 检索源码或二进制。
测试使用 `test/knn/arch35/golden.py` 中的独立 PyTorch CPU 参考实现，
测试和 NPU 运行时都不需要安装 torch_cluster 包。

##### nanoflann 规则

树分区、遍历顺序和距离算术规则改编自
[pytorch_cluster 所附 nanoflann 头文件](https://github.com/rusty1s/pytorch_cluster/blob/e9a855c284b45edcbf0282cf70ac09bee0ce4e49/csrc/cpu/utils/nanoflann.hpp)，
原始提交为 `e9a855c284b45edcbf0282cf70ac09bee0ce4e49`，
路径为 `csrc/cpu/utils/nanoflann.hpp`（BSD，Jose Luis Blanco）。

| 本地文件和函数 | 上游对应规则 |
| --- | --- |
| `csrc/npu/knn/op_kernel/arch35/knn_tree.h` 的 `SelectSplit`、`SerialPartition`、`CooperativePartition` | `middleSplit_`、`planeSplit` 的选轴、切分阈值和双指针置换 |
| `csrc/npu/knn/op_kernel/arch35/knn_tree.h` 的建树函数和 `Earlier` | `divideTree`、`searchLevel` 的包围盒、叶子排列和访问顺序 |
| `csrc/npu/knn/op_kernel/arch35/knn_kernel.cpp` 的 `L2Distance`、`SquaredDifference` | `L2_Adaptor::evalMetric` 的四项分组算术 |
| `test/knn/arch35/golden.py` 的 `_TieOrder` | 独立实现的 CPU 分区和遍历规则，仅用于等距点顺序 |

上述为面向 NPU 的改编，未复制整个 nanoflann 头文件。NPU 内存布局、
协作归约/前缀、kernel 派发和流式 Top-K 在本实现中完成。
涉及第三方接口/规则的生产文件共三个：`knn.py`、`knn_tree.h` 和
`knn_kernel.cpp`，不是对整个项目作第三方归属声明。
不按 SCA 文本相似度推断改编范围；MIT 和 BSD 分别对应两个来源，不能
互相替代。涉及的源文件均保留原作者声明。

#### 构建与测试

实测环境：Ascend950PR (`dav-3510`)、CANN 9.1.0、PyTorch 2.7.1、
torch_npu 2.7.1.post8、Python 3.12.13。其他平台不记为已验证。
从仓库根目录、已配置的 Python 环境执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S . -B build/cmake_python -DWITH_PYTHON=ON \
      -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH=dav-3510
cmake --build build/cmake_python -j 4
export PYTHONPATH="$PWD/python"
NPU_DEVICE_ID=0 python -m pytest test/knn/arch35 -v --maxfail=1
python test/knn/arch35/benchmark_knn.py --warmup 20 --iters 100 \
       --json-output /path/to/new-run/benchmark.json
```

测试目录包含功能测试、接口契约、独立 CPU 标杆、CPU 标杆自测和性能脚本。
`golden.py` 每次处理至多 32 条查询，穷举全部候选点，保留 FP16/FP32
四项分组舍入和 L2 树遍历等距顺序；树元数据仅用于等距排序，不剪枝。
实现不导入 `ops_gnn`、`torch_npu` 或 `torch_cluster`。
标杆自测已合入唯一测试入口 `test_knn.py`。无 NPU 及其扩展时，只选择
这些自测并绕过共享的设备架构 conftest：

```bash
python -m pytest test/knn/arch35/test_knn.py \
       --confcutdir=test/knn/arch35 -k test_golden_ -q
```

功能测试严格比较全部近邻边的多重集合；余弦距离使用 CPU double 标杆。
性能脚本覆盖 32 个形状、两种 dtype，每项要求
`reference_ms / measured_ms >= 0.45`。`--quick` 只用于诊断。
请将报告输出到仓库外的独立目录；运行结果随 PR 提供，不存放在源码树中。

#### 调用示例

```python
import torch
import torch_npu
from ops_gnn import knn, knn_graph

torch.npu.set_device(0)
x = torch.randn(128, 3, device="npu", dtype=torch.float32)
y = torch.randn(32, 3, device="npu", dtype=torch.float32)
edge = knn(x, y, k=8)           # [2, 256]，row 为 y，col 为 x
graph = knn_graph(x, k=8)       # 无自环，source_to_target
torch.npu.synchronize()
```

## 三、测试指南

### 运行测试

```sh
# 运行当前机器芯片型号对应的所有测试（pytest test/ 自动只收集对应目录）
pytest test/ -v

# 运行单个算子测试（950 上为 arch35，A2/A3 上为 arch22）
pytest test/spmm_max/arch22/test_spmm_max.py -v
pytest test/spmm_min/arch22/test_spmm_min.py -v
pytest test/spmm_sum/arch22/test_spmm_sum.py -v
pytest test/unified_spmm/arch22/test_unified_spmm.py -v
pytest test/gather_csr/arch35/test_gather_csr.py -v
pytest test/segment_max_csr/arch35/test_segment_max_csr.py -v
pytest test/graclus_cluster/arch35/test_graclus_cluster.py -v
python -m pytest test/gather_coo/arch35/test_gather_coo.py -v
pytest test/random_walk/arch35/test_random_walk.py -v
pytest test/radius/arch35/test_radius.py -v
pytest test/sparse/arch35/test_sparse.py -v

# 运行 random_walk 性能测试
NPU_DEVICE_ID=<device_id> python test/random_walk/arch35/benchmark_random_walk.py

# 运行 SpMM 性能测试
NPU_DEVICE_ID=<device_id> python test/spmm_max/arch22/benchmark_spmm_max.py
NPU_DEVICE_ID=<device_id> python test/spmm_min/arch22/benchmark_spmm_min.py
NPU_DEVICE_ID=<device_id> python test/spmm_sum/arch22/benchmark_spmm_sum.py
# 三维 BSpMM 示例；max/min 脚本同样支持 --batches
NPU_DEVICE_ID=<device_id> python test/spmm_sum/arch22/benchmark_spmm_sum.py --batches 4
NPU_DEVICE_ID=<device_id> python test/unified_spmm/arch22/benchmark_unified_spmm.py
# 高维二元广播：[N,3,1] 与 [E,2,3,4] 聚合为 [M,2,3,4]
NPU_DEVICE_ID=<device_id> python test/unified_spmm/arch22/benchmark_unified_spmm.py \
  --op mul --reduce mean --lhs-shape 3,1 --rhs-shape 2,3,4

# 运行 radius 官方标杆性能测试
python test/radius/arch35/benchmark_radius.py

# 运行单个测试用例
pytest test/segment_max_csr/arch35/test_segment_max_csr.py::test_segment_max_csr_basic -v
```

### 测试编写模板

```python
import pytest
import torch
import ops_gnn

def test_my_operator():
    """测试基本功能"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)          # 1. 指定 NPU 设备
    torch.manual_seed(42)            # 2. 设置随机种子

    # 3. 创建 NPU Tensor
    src = torch.tensor([...], dtype=torch.float32, device=device)

    # 4. 调用算子
    result = ops_gnn.my_op(src)

    # 5. 验证：设备、形状、数值
    assert result.device.type == 'npu'
    assert result.shape == expected_shape
    assert torch.allclose(result, expected)
```

---

### segment_coo — 排序 COO 分段归约

`segment_coo(src, index, out=None, dim_size=None, reduce="sum")` 返回归约值。
`segment_sum_coo`、`segment_add_coo`、`segment_mean_coo` 参数相同但不接受
reduce；`segment_min_coo`、`segment_max_coo` 返回 `(values, indices)`。

index 为排序的非负 int64，归约维为 `index.dim()-1`，前部维度支持广播。
src rank 为 1–8，支持 float16/float32/bfloat16/int8/uint8/int32/int64。
支持非连续输入和 out；对齐暂存的结果写回原 out。提供 out 时由它确定
输出段数，否则使用 dim_size 或最大 index 加一。mul 不支持。

sum/add 覆盖 out；mean 使用 out 初值，min/max 与初值比较。无 out 的
空段返回零，有 out 时 mean/min/max 保留空段初值。极值平局选择较后位置；
空段索引为归约维长度；归约维长度为零时，极值索引为零。

```python
src = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device="npu")
index = torch.tensor([0, 0, 2], dtype=torch.int64, device="npu")
values = ops_gnn.segment_coo(src, index, dim_size=4, reduce="sum")
# [[4., 6.], [0., 0.], [5., 6.], [0., 0.]]
```

## 四、返回主文档

- **[返回 README](../../README.md)**
- **[查看开发指导](development_guide.md)**

## torch_sparse 工具接口

### non_diag_mask

```text
non_diag_mask(row, col, M, N, k) -> Tensor
```

`row`、`col` 为同一 NPU 上长度相同的一维 int64 COO 索引，M、N 为非负矩阵尺寸，k 为有符号对角偏移。允许非连续输入，Host 在原设备上生成连续副本。

每个元素须满足 `0 <= row[i] < M`、`0 <= col[i] < N`。任一维度为零时输入必须没有元素；索引无需排序，可重复。Host 即时检查元数据，索引内容在 NPU 上检查，因此越界错误可能在 `torch.npu.synchronize()` 等同步点才以设备异常暴露。同步前返回 Tensor 不代表非法输入成功；发生错误后不能继续使用结果或复用该上下文计时。

输出为 bool 张量，长度为 `E + num_diag`。当 `k >= 0` 时，`num_diag = max(0, min(M, N-k))`；否则为 `max(0, min(M+k, N))`。前 E 个元素为 `col-row != k`，尾部补 `num_diag` 个 False。

这是任务定义的公共掩码排列。`SparseTensor.set_diag/fill_diag` 的 NPU 适配器另外完成 COO 排序与 CSR 缓存重建，不将该排列直接当作上游有序插入的散布索引。CPU/CUDA 前端保持原实现。

### 可选 SparseTensor 前端的替换值

有权输入的 `values` 可为 None、Python 数值标量，或能广播至 `[num_diag, *value.shape[1:]]` 的 Tensor。Tensor 必须与原权重处于同一设备（包括设备编号）；多元素及空 Tensor 的 dtype 也必须相同。单元素 Tensor 保留 PyTorch 标量转换，包含零维及 `[1,1]` 等形状；不能表示的转换或不兼容广播由框架报错。类型、设备和 dtype 检查发生在 native mask 调用之前，空对角线也不绕过检查。

无权输入忽略传入 values，输出继续无权。对于有权输入，None 将各对角 feature 填为 1；这是既有 NPU 扩展行为，不声称与上游所有广播边界完全等价。底层 Tensor 计算可在不安装可选 SparseTensor 包时独立测试。


### torch_sparse_random_walk

```text
torch_sparse_random_walk(rowptr, col, start, walk_length) -> Tensor
torch_sparse_random_walk_with_rand(rowptr, col, start, walk_length, rand) -> Tensor
```

前三个参数为同一 NPU 上的一维 int64 张量，分别为 CSR 指针、列索引和起点。`walk_length` 非负；输出 int64 `[S, walk_length+1]`，首列为起点。每一步均匀选择当前节点的邻居；遇到合法零度节点则终止，该行后续元素为 -1。非法 CSR、越界索引或非法起点会报错，不按合法死端处理。

公共入口从 NPU 生成器取得随机状态，在设备上执行 Philox。`torch_sparse_random_walk_with_rand` 接受同设备 float32 `[S, walk_length]` 随机数，供共享随机输入的数值对照。随机值须有限且处于 `[0, 1)`；索引乘法的 float32 舍入可能得到 degree，此时限制为 degree-1。

相同 seed 不意味着不同后端的随机序列一致。单邻居图和共享 rand 的路径可做逐元素对照；公共多邻居接口的跨后端一致性需同时对齐 RNG 算法及消费顺序。


### 实现与错误处理

- Host 校验 dtype、维度、设备、尺寸和输出长度溢出；内核融合内容范围检查，输出与连续副本始终留在 NPU。
- 内核提交到调用者当前 NPU stream。元信息错误立即抛出；设备内容错误通过设备异常在同步点暴露。负例应在独立进程中运行并显式同步，不能将异常后的设备上下文继续用作后续计时环境。
- 内核分别位于 `csrc/npu/non_diag_mask` 和 `csrc/npu/torch_sparse_random_walk`，避免与已有 random_walk 内核和构建对象重名。
- 公开 Host/Kernel C++ 接口统一处于 `opsgnn`；工具类游走参数命名为 `TorchSparseRandomWalkLaunchParams`，与原 node2vec 算子的参数类型区分。Dispatcher 命名空间与 C++ 命名空间分别维护。
- 兼容前端位于 `python/ops_gnn/torch_sparse_utils.py`。dispatcher 注册保留 `torch_sparse` 和辅助 `ops_gnn` 接口，不提供 CPU 计算回退。
- 内核不申请算法 workspace；随机输入、输出、连续副本及 SparseTensor 前端临时张量仍占用内存，不能据此称全调用零额外内存。

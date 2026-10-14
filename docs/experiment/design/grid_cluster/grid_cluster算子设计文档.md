# grid_cluster 算子设计文档

## 修订记录

| 版本 | 日期 | 修改人 | 修改内容 |
| --- | --- | --- | --- |
| V0.1 | 2026-09-07 | flying_whj | 根据任务书和设计模板形成初稿 |
| V1.0 | 2026-09-07 | flying_whj | 完成核心实现，补充精度语义与专用 SIMT 路径 |
| V1.1 | 2026-09-08 | flying_whj | 清理失效引用，完善实现设计描述 |

# 需求背景（required）

## 需求来源

本设计针对 Ascend 950 系列设备，使用 Ascend C 实现与 `torch_cluster.grid_cluster` 兼容的 NPU 算子。

参考语义和目标位置：

- Python 接口：`torch_cluster/grid.py`；
- CPU 标杆：`torch_cluster/csrc/cpu/grid_cpu.cpp`；
- CUDA 参考：`torch_cluster/csrc/cuda/grid_cuda.cu`；
- 目标仓库：`cann/ops-gnn`；
- 代码目录：`csrc/npu/grid_cluster/`；
- 目标硬件：Ascend 950 系列，`arch35 / dav-3510`。

## 背景介绍

`grid_cluster` 将 N 个 D 维点映射到规则网格。每个点先计算各维网格坐标，再按混合进制线性化为一个 `int64` cluster ID。同一网格单元内的点得到相同 ID，常用于点云体素化、图构建前的空间分桶和邻域检索预处理。

点之间相互独立；单点内部需要依次处理 D 个维度，并严格保持 CPU 标杆的减法、真除法、向零截断和 `int64` 溢出语义。因此实现重点是：

1. 精确复现不同 dtype 的中间舍入和溢出；
2. 对任务书高频维度 D=3/16/32/64 建立专用并行路径；
3. 避免 Host 同步和多余中间 Tensor；
4. 保证全部性能用例的 `标杆耗时 / 实测耗时 >= 0.45`。

调用链如下：

```text
ops_gnn.grid_cluster
        │
        ▼
_pybind.grid_cluster
        │
        ▼
PyBind / NPU Host 实现
        │
        ▼
Ascend C arch35 SIMT Kernel
```

# 需求分析（required）

## 算子接口

### Python 函数签名

```python
def grid_cluster(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ...
```

### PyBind 与 C++ 入口

Python 调用 `_pybind.grid_cluster(pos, size, start, end)`；PyBind 绑定到 `opsgnn::grid_cluster_npu`。

### 参数说明

| 参数 | 输入/输出 | shape | dtype | 约束与说明 |
| --- | --- | --- | --- | --- |
| `pos` | 输入 | `[N,D]` | float16、bfloat16、float32、int8、uint8、int16、int32 | NPU Tensor，`N>=0`，`D>0` |
| `size` | 输入 | `[D]` | 支持与 `pos` 同类型；混合类型走 ATen 兼容路径 | 与 `pos` 同设备，元素为正有限值 |
| `start` | 可选输入 | `[D]` | 同上 | 缺省时在 NPU 上取 `pos.min(0)` |
| `end` | 可选输入 | `[D]` | 同上 | 缺省时在 NPU 上取 `pos.max(0)` |
| `cluster` | 输出 | `[N]` | int64 | 与 CPU 标杆 bit-wise 一致 |

## 功能要求拆解

1. 支持七种输入 dtype。
2. 支持 `start/end` 均缺省、仅指定一个、或同时指定。
3. 支持空输入，直接返回空 `torch.long` Tensor。
4. 支持非连续 `pos`，Host 侧转为 contiguous。
5. 使用调用者当前 NPU stream 异步执行。
6. 同 dtype 输入走融合 Ascend C Kernel；混合 dtype 遵循 ATen 类型提升规则。
7. 输出是确定性整数结果，与 CPU torch_cluster 标杆逐元素一致。
8. 任务规格覆盖 D=1～64；通用路径本身不设置固定 D 上限。
9. float16、float32、int16、int32 的全部任务性能数据点达到 0.45 门限。

# 详细设计（required）

## 算子分析

### 数学定义

对点 `i` 和维度 `d`：

\[
c_{i,d}=\operatorname{trunc}\left(\frac{pos_{i,d}-start_d}{size_d}\right)
\]

\[
n_d=\operatorname{trunc}\left(\frac{end_d-start_d}{size_d}\right)+1
\]

其中 `trunc` 表示向零截断。线性化 stride 为：

\[
stride_0=1,\qquad
stride_d=\prod_{k=0}^{d-1}n_k
\]

最终：

\[
cluster_i=\sum_{d=0}^{D-1}c_{i,d}\times stride_d
\]

CPU 等价伪代码：

```cpp
for (int64_t i = 0; i < N; ++i) {
    uint64_t result = 0;
    uint64_t stride = 1;
    for (int64_t d = 0; d < D; ++d) {
        T delta = dtype_sub(pos[i, d], start[d]);
        T extent = dtype_sub(end[d], start[d]);
        int64_t coord = trunc_to_long(dtype_div(delta, size[d]));
        int64_t count = trunc_to_long(dtype_div(extent, size[d])) + 1;
        result += uint64_t(coord) * stride;
        stride *= uint64_t(count);
    }
    output[i] = int64_t(result);
}
```

### 精度语义

- `long()` 是向零截断，不是 floor；负偏移必须按向零截断处理。
- 减法先在输入 dtype 中物化。int32 使用无符号减法后转回 int32，以获得定义明确的模 (2^{32}) 语义。
- float16/bfloat16 的除法结果在转 `int64` 前回写到原 dtype，复现低精度中间舍入。
- 除法使用 float32 商，并通过 FMA 余差做一次修正，降低整数边界附近少一个 voxel 的风险。
- stride 和累加使用 `uint64_t`，以定义明确的模 (2^{64}) 语义完成计算，再按位转换为 `int64_t`。

### dtype 映射

| PyTorch dtype | Kernel 类型 | `Round` | 处理方式 |
| --- | --- | ---: | --- |
| uint8 | `uint8_t` | false | dtype 减法后转 float32 求商 |
| int8 | `int8_t` | false | dtype 减法后转 float32 求商 |
| int16 | `int16_t` | false | dtype 减法后转 float32 求商 |
| int32 | `int32_t` | false | 无符号减法模拟 int32 wrap，再转 float32 |
| float16 | `half` | true | float32 辅助计算，商回写 half 后转 int64 |
| float32 | `float` | false | float32 计算后向零转 int64 |
| bfloat16 | `bfloat16_t` | true | float32 辅助计算，商回写 bf16 后转 int64 |

## 实现路径选择

- [x] Ascend C Kernel
- [ ] CATLASS 模板库
- [ ] ACLNN 封装作为主体实现

选择 Ascend C SIMT Kernel。原因是每个输出点独立、每点计算由多维标量索引和整数线性化构成，不属于矩阵乘法，也不需要跨点归约。Host 仅使用 ATen/ACLNN 完成缺省边界和混合 dtype 兼容处理。

## Host 侧设计

Host 入口：

```cpp
at::Tensor grid_cluster_npu(
    at::Tensor pos,
    at::Tensor size,
    std::optional<at::Tensor> start,
    std::optional<at::Tensor> end);
```

执行流程：

1. 校验 `pos.dim()==2` 且 `D>0`。
2. 校验 `pos` dtype 属于七种支持类型。
3. 校验 `size/start/end` 与 `pos` 同设备，且为长度 D 的一维向量。
4. 在当前 NPU 上检查全部 `size[d]` 为正有限值；即使 `N==0`，非法 `size` 也显式报错。
5. 创建 `[N]`、`int64` 输出；`N==0` 时直接返回。
6. 将输入转为 contiguous。
7. 缺省 `start/end` 时分别计算逐维 min/max。
8. 对 int8/uint8/int16，先无损提升为 int32 做 min/max，再转回输入 dtype，以绕过当前 950 环境 `aclnnMinDim/aclnnMaxDim` 对窄整数不支持的问题。
9. 若四个输入 dtype 相同，获取当前 NPU stream，调用 `GridCluster`。
10. 若元数据 dtype 不同，使用 `true_divide/cumprod/sum` 的 ATen 组合路径，保留 PyTorch 类型提升语义。

`size>0` 是公开接口契约。Host 使用 `.all().item<bool>()` 读取标量校验结果；这会引入 NPU 到 Host 的同步，性能需要重新测量。

## Kernel 侧设计

### 通用路径

`GridSimt<T,Round>` 使用 256 个线程。一个逻辑线程通过 grid-stride loop 处理多个点，并在运行时循环 D 维：

```text
tid  = blockIdx * threadNum + threadIdx
step = blockNum * threadNum
for i = tid; i < N; i += step:
    逐维计算 coord、count、stride 和 result
    out[i] = result
```

该路径负责非 D=3/16/32/64 的形状，也作为超大地址范围的安全回退。

### D=3 专用路径

`Grid3Simt` 使用 512 个线程，一个线程处理一个点。Kernel 在点循环外读取三维 `start/size`、计算倒数和线性化 stride，点循环内完全展开三维计算：

```text
cluster = c0 + n0*c1 + n0*n1*c2
```

当 `N*D <= UINT32_MAX` 时，`tid/step/i` 使用 uint32，减少三维大规模用例的地址计算开销；超出时入口回退到 int64 下标通用路径。cluster ID 和 stride 始终保持 64 位。

### D=16/32/64 宽维路径

一个 lane group 协作处理一个点：

| D | group 宽度 W | 每 lane 维数 | 线程数 |
| ---: | ---: | ---: | ---: |
| 16 | 2 | 8 | 512 |
| 32，float16/bfloat16 | 4 | 8 | 512 |
| 32，其他 dtype | 4 | 8 | 256 |
| 64 | 4 | 16 | 256 |

每个 lane 在点循环外缓存自己负责维度的 `origin/size/reciprocal/stride`，循环内计算局部 `uint64` 累加值，再通过 `WarpShflDownSync` 规约，由 lane 0 写出。

D=16 的整数输入若所有 `size[d]` 均为正的 2 的幂，使用倒数乘法快速路径。该路径仍先执行 CPU 兼容的 dtype 减法；在任务书 `size=8` 场景中可减少 FMA 修正开销。

D>16 时，如果前 16 维 radix 的前缀乘积在模 (2^{64}) 下已为 0，则后续 stride 恒为 0，后续维度对输出没有贡献。此时仅计算前 16 维；该优化不做 hash 或重编号，不改变 `int64` bit pattern。

### 路径分发

```text
N*D > UINT32_MAX ? 通用 int64 下标路径
D == 3          ? Grid3Simt<512>
D == 16         ? GridWide512Simt<16, W=2>
D == 32         ? Round ? GridWide512Simt<32,W=4>
                        : GridWideSimt<32,W=4>
D == 64         ? GridWideSimt<64,W=4>
其他             ? GridSimt<256>
```

## Tiling 策略

本算子没有独立 `TilingData`，也不使用传统的 GM→UB→GM tile。原因是各维只读取一个标量坐标并立刻参与索引计算，宽维路径的元数据常驻寄存器，直接 GM 标量访问和 SIMT 并行更适合当前实现。

### Block 级切分

Host 计算：

```text
needed_blocks = ceil(N / 256)
block_num = ResolveAivCoreNum(needed_blocks)
```

各路径使用 grid-stride loop；宽维路径按 `W` 个 lane 组成一个逻辑点。该分配确保每个点恰好处理一次，尾部由循环条件自然处理，无需额外尾核参数。

### 核内切分

| 路径 | 逻辑点映射 | 尾部处理 |
| --- | --- | --- |
| D=3 | 1 thread / point | `i < N` |
| D=16 | 2 lanes / point | `i < N`，lane 0 写回 |
| D=32/64 | 4 lanes / point | `i < N`，lane 0 写回 |
| 通用 | 1 thread / point | `i < N` |

### UB 级 Tiling

不适用。最终路径不建立 `TPipe/TQue/TBuf`，UB 使用量为 0。设计模板中的 32B UB 对齐、double buffer 和 UB 容量公式不适用于本实现。

## 数据流与内存管理

```text
GM(pos, size, start, end)
        │ SIMT 标量加载
        ▼
寄存器：delta / quotient / reciprocal / stride / partial sum
        │ 宽维路径使用 lane shuffle
        ▼
GM(cluster)
```

| 区域 | 大小 | 说明 |
| --- | ---: | --- |
| 输入 GM | 外部输入 | `pos/size/start/end` |
| 输出 GM | `N * 8` bytes | `int64` cluster ID |
| UB Queue/Buffer | 0 | 不使用 UB staging |
| 自定义 Workspace | 0 | Kernel launch 的 workspace 参数为 `nullptr` |

## Workspace 需求

无自定义 workspace。Host 只分配最终输出；缺省边界和混合 dtype 路径产生的 ATen 临时 Tensor 由 PyTorch 运行时管理，不属于自定义 Kernel workspace。

## Ascend C API/机制映射

| 步骤 | API/机制 | 用途 |
| --- | --- | --- |
| SIMT 调用 | `AscendC::Simt::VF_CALL` | 从 AIV Kernel 启动 Vector Function |
| 线程索引 | `GetBlockIdx/GetBlockNum/GetThreadIdx/GetThreadNum` | grid-stride 分工 |
| 除法修正 | `AscendC::Simt::Fma` | 余差修正，稳定整数边界 |
| lane 规约 | `WarpShflDownSync` | 宽维局部和规约 |
| 异步启动 | `<<<cores, nullptr, stream>>>` | 在当前 NPU stream 提交 |

## 确定性与并发安全

- 每个输出位置只由一个线程或 lane group 写入。
- 无原子操作、无跨 block 依赖、无核间同步。
- lane 规约顺序固定。
- 输出是整数 bit pattern；同一输入重复执行结果一致。

# 性能优化

## 关键优化点

1. D=3/16/32/64 编译期特化，消除动态维度循环。
2. D=3 与安全范围内的宽维路径使用 uint32 下标。
3. `1/size[d]` 在点循环外预计算并复用。
4. D=16 将一个点拆为 2 lane，D=32/64 拆为 4 lane，降低单线程工作量和寄存器压力。
5. D=16 整数 2 的幂边长走精确倒数乘法路径。
6. D>16 在 prefix 模 (2^{64}) 为 0 时跳过后续零贡献。
7. Host 正常快路径不回读设备数值，不引入同步。

## 性能特性

| 项目 | 结论 |
| --- | --- |
| 计算模式 | 标量索引计算，D 较大时计算和寄存器压力增加 |
| 访存模式 | `pos` 顺序读取，`size/start/end` 小向量重复复用，输出顺序写 |
| 并行性 | 点间高并行；宽维由 lane group 协作 |
| 主要风险 | 小规模启动开销、大 D 寄存器/栈压力、共享设备性能波动 |

# 支持硬件

| 芯片 | 架构 | 状态 |
| --- | --- | --- |
| Ascend950PR | `arch35 / dav-3510` | 实机编译、精度和性能验证通过 |
| 其他芯片 | — | 未验证 |

验证环境：CANN 9.1.0、Python 3.12、PyTorch 2.7.1、torch_npu 2.7.1.post10。

# 算子约束限制

1. `pos` 必须是 `[N,D]`，`D>0`。
2. `size/start/end` 必须与 `pos` 同设备且为 `[D]`。
3. `size[d]` 必须为正有限值；零、负值及 NaN 显式报错，正无穷大仍属于不支持的输入。
4. `start/end` 只定义网格原点和范围，不裁剪 `pos`。
5. 快速自定义 Kernel 要求四个输入 dtype 相同；混合 dtype 走 ATen 路径。
6. 高维网格总数超过 int64 时，CPU 标杆会出现模 (2^{64}) 溢出；本实现保留其 bit-wise 结果，不能同时保证输出非负和 CPU bit-wise 一致。
7. 任务测试覆盖 D=1～64；通用 Kernel 可接受更大 D，但不作为当前验收承诺。
8. 大规模性能测试仅抽样 4096 行校验输出；全量逐元素精度由小中规模功能测试覆盖。

# 可维可测分析

## 精度标准

输出 `int64` cluster ID 与 CPU torch_cluster 标杆逐元素 `torch.equal`，即 bit-wise 一致。覆盖：

- 七种真实 NPU dtype；
- D=1/2/3/16/32/64/65；
- 四种可选边界组合；
- 空输入、相同点、非连续输入；
- nextafter 边界、负偏移向零截断；
- 整数减法溢出、高维 stride 溢出；
- D=16 幂边长和非零高维 prefix；
- 当前 stream、确定性和参数校验。

## 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| FP16/BF16 边界舍入不同 | ID 偏差 | 商转 long 前回写输入 dtype，增加 nextafter 用例 |
| int32 减法溢出 | C++ 有符号溢出未定义 | uint32 减法后转回 int32 |
| int64 stride 溢出 | 未定义行为或结果不一致 | uint64 模 (2^{64}) 运算 |
| 窄整数 min/max API 不支持 | 缺省边界失败 | int32 无损归约后转回 |
| 大 D 单线程压力 | 性能回退 | D=16/32/64 lane group 协作 |
| 共享服务器外部负载 | 性能波动 | 测试前后检查 `npu-smi`，保留原始 CSV 和日志 |

# 代码结构

```text
ops-gnn/
├── csrc/npu/grid_cluster/
│   ├── op_host/
│   │   ├── grid_cluster.cpp
│   │   └── grid_cluster.h
│   └── op_kernel/arch35/
│       ├── grid_cluster.cpp
│       └── grid_cluster.h
├── python/ops_gnn/
│   ├── __init__.py
│   └── grid_cluster.py
├── docs/experiment/design/grid_cluster/
│   └── grid_cluster算子设计文档.md
└── test/grid_cluster/arch35/
    ├── grid_cluster_golden.py
    ├── test_grid_cluster.py
    └── benchmark_grid_cluster.py
```

# 参考资料

- torch_cluster CPU/CUDA/Python 实现；
- 社区任务书 `grid_cluster_task_doc.md`；
- 社区算子设计文档模板；
- Ascend C 索引类 Tiling 与通用 Tiling 设计原则；

# ops-gnn 开发指导

本文档提供 ops-gnn 库的开发环境构建、算子开发指南和测试说明。

## 一、环境构建

### 安装 CANN 软件

#### 安装前准备

离线安装时，请单击[获取链接](https://www.hiascend.com/developer/download/community/result?module=cann)下载 CANN 软件包，并上传到安装环境任意路径。

#### 安装 CANN

请使用 CANN 9.1.0 及以上版本，并配套 HDK（驱动/固件）25.7.rc1 及以上版本，其他版本暂不支持。

```sh
chmod +x Ascend-cann-toolkit_${VERSION}_linux-$(arch).run
./Ascend-cann-toolkit_${VERSION}_linux-$(arch).run --install
```

其中 `${VERSION}` 表示对应的 CANN 版本（如 9.1.0），`$(arch)` 表示 CPU 架构。

#### 安装后配置

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

如果 CANN 安装在自定义路径，请执行：

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

### CANN 详细安装指南

开发者可访问[昇腾文档-昇腾社区](https://www.hiascend.com/document) → CANN 社区版 → 软件安装，查看 CANN 软件安装引导，根据机器环境、操作系统和业务场景选择后阅读详细安装步骤。

### 依赖安装

ops-gnn 依赖以下组件：

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.9+ | |
| CMake | 3.18+ | C++ 构建系统 |
| GCC | 7.0+ | C++17 编译器 |
| PyTorch | 2.7+ | 需从昇腾社区获取 NPU 适配版本 |
| torch_npu | 与 PyTorch、CANN 匹配 | PyTorch NPU 设备后端 |
| CANN Toolkit | 9.1.0+ | AscendC 编译器、Bisheng 编译器、运行时库 |
| HDK（驱动/固件） | 25.7.rc1+ | Ascend 硬件驱动和固件 |

`torch` 和 `torch_npu` 需从[昇腾社区下载页面](https://www.hiascend.com/developer/download/community)获取适配 Ascend 硬件的版本。

支持平台为 Ascend 950（arch35）与 A2/A3（910B/910C，arch22），其他平台不支持。

glibc 2.41 版本暂不支持。

```sh
# 系统工具链
sudo apt install build-essential cmake

# 测试依赖（可选）
pip install pytest pytest-cov
```

### 安装 ops-gnn

```sh
# 方式一：pip 开发模式安装
python3 -m pip install --no-build-isolation --no-deps -e .

# 方式二：构建脚本安装
cd scripts
./build.sh python
cd ..

# 方式三：CMake 手动编译
mkdir build_cmake && cd build_cmake
cmake ..
cmake --build .
cd ..
```

## 二、项目架构

### 分层架构设计

ops-gnn 采用 PyTorch 扩展 + AscendC 内核的分层架构：

```text
┌─────────────────────────────────────────────┐
│              Python 接口层                   │
│   python/ops_gnn/<op>.py                    │
│   (类型注解、参数预处理、文档)                 │
├─────────────────────────────────────────────┤
│              PyTorch 绑定层                  │
│   csrc/pybind.cpp                           │
│   (PYBIND11_MODULE 注册、参数映射)            │
├─────────────────────────────────────────────┤
│              Host 端算子层                   │
│   csrc/npu/<op>/op_host/<op>.cpp/.h            │
│   (参数解析、Tiling 计算、Kernel 启动、流管理)  │
├─────────────────────────────────────────────┤
│              AscendC Kernel 层              │
│   csrc/npu/<op>/op_kernel/<arch>/<op>.cpp/.h          │
│   (AscendC SIMT/VF 编程、向量计算、数据搬移)    │
└─────────────────────────────────────────────┘
```

**架构说明：**

- **Python 接口层**：提供用户友好的 Python API，包含类型注解、docstring 和使用示例
- **PyTorch 绑定层**：通过 pybind11 将 C++ 函数暴露给 Python，处理 Tensor 参数传递
- **Host 端算子层**：运行在主机 CPU，负责参数校验、Tiling 数据计算、AscendC Kernel 启动和 Stream 同步
- **AscendC Kernel 层**：运行在 NPU 设备端（AIV 核），实现核心计算逻辑

### 目录结构

```text
ops-gnn
├── csrc/                           # C++/AscendC 源码目录
│   ├── pybind.cpp                  # PyTorch 绑定代码
│   └── npu/                        # NPU 相关代码（按算子分类）
├── docs/                           # 文档目录（API 说明见 docs/*/api_reference.md）
├── python/                         # Python 源码目录
│   └── ops_gnn/                    # Python 包目录
├── test/                           # 测试目录（按 <算子>/<架构> 组织）
├── scripts/                        # 构建脚本目录
│   └── build.sh                    # 统一构建/测试脚本（python/cpp/all/test）
├── cmake/                          # CMake 配置
│   └── OpsGNNConfig.cmake.in       # CMake 包配置模板
├── CMakeLists.txt                  # CMake 构建配置
├── setup.py                        # Python 安装脚本（setuptools + cmake）
├── setup.cfg                       # setuptools 配置
├── pyproject.toml                  # 现代 Python 项目配置
├── MANIFEST.in                     # 打包清单
└── LICENSE                         # CANN 许可证
```

`csrc/npu` 下各算子目录包含 `op_host` 和 `op_kernel/<arch>`；测试按 `test/<算子>/<arch>` 组织，各架构子目录包含 `golden.py`、功能测试文件和性能测试文件。其中，`<arch>` 表示目标架构对应的目录名。默认编译 `arch22` 和 `arch35`；测试只处理当前机器芯片型号对应的目录：950 → `arch35`，A2(910B)/A3(910C) → `arch22`。

### 核心文件说明

| 文件 | 功能说明 |
|------|---------|
| `csrc/pybind.cpp` | PyTorch 绑定入口，通过 `PYBIND11_MODULE` 注册所有 C++ 算子到 Python |
| `csrc/npu/<op>/op_host/<op>.h` | Host 端算子接口声明，定义函数签名 |
| `csrc/npu/<op>/op_host/<op>.cpp` | Host 端算子实现：Tensor 维度解析、Tiling 参数计算、dtype 分发、Stream 管理 |
| `csrc/npu/<op>/op_kernel/<arch>/<op>.h` | Kernel Launch 函数声明（Host 端调用入口） |
| `csrc/npu/<op>/op_kernel/<arch>/<op>.cpp` | Kernel Launch 实现 + 模板显式实例化 + `<<<>>>` 启动语法 |
| `csrc/npu/<op>/op_kernel/<arch>/<op>_kernel.h` | AscendC Kernel 核心类实现（Init → Process → Compute 流水线） |
| `csrc/npu/<op>/op_kernel/<arch>/<op>_tiling.h` | Tiling 数据结构定义（传递给 Device 端的参数结构体） |
| `python/ops_gnn/<op>.py` | Python 接口封装：类型注解、参数默认值处理、调用 `_pybind.<op>` |
| `python/ops_gnn/__init__.py` | 包入口，从各模块导入并注册到 `__all__` |
| `python/ops_gnn/typing.py` | 类型别名（`Tensor`, `OptTensor`） |

## 三、代码规范

### 命名规范

**C++ 层：**

- 文件名：小写下划线分隔，如 `segment_max_csr_kernel.h`
- 函数名：大驼峰，如 `LaunchSegmentMaxCsrKernel`
- 结构体名：大驼峰，如 `SegmentMaxCsrTilingData`
- 类名：大驼峰，如 `SegmentMaxCsrKernel`
- 模板参数：大驼峰或单字母大写，如 `typename T`

**Python 层：**

- 文件名：小写下划线分隔，如 `segment_max_csr.py`
- 函数名：小写下划线分隔，如 `segment_max_csr`
- 与 PyTorch 风格保持一致

### 代码风格

- C++ 遵循 C++17 标准，使用 `#pragma once` 头文件保护
- AscendC Kernel 使用 `__aicore__`、`__global__`、`__gm__` 等修饰符
- 使用 namespace `AscendC` 下的 API
- Python 接口使用类型注解（`Tensor`, `OptTensor`, `Optional[Tensor]`）
- 每个文件顶部包含 CANN Open Software License 版权声明

### 文件组织规范

每个算子严格遵循以下文件拆分：

```text
csrc/npu/<op>/
├── op_host/
│   ├── <op>.h          # Host 接口声明
│   └── <op>.cpp        # Host 实现
└── op_kernel/
    └── <arch>/
        ├── <op>.h               # Kernel Launch 声明
        ├── <op>.cpp             # Kernel Launch 实现 + 模板实例化
        ├── <op>_kernel.h        # Kernel 核心类（简单算子可合并到 <op>.cpp）
        └── <op>_tiling.h        # Tiling 结构体
```

## 四、构建系统

### CMake 构建

CMakeLists.txt 负责编译 AscendC Kernel 和 PyTorch 绑定库：

1. 使用 bisheng 编译器将 Kernel 源文件编译为 `libopsgnn_npu_kernel.so`
2. 将 `csrc/pybind.cpp` + host 端代码编译为 `_pybind.so`
3. 产物安装到 `output/kernel/`

关键 CMake 变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NPU_ARCH` | NPU 架构 | 默认构建两种架构；设置 `dav-3510` 仅构建 950，设置 `dav-2201` 构建 A2/A3 |
| `WITH_PYTHON` | 是否编译 Python 绑定 | `ON` |
| `ASCEND_HOME_PATH` | CANN 安装路径 | 从环境变量读取 |
| `Python3_ROOT_DIR` | Python 根目录 | build.sh 自动导出 |

### setuptools 构建

`setup.py` 调用 CMake 编译并生成 wheel 包。`build_with_cmake()` 构建前会自动清理旧 `CMakeCache.txt`，避免 pip 临时目录导致路径不一致。

## 五、测试指南

### 运行功能测试

```sh
# 运行当前机器芯片型号对应的所有功能测试（950→arch35，A2/A3→arch22）
pytest test/ -v
# 或直接指定目录
pytest test/spmm_max/arch22/test_spmm_max.py -v                       # arch22 单个算子测试
pytest test/segment_max_csr/arch35/test_segment_max_csr.py -v             # 单个算子测试
pytest test/segment_max_csr/arch35/test_segment_max_csr.py::test_func -v  # 单个测试用例
```

> 说明：`pytest test/` 会依据本地芯片型号只收集对应目录 —— 950 上只跑 `arch35`（`arch22` 被忽略），A2/A3 上只跑 `arch22`（`arch35` 被忽略）。

### 运行性能测试

```sh
NPU_DEVICE_ID=<device_id> python test/<op>/<arch>/benchmark_<op>.py
```

其中，`<arch>` 按芯片型号取 `arch35`（950）或 `arch22`（A2/A3），`<op>` 表示算子名，`<device_id>` 表示设备 ID。

### 功能测试编写模式

1. `torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))` 指定 NPU 设备
2. `torch.manual_seed(42)` 保证可复现
3. 用 `.npu()` 方法创建 NPU Tensor（如 `torch.randint(...).npu()`）
4. 调用 `ops_gnn.<op>(...)` 
5. 验证结果：设备类型、形状、数值正确性

### 功能测试覆盖要求

| 场景 | 说明 |
|------|------|
| 基本功能 | 典型输入 |
| 不同 dtype | float32, float16, int32, int16 |
| 边界情况 | 空 segment、单元素 |
| 可选参数 | optional_out |
| 多维输入 | 2D/3D Tensor |
| 广播场景 | indptr 广播 |

## 六、算子开发实例

本节以 `segment_max_csr` 为例，说明开发新算子的流程。

### 调用链分析

```text
用户代码
    └─→ ops_gnn.segment_max_csr(src, indptr, optional_out)
            │  python/ops_gnn/segment_max_csr.py
            │  （旧接口参数默认值处理；Gather COO 的 None 由 pybind 显式传递）
            └─→ _pybind.segment_max_csr(src, indptr, optional_out)
                    │  csrc/pybind.cpp
                    │  (PYBIND11_MODULE 注册)
                    └─→ segment_max_csr()  [Host]
                            │  csrc/npu/segment_max_csr/op_host/segment_max_csr.cpp
                            │  (维度解析、Tiling填充、dtype分发、Stream管理)
                            └─→ LaunchSegmentMaxCsrKernel<T>()
                                    │  csrc/npu/segment_max_csr/op_kernel/<arch>/segment_max_csr.cpp
                                    │  (获取AIV核数、<<<>>>启动)
                                    └─→ segment_max_csr_kernel<T>  [Device]
                                            │  segment_max_csr_kernel.h
                                            │  (Init → Process → Compute 流水线)
```

每层职责：

- **Python**（`python/ops_gnn/<op>.py`）：类型注解、可选参数默认值、调用 `_pybind.<op>`
- **PyBind**（`csrc/pybind.cpp`）：`m.def("<op>", &func, py::arg(...), ...)` 注册 C++ 函数
- **Host**（`csrc/npu/<op>/op_host/`）：Tensor 维度解析 → Tiling 参数计算 → `torch::empty` 创建输出 → 按 `scalar_type` 分发模板 Launch。旧算子可以自行管理 Stream；Gather COO 必须复用 PyTorch 当前 NPU stream，不创建、同步或销毁私有 ACL Stream。
- **Kernel Launch**（`csrc/npu/<op>/op_kernel/<arch>/<op>.cpp`）：`GetCoreNumAiv()` 获取核数 → `<<<coreNum, nullptr, stream>>>` 启动 → 显式模板实例化
- **Kernel 实现**（`<op>_kernel.h`）：`Init()` 解析 Tiling + 分配 Buffer/Event → `Process()` 按 Block 分配工作范围 → `Compute()` 双缓冲流水线（DataCopy → Max/Add → DataCopy）

### 新增算子文件清单

以 `segment_max_csr` 为模板，每个新算子需创建：

```text
csrc/npu/<op>/op_host/
├── <op>.h                   # torch::Tensor <op>(torch::Tensor ...);
└── <op>.cpp                 # Host 实现

csrc/npu/<op>/op_kernel/<arch>/
├── <op>.h                   # template<typename T> void Launch<Op>Kernel(...);
├── <op>.cpp                 # Launch 实现 + 模板实例化
├── <op>_kernel.h            # Kernel 核心类 (Init/Process/Compute)
└── <op>_tiling.h            # struct <Op>TilingData { uint32_t/uint64_t ... };

python/ops_gnn/
└── <op>.py                  # Python 接口

test/<op>/<arch>/
├── golden.py                # CPU/参考实现
├── test_<op>.py             # 功能测试
└── benchmark_<op>.py        # 性能测试
```

其中 `<arch>` 按芯片型号取 `arch35`（950）或 `arch22`（A2/A3）。

此外需修改两个文件：

- `csrc/pybind.cpp`：`#include "<op>/op_host/<op>.h"` + `m.def("<op>", ...)`
- `python/ops_gnn/__init__.py`：`from .<op> import <op>` + 加入 `__all__`

简单算子可合并文件：无 Tiling 时省略 `_tiling.h`，逻辑简单时 `<op>_kernel.h` 可合并到 `<op>.cpp`（参考 `ptr2ind`）。

### 两种 Kernel 模式

| 特性 | ptr2ind | segment_max_csr |
|------|-----------|----------------|
| 文件数 | 2个（.h + .cpp） | launch + kernel + tiling（4文件） |
| 编程模型 | SIMT (`__simt_vf__` + `VF_CALL`) | Kernel类 (TPipe + Buffer + Event) |
| 数据搬移 | 直接 GM 读写 | DataCopy + 双缓冲流水线 |
| Tiling | 无（标量参数直传） | Tiling 结构体 |
| 适用 | 逐元素操作 | 规约、分段、多级流水线 |

### 开发流程总结

1. 参照 `segment_max_csr` 创建目录结构和文件
2. 定义 Tiling 结构体（不含指针；形状、长度和地址相关字段按需使用 64-bit）
3. 实现 Kernel 核心类（Init → Process → Compute），用 TPipe + Event 做双缓冲流水线
4. 实现 Launch 函数，`<<<>>>` 用 `#pragma GCC diagnostic` 包围
5. Host 端：维度解析 → Tiling 填充 → `torch::empty` → dtype 分发 Launch；需要异步语义的算子复用当前 PyTorch NPU stream，不在算子内部创建或同步私有 Stream
6. 在 `pybind.cpp` 中 `m.def(...)` 注册
7. 编写 Python 接口（类型注解 + docstring + 默认值处理）
8. 更新 `__init__.py` 导出
9. 编写测试（至少覆盖基本功能、多个 dtype、边界情况）

### segment_csr 文件职责

Host 校验 Tensor 元数据，并在当前 NPU stream 上选择计算路径。
Kernel 入口为 `SegmentCsr` 和 `SegmentCsrVector`，Python 继续导出
`segment_*_csr` 算子族。Kernel 文件位于 `csrc/npu/segment_csr/op_kernel/arch35/`：

| 文件 | 职责 |
| --- | --- |
| `segment_csr.h` | 枚举、dtype 分发、向量路径条件和入口声明 |
| `segment_csr.cpp` | 通用 SIMT 归约及 packed 访存 |
| `segment_csr_vec_kernel.cpp` | AIV 搬运、同步和寄存器路径分发 |
| `segment_csr_extrema.h` | FP16/INT64 极值及首次出现索引 |
| `segment_csr_sum.h` | sum/mean 计算及各 dtype 的数值行为 |
| `segment_csr_uniform.h` | 快路径启用前逐项检查所有分段指针 |

主头文件和主翻译单元按算子名命名，与同仓算子及产品评审要求对齐。
SIMT 和 AIV 两种执行模型继续分别编译；本轮目录整改不改变计算与分发逻辑。

产品测试采用 `test/segment_csr/arch35/`，只保留三个文件：

| 文件 | 职责 |
| --- | --- |
| `test_segment_csr.py` | 将 API、跨调用语义和向量路径用例集中在一个参数化测试套件 |
| `golden.py` | 纯 PyTorch CPU 值与首次极值索引参考 |
| `benchmark_segment_csr.py` | 独立的 29 组形状计时，输入和日志辅助函数内聚于该文件 |

合并保留所有原测试函数、断言和参数组合。扩展测试与向量回归的 dtype
常量分别命名，保留各自不同的支持范围。非连续输入继续在 NPU 上构造
并断言 stride，内部格式告警继续按失败处理。

任务专用基线表、验收报告编排器和 profiler 脚本放在 PR 验证附件的
`validation_inputs/` 中。完整的 464 项正确性与性能门禁独立于日常 pytest
收集。解压附件后，在产品仓根目录执行：

```bash
PYTHONPATH=python python -m pytest test/segment_csr/arch35 test/segment_max_csr -q
CSR_PRODUCT_ROOT="$PWD" PYTHONPATH=python python /path/to/evidence/validation_inputs/run_segment_csr_acceptance.py --all --warmup 20 --iter 100
```

外部验收脚本仅通过 `CSR_PRODUCT_ROOT` 定位产品源码和二进制；输入、参考
计算、计时方式及 0.45 门槛保持原样。Profiler 独立运行，用于诊断，不能替代接口计时。


### segment_coo 验证入口

功能用例集中在一个文件，参考和性能工具分别独立；验证环境需安装真实的
CPU torch_scatter 2.1.2（选择与当前 Python/PyTorch 匹配的 CPU wheel）。
测试不会用跳过或 mock 代替该参考。执行前加载 CANN 环境并构建扩展：

```bash
NPU_ARCH=dav-3510 python setup.py build_ext --inplace
PYTHONPATH=python python -m pytest test/segment_coo/arch35 -q -ra
for reduce in sum mean min max; do
  PYTHONPATH=python python test/segment_coo/arch35/benchmark_segment_coo.py \
    --reduce "$reduce" --warmup 20 --iter 100
done
```

### torch_sparse_spmm

接口约束和示例见 [API 文档](api_reference.md#torch_sparse_spmm)。

#### 文件职责

| 文件 | 职责 |
| --- | --- |
| `python/ops_gnn/torch_sparse_spmm.py` | torch_sparse 适配、输入检查、拓扑缓存与一阶梯度 |
| `csrc/npu/torch_sparse_spmm/op_host/torch_sparse_spmm.cpp` | ATen 接口、参数校验及当前 NPU stream 上的分发 |
| `csrc/npu/torch_sparse_spmm/op_kernel/arch35/torch_sparse_spmm.h` / `.cpp` | `TorchSparseSpmm` 及 ValueGrad、Validate、Convert、PermuteValues 辅助入口声明与实现 |
| `torch_sparse_spmm_kernel.h` / `torch_sparse_spmm_tiling.h` | 计算模板和形状参数 |
| `csrc/pybind.cpp` | 六个中央绑定，复用 ops_gnn 包 |

入口采用同仓 `SegmentCsr`、`SegmentCoo` 的算子名称格式。辅助入口使用 `TorchSparseSpmm` 前缀。

#### 构建与依赖

使用 Ascend 950（dav-3510）、CANN 9.1.0 及匹配的 PyTorch/torch_npu。
需要安装与 PyTorch 匹配的 torch_sparse、torch_scatter；PyG 测试另需
torch_geometric。Python 接口兼容仓库的 Python 3.9+ 范围。

在 ops-gnn 仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CANN_TARGET=950 NPU_ARCH=dav-3510 MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps -e .
```

更新原生代码后重新构建。`dav-2201` 构建不提供此模块。
Host、Kernel 通过仓库 CMake 的架构目录扫描编译；绑定集中在
`csrc/pybind.cpp`，不创建独立扩展包。

#### 测试与性能测试

在仓库根目录执行，测试要求安装本模块和 NPU 依赖：

```bash
python -m pytest -q test/torch_sparse_spmm/arch35/test_torch_sparse_spmm.py
python test/torch_sparse_spmm/arch35/benchmark_torch_sparse_spmm.py --reduce sum --iter 100 --warmup 20
python test/torch_sparse_spmm/arch35/benchmark_torch_sparse_spmm.py --format coo --dtype fp32
```

测试目录仅包含 benchmark、golden 和功能测试三个文件。功能测试覆盖七种
精度组合、四种归约、梯度、参数防护、缓存失效和 PyG GCN/GIN。
PyG 依赖缺失时仅跳过 PyG 用例，其余功能测试继续执行。
benchmark 在脚本内生成输入，报告同步后的平均调用耗时（毫秒）。
COO 测量预热后的调用，包含边权重排和 SpMM；首次转换不计入该平均值。
任务验收用例、ATK 和 profiler 工具保存在 PR 的独立验证附件中。

## 七、更多资源

- **[返回 README](../../README.md)**
- **[AscendC 编程指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/developerguide/ascendcdev/ascendcdev_0001.html)**
- **[PyTorch C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)**
- **[CANN 社区版文档](https://www.hiascend.com/document)**

## torch_sparse 工具接口

### 构建与启用

本模块面向 Ascend 950（`dav-3510`），使用仓库统一的 CMake 构建、`ops_gnn._pybind` 扩展和 `libopsgnn_npu_kernel.so`。在已加载 CANN 环境且安装匹配的 PyTorch、torch_npu 后，从仓库根目录安装：

```bash
NPU_ARCH=dav-3510 python3 -m pip install . --no-build-isolation
```

安装 `torch_sparse` 可启用其 `SparseTensor` 前端；导入 ops_gnn 安装 NPU 实现：

```python
import torch
import torch_npu
import ops_gnn

row = torch.tensor([0, 0, 1], dtype=torch.int64, device="npu")
col = torch.tensor([0, 1, 1], dtype=torch.int64, device="npu")
mask = ops_gnn.non_diag_mask(row, col, 2, 2, 0)
```

没有独立的 `_C.so` 或额外构建脚本。工具算子的随机游走合同与 `ops_gnn.random_walk` 的 node2vec/p/q 接口分别维护；后者的现有接口不变。950 以外架构不提供本模块的内核。


### 测试

`test/torch_sparse_utils/arch35/` 仅保留功能、精度测试 `test_torch_sparse_utils.py`、`test_functional.py` 和独立 CPU 参考 `golden.py`。用例覆盖关键精度、元数据拦截、死端、共享随机数游走、原生 RNG 重复性、替换值合同、公开关键字参数、空起点、孤立图、CSR 转换和空稀疏输入；依赖 PyTorch、torch_npu 和仓内扩展，不要求第三方 torch_sparse 包。通过 `NPU_DEVICE_ID` 选择设备。

```bash
NPU_DEVICE_ID=0 python -m pytest test/torch_sparse_utils/arch35 -q
```

性能看护由后续流水线执行，产品测试目录不包含性能脚本。任务书完整性能矩阵及既有逐例记录保留在独立验收材料中。

任务全量 fixtures、GPU 基线、400 项功能/性能工具、profiler 和可选上游兼容测试保存在 PR 验证附件。fixed2 是 mask200/walk200 的子集，性能唯一用例为 400 项而非 402 项；功能400使用另一套生成矩阵，不能把公共多邻居游走称为跨后端逐元素 EXACT。复现附件时保留原任务计时/内存口径、完整分母、失败/跳过及源码/二进制身份。

[返回 API 参考](api_reference.md)

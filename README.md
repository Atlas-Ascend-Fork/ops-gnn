# ops-gnn

[English](README_en.md) | 简体中文

ops-gnn 是昇腾生态下针对图神经网络（GNN）推出的一款算子库，支持基于NPU对图神经网络进行加速。

## 项目结构

```text
ops-gnn/
├── csrc/                       # C++/AscendC源码目录
│   ├── pybind.cpp              # PyTorch绑定代码
│   └── npu/                    # NPU相关代码（按算子分类）
├── docs/                       # 文档目录（API 说明见 docs/*/api_reference.md）
├── python/                     # Python源码目录
│   └── ops_gnn/                # Python包目录
├── test/                       # 测试目录（按 <算子>/<架构> 组织）
├── scripts/                    # 构建脚本目录
│   └── build.sh                # 统一构建脚本
├── cmake/                      # CMake配置
│   └── OpsGNNConfig.cmake.in
├── CMakeLists.txt              # CMake构建配置
├── setup.py                    # Python 安装脚本（setuptools + CMake）
├── setup.cfg                   # setuptools配置
├── pyproject.toml              # 现代Python项目配置
├── MANIFEST.in                 # 打包清单
└── LICENSE                     # CANN 许可证
```

`csrc/npu` 下各算子目录包含 `op_host` 和 `op_kernel/<arch>`；测试按 `test/<算子>/<arch>` 组织，各架构子目录包含 `golden.py`、功能测试文件和性能测试文件。其中，`<arch>` 表示目标架构对应的目录名。编译与测试均只处理当前机器芯片型号对应的目录：950 → `arch35`，A2/A3 → `arch22`，其他芯片型号不支持。

## 环境要求

- Python 3.9+
- CMake 3.18+
- CANN 9.1.0 及以上 + HDK（驱动/固件）25.7.rc1 及以上
- PyTorch 2.7+
- 与 PyTorch、CANN 匹配的 torch_npu
- CANN Toolkit (AscendC 编译器)
- C++17 或更高版本编译器
- 支持平台：Ascend 950（arch35）与 A2/A3（arch22）；其他平台不支持
- glibc 2.41 版本暂不支持

### CANN 环境配置

```bash
# 激活 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

如果 CANN 安装在自定义路径，请执行：

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

## 安装方法

### 方法1：使用pip直接安装

```bash
# 安装开发模式
python3 -m pip install --no-build-isolation --no-deps -e .
```

## 方法2：使用构建脚本

```bash
cd scripts

# 构建 Python 包
./build.sh python
cd ..
```

## 方法3：使用CMake（Linux）

```bash
mkdir -p build_cmake
cd build_cmake
cmake ..
cmake --build .
cd ..
```

### 构建产物与目标设备

三种方法均将可分发的 wheel 包保存到 `output/whl/`。方法 1、2 同时完成开发模式安装；
方法 3 在编译共享库后自动打包。
wheel 文件名按目标设备区分：

| 目标设备 | wheel 文件名 |
|----------|--------------|
| A2 | `ops_gnn-0.1.0+cann_a2-*.whl` |
| A3 | `ops_gnn-0.1.0+cann_a3-*.whl` |
| 950 | `ops_gnn-0.1.0+cann_950-*.whl` |

默认识别本机芯片，检测不到设备时默认选择 950。需要手动指定时，方法 1 设置环境变量
`CANN_TARGET`，方法 2 使用 `--cann-target`，方法 3 使用 `-DCANN_TARGET`；取值为 `a2`、`a3` 或 `950`。

## 功能测试

Ascend 950 的 torch_sparse SpMM 随 `ops_gnn` 一起构建和安装。安装
`torch_sparse` 后，在导入其前端函数前执行
`from ops_gnn import torch_sparse_spmm` 注册 NPU 路径。接口、依赖与复现
接口与约束见 [SpMM API 文档](docs/zh/api_reference.md#torch_sparse_spmm)，构建与测试命令见 [开发指南](docs/zh/development_guide.md#torch_sparse_spmm)。


```bash
# 安装测试依赖
pip install pytest pytest-cov

# 快速开始：先跑当前架构的导入用例（950 上为 arch35，A2/A3 上为 arch22）
pytest test/test_import.py -v

# 运行当前机器芯片型号对应的所有功能测试（950→arch35，A2/A3→arch22）
pytest test/ -v
```

运行单个算子的功能测试：

```bash
NPU_DEVICE_ID=<device_id> python3 -m pytest test/<op>/<arch>/test_<op>.py -v
```

其中，`<arch>` 按芯片型号取 `arch35`（950）或 `arch22`（A2/A3），`<op>` 表示算子名，`<device_id>` 表示设备 ID。

## 性能测试

运行单个算子的性能测试：

```bash
NPU_DEVICE_ID=<device_id> python3 test/<op>/<arch>/benchmark_<op>.py
```

其中，`<arch>` 按芯片型号取 `arch35`（950）或 `arch22`（A2/A3），`<op>` 表示算子名，`<device_id>` 表示设备 ID。

## 使用示例

```python
import torch
import ops_gnn

# CSR 格式的分段最大值运算
src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
result = ops_gnn.segment_max_csr(src, indptr)
print(result)  # 输出: tensor([[3, 4], [7, 8]], device='npu:0')

# torch_sparse 兼容：row indices <-> CSR rowptr
row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
rowptr = ops_gnn.ind2ptr(row, 8)          # 替换 torch.ops.torch_sparse.ind2ptr(row, 8)
row2 = ops_gnn.ptr2ind(rowptr, 6)         # 替换 torch.ops.torch_sparse.ptr2ind(rowptr, 6)

# COO 行扩展；index 必须为有序 int64
src = torch.arange(20, dtype=torch.float32, device='npu').reshape(5, 4)
index = torch.tensor([0, 1, 1, 4], dtype=torch.int64, device='npu')
result = ops_gnn.gather_coo(src, index)
print(result.shape)  # 输出: torch.Size([4, 4])

# 按 CSR 指针展开 segment 特征
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16, device='npu')
indptr = torch.tensor([0, 2, 5], dtype=torch.int64, device='npu')
result = ops_gnn.gather_csr(src, indptr)
print(result.shape)  # torch.Size([5, 2])

# torch_scatter 兼容的 Scatter 归约
src = torch.randn(10, 6, 64, dtype=torch.float32, device='npu')
index = torch.randint(0, 4, (10,), dtype=torch.long, device='npu')
result = ops_gnn.scatter(src, index, dim=0, dim_size=4, reduce='sum')
print(result.shape)  # torch.Size([4, 6, 64])
```

## 算子列表

| 算子 | 功能 | 设备支持 | API 文档 |
|------|------|----------|----------|
| `gather_coo` | 按有序 COO 索引扩展源行 | NPU | [gather_coo — COO 行扩展](docs/zh/api_reference.md#gather_coo--coo-行扩展) |
| `gather_csr` | 按 CSR 指针展开 segment 特征 | NPU | [gather_csr - CSR 分段展开](docs/zh/api_reference.md#gather_csr---csr-分段展开) |
| `fps` | 与 torch_cluster 对齐的最远点采样 | NPU（Ascend 950） | [fps — 最远点采样](docs/zh/api_reference.md#fps--最远点采样) |
| `random_walk` | COO 图上的均匀或 node2vec 偏置随机游走 | NPU | [random_walk — NPU 随机游走](docs/zh/api_reference.md#random_walk--npu-随机游走) |
| `segment_max_csr` | CSR 格式的分段最大值运算 | NPU | [segment_max_csr — CSR 分段最大值](docs/zh/api_reference.md#segment_max_csr--csr-分段最大值) |
| `radius` / `radius_graph` | 半径内邻居搜索（torch_cluster 兼容，Ascend 950PR） | NPU | [radius / radius_graph — 半径内邻居搜索](docs/zh/api_reference.md#radius--radius_graph--半径内邻居搜索) |
| `knn` / `knn_graph` | 精确近邻检索与图构建（Ascend 950PR） | NPU | [knn / knn_graph — 精确近邻搜索](docs/zh/api_reference.md#knn--knn_graph--精确近邻搜索) |
| `nearest` | 同batch最近y点的全局索引 | NPU（950，arch35） | [nearest](docs/zh/api_reference.md) |
| `grid_cluster` | 将点坐标映射为规则网格的线性 int64 ID（torch_cluster 兼容） | NPU（950，arch35） | [grid_cluster — 规则网格聚类](docs/zh/api_reference.md) |
| `graclus_cluster` | 图贪心聚类 | NPU / CPU float64 回退 | [graclus_cluster — 图贪心聚类](docs/zh/api_reference.md#graclus_cluster--图贪心聚类) |
| `ind2ptr` | 有序行索引转 CSR 行指针（对齐 torch_sparse） | NPU | [ind2ptr — 行索引转 CSR 行指针](docs/zh/api_reference.md#ind2ptr--行索引转-csr-行指针) |
| `ptr2ind` | CSR 行指针转行索引（对齐 torch_sparse） | NPU | [ptr2ind — CSR 行指针转行索引](docs/zh/api_reference.md#ptr2ind--csr-行指针转行索引) |
| `scatter` / `scatter_*` | 与 torch_scatter 对齐的索引分组归约 | NPU / CPU float64、int64 回退 | [scatter — Scatter 系列归约](docs/zh/api_reference.md#scatter--scatter-系列归约) |
| `spmm` | CSR copy/binary 消息与 sum/max/min/mean 聚合 | NPU（A2/A3，arch22） | [spmm — 一维与二维 CSR 聚合](docs/zh/api_reference.md#spmm--csr-通用聚合) |
| `bspmm` | 三维及更高维 CSR copy/binary 消息聚合 | NPU（A2/A3，arch22） | [bspmm — 高维 CSR 聚合](docs/zh/api_reference.md#bspmm--批量-csr-通用聚合) |

## 开发指南

### 添加新的 NPU 算子

1. 在 `csrc/npu/<op>/op_kernel/<arch>/` 中创建 AscendC 内核文件（`.cpp`）和头文件（`.h`）
2. 在 `csrc/npu/<op>/op_host/` 中创建算子接口实现（`.cpp`）和头文件（`.h`）
3. 在 `csrc/pybind.cpp` 中添加 PyTorch 绑定
4. 在 `python/ops_gnn/` 中创建 Python 接口声明文件
5. 更新 `python/ops_gnn/__init__.py` 导出新函数
6. 在 `test/<op>/<arch>/` 中添加 `golden.py`、`test_<op>.py` 和 `benchmark_<op>.py`

其中，`<op>` 表示算子名，`<arch>` 按芯片型号取 `arch35`（950）或 `arch22`（A2/A3）。

## 许可证

CANN Open Software License Agreement Version 2.0

第三方来源的版权和许可全文集中保留于 [LICENSE](LICENSE)，
KNN 的改编范围见[API 文档](docs/zh/api_reference.md#knn-第三方来源)。

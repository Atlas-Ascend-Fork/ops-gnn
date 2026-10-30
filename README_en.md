# ops-gnn

English | [简体中文](README.md)

ops-gnn is an operator library for Graph Neural Networks (GNNs) in the Huawei Ascend ecosystem, designed to accelerate GNN computation on NPUs.

## Project Structure

```text
ops-gnn/
├── csrc/                       # C++/AscendC source code directory
│   ├── pybind.cpp              # PyTorch binding code
│   └── npu/                    # NPU-related code (organized by operator)
├── docs/                       # Docs (API reference: docs/*/api_reference.md)
├── python/                     # Python source directory
│   └── ops_gnn/                # Python package directory
├── test/                       # Tests organized by <operator>/<arch>
├── scripts/                    # Build scripts directory
│   └── build.sh                # Unified build script
├── cmake/                      # CMake configuration
│   └── OpsGNNConfig.cmake.in
├── CMakeLists.txt              # CMake build configuration
├── setup.py                    # Python installer (setuptools + CMake)
├── setup.cfg                   # setuptools configuration
├── pyproject.toml              # Modern Python project configuration
├── MANIFEST.in                 # Packaging manifest
└── LICENSE                     # CANN License
```

Each operator directory under `csrc/npu` contains `op_host` and `op_kernel/<arch>`. Each operator directory under `test` contains `golden.py`, a functional test file, and a performance test file. Here, `<arch>` is the directory name for the target architecture.

## Requirements

- Python 3.9+
- CMake 3.18+
- CANN 9.1.0 or later + HDK (driver/firmware) 25.7.rc1 or later
- PyTorch 2.7+
- A torch_npu build matched to PyTorch and CANN
- CANN Toolkit (AscendC compiler)
- C++17 or later compiler
- Supported platform: Ascend 950 (arch35) and A2/A3 (arch22); other platforms are not supported
- glibc 2.41 is not supported

### CANN Environment Setup

```bash
# Activate CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

If CANN is installed in a custom path, run:

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

## Installation

### Method 1: Install via pip

```bash
# Install in development mode
python3 -m pip install --no-build-isolation --no-deps -e .
```

## Method 2: Using the build script

```bash
cd scripts

# Build Python package
./build.sh python
cd ..
```

## Method 3: Using CMake (Linux)

```bash
mkdir -p build_cmake
cd build_cmake
cmake ..
cmake --build .
cd ..
```

### Build Outputs and Target Devices

All three methods save distributable wheels to `output/whl/`. Methods 1 and 2 also install in
development mode; Method 3 packages the wheel after building shared libraries.
The wheel has a device-neutral version.
By default it builds arch22 for A2/A3 and arch35 for 950, bundling
`libopsgnn_npu_kernel-A2-A3.so` (shared by A2/A3) and
`libopsgnn_npu_kernel-950.so`. On import, `ops_gnn` detects the local device
with `npu-smi` and loads the matching native extension and kernel. Set
`CANN_TARGET=a2/a3/950` to select a target explicitly. A2/A3 share the native
extension and kernel in `python/ops_gnn/A2-A3/`; 950 uses `python/ops_gnn/950/`.
A wheel built for either a2 or a3 supports both A2 and A3.

For a single-device build, set `CANN_TARGET` with Method 1, pass
`--cann-target a2/a3/950` with Method 2, or pass
`-DCANN_TARGET=a2/a3/950` with Method 3. `NPU_ARCH=dav-2201` builds
both A2 and A3; `NPU_ARCH=dav-3510` builds 950 only.

## Functional Tests

On Ascend 950, torch_sparse SpMM is built and installed with `ops_gnn`. After
installing `torch_sparse`, run `from ops_gnn import torch_sparse_spmm` before
importing its frontend functions. See the [SpMM API reference](docs/en/api_reference.md#torch_sparse_spmm)
for interfaces and the [development guide](docs/en/development_guide.md#torch_sparse_spmm)
for dependencies, build commands and reproduction steps.


```bash
# Install test dependencies
pip install pytest pytest-cov scipy

# Quick start: run the import tests for the current architecture (arch35 on 950, arch22 on A2/A3)
pytest test/test_import.py -v

# Run all functional tests for the local chip model (950→arch35, A2/A3→arch22)
pytest test/ -v
```

Run the functional tests for one operator:

```bash
NPU_DEVICE_ID=<device_id> python3 -m pytest test/<op>/<arch>/test_<op>.py -v
```

Here, `<arch>` is `arch35` (950) or `arch22` (A2/A3) according to the chip model, `<op>` is the operator name, and `<device_id>` is the device ID.

## Performance Tests

Run the performance tests for one operator:

```bash
NPU_DEVICE_ID=<device_id> python3 test/<op>/<arch>/benchmark_<op>.py
```

Here, `<arch>` is `arch35` (950) or `arch22` (A2/A3) according to the chip model, `<op>` is the operator name, and `<device_id>` is the device ID.

## Usage Examples

```python
import torch
import ops_gnn

# CSR segment max operation
src = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, device='npu')
indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device='npu')
result = ops_gnn.segment_max_csr(src, indptr)
print(result)  # output: tensor([[3, 4], [7, 8]], device='npu:0')

# torch_sparse compatibility: row indices <-> CSR rowptr
row = torch.tensor([2, 2, 4, 5, 5, 6], dtype=torch.long, device='npu')
rowptr = ops_gnn.ind2ptr(row, 8)          # replaces torch.ops.torch_sparse.ind2ptr(row, 8)
row2 = ops_gnn.ptr2ind(rowptr, 6)         # replaces torch.ops.torch_sparse.ptr2ind(rowptr, 6)

# COO row expansion; index must be sorted int64
src = torch.arange(20, dtype=torch.float32, device='npu').reshape(5, 4)
index = torch.tensor([0, 1, 1, 4], dtype=torch.int64, device='npu')
result = ops_gnn.gather_coo(src, index)
print(result.shape)  # output: torch.Size([4, 4])

# Expand segment features using CSR pointers
src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16, device='npu')
indptr = torch.tensor([0, 2, 5], dtype=torch.int64, device='npu')
result = ops_gnn.gather_csr(src, indptr)
print(result.shape)  # torch.Size([5, 2])

# torch_scatter-compatible indexed reduction
src = torch.randn(10, 6, 64, dtype=torch.float32, device='npu')
index = torch.randint(0, 4, (10,), dtype=torch.long, device='npu')
result = ops_gnn.scatter(src, index, dim=0, dim_size=4, reduce='sum')
print(result.shape)  # torch.Size([4, 6, 64])
```

## Operator List

| Operator | Description | Device Support | API Reference |
|----------|-------------|----------------|---------------|
| `gather_coo` | Expand source rows by sorted COO indices | NPU | [gather_coo — COO Row Expansion](docs/en/api_reference.md#gather_coo--coo-row-expansion) |
| `gather_csr` | Expands segment features using CSR pointers | NPU | [gather_csr - CSR Segment Expansion](docs/en/api_reference.md#gather_csr---csr-segment-expansion) |
| `fps` | torch_cluster-compatible farthest point sampling | NPU (Ascend 950) | [fps — Farthest Point Sampling](docs/en/api_reference.md#fps--farthest-point-sampling) |
| `random_walk` | Uniform or node2vec-biased random walks on COO graphs | NPU | [random_walk — NPU Random Walk](docs/en/api_reference.md#random_walk--npu-random-walk) |
| `segment_max_csr` | Segmented max reduction on CSR format | NPU | [segment_max_csr — CSR Segmented Max](docs/en/api_reference.md#segment_max_csr--csr-segmented-max) |
| `radius` / `radius_graph` | Radius neighbor search (torch_cluster compatible, Ascend 950PR) | NPU | [radius / radius_graph — Radius Neighbor Search](docs/en/api_reference.md#radius--radius_graph--radius-neighbor-search) |
| `knn` / `knn_graph` | Exact neighbor search and graph construction (Ascend 950PR) | NPU | [knn / knn_graph — Exact Neighbor Search](docs/en/api_reference.md#knn--knn_graph--exact-neighbor-search) |
| `nearest` | Global index of the nearest y point in the same batch | NPU (950, arch35) | [nearest](docs/en/api_reference.md) |
| `grid_cluster` | Map point coordinates to linear int64 regular-grid IDs (torch_cluster compatible) | NPU (950, arch35) | [grid_cluster — Regular Grid Clustering](docs/en/api_reference.md) |
| `graclus_cluster` | Greedy graph clustering | NPU / CPU fallback for float64 | [graclus_cluster — Greedy Graph Clustering](docs/en/api_reference.md#graclus_cluster--greedy-graph-clustering) |
| `ind2ptr` | Sorted row indices to CSR row pointer (torch_sparse-aligned) | NPU | [ind2ptr — Sorted Row Indices to CSR Row Pointer](docs/en/api_reference.md#ind2ptr--sorted-row-indices-to-csr-row-pointer) |
| `ptr2ind` | CSR row pointer to row indices (torch_sparse-aligned) | NPU | [ptr2ind — CSR Row Pointer to Row Indices](docs/en/api_reference.md#ptr2ind--csr-row-pointer-to-row-indices) |
| `scatter` / `scatter_*` | torch_scatter-compatible indexed reductions | NPU / CPU fallback for float64 and int64 | [scatter — Scatter Reductions](docs/en/api_reference.md#scatter--scatter-reductions) |
| `spmm` | CSR copy/binary messages with sum/max/min/mean aggregation | NPU (A2/A3, arch22) | [spmm — Rank-1 and Rank-2 CSR Aggregation](docs/en/api_reference.md#spmm--general-csr-aggregation) |
| `bspmm` | Rank-3-and-higher CSR copy/binary message aggregation | NPU (A2/A3, arch22) | [bspmm — Higher-Rank CSR Aggregation](docs/en/api_reference.md#bspmm--general-batched-csr-aggregation) |

## Development Guide

### Adding a New NPU Operator

1. Create AscendC kernel files (`.cpp`) and headers (`.h`) in `csrc/npu/<op>/op_kernel/<arch>/`
2. Create operator interface implementation (`.cpp`) and header (`.h`) in `csrc/npu/<op>/op_host/`
3. Add PyTorch bindings in `csrc/pybind.cpp`
4. Create Python interface declaration files in `python/ops_gnn/`
5. Update `python/ops_gnn/__init__.py` to export the new function
6. Add `golden.py`, `test_<op>.py`, and `benchmark_<op>.py` under `test/<op>/<arch>/`

Here, `<op>` is the operator name, and `<arch>` is `arch35` (950) or `arch22` (A2/A3) according to the chip model.

## License

CANN Open Software License Agreement Version 2.0

Third-party copyrights and complete licenses are retained in [LICENSE](LICENSE).
KNN adaptation scope is documented in the [API reference](docs/en/api_reference.md#knn-third-party-sources).

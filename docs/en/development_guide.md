# ops-gnn Development Guide

This document provides guidance on development environment setup, operator development, and testing for the ops-gnn library.

## Environment Setup

### Install CANN

#### Pre-installation

For offline installation, click [here](https://www.hiascend.com/developer/download/community/result?module=cann) to download the CANN package and upload it to any path on the installation environment.

#### Install CANN

CANN 9.1.0 or later and HDK (driver/firmware) 25.7.rc1 or later are required. Other versions are not currently supported.

```sh
chmod +x Ascend-cann-toolkit_${VERSION}_linux-$(arch).run
./Ascend-cann-toolkit_${VERSION}_linux-$(arch).run --install
```

Where `${VERSION}` is the CANN version (e.g., 9.1.0) and `$(arch)` is the CPU architecture.

#### Post-installation Configuration

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

If CANN is installed in a custom path, run:

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

### Detailed CANN Installation Guide

Visit [Ascend Documentation](https://www.hiascend.com/document) → CANN Community Edition → Software Installation for detailed installation steps based on your environment, OS, and use case.

### Dependencies

ops-gnn depends on the following components:

| Component | Version | Description |
|-----------|---------|-------------|
| Python | 3.9+ | |
| CMake | 3.18+ | C++ build system |
| GCC | 7.0+ | C++17 compiler |
| PyTorch | 2.7+ | Ascend-adapted version from Ascend Community |
| torch_npu | Matched to PyTorch and CANN | PyTorch NPU device backend |
| CANN Toolkit | 9.1.0+ | AscendC compiler, Bisheng compiler, runtime libraries |
| HDK (driver/firmware) | 25.7.rc1+ | Ascend hardware driver and firmware |

`torch` and `torch_npu` must be obtained from the [Ascend Community Download Page](https://www.hiascend.com/developer/download/community) for Ascend hardware-adapted versions.

The supported platforms are Ascend 950 (arch35) and A2/A3 (910B/910C, arch22); other platforms are not supported.

```sh
# System toolchain
sudo apt install build-essential cmake

# Test dependencies (optional)
pip install pytest pytest-cov
```

### Install ops-gnn

```sh
# Method 1: pip development mode
python3 -m pip install --no-build-isolation --no-deps -e .

# Method 2: Build script
cd scripts
./build.sh python
cd ..

# Method 3: CMake manual build
mkdir build_cmake && cd build_cmake
cmake ..
cmake --build .
cd ..
```

## Project Architecture

### Layered Architecture

ops-gnn adopts a layered architecture of PyTorch extension + AscendC kernel:

```text
+---------------------------------------------+
|              Python Interface Layer           |
|   python/ops_gnn/<op>.py                    |
|   (Type annotations, parameter preprocessing,|
|    documentation)                            |
+---------------------------------------------+
|              PyTorch Binding Layer            |
|   csrc/pybind.cpp                           |
|   (PYBIND11_MODULE registration, param map)  |
+---------------------------------------------+
|              Host-side Operator Layer         |
|   csrc/npu/<op>/op_host/<op>.cpp/.h            |
|   (Argument parsing, Tiling calc,            |
|    Kernel launch, stream management)         |
+---------------------------------------------+
|              AscendC Kernel Layer             |
|   csrc/npu/<op>/op_kernel/<arch>/<op>.cpp/.h          |
|   (AscendC SIMT/VF programming,              |
|    vector computation, data movement)        |
+---------------------------------------------+
```

**Architecture Description:**

- **Python Interface Layer**: Provides user-friendly Python API with type annotations, docstrings, and usage examples
- **PyTorch Binding Layer**: Exposes C++ functions to Python via pybind11, handles Tensor parameter passing
- **Host-side Operator Layer**: Runs on host CPU, responsible for parameter validation, Tiling data computation, AscendC Kernel launch, and Stream synchronization
- **AscendC Kernel Layer**: Runs on NPU device (AIV cores), implements core computation logic

### Directory Structure

```text
ops-gnn
├── csrc/                           # C++/AscendC source code
│   ├── pybind.cpp                  # PyTorch binding code
│   └── npu/                        # NPU-related code (organized by operator)
├── docs/                           # Docs (API reference: docs/*/api_reference.md)
├── python/                         # Python source code
│   └── ops_gnn/                    # Python package
├── test/                           # Tests organized as <operator>/<arch>
├── scripts/                        # Build scripts
│   └── build.sh                    # Unified build/test script (python/cpp/all/test)
├── cmake/                          # CMake configuration
│   └── OpsGNNConfig.cmake.in       # CMake package config template
├── CMakeLists.txt                  # CMake build configuration
├── setup.py                        # Python install script (setuptools + cmake)
├── setup.cfg                       # setuptools configuration
├── pyproject.toml                  # Modern Python project config
├── MANIFEST.in                     # Packaging manifest
└── LICENSE                         # CANN License
```

Each operator directory under `csrc/npu` contains `op_host` and `op_kernel/<arch>`. Each operator directory under `test` contains `golden.py`, a functional test file, and a performance test file. Here, `<arch>` is the directory name for the target architecture.

### Core File Descriptions

| File | Description |
|------|-------------|
| `csrc/pybind.cpp` | PyTorch binding entry, registers all C++ operators to Python via `PYBIND11_MODULE` |
| `csrc/npu/<op>/op_host/<op>.h` | Host-side operator interface declaration |
| `csrc/npu/<op>/op_host/<op>.cpp` | Host-side operator implementation: Tensor dimension parsing, Tiling calc, dtype dispatch, Stream management |
| `csrc/npu/<op>/op_kernel/<arch>/<op>.h` | Kernel Launch function declaration (Host-side call entry) |
| `csrc/npu/<op>/op_kernel/<arch>/<op>.cpp` | Kernel Launch implementation + explicit template instantiation + `<<<>>>` launch syntax |
| `csrc/npu/<op>/op_kernel/<arch>/<op>_kernel.h` | AscendC Kernel core class implementation (Init → Process → Compute pipeline) |
| `csrc/npu/<op>/op_kernel/<arch>/<op>_tiling.h` | Tiling data structure definition (parameter struct passed to Device side) |
| `python/ops_gnn/<op>.py` | Python interface: type annotations, default value handling, calls `_pybind.<op>` |
| `python/ops_gnn/__init__.py` | Package entry, imports from modules and registers to `__all__` |
| `python/ops_gnn/typing.py` | Type aliases (`Tensor`, `OptTensor`) |

## Coding Standards

### Naming Conventions

**C++ Layer:**

- File names: lowercase underscore-separated, e.g., `segment_max_csr_kernel.h`
- Function names: PascalCase, e.g., `LaunchSegmentMaxCsrKernel`
- Struct names: PascalCase, e.g., `SegmentMaxCsrTilingData`
- Class names: PascalCase, e.g., `SegmentMaxCsrKernel`
- Template parameters: PascalCase or single uppercase letter, e.g., `typename T`

**Python Layer:**

- File names: lowercase underscore-separated, e.g., `segment_max_csr.py`
- Function names: lowercase underscore-separated, e.g., `segment_max_csr`
- Consistent with PyTorch style

### Code Style

- C++ follows C++17 standard, uses `#pragma once` for header guards
- AscendC Kernel uses `__aicore__`, `__global__`, `__gm__` modifiers
- Use APIs under namespace `AscendC`
- Python interfaces use type annotations (`Tensor`, `OptTensor`, `Optional[Tensor]`)
- Each file includes CANN Open Software License copyright header

### File Organization

Each operator strictly follows the following file split:

```text
csrc/npu/<op>/
├── op_host/
│   ├── <op>.h          # Host interface declaration
│   └── <op>.cpp        # Host implementation
└── op_kernel/
    └── <arch>/
        ├── <op>.h               # Kernel Launch declaration
        ├── <op>.cpp             # Kernel Launch implementation + template instantiation
        ├── <op>_kernel.h        # Kernel core class (simple ops can merge into <op>.cpp)
        └── <op>_tiling.h        # Tiling struct
```

## Build System

### CMake Build

CMakeLists.txt compiles AscendC Kernels and PyTorch binding library:

1. Uses Bisheng compiler to compile Kernel source files into `libopsgnn_npu_kernel.so`
2. Compiles `csrc/pybind.cpp` + host-side code into `_pybind.so`
3. Outputs artifacts to `output/kernel/`

Key CMake variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NPU_ARCH` | NPU architecture | Detected by chip model: `dav-3510` (950 → arch35) / `dav-2201` (A2 (910B)/A3 (910C) → arch22); can be set to override detection |
| `WITH_PYTHON` | Enable Python binding build | `ON` |
| `ASCEND_HOME_PATH` | CANN installation path | Read from env |
| `Python3_ROOT_DIR` | Python root directory | Auto-exported by build.sh |

### setuptools Build

`setup.py` invokes CMake to compile and generate a wheel package. `build_with_cmake()` cleans old `CMakeCache.txt` before building to avoid path inconsistencies from pip temp directories.

## Testing Guide

### Running Functional Tests

```sh
# Run all functional tests for the local chip model (950→arch35, A2/A3→arch22)
pytest test/ -v
# Or target a directory directly
pytest test/spmm_max/arch22/test_spmm_max.py -v                       # One arch22 operator
pytest test/segment_max_csr/arch35/test_segment_max_csr.py -v             # Single operator test
pytest test/segment_max_csr/arch35/test_segment_max_csr.py::test_func -v  # Single test case
```

> Note: `pytest test/` only collects the directory matching the local chip model — on 950 only `arch35` runs (`arch22` is ignored), on A2/A3 only `arch22` runs (`arch35` is ignored).

### Running Performance Tests

```sh
NPU_DEVICE_ID=<device_id> python test/<op>/<arch>/benchmark_<op>.py
```

Here, `<op>` is the operator name, and `<device_id>` is the device ID.

### Functional Test Writing Pattern

1. `torch.npu.set_device(int(os.environ.get("NPU_DEVICE_ID", 0)))` to specify NPU device
2. `torch.manual_seed(42)` for reproducibility
3. Use `.npu()` method to create NPU Tensor (e.g., `torch.randint(...).npu()`)
4. Call `ops_gnn.<op>(...)`
5. Verify results: device type, shape, numerical correctness

### Functional Test Coverage Requirements

| Scenario | Description |
|----------|-------------|
| Basic functionality | Typical input |
| Different dtypes | float32, float16, int32, int16 |
| Edge cases | Empty segment, single element |
| Optional parameters | optional_out |
| Multi-dimensional input | 2D/3D Tensor |
| Broadcast scenarios | indptr broadcasting |

## Operator Development Walkthrough

This section uses `segment_max_csr` as an example to illustrate the operator development process.

### Call Chain Analysis

```text
User Code
    └─→ ops_gnn.segment_max_csr(src, indptr, optional_out)
            │  python/ops_gnn/segment_max_csr.py
            │  (legacy optional-argument handling; Gather COO passes None explicitly through pybind)
            └─→ _pybind.segment_max_csr(src, indptr, optional_out)
                    │  csrc/pybind.cpp
                    │  (PYBIND11_MODULE registration)
                    └─→ segment_max_csr()  [Host]
                            │  csrc/npu/segment_max_csr/op_host/segment_max_csr.cpp
                            │  (dimension parsing, Tiling fill, dtype dispatch, Stream management)
                            └─→ LaunchSegmentMaxCsrKernel<T>()
                                    │  csrc/npu/segment_max_csr/op_kernel/<arch>/segment_max_csr.cpp
                                    │  (get AIV core count, <<<>>> launch)
                                    └─→ segment_max_csr_kernel<T>  [Device]
                                            │  segment_max_csr_kernel.h
                                            │  (Init → Process → Compute pipeline)
```

Each layer's responsibilities:

- **Python** (`python/ops_gnn/<op>.py`): Type annotations, optional param defaults, calls `_pybind.<op>`
- **PyBind** (`csrc/pybind.cpp`): `m.def("<op>", &func, py::arg(...), ...)` registers C++ function
- **Host** (`csrc/npu/<op>/op_host/`): Tensor dimension parsing → Tiling calc → `torch::empty` create output → dtype dispatch launch. Legacy operators may manage their own stream; Gather COO must reuse PyTorch's current NPU stream and must not create, synchronize, or destroy a private ACL stream.
- **Kernel Launch** (`csrc/npu/<op>/op_kernel/<arch>/<op>.cpp`): `GetCoreNumAiv()` get core count → `<<<coreNum, nullptr, stream>>>` launch → explicit template instantiation
- **Kernel Impl** (`<op>_kernel.h`): `Init()` parse Tiling + allocate Buffer/Event → `Process()` assign work range per Block → `Compute()` double-buffer pipeline (DataCopy → Max/Add → DataCopy)

### New Operator File Checklist

Using `segment_max_csr` as a template, each new operator needs:

```text
csrc/npu/<op>/op_host/
├── <op>.h                   # torch::Tensor <op>(torch::Tensor ...);
└── <op>.cpp                 # Host implementation

csrc/npu/<op>/op_kernel/<arch>/
├── <op>.h                   # template<typename T> void Launch<Op>Kernel(...);
├── <op>.cpp                 # Launch impl + template instantiation
├── <op>_kernel.h            # Kernel core class (Init/Process/Compute)
└── <op>_tiling.h            # struct <Op>TilingData { uint32_t/uint64_t ... };

python/ops_gnn/
└── <op>.py                  # Python interface

test/<op>/<arch>/
├── golden.py                # CPU/reference implementation
├── test_<op>.py             # Functional tests
└── benchmark_<op>.py        # Performance benchmark
```

Here, `<arch>` is `arch35` (950) or `arch22` (A2/A3) according to the chip model.

Additionally, two files must be modified:

- `csrc/pybind.cpp`: `#include "<op>/op_host/<op>.h"` + `m.def("<op>", ...)`
- `python/ops_gnn/__init__.py`: `from .<op> import <op>` + add to `__all__`

Simple operators can merge files: omit `_tiling.h` when no Tiling, merge `<op>_kernel.h` into `<op>.cpp` when logic is simple (see `ptr2ind`).

### Two Kernel Modes

| Feature | ptr2ind | segment_max_csr |
|---------|-----------|----------------|
| File count | 2 files (.h + .cpp) | launch + kernel + tiling (4 files) |
| Programming model | SIMT (`__simt_vf__` + `VF_CALL`) | Kernel class (TPipe + Buffer + Event) |
| Data movement | Direct GM read/write | DataCopy + double-buffer pipeline |
| Tiling | None (scalar params direct) | Tiling struct |
| Applicable | Element-wise ops | Reductions, segmentation, multi-stage pipelines |

### Development Process Summary

1. Create directory structure and files following `segment_max_csr`
2. Define the Tiling struct (no pointers; use 64-bit fields for shape, length, and address values where required)
3. Implement Kernel core class (Init → Process → Compute), use TPipe + Event for double-buffer pipeline
4. Implement Launch function, wrap `<<<>>>` with `#pragma GCC diagnostic`
5. Host side: dimension parsing → Tiling fill → `torch::empty` → dtype dispatch launch; operators requiring asynchronous semantics reuse PyTorch's current NPU stream instead of creating or synchronizing a private stream
6. Register with `m.def(...)` in `pybind.cpp`
7. Write Python interface (type annotations + docstring + default handling)
8. Update `__init__.py` exports
9. Write tests (at minimum: basic functionality, multiple dtypes, edge cases)

### segment_csr file responsibilities

The Host validates tensor metadata and selects the current-stream compute path.
Kernel entry points are `SegmentCsr` and `SegmentCsrVector`; Python exposes the
`segment_*_csr` family. Kernel files under `csrc/npu/segment_csr/op_kernel/arch35/`:

| File | Responsibility |
| --- | --- |
| `segment_csr.h` | Enums, dtype dispatch, vector eligibility and entry declarations |
| `segment_csr.cpp` | General SIMT reduction and packed accesses |
| `segment_csr_vec_kernel.cpp` | AIV transfers, synchronization and register dispatch |
| `segment_csr_extrema.h` | FP16/INT64 extrema and first-occurrence indices |
| `segment_csr_sum.h` | Sum/mean arithmetic and dtype-specific numeric behavior |
| `segment_csr_uniform.h` | Validate every pointer before selecting uniform segments |

The main header and translation unit use the operator name, matching neighboring
operators and the product-review request. The SIMT and AIV execution models keep
separate translation units. Their arithmetic and dispatch are unchanged by this
layout revision.

The product test directory follows `test/segment_csr/arch35/` and contains three files:

| File | Responsibility |
| --- | --- |
| `test_segment_csr.py` | All API, cross-call semantic and vector-route cases in one parameterized suite |
| `golden.py` | Pure PyTorch CPU values and first-extremum index reference |
| `benchmark_segment_csr.py` | Standalone 29-shape timing; input and logging helpers are included here |

Consolidation retains every original test function, assertion and parameter
combination. Extended-suite and vector-suite dtype constants have distinct names
to preserve their different domains. On-device strided views remain explicitly
checked and internal-format warnings remain errors.

Task-specific baseline tables, acceptance/report drivers and profiler scripts
are distributed in the PR validation attachment under `validation_inputs/`.
They are not product test modules. Their complete 464-case correctness/performance
gate remains independent of routine pytest collection. To reproduce from the
product checkout after extracting that attachment:

```bash
PYTHONPATH=python python -m pytest test/segment_csr/arch35 test/segment_max_csr -q
CSR_PRODUCT_ROOT="$PWD" PYTHONPATH=python python /path/to/evidence/validation_inputs/run_segment_csr_acceptance.py --all --warmup 20 --iter 100
```

The external driver uses `CSR_PRODUCT_ROOT` solely to locate product sources and
binaries. Inputs, reference calculations, timing and the 0.45 threshold are retained.
Profiler runs remain a separate diagnostic and do not replace interface timing.


### segment_coo validation

Functional cases are consolidated into one module, with separate reference
and benchmark entry points. Install real CPU torch_scatter 2.1.2 using a CPU
wheel matching the Python/PyTorch environment; tests do not replace it with
skips or mocks. Load CANN and build the extension before running:

```bash
NPU_ARCH=dav-3510 python setup.py build_ext --inplace
PYTHONPATH=python python -m pytest test/segment_coo/arch35 -q -ra
for reduce in sum mean min max; do
  PYTHONPATH=python python test/segment_coo/arch35/benchmark_segment_coo.py \
    --reduce "$reduce" --warmup 20 --iter 100
done
```

### torch_sparse_spmm

See the [API reference](api_reference.md#torch_sparse_spmm) for constraints and examples.

#### File responsibilities

| File | Responsibility |
| --- | --- |
| `python/ops_gnn/torch_sparse_spmm.py` | torch_sparse adaptation, validation, topology caches and first-order gradients |
| `csrc/npu/torch_sparse_spmm/op_host/torch_sparse_spmm.cpp` | ATen APIs, argument validation and dispatch on the current NPU stream |
| `csrc/npu/torch_sparse_spmm/op_kernel/arch35/torch_sparse_spmm.h` / `.cpp` | Declarations and implementations of `TorchSparseSpmm` and its ValueGrad, Validate, Convert and PermuteValues helpers |
| `torch_sparse_spmm_kernel.h` / `torch_sparse_spmm_tiling.h` | Compute templates and shape parameters |
| `csrc/pybind.cpp` | Six central bindings in the ops_gnn package |

Entry names follow the neighboring `SegmentCsr` and `SegmentCoo` operators. Helper entries share the `TorchSparseSpmm` prefix.

#### Build and dependencies

Use Ascend 950 (dav-3510), CANN 9.1.0 and compatible PyTorch/torch_npu versions.
Install matching torch_sparse and torch_scatter packages; PyG tests additionally
require torch_geometric. Python annotations support the repository's Python 3.9+
range. From the ops-gnn repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CANN_TARGET=950 NPU_ARCH=dav-3510 MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps -e .
```

Rebuild after native source changes. This module is unavailable in dav-2201
builds. CMake discovers the host and architecture-specific kernel directories.
Bindings are defined in csrc/pybind.cpp; there is no independent extension package.

#### Tests and benchmark

Run from the repository root with the extension and NPU dependencies installed:

```bash
python -m pytest -q test/torch_sparse_spmm/arch35/test_torch_sparse_spmm.py
python test/torch_sparse_spmm/arch35/benchmark_torch_sparse_spmm.py --reduce sum --iter 100 --warmup 20
python test/torch_sparse_spmm/arch35/benchmark_torch_sparse_spmm.py --format coo --dtype fp32
```

The test directory contains only the benchmark, golden and functional test files.
Tests cover seven precision combinations, four reductions, gradients, validation,
cache invalidation and PyG GCN/GIN. Missing PyG dependencies skip only PyG cases.
The benchmark generates inputs in the script and reports synchronized average
call latency in milliseconds. Warm COO timing includes weight permutation and
SpMM; initial topology conversion is outside this average.
Task acceptance inputs, ATK and profiler tools are in a separate PR validation
attachment.

## More Resources

- **[Back to README](../../README_en.md)**
- **[AscendC Programming Guide](https://www.hiascend.com/document)**
- **[PyTorch C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)**
- **[CANN Community Edition Docs](https://www.hiascend.com/document)**

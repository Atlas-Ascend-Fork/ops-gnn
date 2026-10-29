#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

###############################################################################
# ops-gnn 构建脚本
# 支持生成 Python 包和 C++ 二进制文件
# Keep this script in LF format for Linux build environments.
###############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认值
BUILD_TYPE="Release"
# 命令行 --npu-arch 优先；否则沿用环境变量，最后自动检测设备。
NPU_ARCH="${NPU_ARCH:-${TARGET_NPU_ARCH:-}}"
CANN_TARGET="${CANN_TARGET:-}"
WITH_PYTHON="ON"
INSTALL_PREFIX="${PROJECT_ROOT}/output"

###############################################################################
# 帮助信息
###############################################################################
usage() {
    cat << EOF
${BLUE}ops-gnn 构建脚本${NC}

用法: $0 [选项] [构建目标]

构建目标:
    python           构建 Python 包 (默认)
    cpp              构建 C++ 二进制文件
    all              同时构建 Python 包和 C++ 二进制
    test             运行当前 NPU 架构支持的 pytest 用例

选项:
    -t, --type TYPE      构建类型 (Debug/Release) [默认: Release]
    --npu-arch ARCH      NPU 架构 (dav-3510/dav-2201；dav-2201 覆盖 A2/A3)
                         [默认: 环境变量 NPU_ARCH/TARGET_NPU_ARCH，未设置时按芯片型号自动检测]
    --cann-target TARGET 目标设备 (950/a2/a3)，同时选择架构和 wheel 版本标记
    --no-python          禁用 Python 绑定 (仅 C++ 构建)
    -c, --clean          清理构建缓存 (删除 build 和 output 目录)
    -h, --help           显示帮助信息

示例:
    # 构建 Python 包
    $0 python

    # 构建 C++ 二进制
    $0 cpp

    # 清理缓存后构建
    $0 cpp --clean

    # 同时构建两者
    $0 all -t Debug

    # C++ 构建时禁用 Python 绑定
    $0 cpp --no-python

    # 运行整体测试（自动发现 test/ 下全部用例，含 test/sparse/）
    $0 test

EOF
    exit "${1:-0}"
}

###############################################################################
# 日志函数
###############################################################################
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

###############################################################################
# 参数解析
###############################################################################
BUILD_TARGET="python"

CLEAN_BUILD="OFF"

while [[ $# -gt 0 ]]; do
    case $1 in
        python|cpp|all|test)
            BUILD_TARGET="$1"
            shift
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --npu-arch)
            NPU_ARCH="$2"
            shift 2
            ;;
        --cann-target)
            CANN_TARGET="$2"
            shift 2
            ;;
        --no-python)
            WITH_PYTHON="OFF"
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD="ON"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} 未知选项: $1" >&2
            echo "" >&2
            usage 1
            ;;
    esac
done

detect_cann_target() {
    local npu_smi_bin="${NPU_SMI_BIN:-npu-smi}"
    local device_info
    if ! command -v "$npu_smi_bin" >/dev/null 2>&1; then
        log_warn "未找到 npu-smi，默认以 Ascend 950 设备进行编译；可用 --cann-target 指定目标"
        CANN_TARGET="950"
    elif ! device_info=$("$npu_smi_bin" info -m 2>&1); then
        log_warn "npu-smi info 执行失败，默认以 Ascend 950 设备进行编译；可用 --cann-target 指定目标"
        CANN_TARGET="950"
    elif grep -qiE '910_93|910c' <<< "$device_info"; then
        CANN_TARGET="a3"
    elif grep -qi '910b' <<< "$device_info"; then
        CANN_TARGET="a2"
    elif grep -qi '950' <<< "$device_info"; then
        CANN_TARGET="950"
    else
        log_error "不支持的芯片型号，仅支持 950 / A2 / A3"
    fi
}

case "$NPU_ARCH" in
    ""|dav-3510|dav-2201) ;;
    *) log_error "不支持的 NPU 架构: $NPU_ARCH（支持 dav-3510、dav-2201）" ;;
esac
if [ -z "$CANN_TARGET" ]; then
    if [ "$NPU_ARCH" = "dav-3510" ]; then
        CANN_TARGET="950"
    else
        detect_cann_target
    fi
fi
CANN_TARGET="${CANN_TARGET,,}"
case "$CANN_TARGET" in
    950) TARGET_ARCH="dav-3510" ;;
    a2|a3) TARGET_ARCH="dav-2201" ;;
    *) log_error "不支持的 CANN_TARGET: $CANN_TARGET（支持 950、a2、a3）" ;;
esac
if [ -n "$NPU_ARCH" ] && [ "$NPU_ARCH" != "$TARGET_ARCH" ]; then
    log_error "NPU_ARCH=$NPU_ARCH 与 CANN_TARGET=$CANN_TARGET 不匹配"
fi
NPU_ARCH="$TARGET_ARCH"

# Python 构建由 setup.py 启动新的 CMake 进程，因此必须导出该值。
export NPU_ARCH CANN_TARGET

###############################################################################
# 清理构建缓存
###############################################################################
clean_build() {
    if [ "$CLEAN_BUILD" = "ON" ]; then
        log_info "清理构建缓存..."
        rm -rf "$PROJECT_ROOT/build"
        rm -rf "$PROJECT_ROOT/output"
        log_info "缓存清理完成"
    fi
}

###############################################################################
# 版本比较函数
###############################################################################
version_ge() {
    [ "$(printf '%s\n' "$@" | sort -V | head -n1)" != "$1" ] || [ "$1" = "$2" ]
}

###############################################################################
# 查找 Python 解释器
###############################################################################
find_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python 未安装，请安装 Python 3.9+"
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_info "Python 解释器: $PYTHON_CMD (版本: $PYTHON_VERSION)"

    # 设置 Python3_ROOT_DIR，确保 CMake 找到正确的 Python 版本
    PYTHON_PREFIX=$($PYTHON_CMD -c "import sys; print(sys.prefix)")
    export Python3_ROOT_DIR="$PYTHON_PREFIX"
    log_info "Python3_ROOT_DIR: $Python3_ROOT_DIR"

    if ! version_ge "$PYTHON_VERSION" "3.9"; then
        log_error "Python 版本需要 >= 3.9，当前版本: $PYTHON_VERSION"
    fi
}

###############################################################################
# 环境检查
###############################################################################
check_env() {
    log_info "检查环境..."

    # Python 检查
    if [ "$BUILD_TARGET" != "cpp" ] || [ "$WITH_PYTHON" = "ON" ]; then
        find_python

        if ! $PYTHON_CMD -c "import torch" &> /dev/null; then
            log_error "PyTorch 未安装，请先安装: $PYTHON_CMD -m pip install torch"
        fi

        PYTORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)")
        log_info "PyTorch 版本: $PYTORCH_VERSION"
    fi

    # CMake 检查
    if [ "$BUILD_TARGET" = "cpp" ] || [ "$BUILD_TARGET" = "all" ]; then
        if ! command -v cmake &> /dev/null; then
            log_error "CMake 未安装，请先安装: sudo apt install cmake"
        fi

        CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
        log_info "CMake 版本: $CMAKE_VERSION"

        if ! version_ge "$CMAKE_VERSION" "3.18"; then
            log_error "CMake 版本需要 >= 3.18，当前版本: $CMAKE_VERSION"
            log_error "请升级CMake: https://cmake.org/download/"
        fi
    fi

    # 编译器检查 (Linux)
    if [ "$BUILD_TARGET" = "cpp" ] || [ "$BUILD_TARGET" = "all" ]; then
        if ! command -v g++ &> /dev/null; then
            log_error "GCC 编译器未安装，请先安装: sudo apt install build-essential"
        fi

        GCC_VERSION=$(g++ --version | head -n1 | awk '{print $4}')
        log_info "GCC 版本: $GCC_VERSION"

        if ! version_ge "$GCC_VERSION" "7.0"; then
            log_warn "GCC 版本建议 >= 7.0，当前版本: $GCC_VERSION"
        fi
    fi
}

###############################################################################
# 构建 Python 包
###############################################################################
build_python() {
    log_info "开始构建 Python 包..."

    cd "$PROJECT_ROOT"

    # 开发模式安装同时编译原生库并保存可分发的 wheel 包。
    log_info "安装 Python 包到开发模式..."
    if pip_output=$($PYTHON_CMD -m pip install -e . --no-build-isolation 2>&1); then
        :
    elif [[ "$pip_output" == *"externally-managed-environment"* ]]; then
        log_warn "标准安装受到 externally-managed-environment 限制，尝试使用 --break-system-packages..."
        $PYTHON_CMD -m pip install -e . --no-build-isolation --break-system-packages || \
            log_error "Python 开发模式安装失败"
    else
        printf '%s\n' "$pip_output" >&2
        log_error "Python 开发模式安装失败"
    fi

    # 显示 wheel 包位置
    WHL_FILE=$(find "$PROJECT_ROOT/output/whl" -name "ops_gnn-*+cann_${CANN_TARGET}-*.whl" 2>/dev/null | head -n1)
    if [ -n "$WHL_FILE" ]; then
        log_info "wheel 包生成成功: $WHL_FILE"
    else
        log_error "wheel 包未找到: $PROJECT_ROOT/output/whl（目标: $CANN_TARGET）"
    fi

    log_info "Python 包构建完成!"
    log_info "运行整体测试: $0 test  或  $PYTHON_CMD -m pytest test/"
}

###############################################################################
# 运行代码仓整体测试
###############################################################################
run_tests() {
    log_info "开始运行当前 NPU 架构支持的测试..."
    cd "$PROJECT_ROOT"

    if [ -z "${PYTHON_CMD:-}" ]; then
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        elif command -v python &> /dev/null; then
            PYTHON_CMD="python"
        else
            log_error "未找到 Python 解释器"
        fi
    fi

    export PYTHONPATH="${PROJECT_ROOT}/python:${PYTHONPATH:-}"

    # 测试按 test/<operator>/<arch>/ 组织；根 conftest.py 会根据
    # NPU_ARCH 忽略不匹配的架构子目录。
    log_info "测试目录: test（架构: ${NPU_ARCH}）"

    if ! $PYTHON_CMD -m pytest test -v; then
        log_error "整体测试失败: test（架构: ${NPU_ARCH}）"
    fi

    log_info "整体测试完成!"
}

###############################################################################
# 构建 C++ 二进制
###############################################################################
build_cpp() {
    log_info "开始构建 C++ 二进制..."

    BUILD_DIR="$PROJECT_ROOT/build/cpp_${NPU_ARCH}"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # CMake 配置
    log_info "配置 CMake..."
    cmake \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DWITH_PYTHON="$WITH_PYTHON" \
        -DNPU_ARCH="$NPU_ARCH" \
        -DCANN_TARGET="$CANN_TARGET" \
        "$PROJECT_ROOT"

    # 编译
    log_info "编译 C++ 代码..."
    cmake --build . --config "$BUILD_TYPE" -j$(nproc)

    # 安装到 output/kernel
    log_info "安装到 output/kernel..."
    cmake --install .

    # 更新 Python 包中的 so 文件
    if [ "$WITH_PYTHON" = "ON" ]; then
        PYTHON_SO_PATH="$PROJECT_ROOT/python/ops_gnn/_pybind.so"
        OUTPUT_SO_PATH="$INSTALL_PREFIX/kernel/lib_pybind.so"
        if [ -f "$OUTPUT_SO_PATH" ]; then
            log_info "更新 Python 包中的 so 文件..."
            cp "$OUTPUT_SO_PATH" "$PYTHON_SO_PATH"
            log_info "so 文件已更新: $PYTHON_SO_PATH"
        else
            log_warn "so 文件未找到: $OUTPUT_SO_PATH"
        fi
    fi

    log_info "C++ 二进制构建完成!"
    log_info "输出目录: $BUILD_DIR"
    log_info "安装目录: $INSTALL_PREFIX/kernel"
}

###############################################################################
# 主函数
###############################################################################
main() {
    log_info "ops-gnn 构建脚本启动"
    log_info "构建目标: $BUILD_TARGET"
    log_info "构建类型: $BUILD_TYPE"
    log_info "NPU 架构: $NPU_ARCH"
    log_info "CANN 目标设备: $CANN_TARGET"
    log_info "Python 绑定: $WITH_PYTHON"
    log_info "清理缓存: $CLEAN_BUILD"

    clean_build
    check_env

    case $BUILD_TARGET in
        python)
            build_python
            ;;
        cpp)
            build_cpp
            ;;
        all)
            build_python
            log_info ""
            build_cpp
            ;;
        test)
            run_tests
            ;;
    esac

    if [ "$BUILD_TARGET" = "test" ]; then
        log_info "测试流程结束!"
    else
        log_info "构建完成!"
    fi
}

main

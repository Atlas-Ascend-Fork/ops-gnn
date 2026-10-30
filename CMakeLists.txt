# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPSGNN_VERSION 0.1.0)

# Without a target selection, build all device variants in isolated directories.
# The final wheel contains a single Python distribution and the three native pairs.
set(_REQUESTED_NPU_ARCH "${NPU_ARCH}")
if(_REQUESTED_NPU_ARCH STREQUAL "")
    if(NOT "$ENV{NPU_ARCH}" STREQUAL "")
        set(_REQUESTED_NPU_ARCH "$ENV{NPU_ARCH}")
    else()
        set(_REQUESTED_NPU_ARCH "$ENV{TARGET_NPU_ARCH}")
    endif()
endif()
if(NOT OPSGNN_SINGLE_TARGET
   AND NOT DEFINED CANN_TARGET AND "$ENV{CANN_TARGET}" STREQUAL ""
   AND (_REQUESTED_NPU_ARCH STREQUAL "" OR
        _REQUESTED_NPU_ARCH STREQUAL "dav-2201"))
    project(opsgnn_all LANGUAGES NONE)
    option(WITH_PYTHON "Link to Python when building" ON)
    option(OPSGNN_BUILD_WHEEL "Build a distributable Python wheel" ON)
    if(_REQUESTED_NPU_ARCH STREQUAL "dav-2201")
        set(_DEVICE_TARGETS a2 a3)
    else()
        set(_DEVICE_TARGETS a2 a3 950)
    endif()
    set(_VARIANT_TARGETS)
    foreach(DEVICE_TARGET IN LISTS _DEVICE_TARGETS)
        list(APPEND _VARIANT_TARGETS variant_${DEVICE_TARGET})
        if(DEVICE_TARGET STREQUAL "950")
            set(DEVICE_ARCH dav-3510)
        else()
            set(DEVICE_ARCH dav-2201)
        endif()
        add_custom_target(variant_${DEVICE_TARGET} ALL
            COMMAND ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_SOURCE_DIR}"
                -B "${CMAKE_CURRENT_BINARY_DIR}/variant_${DEVICE_TARGET}"
                -DOPSGNN_SINGLE_TARGET=ON
                -DOPSGNN_BUILD_WHEEL=OFF
                -DWITH_PYTHON=${WITH_PYTHON}
                -DNPU_ARCH=${DEVICE_ARCH}
                -DCANN_TARGET=${DEVICE_TARGET}
            COMMAND ${CMAKE_COMMAND} --build
                "${CMAKE_CURRENT_BINARY_DIR}/variant_${DEVICE_TARGET}"
            VERBATIM
        )
    endforeach()
    # A2 and A3 share the output kernel filename; avoid concurrent writes.
    add_dependencies(variant_a3 variant_a2)
    if(WITH_PYTHON AND OPSGNN_BUILD_WHEEL)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_custom_target(wheel ALL
            COMMAND ${CMAKE_COMMAND} -E env
                "NPU_ARCH=${_REQUESTED_NPU_ARCH}" "TARGET_NPU_ARCH=" "CANN_TARGET="
                "OPSGNN_PREBUILT_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
                "OPSGNN_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
                "${Python3_EXECUTABLE}" -m pip wheel . --no-deps --no-build-isolation
                -w "${CMAKE_CURRENT_SOURCE_DIR}/output/whl"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            DEPENDS ${_VARIANT_TARGETS}
            VERBATIM
        )
    endif()
    return()
endif()

###############################################################################
# Configuration Options
###############################################################################

if(NOT DEFINED NPU_ARCH OR NPU_ARCH STREQUAL "")
    if(NOT "$ENV{NPU_ARCH}" STREQUAL "")
        set(NPU_ARCH "$ENV{NPU_ARCH}")
    else()
        set(NPU_ARCH "$ENV{TARGET_NPU_ARCH}")
    endif()
endif()
set(VALID_NPU_ARCHS dav-3510 dav-2201)
if(NOT NPU_ARCH STREQUAL "" AND NOT "${NPU_ARCH}" IN_LIST VALID_NPU_ARCHS)
    message(FATAL_ERROR "Unsupported NPU_ARCH: ${NPU_ARCH}")
endif()
if(NOT DEFINED CANN_TARGET OR CANN_TARGET STREQUAL "")
    set(CANN_TARGET "$ENV{CANN_TARGET}")
endif()
if(CANN_TARGET STREQUAL "")
    if(NPU_ARCH STREQUAL "dav-3510")
        set(CANN_TARGET "950")
    else()
        if(NOT "$ENV{NPU_SMI_BIN}" STREQUAL "")
            find_program(NPU_SMI_EXECUTABLE NAMES "$ENV{NPU_SMI_BIN}")
        else()
            find_program(NPU_SMI_EXECUTABLE NAMES npu-smi)
        endif()
        set(NPU_SMI_RESULT "not found")
        if(NPU_SMI_EXECUTABLE)
            execute_process(
                COMMAND "${NPU_SMI_EXECUTABLE}" info -m
                RESULT_VARIABLE NPU_SMI_RESULT
                OUTPUT_VARIABLE NPU_SMI_OUTPUT
                ERROR_VARIABLE NPU_SMI_ERROR
                TIMEOUT 10
            )
        endif()
        if(NOT NPU_SMI_RESULT STREQUAL "0")
            message(WARNING "检测不到 NPU 设备，默认以 Ascend 950 设备进行编译；可用 -DCANN_TARGET=950/a2/a3 指定目标")
            set(CANN_TARGET "950")
        else()
            string(TOLOWER "${NPU_SMI_OUTPUT}" NPU_SMI_OUTPUT_LOWER)
            if(NPU_SMI_OUTPUT_LOWER MATCHES "910_93|910c")
                set(CANN_TARGET "a3")
            elseif(NPU_SMI_OUTPUT_LOWER MATCHES "910b")
                set(CANN_TARGET "a2")
            elseif(NPU_SMI_OUTPUT_LOWER MATCHES "950")
                set(CANN_TARGET "950")
            else()
                message(FATAL_ERROR "不支持的芯片型号，仅支持 950 / A2 / A3")
            endif()
        endif()
    endif()
endif()
string(TOLOWER "${CANN_TARGET}" CANN_TARGET)
if(CANN_TARGET STREQUAL "950")
    set(TARGET_ARCH "dav-3510")
elseif(CANN_TARGET STREQUAL "a2" OR CANN_TARGET STREQUAL "a3")
    set(TARGET_ARCH "dav-2201")
else()
    message(FATAL_ERROR "Unsupported CANN_TARGET: ${CANN_TARGET}; use 950/a2/a3")
endif()
if(NOT NPU_ARCH STREQUAL "" AND NOT NPU_ARCH STREQUAL TARGET_ARCH)
    message(FATAL_ERROR "NPU_ARCH=${NPU_ARCH} conflicts with CANN_TARGET=${CANN_TARGET}")
endif()
set(NPU_ARCH "${TARGET_ARCH}")
set(NPU_ARCH "${NPU_ARCH}" CACHE STRING "NPU architecture" FORCE)
set(CANN_TARGET "${CANN_TARGET}" CACHE STRING "CANN target device (950/a2/a3)" FORCE)
set_property(CACHE CANN_TARGET PROPERTY STRINGS 950 a2 a3)

option(WITH_PYTHON "Link to Python when building" ON)
option(OPSGNN_BUILD_WHEEL "Build a distributable Python wheel" ON)
message(STATUS "ops_gnn build: NPU_ARCH=${NPU_ARCH}, CANN_TARGET=${CANN_TARGET}")

###############################################################################
# System Detection
###############################################################################

if(UNIX)
    set(_ascend_processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(NOT _ascend_processor)
        execute_process(
            COMMAND uname -m
            OUTPUT_VARIABLE _ascend_processor
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    set(SYSTEM_PREFIX "${_ascend_processor}-linux")
endif()

###############################################################################
# Ascend/CANN Environment
###############################################################################

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable not set. Please source the CANN setenv.bash first.")
endif()
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")

if(EXISTS "${ASCEND_HOME_PATH}/bin/bisheng")
    set(BISHENG "${ASCEND_HOME_PATH}/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
elseif(EXISTS "${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng")
    set(BISHENG "${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
else()
    message(FATAL_ERROR
        "Bisheng compiler not found under ${ASCEND_HOME_PATH}/bin/bisheng or "
        "${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng")
endif()
message(STATUS "Bisheng compiler path: ${BISHENG}")
set(CMAKE_ASC_COMPILER "${BISHENG}" CACHE FILEPATH "ASC compiler" FORCE)

set(CMAKE_PREFIX_PATH ${ASCEND_HOME_PATH} ${CMAKE_PREFIX_PATH})
find_package(ASC REQUIRED)

project(opsgnn LANGUAGES CXX ASC)

# ASC's fwk_modules/config.cmake clears the normal CXX DEBUG/RELEASE flags.
# Restore the compiler/user cache values after language initialization so a
# Release wheel really optimizes the PyTorch Host and bindings. Device kernels
# retain their explicit ASC flags below.
foreach(OPSGNN_HOST_CONFIG DEBUG RELEASE)
    get_property(OPSGNN_HOST_CONFIG_FLAGS CACHE CMAKE_CXX_FLAGS_${OPSGNN_HOST_CONFIG} PROPERTY VALUE)
    set(CMAKE_CXX_FLAGS_${OPSGNN_HOST_CONFIG} "${OPSGNN_HOST_CONFIG_FLAGS}")
endforeach()
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O2")

set(ASCEND_INCLUDE_DIRS
    ${ASCEND_HOME_PATH}/include/ascendc/basic_api
    ${ASCEND_HOME_PATH}/include/ascendc
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/simt_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/utils/base
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/tiling
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/tiling/platform
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/interface
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/basic_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ascendc/include/basic_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ascendc/include/basic_api/impl
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/basic_api
    ${ASCEND_HOME_PATH}/acllib/include
)

set(ASCEND_LINK_DIRS
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ccec_compiler/lib
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/lib64
    ${ASCEND_HOME_PATH}/acllib/lib64
)

set(ASCEND_LIBRARIES
    platform
    ascendcl
    runtime
    ascendc_runtime
    tiling_api
    unified_dlog
    dl
    stdc++
    m
)

###############################################################################
# Python3 Configuration
###############################################################################

if(WITH_PYTHON)
    if(DEFINED ENV{Python3_ROOT_DIR})
        set(Python3_ROOT_DIR "$ENV{Python3_ROOT_DIR}" CACHE PATH "Python3 root directory")
        message(STATUS "Python3_ROOT_DIR set from environment: ${Python3_ROOT_DIR}")
    else()
        execute_process(
            COMMAND python3 -c "import sys; print(sys.prefix)"
            OUTPUT_VARIABLE Python3_ROOT_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(Python3_ROOT_DIR "${Python3_ROOT_DIR}" CACHE PATH "Python3 root directory")
        message(STATUS "Python3_ROOT_DIR auto-detected: ${Python3_ROOT_DIR}")
    endif()

    execute_process(
        COMMAND "${Python3_ROOT_DIR}/bin/python3" -c "import sysconfig; print(sysconfig.get_paths()['include'])"
        OUTPUT_VARIABLE Python3_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND "${Python3_ROOT_DIR}/bin/python3" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
        OUTPUT_VARIABLE Python3_LIBDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND "${Python3_ROOT_DIR}/bin/python3" -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))"
        OUTPUT_VARIABLE Python3_LDLIBRARY
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(Python3_LIBRARIES "${Python3_LIBDIR}/${Python3_LDLIBRARY}")
    # Some Conda Python builds retain a removed static LDLIBRARY in sysconfig.
    # Relocatable builds may also retain their original LIBDIR. Search both
    # locations for the shared runtime before creating the imported target.
    if(NOT EXISTS "${Python3_LIBRARIES}" OR Python3_LDLIBRARY MATCHES "\\.a$")
        execute_process(
            COMMAND "${Python3_ROOT_DIR}/bin/python3" -c
                "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))"
            OUTPUT_VARIABLE Python3_LDVERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND "${Python3_ROOT_DIR}/bin/python3" -c "import sys; print(sys.base_prefix)"
            OUTPUT_VARIABLE Python3_BASE_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(OPSGNN_SAVED_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
        find_library(OPSGNN_PYTHON_SHARED_LIBRARY
            NAMES "python${Python3_LDVERSION}"
            PATHS "${Python3_LIBDIR}" "${Python3_BASE_PREFIX}/lib"
            NO_DEFAULT_PATH REQUIRED)
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${OPSGNN_SAVED_LIBRARY_SUFFIXES}")
        set(Python3_LIBRARIES "${OPSGNN_PYTHON_SHARED_LIBRARY}")
    endif()
    set(Python3_FOUND TRUE)
    set(OPSGNN_PYTHON_EXECUTABLE "${Python3_ROOT_DIR}/bin/python3")

    add_library(Python3::Python SHARED IMPORTED)
    set_property(TARGET Python3::Python PROPERTY IMPORTED_LOCATION "${Python3_LIBRARIES}")
    set_property(TARGET Python3::Python PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Python3_INCLUDE_DIRS}")

    message(STATUS "Manually detected Python3:")
    message(STATUS "  Include: ${Python3_INCLUDE_DIRS}")
    message(STATUS "  Library: ${Python3_LIBRARIES}")
else()
    set(OPSGNN_PYTHON_EXECUTABLE python3)
endif()

###############################################################################
# PyTorch Configuration
###############################################################################

execute_process(
    COMMAND "${OPSGNN_PYTHON_EXECUTABLE}" -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

execute_process(
    COMMAND "${OPSGNN_PYTHON_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_INSTALL_PREFIX)
    set(ENV{TORCH_INSTALL_PREFIX} "${TORCH_INSTALL_PREFIX}")
endif()

if(TORCH_CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}" "${CMAKE_PREFIX_PATH}")
endif()

# Suppress kineto warning for CPU torch: pre-set to an existing torch lib
# so find_library skips the search and no "not found" warning is emitted
set(kineto_LIBRARY "${TORCH_INSTALL_PREFIX}/lib/libtorch.so" CACHE FILEPATH "" FORCE)
find_package(Torch REQUIRED)

message(STATUS "Found PyTorch: ${TORCH_INSTALL_PREFIX}")

execute_process(
    COMMAND "${OPSGNN_PYTHON_EXECUTABLE}" -c "import importlib.util, os; spec = importlib.util.find_spec('torch_npu'); print(os.path.dirname(spec.origin) if spec and spec.origin else '')"
    OUTPUT_VARIABLE TORCH_NPU_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT TORCH_NPU_INSTALL_PREFIX)
    message(FATAL_ERROR "torch_npu is required when building ops_gnn NPU extension")
endif()
set(TORCH_NPU_INCLUDE_DIR "${TORCH_NPU_INSTALL_PREFIX}/include")
set(TORCH_NPU_LIBRARY_DIR "${TORCH_NPU_INSTALL_PREFIX}/lib")
if(NOT EXISTS "${TORCH_NPU_INCLUDE_DIR}")
    message(FATAL_ERROR "torch_npu include directory not found: ${TORCH_NPU_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${TORCH_NPU_LIBRARY_DIR}")
    message(FATAL_ERROR "torch_npu library directory not found: ${TORCH_NPU_LIBRARY_DIR}")
endif()
message(STATUS "Found Torch NPU: ${TORCH_NPU_INSTALL_PREFIX}")

###############################################################################
# Source Files
###############################################################################

function(collect_npu_arch_sources ARCH_NAME OUT_KERNEL_FILES OUT_KERNEL_HEADERS OUT_HOST_FILES)
    file(GLOB_RECURSE ARCH_KERNEL_FILES
        CONFIGURE_DEPENDS
        csrc/npu/*/op_kernel/${ARCH_NAME}/*.asc
        csrc/npu/*/op_kernel/${ARCH_NAME}/*.cpp
    )
    file(GLOB_RECURSE ARCH_KERNEL_HEADERS
        CONFIGURE_DEPENDS
        csrc/npu/*/op_kernel/${ARCH_NAME}/*.h
    )
    file(GLOB ARCH_KERNEL_DIRS
        CONFIGURE_DEPENDS
        LIST_DIRECTORIES true
        csrc/npu/*/op_kernel/${ARCH_NAME}
    )

    set(ARCH_HOST_FILES)
    foreach(KERNEL_DIR IN LISTS ARCH_KERNEL_DIRS)
        get_filename_component(OP_KERNEL_DIR "${KERNEL_DIR}" DIRECTORY)
        get_filename_component(OPERATOR_DIR "${OP_KERNEL_DIR}" DIRECTORY)
        file(GLOB OPERATOR_HOST_FILES
            CONFIGURE_DEPENDS
            "${OPERATOR_DIR}/op_host/*.cpp"
        )
        list(APPEND ARCH_HOST_FILES ${OPERATOR_HOST_FILES})
    endforeach()
    list(REMOVE_DUPLICATES ARCH_HOST_FILES)

    set(${OUT_KERNEL_FILES} ${ARCH_KERNEL_FILES} PARENT_SCOPE)
    set(${OUT_KERNEL_HEADERS} ${ARCH_KERNEL_HEADERS} PARENT_SCOPE)
    set(${OUT_HOST_FILES} ${ARCH_HOST_FILES} PARENT_SCOPE)
endfunction()

collect_npu_arch_sources(
    arch22 NPU_ARCH22_FILES NPU_ARCH22_HEADERS NPU_ARCH22_HOST_FILES)
collect_npu_arch_sources(
    arch35 NPU_ARCH35_FILES NPU_ARCH35_HEADERS NPU_ARCH35_HOST_FILES)

if(NPU_ARCH STREQUAL "dav-2201")
    # A2/A3: 只编译 arch22 算子
    set(NPU_ASC_FILES ${NPU_ARCH22_FILES})
    set(NPU_KERNEL_HEADERS ${NPU_ARCH22_HEADERS})
    set(NPU_CPP_INTERFACE_FILES ${NPU_ARCH22_HOST_FILES})
else()
    # 950: 只编译 arch35 算子
    set(NPU_ASC_FILES ${NPU_ARCH35_FILES})
    set(NPU_KERNEL_HEADERS ${NPU_ARCH35_HEADERS})
    set(NPU_CPP_INTERFACE_FILES ${NPU_ARCH35_HOST_FILES})
endif()

set(OPERATOR_SOURCES
    csrc/pybind.cpp
    ${NPU_CPP_INTERFACE_FILES}
)

###############################################################################
# NPU Kernel Compilation
###############################################################################

set(NPU_KERNEL_LIB "")

if(NPU_ASC_FILES)
    if(NPU_ARCH STREQUAL "dav-2201")
        set(KERNEL_LIBRARY_LABEL "A2-A3")
    else()
        set(KERNEL_LIBRARY_LABEL "950")
    endif()
    set(KERNEL_DEVICE_LABEL "${KERNEL_LIBRARY_LABEL}")
    set(NPU_KERNEL_LIB_NAME "opsgnn_npu_kernel-${KERNEL_LIBRARY_LABEL}")
    set(OUTPUT_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output/kernel")
    set(NPU_KERNEL_LIB "${OUTPUT_KERNEL_DIR}/lib${NPU_KERNEL_LIB_NAME}.so")

    set(NPU_KERNEL_OBJECTS "")
    foreach(ASC_FILE ${NPU_ASC_FILES})
        get_filename_component(FILE_NAME ${ASC_FILE} NAME_WE)
        get_filename_component(FILE_EXT ${ASC_FILE} EXT)
        set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.o")
        list(APPEND NPU_KERNEL_OBJECTS ${OBJ_FILE})

        set(ASC_LANG_FLAG "")
        if("${FILE_EXT}" STREQUAL ".cpp")
            set(ASC_LANG_FLAG "--asc-aicore-lang")
        endif()
        if(ASC_FILE MATCHES "/csrc/npu/torch_sparse_spmm/")
            list(APPEND ASC_LANG_FLAG "-std=c++17")
        endif()

        set(ASC_INCLUDE_FLAGS "")
        foreach(INC_DIR ${ASCEND_INCLUDE_DIRS})
            list(APPEND ASC_INCLUDE_FLAGS "-I${INC_DIR}")
        endforeach()
        list(APPEND ASC_INCLUDE_FLAGS
            "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc/npu"
        )

        add_custom_command(
            OUTPUT ${OBJ_FILE}
            COMMAND ${CMAKE_ASC_COMPILER}
            ARGS ${ASC_LANG_FLAG}
                -c ${ASC_FILE}
                --npu-arch=${NPU_ARCH}
                -O2
                -fPIC
                ${ASC_INCLUDE_FLAGS}
                -o ${OBJ_FILE}
            MAIN_DEPENDENCY ${ASC_FILE}
            DEPENDS ${NPU_KERNEL_HEADERS}
            COMMENT "Compiling ASC kernel: ${FILE_NAME}${FILE_EXT}"
        )
    endforeach()

    set(ASC_LINK_FLAGS "")
    foreach(LINK_DIR ${ASCEND_LINK_DIRS})
        list(APPEND ASC_LINK_FLAGS "-L${LINK_DIR}")
    endforeach()
    foreach(LINK_DIR ${ASCEND_LINK_DIRS})
        list(APPEND ASC_LINK_FLAGS "-Wl,-rpath,${LINK_DIR}")
    endforeach()
    foreach(LIB_NAME ${ASCEND_LIBRARIES})
        list(APPEND ASC_LINK_FLAGS "-l${LIB_NAME}")
    endforeach()

    add_custom_command(
        OUTPUT ${NPU_KERNEL_LIB}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_KERNEL_DIR}
        COMMAND ${CMAKE_ASC_COMPILER}
        ARGS -shared -fPIC -o ${NPU_KERNEL_LIB}
            -Wl,-soname,lib${NPU_KERNEL_LIB_NAME}.so
            ${NPU_KERNEL_OBJECTS}
            ${ASC_LINK_FLAGS}
        DEPENDS ${NPU_KERNEL_OBJECTS}
        COMMENT "Linking NPU kernel library: ${NPU_KERNEL_LIB}"
    )

    add_custom_target(npu_kernel_target ALL DEPENDS ${NPU_KERNEL_LIB})
endif()

###############################################################################
# Main Library Target
###############################################################################

add_library(_pybind SHARED ${OPERATOR_SOURCES})
if(NPU_ARCH STREQUAL "dav-2201")
    target_compile_definitions(_pybind PRIVATE OPSGNN_DAV_2201)
endif()

target_compile_definitions(_pybind PRIVATE 
    WITH_PYTHON WITH_NPU AT_PARALLEL_OPENMP 
    TORCH_API_INCLUDE_EXTENSION_H TORCH_EXTENSION_NAME=_pybind
)

target_include_directories(_pybind PRIVATE
    ${ASCEND_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/npu
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/pybind11
    ${TORCH_NPU_INCLUDE_DIR}
)

if(WITH_PYTHON)
    target_include_directories(_pybind PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries(_pybind PRIVATE Python3::Python)
endif()

target_link_libraries(_pybind PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so
    ${TORCH_NPU_LIBRARY_DIR}/libtorch_npu.so
)

if(TORCH_NPU_LIBRARY_DIR)
    target_link_directories(_pybind PRIVATE ${TORCH_NPU_LIBRARY_DIR})
    target_link_options(_pybind PRIVATE "-Wl,-rpath,${TORCH_NPU_LIBRARY_DIR}")
endif()

target_link_directories(_pybind PRIVATE ${ASCEND_LINK_DIRS})
target_link_libraries(_pybind PRIVATE ${ASCEND_LIBRARIES})

if(NPU_KERNEL_LIB)
    add_dependencies(_pybind npu_kernel_target)
    target_link_libraries(_pybind PRIVATE ${NPU_KERNEL_LIB})
endif()

set_target_properties(_pybind PROPERTIES 
    EXPORT_NAME OpsGNN
    LINK_FLAGS "-s"
    LIBRARY_OUTPUT_NAME "_pybind"
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

add_custom_command(TARGET _pybind POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/python/ops_gnn/${KERNEL_DEVICE_LABEL}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:_pybind>"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/ops_gnn/${KERNEL_DEVICE_LABEL}/_pybind.so"
    COMMENT "Copying _pybind.so to python/ops_gnn/${KERNEL_DEVICE_LABEL}/"
)

if(NPU_KERNEL_LIB)
    add_custom_command(TARGET _pybind POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/python/ops_gnn/${KERNEL_DEVICE_LABEL}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NPU_KERNEL_LIB}"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/ops_gnn/${KERNEL_DEVICE_LABEL}/lib${NPU_KERNEL_LIB_NAME}.so"
        COMMENT "Copying NPU kernel library to python/ops_gnn/"
    )
endif()

###############################################################################
# Installation
###############################################################################

if(WITH_PYTHON AND OPSGNN_BUILD_WHEEL)
    add_custom_target(wheel ALL
        COMMAND ${CMAKE_COMMAND} -E env
            "NPU_ARCH=${NPU_ARCH}" "CANN_TARGET=${CANN_TARGET}"
            "OPSGNN_PREBUILT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "${OPSGNN_PYTHON_EXECUTABLE}" -m pip wheel . --no-deps --no-build-isolation
            -w "${CMAKE_CURRENT_SOURCE_DIR}/output/whl"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DEPENDS _pybind
        COMMENT "Packaging ops-gnn wheel for ${CANN_TARGET}"
        VERBATIM
    )
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/output" CACHE PATH "Install path prefix" FORCE)
set(OPSGNN_CMAKECONFIG_INSTALL_DIR "share/cmake/OpsGNN" CACHE STRING "install path for OpsGNNConfig.cmake")

configure_package_config_file(cmake/OpsGNNConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfig.cmake"
    INSTALL_DESTINATION ${OPSGNN_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfigVersion.cmake
    VERSION ${OPSGNN_VERSION}
    COMPATIBILITY AnyNewerVersion)

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfigVersion.cmake
    DESTINATION ${OPSGNN_CMAKECONFIG_INSTALL_DIR})

install(TARGETS _pybind
    EXPORT OpsGNNTargets
    LIBRARY DESTINATION kernel
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if(NPU_KERNEL_LIB)
    install(FILES ${NPU_KERNEL_LIB} DESTINATION kernel
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()

install(EXPORT OpsGNNTargets
    NAMESPACE OpsGNN::
    DESTINATION ${OPSGNN_CMAKECONFIG_INSTALL_DIR})

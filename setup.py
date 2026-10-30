"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""
import os
import shutil
import subprocess

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel

__version__ = '0.1.0'
URL = 'https://gitcode.com/cann/ops-gnn'

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'


TARGET_ARCHS = {'a2': 'dav-2201', 'a3': 'dav-2201', '950': 'dav-3510'}
KERNEL_LABELS = {'a2': 'A2-A3', 'a3': 'A2-A3', '950': '950'}


def resolve_build_targets():
    npu_arch = os.getenv('NPU_ARCH') or os.getenv('TARGET_NPU_ARCH')
    cann_target = os.getenv('CANN_TARGET', '').lower()
    if npu_arch and npu_arch not in TARGET_ARCHS.values():
        raise ValueError(f'Unsupported NPU_ARCH: {npu_arch}')
    if cann_target and cann_target not in TARGET_ARCHS:
        raise ValueError(f'Unsupported CANN_TARGET: {cann_target}; use 950/a2/a3')
    if cann_target and npu_arch and TARGET_ARCHS[cann_target] != npu_arch:
        raise ValueError(f'NPU_ARCH={npu_arch} conflicts with CANN_TARGET={cann_target}')
    if cann_target:
        return (cann_target,)
    if npu_arch:
        return ('950',) if npu_arch == 'dav-3510' else ('a2', 'a3')
    return ('a2', 'a3', '950')


def build_with_cmake(cann_target):
    npu_arch = TARGET_ARCHS[cann_target]
    if 'ASCEND_HOME_PATH' not in os.environ:
        raise EnvironmentError("ASCEND_HOME_PATH environment variable not set. Please source set_env.sh first.")

    print(f'NPU 架构: {npu_arch}')
    cmake_build_dir = f'build/cmake_python_{cann_target}'
    # 删除 CMakeCache.txt，避免 pip 拷贝源码到临时目录后
    # cmake 检测到缓存的源路径与当前路径不一致而报错
    cache_file = os.path.join(cmake_build_dir, 'CMakeCache.txt')
    if os.path.exists(cache_file):
        os.remove(cache_file)
    os.makedirs(cmake_build_dir, exist_ok=True)
    
    cmake_cmd = [
        'cmake',
        '-DWITH_PYTHON=ON',
        '-DOPSGNN_BUILD_WHEEL=OFF',
        '-DCMAKE_BUILD_TYPE=Release',
        f'-DNPU_ARCH={npu_arch}',
        f'-DCANN_TARGET={cann_target}',
        '-S', '.',
        '-B', cmake_build_dir,
    ]
    
    print(f'Running CMake: {" ".join(cmake_cmd)}')
    subprocess.run(cmake_cmd, check=True)
    
    make_cmd = ['make', '-C', cmake_build_dir, '-j4']
    print(f'Running Make: {" ".join(make_cmd)}')
    subprocess.run(make_cmd, check=True)
    
    npu_kernel_lib = os.path.join('output', 'kernel', f'libopsgnn_npu_kernel-{KERNEL_LABELS[cann_target]}.so')
    if os.path.exists(npu_kernel_lib):
        print(f'NPU kernel library generated at: {npu_kernel_lib}')
    
    pybind_lib = os.path.join(cmake_build_dir, 'lib_pybind.so')
    if os.path.exists(pybind_lib):
        pybind_dest = os.path.join('python', 'ops_gnn', KERNEL_LABELS[cann_target], '_pybind.so')
        os.makedirs(os.path.dirname(pybind_dest), exist_ok=True)
        shutil.copy2(pybind_lib, pybind_dest)
        print(f'Copied _pybind.so to: {pybind_dest}')


def reuse_cmake_libraries(build_dir):
    with open(os.path.join(build_dir, 'CMakeCache.txt'), encoding='utf-8') as cache_file:
        cache = {}
        for line in cache_file:
            if line.startswith(('#', '//')) or '=' not in line:
                continue
            key, value = line.strip().split('=', 1)
            cache[key.split(':', 1)[0]] = value
    expected = {
        'NPU_ARCH': TARGET_ARCHS[BUILD_TARGETS[0]],
        'CANN_TARGET': BUILD_TARGETS[0],
    }
    for key, value in expected.items():
        if cache.get(key) != value:
            raise ValueError(f'Prebuilt CMake {key} does not match this wheel build')
    source_dir = cache.get('CMAKE_HOME_DIRECTORY')
    if not source_dir or not os.path.isdir(source_dir):
        raise ValueError('Prebuilt CMake source directory is missing or unavailable')
    # Older pip versions package a temporary source copy. Read native artifacts
    # from the original CMake source/build directories instead of the copy.
    label = KERNEL_LABELS[BUILD_TARGETS[0]]
    destination = os.path.join('python', 'ops_gnn', label)
    os.makedirs(destination, exist_ok=True)
    kernel_filename = f'libopsgnn_npu_kernel-{KERNEL_LABELS[BUILD_TARGETS[0]]}.so'
    libraries = (
        (os.path.join(build_dir, 'lib_pybind.so'), '_pybind.so'),
        (os.path.join(source_dir, 'output', 'kernel', kernel_filename), kernel_filename),
    )
    for source, filename in libraries:
        shutil.copy2(source, os.path.join(destination, filename))
    print(f'Reusing native libraries from: {build_dir}')


def prepare_native_libraries():
    prebuilt_root = os.getenv('OPSGNN_PREBUILT_ROOT')
    if prebuilt_root:
        for target in BUILD_TARGETS:
            label = KERNEL_LABELS[target]
            destination = os.path.join('python', 'ops_gnn', label)
            os.makedirs(destination, exist_ok=True)
            libraries = (
                (os.path.join(prebuilt_root, f'variant_{target}', 'lib_pybind.so'), '_pybind.so'),
                (os.path.join(os.getenv('OPSGNN_SOURCE_ROOT', '.'), 'output', 'kernel',
                              f'libopsgnn_npu_kernel-{KERNEL_LABELS[target]}.so'),
                 f'libopsgnn_npu_kernel-{KERNEL_LABELS[target]}.so'),
            )
            for source, filename in libraries:
                shutil.copy2(source, os.path.join(destination, filename))
    prebuilt_dir = os.getenv('OPSGNN_PREBUILT_DIR')
    if prebuilt_dir:
        reuse_cmake_libraries(prebuilt_dir)
    elif not prebuilt_root:
        for target in BUILD_TARGETS:
            build_with_cmake(target)


BUILD_TARGETS = resolve_build_targets()

if not BUILD_DOCS:
    prepare_native_libraries()


class CannEditableWheel(editable_wheel):
    def run(self):
        # Package the already compiled libraries before creating the editable wheel.
        wheel_command = self.reinitialize_command('bdist_wheel')
        wheel_command.dist_dir = os.path.abspath(os.path.join('output', 'whl'))
        self.run_command('bdist_wheel')
        super().run()


class CannDevelop(develop):
    def run(self):
        # pip versions before PEP 660 use setup.py develop for editable installs.
        wheel_command = self.reinitialize_command('bdist_wheel')
        wheel_command.dist_dir = os.path.abspath(os.path.join('output', 'whl'))
        self.run_command('bdist_wheel')
        super().run()


class CustomBuildPy(build_py):
    def run(self):
        # A previous build may have populated other device directories.
        for label in ('A2', 'A3', 'A2-A3', '950'):
            shutil.rmtree(os.path.join(self.build_lib, 'ops_gnn', label), ignore_errors=True)
        build_py.run(self)
        for label in ('A2', 'A3', 'A2-A3', '950'):
            if label not in {KERNEL_LABELS[target] for target in BUILD_TARGETS}:
                shutil.rmtree(os.path.join(self.build_lib, 'ops_gnn', label), ignore_errors=True)

        for target in BUILD_TARGETS:
            label = KERNEL_LABELS[target]
            for library_name in ('_pybind.so', f'libopsgnn_npu_kernel-{KERNEL_LABELS[target]}.so'):
                library_src = os.path.join('python', 'ops_gnn', label, library_name)
                library_dest = os.path.join(self.build_lib, 'ops_gnn', label, library_name)
                os.makedirs(os.path.dirname(library_dest), exist_ok=True)
                shutil.copy2(library_src, library_dest)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name='ops-gnn',
    version=__version__,
    license='CANN Open Software License Agreement Version 2.0',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    python_requires='>=3.9',
    ext_modules=[],
    distclass=BinaryDistribution,
    cmdclass={
        'build_py': CustomBuildPy,
        'editable_wheel': CannEditableWheel,
        'develop': CannDevelop,
    },
    packages=find_packages('python'),
    package_dir={'': 'python'},
    package_data={'ops_gnn': [
        f'{KERNEL_LABELS[target]}/{filename}'
        for target in BUILD_TARGETS
        for filename in ('_pybind.so', f'libopsgnn_npu_kernel-{KERNEL_LABELS[target]}.so')
    ]},
    include_package_data=True,
)

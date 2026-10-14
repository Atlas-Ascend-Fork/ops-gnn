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
import warnings

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel
from wheel.bdist_wheel import bdist_wheel

__version__ = '0.1.0'
URL = 'https://gitcode.com/cann/ops-gnn'

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'


def detect_cann_target():
    npu_smi = shutil.which(os.getenv('NPU_SMI_BIN', 'npu-smi'))
    reason = '未找到 npu-smi'
    if npu_smi is not None:
        try:
            result = subprocess.run(
                [npu_smi, 'info', '-m'], check=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=10)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            reason = f'npu-smi info 执行失败：{exc}'
        else:
            product_info = result.stdout.lower()
            if '910_93' in product_info or '910c' in product_info:
                return 'a3'
            if '910b' in product_info:
                return 'a2'
            if '950' in product_info:
                return '950'
            raise ValueError('不支持的芯片型号，仅支持 950 / A2 / A3。')
    warnings.warn(
        f'由于检测不到 NPU 设备（{reason}），当前默认以 Ascend 950 设备进行编译'
        '（dav-3510 / arch35）。可设置 CANN_TARGET=950/a2/a3 显式指定目标设备。',
        RuntimeWarning, stacklevel=2)
    return '950'


def resolve_npu_target():
    target_archs = {'950': 'dav-3510', 'a2': 'dav-2201', 'a3': 'dav-2201'}
    npu_arch = os.getenv('NPU_ARCH') or os.getenv('TARGET_NPU_ARCH')
    cann_target = os.getenv('CANN_TARGET')
    if npu_arch and npu_arch not in target_archs.values():
        raise ValueError(f'Unsupported NPU_ARCH: {npu_arch}')
    if cann_target:
        cann_target = cann_target.lower()
        if cann_target not in target_archs:
            raise ValueError(f'Unsupported CANN_TARGET: {cann_target}; use 950/a2/a3')
    else:
        # dav-2201 is shared by A2/A3 and requires a device label or detection.
        cann_target = '950' if npu_arch == 'dav-3510' else detect_cann_target()
    target_arch = target_archs.get(cann_target)
    if target_arch is None:
        raise ValueError(f'Unsupported CANN_TARGET: {cann_target}; use 950/a2/a3')
    if npu_arch and npu_arch != target_arch:
        raise ValueError(
            f'NPU_ARCH={npu_arch} 与 CANN_TARGET={cann_target} 不匹配；'
            '使用 CANN_TARGET=950/a2/a3 指定目标设备。')
    return target_arch, cann_target


def build_with_cmake(npu_arch):
    if 'ASCEND_HOME_PATH' not in os.environ:
        raise EnvironmentError("ASCEND_HOME_PATH environment variable not set. Please source set_env.sh first.")

    print(f'NPU 架构: {npu_arch}')
    cmake_build_dir = f'build/cmake_python_{npu_arch}'
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
        f'-DCANN_TARGET={CANN_TARGET}',
        '-S', '.',
        '-B', cmake_build_dir,
    ]
    
    print(f'Running CMake: {" ".join(cmake_cmd)}')
    subprocess.run(cmake_cmd, check=True)
    
    make_cmd = ['make', '-C', cmake_build_dir, '-j4']
    print(f'Running Make: {" ".join(make_cmd)}')
    subprocess.run(make_cmd, check=True)
    
    npu_kernel_lib = os.path.join('output', 'kernel', 'libopsgnn_npu_kernel.so')
    if os.path.exists(npu_kernel_lib):
        print(f'NPU kernel library generated at: {npu_kernel_lib}')
    
    pybind_lib = os.path.join(cmake_build_dir, 'lib_pybind.so')
    if os.path.exists(pybind_lib):
        pybind_dest = os.path.join('python', 'ops_gnn', '_pybind.so')
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
        'NPU_ARCH': NPU_ARCH,
        'CANN_TARGET': CANN_TARGET,
    }
    for key, value in expected.items():
        if cache.get(key) != value:
            raise ValueError(f'Prebuilt CMake {key} does not match this wheel build')
    source_dir = cache.get('CMAKE_HOME_DIRECTORY')
    if not source_dir or not os.path.isdir(source_dir):
        raise ValueError('Prebuilt CMake source directory is missing or unavailable')
    # Older pip versions package a temporary source copy. Read native artifacts
    # from the original CMake source/build directories instead of the copy.
    libraries = (
        (os.path.join(build_dir, 'lib_pybind.so'), '_pybind.so'),
        (os.path.join(source_dir, 'output', 'kernel', 'libopsgnn_npu_kernel.so'), 'libopsgnn_npu_kernel.so'),
    )
    for source, filename in libraries:
        shutil.copy2(source, os.path.join('python', 'ops_gnn', filename))
    print(f'Reusing native libraries from: {build_dir}')


NPU_ARCH, CANN_TARGET = resolve_npu_target()

if not BUILD_DOCS:
    prebuilt_dir = os.getenv('OPSGNN_PREBUILT_DIR')
    if prebuilt_dir:
        reuse_cmake_libraries(prebuilt_dir)
    else:
        build_with_cmake(NPU_ARCH)


class CannBdistWheel(bdist_wheel):
    def run(self):
        super().run()
        # Preserve canonical internal metadata and use underscores in filenames.
        for index, (command, python_version, path) in enumerate(self.distribution.dist_files):
            if command != 'bdist_wheel':
                continue
            filename = os.path.basename(path).replace('+cann.', '+cann_')
            renamed = os.path.join(os.path.dirname(path), filename)
            if renamed != path:
                os.replace(path, renamed)
                self.distribution.dist_files[index] = (command, python_version, renamed)


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
        build_py.run(self)

        for library_name in ('_pybind.so', 'libopsgnn_npu_kernel.so'):
            library_src = os.path.join('python', 'ops_gnn', library_name)
            if os.path.exists(library_src):
                library_dest = os.path.join(self.build_lib, 'ops_gnn', library_name)
                os.makedirs(os.path.dirname(library_dest), exist_ok=True)
                shutil.copy2(library_src, library_dest)
                print(f'Copied {library_name} to build lib: {library_dest}')


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name='ops-gnn',
    version=f'{__version__}+cann.{CANN_TARGET}',
    license='CANN Open Software License Agreement Version 2.0',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    python_requires='>=3.9',
    ext_modules=[],
    distclass=BinaryDistribution,
    cmdclass={
        'build_py': CustomBuildPy,
        'bdist_wheel': CannBdistWheel,
        'editable_wheel': CannEditableWheel,
        'develop': CannDevelop,
    },
    packages=find_packages('python'),
    package_dir={'': 'python'},
    package_data={'ops_gnn': ['*.so']},
    include_package_data=True,
)

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Architecture-independent build and wheel packaging regressions."""

import contextlib
from email.parser import Parser
import importlib.util
import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERNEL_LABELS = {"A2": "A2-A3", "A3": "A2-A3", "A2-A3": "A2-A3", "950": "950"}
BUILD_ENVIRONMENT = (
    "BUILD_DOCS", "NPU_ARCH", "TARGET_NPU_ARCH", "CANN_TARGET", "NPU_SMI_BIN",
    "OPSGNN_PREBUILT_DIR", "OPSGNN_PREBUILT_ROOT", "OPSGNN_SOURCE_ROOT",
)


@unittest.skipUnless(importlib.util.find_spec("setuptools"), "packaging tests require setuptools")
class TestPackaging(unittest.TestCase):
    @staticmethod
    def write_prebuilt_tree(directory, target, architecture):
        source = directory / "native_source"
        kernel_directory = source / "output" / "kernel"
        kernel_directory.mkdir(parents=True)
        build = directory / "prebuilt"
        build.mkdir()
        cache = (
            f"NPU_ARCH:STRING={architecture}\n"
            f"CANN_TARGET:STRING={target}\n"
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n"
        )
        (build / "CMakeCache.txt").write_text(cache, encoding="utf-8")
        (build / "lib_pybind.so").write_bytes(b"selected pybind")
        filename = f"libopsgnn_npu_kernel-{KERNEL_LABELS[target.upper()]}.so"
        (kernel_directory / filename).write_bytes(b"selected kernel")
        return build

    @staticmethod
    def write_wheel_fixture(directory):
        package = directory / "python" / "ops_gnn"
        for label in ("A2-A3", "950"):
            native = package / label
            native.mkdir()
            (native / "_pybind.so").write_bytes(f"pybind:{label}".encode())
            filename = f"libopsgnn_npu_kernel-{KERNEL_LABELS[label]}.so"
            (native / filename).write_bytes(f"kernel:{KERNEL_LABELS[label]}".encode())
        for label in ("A2", "A3"):
            legacy = package / label
            legacy.mkdir()
            (legacy / "_pybind.so").write_bytes(b"obsolete device binding")
            (legacy / f"libopsgnn_npu_kernel-{label}.so").write_bytes(b"obsolete device kernel")
        (package / "_pybind.so").write_bytes(b"obsolete unlabelled pybind")
        (package / "libopsgnn_npu_kernel.so").write_bytes(b"obsolete unlabelled kernel")

    def setUp(self):
        self.contexts = contextlib.ExitStack()
        self.addCleanup(self.contexts.close)
        self.temporary = Path(self.contexts.enter_context(tempfile.TemporaryDirectory(prefix="opsgnn-packaging-")))
        self.contexts.callback(os.chdir, Path.cwd())
        self.contexts.enter_context(patch.dict(os.environ, dict(os.environ), clear=True))
        self.output = io.StringIO()
        self.contexts.enter_context(contextlib.redirect_stdout(self.output))
        self.contexts.enter_context(contextlib.redirect_stderr(self.output))
        self.case_number = 0

    def load_setup(self, environment=None):
        """Load the real setup script into an isolated package without compiling."""
        import setuptools

        self.case_number += 1
        directory = self.temporary / str(self.case_number)
        package = directory / "python" / "ops_gnn"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('"""Packaging fixture."""\n', encoding="utf-8")
        (directory / "setup.py").write_text("# Packaging fixture\n", encoding="utf-8")
        for name in ("MANIFEST.in", "README.md", "LICENSE"):
            shutil.copy2(PROJECT_ROOT / name, directory / name)
        os.chdir(directory)
        for variable in BUILD_ENVIRONMENT:
            os.environ.pop(variable, None)
        os.environ["BUILD_DOCS"] = "1"
        os.environ.update(environment or {})
        metadata = {}

        def capture_setup(**attributes):
            metadata.update(attributes)

        with patch.object(setuptools, "setup", capture_setup):
            spec = importlib.util.spec_from_file_location("packaging_setup", PROJECT_ROOT / "setup.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module, metadata, directory

    def build_wheel(self, module, metadata, directory):
        distribution = module.BinaryDistribution({
            name: value for name, value in metadata.items() if name != "distclass"
        })
        distribution.script_name = "setup.py"
        build = distribution.get_command_obj("build")
        build.ensure_finalized()
        for label in ("A2", "A3", "A2-A3", "950"):
            stale = Path(build.build_lib) / "ops_gnn" / label / "stale.so"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_bytes(b"previous build")
        wheel = distribution.get_command_obj("bdist_wheel")
        wheel.dist_dir = str(directory / "wheelhouse")
        wheel.ensure_finalized()
        wheel.run()
        artifacts = list((directory / "wheelhouse").glob("*.whl"))
        self.assertEqual(len(artifacts), 1)
        return artifacts[0]

    def assert_wheel_contents(self, wheel_path, version, labels):
        self.assertTrue(wheel_path.name.startswith(f"ops_gnn-{version}-"))
        with ZipFile(wheel_path) as archive:
            native_files = {name for name in archive.namelist() if name.endswith(".so")}
            expected = {
                f"ops_gnn/{label}/{filename}"
                for label in labels
                for filename in ("_pybind.so", f"libopsgnn_npu_kernel-{KERNEL_LABELS[label]}.so")
            }
            self.assertEqual(native_files, expected)
            for label in labels:
                self.assertEqual(archive.read(f"ops_gnn/{label}/_pybind.so"), f"pybind:{label}".encode())
                filename = f"ops_gnn/{label}/libopsgnn_npu_kernel-{KERNEL_LABELS[label]}.so"
                self.assertEqual(archive.read(filename), f"kernel:{KERNEL_LABELS[label]}".encode())
            metadata_path = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            wheel_metadata = Parser().parsestr(archive.read(metadata_path).decode())
            self.assertEqual(wheel_metadata["Version"], version)

    def test_resolve_build_targets(self):
        cases = [
            ({}, ("a2", "a3", "950")),
            ({"CANN_TARGET": "a2"}, ("a2",)),
            ({"CANN_TARGET": "A3"}, ("a3",)),
            ({"CANN_TARGET": "950"}, ("950",)),
            ({"NPU_ARCH": "dav-2201"}, ("a2", "a3")),
            ({"NPU_ARCH": "dav-3510"}, ("950",)),
            ({"TARGET_NPU_ARCH": "dav-2201"}, ("a2", "a3")),
            ({"TARGET_NPU_ARCH": "dav-3510"}, ("950",)),
            ({"NPU_ARCH": "dav-2201", "CANN_TARGET": "a3"}, ("a3",)),
            ({"NPU_ARCH": "dav-3510", "CANN_TARGET": "950"}, ("950",)),
            ({"NPU_ARCH": "dav-2201", "TARGET_NPU_ARCH": "dav-3510"}, ("a2", "a3")),
        ]
        for environment, expected in cases:
            with self.subTest(environment=environment):
                module, metadata, _ = self.load_setup(environment)
                self.assertEqual(module.BUILD_TARGETS, expected)
                self.assertEqual(metadata["version"], module.__version__)
                self.assertNotIn("+", metadata["version"])

    def test_invalid_or_conflicting_targets_fail(self):
        cases = [
            {"CANN_TARGET": "unsupported"},
            {"NPU_ARCH": "arch22"},
            {"TARGET_NPU_ARCH": "unsupported"},
            {"NPU_ARCH": "dav-2201", "CANN_TARGET": "950"},
            {"NPU_ARCH": "dav-3510", "CANN_TARGET": "a2"},
            {"NPU_ARCH": "dav-3510", "CANN_TARGET": "a3"},
        ]
        for environment in cases:
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                self.load_setup(environment)

    def test_native_build_only_schedules_requested_targets(self):
        cases = [
            ({}, ["a2", "a3", "950"]), ({"CANN_TARGET": "a2"}, ["a2"]),
            ({"CANN_TARGET": "a3"}, ["a3"]), ({"CANN_TARGET": "950"}, ["950"]),
        ]
        for environment, expected in cases:
            with self.subTest(environment=environment):
                module, _, _ = self.load_setup(environment)
                build = Mock()
                with patch.object(module, "build_with_cmake", build):
                    module.prepare_native_libraries()
                self.assertEqual([call.args[0] for call in build.call_args_list], expected)

    def test_cmake_commands_use_matching_architecture(self):
        for target, architecture in (("a2", "dav-2201"), ("a3", "dav-2201"), ("950", "dav-3510")):
            with self.subTest(target=target):
                module, _, directory = self.load_setup({"CANN_TARGET": target})
                os.environ["ASCEND_HOME_PATH"] = str(directory / "cann")
                with patch.object(module.subprocess, "run") as run:
                    module.build_with_cmake(target)
                configure = run.call_args_list[0]
                self.assertIn(f"-DCANN_TARGET={target}", configure.args[0])
                self.assertIn(f"-DNPU_ARCH={architecture}", configure.args[0])
                self.assertEqual(configure.args[0][-1], f"build/cmake_python_{target}")
                self.assertTrue(configure.kwargs["check"])
                self.assertEqual(
                    run.call_args_list[1].args[0], ["make", "-C", f"build/cmake_python_{target}", "-j4"]
                )

    def test_prebuilt_reuse_keeps_device_names(self):
        for target, architecture in (("a2", "dav-2201"), ("a3", "dav-2201"), ("950", "dav-3510")):
            with self.subTest(target=target):
                module, _, directory = self.load_setup({"CANN_TARGET": target})
                build = self.write_prebuilt_tree(directory, target, architecture)
                module.reuse_cmake_libraries(str(build))
                destination = directory / "python" / "ops_gnn" / KERNEL_LABELS[target.upper()]
                filename = f"libopsgnn_npu_kernel-{KERNEL_LABELS[target.upper()]}.so"
                self.assertEqual((destination / "_pybind.so").read_bytes(), b"selected pybind")
                self.assertEqual((destination / filename).read_bytes(), b"selected kernel")
                self.assertEqual({path.name for path in destination.iterdir()}, {"_pybind.so", filename})

    def test_prebuilt_reuse_rejects_wrong_device_or_architecture(self):
        for cached_target, cached_architecture in (("a3", "dav-2201"), ("a2", "dav-3510")):
            with self.subTest(target=cached_target, architecture=cached_architecture):
                module, _, directory = self.load_setup({"CANN_TARGET": "a2"})
                build = self.write_prebuilt_tree(directory, cached_target, cached_architecture)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    module.reuse_cmake_libraries(str(build))

    def test_all_prebuilt_variants_are_staged_without_rebuilding(self):
        module, _, directory = self.load_setup()
        prebuilt = directory / "prebuilt"
        source = directory / "native_source"
        kernel_directory = source / "output" / "kernel"
        kernel_directory.mkdir(parents=True)
        for target in ("a2", "a3", "950"):
            variant = prebuilt / f"variant_{target}"
            variant.mkdir(parents=True)
            (variant / "lib_pybind.so").write_bytes(KERNEL_LABELS[target.upper()].encode())
            filename = f"libopsgnn_npu_kernel-{KERNEL_LABELS[target.upper()]}.so"
            (kernel_directory / filename).write_bytes(KERNEL_LABELS[target.upper()].encode())
        os.environ["OPSGNN_PREBUILT_ROOT"] = str(prebuilt)
        os.environ["OPSGNN_SOURCE_ROOT"] = str(source)
        build = Mock(side_effect=AssertionError("prebuilt variants must not be rebuilt"))
        with patch.object(module, "build_with_cmake", build):
            module.prepare_native_libraries()
        for label in ("A2-A3", "950"):
            destination = directory / "python" / "ops_gnn" / label
            self.assertEqual((destination / "_pybind.so").read_bytes(), label.encode())
            filename = f"libopsgnn_npu_kernel-{KERNEL_LABELS[label]}.so"
            self.assertEqual((destination / filename).read_bytes(), KERNEL_LABELS[label].encode())
        build.assert_not_called()

    def test_wheel_name_and_contents_match_selected_devices(self):
        cases = [
            ({}, ("A2-A3", "950")),
            ({"CANN_TARGET": "a2"}, ("A2-A3",)),
            ({"CANN_TARGET": "a3"}, ("A2-A3",)),
            ({"CANN_TARGET": "950"}, ("950",)),
            ({"NPU_ARCH": "dav-2201"}, ("A2-A3",)),
        ]
        for environment, labels in cases:
            with self.subTest(environment=environment):
                module, metadata, directory = self.load_setup(environment)
                self.write_wheel_fixture(directory)
                artifact = self.build_wheel(module, metadata, directory)
                self.assert_wheel_contents(artifact, module.__version__, labels)


if __name__ == "__main__":
    unittest.main()

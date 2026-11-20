# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only argument-preservation checks for generated ASC build commands."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")
MAKE = shutil.which("make")


def _extract_custom_command(source: str, marker: str) -> str:
    start = source.index(marker)
    closing = re.search(r"\n[ \t]+\)", source[start:])
    if closing is None:
        raise ValueError(f"unterminated custom command: {marker}")
    end = start + closing.end()
    return source[start:end]


def _write_fake_tools(root: Path):
    tool_dir = root / "tools"
    tool_dir.mkdir()
    sentinel = tool_dir / "opsgnn_probe_sentinel"
    sentinel.write_text(
        "#!/usr/bin/python3\n"
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['PROBE_SENTINEL_FILE']).touch()\n"
        "print('SENTINEL_EXECUTED')\n",
        encoding="utf-8",
    )
    sentinel.chmod(0o755)

    compiler = root / "fake_compiler.py"
    compiler.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['PROBE_ARGUMENTS'], 'a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "output = Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "output.touch()\n",
        encoding="utf-8",
    )
    compiler.chmod(0o755)
    return tool_dir, compiler, root / "sentinel-ran"


def _write_probe_project(source_dir: Path, compiler: Path, compile_command: str,
                         link_command: str) -> None:
    include_dir = source_dir / "include"
    include_dir.mkdir()
    (source_dir / "kernel.cpp").write_text(
        "// CPU-only fixture; the fake compiler records arguments.\n",
        encoding="utf-8",
    )
    preamble = f'''cmake_minimum_required(VERSION 3.18)
project(ops_gnn_argument_probe LANGUAGES NONE)
set(CMAKE_ASC_COMPILER "{compiler}")
set(ASC_LANG_FLAG --asc-aicore-lang)
set(ASC_FILE "${{CMAKE_CURRENT_SOURCE_DIR}}/kernel.cpp")
set(NPU_ARCH dav-3510)
set(ASC_INCLUDE_FLAGS)
list(APPEND ASC_INCLUDE_FLAGS "-I${{CMAKE_CURRENT_SOURCE_DIR}}/include")
set(OBJ_FILE "${{CMAKE_CURRENT_BINARY_DIR}}/kernel.o")
set(FILE_NAME kernel)
set(FILE_EXT .cpp)
set(OUTPUT_KERNEL_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/kernel-output")
set(NPU_KERNEL_LIB "${{OUTPUT_KERNEL_DIR}}/libopsgnn_npu_kernel-950.so")
set(NPU_KERNEL_OBJECTS "${{OBJ_FILE}}")
set(ASC_LINK_FLAGS "")
'''
    cmake_lists = "\n".join((preamble, compile_command, link_command,
                             'add_custom_target(probe ALL DEPENDS "${NPU_KERNEL_LIB}")', ""))
    (source_dir / "CMakeLists.txt").write_text(cmake_lists, encoding="utf-8")


def _build_probe(source_dir: Path, build_dir: Path, env):
    configure = subprocess.run(
        [CMAKE, "-S", str(source_dir), "-B", str(build_dir), "-G", "Unix Makefiles"],
        env=env, capture_output=True, text=True,
    )
    if configure.returncode:
        raise AssertionError(configure.stdout + configure.stderr)
    result = subprocess.run(
        [CMAKE, "--build", str(build_dir)], env=env, capture_output=True, text=True,
    )
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result


@unittest.skipUnless(CMAKE and MAKE,
                     "CMake and Make are required for generated-command checks")
class TestCMakeCommandEscaping(unittest.TestCase):
    def test_special_source_paths_remain_single_arguments(self):
        source_text = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        compile_command = _extract_custom_command(
            source_text, '        add_custom_command(\n            OUTPUT "${OBJ_FILE}"'
        )
        link_command = _extract_custom_command(
            source_text, '    add_custom_command(\n        OUTPUT "${NPU_KERNEL_LIB}"'
        )

        with tempfile.TemporaryDirectory(prefix="opsgnn-cmake-args-") as temp:
            root = Path(temp)
            tool_dir, compiler, sentinel_result = _write_fake_tools(root)
            env = dict(os.environ)
            env["PATH"] = str(tool_dir) + os.pathsep + env.get("PATH", "")
            env["PROBE_SENTINEL_FILE"] = str(sentinel_result)

            marker = "`opsgnn_probe_sentinel`"
            case_root = root / "case-0"
            source_dir = case_root / ("source" + marker)
            source_dir.mkdir(parents=True)
            include_dir = source_dir / "include"
            build_dir = case_root / "build"
            argument_log = case_root / "arguments.jsonl"
            env["PROBE_ARGUMENTS"] = str(argument_log)
            _write_probe_project(source_dir, compiler, compile_command, link_command)
            result = _build_probe(source_dir, build_dir, env)
            self.assertFalse(sentinel_result.exists(), result.stdout + result.stderr)

            invocations = [json.loads(line) for line in argument_log.read_text().splitlines()]
            self.assertEqual(len(invocations), 2)
            compile_args = next(args for args in invocations if "-c" in args)
            self.assertEqual(compile_args[compile_args.index("-c") + 1],
                             str(source_dir / "kernel.cpp"),
                             result.stdout + result.stderr + repr(compile_args))
            self.assertIn("-I" + str(include_dir), compile_args)
            link_args = next(args for args in invocations if "-shared" in args)
            self.assertEqual(link_args[link_args.index("-o") + 1],
                             str(build_dir
                                 / "kernel-output/libopsgnn_npu_kernel-950.so"))


if __name__ == "__main__":
    unittest.main()

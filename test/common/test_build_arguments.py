# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build script argument checks without running a build."""

from pathlib import Path
import subprocess
import unittest


BUILD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build.sh"


class TestBuildArguments(unittest.TestCase):
    @staticmethod
    def run_build(*arguments):
        return subprocess.run(
            [str(BUILD_SCRIPT), *arguments],
            capture_output=True, text=True, encoding="utf-8", timeout=10, check=False,
        )

    def test_missing_option_values(self):
        for option in ("-t", "--type", "--npu-arch", "--cann-target"):
            for suffix in ((), ("--help", "--help"), ("", "--help")):
                with self.subTest(option=option, suffix=suffix):
                    result = self.run_build(option, *suffix)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(f"选项 {option} 缺少参数值", result.stderr)

    def test_supplying_values_preserves_help(self):
        result = self.run_build("--type", "Debug", "--npu-arch", "dav-2201", "--cann-target", "a2", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops-gnn", result.stdout)


if __name__ == "__main__":
    unittest.main()

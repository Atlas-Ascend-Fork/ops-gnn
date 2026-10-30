# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Device selection regressions without PyTorch or Ascend hardware."""

import importlib.util
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestDeviceSelection(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, dict(os.environ), clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("CANN_TARGET", None)
        os.environ.pop("NPU_SMI_BIN", None)
        spec = importlib.util.spec_from_file_location(
            "packaging_device_selector", PROJECT_ROOT / "python" / "ops_gnn" / "_device.py"
        )
        self.selector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.selector)

    def test_explicit_device_override_skips_detection(self):
        cases = [("a2", "A2-A3"), ("A2", "A2-A3"), ("a3", "A2-A3"), ("A3", "A2-A3"), ("950", "950")]
        for target, expected in cases:
            with self.subTest(target=target), patch.dict(os.environ, {"CANN_TARGET": target}):
                which = Mock(side_effect=AssertionError("explicit target must not trigger npu-smi"))
                with patch.object(self.selector.shutil, "which", which):
                    self.assertEqual(self.selector.select_target(), expected)
                which.assert_not_called()

    def test_detect_local_device(self):
        cases = [
            ("Ascend 910B", "A2-A3"), ("ASCEND 910b", "A2-A3"),
            ("Ascend 910C", "A2-A3"), ("Ascend 910_9391", "A2-A3"), ("Ascend 950", "950"),
        ]
        for product, expected in cases:
            with self.subTest(product=product):
                os.environ["NPU_SMI_BIN"] = "custom-npu-smi"
                which = Mock(return_value="/mock/custom-npu-smi")
                run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=product))
                lookup_patch = patch.object(self.selector.shutil, "which", which)
                command_patch = patch.object(self.selector.subprocess, "run", run)
                with lookup_patch, command_patch:
                    self.assertEqual(self.selector.select_target(), expected)
                which.assert_called_once_with("custom-npu-smi")
                run.assert_called_once_with(
                    ["/mock/custom-npu-smi", "info", "-m"],
                    check=True, capture_output=True, text=True, timeout=10,
                )

    def test_invalid_device_override_fails(self):
        for target in ("all", "310p", "dav-2201"):
            with self.subTest(target=target), patch.dict(os.environ, {"CANN_TARGET": target}):
                with self.assertRaisesRegex(ValueError, "CANN_TARGET"):
                    self.selector.select_target()

    def test_missing_device_tool_reports_override_option(self):
        with patch.object(self.selector.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "CANN_TARGET=a2/a3/950"):
                self.selector.select_target()

    def test_unsupported_device_is_not_silently_mapped_to_950(self):
        for product in ("", "Ascend 310P"):
            with self.subTest(product=product):
                run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=product))
                lookup_patch = patch.object(self.selector.shutil, "which", return_value="/mock/npu-smi")
                command_patch = patch.object(self.selector.subprocess, "run", run)
                with lookup_patch, command_patch, self.assertRaisesRegex(RuntimeError, "Unsupported Ascend device"):
                    self.selector.select_target()

    def test_device_tool_failure_reports_detection_error(self):
        failures = [
            OSError("device tool unavailable"),
            subprocess.CalledProcessError(1, ["npu-smi"]),
            subprocess.TimeoutExpired(["npu-smi"], 10),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                lookup_patch = patch.object(self.selector.shutil, "which", return_value="/mock/npu-smi")
                command_patch = patch.object(self.selector.subprocess, "run", side_effect=failure)
                failure_check = self.assertRaisesRegex(RuntimeError, "device detection failed")
                with lookup_patch, command_patch, failure_check as error:
                    self.selector.select_target()
                self.assertIs(error.exception.__cause__, failure)


if __name__ == "__main__":
    unittest.main()

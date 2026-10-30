# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Select the native kernel matching the installed Ascend device."""
import os
import shutil
import subprocess


def select_target():
    override = os.getenv("CANN_TARGET", "").lower()
    if override:
        if override not in ("a2", "a3", "950"):
            raise ValueError("CANN_TARGET must be a2, a3, or 950")
        return "950" if override == "950" else "A2-A3"
    npu_smi = shutil.which(os.getenv("NPU_SMI_BIN", "npu-smi"))
    if not npu_smi:
        raise RuntimeError("npu-smi is unavailable; set CANN_TARGET=a2/a3/950")
    try:
        result = subprocess.run(
            [npu_smi, "info", "-m"], check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"npu-smi device detection failed: {exc}") from exc
    product = result.stdout.lower()
    if "910_93" in product or "910c" in product:
        return "A2-A3"
    if "910b" in product:
        return "A2-A3"
    if "950" in product:
        return "950"
    raise RuntimeError("Unsupported Ascend device; supported devices: A2, A3, 950")

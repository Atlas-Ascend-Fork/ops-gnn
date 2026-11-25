# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark ``index_sort`` for the arch35 build (950, dav-3510).

The measurement loop is architecture independent, so the arch22 module holds the
one implementation of it and this module only re-binds the entry point for the
950 build; the reported numbers are per platform.
"""

import runpy
from pathlib import Path

_SHARED = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "arch22"
        / "benchmark_index_sort.py"))


if __name__ == "__main__":
    _SHARED["main"]()

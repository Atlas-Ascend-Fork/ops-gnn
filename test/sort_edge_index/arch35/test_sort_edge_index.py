# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional tests of ``sort_edge_index`` for the arch35 build (950, dav-3510).

The cases and the edge-list reference are architecture independent, so the arch22
module holds the one implementation of the suite and this module only re-binds it
for the 950 build; a second copy would let the two platforms drift apart.
"""

import runpy
from pathlib import Path

_SHARED = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "arch22"
        / "test_sort_edge_index.py"))


class TestSortEdgeIndex(_SHARED["TestSortEdgeIndex"]):
    """``sort_edge_index`` against the CPU composite-key reference on 950."""


# The argument-validation case is a module-level test rather than a method, so
# the subclass above does not carry it over; rebinding the already-parametrised
# function keeps 950 running the same set of cases as arch22.
test_invalid_arguments = _SHARED["test_invalid_arguments"]

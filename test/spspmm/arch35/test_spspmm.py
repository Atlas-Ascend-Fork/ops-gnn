# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional tests of ``spspmm`` for the arch35 build (950, dav-3510).

The cases and the float64 reference are architecture independent, so the arch22
module holds the one implementation of them and this module only re-binds the
suite for the 950 build; a second copy would let the two platforms drift apart.
The long-row case runs here as well: the 950 merges it in the main pass instead
of the spill fallback, and the product it must produce is the same.
"""

import runpy
from pathlib import Path

_SHARED = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "arch22" / "test_spspmm.py"))

# Every case is a module-level function, so the suite is rebound by name. The
# functions keep the arch22 module globals (helpers and tolerances) they were
# defined with; only their collection happens under this module.
globals().update({name: value for name, value in _SHARED.items() if name.startswith("test_")})

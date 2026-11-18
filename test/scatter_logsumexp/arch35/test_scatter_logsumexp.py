# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functional tests of ``scatter_logsumexp`` for the arch35 build (Ascend 950, dav-3510)."""

from common.test_grouped_log_softmax_suite import LogSumExpSuite
from common.test_grouped_softmax_suite import load_sibling_golden

_golden = load_sibling_golden(__file__)


class TestScatterLogSumExp(LogSumExpSuite):
    """``scatter_logsumexp`` against its float64 grouped log-sum-exp golden."""

    api_name = "scatter_logsumexp"
    reference = staticmethod(_golden.reference)

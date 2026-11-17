# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Log-domain cases of the grouped-softmax operators.

``scatter_log_softmax`` has two cases that have no softmax counterpart: the
agreement with ``log(scatter_softmax)`` and the all-``-inf`` group. They live
here rather than in the operator module, which only binds the operator name.
"""

import pytest
import torch

from common.grouped_softmax_cases import source
from common.grouped_softmax_golden import assert_close
from common.test_grouped_softmax_suite import GroupedSoftmaxSuite, check


class LogSoftmaxSuite(GroupedSoftmaxSuite):
    """The shared cases plus the two log-domain-only ones."""

    log = True

    @pytest.mark.npu
    def test_matches_log_of_softmax(self, npu_api):
        """The log domain must agree with ``log(scatter_softmax)``."""
        src = source((32, 16), torch.float32)
        index = torch.randint(
            0, 5, (32,), generator=torch.Generator().manual_seed(7)).npu()

        log_out = self.call(npu_api, src.npu(), index, dim_size=5)
        softmax_out = npu_api.scatter_softmax(src.npu(), index, dim_size=5)

        torch.testing.assert_close(
            log_out.cpu(), torch.log(softmax_out).cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.npu
    def test_all_negative_infinity_group_is_nan(self, npu_api):
        """A group with no finite element is all NaN, like ``log_softmax``."""
        src = torch.full((4, 2), float("-inf"), dtype=torch.float32)
        index = torch.zeros(4, dtype=torch.int64)

        result = self.call(npu_api, src.npu(), index.npu(), dim_size=1)

        check(bool(torch.isnan(result).all()),
              f"got {result.flatten().tolist()}")
        assert_close(result, self.expected(src, index, 0, 1), torch.float32)

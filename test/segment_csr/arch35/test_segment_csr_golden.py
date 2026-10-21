# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden mean regressions in the segment CSR architecture suite."""

import pytest
import torch

from golden import segment_csr_cpu


class TestSegmentCsrGoldenIntegerMean:
    @staticmethod
    @pytest.mark.parametrize(
        "value",
        [16_777_217, -16_777_217, 9_007_199_254_740_993, -9_007_199_254_740_993],
    )
    def test_single_int64_mean_preserves_exact_integer(value):
        source = torch.tensor([value], dtype=torch.int64)
        actual = segment_csr_cpu(source, torch.tensor([0, 1]), "mean")
        assert torch.equal(actual, source)

    @staticmethod
    @pytest.mark.parametrize(
        "dtype,count,expected",
        [(torch.int8, 200, -1), (torch.int8, 256, 0), (torch.uint8, 256, 0)],
    )
    def test_narrow_mean_wraps_count_before_integer_division(dtype, count, expected):
        source = torch.zeros(count, dtype=dtype)
        source[0] = 100
        actual = segment_csr_cpu(source, torch.tensor([0, count]), "mean")
        assert torch.equal(actual, torch.tensor([expected], dtype=dtype))

    @staticmethod
    def test_integer_mean_truncates_toward_zero_and_float_control():
        ptr = torch.tensor([0, 2])
        assert torch.equal(
            segment_csr_cpu(torch.tensor([-1, -2]), ptr, "mean"), torch.tensor([-1])
        )
        assert torch.equal(
            segment_csr_cpu(torch.tensor([2.0, 4.0]), ptr, "mean"), torch.tensor([3.0])
        )

    @staticmethod
    def test_int8_sum_is_wrapped_before_division():
        source = torch.tensor([100, 100], dtype=torch.int8)
        actual = segment_csr_cpu(source, torch.tensor([0, 2]), "mean")
        assert torch.equal(actual, torch.tensor([-28], dtype=torch.int8))

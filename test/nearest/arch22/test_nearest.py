# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2/A3 shared interface tests for nearest."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "nearest_arch35_tests", Path(__file__).parents[1] / "arch35/test_nearest.py")
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

device = _shared.device
report_parameters = _shared.report_parameters
test_task_shapes = _shared.test_task_shapes
test_independent_seeds = _shared.test_independent_seeds
test_1d_noncontiguous = _shared.test_1d_noncontiguous
test_noncontiguous = _shared.test_noncontiguous
test_batch_global_index_gaps_imbalance = _shared.test_batch_global_index_gaps_imbalance
test_single_node_per_batch = _shared.test_single_node_per_batch
test_ties_across_1024_and_tiles = _shared.test_ties_across_1024_and_tiles
test_fp16_rounding_counterexample = _shared.test_fp16_rounding_counterexample
test_batch_errors = _shared.test_batch_errors
test_missing_batch_means_zero = _shared.test_missing_batch_means_zero
test_single_batch_matches_unbatched = _shared.test_single_batch_matches_unbatched
test_single_batch_close_candidates = _shared.test_single_batch_close_candidates
test_nondefault_stream_and_input_mutation = _shared.test_nondefault_stream_and_input_mutation
test_dispatcher_entry = _shared.test_dispatcher_entry
test_dispatcher_rejects_malformed_pointers = _shared.test_dispatcher_rejects_malformed_pointers
test_dispatcher_accepts_empty_matched_segments = _shared.test_dispatcher_accepts_empty_matched_segments
test_existing_dispatcher_schema = _shared.test_existing_dispatcher_schema
test_empty_x_and_empty_y = _shared.test_empty_x_and_empty_y
test_nonfinite_rejected = _shared.test_nonfinite_rejected
test_large_axis_with_small_work = _shared.test_large_axis_with_small_work
test_fp16_matches_promoted_inputs_and_scipy = _shared.test_fp16_matches_promoted_inputs_and_scipy
test_fp16_tie_order_across_simt_simd_and_tiles = _shared.test_fp16_tie_order_across_simt_simd_and_tiles
test_fp16_batch_global_smallest_index = _shared.test_fp16_batch_global_smallest_index
test_fp16_underflow_and_distance_overflow = _shared.test_fp16_underflow_and_distance_overflow
test_fp16_1d_and_single_point_batch_indices = _shared.test_fp16_1d_and_single_point_batch_indices
test_fp16_nondefault_stream_and_input_mutation = _shared.test_fp16_nondefault_stream_and_input_mutation
test_fp16_rounding_contract_and_scipy_difference = _shared.test_fp16_rounding_contract_and_scipy_difference
TestAcceptanceLogRegressions = _shared.TestAcceptanceLogRegressions
test_reference_semantics = _shared.test_reference_semantics
TestNearestOfficial = _shared.TestNearestOfficial
test_scalar_half_large_candidates = _shared.test_scalar_half_large_candidates
test_scalar_half_large_candidate_ties = _shared.test_scalar_half_large_candidate_ties
test_scalar_half_large_candidates_nondefault_stream = (
    _shared.test_scalar_half_large_candidates_nondefault_stream
)

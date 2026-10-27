/*
 * Copyright (c) 2026 Starlink_. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 */

#ifndef OPS_GNN_SEGMENT_CSR_HOST_H
#define OPS_GNN_SEGMENT_CSR_HOST_H

#include <string>
#include <vector>

#include <c10/util/Optional.h>
#include <torch/types.h>

namespace ops_gnn {

// Private C++ module initializer; registers only CPU-side observation on arch22.
// Not a Python API and a no-op on arch35.
void initialize_segment_csr_proof_observer();

// Internal boundary for other ops_gnn kernels that bypass the ATen dispatcher.
void invalidate_segment_csr_proofs();

// torch_scatter-compatible segment_csr forward on NPU.
// reduce: "sum" | "add" | "mean" | "min" | "max".
// Returns {out} for sum/add/mean and {out, arg_out} for min/max.
std::vector<torch::Tensor> segment_csr(
    torch::Tensor src, torch::Tensor indptr, c10::optional<torch::Tensor> out, const std::string& reduce, bool returnArg = true);

} // namespace ops_gnn

#endif // OPS_GNN_SEGMENT_CSR_HOST_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SEGMENT_LOGSUMEXP_OP_HOST_H
#define OPS_GNN_SEGMENT_LOGSUMEXP_OP_HOST_H

#include <torch/extension.h>

namespace opsgnn {

// Numerically stable log-sum-exp over CSR segments. ``src`` must be
// contiguous, ``indptr`` a contiguous int32 vector of ``segments + 1``
// boundaries that ends at ``src.size(0)``. The returned tensor has shape
// ``(indptr.numel() - 1,) + src.sizes()[1:]`` and the dtype of ``src``.
torch::Tensor segment_logsumexp_forward(const torch::Tensor& src,
                                        const torch::Tensor& indptr);

}  // namespace opsgnn

#endif  // OPS_GNN_SEGMENT_LOGSUMEXP_OP_HOST_H

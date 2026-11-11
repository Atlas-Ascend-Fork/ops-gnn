/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "segment_logsumexp/op_host/segment_logsumexp.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "c10/core/DeviceGuard.h"
#include "common/aiv_launch_utils.h"
#include "segment_logsumexp/op_kernel/common/segment_logsumexp_launch.h"
#include "segment_logsumexp/op_kernel/common/segment_logsumexp_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

namespace {

SegmentLogSumExpDType ToSegmentLogSumExpDType(at::ScalarType dtype)
{
    switch (dtype) {
        case at::ScalarType::Half:
            return SEGMENT_LOGSUMEXP_F16;
        case at::ScalarType::BFloat16:
            return SEGMENT_LOGSUMEXP_BF16;
        case at::ScalarType::Float:
            return SEGMENT_LOGSUMEXP_F32;
        default:
            TORCH_CHECK(false, "segment_logsumexp: unsupported source dtype");
    }
}

void ValidateInputs(const torch::Tensor& src, const torch::Tensor& indptr)
{
    TORCH_CHECK(src.is_contiguous(), "segment_logsumexp: src must be contiguous");
    TORCH_CHECK(indptr.is_contiguous(), "segment_logsumexp: indptr must be contiguous");
    TORCH_CHECK(indptr.scalar_type() == at::ScalarType::Int,
                "segment_logsumexp: indptr must have int32 dtype");
    TORCH_CHECK(indptr.device() == src.device(),
                "segment_logsumexp: src and indptr must be on the same device");
    TORCH_CHECK(src.dim() >= 1, "segment_logsumexp: src must have at least one dimension");
    TORCH_CHECK(indptr.dim() == 1, "segment_logsumexp: indptr must be 1-D");
    TORCH_CHECK(indptr.numel() >= 1,
                "segment_logsumexp: indptr must contain at least one boundary");
    switch (src.scalar_type()) {
        case at::ScalarType::Half:
        case at::ScalarType::BFloat16:
        case at::ScalarType::Float:
            break;
        default:
            TORCH_CHECK(false,
                        "segment_logsumexp: src dtype must be float16, bfloat16 or float32");
    }
}

// The trailing axes are reduced feature by feature, so they are flattened into
// one feature count while the rows stay along axis 0.
uint64_t ColumnTile(uint64_t features)
{
    const uint64_t padded =
        (features + SEGMENT_LOGSUMEXP_COL_ALIGN - 1) / SEGMENT_LOGSUMEXP_COL_ALIGN *
        SEGMENT_LOGSUMEXP_COL_ALIGN;
    return std::min(SEGMENT_LOGSUMEXP_MAX_COL_TILE,
                    std::max(padded, SEGMENT_LOGSUMEXP_COL_ALIGN));
}

// One core walks the tasks of a phase in a strided order, so the number of
// tasks bounds the busy cores. Splitting the rows of a segment is what fills a
// machine that has more cores than segments; the piece count is bounded so that
// each piece still holds enough rows to pay for the extra passes.
uint64_t RowChunks(uint64_t rows, uint64_t segments, uint64_t baseTasks, uint32_t coreNum)
{
    if (segments == 0 || baseTasks == 0 || baseTasks >= coreNum) {
        return 1;
    }
    const uint64_t wanted = (static_cast<uint64_t>(coreNum) + baseTasks - 1) / baseTasks;
    const uint64_t affordable = rows / (segments * SEGMENT_LOGSUMEXP_MIN_ROWS_PER_CHUNK);
    const uint64_t chunks =
        std::min(std::min(wanted, affordable), static_cast<uint64_t>(SEGMENT_LOGSUMEXP_MAX_ROW_CHUNKS));
    return std::max<uint64_t>(chunks, 1);
}

// The platform core count is queried once per call, so the two grids are capped
// against it instead of asking the platform again.
uint32_t ClampBlockNum(uint32_t coreNum, uint64_t tasks)
{
    if (tasks == 0) {
        return 1;
    }
    const uint64_t capped = std::min<uint64_t>(tasks, static_cast<uint64_t>(coreNum));
    return static_cast<uint32_t>(capped == 0 ? 1 : capped);
}

SegmentLogSumExpTilingData BuildTiling(const torch::Tensor& src,
                                       const torch::Tensor& indptr)
{
    SegmentLogSumExpTilingData tiling{};
    tiling.rows = static_cast<uint64_t>(src.size(0));
    tiling.segments = static_cast<uint64_t>(indptr.numel() - 1);
    tiling.features = 1;
    for (int64_t axis = 1; axis < src.dim(); ++axis) {
        tiling.features *= static_cast<uint64_t>(src.size(axis));
    }

    // Narrow feature rows keep a small unified-buffer footprint; wide rows are
    // tiled. The tile width is a multiple of 8 fp32 lanes so that every vector
    // instruction of the row loop starts on a 32-byte aligned buffer base.
    tiling.colTile = ColumnTile(tiling.features);
    tiling.colTiles = (tiling.features + tiling.colTile - 1) / tiling.colTile;
    tiling.baseTasks = tiling.segments * tiling.colTiles;

    const uint32_t coreNum = ops_gnn::sparse::ResolveAivCoreNum(
        std::numeric_limits<uint32_t>::max());
    tiling.rowChunks = RowChunks(tiling.rows, tiling.segments, tiling.baseTasks, coreNum);
    tiling.tasks = tiling.baseTasks * tiling.rowChunks;
    tiling.blockNum = ClampBlockNum(coreNum, tiling.tasks);
    tiling.baseBlockNum = ClampBlockNum(coreNum, tiling.baseTasks);
    return tiling;
}

}  // namespace

torch::Tensor segment_logsumexp_forward(const torch::Tensor& src,
                                        const torch::Tensor& indptr)
{
    ValidateInputs(src, indptr);
    const c10::DeviceGuard deviceGuard(src.device());

    std::vector<int64_t> outSizes(src.sizes().begin(), src.sizes().end());
    outSizes[0] = static_cast<int64_t>(indptr.numel() - 1);
    torch::Tensor out = torch::empty(outSizes, src.options());

    const SegmentLogSumExpTilingData tiling = BuildTiling(src, indptr);
    if (tiling.tasks == 0) {
        return out;
    }

    // The split phases exchange the per-chunk maximum and exponential sum through
    // one fp32 workspace; every entry is written before it is read, so it needs
    // no initialisation. A fused call keeps everything in the unified buffer and
    // therefore allocates nothing.
    float* splitMax = nullptr;
    float* splitSum = nullptr;
    torch::Tensor workspace;
    if (tiling.rowChunks > 1) {
        const int64_t elements = static_cast<int64_t>(tiling.tasks * tiling.colTile);
        workspace = torch::empty({2 * elements},
                                 src.options().dtype(at::ScalarType::Float));
        splitMax = workspace.data_ptr<float>();
        splitSum = splitMax + elements;
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream(src.get_device());
    SegmentLogSumExp(src.data_ptr(), indptr.data_ptr<int32_t>(), out.data_ptr(),
                     splitMax, splitSum, ToSegmentLogSumExpDType(src.scalar_type()),
                     tiling, stream);
    return out;
}

}  // namespace opsgnn

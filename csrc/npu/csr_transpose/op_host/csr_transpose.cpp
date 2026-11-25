/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "csr_transpose/op_host/csr_transpose.h"

#include "csr_transpose/op_kernel/common/csr_transpose_launch.h"

#include <limits>

#include "common/aiv_launch_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

namespace {

const char* const kApi = "csr_transpose";

// A full-grid launch only pays off once every AIV owns a reasonable number of
// edges; smaller matrices stay on fewer AIVs.
constexpr int64_t kEdgesPerBlock = 4096;

// UB capacity the platform reports for the current device.
uint32_t QueryUbBytes()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, kApi, ": platform information unavailable");
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    TORCH_CHECK(ubBytes > 0 && ubBytes <= std::numeric_limits<uint32_t>::max(),
                kApi, ": invalid platform UB capacity");
    return static_cast<uint32_t>(ubBytes);
}

// One AIV per kEdgesPerBlock edges, capped by the platform AIV count and by the
// number of columns: an AIV without a column has nothing to count or to write.
// The column split of every pass walks 32-byte boundaries, and neighbouring
// ranges are at least kTransposeAlignment columns apart, so the number of AIVs is
// also capped by `numColumns / kTransposeAlignment`. That keeps the last range of
// the prefix-sum pass non-empty, which is what emits the closing boundary.
uint32_t ResolveBlocks(int64_t numEdges, int64_t numColumns)
{
    const int64_t wanted = (numEdges + kEdgesPerBlock - 1) / kEdgesPerBlock;
    const int64_t bounded = wanted < 1 ? 1 : (wanted > std::numeric_limits<uint32_t>::max()
                                                  ? std::numeric_limits<uint32_t>::max()
                                                  : wanted);
    uint32_t blocks = ops_gnn::sparse::ResolveAivCoreNum(static_cast<uint32_t>(bounded));
    if (numColumns > 0) {
        const int64_t usable = numColumns / kTransposeAlignment;
        const int64_t cap = usable < 1 ? 1 : usable;
        if (static_cast<int64_t>(blocks) > cap) {
            blocks = static_cast<uint32_t>(cap);
        }
    }
    return blocks == 0 ? 1u : blocks;
}

void CheckInputs(const torch::Tensor& indptr, const torch::Tensor& indices)
{
    TORCH_CHECK(indptr.device().type() == c10::DeviceType::PrivateUse1,
                kApi, ": indptr must be an NPU tensor");
    TORCH_CHECK(indices.device().type() == c10::DeviceType::PrivateUse1,
                kApi, ": indices must be an NPU tensor");
    TORCH_CHECK(indptr.device() == indices.device(),
                kApi, ": indptr and indices must be on the same NPU");
    TORCH_CHECK(indptr.dim() == 1, kApi, ": indptr must be 1-D");
    TORCH_CHECK(indices.dim() == 1, kApi, ": indices must be 1-D");
    TORCH_CHECK(indptr.scalar_type() == at::kLong, kApi, ": indptr must be int64");
    TORCH_CHECK(indices.scalar_type() == at::kLong, kApi, ": indices must be int64");
    TORCH_CHECK(indptr.numel() >= 1, kApi, ": indptr must hold at least one boundary");
}

// The number of output rows. Without an explicit count the widest column id
// defines it, which needs the one scalar read of the call.
int64_t ResolveNumColumns(const torch::Tensor& indices, const std::optional<int64_t>& numCols)
{
    if (numCols.has_value()) {
        TORCH_CHECK(*numCols >= 0, kApi, ": num_cols must be non-negative");
        return *numCols;
    }
    if (indices.numel() == 0) {
        return 0;
    }
    const int64_t widest = indices.max().item<int64_t>();
    TORCH_CHECK(widest >= 0, kApi, ": indices must be non-negative, got ", widest);
    return widest + 1;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> CsrTranspose(const torch::Tensor& indptr,
                                                      const torch::Tensor& indices,
                                                      std::optional<int64_t> numCols)
{
    CheckInputs(indptr, indices);
    const torch::Tensor rowPtr = indptr.is_contiguous() ? indptr : indptr.contiguous();
    const torch::Tensor colIdx = indices.is_contiguous() ? indices : indices.contiguous();

    const int64_t numRows = rowPtr.numel() - 1;
    const int64_t numEdges = colIdx.numel();
    const int64_t numColumns = ResolveNumColumns(colIdx, numCols);
    TORCH_CHECK(numColumns > 0 || numEdges == 0,
                kApi, ": a matrix with edges needs at least one column, got num_cols=0");
    TORCH_CHECK(numRows > 0 || numEdges == 0,
                kApi, ": indptr must hold at least two boundaries to describe an edge");

    const c10_npu::OptionalNPUGuard deviceGuard(rowPtr.device());
    torch::Tensor outPtr = torch::empty({numColumns + 1}, rowPtr.options());
    torch::Tensor outIdx = torch::empty({numEdges}, rowPtr.options());
    if (numEdges == 0) {
        // No edge means no column has a row: every boundary is zero.
        outPtr.zero_();
        return {outPtr, outIdx};
    }

    const uint32_t ubBytes = QueryUbBytes();
    const uint32_t blockNum = ResolveBlocks(numEdges, numColumns);
    const aclrtStream stream = c10_npu::getCurrentNPUStream(rowPtr.device().index()).stream();

    // The histogram and one range total per AIV are scratch (one 32-byte block per
    // total, see kTransposeBlockStride).
    const int64_t totalSlots = static_cast<int64_t>(blockNum) * kTransposeBlockStride;
    torch::Tensor counts = torch::empty({numColumns}, rowPtr.options());
    torch::Tensor blockTotals = torch::empty({totalSlots}, rowPtr.options());

    const int64_t* indptrData = rowPtr.data_ptr<int64_t>();
    const int64_t* indexData = colIdx.data_ptr<int64_t>();
    int64_t* outPtrData = outPtr.data_ptr<int64_t>();
    int64_t* outIdxData = outIdx.data_ptr<int64_t>();

    CsrTransposeCount(indexData, counts.data_ptr<int64_t>(), blockTotals.data_ptr<int64_t>(),
                      numEdges, numColumns, blockNum, ubBytes, stream);
    CsrTransposeIndptr(counts.data_ptr<int64_t>(), blockTotals.data_ptr<int64_t>(), outPtrData,
                       numColumns, blockNum, stream);
    CsrTransposeWrite(indptrData, indexData, outPtrData, outIdxData, numRows, numEdges, numColumns,
                      blockNum, ubBytes, stream);
    return {outPtr, outIdx};
}

}  // namespace opsgnn

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Coalesce Host orchestration.
//
// Sorting and duplicate detection run on the NPU through the standard tensor
// primitives (sort / comparison / nonzero / gather). The deduplicated segments
// are contiguous in sorted order, so the custom Ascend C kernel
// (op_kernel/arch22/coalesce.cpp) folds each segment of edge_attr into one row.

#include "coalesce.h"
#include "coalesce/op_kernel/arch22/coalesce.h"
#include "coalesce/op_kernel/arch22/coalesce_tiling.h"

#include <algorithm>
#include <limits>
#include <vector>

#include <c10/core/DeviceGuard.h>
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {
namespace {

const char *const kApi = "coalesce";

// Minimum number of deduplicated edges handed to one AIV so small inputs do not
// pay a full-grid launch.
constexpr uint64_t kMinSegmentsPerCore = 1024;

// Reserve kept free in UB for scalar/event bookkeeping.
constexpr uint64_t kUbReserveBytes = 8 * 1024;

constexpr uint64_t kMaxTileRows = 512;

// Square-root bound of INT64_MAX: num_nodes * num_nodes must not overflow the
// 64-bit key product that PyG forms.
constexpr int64_t kMaxNumNodes = 3037000499LL;

int64_t FeatureWidth(const torch::Tensor &attr) {
    int64_t width = 1;
    for (int64_t d = 1; d < attr.dim(); ++d)
        width *= attr.size(d);
    return width;
}

torch::Tensor AsContiguous(const torch::Tensor &tensor) {
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

void ValidateEdgeIndex(const torch::Tensor &edge_index) {
    TORCH_CHECK(edge_index.defined(), kApi, ": edge_index must be defined");
    TORCH_CHECK(edge_index.device().type() == c10::DeviceType::PrivateUse1, kApi,
                ": edge_index must be an NPU tensor");
    TORCH_CHECK(edge_index.scalar_type() == at::kLong, kApi, ": edge_index must be int64");
    TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2, kApi,
                ": edge_index must have shape [2, E]");
    TORCH_CHECK(edge_index.layout() == at::kStrided, kApi, ": edge_index must be strided");
}

void ValidateAttr(const torch::Tensor &attr, const torch::Tensor &edge_index) {
    TORCH_CHECK(attr.device() == edge_index.device(), kApi,
                ": edge_attr and edge_index must be on the same NPU");
    TORCH_CHECK(attr.scalar_type() == at::kHalf || attr.scalar_type() == at::kFloat, kApi,
                ": edge_attr must be float16 or float32");
    TORCH_CHECK(attr.dim() >= 1, kApi, ": edge_attr must have rank >= 1");
    TORCH_CHECK(attr.size(0) == edge_index.size(1), kApi,
                ": edge_attr must have one row per edge");
    TORCH_CHECK(attr.layout() == at::kStrided, kApi, ": edge_attr must be strided");
}

void ValidateReduce(int64_t reduce) {
    TORCH_CHECK(reduce >= kCoalesceSum && reduce <= kCoalesceMax, kApi,
                ": reduce must be sum, mean, min or max");
}

// Resolve the key radix exactly as PyG's maybe_num_nodes: an explicit value, or
// one past the largest endpoint when the input is nonempty.
int64_t ResolveNumNodes(const torch::Tensor &edge_index, const c10::optional<int64_t> &num_nodes) {
    int64_t resolved;
    if (num_nodes.has_value()) {
        resolved = *num_nodes;
    } else if (edge_index.numel() == 0) {
        resolved = 0;
    } else {
        resolved = edge_index.max().item<int64_t>() + 1;
    }
    TORCH_CHECK(resolved >= 0, kApi, ": num_nodes must be non-negative");
    TORCH_CHECK(resolved <= kMaxNumNodes, kApi, ": num_nodes is too large and would overflow the edge key");
    return resolved;
}

struct DeviceCaps {
    uint32_t cores;
    uint64_t ubBytes;
};

DeviceCaps QueryDeviceCaps() {
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, kApi, ": platform information unavailable");
    DeviceCaps caps{std::max<uint32_t>(1, platform->GetCoreNumAiv()), 0};
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, caps.ubBytes);
    TORCH_CHECK(caps.ubBytes > kUbReserveBytes, kApi, ": invalid device UB capacity");
    return caps;
}

// Sort edges by their integer key and gather edge_attr into sorted order.
std::tuple<torch::Tensor, torch::Tensor> SortByKey(const torch::Tensor &key, bool is_sorted) {
    if (!is_sorted)
        return torch::sort(key, /*dim=*/0, /*descending=*/false);
    return {key, torch::arange(key.numel(), key.options())};
}

struct DedupResult {
    torch::Tensor uniqueIndex;
    torch::Tensor indptr;
    int64_t unique;
};

// Build the PyG-compatible integer key and return it sorted with its permutation.
std::tuple<torch::Tensor, torch::Tensor> BuildSortedKeys(
    const torch::Tensor &ei, int64_t numNodes, bool sort_by_row, bool is_sorted) {
    auto row = ei.select(0, sort_by_row ? 0 : 1);
    auto col = ei.select(0, sort_by_row ? 1 : 0);
    auto key = row * numNodes + col;
    return SortByKey(key, is_sorted);
}

// Collapse maximal runs of equal sorted keys into unique edges.
DedupResult DedupEdges(const torch::Tensor &sortedKey, const torch::Tensor &perm,
                       const torch::Tensor &ei, int64_t edges) {
    auto boundary = torch::ones({edges}, ei.options().dtype(at::kBool));
    if (edges > 1) {
        auto changes = sortedKey.narrow(0, 1, edges - 1).ne(sortedKey.narrow(0, 0, edges - 1));
        boundary.narrow(0, 1, edges - 1).copy_(changes);
    }
    auto starts = boundary.nonzero().squeeze(1);
    const int64_t unique = starts.size(0);
    auto indptr = torch::cat({starts, torch::full({1}, edges, ei.options())});
    auto representative = perm.index_select(0, starts);
    auto uniqueIndex = ei.index_select(1, representative);
    return {uniqueIndex, indptr, unique};
}

std::vector<int64_t> AttrOutputShape(const torch::Tensor &attr, int64_t unique) {
    std::vector<int64_t> outShape;
    outShape.push_back(unique);
    for (int64_t d = 1; d < attr.dim(); ++d)
        outShape.push_back(attr.size(d));
    return outShape;
}

// Budget UB tiles (source/work plus fixed acc/out/denom buffers) and launch the
// arch22 segment-reduction kernel.
void LaunchSegmentReduce(const torch::Tensor &attr, int64_t channels, int64_t reduce,
                         const torch::Tensor &sortedAttr, const torch::Tensor &indptr,
                         int64_t edges, int64_t unique, const torch::Tensor &outAttr,
                         aclrtStream stream) {
    const DeviceCaps caps = QueryDeviceCaps();
    const bool half = attr.scalar_type() == at::kHalf;
    const uint64_t itemBytes = half ? sizeof(uint16_t) : sizeof(float);
    const uint32_t blockElems = static_cast<uint32_t>(32 / itemBytes);
    const uint32_t width = static_cast<uint32_t>((channels + blockElems - 1) / blockElems * blockElems);
    const uint64_t perRow = uint64_t(width) * itemBytes + (half ? uint64_t(width) * sizeof(float) : 0);
    // Fixed per-tile buffers: acc (fp32) + out (src dtype) + denom (fp32, mean only).
    const uint64_t fixed = uint64_t(width) * (sizeof(float) + itemBytes +
        (reduce == kCoalesceMean ? sizeof(float) : 0));
    TORCH_CHECK(perRow > 0 && fixed + perRow <= caps.ubBytes,
                kApi, ": feature width exceeds UB capacity");
    const uint64_t budget = caps.ubBytes > (kUbReserveBytes + fixed)
        ? caps.ubBytes - kUbReserveBytes - fixed : 0;
    const uint32_t tileRows = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(kMaxTileRows, budget / perRow)));
    uint64_t needed = (static_cast<uint64_t>(unique) + kMinSegmentsPerCore - 1) / kMinSegmentsPerCore;
    uint32_t blocks = static_cast<uint32_t>(
        std::min<uint64_t>({caps.cores, needed ? needed : 1, static_cast<uint64_t>(unique)}));

    CoalesceTilingData tiling{};
    tiling.E = static_cast<uint64_t>(edges);
    tiling.U = static_cast<uint64_t>(unique);
    tiling.C = static_cast<uint64_t>(channels);
    tiling.srcNumel = static_cast<uint64_t>(edges) * channels;
    tiling.outNumel = static_cast<uint64_t>(unique) * channels;
    tiling.reduce = static_cast<uint32_t>(reduce);
    tiling.dtype = half ? kCoalesceFloat16 : kCoalesceFloat32;
    tiling.usedBlockNum = blocks;
    tiling.tileRows = tileRows;
    tiling.width = width;
    tiling.hasAttr = 1;

    const aclError status = CoalesceSegmentReduce(sortedAttr.data_ptr(), indptr.data_ptr<int64_t>(),
                                                  outAttr.data_ptr(), tiling, stream);
    TORCH_CHECK(status == ACL_SUCCESS, kApi, ": segment reduction launch failed, acl status ", status);
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> CoalesceForward(
    torch::Tensor edge_index, c10::optional<torch::Tensor> edge_attr, c10::optional<int64_t> num_nodes,
    int64_t reduce, bool is_sorted, bool sort_by_row) {
    ValidateEdgeIndex(edge_index);
    ValidateReduce(reduce);
    const bool hasAttr = edge_attr.has_value() && edge_attr->defined();
    if (hasAttr)
        ValidateAttr(*edge_attr, edge_index);

    const int64_t edges = edge_index.size(1);
    const int64_t numNodes = ResolveNumNodes(edge_index, num_nodes);

    c10::DeviceGuard guard(edge_index.device());
    const auto ei = AsContiguous(edge_index);

    auto sorted = BuildSortedKeys(ei, numNodes, sort_by_row, is_sorted);
    auto &sortedKey = std::get<0>(sorted);
    auto &perm = std::get<1>(sorted);

    DedupResult dedup = DedupEdges(sortedKey, perm, ei, edges);

    if (!hasAttr)
        return {dedup.uniqueIndex, torch::Tensor()};

    auto attr = AsContiguous(*edge_attr);
    const int64_t channels = FeatureWidth(attr);
    auto flat = attr.view({edges, channels});
    auto sortedAttr = flat.index_select(0, perm);

    auto outAttr = torch::empty(AttrOutputShape(attr, dedup.unique), attr.options());

    if (dedup.unique > 0 && channels > 0) {
        const aclrtStream stream = c10_npu::getCurrentNPUStream(edge_index.device().index()).stream();
        LaunchSegmentReduce(attr, channels, reduce, sortedAttr, dedup.indptr, edges, dedup.unique,
                            outAttr, stream);
    }

    return {dedup.uniqueIndex, outAttr};
}

} // namespace opsgnn

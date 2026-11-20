/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "disjoint_union_coo/op_host/disjoint_union_coo.h"

#include "disjoint_union_coo/op_kernel/arch22/disjoint_union_coo.h"

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include <c10/core/DeviceGuard.h>

#include "common/aiv_launch_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {
namespace {

const char* const kApi = "disjoint_union_coo";

// The kernel addresses the concatenated output with 32-bit offsets and shifts
// ids with 32-bit vector adds, so every concatenated extent stays in int32.
constexpr int64_t kMaxExtent = std::numeric_limits<int32_t>::max();

// Launch one AIV per this many edges; small matrices stay on fewer cores
// instead of paying a full-grid launch.
constexpr int64_t kMinEdgesPerCore = 1024;

// One input COO matrix together with its accumulated offsets.
struct CooPart {
    torch::Tensor source;
    torch::Tensor dest;
    torch::Tensor data;
    int64_t numRows{0};
    int64_t numCols{0};
    int64_t offsetRows{0};
    int64_t offsetCols{0};
    int64_t offsetEdges{0};
};

// Every input matrix of one call together with the concatenated extents.
struct CooUnion {
    std::vector<CooPart> parts;
    int64_t totalRows{0};
    int64_t totalCols{0};
    int64_t totalEdges{0};
    bool hasData{false};
};

// Typed view of the three output buffers handed to one launch.
template <typename T>
struct CooOutputs {
    T* source{nullptr};
    T* dest{nullptr};
    T* data{nullptr};
    uint32_t totalEdges{0};
};

bool IsIndexDtype(const torch::Tensor& tensor)
{
    return tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong;
}

// Copy a view into contiguous storage so the kernel can read whole tiles.
torch::Tensor AsContiguous(const torch::Tensor& tensor)
{
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

// Reject tensors the kernel cannot consume as a COO index array. `role` names
// the offending operand (source, dest or data) in the error message.
void CheckIndexTensor(const torch::Tensor& tensor, const std::string& part, const char* role)
{
    TORCH_CHECK(tensor.defined(), kApi, part, ": ", role, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, kApi, part, ": ", role,
                " must be NPU tensors");
    TORCH_CHECK(tensor.dim() == 1, kApi, part, ": ", role, " must be 1-D");
    TORCH_CHECK(IsIndexDtype(tensor), kApi, part, ": ", role, " must be int32 or int64");
}

// Validate one matrix: matching index pairs plus their row/column extents.
void CheckMatrix(const CooPart& part, size_t index)
{
    const std::string partName = ": graph " + std::to_string(index);
    CheckIndexTensor(part.source, partName, "source");
    CheckIndexTensor(part.dest, partName, "dest");
    TORCH_CHECK(part.source.device() == part.dest.device(), kApi, partName,
                ": source and dest must share one NPU");
    TORCH_CHECK(part.source.scalar_type() == part.dest.scalar_type(), kApi, partName,
                ": source and dest must have the same dtype");
    TORCH_CHECK(part.numRows >= 0 && part.numCols >= 0, kApi, partName,
                ": num_rows and num_cols must be non-negative");
    TORCH_CHECK(part.numRows <= kMaxExtent && part.numCols <= kMaxExtent, kApi, partName,
                ": num_rows and num_cols must not exceed ", kMaxExtent);
}

// Validate the optional per-edge data array of one matrix.
void CheckData(const CooPart& part, size_t index)
{
    if (!part.data.defined()) {
        return;
    }
    const std::string partName = ": graph " + std::to_string(index);
    CheckIndexTensor(part.data, partName, "data");
    TORCH_CHECK(part.data.device() == part.source.device(), kApi, partName,
                ": data must be on the same NPU as the indices");
    TORCH_CHECK(part.data.scalar_type() == part.source.scalar_type(), kApi, partName,
                ": data dtype must match the source and dest dtype");
    TORCH_CHECK(part.data.numel() == part.source.numel(), kApi, partName,
                ": data must hold one value per edge, expected ", part.source.numel(), " got ",
                part.data.numel());
}

// Reject inputs whose concatenated extents would overflow 32-bit addressing.
void CheckExtents(const CooUnion& union_, const CooPart& part)
{
    TORCH_CHECK(union_.totalRows <= kMaxExtent - part.numRows &&
                    union_.totalCols <= kMaxExtent - part.numCols &&
                    union_.totalEdges <= kMaxExtent - part.source.numel(),
                kApi, ": concatenated rows/cols/nnz must not exceed ", kMaxExtent);
}

// Query the UB capacity the platform reports for the current device.
uint32_t QueryUbBytes()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, kApi, ": platform information unavailable");
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    TORCH_CHECK(ubBytes > 0 && ubBytes <= std::numeric_limits<uint32_t>::max(), kApi,
                ": invalid platform UB capacity");
    return static_cast<uint32_t>(ubBytes);
}

// Spread the edges of one matrix over at least one AIV.
uint32_t BlocksFor(int64_t edges)
{
    const int64_t needed = std::max<int64_t>(1, (edges + kMinEdgesPerCore - 1) / kMinEdgesPerCore);
    return ops_gnn::sparse::ResolveAivCoreNum(static_cast<uint32_t>(needed));
}

// Build one part: contiguous views, declared extents and its offsets.
CooPart MakePart(const torch::Tensor& source, const torch::Tensor& dest,
                 const std::optional<torch::Tensor>& data, int64_t numRows, int64_t numCols)
{
    CooPart part;
    part.source = AsContiguous(source);
    part.dest = AsContiguous(dest);
    if (data.has_value() && data->defined()) {
        part.data = AsContiguous(*data);
    }
    part.numRows = numRows;
    part.numCols = numCols;
    return part;
}

// Validate and collect every matrix of the call, accumulating its offsets.
CooUnion BuildUnion(const std::vector<torch::Tensor>& sources, const std::vector<torch::Tensor>& dests,
                    const std::vector<int64_t>& numRows, const std::vector<int64_t>& numCols,
                    const std::vector<std::optional<torch::Tensor>>& datas)
{
    CooUnion union_;
    union_.parts.reserve(sources.size());
    for (size_t i = 0; i < sources.size(); ++i) {
        const std::optional<torch::Tensor> data = i < datas.size() ? datas[i] : std::nullopt;
        CooPart part = MakePart(sources[i], dests[i], data, numRows[i], numCols[i]);
        CheckMatrix(part, i);
        CheckData(part, i);
        CheckExtents(union_, part);
        part.offsetRows = union_.totalRows;
        part.offsetCols = union_.totalCols;
        part.offsetEdges = union_.totalEdges;
        union_.hasData = union_.hasData || part.data.defined();
        union_.totalRows += part.numRows;
        union_.totalCols += part.numCols;
        union_.totalEdges += part.source.numel();
        union_.parts.push_back(std::move(part));
    }
    return union_;
}

// Launch one matrix: the data-less matrix fallback only needs a valid address,
// because the kernel reads `data` solely when the matrix has one and writes
// `outData` only when some matrix carries data.
template <typename T>
void LaunchPart(const CooPart& part, const CooOutputs<T>& outputs, uint32_t hasDataGlobal,
                aclrtStream stream, uint32_t ubBytes)
{
    const uint32_t edges = static_cast<uint32_t>(part.source.numel());
    T* source = static_cast<T*>(part.source.data_ptr());
    T* dest = static_cast<T*>(part.dest.data_ptr());
    T* data = part.data.defined() ? static_cast<T*>(part.data.data_ptr()) : source;
    T* destination = hasDataGlobal != 0 ? outputs.data : outputs.source;
    const uint32_t blocks = BlocksFor(part.source.numel());
    if constexpr (std::is_same<T, int32_t>::value) {
        DisjointUnionCooInt32(blocks, stream, source, dest, data, outputs.source, outputs.dest, destination,
                              part.offsetRows, part.offsetCols, part.offsetEdges, edges, outputs.totalEdges,
                              part.data.defined() ? 1u : 0u, hasDataGlobal, ubBytes);
    } else {
        DisjointUnionCooInt64(blocks, stream, source, dest, data, outputs.source, outputs.dest, destination,
                              part.offsetRows, part.offsetCols, part.offsetEdges, edges, outputs.totalEdges,
                              part.data.defined() ? 1u : 0u, hasDataGlobal, ubBytes);
    }
}

// Launch every non-empty matrix of the call on the current stream.
template <typename T>
void LaunchUnion(const CooUnion& union_, const CooOutputs<T>& outputs, uint32_t hasDataGlobal,
                 aclrtStream stream, uint32_t ubBytes)
{
    for (const auto& part : union_.parts) {
        if (part.source.numel() == 0) {
            continue;
        }
        LaunchPart<T>(part, outputs, hasDataGlobal, stream, ubBytes);
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t> DisjointUnionCoo(
    const std::vector<torch::Tensor>& sources, const std::vector<torch::Tensor>& dests,
    const std::vector<int64_t>& numRows, const std::vector<int64_t>& numCols,
    const std::vector<std::optional<torch::Tensor>>& datas)
{
    const size_t graphs = sources.size();
    TORCH_CHECK(graphs > 0, kApi, ": requires at least one COO matrix");
    TORCH_CHECK(dests.size() == graphs && numRows.size() == graphs && numCols.size() == graphs, kApi,
                ": sources, dests, num_rows and num_cols must have the same length");
    TORCH_CHECK(datas.empty() || datas.size() == graphs, kApi,
                ": datas must be empty or provide one entry per COO matrix");

    CooUnion union_ = BuildUnion(sources, dests, numRows, numCols, datas);
    const c10::Device device = sources.front().device();
    c10::DeviceGuard guard(device);
    const auto options = sources.front().options();
    torch::Tensor outSource = torch::empty({union_.totalEdges}, options);
    torch::Tensor outDest = torch::empty({union_.totalEdges}, options);
    torch::Tensor outData = union_.hasData ? torch::empty({union_.totalEdges}, options) : torch::Tensor();
    const int64_t totalRows = union_.totalRows;
    const int64_t totalCols = union_.totalCols;
    if (union_.totalEdges == 0) {
        return {std::move(outSource), std::move(outDest), std::move(outData), totalRows, totalCols};
    }

    const uint32_t ubBytes = QueryUbBytes();
    const aclrtStream stream = c10_npu::getCurrentNPUStream(device.index()).stream();
    const uint32_t hasDataGlobal = union_.hasData ? 1u : 0u;
    const uint32_t totalEdges = static_cast<uint32_t>(union_.totalEdges);
    if (sources.front().scalar_type() == at::kInt) {
        const CooOutputs<int32_t> outputs{static_cast<int32_t*>(outSource.data_ptr()),
                                          static_cast<int32_t*>(outDest.data_ptr()),
                                          union_.hasData ? static_cast<int32_t*>(outData.data_ptr()) : nullptr,
                                          totalEdges};
        LaunchUnion<int32_t>(union_, outputs, hasDataGlobal, stream, ubBytes);
    } else {
        const CooOutputs<int64_t> outputs{static_cast<int64_t*>(outSource.data_ptr()),
                                          static_cast<int64_t*>(outDest.data_ptr()),
                                          union_.hasData ? static_cast<int64_t*>(outData.data_ptr()) : nullptr,
                                          totalEdges};
        LaunchUnion<int64_t>(union_, outputs, hasDataGlobal, stream, ubBytes);
    }
    return {std::move(outSource), std::move(outDest), std::move(outData), totalRows, totalCols};
}

}  // namespace opsgnn

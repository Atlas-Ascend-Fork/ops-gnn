/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0.
 * Please refer to the License for details.
 */

#include "gather_coo.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <c10/core/DeviceType.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <tiling/platform/platform_ascendc.h>

#include "gather_coo/op_kernel/arch35/gather_coo.h"
#include "gather_coo/op_kernel/arch35/gather_coo_tiling.h"

namespace opsgnn {

namespace {

struct GatherCooShapeInfo {
    int64_t gatherDim = 0;
    uint32_t elementBytes = 0;
    uint64_t batchCount = 0;
    uint64_t sourceRows = 0;
    uint64_t indexRows = 0;
    uint64_t logicalFeatures = 0;
    uint64_t featureCount = 0;
    uint64_t totalRows = 0;
    uint64_t indexStorageElements = 0;
    uint64_t sourceStorageElements = 0;
    uint64_t outputStorageElements = 0;
    std::vector<int64_t> outputShape;
};

uint64_t CheckedMul(uint64_t lhs, uint64_t rhs, const char* name)
{
    TORCH_CHECK(
        rhs == 0 || lhs <= std::numeric_limits<uint64_t>::max() / rhs,
        "gather_coo ", name, " overflows uint64_t");
    return lhs * rhs;
}

uint64_t Product(const torch::Tensor& tensor, int64_t begin, int64_t end, const char* name)
{
    uint64_t result = 1;
    for (int64_t dim = begin; dim < end; ++dim) {
        result = CheckedMul(result, static_cast<uint64_t>(tensor.size(dim)), name);
    }
    return result;
}

uint32_t ElementBytes(at::ScalarType dtype)
{
    switch (dtype) {
        case at::kChar:
        case at::kByte:
            return 1;
        case at::kShort:
        case at::kHalf:
        case at::kBFloat16:
            return 2;
        case at::kInt:
        case at::kFloat:
            return 4;
        case at::kLong:
        case at::kDouble:
            return 8;
        default:
            TORCH_CHECK(false, "gather_coo does not support dtype ", dtype);
    }
    return 0;
}

void CheckNpuTensor(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name, " must be an NPU tensor");
}

void CheckSameShapeExceptGather(
    const torch::Tensor& out,
    const std::vector<int64_t>& expectedShape,
    const char* name)
{
    TORCH_CHECK(out.dim() == static_cast<int64_t>(expectedShape.size()), name, " rank mismatch");
    for (int64_t dim = 0; dim < out.dim(); ++dim) {
        TORCH_CHECK(
            out.size(dim) == expectedShape[dim],
            name, " shape mismatch at dim ", dim,
            "; expected ", expectedShape[dim], ", got ", out.size(dim));
    }
}

void ValidateInputs(const torch::Tensor& src, const torch::Tensor& index)
{
    CheckNpuTensor(src, "src");
    CheckNpuTensor(index, "index");
    TORCH_CHECK(src.device() == index.device(), "src and index must be on the same NPU");
    TORCH_CHECK(index.scalar_type() == at::kLong, "index must have dtype torch.int64");
    TORCH_CHECK(src.dim() >= 1 && src.dim() <= 8, "src rank must be in [1, 8]");
    TORCH_CHECK(
        index.dim() >= 1 && index.dim() <= src.dim(),
        "index rank must be in [1, src.dim()]");

    const int64_t gatherDim = index.dim() - 1;
    for (int64_t dim = 0; dim < gatherDim; ++dim) {
        TORCH_CHECK(
            src.size(dim) == index.size(dim),
            "src and index prefix shapes must match at dim ", dim);
    }
}

GatherCooShapeInfo BuildShapeInfo(
    const torch::Tensor& src,
    const torch::Tensor& index)
{
    GatherCooShapeInfo info;
    info.gatherDim = index.dim() - 1;
    info.elementBytes = ElementBytes(src.scalar_type());
    const uint32_t laneFactor = info.elementBytes == 8 ? 2 : 1;
    info.batchCount = Product(src, 0, info.gatherDim, "batchCount");
    info.sourceRows = static_cast<uint64_t>(src.size(info.gatherDim));
    info.indexRows = static_cast<uint64_t>(index.size(info.gatherDim));
    info.logicalFeatures = Product(
        src, info.gatherDim + 1, src.dim(), "featureCount");
    info.featureCount = CheckedMul(info.logicalFeatures, laneFactor, "featureCount");
    info.totalRows = CheckedMul(info.batchCount, info.indexRows, "totalRows");
    info.indexStorageElements = CheckedMul(info.totalRows, 2, "indexStorageElements");
    info.sourceStorageElements = CheckedMul(
        CheckedMul(info.batchCount, info.sourceRows, "sourceStorageElements"),
        info.featureCount,
        "sourceStorageElements");
    info.outputStorageElements = CheckedMul(
        info.totalRows, info.featureCount, "outputStorageElements");
    TORCH_CHECK(
        info.totalRows == 0 || info.sourceRows > 0,
        "index contains entries but src has zero rows along the gather dimension");
    info.outputShape.assign(src.sizes().begin(), src.sizes().end());
    info.outputShape[info.gatherDim] = static_cast<int64_t>(info.indexRows);
    return info;
}

torch::Tensor ValidateOptionalOut(
    const c10::optional<torch::Tensor>& optionalOut,
    const torch::Tensor& src,
    const std::vector<int64_t>& outputShape)
{
    if (!optionalOut.has_value()) {
        return {};
    }
    torch::Tensor out = optionalOut.value();
    CheckNpuTensor(out, "out");
    TORCH_CHECK(
        out.device() == src.device(),
        "out must be on the same NPU as src");
    TORCH_CHECK(
        out.scalar_type() == src.scalar_type(),
        "out dtype must match src dtype");
    CheckSameShapeExceptGather(out, outputShape, "out");
    return out;
}

GatherCooTilingData BuildTiling(const GatherCooShapeInfo& info)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t availableAiv = platform->GetCoreNumAiv();
    TORCH_CHECK(availableAiv > 0, "failed to query a valid AIV core count");
    const uint64_t coreCount64 =
        std::min<uint64_t>(availableAiv, info.totalRows);
    const uint32_t coreNum = static_cast<uint32_t>(
        std::max<uint64_t>(coreCount64, 1));

    GatherCooTilingData tiling{};
    tiling.batchCount = info.batchCount;
    tiling.sourceRows = info.sourceRows;
    tiling.indexRows = info.indexRows;
    tiling.featureCount = info.featureCount;
    tiling.logicalFeatureCount = info.logicalFeatures;
    tiling.elementBytes = info.elementBytes;
    tiling.totalRows = info.totalRows;
    tiling.indexStorageElements = info.indexStorageElements;
    tiling.sourceStorageElements = info.sourceStorageElements;
    tiling.outputStorageElements = info.outputStorageElements;
    tiling.rowsPerCore = info.totalRows / coreNum;
    tiling.extraCoreCount = info.totalRows % coreNum;
    tiling.featureTile = static_cast<uint32_t>(
        std::min<uint64_t>(info.featureCount, GATHER_COO_MAX_FEATURE_TILE));
    tiling.coreNum = coreNum;
    tiling.useRunCache = info.featureCount <= tiling.featureTile &&
        info.indexRows > info.sourceRows ? 1 : 0;
    const uint64_t rowBytes = info.logicalFeatures * info.elementBytes;
    const bool originalSlab = info.logicalFeatures == 128 && info.elementBytes <= 4;
    const bool alignedSlab = rowBytes > 0 && rowBytes <= 1024 && rowBytes % 32 == 0 &&
        info.logicalFeatures != 128 && tiling.rowsPerCore >= 4;
    if (originalSlab || alignedSlab) {
        tiling.outputSlabRows = static_cast<uint32_t>(std::min<uint64_t>(
            GATHER_COO_SLAB_MAX_ROWS, GATHER_COO_SLAB_BYTES / rowBytes));
    }
    TORCH_CHECK(tiling.featureTile > 0, "gather_coo feature tile must be positive");
    return tiling;
}

template <typename T>
void LaunchOnCurrentStream(
    const torch::Tensor& src,
    const torch::Tensor& index,
    const torch::Tensor& output,
    const GatherCooTilingData& tiling,
    const c10_npu::NPUStream& currentStream)
{
    // NPUStream::stream() is torch_npu's official queue-to-ACL bridge: it
    // drains pending host tasks before returning the managed ACL stream. The
    // kernel is then appended to that same stream and remains asynchronous;
    // this does not call a device synchronization API.
    aclrtStream stream = currentStream.stream();
    GatherCoo<T>(
        reinterpret_cast<T*>(src.data_ptr()),
        index.data_ptr<int64_t>(),
        reinterpret_cast<T*>(output.data_ptr()),
        tiling,
        stream);
}

void LaunchByDtype(
    const torch::Tensor& src,
    const torch::Tensor& index,
    const torch::Tensor& output,
    const GatherCooTilingData& tiling)
{
    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    switch (src.scalar_type()) {
        case at::kChar:
        case at::kByte:
            LaunchOnCurrentStream<uint8_t>(src, index, output, tiling, currentStream);
            return;
        case at::kShort:
        case at::kHalf:
        case at::kBFloat16:
            LaunchOnCurrentStream<uint16_t>(
                src, index, output, tiling, currentStream);
            return;
        case at::kInt:
        case at::kFloat:
        case at::kLong:
        case at::kDouble:
            LaunchOnCurrentStream<uint32_t>(
                src, index, output, tiling, currentStream);
            return;
        default:
            TORCH_CHECK(false, "gather_coo does not support dtype ", src.scalar_type());
    }
}

torch::Tensor CopyToOptionalOut(
    const c10::optional<torch::Tensor>& optionalOut,
    const torch::Tensor& originalOut,
    const torch::Tensor& output)
{
    if (!optionalOut.has_value()) {
        return output;
    }
    originalOut.copy_(output);
    return originalOut;
}

} // namespace

torch::Tensor gather_coo(
    torch::Tensor src,
    torch::Tensor index,
    c10::optional<torch::Tensor> optional_out)
{
    ValidateInputs(src, index);
    const GatherCooShapeInfo info = BuildShapeInfo(src, index);
    const torch::Tensor originalOut =
        ValidateOptionalOut(optional_out, src, info.outputShape);

    // These copies are enqueued by PyTorch on the caller's current stream.
    // The kernel itself only receives contiguous tensors.
    src = src.contiguous();
    index = index.contiguous();

    const torch::Tensor output = torch::empty(info.outputShape, src.options());
    if (info.totalRows != 0 && info.featureCount != 0) {
        const GatherCooTilingData tiling = BuildTiling(info);
        LaunchByDtype(src, index, output, tiling);
    }
    // Copying through the temporary output keeps non-contiguous and
    // storage-overlapping public out tensors safe.
    return CopyToOptionalOut(optional_out, originalOut, output);
}

}  // namespace opsgnn

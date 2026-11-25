/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_csr.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "gather_csr/op_kernel/arch35/gather_csr.h"
#include "gather_csr/op_kernel/arch35/gather_csr_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

namespace {

constexpr int64_t kMaxRank = 8;
constexpr uint32_t kTileBytes = 16 * 1024;

bool IsSupportedDtype(at::ScalarType dtype)
{
    switch (dtype) {
        case at::ScalarType::Half:
        case at::ScalarType::BFloat16:
        case at::ScalarType::Float:
        case at::ScalarType::Char:
        case at::ScalarType::Short:
        case at::ScalarType::Int:
        case at::ScalarType::Byte:
        case at::ScalarType::Double:
        case at::ScalarType::Long:
            return true;
        default:
            return false;
    }
}

uint64_t CheckedMultiply(uint64_t lhs, uint64_t rhs, const char* name)
{
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    TORCH_CHECK(lhs <= std::numeric_limits<uint64_t>::max() / rhs,
                "gather_csr: ", name, " overflows uint64");
    return lhs * rhs;
}

uint64_t CheckedProduct(at::IntArrayRef sizes, int64_t begin, int64_t end, const char* name)
{
    uint64_t result = 1;
    for (int64_t dim = begin; dim < end; ++dim) {
        const auto size = sizes[dim];
        TORCH_CHECK(size >= 0, "gather_csr: ", name, " contains a negative dimension");
        if (size == 0) {
            return 0;
        }
        result = CheckedMultiply(result, static_cast<uint64_t>(size), name);
    }
    return result;
}

uint64_t CheckedCeilDivide(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        TORCH_CHECK(false, "gather_csr: internal error: division by zero");
        return 0;
    }
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

void CheckNpuTensor(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), "gather_csr: ", name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "gather_csr: ", name, " must be an NPU tensor");
}

struct ValidatedArguments {
    int64_t dim;
    bool hasOut;
    torch::Tensor out;
};

ValidatedArguments ValidateArguments(const torch::Tensor& src, const torch::Tensor& indptr,
                                     const c10::optional<torch::Tensor>& optionalOut)
{
    CheckNpuTensor(src, "src");
    CheckNpuTensor(indptr, "indptr");
    TORCH_CHECK(src.device() == indptr.device(), "gather_csr: src and indptr must be on the same NPU");
    TORCH_CHECK(IsSupportedDtype(src.scalar_type()), "gather_csr: unsupported src dtype ", src.scalar_type());
    TORCH_CHECK(indptr.scalar_type() == at::ScalarType::Long, "gather_csr: indptr dtype must be int64");
    TORCH_CHECK(src.dim() >= 1 && src.dim() <= kMaxRank,
                "gather_csr: src rank must be in [1, ", kMaxRank, "], got ", src.dim());
    TORCH_CHECK(indptr.dim() >= 1 && indptr.dim() <= src.dim(),
                "gather_csr: indptr rank must be in [1, src.dim()], got ", indptr.dim());

    ValidatedArguments args{indptr.dim() - 1, optionalOut.has_value(), torch::Tensor{}};
    if (!args.hasOut) {
        return args;
    }
    args.out = optionalOut.value();
    CheckNpuTensor(args.out, "out");
    TORCH_CHECK(args.out.device() == src.device(), "gather_csr: out and src must be on the same NPU");
    TORCH_CHECK(args.out.scalar_type() == src.scalar_type(), "gather_csr: out dtype must match src dtype");
    TORCH_CHECK(args.out.dim() == src.dim(), "gather_csr: out rank must match src rank");
    for (int64_t axis = 0; axis < src.dim(); ++axis) {
        if (axis != args.dim) {
            TORCH_CHECK(args.out.size(axis) == src.size(axis),
                        "gather_csr: out shape must match src outside dimension ", args.dim);
        }
    }
    return args;
}

void CheckBroadcastShape(const torch::Tensor& src, const torch::Tensor& indptr, int64_t dim)
{
    for (int64_t axis = 0; axis < dim; ++axis) {
        TORCH_CHECK(indptr.size(axis) == 1 || indptr.size(axis) == src.size(axis),
                    "gather_csr: indptr dimension ", axis, " with size ", indptr.size(axis),
                    " cannot broadcast to src size ", src.size(axis));
    }
}

void CheckSegmentCount(const torch::Tensor& src, const torch::Tensor& indptr, int64_t dim)
{
    const int64_t ptrWidth = indptr.size(-1);
    if (ptrWidth == 0) {
        TORCH_CHECK(src.size(dim) == 0, "gather_csr: an empty indptr requires src.size(", dim,
                    ") to be 0, got ", src.size(dim));
        return;
    }

    const int64_t segmentCount = ptrWidth - 1;
    TORCH_CHECK(src.size(dim) == segmentCount, "gather_csr: src.size(", dim,
                ") must equal indptr.size(-1) - 1; got ", src.size(dim), " and ", segmentCount);
}

struct IndptrInfo {
    int64_t outputRows;
    uint64_t maxSegmentLength;
    bool hasUncoveredPrefix;
};

IndptrInfo ValidateIndptr(const torch::Tensor& indptr, uint64_t batchCount, uint64_t segmentCount)
{
    auto cpu = indptr.to(torch::kCPU).contiguous();
    const auto* values = cpu.data_ptr<int64_t>();
    const uint64_t rowWidth = segmentCount + 1;
    TORCH_CHECK(batchCount > 0, "gather_csr: internal error: non-empty src has zero batch count");
    const int64_t commonEndpoint = values[segmentCount];
    TORCH_CHECK(commonEndpoint >= 0, "gather_csr: indptr endpoint must be non-negative");

    uint64_t maxSegmentLength = 0;
    bool hasUncoveredPrefix = false;
    for (uint64_t batch = 0; batch < batchCount; ++batch) {
        const auto* row = values + batch * rowWidth;
        TORCH_CHECK(row[segmentCount] == commonEndpoint,
                    "gather_csr: all broadcasted indptr rows must have the same endpoint; batch ", batch,
                    " has endpoint ", row[segmentCount], " but expected ", commonEndpoint);
        hasUncoveredPrefix = hasUncoveredPrefix || row[0] > 0;
        for (uint64_t segment = 0; segment <= segmentCount; ++segment) {
            TORCH_CHECK(row[segment] >= 0 && row[segment] <= commonEndpoint,
                        "gather_csr: indptr value out of range at batch ", batch, ", index ", segment,
                        ": expected [0, ", commonEndpoint, "], got ", row[segment]);
        }
        for (uint64_t segment = 0; segment < segmentCount; ++segment) {
            TORCH_CHECK(row[segment] <= row[segment + 1],
                        "gather_csr: indptr must be non-decreasing at batch ", batch,
                        ", indices ", segment, " and ", segment + 1);
            maxSegmentLength = std::max(maxSegmentLength,
                static_cast<uint64_t>(row[segment + 1] - row[segment]));
        }
    }
    return {commonEndpoint, maxSegmentLength, hasUncoveredPrefix};
}

torch::Tensor HandleEmptyInput(const torch::Tensor& src, const torch::Tensor& indptr,
                               const ValidatedArguments& args)
{
    int64_t expectedRows = 0;
    bool hasUncoveredPrefix = false;
    if (indptr.numel() > 0) {
        const uint64_t segmentCount = static_cast<uint64_t>(indptr.size(-1) - 1);
        const uint64_t ptrBatchCount = CheckedProduct(indptr.sizes(), 0, args.dim, "indptr");
        const auto ptrInfo = ValidateIndptr(indptr, ptrBatchCount, segmentCount);
        expectedRows = ptrInfo.outputRows;
        hasUncoveredPrefix = ptrInfo.hasUncoveredPrefix;
    }
    if (args.hasOut) {
        TORCH_CHECK(args.out.size(args.dim) == expectedRows,
                    "gather_csr: out.size(", args.dim, ") must equal indptr endpoint ", expectedRows,
                    " for an empty src, got ", args.out.size(args.dim));
        if (hasUncoveredPrefix) {
            args.out.zero_();
        }
        return args.out;
    }
    auto shape = src.sizes().vec();
    shape[args.dim] = expectedRows;
    return torch::zeros(shape, src.options());
}

struct ContiguousInputs {
    torch::Tensor src;
    torch::Tensor indptr;
};

ContiguousInputs PrepareInputs(const torch::Tensor& src, const torch::Tensor& indptr, int64_t dim)
{
    auto expandedShape = indptr.sizes().vec();
    for (int64_t axis = 0; axis < dim; ++axis) {
        expandedShape[axis] = src.size(axis);
    }
    return {src.contiguous(), indptr.expand(expandedShape).contiguous()};
}

struct LaunchShapeInfo {
    uint64_t batchCount;
    uint64_t segmentCount;
    uint64_t featureBytes;
    uint64_t totalSegments;
    uint64_t totalOutputRows;
    IndptrInfo indptr;
    std::vector<int64_t> outShape;
};

LaunchShapeInfo BuildLaunchShapeInfo(const torch::Tensor& src, const torch::Tensor& indptr,
                                     int64_t dim, const torch::Tensor& contiguousIndptr)
{
    TORCH_CHECK(indptr.size(-1) >= 1, "gather_csr: non-empty src requires a non-empty indptr");
    const int64_t segmentCountSigned = indptr.size(-1) - 1;
    TORCH_CHECK(segmentCountSigned > 0, "gather_csr: non-empty src requires at least one segment");
    TORCH_CHECK(src.size(dim) == segmentCountSigned,
                "gather_csr: src.size(", dim, ") must equal indptr.size(-1) - 1; got ",
                src.size(dim), " and ", segmentCountSigned);

    const uint64_t batchCount = CheckedProduct(src.sizes(), 0, dim, "src batch count");
    const uint64_t segmentCount = static_cast<uint64_t>(segmentCountSigned);
    const uint64_t featureElements = CheckedProduct(src.sizes(), dim + 1, src.dim(), "src feature size");
    const uint64_t featureBytes = CheckedMultiply(featureElements, src.element_size(), "feature byte count");
    const uint64_t totalSegments = CheckedMultiply(batchCount, segmentCount, "total segment count");
    CheckedMultiply(batchCount, segmentCount + 1, "expanded indptr element count");
    CheckedMultiply(totalSegments, featureBytes, "src byte span");

    const uint64_t ptrBatchCount = CheckedProduct(indptr.sizes(), 0, dim, "indptr batch count");
    const auto& validationIndptr = ptrBatchCount < batchCount ? indptr : contiguousIndptr;
    const auto ptrInfo = ValidateIndptr(validationIndptr, ptrBatchCount, segmentCount);
    const uint64_t outputRows = static_cast<uint64_t>(ptrInfo.outputRows);
    const uint64_t totalOutputRows = CheckedMultiply(batchCount, outputRows, "total output row count");
    CheckedMultiply(totalOutputRows, featureBytes, "out byte span");
    auto outShape = src.sizes().vec();
    outShape[dim] = ptrInfo.outputRows;
    return {batchCount, segmentCount, featureBytes, totalSegments,
            totalOutputRows, ptrInfo, std::move(outShape)};
}

void PreserveAliasedInputs(const ValidatedArguments& args, const torch::Tensor& src,
                           const torch::Tensor& indptr, ContiguousInputs& inputs)
{
    if (args.hasOut && args.out.is_alias_of(src)) {
        inputs.src = src.clone(at::MemoryFormat::Contiguous);
    }
    if (args.hasOut && args.out.is_alias_of(indptr)) {
        inputs.indptr = inputs.indptr.clone(at::MemoryFormat::Contiguous);
    }
}

torch::Tensor PrepareKernelOutput(const torch::Tensor& src, const ValidatedArguments& args,
                                  const LaunchShapeInfo& shapeInfo)
{
    if (args.hasOut) {
        TORCH_CHECK(args.out.size(args.dim) == shapeInfo.indptr.outputRows,
                    "gather_csr: out.size(", args.dim, ") must equal indptr endpoint ",
                    shapeInfo.indptr.outputRows, ", got ", args.out.size(args.dim));
    }
    torch::Tensor kernelOut = args.hasOut && args.out.is_contiguous()
        ? args.out : torch::empty(shapeInfo.outShape, src.options());
    if (shapeInfo.indptr.hasUncoveredPrefix) {
        kernelOut.zero_();
    }
    return kernelOut;
}

GatherCsrTilingData BuildTiling(const LaunchShapeInfo& shapeInfo)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "gather_csr: failed to initialize AscendC platform information");
    const uint32_t availableCores = std::max<uint32_t>(platform->GetCoreNumAiv(), 1);
    const uint64_t outputRows = static_cast<uint64_t>(shapeInfo.indptr.outputRows);
    const uint64_t averageSegmentLength = CheckedCeilDivide(outputRows, shapeInfo.segmentCount);
    const bool heavilySkewed = averageSegmentLength > 0 &&
        averageSegmentLength <= std::numeric_limits<uint64_t>::max() / 4 &&
        shapeInfo.indptr.maxSegmentLength > 4 * averageSegmentLength;
    const bool useOutputMajor = shapeInfo.totalSegments < availableCores || heavilySkewed;
    const uint64_t totalJobs = useOutputMajor ? shapeInfo.totalOutputRows : shapeInfo.totalSegments;
    const uint32_t activeCores = static_cast<uint32_t>(std::min<uint64_t>(availableCores, totalJobs));
    TORCH_CHECK(activeCores > 0, "gather_csr: non-empty input must contain at least one segment");

    GatherCsrTilingData tiling{};
    tiling.batchCount = shapeInfo.batchCount;
    tiling.segmentCount = shapeInfo.segmentCount;
    tiling.outputRows = outputRows;
    tiling.featureBytes = shapeInfo.featureBytes;
    tiling.totalSegments = shapeInfo.totalSegments;
    tiling.tileBytes = kTileBytes;
    tiling.activeCoreNum = activeCores;
    tiling.scheduleMode = useOutputMajor ? GATHER_CSR_OUTPUT_MAJOR : GATHER_CSR_SEGMENT_MAJOR;
    return tiling;
}

torch::Tensor RunKernel(const torch::Tensor& src, const ValidatedArguments& args,
                        const ContiguousInputs& inputs, const LaunchShapeInfo& shapeInfo)
{
    auto kernelOut = PrepareKernelOutput(src, args, shapeInfo);
    if (kernelOut.numel() > 0) {
        const auto tiling = BuildTiling(shapeInfo);
        aclrtStream stream = c10_npu::getCurrentNPUStream(src.get_device()).stream();
        GatherCsr(inputs.src.data_ptr(), inputs.indptr.data_ptr<int64_t>(),
                              kernelOut.data_ptr(), tiling, stream);
    }
    if (args.hasOut && !args.out.is_contiguous()) {
        args.out.copy_(kernelOut);
    }
    return args.hasOut ? args.out : kernelOut;
}

} // namespace

torch::Tensor gather_csr(torch::Tensor src, torch::Tensor indptr,
                         c10::optional<torch::Tensor> optionalOut)
{
    const auto args = ValidateArguments(src, indptr, optionalOut);
    CheckBroadcastShape(src, indptr, args.dim);
    CheckSegmentCount(src, indptr, args.dim);
    if (src.numel() == 0) {
        return HandleEmptyInput(src, indptr, args);
    }

    auto inputs = PrepareInputs(src, indptr, args.dim);
    const auto shapeInfo = BuildLaunchShapeInfo(src, indptr, args.dim, inputs.indptr);
    PreserveAliasedInputs(args, src, indptr, inputs);
    return RunKernel(src, args, inputs, shapeInfo);
}

}  // namespace opsgnn

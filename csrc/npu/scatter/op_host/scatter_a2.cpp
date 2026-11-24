/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifdef OPSGNN_DAV_2201
#include "scatter.h"
#include "scatter/op_kernel/arch22/scatter.h"
#include "scatter_groups.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include <ATen/MemoryOverlap.h>
#include <ATen/record_function.h>
#include <algorithm>
#include <array>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <torch/extension.h>
#include <unordered_map>

namespace opsgnn {
uint32_t ScatterA2DtypeCode(at::ScalarType dtype, const char *errorMessage) {
    using namespace scatter_a2;
    switch (dtype) {
    case at::kHalf:
        return Float16;
    case at::kBFloat16:
        return BFloat16;
    case at::kFloat:
        return Float32;
    case at::kChar:
        return Int8;
    case at::kShort:
        return Int16;
    case at::kInt:
        return Int32;
    case at::kByte:
        return UInt8;
    default:
        TORCH_CHECK(false, errorMessage);
    }
}

namespace {
using namespace scatter_a2;

struct IndexStatsResource {
    void *host = nullptr;
    aclrtEvent event = nullptr;
    uint64_t capacity = 0;
    int device = 0;
    aclrtStream stream = nullptr;
    bool submitted = false;
    bool recorded = false;
    std::array<c10::Storage, 2> owners;

    ~IndexStatsResource() {
        if (event)
            aclrtDestroyEvent(event);
        if (host)
            aclrtFreeHost(host);
    }
};

struct IndexStatsDevices {
    std::mutex mutex;
    std::unordered_map<int, bool> poisoned;
};

IndexStatsDevices &StatsDevices() {
    // Retain the small registry at shutdown, including TLS destruction.
    static auto *devices = new IndexStatsDevices;
    return *devices;
}

void CheckStatsDevice(int device) {
    auto &state = StatsDevices();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(!state.poisoned.try_emplace(device, false).first->second,
                "scatter index stats resources quarantined after synchronization failure");
}

void QuarantineStatsResource(std::unique_ptr<IndexStatsResource> &resource) noexcept {
    // Keep both DMA endpoints and the event alive even during stack unwinding.
    auto *retained = resource.release();
    if (!retained)
        return;
    try {
        auto &state = StatsDevices();
        std::lock_guard<std::mutex> lock(state.mutex);
        const auto found = state.poisoned.find(retained->device);
        if (found != state.poisoned.end())
            found->second = true;
    } catch (...) {
        // Retaining the resource must not depend on allocating a registry entry.
    }
}

void CompleteStatsResource(IndexStatsResource &resource) {
    resource.submitted = false;
    resource.recorded = false;
    resource.owners = {};
}

void RecoverStatsResource(std::unique_ptr<IndexStatsResource> &resource) noexcept {
    if (!resource || !resource->submitted)
        return;
    try {
        const c10::DeviceGuard guard(c10::Device(c10::DeviceType::PrivateUse1, resource->device));
        bool complete = false;
        if (resource->recorded)
            complete = aclrtSynchronizeEvent(resource->event) == ACL_SUCCESS;
        else {
            uint32_t flags = 0;
            constexpr uint32_t noWait = ACL_STREAM_PERSISTENT | ACL_STREAM_DEVICE_USE_ONLY | ACL_STREAM_CPU_SCHEDULE;
            if (aclrtStreamGetFlags(resource->stream, &flags) == ACL_SUCCESS && !(flags & noWait))
                complete = aclrtSynchronizeStream(resource->stream) == ACL_SUCCESS;
        }
        if (complete)
            CompleteStatsResource(*resource);
        else
            QuarantineStatsResource(resource);
    } catch (...) {
        QuarantineStatsResource(resource);
    }
}

void ReleaseStatsResource(std::unique_ptr<IndexStatsResource> &resource) noexcept {
    RecoverStatsResource(resource);
    if (!resource)
        return;
    try {
        const c10::DeviceGuard guard(c10::Device(c10::DeviceType::PrivateUse1, resource->device));
        resource.reset();
    } catch (...) {
        QuarantineStatsResource(resource);
    }
}

class IndexStatsHostBuffer {
  public:
    ~IndexStatsHostBuffer() { ReleaseStatsResource(resource_); }

    int64_t *Run(const torch::Tensor &index, const torch::Tensor &stats, uint32_t blocks, aclrtStream stream) {
        const uint64_t bytes = blocks * kDmaBytes;
        Prepare(bytes, index.get_device());
        resource_->stream = stream;
        resource_->owners = {index.storage(), stats.storage()};
        auto *source = index.data_ptr<int64_t>(), *destination = stats.data_ptr<int64_t>();
        resource_->submitted = true;
        try {
            TORCH_CHECK(ScatterA2IndexStats(source, index.numel(), destination, blocks, stream) == ACL_SUCCESS,
                        "scatter index stats launch failed");
            TORCH_CHECK(aclrtMemcpyAsync(resource_->host, bytes, destination, bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                                         stream) == ACL_SUCCESS,
                        "scatter index stats transfer failed");
            TORCH_CHECK(aclrtRecordEvent(resource_->event, stream) == ACL_SUCCESS,
                        "scatter index stats event recording failed");
            resource_->recorded = true;
            Synchronize();
        } catch (...) {
            RecoverStatsResource(resource_);
            throw;
        }
        return static_cast<int64_t *>(resource_->host);
    }

  private:
    void Prepare(uint64_t bytes, int device) {
        CheckStatsDevice(device);
        TORCH_CHECK(!resource_ || !resource_->submitted, "scatter index stats buffer is still in use");
        if (resource_ && (resource_->capacity < bytes || resource_->device != device))
            ReleaseStatsResource(resource_);
        CheckStatsDevice(device);
        if (resource_)
            return;
        const c10::DeviceGuard guard(c10::Device(c10::DeviceType::PrivateUse1, device));
        auto fresh = std::make_unique<IndexStatsResource>();
        fresh->device = device;
        fresh->capacity = bytes;
        TORCH_CHECK(aclrtMallocHost(&fresh->host, bytes) == ACL_SUCCESS, "scatter index stats host allocation failed");
        TORCH_CHECK(aclrtCreateEventExWithFlag(&fresh->event, ACL_EVENT_SYNC) == ACL_SUCCESS,
                    "scatter index stats reusable event creation failed");
        resource_ = std::move(fresh);
    }

    void Synchronize() {
        const aclError status = aclrtSynchronizeEvent(resource_->event);
        if (status != ACL_SUCCESS)
            QuarantineStatsResource(resource_);
        TORCH_CHECK(status == ACL_SUCCESS, "scatter index stats synchronization failed");
        CompleteStatsResource(*resource_);
    }

    std::unique_ptr<IndexStatsResource> resource_;
};

uint64_t CheckedProduct(const at::IntArrayRef &sizes, int64_t first, int64_t last) {
    uint64_t product = 1;
    for (int64_t axis = first; axis < last; ++axis) {
        const auto size = static_cast<uint64_t>(sizes[axis]);
        TORCH_CHECK(!size || product <= INT64_MAX / size, "scatter shape product overflows int64");
        product *= size;
    }
    return product;
}

uint64_t SpanElements(const torch::Tensor &tensor) {
    if (!tensor.numel())
        return 0;
    uint64_t span = 1;
    for (int64_t axis = 0; axis < tensor.dim(); ++axis) {
        TORCH_CHECK(tensor.stride(axis) >= 0, "scatter requires nonnegative tensor strides");
        const uint64_t count = tensor.size(axis) - 1, stride = tensor.stride(axis);
        TORCH_CHECK(!count || stride <= (INT64_MAX - span) / count, "scatter tensor span overflows");
        span += count * stride;
    }
    TORCH_CHECK(span <= INT64_MAX / tensor.element_size(), "scatter tensor span bytes overflow");
    return span;
}

bool ByteOverlap(const torch::Tensor &a, const torch::Tensor &b) {
    const uint64_t aBytes = SpanElements(a) * a.element_size();
    const uint64_t bBytes = SpanElements(b) * b.element_size();
    const auto aStart = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto bStart = reinterpret_cast<uintptr_t>(b.data_ptr());
    TORCH_CHECK(aBytes <= UINTPTR_MAX - aStart && bBytes <= UINTPTR_MAX - bStart, "scatter tensor address overflows");
    return aBytes && bBytes && aStart < bStart + bBytes && bStart < aStart + aBytes;
}

void ValidateTensors(const torch::Tensor &src, const torch::Tensor &index, const torch::Tensor &out) {
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "scatter expects NPU tensors");
    TORCH_CHECK(src.device() == index.device() && src.device() == out.device(),
                "src, index and out must use the same NPU device");
    TORCH_CHECK(index.scalar_type() == at::kLong, "scatter index must have int64 dtype");
    TORCH_CHECK(src.scalar_type() == out.scalar_type(), "scatter src and out dtypes must match");
    TORCH_CHECK(src.dim() >= 1 && src.dim() <= 8, "scatter requires 1 to 8 source dimensions");
    TORCH_CHECK(out.dim() == src.dim(), "scatter out and src ranks must match");
    TORCH_CHECK(index.dim() <= src.dim(), "scatter index rank must not exceed src rank");
    for (const auto &tensor : {src, index, out}) {
        const uint64_t elements = static_cast<uint64_t>(tensor.numel());
        TORCH_CHECK(elements <= INT64_MAX / tensor.element_size(), "scatter tensor byte size overflows");
    }
}

torch::Tensor BroadcastIndex(const torch::Tensor &src, const torch::Tensor &index, int64_t dim) {
    auto result = index;
    if (index.dim() == 1) {
        std::vector<int64_t> shape(src.dim(), 1);
        shape[dim] = index.numel();
        result = index.view(shape);
    } else {
        while (result.dim() < src.dim())
            result = result.unsqueeze(-1);
    }
    return result.expand(src.sizes());
}

void ValidateGeometry(const torch::Tensor &src, const torch::Tensor &index, const torch::Tensor &out, int64_t dim,
                      int64_t reduce) {
    TORCH_CHECK(dim >= 0 && dim < src.dim(), "scatter dimension is out of range");
    TORCH_CHECK(reduce == Sum || reduce == Mul || reduce == Mean || reduce == Min || reduce == Max,
                "invalid scatter reduction code");
    for (int64_t axis = 0; axis < src.dim(); ++axis)
        TORCH_CHECK(axis == dim || src.size(axis) == out.size(axis), "scatter out shape is incompatible");
    at::assert_no_internal_overlap(out);
    if (reduce == Sum || reduce == Mean) {
        at::assert_no_overlap(src, out);
        at::assert_no_overlap(BroadcastIndex(src, index, dim), out);
    } else
        TORCH_CHECK(out.is_contiguous(), "scatter mul/min/max require contiguous out");
}

bool CanUseGroups(const ScatterGroupedRequest &request) {
    const auto &src = request.src;
    const auto &index = request.index;
    ValidateTensors(src, index, request.out ? *request.out : src);
    TORCH_CHECK(request.reduce >= Sum && request.reduce <= Max, "invalid scatter reduction code");
    SpanElements(src);
    SpanElements(index);
    if (request.out)
        ValidateGeometry(src, index, *request.out, request.dim, request.reduce);
    if (index.dim() != 1 || !src.numel())
        return false;
    return !request.out || (!ByteOverlap(src, *request.out) && !ByteOverlap(index, *request.out));
}

bool CanUseGroupsBeforeBroadcast(const ScatterGroupedRequest &request) {
    if (request.out || request.index.dim() != 1 || !request.src.numel())
        return false;
    TORCH_CHECK(request.index.numel() == 1 || request.index.numel() == request.src.size(request.dim),
                "scatter compact index length must be one or match the source dimension");
    return CanUseGroups(request);
}

Tiling MakeTiling(const torch::Tensor &src, const torch::Tensor &index, const torch::Tensor &out, int64_t dim,
                  int64_t reduce, bool hasOut) {
    Tiling t{};
    t.before = CheckedProduct(src.sizes(), 0, dim);
    t.length = src.size(dim);
    t.after = CheckedProduct(src.sizes(), dim + 1, src.dim());
    t.buckets = out.size(dim);
    t.srcElements = src.numel();
    t.outElements = out.numel();
    t.indexElements = index.numel();
    t.reduce = reduce;
    t.dtype = ScatterA2DtypeCode(src.scalar_type(), "scatter A2 native path requires an L1 dtype");
    t.hasOut = hasOut;
    t.rank = src.dim();
    t.dim = dim;
    t.sourceSpan = SpanElements(src);
    t.outputSpan = SpanElements(out);
    t.indexSpan = SpanElements(index);
    t.indexMutable = ByteOverlap(index, out);
    t.compactIndex = index.dim() == 1 && (t.indexElements == 1 || t.indexElements == t.length);
    TORCH_CHECK(t.compactIndex || index.sizes() == src.sizes(),
                "native scatter index must be a dimension vector or an expanded "
                "src-shaped tensor");
    for (int64_t axis = 0; axis < index.dim(); ++axis)
        t.indexStrides[axis] = index.stride(axis);
    for (int64_t axis = 0; axis < src.dim(); ++axis) {
        t.sizes[axis] = src.size(axis);
        t.sourceStrides[axis] = src.stride(axis);
        t.outputStrides[axis] = out.stride(axis);
    }
    return t;
}

uint64_t AtomicUbBytes(const Tiling &t, uint32_t itemBytes, bool arguments = false) {
    const uint32_t rows = arguments ? t.argumentRows : t.atomicRows;
    const uint64_t elements = t.atomicWidth * rows;
    uint64_t bytes = elements * itemBytes + (t.dtype == Float32 ? 0 : elements * 4);
    bytes += (rows * 8 + kDmaBytes - 1) / kDmaBytes * kDmaBytes;
    if (t.reduce == Min || t.reduce == Max)
        bytes += (elements + kMaskElementsPerDmaBlock - 1) / kMaskElementsPerDmaBlock *
                 kDmaBytes; // Full-tile comparison mask.
    if (arguments) {
        bytes += elements * 8; // Final extrema and candidate argument positions.
        if (t.hasOut)
            bytes += elements * itemBytes + (t.dtype == Float32 ? 0 : elements * 4);
    }
    return bytes + kUbReserveBytes;
}

void PlanOptimized(Tiling &t, uint64_t ubBytes, uint32_t itemBytes) {
    if (t.reduce == Sum && t.dtype > Float32 && t.compactIndex && t.before == 1 && t.after == 1 &&
        t.length >= kBucketSourceRows) {
        const uint64_t fixed = kUbReserveBytes + kBucketSourceRows * (itemBytes + 8) + kBucketConversionElements * 4;
        TORCH_CHECK(ubBytes > fixed + kBucketConversionElements * 4, "scatter bucket table exceeds device UB");
        const uint64_t capacity = (ubBytes - fixed) / 4 / kBucketTableAlignment * kBucketTableAlignment;
        const uint64_t requested =
            (t.buckets + kBucketTableAlignment - 1) / kBucketTableAlignment * kBucketTableAlignment;
        t.atomicWidth = std::min<uint64_t>(std::min<uint64_t>(capacity, requested), kMaxBucketTableElements);
        t.path = BucketTable;
        t.units = (t.length + kBucketSourceRows - 1) / kBucketSourceRows;
    } else if (t.path == FeatureOwner && (t.reduce == Min || t.reduce == Max) &&
               t.srcElements >= kMinAtomicSourceElements) {
        t.path = Atomic;
        t.atomicWidth = std::min<uint64_t>(
            (t.after + kVectorMaskElements - 1) / kVectorMaskElements * kVectorMaskElements, kMaxAtomicFeatureElements);
        t.atomicRows = kMaxAtomicSourceRows;
        while (AtomicUbBytes(t, itemBytes) > ubBytes && t.atomicRows > 1)
            t.atomicRows /= 2;
        TORCH_CHECK(AtomicUbBytes(t, itemBytes) <= ubBytes, "scatter source tile exceeds device UB");
        t.argumentRows = kMaxAtomicSourceRows;
        while (AtomicUbBytes(t, itemBytes, true) > ubBytes && t.argumentRows > 1)
            t.argumentRows /= 2;
        TORCH_CHECK(AtomicUbBytes(t, itemBytes, true) <= ubBytes, "scatter argument tile exceeds device UB");
        t.units =
            t.before * ((t.after + t.atomicWidth - 1) / t.atomicWidth) * ((t.length + t.atomicRows - 1) / t.atomicRows);
    }
}

void PlanLaunch(Tiling &t, const torch::Tensor &src, const torch::Tensor &index, const torch::Tensor &out) {
    if (!t.outElements) {
        t.blocks = 1;
        return;
    }
    const bool aliases = ByteOverlap(src, out) || ByteOverlap(index, out);
    const bool aligned =
        t.after * out.element_size() % kDmaBytes == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % kDmaBytes == 0;
    const bool vector = !aliases && t.compactIndex && t.after >= kFeatureTileElements && aligned &&
                        t.dtype <= Float32 && t.length <= kExactFloatIndexLimit;
    t.path = aliases ? AliasedOrdered : (vector ? FeatureOwner : OutputOwner);
    t.units = vector ? t.before * ((t.after + kFeatureTileElements - 1) / kFeatureTileElements)
                     : (t.outElements + kScalarOutputElements - 1) / kScalarOutputElements;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    // Source, accumulator, float conversion, arg positions, row positions, INT64
    // arg, mask.
    TORCH_CHECK(ubBytes >=
                    kUbReserveBytes + 7 * kOutputTileElements * sizeof(float) +
                        (kOutputTileElements + kMaskElementsPerDmaBlock - 1) / kMaskElementsPerDmaBlock * kDmaBytes,
                "scatter A2 device UB is too small");
    if (!aliases)
        PlanOptimized(t, ubBytes, out.element_size());
    if (t.reduce == Mean) {
        TORCH_CHECK((t.path == FeatureOwner || t.path == Atomic) && t.srcElements && t.length <= kExactFloatIndexLimit,
                    "native fused mean requires a nonempty compact floating feature path");
        const uint64_t fixed = kUbReserveBytes + kBucketSourceRows * sizeof(int64_t);
        const uint64_t capacity = (ubBytes - fixed) / 4 / kBucketTableAlignment * kBucketTableAlignment;
        const uint64_t requested =
            (t.buckets + kBucketTableAlignment - 1) / kBucketTableAlignment * kBucketTableAlignment;
        t.countWidth = std::min<uint64_t>(std::min<uint64_t>(capacity, requested), kMaxBucketTableElements);
    }
    const uint32_t cores = std::max<uint32_t>(platform->GetCoreNumAiv(), 1);
    t.blocks = aliases ? 1 : static_cast<uint32_t>(std::min<uint64_t>(cores, std::max<uint64_t>(t.units, 1)));
}

void LaunchScatter(const torch::Tensor &src, const torch::Tensor &index, const torch::Tensor &out,
                   const torch::Tensor &arg, const Tiling &t) {
    const bool vector = t.path == FeatureOwner || t.path == Atomic || t.path == BucketTable;
    const auto options = out.options().dtype(at::kFloat);
    const bool directOutput = t.dtype == Float32 && (t.reduce == Sum || t.reduce == Mean) &&
                              (t.path == Atomic || t.path == FeatureOwner || t.path == BucketTable);
    auto workspace = directOutput ? out : torch::empty({vector ? out.numel() : 0}, options);
    auto argWorkspace = torch::empty({vector && arg.numel() ? out.numel() : 0}, options);
    auto counts = torch::empty({t.reduce == Mean ? static_cast<int64_t>(t.buckets) : 0}, out.options().dtype(at::kInt));
    const auto current = c10_npu::getCurrentNPUStream(src.get_device());
    std::array<c10::Storage, 7> owners{src.storage(),       index.storage(),        out.storage(),   arg.storage(),
                                       workspace.storage(), argWorkspace.storage(), counts.storage()};
    for (const auto &owner : owners)
        c10_npu::NPUCachingAllocator::recordStream(owner.data_ptr(), current);
    void *source = src.data_ptr(), *destination = out.data_ptr();
    auto *indices = index.data_ptr<int64_t>();
    auto *arguments = arg.numel() ? arg.data_ptr<int64_t>() : nullptr;
    auto *work = workspace.numel() ? workspace.data_ptr<float>() : nullptr;
    auto *argWork = argWorkspace.numel() ? argWorkspace.data_ptr<float>() : nullptr;
    auto *countValues = counts.numel() ? counts.data_ptr<int32_t>() : nullptr;
    const auto stream = current.stream(false);
    at_npu::native::OpCommand::RunOpApi("scatter_a2", [=, owners = std::move(owners)]() {
        return static_cast<int>(
            ScatterA2(source, indices, destination, arguments, work, argWork, countValues, t, stream));
    });
}
} // namespace

std::tuple<torch::Tensor, torch::Tensor> scatter_forward(torch::Tensor src, torch::Tensor index, int64_t dim,
                                                         torch::Tensor out, int64_t reduce, bool hasOut,
                                                         int64_t hotTarget) {
    ValidateTensors(src, index, out);
    ValidateGeometry(src, index, out, dim, reduce);
    // hot_target selects an optimized path only on arch35. A2 retains and
    // validates the argument for API compatibility; it does not select an A2 path.
    TORCH_CHECK(hotTarget >= -1, "scatter hot_target must be -1 or non-negative");
    const c10::DeviceGuard guard(src.device());
    const auto originalOut = out;
    const bool sourceOverlap = ByteOverlap(src, out), indexOverlap = ByteOverlap(index, out);
    const bool overlaps = (reduce == Sum || reduce == Mean) && (sourceOverlap || indexOverlap);
    const bool lowPrecision = src.scalar_type() == at::kHalf || src.scalar_type() == at::kBFloat16;
    const bool orderedStrides = overlaps && !lowPrecision;
    if (overlaps && lowPrecision) {
        if (sourceOverlap)
            src = src.clone();
        if (indexOverlap)
            index = index.clone();
    }
    if (!orderedStrides) {
        src = src.contiguous();
        out = out.contiguous();
    }
    if (!ByteOverlap(index, out))
        index = index.contiguous();
    auto tiling = MakeTiling(src, index, out, dim, reduce, hasOut);
    PlanLaunch(tiling, src, index, out);
    const auto argShape = reduce == Min || reduce == Max ? out.sizes().vec() : std::vector<int64_t>{0};
    auto arg = torch::empty(argShape, out.options().dtype(at::kLong));
    if (out.numel())
        LaunchScatter(src, index, out, arg, tiling);
    if (!orderedStrides && !originalOut.is_contiguous())
        originalOut.copy_(out);
    return std::make_tuple(originalOut, arg);
}
std::tuple<int64_t, int64_t> scatter_index_stats(torch::Tensor index) {
    TORCH_CHECK(index.device().type() == c10::DeviceType::PrivateUse1 && index.scalar_type() == at::kLong,
                "scatter index statistics require an INT64 NPU tensor");
    if (!index.numel())
        return std::make_tuple(0, -1);
    const c10::DeviceGuard guard(index.device());
    SpanElements(index);
    index = index.contiguous();
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t available = std::max<uint32_t>(1, platform->GetCoreNumAiv());
    const uint32_t blocks = std::min<uint64_t>(
        available, (static_cast<uint64_t>(index.numel()) + kIndexStatsTileElements - 1) / kIndexStatsTileElements);
    auto stats = torch::empty({blocks, kIndexStatsElementsPerCore}, index.options());
    const auto current = c10_npu::getCurrentNPUStream(index.get_device());
    std::array<c10::Storage, 2> owners{index.storage(), stats.storage()};
    for (const auto &owner : owners)
        c10_npu::NPUCachingAllocator::recordStream(owner.data_ptr(), current);
    // Cache only a transfer with proven completion, never an index-value proof.
    static thread_local IndexStatsHostBuffer hostBuffer;
    // Flush pending framework commands before submitting directly on its stream.
    const auto readyStream = current.stream(true);
    const auto *values = hostBuffer.Run(index, stats, blocks, readyStream);
    int64_t minimum = INT64_MAX, maximum = INT64_MIN;
    for (uint32_t core = 0; core < blocks; ++core) {
        minimum = std::min(minimum, values[core * kIndexStatsElementsPerCore]);
        maximum = std::max(maximum, values[core * kIndexStatsElementsPerCore + 1]);
    }
    return std::make_tuple(minimum, maximum);
}
std::tuple<torch::Tensor, torch::Tensor> scatter_a2_forward(torch::Tensor src, torch::Tensor index, int64_t dim,
                                                            std::optional<torch::Tensor> out,
                                                            std::optional<int64_t> dimSize, int64_t reduce) {
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1 && index.device() == src.device(),
                "scatter src and index must share an NPU device");
    TORCH_CHECK(src.dim() >= 1 && src.dim() <= 8 && index.dim() <= src.dim(), "invalid scatter tensor rank");
    TORCH_CHECK(index.scalar_type() == at::kLong, "scatter index must have int64 dtype");
    ScatterA2DtypeCode(src.scalar_type(), "scatter A2 native path requires an L1 dtype");
    if (dim < 0)
        dim += src.dim();
    TORCH_CHECK_INDEX(dim >= 0 && dim < src.dim(), "scatter dimension is out of range");
    const c10::DeviceGuard guard(src.device());
    const ScatterGroupedRequest groupedRequest{src, index, out, dimSize, dim, reduce};
    if (CanUseGroupsBeforeBroadcast(groupedRequest))
        return scatter_grouped_forward(groupedRequest);
    auto expanded = BroadcastIndex(src, index, dim);
    if (CanUseGroups(groupedRequest))
        return scatter_grouped_forward(groupedRequest);
    const auto bounds = expanded.numel() ? scatter_index_stats(index) : std::make_tuple<int64_t, int64_t>(0, -1);
    const int64_t minimum = std::get<0>(bounds), maximum = std::get<1>(bounds);
    TORCH_CHECK_INDEX(minimum >= 0, "scatter index is outside the output dimension");
    torch::Tensor output;
    if (out)
        output = *out;
    else {
        TORCH_CHECK_VALUE(!dimSize || *dimSize >= 0, "scatter dim_size must be nonnegative");
        TORCH_CHECK(maximum < INT64_MAX, "scatter output dimension overflows int64");
        auto shape = src.sizes().vec();
        shape[dim] = dimSize ? *dimSize : maximum + 1;
        TORCH_CHECK_INDEX(maximum < shape[dim], "scatter index is outside the output dimension");
        output = torch::empty(shape, src.options());
    }
    TORCH_CHECK(output.dim() == src.dim(), "scatter out and src ranks must match");
    TORCH_CHECK_INDEX(maximum < output.size(dim), "scatter index is outside the output dimension");
    return scatter_forward(src, index.dim() == 1 ? index : expanded, dim, output, reduce, out.has_value(), -1);
}
} // namespace opsgnn

namespace opsgnn {
namespace {
using namespace scatter_a2;

uint64_t GroupProduct(const at::IntArrayRef &sizes, int64_t first, int64_t last) {
    uint64_t product = 1;
    for (int64_t dim = first; dim < last; ++dim) {
        const uint64_t size = sizes[dim];
        TORCH_CHECK(!size || product <= INT64_MAX / size, "grouped scatter shape overflows");
        product *= size;
    }
    return product;
}
uint64_t AlignDmaBytes(uint64_t bytes) { return (bytes + kDmaBytes - 1) / kDmaBytes * kDmaBytes; }
uint64_t GroupUbBytes(const GroupTiling &t, uint32_t itemBytes) {
    const uint64_t width = t.tileWidth, rows = t.tileRows, groups = t.outputGroup;
    const bool extreme = t.reduce == Min || t.reduce == Max;
    const bool scalar = t.dtype > Float32 || (extreme && t.length > kExactFloatIndexLimit);
    uint64_t bytes = AlignDmaBytes(rows * width * itemBytes) + AlignDmaBytes(groups * width * itemBytes);
    bytes += AlignDmaBytes(rows * sizeof(int64_t)) + AlignDmaBytes(width * sizeof(int32_t));
    bytes += AlignDmaBytes((groups + 1) * sizeof(int64_t));
    bytes += AlignDmaBytes(t.validationBlocks * kGroupValidationBytesPerCore);
    if (!scalar && t.dtype != Float32)
        bytes += AlignDmaBytes(rows * width * sizeof(float));
    if (extreme) {
        bytes += AlignDmaBytes(groups * width * sizeof(int64_t));
        if (!scalar)
            bytes += 2 * AlignDmaBytes(width * sizeof(float)) + AlignDmaBytes((width + 7) / 8);
    }
    if (!scalar && t.reduce == Mean)
        bytes += AlignDmaBytes(width * sizeof(float));
    return bytes + kUbReserveBytes;
}
struct GroupTileOptions {
    uint32_t outputs = kGroupDefaultOutputs;
    uint32_t rows = kGroupMaxSourceRows;
    uint64_t additionalUbBytes = 0;
};
void ChooseGroupTile(GroupTiling &t, uint32_t bytes, uint32_t cores, uint64_t ubBytes, GroupTileOptions options = {}) {
    const uint64_t rowBytes = t.after * bytes;
    const uint32_t alignmentGroups = kDmaBytes / std::gcd<uint64_t>(kDmaBytes, rowBytes);
    t.outputGroup = std::max<uint32_t>(alignmentGroups, options.outputs);
    const uint64_t groups = (t.totalTasks + t.outputGroup - 1) / t.outputGroup;
    const uint64_t wantedFeatures = std::max<uint64_t>(1, (cores + groups - 1) / groups);
    const uint64_t wantedWidth = (t.after + wantedFeatures - 1) / wantedFeatures;
    t.tileWidth = std::min<uint64_t>(kGroupMaxFeatureElements, (wantedWidth + kVectorMaskElements - 1) /
                                                                   kVectorMaskElements * kVectorMaskElements);
    t.tileRows = options.rows;
    while (GroupUbBytes(t, bytes) + options.additionalUbBytes > ubBytes) {
        if (t.tileRows > 1)
            t.tileRows /= 2;
        else {
            TORCH_CHECK(t.tileWidth > kVectorMaskElements, "grouped scatter exceeds device UB");
            t.tileWidth = (t.tileWidth / 2 + kVectorMaskElements - 1) / kVectorMaskElements * kVectorMaskElements;
            t.tileRows = options.rows;
        }
    }
    t.featureTiles = (t.after + t.tileWidth - 1) / t.tileWidth;
    // Nonaligned row tails share a store block with the next row. The same
    // group owner processes all feature tiles in that case.
    t.unitFeatureTiles = rowBytes % kDmaBytes == 0 ? t.featureTiles : 1;
    t.units = groups * t.unitFeatureTiles;
    t.blocks = std::min<uint64_t>(cores, t.units);
}
GroupTiling MakeGroupTiling(const ScatterGroupedRequest &request, const torch::Tensor &source, const torch::Tensor &out,
                            const ScatterGroupUse &use) {
    GroupTiling t{};
    t.length = source.size(request.dim);
    t.after = GroupProduct(source.sizes(), request.dim + 1, source.dim());
    t.buckets = use.plan->buckets;
    t.srcElements = source.numel();
    t.outElements = out.numel();
    t.indexElements = use.plan->indexElements;
    t.totalTasks = t.outElements / t.after;
    t.dtype = ScatterA2DtypeCode(source.scalar_type(), "grouped scatter requires an L1 dtype");
    t.reduce = request.reduce;
    t.hasOut = request.out.has_value();
    t.validationBlocks = use.validationBlocks;
    const auto hardware = GetScatterGroupHardware(source.get_device());
    ChooseGroupTile(t, source.element_size(), hardware.cores, hardware.ubBytes);
    if (reinterpret_cast<uintptr_t>(out.data_ptr()) % kDmaBytes)
        t.blocks = 1;
    return t;
}
void LaunchGroup(const torch::Tensor &source, const torch::Tensor &out, const torch::Tensor &arg,
                 const ScatterGroupUse &use, const GroupTiling &t) {
    const auto current = c10_npu::getCurrentNPUStream(source.get_device());
    std::array<c10::Storage, 6> owners{source.storage(),
                                       out.storage(),
                                       arg.defined() ? arg.storage() : c10::Storage(),
                                       use.plan->rowptr.storage(),
                                       use.plan->rows.storage(),
                                       use.validation.storage()};
    for (const auto &owner : owners)
        if (owner && owner.data_ptr().get())
            c10_npu::NPUCachingAllocator::recordStream(owner.data_ptr(), current);
    void *input = source.data_ptr(), *output = out.data_ptr();
    auto *pointers = use.plan->rowptr.data_ptr<int64_t>(), *rows = use.plan->rows.data_ptr<int64_t>();
    auto *arguments = arg.defined() ? arg.data_ptr<int64_t>() : nullptr;
    auto *flags = use.validationBlocks ? use.validation.data_ptr<int32_t>() : nullptr;
    // Flush queued input/output copies before this direct launch. Recorded
    // allocations remain stream-safe after local Tensor owners are released;
    // the validation ticket itself stays on the calling thread until Finish.
    const auto stream = current.stream(true);
    TORCH_CHECK(ScatterA2Grouped(input, pointers, rows, output, arguments, flags, t, stream) == ACL_SUCCESS,
                "grouped scatter launch failed");
    MarkScatterGroupsOwnerEnqueued(use, stream);
}
bool UsePrivateSum(const ScatterGroupUse &use, const GroupTiling &t, int device,
                   uint32_t maximumOutputs = kGroupDefaultOutputs) {
    if (!use.pendingIndex.defined() || t.dtype != Float32 || t.reduce != Sum || t.hasOut || !t.outElements ||
        !t.blocks || t.after != t.tileWidth || t.featureTiles != 1 || t.outputGroup > maximumOutputs)
        return false;
    constexpr uint64_t equalityChunkElements = 512;
    constexpr uint64_t equalityUbBytes = 8352;
    constexpr uint64_t workingSetUbMultiple = 8;
    const auto hardware = GetScatterGroupHardware(device);
    if (t.indexElements > t.blocks * equalityChunkElements || use.plan->maximumBucketRows > t.tileRows ||
        GroupUbBytes(t, sizeof(float)) + equalityUbBytes > hardware.ubBytes)
        return false;
    // Experimental scheduling bound: avoid waiting for long private producers.
    // The public domain is unchanged; all other geometry uses the guarded path.
    const uint64_t perCoreSource = (t.srcElements + t.blocks - 1) / t.blocks;
    const uint64_t perCoreOutput = (t.outElements + t.blocks - 1) / t.blocks;
    return perCoreSource + perCoreOutput <= hardware.ubBytes / sizeof(float) * workingSetUbMultiple;
}

bool PreparePrivateSum(const ScatterGroupUse &use, GroupTiling &tiling, int device) {
    // Preserve every v1 eligibility condition before trying a wider row group.
    if (!UsePrivateSum(use, tiling, device))
        return false;
    constexpr uint32_t privateOutputs = 8, privateRows = 64;
    constexpr uint64_t equalityUbBytes = 8352;
    const auto hardware = GetScatterGroupHardware(device);
    auto candidate = tiling;
    candidate.outputGroup = privateOutputs;
    candidate.tileWidth = kVectorMaskElements;
    candidate.tileRows = 1;
    if (GroupUbBytes(candidate, sizeof(float)) + equalityUbBytes > hardware.ubBytes)
        return true;
    ChooseGroupTile(candidate, sizeof(float), hardware.cores, hardware.ubBytes,
                    {privateOutputs, privateRows, equalityUbBytes});
    // Keep any original single-core restriction caused by output alignment.
    candidate.blocks = std::min(candidate.blocks, tiling.blocks);
    if (UsePrivateSum(use, candidate, device, privateOutputs))
        tiling = candidate;
    return true;
}

struct GroupOutput {
    torch::Tensor value, arg;
};
GroupOutput CreateGroupOutput(const ScatterGroupedRequest &request, int64_t buckets) {
    auto shape = request.src.sizes().vec();
    shape[request.dim] = buckets;
    auto output = request.out ? request.out->contiguous() : torch::empty(shape, request.src.options());
    const bool extreme = request.reduce == Min || request.reduce == Max;
    TORCH_CHECK(!extreme || static_cast<uint64_t>(output.numel()) <= INT64_MAX / sizeof(int64_t),
                "grouped scatter argument byte size overflows");
    return {output, extreme ? torch::empty(output.sizes(), output.options().dtype(at::kLong)) : torch::Tensor()};
}
} // namespace

std::tuple<torch::Tensor, torch::Tensor> scatter_grouped_forward(const ScatterGroupedRequest &request) {
    const c10::DeviceGuard guard(request.src.device());
    // No-out calls only mutate private allocations. The public binding keeps
    // the external CSR proof boundary; avoid observing each private allocation.
    std::optional<at::DisableRecordFunctionGuard> privateRecords;
    if (!request.out)
        privateRecords.emplace();
    auto source = request.src.contiguous();
    const std::optional<int64_t> buckets =
        request.out ? std::optional<int64_t>(request.out->size(request.dim)) : request.dimSize;
    // Guarded mismatches write nothing; fused mismatches remain private. Retry
    // from a fresh snapshot before any result is exposed to the caller.
    for (uint32_t attempt = 0; attempt < 2; ++attempt) {
        const bool considerPrivate = source.scalar_type() == at::kFloat && request.reduce == Sum && !request.out;
        auto use = BeginScatterGroups(request.index, source.size(request.dim), buckets, attempt != 0, considerPrivate);
        auto output = CreateGroupOutput(request, use.plan->buckets);
        auto tiling = MakeGroupTiling(request, source, output.value, use);
        if (PreparePrivateSum(use, tiling, source.get_device())) {
            LaunchScatterGroupsPrivateSum(use, source, output.value, tiling);
        } else {
            const bool deferred = use.pendingIndex.defined();
            ValidateDeferredScatterGroups(use);
            if (deferred)
                tiling = MakeGroupTiling(request, source, output.value, use);
            if (output.value.numel())
                LaunchGroup(source, output.value, output.arg, use, tiling);
        }
        if (!FinishScatterGroups(use))
            continue;
        if (request.out && !request.out->is_contiguous())
            request.out->copy_(output.value);
        return std::make_tuple(request.out ? *request.out : output.value, output.arg);
    }
    TORCH_CHECK(false, "scatter index metadata could not be rebuilt");
}
} // namespace opsgnn
#endif

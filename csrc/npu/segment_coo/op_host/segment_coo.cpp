/*
 * Copyright (c) 2026.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "segment_coo.h"
#ifdef OPSGNN_DAV_2201
#include "segment_coo/op_kernel/arch22/segment_coo.h"
#else
#include "segment_coo/op_kernel/arch35/segment_coo.h"
#endif
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include <algorithm>
#include <c10/core/DeviceGuard.h>
#include <ATen/MemoryOverlap.h>
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include <array>
#include <limits>
#include <cmath>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <string>

namespace {
#ifdef OPSGNN_DAV_2201
using namespace opsgnn;
#endif
constexpr uintptr_t kSegmentCooPackedAlignment = 32;

struct SegmentShape {
    int64_t dim;
    int64_t before;
    int64_t after;
    int64_t length;
    bool sharedIndex;
};

#ifdef OPSGNN_DAV_2201
struct IndexFacts {
    int64_t maximum = -1;
    int64_t uniformRows = 0;
};

struct IndexCacheEntry {
    c10::weak_intrusive_ptr<c10::TensorImpl> tensor;
    int64_t version, requested, offset;
    std::vector<int64_t> sizes, strides;
    IndexFacts facts;
};

// Only immutable host facts are cached. Weak identity prevents allocator reuse
// from resurrecting a proof; views share version counters with their aliases.
// Inference tensors have no version counter and are always checked again.
IndexFacts ValidateIndex(const torch::Tensor &index, int64_t requested) {
    if (!index.numel())
        return {};
    static std::mutex mutex;
    static std::map<const c10::TensorImpl *, IndexCacheEntry> cache;
    const auto *identity = index.unsafeGetTensorImpl();
    const int64_t version = index.is_inference() ? -1 : index._version();
    if (version >= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = cache.find(identity);
        if (found != cache.end()) {
            const auto &entry = found->second;
            if (entry.tensor.lock().get() == identity && entry.version == version && entry.requested == requested &&
                entry.offset == index.storage_offset() && index.sizes().equals(entry.sizes) &&
                index.strides().equals(entry.strides)) {
                return entry.facts;
            }
        }
    }
    std::vector<torch::Tensor> checks{index.min(), index.max(),
                                      index.slice(-1, 1).lt(index.slice(-1, 0, -1)).any().to(at::kLong)};
    int64_t uniformRows = 0;
    if (index.dim() == 1 && index.is_contiguous() && requested > 0 && index.numel() % requested == 0) {
        uniformRows = index.numel() / requested;
        auto labels = torch::arange(requested, index.options());
        // Sortedness AND the endpoints of every interval prove every label.
        auto first = index.slice(0, 0, index.numel(), uniformRows);
        auto last = index.slice(0, uniformRows - 1, index.numel(), uniformRows);
        checks.push_back(first.eq(labels).logical_and(last.eq(labels)).all().to(at::kLong));
    }
    auto host = torch::stack(checks).cpu();
    const auto *values = host.data_ptr<int64_t>();
    TORCH_CHECK(values[0] >= 0, "index values must be non-negative");
    TORCH_CHECK(values[2] == 0, "index must be sorted along its last dimension");
    TORCH_CHECK(values[1] < std::numeric_limits<int64_t>::max(), "index label too large");
    TORCH_CHECK(requested < 0 || values[1] < requested, "index values exceed output size");
    IndexFacts facts{values[1], checks.size() == 4 && values[3] ? uniformRows : 0};
    if (version >= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        if (cache.size() >= 256)
            cache.erase(cache.begin());
        cache.insert_or_assign(identity,
                               IndexCacheEntry{c10::weak_intrusive_ptr<c10::TensorImpl>(index.getIntrusivePtr()),
                                               version, requested, index.storage_offset(), index.sizes().vec(),
                                               index.strides().vec(), facts});
    }
    return facts;
}

// Each predicate owns its bounded cache and criterion. Eviction follows map
// key order (not LRU); weak identity and version/layout checks govern validity.
struct SourcePredicateEntry {
    c10::weak_intrusive_ptr<c10::TensorImpl> tensor;
    int64_t version, criterion, offset;
    c10::Device device;
    at::ScalarType dtype;
    std::vector<int64_t> sizes, strides;
    bool value;

    bool Matches(const torch::Tensor &src, int64_t currentVersion) const {
        return tensor.lock().get() == src.unsafeGetTensorImpl() && version == currentVersion &&
               offset == src.storage_offset() && device == src.device() && dtype == src.scalar_type() &&
               src.sizes().equals(sizes) && src.strides().equals(strides);
    }
};

bool CheckInt64Range(const torch::Tensor &src, int64_t bound) {
    auto input = src.contiguous();
    if (reinterpret_cast<uintptr_t>(input.data_ptr()) % 32)
        input = input.clone();
    const int blocks = std::max<uint32_t>(1, platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    // Every core owns one complete 32-byte flag block, including DMA padding.
    auto flags = torch::empty({blocks * 8}, src.options().dtype(at::kInt));
    const auto current = c10_npu::getCurrentNPUStream(src.device().index());
    c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(), current);
    c10_npu::NPUCachingAllocator::recordStream(flags.storage().data_ptr(), current);
    const auto stream = current.stream(false);
    at_npu::native::OpCommand::RunOpApi("segment_coo_validate_int64", [input, flags, bound, blocks, stream]() {
        SegmentCooValidateInt64(input.data_ptr(), input.numel(), bound, flags.data_ptr<int32_t>(), blocks, stream);
        return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
    });
    const auto host = flags.cpu();
    const auto *valid = host.data_ptr<int32_t>();
    for (int i = 0; i < blocks; ++i)
        if (valid[i * 8] != 1)
            return false;
    return true;
}

bool ProveSourceRange(const torch::Tensor &src, int64_t length, int64_t reduce) {
    if (!src.numel() || !length)
        return false;
    const bool integer = src.scalar_type() == at::kLong;
    if (integer && length > INT32_MAX)
        return false;
    const int64_t criterion = integer ? (reduce <= SEGMENT_COO_MEAN ? INT32_MAX / length : INT32_MAX) : length;
    static std::mutex mutex;
    static std::map<const c10::TensorImpl *, SourcePredicateEntry> cache;
    const auto *identity = src.unsafeGetTensorImpl();
    const int64_t version = src.is_inference() ? -1 : src._version();
    if (version >= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = cache.find(identity);
        if (found != cache.end()) {
            const auto &e = found->second;
            if (e.criterion == criterion && e.Matches(src, version))
                return e.value;
        }
    }
    // Validation statistics are computed on the current NPU stream. Only the
    // predicate is cached, never source data, an address, or a reduction result.
    // A fresh tensor/version pays this proof inside the public interface.
    at::NoGradGuard noGrad;
    bool bounded;
    if (integer) {
        bounded = CheckInt64Range(src, criterion);
    } else {
        auto checks = torch::stack({src.abs().max(), torch::isfinite(src).all().to(at::kFloat)}).cpu();
        const auto *values = checks.data_ptr<float>();
        bounded = values[1] != 0 && std::isfinite(values[0]) &&
                  static_cast<double>(values[0]) * static_cast<double>(length) <=
                      static_cast<double>(std::numeric_limits<float>::max()) / 4;
    }
    if (version >= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        if (cache.size() >= 128)
            cache.erase(cache.begin());
        cache.insert_or_assign(
            identity, SourcePredicateEntry{c10::weak_intrusive_ptr<c10::TensorImpl>(src.getIntrusivePtr()), version,
                                           criterion, src.storage_offset(), src.device(), src.scalar_type(),
                                           src.sizes().vec(), src.strides().vec(), bounded});
    }
    return bounded;
}

// An immutable numerical-risk decision, bound to the current source version.
// The statistics are NPU reductions inside this API, not cached output values.
struct FloatOrderEntry {
    SourcePredicateEntry source;
    int64_t dimension, uniformRows;
};
bool NeedsCpuOrder(const torch::Tensor &src, int64_t dimension, int64_t uniformRows) {
    const int64_t length = src.size(dimension);
    if (!src.numel() || length <= 1)
        return false;
    const int64_t terms = uniformRows ? uniformRows : length;
    static std::mutex mutex;
    static std::map<const c10::TensorImpl *, FloatOrderEntry> cache;
    const auto *identity = src.unsafeGetTensorImpl();
    const int64_t version = src.is_inference() ? -1 : src._version();
    if (version >= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = cache.find(identity);
        if (found != cache.end()) {
            const auto &entry = found->second;
            const auto &e = entry.source;
            if (entry.dimension == dimension && entry.uniformRows == uniformRows && e.Matches(src, version))
                return e.value;
        }
    }
    bool ordered = true;
    if (terms < (1LL << 20)) {
        at::NoGradGuard noGrad;
        auto magnitudes = src.to(at::kFloat).abs();
        auto sums = uniformRows ? magnitudes.reshape({length / uniformRows, uniformRows, -1}).sum(1)
                                : magnitudes.sum(dimension);
        const double observed = sums.max().item<float>();
        constexpr double unit = 0x1p-24;
        const double gamma = terms * unit / (1.0 - terms * unit);
        // Inflate the positive reduction by its own rounding bound. Include
        // sequential opmath error, compensated final rounding and low-part
        // accumulation. Nonfinite or very long sums use the ordered path.
        const double magnitude = observed / (1.0 - gamma);
        const double error = (gamma + 4 * unit + 8 * gamma * gamma) * magnitude;
        const double absolute =
            src.scalar_type() == at::kFloat ? 0x1p-13 : (src.scalar_type() == at::kHalf ? 0x1p-10 : 0x1p-7);
        ordered = !std::isfinite(error) || error > absolute;
    }
    if (version >= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        if (cache.size() >= 128)
            cache.erase(cache.begin());
        cache.insert_or_assign(
            identity,
            FloatOrderEntry{SourcePredicateEntry{c10::weak_intrusive_ptr<c10::TensorImpl>(src.getIntrusivePtr()),
                                                 version, terms, src.storage_offset(), src.device(), src.scalar_type(),
                                                 src.sizes().vec(), src.strides().vec(), ordered},
                            dimension, uniformRows});
    }
    return ordered;
}

// Extrema only need a sequential CPU argument shadow when no source can
// win (NaNs or a supplied initial output). Cache only the NaN predicate.
bool SourceHasNan(const torch::Tensor &src) {
    if (!src.numel())
        return false;
    static std::mutex mutex;
    static std::map<const c10::TensorImpl *, SourcePredicateEntry> cache;
    const auto *identity = src.unsafeGetTensorImpl();
    const int64_t version = src.is_inference() ? -1 : src._version();
    if (version >= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = cache.find(identity);
        if (found != cache.end()) {
            const auto &e = found->second;
            if (e.Matches(src, version))
                return e.value;
        }
    }
    at::NoGradGuard noGrad;
    const bool hasNan = src.isnan().any().item<bool>();
    if (version >= 0) {
        std::lock_guard<std::mutex> lock(mutex);
        if (cache.size() >= 128)
            cache.erase(cache.begin());
        cache.insert_or_assign(identity,
                               SourcePredicateEntry{c10::weak_intrusive_ptr<c10::TensorImpl>(src.getIntrusivePtr()),
                                                    version, 0, src.storage_offset(), src.device(), src.scalar_type(),
                                                    src.sizes().vec(), src.strides().vec(), hasNan});
    }
    return hasNan;
}

void ValidateOutputStorage(const torch::Tensor &out) {
    std::vector<std::pair<int64_t, int64_t>> layout;
    for (int64_t d = 0; d < out.dim(); ++d)
        layout.emplace_back(out.stride(d), out.size(d));
    std::sort(layout.begin(), layout.end());
    int64_t span = 1;
    for (const auto &axis : layout) {
        if (axis.second <= 1)
            continue;
        TORCH_CHECK(axis.first >= span, "out must have non-overlapping storage");
        span += (axis.second - 1) * axis.first;
    }
    TORCH_CHECK(!out.is_inference() || c10::InferenceMode::is_enabled(),
                "inference out may only be updated in inference_mode");
}
#endif

void ValidateInputs(const torch::Tensor &src, const torch::Tensor &index, int64_t reduce) {
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "src must be NPU");
    TORCH_CHECK(index.device() == src.device(), "index and src must be on the same NPU");
    TORCH_CHECK(index.scalar_type() == at::kLong, "index must be int64");
    TORCH_CHECK(src.dim() >= 1 && src.dim() <= 8 && index.dim() >= 1 && index.dim() <= src.dim(), "invalid rank");
    TORCH_CHECK(src.layout() == at::kStrided && index.layout() == at::kStrided,
                "src and index must be strided tensors");
    TORCH_CHECK(reduce >= SEGMENT_COO_SUM && reduce <= SEGMENT_COO_MAX, "invalid reduction");
}

SegmentCooDtype ResolveDtype(at::ScalarType type) {
    switch (type) {
        case at::kFloat:
            return SEGMENT_COO_FLOAT32;
        case at::kHalf:
            return SEGMENT_COO_FLOAT16;
        case at::kBFloat16:
            return SEGMENT_COO_BFLOAT16;
        case at::kChar:
            return SEGMENT_COO_INT8;
        case at::kByte:
            return SEGMENT_COO_UINT8;
        case at::kInt:
            return SEGMENT_COO_INT32;
        case at::kLong:
            return SEGMENT_COO_INT64;
        default:
            TORCH_CHECK(false, "unsupported src dtype");
    }
}

SegmentShape PrepareIndex(const torch::Tensor &src, torch::Tensor &index) {
    SegmentShape shape{index.dim() - 1, 1, 1, src.size(index.dim() - 1), true};
    std::vector<int64_t> indexShape;
    for (int64_t d = 0; d < shape.dim; ++d) {
        TORCH_CHECK(index.size(d) == 1 || index.size(d) == src.size(d), "index broadcast mismatch");
        shape.before *= src.size(d);
        indexShape.push_back(src.size(d));
        shape.sharedIndex = shape.sharedIndex && index.size(d) == 1;
    }
#ifdef OPSGNN_DAV_2201
    const bool broadcastReduction = index.size(shape.dim) == 1 && shape.length != 1;
    TORCH_CHECK(index.size(shape.dim) == shape.length || broadcastReduction,
                "index length must be one or match reduction dimension");
    if (shape.sharedIndex)
        std::fill(indexShape.begin(), indexShape.end(), 1);
#else
    TORCH_CHECK(index.size(shape.dim) == shape.length, "index length must match reduction dimension");
#endif
    indexShape.push_back(shape.length);
    for (int64_t d = shape.dim + 1; d < src.dim(); ++d) {
        shape.after *= src.size(d);
    }
    if (!shape.sharedIndex
#ifdef OPSGNN_DAV_2201
        || broadcastReduction
#endif
    ) {
        index = index.expand(indexShape);
    }
    index = index.contiguous();
    return shape;
}

torch::Tensor AlignedContiguous(torch::Tensor value) {
    value = value.contiguous();
    // The widest packed access is 32 bytes; contiguous offset views may be
    // unaligned.
    if (reinterpret_cast<uintptr_t>(value.data_ptr()) % kSegmentCooPackedAlignment != 0) {
        value = value.clone();
    }
    return value;
}

int64_t ResolveSegments(const torch::Tensor &src, const torch::Tensor &index,
                        const std::optional<torch::Tensor> &optionalOut, std::optional<int64_t> dimSize,
                        const SegmentShape &shape, int64_t inferredMaximum = -1) {
    if (optionalOut.has_value()) {
        const auto &out = *optionalOut;
        TORCH_CHECK(out.device() == src.device() && out.scalar_type() == src.scalar_type(),
                    "out device and dtype must match src");
        TORCH_CHECK(out.dim() == src.dim(), "out rank mismatch");
        for (int64_t d = 0; d < src.dim(); ++d) {
            if (d != shape.dim) {
                TORCH_CHECK(out.size(d) == src.size(d), "out shape mismatch");
            }
        }
#ifdef OPSGNN_DAV_2201
        TORCH_CHECK(out.layout() == at::kStrided, "out must be strided");
        ValidateOutputStorage(out);
#endif
        return out.size(shape.dim);
    }
    if (dimSize.has_value()) {
        return *dimSize;
    }
    if (index.numel() == 0) {
        return 0;
    }
#ifdef OPSGNN_DAV_2201
    return inferredMaximum + 1;
#else
    return index.select(shape.dim, shape.length - 1).max().item<int64_t>() + 1;
#endif
}

#ifdef OPSGNN_DAV_2201
// The capacity planner derives from m0_52603995/torch_scatter_npu,
// commit b4454bcbd952386c25056b374c1e0c1537035886.
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// 由人工智能与数字经济广东省实验室（深圳）-大湾区昇腾算力应用创新研究院
// 资助（编号：GML-ST-2026-20）。
namespace arch22 {
constexpr uint64_t kUbReserveBytes = 8 * 1024;

// Input facts and intermediate tile choices are private to Host planning.
// Only BuildLaunchPlan's final immutable result crosses the launch boundary.
struct PlanInputs {
    int64_t batches, length, channels, segments, indexBatches;
    SegmentCooReduce reduce;
    SegmentCooDtype dtype;
    int32_t hasOut, blocks;
    int32_t hasArg = 0, sourceFlags = 0;
    uint64_t uniformRows = 0, ubBytes = 0;
    uint32_t tileK = 64, tileRows = 64, outputGroup = 1, batchReduce = 0;
};

inline uint64_t Align32(uint64_t bytes) {
    return (bytes + 31) / 32 * 32;
}
inline uint32_t ItemBytes(uint32_t dtype) {
    return dtype == SegmentFloat32 || dtype == SegmentInt32
               ? 4
               : (dtype == SegmentInt64 ? 8 : (dtype == SegmentInt8 || dtype == SegmentUInt8 ? 1 : 2));
}
inline bool FloatInput(uint32_t dtype) {
    return dtype == SegmentFloat32 || dtype == SegmentFloat16 || dtype == SegmentBFloat16;
}
// Exactly the buffers allocated by SegmentReduceDevice::Init, plus explicit reserve.
inline uint64_t TileBytes(uint32_t dtype, bool hasArg, uint32_t cols, uint32_t rows, bool compensated = false,
                          bool batchReduce = false) {
    const uint32_t item = ItemBytes(dtype);
    uint64_t bytes = Align32(uint64_t(rows) * cols * item) + Align32(cols * 4ULL) + Align32(uint64_t(cols) * item);
    if (FloatInput(dtype) && dtype != SegmentFloat32)
        bytes += Align32(uint64_t(rows) * cols * 4);
    if (!FloatInput(dtype) && dtype != SegmentInt32)
        bytes += Align32(cols * 4ULL);
    if (hasArg)
        bytes += Align32(cols * 4ULL) + Align32(cols * 8ULL) + 2 * Align32((cols + 7ULL) / 8);
    else if (!compensated)
        bytes += Align32((uint64_t(rows) * cols + 7) / 8); // Full source-tile NaN mask.
    if (hasArg && FloatInput(dtype))
        bytes += Align32(cols * 4ULL); // NaN-preserving extrema merge scratch.
    if (compensated) {
        const uint64_t scratch = uint64_t(std::max(1U, rows / 2)) * cols;
        // First-level source lows are implicit zero; all three tree buffers
        // (correction, sum and z) need at most the surviving half of the tile.
        // Batch FP16 reuses the consumed half source tile for float lows.
        const uint64_t buffers = batchReduce && dtype == SegmentFloat16 && rows >= 2 ? 2 : 3;
        bytes += buffers * Align32(scratch * 4) + Align32(cols * 4ULL) + Align32((scratch + 7) / 8);
    }
    // Value-only floating extrema use disjoint ping-pong partials. Half input
    // storage can be reused after Cast; float input needs a half-tile buffer.
    if (batchReduce && !compensated && dtype == SegmentFloat32)
        bytes += Align32(uint64_t(rows / 2) * cols * 4);
    return bytes;
}
struct TilePlan {
    uint32_t cols, rows, outputGroup;
    uint64_t bytes;
};
inline TilePlan PlanTile(uint64_t k, uint32_t dtype, bool hasArg, uint64_t ubBytes, bool compensated = false,
                         bool groupAligned = false, bool batchReduce = false) {
    constexpr uint64_t reserve = kUbReserveBytes;
    if (!k || ubBytes <= reserve)
        throw std::invalid_argument("invalid K or UB capacity");
    // Compare on arch22 ignores a partial repeat. Reserve full 64-lane Work/arg
    // regions even when DMA and arithmetic process only the logical padded row.
    const uint32_t align = 64;
    // Bound DMA/vector instruction counts independently of the full feature width.
    uint32_t cols = static_cast<uint32_t>(std::min<uint64_t>((k + align - 1) / align * align, 4096));
    const uint64_t budget = ubBytes - reserve;
    // Prefer enough rows for amortized DMA. Neither limit is an input-shape whitelist.
    while (cols > align && TileBytes(dtype, hasArg, cols, 32, compensated, batchReduce) > budget)
        cols = std::max(align, (cols / 2 / align) * align);
    if (TileBytes(dtype, hasArg, cols, 1, compensated, batchReduce) > budget)
        throw std::invalid_argument("UB is too small for a minimum segment tile");
    uint32_t rows = 256;
    while (TileBytes(dtype, hasArg, cols, rows, compensated, batchReduce) > budget)
        --rows;
    const uint64_t valueGroup = 32 / std::gcd(uint64_t(32), (k % 32) * ItemBytes(dtype));
    const uint64_t argGroup = hasArg ? 32 / std::gcd(uint64_t(32), (k % 32) * 8) : 1;
    // Adjacent segments can share one strided source transfer when a complete
    // feature row is 32-byte aligned.  Keep the old ownership/alignment group
    // for tails, but permit a bounded group of eight for aligned K as well;
    // the device verifies contiguous bounds before using the grouped path and
    // falls back to one task at a time for arbitrary COO labels.  This is a
    // layout-derived choice, not a K-value whitelist.
    uint64_t outputGroup = std::lcm(valueGroup, argGroup);
    if (groupAligned && k * uint64_t(ItemBytes(dtype)) % 32 == 0 && outputGroup < 8)
        outputGroup = 8;
    return {cols, rows, static_cast<uint32_t>(outputGroup),
            TileBytes(dtype, hasArg, cols, rows, compensated, batchReduce) + reserve};
}
// Transfer-aware instruction-count model for proven uniform, aligned 2D
// sum/mean. Enumerate vector-aligned column tiles together with their exact UB
// row capacity and segment group; account for every extra tile's DMA/store,
// vector submissions and dependent-stage barriers. This is a scheduling
// heuristic, not a cycle prediction; measured transfers remain authoritative.
inline TilePlan PlanCompensatedColumns(uint64_t k, uint32_t dtype, uint64_t ubBytes, uint32_t segmentRows,
                                       uint64_t tasks, uint32_t cores) {
    auto best = PlanTile(k, dtype, false, ubBytes, true, false, true);
    best.outputGroup = 0;
    uint64_t bestCost = UINT64_MAX;
    uint32_t levels = 0;
    for (uint32_t n = segmentRows; n > 1; n /= 2)
        ++levels;
    const uint32_t stages = segmentRows == 2 ? 1 : 6 + 8 * (levels - 1);
    for (uint32_t cols = best.cols;; cols = std::max(64U, (cols / 2 / 64) * 64)) {
        auto plan = PlanTile(cols, dtype, false, ubBytes, true, false, true);
        const uint32_t width = static_cast<uint32_t>(std::min<uint64_t>(k, plan.cols));
        const uint64_t groups = std::min<uint64_t>({32, plan.rows / segmentRows, tasks / (uint64_t(cores) * 2)});
        if (groups >= 1 && 2 * width / 8 <= 255) {
            const uint64_t tiles = (k + plan.cols - 1) / plan.cols;
            const uint64_t batches = (tasks + groups - 1) / groups;
            // Contiguous full rows use one dense transfer. Column tiles
            // submit strided row blocks on both load and store. Charge their
            // fragmentation by actual row bytes and active group rows, not
            // by dtype/K names. Eight bytes per row is a calibrated relative
            // issue-cost weight (not a hardware transaction size or cycles).
            const auto tileCost = [&](uint64_t logicalWidth) {
                const uint64_t rowBytes = logicalWidth * ItemBytes(dtype);
                const uint64_t blocks = groups * (uint64_t(segmentRows) + 1);
                const uint64_t strided = tiles > 1 ? (blocks * 8 + rowBytes - 1) / rowBytes : 0;
                return 2 + stages * ((logicalWidth + 63) / 64 + 1) + strided;
            };
            const uint64_t lastWidth = k - (tiles - 1) * plan.cols;
            const uint64_t cost = batches * ((tiles - 1) * tileCost(width) + tileCost(lastWidth));
            // Retain the existing multi-segment choice whenever it fits.
            // A one-segment tree is a capacity fallback, still consuming the
            // validator's full uniform proof and a complete power-of-two span.
            const bool multiple = groups >= 2;
            const bool bestMultiple = best.outputGroup >= 2;
            if ((multiple && !bestMultiple) || (multiple == bestMultiple && cost < bestCost)) {
                bestCost = cost;
                plan.outputGroup = static_cast<uint32_t>(groups);
                best = plan;
            }
        }
        if (cols == 64)
            break;
    }
    return best;
}

// Every input to MakeSegmentPlan participates in the key. Index/out dtypes and
// address guards are normalized by the common Host; no address is cached here.
struct SegmentPlanKey {
    uint64_t batches, rows, features, segments, uniformRows, ubBytes;
    uint32_t cores, dtype;
    SegmentReduce reduce;
    bool hasArg, hasOut, direct2D;
    bool operator==(const SegmentPlanKey &other) const {
        return std::tie(batches, rows, features, segments, uniformRows, ubBytes, cores, dtype, reduce, hasArg, hasOut,
                        direct2D) == std::tie(other.batches, other.rows, other.features, other.segments,
                                              other.uniformRows, other.ubBytes, other.cores, other.dtype, other.reduce,
                                              other.hasArg, other.hasOut, other.direct2D);
    }
};
struct SegmentPlan {
    TilePlan tile;
    uint32_t batchReduce;
};

inline void SelectBatchedValues(const SegmentPlanKey &key, uint64_t averageRows, bool compensated, SegmentPlan &state) {
    auto &plan = state.tile;
    auto &batchReduce = state.batchReduce;
    // Paired vector repeats consume adjacent rows without packing. Both row
    // strides must fit the hardware repeat field; multiple complete segments
    // must fit UB, with enough groups left to occupy every vector core.
    const bool batchValueExtreme = !key.hasArg && (key.reduce == SegmentMin || key.reduce == SegmentMax);
    if (key.direct2D && (compensated || batchValueExtreme) && !key.hasOut &&
        (key.dtype == SegmentFloat32 || key.dtype == SegmentFloat16) && averageRows >= 2 &&
        (averageRows & (averageRows - 1)) == 0) {
        auto candidate = PlanTile(key.features, key.dtype, false, key.ubBytes, compensated, false, true);
        const uint64_t groups =
            std::min<uint64_t>({32, candidate.rows / averageRows,
                                key.batches * static_cast<uint64_t>(key.segments) / (uint64_t(key.cores) * 2)});
        if (key.features <= candidate.cols && key.features % 8 == 0 && 2 * key.features / 8 <= 255 && groups >= 2) {
            candidate.outputGroup = static_cast<uint32_t>(groups);
            plan = candidate;
            batchReduce = kValueBatch;
        }
    }
}

inline void SelectCompensatedColumns(const SegmentPlanKey &key, uint64_t averageRows, bool compensated,
                                     SegmentPlan &state) {
    auto &plan = state.tile;
    auto &batchReduce = state.batchReduce;
    // Column splitting is limited to the validator's complete uniform proof.
    // Every output row and feature tile owns whole 32-byte blocks. Existing
    // irregular, initial-out and extrema planners remain unchanged.
    if (key.direct2D && compensated && key.uniformRows &&
        (key.dtype == SegmentFloat32 || key.dtype == SegmentFloat16) && averageRows >= 2 && averageRows <= 256 &&
        (averageRows & (averageRows - 1)) == 0 && key.features * ItemBytes(key.dtype) % 32 == 0) {
        const auto candidate =
            PlanCompensatedColumns(key.features, key.dtype, key.ubBytes, static_cast<uint32_t>(averageRows),
                                   key.batches * static_cast<uint64_t>(key.segments), key.cores);
        if (candidate.outputGroup >= 1) {
            plan = candidate;
            batchReduce = kValueBatch;
        }
    }
}

inline void SelectNarrowUniformRows(const SegmentPlanKey &key, bool compensated, SegmentPlan &state) {
    auto &plan = state.tile;
    auto &batchReduce = state.batchReduce;
    // In the proven uniform batch path the logical row is also the physical
    // UB row. Narrow aligned features can therefore use twice as many rows
    // without changing any per-segment TwoSum operation or rounding order.
    // Keep the generic >=64-lane layout whenever a scalar/out fallback exists.
    if (key.direct2D && compensated && !key.hasOut && !key.hasArg && key.uniformRows >= 2 &&
        (key.uniformRows & (key.uniformRows - 1)) == 0 && key.features < 64 &&
        key.features * ItemBytes(key.dtype) % 32 == 0 && (key.dtype == SegmentFloat32 || key.dtype == SegmentFloat16)) {
        const uint32_t cols = static_cast<uint32_t>(key.features);
        uint32_t rows = 512;
        while (rows > 1 && TileBytes(key.dtype, false, cols, rows, true, true) + kUbReserveBytes > key.ubBytes)
            rows /= 2;
        const uint64_t groups =
            std::min<uint64_t>({32, rows / key.uniformRows, key.batches * key.segments / (uint64_t(key.cores) * 2)});
        if (groups >= 2 && groups > plan.outputGroup) {
            plan = {cols, rows, static_cast<uint32_t>(groups),
                    TileBytes(key.dtype, false, cols, rows, true, true) + kUbReserveBytes};
            batchReduce = kValueBatch;
        }
    }
}

inline void SelectLongUniformColumns(const SegmentPlanKey &key, bool compensated, SegmentPlan &state) {
    auto &plan = state.tile;
    auto &batchReduce = state.batchReduce;
    // A long proven segment can fit one compensated tree by splitting its
    // aligned feature columns. This avoids repeated tree finalization/merge
    // when the generic wide row cannot hold the complete segment in UB.
    if (key.direct2D && compensated && !key.hasOut && !key.hasArg && key.uniformRows > plan.rows &&
        key.uniformRows <= 512 && (key.uniformRows & (key.uniformRows - 1)) == 0 && key.features > 0 &&
        key.features * ItemBytes(key.dtype) % 32 == 0 && (key.dtype == SegmentFloat32 || key.dtype == SegmentFloat16)) {
        const uint32_t cols = static_cast<uint32_t>(std::min<uint64_t>(32, key.features));
        const uint32_t rows = 512;
        const uint64_t bytes = TileBytes(key.dtype, false, cols, rows, true, true) + kUbReserveBytes;
        if (bytes <= key.ubBytes) {
            plan = {cols, rows, 1, bytes};
            batchReduce = kValueBatch;
        }
    }
}

inline SegmentPlan MakeSegmentPlan(const SegmentPlanKey &key) {
    // Preserve this backend's compensated floating sum/mean workspace.
    const bool compensated = FloatInput(key.dtype) && (key.reduce == SegmentSum || key.reduce == SegmentMean);
    // Grouping amortizes DMA, provided there are enough segments to give every
    // AIV at least one complete group.
    // Group ownership only amortizes its scalar bound checks when a typical
    // segment occupies a material fraction of the planned row tile.  Derive
    // that cutoff from the actual UB plan instead of an input-K whitelist;
    // short segments keep the ordinary per-task ownership and its parallelism.
    const uint64_t averageRows = key.segments ? key.rows / static_cast<uint64_t>(key.segments) : 0;
    const auto basePlan = PlanTile(key.features, key.dtype, key.hasArg, key.ubBytes, compensated, false);
    const bool groupLengthOk = averageRows * 4 >= basePlan.rows;
    const bool groupAligned =
        !key.hasOut && groupLengthOk && key.batches * static_cast<uint64_t>(key.segments) >= uint64_t(key.cores) * 8;
    auto plan = groupAligned ? PlanTile(key.features, key.dtype, key.hasArg, key.ubBytes, compensated, true) : basePlan;
    SegmentPlan result{plan, kTaskReduction};
    SelectBatchedValues(key, averageRows, compensated, result);
    SelectCompensatedColumns(key, averageRows, compensated, result);
    SelectNarrowUniformRows(key, compensated, result);
    SelectLongUniformColumns(key, compensated, result);
    return result;
}

inline SegmentPlan CachedSegmentPlan(const SegmentPlanKey &key) {
    // A small per-thread ring avoids locking or heap allocation on the hot path.
    // Repeated shapes hit mostRecent directly; misses retain ordinary planning.
    struct Entry {
        SegmentPlanKey key{};
        SegmentPlan plan{};
    };
    constexpr size_t capacity = 16;
    struct Cache {
        std::array<Entry, capacity> entries{};
        size_t size = 0, next = 0, mostRecent = 0;
    };
    static thread_local Cache cache;
    if (cache.size && cache.entries[cache.mostRecent].key == key)
        return cache.entries[cache.mostRecent].plan;
    for (size_t i = 0; i < cache.size; ++i) {
        if (cache.entries[i].key == key) {
            cache.mostRecent = i;
            return cache.entries[i].plan;
        }
    }
    const auto plan = MakeSegmentPlan(key);
    cache.entries[cache.next] = {key, plan};
    cache.mostRecent = cache.next;
    cache.next = (cache.next + 1) % capacity;
    cache.size = std::min(cache.size + 1, capacity);
    return plan;
}
inline void ConfigureIntegerGroups(PlanInputs &p) {
    const uint64_t item = p.dtype == SEGMENT_COO_INT64 ? 8 : 1;
    p.outputGroup = 32 / std::gcd(uint64_t(32), uint64_t(p.channels) * item);
    if (p.uniformRows && p.uniformRows <= p.tileRows && (p.uniformRows & (p.uniformRows - 1)) == 0 &&
        p.channels % 8 == 0 && !p.hasOut) {
        const uint64_t groups = std::min<uint64_t>(
            {32, p.tileRows / p.uniformRows, uint64_t(p.batches * p.segments) / (std::max(1, p.blocks) * 2ULL)});
        p.outputGroup = std::max<uint64_t>(p.outputGroup, std::max<uint64_t>(1, groups));
    }
}

inline void ConfigureIntegerTiles(PlanInputs &p) {
    p.tileK = p.dtype == SEGMENT_COO_INT64 ? std::min<uint64_t>((p.channels + 31) / 32 * 32, 128)
                                           : std::min<uint64_t>((p.channels + 63) / 64 * 64, 128);
    p.tileRows = 128;
    if (p.dtype == SEGMENT_COO_INT64 && p.reduce < SEGMENT_COO_MIN && !p.hasArg) {
        // Exact int64 digit decomposition budgets every live UB buffer.
        for (;;) {
            const uint64_t groups =
                p.uniformRows && p.uniformRows <= p.tileRows ? std::min<uint64_t>(32, p.tileRows / p.uniformRows) : 4;
            const uint64_t bytes =
                uint64_t(p.tileK) * p.tileRows * 41 + uint64_t(p.tileK) * groups * 40 + p.tileK * 8 + kUbReserveBytes;
            if (bytes <= p.ubBytes)
                break;
            p.tileRows /= 2;
        }
    }
    if (p.dtype == SEGMENT_COO_INT64 && p.reduce >= SEGMENT_COO_MIN && !p.hasArg) {
        while (uint64_t(p.tileK) * p.tileRows * 16 + p.tileK * 16 + 16384 > p.ubBytes)
            p.tileRows /= 2;
    }
    ConfigureIntegerGroups(p);
}

inline void ConfigureOrderedGroups(PlanInputs &p, uint64_t perElement, uint64_t item) {
    // Independent segment accumulators can issue together before each
    // row barrier, while every individual accumulator keeps CPU order.
    if (p.uniformRows && !p.hasOut && (p.channels * item) % 32 == 0) {
        for (uint64_t group = 2; group <= 32; group *= 2) {
            const uint64_t columns = (group * p.channels + 63) / 64 * 64;
            const uint64_t rowCapacity = (group * p.uniformRows * p.channels + columns - 1) / columns;
            if (columns > 4096 || rowCapacity > 1024 || p.batches * p.segments < std::max(1, p.blocks) * 2 * group ||
                rowCapacity * columns * perElement + columns * 32 + kUbReserveBytes > p.ubBytes)
                break;
            p.tileK = static_cast<uint32_t>(columns);
            p.outputGroup = static_cast<uint32_t>(group);
        }
    }
}

inline void ConfigureOrderedTiles(PlanInputs &p) {
    // Ordered rows parallelize across features. Preserve full feature width
    // when UB permits; tree-oriented narrow columns multiply row issue cost.
    p.tileK = static_cast<uint32_t>(std::min<uint64_t>((p.channels + 63) / 64 * 64, 4096));
    const uint64_t perElement = p.dtype == SEGMENT_COO_FLOAT32 ? 4 : 6;
    while (p.tileK > 64 && uint64_t(p.tileK) * (perElement + 32) + kUbReserveBytes > p.ubBytes)
        p.tileK /= 2;
    const uint64_t item = p.dtype == SEGMENT_COO_FLOAT32 ? 4 : 2;
    p.outputGroup = 32 / std::gcd(uint64_t(32), uint64_t(p.channels) * item);
    ConfigureOrderedGroups(p, perElement, item);
    const uint64_t reserve = uint64_t(p.tileK) * 32 + kUbReserveBytes;
    if (p.ubBytes <= reserve)
        throw std::invalid_argument("UB is too small for ordered reduction");
    p.tileRows = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(1024, (p.ubBytes - reserve) / (p.tileK * perElement))));
    p.batchReduce = kTaskReduction;
}

void ConfigureTiles(PlanInputs &p) {
    if (p.dtype == SEGMENT_COO_INT64 || p.dtype == SEGMENT_COO_INT8 || p.dtype == SEGMENT_COO_UINT8) {
        ConfigureIntegerTiles(p);
        return;
    }
    if ((p.sourceFlags & kCpuOrderedFloat) && p.dtype <= SEGMENT_COO_BFLOAT16 && p.reduce <= SEGMENT_COO_MEAN) {
        ConfigureOrderedTiles(p);
        return;
    }
    const auto selected =
        CachedSegmentPlan({static_cast<uint64_t>(p.batches), static_cast<uint64_t>(p.length),
                           static_cast<uint64_t>(p.channels), static_cast<uint64_t>(p.segments), p.uniformRows,
                           p.ubBytes, static_cast<uint32_t>(std::max(1, p.blocks)), static_cast<uint32_t>(p.dtype),
                           static_cast<SegmentReduce>(p.reduce), p.hasArg != 0, p.hasOut != 0, p.batches == 1});
    p.tileK = selected.tile.cols;
    p.tileRows = selected.tile.rows;
    p.outputGroup = selected.tile.outputGroup;
    p.batchReduce = selected.batchReduce;
    if (p.hasArg && p.reduce >= SEGMENT_COO_MIN) {
        const auto tile = PlanTile(p.channels, p.dtype, true, p.ubBytes, false, false, false);
        p.tileK = tile.cols;
        p.tileRows = tile.rows;
        p.outputGroup = tile.outputGroup;
        p.batchReduce = kTaskReduction;
    }
}

bool ConfigureInt32Batch(SegmentTilingData &t, uint64_t ubBytes) {
    const uint64_t rows = t.nSegments ? t.M / t.nSegments : 0;
    // Input plus ping-pong scratch use <=144 KiB; mean temporaries add at
    // most 32 KiB, retaining >=16 KiB of the supported 192 KiB UB. There is
    // no data-dependent speculation: indexMode=2 is the Host's full proof.
    const uint32_t cols = static_cast<uint32_t>(std::min<uint64_t>(128, t.K));
    // Six bytes per lane cover the input and half-tile ping-pong scratch.
    // Keep the tuned 144 KiB ceiling; use a smaller budget on smaller UBs.
    constexpr uint64_t kScratchCeiling = 144 * 1024;
    constexpr uint64_t kMeanAndReserve = 48 * 1024;
    if (ubBytes <= kMeanAndReserve)
        return false;
    const uint64_t scratchBytes = std::min(kScratchCeiling, ubBytes - kMeanAndReserve);
    const uint32_t capacityRows = cols ? scratchBytes / (6 * cols) : 0;
    const uint32_t maxGroups = cols <= 64 ? 64 : 32;
    if (t.indexMode == kUniformIndex && !t.hasInitial && !t.hasArg && t.reduceCode <= SegmentMean && t.K % 8 == 0 &&
        rows && (rows & (rows - 1)) == 0 && rows <= capacityRows) {
        t.tileK = cols;
        t.tileRows = capacityRows;
        t.outputGroup = static_cast<uint32_t>(std::min<uint64_t>(maxGroups, capacityRows / rows));
        t.featureTiles = (t.K + cols - 1) / cols;
        t.totalUnits = (t.totalTasks + t.outputGroup - 1) / t.outputGroup * t.featureTiles;
        t.usedBlockNum = std::min<uint64_t>(t.usedBlockNum, t.totalUnits);
        return true;
    }
    return false;
}

bool ConfigureInt64Narrow(SegmentTilingData &t, uint64_t ubBytes) {
    const uint64_t rows = t.nSegments ? t.M / t.nSegments : 0;
    if (!(t.sourceFlags & kSourceRangeProven) || t.hasInitial || t.hasArg || t.indexMode != kUniformIndex ||
        t.E_1 != 1 || rows < 2 || rows > INT32_MAX || (rows & (rows - 1)) || t.K % 8 || t.K > UINT32_MAX / 8) {
        return false;
    }
    // Keep complete feature rows whenever possible to use dense DMA. Budget
    // input + low words, reused input tree scratch, output words and offsets.
    t.tileK = std::min<uint64_t>((t.K + 31) / 32 * 32, 256);
    const uint64_t tasksPerCore = std::max<uint64_t>(1, t.totalTasks / (t.usedBlockNum * 2ULL));
    uint32_t selectedRows = 1, selectedGroups = 1;
    for (uint32_t candidate = 512; candidate >= 1; --candidate) {
        const uint32_t groups = std::min<uint64_t>({32, std::max<uint64_t>(1, candidate / rows), tasksPerCore});
        const uint64_t output = uint64_t(groups) * t.tileK;
        const uint64_t bytes = uint64_t(candidate) * t.tileK * 12 + output * 24 + t.tileK * 4 +
                               ((output * 2 + 7) / 8 + 31) / 32 * 32 + kUbReserveBytes;
        if (bytes <= ubBytes) {
            selectedRows = candidate;
            selectedGroups = groups;
            break;
        }
    }
    // Batches consist of whole power-of-two segments. Single long segments
    // use the exact odd-row carry in the integer tree.
    t.tileRows = selectedRows;
    t.outputGroup = selectedGroups;
    t.featureTiles = (t.K + t.tileK - 1) / t.tileK;
    t.totalUnits =
        t.outputGroup == 1 ? t.totalTasks * t.featureTiles : (t.totalTasks + t.outputGroup - 1) / t.outputGroup;
    t.usedBlockNum = std::min<uint64_t>(t.usedBlockNum, t.totalUnits);
    return true;
}

// This is the only producer of device tiling. Launchers consume this plan
// without changing tile sizes, work ownership or the chosen algorithm.
SegmentCooPlan BuildLaunchPlan(PlanInputs p) {
    ConfigureTiles(p);
    SegmentTilingData t{};
    t.E_1 = p.batches;
    t.M = p.length;
    t.K = p.channels;
    t.nSegments = p.segments;
    t.srcNumel = p.batches * p.length * p.channels;
    t.outNumel = p.batches * p.segments * p.channels;
    t.indexNumel = p.indexBatches * p.length;
    t.indexBatchStride = p.indexBatches == 1 ? 0 : p.length;
    t.indexRowStride = 1;
    t.indexMode = p.uniformRows ? kUniformIndex : kSortedIndex;
    t.reduceCode = p.reduce;
    t.dtypeCode = p.dtype;
    // All reductions include the supplied seed; empty segments preserve it.
    t.hasInitial = p.hasOut;
    t.hasArg = p.hasArg;
    t.tileK = p.tileK;
    t.tileRows = p.tileRows;
    t.outputGroup = p.outputGroup;
    t.batchReduce = p.batchReduce;
    t.sourceFlags = p.sourceFlags;
    t.totalTasks = p.batches * p.segments;
    t.featureTiles = (p.channels + p.tileK - 1) / p.tileK;
    t.totalUnits =
        p.outputGroup == 1 ? t.totalTasks * t.featureTiles : (t.totalTasks + p.outputGroup - 1) / p.outputGroup;
    t.usedBlockNum = std::min<uint64_t>(std::max(1, p.blocks), t.totalUnits);
    SegmentCooPlan plan{t, SegmentKernelPath::General, static_cast<uint32_t>(std::max(1, p.blocks)), p.hasOut};
    if (!t.usedBlockNum)
        return plan;
    if (p.dtype == SEGMENT_COO_INT32 && ConfigureInt32Batch(plan.tiling, p.ubBytes)) {
        plan.path = SegmentKernelPath::Int32Batch;
    } else if (p.dtype == SEGMENT_COO_INT64 && ConfigureInt64Narrow(plan.tiling, p.ubBytes)) {
        plan.path = SegmentKernelPath::Int64Narrow;
    }
    return plan;
}
} // namespace arch22

struct DeviceCaps {
    int32_t blocks;
    uint64_t ubBytes;
};

DeviceCaps QueryDeviceCaps(int device) {
    static thread_local int recentDevice = -1;
    static thread_local DeviceCaps recentCaps{};
    if (recentDevice == device)
        return recentCaps;
    static std::mutex mutex;
    static std::map<int, DeviceCaps> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(device);
    if (found != cache.end()) {
        recentDevice = device;
        recentCaps = found->second;
        return recentCaps;
    }
    const char *soc = aclrtGetSocName();
    TORCH_CHECK(soc && (std::string(soc).find("910B") != std::string::npos ||
                        std::string(soc).find("910_93") != std::string::npos ||
                        std::string(soc).find("910C") != std::string::npos),
                "arch22 Segment COO requires an A2/A3 device");
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    DeviceCaps caps{static_cast<int32_t>(std::max<uint32_t>(1, platform->GetCoreNumAiv())), 0};
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, caps.ubBytes);
    TORCH_CHECK(caps.ubBytes > 8192, "invalid device UB capacity");
    int64_t runtimeCores = 0;
    if (aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &runtimeCores) == ACL_SUCCESS && runtimeCores > 0)
        caps.blocks = static_cast<int32_t>(runtimeCores);
    cache.emplace(device, caps);
    recentDevice = device;
    recentCaps = caps;
    return caps;
}
#endif

#ifdef OPSGNN_DAV_2201
inline SegmentCooPlan CachedLaunchPlan(const arch22::PlanInputs &inputs, int device) {
    // Cache only the complete final plan. The inner geometry cache shares tile
    // search across source predicates; neither cache retains tensors or pointers.
    using Key = std::array<int64_t, 14>;
    const Key key{inputs.batches,
                  inputs.length,
                  inputs.channels,
                  inputs.segments,
                  inputs.indexBatches,
                  inputs.dtype,
                  inputs.reduce,
                  inputs.hasOut,
                  inputs.hasArg,
                  inputs.uniformRows,
                  device,
                  inputs.blocks,
                  static_cast<int64_t>(inputs.ubBytes),
                  inputs.sourceFlags};
    struct Entry {
        Key key{};
        SegmentCooPlan plan{};
    };
    struct PlanCache {
        std::array<Entry, 16> entries{};
        size_t size = 0, next = 0, recent = 0;
    };
    static thread_local PlanCache plans;
    if (plans.size && plans.entries[plans.recent].key == key)
        return plans.entries[plans.recent].plan;
    for (size_t i = 0; i < plans.size; ++i) {
        if (plans.entries[i].key == key) {
            plans.recent = i;
            return plans.entries[i].plan;
        }
    }
    const auto plan = arch22::BuildLaunchPlan(inputs);
    plans.entries[plans.next] = {key, plan};
    plans.recent = plans.next;
    plans.next = (plans.next + 1) % plans.entries.size();
    plans.size = std::min(plans.size + 1, plans.entries.size());
    return plan;
}
#endif

SegmentCooPlan MakePlan(const SegmentShape &shape, int64_t segments, SegmentCooDtype dtype, int64_t reduce, bool hasOut,
                        bool hasArg, int64_t uniformRows, int device, int sourceFlags) {
#ifdef OPSGNN_DAV_2201
    const auto caps = QueryDeviceCaps(device);
    arch22::PlanInputs inputs{shape.before,
                              shape.length,
                              shape.after,
                              segments,
                              shape.sharedIndex ? 1 : shape.before,
                              static_cast<SegmentCooReduce>(reduce),
                              dtype,
                              hasOut,
                              caps.blocks};
    inputs.uniformRows = uniformRows;
    inputs.hasArg = hasArg;
    inputs.sourceFlags = sourceFlags;
    inputs.ubBytes = caps.ubBytes;
    const auto plan = CachedLaunchPlan(inputs, device);
#else
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int32_t blocks = std::max<uint32_t>(1, platform->GetCoreNumAiv());
    SegmentCooPlan plan{shape.before,
                        shape.length,
                        shape.after,
                        segments,
                        shape.sharedIndex ? 1 : shape.before,
                        static_cast<SegmentCooReduce>(reduce),
                        dtype,
                        hasOut,
                        blocks};
    ConfigureSegmentCooPaths(plan);
#endif
    return plan;
}

void EnqueueReduction(torch::Tensor src, torch::Tensor index, torch::Tensor out, torch::Tensor arg,
                      const SegmentCooPlan &plan) {
#ifdef OPSGNN_DAV_2201
    torch::Tensor ptr;
    // The extrema shadow must observe the original seed after the main kernel
    // writes out. This snapshot stays on NPU and is retained by the launch.
    if ((plan.tiling.sourceFlags & kRepairExtrema) && plan.hasOut)
        ptr = out.clone();
#else
    auto ptr =
        plan.directIndex ? torch::Tensor() : torch::empty({plan.indexBatches * (plan.segments + 1)}, index.options());
#endif
    const auto currentStream = c10_npu::getCurrentNPUStream(src.device().index());
#ifdef OPSGNN_DAV_2201
    // Caller-owned inputs may originate on another stream. Fresh outputs,
    // args and pointer workspace are allocated on the current stream, and
    // the closure below retains them until the launch has been submitted.
    // torch_npu 2.7.1 recordStream does not ignore same-stream storage: marking
    // fresh outputs would delay reuse until an event completes and cause new
    // device allocations inside long batches of asynchronous public calls.
    for (const auto *tensor : {&src, &index}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), currentStream);
    }
    if (plan.hasOut)
        c10_npu::NPUCachingAllocator::recordStream(out.storage().data_ptr(), currentStream);
#endif
    auto stream = currentStream.stream(false);
    // Captured tensors retain all temporaries until the queued launch has
    // executed.
    at_npu::native::OpCommand::RunOpApi("segment_coo", [src, index, out, arg, ptr, plan, stream]() {
        SegmentCoo(src.data_ptr(), index.data_ptr<int64_t>(), out.data_ptr(),
                   arg.defined() && arg.numel() ? arg.data_ptr<int64_t>() : nullptr,
#ifdef OPSGNN_DAV_2201
                   ptr.defined() ? reinterpret_cast<int64_t *>(ptr.data_ptr()) : nullptr,
#else
                   ptr.defined() ? ptr.data_ptr<int64_t>() : nullptr,
#endif
                   plan, stream);
        return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
    });
}
#ifdef OPSGNN_DAV_2201
inline auto ValidateIndexForOutput(const torch::Tensor &src, const torch::Tensor &index,
                                   const std::optional<torch::Tensor> &optional_out,
                                   const std::optional<int64_t> &dim_size) {
    // With out supplied, its shape determines the segment count; dim_size is
    // ignored by the reference and the task's no-out shape-inference rule.
    TORCH_CHECK(optional_out.has_value() || !dim_size.has_value() || *dim_size >= 0,
                "dim_size must be non-negative when out is not provided");
    int64_t requested = dim_size.value_or(-1);
    if (optional_out.has_value()) {
        TORCH_CHECK(optional_out->dim() == src.dim(), "out rank mismatch");
        requested = optional_out->size(index.dim() - 1);
    }
    return ValidateIndex(index, requested);
}

inline bool NeedsRangeProof(SegmentCooDtype dtype, int64_t reduce, bool returnArg) {
    return (dtype == SEGMENT_COO_FLOAT32 && reduce <= SEGMENT_COO_MEAN) ||
           (dtype == SEGMENT_COO_INT64 && (reduce <= SEGMENT_COO_MEAN || !returnArg));
}

inline int SourceFlags(const torch::Tensor &src, const SegmentShape &dims, int64_t uniformRows, SegmentCooDtype dtype,
                       int64_t reduce, bool hasOut, bool returnArg) {
    const bool boundedSource =
        !hasOut && NeedsRangeProof(dtype, reduce, returnArg) && ProveSourceRange(src, dims.length, reduce);
    const bool orderedFloat =
        reduce <= SEGMENT_COO_MEAN && src.is_floating_point() && (hasOut || NeedsCpuOrder(src, dims.dim, uniformRows));
    const bool repairExtrema = reduce >= SEGMENT_COO_MIN && (hasOut || returnArg);
    return (boundedSource ? kSourceRangeProven : 0) | (orderedFloat ? kCpuOrderedFloat : 0) |
           (repairExtrema ? kRepairExtrema : 0);
}

inline void ProtectOutputAliases(torch::Tensor &src, torch::Tensor &index, const torch::Tensor &out) {
    if (at::get_overlap_status(src, out) != at::MemOverlapStatus::No)
        src = src.clone();
    if (at::get_overlap_status(index, out) != at::MemOverlapStatus::No)
        index = index.clone();
}
#endif

inline torch::Tensor FinalizeOutput(torch::Tensor out, const std::optional<torch::Tensor> &optional_out) {
    if (optional_out.has_value() && out.data_ptr() != optional_out->data_ptr()) {
        optional_out->copy_(out);
        out = *optional_out;
    }
#ifdef OPSGNN_DAV_2201
    else if (optional_out.has_value()) {
        out.unsafeGetTensorImpl()->bump_version();
    }
#endif
    return out;
}
} // namespace

std::tuple<torch::Tensor, torch::Tensor> segment_coo_forward(torch::Tensor src, torch::Tensor index,
                                                             std::optional<torch::Tensor> optional_out,
                                                             std::optional<int64_t> dim_size, int64_t reduce,
                                                             bool return_arg) {
    ValidateInputs(src, index, reduce);
    c10::DeviceGuard guard(src.device());
    const auto dtype = ResolveDtype(src.scalar_type());
    const auto dims = PrepareIndex(src, index);
#ifdef OPSGNN_DAV_2201
    const auto indexFacts = ValidateIndexForOutput(src, index, optional_out, dim_size);
    const int sourceFlags =
        SourceFlags(src, dims, indexFacts.uniformRows, dtype, reduce, optional_out.has_value(), return_arg);
#endif
    src = AlignedContiguous(src);
    const int64_t segments = ResolveSegments(src, index, optional_out, dim_size, dims
#ifdef OPSGNN_DAV_2201
                                             ,
                                             indexFacts.maximum
#endif
    );
    TORCH_CHECK(segments >= 0, "dim_size must be non-negative");
    auto shape = src.sizes().vec();
    shape[dims.dim] = segments;
    auto out = optional_out.has_value() ? AlignedContiguous(*optional_out) : torch::empty(shape, src.options());
#ifdef OPSGNN_DAV_2201
    if (optional_out.has_value())
        ProtectOutputAliases(src, index, out);
#endif
    auto arg =
        reduce >= SEGMENT_COO_MIN && return_arg ? torch::empty(shape, src.options().dtype(at::kLong)) : torch::Tensor();
    if (out.numel() == 0) {
        return {optional_out.has_value() ? *optional_out : out, arg};
    }
    const auto plan = MakePlan(dims, segments, dtype, reduce, optional_out.has_value(), arg.defined(),
#ifdef OPSGNN_DAV_2201
                               indexFacts.uniformRows,
#else
                               0,
#endif
                               src.device().index(),
#ifdef OPSGNN_DAV_2201
                               sourceFlags
#else
                               false
#endif
    );
    EnqueueReduction(src, index, out, arg, plan);
    return {FinalizeOutput(out, optional_out), arg};
}

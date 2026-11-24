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
#include "scatter_groups.h"

#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <c10/core/DeviceGuard.h>
#include <cstdlib>
#include <list>
#include <mutex>
#include <torch/extension.h>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "scatter/op_kernel/arch22/scatter.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace opsgnn {

namespace {
constexpr size_t kIdleCompletionLimit = 8;
constexpr uint64_t kMappedHostPageBytes = 4096;

struct TicketResources {
    void *host = nullptr;
    void *hostBase = nullptr;
    void *mappedDevice = nullptr;
    uint64_t registeredBytes = 0;
    bool registrationAttempted = false;
    bool hostPending = false;
    aclrtEvent event = nullptr;
    int device = 0;
    uint64_t capacity = 0;
    aclrtStream allocationStream = nullptr;
    c10_npu::NPUCachingAllocator::NPUAllocator *allocator = nullptr;
    torch::Tensor flags;
    std::array<c10::Storage, 8> owners;
};

struct CompletionDeleter {
    void operator()(TicketResources *resource) const noexcept;
};
using CompletionResource = std::unique_ptr<TicketResources, CompletionDeleter>;

struct DeviceCompletions {
    bool poisoned = false;
    std::vector<CompletionResource> idle;
};

struct CompletionPool {
    std::mutex mutex;
    std::unordered_map<int, DeviceCompletions> devices;
};

CompletionPool &Completions() {
    // EQ/event completes before Host reuse. An older Owner may still read GPU
    // flags: reuse therefore retains actual stream, allocator and capacity keys.
    static auto *pool = new CompletionPool;
    return *pool;
}

void QuarantineResource(TicketResources *retained) noexcept {
    if (!retained)
        return;
    try {
        auto &pool = Completions();
        std::lock_guard<std::mutex> lock(pool.mutex);
        const auto found = pool.devices.find(retained->device);
        if (found != pool.devices.end())
            found->second.poisoned = true;
    } catch (...) {
        // A raw retained owner needs no allocation, including during unwinding.
    }
}

void QuarantineCompletion(CompletionResource resource) noexcept { QuarantineResource(resource.release()); }

void CompletionDeleter::operator()(TicketResources *resource) const noexcept {
    if (!resource)
        return;
    try {
        const c10::DeviceGuard guard(c10::Device(c10::DeviceType::PrivateUse1, resource->device));
        if (resource->hostPending) {
            QuarantineResource(resource);
            return;
        }
        if (resource->registrationAttempted) {
            if (aclrtHostUnregister(resource->host) != ACL_SUCCESS) {
                QuarantineResource(resource);
                return;
            }
            resource->registrationAttempted = false;
            resource->mappedDevice = nullptr;
        }
        if (resource->event && aclrtDestroyEvent(resource->event) != ACL_SUCCESS) {
            QuarantineResource(resource);
            return;
        }
        resource->event = nullptr;
        if (resource->hostBase && aclrtFreeHost(resource->hostBase) != ACL_SUCCESS) {
            QuarantineResource(resource);
            return;
        }
        resource->hostBase = resource->host = nullptr;
        // Only now may Tensor/Storage members release their recorded allocations.
        delete resource;
    } catch (...) {
        QuarantineResource(resource);
    }
}

bool OptionalMappingFailure(aclError status) {
    return status == ACL_ERROR_RT_FEATURE_NOT_SUPPORT || status == ACL_ERROR_RT_MEMORY_ALLOCATION ||
           status == ACL_ERROR_RT_RESOURCE_ALLOC_FAIL;
}

bool RuntimeErrorIsClear() {
    // Never consume/clear an unknown caller or asynchronous Runtime error.
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) == ACL_SUCCESS;
}

bool MappedHostSupported(int device) {
    TORCH_CHECK(RuntimeErrorIsClear(), "scatter mapping setup observed an existing Runtime error");
    aclrtHostMemMapCapability capability = ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED;
    const auto status = aclrtHostMemMapCapabilities(device, ACL_RT_HAC_TYPE_AIV, &capability);
    if (status != ACL_SUCCESS) {
        TORCH_CHECK(OptionalMappingFailure(status) && RuntimeErrorIsClear(),
                    "scatter Host mapping capability query failed: ", status);
        return false;
    }
    TORCH_CHECK(RuntimeErrorIsClear() &&
                    (capability == ACL_RT_HOST_MEM_MAP_SUPPORTED || capability == ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED),
                "scatter Host mapping capability state is unknown");
    return capability == ACL_RT_HOST_MEM_MAP_SUPPORTED;
}

void AllocatePlainHost(TicketResources &resource) {
    TORCH_CHECK(aclrtMallocHost(&resource.hostBase, resource.capacity) == ACL_SUCCESS,
                "scatter validation host allocation failed");
    resource.host = resource.hostBase;
}

void RecoverMappingRegistration(CompletionResource &resource, aclError registrationStatus) {
    // A failed register is not proof that nothing was registered. Only a
    // successful unregister establishes that the whole selected range is safe.
    const auto cleanup = aclrtHostUnregister(resource->host);
    if (cleanup != ACL_SUCCESS) {
        QuarantineCompletion(std::move(resource));
        TORCH_CHECK(false, "scatter mapping registration/cleanup failed: ", registrationStatus, "/", cleanup);
    }
    resource->registrationAttempted = false;
    resource->mappedDevice = nullptr;
    if (!OptionalMappingFailure(registrationStatus) || !RuntimeErrorIsClear()) {
        QuarantineCompletion(std::move(resource));
        TORCH_CHECK(false, "scatter mapping registration left an unknown Runtime state: ", registrationStatus);
    }
    // The aligned pinned allocation remains a valid ordinary D2H destination.
}

void AllocateValidationHost(CompletionResource &resource) {
    if (!MappedHostSupported(resource->device)) {
        AllocatePlainHost(*resource);
        return;
    }
    const uint64_t pages = (resource->capacity + kMappedHostPageBytes - 1) / kMappedHostPageBytes;
    resource->registeredBytes = pages * kMappedHostPageBytes;
    const uint64_t bytes = resource->registeredBytes + kMappedHostPageBytes - 1;
    const auto status = aclrtMallocHost(&resource->hostBase, bytes);
    if (status != ACL_SUCCESS) {
        if (!resource->hostBase && OptionalMappingFailure(status) && RuntimeErrorIsClear()) {
            AllocatePlainHost(*resource);
            return;
        }
        QuarantineCompletion(std::move(resource));
        TORCH_CHECK(false, "scatter mapped Host allocation failed: ", status);
    }
    const auto base = reinterpret_cast<uintptr_t>(resource->hostBase);
    TORCH_CHECK(base && base <= UINTPTR_MAX - bytes, "scatter mapped Host address overflows");
    const auto aligned = (base + kMappedHostPageBytes - 1) / kMappedHostPageBytes * kMappedHostPageBytes;
    TORCH_CHECK(aligned >= base && aligned + resource->registeredBytes <= base + bytes,
                "scatter mapped Host registration exceeds its allocation");
    resource->host = reinterpret_cast<void *>(aligned);
    resource->registrationAttempted = true;
    const auto registration =
        aclrtHostRegister(resource->host, resource->registeredBytes, ACL_HOST_REGISTER_MAPPED, &resource->mappedDevice);
    if (registration != ACL_SUCCESS || !resource->mappedDevice)
        RecoverMappingRegistration(resource, registration);
}

CompletionResource AcquireCompletion(const torch::Tensor &index, uint64_t bytes, aclrtStream allocationStream) {
    const int device = index.get_device();
    auto *allocator = c10_npu::NPUCachingAllocator::get();
    auto &pool = Completions();
    {
        std::lock_guard<std::mutex> lock(pool.mutex);
        auto &entry = pool.devices[device];
        TORCH_CHECK(!entry.poisoned, "scatter validation resources quarantined after completion or mapping failure");
        for (auto it = entry.idle.begin(); it != entry.idle.end(); ++it) {
            if ((*it)->capacity >= bytes && (*it)->allocationStream == allocationStream &&
                (*it)->allocator == allocator) {
                auto resource = std::move(*it);
                entry.idle.erase(it);
                return resource;
            }
        }
    }
    CompletionResource resource(new TicketResources);
    resource->device = device;
    resource->capacity = bytes;
    resource->allocationStream = allocationStream;
    resource->allocator = allocator;
    resource->flags = torch::empty({static_cast<int64_t>(bytes / sizeof(int32_t))}, index.options().dtype(at::kInt));
    AllocateValidationHost(resource);
    TORCH_CHECK(aclrtCreateEventExWithFlag(&resource->event, ACL_EVENT_SYNC) == ACL_SUCCESS,
                "scatter validation reusable event creation failed");
    return resource;
}

void ReturnCompletion(CompletionResource resource) {
    if (resource->hostPending) {
        QuarantineCompletion(std::move(resource));
        return;
    }
    auto &pool = Completions();
    std::lock_guard<std::mutex> lock(pool.mutex);
    auto &entry = pool.devices.at(resource->device);
    if (!entry.poisoned && entry.idle.size() < kIdleCompletionLimit) {
        resource->owners = {};
        entry.idle.push_back(std::move(resource));
    }
    // On eviction the custom deleter unregisters only completed Host writes.
    // GPU flags retain the old Owner's recordStream protection on release.
}

bool CompleteUnrecordedStream(aclrtStream stream) {
    uint32_t flags = 0;
    constexpr uint32_t noWait = ACL_STREAM_PERSISTENT | ACL_STREAM_DEVICE_USE_ONLY | ACL_STREAM_CPU_SCHEDULE;
    // These special streams may report synchronization success without waiting.
    // An unknown mode is not a completion proof for pending DMA or kernel reads.
    return aclrtStreamGetFlags(stream, &flags) == ACL_SUCCESS && !(flags & noWait) &&
           aclrtSynchronizeStream(stream) == ACL_SUCCESS;
}
} // namespace

struct ScatterGroupTicket {
    CompletionResource resources;
    aclrtStream stream = nullptr;
    bool submitted = false;
    bool recorded = false;
    bool ownerEnqueued = false;
    std::atomic<bool> completed{false};

    ~ScatterGroupTicket() noexcept {
        if (!resources)
            return;
        try {
            const c10::DeviceGuard guard(c10::Device(c10::DeviceType::PrivateUse1, resources->device));
            bool complete = true;
            if (submitted && !completed.load(std::memory_order_acquire))
                complete = recorded ? aclrtSynchronizeEvent(resources->event) == ACL_SUCCESS
                                    : CompleteUnrecordedStream(stream);
            if (!complete || (!submitted && resources->hostPending)) {
                QuarantineCompletion(std::move(resources));
                return;
            }
            resources->hostPending = false;
            // Reuse only after the owner is in this allocation's actual stream.
            // Host failure before marking discards this lease; GPU recordStream
            // still protects a partially submitted owner on resource destruction.
            if (ownerEnqueued)
                ReturnCompletion(std::move(resources));
            else
                resources.reset();
        } catch (...) {
            QuarantineCompletion(std::move(resources));
        }
    }
};

namespace {
constexpr uint64_t kMetadataCacheBytes = 64ULL * 1024 * 1024;
constexpr size_t kMetadataCacheEntries = 8;
constexpr uint64_t kValidationTileElements = 512;
constexpr uint64_t kValidationBytesPerCore = 32;
constexpr uint64_t kValidationWordsPerCore = kValidationBytesPerCore / sizeof(int32_t);

using GroupPlan = std::shared_ptr<const ScatterGroups>;

struct GroupKey {
    uintptr_t address;
    int64_t axisLength, bucketHint, storageOffset, stream;
    int device;
    std::vector<int64_t> sizes, strides;

    bool operator==(const GroupKey &other) const {
        return std::tie(address, axisLength, bucketHint, storageOffset, stream, device, sizes, strides) ==
               std::tie(other.address, other.axisLength, other.bucketHint, other.storageOffset, other.stream,
                        other.device, other.sizes, other.strides);
    }
};

struct CachedGroups {
    GroupKey key;
    GroupPlan plan;
    uint64_t bytes;
};

struct GroupsCache {
    std::mutex mutex;
    std::list<CachedGroups> entries;
    uint64_t bytes = 0;
};

GroupsCache &Cache() {
    static GroupsCache cache;
    return cache;
}

bool CacheEnabled() {
    // The complete public call always includes exact validation or cold setup.
    // A precise zero selects the diagnostic cold path; no preload API exists.
    const char *option = std::getenv("OPSGNN_SCATTER_GROUP_CACHE");
    return !(option && option[0] == '0' && option[1] == '\0');
}

GroupKey MakeKey(const torch::Tensor &index, int64_t axisLength, std::optional<int64_t> buckets,
                 const c10_npu::NPUStream &stream) {
    // Address is a lookup hint only. Every hit must pass full INT64 equality.
    return {reinterpret_cast<uintptr_t>(index.const_data_ptr()),
            axisLength,
            buckets.value_or(-1),
            index.storage_offset(),
            stream.id(),
            index.get_device(),
            index.sizes().vec(),
            index.strides().vec()};
}

GroupPlan LookupGroups(const GroupKey &key) {
    auto &cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) {
            auto plan = it->plan;
            cache.entries.splice(cache.entries.begin(), cache.entries, it);
            return plan;
        }
    }
    return nullptr;
}

void RememberGroups(const GroupKey &key, GroupPlan plan) {
    const uint64_t bytes = plan->snapshot.nbytes() + plan->rowptr.nbytes() + plan->rows.nbytes();
    auto &cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    for (auto it = cache.entries.begin(); it != cache.entries.end();) {
        if (it->key == key) {
            cache.bytes -= it->bytes;
            it = cache.entries.erase(it);
        } else
            ++it;
    }
    // An oversized replacement must still invalidate the previous same-key plan.
    if (bytes > kMetadataCacheBytes)
        return;
    while (!cache.entries.empty() &&
           (cache.entries.size() >= kMetadataCacheEntries || cache.bytes > kMetadataCacheBytes - bytes)) {
        cache.bytes -= cache.entries.back().bytes;
        cache.entries.pop_back();
    }
    cache.entries.push_front({key, std::move(plan), bytes});
    cache.bytes += bytes;
}

void ValidateIndex(const torch::Tensor &index, int64_t axisLength, std::optional<int64_t> buckets) {
    TORCH_CHECK(index.device().type() == c10::DeviceType::PrivateUse1 && index.scalar_type() == at::kLong,
                "scatter groups require an INT64 NPU index");
    TORCH_CHECK(index.dim() == 1 && axisLength > 0 && (index.numel() == 1 || index.numel() == axisLength),
                "scatter groups require a nonempty compact dimension index");
    TORCH_CHECK_VALUE(!buckets || *buckets >= 0, "scatter dim_size must be nonnegative");
    TORCH_CHECK(axisLength <= INT64_MAX / static_cast<int64_t>(sizeof(int64_t)),
                "scatter row metadata byte size overflows int64");
}

void SetBounds(ScatterGroups &plan, const torch::Tensor &host, std::optional<int64_t> requestedBuckets) {
    const auto *index = host.const_data_ptr<int64_t>();
    const auto bounds = std::minmax_element(index, index + plan.indexElements);
    plan.minimum = *bounds.first;
    plan.maximum = *bounds.second;
    TORCH_CHECK_INDEX(plan.minimum >= 0, "scatter index is outside the output dimension");
    TORCH_CHECK(plan.maximum < INT64_MAX, "scatter output dimension overflows int64");
    plan.buckets = requestedBuckets.value_or(plan.maximum + 1);
    TORCH_CHECK_INDEX(plan.maximum < plan.buckets, "scatter index is outside the output dimension");
    const uint64_t maximumElements = INT64_MAX / sizeof(int64_t);
    const uint64_t used = static_cast<uint64_t>(plan.axisLength) + plan.indexElements;
    TORCH_CHECK(used < maximumElements && static_cast<uint64_t>(plan.buckets) < maximumElements - used,
                "scatter grouping metadata byte size overflows int64");
}

std::array<torch::Tensor, 2> StableRows(ScatterGroups &plan, const torch::Tensor &host) {
    auto rowptr = torch::zeros({plan.buckets + 1}, host.options());
    auto rows = torch::empty({plan.axisLength}, host.options());
    auto *pointers = rowptr.data_ptr<int64_t>();
    auto *positions = rows.data_ptr<int64_t>();
    const auto *index = host.const_data_ptr<int64_t>();
    for (int64_t row = 0; row < plan.axisLength; ++row)
        ++pointers[index[plan.indexElements == 1 ? 0 : row] + 1];
    for (int64_t bucket = 1; bucket <= plan.buckets; ++bucket) {
        plan.maximumBucketRows = std::max(plan.maximumBucketRows, pointers[bucket]);
        pointers[bucket] += pointers[bucket - 1];
    }
    std::vector<int64_t> cursor(pointers, pointers + plan.buckets);
    for (int64_t row = 0; row < plan.axisLength; ++row)
        positions[cursor[index[plan.indexElements == 1 ? 0 : row]]++] = row;
    return {rowptr, rows};
}

GroupPlan BuildGroups(const torch::Tensor &index, int64_t axisLength, std::optional<int64_t> buckets,
                      const c10_npu::NPUStream &current) {
    auto plan = std::make_shared<ScatterGroups>();
    plan->axisLength = axisLength;
    plan->indexElements = index.numel();
    // Both the lookup snapshot and the stable order originate from these exact
    // private bytes. Never build one from a second read of mutable caller data.
    plan->snapshot = index.contiguous().clone();
    const auto host = plan->snapshot.cpu();
    SetBounds(*plan, host, buckets);
    const auto metadata = StableRows(*plan, host);
    plan->rowptr = metadata[0].to(index.options(), false, true);
    plan->rows = metadata[1].to(index.options(), false, true);
    const auto ready = current.stream(true);
    TORCH_CHECK(aclrtSynchronizeStream(ready) == ACL_SUCCESS, "scatter group metadata upload failed");
    // Blocking uploads keep CPU Tensor/vector owners alive until DMA finishes
    // and make publication to another Host thread safe. The actual group
    // consumer records rowptr/rows once at launch; snapshot has no owner use.
    return plan;
}

ScatterGroupUse ValidateGroups(const torch::Tensor &index, GroupPlan plan, const c10_npu::NPUStream &current) {
    auto input = index.contiguous();
    const uint32_t available = GetScatterGroupHardware(index.get_device()).cores;
    const uint32_t blocks = std::min<uint64_t>(
        available, (static_cast<uint64_t>(input.numel()) + kValidationTileElements - 1) / kValidationTileElements);
    auto ticket = std::make_shared<ScatterGroupTicket>();
    const uint64_t bytes = blocks * kValidationBytesPerCore;
    ticket->resources = AcquireCompletion(index, bytes, current.stream(false));
    auto flags = ticket->resources->flags;
    // These strong owners survive until EQ and its Host result complete or quarantine.
    // The lease is exclusive; only its active blocks prefix is read or written.
    ticket->resources->owners = {input.storage(), plan->snapshot.storage(), flags.storage()};
    // A caller may resize the input Storage after an error. Retain its original
    // allocation on this stream even when a quarantined StorageImpl is rebound.
    c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(), current);
    // Flush pending copies and prove that the next flag writer uses the same
    // actual stream as the previous successfully enqueued owner.
    ticket->stream = current.stream(true);
    TORCH_CHECK(ticket->stream == ticket->resources->allocationStream,
                "scatter flag lease stream changed before equality launch");
    ticket->submitted = true;
    ticket->resources->hostPending = true;
    auto *mapped = static_cast<int32_t *>(ticket->resources->mappedDevice);
    const auto status =
        mapped ? ScatterA2IndexEqualMapped(input.data_ptr<int64_t>(), plan->snapshot.data_ptr<int64_t>(), input.numel(),
                                           flags.data_ptr<int32_t>(), mapped, blocks, ticket->stream)
               : ScatterA2IndexEqual(input.data_ptr<int64_t>(), plan->snapshot.data_ptr<int64_t>(), input.numel(),
                                     flags.data_ptr<int32_t>(), blocks, ticket->stream);
    TORCH_CHECK(status == ACL_SUCCESS, "scatter index equality launch failed");
    if (!mapped)
        TORCH_CHECK(aclrtMemcpyAsync(ticket->resources->host, bytes, flags.data_ptr(), bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                                     ticket->stream) == ACL_SUCCESS,
                    "scatter equality result transfer failed");
    TORCH_CHECK(aclrtRecordEvent(ticket->resources->event, ticket->stream) == ACL_SUCCESS,
                "scatter validation event recording failed");
    ticket->recorded = true;
    return {std::move(plan), std::move(flags), blocks, std::move(ticket)};
}
} // namespace

ScatterGroupHardware GetScatterGroupHardware(int device) {
    static thread_local std::unordered_map<int, ScatterGroupHardware> capacities;
    const auto found = capacities.find(device);
    if (found != capacities.end())
        return found->second;
    const auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    ScatterGroupHardware result{std::max<uint32_t>(1, platform->GetCoreNumAiv()), 0};
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, result.ubBytes);
    TORCH_CHECK(result.ubBytes > 0, "scatter device UB capacity query failed");
    capacities.emplace(device, result);
    return result;
}

ScatterGroupUse BeginScatterGroups(torch::Tensor index, int64_t axisLength, std::optional<int64_t> requestedBuckets,
                                   bool rebuild, bool deferValidation) {
    ValidateIndex(index, axisLength, requestedBuckets);
    const c10::DeviceGuard guard(index.device());
    const auto current = c10_npu::getCurrentNPUStream(index.get_device());
    const auto key = MakeKey(index, axisLength, requestedBuckets, current);
    const bool cacheEnabled = CacheEnabled();
    if (cacheEnabled && !rebuild) {
        if (auto plan = LookupGroups(key)) {
            if (deferValidation)
                return {std::move(plan), torch::Tensor(), 0, nullptr, index.contiguous()};
            return ValidateGroups(index, std::move(plan), current);
        }
    }
    auto plan = BuildGroups(index, axisLength, requestedBuckets, current);
    if (cacheEnabled)
        RememberGroups(key, plan);
    return {std::move(plan), torch::empty({0}, index.options().dtype(at::kInt)), 0, nullptr};
}

void ValidateDeferredScatterGroups(ScatterGroupUse &use) {
    if (!use.pendingIndex.defined())
        return;
    const auto current = c10_npu::getCurrentNPUStream(use.pendingIndex.get_device());
    use = ValidateGroups(use.pendingIndex, use.plan, current);
}

void LaunchScatterGroupsPrivateSum(ScatterGroupUse &use, const torch::Tensor &source, const torch::Tensor &output,
                                   const scatter_a2::GroupTiling &tiling) {
    TORCH_CHECK(use.pendingIndex.defined() && !use.ticket && tiling.dtype == scatter_a2::Float32 &&
                    tiling.reduce == scatter_a2::Sum && !tiling.hasOut && !tiling.validationBlocks && tiling.blocks,
                "invalid private grouped Sum request");
    const auto current = c10_npu::getCurrentNPUStream(source.get_device());
    auto ticket = std::make_shared<ScatterGroupTicket>();
    const uint64_t bytes = tiling.blocks * kValidationBytesPerCore;
    ticket->resources = AcquireCompletion(use.pendingIndex, bytes, current.stream(false));
    auto input = use.pendingIndex;
    auto flags = ticket->resources->flags;
    // Retain every actual allocation before any launch can partially submit.
    // Private output/snapshot cannot be rebound by a caller before completion.
    ticket->resources->owners = {
        input.storage(), use.plan->snapshot.storage(), flags.storage(),         source.storage(), output.storage(),
        c10::Storage(),  use.plan->rowptr.storage(),   use.plan->rows.storage()};
    const std::array<c10::Storage, 5> consumers{input.storage(), source.storage(), use.plan->rowptr.storage(),
                                                use.plan->rows.storage(), flags.storage()};
    for (const auto &storage : consumers)
        c10_npu::NPUCachingAllocator::recordStream(storage.data_ptr(), current);
    ticket->stream = current.stream(true);
    TORCH_CHECK(ticket->stream == ticket->resources->allocationStream,
                "scatter flag lease stream changed before private Sum launch");
    use.validation = flags;
    use.validationBlocks = tiling.blocks;
    use.ticket = ticket;
    use.pendingIndex = torch::Tensor();
    auto *mapped = static_cast<int32_t *>(ticket->resources->mappedDevice);
    ticket->submitted = true;
    ticket->resources->hostPending = true;
    TORCH_CHECK(ScatterA2FusedPrivateSum(source.data_ptr(), input.data_ptr<int64_t>(),
                                         use.plan->snapshot.data_ptr<int64_t>(), use.plan->rowptr.data_ptr<int64_t>(),
                                         use.plan->rows.data_ptr<int64_t>(), output.data_ptr(),
                                         flags.data_ptr<int32_t>(), mapped, tiling, ticket->stream) == ACL_SUCCESS,
                "private grouped Sum launch failed");
    if (!mapped)
        TORCH_CHECK(aclrtMemcpyAsync(ticket->resources->host, bytes, flags.data_ptr(), bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                                     ticket->stream) == ACL_SUCCESS,
                    "private grouped Sum equality transfer failed");
    TORCH_CHECK(aclrtRecordEvent(ticket->resources->event, ticket->stream) == ACL_SUCCESS,
                "private grouped Sum event recording failed");
    ticket->recorded = true;
    MarkScatterGroupsOwnerEnqueued(use, ticket->stream);
}

void MarkScatterGroupsOwnerEnqueued(const ScatterGroupUse &use, aclrtStream ownerStream) noexcept {
    if (use.ticket && use.ticket->resources && use.ticket->recorded)
        use.ticket->ownerEnqueued = ownerStream == use.ticket->resources->allocationStream;
}

bool FinishScatterGroups(const ScatterGroupUse &use) {
    TORCH_CHECK(!use.pendingIndex.defined(), "scatter deferred equality has not been submitted");
    if (!use.validationBlocks)
        return true;
    TORCH_CHECK(use.ticket && use.ticket->resources && use.ticket->recorded,
                "scatter group validation ticket is incomplete");
    const c10::DeviceGuard guard(use.plan->snapshot.device());
    aclError status = ACL_SUCCESS;
    if (use.ticket->resources->mappedDevice) {
        // Mapped Host visibility is device-validated after SynchronizeEvent.
        // Keep the D2H query shortcut separate until mapped Query is verified.
        status = aclrtSynchronizeEvent(use.ticket->resources->event);
    } else {
        aclrtEventRecordedStatus completion = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        status = aclrtQueryEventStatus(use.ticket->resources->event, &completion);
        if (status == ACL_SUCCESS && completion != ACL_EVENT_RECORDED_STATUS_COMPLETE)
            status = aclrtSynchronizeEvent(use.ticket->resources->event);
    }
    if (status != ACL_SUCCESS)
        QuarantineCompletion(std::move(use.ticket->resources));
    TORCH_CHECK(status == ACL_SUCCESS, "scatter group validation synchronization failed");
    use.ticket->resources->hostPending = false;
    use.ticket->completed.store(true, std::memory_order_release);
    const auto *flags = static_cast<const int32_t *>(use.ticket->resources->host);
    for (uint32_t core = 0; core < use.validationBlocks; ++core)
        if (flags[core * kValidationWordsPerCore] != 1)
            return false;
    return true;
}

} // namespace opsgnn
#endif

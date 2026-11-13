/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GNN_SPARSE_COMMON_HOST_PROOF_CACHE_H
#define OPS_GNN_SPARSE_COMMON_HOST_PROOF_CACHE_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/record_function.h>

namespace ops_gnn {
namespace sparse {

constexpr size_t kSpmmProofCacheEntries = 256;

// Proof reuse assumes input writes are versioned/recorded PyTorch operations
// or native ops covered by an explicit write scope. Unrecorded raw-pointer or
// foreign-device writes cannot be detected by Tensor versions or this observer.
// Reuse is disabled until EnsureObserver is called at module initialization.
namespace spmm_proof_detail {

// Versions do not cover writes through .data / independent set_(storage)
// aliases. Conservatively invalidate on external ATen operations, as the
// arch22 segment_csr proofs do. Keep profiling enabled: TLS excludes only our
// own allocations and validation from this observer, not from other observers.
inline std::atomic<uint64_t> mutationEpoch{0};
inline std::atomic<uint64_t> activeExternalOps{0};
inline std::atomic<bool> observerReady{false};
inline thread_local unsigned internalDepth = 0;
inline thread_local bool canObserve = true;

struct ExternalOperation : at::ObserverContext {};

inline std::unique_ptr<at::ObserverContext> ObserveStart(const at::RecordFunction&)
{
    if (internalDepth != 0) return nullptr;
    auto context = std::make_unique<ExternalOperation>();
    activeExternalOps.fetch_add(1, std::memory_order_acq_rel);
    mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
    return context;
}

inline void ObserveEnd(const at::RecordFunction&, at::ObserverContext* context)
{
    if (context == nullptr) return;
    mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
    activeExternalOps.fetch_sub(1, std::memory_order_acq_rel);
}

inline void EnsureObserver()
{
    // Call during extension initialization, before concurrent public calls.
    // ATen global callback registration is not supported during parallel work.
    // Retain for process lifetime, avoiding unsafe callback removal at shutdown.
    static const auto handle = at::addGlobalCallback(
        at::RecordFunctionCallback(ObserveStart, ObserveEnd)
            .needsInputs(false).needsOutputs(false).samplingProb(1.0)
            .scopes({at::RecordScope::FUNCTION}));
    (void)handle;
    observerReady.store(true, std::memory_order_release);
}

inline bool CanReuse(uint64_t epoch)
{
    return canObserve && observerReady.load(std::memory_order_acquire) &&
           activeExternalOps.load(std::memory_order_acquire) == 0 &&
           mutationEpoch.load(std::memory_order_acquire) == epoch;
}

// Weak ownership prevents a dead TensorImpl from becoming a new identity.
// Layout and data address also detect metadata changes / .data rebinding that
// do not necessarily bump the original tensor's version counter.
struct TensorStamp {
    c10::weak_intrusive_ptr<c10::TensorImpl> tensor;
    int64_t version;
    int64_t offset;
    const void* data;
    c10::Device device;
    at::ScalarType dtype;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;

    explicit TensorStamp(const at::Tensor& value)
        : tensor(value.getIntrusivePtr()),
          version(value.is_inference() ? -1 : value._version()),
          offset(value.storage_offset()), data(value.const_data_ptr()),
          device(value.device()), dtype(value.scalar_type()),
          sizes(value.sizes().vec()), strides(value.strides().vec()) {}

    bool Matches(const at::Tensor& value) const
    {
        return version >= 0 && !value.is_inference() &&
               tensor.lock().get() == value.unsafeGetTensorImpl() &&
               version == value._version() && offset == value.storage_offset() &&
               data == value.const_data_ptr() && device == value.device() &&
               dtype == value.scalar_type() && value.sizes().equals(sizes) &&
               value.strides().equals(strides);
    }
};

// Bounded key-order eviction (not FIFO/LRU). Each live identity occupies at
// most one entry; a new version replaces the previous proof.
template <typename Key, typename Entry, typename Compare = std::less<Key>>
struct ProofCache {
    std::mutex mutex;
    std::map<Key, Entry, Compare> entries;

    // The caller holds mutex, including the publication-generation check.
    void StoreLocked(const Key& key, Entry entry)
    {
        if (entries.find(key) == entries.end() && entries.size() >= kSpmmProofCacheEntries) {
            entries.erase(entries.begin());
        }
        entries.insert_or_assign(key, std::move(entry));
    }
};

using CsrIdentity = std::pair<const c10::TensorImpl*, const c10::TensorImpl*>;
struct CsrIdentityLess {
    bool operator()(const CsrIdentity& lhs, const CsrIdentity& rhs) const
    {
        // std::less provides a total order for unrelated pointers; pair's
        // built-in pointer comparison does not provide that guarantee in C++17.
        const std::less<const c10::TensorImpl*> less;
        return less(lhs.first, rhs.first) ||
               (lhs.first == rhs.first && less(lhs.second, rhs.second));
    }
};

}  // namespace spmm_proof_detail

// Other native ops may write an independent storage alias without an ATen
// dispatch or a shared version bump (segment_csr even disables recording).
// Their bindings must cover the complete submission interval with this guard.
// Do not suppress their records: other proof observers still need to see them.
class SpmmExternalWriteScope {
public:
    SpmmExternalWriteScope()
    {
        using namespace spmm_proof_detail;
        activeExternalOps.fetch_add(1, std::memory_order_acq_rel);
        mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
    }

    ~SpmmExternalWriteScope()
    {
        using namespace spmm_proof_detail;
        mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
        activeExternalOps.fetch_sub(1, std::memory_order_acq_rel);
    }

    SpmmExternalWriteScope(const SpmmExternalWriteScope&) = delete;
    SpmmExternalWriteScope& operator=(const SpmmExternalWriteScope&) = delete;
};

// Enclose the complete native operation so its internal ATen work does not
// invalidate its own proofs. Supplied out may alias an older cached tensor
// through independent storage aliases: conservatively invalidate all proofs
// across the entire write/submission interval, including concurrent readers.
class SpmmProofScope {
public:
    explicit SpmmProofScope(bool suppliedOutput = false)
        : previousObservation(spmm_proof_detail::canObserve), writesOutput(suppliedOutput)
    {
        using namespace spmm_proof_detail;
        canObserve = previousObservation && at::isRecordFunctionEnabled();
        if (!canObserve || writesOutput) mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
        if (writesOutput) activeExternalOps.fetch_add(1, std::memory_order_acq_rel);
        ++internalDepth;
    }

    ~SpmmProofScope()
    {
        using namespace spmm_proof_detail;
        --internalDepth;
        if (writesOutput) {
            mutationEpoch.fetch_add(1, std::memory_order_acq_rel);
            activeExternalOps.fetch_sub(1, std::memory_order_acq_rel);
        }
        canObserve = previousObservation;
    }

    SpmmProofScope(const SpmmProofScope&) = delete;
    SpmmProofScope& operator=(const SpmmProofScope&) = delete;

private:
    const bool previousObservation;
    const bool writesOutput;
};

inline bool CachedHasNan(const at::Tensor& x)
{
    if (x.numel() == 0) return false;
    SpmmProofScope scope;
    using namespace spmm_proof_detail;
    struct Entry {
        TensorStamp source;
        uint64_t epoch;
        bool hasNan;
    };
    static ProofCache<const c10::TensorImpl*, Entry> cache;
    const auto* identity = x.unsafeGetTensorImpl();
    const auto epoch = mutationEpoch.load(std::memory_order_acquire);
    const TensorStamp source(x);
    if (source.version >= 0 && CanReuse(epoch)) {
        std::lock_guard<std::mutex> lock(cache.mutex);
        const auto found = cache.entries.find(identity);
        if (found != cache.entries.end() && found->second.epoch == epoch &&
            found->second.source.Matches(x) && CanReuse(epoch)) {
            return found->second.hasNan;
        }
    }
    const bool hasNan = x.isnan().any().item<bool>();
    if (source.Matches(x) && CanReuse(epoch)) {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (CanReuse(epoch)) cache.StoreLocked(identity, Entry{source, epoch, hasNan});
    }
    return hasNan;
}

struct SpmmCsrFacts {
    bool endpoints;
    bool pointerRange;
    bool nondecreasing;
    bool indexRange;
};

inline SpmmCsrFacts CachedSpmmCsrFacts(const at::Tensor& indptr, const at::Tensor& indices,
                                     int64_t rows, int64_t sources, int64_t edges,
                                     bool checkIndices)
{
    SpmmProofScope scope;
    using namespace spmm_proof_detail;
    struct Entry {
        TensorStamp pointer;
        TensorStamp index;
        uint64_t epoch;
        int64_t rows;
        int64_t sources;
        int64_t edges;
        bool checkIndices;
        SpmmCsrFacts facts;
    };
    static ProofCache<CsrIdentity, Entry, CsrIdentityLess> cache;
    const CsrIdentity key{indptr.unsafeGetTensorImpl(), indices.unsafeGetTensorImpl()};
    const auto epoch = mutationEpoch.load(std::memory_order_acquire);
    const TensorStamp pointer(indptr), index(indices);
    if (pointer.version >= 0 && index.version >= 0 && CanReuse(epoch)) {
        std::lock_guard<std::mutex> lock(cache.mutex);
        const auto found = cache.entries.find(key);
        if (found != cache.entries.end()) {
            const auto& entry = found->second;
            if (entry.epoch == epoch && entry.rows == rows && entry.sources == sources &&
                entry.edges == edges && entry.checkIndices == checkIndices &&
                entry.pointer.Matches(indptr) && entry.index.Matches(indices) && CanReuse(epoch)) {
                return entry.facts;
            }
        }
    }

    // A valid cold proof makes ONE blocking D2H transfer. Correct endpoints
    // and nondecreasing pointers imply every pointer is in [0, edges], so do
    // not launch a redundant range reduction on the normal cache-miss path.
    std::vector<at::Tensor> checks{
        indptr.select(0, 0).eq(0).logical_and(indptr.select(0, rows).eq(edges)),
        indptr.slice(0, 1).ge(indptr.slice(0, 0, rows)).all()};
    if (checkIndices) checks.push_back(indices.ge(0).logical_and(indices.lt(sources)).all());
    // Predicates are already bool. Preserve their dtype to avoid an extra
    // device conversion/allocation on every cache miss.
    const auto host = at::stack(checks).cpu();
    const auto* flags = host.const_data_ptr<bool>();
    // Invalid endpoint/order cases still need the explicit range predicate:
    // range errors must retain their original precedence over order errors.
    const bool pointerRange = (flags[0] && flags[1]) ||
        indptr.ge(0).logical_and(indptr.le(edges)).all().item<bool>();
    const SpmmCsrFacts facts{flags[0], pointerRange, flags[1],
                            !checkIndices || flags[2]};
    if (pointer.Matches(indptr) && index.Matches(indices) && CanReuse(epoch)) {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (CanReuse(epoch)) {
            cache.StoreLocked(key, Entry{pointer, index, epoch, rows, sources, edges, checkIndices, facts});
        }
    }
    return facts;
}

inline void CheckCachedSpmmCsr(const at::Tensor& indptr, const at::Tensor& indices,
                               int64_t rows, int64_t sources, int64_t edges,
                               const char* api, bool checkIndices = true)
{
    const auto facts = CachedSpmmCsrFacts(indptr, indices, rows, sources, edges, checkIndices);
    TORCH_CHECK(facts.endpoints, api, ": indptr must start at zero and end at nnz");
    TORCH_CHECK(facts.pointerRange, api, ": indptr values out of range");
    TORCH_CHECK(facts.nondecreasing, api, ": indptr must be non-decreasing");
    TORCH_CHECK(facts.indexRange, api, ": indices out of range");
}

}  // namespace sparse
}  // namespace ops_gnn

#endif  // OPS_GNN_SPARSE_COMMON_HOST_PROOF_CACHE_H

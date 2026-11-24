/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#ifdef OPSGNN_DAV_2201
#include "scatter/op_kernel/arch22/scatter.h"
#include <acl/acl.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <torch/types.h>
#include <tuple>

namespace opsgnn {

// Private immutable index metadata. No source values or reduction results are cached.
struct ScatterGroups {
    torch::Tensor snapshot;
    torch::Tensor rowptr;
    torch::Tensor rows;
    int64_t minimum = 0;
    int64_t maximum = -1;
    int64_t axisLength = 0;
    int64_t indexElements = 0;
    int64_t buckets = 0;
    int64_t maximumBucketRows = 0;
};

// Immutable hardware capacities, queried under the caller's DeviceGuard.
// Per-thread/device reuse avoids repeating platform queries for each call.
struct ScatterGroupHardware {
    uint32_t cores;
    uint64_t ubBytes;
};
ScatterGroupHardware GetScatterGroupHardware(int device);

struct ScatterGroupTicket;

struct ScatterGroupUse {
    std::shared_ptr<const ScatterGroups> plan;
    torch::Tensor validation;
    uint32_t validationBlocks = 0;
    std::shared_ptr<ScatterGroupTicket> ticket;
    torch::Tensor pendingIndex;
};

// Call only for nonempty compact indices and nonaliased src/index/out. A hit
// schedules exact comparison and returns an exclusive device-flags lease. The
// bucket owner checks every flag and writes nothing after a mismatch. A deferred
// hit instead requires explicit ordinary validation or a completed private owner.
ScatterGroupUse BeginScatterGroups(torch::Tensor index, int64_t axisLength, std::optional<int64_t> requestedBuckets,
                                   bool rebuild = false, bool deferValidation = false);

// Deferred hits must choose exactly one launch path before Finish.
void ValidateDeferredScatterGroups(ScatterGroupUse &use);
void LaunchScatterGroupsPrivateSum(ScatterGroupUse &use, const torch::Tensor &source, const torch::Tensor &output,
                                   const scatter_a2::GroupTiling &tiling);

// Call only after the actual owner launch succeeds. Matching its actual stream
// permits same-stream flags reuse after Finish; cold uses have no ticket.
void MarkScatterGroupsOwnerEnqueued(const ScatterGroupUse &use, aclrtStream ownerStream) noexcept;

// Queue the guarded or private owner before Finish. False requires rebuilding
// inside the same public call; a computed private result must never be published.
// The caller captures plan/validation storages and records their current stream.
bool FinishScatterGroups(const ScatterGroupUse &use);

} // namespace opsgnn

namespace opsgnn {
struct ScatterGroupedRequest {
    torch::Tensor src, index;
    std::optional<torch::Tensor> out;
    std::optional<int64_t> dimSize;
    int64_t dim, reduce;
};
// Metadata and original src/index/out byte-overlap checks precede this entry.
std::tuple<torch::Tensor, torch::Tensor> scatter_grouped_forward(const ScatterGroupedRequest &request);
} // namespace opsgnn
#endif

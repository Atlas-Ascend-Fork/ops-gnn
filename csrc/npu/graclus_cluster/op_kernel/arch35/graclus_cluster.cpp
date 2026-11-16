/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"
#include "graclus_cluster.h"

struct NeighborSearchContext {
    int64_t start;
    int64_t end;
    __gm__ int64_t* col;
    __gm__ float* weight;
    __gm__ int64_t* cluster;
    uint32_t numNodes;
};

static __aicore__ inline bool IsInvalidNode(int64_t node, uint32_t numNodes)
{
    return node < 0 || node >= static_cast<int64_t>(numNodes);
}

static __aicore__ inline void InitClusters(__gm__ int64_t* cluster, uint32_t numNodes)
{
    for (uint32_t i = 0; i < numNodes; ++i) {
        cluster[i] = -1;
    }
}

static __aicore__ inline bool IsUnavailable(int64_t neighbor, __gm__ int64_t* cluster, uint32_t numNodes)
{
    return neighbor < 0 || neighbor >= static_cast<int64_t>(numNodes) || cluster[neighbor] >= 0;
}

static __aicore__ inline void AssignClusterPair(int64_t node, int64_t neighbor, __gm__ int64_t* cluster)
{
    int64_t clusterValue = node < neighbor ? node : neighbor;
    cluster[node] = clusterValue;
    cluster[neighbor] = clusterValue;
}

static __aicore__ inline void MatchUnweightedNode(int64_t node, int64_t start, int64_t end,
                                                  __gm__ int64_t* col, __gm__ int64_t* cluster,
                                                  uint32_t numNodes)
{
    cluster[node] = node;
    for (int64_t edgeIdx = start; edgeIdx < end; ++edgeIdx) {
        int64_t neighbor = col[edgeIdx];
        if (IsUnavailable(neighbor, cluster, numNodes)) {
            continue;
        }
        AssignClusterPair(node, neighbor, cluster);
        break;
    }
}

static __aicore__ inline int64_t FindWeightedNeighbor(const NeighborSearchContext& ctx, int64_t defaultNeighbor)
{
    int64_t bestNeighbor = defaultNeighbor;
    float bestWeight = 0.0f;
    for (int64_t edgeIdx = ctx.start; edgeIdx < ctx.end; ++edgeIdx) {
        int64_t neighbor = ctx.col[edgeIdx];
        if (IsUnavailable(neighbor, ctx.cluster, ctx.numNodes)) {
            continue;
        }
        float edgeWeight = ctx.weight[edgeIdx];
        if (edgeWeight >= bestWeight) {
            bestWeight = edgeWeight;
            bestNeighbor = neighbor;
        }
    }
    return bestNeighbor;
}

static __aicore__ inline int64_t FindLastAvailableNeighbor(const NeighborSearchContext& ctx, int64_t defaultNeighbor)
{
    for (int64_t edgeIdx = ctx.end - 1; edgeIdx >= ctx.start; --edgeIdx) {
        int64_t neighbor = ctx.col[edgeIdx];
        if (!IsUnavailable(neighbor, ctx.cluster, ctx.numNodes)) {
            return neighbor;
        }
    }
    return defaultNeighbor;
}

static __aicore__ inline int64_t FindCompleteGraphNeighbor(int64_t node, __gm__ int64_t* cluster,
                                                            uint32_t numNodes)
{
    for (int64_t neighbor = static_cast<int64_t>(numNodes) - 1; neighbor >= 0; --neighbor) {
        if (neighbor != node && cluster[neighbor] < 0) {
            return neighbor;
        }
    }
    return node;
}

static __aicore__ inline int64_t SelectWeightedNeighbor(const NeighborSearchContext& ctx, int64_t node,
                                                        uint32_t weightMode)
{
    if (weightMode == 3) {
        return FindCompleteGraphNeighbor(node, ctx.cluster, ctx.numNodes);
    }
    if (weightMode == 2) {
        return FindLastAvailableNeighbor(ctx, node);
    }
    return FindWeightedNeighbor(ctx, node);
}

static __aicore__ inline void ProcessNode(int64_t node, __gm__ int64_t* rowptr, __gm__ int64_t* col,
                                          __gm__ float* weight, __gm__ int64_t* cluster,
                                          uint32_t numNodes, uint32_t hasWeight, uint32_t weightMode)
{
    if (IsInvalidNode(node, numNodes) || cluster[node] >= 0) {
        return;
    }
    int64_t start = rowptr[node];
    int64_t end = rowptr[node + 1];
    if (hasWeight == 0) {
        MatchUnweightedNode(node, start, end, col, cluster, numNodes);
        return;
    }
    NeighborSearchContext ctx = {start, end, col, weight, cluster, numNodes};
    int64_t neighbor = SelectWeightedNeighbor(ctx, node, weightMode);
    AssignClusterPair(node, neighbor, cluster);
}

__attribute__((aiv)) __global__ __aicore__ void GraclusClusterKernel(
    __gm__ int64_t* rowptr, __gm__ int64_t* col, __gm__ float* weight,
    __gm__ int64_t* node_perm, __gm__ int64_t* cluster,
    uint32_t numNodes, uint32_t hasWeight, uint32_t weightMode)
{
    InitClusters(cluster, numNodes);
    for (uint32_t permIdx = 0; permIdx < numNodes; ++permIdx) {
        ProcessNode(node_perm[permIdx], rowptr, col, weight, cluster, numNodes, hasWeight, weightMode);
    }
}

namespace opsgnn {

void GraclusCluster(int64_t* rowptr, int64_t* col, float* weight, int64_t* node_perm,
                                int64_t* cluster, uint32_t numNodes, uint32_t hasWeight,
                                uint32_t weightMode, aclrtStream stream)
{
    GraclusClusterKernel<<<1, nullptr, stream>>>(
        rowptr, col, weight, node_perm, cluster, numNodes, hasWeight, weightMode);
}

}  // namespace opsgnn

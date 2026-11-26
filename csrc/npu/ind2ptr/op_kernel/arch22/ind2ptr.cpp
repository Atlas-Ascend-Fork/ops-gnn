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
#include "common/aiv_launch_utils.h"
#include "ind2ptr/op_kernel/common/ind2ptr_launch.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220
#error "arch22/ind2ptr.cpp must be compiled for dav-2201 (A2/A3)"
#endif

namespace {

using namespace AscendC;

constexpr uint32_t kOutChunkRows = 2048;
constexpr uint32_t kRowsPerBlock = 1024;
constexpr uint32_t kMaxBlocks = 0xffffffffU;

// Each AIV owns a contiguous range of output rows. Rows are non-decreasing, so
// one advancing cursor over `ind` turns the whole range into a linear scan;
// only the first row of the range needs a binary search. Output chunks are
// staged in UB and written with DataCopyPad so concurrent blocks never issue
// overlapping scalar GM stores.
class Ind2PtrKernel {
 public:
  __aicore__ inline void Init(
      __gm__ const int64_t* ind, __gm__ int64_t* out, int64_t M, int64_t numel,
      TPipe* pipe)
  {
    M_ = M;
    numel_ = numel;
    indGm_.SetGlobalBuffer(const_cast<__gm__ int64_t*>(ind), numel);
    outGm_.SetGlobalBuffer(out, M + 1);
    pipe->InitBuffer(outBuf_, kOutChunkRows * sizeof(int64_t));
  }

  __aicore__ inline void Process()
  {
    const uint32_t blockNum = AscendC::GetBlockNum();
    const uint32_t blocks = (blockNum == 0) ? 1U : blockNum;
    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t total = M_ + 1;
    const int64_t begin = blockIdx * total / blocks;
    const int64_t end = (blockIdx + 1) * total / blocks;
    int64_t pos = LowerBound(begin);
    for (int64_t row = begin; row < end; row += kOutChunkRows) {
      const int64_t remain = end - row;
      const uint32_t count =
          static_cast<uint32_t>(remain < kOutChunkRows ? remain : kOutChunkRows);
      pos = FillChunk(row, count, pos);
    }
  }

 private:
  __aicore__ inline int64_t LowerBound(int64_t row)
  {
    int64_t low = 0;
    int64_t high = numel_;
    while (low < high) {
      const int64_t mid = low + (high - low) / 2;
      if (indGm_.GetValue(mid) < row) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    return low;
  }

  __aicore__ inline int64_t FillChunk(int64_t begin, uint32_t count, int64_t pos)
  {
    LocalTensor<int64_t> out = outBuf_.Get<int64_t>();
    for (uint32_t j = 0; j < count; ++j) {
      const int64_t row = begin + j;
      while (pos < numel_ && indGm_.GetValue(pos) < row) {
        ++pos;
      }
      out.SetValue(j, pos);
    }
    // Scalar stores fill the buffer, so wait for scalar->MTE3 ordering before
    // issuing the outbound copy.
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(outGm_[begin], out, cp);
    // The next chunk reuses the same buffer, so wait for the copy to finish
    // before the scalar unit overwrites it.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    return pos;
  }

  GlobalTensor<int64_t> indGm_;
  GlobalTensor<int64_t> outGm_;
  TBuf<TPosition::VECCALC> outBuf_;
  int64_t M_ = 0;
  int64_t numel_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void ind2ptr_kernel(
    __gm__ const int64_t* ind, __gm__ int64_t* out, int64_t M, int64_t numel)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  Ind2PtrKernel op;
  op.Init(ind, out, M, numel, &pipe);
  op.Process();
}

namespace opsgnn {

void Ind2Ptr(const int64_t* ind, int64_t* out, int64_t M, int64_t numel, aclrtStream stream)
{
  const uint64_t outputItems = static_cast<uint64_t>(M) + 1;
  const uint64_t wanted = (outputItems + kRowsPerBlock - 1) / kRowsPerBlock;
  const uint32_t bounded = wanted > kMaxBlocks ? kMaxBlocks : static_cast<uint32_t>(wanted);
  const uint32_t blocks = ops_gnn::sparse::ResolveAivCoreNum(bounded);
  ind2ptr_kernel<<<blocks, nullptr, stream>>>(ind, out, M, numel);
}

}  // namespace opsgnn

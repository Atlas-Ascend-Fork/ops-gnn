/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file csr_row_wise_sampling_num_picks_kernel.h
 * @brief Per-row pick-count helper kernel for weighted CSR row-wise sampling.
 *
 * Splits seed rows evenly across all vector cores and writes the exact number
 * of neighbors with prob > 0 per row. The host turns those counts into
 * nnz-balanced row_split boundaries and a per-row out_starts prefix table.
 */

#pragma once

#include "csr_row_wise_sampling/op_kernel/arch22/csr_row_wise_sampling_common.h"
#include "csr_row_wise_sampling/op_kernel/arch22/csr_row_wise_sampling_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

__aicore__ inline uint32_t MinU32NumPicks(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

}  // namespace

template <typename IdT>
class KernelCsrRowWiseSamplingNumPicks {
 public:
  __aicore__ inline KernelCsrRowWiseSamplingNumPicks() {}

  __aicore__ inline void Init(
      GM_ADDR rows, GM_ADDR indptr, GM_ADDR data, GM_ADDR prob, GM_ADDR out,
      GM_ADDR tiling_ptr, TPipe* pipe) {
    const __gm__ CsrRowWiseSamplingWeightedTiling* tiling =
        (const __gm__ CsrRowWiseSamplingWeightedTiling*)tiling_ptr;
    num_rows_ = tiling->num_rows;
    num_samples_ = tiling->num_samples;
    replace_ = tiling->replace;
    has_data_ = tiling->has_data;
    select_all_ = tiling->select_all;
    num_total_rows_ = tiling->num_total_rows;

    rows_gm_.SetGlobalBuffer((__gm__ IdT*)rows, num_rows_);
    indptr_gm_.SetGlobalBuffer((__gm__ IdT*)indptr, num_total_rows_ + 1);
    if (has_data_) data_gm_.SetGlobalBuffer((__gm__ IdT*)data);
    prob_gm_.SetGlobalBuffer((__gm__ float*)prob);
    out_gm_.SetGlobalBuffer((__gm__ uint32_t*)out, num_rows_);

    // Counts are staged in UB and flushed per chunk through the VECOUT
    // queue: direct GM scalar stores from concurrent blocks are not
    // reliable (measured on 910B3: writes from some blocks never land).
    pipe->InitBuffer(
        out_q_, kQueueDepthWeighted, kChunkElems * sizeof(uint32_t));
  }

  __aicore__ inline void Process() {
    const uint32_t block_id = AscendC::GetBlockIdx();
    const uint32_t block_num = AscendC::GetBlockNum();
    const uint32_t blocks = (block_num == 0) ? 1u : block_num;
    const uint32_t chunk = (num_rows_ + blocks - 1) / blocks;
    const uint32_t start = block_id * chunk;
    const uint32_t end = MinU32NumPicks(start + chunk, num_rows_);
    for (uint32_t base = start; base < end; base += kChunkElems) {
      const uint32_t count = MinU32NumPicks(end - base, kChunkElems);
      ProcessChunk(base, count);
    }
  }

 private:
  static constexpr uint32_t kChunkElems = 2048;  // rows staged per flush

  __aicore__ inline void ProcessChunk(uint32_t base, uint32_t count) {
    LocalTensor<uint32_t> out = out_q_.AllocTensor<uint32_t>();
    for (uint32_t j = 0; j < count; ++j) {
      const IdT rid = rows_gm_.GetValue(base + j);
      uint32_t picks = 0;  // invalid row: consistent with the main kernel
      if (rid >= 0 && rid < static_cast<IdT>(num_total_rows_)) {
        // Edge offsets stay 64-bit: indptr values can exceed 2^32 on int64
        // graphs, and truncating them silently reads the wrong edge.
        const IdT off = indptr_gm_.GetValue(static_cast<uint32_t>(rid));
        const uint32_t deg = static_cast<uint32_t>(
            indptr_gm_.GetValue(static_cast<uint32_t>(rid) + 1) - off);
        uint32_t nnz = 0;
        for (uint32_t t = 0; t < deg; ++t) {
          const IdT eid =
              has_data_ ? data_gm_.GetValue(off + static_cast<IdT>(t))
                        : off + static_cast<IdT>(t);
          if (IsUsableWeight(prob_gm_.GetValue(static_cast<uint64_t>(eid)))) ++nnz;
        }
        if (select_all_) {
          picks = nnz;
        } else if (replace_) {
          picks = (nnz == 0) ? 0 : num_samples_;
        } else {
          picks = (nnz < num_samples_) ? nnz : num_samples_;
        }
      }
      out.SetValue(j, picks);
    }
    DataCopyExtParams cp{
        1, static_cast<uint32_t>(count * sizeof(uint32_t)), 0, 0, 0};
    out_q_.EnQue(out);
    LocalTensor<uint32_t> ready = out_q_.DeQue<uint32_t>();
    DataCopyPad(out_gm_[base], ready, cp);
    out_q_.FreeTensor(ready);
  }

  GlobalTensor<IdT> rows_gm_, indptr_gm_, data_gm_;
  GlobalTensor<float> prob_gm_;
  GlobalTensor<uint32_t> out_gm_;
  TQue<TPosition::VECOUT, kQueueDepthWeighted> out_q_;
  uint32_t num_rows_ = 0, num_samples_ = 0, replace_ = 0, has_data_ = 0;
  uint32_t select_all_ = 0, num_total_rows_ = 0;
};

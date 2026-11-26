/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file csr_row_wise_sampling_kernel.h
 * @brief Multi-core AscendC kernels for weighted (probability/mask) CSR
 *        row-wise sampling.
 *
 * Design (mirrors the uniform/biased v2 multi-core kernels):
 * - A per-row pick-count helper kernel (csr_row_wise_sampling_num_picks_*)
 *   splits seed rows evenly across all vector cores and writes the exact
 *   number of neighbors with prob > 0 per row. The host turns those counts
 *   into nnz-balanced row_split boundaries and a per-row out_starts prefix
 *   table, so every block owns a disjoint output range, no cross-block
 *   reduction is needed, and every buffer is allocated exactly.
 * - The main kernel stages each row's CSR window (indices + optional data)
 *   into UB with DataCopyPad and gathers the per-edge probabilities into one
 *   UB buffer. The gather itself must read prob[edge_id] scalar-wise (prob
 *   is indexed by eid, not by slot; edge ids are arbitrary), but every
 *   subsequent read in the sampling loops hits UB instead of GM.
 * - Picked (row, col, edge_id) triples are staged in UB and flushed with
 *   DataCopyPad; out_ptr accumulates in UB and flushes in meta-sized
 *   chunks. No per-element GM traffic on the hot path (ADR-0013).
 * - Rows whose degree exceeds the UB window fall back to the v1 direct-GM
 *   path (rare hub rows). select_all huge rows are streamed in
 *   window-sized segments with batched writes, since that mode is the
 *   deterministic "pick every neighbor" case.
 *
 * Sampling methods (identical semantics to the CPU TreeSampler):
 *  - replace    : CDF inverse-transform with binary search.
 *  - no-replace : CDF linear scan with per-pick renormalization + marking.
 *  - select_all : every neighbor with prob > 0, CSR order.
 *
 * RNG: xorshift32 per-row seeded; uniform float via state / 2^32.
 *
 * NOTE: AscendC forbids double-precision operations inside __aicore__
 * functions, so all accumulation is done in float. Only float32 probability
 * input is supported (the host launcher normalizes to float32).
 *
 * write_row_pos selects the meaning of out_rows: 0 writes the seed row id
 * (non-fused COOMatrix contract), 1 writes the row position i
 * (CSRRowWiseSamplingFused contract, picked_coo_rows).
 */

#include "csr_row_wise_sampling/op_kernel/arch22/csr_row_wise_sampling_common.h"
#include "csr_row_wise_sampling/op_kernel/arch22/csr_row_wise_sampling_num_picks_kernel.h"
#include "csr_row_wise_sampling/op_kernel/arch22/csr_row_wise_sampling_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

__aicore__ inline uint32_t Xorshift32Weighted(uint32_t& x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

// Uniform float in [0, 1).
// AscendC forbids double and uint<->float casts in __aicore__, so the uint32
// state is converted via a signed int32 intermediate.
__aicore__ inline float RandUnitWeighted(uint32_t& state) {
  uint32_t r = Xorshift32Weighted(state);
  int32_t signed_r = static_cast<int32_t>(r);  // [-2^31, 2^31)
  float f = static_cast<float>(signed_r);
  return (f + 2147483648.0f) / 4294967296.0f;  // [0, 1)
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

}  // namespace

// ---------------------------------------------------------------------------
// Main kernel: weighted row-wise sampling.
// ---------------------------------------------------------------------------

template <typename IdT>
class KernelCsrRowWiseSampling {
 public:
  __aicore__ inline KernelCsrRowWiseSampling() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
      GM_ADDR prob, GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols,
      GM_ADDR out_idxs, GM_ADDR row_split, GM_ADDR out_starts,
      GM_ADDR tiling_ptr, TPipe* pipe) {
    const __gm__ CsrRowWiseSamplingWeightedTiling* tiling =
        (const __gm__ CsrRowWiseSamplingWeightedTiling*)tiling_ptr;
    num_rows_ = tiling->num_rows;
    num_samples_ = tiling->num_samples;
    replace_ = tiling->replace;
    has_data_ = tiling->has_data;
    seed_ = tiling->seed;
    select_all_ = tiling->select_all;
    num_total_rows_ = tiling->num_total_rows;
    write_row_pos_ = tiling->write_row_pos;
    const uint32_t ub_available = tiling->ub_available;

    indptr_p_ = (const __gm__ IdT*)indptr;
    rows_p_ = (const __gm__ IdT*)rows;
    prob_p_ = (const __gm__ float*)prob;
    if (has_data_) data_p_ = (const __gm__ IdT*)data;
    indices_gm_.SetGlobalBuffer((__gm__ IdT*)indices);
    if (has_data_) data_gm_.SetGlobalBuffer((__gm__ IdT*)data);
    out_ptr_gm_.SetGlobalBuffer((__gm__ IdT*)out_ptr, num_rows_ + 1);
    out_rows_gm_.SetGlobalBuffer((__gm__ IdT*)out_rows);
    out_cols_gm_.SetGlobalBuffer((__gm__ IdT*)out_cols);
    out_idxs_gm_.SetGlobalBuffer((__gm__ IdT*)out_idxs);

    const __gm__ uint32_t* row_split_p = (const __gm__ uint32_t*)row_split;
    out_starts_p_ = (const __gm__ uint32_t*)out_starts;
    block_idx_ = AscendC::GetBlockIdx();
    row_begin_ = row_split_p[block_idx_];
    row_end_ = row_split_p[block_idx_ + 1];

    window_elems_ = (ub_available > kMetaUbReserveWeighted)
                        ? (ub_available - kMetaUbReserveWeighted) /
                              kWindowInstancesWeighted / sizeof(IdT)
                        : 0;
    if (window_elems_ == 0) window_elems_ = 1;
    meta_rows_ = kMetaRowsWeighted;

    pipe->InitBuffer(
        win_idx_q_, kQueueDepthWeighted, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(
        win_data_q_, kQueueDepthWeighted, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(prob_buf_, window_elems_ * sizeof(float));
    pipe->InitBuffer(cdf_buf_, window_elems_ * sizeof(float));
    pipe->InitBuffer(used_buf_, window_elems_ * sizeof(uint32_t));
    pipe->InitBuffer(out_r_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_c_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_e_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_q_, kQueueDepthWeighted, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_ptr_buf_, (meta_rows_ + 1) * sizeof(IdT));
  }

  __aicore__ inline void Process() {
    if (row_begin_ >= row_end_) return;  // idle block: owns zero rows

    // No-replace uses per-row generation markers so a row only has to stamp
    // the offsets it picks; clear the buffer once per block.
    LocalTensor<uint32_t> used = used_buf_.Get<uint32_t>();
    for (uint32_t j = 0; j < window_elems_; ++j) used.SetValue(j, 0);

    LocalTensor<IdT> out_ptr_local = out_ptr_buf_.Get<IdT>();
    uint32_t out_ptr_staged = 0;

    for (uint32_t i = row_begin_; i < row_end_; ++i) {
      // out_ptr equals the host-baked prefix table, so the block can stage
      // its entries directly (rows row_begin_..row_end_ inclusive).
      out_ptr_local.SetValue(
          out_ptr_staged, static_cast<IdT>(out_starts_p_[i]));
      ++out_ptr_staged;

      const IdT rid = rows_p_[i];
      const IdT row_label = write_row_pos_ ? static_cast<IdT>(i) : rid;
      if (rid >= 0 && rid < static_cast<IdT>(num_total_rows_)) {
        const IdT off = indptr_p_[rid];
        const uint32_t deg =
            static_cast<uint32_t>(indptr_p_[rid + 1] - off);
        const uint32_t out_pos = out_starts_p_[i];
        const uint32_t num_picks = out_starts_p_[i + 1] - out_pos;
        if (num_picks > 0 && deg > 0) {
          uint32_t state = seed_ ^ (i * kGoldenRatioHashWeighted +
                                    kGoldenRatioOffsetWeighted);
          if (state == 0) state = kRngFallbackSeedWeighted;
          SampleRow(out_pos, row_label, off, deg, num_picks, state, i + 1,
                    used);
        }
      }

      // Flush meta staging before overflow (one spare slot remains for the
      // final prefix entry appended after the row loop).
      if (out_ptr_staged >= meta_rows_) {
        FlushOutPtr(out_ptr_local, out_ptr_staged);
        out_ptr_staged = 0;
      }
    }

    out_ptr_local.SetValue(
        out_ptr_staged, static_cast<IdT>(out_starts_p_[row_end_]));
    ++out_ptr_staged;
    FlushOutPtr(out_ptr_local, out_ptr_staged);
  }

 private:
  __aicore__ inline void SampleRow(
      uint32_t out_pos, IdT row_label, IdT off, uint32_t deg,
      uint32_t num_picks, uint32_t& state, uint32_t gen,
      LocalTensor<uint32_t>& used) {
    if (select_all_) {
      if (deg <= window_elems_) {
        SampleRowThroughUb(out_pos, row_label, off, deg, num_picks, state,
                           gen, used);
      } else {
        SampleRowSegmentedSelectAll(out_pos, row_label, off, deg, num_picks);
      }
      return;
    }
    if (deg <= window_elems_ && num_picks <= window_elems_) {
      SampleRowThroughUb(out_pos, row_label, off, deg, num_picks, state, gen,
                         used);
      return;
    }
    // Rare hub row (degree or pick count beyond the UB window): v1 direct-GM
    // scalar path. Kept simple; the hot path never reaches it.
    SampleRowDirectGm(out_pos, row_label, off, deg, num_picks, state);
  }

  // Loads window [base, base + chunk) of the row into UB: indices, data
  // (edge ids) when present, and the gathered probability of each edge.
  __aicore__ inline void LoadWindow(
      IdT off, uint32_t base, uint32_t chunk, LocalTensor<IdT>& win_idx,
      LocalTensor<IdT>& win_data, LocalTensor<float>& prob_local) {
    const uint32_t copy_bytes = chunk * sizeof(IdT);
    DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
    DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};

    win_idx = win_idx_q_.AllocTensor<IdT>();
    DataCopyPad(win_idx, indices_gm_[off + base], cp, pad);
    win_idx_q_.EnQue(win_idx);
    win_idx = win_idx_q_.DeQue<IdT>();

    if (has_data_) {
      win_data = win_data_q_.AllocTensor<IdT>();
      DataCopyPad(win_data, data_gm_[off + base], cp, pad);
      win_data_q_.EnQue(win_data);
      win_data = win_data_q_.DeQue<IdT>();
    }

    // prob is indexed by eid, not by slot, so the gather is scalar by
    // nature; it happens once per window into UB, and all sampling reads
    // then hit UB.
    for (uint32_t j = 0; j < chunk; ++j) {
      const IdT eid =
          has_data_ ? win_data.GetValue(j) : off + static_cast<IdT>(base + j);
      prob_local.SetValue(j, prob_p_[static_cast<uint64_t>(eid)]);
    }
  }

  __aicore__ inline void ReleaseWindow(LocalTensor<IdT>& win_idx,
                                       LocalTensor<IdT>& win_data) {
    win_idx_q_.FreeTensor(win_idx);
    if (has_data_) win_data_q_.FreeTensor(win_data);
  }

  __aicore__ inline void SampleRowThroughUb(
      uint32_t out_pos, IdT row_label, IdT off, uint32_t deg,
      uint32_t num_picks, uint32_t& state, uint32_t gen,
      LocalTensor<uint32_t>& used) {
    LocalTensor<IdT> win_idx;
    LocalTensor<IdT> win_data;
    LocalTensor<float> prob_local = prob_buf_.Get<float>();
    LoadWindow(off, 0, deg, win_idx, win_data, prob_local);

    LocalTensor<IdT> out_r = out_r_buf_.Get<IdT>();
    LocalTensor<IdT> out_c = out_c_buf_.Get<IdT>();
    LocalTensor<IdT> out_e = out_e_buf_.Get<IdT>();
    uint32_t written = 0;
    if (select_all_) {
      written = SampleSelectAllUb(out_r, out_c, out_e, prob_local, win_idx,
                                  win_data, deg, num_picks, row_label, off);
    } else if (replace_) {
      written = SampleReplaceUb(out_r, out_c, out_e, prob_local, win_idx,
                                win_data, deg, num_picks, row_label, off, state);
    } else {
      written = SampleNoReplaceUb(out_r, out_c, out_e, prob_local, win_idx,
                                  win_data, deg, num_picks, row_label, off,
                                  state, gen, used);
    }
    FlushPicks(out_r, out_c, out_e, out_pos, written);
    ReleaseWindow(win_idx, win_data);
  }

  __aicore__ inline void FlushPicks(LocalTensor<IdT>& out_r,
                                    LocalTensor<IdT>& out_c,
                                    LocalTensor<IdT>& out_e,
                                    uint32_t out_pos, uint32_t written) {
    if (written == 0) return;
    CopyOutStaged(out_r, out_rows_gm_[out_pos], written);
    CopyOutStaged(out_c, out_cols_gm_[out_pos], written);
    CopyOutStaged(out_e, out_idxs_gm_[out_pos], written);
  }

  __aicore__ inline uint32_t SampleSelectAllUb(
      LocalTensor<IdT>& out_r, LocalTensor<IdT>& out_c, LocalTensor<IdT>& out_e,
      LocalTensor<float>& prob_local, LocalTensor<IdT>& win_idx,
      LocalTensor<IdT>& win_data, uint32_t deg, uint32_t num_picks,
      IdT row_label, IdT off) {
    uint32_t written = 0;
    for (uint32_t j = 0; j < deg && written < num_picks; ++j) {
      if (IsUsableWeight(prob_local.GetValue(j))) {
        StagePick(out_r, out_c, out_e, written, row_label, win_idx, win_data,
                  j, off, 0);
        ++written;
      }
    }
    return written;
  }

  __aicore__ inline uint32_t SampleReplaceUb(
      LocalTensor<IdT>& out_r, LocalTensor<IdT>& out_c, LocalTensor<IdT>& out_e,
      LocalTensor<float>& prob_local, LocalTensor<IdT>& win_idx,
      LocalTensor<IdT>& win_data, uint32_t deg, uint32_t num_picks,
      IdT row_label, IdT off, uint32_t& state) {
    LocalTensor<float> cdf = cdf_buf_.Get<float>();
    float total = 0.0f;
    for (uint32_t j = 0; j < deg; ++j) {
      total += SanitizeWeight(prob_local.GetValue(j));
      cdf.SetValue(j, total);
    }
    uint32_t written = 0;
    for (uint32_t p = 0; p < num_picks; ++p) {
      const float target = RandUnitWeighted(state) * total;
      const uint32_t j = FindCdfSlot(cdf, prob_local, deg, target);
      StagePick(out_r, out_c, out_e, written, row_label, win_idx, win_data,
                j, off, 0);
      ++written;
    }
    return written;
  }

  __aicore__ inline uint32_t FindCdfSlot(LocalTensor<float>& cdf,
                                         LocalTensor<float>& prob_local,
                                         uint32_t deg, float target) {
    uint32_t lo = 0, hi = deg;
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (cdf.GetValue(mid) <= target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (lo < deg) return lo;
    // Floating-point fallback: the last slot with a usable weight.
    for (uint32_t j = deg; j > 0; --j) {
      if (IsUsableWeight(prob_local.GetValue(j - 1))) return j - 1;
    }
    return 0;
  }

  __aicore__ inline uint32_t SampleNoReplaceUb(
      LocalTensor<IdT>& out_r, LocalTensor<IdT>& out_c, LocalTensor<IdT>& out_e,
      LocalTensor<float>& prob_local, LocalTensor<IdT>& win_idx,
      LocalTensor<IdT>& win_data, uint32_t deg, uint32_t num_picks,
      IdT row_label, IdT off, uint32_t& state, uint32_t gen,
      LocalTensor<uint32_t>& used) {
    float remaining = 0.0f;
    for (uint32_t j = 0; j < deg; ++j) remaining += SanitizeWeight(prob_local.GetValue(j));
    uint32_t written = 0;
    for (uint32_t p = 0; p < num_picks; ++p) {
      if (remaining <= 0.0f) break;
      const float target = RandUnitWeighted(state) * remaining;
      const uint32_t picked =
          PickUnusedSlot(prob_local, used, gen, deg, target);
      if (picked == deg) break;  // no usable neighbor left
      used.SetValue(picked, gen);
      remaining -= SanitizeWeight(prob_local.GetValue(picked));
      StagePick(out_r, out_c, out_e, written, row_label, win_idx, win_data,
                picked, off, 0);
      ++written;
    }
    return FillRepeatedPicks(out_r, out_c, out_e, written, num_picks,
                             row_label);
  }

  __aicore__ inline uint32_t PickUnusedSlot(
      LocalTensor<float>& prob_local, LocalTensor<uint32_t>& used,
      uint32_t gen, uint32_t deg, float target) {
    float running = 0.0f;
    for (uint32_t j = 0; j < deg; ++j) {
      if (used.GetValue(j) == gen) continue;
      const float pv = prob_local.GetValue(j);
      if (!IsUsableWeight(pv)) continue;
      running += pv;
      if (running > target) return j;
    }
    // Floating-point fallback: first usable offset.
    return FirstUnusedSlot(prob_local, used, gen, deg);
  }

  __aicore__ inline uint32_t FirstUnusedSlot(
      LocalTensor<float>& prob_local, LocalTensor<uint32_t>& used,
      uint32_t gen, uint32_t deg) {
    for (uint32_t j = 0; j < deg; ++j) {
      if (used.GetValue(j) == gen) continue;
      if (IsUsableWeight(prob_local.GetValue(j))) return j;
    }
    return deg;
  }

  __aicore__ inline uint32_t FillRepeatedPicks(
      LocalTensor<IdT>& out_r, LocalTensor<IdT>& out_c,
      LocalTensor<IdT>& out_e, uint32_t written, uint32_t num_picks,
      IdT row_label) {
    if (written == 0 || written >= num_picks) return written;
    // Defensive: an exhausted remaining weight would otherwise leave holes
    // in the CSR row. Repeat the last pick.
    const uint32_t last = written - 1;
    const IdT last_c = out_c.GetValue(last);
    const IdT last_e = out_e.GetValue(last);
    for (uint32_t p = written; p < num_picks; ++p) {
      out_r.SetValue(p, row_label);
      out_c.SetValue(p, last_c);
      out_e.SetValue(p, last_e);
    }
    return num_picks;
  }

  // select_all for rows whose degree exceeds the UB window: stream the row
  // through window-sized segments, still emitting every prob > 0 neighbor
  // with a batched copy-out per segment.
  __aicore__ inline void SampleRowSegmentedSelectAll(
      uint32_t out_pos, IdT row_label, IdT off, uint32_t deg,
      uint32_t num_picks) {
    uint32_t written = 0;
    for (uint32_t base = 0; base < deg && written < num_picks;
         base += window_elems_) {
      const uint32_t chunk = MinU32(deg - base, window_elems_);
      LocalTensor<IdT> win_idx;
      LocalTensor<IdT> win_data;
      LocalTensor<float> prob_local = prob_buf_.Get<float>();
      LoadWindow(off, base, chunk, win_idx, win_data, prob_local);

      LocalTensor<IdT> out_r = out_r_buf_.Get<IdT>();
      LocalTensor<IdT> out_c = out_c_buf_.Get<IdT>();
      LocalTensor<IdT> out_e = out_e_buf_.Get<IdT>();
      uint32_t staged = 0;
      for (uint32_t j = 0; j < chunk && written < num_picks; ++j) {
        if (IsUsableWeight(prob_local.GetValue(j))) {
          StagePick(out_r, out_c, out_e, staged, row_label, win_idx, win_data,
                    j, off, base);
          ++staged;
          ++written;
        }
      }
      if (staged > 0) {
        CopyOutStaged(out_r, out_rows_gm_[out_pos + written - staged], staged);
        CopyOutStaged(out_c, out_cols_gm_[out_pos + written - staged], staged);
        CopyOutStaged(out_e, out_idxs_gm_[out_pos + written - staged], staged);
      }
      ReleaseWindow(win_idx, win_data);
    }
  }

  // v1-style direct GM scalar path for huge rows (replace / no-replace).
  __aicore__ inline void SampleRowDirectGm(
      uint32_t out_pos, IdT row_label, IdT off, uint32_t deg,
      uint32_t num_picks, uint32_t& state) {
    if (replace_) {
      SampleDirectReplace(out_pos, row_label, off, deg, num_picks, state);
    } else {
      SampleDirectNoReplace(out_pos, row_label, off, deg, num_picks, state);
    }
  }

  __aicore__ inline void SampleDirectReplace(
      uint32_t out_pos, IdT row_label, IdT off, uint32_t deg,
      uint32_t num_picks, uint32_t& state) {
    float total = 0.0f;
    for (uint32_t j = 0; j < deg; ++j) total += SanitizeWeight(ProbAt(off, j));
    for (uint32_t p = 0; p < num_picks; ++p) {
      const float target = RandUnitWeighted(state) * total;
      const uint32_t picked = FindDirectSlot(off, deg, target);
      WritePickGm(out_pos + p, row_label, off + static_cast<IdT>(picked));
    }
  }

  __aicore__ inline uint32_t FindDirectSlot(IdT off, uint32_t deg,
                                            float target) {
    float running = 0.0f;
    for (uint32_t j = 0; j < deg; ++j) {
      const float pv = ProbAt(off, j);
      if (!IsUsableWeight(pv)) continue;
      running += pv;
      if (running > target) return j;
    }
    for (uint32_t j = deg; j > 0; --j) {
      if (IsUsableWeight(ProbAt(off, j - 1))) return j - 1;
    }
    return 0;
  }

  // No replacement: renormalize per draw and dedup against the eid values
  // already written for this row (same role the GM workspace used to play,
  // without a host-allocated max_deg buffer).
  __aicore__ inline void SampleDirectNoReplace(
      uint32_t out_pos, IdT row_label, IdT off, uint32_t deg,
      uint32_t num_picks, uint32_t& state) {
    float remaining = 0.0f;
    for (uint32_t j = 0; j < deg; ++j) remaining += SanitizeWeight(ProbAt(off, j));
    uint32_t written = 0;
    for (uint32_t p = 0; p < num_picks; ++p) {
      if (remaining <= 0.0f) break;
      const float target = RandUnitWeighted(state) * remaining;
      const uint32_t picked =
          PickDirectSlot(out_pos, off, written, deg, target);
      if (picked == deg) break;
      WritePickGm(out_pos + written, row_label,
                  off + static_cast<IdT>(picked));
      remaining -= SanitizeWeight(ProbAt(off, picked));
      ++written;
    }
    RepeatDirectPicks(out_pos, written, num_picks);
  }

  __aicore__ inline uint32_t PickDirectSlot(uint32_t out_pos, IdT off,
                                            uint32_t written, uint32_t deg,
                                            float target) {
    float running = 0.0f;
    for (uint32_t j = 0; j < deg; ++j) {
      if (PickedBefore(out_pos, written, SlotId(off, j))) continue;
      const float pv = ProbAt(off, j);
      if (!IsUsableWeight(pv)) continue;
      running += pv;
      if (running > target) return j;
    }
    return FirstUnpickedDirect(out_pos, off, written, deg);
  }

  __aicore__ inline uint32_t FirstUnpickedDirect(uint32_t out_pos, IdT off,
                                                 uint32_t written,
                                                 uint32_t deg) {
    for (uint32_t j = 0; j < deg; ++j) {
      if (PickedBefore(out_pos, written, SlotId(off, j))) continue;
      if (IsUsableWeight(ProbAt(off, j))) return j;
    }
    return deg;
  }

  __aicore__ inline void RepeatDirectPicks(uint32_t out_pos, uint32_t written,
                                           uint32_t num_picks) {
    if (written == 0 || written >= num_picks) return;
    // Defensive: an exhausted remaining weight would otherwise leave holes
    // in the CSR row. Repeat the last pick.
    const uint32_t last = out_pos + written - 1;
    for (uint32_t p = written; p < num_picks; ++p) {
      out_rows_gm_.SetValue(out_pos + p, out_rows_gm_.GetValue(last));
      out_cols_gm_.SetValue(out_pos + p, out_cols_gm_.GetValue(last));
      out_idxs_gm_.SetValue(out_pos + p, out_idxs_gm_.GetValue(last));
    }
  }

  __aicore__ inline float ProbAt(IdT off, uint32_t j) {
    const IdT eid = has_data_ ? data_p_[off + static_cast<IdT>(j)]
                              : off + static_cast<IdT>(j);
    return prob_p_[static_cast<uint64_t>(eid)];
  }

  __aicore__ inline IdT SlotId(IdT off, uint32_t j) {
    return has_data_ ? data_p_[off + static_cast<IdT>(j)]
                     : off + static_cast<IdT>(j);
  }

  __aicore__ inline bool PickedBefore(uint32_t out_pos, uint32_t count,
                                      IdT picked_val) {
    for (uint32_t k = 0; k < count; ++k) {
      if (out_idxs_gm_.GetValue(out_pos + k) == picked_val) return true;
    }
    return false;
  }

  // Stages one picked triple in VECCALC buffers. `base` is the window start
  // within the row (0 for single-window rows).
  __aicore__ inline void StagePick(
      LocalTensor<IdT>& out_r, LocalTensor<IdT>& out_c,
      LocalTensor<IdT>& out_e, uint32_t j, IdT row_label,
      LocalTensor<IdT>& win_idx, LocalTensor<IdT>& win_data, uint32_t local,
      IdT off, uint32_t base) {
    out_r.SetValue(j, row_label);
    out_c.SetValue(j, win_idx.GetValue(local));
    out_e.SetValue(j, has_data_ ? win_data.GetValue(local)
                                : off + static_cast<IdT>(base + local));
  }

  __aicore__ inline void WritePickGm(uint32_t pos, IdT row_label, IdT picked) {
    out_rows_gm_.SetValue(pos, row_label);
    out_cols_gm_.SetValue(
        pos, indices_gm_.GetValue(static_cast<uint64_t>(picked)));
    out_idxs_gm_.SetValue(
        pos, has_data_ ? data_gm_.GetValue(static_cast<uint64_t>(picked))
                       : picked);
  }

  // Writes staged out_ptr entries to GM (contiguous run starting at the
  // block's row_begin_ + already-flushed count).
  __aicore__ inline void FlushOutPtr(
      LocalTensor<IdT>& staging, uint32_t count) {
    if (count == 0) return;
    DataCopyExtParams cp{
        1, static_cast<uint32_t>(count * sizeof(IdT)), 0, 0, 0};
    DataCopyPad(out_ptr_gm_[row_begin_ + out_ptr_flushed_], staging, cp);
    // DataCopyPad is asynchronous: drain MTE3 before the next row
    // overwrites the staging buffer it is still reading from.
    AscendC::PipeBarrier<PIPE_MTE3>();
    out_ptr_flushed_ += count;
  }

  // Copies `count` elements from a VECCALC staging tensor to GM through
  // the VECOUT queue, one complete Alloc/EnQue/DeQue/Free cycle.
  __aicore__ inline void CopyOutStaged(
      LocalTensor<IdT>& staging, GlobalTensor<IdT> dst, uint32_t count) {
    LocalTensor<IdT> out = out_q_.AllocTensor<IdT>();
    for (uint32_t j = 0; j < count; ++j) {
      out.SetValue(j, staging.GetValue(j));
    }
    DataCopyExtParams cp{
        1, static_cast<uint32_t>(count * sizeof(IdT)), 0, 0, 0};
    out_q_.EnQue(out);
    LocalTensor<IdT> ready = out_q_.DeQue<IdT>();
    DataCopyPad(dst, ready, cp);
    out_q_.FreeTensor(ready);
  }

  const __gm__ IdT* indptr_p_ = nullptr;
  const __gm__ IdT* rows_p_ = nullptr;
  const __gm__ IdT* data_p_ = nullptr;
  const __gm__ float* prob_p_ = nullptr;
  const __gm__ uint32_t* out_starts_p_ = nullptr;
  GlobalTensor<IdT> indices_gm_, data_gm_;
  GlobalTensor<IdT> out_ptr_gm_, out_rows_gm_, out_cols_gm_, out_idxs_gm_;
  TQue<TPosition::VECIN, kQueueDepthWeighted> win_idx_q_, win_data_q_;
  TQue<TPosition::VECOUT, kQueueDepthWeighted> out_q_;
  TBuf<TPosition::VECCALC> prob_buf_, cdf_buf_, used_buf_;
  TBuf<TPosition::VECCALC> out_r_buf_, out_c_buf_, out_e_buf_;
  TBuf<TPosition::VECCALC> out_ptr_buf_;
  uint32_t num_rows_ = 0, num_samples_ = 0, replace_ = 0, has_data_ = 0;
  uint32_t seed_ = 0, select_all_ = 0, num_total_rows_ = 0;
  uint32_t write_row_pos_ = 0;
  uint32_t row_begin_ = 0, row_end_ = 0, window_elems_ = 0;
  uint32_t block_idx_ = 0, meta_rows_ = 0, out_ptr_flushed_ = 0;
};

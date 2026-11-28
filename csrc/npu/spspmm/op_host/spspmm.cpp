/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "spspmm.h"

#include <cstdint>
#include <string>

#include <acl/acl.h>
#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "../op_kernel/common/spspmm_common.h"

namespace opsgnn {
namespace {

using at::Tensor;

// The kernels carry int32 shapes, so every dimension and non-zero count is
// checked against this limit before it is narrowed. The limit is one below
// INT32_MAX because the CSR row-pointer arrays are sized ``dim + 1``: accepting
// INT32_MAX would overflow that increment in the signed int32 domain.
constexpr int64_t kShapeLimit = 2147483646;

int DtypeOf(const Tensor& value)
{
    if (value.scalar_type() == at::kFloat) {
        return 0;
    }
    if (value.scalar_type() == at::kHalf) {
        return 1;
    }
    TORCH_CHECK(false, "spspmm supports float32 and float16 values");
    return 0;
}

void CheckSide(const Tensor& index, const Tensor& value, const char* side)
{
    const std::string index_name = std::string("index_") + side;
    const std::string value_name = std::string("value_") + side;
    TORCH_CHECK(index.device().type() == c10::DeviceType::PrivateUse1,
                "spspmm ", index_name, " must be an NPU tensor");
    TORCH_CHECK(index.scalar_type() == at::kLong, "spspmm ", index_name, " dtype must be int64");
    TORCH_CHECK(index.dim() == 2 && index.size(0) == 2, "spspmm ", index_name,
                " must have shape [2, nnz]");
    TORCH_CHECK(value.device() == index.device(), "spspmm ", value_name,
                " must live on the same NPU as its index");
    TORCH_CHECK(value.dim() == 1, "spspmm ", value_name, " must be 1-D");
    TORCH_CHECK(value.numel() == index.size(1), "spspmm ", value_name,
                " length must equal the number of edges");
    TORCH_CHECK(value.scalar_type() == at::kFloat || value.scalar_type() == at::kHalf,
                "spspmm ", value_name, " dtype must be float32 or float16");
}

void CheckArguments(const Tensor& index_a, const Tensor& value_a, const Tensor& index_b,
                    const Tensor& value_b, int64_t m, int64_t k, int64_t n)
{
    CheckSide(index_a, value_a, "a");
    CheckSide(index_b, value_b, "b");
    TORCH_CHECK(value_a.device() == value_b.device(), "spspmm a and b must live on the same NPU");
    TORCH_CHECK(value_a.scalar_type() == value_b.scalar_type(),
                "spspmm a and b must share one value dtype");
    TORCH_CHECK(m >= 0 && k >= 0 && n >= 0, "spspmm m, k and n must be nonnegative");
    TORCH_CHECK(m <= kShapeLimit && k <= kShapeLimit && n <= kShapeLimit,
                "spspmm m, k and n must fit int32");
    TORCH_CHECK(index_a.size(1) <= kShapeLimit && index_b.size(1) <= kShapeLimit,
                "spspmm edge counts must fit int32");
}

int64_t ChunksOf(int64_t count)
{
    return (count + SPSSPMM_SCAN_CHUNK - 1) / SPSSPMM_SCAN_CHUNK;
}

std::tuple<Tensor, Tensor> EmptyResult(const Tensor& value)
{
    return {at::empty({2, 0}, value.options().dtype(at::kLong)),
            at::empty({0}, value.options())};
}

// Every buffer the pipeline reads before writing is cleared on the operator's
// own stream. A host fill would be a separate task and could be scheduled
// around the directly launched kernels, so the clear is an ACL stream task.
void ClearDevice(void* pointer, size_t bytes, void* stream)
{
    if (bytes == 0) {
        return;
    }
    const aclError status =
        aclrtMemsetAsync(pointer, bytes, 0, bytes, static_cast<aclrtStream>(stream));
    TORCH_CHECK(status == ACL_SUCCESS, "spspmm: device clear failed, error code ",
                static_cast<int>(status));
}

// A single-element device read, queued behind the kernels on the same stream so
// the value is complete when the stream drains.
template <typename T>
T ReadScalar(const Tensor& tensor, void* stream)
{
    T host = 0;
    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    aclError status = aclrtMemcpyAsync(&host, sizeof(T), tensor.data_ptr(), sizeof(T),
                                       ACL_MEMCPY_DEVICE_TO_HOST, aclStream);
    TORCH_CHECK(status == ACL_SUCCESS, "spspmm: device readback failed, error code ",
                static_cast<int>(status));
    status = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(status == ACL_SUCCESS, "spspmm: stream synchronisation failed, error code ",
                static_cast<int>(status));
    return host;
}

// Row pointers of both operands, their narrowed columns and the sticky status
// block. Word 0 is raised by any kernel that meets an out-of-range coordinate,
// an unsorted row id or a duplicated column; word 1 marks a row above the arch22
// cursor capacity and stays unused on arch35.
struct CsrStructure {
    Tensor rowPtrA;
    Tensor colA;
    Tensor rowPtrB;
    Tensor colB;
    Tensor status;
};

CsrStructure BuildCsr(const Tensor& index_a, const Tensor& index_b, const SpspmmShape& shape,
                      const at::TensorOptions& i32, void* stream)
{
    const int64_t nnz_a = shape.nnzA;
    const int64_t nnz_b = shape.nnzB;
    const Tensor row_a = at::empty({nnz_a}, i32);
    const Tensor col_a = at::empty({nnz_a}, i32);
    const Tensor row_b = at::empty({nnz_b}, i32);
    const Tensor col_b = at::empty({nnz_b}, i32);
    const Tensor hist_a = at::empty({shape.m}, i32);
    const Tensor hist_b = at::empty({shape.k}, i32);
    const Tensor row_ptr_a = at::empty({static_cast<int64_t>(shape.m) + 1}, i32);
    const Tensor row_ptr_b = at::empty({static_cast<int64_t>(shape.k) + 1}, i32);
    const Tensor status = at::empty({SPSSPMM_ERROR_WORDS}, i32);

    const int64_t chunks_a = ChunksOf(shape.m);
    const int64_t chunks_b = ChunksOf(shape.k);
    const int64_t sums_a = chunks_a * SPSSPMM_CHUNK_SUM_STRIDE_INT32;
    const int64_t sums_b = chunks_b * SPSSPMM_CHUNK_SUM_STRIDE_INT32;
    const Tensor sum_a = at::empty({sums_a}, i32);
    const Tensor prefix_a = at::empty({chunks_a}, i32);
    const Tensor sum_b = at::empty({sums_b}, i32);
    const Tensor prefix_b = at::empty({chunks_b}, i32);

    // The histogram counters and the sticky status words are accumulated, so they
    // start at zero on the stream before the first kernel reads them.
    ClearDevice(hist_a.data_ptr(), static_cast<size_t>(shape.m) * sizeof(int32_t), stream);
    ClearDevice(hist_b.data_ptr(), static_cast<size_t>(shape.k) * sizeof(int32_t), stream);
    ClearDevice(status.data_ptr(), static_cast<size_t>(SPSSPMM_ERROR_WORDS) * sizeof(int32_t),
                stream);

    const int64_t* data_a = static_cast<const int64_t*>(index_a.data_ptr());
    const int64_t* data_b = static_cast<const int64_t*>(index_b.data_ptr());
    SpspmmConvertIndices(data_a, data_a + nnz_a, data_b, data_b + nnz_b, row_a.data_ptr(),
                         col_a.data_ptr(), row_b.data_ptr(), col_b.data_ptr(), shape,
                         status.data_ptr(), stream);
    SpspmmHistogram(row_a.data_ptr(), hist_a.data_ptr(), shape.nnzA, shape.m,
                    status.data_ptr(), stream);
    SpspmmHistogram(row_b.data_ptr(), hist_b.data_ptr(), shape.nnzB, shape.k,
                    status.data_ptr(), stream);
    SpspmmScanInt32(hist_a.data_ptr(), row_ptr_a.data_ptr(), sum_a.data_ptr(), prefix_a.data_ptr(),
                    shape.m, stream);
    SpspmmScanInt32(hist_b.data_ptr(), row_ptr_b.data_ptr(), sum_b.data_ptr(), prefix_b.data_ptr(),
                    shape.k, stream);
    SpspmmValidateRows(row_ptr_a.data_ptr(), col_a.data_ptr(), shape.m, shape.nnzA,
                       status.data_ptr(), stream);
    SpspmmValidateRows(row_ptr_b.data_ptr(), col_b.data_ptr(), shape.k, shape.nnzB,
                       status.data_ptr(), stream);
    return {row_ptr_a, col_a, row_ptr_b, col_b, status};
}

void CheckCoalesced(const Tensor& status, void* stream)
{
    if (ReadScalar<int32_t>(status, stream) != 0) {
        TORCH_CHECK(false, "spspmm requires coalesced inputs: every row id must be in range and "
                           "sorted ascending, and the column ids must be strictly ascending inside "
                           "each row (duplicates must be merged first, see coalesced=False)");
    }
}

// Candidate slot count of every output row and the prefix sum that turns it
// into per-row offsets. The last offset is the total candidate count.
std::tuple<Tensor, int64_t> CountSlots(const CsrStructure& csr, const SpspmmShape& shape,
                                       const at::TensorOptions& i64, void* stream)
{
    const int64_t chunks = ChunksOf(shape.m);
    const Tensor products = at::empty({shape.m}, i64);
    const Tensor offsets = at::empty({static_cast<int64_t>(shape.m) + 1}, i64);
    const Tensor sums = at::empty({chunks * SPSSPMM_CHUNK_SUM_STRIDE_INT64}, i64);
    const Tensor prefix = at::empty({chunks}, i64);

    SpspmmProductCount(csr.rowPtrA.data_ptr(), csr.colA.data_ptr(), csr.rowPtrB.data_ptr(),
                       products.data_ptr(), shape, stream);
    SpspmmScanInt64(products.data_ptr(), offsets.data_ptr(), sums.data_ptr(), prefix.data_ptr(),
                    shape.m, stream);
    return {offsets, ReadScalar<int64_t>(offsets.select(0, shape.m), stream)};
}

// Ordered per-row merge of the candidates, then their compaction into the
// coalesced COO result. An empty product is returned as an empty COO.
std::tuple<Tensor, Tensor> MergeProduct(const CsrStructure& csr, const Tensor& value_a,
                                        const Tensor& value_b, const SpspmmShape& shape, int dtype,
                                        const Tensor& products, int64_t slots, void* stream)
{
    const auto i32 = value_a.options().dtype(at::kInt);
    // ``row_ptr_c`` and ``index_c`` are int64 even though the values are float.
    const auto i64 = value_a.options().dtype(at::kLong);
    const Tensor cursors = at::empty({shape.nnzA}, i32);
    const Tensor candidate_cols = at::empty({slots}, i32);
    const Tensor candidate_vals = at::empty({slots}, value_a.options());
    const Tensor unique_counts = at::empty({shape.m}, i32);
    const Tensor row_ptr_c = at::empty({static_cast<int64_t>(shape.m) + 1}, i64);
    const int64_t chunks = ChunksOf(shape.m);
    const Tensor sums = at::empty({chunks * SPSSPMM_CHUNK_SUM_STRIDE_INT64}, i64);
    const Tensor prefix = at::empty({chunks}, i64);

    SpspmmCompute(csr.rowPtrA.data_ptr(), csr.colA.data_ptr(), value_a.data_ptr(),
                  csr.rowPtrB.data_ptr(), csr.colB.data_ptr(), value_b.data_ptr(),
                  cursors.data_ptr(), products.data_ptr(), candidate_cols.data_ptr(),
                  candidate_vals.data_ptr(), unique_counts.data_ptr(), csr.status.data_ptr(),
                  shape, dtype, stream);
    SpspmmScanInt32ToInt64(unique_counts.data_ptr(), row_ptr_c.data_ptr(), sums.data_ptr(),
                           prefix.data_ptr(), shape.m, stream);

    const int64_t nnz_c = ReadScalar<int64_t>(row_ptr_c.select(0, shape.m), stream);
    if (ReadScalar<int32_t>(csr.status, stream) != 0) {
        TORCH_CHECK(false, "spspmm internal candidate bound exceeded; the input row exceeds the "
                           "recorded product count");
    }
    if (nnz_c == 0) {
        return EmptyResult(value_a);
    }

    const Tensor index_c = at::empty({2, nnz_c}, i64);
    const Tensor value_c = at::empty({nnz_c}, value_a.options());
    SpspmmCopy(products.data_ptr(), candidate_cols.data_ptr(), candidate_vals.data_ptr(),
               unique_counts.data_ptr(), row_ptr_c.data_ptr(), index_c.data_ptr(),
               value_c.data_ptr(), nnz_c, shape, dtype, stream);
    return {index_c, value_c};
}

}  // namespace

std::tuple<Tensor, Tensor> SpspmmForward(Tensor index_a, Tensor value_a, Tensor index_b,
                                         Tensor value_b, int64_t m, int64_t k, int64_t n)
{
    CheckArguments(index_a, value_a, index_b, value_b, m, k, n);

    const int64_t nnz_a = index_a.size(1);
    const int64_t nnz_b = index_b.size(1);
    c10::DeviceGuard guard(value_a.device());
    if (nnz_a == 0 || nnz_b == 0) {
        return EmptyResult(value_a);
    }

    const int dtype = DtypeOf(value_a);

    // Copying to contiguous happens before the stream is resolved: the stream
    // lookup drains torch_npu's host queue, so it must not be overtaken by our
    // direct launches.
    const Tensor ia = index_a.contiguous();
    const Tensor va = value_a.contiguous();
    const Tensor ib = index_b.contiguous();
    const Tensor vb = value_b.contiguous();

    const SpspmmShape shape{static_cast<int32_t>(m), static_cast<int32_t>(k),
                            static_cast<int32_t>(n), static_cast<int32_t>(nnz_a),
                            static_cast<int32_t>(nnz_b)};
    void* stream = c10_npu::getCurrentNPUStream(value_a.device().index()).stream();

    const CsrStructure csr = BuildCsr(ia, ib, shape, ia.options().dtype(at::kInt), stream);
    CheckCoalesced(csr.status, stream);

    const auto [products, slots] = CountSlots(csr, shape, ia.options(), stream);
    TORCH_CHECK(slots <= SPSSPMM_MAX_PRODUCTS, "spspmm needs ", slots,
                " candidate product slots, above the supported limit of ", SPSSPMM_MAX_PRODUCTS);
    if (slots == 0) {
        return EmptyResult(va);
    }
    return MergeProduct(csr, va, vb, shape, dtype, products, slots, stream);
}

}  // namespace opsgnn

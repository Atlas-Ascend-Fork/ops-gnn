/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKey.h>
#include <pybind11/pybind11.h>
#include <string>
#include "segment_coo/op_host/segment_coo.h"
#include "segment_csr/op_host/segment_csr.h"
#ifdef OPSGNN_DAV_2201
#include "spmm_max/op_host/spmm_max.h"
#include "spmm_min/op_host/spmm_min.h"
#include "spmm_sum/op_host/spmm_sum.h"
#include "unified_spmm/op_host/unified_spmm.h"
#else
#include "gather_coo/op_host/gather_coo.h"
#include "gather_csr/op_host/gather_csr.h"
#include "grid_cluster/op_host/grid_cluster.h"
#include "random_walk/op_host/random_walk.h"
#include "segment_max_csr/op_host/segment_max_csr.h"
#include "fps/op_host/fps.h"
#include "radius/op_host/radius.h"
#include "graclus_cluster/op_host/graclus_cluster.h"
#include "ind2ptr/op_host/ind2ptr.h"
#include "ptr2ind/op_host/ptr2ind.h"
#include "scatter/op_host/scatter.h"
#include "torch_sparse_spmm/op_host/torch_sparse_spmm.h"
#include "non_diag_mask/op_host/non_diag_mask.h"
#include "torch_sparse_random_walk/op_host/torch_sparse_random_walk.h"
#include "knn/op_host/knn.h"
#endif

namespace py = pybind11;

namespace opsgnn {
bool HasPrivateUse1Kernel(const std::string& opName)
{
    const auto op = c10::Dispatcher::singleton().findOp(c10::OperatorName(opName, ""));
    return op.has_value() && op->hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1);
}
}  // namespace opsgnn

#ifndef OPSGNN_DAV_2201
TORCH_LIBRARY_FRAGMENT(ops_gnn, m) {
    m.def("fps_single(Tensor src, int out_size, int start_index) -> Tensor",
          TORCH_FN(fps_npu_single));
    m.def("fps_batch(Tensor src, Tensor ptr, Tensor out_ptr, Tensor start) -> Tensor",
          TORCH_FN(fps_npu_batch));
    // This op consumes libc's process-global rand() state.  Conservative
    // alias analysis prevents TorchScript from CSE/reordering identical calls
    // without pretending that the degree tensor itself is mutated.
    m.def(torch::schema(
              "fps_random_start(Tensor degree) -> Tensor",
              c10::AliasAnalysisKind::CONSERVATIVE),
          TORCH_FN(fps_random_start));
}
#endif

namespace {
// Public ops_gnn kernels may write through independent storage aliases without
// going through an ATen dispatcher operation. Invalidate CSR proof generations
// at their boundaries, including exceptional exits. CSR itself manages its
// internal writes and must retain proof reuse across ordinary repeated calls.
struct ExternalCsrProofBoundary {
    ExternalCsrProofBoundary() { ops_gnn::invalidate_segment_csr_proofs(); }
    ~ExternalCsrProofBoundary() { ops_gnn::invalidate_segment_csr_proofs(); }
};


void BindSegmentCoo(py::module_& m)
{
    m.attr("SEGMENT_COO_SUM") = static_cast<int32_t>(SEGMENT_COO_SUM);
    m.attr("SEGMENT_COO_MEAN") = static_cast<int32_t>(SEGMENT_COO_MEAN);
    m.attr("SEGMENT_COO_MIN") = static_cast<int32_t>(SEGMENT_COO_MIN);
    m.attr("SEGMENT_COO_MAX") = static_cast<int32_t>(SEGMENT_COO_MAX);
    m.def("segment_coo_forward", &segment_coo_forward, py::arg("src"), py::arg("index"),
          py::arg("out"), py::arg("dim_size"), py::arg("reduce"), py::arg("return_arg") = true,
          py::call_guard<py::gil_scoped_release, ExternalCsrProofBoundary>());
}

#ifndef OPSGNN_DAV_2201
void RegisterTorchSparseCompat(py::module_& module)
{
    module.def("non_diag_mask", &opsgnn::non_diag_mask_npu,
               py::arg("row"), py::arg("col"), py::arg("M"), py::arg("N"), py::arg("k"));
    module.def("torch_sparse_random_walk", &opsgnn::torch_sparse_random_walk_npu,
               py::arg("rowptr"), py::arg("col"), py::arg("start"), py::arg("walk_length"));
    module.def("torch_sparse_random_walk_with_rand", &opsgnn::torch_sparse_random_walk_with_rand_npu,
               py::arg("rowptr"), py::arg("col"), py::arg("start"), py::arg("walk_length"),
               py::arg("rand"));
}

void RegisterFps(py::module_& m)
{
    m.def("fps_single", &fps_npu_single, py::arg("src"), py::arg("out_size"),
          py::arg("start_index"), "Single-batch farthest point sampling on NPU");
    m.def("fps_batch", &fps_npu_batch, py::arg("src"), py::arg("ptr"),
          py::arg("out_ptr"), py::arg("start"),
          "Batched farthest point sampling on NPU");
    m.def("fps_random_start", &fps_random_start, py::arg("degree"),
          "Generate torch_cluster CPU-compatible random FPS starts");
}
#endif

void RegisterSegmentCsr(py::module_& module)
{
    module.def("segment_csr", &ops_gnn::segment_csr,
        py::arg("src"), py::arg("indptr"), py::arg("out") = py::none(),
        py::arg("reduce") = "sum", py::arg("return_arg") = true,
        py::call_guard<py::gil_scoped_release>());
    module.def("segment_csr_value",
        [](torch::Tensor src, torch::Tensor indptr,
           c10::optional<torch::Tensor> out, const std::string& reduce) {
            return ops_gnn::segment_csr(std::move(src), std::move(indptr),
                                       std::move(out), reduce, false)[0];
        }, py::arg("src"), py::arg("indptr"), py::arg("out") = py::none(),
        py::arg("reduce") = "sum", py::call_guard<py::gil_scoped_release>());
}

#ifndef OPSGNN_DAV_2201
void RegisterKnn(py::module_& m)
{
    m.def("knn", &opsgnn::knn_npu, py::arg("x"), py::arg("y"),
          py::arg("ptr_x") = py::none(), py::arg("ptr_y") = py::none(),
          py::arg("k"), py::arg("cosine") = false,
          py::arg("num_workers") = 1, py::arg("stream_handle") = 0,
          "Exact K-nearest-neighbor search on NPU");
}
#endif
#ifdef OPSGNN_DAV_2201
void BindArch22(py::module_& m)
{
    m.attr("npu_arch") = "dav-2201";
    BindSegmentCoo(m);
    m.def("segment_coo_value",
          [](torch::Tensor src, torch::Tensor index, std::optional<torch::Tensor> out,
             std::optional<int64_t> dimSize, int64_t reduce) {
              return std::get<0>(segment_coo_forward(std::move(src), std::move(index),
                                   std::move(out), dimSize, reduce, false));
          }, py::arg("src"), py::arg("index"), py::arg("out"), py::arg("dim_size"),
          py::arg("reduce"), py::call_guard<py::gil_scoped_release, ExternalCsrProofBoundary>());
    m.def("spmm_max_csr", &opsgnn::SpmmMaxCsr, py::arg("indptr"), py::arg("indices"),
          py::arg("x"), py::arg("out") = py::none(), "CSR SpMM max on Ascend NPU", py::call_guard<ExternalCsrProofBoundary>());
    m.def("spmm_min_csr", &opsgnn::SpmmMinCsr, py::arg("indptr"), py::arg("indices"),
          py::arg("x"), py::arg("out") = py::none(), "CSR SpMM min on Ascend NPU", py::call_guard<ExternalCsrProofBoundary>());
    m.def("spmm_sum_csr", &opsgnn::SpmmSumCsr, py::arg("indptr"), py::arg("indices"),
          py::arg("x"), py::arg("out") = py::none(), "CSR SpMM sum on Ascend NPU", py::call_guard<ExternalCsrProofBoundary>());
    m.def("unified_spmm_csr", &opsgnn::UnifiedSpmmCsr, py::arg("indptr"), py::arg("indices"),
          py::arg("x"), py::arg("op") = "copy_lhs", py::arg("reduce") = "sum",
          py::arg("out") = py::none(), "Unified two-dimensional CSR SpMM on Ascend NPU", py::call_guard<ExternalCsrProofBoundary>());
}
#else
void BindTorchSparseSpmm(py::module_& m)
{
    auto torch_sparse_spmm = m.def_submodule("torch_sparse_spmm", "torch_sparse SpMM on Ascend 950");
    torch_sparse_spmm.def("forward", &opsgnn::TorchSparseSpmmForward, py::arg("ptr"), py::arg("col"), py::arg("value"),
                          py::arg("mat"), py::arg("reduce"), py::arg("outtype"));
    torch_sparse_spmm.def("validate", &opsgnn::TorchSparseSpmmValidate, py::arg("ptr"), py::arg("col"), py::arg("n"));
    torch_sparse_spmm.def("value_backward", &opsgnn::TorchSparseSpmmValueGrad, py::arg("ptr"), py::arg("col"),
                          py::arg("mat"), py::arg("grad"), py::arg("reduce"), py::arg("width"));
    torch_sparse_spmm.def("convert", &opsgnn::TorchSparseSpmmConvert, py::arg("src"), py::arg("output"),
                          py::arg("to_ptr"));
    torch_sparse_spmm.def("permute_values", &opsgnn::TorchSparseSpmmPermuteValues, py::arg("value"), py::arg("perm"));
    torch_sparse_spmm.def("tensor_version", &opsgnn::TorchSparseSpmmTensorVersion, py::arg("tensor"));
}


void BindArch35(py::module_& m)
{
    m.attr("npu_arch") = "dav-3510";
    m.def("grid_cluster", &opsgnn::grid_cluster_npu,
          py::arg("pos"), py::arg("size"), py::arg("start") = py::none(),
          py::arg("end") = py::none(), "Regular grid clustering on NPU");
    BindTorchSparseSpmm(m);
    BindSegmentCoo(m);
    m.def("random_walk", &opsgnn::random_walk_npu,
          py::arg("rowptr"), py::arg("col"), py::arg("start"), py::arg("walk_length"),
          py::arg("p") = 1.0, py::arg("q") = 1.0, py::arg("return_edge_indices") = false,
          py::arg("neighbors_sorted") = true,
          "Random walk on a CSR graph (NPU)");
    m.def("gather_csr", &opsgnn::gather_csr, py::arg("src"), py::arg("indptr"),
          py::arg("out") = py::none(), "Gather CSR (NPU)");
    m.def("segment_max_csr", &opsgnn::segment_max_csr, py::arg("src"), py::arg("indptr"),
          py::arg("optional_out") = torch::Tensor(), "Segment max csr(NPU)");
    RegisterFps(m);
    m.def("radius", &opsgnn::radius_npu, py::arg("x"), py::arg("y"),
          py::arg("ptr_x") = py::none(), py::arg("ptr_y") = py::none(),
          py::arg("r"), py::arg("max_num_neighbors"),
          py::arg("num_workers"), py::arg("ignore_same_index") = false,
          py::arg("stream_handle") = 0,
          "Radius neighbor search on NPU");
    RegisterKnn(m);
    m.def("graclus_cluster_npu", &opsgnn::graclus_cluster_npu, py::arg("rowptr"), py::arg("col"),
          py::arg("weight"), py::arg("node_perm"), py::arg("num_nodes"), py::arg("has_weight"),
          py::arg("weight_mode"), "Graclus greedy clustering core(NPU)");
    m.def("ind2ptr", &opsgnn::ind2ptr, py::arg("ind"), py::arg("M"),
          "Convert sorted row indices to CSR row pointer (NPU), same as torch_sparse.ind2ptr");
    m.def("ptr2ind", &opsgnn::ptr2ind, py::arg("ptr"), py::arg("E"),
          "Convert CSR row pointer to row indices (NPU), same as torch_sparse.ptr2ind");
    m.def(
        "gather_coo",
        [](torch::Tensor src, torch::Tensor index, py::object outObject) {
            c10::optional<torch::Tensor> optionalOut = c10::nullopt;
            if (!outObject.is_none()) {
                optionalOut = outObject.cast<torch::Tensor>();
            }
            return opsgnn::gather_coo(src, index, optionalOut);
        },
        py::arg("src"),
        py::arg("index"),
        py::arg("out") = py::none(),
        "Gather rows using a COO index on the current NPU stream");
    m.def("scatter_forward", &opsgnn::scatter_forward, py::arg("src"), py::arg("index"),
          py::arg("dim"), py::arg("out"), py::arg("reduce"), py::arg("has_out"),
          py::arg("hot_target") = -1, "torch_scatter-compatible forward reduction(NPU)");
}
#endif

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "ops_gnn: NPU extension";
    m.def("has_private_use1_kernel", &opsgnn::HasPrivateUse1Kernel, py::arg("op_name"),
          "True if the operator already has a PrivateUse1 (NPU) kernel, e.g. a pre-existing "
          "third-party registration");
#ifdef OPSGNN_DAV_2201
    ops_gnn::initialize_segment_csr_proof_observer();
#endif
    RegisterSegmentCsr(m);
#ifdef OPSGNN_DAV_2201
    BindArch22(m);
#else
    BindArch35(m);
    RegisterTorchSparseCompat(m);
#endif
}

#ifndef OPSGNN_DAV_2201
// Keep these direct operators available even when upstream torch_sparse is not
// installed. The upstream namespace is implementation-only: torch_sparse
// remains responsible for defining its own schemas during import.
TORCH_LIBRARY_FRAGMENT(ops_gnn, m) {
    m.def("non_diag_mask(Tensor row, Tensor col, int M, int N, int k) -> Tensor");
    m.def("torch_sparse_random_walk(Tensor rowptr, Tensor col, Tensor start, int walk_length) -> Tensor");
    m.def("torch_sparse_random_walk_with_rand(Tensor rowptr, Tensor col, Tensor start, int walk_length, Tensor rand) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_gnn, PrivateUse1, m) {
    m.impl("non_diag_mask", &opsgnn::non_diag_mask_npu);
    m.impl("torch_sparse_random_walk", &opsgnn::torch_sparse_random_walk_npu);
    m.impl("torch_sparse_random_walk_with_rand", &opsgnn::torch_sparse_random_walk_with_rand_npu);
}

TORCH_LIBRARY_IMPL(torch_sparse, PrivateUse1, m) {
    m.impl("non_diag_mask", &opsgnn::non_diag_mask_npu);
    m.impl("random_walk", &opsgnn::torch_sparse_random_walk_npu);
}
#endif

"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import importlib

import torch

__version__ = '0.1.0'

try:
    import torch_npu
except ImportError:
    torch_npu = None

try:
    _pybind = importlib.import_module('._pybind', __name__)
except (ImportError, OSError) as exc:
    raise ImportError(
        "无法导入 ops_gnn._pybind 原生扩展，NPU 算子不可用。请确认："
        "(1) ops_gnn 与当前 Python、PyTorch 和 torch_npu 版本匹配；"
        "(2) 已加载 CANN 环境（例如执行 set_env.sh，且动态库搜索路径正确）；"
        "(3) 安装包是为当前 NPU 架构编译的。"
        "如从源码安装，请设置正确的 NPU_ARCH 后重新编译。"
        f"原始错误：{exc}"
    ) from exc

if _pybind.npu_arch == 'dav-2201':
    from .segment_csr import (
        segment_csr, segment_sum_csr, segment_add_csr,
        segment_mean_csr, segment_min_csr, segment_max_csr,
    )
    from .segment_coo import (
        segment_coo, segment_sum_coo, segment_add_coo,
        segment_mean_coo, segment_min_coo, segment_max_coo,
    )
    from .bspmm import bspmm
    from .spmm import spmm
    __all__ = [
        'spmm',
        'bspmm',
        'segment_csr', 'segment_sum_csr', 'segment_add_csr',
        'segment_mean_csr', 'segment_min_csr', 'segment_max_csr',
        'segment_coo', 'segment_sum_coo', 'segment_add_coo',
        'segment_mean_coo', 'segment_min_coo', 'segment_max_coo',
    ]
else:
    from .segment_csr import (
        segment_csr, segment_sum_csr, segment_add_csr,
        segment_mean_csr, segment_min_csr, segment_max_csr,
    )
    from .segment_coo import (
        segment_coo, segment_sum_coo, segment_add_coo,
        segment_mean_coo, segment_min_coo, segment_max_coo,
    )
    from .gather_csr import gather_csr
    from .fps import fps
    from .graclus_cluster import graclus_cluster
    from .gather_coo import gather_coo
    from .grid_cluster import grid_cluster
    from .ind2ptr import ind2ptr
    from .ptr2ind import ptr2ind
    from .random_walk import random_walk
    from .scatter import (
        native_available,
        scatter,
        scatter_add,
        scatter_max,
        scatter_mean,
        scatter_min,
        scatter_mul,
        scatter_sum,
    )
    from .segment_max_csr import segment_max_csr as segment_max_csr_legacy
    from .radius import radius, radius_graph
    from .knn import knn, knn_graph
    from .nearest import nearest
    from .torch_sparse_utils import (
        non_diag_mask, torch_sparse_random_walk, torch_sparse_random_walk_with_rand,
    )
    from .typing import Tensor, OptTensor

    __all__ = [
        'non_diag_mask',
        'torch_sparse_random_walk',
        'torch_sparse_random_walk_with_rand',
        'gather_csr',
        'fps',
        'graclus_cluster',
        'gather_coo',
        'grid_cluster',
        'random_walk',
        'scatter',
        'scatter_sum',
        'scatter_add',
        'scatter_mul',
        'scatter_mean',
        'scatter_min',
        'scatter_max',
        'native_available',
        'segment_max_csr',
        'segment_max_csr_legacy',
        'segment_csr',
        'segment_sum_csr',
        'segment_add_csr',
        'segment_mean_csr',
        'segment_min_csr',
        'segment_coo',
        'segment_sum_coo',
        'segment_add_coo',
        'segment_mean_coo',
        'segment_min_coo',
        'segment_max_coo',
        'ind2ptr',
        'ptr2ind',
        'radius',
        'radius_graph',
        'knn',
        'knn_graph',
        'nearest',
        'Tensor',
        'OptTensor',
    ]

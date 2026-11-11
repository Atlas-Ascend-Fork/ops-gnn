# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import contextlib
import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


def test_import():
    """测试基本导入"""
    import ops_gnn

    assert (
        (
            hasattr(ops_gnn, 'spmm')
            and hasattr(ops_gnn, 'bspmm')
        )
        or (
            hasattr(ops_gnn, 'graclus_cluster')
            and hasattr(ops_gnn, 'gather_coo')
        )
    )


def test_version():
    """测试版本号"""
    import ops_gnn
    assert ops_gnn.__version__ == '0.1.0'


def test_arch22_index_exports():
    import ops_gnn

    if 'bspmm' not in ops_gnn.__all__:
        return
    for name in ('ind2ptr', 'ptr2ind'):
        assert name in ops_gnn.__all__
        assert callable(getattr(ops_gnn, name))


def test_arch22_index_optional_dependency():
    import ops_gnn

    if 'bspmm' not in ops_gnn.__all__:
        return
    source = r'''
import importlib.abc
import sys
import torch
import torch_npu

if sys.argv[1] == 'absent':
    class BlockSparse(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'torch_sparse' or fullname.startswith('torch_sparse.'):
                raise ModuleNotFoundError('optional dependency disabled for this test')
    sys.meta_path.insert(0, BlockSparse())
else:
    import torch_sparse

def registration():
    return {name: torch._C._dispatch_has_kernel_for_dispatch_key(
        'torch_sparse::' + name, 'PrivateUse1')
        if hasattr(torch.ops.torch_sparse, name) else None
        for name in ('ind2ptr', 'ptr2ind')}

before = registration()
import ops_gnn
ind = torch.tensor([0, 0, 2], dtype=torch.int64, device='npu')
ptr = ops_gnn.ind2ptr(ind, 3)
torch.testing.assert_close(ptr.cpu(), torch.tensor([0, 2, 2, 3]))
torch.testing.assert_close(ops_gnn.ptr2ind(ptr, 3), ind)
assert registration() == before
if sys.argv[1] == 'absent':
    assert 'torch_sparse' not in sys.modules
'''
    modes = ['absent']
    if importlib.util.find_spec('torch_sparse') is not None:
        modes.append('present')
    for mode in modes:
        subprocess.run([sys.executable, '-c', source, mode],
                       check=True, timeout=120)


class _NpuDependencyGuardReached(Exception):
    """Stop collection after the CPU reference has loaded, before NPU dependencies."""


@unittest.skipUnless(
    importlib.util.find_spec("pytest") and importlib.util.find_spec("torch"),
    "golden import regressions require pytest and PyTorch",
)
class TestGoldenImportIsolation(unittest.TestCase):
    def check_foreign_golden_collision(self, use_cached_module):
        import pytest

        test_path = Path(__file__).resolve().parents[1] / "torch_sparse_spmm" / "arch35" / "test_torch_sparse_spmm.py"
        with contextlib.ExitStack() as contexts:
            directory = Path(contexts.enter_context(tempfile.TemporaryDirectory(prefix="opsgnn-golden-")))
            contexts.callback(os.chdir, Path.cwd())
            contexts.enter_context(patch.object(sys, "path", [str(directory), *sys.path]))
            original_present = "golden" in sys.modules
            original_module = sys.modules.get("golden")

            def restore_golden_module():
                if original_present:
                    sys.modules["golden"] = original_module
                else:
                    sys.modules.pop("golden", None)

            contexts.callback(restore_golden_module)
            os.chdir(directory)
            foreign_path = directory / "golden.py"
            foreign_path.write_text(
                "raise AssertionError('foreign golden.py was imported')\n",
                encoding="utf-8",
            )
            foreign_module = ModuleType("golden")
            foreign_module.__file__ = str(foreign_path)
            if use_cached_module:
                sys.modules["golden"] = foreign_module
            else:
                sys.modules.pop("golden", None)

            def stop_at_npu_dependency(module_name, *_args, **_kwargs):
                self.assertEqual(module_name, "torch_npu")
                raise _NpuDependencyGuardReached

            contexts.enter_context(patch.object(pytest, "importorskip", stop_at_npu_dependency))
            with self.assertRaises(_NpuDependencyGuardReached):
                runpy.run_path(str(test_path))

            if use_cached_module:
                self.assertIs(sys.modules["golden"], foreign_module)
            else:
                self.assertNotIn("golden", sys.modules)

    def test_spmm_collection_ignores_cached_golden(self):
        self.check_foreign_golden_collision(use_cached_module=True)

    def test_spmm_collection_ignores_search_path_golden(self):
        self.check_foreign_golden_collision(use_cached_module=False)


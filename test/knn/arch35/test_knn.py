# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NPU correctness, public API/graph contracts and independent CPU golden tests."""
import builtins
import importlib
import inspect
import itertools
import logging
import os
from pathlib import Path
import runpy
import struct
from types import SimpleNamespace

import pytest
import torch

GOLDEN_PATH = Path(__file__).with_name('golden.py')
golden_functions = runpy.run_path(GOLDEN_PATH)
golden = SimpleNamespace(**{name: golden_functions[name] for name in ('knn', 'knn_graph')})

DTYPES = [torch.float16, torch.float32]
API_NAMES = ('knn', 'knn_graph')
FLOWS = ('source_to_target', 'target_to_source')
EDGES = (
    (),
    ((0, 0),),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 1), (1, 2), (2, 0)),
    ((2, 1), (0, 0), (2, 1), (1, 2), (1, 1)),
    ((0, 2), (1, 0), (2, 1), (0, 1)),
)
LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def ops_gnn():
    return importlib.import_module('ops_gnn')


@pytest.fixture(scope='module')
def knn_module():
    return importlib.import_module('ops_gnn.knn')


@pytest.fixture
def graph_options(loop, flow):
    return SimpleNamespace(loop=loop, flow=flow)


@pytest.fixture(scope='module')
def npu_environment(ops_gnn):
    importlib.import_module('torch_npu')
    torch.npu.set_device(int(os.environ.get('NPU_DEVICE_ID', 0)))
    torch.set_num_threads(2)
    LOGGER.info('INDEPENDENT_CPU_REFERENCE %s', Path(__file__).with_name('golden.py'))
    LOGGER.info('NPU_IMPLEMENTATION %s', ops_gnn.__file__)


def assert_edges(actual, expected, n, m):
    assert actual.device.type == 'npu'
    assert actual.dtype == torch.int64
    actual = actual.cpu()
    assert actual.shape == expected.shape
    if actual.numel():
        assert bool(((actual[0] >= 0) & (actual[0] < m)).all())
        assert bool(((actual[1] >= 0) & (actual[1] < n)).all())
    # Multiset equality also detects missing or duplicate edges.
    a = (actual[0] * (n + 1) + actual[1]).sort().values
    b = (expected[0] * (n + 1) + expected[1]).sort().values
    assert torch.equal(a, b), f'{int((a != b).sum())} mismatching sorted edge entries'


def random_inputs(shape, dtype, seed, distribution='normal'):
    n, m, f = shape
    torch.manual_seed(seed)
    if distribution == 'uniform':
        return ((torch.rand(n, f) * 10 - 5).to(dtype),
                (torch.rand(m, f) * 10 - 5).to(dtype))
    return torch.randn(n, f).to(dtype), torch.randn(m, f).to(dtype)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('seed', [0, 42])
def test_cpu_oracle_smoke(ops_gnn, dtype, seed):
    x, y = random_inputs((256, 32, 32), dtype, seed)
    expected = golden.knn(x, y, 8)
    assert_edges(ops_gnn.knn(x.npu(), y.npu(), 8), expected, 256, 32)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('f', [1, 3, 4, 5, 7, 8, 16, 32, 64, 128, 768])
@pytest.mark.parametrize('distribution', ['normal', 'uniform'])
def test_feature_and_distribution(ops_gnn, dtype, f, distribution):
    x, y = random_inputs((256, 32, f), dtype, 42, distribution)
    assert_edges(ops_gnn.knn(x.npu(), y.npu(), 8),
                 golden.knn(x, y, 8), 256, 32)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('k', [1, 4, 8, 16, 32, 33, 64, 65, 100, 101, 128, 129, 257])
@pytest.mark.parametrize('f', [3, 32])
def test_k_boundaries(ops_gnn, dtype, k, f):
    x, y = random_inputs((257, 7, f), dtype, 7)
    assert_edges(ops_gnn.knn(x.npu(), y.npu(), k),
                 golden.knn(x, y, k), 257, 7)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kind', ['all_equal', 'duplicate', 'quantized'])
@pytest.mark.parametrize('n,k', [(8, 4), (11, 4), (128, 8), (257, 65)])
def test_exact_cpu_tie_selection(ops_gnn, dtype, kind, n, k):
    torch.manual_seed(9)
    x = torch.zeros(n, 3, dtype=dtype)
    if kind == 'duplicate':
        x[::3, 0] = 1
        x[1::3, 0] = -1
    elif kind == 'quantized':
        x = torch.randint(-2, 3, (n, 3)).to(dtype)
    y = torch.zeros(5, 3, dtype=dtype)
    assert_edges(ops_gnn.knn(x.npu(), y.npu(), k),
                 golden.knn(x, y, k), n, 5)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('k', [4, 65, 101])
@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('f', [3, 32])
def test_uneven_batches(ops_gnn, dtype, k, explicit, f):
    x, y = random_inputs((407, 12, f), dtype, 17)
    bx = torch.repeat_interleave(torch.tensor([0, 2, 5]), torch.tensor([129, 137, 141]))
    by = torch.repeat_interleave(torch.tensor([0, 2, 5]), torch.tensor([3, 5, 4]))
    kwargs = {'batch_size': 7} if explicit else {}
    actual = ops_gnn.knn(x.npu(), y.npu(), k, bx.npu(), by.npu(), **kwargs)
    expected = golden.knn(x, y, k, bx, by, **kwargs)
    assert_edges(actual, expected, 407, 12)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('k', [4, 64, 100])
@pytest.mark.parametrize('loop', [False, True])
@pytest.mark.parametrize('flow', ['source_to_target', 'target_to_source'])
@pytest.mark.parametrize('f', [3, 16])
def test_graph_cpu_reference(ops_gnn, dtype, k, graph_options, f):
    loop, flow = graph_options.loop, graph_options.flow
    x, _ = random_inputs((258, 1, f), dtype, 3)
    batch = torch.arange(2).repeat_interleave(129)
    expected = golden.knn_graph(x, k, batch, loop, flow)
    actual = ops_gnn.knn_graph(x.npu(), k, batch.npu(), loop, flow)
    assert_edges(actual, expected, 258, 258)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('f', [3, 32, 768])
def test_cosine_independent_cpu(ops_gnn, dtype, f):
    x, y = random_inputs((129, 11, f), dtype, 23)
    x[0] = 0
    y[0] = 0
    a = torch.nn.functional.normalize(x.double(), dim=1)
    b = torch.nn.functional.normalize(y.double(), dim=1)
    distance = 1 - b @ a.T
    cols = distance.argsort(dim=1, stable=True)[:, :8].flatten()
    rows = torch.arange(11).repeat_interleave(8)
    assert_edges(ops_gnn.knn(x.npu(), y.npu(), 8, cosine=True),
                 torch.stack([rows, cols]), 129, 11)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('k', [4, 101])
@pytest.mark.parametrize('f', [3, 32])
def test_nondefault_stream(ops_gnn, dtype, k, f):
    x, y = random_inputs((129, 19, f), dtype, 13)
    expected = golden.knn(x, y, k)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        xn, yn = x.npu(), y.npu()
        outputs = [ops_gnn.knn(xn, yn, k).clone() for _ in range(8)]
    stream.synchronize()
    for actual in outputs:
        assert_edges(actual, expected, 129, 19)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
def test_1d_and_noncontiguous(ops_gnn, dtype):
    x, y = random_inputs((129, 17, 4), dtype, 31)
    for xc, yc in [(x[:, 0], y[:, 0]), (x[:, ::2], y[:, ::2])]:
        assert_edges(ops_gnn.knn(xc.npu(), yc.npu(), 65),
                     golden.knn(xc, yc, 65), 129, 17)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n,m', [(0, 8), (8, 0), (0, 0)])
def test_empty_inputs(ops_gnn, dtype, n, m):
    x, y = torch.empty(n, 3, dtype=dtype), torch.empty(m, 3, dtype=dtype)
    assert_edges(ops_gnn.knn(x.npu(), y.npu(), 4),
                 golden.knn(x, y, 4), n, m)
    if n == 0:
        assert_edges(ops_gnn.knn_graph(x.npu(), 4),
                     golden.knn_graph(x, 4), 0, 0)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n,f', [(1024, 64), (4096, 128), (8192, 256),
                                (512, 768), (10000, 3), (20000, 128), (32000, 3)])
def test_large_taskbook_shapes(ops_gnn, dtype, n, f):
    x, y = random_inputs((n, n, f), dtype, 42)
    assert_edges(ops_gnn.knn(x.npu(), y.npu(), 4),
                 golden.knn(x, y, 4), n, n)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
def test_many_queries_large_k(ops_gnn, dtype):
    x = torch.arange(101, dtype=dtype).reshape(-1, 1)
    y = torch.linspace(0, 100, 14537).to(dtype).reshape(-1, 1)
    assert_edges(ops_gnn.knn(x.npu(), y.npu(), 101),
                 golden.knn(x, y, 101), 101, 14537)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n', [8191, 8192, 8193, 16383, 16384, 16385, 32000])
@pytest.mark.parametrize('k', [8, 65])
@pytest.mark.parametrize('kind', ['normal', 'quantized'])
def test_large_tree_small_query_grid(ops_gnn, dtype, n, k, kind):
    x, y = random_inputs((n, 11, 3), dtype, 71)
    if kind == 'quantized':
        x = x.round()
        y = y.round()
    assert_edges(ops_gnn.knn(x.npu(), y.npu(), k),
                 golden.knn(x, y, k), n, 11)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('f', [1, 2, 16, 32])
def test_large_tree_feature_widths(ops_gnn, dtype, f):
    x, y = random_inputs((8192, 11, f), dtype, 73)
    assert_edges(ops_gnn.knn(x.npu(), y.npu(), 8),
                 golden.knn(x, y, 8), 8192, 11)


@pytest.mark.npu
@pytest.mark.usefixtures('npu_environment')
@pytest.mark.parametrize('dtype', DTYPES)
def test_large_tree_uneven_batches(ops_gnn, dtype):
    sizes = torch.tensor([8192, 8193, 11])
    n = int(sizes.sum())
    x, y = random_inputs((n, 15, 3), dtype, 79)
    bx = torch.repeat_interleave(torch.tensor([0, 2, 5]), sizes)
    by = torch.tensor([0, 2, 5]).repeat_interleave(5)
    actual = ops_gnn.knn(x.npu(), y.npu(), 4, bx.npu(), by.npu(), batch_size=7)
    expected = golden.knn(x, y, 4, bx, by, batch_size=7)
    assert_edges(actual, expected, n, 15)


def reference_signature(name):
    return inspect.signature(getattr(golden, name))


@pytest.fixture
def capture_implementation(monkeypatch, knn_module):
    """Record bound arguments without touching a device or computing neighbors."""
    monkeypatch.setattr(knn_module, '_knn_impl', lambda query: query)
    monkeypatch.setattr(knn_module, '_knn_graph_impl', lambda graph: graph)


@pytest.mark.parametrize('name', API_NAMES)
def test_public_metadata_matches_reference(knn_module, name):
    implementation = getattr(knn_module, name)
    reference = getattr(golden, name)
    assert inspect.signature(implementation) == inspect.signature(reference)
    assert implementation.__annotations__ == reference.__annotations__
    assert implementation.__name__ == reference.__name__


@pytest.mark.usefixtures('capture_implementation')
@pytest.mark.parametrize('name', API_NAMES)
@pytest.mark.parametrize('positional_count', range(9))
def test_every_positional_keyword_split(knn_module, name, positional_count):
    signature = reference_signature(name)
    names = tuple(signature.parameters)
    values = tuple(object() for _ in names)
    keyword = dict(reversed(list(zip(names[positional_count:], values[positional_count:]))))
    actual = getattr(knn_module, name)(*values[:positional_count], **keyword)
    assert tuple(actual) == values


def optional_subsets():
    for name in API_NAMES:
        signature = reference_signature(name)
        optional = [key for key, value in signature.parameters.items()
                    if value.default is not inspect.Parameter.empty]
        for mask in itertools.product((False, True), repeat=len(optional)):
            yield name, tuple(key for key, selected in zip(optional, mask) if selected)


@pytest.mark.usefixtures('capture_implementation')
@pytest.mark.parametrize('name,provided', list(optional_subsets()))
def test_every_optional_keyword_subset(knn_module, name, provided):
    signature = reference_signature(name)
    keyword = {key: object() for key, value in signature.parameters.items()
               if value.default is inspect.Parameter.empty or key in provided}
    expected = signature.bind(**keyword)
    expected.apply_defaults()
    actual = getattr(knn_module, name)(**keyword)
    assert actual._asdict() == expected.arguments


@pytest.mark.usefixtures('capture_implementation')
@pytest.mark.parametrize('name', API_NAMES)
@pytest.mark.parametrize('position', range(8))
def test_duplicate_arguments_rejected(knn_module, name, position):
    names = tuple(reference_signature(name).parameters)
    values = tuple(object() for _ in names)
    with pytest.raises(TypeError):
        getattr(knn_module, name)(*values, **{names[position]: object()})


def missing_required_cases():
    for name in API_NAMES:
        for key, value in reference_signature(name).parameters.items():
            if value.default is inspect.Parameter.empty:
                yield name, key


@pytest.mark.usefixtures('capture_implementation')
@pytest.mark.parametrize('name,missing', list(missing_required_cases()))
def test_missing_required_rejected(knn_module, name, missing):
    keyword = {key: object() for key in reference_signature(name).parameters if key != missing}
    with pytest.raises(TypeError):
        getattr(knn_module, name)(**keyword)


@pytest.mark.usefixtures('capture_implementation')
@pytest.mark.parametrize('name', API_NAMES)
@pytest.mark.parametrize('kind', ('extra_positional', 'unknown_keyword'))
def test_invalid_call_rejected(knn_module, name, kind):
    values = tuple(object() for _ in reference_signature(name).parameters)
    with pytest.raises(TypeError):
        if kind == 'extra_positional':
            getattr(knn_module, name)(*values, object())
        else:
            getattr(knn_module, name)(*values, unsupported=object())


def edge_tensor(pairs):
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()


@pytest.mark.parametrize('flow', FLOWS)
@pytest.mark.parametrize('loop', (False, True))
@pytest.mark.parametrize('pairs', EDGES)
def test_graph_preserves_edge_order_and_filters_only_self_edges(monkeypatch, knn_module, flow, loop, pairs):
    coordinates = torch.zeros(3, 2)
    batch = torch.tensor([0, 0, 0])
    raw = edge_tensor(pairs)
    before = raw.clone()
    calls = []

    def dispatch(query):
        calls.append(query)
        return raw

    monkeypatch.setattr(knn_module, '_knn_impl', dispatch)
    result = knn_module.knn_graph(coordinates, 2, batch, loop, flow,
                                  cosine=True, num_workers=3, batch_size=1)
    kept = [pair for pair in pairs if loop or pair[0] != pair[1]]
    expected = [(pair[1], pair[0]) if flow == FLOWS[0] else pair for pair in kept]
    assert torch.equal(result, edge_tensor(expected))
    assert torch.equal(raw, before)
    assert result.dtype == torch.long
    assert result.device == raw.device
    assert result.is_contiguous()
    assert len(calls) == 1
    query = calls[0]
    assert query.x is coordinates and query.y is coordinates
    assert query.batch_x is batch and query.batch_y is batch
    assert query.k == 2 + int(not loop)
    assert query.cosine is True
    assert query.num_workers == 3 and query.batch_size == 1


@pytest.mark.parametrize('flow', FLOWS)
def test_graph_binds_public_arguments_only_once(monkeypatch, knn_module, flow):
    bind = inspect.Signature.bind
    graph_signature = inspect.signature(knn_module.knn_graph)
    bindings = []

    def count_bindings(public_signature, *args, **kwargs):
        bindings.append(public_signature)
        return bind(public_signature, *args, **kwargs)

    monkeypatch.setattr(inspect.Signature, 'bind', count_bindings)
    monkeypatch.setattr(knn_module, '_knn_impl', lambda query: edge_tensor(()))
    knn_module.knn_graph(torch.empty(0, 2), 2, flow=flow)
    assert bindings == [graph_signature]


def test_invalid_graph_flow_rejected_before_dispatch(monkeypatch, knn_module):
    calls = []
    monkeypatch.setattr(knn_module, '_knn_impl', calls.append)
    with pytest.raises(ValueError, match='flow'):
        knn_module.knn_graph(torch.zeros(3, 2), 2, flow='invalid')
    assert not calls


def test_golden_reference_does_not_import_extensions(monkeypatch):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch_cluster', 'ops_gnn', 'torch_npu'):
            raise AssertionError(f'golden imported {name}')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    reference = runpy.run_path(GOLDEN_PATH)
    actual = reference['knn'](torch.tensor([0., 2., 4.]), torch.tensor([3.5]), 1)
    assert torch.equal(actual, torch.tensor([[0], [2]]))


@pytest.mark.parametrize('dtype', DTYPES)
def test_golden_known_nearest_neighbors(dtype):
    x = torch.tensor([0., 2., 5., 9.], dtype=dtype)
    y = torch.tensor([1.5, 8.], dtype=dtype)
    assert torch.equal(golden.knn(x, y, 2), torch.tensor([[0, 0, 1, 1], [1, 0, 3, 2]]))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('features', [1, 3, 4, 5, 7, 8, 17])
def test_golden_grouped_distance_matches_scalar_rounding(dtype, features):
    torch.manual_seed(71)
    x, y = torch.randn(9, features).to(dtype), torch.randn(3, features).to(dtype)
    code = 'e' if dtype == torch.float16 else 'f'

    def rounded(value):
        return struct.unpack(code, struct.pack(code, value))[0]

    expected = []
    for query in y.tolist():
        row = []
        for point in x.tolist():
            square = [rounded(rounded(a - b) ** 2) for a, b in zip(query, point)]
            distance = 0.
            grouped = features - features % 4
            for start in range(0, grouped, 4):
                group = rounded(square[start] + square[start + 1])
                group = rounded(group + square[start + 2])
                group = rounded(group + square[start + 3])
                distance = rounded(distance + group)
            for value in square[grouped:]:
                distance = rounded(distance + value)
            row.append(distance)
        expected.append(row)
    assert torch.equal(golden_functions['_l2_distances'](x, y), torch.tensor(expected, dtype=dtype))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n,expected', [(8, [0, 1, 2, 3]), (11, [5, 6, 7, 8])])
def test_golden_equal_distance_leaf_and_split_order(dtype, n, expected):
    actual = golden.knn(torch.zeros(n, 3, dtype=dtype), torch.zeros(1, 3, dtype=dtype), 4)
    assert actual[1].tolist() == expected


@pytest.mark.parametrize('dtype', DTYPES)
def test_golden_noncontiguous_inputs_and_query_blocks(dtype):
    torch.manual_seed(17)
    x, y = torch.randn(129, 8).to(dtype)[:, ::2], torch.randn(73, 8).to(dtype)[:, ::2]
    actual = golden.knn(x, y, 8)
    expected = golden.knn(x.contiguous(), y.contiguous(), 8)
    assert torch.equal(actual, expected)
    pieces = []
    for first in range(0, len(y), 11):
        part = golden.knn(x, y[first:first + 11], 8)
        part[0] += first
        pieces.append(part)
    assert torch.equal(actual, torch.cat(pieces, dim=1))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('explicit', [False, True])
def test_golden_sparse_batches_remain_isolated(dtype, explicit):
    x = torch.tensor([0., 2., 1., 3., -1., 4.], dtype=dtype)
    y = torch.tensor([1.5, 2.5, 0.], dtype=dtype)
    bx, by = torch.tensor([0, 0, 2, 2, 5, 5]), torch.tensor([0, 2, 5])
    kwargs = {'batch_size': 7} if explicit else {}
    actual = golden.knn(x, y, 2, bx, by, **kwargs)
    assert torch.equal(actual, torch.tensor([[0, 0, 1, 1, 2, 2], [1, 0, 3, 2, 4, 5]]))
    assert torch.equal(bx[actual[1]], by[actual[0]])


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n,m', [(0, 8), (8, 0), (0, 0)])
def test_golden_empty_inputs(dtype, n, m):
    actual = golden.knn(torch.empty(n, 3, dtype=dtype), torch.empty(m, 3, dtype=dtype), 4)
    assert actual.shape == (2, 0) and actual.dtype == torch.long
    if not n:
        assert golden.knn_graph(torch.empty(n, 3, dtype=dtype), 4).shape == (2, 0)


@pytest.mark.parametrize('dtype', DTYPES)
def test_golden_cosine_zero_vectors_and_index_ties(dtype):
    x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [-1., 0.]], dtype=dtype)
    y = torch.tensor([[0., 0.], [1., 0.]], dtype=dtype)
    assert torch.equal(golden.knn(x, y, 2, cosine=True), torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]]))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('loop', [False, True])
@pytest.mark.parametrize('flow', ['source_to_target', 'target_to_source'])
def test_golden_graph_filter_and_flow(dtype, loop, flow):
    x = torch.tensor([0., 2., 5., 9.], dtype=dtype)
    raw = golden.knn(x, x, 2 + int(not loop))
    expected = raw if loop else raw[:, raw[0] != raw[1]]
    if flow == 'source_to_target':
        expected = expected.flip(0)
    assert torch.equal(golden.knn_graph(x, 2, loop=loop, flow=flow), expected)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('k', [1, 64, 65, 100, 101, 128, 129, 257])
def test_golden_large_k_retains_exact_unique_neighbors(dtype, k):
    x, y = torch.arange(257, dtype=dtype), torch.tensor([-1.], dtype=dtype)
    actual = golden.knn(x, y, k)
    assert actual.shape == (2, k)
    assert actual[1].tolist() == list(range(k))


@pytest.mark.parametrize('kind', ['k_zero', 'k_too_large', 'features', 'dtype',
                                'batch_length', 'batch_order', 'batch_size', 'nonfinite'])
def test_golden_invalid_reference_inputs(kind):
    x, y, k = torch.zeros(3, 2), torch.zeros(2, 2), 2
    kwargs = {}
    if kind == 'k_zero':
        k = 0
    elif kind == 'k_too_large':
        k = 4
    elif kind == 'features':
        y = torch.zeros(2, 3)
    elif kind == 'dtype':
        y = y.half()
    elif kind == 'batch_length':
        kwargs['batch_x'] = torch.tensor([0, 0])
    elif kind == 'batch_order':
        kwargs['batch_x'] = torch.tensor([1, 0, 1])
    elif kind == 'batch_size':
        kwargs['batch_size'] = 0
    else:
        x[0, 0] = float('nan')
    with pytest.raises(ValueError):
        golden.knn(x, y, k, **kwargs)


def test_golden_invalid_graph_flow():
    with pytest.raises(ValueError, match='flow'):
        golden.knn_graph(torch.zeros(3, 2), 2, flow='invalid')


@pytest.mark.parametrize('name', ['knn', 'knn_graph'])
def test_golden_public_argument_record_preserves_keyword_calls(name):
    x = torch.tensor([0., 2., 5., 9.])
    if name == 'knn':
        actual = golden.knn(x=x, y=x, k=2, num_workers=4)
        expected = golden.knn(x, x, 2, None, None, False, 4)
    else:
        actual = golden.knn_graph(x=x, k=2, loop=True, flow='target_to_source', num_workers=4)
        expected = golden.knn_graph(x, 2, None, True, 'target_to_source', False, 4)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('name', ['knn', 'knn_graph'])
@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unknown', 'excess'])
def test_golden_public_argument_record_rejects_invalid_calls(name, kind):
    x = torch.tensor([0., 2., 5., 9.])
    positional = (x, x, 2) if name == 'knn' else (x, 2)
    keyword = {}
    if kind == 'missing':
        positional = positional[:-1]
    elif kind == 'duplicate':
        keyword['x'] = x
    elif kind == 'unknown':
        keyword['unsupported'] = True
    else:
        positional += (None,) * (9 - len(positional))
    with pytest.raises(TypeError):
        golden_functions[name](*positional, **keyword)

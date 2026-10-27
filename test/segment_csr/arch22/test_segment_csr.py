# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2 CSR API, numerical and deterministic-result tests.

Run with pytest; use --junitxml for reports. NPU cases require Ascend A2.
The standalone PyTorch reference and comparator self-checks also run on CPU.

Execute this file directly for the separate large-allocation mean check. Its
default allocates slightly over 8 GiB and scans over two billion rows on one
core. Use --rows, --channels, --dtype, --irregular and --generic to configure it.
"""

import argparse
from dataclasses import dataclass
import gc
import inspect
import json
import logging
import math
import os
from pathlib import Path
import runpy
import sys
import time

import pytest
import torch

_GOLDEN = runpy.run_path(Path(__file__).with_name("golden.py"))
cpu_reference = _GOLDEN["cpu_reference"]
precision_metrics = _GOLDEN["precision_metrics"]
precision_passed = _GOLDEN["precision_passed"]
assert_close = _GOLDEN["assert_close"]
DTYPES = _GOLDEN["DTYPES"]
REDUCES = _GOLDEN["REDUCES"]
FLOATS = (torch.float16, torch.float32, torch.bfloat16)
INTEGERS = (torch.int8, torch.uint8, torch.int32, torch.int64)
TEST_SHAPES = ((128,), (1024,), (4096,), (8192,), (32, 512), (64, 768), (8, 16, 64), (4, 128, 256))
GENERAL_SHAPES = ((1,), (4,), (2, 2), (1, 128), (512, 768), (512, 1024), (1024, 768), (1024, 1024))
SEED = 42


@dataclass
class RunOptions:
    out: object = None
    api: str = None
    layout: str = "contiguous"
    expected: object = None
    expected_arg: object = None
    device_inputs: object = None
    check_source: bool = True
    check_pointer: bool = True


def _run(npu_api, src, indptr, reduce="sum", options=None):
    options = options or RunOptions()
    out, api, layout = options.out, options.api, options.layout
    expected, expected_arg = options.expected, options.expected_arg
    device_inputs = options.device_inputs
    api = api or f"segment_{reduce}_csr"
    want_arg = api in ("segment_min_csr", "segment_max_csr")
    if expected is None:
        oracle = cpu_reference(src, indptr, out=out, reduce=reduce, return_arg=want_arg)
        expected, expected_arg = oracle if want_arg else (oracle, None)
    source, pointers, output = device_inputs or (
        _device_view(src, layout),
        _device_view(indptr, layout),
        None if out is None else _device_view(out, layout),
    )
    kwargs = {"out": output}
    if api == "segment_csr":
        kwargs["reduce"] = reduce
    result = getattr(npu_api, api)(source, pointers, **kwargs)
    assert isinstance(result, tuple) == want_arg
    actual, arguments = result if want_arg else (result, None)
    assert actual.device.type == "npu" and actual.device == source.device
    if output is not None:
        assert actual is output
    assert_close(actual, expected)
    if want_arg:
        assert arguments.device == source.device and expected_arg is not None
        assert_close(arguments, expected_arg)
    if options.check_pointer:
        assert torch.equal(pointers.cpu(), indptr)
    if options.check_source:
        torch.testing.assert_close(source.cpu(), src, rtol=0, atol=0, equal_nan=True)
    return result


@pytest.fixture(scope="module")
def npu_api():
    import torch_npu  # noqa: F401  # Register the NPU device.
    import ops_gnn

    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))
    assert torch.npu.get_device_name().startswith("Ascend910B"), "acceptance requires Ascend A2"
    torch.manual_seed(SEED)
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    return ops_gnn


def _source(shape, dtype, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=generator).to(dtype)
    high = 2 if dtype in (torch.int8, torch.uint8) else 128
    return torch.randint(0, high, shape, generator=generator).to(dtype)


def _indptr(length, segments):
    sizes = torch.full((segments,), length // segments, dtype=torch.int64)
    sizes[-1] += length - length // segments * segments
    return torch.cat((torch.zeros(1, dtype=torch.int64), sizes.cumsum(0)))


def _last_pointer(pointer, src):
    return pointer.view(*[1] * (src.ndim - 1), pointer.numel())


def _device_view(value, layout="contiguous"):
    return _device_storage(value, layout)[0]


def _device_storage(value, layout):
    if layout == "strided":
        shape = list(value.shape)
        shape[-1] = 2 * shape[-1] + 1
        storage = torch.full(shape, 23, dtype=value.dtype, device="npu")
        view = storage[..., 1::2]
        view.copy_(value.npu())
        if value.numel() > 1:
            assert not view.is_contiguous()
        assert view.storage_offset() > 0
        return view, storage
    return value.npu(), None


class SavedTensorVersion(torch.autograd.Function):
    """Use autograd's public saved-tensor check to observe version changes."""

    @staticmethod
    def forward(*args, **kwargs):
        if kwargs:
            raise TypeError("version probe accepts positional arguments only")
        ctx, tensor, token = args
        ctx.save_for_backward(tensor)
        return token.clone()

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Unpacking raises if a writer changed the saved tensor's version.
        assert len(ctx.saved_tensors) == 1
        return None, grad_outputs[0]


def _version_probe(tensor):
    if tensor.is_inference():
        return None
    token = torch.ones((), device="cpu", requires_grad=True)
    return SavedTensorVersion.apply(tensor, token)


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_original_shapes(npu_api, shape, reduce, dtype):
    src = _source(shape, dtype)
    pointer = _last_pointer(_indptr(shape[-1], max(2, shape[-1] // 32)), src)
    _run(npu_api, src, pointer, reduce)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_original_general_shapes(npu_api, shape, reduce, dtype):
    src = _source(shape, dtype)
    pointer = _last_pointer(_indptr(shape[-1], max(2, shape[-1] // 16)), src)
    _run(npu_api, src, pointer, reduce)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
def test_all_public_interfaces(npu_api, dtype, reduce):
    src = _source((31, 7), dtype)
    pointer = torch.tensor([0, 1, 1, 9, 17, 31])
    specialized = _run(npu_api, src, pointer, reduce)
    generic = _run(npu_api, src, pointer, reduce, options=RunOptions(api="segment_csr"))
    value = specialized[0] if isinstance(specialized, tuple) else specialized
    assert torch.equal(value.cpu(), generic.cpu())
    if reduce == "add":
        assert torch.equal(generic.cpu(), npu_api.segment_sum_csr(src.npu(), pointer.npu()).cpu())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("layout", ("contiguous", "strided"))
def test_out_overwrite_and_storage(npu_api, dtype, reduce, layout):
    src = _source((3, 29, 7), dtype)
    pointer = torch.tensor([[2, 2, 5, 19, 23]])
    out = torch.full((3, 4, 7), 11, dtype=dtype)
    source, pointers = (_device_view(t, layout) for t in (src, pointer))
    output, storage = _device_storage(out, layout)
    _run(
        npu_api,
        src,
        pointer,
        reduce,
        options=RunOptions(out=out, device_inputs=(source, pointers, output)),
    )
    if layout == "strided":
        assert bool((storage[..., ::2].cpu() == 23).all()), "strided out clobbered neighbors"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize(
    "shape,pointer", [((0,), [0]), ((0,), [0, 0, 0]), ((8,), [3, 3, 3, 3]), ((8, 0), [0, 3, 8])]
)
def test_empty_inputs_and_segments(npu_api, dtype, reduce, shape, pointer):
    src = _source(shape, dtype)
    pointer = torch.tensor(pointer)
    _run(npu_api, src, pointer, reduce)
    out_shape = list(shape)
    out_shape[0] = pointer.numel() - 1
    out = torch.full(out_shape, 11, dtype=dtype)
    _run(npu_api, src, pointer, reduce, options=RunOptions(out=out))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
def test_zero_segments_nonempty_source(npu_api, dtype, reduce):
    src, pointer = (_source((13, 7), dtype), torch.tensor([5]))
    expected = _GOLDEN["zero_segment_task_reference"](src, pointer)
    _run(
        npu_api,
        src,
        pointer,
        reduce,
        options=RunOptions(
            expected=expected, expected_arg=torch.empty(expected.shape, dtype=torch.int64)
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("shared", (False, True))
def test_batch_broadcast_and_feature_tails(npu_api, dtype, reduce, shared):
    src = _source((2, 3, 19, 2, 17), dtype)
    pointer = torch.tensor([[[1, 1, 3, 10, 18]], [[0, 2, 9, 9, 19]]])
    if shared:
        pointer = pointer[:1]
    _run(npu_api, src, pointer, reduce, options=RunOptions(layout="strided"))


@pytest.mark.parametrize("rank", range(1, 9))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
def test_rank_coverage(npu_api, rank, dtype, reduce):
    shape = (2,) * (rank - 1) + (7,)
    src = _source(shape, dtype)
    pointer = _last_pointer(torch.tensor([0, 1, 1, 6, 7]), src)
    _run(npu_api, src, pointer, reduce)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", ("min", "max"))
def test_first_extremum_and_original_coordinate(npu_api, dtype, reduce):
    src = torch.tensor([[9, 2], [3, 7], [3, 7], [8, 1], [8, 1], [4, 6]], dtype=dtype)
    pointer = torch.tensor([1, 1, 3, 5])
    _run(npu_api, src, pointer, reduce)


@pytest.mark.parametrize("dtype", INTEGERS)
@pytest.mark.parametrize("reduce", REDUCES)
def test_integer_boundaries_overflow_and_mean(npu_api, dtype, reduce):
    info = torch.iinfo(dtype)
    if dtype == torch.uint8:
        values = [info.max, 1, 2, 253, 17, 4, 9, 9, 3, 0]
    else:
        values = [info.max, 1, info.min, -1, -7, 2, -2, -3, 5, 5]
    src = torch.tensor(values, dtype=dtype).view(5, 2)
    pointer = torch.tensor([0, 2, 4, 5])
    _run(npu_api, src, pointer, reduce)


@pytest.mark.parametrize("reduce", REDUCES)
def test_int64_adjacent_above_float_precision(npu_api, reduce):
    src = torch.tensor(
        [
            2**53 + 1,
            2**53 + 3,
            -(2**53 + 7),
            -(2**53 + 5),
            2**24 + 1,
            2**24 + 3,
            2**62 + 7,
            2**62 + 9,
        ],
        dtype=torch.int64,
    )
    _run(npu_api, src, torch.tensor([0, 2, 4, 6, 8]), reduce)


@pytest.mark.parametrize("dtype", (torch.int8, torch.uint8))
@pytest.mark.parametrize("length", (127, 128, 129, 255, 257, 511))
@pytest.mark.parametrize("reduce", REDUCES)
def test_byte_accumulation_and_count_narrowing(npu_api, dtype, length, reduce):
    src = torch.arange(length).remainder(7).to(dtype).reshape(length, 1).expand(length, 3).clone()
    if dtype == torch.int8:
        src.sub_(3)
    _run(npu_api, src, torch.tensor([0, length]), reduce)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("reduce", ("sum", "add", "mean"))
@pytest.mark.parametrize("pattern", ("long", "cancellation", "rounding", "extreme"))
def test_floating_precision(npu_api, dtype, reduce, pattern):
    if pattern == "long":
        src = _source((65537, 3), dtype)
    elif pattern == "cancellation":
        scale = 2048.0 if dtype == torch.float16 else 2**20
        src = (
            torch.tensor([scale, 1.0, -scale, 3.0, -3.0, 0.125], dtype=dtype)
            .repeat(257)
            .reshape(-1, 1)
        )
    elif pattern == "rounding":
        src = torch.tensor(
            [1.0, 2 ** (-8), 1.0, 2 ** (-11), -1.0, 2 ** (-8), -1.0, 2 ** (-11)], dtype=dtype
        ).reshape(-1, 1)
    else:
        scale = torch.finfo(dtype).max / 8
        src = torch.tensor([scale, -scale, 1.0, scale, -scale, 2.0], dtype=dtype).reshape(-1, 1)
    pointer = torch.tensor([0, src.size(0)])
    _run(npu_api, src, pointer, reduce)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("reduce", ("sum", "mean", "min", "max"))
def test_special_float_values_defined_extrema(npu_api, dtype, reduce):
    src = torch.tensor(
        [float("nan"), 3.0, -2.0, float("inf"), 7.0, float("-inf"), 4.0, -0.0], dtype=dtype
    )
    _run(npu_api, src, torch.tensor([0, 3, 5, 8]), reduce)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
def test_pointer_mutation_alias_and_output_independence(npu_api, dtype, reduce):
    src = _source((64, 9), dtype)
    pointer = torch.tensor([0, 16, 32, 48, 64])
    source, pointers = (src.npu(), pointer.npu())
    first = _run(
        npu_api, src, pointer, reduce, options=RunOptions(device_inputs=(source, pointers, None))
    )
    before = (first[0] if isinstance(first, tuple) else first).cpu().clone()
    alias = pointers.view(-1)
    alias[1:4].copy_(torch.tensor([7, 7, 61], device="npu"))
    pointer[1:4] = torch.tensor([7, 7, 61])
    second = _run(
        npu_api, src, pointer, reduce, options=RunOptions(device_inputs=(source, pointers, None))
    )
    a = first[0] if isinstance(first, tuple) else first
    b = second[0] if isinstance(second, tuple) else second
    assert a.data_ptr() != b.data_ptr()
    assert torch.equal(before, a.cpu()), "new call modified an earlier output"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
def test_nondefault_stream_and_temporary_lifetime(npu_api, dtype, reduce):
    src = _source((1031, 11), dtype)
    pointer = torch.tensor([0, 2, 17, 513, 513, 1031])
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        result = getattr(npu_api, f"segment_{reduce}_csr")(src.npu().clone(), pointer.npu().clone())
        garbage = [torch.empty((1031, 11), dtype=dtype, device="npu") for _ in range(16)]
        del garbage
        gc.collect()
    torch.npu.current_stream().wait_stream(stream)
    expected = cpu_reference(src, pointer, reduce=reduce, return_arg=reduce in ("min", "max"))
    outputs = result if isinstance(result, tuple) else (result,)
    references = expected if isinstance(expected, tuple) else (expected,)
    for actual, oracle in zip(outputs, references):
        metrics, _ = precision_metrics(actual, oracle)
        passed = precision_passed(metrics)
        assert passed


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
def test_coo_value_agreement_where_contracts_match(npu_api, dtype, reduce):
    src = torch.arange(1, 13).reshape(6, 2).to(dtype)
    pointer, index = (torch.tensor([0, 2, 4, 6]), torch.tensor([0, 0, 1, 1, 2, 2]))
    csr = _run(npu_api, src, pointer, reduce, options=RunOptions(api="segment_csr"))
    coo = npu_api.segment_coo(src.npu(), index.npu(), reduce=reduce)
    assert torch.equal(csr.cpu(), coo.cpu())


@pytest.mark.parametrize("reduce", REDUCES)
def test_out_alias_source(npu_api, reduce):
    src = _source((8, 7), torch.float32)
    pointer = torch.tensor([0, 2, 4, 6, 8])
    source = src.npu()
    output = source[:4]
    _run(
        npu_api,
        src,
        pointer,
        reduce,
        options=RunOptions(
            out=src[:4].clone(), device_inputs=(source, pointer.npu(), output), check_source=False
        ),
    )


def test_exact_signature_and_negative_metadata(npu_api):
    for reduce in REDUCES:
        names = tuple(inspect.signature(getattr(npu_api, f"segment_{reduce}_csr")).parameters)
        assert names == ("src", "indptr", "out")
    assert tuple(inspect.signature(npu_api.segment_csr).parameters) == (
        "src",
        "indptr",
        "out",
        "reduce",
    )
    src, pointer = (torch.ones(8, device="npu"), torch.tensor([0, 3, 8], device="npu"))
    with pytest.raises(ValueError):
        npu_api.segment_csr(src, pointer, reduce="mul")
    with pytest.raises((TypeError, RuntimeError, ValueError)):
        npu_api.segment_csr(src, pointer, dim_size=2)
    for bad in (pointer.int(), torch.empty(0, dtype=torch.int64, device="npu"), pointer.view(1, 3)):
        with pytest.raises((RuntimeError, ValueError)):
            npu_api.segment_csr(src, bad)
    for out in (torch.zeros(3, device="npu"), torch.zeros(2, dtype=torch.int64, device="npu")):
        with pytest.raises((RuntimeError, ValueError)):
            npu_api.segment_csr(src, pointer, out=out)


@pytest.mark.parametrize(
    "pointer", ([0, 5, 4, 8], [-1, 3, 8], [0, 3, 9], [0, 2**32 + 8], [2**53 + 1, 2**53 + 2])
)
def test_invalid_pointer_values_rejected(npu_api, pointer):
    source = torch.ones(8, device="npu")
    with pytest.raises((RuntimeError, ValueError)):
        npu_api.segment_csr(source, torch.tensor(pointer, dtype=torch.int64, device="npu"))
    valid = npu_api.segment_csr(source, torch.tensor([0, 8], device="npu"))
    assert valid.cpu().item() == 8


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
def test_transposed_source_and_pointer(npu_api, dtype, reduce):
    source = _source((64, 128), dtype)
    pointer = torch.tensor([0, 8, 32, 64])
    device_source = source.t().contiguous().npu().t()
    assert not device_source.is_contiguous()
    _run(
        npu_api,
        source,
        pointer,
        reduce,
        options=RunOptions(device_inputs=(device_source, _device_view(pointer, "strided"), None)),
    )


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("reduce", ("sum", "mean"))
@pytest.mark.parametrize("geometry", ((16, 32), (16, 128), (32, 32), (32, 128)))
@pytest.mark.parametrize(
    "pattern", ("extreme_cancellation", "moderate_cancellation", "signed_random")
)
def test_uniform_fast_branch_adversarial(npu_api, dtype, reduce, geometry, pattern):
    rows, channels = geometry
    segments = 2053
    if pattern == "extreme_cancellation":
        scale = torch.finfo(dtype).max / 8
        group = torch.zeros((rows, channels), dtype=dtype)
        group[:6] = torch.tensor([scale, -scale, 1.0, scale, -scale, 2.0], dtype=dtype).view(6, 1)
        src = group.repeat(segments, 1)
    elif pattern == "moderate_cancellation":
        scale, tiny = (32768.0, 2 ** (-10)) if dtype == torch.float16 else (2**25, 1.0)
        group = torch.zeros((rows, channels), dtype=dtype)
        cycle = torch.tensor([scale, -scale, tiny, scale, -scale], dtype=dtype)
        repeated = cycle.repeat(rows // cycle.numel())
        group[: repeated.numel()] = repeated.view(-1, 1)
        src = group.repeat(segments, 1)
    else:
        generator = torch.Generator().manual_seed(SEED)
        src = (torch.randn((segments * rows, channels), generator=generator) * 7).to(dtype)
    pointer = torch.arange(0, src.size(0) + 1, rows, dtype=torch.int64)
    _run(npu_api, src, pointer, reduce, options=RunOptions(api="segment_csr"))


@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("inference", (False, True))
def test_int64_uniform_fast_branch_source_mutation(npu_api, reduce, inference):
    shape, rows = ((4096, 32), 16)
    generator = torch.Generator().manual_seed(SEED)
    original = torch.randint(-100, 100, shape, generator=generator, dtype=torch.int64)
    widened = original.clone()
    widened[0, 0], widened[1, 0] = (2**53 + 1, 2**53 + 3)
    widened[2, 1], widened[3, 1] = (-(2**53 + 5), -(2**53 + 7))
    widened[-2, 2], widened[-1, 2] = (2**62 + 3, 2**62 + 5)
    pointer = torch.arange(0, shape[0] + 1, rows, dtype=torch.int64)
    with torch.inference_mode(inference):
        source, pointers = (original.npu(), pointer.npu())
        alias = source.view(-1)
        for _, cpu in (
            ("small", original),
            ("wide_adjacent", widened),
            ("small_again", original),
        ):
            alias.copy_(cpu.reshape(-1).npu())
            _run(
                npu_api,
                cpu,
                pointer,
                reduce,
                options=RunOptions(api="segment_csr", device_inputs=(source, pointers, None)),
            )


@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.bfloat16))
@pytest.mark.parametrize("reduce", ("sum", "mean"))
@pytest.mark.parametrize("inference", (False, True))
def test_uniform_float_source_mutation(npu_api, dtype, reduce, inference):
    segments, rows, channels = (2053, 16, 32)
    original = _source((segments * rows, channels), dtype)
    widened = torch.zeros_like(original).reshape(segments, rows, channels)
    scale = torch.finfo(dtype).max / 8
    widened[:, :6, :] = torch.tensor([scale, -scale, 1.0, scale, -scale, 2.0], dtype=dtype).view(
        1, 6, 1
    )
    widened = widened.reshape_as(original)
    pointer = torch.arange(0, original.size(0) + 1, rows, dtype=torch.int64)
    with torch.inference_mode(inference):
        source, pointers = (original.npu(), pointer.npu())
        alias = source.view(-1)
        for _, cpu in (("small", original), ("extreme", widened), ("small_again", original)):
            alias.copy_(cpu.reshape(-1).npu())
            _run(
                npu_api,
                cpu,
                pointer,
                reduce,
                options=RunOptions(device_inputs=(source, pointers, None)),
            )


@pytest.mark.parametrize("dtype", (torch.float32, torch.int64))
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("inference", (False, True))
def test_uniform_pointer_internal_mutation_invalidates_plan(npu_api, dtype, reduce, inference):
    segments, rows, channels = (2053, 16, 32)
    src = _source((segments * rows, channels), dtype)
    original = torch.arange(0, src.size(0) + 1, rows, dtype=torch.int64)
    changed = original.clone()
    changed[1] = changed[0]
    changed[segments // 2] += 3
    with torch.inference_mode(inference):
        source, pointers = (src.npu(), original.npu())
        alias = pointers.view(-1)
        outputs = []
        for _, pointer in (
            ("uniform", original),
            ("internal_change", changed),
            ("uniform_again", original),
        ):
            alias.copy_(pointer.npu())
            result = _run(
                npu_api,
                src,
                pointer,
                reduce,
                options=RunOptions(device_inputs=(source, pointers, None)),
            )
            outputs.append(result[0] if isinstance(result, tuple) else result)
        assert torch.equal(outputs[0].cpu(), outputs[2].cpu())
        assert outputs[0].data_ptr() != outputs[2].data_ptr()


@pytest.mark.parametrize("inference", (False, True))
@pytest.mark.parametrize("api", ("segment_csr", "segment_sum_csr"))
def test_out_alias_indptr_revalidates_pointer(npu_api, inference, api):
    src = torch.tensor([0, 0, 1, 0, 2, 2], dtype=torch.int64)
    initial = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
    with torch.inference_mode(inference):
        source, pointers = (src.npu(), initial.npu())
        output = pointers[:-1]
        for _ in range(3):
            pointer = pointers.cpu().clone()
            expected = cpu_reference(src, pointer, reduce="sum")
            result = _run(
                npu_api,
                src,
                pointer,
                "sum",
                options=RunOptions(
                    api=api,
                    out=pointer[:-1].clone(),
                    expected=expected,
                    device_inputs=(source, pointers, output),
                    check_pointer=False,
                ),
            )
            assert result is output
            assert int(pointers[-1].cpu()) == src.size(0)
            assert torch.equal(pointers[:-1].cpu(), expected)
            assert bool((pointers.cpu()[1:] >= pointers.cpu()[:-1]).all())


@pytest.mark.parametrize("dtype", DTYPES)
def test_internal_output_overlap_rejected(npu_api, dtype):
    source = torch.ones((8, 3), dtype=dtype, device="npu")
    pointer = torch.tensor([0, 4, 8], dtype=torch.int64, device="npu")
    output = torch.zeros((1, 3), dtype=dtype, device="npu").expand(2, 3)
    with pytest.raises((RuntimeError, ValueError)):
        npu_api.segment_sum_csr(source, pointer, out=output)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
@pytest.mark.parametrize("batched", (False, True))
def test_interleaved_nonoverlapping_out(npu_api, dtype, reduce, batched):
    shape = (2, 4, 3) if batched else (4, 3)
    src = torch.arange(1, math.prod(shape) + 1).reshape(shape).to(dtype)
    pointer = torch.tensor([[0, 2, 4]]) if batched else torch.tensor([0, 2, 4])
    out_shape = (2, 2, 3) if batched else (2, 3)
    strides = (8, 3, 2) if batched else (3, 2)
    storage = torch.full((16 if batched else 8,), 23, dtype=dtype, device="npu")
    output = storage.as_strided(out_shape, strides)
    output.fill_(11)
    _run(
        npu_api,
        src,
        pointer,
        reduce,
        options=RunOptions(
            out=torch.full(out_shape, 11, dtype=dtype),
            device_inputs=(src.npu(), pointer.npu(), output),
        ),
    )
    holes = [1, 6, 9, 14] if batched else [1, 6]
    assert torch.equal(storage.cpu()[holes], torch.full((len(holes),), 23, dtype=dtype))


@pytest.mark.parametrize("inference", (False, True))
def test_independent_storage_alias_out_invalidates_pointer_cache(npu_api, inference):
    pointer = torch.tensor([0, 2, 4], dtype=torch.int64)
    first_src = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    with torch.inference_mode(inference):
        pointers = pointer.npu()
        output = torch.empty(0, dtype=torch.int64, device="npu").set_(
            pointers.untyped_storage(), 0, (2,), (1,)
        )
        version_probe = _version_probe(pointers)
        _run(
            npu_api,
            first_src,
            pointer,
            "sum",
            options=RunOptions(
                out=pointer[:2].clone(),
                device_inputs=(first_src.npu(), pointers, output),
                check_pointer=False,
            ),
        )
        updated = torch.tensor([0, 1, 4], dtype=torch.int64)
        assert torch.equal(pointers.cpu(), updated)
        if version_probe is not None:
            version_probe.backward()
        src = torch.tensor([1, 10, 100, 1000], dtype=torch.int64)
        result = _run(
            npu_api,
            src,
            updated,
            "sum",
            options=RunOptions(device_inputs=(src.npu(), pointers, None)),
        )
        assert torch.equal(result.cpu(), torch.tensor([1, 1110]))


@pytest.mark.parametrize("inference", (False, True))
@pytest.mark.parametrize("reduce", ("sum", "mean", "min", "max"))
def test_independent_out_alias_invalidates_other_source_proof(npu_api, inference, reduce):
    shape, rows = ((4096, 32), 16)
    small = _source(shape, torch.int64)
    wide = small + (2**53 + 1)
    wide[1::2, 1] *= -1
    pointer = torch.arange(0, shape[0] + 1, rows, dtype=torch.int64)
    writer_pointer = torch.arange(shape[0] + 1, dtype=torch.int64)
    with torch.inference_mode(inference):
        source, pointers = (small.npu(), pointer.npu())
        _run(
            npu_api,
            small,
            pointer,
            reduce,
            options=RunOptions(api="segment_csr", device_inputs=(source, pointers, None)),
        )
        alias = torch.empty(0, dtype=torch.int64, device="npu").set_(
            source.untyped_storage(), 0, shape, (shape[1], 1)
        )
        version_probe = _version_probe(source)
        written = npu_api.segment_sum_csr(wide.npu(), writer_pointer.npu(), out=alias)
        assert written is alias
        assert torch.equal(source.cpu(), wide)
        if version_probe is not None:
            version_probe.backward()
        _run(
            npu_api,
            wide,
            pointer,
            reduce,
            options=RunOptions(api="segment_csr", device_inputs=(source, pointers, None)),
        )


class TestCpuReference:
    """Run with -k TestCpuReference without touching an NPU."""

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("rows", (2**31 - 1, 2**31, 2**31 + 1, 2**32 + 1))
    @staticmethod
    def test_float_mean_full_segment_count(dtype, rows):
        # Test finalization without allocating billions of source elements.
        # These nearby counts round to 2**31 or 2**32 at every float dtype.
        numerator = torch.tensor([8.0, -8.0], dtype=dtype)
        actual = _GOLDEN["_divide_mean"](numerator, rows)
        scale = 2.0 ** (-28 if rows < 2**32 else -29)
        expected = torch.tensor([scale, -scale], dtype=dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("shared_version", (False, True))
    @staticmethod
    def test_saved_tensor_detects_shared_version(shared_version):
        source = torch.tensor([0, 2, 4])
        probe = _version_probe(source)
        alias = (
            source.view(-1)
            if shared_version
            else torch.empty(0, dtype=source.dtype).set_(
                source.untyped_storage(), 0, source.shape, source.stride()
            )
        )
        alias.add_(1)
        if shared_version:
            with pytest.raises(RuntimeError, match="version"):
                probe.backward()
        else:
            probe.backward()

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @staticmethod
    def test_reference_out_partial_empty_and_first_tie(dtype, reduce):
        source = torch.tensor([8, 3, 3, 7], dtype=dtype)
        pointer = torch.tensor([1, 1, 3, 4])
        output = torch.full((3,), 11, dtype=dtype)
        result = cpu_reference(source, pointer, output, reduce, return_arg=reduce in ("min", "max"))
        values, args = result if isinstance(result, tuple) else (result, None)
        expected = [0, 6 if reduce in ("sum", "add") else 3, 7]
        assert torch.equal(values, torch.tensor(expected, dtype=dtype))
        if args is not None:
            assert torch.equal(args, torch.tensor([4, 1, 3]))
        assert torch.equal(
            output, torch.full_like(output, 11)
        ), "oracle must preserve caller's CPU seed"

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @staticmethod
    def test_reference_empty_source_preserves_out(dtype, reduce):
        actual = cpu_reference(
            torch.empty(0, dtype=dtype),
            torch.tensor([0, 0, 0]),
            torch.full((2,), 11, dtype=dtype),
            reduce,
        )
        assert torch.equal(actual, torch.full((2,), 11, dtype=dtype))

    @staticmethod
    def test_reference_int64_never_rounds_through_float():
        source = torch.tensor([2**53 + 1, 2**53 + 3], dtype=torch.int64)
        assert cpu_reference(source, torch.tensor([0, 2]), reduce="mean").item() == 2**53 + 2
        assert (
            cpu_reference(torch.tensor([-7, 2]), torch.tensor([0, 2]), reduce="mean").item() == -2
        )

    @staticmethod
    def test_reference_half_mean_narrows_before_division():
        source = torch.tensor([65504.0, 65504.0], dtype=torch.float16)
        assert torch.isposinf(cpu_reference(source, torch.tensor([0, 2]), reduce="mean")).item()

    @pytest.mark.parametrize("dtype", (torch.int8, torch.uint8))
    @pytest.mark.parametrize("count", (127, 128, 129, 255, 257, 511))
    @staticmethod
    def test_reference_byte_wrapped_sum_and_count(dtype, count):
        source = torch.arange(count).remainder(7).to(dtype)
        if dtype == torch.int8:
            source.sub_(3)
        numerator = sum(source.tolist()) % 256
        denominator = count % 256
        if dtype == torch.int8:
            numerator = numerator - 256 if numerator >= 128 else numerator
            denominator = denominator - 256 if denominator >= 128 else denominator
        sign = -1 if (numerator < 0) != (denominator < 0) else 1
        expected = sign * (abs(numerator) // abs(denominator))
        actual = cpu_reference(source, torch.tensor([0, count]), reduce="mean")
        assert actual.item() == expected

    @staticmethod
    def test_guard_undefined_native_cases():
        with pytest.raises(ValueError, match="zero divisor"):
            cpu_reference(torch.ones(256, dtype=torch.int8), torch.tensor([0, 256]), reduce="mean")
        with pytest.raises(ValueError, match="zero segments"):
            cpu_reference(torch.ones(8), torch.tensor([3]))

    @staticmethod
    def test_precision_hard_limit_and_special_values():
        actual, expected = (torch.zeros(10000), torch.zeros(10000))
        actual[7] = 0.02
        metrics, limits = precision_metrics(actual, expected)
        assert metrics["matched_ratio"] >= limits["required_matched_ratio"]
        assert not precision_passed(metrics), "99% matches cannot conceal the hard-limit failure"
        special = torch.tensor([float("inf"), float("-inf"), float("nan")])
        assert precision_passed(precision_metrics(special, special)[0])
        assert not precision_passed(precision_metrics(special, torch.zeros(3))[0])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", REDUCES)
def test_repeat_bitwise_determinism(npu_api, dtype, reduce):
    src = _source((4099, 17), dtype)
    pointer = torch.tensor([0, 1, 2, 17, 1023, 1023, 2048, 4099])
    first = _run(npu_api, src, pointer, reduce)
    first = first if isinstance(first, tuple) else (first,)
    expected = [value.cpu().contiguous().view(torch.uint8) for value in first]
    source, pointers = src.npu(), pointer.npu()
    for _ in range(10):
        result = getattr(npu_api, f"segment_{reduce}_csr")(source, pointers)
        result = result if isinstance(result, tuple) else (result,)
        for before, after in zip(expected, result):
            assert torch.equal(before, after.cpu().contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", (torch.int8, torch.uint8))
@pytest.mark.parametrize("length", (256, 512, 1024))
@pytest.mark.parametrize("layout", ("contiguous", "strided"))
def test_byte_mean_zero_divisor_contract(npu_api, dtype, length, layout):
    # NPU explicitly returns zero; do not invoke the CPU's undefined division.
    src = torch.arange(length * 3).reshape(length, 3).remainder(7).to(dtype)
    if dtype == torch.int8:
        src.sub_(3)
    expected = torch.zeros((3, 3), dtype=dtype)
    pointer = torch.tensor([0, length, length, length])
    _run(
        npu_api,
        src,
        pointer,
        "mean",
        options=RunOptions(out=torch.full_like(expected, 11), layout=layout, expected=expected),
    )
    _run(
        npu_api,
        src,
        pointer,
        "mean",
        options=RunOptions(api="segment_csr", layout=layout, expected=expected),
    )


@pytest.mark.parametrize("inference", (False, True))
@pytest.mark.parametrize("alias_kind", ("data", "storage"))
@pytest.mark.parametrize("reduce", ("sum", "mean"))
def test_external_alias_rechecks_bounded_half_source(npu_api, inference, alias_kind, reduce):
    with torch.inference_mode(inference):
        src = torch.ones((2053 * 16, 32), dtype=torch.float16)
        pointer = torch.arange(2054, dtype=torch.int64) * 16
        source, pointers = src.npu(), pointer.npu()
        _run(
            npu_api,
            src,
            pointer,
            reduce,
            options=RunOptions(device_inputs=(source, pointers, None)),
        )
        alias = (
            source.data
            if alias_kind == "data"
            else torch.empty(0, dtype=source.dtype, device=source.device).set_(
                source.untyped_storage(), 0, source.shape, source.stride()
            )
        )
        for pattern in ([32768.0, 2**-9, -32768.0] + [0.0] * 13, [1.0] * 16):
            src.copy_(torch.tensor(pattern, dtype=src.dtype).repeat(2053).view(-1, 1))
            alias.copy_(src.npu())
            _run(
                npu_api,
                src,
                pointer,
                reduce,
                options=RunOptions(device_inputs=(source, pointers, None)),
            )


def _check_arg_carry(npu_api, source, reduce, channels):
    source = source.reshape(2, 3, 1).expand(2, 3, channels).clone()
    pointer = torch.tensor([[0, 1, 1, 3]])
    _run(npu_api, source, pointer, reduce)
    _run(npu_api, source, pointer, reduce, options=RunOptions(api="segment_csr"))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduce", ("min", "max"))
@pytest.mark.parametrize("channels", (1, 17, 64, 129))
def test_extreme_identity_arg_carry_across_batches(npu_api, dtype, reduce, channels):
    limits = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    identity = limits.max if reduce == "min" else limits.min
    source = torch.tensor([identity, 7, 1, identity, identity, identity], dtype=dtype)
    _check_arg_carry(npu_api, source, reduce, channels)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("reduce", ("min", "max"))
@pytest.mark.parametrize("channels", (1, 17, 64, 129))
def test_nan_only_segment_carries_arg(npu_api, dtype, reduce, channels):
    source = torch.tensor(
        [float("nan"), 7, 1, float("nan"), float("nan"), float("nan")], dtype=dtype
    )
    _check_arg_carry(npu_api, source, reduce, channels)


@pytest.mark.parametrize("dtype", FLOATS)
def test_mean_overflow_narrows_before_division(npu_api, dtype):
    source = torch.full((2, 17), torch.finfo(dtype).max, dtype=dtype)
    _run(npu_api, source, torch.tensor([0, 2]), "mean")


class TestPrecisionStandard:
    """Comparator checks for opbase mixed_tolerance_standard.md sections 2.1-2.3."""

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("kind", ("zero", "subnormal", "normal"))
    @staticmethod
    def test_ulp_spacing(dtype, kind):
        info = torch.finfo(dtype)
        value = {"zero": 0.0, "subnormal": info.tiny / 2, "normal": 1.0}[kind]
        expected = info.eps if kind == "normal" else info.tiny * info.eps
        spacing = _GOLDEN["_ulp_at_output_precision"](torch.tensor([value]).double(), dtype)
        assert spacing.item() == expected

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("above", (False, True))
    @staticmethod
    def test_element_tolerance_boundary(dtype, above):
        tolerance = {torch.float16: 2**-10, torch.bfloat16: 2**-7, torch.float32: 2**-13}[dtype]
        actual = torch.tensor([tolerance], dtype=dtype)
        if above:
            actual = torch.nextafter(actual, torch.full_like(actual, float("inf")))
        metrics, _ = precision_metrics(actual, torch.zeros_like(actual))
        assert metrics["matched_ratio"] == (0.0 if above else 1.0)

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("mismatches", (1, 2))
    @staticmethod
    def test_matched_ratio_boundary(dtype, mismatches):
        actual = torch.zeros(100, dtype=dtype)
        actual[:mismatches] = (
            2 * {torch.float16: 2**-10, torch.bfloat16: 2**-7, torch.float32: 2**-13}[dtype]
        )
        if mismatches == 1:
            assert_close(actual, torch.zeros_like(actual))
        else:
            with pytest.raises(AssertionError):
                assert_close(actual, torch.zeros_like(actual))

    @pytest.mark.parametrize("other_value", (0.0, 1024.0))
    @staticmethod
    def test_zero_worst_point_hard_limit(other_value):
        expected = torch.zeros(100, dtype=torch.bfloat16)
        expected[-1] = other_value
        actual = expected.clone()
        actual[0] = 0.1201171875
        metrics, _ = precision_metrics(actual, expected)
        assert metrics["matched_ratio"] == 0.99
        assert metrics["hard_limit"] == 0.1
        assert not metrics["hard_limit_pass"]
        with pytest.raises(AssertionError):
            assert_close(actual, expected)

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("above", (False, True))
    @staticmethod
    def test_ulp_hard_limit_boundary(dtype, above):
        expected = torch.full((100,), 4096.0, dtype=dtype)
        actual = expected.clone()
        actual[0] += 32 * (4096 * torch.finfo(dtype).eps)
        if above:
            actual[0] = torch.nextafter(actual[0], torch.tensor(float("inf"), dtype=dtype))
        metrics, _ = precision_metrics(actual, expected)
        assert metrics["matched_ratio"] >= 0.99
        assert metrics["hard_limit_pass"] != above

    @pytest.mark.parametrize("dtype", FLOATS)
    @pytest.mark.parametrize("sign", (-1, 1))
    @staticmethod
    def test_higher_precision_reference_overflow(dtype, sign):
        expected = torch.tensor([sign * torch.finfo(dtype).max * 2], dtype=torch.float64)
        actual = expected.to(dtype)
        assert_close(actual, expected)
        with pytest.raises(AssertionError):
            assert_close(-actual, expected)

    @pytest.mark.parametrize("dtype", FLOATS)
    @staticmethod
    def test_higher_precision_reference_keeps_finite_error(dtype):
        expected = torch.tensor([1.001], dtype=torch.float64)
        actual = expected.to(dtype)
        metrics, _ = precision_metrics(actual, expected)
        assert metrics["max_abs_error"] == abs(actual.double().item() - 1.001)
        assert_close(actual, expected)

    @pytest.mark.parametrize("dtype", FLOATS)
    @staticmethod
    def test_special_values_and_empty(dtype):
        expected = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=dtype)
        assert_close(expected.clone(), expected)
        with pytest.raises(AssertionError):
            assert_close(torch.zeros_like(expected), expected)
        assert_close(torch.empty(0, dtype=dtype), torch.empty(0, dtype=dtype))


def check(args):
    import torch_npu  # noqa: F401  # Register the NPU backend.
    import ops_gnn

    torch.npu.set_device(int(os.getenv("NPU_DEVICE_ID", "0")))
    dtype = getattr(torch, args.dtype)
    length = args.rows + int(args.irregular)
    source = torch.zeros((1, length, args.channels), dtype=dtype, device="npu")
    source[0, 0] = 8
    bounds = [0, args.rows, length] if args.irregular else [0, args.rows]
    pointer = torch.tensor([bounds], dtype=torch.int64, device=source.device)
    torch.npu.synchronize()
    logging.info(
        "%s",
        json.dumps({"phase": "allocated", "bytes": source.numel() * source.element_size()}),
    )
    started = time.monotonic()
    actual = (
        ops_gnn.segment_csr(source, pointer, reduce="mean")
        if args.generic
        else ops_gnn.segment_mean_csr(source, pointer)
    )
    actual = actual.cpu()
    elapsed = time.monotonic() - started
    # Independent closed-form reference, including the source-type divisor.
    divisor = torch.tensor(float(args.rows), dtype=torch.float32).to(dtype).item()
    expected = torch.zeros_like(actual)
    expected[0, 0] = 8.0 / divisor
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return dict(
        rows=args.rows,
        channels=args.channels,
        dtype=args.dtype,
        irregular=args.irregular,
        generic=args.generic,
        actual=actual.tolist(),
        expected=expected.tolist(),
        seconds=elapsed,
        status="PASS",
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2**31 + 1)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--irregular", action="store_true")
    parser.add_argument("--generic", action="store_true")
    args = parser.parse_args()
    if args.rows < 1 or args.channels < 1:
        parser.error("rows and channels must be positive")
    logging.info("%s", json.dumps(check(args)))


if __name__ == "__main__":
    main()

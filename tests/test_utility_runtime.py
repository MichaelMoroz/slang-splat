from __future__ import annotations

from contextlib import contextmanager

import slangpy as spy

from src.utility import (
    INDIRECT_BUFFER_USAGE,
    RW_BUFFER_USAGE,
    SHADER_ROOT,
    alloc_buffer,
    alloc_texture_2d,
    clear_debug_resource_allocations,
    defer_resource_release,
    drain_deferred_resource_releases,
    dispatch,
    dispatch_indirect,
    grow_capacity,
    load_compute_items,
)


def test_grow_capacity_scales_monotonically() -> None:
    assert grow_capacity(0, 0) == 1
    assert grow_capacity(1, 0) == 1
    assert grow_capacity(5, 2) == 5
    assert grow_capacity(6, 4) == 6
    assert grow_capacity(5, 10) == 15


def test_alloc_helpers_create_minimum_sized_resources(device: spy.Device) -> None:
    buffer = alloc_buffer(device, name="test.buffer", size=0, usage=RW_BUFFER_USAGE)
    texture = alloc_texture_2d(device, name="test.texture", format=spy.Format.rgba32_float, width=0, height=0, usage=spy.TextureUsage.shader_resource)

    assert buffer is not None
    assert texture is not None
    assert int(texture.width) == 1
    assert int(texture.height) == 1


def test_deferred_resource_release_waits_for_min_age() -> None:
    clear_debug_resource_allocations()
    resource = object()
    defer_resource_release(resource)

    assert drain_deferred_resource_releases(max_bytes=1_000_000, min_age=2) == (0, 0)
    assert drain_deferred_resource_releases(max_bytes=1_000_000, min_age=2) == (1, 0)
    clear_debug_resource_allocations()


def test_load_compute_items_loads_kernel_and_pipeline(device: spy.Device) -> None:
    items = load_compute_items(
        device,
        {
            "blur_horizontal": ("kernel", SHADER_ROOT / "utility" / "blur" / "separable_gaussian_blur.slang", "csGaussianBlurHorizontal"),
            "prefix_scan": ("pipeline", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csPrefixScanBlocks"),
        },
    )

    assert set(items) == {"blur_horizontal", "prefix_scan"}
    assert items["blur_horizontal"] is not None
    assert items["prefix_scan"] is not None


class _FakeKernel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def dispatch(self, *, thread_count, vars, command_encoder) -> None:
        self.calls.append({"thread_count": thread_count, "vars": vars, "command_encoder": command_encoder})


class _FakePipeline:
    pass


class _FakeCursor:
    pass


class _FakeComputePass:
    def __init__(self) -> None:
        self.bound_pipeline = None
        self.indirect_args = None

    def bind_pipeline(self, pipeline):
        self.bound_pipeline = pipeline
        return pipeline

    def dispatch_compute_indirect(self, args) -> None:
        self.indirect_args = args


class _FakeComputePassScope:
    def __init__(self, compute_pass: _FakeComputePass) -> None:
        self.compute_pass = compute_pass

    def __enter__(self) -> _FakeComputePass:
        return self.compute_pass

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeEncoder:
    def __init__(self) -> None:
        self.groups: list[tuple[str, object]] = []
        self.compute_pass = _FakeComputePass()

    def push_debug_group(self, label, color) -> None:
        self.groups.append(("push", label))

    def pop_debug_group(self) -> None:
        self.groups.append(("pop", None))

    def begin_compute_pass(self):
        return _FakeComputePassScope(self.compute_pass)


def test_dispatch_helpers_wrap_debug_groups(monkeypatch) -> None:
    monkeypatch.setattr(spy, "ShaderCursor", lambda _bound: _FakeCursor())
    monkeypatch.setattr(spy, "BufferOffsetPair", lambda buffer, offset: (buffer, offset))

    kernel = _FakeKernel()
    pipeline = _FakePipeline()
    encoder = _FakeEncoder()
    args_buffer = object()

    dispatch(
        kernel=kernel,
        thread_count=spy.uint3(3, 1, 1),
        vars={"g_Test": 1},
        command_encoder=encoder,
        debug_label="Kernel Dispatch",
        debug_color_index=1,
    )
    dispatch_indirect(
        pipeline=pipeline,
        args_buffer=args_buffer,
        vars={"g_Value": 5},
        command_encoder=encoder,
        arg_offset=3,
        debug_label="Pipeline Dispatch",
        debug_color_index=2,
    )

    assert len(kernel.calls) == 1
    assert encoder.groups == [
        ("push", "Kernel Dispatch"),
        ("pop", None),
        ("push", "Pipeline Dispatch"),
        ("pop", None),
    ]
    assert encoder.compute_pass.bound_pipeline is pipeline
    assert encoder.compute_pass.indirect_args == (args_buffer, 12)


def test_dispatch_requires_debug_color_when_labeled() -> None:
    kernel = _FakeKernel()
    with _raises(ValueError):
        dispatch(
            kernel=kernel,
            thread_count=spy.uint3(1, 1, 1),
            vars={},
            command_encoder=_FakeEncoder(),
            debug_label="Missing Color",
        )


@contextmanager
def _raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} to be raised.")

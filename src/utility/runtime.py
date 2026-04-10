from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

import slangpy as spy
from slangpy import math as smath


def thread_count_1d(count: int) -> spy.uint3:
    return spy.uint3(int(count), 1, 1)


def thread_count_2d(width: int, height: int, depth: int = 1) -> spy.uint3:
    return spy.uint3(int(width), int(height), int(depth))


def debug_color(index: int, weight: float = 0.05) -> spy.float3:
    phase = float(int(index)) * 1.7320508075688772
    return spy.float3(
        smath.lerp(1.0, smath.sin(phase), weight),
        smath.lerp(1.0, smath.sin(phase + 2.0943951023931953), weight),
        smath.lerp(1.0, smath.sin(phase + 4.1887902047863905), weight),
    )


@contextmanager
def debug_group(target: object, label: str, color: spy.float3):
    push = getattr(target, "push_debug_group", None)
    pop = getattr(target, "pop_debug_group", None)
    active = callable(push) and callable(pop)
    if active:
        push(str(label), color)
    try:
        yield target
    finally:
        if active:
            pop()


@contextmanager
def debug_region(target: object, label: str, color_index: int):
    with debug_group(target, str(label), debug_color(int(color_index))) as active_target:
        yield active_target


def dispatch(
    *,
    kernel: spy.ComputeKernel,
    thread_count: spy.uint3,
    vars: dict[str, Any],
    command_encoder: spy.CommandEncoder,
    debug_label: str | None = None,
    debug_color_value: spy.float3 | None = None,
    debug_color_index: int | None = None,
) -> None:
    if debug_label is not None:
        color = debug_color_value
        if color is None:
            if debug_color_index is None:
                raise ValueError("debug_color_value or debug_color_index is required when debug_label is provided.")
            color = debug_color(int(debug_color_index))
        with debug_group(command_encoder, debug_label, color):
            dispatch(
                kernel=kernel,
                thread_count=thread_count,
                vars=vars,
                command_encoder=command_encoder,
            )
        return
    kernel.dispatch(thread_count=thread_count, vars=vars, command_encoder=command_encoder)


def dispatch_indirect(
    *,
    pipeline: spy.ComputePipeline,
    args_buffer: spy.Buffer,
    vars: dict[str, Any],
    command_encoder: spy.CommandEncoder,
    arg_offset: int = 0,
    resource_binder: Callable[[Any], Any] | None = None,
    debug_label: str | None = None,
    debug_color_value: spy.float3 | None = None,
    debug_color_index: int | None = None,
) -> None:
    if debug_label is not None:
        color = debug_color_value
        if color is None:
            if debug_color_index is None:
                raise ValueError("debug_color_value or debug_color_index is required when debug_label is provided.")
            color = debug_color(int(debug_color_index))
        with debug_group(command_encoder, debug_label, color):
            dispatch_indirect(
                pipeline=pipeline,
                args_buffer=args_buffer,
                vars=vars,
                command_encoder=command_encoder,
                arg_offset=arg_offset,
                resource_binder=resource_binder,
            )
        return
    with command_encoder.begin_compute_pass() as compute_pass:
        cursor = spy.ShaderCursor(compute_pass.bind_pipeline(pipeline))
        for name, value in vars.items():
            setattr(cursor, name, resource_binder(value) if resource_binder is not None else value)
        compute_pass.dispatch_compute_indirect(spy.BufferOffsetPair(args_buffer, int(arg_offset) * 4))

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from inspect import Signature, signature
from typing import Any, Callable, Iterator, TypeVar

import slangpy as spy

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def debug_group(command_encoder: spy.CommandEncoder, label: str, color: spy.float3) -> Iterator[None]:
    command_encoder.push_debug_group(label, color)
    try:
        yield
    finally:
        command_encoder.pop_debug_group()


def _resolve_command_encoder(
    func_signature: Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    command_encoder_arg: str | None,
    command_encoder_index: int | None,
) -> Any:
    if command_encoder_arg is not None:
        bound = func_signature.bind_partial(*args, **kwargs)
        if command_encoder_arg in bound.arguments:
            return bound.arguments[command_encoder_arg]
    if command_encoder_index is not None and command_encoder_index < len(args):
        return args[command_encoder_index]
    raise TypeError("Unable to resolve command encoder for debug-group decorator.")


def with_debug_group(
    label: str | Callable[..., str],
    color: spy.float3,
    *,
    command_encoder_arg: str | None = "command_encoder",
    command_encoder_index: int | None = None,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        func_signature = signature(func)

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            command_encoder = _resolve_command_encoder(
                func_signature,
                args,
                kwargs,
                command_encoder_arg,
                command_encoder_index,
            )
            resolved_label = label(*args, **kwargs) if callable(label) else label
            with debug_group(command_encoder, resolved_label, color):
                return func(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator

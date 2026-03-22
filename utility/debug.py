from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import slangpy as spy


@contextmanager
def debug_group(command_encoder: spy.CommandEncoder, label: str, color: spy.float3) -> Iterator[None]:
    command_encoder.push_debug_group(label, color)
    try:
        yield
    finally:
        command_encoder.pop_debug_group()

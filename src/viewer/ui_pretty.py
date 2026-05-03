"""Compact, color-coded ImGui pretty printer for inline label=value structs.

Sections are sequences of `(header, fields)` where `header` is a string
(empty for no header) and `fields` is a sequence of `(label, value)` pairs.
Values may be scalars (number/bool/str/None) or short numeric vectors
(tuple/list/ndarray/spy.float* with .x/.y/[.z[.w]] attributes).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

from imgui_bundle import imgui

Field = tuple[str, Any]
Section = tuple[str, Sequence[Field]]

_COL_HEADER = (0.55, 0.85, 1.00, 1.00)
_COL_LABEL = (0.62, 0.78, 0.92, 1.00)
_COL_EQ = (0.45, 0.50, 0.58, 1.00)
_COL_PIPE = (0.40, 0.45, 0.52, 1.00)
_COL_NUM = (0.97, 0.85, 0.45, 1.00)
_COL_STR = (0.62, 0.90, 0.62, 1.00)
_COL_TRUE = (0.55, 0.92, 0.55, 1.00)
_COL_FALSE = (0.92, 0.55, 0.55, 1.00)
_COL_NONE = (0.85, 0.40, 0.40, 1.00)
_COL_PUNCT = (0.55, 0.60, 0.68, 1.00)

_INDENT_PX = 8.0


def _format_scalar(value: Any) -> tuple[str, tuple[float, float, float, float]]:
    if value is None:
        return "n/a", _COL_NONE
    if isinstance(value, bool):
        return ("true", _COL_TRUE) if value else ("false", _COL_FALSE)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}", _COL_NUM
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not math.isfinite(f):
            return "n/a", _COL_NONE
        af = abs(f)
        if af != 0.0 and (af < 1e-3 or af >= 1e6):
            return f"{f:.3e}", _COL_NUM
        return f"{f:.4g}", _COL_NUM
    if isinstance(value, str):
        return value, _COL_STR
    return str(value), _COL_STR


def _vector_components(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, (list, tuple)) and 2 <= len(value) <= 4:
        if all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) for x in value):
            return tuple(float(x) for x in value)
        return None
    if isinstance(value, np.ndarray) and value.ndim == 1 and 2 <= value.size <= 4:
        return tuple(float(x) for x in value)
    if hasattr(value, "x") and hasattr(value, "y") and not isinstance(value, str):
        comps = [float(value.x), float(value.y)]
        if hasattr(value, "z"):
            comps.append(float(value.z))
            if hasattr(value, "w"):
                comps.append(float(value.w))
        return tuple(comps)
    return None


def _value_tokens(value: Any) -> list[tuple[str, tuple[float, float, float, float]]]:
    vec = _vector_components(value)
    if vec is None:
        text, color = _format_scalar(value)
        return [(text, color)]
    tokens: list[tuple[str, tuple[float, float, float, float]]] = [("(", _COL_PUNCT)]
    for i, comp in enumerate(vec):
        if i > 0:
            tokens.append((", ", _COL_PUNCT))
        text, color = _format_scalar(float(comp))
        tokens.append((text, color))
    tokens.append((")", _COL_PUNCT))
    return tokens


def _pair_tokens(label: str, value: Any, *, leading_pipe: bool) -> list[tuple[str, tuple[float, float, float, float]]]:
    tokens: list[tuple[str, tuple[float, float, float, float]]] = []
    if leading_pipe:
        tokens.append((" | ", _COL_PIPE))
    tokens.append((label, _COL_LABEL))
    tokens.append(("=", _COL_EQ))
    tokens.extend(_value_tokens(value))
    return tokens


def _group_width(tokens: Iterable[tuple[str, tuple[float, float, float, float]]]) -> float:
    return sum(float(imgui.calc_text_size(text).x) for text, _ in tokens)


class _Flow:
    __slots__ = ("max", "indent", "x", "fresh")

    def __init__(self, max_width: float, indent: float):
        self.max = max(float(max_width), 1.0)
        self.indent = float(indent)
        self.x = 0.0
        self.fresh = True

    def emit_group(self, tokens: list[tuple[str, tuple[float, float, float, float]]]) -> None:
        width = _group_width(tokens)
        if not self.fresh and self.x + width > self.max:
            self.fresh = True
            self.x = 0.0
        if self.fresh:
            if self.indent > 0.0:
                imgui.dummy(imgui.ImVec2(self.indent, 0.0))
                imgui.same_line(0.0, 0.0)
            self.fresh = False
        else:
            imgui.same_line(0.0, 0.0)
        for i, (text, color) in enumerate(tokens):
            if i > 0:
                imgui.same_line(0.0, 0.0)
            imgui.text_colored(imgui.ImVec4(*color), text)
        self.x += width


def draw_struct_sections(sections: Sequence[Section], *, max_width: float | None = None) -> None:
    if max_width is None:
        avail = imgui.get_content_region_avail()
        width = max(float(avail.x), 80.0)
    else:
        width = max(float(max_width), 80.0)
    for header, fields in sections:
        if header:
            imgui.text_colored(imgui.ImVec4(*_COL_HEADER), header)
            indent = _INDENT_PX
        else:
            indent = 0.0
        if not fields:
            continue
        flow = _Flow(max_width=width - indent, indent=indent)
        for i, (label, value) in enumerate(fields):
            flow.emit_group(_pair_tokens(label, value, leading_pipe=(i > 0)))


def measure_struct_sections(sections: Sequence[Section], *, max_width: float) -> int:
    """Return the number of text lines `draw_struct_sections` would render."""
    width = max(float(max_width), 1.0)
    lines = 0
    for header, fields in sections:
        if header:
            lines += 1
            indent = _INDENT_PX
        else:
            indent = 0.0
        if not fields:
            continue
        body_width = max(width - indent, 1.0)
        x = 0.0
        first = True
        for i, (label, value) in enumerate(fields):
            group = _group_width(_pair_tokens(label, value, leading_pipe=(i > 0)))
            if first or x + group > body_width:
                lines += 1
                x = group
                first = False
            else:
                x += group
    return lines


def format_struct_sections_text(sections: Sequence[Section]) -> str:
    """Plain-text fallback (no colors, no wrapping). Useful for tests/logs."""
    out: list[str] = []
    for header, fields in sections:
        if header:
            out.append(header)
        if not fields:
            continue
        parts: list[str] = []
        for label, value in fields:
            vec = _vector_components(value)
            if vec is not None:
                inner = ", ".join(_format_scalar(float(c))[0] for c in vec)
                parts.append(f"{label}=({inner})")
            else:
                text, _ = _format_scalar(value)
                parts.append(f"{label}={text}")
        out.append(" | ".join(parts))
    return "\n".join(out)

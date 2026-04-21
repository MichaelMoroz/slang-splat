from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..utility import ResourceAllocation, resource_allocation

_MAX_WALK_DEPTH = 12
_MAX_WALK_NODES = 20000
_STATE_RESOURCE_ATTRS = (
    "viewport_texture",
    "loss_debug_texture",
    "debug_target_texture",
    "debug_present_texture",
    "debug_dssim_blur",
    "debug_dssim_moments",
    "debug_dssim_blurred_moments",
)
_SKIP_ATTRS = {
    "app",
    "callbacks",
    "ctx",
    "device",
    "frames",
    "loss_debug_view_options",
    "renderer",
    "toolkit",
    "ui",
}


@dataclass(frozen=True, slots=True)
class ResourceDebugRow:
    kind: str
    name: str
    owner: str
    byte_size: int
    details: str
    usage: str
    order: int


@dataclass(frozen=True, slots=True)
class ResourceDebugSnapshot:
    rows: tuple[ResourceDebugRow, ...]
    total_consumption: int
    buffer_count: int
    buffer_total: int
    buffer_mean: float
    buffer_median: float
    texture_count: int
    texture_total: int


def format_resource_bytes(byte_count: float | int) -> str:
    value = float(max(byte_count, 0.0))
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or suffix == "GiB":
            return f"{value:.0f} {suffix}" if suffix == "B" else f"{value:.2f} {suffix}"
        value /= 1024.0
    return f"{value:.2f} GiB"


def write_resource_debug_log(snapshot: ResourceDebugSnapshot, directory: Path | str | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    log_dir = root / "temp" / "resource_logs" if directory is None else Path(directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"resource_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(format_resource_debug_log(snapshot), encoding="utf-8")
    return path


def format_resource_debug_log(snapshot: ResourceDebugSnapshot) -> str:
    rows = tuple(sorted(snapshot.rows, key=lambda item: (-item.byte_size, item.order)))
    lines = [
        "Slang Splat Resource Debug Log",
        "",
        f"Total Consumption: {format_resource_bytes(snapshot.total_consumption)} ({snapshot.total_consumption:,} bytes)",
        (
            f"Buffers: {snapshot.buffer_count:,} | total={format_resource_bytes(snapshot.buffer_total)} ({snapshot.buffer_total:,} bytes) | "
            f"mean={format_resource_bytes(snapshot.buffer_mean)} | median={format_resource_bytes(snapshot.buffer_median)}"
        ),
        f"Textures: {snapshot.texture_count:,} | total={format_resource_bytes(snapshot.texture_total)} ({snapshot.texture_total:,} bytes)",
        "",
        "Duplicate-Looking Groups",
        "Count\tTotal Size\tSingle Size\tType\tDetails\tName",
    ]
    duplicate_lines = _duplicate_group_lines(rows)
    lines.extend(duplicate_lines if duplicate_lines else ("<none>",))
    lines.extend(
        (
            "",
            "Resource Table",
            "Size\tType\tDetails\tName\tOwner\tUsage",
        )
    )
    for row in rows:
        lines.append(
            "\t".join(
                (
                    f"{format_resource_bytes(row.byte_size)} ({row.byte_size:,} bytes)",
                    row.kind,
                    row.details,
                    row.name,
                    row.owner,
                    row.usage,
                )
            )
        )
    return "\n".join(lines) + "\n"


def _duplicate_group_lines(rows: tuple[ResourceDebugRow, ...]) -> tuple[str, ...]:
    groups: dict[tuple[str, str, int, str], list[ResourceDebugRow]] = defaultdict(list)
    for row in rows:
        groups[(row.kind, row.name, row.byte_size, row.details)].append(row)
    duplicate_groups = sorted(
        (group for group in groups.values() if len(group) > 1),
        key=lambda group: (-len(group) * group[0].byte_size, -len(group), group[0].name),
    )
    return tuple(
        "\t".join(
            (
                f"{len(group):,}",
                f"{format_resource_bytes(len(group) * group[0].byte_size)} ({len(group) * group[0].byte_size:,} bytes)",
                f"{format_resource_bytes(group[0].byte_size)} ({group[0].byte_size:,} bytes)",
                group[0].kind,
                group[0].details,
                group[0].name,
            )
        )
        for group in duplicate_groups
    )


def collect_resource_debug_snapshot(viewer: object) -> ResourceDebugSnapshot:
    found: dict[int, tuple[ResourceAllocation, list[str]]] = {}
    visited: set[int] = set()
    node_count = [0]
    for root_name, root in _viewer_resource_roots(viewer):
        _walk_resource_graph(root, root_name, found, visited, node_count, _MAX_WALK_DEPTH)
    rows = tuple(
        sorted(
            (
                ResourceDebugRow(
                    kind=allocation.kind,
                    name=allocation.name,
                    owner=min(paths, key=len),
                    byte_size=allocation.byte_size,
                    details=allocation.details,
                    usage=allocation.usage,
                    order=allocation.order,
                )
                for allocation, paths in found.values()
            ),
            key=lambda row: (-row.byte_size, row.order),
        )
    )
    return _snapshot_from_rows(rows)


def _viewer_resource_roots(viewer: object) -> tuple[tuple[str, object], ...]:
    state = getattr(viewer, "s", None)
    if state is None:
        return ()
    roots: list[tuple[str, object]] = []
    for attr, label in (
        ("renderer", "main_renderer"),
        ("training_renderer", "training_renderer"),
        ("debug_renderer", "debug_renderer"),
        ("trainer", "trainer"),
    ):
        value = getattr(state, attr, None)
        if value is not None:
            roots.append((f"viewer.{label}", value))
    state_resources = {attr: getattr(state, attr, None) for attr in _STATE_RESOURCE_ATTRS if getattr(state, attr, None) is not None}
    progress = getattr(state, "colmap_import_progress", None)
    if progress is not None:
        state_resources["colmap_import_textures"] = getattr(progress, "native_textures", ())
    if state_resources:
        roots.append(("viewer.state", SimpleNamespace(**state_resources)))
    return tuple(roots)


def _snapshot_from_rows(rows: tuple[ResourceDebugRow, ...]) -> ResourceDebugSnapshot:
    buffer_sizes = [row.byte_size for row in rows if row.kind == "Buffer"]
    texture_sizes = [row.byte_size for row in rows if row.kind == "Texture"]
    buffer_total = int(sum(buffer_sizes))
    texture_total = int(sum(texture_sizes))
    return ResourceDebugSnapshot(
        rows=rows,
        total_consumption=buffer_total + texture_total,
        buffer_count=len(buffer_sizes),
        buffer_total=buffer_total,
        buffer_mean=float(buffer_total / len(buffer_sizes)) if buffer_sizes else 0.0,
        buffer_median=float(median(buffer_sizes)) if buffer_sizes else 0.0,
        texture_count=len(texture_sizes),
        texture_total=texture_total,
    )


def _walk_resource_graph(value: object, path: str, found: dict[int, tuple[ResourceAllocation, list[str]]], visited: set[int], node_count: list[int], depth: int) -> None:
    allocation = resource_allocation(value)
    if allocation is not None:
        entry = found.setdefault(id(value), (allocation, []))
        entry[1].append(path)
        return
    if depth <= 0 or _is_terminal(value):
        return
    value_id = id(value)
    if value_id in visited:
        return
    visited.add(value_id)
    node_count[0] += 1
    if node_count[0] > _MAX_WALK_NODES:
        return
    for child_name, child in _iter_children(value):
        _walk_resource_graph(child, f"{path}.{child_name}", found, visited, node_count, depth - 1)


def _is_terminal(value: object) -> bool:
    return value is None or isinstance(value, (str, bytes, bytearray, bool, int, float, complex, np.ndarray))


def _iter_children(value: object) -> tuple[tuple[str, object], ...]:
    if isinstance(value, dict):
        return tuple((str(key), child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return tuple((f"[{idx}]", child) for idx, child in enumerate(value))
    if isinstance(value, (set, frozenset)):
        return tuple((f"[{idx}]", child) for idx, child in enumerate(value))
    attrs = _object_attrs(value)
    return tuple((name, child) for name, child in attrs.items() if name not in _SKIP_ATTRS)


def _object_attrs(value: object) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    attrs: dict[str, Any] = {}
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot.startswith("__"):
                continue
            try:
                attrs[slot] = getattr(value, slot)
            except AttributeError:
                pass
    return attrs

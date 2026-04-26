from __future__ import annotations

import importlib
from pathlib import Path

from imgui_bundle import imgui, imgui_md

from .constants import _WINDOW_TITLE

_DOC_MAX_WIDTH = 104
_SHORTCUTS_TEXT = "Controls: LMB drag look | WASDQE move | wheel speed"


def _read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _imgui_bundle_assets_path() -> Path:
    package = importlib.import_module("imgui_bundle")
    return Path(package.__file__).resolve().parent / "assets"


def _markdown_font_base_path() -> Path | None:
    path = _imgui_bundle_assets_path() / "fonts" / "Roboto" / "Roboto"
    return path if path.with_name(path.name + "-Regular.ttf").exists() else None


def _status_suffix(text: str) -> str:
    value = str(text).strip()
    return value.split(": ", 1)[-1] if ": " in value else value


def _draw_disabled_wrapped_text(text: str) -> None:
    value = _status_suffix(text)
    if not value:
        return
    imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x)
    imgui.begin_disabled()
    imgui.text_unformatted(value)
    imgui.end_disabled()
    imgui.pop_text_wrap_pos()


def _draw_markdown_text(text: str) -> None:
    value = str(text).strip()
    if not value:
        return
    try:
        imgui_md.render_unindented(value)
    except Exception:
        imgui.push_text_wrap_pos(_DOC_MAX_WIDTH * imgui.get_font_size() * 0.5)
        imgui.text_unformatted(value)
        imgui.pop_text_wrap_pos()


def _build_about_text() -> str:
    return "\n".join(
        (
            f"# {_WINDOW_TITLE}",
            "",
            "Single-window Gaussian splat viewer and trainer built on **Slangpy**.",
            "",
            "The scene is presented inside a docked viewport window with the **imgui-bundle** UI around it.",
            "",
            f"**Controls:** {_SHORTCUTS_TEXT.split(': ', 1)[-1]}",
        )
    )


def _build_documentation_text() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    parts = [
        "Viewer Documentation",
        "",
        _read_text_if_exists(repo_root / "doc" / "Viewer.md").strip(),
    ]
    text = "\n\n".join(part for part in parts if part)
    return text if text else "Documentation is unavailable."
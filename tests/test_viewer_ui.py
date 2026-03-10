from __future__ import annotations

from src.viewer import ui


def test_about_text_mentions_single_window_viewer() -> None:
    text = ui._build_about_text()

    assert "Slang Splat Viewer" in text
    assert "Single-window" in text
    assert "WASDQE" in text


def test_documentation_text_loads_local_viewer_doc() -> None:
    text = ui._build_documentation_text()

    assert "Viewer Documentation" in text
    assert "Frame Flow" in text
    assert "Input Routing" in text


def test_panel_rect_starts_below_menu_bar() -> None:
    x, y, w, h = ui._panel_rect(1600, 900, 24.0)

    assert x == 0.0
    assert y == 24.0
    assert w == 280.0
    assert h == 876.0

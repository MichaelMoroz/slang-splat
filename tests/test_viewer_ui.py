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

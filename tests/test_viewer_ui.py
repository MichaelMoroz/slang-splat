from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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


def test_build_ui_initializes_histogram_controls() -> None:
    renderer = SimpleNamespace(
        radius_scale=1.0,
        alpha_cutoff=1.0 / 255.0,
        max_splat_steps=32768,
        transmittance_threshold=0.005,
        sampled5_safety_scale=1.0,
        cached_raster_grad_atomic_mode="float",
        cached_raster_grad_fixed_scale=0.125,
        debug_show_ellipses=False,
        debug_show_processed_count=False,
        debug_show_grad_norm=False,
        debug_grad_norm_threshold=2e-4,
    )

    viewer_ui = ui.build_ui(renderer)

    assert viewer_ui._values["show_histograms"] is False
    assert viewer_ui._values["hist_auto_refresh"] is True
    assert viewer_ui._values["hist_bin_count"] == 64
    assert viewer_ui._values["hist_y_limit"] == 1.0
    assert viewer_ui._values["cached_raster_grad_fixed_scale"] == 0.125
    assert viewer_ui._values["_histogram_update_y_limit"] is True
    assert viewer_ui._values["_histogram_update_log_range"] is False


def test_histogram_log_range_from_ranges_uses_nonzero_finite_extrema() -> None:
    payload = SimpleNamespace(
        min_values=np.array([0.0, -1e-4, -1.0, np.nan], dtype=np.float32),
        max_values=np.array([0.0, 1e-2, 10.0, np.inf], dtype=np.float32),
    )

    lo, hi = ui._histogram_log_range_from_ranges(payload)

    assert np.isclose(lo, -2.0)
    assert np.isclose(hi, 1.0)

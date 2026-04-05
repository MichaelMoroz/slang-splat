from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.viewer import ui
from src.viewer.constants import _WINDOW_TITLE


def test_about_text_mentions_single_window_viewer() -> None:
    text = ui._build_about_text()

    assert _WINDOW_TITLE in text
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
        cached_raster_grad_atomic_mode="fixed",
        cached_raster_grad_fixed_ro_local_range=0.01,
        cached_raster_grad_fixed_scale_range=0.01,
        cached_raster_grad_fixed_quat_range=0.01,
        cached_raster_grad_fixed_color_range=0.2,
        cached_raster_grad_fixed_opacity_range=0.2,
        debug_show_ellipses=False,
        debug_show_processed_count=False,
        debug_show_grad_norm=False,
        debug_grad_norm_threshold=2e-4,
        debug_clone_count_range=(0.0, 16.0),
        debug_contribution_range=(0.001, 1.0),
    )

    viewer_ui = ui.build_ui(renderer)

    assert viewer_ui._values["show_histograms"] is False
    assert viewer_ui._values["hist_auto_refresh"] is True
    assert viewer_ui._values["hist_bin_count"] == 64
    assert viewer_ui._values["hist_y_limit"] == 1.0
    assert viewer_ui._values["cached_raster_grad_fixed_ro_local_range"] == 0.01
    assert viewer_ui._values["cached_raster_grad_fixed_color_range"] == 0.2
    assert viewer_ui._values["cached_raster_grad_atomic_mode"] == 1
    assert viewer_ui._values["render_background_mode"] == 1
    assert viewer_ui._values["render_background_color"] == (0.0, 0.0, 0.0)
    assert viewer_ui._values["debug_clone_count_min"] == 0.0
    assert viewer_ui._values["debug_clone_count_max"] == 16.0
    assert viewer_ui._values["debug_contribution_min"] == 0.001
    assert viewer_ui._values["debug_contribution_max"] == 1.0
    assert viewer_ui._values["lr_scale_mul"] == 5.0
    assert viewer_ui._values["lr_color_mul"] == 5.0
    assert viewer_ui._values["lr_opacity_mul"] == 5.0
    assert viewer_ui._values["lr_schedule_enabled"] is True
    assert viewer_ui._values["lr_schedule_start_lr"] == 1e-3
    assert viewer_ui._values["lr_schedule_end_lr"] == 1e-4
    assert viewer_ui._values["lr_schedule_steps"] == 30000
    assert viewer_ui._values["position_random_step_noise_lr"] == 5e5
    assert viewer_ui._values["position_random_step_opacity_gate_center"] == 0.005
    assert viewer_ui._values["position_random_step_opacity_gate_sharpness"] == 100.0
    assert viewer_ui._values["background_mode"] == 1
    assert viewer_ui._values["train_background_color"] == (1.0, 1.0, 1.0)
    assert viewer_ui._values["use_sh"] is True
    assert viewer_ui._values["sh1_reg"] == 0.01
    assert viewer_ui._values["maintenance_interval"] == 200
    assert viewer_ui._values["maintenance_growth_ratio"] == 0.02
    assert viewer_ui._values["maintenance_growth_start_step"] == 500
    assert viewer_ui._values["maintenance_alpha_cull_threshold"] == 1e-2
    assert viewer_ui._values["maintenance_contribution_cull_threshold"] == 0.001
    assert viewer_ui._values["density_regularizer"] == 0.05
    assert viewer_ui._values["max_allowed_density"] == 12.0
    assert viewer_ui._values["max_anisotropy"] == 32.0
    assert viewer_ui._values["max_gaussians"] == 1000000
    assert viewer_ui._values["training_steps_per_frame"] == 3
    assert viewer_ui._values["colmap_init_mode"] == 1
    assert viewer_ui._values["colmap_image_downscale_mode"] == 0
    assert viewer_ui._values["colmap_image_target_width"] == 2048
    assert viewer_ui._values["colmap_image_scale"] == 1.0
    assert viewer_ui._values["colmap_nn_radius_scale_coef"] == 0.5
    assert viewer_ui._values["_histogram_update_y_limit"] is True
    assert viewer_ui._values["_histogram_update_log_range"] is False


def test_optimizer_regularization_tab_includes_density_controls() -> None:
    assert "sh1_reg" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "density_regularizer" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "max_allowed_density" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "position_random_step_noise_lr" in ui._OPTIMIZER_TAB_KEYS["Learning Rates"]
    assert "position_random_step_opacity_gate_center" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "position_random_step_opacity_gate_sharpness" in ui._OPTIMIZER_TAB_KEYS["Regularization"]


def test_debug_mode_labels_include_contribution_amount() -> None:
    assert "contribution_amount" in ui._DEBUG_MODE_VALUES
    assert "Contribution Amount" in ui._DEBUG_MODE_LABELS


def test_contribution_amount_debug_mode_exposes_no_extra_range_controls() -> None:
    assert ui._renderer_debug_control_keys("contribution_amount") == ("debug_mode", "debug_contribution_min", "debug_contribution_max")
    assert ui._renderer_debug_control_keys("processed_count") == ("debug_mode",)
    assert ui._renderer_debug_control_keys("splat_density") == ("debug_mode", "debug_density_min", "debug_density_max")


def test_contribution_amount_colorbar_ticks_use_log_scale() -> None:
    viewer_ui = SimpleNamespace(_values={"debug_contribution_min": 0.001, "debug_contribution_max": 1.0})

    lo = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), "contribution_amount", 0.0, viewer_ui))
    hi = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), "contribution_amount", 1.0, viewer_ui))

    assert np.isclose(lo, 0.001, rtol=0.0, atol=1e-9)
    assert np.isclose(hi, 1.0, rtol=0.0, atol=1e-6)


def test_histogram_log_range_from_ranges_uses_nonzero_finite_extrema() -> None:
    payload = SimpleNamespace(
        min_values=np.array([0.0, -1e-4, -1.0, np.nan], dtype=np.float32),
        max_values=np.array([0.0, 1e-2, 10.0, np.inf], dtype=np.float32),
    )

    lo, hi = ui._histogram_log_range_from_ranges(payload)

    assert np.isclose(lo, -2.0)
    assert np.isclose(hi, 1.0)

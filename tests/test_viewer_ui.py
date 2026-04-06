from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import slangpy as spy

from src.viewer import ui
from src.viewer.constants import _WINDOW_TITLE


def _dummy_renderer() -> SimpleNamespace:
    return SimpleNamespace(
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


def test_about_text_mentions_single_window_viewer() -> None:
    text = ui._build_about_text()

    assert _WINDOW_TITLE in text
    assert "Single-window" in text
    assert "docked viewport" in text
    assert "WASDQE" in text


def test_documentation_text_loads_local_viewer_doc() -> None:
    text = ui._build_documentation_text()

    assert "Viewer Documentation" in text
    assert "Frame Flow" in text
    assert "Input Routing" in text


def test_panel_rect_starts_below_menu_bar() -> None:
    x, y, w, h = ui._panel_rect(1600, 900, 24.0)

    assert x == 1320.0
    assert y == 24.0
    assert w == 280.0
    assert h == 876.0


def test_clamp_viewport_size_rounds_and_clamps() -> None:
    assert ui._clamp_viewport_size(511.6, 287.4) == (512, 287)
    assert ui._clamp_viewport_size(0.1, 0.1) == (1, 1)


def test_rect_contains_matches_viewport_bounds() -> None:
    rect = (10.0, 20.0, 100.0, 50.0)

    assert ui._rect_contains(rect, (10.0, 20.0))
    assert ui._rect_contains(rect, (109.9, 69.9))
    assert not ui._rect_contains(rect, (110.0, 70.0))
    assert not ui._rect_contains(rect, None)


def test_keyboard_capture_passes_through_for_focused_viewport_without_active_ui_item() -> None:
    assert ui._should_capture_keyboard_for_ui(True, viewport_input_active=True, want_text_input=False) is False
    assert ui._should_capture_keyboard_for_ui(True, viewport_input_active=True, want_text_input=True) is True
    assert ui._should_capture_keyboard_for_ui(True, viewport_input_active=False, want_text_input=False) is True
    assert ui._should_capture_keyboard_for_ui(False, viewport_input_active=True, want_text_input=False) is False


def test_status_suffix_strips_presenter_prefix() -> None:
    assert ui._status_suffix("Train Res: 2473x1643 (N=1)") == "2473x1643 (N=1)"
    assert ui._status_suffix("Refinement: every 200 | growth=2.00%") == "every 200 | growth=2.00%"
    assert ui._status_suffix("Manual 1x") == "Manual 1x"


def test_build_ui_initializes_histogram_controls() -> None:
    viewer_ui = ui.build_ui(_dummy_renderer())

    assert viewer_ui._values["show_histograms"] is False
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
    assert viewer_ui._values["debug_depth_local_mismatch_min"] == 0.0
    assert viewer_ui._values["debug_depth_local_mismatch_max"] == 0.5
    assert viewer_ui._values["debug_depth_local_mismatch_smooth_radius"] == 2.0
    assert viewer_ui._values["debug_depth_local_mismatch_reject_radius"] == 4.0
    assert viewer_ui._values["loss_debug_view"] == 0
    assert viewer_ui._values["lr_scale_mul"] == 5.0
    assert viewer_ui._values["lr_color_mul"] == 5.0
    assert viewer_ui._values["lr_opacity_mul"] == 5.0
    assert viewer_ui._values["lr_schedule_enabled"] is True
    assert viewer_ui._values["lr_schedule_start_lr"] == 0.005
    assert viewer_ui._values["lr_schedule_end_lr"] == 1e-4
    assert viewer_ui._values["lr_schedule_steps"] == 30000
    assert viewer_ui._values["lr_schedule_stage1_step"] == 2000
    assert viewer_ui._values["lr_schedule_stage2_step"] == 5000
    assert viewer_ui._values["position_random_step_noise_lr"] == 5e5
    assert viewer_ui._values["position_random_step_noise_end_step"] == 30000
    assert viewer_ui._values["position_random_step_opacity_gate_center"] == 0.005
    assert viewer_ui._values["position_random_step_opacity_gate_sharpness"] == 100.0
    assert viewer_ui._values["background_mode"] == 1
    assert viewer_ui._values["train_background_color"] == (1.0, 1.0, 1.0)
    assert viewer_ui._values["use_sh"] is True
    assert viewer_ui._values["sh_start_step"] == 5000
    assert viewer_ui._values["sh1_reg"] == 0.01
    assert viewer_ui._values["refinement_interval"] == 200
    assert viewer_ui._values["refinement_growth_ratio"] == 0.075
    assert viewer_ui._values["refinement_growth_start_step"] == 500
    assert viewer_ui._values["refinement_alpha_cull_threshold"] == 1e-2
    assert viewer_ui._values["refinement_min_contribution_percent"] == 1e-05
    assert viewer_ui._values["refinement_min_contribution_decay"] == 0.995
    assert viewer_ui._values["density_regularizer"] == 0.02
    assert viewer_ui._values["depth_ratio_weight"] == 1.0
    assert viewer_ui._values["depth_ratio_schedule_step1"] == 1000
    assert viewer_ui._values["depth_ratio_schedule_step2"] == 2000
    assert viewer_ui._values["depth_ratio_schedule_step3"] == 5000
    assert viewer_ui._values["depth_ratio_grad_min"] == 0.0
    assert viewer_ui._values["depth_ratio_grad_max"] == 0.1
    assert viewer_ui._values["max_allowed_density"] == 12.0
    assert viewer_ui._values["max_anisotropy"] == 32.0
    assert viewer_ui._values["max_gaussians"] == 1000000
    assert viewer_ui._values["training_steps_per_frame"] == 3
    assert viewer_ui._values["colmap_init_mode"] == 1
    assert viewer_ui._values["colmap_image_downscale_mode"] == 1
    assert viewer_ui._values["colmap_image_max_size"] == 2048
    assert viewer_ui._values["colmap_image_scale"] == 1.0
    assert viewer_ui._values["colmap_nn_radius_scale_coef"] == 0.5
    assert viewer_ui._values["_histogram_update_y_limit"] is True
    assert viewer_ui._values["_histogram_update_log_range"] is False
    assert viewer_ui._values["_show_histograms_prev"] is False


def test_toolkit_window_render_draws_non_background_pixels(device) -> None:
    viewer_ui = ui.build_ui(_dummy_renderer())
    toolkit = ui.create_toolkit_window(device, 640, 360)
    surface = device.create_texture(
        format=spy.Format.rgba32_float,
        width=640,
        height=360,
        usage=spy.TextureUsage.render_target | spy.TextureUsage.copy_source,
    )
    viewport = device.create_texture(
        format=spy.Format.rgba32_float,
        width=320,
        height=180,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )
    viewport.copy_from_numpy(np.full((180, 320, 4), [0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    try:
        encoder = device.create_command_encoder()
        encoder.clear_texture_float(surface, clear_value=spy.float4(0.0, 0.0, 0.0, 1.0))
        toolkit.render(viewer_ui, surface, encoder, viewport_texture=viewport)
        device.submit_command_buffer(encoder.finish())
        device.wait()
        image = np.asarray(surface.to_numpy(), dtype=np.float32)
    finally:
        toolkit.shutdown()

    assert np.any(np.abs(image[..., :3]) > 1e-4)


def test_colmap_import_window_docks_into_toolkit_tab(monkeypatch) -> None:
    dock_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(ui.imgui, "set_next_window_dock_id", lambda dock_id, cond: dock_calls.append((int(dock_id), int(cond))))
    monkeypatch.setattr(ui.imgui, "set_next_window_pos", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "set_next_window_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "begin", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(ui.imgui, "end", lambda: None)
    toolkit = SimpleNamespace(_show_colmap_import=True, _menu_bar_height=24.0, _toolkit_dock_id=17, _dockspace_id=9, _dock_tool_window=lambda cond: ui.imgui.set_next_window_dock_id(17, cond))

    ui.ToolkitWindow._draw_colmap_import_window(toolkit, SimpleNamespace(_values={}, _texts={}))

    assert dock_calls[0] == (17, int(ui.imgui.Cond_.appearing.value))


def test_help_windows_dock_into_toolkit_tabs(monkeypatch) -> None:
    dock_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(ui.imgui, "set_next_window_dock_id", lambda dock_id, cond: dock_calls.append((int(dock_id), int(cond))))
    monkeypatch.setattr(ui.imgui, "set_next_window_pos", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "set_next_window_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "begin", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(ui.imgui, "end", lambda: None)
    toolkit = SimpleNamespace(
        _show_about=True,
        _show_docs=True,
        _menu_bar_height=24.0,
        _toolkit_dock_id=17,
        _about_text="about",
        _documentation_text="docs",
        _dock_tool_window=lambda cond: ui.imgui.set_next_window_dock_id(17, cond),
    )

    ui.ToolkitWindow._draw_about_window(toolkit)
    ui.ToolkitWindow._draw_documentation_window(toolkit)

    assert dock_calls == [(17, int(ui.imgui.Cond_.appearing.value)), (17, int(ui.imgui.Cond_.appearing.value))]


def test_histogram_window_docks_and_requests_refresh_on_open(monkeypatch) -> None:
    dock_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(ui.imgui, "set_next_window_dock_id", lambda dock_id, cond: dock_calls.append((int(dock_id), int(cond))))
    monkeypatch.setattr(ui.imgui, "set_next_window_pos", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "set_next_window_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "begin", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(ui.imgui, "end", lambda: None)
    toolkit = SimpleNamespace(_menu_bar_height=24.0, _toolkit_dock_id=17, _draw_histogram_controls=lambda ui_obj: None, _dock_tool_window=lambda cond: ui.imgui.set_next_window_dock_id(17, cond))
    viewer_ui = SimpleNamespace(_values={"show_histograms": True, "_show_histograms_prev": False}, _texts={})

    ui.ToolkitWindow._draw_histogram_window(toolkit, viewer_ui)

    assert dock_calls == [(17, int(ui.imgui.Cond_.appearing.value))]
    assert viewer_ui._values["_histograms_refresh_requested"] is True
    assert viewer_ui._values["_show_histograms_prev"] is True


def test_renderer_debug_window_docks_into_toolkit_tabs(monkeypatch) -> None:
    dock_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(ui.imgui, "set_next_window_dock_id", lambda dock_id, cond: dock_calls.append((int(dock_id), int(cond))))
    monkeypatch.setattr(ui.imgui, "set_next_window_pos", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "set_next_window_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "begin", lambda *args, **kwargs: (False, True))
    monkeypatch.setattr(ui.imgui, "end", lambda: None)
    toolkit = SimpleNamespace(_menu_bar_height=24.0, _toolkit_rect=(0.0, 24.0, 280.0, 876.0), _toolkit_dock_id=17, _dock_tool_window=lambda cond: ui.imgui.set_next_window_dock_id(17, cond))
    viewer_ui = SimpleNamespace(_values={"show_renderer_debug": True}, _texts={})

    ui.ToolkitWindow._draw_renderer_debug_window(toolkit, viewer_ui)

    assert dock_calls == [(17, int(ui.imgui.Cond_.appearing.value))]


def test_optimizer_regularization_tab_includes_density_controls() -> None:
    assert "sh1_reg" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "density_regularizer" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_weight" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_schedule_step1" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_schedule_step2" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_schedule_step3" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_grad_min" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_grad_max" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "max_allowed_density" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "position_random_step_noise_lr" in ui._OPTIMIZER_TAB_KEYS["Learning Rates"]
    assert "lr_schedule_stage1_step" in ui._OPTIMIZER_TAB_KEYS["Learning Rates"]
    assert "lr_schedule_stage2_step" in ui._OPTIMIZER_TAB_KEYS["Learning Rates"]
    assert "position_random_step_noise_end_step" in ui._OPTIMIZER_TAB_KEYS["Learning Rates"]
    assert "position_random_step_opacity_gate_center" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "position_random_step_opacity_gate_sharpness" in ui._OPTIMIZER_TAB_KEYS["Regularization"]


def test_schedule_step_slider_max_tracks_schedule_steps() -> None:
    assert ui._control_bound(SimpleNamespace(_values={"lr_schedule_steps": 1234}), SimpleNamespace(kwargs={"max_from": "lr_schedule_steps"}), "max_from", 0) == 1234


def test_debug_mode_labels_include_contribution_amount() -> None:
    assert "contribution_amount" in ui._DEBUG_MODE_VALUES
    assert "Contribution Amount" in ui._DEBUG_MODE_LABELS
    assert "depth_local_mismatch" in ui._DEBUG_MODE_VALUES
    assert "Depth Local Mismatch" in ui._DEBUG_MODE_LABELS


def test_contribution_amount_debug_mode_exposes_no_extra_range_controls() -> None:
    assert ui._renderer_debug_control_keys("contribution_amount") == ("debug_mode", "debug_contribution_min", "debug_contribution_max")
    assert ui._renderer_debug_control_keys("processed_count") == ("debug_mode",)
    assert ui._renderer_debug_control_keys("splat_density") == ("debug_mode", "debug_density_min", "debug_density_max")
    assert ui._renderer_debug_control_keys("depth_local_mismatch") == ("debug_mode", "debug_depth_local_mismatch_min", "debug_depth_local_mismatch_max", "debug_depth_local_mismatch_smooth_radius", "debug_depth_local_mismatch_reject_radius")


def test_contribution_amount_colorbar_ticks_use_log_scale() -> None:
    viewer_ui = SimpleNamespace(_values={"debug_contribution_min": 0.001, "debug_contribution_max": 1.0})

    lo = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), "contribution_amount", 0.0, viewer_ui))
    hi = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), "contribution_amount", 1.0, viewer_ui))

    assert np.isclose(lo, 0.001, rtol=0.0, atol=1e-9)
    assert np.isclose(hi, 1.0, rtol=0.0, atol=1e-6)


def test_depth_local_mismatch_colorbar_ticks_use_local_range() -> None:
    viewer_ui = SimpleNamespace(_values={"debug_depth_local_mismatch_min": 0.05, "debug_depth_local_mismatch_max": 0.35})

    lo = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), "depth_local_mismatch", 0.0, viewer_ui))
    hi = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), "depth_local_mismatch", 1.0, viewer_ui))

    assert np.isclose(lo, 0.05, rtol=0.0, atol=1e-9)
    assert np.isclose(hi, 0.35, rtol=0.0, atol=1e-9)


def test_histogram_log_range_from_ranges_uses_nonzero_finite_extrema() -> None:
    payload = SimpleNamespace(
        min_values=np.array([0.0, -1e-4, -1.0, np.nan], dtype=np.float32),
        max_values=np.array([0.0, 1e-2, 10.0, np.inf], dtype=np.float32),
    )

    lo, hi = ui._histogram_log_range_from_ranges(payload)

    assert np.isclose(lo, -2.0)
    assert np.isclose(hi, 1.0)

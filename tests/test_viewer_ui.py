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
        debug_adam_momentum_range=(1e-5, 0.1),
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

    assert x == 1300.0
    assert y == 24.0
    assert w == 300.0
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
    assert viewer_ui._values["show_training_views"] is False
    assert viewer_ui._values["show_camera_overlays"] is True
    assert viewer_ui._values["show_camera_labels"] is False
    assert viewer_ui._values["show_training_cameras"] is False
    assert viewer_ui._values["hist_bin_count"] == 64
    assert viewer_ui._values["hist_y_limit"] == 1.0
    assert viewer_ui._values["cached_raster_grad_fixed_ro_local_range"] == 0.01
    assert viewer_ui._values["cached_raster_grad_fixed_color_range"] == 0.2
    assert viewer_ui._values["cached_raster_grad_atomic_mode"] == 1
    assert viewer_ui._values["render_background_mode"] == 1
    assert viewer_ui._values["render_background_color"] == (0.0, 0.0, 0.0)
    assert viewer_ui._values["debug_mode"] == ui._DEBUG_MODE_VALUES.index("normal")
    assert viewer_ui._values["debug_clone_count_min"] == 0.0
    assert viewer_ui._values["debug_clone_count_max"] == 16.0
    assert viewer_ui._values["debug_contribution_min"] == 0.001
    assert viewer_ui._values["debug_contribution_max"] == 1.0
    assert viewer_ui._values["debug_adam_momentum_threshold"] == 1e-2
    assert viewer_ui._values["debug_sh_coeff_index"] == 0
    assert viewer_ui._values["debug_depth_local_mismatch_min"] == 0.0
    assert viewer_ui._values["debug_depth_local_mismatch_max"] == 0.5
    assert viewer_ui._values["debug_depth_local_mismatch_smooth_radius"] == 2.0
    assert viewer_ui._values["debug_depth_local_mismatch_reject_radius"] == 4.0
    assert viewer_ui._values["loss_debug_view"] == 0
    assert viewer_ui._values["lr_scale_mul"] == 20.0
    assert viewer_ui._values["lr_color_mul"] == 5.0
    assert viewer_ui._values["lr_opacity_mul"] == 5.0
    assert viewer_ui._values["lr_schedule_enabled"] is True
    assert viewer_ui._values["lr_schedule_start_lr"] == 0.005
    assert viewer_ui._values["lr_schedule_stage1_lr"] == 0.002
    assert viewer_ui._values["lr_schedule_stage2_lr"] == 0.001
    assert viewer_ui._values["lr_schedule_end_lr"] == 1.5e-4
    assert viewer_ui._values["lr_pos_mul"] == 1.0
    assert viewer_ui._values["lr_pos_stage1_mul"] == 0.75
    assert viewer_ui._values["lr_pos_stage2_mul"] == 0.2
    assert viewer_ui._values["lr_pos_stage3_mul"] == 0.2
    assert viewer_ui._values["lr_sh_mul"] == 0.05
    assert viewer_ui._values["lr_sh_stage1_mul"] == 0.05
    assert viewer_ui._values["lr_sh_stage2_mul"] == 0.05
    assert viewer_ui._values["lr_sh_stage3_mul"] == 0.05
    assert viewer_ui._values["lr_schedule_steps"] == 30000
    assert viewer_ui._values["lr_schedule_stage1_step"] == 3000
    assert viewer_ui._values["lr_schedule_stage2_step"] == 14000
    assert viewer_ui._values["position_random_step_noise_lr"] == 5e5
    assert np.isclose(viewer_ui._values["position_random_step_noise_stage1_lr"], 466666.6666666667)
    assert np.isclose(viewer_ui._values["position_random_step_noise_stage2_lr"], 416666.6666666667)
    assert viewer_ui._values["position_random_step_noise_stage3_lr"] == 0.0
    assert viewer_ui._values["position_random_step_opacity_gate_center"] == 0.005
    assert viewer_ui._values["position_random_step_opacity_gate_sharpness"] == 100.0
    assert viewer_ui._values["background_mode"] == 1
    assert viewer_ui._values["use_target_alpha_mask"] is False
    assert viewer_ui._values["train_background_color"] == (1.0, 1.0, 1.0)
    assert viewer_ui._values["sh_band"] == 0
    assert viewer_ui._values["sh_band_stage1"] == 1
    assert viewer_ui._values["sh_band_stage2"] == 1
    assert viewer_ui._values["sh_band_stage3"] == 1
    assert viewer_ui._values["sh1_reg"] == 0.01
    assert viewer_ui._values["refinement_interval"] == 200
    assert viewer_ui._values["refinement_growth_ratio"] == 0.05
    assert viewer_ui._values["refinement_growth_start_step"] == 500
    assert viewer_ui._values["refinement_alpha_cull_threshold"] == 1e-2
    assert viewer_ui._values["refinement_min_contribution_percent"] == 1e-05
    assert viewer_ui._values["refinement_min_contribution_decay"] == 0.995
    assert viewer_ui._values["refinement_opacity_mul"] == 1.0
    assert viewer_ui._values["refinement_loss_weight"] == 0.25
    assert viewer_ui._values["refinement_target_edge_weight"] == 0.75
    assert viewer_ui._values["density_regularizer"] == 0.02
    assert viewer_ui._values["depth_ratio_weight"] == 1.0
    assert viewer_ui._values["depth_ratio_stage1_weight"] == 0.05
    assert viewer_ui._values["depth_ratio_stage2_weight"] == 0.01
    assert viewer_ui._values["depth_ratio_stage3_weight"] == 0.001
    assert viewer_ui._values["depth_ratio_grad_min"] == 0.0
    assert viewer_ui._values["depth_ratio_grad_max"] == 0.1
    assert viewer_ui._values["max_allowed_density"] == 12.0
    assert viewer_ui._values["max_anisotropy"] == 32.0
    assert viewer_ui._values["max_gaussians"] == 1000000
    assert viewer_ui._values["training_steps_per_frame"] == 3
    assert viewer_ui._values["train_subsample_factor"] == 0
    assert viewer_ui._values["colmap_init_mode"] == 0
    assert viewer_ui._values["colmap_depth_root"] == ""
    assert viewer_ui._values["colmap_depth_value_mode"] == 1
    assert viewer_ui._values["colmap_image_downscale_mode"] == 0
    assert viewer_ui._values["colmap_image_max_size"] == 2048
    assert viewer_ui._values["colmap_image_scale"] == 1.0
    assert viewer_ui._values["colmap_nn_radius_scale_coef"] == 0.5
    assert viewer_ui._values["colmap_depth_point_count"] == 100000
    assert viewer_ui._values["_histogram_update_y_limit"] is True
    assert viewer_ui._values["_histogram_update_log_range"] is False
    assert viewer_ui._values["_show_histograms_prev"] is False
    assert viewer_ui._values["_training_views_rows"] == ()
    assert viewer_ui._values["_training_view_overlay_segments"] == ()
    assert viewer_ui._values["_viewport_sh_band"] == 0
    assert viewer_ui._values["_viewport_sh_control_key"] == "sh_band"
    assert viewer_ui._values["_viewport_sh_stage_label"] == "Stage 0"
    assert "show_renderer_debug" not in viewer_ui._values


def test_colmap_init_mode_labels_append_depth_only_for_valid_depth_root(tmp_path) -> None:
    viewer_ui = ui.build_ui(_dummy_renderer())
    viewer_ui._values["colmap_depth_root"] = ""
    assert ui._colmap_init_mode_labels(False) == ui._COLMAP_INIT_MODE_LABELS
    assert ui._colmap_init_mode_label(viewer_ui) == "COLMAP Pointcloud"

    viewer_ui._values["colmap_depth_root"] = str(tmp_path)
    viewer_ui._values["colmap_init_mode"] = 3

    assert ui._colmap_init_mode_labels(True) == ui._COLMAP_INIT_MODE_LABELS + ("From Depth",)
    assert ui._colmap_init_mode_label(viewer_ui) == "From Depth"


def test_colmap_depth_value_mode_labels_cover_distance_and_z_depth() -> None:
    assert ui._COLMAP_DEPTH_VALUE_MODE_LABELS == ("Depth Is Distance", "Depth Is Z-Depth")


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


def test_viewport_view_menu_left_aligns_view_mode_button(monkeypatch) -> None:
    button_labels: list[str] = []
    cursor_positions: list[tuple[float, float]] = []
    mode_text: list[str] = []
    same_line_calls: list[tuple[float, float]] = []
    pushed_colors: list[tuple[int, tuple[float, float, float, float]]] = []
    filled_rects: list[tuple[float, float, float, float, float]] = []

    class _DrawList:
        def add_rect_filled(self, p0, p1, _color, rounding):
            filled_rects.append((float(p0.x), float(p0.y), float(p1.x), float(p1.y), float(rounding)))

    monkeypatch.setattr(ui.imgui, "get_style", lambda: SimpleNamespace(frame_padding=ui.imgui.ImVec2(4.0, 3.0)))
    monkeypatch.setattr(ui.imgui, "calc_text_size", lambda text: ui.imgui.ImVec2(72.0 if text == "View Mode" else 84.0, 14.0))
    monkeypatch.setattr(ui.imgui, "push_id", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_id", lambda: None)
    monkeypatch.setattr(ui.imgui, "set_cursor_screen_pos", lambda pos: cursor_positions.append((float(pos.x), float(pos.y))))
    monkeypatch.setattr(ui.imgui, "small_button", lambda label: button_labels.append(label) or False)
    monkeypatch.setattr(ui.imgui, "same_line", lambda offset=0.0, spacing=-1.0: same_line_calls.append((float(offset), float(spacing))))
    monkeypatch.setattr(ui.imgui, "get_cursor_screen_pos", lambda: ui.imgui.ImVec2(157.0, 72.0))
    monkeypatch.setattr(ui.imgui, "get_window_draw_list", lambda: _DrawList())
    monkeypatch.setattr(ui.imgui, "push_style_color", lambda idx, color: pushed_colors.append((int(idx), (float(color.x), float(color.y), float(color.z), float(color.w)))))
    monkeypatch.setattr(ui.imgui, "pop_style_color", lambda count=1: None)
    monkeypatch.setattr(ui.imgui, "set_next_item_width", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "push_style_var", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_style_var", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "text_unformatted", lambda text: mode_text.append(text))
    monkeypatch.setattr(ui.imgui, "begin_popup", lambda *_args: False)
    monkeypatch.setattr(ui.imgui, "begin_combo", lambda *_args: False)
    toolkit = SimpleNamespace(_viewport_content_rect=(50.0, 60.0, 400.0, 240.0), _interface_scale_factor=lambda _ui_obj: 1.5)
    viewer_ui = SimpleNamespace(_values={"debug_mode": ui._DEBUG_MODE_VALUES.index("depth_std"), "show_camera_overlays": True, "show_camera_labels": False, "show_training_cameras": False, "_viewport_sh_band": 0, "_viewport_sh_control_key": "sh_band", "sh_band": 0})

    origin = ui.ToolkitWindow._draw_viewport_view_menu(toolkit, viewer_ui, ui.imgui.ImVec2(50.0, 60.0))

    assert button_labels == ["View Mode", "Cameras On", "Labels Off", "Training Cameras Off"]
    assert cursor_positions == [(62.0, 72.0)]
    assert same_line_calls == [(0.0, 15.0), (0.0, 15.0), (0.0, 15.0), (0.0, 15.0), (0.0, 15.0)]
    assert filled_rects == [(148.0, 69.0, 250.0, 89.0, 6.0)]
    assert len(pushed_colors) == 1
    assert pushed_colors[0][0] == int(ui.imgui.Col_.text.value)
    np.testing.assert_allclose(np.array(pushed_colors[0][1]), np.array((0.985, 0.992, 1.0, 1.0)), rtol=0.0, atol=1e-6)
    assert mode_text == ["Depth Std"]
    assert np.isclose(origin.x, 62.0)
    assert origin.y > 72.0


def test_viewport_view_menu_toggles_active_sh_control(monkeypatch) -> None:
    button_labels: list[str] = []
    select_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(ui.imgui, "get_style", lambda: SimpleNamespace(frame_padding=ui.imgui.ImVec2(4.0, 3.0)))
    monkeypatch.setattr(ui.imgui, "calc_text_size", lambda text: ui.imgui.ImVec2(72.0 if text == "View Mode" else 84.0, 14.0))
    monkeypatch.setattr(ui.imgui, "push_id", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_id", lambda: None)
    monkeypatch.setattr(ui.imgui, "set_cursor_screen_pos", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "small_button", lambda label: button_labels.append(label) or False)
    monkeypatch.setattr(ui.imgui, "same_line", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "get_cursor_screen_pos", lambda: ui.imgui.ImVec2(157.0, 72.0))
    monkeypatch.setattr(ui.imgui, "push_style_color", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_style_color", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "set_next_item_width", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "push_style_var", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_style_var", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "text_unformatted", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "begin_popup", lambda *_args: False)
    monkeypatch.setattr(ui.imgui, "begin_combo", lambda *_args: True)
    monkeypatch.setattr(ui.imgui, "selectable", lambda label, selected=False: select_calls.append((str(label), bool(selected))) or (str(label) == "SH2",))
    monkeypatch.setattr(ui.imgui, "set_item_default_focus", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "end_combo", lambda: None)

    class _DrawList:
        def add_rect_filled(self, *_args):
            return None

    monkeypatch.setattr(ui.imgui, "get_window_draw_list", lambda: _DrawList())
    toolkit = SimpleNamespace(_viewport_content_rect=(50.0, 60.0, 400.0, 240.0), _interface_scale_factor=lambda _ui_obj: 1.0)
    viewer_ui = SimpleNamespace(
        _values={
            "debug_mode": ui._DEBUG_MODE_VALUES.index("normal"),
            "show_camera_overlays": True,
            "show_camera_labels": False,
            "show_training_cameras": False,
            "_viewport_sh_band": 0,
            "_viewport_sh_control_key": "sh_band_stage2",
            "sh_band": 0,
            "sh_band_stage2": 0,
        }
    )

    ui.ToolkitWindow._draw_viewport_view_menu(toolkit, viewer_ui, ui.imgui.ImVec2(50.0, 60.0))

    assert button_labels == ["View Mode", "Cameras On", "Labels Off", "Training Cameras Off"]
    assert select_calls == [("SH0", True), ("SH1", False), ("SH2", False), ("SH3", False)]
    assert viewer_ui._values["sh_band"] == 0
    assert viewer_ui._values["sh_band_stage2"] == 2
    assert viewer_ui._values["_viewport_sh_band"] == 2


def test_viewport_camera_overlays_draw_lines_when_enabled(monkeypatch) -> None:
    lines: list[tuple[float, float, float, float, float]] = []
    polylines: list[tuple[list[tuple[float, float]], int, float]] = []
    texts: list[tuple[float, float, float, str]] = []
    rects: list[tuple[float, float, float, float]] = []

    class _DrawList:
        def add_polyline(self, points, _color, _flags, thickness):
            polylines.append(([(float(p.x), float(p.y)) for p in points], int(_flags), float(thickness)))

        def add_line(self, p0, p1, _color, thickness):
            lines.append((float(p0.x), float(p0.y), float(p1.x), float(p1.y), float(thickness)))

        def add_rect_filled(self, p0, p1, _color, _rounding):
            rects.append((float(p0.x), float(p0.y), float(p1.x), float(p1.y)))

        def add_text(self, *args):
            if len(args) == 3:
                pos, _color, text = args
                texts.append((0.0, float(pos.x), float(pos.y), str(text)))
            else:
                _font, font_size, pos, _color, text = args[:5]
                texts.append((float(font_size), float(pos.x), float(pos.y), str(text)))

    monkeypatch.setattr(ui.imgui, "get_window_draw_list", lambda: _DrawList())
    monkeypatch.setattr(ui.imgui, "calc_text_size", lambda text: ui.imgui.ImVec2(9.0 * len(str(text)), 14.0))
    monkeypatch.setattr(ui.imgui, "get_font", lambda: object())
    monkeypatch.setattr(ui.imgui, "get_font_size", lambda: 20.0)
    toolkit = SimpleNamespace(_interface_scale_factor=lambda _ui_obj: 1.0)
    viewer_ui = SimpleNamespace(
        _values={
            "show_camera_overlays": True,
            "show_camera_labels": True,
            "_training_view_overlay_segments": (
                (
                    ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)),
                    ((9.0, 10.0), (11.0, 12.0), (13.0, 14.0), (15.0, 16.0)),
                    ((1.0, 2.0, 9.0, 10.0), (3.0, 4.0, 11.0, 12.0), (5.0, 6.0, 13.0, 14.0), (7.0, 8.0, 15.0, 16.0)),
                    (13.0, 14.0),
                    "frame.png | 32.50 dB",
                    (0.1, 0.2, 0.3, 0.4),
                    1.5,
                ),
            ),
        }
    )

    ui.ToolkitWindow._draw_viewport_camera_overlays(toolkit, viewer_ui, ui.imgui.ImVec2(10.0, 20.0))

    assert polylines == [
        ([(11.0, 22.0), (13.0, 24.0), (15.0, 26.0), (17.0, 28.0)], int(ui.imgui.ImDrawFlags_.closed.value), 1.5),
        ([(19.0, 30.0), (21.0, 32.0), (23.0, 34.0), (25.0, 36.0)], int(ui.imgui.ImDrawFlags_.closed.value), 1.5),
    ]
    assert lines == [
        (11.0, 22.0, 19.0, 30.0, 1.5),
        (13.0, 24.0, 21.0, 32.0, 1.5),
        (15.0, 26.0, 23.0, 34.0, 1.5),
        (17.0, 28.0, 25.0, 36.0, 1.5),
    ]
    assert len(rects) == 1
    assert texts == [(18.0, 29.0, 16.0, "frame.png | 32.50 dB")]


def test_viewport_camera_overlays_skip_lines_when_disabled(monkeypatch) -> None:
    lines: list[tuple[float, float, float, float, float]] = []
    polylines: list[tuple[list[tuple[float, float]], int, float]] = []
    texts: list[tuple[float, float, float, str]] = []

    class _DrawList:
        def add_polyline(self, points, _color, _flags, thickness):
            polylines.append(([(float(p.x), float(p.y)) for p in points], int(_flags), float(thickness)))

        def add_line(self, p0, p1, _color, thickness):
            lines.append((float(p0.x), float(p0.y), float(p1.x), float(p1.y), float(thickness)))

        def add_text(self, *args):
            if len(args) == 3:
                pos, _color, text = args
                texts.append((0.0, float(pos.x), float(pos.y), str(text)))
            else:
                _font, font_size, pos, _color, text = args[:5]
                texts.append((float(font_size), float(pos.x), float(pos.y), str(text)))

    monkeypatch.setattr(ui.imgui, "get_window_draw_list", lambda: _DrawList())
    monkeypatch.setattr(ui.imgui, "get_font", lambda: object())
    monkeypatch.setattr(ui.imgui, "get_font_size", lambda: 20.0)
    toolkit = SimpleNamespace(_interface_scale_factor=lambda _ui_obj: 1.0)
    viewer_ui = SimpleNamespace(
        _values={
            "show_camera_overlays": False,
            "_training_view_overlay_segments": (
                (
                    ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)),
                    ((9.0, 10.0), (11.0, 12.0), (13.0, 14.0), (15.0, 16.0)),
                    ((1.0, 2.0, 9.0, 10.0), (3.0, 4.0, 11.0, 12.0), (5.0, 6.0, 13.0, 14.0), (7.0, 8.0, 15.0, 16.0)),
                    (13.0, 14.0),
                    "frame.png | 32.50 dB",
                    (0.1, 0.2, 0.3, 0.4),
                    1.5,
                ),
            ),
        }
    )

    ui.ToolkitWindow._draw_viewport_camera_overlays(toolkit, viewer_ui, ui.imgui.ImVec2(10.0, 20.0))

    assert polylines == []
    assert lines == []
    assert texts == []


def test_training_views_window_docks_and_uses_imgui_table(monkeypatch) -> None:
    dock_calls: list[tuple[int, int]] = []
    table_columns: list[str] = []
    cells: list[str] = []
    table_calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(ui.imgui, "set_next_window_dock_id", lambda dock_id, cond: dock_calls.append((int(dock_id), int(cond))))
    monkeypatch.setattr(ui.imgui, "set_next_window_pos", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "set_next_window_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "begin", lambda *args, **kwargs: (True, True))
    monkeypatch.setattr(ui.imgui, "end", lambda: None)
    monkeypatch.setattr(ui.imgui, "begin_table", lambda name, columns, flags: table_calls.append((name, int(columns), int(flags))) or True)
    monkeypatch.setattr(ui.imgui, "table_setup_column", lambda name, *_args: table_columns.append(name))
    monkeypatch.setattr(ui.imgui, "table_headers_row", lambda: None)
    monkeypatch.setattr(ui.imgui, "table_next_row", lambda: None)
    monkeypatch.setattr(ui.imgui, "table_next_column", lambda: None)
    monkeypatch.setattr(ui.imgui, "text_unformatted", lambda text: cells.append(text))
    monkeypatch.setattr(ui.imgui, "end_table", lambda: None)
    toolkit = SimpleNamespace(_menu_bar_height=24.0, _toolkit_dock_id=17, _dock_tool_window=lambda cond: ui.imgui.set_next_window_dock_id(17, cond), _training_views_value_text=ui.ToolkitWindow._training_views_value_text)
    viewer_ui = SimpleNamespace(
        _values={
            "show_training_views": True,
            "_training_views_rows": (
                {
                    "image_name": "frame.png",
                    "resolution": "640x360",
                    "fx": 525.0,
                    "fy": 520.0,
                    "cx": 320.0,
                    "cy": 180.0,
                    "near": 0.1,
                    "far": 100.0,
                    "loss": 0.25,
                    "psnr": 32.5,
                },
            ),
        },
        _texts={},
    )

    ui.ToolkitWindow._draw_training_views_window(toolkit, viewer_ui)

    assert dock_calls == [(17, int(ui.imgui.Cond_.appearing.value))]
    assert table_calls and table_calls[0][0] == "##training_views" and table_calls[0][1] == 10
    assert table_columns == ["Image", "Res", "Fx", "Fy", "Cx", "Cy", "Near", "Far", "Loss", "PSNR"]
    assert cells == ["frame.png", "640x360", "525.000", "520.000", "320.000", "180.000", "0.10", "100.00", "0.2500", "32.50"]


def test_viewport_debug_overlay_draws_mode_specific_controls(monkeypatch) -> None:
    drawn: list[tuple[str, bool]] = []
    child_sizes: list[tuple[float, float]] = []
    monkeypatch.setattr(ui.imgui, "get_frame_height", lambda: 20.0)
    monkeypatch.setattr(ui.imgui, "get_style", lambda: SimpleNamespace(item_spacing=ui.imgui.ImVec2(8.0, 6.0)))
    monkeypatch.setattr(ui.imgui, "calc_text_size", lambda _text: ui.imgui.ImVec2(16.0, 14.0))
    monkeypatch.setattr(ui.imgui, "push_style_color", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_style_color", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "push_style_var", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_style_var", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "set_cursor_screen_pos", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "begin_child", lambda _name, size, *_args: child_sizes.append((float(size.x), float(size.y))) or True)
    monkeypatch.setattr(ui.imgui, "end_child", lambda: None)
    monkeypatch.setattr(ui.imgui, "push_item_width", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_item_width", lambda: None)
    toolkit = SimpleNamespace(
        _viewport_content_rect=(0.0, 0.0, 640.0, 360.0),
        _interface_scale_factor=lambda _ui_obj: 1.0,
        _draw_control=lambda _ui_obj, spec, compact=False: drawn.append((spec.key, compact)),
    )
    viewer_ui = SimpleNamespace(_values={"debug_mode": ui._DEBUG_MODE_VALUES.index("depth_local_mismatch")}, _texts={})

    ui.ToolkitWindow._draw_viewport_debug_overlay(toolkit, viewer_ui, ui.imgui.ImVec2(12.0, 34.0))

    assert child_sizes and child_sizes[0][0] >= 220.0
    assert child_sizes[0][1] >= 200.0
    assert drawn == [
        ("debug_depth_local_mismatch_min", True),
        ("debug_depth_local_mismatch_max", True),
        ("debug_depth_local_mismatch_smooth_radius", True),
        ("debug_depth_local_mismatch_reject_radius", True),
    ]


def test_viewport_debug_overlay_draws_training_camera_controls(monkeypatch) -> None:
    child_sizes: list[tuple[float, float]] = []
    combo_labels: list[tuple[str, str]] = []
    slider_calls: list[tuple[str, int, int, int]] = []
    disabled_text: list[str] = []
    monkeypatch.setattr(ui.imgui, "get_text_line_height_with_spacing", lambda: 18.0)
    monkeypatch.setattr(ui.imgui, "get_frame_height", lambda: 20.0)
    monkeypatch.setattr(ui.imgui, "get_style", lambda: SimpleNamespace(item_spacing=ui.imgui.ImVec2(8.0, 6.0)))
    monkeypatch.setattr(ui.imgui, "calc_text_size", lambda _text: ui.imgui.ImVec2(16.0, 14.0))
    monkeypatch.setattr(ui.imgui, "push_style_color", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_style_color", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "push_style_var", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_style_var", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "set_cursor_screen_pos", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "begin_child", lambda _name, size, *_args: child_sizes.append((float(size.x), float(size.y))) or True)
    monkeypatch.setattr(ui.imgui, "end_child", lambda: None)
    monkeypatch.setattr(ui.imgui, "push_item_width", lambda *_args: None)
    monkeypatch.setattr(ui.imgui, "pop_item_width", lambda: None)
    monkeypatch.setattr(ui.imgui, "begin_combo", lambda label, preview: combo_labels.append((label, preview)) or False)
    monkeypatch.setattr(ui.imgui, "slider_int", lambda label, value, lo, hi: slider_calls.append((label, int(value), int(lo), int(hi))) or (False, value))
    monkeypatch.setattr(ui.imgui, "text_disabled", lambda text: disabled_text.append(text))
    monkeypatch.setattr(ui.imgui, "separator", lambda: None)
    toolkit = SimpleNamespace(
        _viewport_content_rect=(0.0, 0.0, 640.0, 360.0),
        _interface_scale_factor=lambda _ui_obj: 1.0,
        _draw_control=lambda *_args, **_kwargs: None,
        _training_camera_debug_section_height=lambda ui_obj: ui.ToolkitWindow._training_camera_debug_section_height(toolkit, ui_obj),
        _draw_training_camera_debug_controls=lambda ui_obj: ui.ToolkitWindow._draw_training_camera_debug_controls(toolkit, ui_obj),
    )
    viewer_ui = SimpleNamespace(
        _values={"debug_mode": ui._DEBUG_MODE_VALUES.index("normal"), "show_training_cameras": True, "loss_debug_view": 0, "loss_debug_frame": 3, "_loss_debug_frame_max": 12},
        _texts={"loss_debug_view": "View: Rendered", "loss_debug_frame": "Frame[3]: frame.png"},
    )

    ui.ToolkitWindow._draw_viewport_debug_overlay(toolkit, viewer_ui, ui.imgui.ImVec2(12.0, 34.0))

    assert child_sizes and child_sizes[0][0] >= 220.0
    assert combo_labels == [("##training_camera_view", "Rendered")]
    assert slider_calls == [("##training_camera_frame", 3, 0, 12)]
    assert disabled_text == ["View: Rendered", "frame.png"]


def test_training_setup_section_draws_subsampling_control(monkeypatch) -> None:
    drawn: list[str] = []
    reset_calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(ui.imgui, "collapsing_header", lambda _label: True)
    monkeypatch.setattr(ui, "_draw_disabled_wrapped_text", lambda _text: None)
    monkeypatch.setattr(ui.imgui, "separator", lambda: None)
    toolkit = SimpleNamespace(
        _draw_control=lambda _ui_obj, spec, compact=False: drawn.append(spec.key),
        _ctx_reset=lambda name, _ui_obj, keys: reset_calls.append((name, tuple(keys))),
    )
    viewer_ui = SimpleNamespace(
        _values={"background_mode": 1, "train_downscale_mode": 1},
        _texts={"training_resolution": "", "training_downscale": "", "training_schedule": "", "training_refinement": ""},
    )

    ui.ToolkitWindow._section_training_setup(toolkit, viewer_ui)

    assert "train_subsample_factor" in drawn
    assert drawn.index("train_subsample_factor") == drawn.index("train_downscale_mode") + 1
    assert reset_calls == [("train_setup_ctx", tuple(spec.key for spec in ui.GROUP_SPECS["Train Setup"]))]


def test_draw_control_compact_uses_stacked_hidden_labels(monkeypatch) -> None:
    labels: list[str] = []
    widget_labels: list[str] = []
    monkeypatch.setattr(ui.imgui, "text_unformatted", lambda label: labels.append(label))
    monkeypatch.setattr(ui.imgui, "input_float", lambda label, value, *_args: widget_labels.append(label) or (False, value))
    monkeypatch.setattr(ui.imgui, "is_item_hovered", lambda: False)
    viewer_ui = SimpleNamespace(_values={"debug_depth_std_min": 0.0}, _texts={})
    spec = next(spec for spec in ui.DEBUG_RENDER_SPECS if spec.key == "debug_depth_std_min")

    ui.ToolkitWindow._draw_control(SimpleNamespace(_TOOLTIPS={}), viewer_ui, spec, compact=True)

    assert labels == ["Depth Std Min"]
    assert widget_labels == ["##debug_depth_std_min"]


def test_debug_colorbar_height_scales_with_interface_scale(monkeypatch) -> None:
    boxes: list[tuple[float, float]] = []

    class _DrawList:
        def add_rect_filled(self, p0, p1, *_args):
            boxes.append((float(p1.y - p0.y), float(p1.x - p0.x)))

        def add_text(self, *_args):
            return None

        def add_rect(self, *_args):
            return None

    monkeypatch.setattr(ui, "_debug_colorbar_mode", lambda _viewer_ui: "depth_std")
    monkeypatch.setattr(ui.imgui, "get_foreground_draw_list", lambda: _DrawList())
    toolkit = SimpleNamespace(
        _viewport_content_rect=(0.0, 0.0, 800.0, 600.0),
        _interface_scale_factor=lambda viewer_ui: float(viewer_ui._values["scale"]),
        _debug_colorbar_title=lambda _mode: "Depth Std",
        _draw_debug_colorbar_gradient=lambda *_args: None,
        _draw_debug_colorbar_ticks=lambda *_args: None,
    )

    ui.ToolkitWindow._draw_debug_colorbar(toolkit, SimpleNamespace(_values={"scale": 1.0}, _texts={}))
    ui.ToolkitWindow._draw_debug_colorbar(toolkit, SimpleNamespace(_values={"scale": 2.0}, _texts={}))

    assert len(boxes) == 2
    assert boxes[1][0] > boxes[0][0] * 1.5


def test_optimizer_regularization_tab_includes_density_controls() -> None:
    assert "sh1_reg" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "density_regularizer" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_grad_min" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_grad_max" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "max_allowed_density" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_weight" not in ui._OPTIMIZER_TAB_KEYS["Schedule"]
    assert "position_random_step_noise_lr" not in ui._OPTIMIZER_TAB_KEYS["Schedule"]
    assert "lr_schedule_start_lr" not in ui._OPTIMIZER_TAB_KEYS["Schedule"]
    assert "lr_pos_mul" not in ui._OPTIMIZER_TAB_KEYS["Schedule"]
    assert "position_random_step_opacity_gate_center" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "position_random_step_opacity_gate_sharpness" in ui._OPTIMIZER_TAB_KEYS["Regularization"]


def test_schedule_stage_specs_clone_same_group_shape() -> None:
    assert tuple(ui.SCHEDULE_STAGE_SPECS) == ("Stage 0", "Stage 1", "Stage 2", "Stage 3")
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 0"]["lr"] == "lr_schedule_start_lr"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 0"]["lr_pos_mul"] == "lr_pos_mul"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 0"]["lr_sh_mul"] == "lr_sh_mul"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 0"]["sh_band"] == "sh_band"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 1"]["lr"] == "lr_schedule_stage1_lr"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 1"]["lr_pos_mul"] == "lr_pos_stage1_mul"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 1"]["lr_sh_mul"] == "lr_sh_stage1_mul"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 2"]["depth_ratio_weight"] == "depth_ratio_stage2_weight"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 3"]["noise_lr"] == "position_random_step_noise_stage3_lr"
    assert tuple(spec.label for spec in ui.SCHEDULE_STAGE_SPECS["Stage 0"]) == ("LR Target", "LR Mul Position", "LR Mul SH", "Depth Ratio Reg", "Noise LR", "SH Band")
    assert tuple(spec.label for spec in ui.SCHEDULE_STAGE_SPECS["Stage 1"]) == ("End Step", "LR Target", "LR Mul Position", "LR Mul SH", "Depth Ratio Reg", "Noise LR", "SH Band")
    assert tuple(spec.label for spec in ui.SCHEDULE_STAGE_SPECS["Stage 2"]) == ("End Step", "LR Target", "LR Mul Position", "LR Mul SH", "Depth Ratio Reg", "Noise LR", "SH Band")
    assert tuple(spec.label for spec in ui.SCHEDULE_STAGE_SPECS["Stage 3"]) == ("End Step", "LR Target", "LR Mul Position", "LR Mul SH", "Depth Ratio Reg", "Noise LR", "SH Band")


def test_schedule_step_slider_max_tracks_schedule_steps() -> None:
    assert ui._control_bound(SimpleNamespace(_values={"lr_schedule_steps": 1234}), SimpleNamespace(kwargs={"max_from": "lr_schedule_steps"}), "max_from", 0) == 1234


def test_debug_mode_labels_include_contribution_amount() -> None:
    assert ui._DEBUG_MODE_VALUES[0] == "normal"
    assert ui._DEBUG_MODE_LABELS[0] == "Normal"
    assert "contribution_amount" in ui._DEBUG_MODE_VALUES
    assert "Contribution Amount" in ui._DEBUG_MODE_LABELS
    assert "adam_momentum" in ui._DEBUG_MODE_VALUES
    assert "Adam Momentum" in ui._DEBUG_MODE_LABELS
    assert "sh_view_dependent" in ui._DEBUG_MODE_VALUES
    assert "SH View-Dependent" in ui._DEBUG_MODE_LABELS
    assert "sh_coefficient" in ui._DEBUG_MODE_VALUES
    assert "SH Coefficient" in ui._DEBUG_MODE_LABELS
    assert "depth_local_mismatch" in ui._DEBUG_MODE_VALUES
    assert "Depth Local Mismatch" in ui._DEBUG_MODE_LABELS


def test_contribution_amount_debug_mode_exposes_no_extra_range_controls() -> None:
    viewer_ui = SimpleNamespace(_values={"debug_mode": ui._DEBUG_MODE_VALUES.index("normal")})

    assert ui._debug_colorbar_mode(viewer_ui) is None
    assert ui._renderer_debug_control_keys("contribution_amount") == ("debug_mode", "debug_contribution_min", "debug_contribution_max")
    assert ui._renderer_debug_control_keys("adam_momentum") == ("debug_mode", "debug_grad_norm_threshold")
    assert ui._renderer_debug_control_keys("sh_coefficient") == ("debug_mode", "debug_sh_coeff_index")
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


def test_adam_momentum_colorbar_ticks_use_grad_norm_log_band() -> None:
    viewer_ui = SimpleNamespace(_values={"debug_grad_norm_threshold": 2e-4})

    lo = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), "adam_momentum", 0.0, viewer_ui))
    hi = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), "adam_momentum", 1.0, viewer_ui))

    assert np.isclose(lo, 2e-7, rtol=0.0, atol=1e-12)
    assert np.isclose(hi, 2e-3, rtol=0.0, atol=1e-12)


def test_histogram_log_range_from_histogram_keeps_central_99_percent_of_counts() -> None:
    payload = SimpleNamespace(
        bin_centers_log10=np.array([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0], dtype=np.float64),
        counts=np.array([[1.0, 1.0, 120.0, 200.0, 120.0, 1.0]], dtype=np.float64),
    )

    lo, hi = ui._histogram_log_range_from_histogram(payload)

    assert np.isclose(lo, -4.0)
    assert np.isclose(hi, -2.0)


def test_histogram_log_range_from_ranges_uses_nonzero_finite_extrema_as_fallback() -> None:
    payload = SimpleNamespace(
        min_values=np.array([0.0, -1e-4, -1.0, np.nan], dtype=np.float32),
        max_values=np.array([0.0, 1e-2, 10.0, np.inf], dtype=np.float32),
    )

    lo, hi = ui._histogram_log_range_from_ranges(payload)

    assert np.isclose(lo, -2.0)
    assert np.isclose(hi, 1.0)

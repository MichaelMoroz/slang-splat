from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import slangpy as spy

from src.app.training_controls import SCHEDULE_STAGE_CONTROL_DEFS, SCHEDULE_STAGE_GROUPS, TRAIN_SETUP_CONTROL_DEFS, TRAIN_SETUP_PRIMARY_KEYS
from src.viewer import ui
from src.viewer.buffer_debug import ResourceDebugRow, ResourceDebugSnapshot
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
        cached_raster_grad_fixed_color_range=0.2,
        cached_raster_grad_fixed_opacity_range=0.2,
        debug_show_ellipses=False,
        debug_show_processed_count=False,
        debug_show_grad_norm=False,
        debug_grad_norm_threshold=2e-4,
        debug_splat_age_range=(0.0, 1.0),
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


def test_view_scale_options_extend_to_300_percent() -> None:
    assert ui._INTERFACE_SCALE_OPTIONS[-1] == ("300%", 3.0)


def test_view_panel_defaults_only_expose_interface_scale() -> None:
    assert tuple(ui.default_control_values("View")) == ("interface_scale",)


def test_build_ui_initializes_control_groups_and_internal_state() -> None:
    viewer_ui = ui.build_ui(_dummy_renderer())

    for key in (
        "interface_scale",
        "theme",
        "show_histograms",
        "show_training_views",
        "show_camera_overlays",
        "show_camera_labels",
        "show_camera_min_dist_spheres",
        "show_training_cameras",
        "hist_bin_count",
        "hist_y_limit",
        "render_background_mode",
        "render_background_color",
        "loss_debug_view",
        "refinement_sample_radius",
        "refinement_clone_scale_mul",
        "refinement_use_compact_split",
        "refinement_solve_opacity",
        "refinement_split_beta",
        "refinement_grad_variance_weight_exponent",
        "refinement_contribution_weight_exponent",
        "colmap_init_mode",
        "colmap_depth_root",
        "colmap_fibonacci_sphere_point_count",
        "colmap_fibonacci_sphere_radius",
        "colmap_selected_camera_ids",
        "debug_gaussian_scale_multiplier",
        "debug_min_opacity",
        "debug_opacity_multiplier",
        "debug_ellipse_scale_multiplier",
    ):
        assert key in viewer_ui._values

    assert 0 <= int(viewer_ui._values["debug_mode"]) < len(ui._DEBUG_MODE_VALUES)
    assert "refinement_loss_weight" not in viewer_ui._values
    assert "refinement_target_edge_weight" not in viewer_ui._values

    assert viewer_ui._values["_histogram_update_y_limit"] is True
    assert viewer_ui._values["_histogram_update_range"] is False
    assert viewer_ui._values["_show_histograms_prev"] is False
    assert viewer_ui._values["_training_views_rows"] == ()
    assert viewer_ui._values["_training_view_overlay_segments"] == ()
    assert viewer_ui._values["_viewport_sh_band"] == 3
    assert viewer_ui._values["_viewport_sh_control_key"] in {"sh_band", "sh_band_stage1", "sh_band_stage2", "sh_band_stage3"}
    assert viewer_ui._values["_viewport_sh_stage_label"] in ui.SCHEDULE_STAGE_SPECS
    assert viewer_ui._values["_colmap_camera_rows"] == ()
    assert "show_renderer_debug" not in viewer_ui._values


def test_build_ui_exposes_refinement_sample_radius_default() -> None:
    viewer_ui = ui.build_ui(_dummy_renderer())

    assert "refinement_sample_radius" in viewer_ui._values


def test_build_ui_exposes_refinement_clone_scale_mul_default() -> None:
    viewer_ui = ui.build_ui(_dummy_renderer())

    assert "refinement_clone_scale_mul" in viewer_ui._values


def test_export_repo_defaults_writes_cached_raster_grad_training_render_defaults() -> None:
    viewer_ui = ui.build_ui(_dummy_renderer())

    viewer_ui._values["cached_raster_grad_atomic_mode"] = 0
    viewer_ui._values["cached_raster_grad_fixed_ro_local_range"] = 3.0
    viewer_ui._values["cached_raster_grad_fixed_scale_range"] = 512.0
    viewer_ui._values["cached_raster_grad_fixed_color_range"] = 9.0
    viewer_ui._values["cached_raster_grad_fixed_opacity_range"] = 10.0

    exported = ui.export_repo_defaults_from_ui_values(viewer_ui._values)

    assert exported["renderer"]["cached_raster_grad_atomic_mode"] == "float"
    assert exported["renderer"]["cached_raster_grad_fixed_ro_local_range"] == 3.0
    assert exported["renderer"]["cached_raster_grad_fixed_scale_range"] == 512.0
    assert exported["renderer"]["cached_raster_grad_fixed_color_range"] == 9.0
    assert exported["renderer"]["cached_raster_grad_fixed_opacity_range"] == 10.0
    assert exported["cli"]["common_render"]["cached_raster_grad_atomic_mode"] == "float"
    assert exported["cli"]["common_render"]["cached_raster_grad_fixed_ro_local_range"] == 3.0
    assert exported["cli"]["common_render"]["cached_raster_grad_fixed_scale_range"] == 512.0
    assert exported["cli"]["common_render"]["cached_raster_grad_fixed_color_range"] == 9.0
    assert exported["cli"]["common_render"]["cached_raster_grad_fixed_opacity_range"] == 10.0


def test_train_schedule_exposes_sorting_order_dithering_controls() -> None:
    stage_controls = {stage: {control.key: control for control in controls} for stage, controls in SCHEDULE_STAGE_CONTROL_DEFS.items()}
    expected = {
        "Stage 0": ("sorting_order_dithering", 0.10000000149011612),
        "Stage 1": ("sorting_order_dithering_stage1", 0.05000000074505806),
        "Stage 2": ("sorting_order_dithering_stage2", 0.05),
        "Stage 3": ("sorting_order_dithering_stage3", 0.0),
    }

    assert "sorting_order_dithering" not in {control.key for control in TRAIN_SETUP_CONTROL_DEFS}
    assert "sorting_order_dithering" not in TRAIN_SETUP_PRIMARY_KEYS
    for stage, (key, value) in expected.items():
        control = stage_controls[stage][key]
        assert control.kind == "input_float"
        assert control.label == "Sort Dither"
        assert control.kwargs["value"] == value
        assert control.kwargs["step"] == 1e-3
        assert control.kwargs["step_fast"] == 1e-2
        assert control.build_args == (key,)
        assert SCHEDULE_STAGE_GROUPS[stage]["sort_dither"] == key


def test_colmap_init_mode_labels_append_depth_only_for_valid_depth_root(tmp_path) -> None:
    viewer_ui = ui.build_ui(_dummy_renderer())
    viewer_ui._values["colmap_depth_root"] = ""
    viewer_ui._values["colmap_init_mode"] = 0
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
    toolkit = SimpleNamespace(_show_colmap_import=True, _menu_bar_height=24.0, _toolkit_dock_id=17, _dockspace_id=9, _applied_interface_scale=1.0, _dock_tool_window=lambda cond: ui.imgui.set_next_window_dock_id(17, cond))

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
        _applied_interface_scale=1.0,
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
    toolkit = SimpleNamespace(_menu_bar_height=24.0, _toolkit_dock_id=17, _applied_interface_scale=1.0, _draw_histogram_controls=lambda ui_obj: None, _dock_tool_window=lambda cond: ui.imgui.set_next_window_dock_id(17, cond))
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
    viewer_ui = SimpleNamespace(_values={"debug_mode": ui._DEBUG_MODE_VALUES.index("depth_std"), "show_camera_overlays": True, "show_camera_labels": False, "show_camera_min_dist_spheres": True, "show_training_cameras": False, "_viewport_sh_band": 0, "_viewport_sh_control_key": "sh_band", "sh_band": 0})

    origin = ui.ToolkitWindow._draw_viewport_view_menu(toolkit, viewer_ui, ui.imgui.ImVec2(50.0, 60.0))

    assert button_labels == ["View Mode", "Cameras On", "Labels Off", "Min Dist On", "Training Cameras Off"]
    assert cursor_positions == [(62.0, 72.0)]
    assert same_line_calls == [(0.0, 15.0), (0.0, 15.0), (0.0, 15.0), (0.0, 15.0), (0.0, 15.0), (0.0, 15.0)]
    assert filled_rects == [(148.0, 69.0, 250.0, 89.0, 6.0)]
    assert len(pushed_colors) == 1
    assert pushed_colors[0][0] == int(ui.imgui.Col_.text.value)
    np.testing.assert_allclose(np.array(pushed_colors[0][1]), np.array((0.985, 0.992, 1.0, 1.0)), rtol=0.0, atol=1e-6)
    assert mode_text == ["Depth Std"]
    assert np.isclose(origin.x, 62.0)
    assert origin.y > 72.0


def test_viewport_view_menu_keeps_training_sh_controls_unchanged(monkeypatch) -> None:
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
            "show_camera_min_dist_spheres": True,
            "show_training_cameras": False,
            "_viewport_sh_band": 0,
            "_viewport_sh_control_key": "sh_band_stage2",
            "sh_band": 0,
            "sh_band_stage2": 0,
        }
    )

    ui.ToolkitWindow._draw_viewport_view_menu(toolkit, viewer_ui, ui.imgui.ImVec2(50.0, 60.0))

    assert button_labels == ["View Mode", "Cameras On", "Labels Off", "Min Dist On", "Training Cameras Off"]
    assert select_calls == [("SH0", True), ("SH1", False), ("SH2", False), ("SH3", False)]
    assert viewer_ui._values["_viewport_sh_band"] == 2
    assert viewer_ui._values["sh_band"] == 0
    assert viewer_ui._values["sh_band_stage2"] == 0


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
            "show_camera_min_dist_spheres": True,
            "_training_view_overlay_segments": (
                (
                    ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)),
                    ((9.0, 10.0), (11.0, 12.0), (13.0, 14.0), (15.0, 16.0)),
                    ((1.0, 2.0, 9.0, 10.0), (3.0, 4.0, 11.0, 12.0), (5.0, 6.0, 13.0, 14.0), (7.0, 8.0, 15.0, 16.0)),
                    (
                        ((2.0, 3.0), (4.0, 5.0), (6.0, 7.0)),
                    ),
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
        ([(12.0, 23.0), (14.0, 25.0), (16.0, 27.0)], int(ui.imgui.ImDrawFlags_.closed.value), 1.275),
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
                    (),
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
    toolkit = SimpleNamespace(_menu_bar_height=24.0, _toolkit_dock_id=17, _applied_interface_scale=1.0, _dock_tool_window=lambda cond: ui.imgui.set_next_window_dock_id(17, cond), _training_views_value_text=ui.ToolkitWindow._training_views_value_text)
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
                    "camera_min_dist": 0.1,
                    "loss": 0.25,
                    "psnr": 32.5,
                },
            ),
        },
        _texts={},
    )

    ui.ToolkitWindow._draw_training_views_window(toolkit, viewer_ui)

    assert dock_calls == [(17, int(ui.imgui.Cond_.appearing.value))]
    assert table_calls and table_calls[0][0] == "##training_views" and table_calls[0][1] == 9
    assert table_columns == ["Image", "Res", "Fx", "Fy", "Cx", "Cy", "Min Dist", "Loss", "PSNR"]
    assert cells == ["frame.png", "640x360", "525.000", "520.000", "320.000", "180.000", "0.100", "0.2500", "32.50"]


def test_resource_debug_window_draws_largest_first_table(monkeypatch) -> None:
    dock_calls: list[tuple[int, int]] = []
    table_columns: list[str] = []
    cells: list[str] = []
    summary: list[str] = []
    table_calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(ui.imgui, "set_next_window_dock_id", lambda dock_id, cond: dock_calls.append((int(dock_id), int(cond))))
    monkeypatch.setattr(ui.imgui, "set_next_window_pos", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "set_next_window_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "begin", lambda *args, **kwargs: (True, True))
    monkeypatch.setattr(ui.imgui, "end", lambda: None)
    monkeypatch.setattr(ui.imgui, "separator", lambda: None)
    monkeypatch.setattr(ui.imgui, "button", lambda _label: False)
    monkeypatch.setattr(ui.imgui, "same_line", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.imgui, "text_disabled", lambda text: summary.append(text))
    monkeypatch.setattr(ui.imgui, "begin_table", lambda name, columns, flags: table_calls.append((name, int(columns), int(flags))) or True)
    monkeypatch.setattr(ui.imgui, "table_setup_column", lambda name, *_args: table_columns.append(name))
    monkeypatch.setattr(ui.imgui, "table_headers_row", lambda: None)
    monkeypatch.setattr(ui.imgui, "table_next_row", lambda: None)
    monkeypatch.setattr(ui.imgui, "table_next_column", lambda: None)
    monkeypatch.setattr(ui.imgui, "text_unformatted", lambda text: summary.append(text) if str(text).startswith(("Total", "Buffers:", "Textures:")) else cells.append(text))
    monkeypatch.setattr(ui.imgui, "end_table", lambda: None)
    snapshot = ResourceDebugSnapshot(
        rows=(
            ResourceDebugRow("Texture", "target", "viewer.state.viewport_texture", 1024, "64x4 rgba32_float", "srv", 2),
            ResourceDebugRow("Buffer", "large", "viewer.renderer.large", 4096, "1,024 elements x 4 B", "rw", 1),
        ),
        total_consumption=5120,
        buffer_count=1,
        buffer_total=4096,
        buffer_mean=4096.0,
        buffer_median=4096.0,
        texture_count=1,
        texture_total=1024,
    )
    toolkit = SimpleNamespace(_menu_bar_height=24.0, _toolkit_dock_id=17, _applied_interface_scale=1.0, _dock_tool_window=lambda cond: ui.imgui.set_next_window_dock_id(17, cond))
    toolkit._draw_resource_debug_summary = lambda snapshot: ui.ToolkitWindow._draw_resource_debug_summary(toolkit, snapshot)
    toolkit._draw_resource_debug_table = lambda snapshot: ui.ToolkitWindow._draw_resource_debug_table(toolkit, snapshot)
    viewer_ui = SimpleNamespace(_values={"show_resource_debug": True, "_resource_debug_snapshot": snapshot}, _texts={"resource_debug_status": ""})

    ui.ToolkitWindow._draw_resource_debug_window(toolkit, viewer_ui)

    assert dock_calls == [(17, int(ui.imgui.Cond_.appearing.value))]
    assert summary == [
        "Total Consumption: 5.00 KiB",
        "Buffers: 1 | total=4.00 KiB | mean=4.00 KiB | median=4.00 KiB",
        "Textures: 1 | total=1.00 KiB",
    ]
    assert table_calls and table_calls[0][0] == "##resource_debug" and table_calls[0][1] == 6
    assert table_columns == ["Size", "Type", "Details", "Name", "Owner", "Usage"]
    assert cells[:6] == ["4.00 KiB", "Buffer", "1,024 elements x 4 B", "large", "viewer.renderer.large", "rw"]
    assert cells[6:] == ["1.00 KiB", "Texture", "64x4 rgba32_float", "target", "viewer.state.viewport_texture", "srv"]


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
        _texts={"loss_debug_view": "View: Rendered", "loss_debug_frame": "Frame[3]: frame.png", "loss_debug_psnr": "PSNR: 32.50 dB"},
    )

    ui.ToolkitWindow._draw_viewport_debug_overlay(toolkit, viewer_ui, ui.imgui.ImVec2(12.0, 34.0))

    assert child_sizes and child_sizes[0][0] >= 220.0
    assert combo_labels == [("##training_camera_view", "Rendered")]
    assert slider_calls == [("##training_camera_frame", 3, 0, 12)]
    assert disabled_text == ["View: Rendered", "frame.png", "32.50 dB"]


def test_build_ui_initializes_loss_debug_psnr_text() -> None:
    viewer_ui = ui.build_ui(_dummy_renderer())

    assert viewer_ui._texts["loss_debug_psnr"] == ""


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


def test_subsampling_control_exposes_one_eighth() -> None:
    spec = next(spec for spec in ui.GROUP_SPECS["Train Setup"] if spec.key == "train_subsample_factor")

    assert spec.kwargs["options"] == ("Auto", "Off", "1/2", "1/3", "1/4", "1/5", "1/6", "1/7", "1/8")


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


def test_apply_visual_state_applies_theme_scale(monkeypatch) -> None:
    scale_calls: list[float] = []
    style = SimpleNamespace(font_scale_main=1.0, scale_all_sizes=lambda value: scale_calls.append(float(value)))
    monkeypatch.setattr(ui.imgui, "get_style", lambda: style)
    toolkit = SimpleNamespace(
        _applied_interface_scale=2.0,
        _applied_theme_index=0,
        _set_current_context=lambda: None,
        _apply_theme=lambda _theme_index: None,
    )

    ui.ToolkitWindow._apply_visual_state(toolkit, 2.0, 1)

    assert scale_calls == [2.0]
    assert np.isclose(style.font_scale_main, 2.0 * (ui._BASE_FONT_SIZE_PX / ui._FONT_ATLAS_SIZE_PX))


def test_performance_plot_heights_scale_with_interface_scale(monkeypatch) -> None:
    plot_sizes: list[tuple[str, float, float]] = []
    monkeypatch.setattr(ui.imgui, "collapsing_header", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(ui.imgui, "text_disabled", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui.imgui, "separator_text", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui.implot, "begin_plot", lambda label, size: plot_sizes.append((str(label), float(size.x), float(size.y))) or False)
    toolkit = SimpleNamespace(
        tk=SimpleNamespace(
            fps_history=[60.0, 58.0],
            loss_history=[1.0, 0.5],
            psnr_history=[20.0, 22.0],
            step_history=[0.0, 1.0],
            step_time_history=[0.0, 1.0],
        ),
        _iters_per_second=lambda *_args: 1.0,
        _plot_scale=lambda viewer_ui: float(viewer_ui._values["scale"]),
    )

    ui.ToolkitWindow._section_performance(toolkit, SimpleNamespace(_values={"scale": 2.0}, _texts={}))

    assert plot_sizes == [("##FPS", -1.0, 220.0), ("##Loss", -1.0, 360.0), ("##PSNR", -1.0, 360.0)]


def test_histogram_plot_height_scales_with_interface_scale(monkeypatch) -> None:
    plot_sizes: list[tuple[str, float, float]] = []
    monkeypatch.setattr(ui.imgui, "text_disabled", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui.implot, "begin_plot", lambda label, size: plot_sizes.append((str(label), float(size.x), float(size.y))) or False)
    toolkit = SimpleNamespace(_plot_scale=lambda viewer_ui: float(viewer_ui._values["scale"]))

    ui.ToolkitWindow._draw_histogram_plot(toolkit, SimpleNamespace(_values={"scale": 1.5}, _texts={}), "test", np.array([0.0, 1.0], dtype=np.float64), np.array([1.0, 2.0], dtype=np.float64), 10.0)

    assert plot_sizes == [("##plot_test", -1.0, 345.0)]


def test_optimizer_regularization_tab_includes_density_controls() -> None:
    assert "sh1_reg" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "density_regularizer" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "color_non_negative_reg" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "ssim_c2" in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_grad_min" not in ui._OPTIMIZER_TAB_KEYS["Regularization"]
    assert "depth_ratio_grad_max" not in ui._OPTIMIZER_TAB_KEYS["Regularization"]
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
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 0"]["max_visible_angle_deg"] == "max_visible_angle_deg"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 0"]["sh_band"] == "sh_band"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 1"]["lr"] == "lr_schedule_stage1_lr"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 1"]["lr_pos_mul"] == "lr_pos_stage1_mul"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 1"]["lr_sh_mul"] == "lr_sh_stage1_mul"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 2"]["colorspace_mod"] == "colorspace_mod_stage2"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 2"]["max_visible_angle_deg"] == "max_visible_angle_deg_stage2"
    assert ui._SCHEDULE_STAGE_GROUPS["Stage 3"]["noise_lr"] == "position_random_step_noise_stage3_lr"
    assert all(ui.SCHEDULE_STAGE_SPECS[stage] for stage in ui.SCHEDULE_STAGE_SPECS)
    assert all(spec.key in ui._SCHEDULE_STAGE_GROUPS[stage].values() for stage, specs in ui.SCHEDULE_STAGE_SPECS.items() for spec in specs)


def test_schedule_step_slider_max_tracks_schedule_steps() -> None:
    assert ui._control_bound(SimpleNamespace(_values={"lr_schedule_steps": 1234}), SimpleNamespace(kwargs={"max_from": "lr_schedule_steps"}), "max_from", 0) == 1234


def test_debug_mode_labels_include_contribution_amount() -> None:
    assert ui._DEBUG_MODE_VALUES[0] == "normal"
    assert ui._DEBUG_MODE_LABELS[0] == "Normal"
    assert "contribution_amount" in ui._DEBUG_MODE_VALUES
    assert "Contribution Amount" in ui._DEBUG_MODE_LABELS
    assert "adam_momentum" in ui._DEBUG_MODE_VALUES
    assert "Adam Momentum" in ui._DEBUG_MODE_LABELS
    assert "adam_second_moment" in ui._DEBUG_MODE_VALUES
    assert "Adam Second Moment" in ui._DEBUG_MODE_LABELS
    assert "grad_variance" in ui._DEBUG_MODE_VALUES
    assert "Grad Variance" in ui._DEBUG_MODE_LABELS
    assert "sh_view_dependent" in ui._DEBUG_MODE_VALUES
    assert "SH View-Dependent" in ui._DEBUG_MODE_LABELS
    assert "sh_coefficient" in ui._DEBUG_MODE_VALUES
    assert "SH Coefficient" in ui._DEBUG_MODE_LABELS
    assert "black_negative" in ui._DEBUG_MODE_VALUES
    assert "Black/Negative Regions" in ui._DEBUG_MODE_LABELS
    assert "depth_local_mismatch" in ui._DEBUG_MODE_VALUES
    assert "Depth Local Mismatch" in ui._DEBUG_MODE_LABELS


def test_contribution_amount_debug_mode_exposes_no_extra_range_controls() -> None:
    viewer_ui = SimpleNamespace(_values={"debug_mode": ui._DEBUG_MODE_VALUES.index("normal")})

    assert ui._debug_colorbar_mode(viewer_ui) is None
    assert ui._renderer_debug_control_keys("ellipse_outlines") == ("debug_mode", "debug_ellipse_thickness_px", "debug_gaussian_scale_multiplier", "debug_min_opacity", "debug_opacity_multiplier", "debug_ellipse_scale_multiplier")
    assert ui._renderer_debug_control_keys("contribution_amount") == ("debug_mode", "debug_contribution_min", "debug_contribution_max")
    assert ui._renderer_debug_control_keys("adam_momentum") == ("debug_mode", "debug_grad_norm_threshold")
    assert ui._renderer_debug_control_keys("adam_second_moment") == ("debug_mode", "debug_grad_norm_threshold")
    assert ui._renderer_debug_control_keys("grad_variance") == ("debug_mode", "debug_grad_norm_threshold")
    assert ui._renderer_debug_control_keys("sh_coefficient") == ("debug_mode", "debug_sh_coeff_index")
    assert ui._renderer_debug_control_keys("black_negative") == ("debug_mode",)
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

    for mode in ("adam_momentum", "adam_second_moment"):
        lo = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), mode, 0.0, viewer_ui))
        hi = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), mode, 1.0, viewer_ui))

        assert np.isclose(lo, 2e-7, rtol=0.0, atol=1e-12)
        assert np.isclose(hi, 2e-3, rtol=0.0, atol=1e-12)


def test_grad_variance_colorbar_ticks_square_grad_norm_band() -> None:
    viewer_ui = SimpleNamespace(_values={"debug_grad_norm_threshold": 2e-4})

    lo = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), "grad_variance", 0.0, viewer_ui))
    hi = float(ui.ToolkitWindow._debug_colorbar_tick_label(SimpleNamespace(), "grad_variance", 1.0, viewer_ui))

    assert np.isclose(lo, 4e-14, rtol=0.0, atol=1e-20)
    assert np.isclose(hi, 4e-6, rtol=0.0, atol=1e-12)


def test_histogram_range_from_histogram_keeps_central_99_percent_of_counts() -> None:
    payload = SimpleNamespace(
        bin_centers=np.array([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0], dtype=np.float64),
        counts=np.array([[1.0, 1.0, 120.0, 200.0, 120.0, 1.0]], dtype=np.float64),
    )

    lo, hi = ui._histogram_range_from_histogram(payload)

    assert np.isclose(lo, -4.0)
    assert np.isclose(hi, -2.0)


def test_histogram_range_from_ranges_uses_finite_extrema_as_fallback() -> None:
    payload = SimpleNamespace(
        min_values=np.array([0.0, -1e-4, -1.0, np.nan], dtype=np.float32),
        max_values=np.array([0.0, 1e-2, 10.0, np.inf], dtype=np.float32),
    )

    lo, hi = ui._histogram_range_from_ranges(payload)

    assert np.isclose(lo, -1.0)
    assert np.isclose(hi, 10.0)

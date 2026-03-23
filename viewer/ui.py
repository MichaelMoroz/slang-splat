from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import slangpy as spy
import slangpy.ui.imgui_bundle as simgui
from imgui_bundle import imgui, implot

from .state import DEBUG_MODE_DEPTH_MEAN, DEBUG_MODE_DEPTH_STD, DEBUG_MODE_NORMAL, DEBUG_MODE_PROCESSED_COUNT
from .training import TrainingController

_WINDOW_TITLE = "Slang Splat Viewer"
_DOC_PATH = Path(__file__).resolve().parents[1] / "README.md"
_INTERFACE_SCALE_KEY = "interface_scale"
_INTERFACE_SCALE_OPTIONS = (
    ("75%", 0.75),
    ("100%", 1.0),
    ("125%", 1.25),
    ("150%", 1.5),
    ("175%", 1.75),
    ("200%", 2.0),
)
_DEFAULT_INTERFACE_SCALE_INDEX = 3
_BASE_FONT_SIZE_PX = 16.0
_FONT_ATLAS_SIZE_PX = _BASE_FONT_SIZE_PX * _INTERFACE_SCALE_OPTIONS[-1][1]
_LOSS_HISTORY_SIZE = 512
_FPS_HISTORY_SIZE = 128
_LOSS_DEBUG_OPTIONS = (("rendered", "Rendered"), ("target", "Target"), ("abs_diff", "Abs Diff"))
_LOSS_DEBUG_ABS_SCALE_KEY = "loss_debug_abs_scale"
_LOSS_DEBUG_ABS_SCALE_DEFAULT = 1.0
_LOSS_DEBUG_ABS_SCALE_MIN = 0.125
_LOSS_DEBUG_ABS_SCALE_MAX = 64.0
_DEBUG_COLORBAR_HEIGHT = 18.0
_DEBUG_COLORBAR_TICKS = 5
_DEBUG_COLORBAR_STEPS = 64
_TOOLKIT_WIDTH_FRACTION = 0.22
_DEBUG_MODE_OPTIONS = (
    (DEBUG_MODE_NORMAL, "Normal"),
    (DEBUG_MODE_PROCESSED_COUNT, "Processed Count"),
    (DEBUG_MODE_DEPTH_MEAN, "Depth Mean"),
    (DEBUG_MODE_DEPTH_STD, "Depth Std"),
)


def _noop() -> None:
    return None


def _default_font_path() -> Path | None:
    package = importlib.import_module("imgui_bundle")
    path = Path(package.__file__).resolve().parent / "assets" / "fonts" / "DroidSans.ttf"
    return path if path.exists() else None


def _menu_item(label: str, shortcut: str = "", selected: bool = False, enabled: bool = True) -> bool:
    return bool(imgui.menu_item(label, shortcut, selected, enabled)[0])


def _saturate(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _jet_colormap(value: float) -> tuple[float, float, float]:
    t = _saturate(value)
    return (
        _saturate(1.5 - abs(4.0 * t - 3.0)),
        _saturate(1.5 - abs(4.0 * t - 2.0)),
        _saturate(1.5 - abs(4.0 * t - 1.0)),
    )


def _color_u32(r: float, g: float, b: float, a: float = 1.0) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(float(r), float(g), float(b), float(a)))


def _processed_count_tick_value(t: float, max_splat_steps: int) -> float:
    return math.pow(2.0, _saturate(t) * math.log2(max(max_splat_steps, 0) + 1.0)) - 1.0


def _format_metric(value: float | None, precision: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{precision}f}"


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"


def _draw_path_row(label: str, value: str, button_label: str) -> bool:
    imgui.text(label)
    width = max(float(imgui.get_content_region_avail().x) - 170.0, 120.0)
    imgui.begin_group()
    imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + width)
    imgui.text_wrapped(value or "<none>")
    imgui.pop_text_wrap_pos()
    imgui.end_group()
    imgui.same_line()
    return bool(imgui.button(button_label))


@dataclass(slots=True)
class ViewerUI:
    values: dict[str, object] = field(
        default_factory=lambda: {
            _INTERFACE_SCALE_KEY: _DEFAULT_INTERFACE_SCALE_INDEX,
            "move_speed": 2.0,
            "fov": 60.0,
            "near": 0.0,
            "far": 1000.0,
            "radius_scale": 1.0,
            "max_anisotropy": 12.0,
            "alpha_cutoff": 0.01,
            "trans_threshold": 0.005,
            "debug_mode": DEBUG_MODE_NORMAL,
            "debug_depth_mean_min": 0.0,
            "debug_depth_mean_max": 20.0,
            "debug_depth_std_min": 0.0,
            "debug_depth_std_max": 2.0,
            "loss_debug": False,
            "loss_debug_view": 0,
            "loss_debug_frame": 0,
            _LOSS_DEBUG_ABS_SCALE_KEY: _LOSS_DEBUG_ABS_SCALE_DEFAULT,
            "_loss_debug_frame_max": 0,
            "background_r": 0.0,
            "background_g": 0.0,
            "background_b": 0.0,
        }
    )
    texts: dict[str, str] = field(
        default_factory=lambda: {
            "scene": "",
            "status": "",
            "error": "",
            "fps": "",
            "render_stats": "",
            "training": "",
            "training_time": "",
            "training_iters_avg": "",
            "training_loss": "",
            "training_mse": "",
            "training_psnr": "",
            "loss_debug_frame": "",
            "training_resolution": "",
            "training_downscale": "",
            "max_splat_steps": "0",
        }
    )


@dataclass(slots=True)
class ToolkitState:
    loss_history: deque = field(default_factory=lambda: deque(maxlen=_LOSS_HISTORY_SIZE))
    fps_history: deque = field(default_factory=lambda: deque(maxlen=_FPS_HISTORY_SIZE))
    psnr_history: deque = field(default_factory=lambda: deque(maxlen=_LOSS_HISTORY_SIZE))
    step_history: deque = field(default_factory=lambda: deque(maxlen=_LOSS_HISTORY_SIZE))
    step_time_history: deque = field(default_factory=lambda: deque(maxlen=_LOSS_HISTORY_SIZE))


class ToolkitWindow:
    @staticmethod
    def _iters_per_second(step_history: deque, step_time_history: deque) -> float:
        if len(step_history) < 2 or len(step_time_history) < 2:
            return 0.0
        sample_count = min(len(step_history), len(step_time_history), 16)
        steps = np.array(list(step_history)[-sample_count:], dtype=np.float64)
        times = np.array(list(step_time_history)[-sample_count:], dtype=np.float64)
        dt = float(times[-1] - times[0])
        if dt <= 1e-6:
            return 0.0
        return float(max(steps[-1] - steps[0], 0.0) / dt)

    def __init__(self, device: spy.Device, width: int, height: int) -> None:
        self.device = device
        self.ctx = simgui.create_imgui_context(width, height)
        imgui.set_current_context(self.ctx)
        implot.create_context()
        imgui.get_io().config_flags |= imgui.ConfigFlags_.docking_enable.value
        self.renderer = spy.ui.Context(device)
        self.callbacks = SimpleNamespace(load_ply=_noop, browse_training_dataset=_noop, browse_training_images=_noop)
        self.tk = ToolkitState()
        self._alive = True
        self._frame_textures: list[spy.Texture] = []
        self._show_about = False
        self._show_docs = False
        self._show_training = True
        self._show_render_debug = False
        self._last_frame_time = time.perf_counter()
        self._menu_bar_height = 0.0
        self._applied_interface_scale = 1.0
        self._toolkit_window_open = True
        self._toolkit_rect = (0.0, 0.0, max(float(width) * _TOOLKIT_WIDTH_FRACTION, 300.0), float(height))
        self._configure_default_font()
        self._apply_theme()
        self._set_interface_scale(_INTERFACE_SCALE_OPTIONS[_DEFAULT_INTERFACE_SCALE_INDEX][1])

    def _apply_theme(self) -> None:
        imgui.style_colors_dark()
        style = imgui.get_style()
        style.window_rounding = 4.0
        style.frame_rounding = 3.0
        style.grab_rounding = 2.0
        style.scrollbar_rounding = 4.0
        style.tab_rounding = 3.0
        style.popup_rounding = 3.0
        style.child_rounding = 3.0
        style.window_border_size = 1.0
        style.frame_border_size = 0.0
        style.item_spacing = imgui.ImVec2(8, 5)
        style.item_inner_spacing = imgui.ImVec2(6, 4)
        style.frame_padding = imgui.ImVec2(6, 4)
        style.indent_spacing = 18.0
        style.scrollbar_size = 12.0
        set_color = style.set_color_
        set_color(imgui.Col_.title_bg_active, imgui.ImVec4(0.18, 0.35, 0.58, 1.0))
        set_color(imgui.Col_.header, imgui.ImVec4(0.22, 0.40, 0.62, 0.55))
        set_color(imgui.Col_.header_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.80))
        set_color(imgui.Col_.header_active, imgui.ImVec4(0.26, 0.50, 0.74, 1.0))
        set_color(imgui.Col_.button, imgui.ImVec4(0.20, 0.40, 0.60, 0.62))
        set_color(imgui.Col_.button_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.80))
        set_color(imgui.Col_.button_active, imgui.ImVec4(0.20, 0.40, 0.60, 1.0))
        set_color(imgui.Col_.tab, imgui.ImVec4(0.14, 0.28, 0.45, 0.86))
        set_color(imgui.Col_.tab_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.80))
        set_color(imgui.Col_.tab_selected, imgui.ImVec4(0.20, 0.42, 0.68, 1.0))
        set_color(imgui.Col_.separator, imgui.ImVec4(0.35, 0.35, 0.45, 0.50))
        set_color(imgui.Col_.frame_bg, imgui.ImVec4(0.12, 0.16, 0.22, 0.90))
        set_color(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.18, 0.28, 0.42, 1.0))
        set_color(imgui.Col_.frame_bg_active, imgui.ImVec4(0.20, 0.34, 0.52, 1.0))
        set_color(imgui.Col_.slider_grab, imgui.ImVec4(0.30, 0.55, 0.82, 0.78))
        set_color(imgui.Col_.slider_grab_active, imgui.ImVec4(0.36, 0.62, 0.90, 1.0))
        set_color(imgui.Col_.check_mark, imgui.ImVec4(0.40, 0.75, 1.00, 1.0))
        set_color(imgui.Col_.scrollbar_grab, imgui.ImVec4(0.24, 0.36, 0.52, 0.60))
        set_color(imgui.Col_.scrollbar_grab_hovered, imgui.ImVec4(0.30, 0.46, 0.64, 0.80))
        set_color(imgui.Col_.scrollbar_grab_active, imgui.ImVec4(0.30, 0.46, 0.64, 1.0))
        set_color(imgui.Col_.separator_hovered, imgui.ImVec4(0.30, 0.55, 0.82, 0.78))
        set_color(imgui.Col_.resize_grip, imgui.ImVec4(0.26, 0.50, 0.74, 0.25))
        set_color(imgui.Col_.resize_grip_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.67))
        set_color(imgui.Col_.resize_grip_active, imgui.ImVec4(0.26, 0.50, 0.74, 0.95))

    def _configure_default_font(self) -> None:
        io = imgui.get_io()
        atlas = io.fonts
        atlas.clear()
        font_path = _default_font_path()
        io.font_default = atlas.add_font_from_file_ttf(str(font_path), _FONT_ATLAS_SIZE_PX) if font_path is not None else atlas.add_font_default()
        if atlas.tex_data is not None:
            atlas.tex_data.get_pixels_array()

    def _interface_scale_factor(self, ui: ViewerUI) -> float:
        idx = min(max(int(ui.values.get(_INTERFACE_SCALE_KEY, _DEFAULT_INTERFACE_SCALE_INDEX)), 0), len(_INTERFACE_SCALE_OPTIONS) - 1)
        return float(_INTERFACE_SCALE_OPTIONS[idx][1])

    def _set_interface_scale(self, scale: float) -> None:
        clamped_scale = max(float(scale), 0.5)
        if abs(clamped_scale - self._applied_interface_scale) <= 1e-6:
            return
        style = imgui.get_style()
        style.scale_all_sizes(clamped_scale / self._applied_interface_scale)
        style.font_scale_main = clamped_scale * (_BASE_FONT_SIZE_PX / _FONT_ATLAS_SIZE_PX)
        self._applied_interface_scale = clamped_scale

    def _sync_interface_scale(self, ui: ViewerUI) -> None:
        self._set_interface_scale(self._interface_scale_factor(ui))

    def handle_keyboard_event(self, event) -> bool:
        if not self._alive:
            return False
        imgui.set_current_context(self.ctx)
        return bool(simgui.handle_keyboard_event(event))

    def handle_mouse_event(self, event) -> bool:
        if not self._alive:
            return False
        imgui.set_current_context(self.ctx)
        return bool(simgui.handle_mouse_event(event))

    def render(self, ui: ViewerUI, training: TrainingController, surface_texture: spy.Texture, command_encoder: spy.CommandEncoder) -> None:
        if not self._alive:
            return
        width = int(surface_texture.width)
        height = int(surface_texture.height)
        now = time.perf_counter()
        dt = max(now - self._last_frame_time, 1e-5)
        self._last_frame_time = now
        imgui.set_current_context(self.ctx)
        self._sync_interface_scale(ui)
        simgui.begin_frame(width, height, dt)
        self._menu_bar_height = self._draw_main_menu_bar(ui)
        imgui.dock_space_over_viewport(viewport=imgui.get_main_viewport(), flags=imgui.DockNodeFlags_.passthru_central_node.value)
        self._draw_panel(ui, training, width, height)
        self._draw_render_debug_window(ui)
        self._draw_about()
        self._draw_docs()
        imgui.render()
        draw_data = imgui.get_draw_data()
        self._frame_textures = simgui.sync_draw_data_textures(self.device, self.renderer, draw_data)
        simgui.render_imgui_draw_data(self.renderer, draw_data, surface_texture, command_encoder)

    def _draw_main_menu_bar(self, ui: ViewerUI) -> float:
        if not imgui.begin_main_menu_bar():
            return 0.0
        if imgui.begin_menu("File"):
            if _menu_item("Load PLY..."):
                self.callbacks.load_ply()
            if _menu_item("Training"):
                self._show_training = True
            imgui.end_menu()
        if imgui.begin_menu("View"):
            active_idx = min(max(int(ui.values.get(_INTERFACE_SCALE_KEY, _DEFAULT_INTERFACE_SCALE_INDEX)), 0), len(_INTERFACE_SCALE_OPTIONS) - 1)
            for idx, (label, _) in enumerate(_INTERFACE_SCALE_OPTIONS):
                if _menu_item(label, selected=idx == active_idx):
                    ui.values[_INTERFACE_SCALE_KEY] = idx
            imgui.separator()
            if _menu_item("Reset Interface Scale"):
                ui.values[_INTERFACE_SCALE_KEY] = _DEFAULT_INTERFACE_SCALE_INDEX
            imgui.end_menu()
        if imgui.begin_menu("Debug"):
            if _menu_item("Rendering"):
                self._show_render_debug = True
            imgui.end_menu()
        if imgui.begin_menu("Help"):
            if _menu_item("Documentation"):
                self._show_docs = True
            if _menu_item("About"):
                self._show_about = True
            imgui.end_menu()
        height = float(imgui.get_window_height())
        imgui.end_main_menu_bar()
        return height

    def _draw_panel(self, ui: ViewerUI, training: TrainingController, width: int, height: int) -> None:
        panel_width = max(float(width) * _TOOLKIT_WIDTH_FRACTION, 320.0)
        imgui.set_next_window_pos(imgui.ImVec2(0.0, self._menu_bar_height), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(panel_width, max(float(height) - self._menu_bar_height, 1.0)), imgui.Cond_.first_use_ever.value)
        opened, self._toolkit_window_open = imgui.begin("Toolkit", self._toolkit_window_open, flags=imgui.WindowFlags_.no_collapse.value)
        if opened:
            pos = imgui.get_window_pos()
            size = imgui.get_window_size()
            self._toolkit_rect = (float(pos.x), float(pos.y), float(size.x), float(size.y))
            self._section_status(ui)
            self._section_camera(ui)
            self._section_view_renderer(ui)
        imgui.end()
        self._draw_training_window(ui, training, width, height)

    def _draw_training_window(self, ui: ViewerUI, training: TrainingController, width: int, height: int) -> None:
        if not self._show_training:
            return
        panel_width = max(float(width) * 0.28, 420.0)
        panel_height = max(float(height) * 0.82, 520.0)
        imgui.set_next_window_pos(
            imgui.ImVec2(max(float(width) - panel_width - 24.0, 24.0), self._menu_bar_height + 24.0),
            imgui.Cond_.first_use_ever.value,
        )
        imgui.set_next_window_size(imgui.ImVec2(panel_width, panel_height), imgui.Cond_.first_use_ever.value)
        opened, self._show_training = imgui.begin("Training", self._show_training, flags=imgui.WindowFlags_.no_collapse.value)
        if opened:
            self._section_plots(ui)
            self._section_training_control(ui, training)
            self._section_dataset(training)
            self._section_training_renderer(training)
            self._section_optimization(training)
            self._section_debug_stats(training.snapshot())
        imgui.end()

    def _section_plots(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Plots", flags=imgui.TreeNodeFlags_.default_open.value):
            return
        iters_per_second = self._iters_per_second(self.tk.step_history, self.tk.step_time_history)
        if iters_per_second > 0.0:
            imgui.text_disabled(f"iters/s {iters_per_second:.1f}")

        fps_arr = np.array(self.tk.fps_history, dtype=np.float64)
        if len(fps_arr) >= 2:
            imgui.text_disabled(f"avg {np.mean(fps_arr):.1f}  min {np.min(fps_arr):.1f}  max {np.max(fps_arr):.1f}")
            if implot.begin_plot("##FPS", imgui.ImVec2(-1, 110)):
                implot.setup_axes("", "FPS", implot.AxisFlags_.no_tick_labels.value, implot.AxisFlags_.auto_fit.value)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, 0, len(fps_arr) - 1, implot.Cond_.always.value)
                shade_spec = implot.Spec()
                shade_spec.fill_alpha = 0.25
                implot.plot_shaded("FPS", fps_arr, 0.0, spec=shade_spec)
                implot.plot_line("FPS", fps_arr)
                implot.annotation(float(len(fps_arr) - 1), float(fps_arr[-1]), imgui.ImVec4(0.4, 0.75, 1.0, 1.0), imgui.ImVec2(-10, -10), True, f"{fps_arr[-1]:.0f}")
                implot.end_plot()
        else:
            imgui.text_disabled("Waiting for FPS data...")

        loss_arr = np.array(self.tk.loss_history, dtype=np.float64)
        step_arr = np.array(self.tk.step_history, dtype=np.float64)
        if len(loss_arr) >= 2 and len(step_arr) >= 2:
            min_len = min(len(loss_arr), len(step_arr))
            s, l = step_arr[:min_len], loss_arr[:min_len]
            imgui.separator_text("Loss")
            loss_spec = implot.Spec()
            loss_spec.line_color = imgui.ImVec4(1.0, 0.6, 0.2, 1.0)
            if implot.begin_plot("##Loss", imgui.ImVec2(-1, 180)):
                implot.setup_axes("step", "loss", 0, implot.AxisFlags_.auto_fit.value)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(s[0]), float(s[-1]), implot.Cond_.always.value)
                implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
                implot.plot_line("Avg Loss", s, l, spec=loss_spec)
                implot.annotation(float(s[-1]), float(l[-1]), imgui.ImVec4(1.0, 0.6, 0.2, 1.0), imgui.ImVec2(-10, -10), True, f"{l[-1]:.2e}")
                implot.end_plot()

        psnr_arr = np.array(self.tk.psnr_history, dtype=np.float64)
        if len(psnr_arr) >= 2 and len(step_arr) >= 2:
            min_len = min(len(psnr_arr), len(step_arr))
            s, p = step_arr[:min_len], psnr_arr[:min_len]
            imgui.separator_text("PSNR")
            psnr_spec = implot.Spec()
            psnr_spec.line_color = imgui.ImVec4(0.3, 0.85, 0.5, 1.0)
            if implot.begin_plot("##PSNR", imgui.ImVec2(-1, 180)):
                implot.setup_axes("step", "PSNR (dB)", 0, implot.AxisFlags_.auto_fit.value)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(s[0]), float(s[-1]), implot.Cond_.always.value)
                implot.plot_line("PSNR", s, p, spec=psnr_spec)
                implot.annotation(float(s[-1]), float(p[-1]), imgui.ImVec4(0.3, 0.85, 0.5, 1.0), imgui.ImVec2(-10, -10), True, f"{p[-1]:.2f}")
                implot.end_plot()
        imgui.separator()

    def _section_status(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Status", flags=imgui.TreeNodeFlags_.default_open.value):
            return
        for key in ("fps", "scene", "status", "render_stats", "training", "training_time", "training_iters_avg", "training_loss", "training_mse", "training_psnr"):
            text = ui.texts.get(key, "")
            if text:
                imgui.text_wrapped(text)
        error = ui.texts.get("error", "")
        if error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.45, 0.45, 1.0), error)
        imgui.separator()

    def _section_camera(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Camera"):
            return
        changed, value = imgui.slider_float("Move Speed", float(ui.values["move_speed"]), 0.1, 20.0)
        if changed:
            ui.values["move_speed"] = value
        changed, value = imgui.slider_float("FOV", float(ui.values["fov"]), 20.0, 100.0)
        if changed:
            ui.values["fov"] = value
        changed, value = imgui.input_float("Near", float(ui.values["near"]), 0.01, 0.1, "%.4f")
        if changed:
            ui.values["near"] = max(value, 0.0)
        changed, value = imgui.input_float("Far", float(ui.values["far"]), 1.0, 10.0, "%.3f")
        if changed:
            ui.values["far"] = max(value, float(ui.values["near"]))
        imgui.text_disabled("Controls: LMB drag look | WASDQE move | wheel speed")
        imgui.separator()

    def _section_training_control(self, ui: ViewerUI, training: TrainingController) -> None:
        if not imgui.collapsing_header("Training", flags=imgui.TreeNodeFlags_.default_open.value):
            return
        snapshot = training.snapshot()
        cfg = training.config
        imgui.text_wrapped(f"Status: {snapshot.status}")
        imgui.text_wrapped(f"Heartbeat: {snapshot.heartbeat}")
        imgui.text_wrapped(f"Iteration: {snapshot.iteration:,}")
        imgui.text_wrapped(f"Scene: {cfg.scene_path or '<none>'}")
        imgui.text_wrapped(f"Cameras: train {snapshot.train_camera_count} | test {snapshot.test_camera_count}")
        resolution_text = ui.texts.get("training_resolution", "")
        if resolution_text:
            imgui.text_wrapped(resolution_text)
        downscale_text = ui.texts.get("training_downscale", "")
        if downscale_text:
            imgui.text_wrapped(downscale_text)
        if snapshot.running and not snapshot.paused:
            imgui.begin_disabled()
        if imgui.button("Reinitialize"):
            if not cfg.scene_path:
                self.callbacks.browse_training_dataset()
            training.reinitialize()
        if snapshot.running and not snapshot.paused:
            imgui.end_disabled()
        imgui.same_line()
        action_label = "Pause" if snapshot.running and not snapshot.paused else "Start"
        if imgui.button(action_label):
            if snapshot.running and not snapshot.paused:
                training.toggle_pause()
            else:
                if not cfg.scene_path:
                    self.callbacks.browse_training_dataset()
                training.start()
        if snapshot.latest is not None:
            latest = snapshot.latest
            imgui.separator()
            imgui.text_wrapped(
                f"Last: loss {latest.loss:.5f} | mse {snapshot.last_mse:.5f} | psnr {_format_metric(snapshot.last_psnr, 2)} | "
                f"eval {_format_metric(latest.test_psnr, 2)} | points {latest.point_count:,}"
            )
            imgui.text_wrapped(
                f"Smooth: loss {_format_metric(snapshot.avg_loss, 5)} | mse {_format_metric(snapshot.avg_mse, 5)} | "
                f"psnr {_format_metric(snapshot.avg_psnr, 2)} | step {latest.elapsed_ms:.2f} ms"
            )
            imgui.text_wrapped(
                f"XYZ LR {latest.xyz_lr:.6g} | camera {latest.camera_name} | "
                f"bg ({latest.background[0]:.2f}, {latest.background[1]:.2f}, {latest.background[2]:.2f})"
            )
        if snapshot.phase_log:
            imgui.separator_text("Phase Log")
            for line in snapshot.phase_log:
                imgui.text_wrapped(line)
        imgui.separator()

    def _section_dataset(self, training: TrainingController) -> None:
        if not imgui.collapsing_header("Dataset"):
            return
        cfg = training.config
        if _draw_path_row("Dataset Folder", cfg.scene_path, "Select Dataset Folder"):
            self.callbacks.browse_training_dataset()
        if _draw_path_row("Images Folder / Subdir", cfg.image_dir, "Select Images Folder"):
            self.callbacks.browse_training_images()
        changed, value = imgui.checkbox("Eval Split", bool(cfg.eval_split))
        if changed:
            cfg.eval_split = bool(value)
        changed, value = imgui.checkbox("Preload CUDA", bool(cfg.preload_cuda))
        if changed:
            cfg.preload_cuda = bool(value)
        changed, value = imgui.checkbox("White Background", bool(cfg.white_background))
        if changed:
            cfg.white_background = bool(value)
        changed, value = imgui.input_int("LLFF Hold", int(cfg.llff_hold), 1, 8)
        if changed:
            cfg.llff_hold = max(int(value), 1)
        changed, value = imgui.input_int("Update Period", int(cfg.update_period), 1, 10)
        if changed:
            cfg.update_period = max(int(value), 1)
        imgui.separator()

    def _section_view_renderer(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Rendering"):
            return
        changed, value = imgui.slider_float("Radius Scale", float(ui.values["radius_scale"]), 0.25, 4.0)
        if changed:
            ui.values["radius_scale"] = value
        changed, value = imgui.slider_float("Max Anisotropy", float(ui.values["max_anisotropy"]), 1.0, 64.0, "%.1f")
        if changed:
            ui.values["max_anisotropy"] = value
        changed, value = imgui.slider_float("Alpha Cutoff", float(ui.values["alpha_cutoff"]), 1e-4, 0.1, "%.4g")
        if changed:
            ui.values["alpha_cutoff"] = value
        changed, value = imgui.slider_float("Trans Threshold", float(ui.values["trans_threshold"]), 1e-4, 0.1, "%.4g")
        if changed:
            ui.values["trans_threshold"] = value
        bg = imgui.ImVec4(float(ui.values["background_r"]), float(ui.values["background_g"]), float(ui.values["background_b"]), 1.0)
        changed, value = imgui.color_edit3("Background", bg, flags=imgui.ColorEditFlags_.float.value)
        if changed:
            ui.values["background_r"] = float(value.x)
            ui.values["background_g"] = float(value.y)
            ui.values["background_b"] = float(value.z)
        imgui.separator()

    def _section_training_renderer(self, training: TrainingController) -> None:
        if not imgui.collapsing_header("Train Renderer"):
            return
        cfg = training.config
        changed, value = imgui.slider_float("Train Radius Scale", float(cfg.radius_scale), 0.25, 4.0)
        if changed:
            cfg.radius_scale = float(value)
        changed, value = imgui.slider_float("Train Max Anisotropy", float(cfg.max_anisotropy), 1.0, 64.0, "%.1f")
        if changed:
            cfg.max_anisotropy = float(value)
        changed, value = imgui.slider_float("Train Alpha Cutoff", float(cfg.alpha_cutoff), 1e-4, 0.1, "%.4g")
        if changed:
            cfg.alpha_cutoff = float(value)
        changed, value = imgui.slider_float("Train Trans Threshold", float(cfg.trans_threshold), 1e-4, 0.1, "%.4g")
        if changed:
            cfg.trans_threshold = float(value)
        imgui.separator()

    def _section_optimization(self, training: TrainingController) -> None:
        if not imgui.collapsing_header("Optimizer"):
            return
        cfg = training.config
        if imgui.begin_tab_bar("##optim_tabs"):
            if imgui.begin_tab_item("Schedule")[0]:
                changed, value = imgui.input_int("Eval Interval", int(cfg.eval_interval), 10, 100)
                if changed:
                    cfg.eval_interval = max(int(value), 1)
                changed, value = imgui.input_int("Densify From", int(cfg.densify_from_iter), 100, 1000)
                if changed:
                    cfg.densify_from_iter = max(int(value), 0)
                changed, value = imgui.input_int("Densify Until", int(cfg.densify_until_iter), 100, 1000)
                if changed:
                    cfg.densify_until_iter = max(int(value), cfg.densify_from_iter)
                changed, value = imgui.input_int("Densify Interval", int(cfg.densification_interval), 10, 100)
                if changed:
                    cfg.densification_interval = max(int(value), 1)
                changed, value = imgui.input_int("Cap Max", int(cfg.cap_max), 1000, 10000)
                if changed:
                    cfg.cap_max = max(int(value), 1)
                changed, value = imgui.input_int("Seed", int(cfg.seed), 1, 10)
                if changed:
                    cfg.seed = int(value)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Init")[0]:
                changed, value = imgui.input_int("Init Points", int(cfg.init_points), 1000, 10000)
                if changed:
                    cfg.init_points = max(int(value), 1)
                changed, value = imgui.input_float("Init Scale Ratio", float(cfg.init_scale_spacing_ratio), 0.01, 0.1, "%.4f")
                if changed:
                    cfg.init_scale_spacing_ratio = max(float(value), 1e-4)
                changed, value = imgui.input_float("Init Scale Mult", float(cfg.init_scale_multiplier), 0.1, 1.0, "%.4f")
                if changed:
                    cfg.init_scale_multiplier = max(float(value), 1e-4)
                changed, value = imgui.input_float("Init Opacity", float(cfg.init_opacity), 0.01, 0.1, "%.4f")
                if changed:
                    cfg.init_opacity = min(max(float(value), 1e-4), 0.9999)
                changed, value = imgui.checkbox("Random Background", bool(cfg.random_background))
                if changed:
                    cfg.random_background = bool(value)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Learning Rates")[0]:
                changed, value = imgui.input_float("Position LR Init", float(cfg.position_lr_init), 1e-5, 1e-4, "%.6g")
                if changed:
                    cfg.position_lr_init = max(float(value), 0.0)
                changed, value = imgui.input_float("Position LR Final", float(cfg.position_lr_final), 1e-6, 1e-5, "%.6g")
                if changed:
                    cfg.position_lr_final = max(float(value), 0.0)
                changed, value = imgui.input_float("Position LR Delay", float(cfg.position_lr_delay_mult), 0.001, 0.01, "%.6g")
                if changed:
                    cfg.position_lr_delay_mult = max(float(value), 0.0)
                changed, value = imgui.input_int("Position LR Steps", int(cfg.position_lr_max_steps), 100, 1000)
                if changed:
                    cfg.position_lr_max_steps = max(int(value), 1)
                changed, value = imgui.input_float("Feature LR", float(cfg.feature_lr), 1e-4, 1e-3, "%.6g")
                if changed:
                    cfg.feature_lr = max(float(value), 0.0)
                changed, value = imgui.input_float("Opacity LR", float(cfg.opacity_lr), 1e-3, 1e-2, "%.6g")
                if changed:
                    cfg.opacity_lr = max(float(value), 0.0)
                changed, value = imgui.input_float("Scaling LR", float(cfg.scaling_lr), 1e-4, 1e-3, "%.6g")
                if changed:
                    cfg.scaling_lr = max(float(value), 0.0)
                changed, value = imgui.input_float("Rotation LR", float(cfg.rotation_lr), 1e-4, 1e-3, "%.6g")
                if changed:
                    cfg.rotation_lr = max(float(value), 0.0)
                changed, value = imgui.input_float("Noise LR", float(cfg.noise_lr), 100.0, 1000.0, "%.4g")
                if changed:
                    cfg.noise_lr = max(float(value), 0.0)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Loss / Reg")[0]:
                changed, value = imgui.input_float("Lambda DSSIM", float(cfg.lambda_dssim), 0.01, 0.1, "%.4f")
                if changed:
                    cfg.lambda_dssim = max(float(value), 0.0)
                changed, value = imgui.input_float("Opacity Reg", float(cfg.opacity_reg), 0.001, 0.01, "%.5f")
                if changed:
                    cfg.opacity_reg = max(float(value), 0.0)
                changed, value = imgui.input_float("Scale Reg", float(cfg.scale_reg), 0.001, 0.01, "%.5f")
                if changed:
                    cfg.scale_reg = max(float(value), 0.0)
                changed, value = imgui.input_float("Train Near", float(cfg.near), 0.01, 0.1, "%.4f")
                if changed:
                    cfg.near = max(float(value), 0.0)
                changed, value = imgui.input_float("Train Far", float(cfg.far), 1.0, 10.0, "%.3f")
                if changed:
                    cfg.far = max(float(value), cfg.near)
                imgui.end_tab_item()
            imgui.end_tab_bar()
        imgui.separator()

    def _section_debug_stats(self, snapshot) -> None:
        latest = snapshot.latest
        if latest is None:
            return
        if imgui.collapsing_header("Gradient Stats"):
            for name, stats in latest.grad_stats.items():
                imgui.text_wrapped(f"{name}: mean|g| {stats.mean_abs:.4e} | max|g| {stats.max_abs:.4e} | l2 {stats.l2:.4e}")
        if imgui.collapsing_header("Splat Stats"):
            stats = latest.splat_stats
            imgui.text_wrapped(f"Opacity: min {stats.opacity_min:.4f} | mean {stats.opacity_mean:.4f} | max {stats.opacity_max:.4f}")
            imgui.text_wrapped(f"Scale: min {stats.scale_min:.4e} | mean {stats.scale_mean:.4e} | max {stats.scale_max:.4e}")
            imgui.text_wrapped(f"Anisotropy: mean {stats.anisotropy_mean:.4f} | max {stats.anisotropy_max:.4f}")
            imgui.text_wrapped(
                f"Bounds min ({stats.position_min[0]:.3f}, {stats.position_min[1]:.3f}, {stats.position_min[2]:.3f}) | "
                f"max ({stats.position_max[0]:.3f}, {stats.position_max[1]:.3f}, {stats.position_max[2]:.3f})"
            )

    def _draw_render_debug_window(self, ui: ViewerUI) -> None:
        if not self._show_render_debug:
            return
        opened, self._show_render_debug = imgui.begin("Rendering Debug", True)
        if opened:
            imgui.separator_text("View Debugger")
            changed, value = imgui.checkbox("Visual Loss Debug", bool(ui.values["loss_debug"]))
            if changed:
                ui.values["loss_debug"] = bool(value)
            imgui.begin_disabled(not bool(ui.values["loss_debug"]))
            view_idx = min(max(int(ui.values["loss_debug_view"]), 0), len(_LOSS_DEBUG_OPTIONS) - 1)
            if imgui.begin_combo("Mode", _LOSS_DEBUG_OPTIONS[view_idx][1]):
                for idx, (_, label) in enumerate(_LOSS_DEBUG_OPTIONS):
                    selected = idx == view_idx
                    if imgui.selectable(label, selected)[0]:
                        ui.values["loss_debug_view"] = idx
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            frame_max = max(int(ui.values.get("_loss_debug_frame_max", 0)), 0)
            changed, value = imgui.slider_int("Frame", int(ui.values["loss_debug_frame"]), 0, frame_max)
            if changed:
                ui.values["loss_debug_frame"] = value
            frame_text = ui.texts.get("loss_debug_frame", "")
            if frame_text:
                imgui.text_disabled(frame_text.split(": ", 1)[-1] if ": " in frame_text else frame_text)
            if _LOSS_DEBUG_OPTIONS[view_idx][0] == "abs_diff":
                changed, value = imgui.slider_float(
                    "Abs Diff Scale",
                    float(ui.values[_LOSS_DEBUG_ABS_SCALE_KEY]),
                    _LOSS_DEBUG_ABS_SCALE_MIN,
                    _LOSS_DEBUG_ABS_SCALE_MAX,
                    "%.3gx",
                    imgui.SliderFlags_.logarithmic.value,
                )
                if changed:
                    ui.values[_LOSS_DEBUG_ABS_SCALE_KEY] = float(value)
            imgui.end_disabled()
            imgui.separator_text("Mode")
            mode_value = int(ui.values["debug_mode"])
            mode_index = next((idx for idx, (value, _) in enumerate(_DEBUG_MODE_OPTIONS) if value == mode_value), 0)
            if imgui.begin_combo("Renderer Debug Mode", _DEBUG_MODE_OPTIONS[mode_index][1]):
                for idx, (value, label) in enumerate(_DEBUG_MODE_OPTIONS):
                    selected = idx == mode_index
                    if imgui.selectable(label, selected)[0]:
                        ui.values["debug_mode"] = value
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            changed, value = imgui.input_float("Depth Mean Min", float(ui.values["debug_depth_mean_min"]), 0.1, 1.0, "%.4g")
            if changed:
                ui.values["debug_depth_mean_min"] = float(value)
            changed, value = imgui.input_float("Depth Mean Max", float(ui.values["debug_depth_mean_max"]), 0.1, 1.0, "%.4g")
            if changed:
                ui.values["debug_depth_mean_max"] = float(value)
            changed, value = imgui.input_float("Depth Std Min", float(ui.values["debug_depth_std_min"]), 0.01, 0.1, "%.4g")
            if changed:
                ui.values["debug_depth_std_min"] = float(value)
            changed, value = imgui.input_float("Depth Std Max", float(ui.values["debug_depth_std_max"]), 0.01, 0.1, "%.4g")
            if changed:
                ui.values["debug_depth_std_max"] = float(value)
            if int(ui.values["debug_mode"]) in (DEBUG_MODE_PROCESSED_COUNT, DEBUG_MODE_DEPTH_MEAN, DEBUG_MODE_DEPTH_STD):
                self._draw_debug_colorbar(ui)
        imgui.end()

    def _draw_debug_colorbar(self, ui: ViewerUI) -> None:
        debug_mode = int(ui.values.get("debug_mode", DEBUG_MODE_NORMAL))
        if debug_mode not in (DEBUG_MODE_PROCESSED_COUNT, DEBUG_MODE_DEPTH_MEAN, DEBUG_MODE_DEPTH_STD):
            return
        imgui.separator()
        title = "Processed Count Scale"
        tick_value = lambda t: _processed_count_tick_value(t, max(int(ui.texts.get("max_splat_steps", "0")), 0))
        if debug_mode == DEBUG_MODE_DEPTH_MEAN:
            title = "Depth Mean Scale"
            depth_min = float(ui.values["debug_depth_mean_min"])
            depth_max = float(ui.values["debug_depth_mean_max"])
            tick_value = lambda t: depth_min + t * (depth_max - depth_min)
        elif debug_mode == DEBUG_MODE_DEPTH_STD:
            title = "Depth Std Scale"
            depth_min = float(ui.values["debug_depth_std_min"])
            depth_max = float(ui.values["debug_depth_std_max"])
            tick_value = lambda t: depth_min + t * (depth_max - depth_min)
        imgui.text(title)
        draw_list = imgui.get_window_draw_list()
        width = max(float(imgui.get_content_region_avail().x), 120.0)
        imgui.dummy(imgui.ImVec2(width, _DEBUG_COLORBAR_HEIGHT + 22.0))
        rect_min = imgui.get_item_rect_min()
        x0 = rect_min.x
        y0 = rect_min.y
        x1 = x0 + width
        y1 = y0 + _DEBUG_COLORBAR_HEIGHT
        for idx in range(_DEBUG_COLORBAR_STEPS):
            t0 = idx / float(_DEBUG_COLORBAR_STEPS)
            t1 = (idx + 1) / _DEBUG_COLORBAR_STEPS
            rgb = _jet_colormap(0.5 * (t0 + t1))
            draw_list.add_rect_filled(imgui.ImVec2(x0 + t0 * width, y0), imgui.ImVec2(x0 + t1 * width, y1), _color_u32(*rgb))
        draw_list.add_rect(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1), _color_u32(0.95, 0.97, 1.0, 0.95), 2.0, 0, 1.0)
        for idx in range(_DEBUG_COLORBAR_TICKS):
            t = idx / max(_DEBUG_COLORBAR_TICKS - 1, 1)
            x = x0 + t * (x1 - x0)
            draw_list.add_line(imgui.ImVec2(x, y1 + 2.0), imgui.ImVec2(x, y1 + 8.0), _color_u32(0.85, 0.88, 0.92, 0.9), 1.0)
            value = tick_value(t)
            label = f"{value:.2f}" if debug_mode in (DEBUG_MODE_DEPTH_MEAN, DEBUG_MODE_DEPTH_STD) else f"{int(round(value)):,}"
            draw_list.add_text(imgui.ImVec2(x - 8.0, y1 + 10.0), _color_u32(0.85, 0.88, 0.92, 0.95), label)

    def _draw_about(self) -> None:
        if not self._show_about:
            return
        opened, self._show_about = imgui.begin("About", True)
        if opened:
            imgui.text_wrapped("Single-window Gaussian splat viewer built on Slangpy and imgui_bundle.")
        imgui.end()

    def _draw_docs(self) -> None:
        if not self._show_docs:
            return
        opened, self._show_docs = imgui.begin("Documentation", True)
        if opened:
            text = _DOC_PATH.read_text(encoding="utf-8") if _DOC_PATH.exists() else "Documentation unavailable."
            if imgui.begin_child("docs"):
                imgui.text_wrapped(text)
                imgui.end_child()
        imgui.end()

    def shutdown(self) -> None:
        if not self._alive:
            return
        self._alive = False
        self._frame_textures.clear()
        imgui.set_current_context(self.ctx)
        implot.destroy_context()
        imgui.destroy_context(self.ctx)


def build_ui() -> ViewerUI:
    return ViewerUI()


def create_toolkit_window(device: spy.Device, width: int, height: int) -> ToolkitWindow:
    return ToolkitWindow(device=device, width=width, height=height)

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
import time

import slangpy as spy
import slangpy.ui.imgui_bundle as simgui
from imgui_bundle import imgui

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
    ("250%", 2.5),
    ("300%", 3.0),
    ("350%", 3.5),
    ("400%", 4.0),
)
_DEFAULT_INTERFACE_SCALE_INDEX = 3
_BASE_FONT_SIZE_PX = 16.0
_FONT_ATLAS_SIZE_PX = _BASE_FONT_SIZE_PX * _INTERFACE_SCALE_OPTIONS[-1][1]
_DEBUG_MODE_NORMAL = 0
_DEBUG_MODE_PROCESSED_COUNT = 1
_DEBUG_COLORBAR_HEIGHT = 18.0
_DEBUG_COLORBAR_TICKS = 5


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


@dataclass(slots=True)
class ViewerUI:
    values: dict[str, object] = field(default_factory=lambda: {
        _INTERFACE_SCALE_KEY: _DEFAULT_INTERFACE_SCALE_INDEX,
        "move_speed": 2.0,
        "fov": 60.0,
        "near": 0.0,
        "far": 1000.0,
        "radius_scale": 1.0,
        "max_anisotropy": 12.0,
        "alpha_cutoff": 0.02,
        "trans_threshold": 0.005,
        "debug_mode": _DEBUG_MODE_NORMAL,
        "background_r": 0.0,
        "background_g": 0.0,
        "background_b": 0.0,
    })
    texts: dict[str, str] = field(default_factory=dict)


class ToolkitWindow:
    def __init__(self, device: spy.Device, width: int, height: int) -> None:
        self.device = device
        self.ctx = simgui.create_imgui_context(width, height)
        imgui.set_current_context(self.ctx)
        imgui.get_io().config_flags |= imgui.ConfigFlags_.docking_enable.value
        self.renderer = spy.ui.Context(device)
        self.callbacks = SimpleNamespace(load_ply=_noop)
        self._alive = True
        self._show_about = False
        self._show_docs = False
        self._show_render_debug = False
        self._last_frame_time = time.perf_counter()
        self._menu_bar_height = 0.0
        self._applied_interface_scale = 1.0
        self._toolkit_window_open = True
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

    def shutdown(self) -> None:
        self._alive = False

    def render(self, ui: ViewerUI, surface_texture: spy.Texture, command_encoder: spy.CommandEncoder) -> None:
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
        self._draw_toolkit_window(ui, width, height)
        self._draw_render_debug_window(ui)
        self._draw_about()
        self._draw_docs()
        imgui.render()
        draw_data = imgui.get_draw_data()
        simgui.sync_draw_data_textures(self.device, self.renderer, draw_data)
        simgui.render_imgui_draw_data(self.renderer, draw_data, surface_texture, command_encoder)

    def _draw_main_menu_bar(self, ui: ViewerUI) -> float:
        if not imgui.begin_main_menu_bar():
            return 0.0
        if imgui.begin_menu("File"):
            if _menu_item("Load PLY..."):
                self.callbacks.load_ply()
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

    def _draw_toolkit_window(self, ui: ViewerUI, width: int, height: int) -> None:
        imgui.set_next_window_pos(imgui.ImVec2(0.0, self._menu_bar_height), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(max(float(width) * 0.22, 300.0), max(float(height) - self._menu_bar_height, 1.0)), imgui.Cond_.first_use_ever.value)
        opened, self._toolkit_window_open = imgui.begin("Toolkit", self._toolkit_window_open, flags=imgui.WindowFlags_.no_collapse.value)
        if opened:
            if ui.texts.get("scene"):
                imgui.text_wrapped(ui.texts["scene"])
            if ui.texts.get("status"):
                imgui.text_wrapped(ui.texts["status"])
            if ui.texts.get("error"):
                imgui.text_colored(imgui.ImVec4(1.0, 0.45, 0.45, 1.0), ui.texts["error"])
            imgui.separator()
            if imgui.collapsing_header("Camera", flags=imgui.TreeNodeFlags_.default_open.value):
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
            if imgui.collapsing_header("Rendering", flags=imgui.TreeNodeFlags_.default_open.value):
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
                bg = imgui.ImVec4(
                    float(ui.values["background_r"]),
                    float(ui.values["background_g"]),
                    float(ui.values["background_b"]),
                    1.0,
                )
                changed, value = imgui.color_edit3(
                    "Background",
                    bg,
                    flags=imgui.ColorEditFlags_.float.value,
                )
                if changed:
                    ui.values["background_r"] = float(value.x)
                    ui.values["background_g"] = float(value.y)
                    ui.values["background_b"] = float(value.z)
            imgui.separator()
            imgui.text_disabled("Controls: LMB drag look | WASDQE move | wheel speed")
        imgui.end()

    def _draw_render_debug_window(self, ui: ViewerUI) -> None:
        if not self._show_render_debug:
            return
        opened, self._show_render_debug = imgui.begin("Rendering Debug", True)
        if opened:
            current = int(ui.values["debug_mode"])
            imgui.text("Mode")
            if imgui.radio_button("Normal", current == _DEBUG_MODE_NORMAL):
                ui.values["debug_mode"] = _DEBUG_MODE_NORMAL
            if imgui.radio_button("Splats Per Tile Processed", current == _DEBUG_MODE_PROCESSED_COUNT):
                ui.values["debug_mode"] = _DEBUG_MODE_PROCESSED_COUNT
            self._draw_debug_colorbar(ui)
        imgui.end()

    def _draw_debug_colorbar(self, ui: ViewerUI) -> None:
        if int(ui.values["debug_mode"]) != _DEBUG_MODE_PROCESSED_COUNT:
            return
        imgui.separator()
        imgui.text("Processed Count Scale")
        draw_list = imgui.get_window_draw_list()
        width = max(float(imgui.get_content_region_avail().x), 120.0)
        imgui.dummy(imgui.ImVec2(width, _DEBUG_COLORBAR_HEIGHT + 22.0))
        rect_min = imgui.get_item_rect_min()
        x0 = rect_min.x
        y0 = rect_min.y
        x1 = x0 + width
        y1 = y0 + _DEBUG_COLORBAR_HEIGHT
        for idx in range(64):
            t0 = idx / 64.0
            t1 = (idx + 1) / 64.0
            rgb = _jet_colormap(0.5 * (t0 + t1))
            draw_list.add_rect_filled(imgui.ImVec2(x0 + t0 * width, y0), imgui.ImVec2(x0 + t1 * width, y1), _color_u32(*rgb))
        draw_list.add_rect(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1), _color_u32(0.95, 0.97, 1.0, 0.95), 2.0, 0, 1.0)
        max_splat_steps = max(int(ui.texts.get("max_splat_steps", "0")), 0)
        for idx in range(_DEBUG_COLORBAR_TICKS):
            t = idx / max(_DEBUG_COLORBAR_TICKS - 1, 1)
            x = x0 + t * (x1 - x0)
            draw_list.add_line(imgui.ImVec2(x, y1 + 2.0), imgui.ImVec2(x, y1 + 8.0), _color_u32(0.85, 0.88, 0.92, 0.9), 1.0)
            value = _processed_count_tick_value(t, max_splat_steps)
            draw_list.add_text(imgui.ImVec2(x - 8.0, y1 + 10.0), _color_u32(0.85, 0.88, 0.92, 0.95), f"{int(round(value)):,}")

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


def build_ui() -> ViewerUI:
    return ViewerUI()


def create_toolkit_window(device: spy.Device, width: int, height: int) -> ToolkitWindow:
    return ToolkitWindow(device, width, height)

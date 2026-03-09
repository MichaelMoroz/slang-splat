from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace

import numpy as np
import glfw
import OpenGL.GL as gl
from imgui_bundle import imgui, implot

from .state import DEFAULT_IMAGE_SUBDIR_INDEX, IMAGE_SUBDIR_OPTIONS, LOSS_DEBUG_OPTIONS

TOOLKIT_WIDTH_FRACTION = 0.15
VIEW_WIDTH_FRACTION = 0.85
LOSS_HISTORY_SIZE = 512
FPS_HISTORY_SIZE = 128


@dataclass(frozen=True, slots=True)
class ControlSpec:
    key: str
    kind: str
    label: str
    kwargs: dict[str, object]


GROUP_SPECS = {
    "Main": (
        ControlSpec("images_subdir", "slider_int", "Image Dir", {"value": DEFAULT_IMAGE_SUBDIR_INDEX, "min": 0, "max": len(IMAGE_SUBDIR_OPTIONS) - 1}),
        ControlSpec("loss_debug", "checkbox", "Visual Loss Debug", {"value": False}),
        ControlSpec("loss_debug_view", "slider_int", "Debug View", {"value": 2, "min": 0, "max": len(LOSS_DEBUG_OPTIONS) - 1}),
        ControlSpec("loss_debug_frame", "slider_int", "Debug Frame", {"value": 0, "min": 0, "max": 10000}),
    ),
    "Camera": (
        ControlSpec("move_speed", "slider_float", "Move Speed", {"value": 2.0, "min": 0.1, "max": 20.0}),
        ControlSpec("fov", "slider_float", "FOV", {"value": 60.0, "min": 25.0, "max": 100.0}),
    ),
    "Train Setup": (
        ControlSpec("max_gaussians", "slider_int", "Max Gaussians", {"value": 5900000, "min": 1000, "max": 10000000}),
        ControlSpec("seed", "slider_int", "Shuffle Seed", {"value": 1234, "min": 0, "max": 1000000}),
        ControlSpec("init_opacity", "input_float", "Init Opacity", {"value": 0.5, "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
    ),
    "Train Optimizer": (
        ControlSpec("lr_base", "input_float", "Base LR", {"value": 1e-3, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("lr_pos_mul", "input_float", "LR Mul Position", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_scale_mul", "input_float", "LR Mul Scale", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_rot_mul", "input_float", "LR Mul Rotation", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_color_mul", "input_float", "LR Mul Color", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_opacity_mul", "input_float", "LR Mul Opacity", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("beta1", "input_float", "Beta1", {"value": 0.9, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ControlSpec("beta2", "input_float", "Beta2", {"value": 0.999, "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ControlSpec("scale_l2", "input_float", "Scale Log Reg", {"value": 0.0, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("scale_abs_reg", "input_float", "Scale Abs Reg", {"value": 0.01, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
        ControlSpec("opacity_reg", "input_float", "Opacity Reg", {"value": 0.01, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
        ControlSpec("max_anisotropy", "input_float", "Max Anisotropy", {"value": 10.0, "step": 0.1, "step_fast": 0.5, "format": "%.6f"}),
        ControlSpec("grad_clip", "input_float", "Grad Clip", {"value": 10.0, "step": 0.1, "step_fast": 1.0, "format": "%.4f"}),
        ControlSpec("grad_norm_clip", "input_float", "Grad Norm Clip", {"value": 10.0, "step": 0.1, "step_fast": 1.0, "format": "%.4f"}),
        ControlSpec("max_update", "input_float", "Max Update", {"value": 0.05, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
    ),
    "Train Stability": (
        ControlSpec("min_scale", "input_float", "Min Scale", {"value": 1e-3, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("max_scale", "input_float", "Max Scale", {"value": 3.0, "step": 1e-2, "step_fast": 0.1, "format": "%.5f"}),
        ControlSpec("min_opacity", "input_float", "Min Opacity", {"value": 1e-4, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("max_opacity", "input_float", "Max Opacity", {"value": 0.9999, "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ControlSpec("position_abs_max", "input_float", "Pos Abs Max", {"value": 1e4, "step": 10.0, "step_fast": 100.0, "format": "%.3f"}),
        ControlSpec("train_near", "input_float", "Train Near", {"value": 0.1, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ControlSpec("train_far", "input_float", "Train Far", {"value": 120.0, "step": 1.0, "step_fast": 10.0, "format": "%.3f"}),
    ),
}


def default_control_values(*group_names: str) -> dict[str, object]:
    groups = GROUP_SPECS.values() if not group_names else (GROUP_SPECS[name] for name in group_names)
    return {spec.key: spec.kwargs["value"] for specs in groups for spec in specs if "value" in spec.kwargs}


RENDER_PARAM_SPECS = (
    ControlSpec("radius_scale", "slider_float", "Radius Scale", {"value": 2.6, "min": 0.5, "max": 4.0, "format": "%.3g"}),
    ControlSpec("alpha_cutoff", "slider_float", "Alpha Cutoff", {"value": 0.0039, "min": 0.0001, "max": 0.1, "format": "%.2e"}),
    ControlSpec("max_splat_steps", "slider_int", "Max Splat Steps", {"value": 32768, "min": 16, "max": 32768}),
    ControlSpec("trans_threshold", "slider_float", "Trans Threshold", {"value": 0.005, "min": 0.001, "max": 0.2, "format": "%.2e"}),
    ControlSpec("sampled5_safety", "slider_float", "MVEE Safety", {"value": 1.0, "min": 1.0, "max": 1.2}),
    ControlSpec("debug_ellipse", "checkbox", "Debug Ellipse Outlines", {"value": False}),
    ControlSpec("debug_processed_count", "checkbox", "Debug Processed Count", {"value": False}),
    ControlSpec("debug_grad_norm", "checkbox", "Debug Grad Norm", {"value": False}),
)

_ALL_DEFAULTS = {spec.key: spec.kwargs["value"] for group in GROUP_SPECS.values() for spec in group if "value" in spec.kwargs}
_ALL_DEFAULTS.update({spec.key: spec.kwargs["value"] for spec in RENDER_PARAM_SPECS if "value" in spec.kwargs})


class _ControlProxy:
    """Minimal proxy giving .value attribute for backward compatibility."""
    __slots__ = ("_store", "_key")

    def __init__(self, store: dict, key: str) -> None:
        self._store = store
        self._key = key

    @property
    def value(self):
        return self._store[self._key]

    @value.setter
    def value(self, v):
        self._store[self._key] = v


class _TextProxy:
    """Minimal proxy giving .text attribute for backward compatibility."""
    __slots__ = ("_store", "_key")

    def __init__(self, store: dict, key: str) -> None:
        self._store = store
        self._key = key

    @property
    def text(self):
        return self._store.get(self._key, "")

    @text.setter
    def text(self, v):
        self._store[self._key] = v


@dataclass(slots=True)
class ViewerUI:
    """Backward-compatible wrapper over plain dicts — provides .controls[key].value and .texts[key].text."""
    _values: dict[str, object] = field(default_factory=dict)
    _texts: dict[str, str] = field(default_factory=dict)

    @property
    def controls(self):
        return {k: _ControlProxy(self._values, k) for k in self._values}

    @property
    def texts(self):
        return {k: _TextProxy(self._texts, k) for k in self._texts}

    def control(self, key: str) -> _ControlProxy:
        return _ControlProxy(self._values, key)

    def text(self, key: str) -> _TextProxy:
        return _TextProxy(self._texts, key)


@dataclass(slots=True)
class ToolkitState:
    loss_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))
    fps_history: deque = field(default_factory=partial(deque, maxlen=FPS_HISTORY_SIZE))
    mse_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))
    step_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))


def _noop() -> None:
    return None


class ToolkitWindow:
    """GLFW + OpenGL3 + Dear ImGui toolkit window (left panel)."""

    def __init__(self, width: int, height: int, x: int, y: int, title: str = "Splat Toolkit"):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.set_window_pos(self.window, x, y)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        self.ctx = imgui.create_context()
        implot.create_context()
        imgui.set_current_context(self.ctx)
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.docking_enable.value

        from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
        self.impl = GlfwRenderer(self.window, attach_callbacks=True)

        self._apply_theme()

        self.callbacks = SimpleNamespace(
            load_ply=_noop,
            load_colmap=_noop,
            reload=_noop,
            reinitialize=_noop,
            start_training=_noop,
            stop_training=_noop,
        )
        self.tk = ToolkitState()
        self._alive = True

    def _apply_theme(self):
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
        _c = style.set_color_
        _c(imgui.Col_.title_bg_active, imgui.ImVec4(0.18, 0.35, 0.58, 1.0))
        _c(imgui.Col_.header, imgui.ImVec4(0.22, 0.40, 0.62, 0.55))
        _c(imgui.Col_.header_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.80))
        _c(imgui.Col_.header_active, imgui.ImVec4(0.26, 0.50, 0.74, 1.00))
        _c(imgui.Col_.button, imgui.ImVec4(0.20, 0.40, 0.60, 0.62))
        _c(imgui.Col_.button_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.80))
        _c(imgui.Col_.button_active, imgui.ImVec4(0.20, 0.40, 0.60, 1.00))
        _c(imgui.Col_.tab, imgui.ImVec4(0.14, 0.28, 0.45, 0.86))
        _c(imgui.Col_.tab_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.80))
        _c(imgui.Col_.tab_selected, imgui.ImVec4(0.20, 0.42, 0.68, 1.00))
        _c(imgui.Col_.separator, imgui.ImVec4(0.35, 0.35, 0.45, 0.50))
        _c(imgui.Col_.plot_lines, imgui.ImVec4(0.40, 0.75, 1.00, 1.00))
        _c(imgui.Col_.frame_bg, imgui.ImVec4(0.12, 0.16, 0.22, 0.90))
        _c(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.18, 0.28, 0.42, 1.00))
        _c(imgui.Col_.frame_bg_active, imgui.ImVec4(0.20, 0.34, 0.52, 1.00))
        _c(imgui.Col_.slider_grab, imgui.ImVec4(0.30, 0.55, 0.82, 0.78))
        _c(imgui.Col_.slider_grab_active, imgui.ImVec4(0.36, 0.62, 0.90, 1.00))
        _c(imgui.Col_.check_mark, imgui.ImVec4(0.40, 0.75, 1.00, 1.00))
        _c(imgui.Col_.scrollbar_grab, imgui.ImVec4(0.24, 0.36, 0.52, 0.60))
        _c(imgui.Col_.scrollbar_grab_hovered, imgui.ImVec4(0.30, 0.46, 0.64, 0.80))
        _c(imgui.Col_.scrollbar_grab_active, imgui.ImVec4(0.30, 0.46, 0.64, 1.00))
        _c(imgui.Col_.separator_hovered, imgui.ImVec4(0.30, 0.55, 0.82, 0.78))
        _c(imgui.Col_.resize_grip, imgui.ImVec4(0.26, 0.50, 0.74, 0.25))
        _c(imgui.Col_.resize_grip_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.67))
        _c(imgui.Col_.resize_grip_active, imgui.ImVec4(0.26, 0.50, 0.74, 0.95))

    @property
    def alive(self) -> bool:
        return self._alive and not glfw.window_should_close(self.window)

    def tick(self, ui: ViewerUI) -> None:
        if not self.alive:
            return
        glfw.make_context_current(self.window)
        glfw.poll_events()
        self.impl.process_inputs()
        imgui.new_frame()
        self._draw_panel(ui)
        imgui.render()
        w, h = glfw.get_framebuffer_size(self.window)
        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0.08, 0.08, 0.10, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)

    def _draw_panel(self, ui: ViewerUI) -> None:
        vp = imgui.get_main_viewport()
        imgui.set_next_window_pos(vp.pos)
        imgui.set_next_window_size(vp.size)
        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_collapse.value
            | imgui.WindowFlags_.no_bring_to_front_on_focus.value
        )
        imgui.begin("##Toolkit", flags=flags)
        self._section_status(ui)
        self._section_scene_io(ui)
        self._section_camera(ui)
        self._section_training_control(ui)
        self._section_training_setup(ui)
        self._section_optimizer(ui)
        self._section_stability(ui)
        self._section_render_params(ui)
        self._section_performance(ui)
        imgui.end()

    # -- Sections --

    def _section_status(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Status", imgui.TreeNodeFlags_.default_open.value):
            return
        # Training state indicator
        training_text = ui._texts.get("training", "Training: not initialized")
        if "running" in training_text:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(0.2, 0.9, 0.3, 1.0))
            imgui.bullet_text("Training active")
            imgui.pop_style_color()
        elif "paused" in training_text:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(1.0, 0.85, 0.2, 1.0))
            imgui.bullet_text("Training paused")
            imgui.pop_style_color()
        else:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(0.5, 0.5, 0.5, 1.0))
            imgui.bullet_text("Idle")
            imgui.pop_style_color()

        # FPS with color gradient (green=good, yellow=ok, red=bad)
        fps = self.tk.fps_history[-1] if self.tk.fps_history else 0.0
        fps_color = imgui.ImVec4(0.2, 0.9, 0.3, 1.0) if fps >= 30 else imgui.ImVec4(1.0, 0.85, 0.2, 1.0) if fps >= 15 else imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
        imgui.text("FPS: ")
        imgui.same_line(0, 0)
        imgui.push_style_color(imgui.Col_.text.value, fps_color)
        imgui.text(f"{fps:.1f}")
        imgui.pop_style_color()

        # Stats table
        tbl_flags = imgui.TableFlags_.sizing_stretch_same.value | imgui.TableFlags_.no_borders_in_body.value
        if imgui.begin_table("##status_tbl", 2, tbl_flags):
            imgui.table_setup_column("Label", imgui.TableColumnFlags_.width_fixed.value, 60)
            imgui.table_setup_column("Value")
            for label, key, fallback in (
                ("Scene", "path", "Scene: <none>"),
                ("Splats", "scene_stats", "Splats: 0"),
            ):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(label)
                imgui.table_next_column()
                raw = ui._texts.get(key, fallback)
                imgui.text(raw.split(": ", 1)[-1] if ": " in raw else raw)
            imgui.end_table()

        # Render stats with tooltip
        render_stats = ui._texts.get("render_stats", "")
        if render_stats:
            imgui.text_disabled("Render:")
            imgui.same_line()
            imgui.text(render_stats.split(": ", 1)[-1] if "Generated:" in render_stats else render_stats)
            if imgui.is_item_hovered():
                imgui.set_item_tooltip("Generated = splat entries created, Written = pixels rasterized")

        error = ui._texts.get("error", "")
        if error:
            imgui.spacing()
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
            imgui.text_wrapped(error)
            imgui.pop_style_color()
        imgui.separator()

    def _section_scene_io(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Scene I/O", imgui.TreeNodeFlags_.default_open.value):
            return
        btn_w = imgui.get_content_region_avail().x
        if imgui.button("Load PLY...", imgui.ImVec2(btn_w, 0)):
            self.callbacks.load_ply()
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Load a pre-trained .ply splat file for viewing")
        if imgui.button("Load COLMAP...", imgui.ImVec2(btn_w, 0)):
            self.callbacks.load_colmap()
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Load a COLMAP dataset folder for training")
        if imgui.button("Reload", imgui.ImVec2(btn_w, 0)):
            self.callbacks.reload()
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Reload the current scene from disk")

        # Combo for image directory selection
        imgui.spacing()
        idx = min(max(int(ui._values["images_subdir"]), 0), len(IMAGE_SUBDIR_OPTIONS) - 1)
        imgui.push_item_width(-1)
        if imgui.begin_combo("##imgdir", IMAGE_SUBDIR_OPTIONS[idx]):
            for i, name in enumerate(IMAGE_SUBDIR_OPTIONS):
                selected = (i == idx)
                if imgui.selectable(name, selected)[0]:
                    ui._values["images_subdir"] = i
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        imgui.pop_item_width()
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Training image resolution (images_8 = 1/8th, images = full)")
        self._ctx_reset("scene_io_ctx", ui, ("images_subdir",))
        imgui.separator()

    def _section_camera(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Camera", imgui.TreeNodeFlags_.default_open.value):
            return
        changed, val = imgui.drag_float(
            "Move Speed", float(ui._values["move_speed"]),
            0.05, 0.1, 20.0, "%.2f", imgui.SliderFlags_.logarithmic.value
        )
        if changed:
            ui._values["move_speed"] = max(0.1, min(val, 20.0))
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Camera movement speed (scroll wheel also adjusts)")
        changed, val = imgui.slider_float("FOV", float(ui._values["fov"]), 25.0, 100.0, "%.1f\u00b0")
        if changed:
            ui._values["fov"] = val
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Vertical field of view in degrees")
        imgui.spacing()
        imgui.text_disabled("LMB drag=look | WASDQE=move | Wheel=speed")
        self._ctx_reset("camera_ctx", ui, ("move_speed", "fov"))
        imgui.separator()

    def _section_training_control(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Training", imgui.TreeNodeFlags_.default_open.value):
            return
        # Training info in compact table
        training_text = ui._texts.get("training", "Training: not initialized")
        loss_text = ui._texts.get("training_loss", "Loss Avg: n/a")
        mse_text = ui._texts.get("training_mse", "MSE: n/a")
        if imgui.begin_table("##train_info", 2, imgui.TableFlags_.sizing_stretch_same.value):
            imgui.table_setup_column("L", imgui.TableColumnFlags_.width_fixed.value, 50)
            imgui.table_setup_column("V")
            for label, text in (("Step", training_text), ("Loss", loss_text), ("MSE", mse_text)):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(label)
                imgui.table_next_column()
                imgui.text(text.split(": ", 1)[-1] if ": " in text else text)
            imgui.end_table()

        instability = ui._texts.get("training_instability", "")
        if instability:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(1.0, 0.85, 0.2, 1.0))
            imgui.text_wrapped(instability)
            imgui.pop_style_color()

        # Colored Start/Stop buttons
        imgui.spacing()
        half_w = imgui.get_content_region_avail().x * 0.48
        imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(0.15, 0.55, 0.22, 0.80))
        imgui.push_style_color(imgui.Col_.button_hovered.value, imgui.ImVec4(0.20, 0.68, 0.28, 0.90))
        imgui.push_style_color(imgui.Col_.button_active.value, imgui.ImVec4(0.15, 0.55, 0.22, 1.00))
        if imgui.button("Start", imgui.ImVec2(half_w, 0)):
            self.callbacks.start_training()
        imgui.pop_style_color(3)
        imgui.same_line()
        imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(0.65, 0.15, 0.15, 0.80))
        imgui.push_style_color(imgui.Col_.button_hovered.value, imgui.ImVec4(0.80, 0.20, 0.20, 0.90))
        imgui.push_style_color(imgui.Col_.button_active.value, imgui.ImVec4(0.65, 0.15, 0.15, 1.00))
        if imgui.button("Stop", imgui.ImVec2(half_w, 0)):
            self.callbacks.stop_training()
        imgui.pop_style_color(3)

        if imgui.button("Reinitialize Gaussians", imgui.ImVec2(imgui.get_content_region_avail().x, 0)):
            self.callbacks.reinitialize()
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Re-initialize all gaussians from the point cloud")

        # Debug section with indent
        imgui.spacing()
        changed, val = imgui.checkbox("Visual Loss Debug", bool(ui._values["loss_debug"]))
        if changed:
            ui._values["loss_debug"] = val

        if bool(ui._values["loss_debug"]):
            imgui.indent()
            # Combo for debug view selection
            view_idx = min(max(int(ui._values["loss_debug_view"]), 0), len(LOSS_DEBUG_OPTIONS) - 1)
            if imgui.begin_combo("##debugview", LOSS_DEBUG_OPTIONS[view_idx][1]):
                for i, (key, name) in enumerate(LOSS_DEBUG_OPTIONS):
                    if imgui.selectable(name, i == view_idx)[0]:
                        ui._values["loss_debug_view"] = i
                    if i == view_idx:
                        imgui.set_item_default_focus()
                imgui.end_combo()

            frame_max = max(int(ui._values.get("_loss_debug_frame_max", 0)), 0)
            changed, val = imgui.slider_int(
                "Frame", int(ui._values["loss_debug_frame"]), 0, frame_max
            )
            if changed:
                ui._values["loss_debug_frame"] = val
            frame_text = ui._texts.get("loss_debug_frame", "")
            if frame_text:
                imgui.text_disabled(frame_text.split(": ", 1)[-1] if ": " in frame_text else frame_text)
            imgui.unindent()
        imgui.separator()

    def _section_training_setup(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Train Setup"):
            return
        for spec in GROUP_SPECS["Train Setup"]:
            self._draw_control(ui, spec)
        imgui.text_disabled("COLMAP init uses direct points + NN scales")
        self._ctx_reset("train_setup_ctx", ui, [s.key for s in GROUP_SPECS["Train Setup"]])
        imgui.separator()

    def _section_optimizer(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Optimizer"):
            return
        if imgui.begin_tab_bar("##optim_tabs"):
            if imgui.begin_tab_item("Learning Rates")[0]:
                imgui.spacing()
                for spec in GROUP_SPECS["Train Optimizer"][:6]:
                    self._draw_control(ui, spec)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Adam")[0]:
                imgui.spacing()
                for spec in GROUP_SPECS["Train Optimizer"][6:8]:
                    self._draw_control(ui, spec)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Regularization")[0]:
                imgui.spacing()
                for spec in GROUP_SPECS["Train Optimizer"][8:]:
                    self._draw_control(ui, spec)
                imgui.end_tab_item()
            imgui.end_tab_bar()
        self._ctx_reset("optimizer_ctx", ui, [s.key for s in GROUP_SPECS["Train Optimizer"]])
        imgui.separator()

    def _section_stability(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Stability"):
            return
        # Paired min/max controls in a two-column table
        pairs = (
            ("min_scale", "max_scale", "Scale"),
            ("min_opacity", "max_opacity", "Opacity"),
        )
        if imgui.begin_table("##stab_pairs", 2, imgui.TableFlags_.sizing_stretch_same.value | imgui.TableFlags_.no_borders_in_body.value):
            imgui.table_setup_column("Min")
            imgui.table_setup_column("Max")
            for min_key, max_key, label in pairs:
                min_spec = next(s for s in GROUP_SPECS["Train Stability"] if s.key == min_key)
                max_spec = next(s for s in GROUP_SPECS["Train Stability"] if s.key == max_key)
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.push_item_width(-1)
                self._draw_control(ui, min_spec)
                imgui.pop_item_width()
                imgui.table_next_column()
                imgui.push_item_width(-1)
                self._draw_control(ui, max_spec)
                imgui.pop_item_width()
            imgui.end_table()

        # Remaining non-paired controls
        paired_keys = {k for min_k, max_k, _ in pairs for k in (min_k, max_k)}
        for spec in GROUP_SPECS["Train Stability"]:
            if spec.key not in paired_keys:
                self._draw_control(ui, spec)
        imgui.text_disabled("Bounds and anisotropy are clamped after each ADAM step")
        self._ctx_reset("stability_ctx", ui, [s.key for s in GROUP_SPECS["Train Stability"]])
        imgui.separator()

    def _section_render_params(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Render Params"):
            return
        # Main render parameters
        for spec in RENDER_PARAM_SPECS[:5]:
            self._draw_control(ui, spec)

        # Debug overlay group
        imgui.separator_text("Debug Overlays")
        for spec in RENDER_PARAM_SPECS[5:]:
            self._draw_control(ui, spec)
        self._ctx_reset("render_ctx", ui, [s.key for s in RENDER_PARAM_SPECS])
        imgui.separator()

    def _section_performance(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Performance"):
            return

        fps_arr = np.array(self.tk.fps_history, dtype=np.float64)
        if len(fps_arr) >= 2:
            imgui.text_disabled(f"avg {np.mean(fps_arr):.1f}  min {np.min(fps_arr):.1f}  max {np.max(fps_arr):.1f}")
            if implot.begin_plot("##FPS", imgui.ImVec2(-1, 110)):
                implot.setup_axes("", "FPS", implot.AxisFlags_.no_tick_labels.value, 0)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, 0, len(fps_arr) - 1, implot.Cond_.always.value)
                implot.setup_axis_limits(implot.ImAxis_.y1.value, 0, float(np.max(fps_arr)) * 1.15, implot.Cond_.always.value)
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
            if implot.begin_plot("##Loss", imgui.ImVec2(-1, 130)):
                implot.setup_axes("step", "loss")
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(s[0]), float(s[-1]), implot.Cond_.always.value)
                implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
                implot.plot_line("Avg Loss", s, l, spec=loss_spec)
                implot.annotation(float(s[-1]), float(l[-1]), imgui.ImVec4(1.0, 0.6, 0.2, 1.0), imgui.ImVec2(-10, -10), True, f"{l[-1]:.2e}")
                implot.end_plot()

        mse_arr = np.array(self.tk.mse_history, dtype=np.float64)
        if len(mse_arr) >= 2 and len(step_arr) >= 2:
            min_len = min(len(mse_arr), len(step_arr))
            s, m = step_arr[:min_len], mse_arr[:min_len]
            imgui.separator_text("MSE")
            mse_spec = implot.Spec()
            mse_spec.line_color = imgui.ImVec4(0.3, 0.85, 0.5, 1.0)
            if implot.begin_plot("##MSE", imgui.ImVec2(-1, 130)):
                implot.setup_axes("step", "MSE")
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(s[0]), float(s[-1]), implot.Cond_.always.value)
                implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
                implot.plot_line("MSE", s, m, spec=mse_spec)
                implot.annotation(float(s[-1]), float(m[-1]), imgui.ImVec4(0.3, 0.85, 0.5, 1.0), imgui.ImVec2(-10, -10), True, f"{m[-1]:.2e}")
                implot.end_plot()

    # -- Helpers --

    _TOOLTIPS = {
        "radius_scale": "Multiplier on gaussian splat radius for rendering",
        "alpha_cutoff": "Minimum alpha threshold — splats below this are skipped",
        "max_splat_steps": "Maximum rasterization steps per pixel ray",
        "trans_threshold": "Transmittance threshold for early ray termination",
        "sampled5_safety": "Safety margin for MVEE bounding ellipsoid",
        "debug_ellipse": "Show ellipse outlines around each gaussian",
        "debug_processed_count": "Heatmap of processed splats per pixel",
        "debug_grad_norm": "Heatmap of gradient norms per pixel",
        "lr_base": "Base learning rate for all parameters",
        "lr_pos_mul": "Learning rate multiplier for position",
        "lr_scale_mul": "Learning rate multiplier for scale",
        "lr_rot_mul": "Learning rate multiplier for rotation",
        "lr_color_mul": "Learning rate multiplier for color/SH",
        "lr_opacity_mul": "Learning rate multiplier for opacity",
        "beta1": "Adam first moment decay (momentum)",
        "beta2": "Adam second moment decay (RMSprop)",
        "grad_clip": "Per-parameter gradient clipping threshold",
        "grad_norm_clip": "Global gradient norm clipping threshold",
        "max_update": "Maximum per-step parameter update magnitude",
        "scale_l2": "L2 regularization on log-scale",
        "scale_abs_reg": "Absolute scale regularization weight",
        "opacity_reg": "Opacity regularization weight (pushes toward 0 or 1)",
        "max_anisotropy": "Maximum ratio between largest and smallest scale axes",
        "min_scale": "Floor for gaussian scale (prevents degenerate splats)",
        "max_scale": "Ceiling for gaussian scale",
        "min_opacity": "Floor for opacity",
        "max_opacity": "Ceiling for opacity",
        "position_abs_max": "Absolute position bounding box (per axis)",
        "train_near": "Near clip plane for training camera",
        "train_far": "Far clip plane for training camera",
        "max_gaussians": "Maximum number of gaussians in the scene",
        "seed": "Random seed for training frame shuffle order",
        "init_opacity": "Initial opacity for new gaussians",
    }

    def _draw_control(self, ui: ViewerUI, spec: ControlSpec) -> None:
        key = spec.key
        if key not in ui._values:
            ui._values[key] = spec.kwargs.get("value", 0)
        if spec.kind == "slider_float":
            changed, val = imgui.slider_float(
                spec.label, float(ui._values[key]),
                float(spec.kwargs.get("min", 0.0)), float(spec.kwargs.get("max", 1.0)),
                spec.kwargs.get("format", "%.3f"),
                imgui.SliderFlags_.logarithmic.value if spec.kwargs.get("logarithmic") else 0
            )
            if changed:
                ui._values[key] = val
        elif spec.kind == "slider_int":
            changed, val = imgui.slider_int(
                spec.label, int(ui._values[key]),
                int(spec.kwargs.get("min", 0)), int(spec.kwargs.get("max", 100))
            )
            if changed:
                ui._values[key] = val
        elif spec.kind == "input_float":
            changed, val = imgui.input_float(
                spec.label, float(ui._values[key]),
                float(spec.kwargs.get("step", 0.0)),
                float(spec.kwargs.get("step_fast", 0.0)),
                spec.kwargs.get("format", "%.3f")
            )
            if changed:
                ui._values[key] = val
        elif spec.kind == "checkbox":
            changed, val = imgui.checkbox(spec.label, bool(ui._values[key]))
            if changed:
                ui._values[key] = val
        tip = self._TOOLTIPS.get(key)
        if tip and imgui.is_item_hovered():
            imgui.set_item_tooltip(tip)

    def _ctx_reset(self, ctx_id: str, ui: ViewerUI, keys) -> None:
        if imgui.begin_popup_context_item(ctx_id):
            if imgui.selectable("Reset to Defaults")[0]:
                for key in keys:
                    if key in _ALL_DEFAULTS:
                        ui._values[key] = _ALL_DEFAULTS[key]
            imgui.separator()
            if imgui.selectable("Copy Values to Clipboard")[0]:
                lines = [f"{k}={ui._values.get(k, '?')}" for k in keys if k in ui._values]
                imgui.set_clipboard_text("\n".join(lines))
            imgui.end_popup()

    def shutdown(self) -> None:
        if not self._alive:
            return
        self._alive = False
        imgui.set_current_context(self.ctx)
        implot.destroy_context()
        self.impl.shutdown()
        imgui.destroy_context(self.ctx)
        glfw.destroy_window(self.window)


def build_ui(renderer) -> ViewerUI:
    """Build a ViewerUI with default control values populated from renderer."""
    values: dict[str, object] = {}
    for specs in GROUP_SPECS.values():
        for spec in specs:
            if "value" in spec.kwargs:
                values[spec.key] = spec.kwargs["value"]
    for spec in RENDER_PARAM_SPECS:
        values[spec.key] = spec.kwargs.get("value", 0)
    values["radius_scale"] = float(renderer.radius_scale)
    values["alpha_cutoff"] = float(renderer.alpha_cutoff)
    values["max_splat_steps"] = int(renderer.max_splat_steps)
    values["trans_threshold"] = float(renderer.transmittance_threshold)
    values["sampled5_safety"] = float(renderer.sampled5_safety_scale)
    values["debug_ellipse"] = bool(renderer.debug_show_ellipses)
    values["debug_processed_count"] = bool(renderer.debug_show_processed_count)
    values["debug_grad_norm"] = bool(renderer.debug_show_grad_norm)
    values["_loss_debug_frame_max"] = 0

    texts: dict[str, str] = {
        key: "" for key in (
            "fps", "path", "scene_stats", "render_stats", "training",
            "training_loss", "training_mse", "training_instability", "error",
            "images_subdir", "loss_debug_view", "loss_debug_frame",
            "setup_hint", "stability_hint",
        )
    }
    return ViewerUI(_values=values, _texts=texts)


def create_toolkit_window() -> ToolkitWindow:
    """Create the ImGui toolkit window positioned on the left side of the primary monitor."""
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW for monitor query")
    monitor = glfw.get_primary_monitor()
    if monitor:
        mode = glfw.get_video_mode(monitor)
        screen_w, screen_h = mode.size.width, mode.size.height
    else:
        screen_w, screen_h = 1920, 1080
    tk_w = max(int(screen_w * TOOLKIT_WIDTH_FRACTION), 280)
    tk_h = screen_h - 80
    return ToolkitWindow(width=tk_w, height=tk_h, x=0, y=0)

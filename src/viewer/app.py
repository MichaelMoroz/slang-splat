from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import slangpy as spy
from slangpy import math as smath

from .. import create_default_device
from ..app.shared import RendererParams, build_init_params, build_training_params, fit_camera
from ..common import normalize3
from ..renderer import Camera, GaussianRenderer
from ..scene import GaussianScene, save_gaussian_ply
from . import presenter, session
from .constants import _WINDOW_TITLE
from .state import (
    DEFAULT_LIST_CAPACITY_MULTIPLIER,
    DEFAULT_MAX_PREPASS_MEMORY_MB,
    LOSS_DEBUG_OPTIONS, ViewerState,
)
from .ui import build_ui, create_toolkit_window, default_control_values

_VIEW_VEC_EPS = 1e-6
_SCROLL_SPEED_BASE = 1.1
_LOOK_SMOOTH = 12.0
_MOVE_SMOOTH = 10.0
_PITCH_LIMIT = math.radians(89.0)
_TRAINING_PARAM_KEYS = {
    "base_lr": "lr_base",
    "lr_pos_mul": "lr_pos_mul",
    "lr_scale_mul": "lr_scale_mul",
    "lr_rot_mul": "lr_rot_mul",
    "lr_color_mul": "lr_color_mul",
    "lr_opacity_mul": "lr_opacity_mul",
    "beta1": "beta1",
    "beta2": "beta2",
    "grad_clip": "grad_clip",
    "grad_norm_clip": "grad_norm_clip",
    "max_update": "max_update",
    "min_scale": "min_scale",
    "max_scale": "max_scale",
    "max_anisotropy": "max_anisotropy",
    "min_opacity": "min_opacity",
    "max_opacity": "max_opacity",
    "position_abs_max": "position_abs_max",
    "near": "train_near",
    "far": "train_far",
    "scale_l2_weight": "scale_l2",
    "scale_abs_reg_weight": "scale_abs_reg",
    "sh1_reg_weight": "sh1_reg",
    "opacity_reg_weight": "opacity_reg",
    "density_regularizer": "density_regularizer",
    "max_allowed_density": "max_allowed_density",
    "position_random_step_noise_lr": "position_random_step_noise_lr",
    "position_random_step_opacity_gate_center": "position_random_step_opacity_gate_center",
    "position_random_step_opacity_gate_sharpness": "position_random_step_opacity_gate_sharpness",
    "max_gaussians": "max_gaussians",
    "lr_schedule_enabled": "lr_schedule_enabled",
    "lr_schedule_start_lr": "lr_schedule_start_lr",
    "lr_schedule_end_lr": "lr_schedule_end_lr",
    "lr_schedule_steps": "lr_schedule_steps",
    "maintenance_interval": "maintenance_interval",
    "maintenance_growth_ratio": "maintenance_growth_ratio",
    "maintenance_growth_start_step": "maintenance_growth_start_step",
    "maintenance_alpha_cull_threshold": "maintenance_alpha_cull_threshold",
    "maintenance_contribution_cull_threshold": "maintenance_contribution_cull_threshold",
    "maintenance_contribution_cull_decay": "maintenance_contribution_cull_decay",
    "background_mode": "background_mode",
    "use_sh": "use_sh",
    "train_downscale_mode": "train_downscale_mode",
    "train_auto_start_downscale": "train_auto_start_downscale",
    "train_downscale_base_iters": "train_downscale_base_iters",
    "train_downscale_iter_step": "train_downscale_iter_step",
    "train_downscale_max_iters": "train_downscale_max_iters",
}
_TRAIN_SETUP_DEFAULTS = default_control_values("Train Setup")
_TRAINING_DEFAULTS = default_control_values("Train Optimizer", "Train Stability")
_CACHED_RASTER_GRAD_ATOMIC_MODE_VALUES = ("float", "fixed")
_DEBUG_MODE_VALUES = (
    GaussianRenderer.DEBUG_MODE_NORMAL,
    GaussianRenderer.DEBUG_MODE_PROCESSED_COUNT,
    GaussianRenderer.DEBUG_MODE_CLONE_COUNT,
    GaussianRenderer.DEBUG_MODE_ELLIPSE_OUTLINES,
    GaussianRenderer.DEBUG_MODE_SPLAT_DENSITY,
    GaussianRenderer.DEBUG_MODE_SPLAT_SPATIAL_DENSITY,
    GaussianRenderer.DEBUG_MODE_SPLAT_SCREEN_DENSITY,
    GaussianRenderer.DEBUG_MODE_CONTRIBUTION_AMOUNT,
    GaussianRenderer.DEBUG_MODE_DEPTH_MEAN,
    GaussianRenderer.DEBUG_MODE_DEPTH_STD,
    GaussianRenderer.DEBUG_MODE_GRAD_NORM,
)

def _training_kwargs(value_for) -> dict[str, object]:
    return {name: value_for(control) for name, control in _TRAINING_PARAM_KEYS.items()}


def _default_training_control_value(control: str) -> object:
    return _TRAIN_SETUP_DEFAULTS[control] if control in _TRAIN_SETUP_DEFAULTS else _TRAINING_DEFAULTS[control]


def default_renderer_params(max_prepass_memory_mb: int = DEFAULT_MAX_PREPASS_MEMORY_MB) -> RendererParams:
    return RendererParams(list_capacity_multiplier=DEFAULT_LIST_CAPACITY_MULTIPLIER, max_prepass_memory_mb=max(int(max_prepass_memory_mb), 1))


def default_init_params():
    return build_init_params(None, None, None, _TRAIN_SETUP_DEFAULTS["init_opacity"], _TRAIN_SETUP_DEFAULTS["seed"])


def _training_background_value(value_for) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.asarray(value_for("train_background_color"), dtype=np.float32).reshape(3))


def _viewer_background_value(value_for) -> tuple[float, float, float]:
    if int(value_for("render_background_mode")) == 0:
        return _training_background_value(value_for)
    return tuple(float(v) for v in np.asarray(value_for("render_background_color"), dtype=np.float32).reshape(3))


def default_training_params(background=(1.0, 1.0, 1.0)):
    return build_training_params(background=background, **_training_kwargs(_default_training_control_value))


class SplatViewer(spy.AppWindow):

    def c(self, key: str):
        return self.ui.control(key)

    def t(self, key: str):
        return self.ui.text(key)

    def renderer_params(self, allow_debug_overlays: bool) -> RendererParams:
        atomic_mode_index = min(max(int(self.c("cached_raster_grad_atomic_mode").value), 0), len(_CACHED_RASTER_GRAD_ATOMIC_MODE_VALUES) - 1)
        debug_mode_index = min(max(int(self.c("debug_mode").value), 0), len(_DEBUG_MODE_VALUES) - 1)
        debug_mode = _DEBUG_MODE_VALUES[debug_mode_index] if allow_debug_overlays else GaussianRenderer.DEBUG_MODE_NORMAL
        return RendererParams(
            radius_scale=float(self.c("radius_scale").value),
            alpha_cutoff=float(self.c("alpha_cutoff").value),
            max_anisotropy=float(self.c("max_anisotropy").value),
            transmittance_threshold=float(self.c("trans_threshold").value),
            list_capacity_multiplier=self.s.list_capacity_multiplier,
            max_prepass_memory_mb=self.s.max_prepass_memory_mb,
            cached_raster_grad_atomic_mode=_CACHED_RASTER_GRAD_ATOMIC_MODE_VALUES[atomic_mode_index],
            cached_raster_grad_fixed_ro_local_range=float(self.c("cached_raster_grad_fixed_ro_local_range").value),
            cached_raster_grad_fixed_scale_range=float(self.c("cached_raster_grad_fixed_scale_range").value),
            cached_raster_grad_fixed_quat_range=float(self.c("cached_raster_grad_fixed_quat_range").value),
            cached_raster_grad_fixed_color_range=float(self.c("cached_raster_grad_fixed_color_range").value),
            cached_raster_grad_fixed_opacity_range=float(self.c("cached_raster_grad_fixed_opacity_range").value),
            debug_mode=debug_mode,
            debug_grad_norm_threshold=float(self.c("debug_grad_norm_threshold").value),
            debug_ellipse_thickness_px=float(self.c("debug_ellipse_thickness_px").value),
            debug_clone_count_range=(float(self.c("debug_clone_count_min").value), float(self.c("debug_clone_count_max").value)),
            debug_density_range=(float(self.c("debug_density_min").value), float(self.c("debug_density_max").value)),
            debug_contribution_range=(float(self.c("debug_contribution_min").value), float(self.c("debug_contribution_max").value)),
            debug_depth_mean_range=(float(self.c("debug_depth_mean_min").value), float(self.c("debug_depth_mean_max").value)),
            debug_depth_std_range=(float(self.c("debug_depth_std_min").value), float(self.c("debug_depth_std_max").value)),
            debug_show_ellipses=debug_mode == GaussianRenderer.DEBUG_MODE_ELLIPSE_OUTLINES,
            debug_show_processed_count=debug_mode == GaussianRenderer.DEBUG_MODE_PROCESSED_COUNT,
            debug_show_grad_norm=debug_mode == GaussianRenderer.DEBUG_MODE_GRAD_NORM,
        )

    def init_params(self):
        return build_init_params(None, None, None, self.c("init_opacity").value, self.c("seed").value)

    def _forward(self) -> spy.float3:
        cy, sy = math.cos(self.s.yaw), math.sin(self.s.yaw)
        cp, sp = math.cos(self.s.pitch), math.sin(self.s.pitch)
        return normalize3(spy.float3(cp * sy, sp, cp * cy), eps=_VIEW_VEC_EPS)

    def camera(self) -> Camera:
        forward = self._forward()
        return Camera.look_at(position=self.s.camera_pos, target=self.s.camera_pos + forward, up=self.s.up, fov_y_degrees=float(self.s.fov_y), near=float(self.s.near), far=float(self.s.far))

    def apply_camera_fit(self, bounds) -> None:
        fit = fit_camera(bounds, self.s.fov_y)
        self.s.camera_pos = fit.position
        self.s.near = fit.near
        self.s.far = fit.far
        self.s.move_speed = fit.move_speed
        self.c("move_speed").value = float(fit.move_speed)
        self.s.yaw = 0.0
        self.s.pitch = 0.0
        self.s.move_vel = spy.float3(0.0, 0.0, 0.0)
        self.s.rot_vel = spy.float2(0.0, 0.0)

    def on_keyboard_event(self, event) -> None:
        if self.toolkit.handle_keyboard_event(event):
            if event.type == spy.KeyboardEventType.key_release:
                self.s.keys[event.key] = False
            return
        if event.type in (spy.KeyboardEventType.key_press, spy.KeyboardEventType.key_release):
            self.s.keys[event.key] = event.type == spy.KeyboardEventType.key_press

    def training_params(self):
        return build_training_params(background=_training_background_value(self._training_control_value), **_training_kwargs(self._training_control_value))

    def render_background(self) -> spy.float3:
        return spy.float3(*_viewer_background_value(self._training_control_value))

    def _training_control_value(self, control: str) -> object:
        return self.c(control).value

    def _apply_resize(self, width: int, height: int) -> None:
        target_width, target_height = int(width), int(height)
        self.device.wait()
        if target_width > 0 and target_height > 0 and (self.s.renderer.width, self.s.renderer.height) != (target_width, target_height):
            session.recreate_renderer(self, target_width, target_height)
        self.s.last_resize_exception = ""
        self.s.last_error = ""

    def on_resize(self, width: int, height: int) -> None:
        try:
            self._apply_resize(width, height)
        except Exception as exc:
            self.s.last_resize_exception = str(exc)
            self.s.last_error = self.s.last_resize_exception

    def on_mouse_event(self, event) -> None:
        if self.toolkit.handle_mouse_event(event):
            if event.type == spy.MouseEventType.move:
                self.s.mx = event.pos.x
                self.s.my = event.pos.y
                self.s.mouse_delta = spy.float2(0.0, 0.0)
            elif event.type in (spy.MouseEventType.button_down, spy.MouseEventType.button_up) and event.button == spy.MouseButton.left:
                self.s.mouse_left = False
            return
        if event.type in (spy.MouseEventType.button_down, spy.MouseEventType.button_up) and event.button == spy.MouseButton.left:
            self.s.mouse_left = event.type == spy.MouseEventType.button_down
        if event.type == spy.MouseEventType.move:
            if self.s.mx is not None and self.s.my is not None:
                self.s.mouse_delta += spy.float2(event.pos.x - self.s.mx, event.pos.y - self.s.my)
            self.s.mx = event.pos.x
            self.s.my = event.pos.y
        if event.type == spy.MouseEventType.scroll:
            self.s.scroll_delta += float(event.scroll.y)

    def __init__(self, app: spy.App, width: int = 1280, height: int = 720, title: str = _WINDOW_TITLE, max_prepass_memory_mb: int = 4096) -> None:
        super().__init__(app, width=width, height=height, title=title, resizable=True, enable_vsync=False)
        self.loss_debug_view_options = LOSS_DEBUG_OPTIONS
        self.s = ViewerState(max_prepass_memory_mb=max(int(max_prepass_memory_mb), 1))
        self.s.renderer = GaussianRenderer(self.device, width=width, height=height, list_capacity_multiplier=self.s.list_capacity_multiplier, max_prepass_memory_mb=self.s.max_prepass_memory_mb)
        self.ui = build_ui(self.s.renderer)
        self.toolkit = create_toolkit_window(self.device, width, height)
        self._bind_toolkit_callbacks()
        session.create_debug_shaders(self)

    def _bind_toolkit_callbacks(self) -> None:
        cb = self.toolkit.callbacks
        cb.load_ply = self._load_ply_callback
        cb.export_ply = self._export_ply_callback
        cb.browse_colmap_root = self._browse_colmap_root_callback
        cb.browse_colmap_images = self._browse_colmap_images_callback
        cb.browse_colmap_ply = self._browse_colmap_ply_callback
        cb.import_colmap = self._import_colmap_callback
        cb.reload = self._reload_callback
        cb.reinitialize = self._reinitialize_callback
        cb.start_training = self._start_training_callback
        cb.stop_training = self._stop_training_callback

    def _run_action(self, action, *, close_colmap_import: bool = False) -> None:
        try:
            action()
        except Exception as exc:
            self.s.last_error = str(exc)
        else:
            self.s.last_error = ""
            if close_colmap_import:
                self.toolkit.close_colmap_import_window()

    def _load_ply_callback(self) -> None:
        path = spy.platform.open_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")])
        if path:
            self._run_action(lambda: session.load_scene(self, Path(path)))

    def _export_source_scene(self) -> GaussianScene:
        if self.s.trainer is not None:
            return self.s.trainer.read_live_scene()
        if isinstance(self.s.scene, GaussianScene):
            return self.s.scene
        raise RuntimeError("No gaussian scene is available to export.")

    def _export_ply_callback(self) -> None:
        path = spy.platform.save_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")])
        if path is None:
            return
        def _export() -> None:
            scene = SplatViewer._export_source_scene(self)
            export_path = Path(path)
            export_path = export_path.with_suffix(".ply") if export_path.suffix.lower() != ".ply" else export_path
            saved_path = save_gaussian_ply(export_path, scene)
            print(f"Exported scene: {saved_path} ({scene.count:,} splats)")
        self._run_action(_export)

    def _browse_colmap_root_callback(self) -> None:
        path = spy.platform.choose_folder_dialog()
        if path:
            self._run_action(lambda: session.choose_colmap_root(self, Path(path)))

    def _browse_colmap_images_callback(self) -> None:
        path = spy.platform.choose_folder_dialog()
        if path:
            self._run_action(lambda: session.choose_colmap_images_root(self, Path(path)))

    def _browse_colmap_ply_callback(self) -> None:
        path = spy.platform.open_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")])
        if path:
            self._run_action(lambda: session.choose_colmap_custom_ply(self, Path(path)))

    def _import_colmap_callback(self) -> None:
        self._run_action(lambda: session.import_colmap_from_ui(self))

    def _reload_callback(self) -> None:
        if self.s.scene_path is not None:
            self._run_action(lambda: session.load_scene(self, self.s.scene_path))
        elif self.s.colmap_root is not None:
            import_cfg = self.s.colmap_import
            if import_cfg.images_root is None:
                self.s.last_error = "COLMAP reload requires a stored image folder."
                return
            self._run_action(
                lambda: session.import_colmap_dataset(
                    self,
                    colmap_root=self.s.colmap_root,
                    database_path=import_cfg.database_path,
                    images_root=import_cfg.images_root,
                    init_mode=import_cfg.init_mode,
                    custom_ply_path=import_cfg.custom_ply_path,
                    image_downscale_mode=import_cfg.image_downscale_mode,
                    image_downscale_target_width=import_cfg.image_downscale_target_width,
                    image_downscale_scale=import_cfg.image_downscale_scale,
                    nn_radius_scale_coef=import_cfg.nn_radius_scale_coef,
                    diffused_point_count=import_cfg.diffused_point_count,
                    diffusion_radius=import_cfg.diffusion_radius,
                )
            )

    def _reinitialize_callback(self) -> None:
        self.s.training_active = False
        self.s.pending_training_reinitialize = True

    def _start_training_callback(self) -> None:
        session.set_training_active(self, True)

    def _stop_training_callback(self) -> None:
        session.set_training_active(self, False)

    def _render_frame(self, render_context) -> None:
        presenter.render_frame(self, render_context)
        self.toolkit.render(self.ui, render_context.surface_texture, render_context.command_encoder)

    def render(self, render_context) -> None:
        try:
            self._render_frame(render_context)
        except Exception as exc:
            self.s.training_active = False
            self.s.last_error = str(exc)
            self.s.last_render_exception = self.s.last_error

    def update_camera(self, dt: float) -> None:
        self.s.move_speed, self.s.fov_y = float(self.c("move_speed").value), float(self.c("fov").value)
        if abs(self.s.scroll_delta) > 1e-5:
            self.s.move_speed = max(self.s.move_speed * (_SCROLL_SPEED_BASE ** self.s.scroll_delta), 0.0)
            self.c("move_speed").value, self.s.scroll_delta = self.s.move_speed, 0.0
        target_rot = self.s.mouse_delta * self.s.look_speed if self.s.mouse_left else spy.float2(0.0, 0.0)
        self.s.rot_vel += (target_rot - self.s.rot_vel) * min(1.0, _LOOK_SMOOTH * dt)
        self.s.mouse_delta = spy.float2(0.0, 0.0)
        if float(smath.length(self.s.rot_vel)) > _VIEW_VEC_EPS:
            self.s.yaw += float(self.s.rot_vel.x)
            self.s.pitch = min(max(self.s.pitch + float(self.s.rot_vel.y), -_PITCH_LIMIT), _PITCH_LIMIT)
        forward = self._forward()
        right, up = normalize3(smath.cross(self.s.up, forward), eps=_VIEW_VEC_EPS), normalize3(smath.cross(forward, normalize3(smath.cross(self.s.up, forward), eps=_VIEW_VEC_EPS)), eps=_VIEW_VEC_EPS)
        move = spy.float3(float(self.s.keys.get(spy.KeyCode.e, False)) - float(self.s.keys.get(spy.KeyCode.q, False)), float(self.s.keys.get(spy.KeyCode.d, False)) - float(self.s.keys.get(spy.KeyCode.a, False)), float(self.s.keys.get(spy.KeyCode.w, False)) - float(self.s.keys.get(spy.KeyCode.s, False)))
        move_length = float(smath.length(move))
        target_move = move * (self.s.move_speed / max(move_length, _VIEW_VEC_EPS)) if move_length > _VIEW_VEC_EPS else spy.float3(0.0, 0.0, 0.0)
        self.s.move_vel += (target_move - self.s.move_vel) * min(1.0, _MOVE_SMOOTH * dt)
        self.s.camera_pos += (up * self.s.move_vel.x + right * self.s.move_vel.y + forward * self.s.move_vel.z) * dt


def _compute_view_geometry() -> tuple[int, int]:
    """Pick a single-window viewer size based on the current desktop size."""
    import sys
    if sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32
        screen_width = int(user32.GetSystemMetrics(0))
        screen_height = int(user32.GetSystemMetrics(1))
        return max(min(int(screen_width * 0.9), 1920), 1280), max(min(int(screen_height * 0.9), 1200), 720)
    return 1600, 900

def main() -> int:
    view_w, view_h = _compute_view_geometry()
    device = create_default_device(enable_debug_layers=False)
    app = spy.App(device=device)
    viewer = SplatViewer(app, width=view_w, height=view_h, title=_WINDOW_TITLE, max_prepass_memory_mb=4096)
    try:
        app.run()
    finally:
        viewer.toolkit.shutdown()
    return 0

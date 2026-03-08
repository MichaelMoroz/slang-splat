from __future__ import annotations

import math
from pathlib import Path

import slangpy as spy
from slangpy import math as smath

from .. import create_default_device
from ..app.shared import RendererParams, build_init_params, build_training_params, fit_camera
from ..common import normalize3
from ..renderer import Camera, GaussianRenderer
from . import presenter, session
from .state import DEFAULT_IMAGE_SUBDIR_INDEX, DEFAULT_LIST_CAPACITY_MULTIPLIER, DEFAULT_MAX_PREPASS_MEMORY_MB, DEFAULT_VIEWER_BACKGROUND, IMAGE_SUBDIR_OPTIONS, LOSS_DEBUG_OPTIONS, ViewerState
from .ui import build_ui, default_control_values

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
    "epsilon": "eps",
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
    "mcmc_position_noise_scale": "mcmc_pos_noise_scale",
    "mcmc_opacity_gate_sharpness": "mcmc_opacity_k",
    "mcmc_opacity_gate_center": "mcmc_opacity_t",
}
_TRAIN_INIT_DEFAULTS = default_control_values("Train Init")
_TRAINING_DEFAULTS = default_control_values("Train Optimizer", "Train Stability")


default_images_subdir = lambda: IMAGE_SUBDIR_OPTIONS[DEFAULT_IMAGE_SUBDIR_INDEX]
default_renderer_params = lambda max_prepass_memory_mb=DEFAULT_MAX_PREPASS_MEMORY_MB: RendererParams(list_capacity_multiplier=DEFAULT_LIST_CAPACITY_MULTIPLIER, max_prepass_memory_mb=max(int(max_prepass_memory_mb), 1))
default_init_params = lambda: build_init_params(_TRAIN_INIT_DEFAULTS["init_pos_jitter"], _TRAIN_INIT_DEFAULTS["init_scale"], _TRAIN_INIT_DEFAULTS["init_scale_jitter"], _TRAIN_INIT_DEFAULTS["init_opacity"], _TRAIN_INIT_DEFAULTS["gaussian_count"], _TRAIN_INIT_DEFAULTS["seed"])
default_training_params = lambda background=DEFAULT_VIEWER_BACKGROUND: build_training_params(background=background, mcmc_position_noise_enabled=bool(_TRAINING_DEFAULTS["mcmc_pos_noise_enabled"]), low_quality_reinit_enabled=bool(_TRAINING_DEFAULTS["low_quality_reinit"]), **{name: _TRAINING_DEFAULTS[control] for name, control in _TRAINING_PARAM_KEYS.items()})


class SplatViewer(spy.AppWindow):
    c = lambda self, key: self.ui.controls[key]
    t = lambda self, key: self.ui.texts[key]
    _selected_images_subdir = lambda self: self.image_subdir_options[min(max(int(self.c("images_subdir").value), 0), len(self.image_subdir_options) - 1)]
    render = lambda self, render_context: presenter.render_frame(self, render_context)
    renderer_params = lambda self, allow_debug_overlays: RendererParams(radius_scale=float(self.c("radius_scale").value), alpha_cutoff=float(self.c("alpha_cutoff").value), max_splat_steps=int(self.c("max_splat_steps").value), transmittance_threshold=float(self.c("trans_threshold").value), sampled5_safety_scale=float(self.c("sampled5_safety").value), list_capacity_multiplier=self.s.list_capacity_multiplier, max_prepass_memory_mb=self.s.max_prepass_memory_mb, debug_show_ellipses=bool(self.c("debug_ellipse").value) if allow_debug_overlays else False, debug_show_processed_count=bool(self.c("debug_processed_count").value) if allow_debug_overlays else False)
    init_params = lambda self: build_init_params(self.c("init_pos_jitter").value, self.c("init_scale").value, self.c("init_scale_jitter").value, self.c("init_opacity").value, self.c("gaussian_count").value, self.c("seed").value)
    _forward = lambda self: (lambda cy, sy, cp, sp: normalize3(spy.float3(cp * sy, sp, cp * cy), eps=_VIEW_VEC_EPS))(math.cos(self.s.yaw), math.sin(self.s.yaw), math.cos(self.s.pitch), math.sin(self.s.pitch))
    camera = lambda self: (lambda forward: Camera.look_at(position=self.s.camera_pos, target=self.s.camera_pos + forward, up=self.s.up, fov_y_degrees=float(self.s.fov_y), near=float(self.s.near), far=float(self.s.far)))(self._forward())
    apply_camera_fit = lambda self, bounds: (lambda fit: (setattr(self.s, "camera_pos", fit.position), setattr(self.s, "near", fit.near), setattr(self.s, "far", fit.far), setattr(self.s, "move_speed", fit.move_speed), setattr(self.c("move_speed"), "value", float(fit.move_speed)), setattr(self.s, "yaw", 0.0), setattr(self.s, "pitch", 0.0), setattr(self.s, "move_vel", spy.float3(0.0, 0.0, 0.0)), setattr(self.s, "rot_vel", spy.float2(0.0, 0.0)) ))(fit_camera(bounds, self.s.fov_y))
    on_keyboard_event = lambda self, event: self.s.keys.__setitem__(event.key, event.type == spy.KeyboardEventType.key_press) if event.type in (spy.KeyboardEventType.key_press, spy.KeyboardEventType.key_release) else None
    training_params = lambda self: build_training_params(background=self.s.background, mcmc_position_noise_enabled=bool(self.c("mcmc_pos_noise_enabled").value), low_quality_reinit_enabled=bool(self.c("low_quality_reinit").value), **{name: self.c(control).value for name, control in _TRAINING_PARAM_KEYS.items()})
    on_resize = lambda self, width, height: (self.device.wait(), session.recreate_renderer(self, int(width), int(height)) if width > 0 and height > 0 and (self.s.renderer.width, self.s.renderer.height) != (int(width), int(height)) else None, setattr(self.s, "last_resize_exception", ""), setattr(self.s, "last_error", ""))
    on_mouse_event = lambda self, event: (setattr(self.s, "mouse_left", event.type == spy.MouseEventType.button_down) if event.type in (spy.MouseEventType.button_down, spy.MouseEventType.button_up) and event.button == spy.MouseButton.left else None, self.s.mouse_delta.__iadd__(spy.float2(event.pos.x - self.s.mx, event.pos.y - self.s.my)) if event.type == spy.MouseEventType.move and self.s.mx is not None and self.s.my is not None else None, setattr(self.s, "mx", event.pos.x) if event.type == spy.MouseEventType.move else None, setattr(self.s, "my", event.pos.y) if event.type == spy.MouseEventType.move else None, setattr(self.s, "scroll_delta", self.s.scroll_delta + float(event.scroll.y)) if event.type == spy.MouseEventType.scroll else None)

    def __init__(self, app: spy.App, width: int = 1280, height: int = 720, title: str = "Slang Splat Viewer", max_prepass_memory_mb: int = 4096) -> None:
        super().__init__(app, width=width, height=height, title=title, resizable=True, enable_vsync=False)
        self.image_subdir_options, self.loss_debug_view_options = IMAGE_SUBDIR_OPTIONS, LOSS_DEBUG_OPTIONS
        self.s = ViewerState(max_prepass_memory_mb=max(int(max_prepass_memory_mb), 1))
        self.s.renderer = GaussianRenderer(self.device, width=width, height=height, list_capacity_multiplier=self.s.list_capacity_multiplier, max_prepass_memory_mb=self.s.max_prepass_memory_mb)
        self.ui = build_ui(self.screen, self, self.s.renderer)
        self.c("images_subdir").value = int(DEFAULT_IMAGE_SUBDIR_INDEX)
        session.create_debug_shaders(self)

    def update_camera(self, dt: float) -> None:
        self.s.move_speed, self.s.fov_y = float(self.c("move_speed").value), float(self.c("fov").value)
        if abs(self.s.scroll_delta) > 1e-5:
            self.s.move_speed = min(max(self.s.move_speed * (_SCROLL_SPEED_BASE ** self.s.scroll_delta), 0.1), 20.0)
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

main = lambda: (lambda app: (SplatViewer(app, width=1280, height=720, max_prepass_memory_mb=4096), app.run(), 0)[-1])(spy.App(device=create_default_device(enable_debug_layers=False)))

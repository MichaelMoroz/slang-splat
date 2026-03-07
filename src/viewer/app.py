from __future__ import annotations
from pathlib import Path

import numpy as np
import slangpy as spy

from .. import create_default_device
from ..app.shared import RendererParams, build_init_params, build_training_params, fit_camera
from ..renderer import Camera, GaussianRenderer
from . import presenter, session
from .state import DEFAULT_IMAGE_SUBDIR_INDEX, IMAGE_SUBDIR_OPTIONS, LOSS_DEBUG_OPTIONS, ViewerState
from .ui import build_ui


def _normalize(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float32) / np.maximum(np.linalg.norm(v), 1e-8)


class SplatViewer(spy.AppWindow):
    def __init__(self, app: spy.App, width: int = 1280, height: int = 720, title: str = "Slang Splat Viewer", max_prepass_memory_mb: int = 4096) -> None:
        super().__init__(app, width=width, height=height, title=title, resizable=True, enable_vsync=False)
        self.image_subdir_options = IMAGE_SUBDIR_OPTIONS
        self.loss_debug_view_options = LOSS_DEBUG_OPTIONS
        self.s = ViewerState(max_prepass_memory_mb=max(int(max_prepass_memory_mb), 1))
        self.s.renderer = GaussianRenderer(
            self.device,
            width=width,
            height=height,
            list_capacity_multiplier=self.s.list_capacity_multiplier,
            max_prepass_memory_mb=self.s.max_prepass_memory_mb,
        )
        self.ui = build_ui(self.screen, self, self.s.renderer)
        self.c("images_subdir").value = int(DEFAULT_IMAGE_SUBDIR_INDEX)
        session.create_debug_shaders(self)

    def c(self, key: str):
        return self.ui.controls[key]

    def t(self, key: str):
        return self.ui.texts[key]

    def renderer_params(self, allow_debug_overlays: bool) -> RendererParams:
        return RendererParams(
            radius_scale=float(self.c("radius_scale").value),
            alpha_cutoff=float(self.c("alpha_cutoff").value),
            max_splat_steps=int(self.c("max_splat_steps").value),
            transmittance_threshold=float(self.c("trans_threshold").value),
            sampled5_safety_scale=float(self.c("sampled5_safety").value),
            list_capacity_multiplier=self.s.list_capacity_multiplier,
            max_prepass_memory_mb=self.s.max_prepass_memory_mb,
            debug_show_ellipses=bool(self.c("debug_ellipse").value) if allow_debug_overlays else False,
            debug_show_processed_count=bool(self.c("debug_processed_count").value) if allow_debug_overlays else False,
        )

    def init_params(self):
        return build_init_params(
            position_jitter_std=self.c("init_pos_jitter").value,
            base_scale=self.c("init_scale").value,
            scale_jitter_ratio=self.c("init_scale_jitter").value,
            initial_opacity=self.c("init_opacity").value,
            gaussian_count=self.c("gaussian_count").value,
            seed=self.c("seed").value,
        )

    def training_params(self):
        return build_training_params(
            background=self.s.background,
            base_lr=self.c("lr_base").value,
            lr_pos_mul=self.c("lr_pos_mul").value,
            lr_scale_mul=self.c("lr_scale_mul").value,
            lr_rot_mul=self.c("lr_rot_mul").value,
            lr_color_mul=self.c("lr_color_mul").value,
            lr_opacity_mul=self.c("lr_opacity_mul").value,
            beta1=self.c("beta1").value,
            beta2=self.c("beta2").value,
            epsilon=self.c("eps").value,
            grad_clip=self.c("grad_clip").value,
            grad_norm_clip=self.c("grad_norm_clip").value,
            max_update=self.c("max_update").value,
            min_scale=self.c("min_scale").value,
            max_scale=self.c("max_scale").value,
            max_anisotropy=self.c("max_anisotropy").value,
            min_opacity=self.c("min_opacity").value,
            max_opacity=self.c("max_opacity").value,
            position_abs_max=self.c("position_abs_max").value,
            near=self.c("train_near").value,
            far=self.c("train_far").value,
            scale_l2_weight=self.c("scale_l2").value,
            mcmc_position_noise_enabled=bool(self.c("mcmc_pos_noise_enabled").value),
            mcmc_position_noise_scale=self.c("mcmc_pos_noise_scale").value,
            mcmc_opacity_gate_sharpness=self.c("mcmc_opacity_k").value,
            mcmc_opacity_gate_center=self.c("mcmc_opacity_t").value,
            low_quality_reinit_enabled=bool(self.c("low_quality_reinit").value),
        )

    def forward(self) -> np.ndarray:
        cy, sy = np.cos(self.s.yaw), np.sin(self.s.yaw)
        cp, sp = np.cos(self.s.pitch), np.sin(self.s.pitch)
        return _normalize(np.array([cp * sy, sp, cp * cy], dtype=np.float32))

    def camera(self) -> Camera:
        return Camera.look_at(
            position=self.s.camera_pos,
            target=self.s.camera_pos + self.forward(),
            up=self.s.up,
            fov_y_degrees=float(self.s.fov_y),
            near=float(self.s.near),
            far=float(self.s.far),
        )

    def update_camera(self, dt: float) -> None:
        self.s.move_speed = float(self.c("move_speed").value)
        self.s.fov_y = float(self.c("fov").value)
        if abs(self.s.scroll_delta) > 1e-5:
            self.s.move_speed = float(np.clip(self.s.move_speed * np.power(1.1, self.s.scroll_delta), 0.1, 20.0))
            self.c("move_speed").value = self.s.move_speed
            self.s.scroll_delta = 0.0
        target_rot = self.s.mouse_delta * self.s.look_speed if self.s.mouse_left else np.zeros((2,), dtype=np.float32)
        self.s.rot_vel += (target_rot - self.s.rot_vel) * min(1.0, 12.0 * dt)
        self.s.mouse_delta[:] = 0.0
        if np.linalg.norm(self.s.rot_vel) > 1e-6:
            self.s.yaw += float(self.s.rot_vel[0])
            self.s.pitch = float(np.clip(self.s.pitch + float(self.s.rot_vel[1]), -float(np.deg2rad(89.0)), float(np.deg2rad(89.0))))
        forward = self.forward()
        right = _normalize(np.cross(self.s.up, forward))
        up = _normalize(np.cross(forward, right))
        move = np.array(
            [
                float(self.s.keys.get(spy.KeyCode.e, False)) - float(self.s.keys.get(spy.KeyCode.q, False)),
                float(self.s.keys.get(spy.KeyCode.d, False)) - float(self.s.keys.get(spy.KeyCode.a, False)),
                float(self.s.keys.get(spy.KeyCode.w, False)) - float(self.s.keys.get(spy.KeyCode.s, False)),
            ],
            dtype=np.float32,
        )
        target_move = move * (self.s.move_speed / max(np.linalg.norm(move), 1e-6)) if np.linalg.norm(move) > 1e-6 else np.zeros((3,), dtype=np.float32)
        self.s.move_vel += (target_move - self.s.move_vel) * min(1.0, 10.0 * dt)
        self.s.camera_pos += (up * self.s.move_vel[0] + right * self.s.move_vel[1] + forward * self.s.move_vel[2]) * dt

    def apply_camera_fit(self, bounds: object) -> None:
        fit = fit_camera(bounds, self.s.fov_y)
        self.s.camera_pos = fit.position
        self.s.near = fit.near
        self.s.far = fit.far
        self.s.move_speed = fit.move_speed
        self.c("move_speed").value = float(fit.move_speed)
        self.s.yaw = 0.0
        self.s.pitch = 0.0
        self.s.move_vel[:] = 0.0
        self.s.rot_vel[:] = 0.0

    def _browse_load_ply(self) -> None:
        path = spy.platform.open_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")])
        if path:
            session.load_scene(self, Path(path))

    def _browse_load_colmap(self) -> None:
        path = spy.platform.choose_folder_dialog()
        if path:
            images_subdir = self.image_subdir_options[int(np.clip(int(self.c("images_subdir").value), 0, len(self.image_subdir_options) - 1))]
            session.load_colmap_dataset(self, Path(path), images_subdir)

    def _reload_scene(self) -> None:
        if self.s.scene_path is not None:
            session.load_scene(self, self.s.scene_path)
        elif self.s.colmap_root is not None:
            images_subdir = self.image_subdir_options[int(np.clip(int(self.c("images_subdir").value), 0, len(self.image_subdir_options) - 1))]
            session.load_colmap_dataset(self, self.s.colmap_root, images_subdir)

    def _initialize_training_scene(self) -> None:
        session.initialize_training_scene(self)

    def _start_training(self) -> None:
        session.start_training(self)

    def _stop_training(self) -> None:
        session.stop_training(self)

    def on_resize(self, width: int, height: int) -> None:
        try:
            self.device.wait()
            if width > 0 and height > 0 and (self.s.renderer.width, self.s.renderer.height) != (int(width), int(height)):
                session.recreate_renderer(self, int(width), int(height))
            self.s.last_resize_exception = ""
            self.s.last_error = ""
        except Exception as exc:
            self.s.last_error = f"Resize failed: {exc}"
            if self.s.last_resize_exception != self.s.last_error:
                print(self.s.last_error)
            self.s.last_resize_exception = self.s.last_error

    def on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if event.type == spy.KeyboardEventType.key_press:
            self.s.keys[event.key] = True
        elif event.type == spy.KeyboardEventType.key_release:
            self.s.keys[event.key] = False

    def on_mouse_event(self, event: spy.MouseEvent) -> None:
        if event.type == spy.MouseEventType.button_down and event.button == spy.MouseButton.left:
            self.s.mouse_left = True
        elif event.type == spy.MouseEventType.button_up and event.button == spy.MouseButton.left:
            self.s.mouse_left = False
        elif event.type == spy.MouseEventType.move:
            if self.s.mx is not None and self.s.my is not None:
                self.s.mouse_delta += np.array([event.pos.x - self.s.mx, event.pos.y - self.s.my], dtype=np.float32)
            self.s.mx, self.s.my = event.pos.x, event.pos.y
        elif event.type == spy.MouseEventType.scroll:
            self.s.scroll_delta += float(event.scroll.y)

    def render(self, render_context: spy.AppWindow.RenderContext) -> None:
        presenter.render_frame(self, render_context)


def main() -> int:
    device = create_default_device(enable_debug_layers=False)
    app = spy.App(device=device)
    SplatViewer(app, width=1280, height=720, max_prepass_memory_mb=4096)
    app.run()
    return 0

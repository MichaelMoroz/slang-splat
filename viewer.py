from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np
import slangpy as spy

from src import create_default_device
from src.common import SHADER_ROOT
from src.renderer import Camera, GaussianRenderer
from src.scene import (
    ColmapReconstruction,
    GaussianInitHyperParams,
    GaussianScene,
    build_training_frames,
    load_colmap_reconstruction,
    load_gaussian_ply,
    suggest_colmap_init_hparams,
)
from src.training import AdamHyperParams, GaussianTrainer, StabilityHyperParams, TrainingHyperParams


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v), 1e-8)


@dataclass(slots=True)
class _SceneCountProxy:
    count: int


class SplatViewer(spy.AppWindow):
    def __init__(
        self,
        app: spy.App,
        width: int = 1280,
        height: int = 720,
        title: str = "Slang Splat Viewer",
        max_prepass_memory_mb: int = 4096,
    ) -> None:
        super().__init__(
            app,
            width=width,
            height=height,
            title=title,
            resizable=True,
            enable_vsync=False,
        )

        self.list_capacity_multiplier = 16
        self.max_prepass_memory_mb = max(int(max_prepass_memory_mb), 1)
        self.renderer = GaussianRenderer(
            self.device,
            width=width,
            height=height,
            list_capacity_multiplier=self.list_capacity_multiplier,
            max_prepass_memory_mb=self.max_prepass_memory_mb,
        )
        self.training_renderer: GaussianRenderer | None = None
        self.debug_renderer: GaussianRenderer | None = None
        self._debug_renderer_size: tuple[int, int] | None = None
        self.scene: GaussianScene | _SceneCountProxy | None = None
        self.scene_path: Path | None = None
        self.stats: dict[str, int | bool | float] = {}

        self.colmap_root: Path | None = None
        self.colmap_recon: ColmapReconstruction | None = None
        self.training_frames = []
        self._colmap_point_positions_buffer: spy.Buffer | None = None
        self._colmap_point_colors_buffer: spy.Buffer | None = None
        self._colmap_point_count = 0
        self.trainer: GaussianTrainer | None = None
        self.training_active = False
        self.loss_debug_view_options = [("rendered", "Rendered"), ("target", "Target"), ("abs_diff", "Abs Diff")]
        self._loss_debug_texture: spy.Texture | None = None
        self._debug_abs_diff_kernel: spy.ComputeKernel | None = None
        self._debug_letterbox_kernel: spy.ComputeKernel | None = None
        self._debug_present_texture: spy.Texture | None = None
        self._synced_step_main: int = -1
        self._synced_step_debug: int = -1
        self._scene_init_signature: tuple[object, ...] | None = None

        self.image_subdir_options = ["images_8", "images_4", "images_2", "images"]
        self.default_image_subdir_index = 1

        self.camera_pos = np.array([0.0, 0.0, -3.0], dtype=np.float32)
        self.yaw = 0.0
        self.pitch = 0.0
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fov_y = 60.0
        self.near = 0.1
        self.far = 120.0
        self.move_speed = 2.0
        self.look_speed = 0.003
        self.background = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.keys: dict[spy.KeyCode, bool] = {}
        self.mouse_left = False
        self.mouse_delta = np.zeros((2,), dtype=np.float32)
        self.scroll_delta = 0.0
        self.move_vel = np.zeros((3,), dtype=np.float32)
        self.rot_vel = np.zeros((2,), dtype=np.float32)
        self._mx: float | None = None
        self._my: float | None = None

        self.last_time = time.perf_counter()
        self.fps_smooth = 60.0
        self.last_error = ""
        self._last_resize_exception = ""
        self._last_render_exception = ""

        self._create_debug_shaders()
        self._build_ui()

    def _reset_loss_debug_state(self) -> None:
        self._loss_debug_texture = None
        self._debug_present_texture = None
        if self.debug_renderer is not None:
            old_renderer = self.debug_renderer
            self.debug_renderer = None
            del old_renderer
        self._debug_renderer_size = None
        self._synced_step_debug = -1

    def _create_debug_shaders(self) -> None:
        shader_path = str(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang")
        abs_diff_program = self.device.load_program(shader_path, ["csComposeAbsDiffDebug"])
        letterbox_program = self.device.load_program(shader_path, ["csComposeLetterboxDebug"])
        self._debug_abs_diff_kernel = self.device.create_compute_kernel(abs_diff_program)
        self._debug_letterbox_kernel = self.device.create_compute_kernel(letterbox_program)

    def _build_ui(self) -> None:
        log_flags = spy.ui.SliderFlags.logarithmic
        panel = spy.ui.Window(self.screen, "Splat Viewer + Trainer", size=spy.float2(520, 760))
        self.fps_text = spy.ui.Text(panel, "FPS: 0.0")
        self.path_text = spy.ui.Text(panel, "Scene: <none>")
        self.scene_stats_text = spy.ui.Text(panel, "Splats: 0")
        self.render_stats_text = spy.ui.Text(panel, "Generated: 0 | Written: 0")
        self.training_text = spy.ui.Text(panel, "Training: idle")
        self.training_loss_text = spy.ui.Text(panel, "Loss: n/a")
        self.error_text = spy.ui.Text(panel, "")

        load_group = spy.ui.Group(panel, "Main")
        spy.ui.Button(load_group, "Load PLY...", callback=self._browse_load_ply)
        spy.ui.Button(load_group, "Load COLMAP...", callback=self._browse_load_colmap)
        spy.ui.Button(load_group, "Reload", callback=self._reload_scene)
        self.images_subdir_slider = spy.ui.SliderInt(
            load_group,
            "Image Dir",
            value=int(self.default_image_subdir_index),
            min=0,
            max=len(self.image_subdir_options) - 1,
        )
        self.images_subdir_text = spy.ui.Text(load_group, "Train images: images_4")
        self.loss_debug_checkbox = spy.ui.CheckBox(load_group, "Visual Loss Debug", value=False)
        self.loss_debug_view_slider = spy.ui.SliderInt(
            load_group,
            "Debug View",
            value=2,
            min=0,
            max=len(self.loss_debug_view_options) - 1,
        )
        self.loss_debug_view_text = spy.ui.Text(load_group, "View: Abs Diff")
        self.loss_debug_frame_slider = spy.ui.SliderInt(load_group, "Debug Frame", value=0, min=0, max=10000)
        self.loss_debug_frame_text = spy.ui.Text(load_group, "Frame: <none>")
        spy.ui.Button(load_group, "Reinitialize Gaussians", callback=self._initialize_training_scene)
        spy.ui.Button(load_group, "Start Training", callback=self._start_training)
        spy.ui.Button(load_group, "Stop Training", callback=self._stop_training)

        cam_group = spy.ui.Group(panel, "Camera")
        self.move_speed_slider = spy.ui.SliderFloat(
            cam_group,
            "Move Speed",
            value=float(self.move_speed),
            min=0.1,
            max=20.0,
            flags=log_flags,
            format="%.3g",
        )
        self.fov_slider = spy.ui.SliderFloat(
            cam_group,
            "FOV",
            value=float(self.fov_y),
            min=25.0,
            max=100.0,
        )

        init_group = spy.ui.Group(panel, "Train Init")
        self.gaussian_count_slider = spy.ui.SliderInt(
            init_group,
            "Gaussian Count",
            value=50000,
            min=1000,
            max=2000000,
            flags=log_flags,
        )
        self.seed_slider = spy.ui.SliderInt(init_group, "Seed", value=1234, min=0, max=1000000)
        self.init_pos_jitter_slider = spy.ui.InputFloat(
            init_group,
            "Pos Jitter",
            value=0.01,
            step=1e-4,
            step_fast=1e-3,
            format="%.6f",
        )
        self.init_scale_slider = spy.ui.InputFloat(
            init_group,
            "Base Scale",
            value=0.03,
            step=1e-4,
            step_fast=1e-3,
            format="%.6f",
        )
        self.init_scale_jitter_slider = spy.ui.InputFloat(
            init_group,
            "Scale Jitter",
            value=0.2,
            step=1e-3,
            step_fast=1e-2,
            format="%.4f",
        )
        self.init_opacity_slider = spy.ui.InputFloat(
            init_group,
            "Init Opacity",
            value=0.5,
            step=1e-3,
            step_fast=1e-2,
            format="%.5f",
        )

        opt_group = spy.ui.Group(panel, "Train Optimizer")
        self.lr_base_slider = spy.ui.InputFloat(
            opt_group,
            "Base LR",
            value=1e-3,
            step=1e-5,
            step_fast=1e-4,
            format="%.8f",
        )
        self.lr_pos_mul_slider = spy.ui.InputFloat(
            opt_group,
            "LR Mul Position",
            value=1.0,
            step=1e-2,
            step_fast=1e-1,
            format="%.8f",
        )
        self.lr_scale_mul_slider = spy.ui.InputFloat(
            opt_group,
            "LR Mul Scale",
            value=1.0,
            step=1e-2,
            step_fast=1e-1,
            format="%.8f",
        )
        self.lr_rot_mul_slider = spy.ui.InputFloat(
            opt_group,
            "LR Mul Rotation",
            value=1.0,
            step=1e-2,
            step_fast=1e-1,
            format="%.8f",
        )
        self.lr_color_mul_slider = spy.ui.InputFloat(
            opt_group,
            "LR Mul Color",
            value=1.0,
            step=1e-2,
            step_fast=1e-1,
            format="%.8f",
        )
        self.lr_opacity_mul_slider = spy.ui.InputFloat(
            opt_group,
            "LR Mul Opacity",
            value=1.0,
            step=1e-2,
            step_fast=1e-1,
            format="%.8f",
        )
        self.beta1_slider = spy.ui.InputFloat(opt_group, "Beta1", value=0.9, step=1e-3, step_fast=1e-2, format="%.6f")
        self.beta2_slider = spy.ui.InputFloat(opt_group, "Beta2", value=0.999, step=1e-4, step_fast=1e-3, format="%.6f")
        self.eps_slider = spy.ui.InputFloat(
            opt_group,
            "Adam Eps",
            value=1e-8,
            step=1e-9,
            step_fast=1e-8,
            format="%.10f",
        )
        self.scale_l2_slider = spy.ui.InputFloat(
            opt_group,
            "Scale L2",
            value=1e-3,
            step=1e-5,
            step_fast=1e-4,
            format="%.8f",
        )
        self.scale_aniso_slider = spy.ui.InputFloat(
            opt_group,
            "Scale Aniso",
            value=1e-3,
            step=1e-3,
            step_fast=1e-2,
            format="%.6f",
        )
        self.mcmc_pos_noise_enabled_checkbox = spy.ui.CheckBox(opt_group, "MCMC Pos Noise", value=True)
        self.mcmc_pos_noise_scale_slider = spy.ui.InputFloat(
            opt_group,
            "MCMC Noise Scale",
            value=1.0,
            step=1e-3,
            step_fast=1e-2,
            format="%.6f",
        )
        self.mcmc_opacity_k_slider = spy.ui.InputFloat(
            opt_group,
            "MCMC Opacity K",
            value=100.0,
            step=0.5,
            step_fast=5.0,
            format="%.4f",
        )
        self.mcmc_opacity_t_slider = spy.ui.InputFloat(
            opt_group,
            "MCMC Opacity T",
            value=0.995,
            step=1e-4,
            step_fast=1e-3,
            format="%.6f",
        )
        self.low_quality_reinit_checkbox = spy.ui.CheckBox(opt_group, "Low-Quality Reinit", value=True)
        self.grad_clip_slider = spy.ui.InputFloat(
            opt_group,
            "Grad Clip",
            value=10.0,
            step=0.1,
            step_fast=1.0,
            format="%.4f",
        )
        self.grad_norm_clip_slider = spy.ui.InputFloat(
            opt_group,
            "Grad Norm Clip",
            value=10.0,
            step=0.1,
            step_fast=1.0,
            format="%.4f",
        )
        self.max_update_slider = spy.ui.InputFloat(
            opt_group,
            "Max Update",
            value=0.05,
            step=1e-4,
            step_fast=1e-3,
            format="%.8f",
        )

        stab_group = spy.ui.Group(panel, "Train Stability")
        self.min_scale_slider = spy.ui.InputFloat(
            stab_group,
            "Min Scale",
            value=1e-3,
            step=1e-5,
            step_fast=1e-4,
            format="%.8f",
        )
        self.max_scale_slider = spy.ui.InputFloat(
            stab_group,
            "Max Scale",
            value=3.0,
            step=1e-2,
            step_fast=0.1,
            format="%.5f",
        )
        self.min_opacity_slider = spy.ui.InputFloat(
            stab_group,
            "Min Opacity",
            value=1e-4,
            step=1e-5,
            step_fast=1e-4,
            format="%.8f",
        )
        self.max_opacity_slider = spy.ui.InputFloat(
            stab_group,
            "Max Opacity",
            value=0.9999,
            step=1e-4,
            step_fast=1e-3,
            format="%.6f",
        )
        self.reinit_threshold_text = spy.ui.Text(stab_group, "Reinit thresholds: Min Scale + Min Opacity")
        self.position_abs_max_slider = spy.ui.InputFloat(
            stab_group,
            "Pos Abs Max",
            value=1e4,
            step=10.0,
            step_fast=100.0,
            format="%.3f",
        )
        self.train_near_slider = spy.ui.InputFloat(
            stab_group,
            "Train Near",
            value=0.1,
            step=1e-3,
            step_fast=1e-2,
            format="%.6f",
        )
        self.train_far_slider = spy.ui.InputFloat(
            stab_group,
            "Train Far",
            value=120.0,
            step=1.0,
            step_fast=10.0,
            format="%.3f",
        )

        params_group = spy.ui.Group(panel, "Render Params")
        self.radius_slider = spy.ui.SliderFloat(
            params_group,
            "Radius Scale",
            value=float(self.renderer.radius_scale),
            min=0.5,
            max=4.0,
            flags=log_flags,
            format="%.3g",
        )
        self.alpha_slider = spy.ui.SliderFloat(
            params_group,
            "Alpha Cutoff",
            value=float(self.renderer.alpha_cutoff),
            min=0.0001,
            max=0.1,
            flags=log_flags,
            format="%.2e",
        )
        self.max_steps_slider = spy.ui.SliderInt(
            params_group,
            "Max Splat Steps",
            value=int(self.renderer.max_splat_steps),
            min=16,
            max=32768,
        )
        self.trans_slider = spy.ui.SliderFloat(
            params_group,
            "Trans Threshold",
            value=float(self.renderer.transmittance_threshold),
            min=0.001,
            max=0.2,
            flags=log_flags,
            format="%.2e",
        )
        self.sampled5_safety_slider = spy.ui.SliderFloat(
            params_group,
            "MVEE Safety",
            value=float(self.renderer.sampled5_safety_scale),
            min=1.0,
            max=1.2,
        )
        self.debug_ellipse_checkbox = spy.ui.CheckBox(
            params_group,
            "Debug Ellipse Outlines",
            value=bool(self.renderer.debug_show_ellipses),
        )
        self.debug_processed_count_checkbox = spy.ui.CheckBox(
            params_group,
            "Debug Processed Count",
            value=bool(self.renderer.debug_show_processed_count),
        )

        spy.ui.Text(panel, "Controls: LMB drag=look | WASDQE=move | Wheel=speed")

    def on_resize(self, width: int, height: int) -> None:
        try:
            self.device.wait()
            if width <= 0 or height <= 0:
                return
            if self.renderer.width == int(width) and self.renderer.height == int(height):
                return
            self._recreate_renderer(int(width), int(height))
            self._last_resize_exception = ""
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"Resize failed: {exc}"
            if self._last_resize_exception != self.last_error:
                print(self.last_error)
            self._last_resize_exception = self.last_error

    def on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if event.type == spy.KeyboardEventType.key_press:
            self.keys[event.key] = True
        elif event.type == spy.KeyboardEventType.key_release:
            self.keys[event.key] = False

    def on_mouse_event(self, event: spy.MouseEvent) -> None:
        if event.type == spy.MouseEventType.button_down and event.button == spy.MouseButton.left:
            self.mouse_left = True
        elif event.type == spy.MouseEventType.button_up and event.button == spy.MouseButton.left:
            self.mouse_left = False
        elif event.type == spy.MouseEventType.move:
            if self._mx is not None and self._my is not None:
                self.mouse_delta += np.array([event.pos.x - self._mx, event.pos.y - self._my], dtype=np.float32)
            self._mx = event.pos.x
            self._my = event.pos.y
        elif event.type == spy.MouseEventType.scroll:
            self.scroll_delta += float(event.scroll.y)

    def _create_renderer(self, width: int, height: int, allow_debug_overlays: bool) -> GaussianRenderer:
        return GaussianRenderer(
            self.device,
            width=width,
            height=height,
            radius_scale=float(self.radius_slider.value),
            alpha_cutoff=float(self.alpha_slider.value),
            max_splat_steps=int(self.max_steps_slider.value),
            transmittance_threshold=float(self.trans_slider.value),
            sampled5_safety_scale=float(self.sampled5_safety_slider.value),
            list_capacity_multiplier=self.list_capacity_multiplier,
            max_prepass_memory_mb=self.max_prepass_memory_mb,
            debug_show_ellipses=bool(self.debug_ellipse_checkbox.value) if allow_debug_overlays else False,
            debug_show_processed_count=bool(self.debug_processed_count_checkbox.value) if allow_debug_overlays else False,
        )

    def _recreate_renderer(self, width: int, height: int) -> None:
        old_renderer = self.renderer
        self.renderer = self._create_renderer(width, height, allow_debug_overlays=True)
        del old_renderer
        if isinstance(self.scene, GaussianScene):
            self.renderer.set_scene(self.scene)
        self._synced_step_main = -1
        self._reset_loss_debug_state()

    def _ensure_training_renderer(self, width: int, height: int) -> GaussianRenderer:
        if self.training_renderer is not None and self.training_renderer.width == int(width) and self.training_renderer.height == int(height):
            return self.training_renderer
        old_renderer = self.training_renderer
        self.training_renderer = self._create_renderer(int(width), int(height), allow_debug_overlays=False)
        if isinstance(self.scene, GaussianScene):
            self.training_renderer.set_scene(self.scene)
        if old_renderer is not None:
            del old_renderer
        self._synced_step_main = -1
        self._synced_step_debug = -1
        return self.training_renderer

    def _ensure_debug_renderer(self, width: int, height: int) -> GaussianRenderer:
        target_size = (int(width), int(height))
        if self.debug_renderer is not None and self._debug_renderer_size == target_size:
            return self.debug_renderer
        old_renderer = self.debug_renderer
        self.debug_renderer = self._create_renderer(target_size[0], target_size[1], allow_debug_overlays=False)
        self._debug_renderer_size = target_size
        if isinstance(self.scene, GaussianScene):
            self.debug_renderer.set_scene(self.scene)
        if old_renderer is not None:
            del old_renderer
        self._synced_step_debug = -1
        return self.debug_renderer

    def _sync_scene_from_training_renderer(self, dst_renderer: GaussianRenderer, target: str, force: bool = False) -> None:
        if self.training_renderer is None or self.trainer is None:
            return
        step = int(self.trainer.state.step)
        if target == "main" and not force and self._synced_step_main == step:
            return
        if target == "debug" and not force and self._synced_step_debug == step:
            return
        enc = self.device.create_command_encoder()
        self.training_renderer.copy_scene_state_to(enc, dst_renderer)
        self.device.submit_command_buffer(enc.finish())
        if target == "main":
            self._synced_step_main = step
        else:
            self._synced_step_debug = step

    def _forward(self) -> np.ndarray:
        yaw = self.yaw
        pitch = self.pitch
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        f = np.array([cp * sy, sp, cp * cy], dtype=np.float32)
        return _normalize(f)

    def _camera(self) -> Camera:
        forward = self._forward()
        return Camera.look_at(
            position=self.camera_pos,
            target=self.camera_pos + forward,
            up=self.up,
            fov_y_degrees=float(self.fov_y),
            near=float(self.near),
            far=float(self.far),
        )

    def _update_camera(self, dt: float) -> None:
        self.move_speed = float(self.move_speed_slider.value)
        self.fov_y = float(self.fov_slider.value)
        if abs(self.scroll_delta) > 1e-5:
            self.move_speed = float(np.clip(self.move_speed * np.power(1.1, self.scroll_delta), 0.1, 20.0))
            self.move_speed_slider.value = self.move_speed
            self.scroll_delta = 0.0

        target_rot = (
            np.array([self.mouse_delta[0], self.mouse_delta[1]], dtype=np.float32) * self.look_speed
            if self.mouse_left
            else np.zeros((2,), dtype=np.float32)
        )
        self.rot_vel += (target_rot - self.rot_vel) * min(1.0, 12.0 * dt)
        self.mouse_delta[:] = 0.0
        if np.linalg.norm(self.rot_vel) > 1e-6:
            self.yaw += float(self.rot_vel[0])
            self.pitch += float(self.rot_vel[1])
            limit = float(np.deg2rad(89.0))
            self.pitch = float(np.clip(self.pitch, -limit, limit))

        forward = self._forward()
        right = _normalize(np.cross(self.up, forward))
        up = _normalize(np.cross(forward, right))
        move = np.array(
            [
                float(self.keys.get(spy.KeyCode.e, False)) - float(self.keys.get(spy.KeyCode.q, False)),
                float(self.keys.get(spy.KeyCode.d, False)) - float(self.keys.get(spy.KeyCode.a, False)),
                float(self.keys.get(spy.KeyCode.w, False)) - float(self.keys.get(spy.KeyCode.s, False)),
            ],
            dtype=np.float32,
        )
        move_len = np.linalg.norm(move)
        if move_len > 1e-6:
            target_move = move * (self.move_speed / move_len)
        else:
            target_move = np.zeros((3,), dtype=np.float32)
        self.move_vel += (target_move - self.move_vel) * min(1.0, 10.0 * dt)
        self.camera_pos += (up * self.move_vel[0] + right * self.move_vel[1] + forward * self.move_vel[2]) * dt

    def _browse_load_ply(self) -> None:
        path = spy.platform.open_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")])
        if path:
            self.load_scene(Path(path))

    def _browse_load_colmap(self) -> None:
        path = spy.platform.choose_folder_dialog()
        if path:
            self.load_colmap_dataset(Path(path), self._selected_images_subdir())

    def _reload_scene(self) -> None:
        if self.scene_path is not None:
            self.load_scene(self.scene_path)
            return
        if self.colmap_root is not None:
            self.load_colmap_dataset(self.colmap_root, self._selected_images_subdir())

    def _selected_images_subdir(self) -> str:
        idx = int(np.clip(int(self.images_subdir_slider.value), 0, len(self.image_subdir_options) - 1))
        return self.image_subdir_options[idx]

    def _selected_loss_debug_view(self) -> tuple[str, str]:
        idx = int(np.clip(int(self.loss_debug_view_slider.value), 0, len(self.loss_debug_view_options) - 1))
        return self.loss_debug_view_options[idx]

    def _selected_loss_debug_frame_index(self) -> int:
        if not self.training_frames:
            return 0
        return int(np.clip(int(self.loss_debug_frame_slider.value), 0, len(self.training_frames) - 1))

    def _update_debug_frame_slider_range(self) -> None:
        max_index = max(len(self.training_frames) - 1, 0)
        try:
            self.loss_debug_frame_slider.min = 0
            self.loss_debug_frame_slider.max = int(max_index)
        except Exception:
            pass
        self.loss_debug_frame_slider.value = int(np.clip(int(self.loss_debug_frame_slider.value), 0, max_index))

    def _sync_render_params_to_renderer(self, renderer: GaussianRenderer, allow_debug_overlays: bool) -> None:
        renderer.radius_scale = float(self.radius_slider.value)
        renderer.alpha_cutoff = float(self.alpha_slider.value)
        renderer.max_splat_steps = int(self.max_steps_slider.value)
        renderer.transmittance_threshold = float(self.trans_slider.value)
        renderer.sampled5_safety_scale = float(self.sampled5_safety_slider.value)
        if allow_debug_overlays:
            renderer.debug_show_ellipses = bool(self.debug_ellipse_checkbox.value)
            renderer.debug_show_processed_count = bool(self.debug_processed_count_checkbox.value)
        else:
            renderer.debug_show_ellipses = False
            renderer.debug_show_processed_count = False

    def _ensure_loss_debug_texture(self, width: int, height: int) -> spy.Texture:
        if (
            self._loss_debug_texture is not None
            and int(self._loss_debug_texture.width) == int(width)
            and int(self._loss_debug_texture.height) == int(height)
        ):
            return self._loss_debug_texture
        self._loss_debug_texture = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=int(width),
            height=int(height),
            usage=(
                spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access
                | spy.TextureUsage.copy_destination
            ),
        )
        return self._loss_debug_texture

    def _ensure_debug_present_texture(self, width: int, height: int) -> spy.Texture:
        if (
            self._debug_present_texture is not None
            and int(self._debug_present_texture.width) == int(width)
            and int(self._debug_present_texture.height) == int(height)
        ):
            return self._debug_present_texture
        self._debug_present_texture = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=int(width),
            height=int(height),
            usage=(
                spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access
                | spy.TextureUsage.copy_destination
            ),
        )
        return self._debug_present_texture

    def _dispatch_debug_abs_diff(
        self,
        encoder: spy.CommandEncoder,
        rendered_tex: spy.Texture,
        target_tex: spy.Texture,
        width: int,
        height: int,
    ) -> spy.Texture:
        if self._debug_abs_diff_kernel is None:
            raise RuntimeError("Debug abs-diff kernel is not initialized.")
        out_tex = self._ensure_loss_debug_texture(width, height)
        self._debug_abs_diff_kernel.dispatch(
            thread_count=spy.uint3(int(width), int(height), 1),
            vars={
                "g_DebugRendered": rendered_tex,
                "g_DebugTarget": target_tex,
                "g_DebugOutput": out_tex,
                "g_DebugWidth": int(width),
                "g_DebugHeight": int(height),
                "g_Stability": {
                    "gradComponentClip": 0.0,
                    "gradNormClip": 0.0,
                    "maxUpdate": 0.0,
                    "minScale": 0.0,
                    "maxScale": 0.0,
                    "minOpacity": 0.0,
                    "maxOpacity": 0.0,
                    "positionAbsMax": 0.0,
                    "hugeValue": 1e8,
                },
            },
            command_encoder=encoder,
        )
        return out_tex

    def _dispatch_debug_letterbox(
        self,
        encoder: spy.CommandEncoder,
        source_tex: spy.Texture,
        source_width: int,
        source_height: int,
        output_width: int,
        output_height: int,
    ) -> spy.Texture:
        if self._debug_letterbox_kernel is None:
            raise RuntimeError("Debug letterbox kernel is not initialized.")
        out_tex = self._ensure_debug_present_texture(output_width, output_height)
        self._debug_letterbox_kernel.dispatch(
            thread_count=spy.uint3(int(output_width), int(output_height), 1),
            vars={
                "g_LetterboxSource": source_tex,
                "g_LetterboxOutput": out_tex,
                "g_LetterboxSourceWidth": int(source_width),
                "g_LetterboxSourceHeight": int(source_height),
                "g_LetterboxOutputWidth": int(output_width),
                "g_LetterboxOutputHeight": int(output_height),
            },
            command_encoder=encoder,
        )
        return out_tex

    def _recenter_camera(self, scene: GaussianScene) -> None:
        pos = np.asarray(scene.positions, dtype=np.float32)
        scales = np.asarray(scene.scales, dtype=np.float32)
        opac = np.asarray(scene.opacities, dtype=np.float32).reshape(-1)
        finite = np.isfinite(pos).all(axis=1)
        if not np.any(finite):
            center = np.zeros((3,), dtype=np.float32)
            radius = 1.0
        else:
            pos_f = pos[finite]
            scales_f = scales[finite]
            opac_f = np.clip(opac[finite], 1e-3, 1.0)
            core_mask = opac_f > np.quantile(opac_f, 0.7)
            if np.count_nonzero(core_mask) > 2048:
                pos_c = pos_f[core_mask]
                scales_c = scales_f[core_mask]
                opac_c = opac_f[core_mask]
            else:
                pos_c = pos_f
                scales_c = scales_f
                opac_c = opac_f

            weight_sum = float(np.sum(opac_c))
            if weight_sum > 1e-6:
                center = (np.sum(pos_c * opac_c[:, None], axis=0) / weight_sum).astype(np.float32)
            else:
                center = np.mean(pos_c, axis=0).astype(np.float32)

            rel = pos_c - center[None, :]
            dist = np.linalg.norm(rel, axis=1)
            splat_extent = np.max(scales_c, axis=1)
            effective = dist + 2.0 * splat_extent
            core_radius = max(float(np.percentile(effective, 90.0)), 1.0)
            q_lo = np.percentile(pos_c, 5.0, axis=0)
            q_hi = np.percentile(pos_c, 95.0, axis=0)
            quant_extent = 0.5 * np.linalg.norm((q_hi - q_lo).astype(np.float32))
            radius = max(core_radius, float(quant_extent), 1.0)

        fov_rad = float(np.deg2rad(self.fov_y))
        fit_distance = radius / max(np.tan(0.5 * fov_rad), 1e-4)
        cam_distance = max(fit_distance * 0.95, radius * 1.35, 1.0)
        self.camera_pos = center + np.array([0.0, 0.0, -cam_distance], dtype=np.float32)
        self.near = max(0.01, cam_distance * 0.0015)
        self.far = max(cam_distance + radius * 4.0, 80.0)
        self.move_speed = max(0.25, radius * 0.15)
        self.move_speed_slider.value = float(self.move_speed)
        self.yaw = 0.0
        self.pitch = 0.0
        self.move_vel[:] = 0.0
        self.rot_vel[:] = 0.0

    def _recenter_camera_from_points(self, points_xyz: np.ndarray) -> None:
        pos = np.asarray(points_xyz, dtype=np.float32)
        finite = np.isfinite(pos).all(axis=1)
        if not np.any(finite):
            center = np.zeros((3,), dtype=np.float32)
            radius = 1.0
        else:
            pos_f = pos[finite]
            center = np.mean(pos_f, axis=0).astype(np.float32)
            rel = pos_f - center[None, :]
            dist = np.linalg.norm(rel, axis=1)
            radius = max(float(np.percentile(dist, 90.0)), 1.0)

        fov_rad = float(np.deg2rad(self.fov_y))
        fit_distance = radius / max(np.tan(0.5 * fov_rad), 1e-4)
        cam_distance = max(fit_distance * 0.95, radius * 1.35, 1.0)
        self.camera_pos = center + np.array([0.0, 0.0, -cam_distance], dtype=np.float32)
        self.near = max(0.01, cam_distance * 0.0015)
        self.far = max(cam_distance + radius * 4.0, 80.0)
        self.move_speed = max(0.25, radius * 0.15)
        self.move_speed_slider.value = float(self.move_speed)
        self.yaw = 0.0
        self.pitch = 0.0
        self.move_vel[:] = 0.0
        self.rot_vel[:] = 0.0

    def _upload_colmap_pointcloud_buffers(self, recon: ColmapReconstruction) -> None:
        xyz_table = getattr(recon, "point_xyz_table", None)
        rgb_table = getattr(recon, "point_rgb_table", None)
        if xyz_table is None or rgb_table is None:
            points = list(recon.points3d.values())
            if not points:
                raise RuntimeError("COLMAP point tables are missing and points3d is empty.")
            xyz = np.ascontiguousarray(np.stack([point.xyz for point in points], axis=0), dtype=np.float32)
            rgb = np.ascontiguousarray(np.stack([point.rgb for point in points], axis=0), dtype=np.float32)
        else:
            xyz = np.ascontiguousarray(xyz_table, dtype=np.float32)
            rgb = np.ascontiguousarray(rgb_table, dtype=np.float32)
        if xyz.shape[0] != rgb.shape[0] or xyz.shape[0] == 0:
            raise RuntimeError("COLMAP point tables are empty or mismatched.")
        pos4 = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        col4 = np.zeros((rgb.shape[0], 4), dtype=np.float32)
        pos4[:, :3] = xyz
        col4[:, :3] = rgb
        usage = (
            spy.BufferUsage.shader_resource
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        )
        self._colmap_point_positions_buffer = self.device.create_buffer(size=xyz.shape[0] * 16, usage=usage)
        self._colmap_point_colors_buffer = self.device.create_buffer(size=rgb.shape[0] * 16, usage=usage)
        self._colmap_point_positions_buffer.copy_from_numpy(pos4)
        self._colmap_point_colors_buffer.copy_from_numpy(col4)
        self._colmap_point_count = int(xyz.shape[0])

    def _apply_dataset_init_defaults(self) -> None:
        if self.colmap_recon is None:
            return
        gaussian_count = int(np.clip(int(self.gaussian_count_slider.value), 1, 10_000_000))
        suggested = suggest_colmap_init_hparams(self.colmap_recon, gaussian_count)
        self.init_pos_jitter_slider.value = float(suggested.position_jitter_std)
        self.init_scale_slider.value = float(suggested.base_scale)
        self.init_scale_jitter_slider.value = float(suggested.scale_jitter_ratio)
        self.init_opacity_slider.value = float(suggested.initial_opacity)

    def load_scene(self, path: Path) -> None:
        try:
            scene = load_gaussian_ply(path)
            self.scene = scene
            self.scene_path = path
            self.colmap_root = None
            self.colmap_recon = None
            self._scene_init_signature = None
            self.training_frames = []
            self._colmap_point_positions_buffer = None
            self._colmap_point_colors_buffer = None
            self._colmap_point_count = 0
            self.trainer = None
            self.training_active = False
            if self.training_renderer is not None:
                old_renderer = self.training_renderer
                self.training_renderer = None
                del old_renderer
            self._update_debug_frame_slider_range()
            self._reset_loss_debug_state()
            self.renderer.set_scene(scene)
            self._recenter_camera(scene)
            self.last_error = ""
            print(f"Loaded scene: {path} ({scene.count:,} splats)")
        except Exception as exc:
            self.last_error = str(exc)
            print(f"Failed to load scene {path}: {exc}")

    def load_colmap_dataset(self, root: Path, images_subdir: str) -> None:
        try:
            recon = load_colmap_reconstruction(root)
            frames = build_training_frames(recon, images_subdir=images_subdir)
            self.colmap_root = Path(root)
            self.colmap_recon = recon
            self.training_frames = frames
            self._upload_colmap_pointcloud_buffers(recon)
            self.scene_path = None
            self.trainer = None
            self.training_active = False
            if self.training_renderer is not None:
                old_renderer = self.training_renderer
                self.training_renderer = None
                del old_renderer
            self._update_debug_frame_slider_range()
            self._reset_loss_debug_state()
            self._apply_dataset_init_defaults()
            print(f"Loaded COLMAP: {root} frames={len(frames)} images={images_subdir}")
            self.last_error = ""
            recenter_xyz = getattr(recon, "point_xyz_table", None)
            if recenter_xyz is None:
                points = list(recon.points3d.values())
                if points:
                    recenter_xyz = np.stack([point.xyz for point in points], axis=0).astype(np.float32)
                else:
                    recenter_xyz = np.zeros((0, 3), dtype=np.float32)
            self._recenter_camera_from_points(recenter_xyz)
            self._initialize_training_scene()
        except Exception as exc:
            self.last_error = str(exc)
            print(f"Failed to load COLMAP {root}: {exc}")

    def _collect_training_hparams(self) -> tuple[AdamHyperParams, StabilityHyperParams, TrainingHyperParams]:
        clamp = lambda v, lo, hi: float(np.clip(float(v), float(lo), float(hi)))
        base_lr = clamp(self.lr_base_slider.value, 1e-8, 1.0)
        lr_pos_mul = clamp(self.lr_pos_mul_slider.value, 0.1, 10.0)
        lr_scale_mul = clamp(self.lr_scale_mul_slider.value, 0.1, 10.0)
        lr_rot_mul = clamp(self.lr_rot_mul_slider.value, 0.1, 10.0)
        lr_color_mul = clamp(self.lr_color_mul_slider.value, 0.1, 10.0)
        lr_opacity_mul = clamp(self.lr_opacity_mul_slider.value, 0.1, 10.0)
        adam = AdamHyperParams(
            position_lr=base_lr * lr_pos_mul,
            scale_lr=base_lr * lr_scale_mul,
            rotation_lr=base_lr * lr_rot_mul,
            color_lr=base_lr * lr_color_mul,
            opacity_lr=base_lr * lr_opacity_mul,
            beta1=clamp(self.beta1_slider.value, 0.0, 0.99999),
            beta2=clamp(self.beta2_slider.value, 0.0, 0.999999),
            epsilon=clamp(self.eps_slider.value, 1e-12, 1e-2),
        )
        stability = StabilityHyperParams(
            grad_component_clip=clamp(self.grad_clip_slider.value, 1e-5, 1e6),
            grad_norm_clip=clamp(self.grad_norm_clip_slider.value, 1e-5, 1e6),
            max_update=clamp(self.max_update_slider.value, 1e-8, 10.0),
            min_scale=clamp(self.min_scale_slider.value, 1e-8, 1e3),
            max_scale=clamp(self.max_scale_slider.value, 1e-8, 1e4),
            min_opacity=clamp(self.min_opacity_slider.value, 0.0, 1.0),
            max_opacity=clamp(self.max_opacity_slider.value, 0.0, 1.0),
            position_abs_max=clamp(self.position_abs_max_slider.value, 1e-3, 1e9),
            loss_grad_clip=clamp(self.grad_clip_slider.value, 1e-5, 1e6),
        )
        if stability.max_scale < stability.min_scale:
            stability.max_scale = stability.min_scale
        if stability.max_opacity < stability.min_opacity:
            stability.max_opacity = stability.min_opacity
        training = TrainingHyperParams(
            background=tuple(float(v) for v in self.background.tolist()),
            near=clamp(self.train_near_slider.value, 1e-6, 1e4),
            far=clamp(self.train_far_slider.value, 1e-5, 1e6),
            ema_decay=0.95,
            scale_l2_weight=clamp(self.scale_l2_slider.value, 0.0, 1e4),
            scale_aniso_weight=clamp(self.scale_aniso_slider.value, 0.0, 1e4),
            mcmc_position_noise_enabled=bool(self.mcmc_pos_noise_enabled_checkbox.value),
            mcmc_position_noise_scale=clamp(self.mcmc_pos_noise_scale_slider.value, 0.0, 1e4),
            mcmc_opacity_gate_sharpness=clamp(self.mcmc_opacity_k_slider.value, 0.0, 1e6),
            mcmc_opacity_gate_center=clamp(self.mcmc_opacity_t_slider.value, 0.0, 1.0),
            low_quality_reinit_enabled=bool(self.low_quality_reinit_checkbox.value),
        )
        if training.far <= training.near:
            training.far = training.near + 1e-3
        return adam, stability, training

    def _collect_init_hparams(self) -> tuple[GaussianInitHyperParams, int, int]:
        clamp = lambda v, lo, hi: float(np.clip(float(v), float(lo), float(hi)))
        init_hparams = GaussianInitHyperParams(
            position_jitter_std=clamp(self.init_pos_jitter_slider.value, 0.0, 10.0),
            base_scale=clamp(self.init_scale_slider.value, 1e-8, 1e3),
            scale_jitter_ratio=clamp(self.init_scale_jitter_slider.value, 0.0, 10.0),
            initial_opacity=clamp(self.init_opacity_slider.value, 0.0, 1.0),
            color_jitter_std=0.0,
        )
        gaussian_count = int(np.clip(int(self.gaussian_count_slider.value), 1, 10_000_000))
        seed = int(np.clip(int(self.seed_slider.value), 0, 1_000_000_000))
        return init_hparams, gaussian_count, seed

    def _current_scene_init_signature(self) -> tuple[object, ...] | None:
        if self.colmap_root is None or self.colmap_recon is None or not self.training_frames:
            return None
        init_hparams, gaussian_count, seed = self._collect_init_hparams()
        return (
            str(self.colmap_root.resolve()),
            int(len(self.training_frames)),
            int(gaussian_count),
            int(seed),
            round(float(init_hparams.position_jitter_std), 8),
            round(float(init_hparams.base_scale), 8),
            round(float(init_hparams.scale_jitter_ratio), 8),
            round(float(init_hparams.initial_opacity), 8),
        )

    def _initialize_training_scene(self) -> None:
        if self.colmap_recon is None or not self.training_frames:
            self.last_error = "Load COLMAP dataset first."
            return
        try:
            init_hparams, gaussian_count, seed = self._collect_init_hparams()
            if (
                self._colmap_point_positions_buffer is None
                or self._colmap_point_colors_buffer is None
                or self._colmap_point_count <= 0
            ):
                raise RuntimeError("COLMAP pointcloud buffers are not initialized.")
            frame_width = int(self.training_frames[0].width)
            frame_height = int(self.training_frames[0].height)
            training_renderer = self._ensure_training_renderer(frame_width, frame_height)
            adam, stability, training = self._collect_training_hparams()
            if self.trainer is None:
                self.trainer = GaussianTrainer(
                    device=self.device,
                    renderer=training_renderer,
                    scene=None,
                    scene_count=gaussian_count,
                    upload_initial_scene=False,
                    frames=self.training_frames,
                    adam_hparams=adam,
                    stability_hparams=stability,
                    training_hparams=training,
                    seed=seed,
                    init_point_positions_buffer=self._colmap_point_positions_buffer,
                    init_point_colors_buffer=self._colmap_point_colors_buffer,
                    init_point_count=self._colmap_point_count,
                )
            else:
                self.trainer.renderer = training_renderer
                self.trainer.update_hyperparams(adam, stability, training)
            self.trainer.initialize_scene_from_pointcloud(
                splat_count=gaussian_count,
                init_hparams=init_hparams,
                seed=seed,
            )
            self.scene = _SceneCountProxy(gaussian_count)
            enc = self.device.create_command_encoder()
            training_renderer.copy_scene_state_to(enc, self.renderer)
            self.device.submit_command_buffer(enc.finish())
            self.training_active = False
            self._synced_step_main = -1
            self._synced_step_debug = -1
            self._scene_init_signature = self._current_scene_init_signature()
            self._update_debug_frame_slider_range()
            self._reset_loss_debug_state()
            self.last_error = ""
            print(f"Initialized training scene ({gaussian_count:,} gaussians)")
        except Exception as exc:
            self.last_error = str(exc)
            self.trainer = None
            self.training_active = False
            print(f"Training scene init failed: {exc}")

    def _start_training(self) -> None:
        current_signature = self._current_scene_init_signature()
        if self.trainer is None or (current_signature is not None and current_signature != self._scene_init_signature):
            self._initialize_training_scene()
        self.training_active = self.trainer is not None

    def _stop_training(self) -> None:
        self.training_active = False

    def _apply_render_params(self) -> None:
        self._sync_render_params_to_renderer(self.renderer, allow_debug_overlays=True)
        if self.training_renderer is not None:
            self._sync_render_params_to_renderer(self.training_renderer, allow_debug_overlays=False)
        if self.debug_renderer is not None:
            self._sync_render_params_to_renderer(self.debug_renderer, allow_debug_overlays=False)

    def _apply_training_params(self) -> None:
        if self.trainer is None:
            return
        adam, stability, training = self._collect_training_hparams()
        self.trainer.update_hyperparams(adam, stability, training)

    def _update_ui_text(self, dt: float) -> None:
        self.fps_smooth += (1.0 / max(dt, 1e-5) - self.fps_smooth) * min(dt * 5.0, 1.0)
        self.fps_text.text = f"FPS: {self.fps_smooth:.1f}"
        self.images_subdir_text.text = f"Train images: {self._selected_images_subdir()}"
        self._update_debug_frame_slider_range()
        _, debug_view_label = self._selected_loss_debug_view()
        self.loss_debug_view_text.text = f"View: {debug_view_label}"
        if self.training_frames:
            frame_idx = self._selected_loss_debug_frame_index()
            frame_name = Path(self.training_frames[frame_idx].image_path).name
            self.loss_debug_frame_text.text = f"Frame[{frame_idx}]: {frame_name}"
        else:
            self.loss_debug_frame_text.text = "Frame: <none>"

        if self.scene_path is not None:
            self.path_text.text = f"Scene: {self.scene_path.name} [PLY]"
        elif self.colmap_root is not None:
            self.path_text.text = f"Scene: {self.colmap_root.name} [COLMAP]"
        else:
            self.path_text.text = "Scene: <none>"

        scene_count = self.scene.count if self.scene is not None else 0
        self.scene_stats_text.text = f"Splats: {scene_count:,}"
        if self.stats:
            delayed_tag = " (delayed)" if bool(self.stats.get("stats_latency_frames", 0)) else ""
            validity_tag = "" if bool(self.stats.get("stats_valid", True)) else " [warming]"
            cap_tag = " [cap]" if bool(self.stats.get("capacity_limited", False)) else ""
            self.render_stats_text.text = (
                f"Generated: {int(self.stats['generated_entries']):,} | "
                f"Written: {int(self.stats['written_entries']):,} | "
                f"Overflow: {bool(self.stats['overflow'])}{cap_tag}{delayed_tag}{validity_tag}"
            )
        else:
            self.render_stats_text.text = "Generated: 0 | Written: 0"

        if self.trainer is None:
            self.training_text.text = "Training: not initialized"
            self.training_loss_text.text = "Loss: n/a"
        else:
            state = self.trainer.state
            status = "running" if self.training_active else "paused"
            psnr_text = f"{state.ema_psnr:.2f} dB" if np.isfinite(state.ema_psnr) else "n/a"
            self.training_text.text = (
                f"Training: {status} | step={state.step:,} | frame={state.last_frame_index}"
            )
            self.training_loss_text.text = (
                f"Loss: {state.last_loss:.6e} | EMA: {state.ema_loss:.6e} | PSNR: {psnr_text} | {state.last_instability}"
            )
        self.error_text.text = f"Error: {self.last_error}" if self.last_error else ""

    def render(self, render_context: spy.AppWindow.RenderContext) -> None:
        image = render_context.surface_texture
        encoder = render_context.command_encoder

        now = time.perf_counter()
        dt = max(now - self.last_time, 1e-5)
        self.last_time = now

        self._update_camera(dt)
        self._apply_render_params()
        self._apply_training_params()

        iw = int(image.width)
        ih = int(image.height)
        if self.renderer.width != iw or self.renderer.height != ih:
            self._recreate_renderer(iw, ih)

        if self.scene is not None:
            try:
                capture_loss_debug = bool(self.loss_debug_checkbox.value) and self.trainer is not None and bool(
                    self.training_frames
                )
                if self.training_active and self.trainer is not None:
                    self.trainer.step()
                if capture_loss_debug:
                    frame_idx = self._selected_loss_debug_frame_index()
                    debug_width, debug_height = self.trainer.frame_size(frame_idx)
                    debug_renderer = self._ensure_debug_renderer(debug_width, debug_height)
                    self._sync_scene_from_training_renderer(debug_renderer, target="debug")
                    frame_camera = self.trainer.make_frame_camera(frame_idx, debug_width, debug_height)
                    debug_render_tex, stats = debug_renderer.render_to_texture(
                        frame_camera,
                        background=self.background,
                        read_stats=True,
                    )
                    target_tex = self.trainer.get_frame_target_texture(frame_idx, native_resolution=True)
                    self.stats = stats
                    debug_view_key, _ = self._selected_loss_debug_view()
                    debug_source_tex: spy.Texture
                    if debug_view_key == "rendered":
                        debug_source_tex = debug_render_tex
                    elif debug_view_key == "target":
                        debug_source_tex = target_tex
                    else:
                        debug_source_tex = self._dispatch_debug_abs_diff(
                            encoder=encoder,
                            rendered_tex=debug_render_tex,
                            target_tex=target_tex,
                            width=debug_width,
                            height=debug_height,
                        )
                    present_tex = self._dispatch_debug_letterbox(
                        encoder=encoder,
                        source_tex=debug_source_tex,
                        source_width=debug_width,
                        source_height=debug_height,
                        output_width=iw,
                        output_height=ih,
                    )
                    encoder.blit(image, present_tex)
                else:
                    if self.trainer is not None and self.training_renderer is not None:
                        self._sync_scene_from_training_renderer(self.renderer, target="main")
                    out_tex, stats = self.renderer.render_to_texture(
                        self._camera(),
                        background=self.background,
                        read_stats=True,
                    )
                    self.stats = stats
                    encoder.blit(image, out_tex)
                self._last_render_exception = ""
            except Exception as exc:
                self.training_active = False
                self.last_error = str(exc)
                if self._last_render_exception != self.last_error:
                    print(f"Render/training error: {self.last_error}")
                self._last_render_exception = self.last_error
                encoder.clear_texture_float(image, clear_value=[0.0, 0.0, 0.0, 1.0])
        else:
            encoder.clear_texture_float(image, clear_value=[0.1, 0.1, 0.12, 1.0])

        self._update_ui_text(dt)


def main() -> int:
    device = create_default_device(enable_debug_layers=False)
    app = spy.App(device=device)
    viewer = SplatViewer(
        app,
        width=1280,
        height=720,
        max_prepass_memory_mb=4096,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

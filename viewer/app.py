from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import slangpy as spy
import torch
from module.splatting import SplattingContext
from utility.ply import load_gaussian_ply
from .camera import Camera
from .state import DEBUG_MODE_PROCESSED_COUNT, ViewerState
from .training import TrainingController
from .ui import _WINDOW_TITLE, build_ui, create_toolkit_window

_ROOT = Path(__file__).resolve().parent
_SHADERS = _ROOT / "shaders"
_VIEW_VEC_EPS = 1e-6
_SCROLL_SPEED_BASE = 1.1
_LOOK_SMOOTH = 12.0
_MOVE_SMOOTH = 10.0
_PITCH_LIMIT = math.radians(89.0)
_ALPHA_EPS = 1e-6
_LOSS_DEBUG_VIEW_KEYS = ("rendered", "target", "loss", "abs_diff")
_MAX_SURFACE_DIM = 8192
_MAX_SURFACE_PIXELS = 33_554_432


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return (v / max(norm, _VIEW_VEC_EPS)).astype(np.float32)


def _pack_params(splats: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(splats.copy())
    alpha = np.clip(packed[13], _ALPHA_EPS, 1.0 - _ALPHA_EPS)
    packed[13] = np.log(alpha / np.clip(1.0 - alpha, _ALPHA_EPS, 1.0))
    return packed


def _pack_params_torch(splats: torch.Tensor) -> torch.Tensor:
    packed = splats.contiguous().clone()
    alpha = torch.clamp(packed[13], _ALPHA_EPS, 1.0 - _ALPHA_EPS)
    packed[13] = torch.log(alpha / torch.clamp(1.0 - alpha, _ALPHA_EPS, 1.0))
    return packed.reshape(-1)


def _camera_dict(camera: torch.Tensor, image_size: tuple[int, int]) -> dict[str, Any]:
    c = camera.detach()
    q = c[0:4] / torch.clamp(torch.linalg.norm(c[0:4]), min=1e-12)
    w, x, y, z = [float(v.item()) for v in q]
    rot = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        device=c.device,
        dtype=c.dtype,
    )
    cam_pos = (-rot.T @ c[4:7]).cpu().tolist()
    return {
        "camPos": spy.float3(*cam_pos),
        "camBasis": spy.float3x3(rot.cpu().numpy()),
        "focalPixels": spy.float2(float(c[7].item()), float(c[8].item())),
        "principalPoint": spy.float2(float(c[9].item()), float(c[10].item())),
        "viewport": spy.float2(*map(float, image_size)),
        "nearDepth": float(c[11].item()),
        "farDepth": float(c[12].item()),
        "k1": float(c[13].item()),
        "k2": float(c[14].item()),
    }


def _is_valid_surface_size(width: int, height: int) -> bool:
    width = int(width)
    height = int(height)
    return (
        width > 0
        and height > 0
        and width <= _MAX_SURFACE_DIM
        and height <= _MAX_SURFACE_DIM
        and width * height <= _MAX_SURFACE_PIXELS
    )


def _fit_camera(splats: np.ndarray, fov_y_degrees: float) -> tuple[spy.float3, np.ndarray, float, float, float]:
    positions = splats[0:3].T.astype(np.float32, copy=False)
    if positions.shape[0] == 0:
        return spy.float3(0.0, 0.0, -3.0), np.zeros((3,), dtype=np.float32), 0.1, 100.0, 2.0
    scales = np.exp(splats[3:6]).astype(np.float32, copy=False)
    extents = 3.0 * np.max(scales, axis=0)
    center = np.mean(positions, axis=0, dtype=np.float32)
    radius = max(float(np.percentile(np.linalg.norm(positions - center[None, :], axis=1) + extents, 90.0)), 0.25)
    distance = max(radius / max(math.tan(0.5 * math.radians(float(fov_y_degrees))), 1e-4) * 0.95, radius * 1.8, 0.5)
    position = center + np.array([0.0, 0.0, -distance], dtype=np.float32)
    near = max(0.01, distance * 0.05)
    far = max(distance + radius * 6.0, 10.0)
    move_speed = max(0.25, radius)
    return spy.float3(*position.tolist()), center, near, far, move_speed


def _fit_camera_to_points(points: np.ndarray, fov_y_degrees: float) -> tuple[spy.float3, np.ndarray, float, float, float]:
    if points.size == 0:
        return spy.float3(0.0, 0.0, -3.0), np.zeros((3,), dtype=np.float32), 0.1, 100.0, 2.0
    center = np.mean(points, axis=0, dtype=np.float32)
    radius = max(float(np.percentile(np.linalg.norm(points - center[None, :], axis=1), 90.0)), 0.25)
    distance = max(radius / max(math.tan(0.5 * math.radians(float(fov_y_degrees))), 1e-4) * 0.95, radius * 1.8, 0.5)
    position = center + np.array([0.0, 0.0, -distance], dtype=np.float32)
    near = max(0.0, distance * 0.01)
    far = max(distance + radius * 10.0, 1000.0)
    move_speed = max(0.25, radius)
    return spy.float3(*position.tolist()), center, near, far, move_speed


class SplatViewer(spy.AppWindow):
    def __init__(self, app: spy.App, width: int = 1600, height: int = 900, title: str = _WINDOW_TITLE) -> None:
        super().__init__(app, width=width, height=height, title=title, resizable=True, enable_vsync=False)
        self.s = ViewerState()
        self.renderer = SplattingContext(device=self.device)
        self._render_frames = 0
        self.blit_kernel = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csBlitOutput"]))
        self.debug_letterbox_kernel = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csDebugLetterboxTensor"]))
        self.debug_loss_kernel = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csDebugLossLetterbox"]))
        self.debug_abs_diff_kernel = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csDebugAbsDiffLetterbox"]))
        self.ui = build_ui()
        self.toolkit = create_toolkit_window(self.device, width, height)
        self.toolkit.callbacks.load_ply = self._load_ply_callback
        self.toolkit.callbacks.browse_training_dataset = self._browse_training_dataset_callback
        self.toolkit.callbacks.browse_training_images = self._browse_training_images_callback
        self.training = TrainingController()
        self._present_texture: spy.Texture | None = None
        self._debug_target_tensors: dict[int, spy.Tensor] = {}
        self._surface_size = (int(width), int(height))
        self._skip_present_frames = 0

    def on_resize(self, width: int, height: int) -> None:
        try:
            super().on_resize(width, height)
        except Exception:
            pass
        self._surface_size = (max(int(width), 0), max(int(height), 0))
        self._present_texture = None
        self._skip_present_frames = 2

    def _ensure_present_texture(self, width: int, height: int) -> spy.Texture:
        width = max(int(width), 1)
        height = max(int(height), 1)
        if self._present_texture is not None and int(self._present_texture.width) == int(width) and int(self._present_texture.height) == int(height):
            return self._present_texture
        self._present_texture = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=int(width),
            height=int(height),
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_source | spy.TextureUsage.copy_destination,
        )
        return self._present_texture

    def _debug_view_key(self) -> str:
        index = min(max(int(self.ui.values.get("loss_debug_view", 0)), 0), len(_LOSS_DEBUG_VIEW_KEYS) - 1)
        return _LOSS_DEBUG_VIEW_KEYS[index]

    def _debug_frame_index(self) -> int:
        frame_count = self.training.frame_count()
        if frame_count <= 0:
            self.ui.values["_loss_debug_frame_max"] = 0
            return 0
        self.ui.values["_loss_debug_frame_max"] = frame_count - 1
        index = min(max(int(self.ui.values.get("loss_debug_frame", 0)), 0), frame_count - 1)
        self.ui.values["loss_debug_frame"] = index
        return index

    def _target_tensor(self, frame_index: int) -> spy.Tensor | None:
        cached = self._debug_target_tensors.get(int(frame_index))
        if cached is not None:
            return cached
        target = self.training.target_image(frame_index)
        if target is None:
            return None
        h, w = int(target.shape[0]), int(target.shape[1])
        rgba = torch.cat((target, torch.ones((h, w, 1), device=target.device, dtype=target.dtype)), dim=2).contiguous()
        tensor = spy.Tensor.empty(
            self.device,
            shape=(h, w),
            dtype=spy.float4,
            usage=spy.BufferUsage.shared | spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )
        tensor.copy_from_torch(rgba)
        self._debug_target_tensors[int(frame_index)] = tensor
        return tensor

    def _forward(self) -> np.ndarray:
        cy, sy = math.cos(self.s.yaw), math.sin(self.s.yaw)
        cp, sp = math.cos(self.s.pitch), math.sin(self.s.pitch)
        return _normalize(np.array([cp * sy, sp, cp * cy], dtype=np.float32))

    def camera(self) -> Camera:
        forward = self._forward()
        position = np.array([float(self.s.camera_pos.x), float(self.s.camera_pos.y), float(self.s.camera_pos.z)], dtype=np.float32)
        up = np.array([float(self.s.up.x), float(self.s.up.y), float(self.s.up.z)], dtype=np.float32)
        return Camera.look_at(position=position, target=position + forward, up=up, fov_y_degrees=float(self.s.fov_y), near=float(self.s.near), far=float(self.s.far))

    def _upload_scene_if_needed(self) -> None:
        preview = self.training.consume_preview()
        if preview is not None:
            params = self.renderer.scene["g_Params"].to_torch()
            packed_preview = _pack_params_torch(preview)
            params[: packed_preview.numel()].copy_(packed_preview)
            self.renderer.device.sync_to_device()
            self.s.scene_dirty = False
            self.s.splat_count = int(preview.shape[1])
            return
        if not self.s.scene_dirty:
            return
        self.renderer.scene["g_Params"].copy_from_numpy(self.s.packed_splats.reshape(-1))
        self.renderer.device.sync_to_cuda()
        self.s.scene_dirty = False

    def _load_scene(self, path: Path) -> None:
        splats = load_gaussian_ply(path)
        self.s.scene_path = path.resolve()
        self.s.splats = splats
        self.s.packed_splats = _pack_params(splats)
        self.s.splat_count = int(splats.shape[1])
        self.s.scene_dirty = True
        self.s.camera_pos, center, self.s.near, self.s.far, self.s.move_speed = _fit_camera(splats, self.s.fov_y)
        position = np.array([self.s.camera_pos.x, self.s.camera_pos.y, self.s.camera_pos.z], dtype=np.float32)
        forward = _normalize(center - position)
        self.s.yaw = math.atan2(float(forward[0]), float(forward[2]))
        self.s.pitch = math.asin(float(np.clip(forward[1], -1.0, 1.0)))
        self.s.move_vel = spy.float3(0.0, 0.0, 0.0)
        self.s.rot_vel = spy.float2(0.0, 0.0)
        self.s.last_error = ""

    def _load_ply_callback(self) -> None:
        path = spy.platform.open_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")])
        if path is None:
            return
        try:
            self._load_scene(Path(path))
        except Exception as exc:
            self.s.last_error = str(exc)

    def _browse_training_dataset_callback(self) -> None:
        path = spy.platform.choose_folder_dialog()
        if path is None:
            return
        try:
            self._debug_target_tensors.clear()
            self.training.set_dataset_folder(Path(path))
            self.s.last_error = ""
        except Exception as exc:
            self.s.last_error = str(exc)

    def _browse_training_images_callback(self) -> None:
        path = spy.platform.choose_folder_dialog()
        if path is None:
            return
        try:
            self._debug_target_tensors.clear()
            self.training.set_images_folder(Path(path))
            self.s.last_error = ""
        except Exception as exc:
            self.s.last_error = str(exc)

    def on_keyboard_event(self, event) -> None:
        if self.toolkit.handle_keyboard_event(event):
            if event.type == spy.KeyboardEventType.key_release:
                self.s.keys[event.key] = False
            return
        if event.type in (spy.KeyboardEventType.key_press, spy.KeyboardEventType.key_release):
            self.s.keys[event.key] = event.type == spy.KeyboardEventType.key_press

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

    def update_camera(self, dt: float) -> None:
        self.s.move_speed = float(self.ui.values["move_speed"])
        self.s.fov_y = float(self.ui.values["fov"])
        self.s.near = float(self.ui.values["near"])
        self.s.far = float(self.ui.values["far"])
        self.s.radius_scale = float(self.ui.values["radius_scale"])
        self.s.max_anisotropy = float(self.ui.values["max_anisotropy"])
        self.s.alpha_cutoff = float(self.ui.values["alpha_cutoff"])
        self.s.trans_threshold = float(self.ui.values["trans_threshold"])
        self.s.debug_mode = int(self.ui.values.get("debug_mode", 0))
        self.s.background = (
            float(self.ui.values["background_r"]),
            float(self.ui.values["background_g"]),
            float(self.ui.values["background_b"]),
        )
        if abs(self.s.scroll_delta) > 1e-5:
            self.s.move_speed = min(max(self.s.move_speed * (_SCROLL_SPEED_BASE ** self.s.scroll_delta), 0.1), 100.0)
            self.ui.values["move_speed"] = self.s.move_speed
            self.s.scroll_delta = 0.0
        target_rot = self.s.mouse_delta * self.s.look_speed if self.s.mouse_left else spy.float2(0.0, 0.0)
        self.s.rot_vel += (target_rot - self.s.rot_vel) * min(1.0, _LOOK_SMOOTH * dt)
        self.s.mouse_delta = spy.float2(0.0, 0.0)
        self.s.yaw += float(self.s.rot_vel.x)
        self.s.pitch = min(max(self.s.pitch + float(self.s.rot_vel.y), -_PITCH_LIMIT), _PITCH_LIMIT)
        forward = self._forward()
        right = _normalize(np.cross(np.array([self.s.up.x, self.s.up.y, self.s.up.z], dtype=np.float32), forward))
        up = _normalize(np.cross(forward, right))
        move = np.array(
            [
                float(self.s.keys.get(spy.KeyCode.e, False)) - float(self.s.keys.get(spy.KeyCode.q, False)),
                float(self.s.keys.get(spy.KeyCode.d, False)) - float(self.s.keys.get(spy.KeyCode.a, False)),
                float(self.s.keys.get(spy.KeyCode.w, False)) - float(self.s.keys.get(spy.KeyCode.s, False)),
            ],
            dtype=np.float32,
        )
        move_len = float(np.linalg.norm(move))
        target_move = move * (self.s.move_speed / max(move_len, _VIEW_VEC_EPS)) if move_len > _VIEW_VEC_EPS else np.zeros((3,), dtype=np.float32)
        move_vel = np.array([self.s.move_vel.x, self.s.move_vel.y, self.s.move_vel.z], dtype=np.float32)
        move_vel += (target_move - move_vel) * min(1.0, _MOVE_SMOOTH * dt)
        self.s.move_vel = spy.float3(*move_vel.tolist())
        position = np.array([self.s.camera_pos.x, self.s.camera_pos.y, self.s.camera_pos.z], dtype=np.float32)
        position += (up * move_vel[0] + right * move_vel[1] + forward * move_vel[2]) * dt
        self.s.camera_pos = spy.float3(*position.tolist())

    def _update_ui(self, training_snapshot, dt: float) -> None:
        self.s.fps_smooth += (1.0 / dt - self.s.fps_smooth) * min(dt * 5.0, 1.0)
        self.ui.values["_loss_debug_frame_max"] = max(self.training.frame_count() - 1, 0)
        frame_index = self._debug_frame_index()
        frame_name = self.training.frame_name(frame_index)
        active_scene = None if self.s.scene_path is None else str(self.s.scene_path)
        if active_scene is None and training_snapshot.scene_path:
            active_scene = training_snapshot.scene_path
        self.ui.texts["fps"] = f"FPS: {self.s.fps_smooth:.1f}"
        self.ui.texts["scene"] = "Scene: <none>" if active_scene is None else f"Scene: {active_scene}"
        self.ui.texts["status"] = f"Status: {'Loss Debug' if bool(self.ui.values.get('loss_debug', False)) else 'Viewer'} | Splats: {self.s.splat_count:,}"
        self.ui.texts["render_stats"] = f"Generated: {int(getattr(self.renderer, '_last_required_total', getattr(self.renderer, '_last_total', 0))):,} | Debug mode: {int(self.ui.values.get('debug_mode', 0))}"
        train_mode = "running" if training_snapshot.running and not training_snapshot.paused else "paused" if training_snapshot.paused else "idle"
        self.ui.texts["training"] = f"Training: {train_mode} | step={training_snapshot.iteration:,} | splats={training_snapshot.point_count:,}"
        self.ui.texts["training_time"] = f"Time: {time.strftime('%H:%M:%S', time.gmtime(training_snapshot.elapsed_seconds))}" if training_snapshot.elapsed_seconds > 0 else "Time: 00:00"
        self.ui.texts["training_iters_avg"] = (
            f"Avg it/s: {training_snapshot.iteration / training_snapshot.elapsed_seconds:.2f}"
            if training_snapshot.elapsed_seconds > 1e-6 and training_snapshot.iteration > 0
            else "Avg it/s: n/a"
        )
        self.ui.texts["training_loss"] = f"Loss Avg: {training_snapshot.avg_loss:.6e}" if math.isfinite(training_snapshot.avg_loss) else "Loss Avg: n/a"
        self.ui.texts["training_mse"] = f"MSE Avg: {training_snapshot.avg_mse:.6e}" if math.isfinite(training_snapshot.avg_mse) else "MSE Avg: n/a"
        self.ui.texts["training_psnr"] = f"PSNR Avg: {training_snapshot.avg_psnr:.3f} dB" if math.isfinite(training_snapshot.avg_psnr) else "PSNR Avg: n/a"
        self.ui.texts["loss_debug_frame"] = f"Frame[{frame_index}]: {frame_name}" if self.training.frame_count() > 0 else "Frame: <none>"
        sample = self.training.camera_sample(0)
        self.ui.texts["training_resolution"] = (
            f"Train Res: {sample.image_size[0]}x{sample.image_size[1]}" if sample is not None else "Train Res: n/a"
        )
        self.ui.texts["training_downscale"] = "Downscale: 1x"
        self.ui.texts["error"] = f"Error: {self.s.last_error}" if self.s.last_error else ""
        self.ui.texts["max_splat_steps"] = str(getattr(self.renderer, "_last_required_total", getattr(self.renderer, "_last_total", 0)))
        tk = self.toolkit.tk
        tk.fps_history.append(self.s.fps_smooth)
        if training_snapshot.iteration > 0 and (not tk.step_history or training_snapshot.iteration != tk.step_history[-1]):
            tk.step_history.append(float(training_snapshot.iteration))
            tk.step_time_history.append(float(self.s.last_time))
            if math.isfinite(training_snapshot.avg_loss) and training_snapshot.avg_loss > 0.0:
                tk.loss_history.append(float(training_snapshot.avg_loss))
            elif tk.loss_history:
                tk.loss_history.append(float(tk.loss_history[-1]))
            if math.isfinite(training_snapshot.avg_psnr):
                tk.psnr_history.append(float(training_snapshot.avg_psnr))
            elif tk.psnr_history:
                tk.psnr_history.append(float(tk.psnr_history[-1]))

    def _render_debug_view(self, image: spy.Texture, encoder: spy.CommandEncoder, active_splat_count: int, render_seed: int) -> None:
        frame_index = self._debug_frame_index()
        sample = self.training.camera_sample(frame_index)
        if sample is None:
            encoder.clear_texture_float(image, clear_value=[*self.s.background, 1.0])
            return
        debug_width, debug_height = map(int, sample.image_size)
        self.renderer.radius_scale = self.s.radius_scale
        self.renderer.dither_strength = float(self.ui.values["dither_strength"])
        self.renderer.max_anisotropy = self.s.max_anisotropy
        self.renderer.alpha_cutoff = self.s.alpha_cutoff
        self.renderer.trans_threshold = self.s.trans_threshold
        self.renderer.render_seed = int(render_seed)
        self.renderer.debug_mode = self.s.debug_mode
        self.renderer.debug_depth_mean_range = (
            float(self.ui.values["debug_depth_mean_min"]),
            float(self.ui.values["debug_depth_mean_max"]),
        )
        self.renderer.debug_depth_std_range = (
            float(self.ui.values["debug_depth_std_min"]),
            float(self.ui.values["debug_depth_std_max"]),
        )
        self.renderer.prepare(active_splat_count, (debug_width, debug_height), self.s.background)
        self._upload_scene_if_needed()
        self.renderer.render(_camera_dict(sample.camera_params, sample.image_size), active_splat_count, command_encoder=encoder)
        self._render_frames += 1
        self.ui.texts["max_splat_steps"] = str(getattr(self.renderer, "_last_required_total", getattr(self.renderer, "_last_total", 0)))
        present = self._ensure_present_texture(int(image.width), int(image.height))
        mode = self._debug_view_key()
        if mode in ("loss", "abs_diff"):
            target = self._target_tensor(frame_index)
            if target is None:
                encoder.clear_texture_float(image, clear_value=[*self.s.background, 1.0])
                return
            kernel = self.debug_loss_kernel if mode == "loss" else self.debug_abs_diff_kernel
            kernel.dispatch(
                thread_count=spy.uint3(int(image.width) * int(image.height), 1, 1),
                vars={
                    "g_DebugRendered": self.renderer.frame["g_Output"],
                    "g_DebugTarget": target,
                    "g_Surface": present,
                    "g_LetterboxSourceWidth": debug_width,
                    "g_LetterboxSourceHeight": debug_height,
                    "g_LetterboxOutputWidth": int(image.width),
                    "g_LetterboxOutputHeight": int(image.height),
                    "g_DebugDiffScale": float(self.ui.values["loss_debug_abs_scale"]),
                },
                command_encoder=encoder,
            )
        else:
            source = self.renderer.frame["g_Output"] if mode == "rendered" else self._target_tensor(frame_index)
            if source is None:
                encoder.clear_texture_float(image, clear_value=[*self.s.background, 1.0])
                return
            self.debug_letterbox_kernel.dispatch(
                thread_count=spy.uint3(int(image.width) * int(image.height), 1, 1),
                vars={
                    "g_LetterboxSource": source,
                    "g_Surface": present,
                    "g_LetterboxSourceWidth": debug_width,
                    "g_LetterboxSourceHeight": debug_height,
                    "g_LetterboxOutputWidth": int(image.width),
                    "g_LetterboxOutputHeight": int(image.height),
                },
                command_encoder=encoder,
            )
        encoder.blit(image, present)

    def render(self, render_context) -> None:
        image = render_context.surface_texture
        encoder = render_context.command_encoder
        now = time.perf_counter()
        dt = max(now - self.s.last_time, 1e-5)
        self.s.last_time = now
        width = int(image.width)
        height = int(image.height)
        if not _is_valid_surface_size(width, height):
            self._present_texture = None
            self.s.last_error = f"Skipping invalid surface size {width}x{height}"
            return
        if (width, height) != self._surface_size:
            self._surface_size = (width, height)
            self._present_texture = None
            self._skip_present_frames = max(self._skip_present_frames, 2)
        if self._skip_present_frames > 0:
            self._skip_present_frames -= 1
            encoder.clear_texture_float(image, clear_value=[*self.s.background, 1.0])
            return
        self.training.update()
        training_snapshot = self.training.snapshot()
        fit_points = self.training.consume_fit_points()
        if fit_points is not None:
            self.s.camera_pos, center, self.s.near, self.s.far, self.s.move_speed = _fit_camera_to_points(fit_points, self.s.fov_y)
            position = np.array([self.s.camera_pos.x, self.s.camera_pos.y, self.s.camera_pos.z], dtype=np.float32)
            forward = _normalize(center - position)
            self.s.yaw = math.atan2(float(forward[0]), float(forward[2]))
            self.s.pitch = math.asin(float(np.clip(forward[1], -1.0, 1.0)))
        try:
            self.update_camera(dt)
            if self._render_frames > 0 and self._render_frames % 64 == 0:
                self.renderer.readback_and_reallocate_buffers(refresh_buffers=True)
            has_training_preview = training_snapshot.preview_count > 0
            if self.s.splats is None and not has_training_preview:
                encoder.clear_texture_float(image, clear_value=[*self.s.background, 1.0])
            else:
                active_splat_count = int(training_snapshot.preview_count) if has_training_preview else int(self.s.splat_count)
                render_seed = int(training_snapshot.iteration) if has_training_preview else 0
                if bool(self.ui.values.get("loss_debug", False)) and self.training.frame_count() > 0:
                    self._render_debug_view(image, encoder, active_splat_count, render_seed)
                else:
                    self.renderer.radius_scale = self.s.radius_scale
                    self.renderer.dither_strength = float(self.ui.values["dither_strength"])
                    self.renderer.max_anisotropy = self.s.max_anisotropy
                    self.renderer.alpha_cutoff = self.s.alpha_cutoff
                    self.renderer.trans_threshold = self.s.trans_threshold
                    self.renderer.render_seed = render_seed
                    self.renderer.debug_mode = self.s.debug_mode
                    self.renderer.debug_depth_mean_range = (
                        float(self.ui.values["debug_depth_mean_min"]),
                        float(self.ui.values["debug_depth_mean_max"]),
                    )
                    self.renderer.debug_depth_std_range = (
                        float(self.ui.values["debug_depth_std_min"]),
                        float(self.ui.values["debug_depth_std_max"]),
                    )
                    self.renderer.prepare(active_splat_count, (width, height), self.s.background)
                    self._upload_scene_if_needed()
                    self.renderer.render(self.camera().gpu_params(width, height), active_splat_count, command_encoder=encoder)
                    self._render_frames += 1
                    self.ui.texts["max_splat_steps"] = str(getattr(self.renderer, "_last_required_total", getattr(self.renderer, "_last_total", 0)))
                    present = self._ensure_present_texture(width, height)
                    self.blit_kernel.dispatch(
                        thread_count=spy.uint3(width * height, 1, 1),
                        vars={"g_ViewImage": self.renderer.frame["g_Output"], "g_Surface": present, "g_Viewport": spy.uint2(width, height)},
                        command_encoder=encoder,
                    )
                    encoder.blit(image, present)
        except Exception as exc:
            self.s.last_error = str(exc)
            encoder.clear_texture_float(image, clear_value=[0.0, 0.0, 0.0, 1.0])
        self._update_ui(training_snapshot, dt)
        self.toolkit.render(self.ui, self.training, image, encoder)


def _compute_view_geometry() -> tuple[int, int]:
    import sys
    if sys.platform == "win32":
        import ctypes
        user32 = ctypes.windll.user32
        return max(min(int(user32.GetSystemMetrics(0) * 0.9), 1920), 1280), max(min(int(user32.GetSystemMetrics(1) * 0.9), 1200), 720)
    return 1600, 900


def _create_device() -> spy.Device:
    include_paths = [str(path) for path in (_SHADERS, Path(__file__).resolve().parents[1] / "module" / "shaders", Path(__file__).resolve().parents[1] / "utility" / "shaders", Path(__file__).resolve().parents[1] / "shaders")]
    return spy.create_torch_device(type=spy.DeviceType.vulkan, include_paths=include_paths, enable_hot_reload=False)


def main() -> int:
    width, height = _compute_view_geometry()
    device = _create_device()
    app = spy.App(device=device)
    viewer = SplatViewer(app, width=width, height=height)
    try:
        app.run()
    finally:
        viewer.training.shutdown()
        viewer.toolkit.shutdown()
    return 0

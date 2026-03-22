from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import slangpy as spy

from module.splatting import SplattingContext
from utility.ply import load_gaussian_ply
from .camera import Camera
from .state import ViewerState
from .ui import _WINDOW_TITLE, build_ui, create_toolkit_window

_ROOT = Path(__file__).resolve().parent
_SHADERS = _ROOT / "shaders"
_VIEW_VEC_EPS = 1e-6
_SCROLL_SPEED_BASE = 1.1
_LOOK_SMOOTH = 12.0
_MOVE_SMOOTH = 10.0
_PITCH_LIMIT = math.radians(89.0)
_ALPHA_EPS = 1e-6


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return (v / max(norm, _VIEW_VEC_EPS)).astype(np.float32)


def _pack_params(splats: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(splats.copy())
    alpha = np.clip(packed[13], _ALPHA_EPS, 1.0 - _ALPHA_EPS)
    packed[13] = np.log(alpha / np.clip(1.0 - alpha, _ALPHA_EPS, 1.0))
    return packed


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


class SplatViewer(spy.AppWindow):
    def __init__(self, app: spy.App, width: int = 1600, height: int = 900, title: str = _WINDOW_TITLE) -> None:
        super().__init__(app, width=width, height=height, title=title, resizable=True, enable_vsync=False)
        self.s = ViewerState()
        self.renderer = SplattingContext(device=self.device)
        self.blit_kernel = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csBlitOutput"]))
        self.ui = build_ui()
        self.toolkit = create_toolkit_window(self.device, width, height)
        self.toolkit.callbacks.load_ply = self._load_ply_callback
        self._present_texture: spy.Texture | None = None

    def _ensure_present_texture(self, width: int, height: int) -> spy.Texture:
        if self._present_texture is not None and int(self._present_texture.width) == int(width) and int(self._present_texture.height) == int(height):
            return self._present_texture
        self._present_texture = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=int(width),
            height=int(height),
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_source | spy.TextureUsage.copy_destination,
        )
        return self._present_texture

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
        self.s.alpha_cutoff = float(self.ui.values["alpha_cutoff"])
        self.s.trans_threshold = float(self.ui.values["trans_threshold"])
        self.s.debug_mode = int(self.ui.values["debug_mode"])
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

    def render(self, render_context) -> None:
        image = render_context.surface_texture
        encoder = render_context.command_encoder
        now = time.perf_counter()
        dt = max(now - self.s.last_time, 1e-5)
        self.s.last_time = now
        self.s.fps_smooth += (1.0 / dt - self.s.fps_smooth) * min(dt * 5.0, 1.0)
        self.ui.texts["status"] = f"FPS: {self.s.fps_smooth:.1f} | Splats: {self.s.splat_count:,}"
        self.ui.texts["scene"] = "Scene: <none>" if self.s.scene_path is None else f"Scene: {self.s.scene_path}"
        self.ui.texts["error"] = f"Error: {self.s.last_error}" if self.s.last_error else ""
        self.ui.texts["max_splat_steps"] = str(getattr(self.renderer, "_last_total", 0))
        try:
            self.update_camera(dt)
            if self.s.splats is None:
                encoder.clear_texture_float(image, clear_value=[*self.s.background, 1.0])
            else:
                width, height = int(image.width), int(image.height)
                self.renderer.radius_scale = self.s.radius_scale
                self.renderer.alpha_cutoff = self.s.alpha_cutoff
                self.renderer.trans_threshold = self.s.trans_threshold
                self.renderer.debug_mode = self.s.debug_mode
                self.renderer.prepare(self.s.splat_count, (width, height), self.s.background)
                self._upload_scene_if_needed()
                self.renderer.render(self.camera().gpu_params(width, height), self.s.splat_count, command_encoder=encoder)
                self.ui.texts["max_splat_steps"] = str(getattr(self.renderer, "_last_total", 0))
                present = self._ensure_present_texture(width, height)
                self.blit_kernel.dispatch(
                    thread_count=spy.uint3(width, height, 1),
                    vars={"g_ViewImage": self.renderer.frame["g_Output"], "g_Surface": present, "g_Viewport": spy.uint2(width, height)},
                    command_encoder=encoder,
                )
                encoder.blit(image, present)
        except Exception as exc:
            self.s.last_error = str(exc)
            encoder.clear_texture_float(image, clear_value=[0.0, 0.0, 0.0, 1.0])
        self.toolkit.render(self.ui, image, encoder)


def _compute_view_geometry() -> tuple[int, int]:
    import sys
    if sys.platform == "win32":
        import ctypes
        user32 = ctypes.windll.user32
        return max(min(int(user32.GetSystemMetrics(0) * 0.9), 1920), 1280), max(min(int(user32.GetSystemMetrics(1) * 0.9), 1200), 720)
    return 1600, 900


def _create_device() -> spy.Device:
    include_paths = [str(path) for path in (_SHADERS, Path(__file__).resolve().parents[1] / "module" / "shaders", Path(__file__).resolve().parents[1] / "utility" / "shaders", Path(__file__).resolve().parents[1] / "shaders")]
    return spy.create_device(type=spy.DeviceType.vulkan, include_paths=include_paths, enable_cuda_interop=False)


def main() -> int:
    width, height = _compute_view_geometry()
    device = _create_device()
    app = spy.App(device=device)
    viewer = SplatViewer(app, width=width, height=height)
    try:
        app.run()
    finally:
        viewer.toolkit.shutdown()
    return 0

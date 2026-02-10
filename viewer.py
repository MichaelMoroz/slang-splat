from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import slangpy as spy

from src import create_default_device
from src.renderer import Camera, GaussianRenderer
from src.scene import GaussianScene, load_gaussian_ply


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v), 1e-8)


class SplatViewer(spy.AppWindow):
    def __init__(
        self,
        app: spy.App,
        width: int = 1280,
        height: int = 720,
        title: str = "Slang Splat Viewer",
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
        self.renderer = GaussianRenderer(
            self.device,
            width=width,
            height=height,
            list_capacity_multiplier=self.list_capacity_multiplier,
        )
        self.scene: GaussianScene | None = None
        self.scene_path: Path | None = None
        self.stats: dict[str, int | bool | float] = {}

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

        self._build_ui()

    def _build_ui(self) -> None:
        panel = spy.ui.Window(self.screen, "PLY Viewer", size=spy.float2(430, 360))
        self.fps_text = spy.ui.Text(panel, "FPS: 0.0")
        self.path_text = spy.ui.Text(panel, "Scene: <none>")
        self.scene_stats_text = spy.ui.Text(panel, "Splats: 0")
        self.render_stats_text = spy.ui.Text(panel, "Generated: 0 | Written: 0")
        self.error_text = spy.ui.Text(panel, "")

        load_group = spy.ui.Group(panel, "Scene")
        spy.ui.Button(load_group, "Load PLY...", callback=self._browse_load_ply)
        spy.ui.Button(load_group, "Reload", callback=self._reload_scene)

        params_group = spy.ui.Group(panel, "Render Params")
        self.radius_slider = spy.ui.SliderFloat(
            params_group,
            "Radius Scale",
            value=float(self.renderer.radius_scale),
            min=0.5,
            max=4.0,
            flags=spy.ui.SliderFlags.logarithmic,
        )
        self.max_radius_slider = spy.ui.SliderFloat(
            params_group,
            "Max Radius (px)",
            value=float(self.renderer.max_splat_radius_px),
            min=8.0,
            max=2048.0,
            flags=spy.ui.SliderFlags.logarithmic,
        )
        self.alpha_slider = spy.ui.SliderFloat(
            params_group,
            "Alpha Cutoff",
            value=float(self.renderer.alpha_cutoff),
            min=0.0001,
            max=0.1,
            flags=spy.ui.SliderFlags.logarithmic,
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
            flags=spy.ui.SliderFlags.logarithmic,
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

        cam_group = spy.ui.Group(panel, "Camera")
        self.move_speed_slider = spy.ui.SliderFloat(
            cam_group,
            "Move Speed",
            value=float(self.move_speed),
            min=0.1,
            max=20.0,
            flags=spy.ui.SliderFlags.logarithmic,
        )
        self.fov_slider = spy.ui.SliderFloat(
            cam_group,
            "FOV",
            value=float(self.fov_y),
            min=25.0,
            max=100.0,
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

    def _recreate_renderer(self, width: int, height: int) -> None:
        old_renderer = self.renderer
        self.renderer = GaussianRenderer(
            self.device,
            width=width,
            height=height,
            radius_scale=float(self.radius_slider.value),
            max_splat_radius_px=float(self.max_radius_slider.value),
            alpha_cutoff=float(self.alpha_slider.value),
            max_splat_steps=int(self.max_steps_slider.value),
            transmittance_threshold=float(self.trans_slider.value),
            sampled5_safety_scale=float(self.sampled5_safety_slider.value),
            list_capacity_multiplier=self.list_capacity_multiplier,
            debug_show_ellipses=bool(self.debug_ellipse_checkbox.value),
            debug_show_processed_count=bool(self.debug_processed_count_checkbox.value),
        )
        del old_renderer
        if self.scene is not None:
            self.renderer.set_scene(self.scene)

    def _forward(self) -> np.ndarray:
        yaw = self.yaw
        pitch = self.pitch
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        f = np.array(
            [cp * sy, sp, cp * cy],
            dtype=np.float32,
        )
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

    def _reload_scene(self) -> None:
        if self.scene_path:
            self.load_scene(self.scene_path)

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

            core_radius = float(np.percentile(effective, 90.0))
            core_radius = max(core_radius, 1.0)

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
        print(
            f"Camera auto-frame: center={center.tolist()} radius={radius:.2f} "
            f"distance={cam_distance:.2f} near={self.near:.3f} far={self.far:.2f}"
        )

    def load_scene(self, path: Path) -> None:
        try:
            scene = load_gaussian_ply(path)
            self.scene = scene
            self.scene_path = path
            self.renderer.set_scene(scene)
            self._recenter_camera(scene)
            self.last_error = ""
            print(f"Loaded scene: {path} ({scene.count:,} splats)")
        except Exception as exc:
            self.last_error = str(exc)
            print(f"Failed to load scene {path}: {exc}")

    def _apply_render_params(self) -> None:
        self.renderer.radius_scale = float(self.radius_slider.value)
        self.renderer.max_splat_radius_px = float(self.max_radius_slider.value)
        self.renderer.alpha_cutoff = float(self.alpha_slider.value)
        self.renderer.max_splat_steps = int(self.max_steps_slider.value)
        self.renderer.transmittance_threshold = float(self.trans_slider.value)
        self.renderer.sampled5_safety_scale = float(self.sampled5_safety_slider.value)
        self.renderer.debug_show_ellipses = bool(self.debug_ellipse_checkbox.value)
        self.renderer.debug_show_processed_count = bool(self.debug_processed_count_checkbox.value)

    def _update_ui_text(self, dt: float) -> None:
        self.fps_smooth += (1.0 / max(dt, 1e-5) - self.fps_smooth) * min(dt * 5.0, 1.0)
        self.fps_text.text = f"FPS: {self.fps_smooth:.1f}"
        if self.scene_path:
            self.path_text.text = f"Scene: {self.scene_path.name}"
        else:
            self.path_text.text = "Scene: <none>"
        scene_count = self.scene.count if self.scene is not None else 0
        self.scene_stats_text.text = f"Splats: {scene_count:,}"
        if self.stats:
            self.render_stats_text.text = (
                f"Generated: {int(self.stats['generated_entries']):,} | "
                f"Written: {int(self.stats['written_entries']):,} | "
                f"Overflow: {bool(self.stats['overflow'])}"
            )
        else:
            self.render_stats_text.text = "Generated: 0 | Written: 0"
        self.error_text.text = f"Error: {self.last_error}" if self.last_error else ""

    def render(self, render_context: spy.AppWindow.RenderContext) -> None:
        image = render_context.surface_texture
        encoder = render_context.command_encoder

        now = time.perf_counter()
        dt = max(now - self.last_time, 1e-5)
        self.last_time = now

        self._update_camera(dt)
        self._apply_render_params()

        iw = int(image.width)
        ih = int(image.height)
        if self.renderer.width != iw or self.renderer.height != ih:
            self._recreate_renderer(iw, ih)

        if self.scene is not None:
            try:
                out_tex, stats = self.renderer.render_to_texture(self._camera(), background=self.background)
                self.stats = stats
                encoder.blit(image, out_tex)
                self._last_render_exception = ""
            except Exception as exc:
                self.last_error = str(exc)
                if self._last_render_exception != self.last_error:
                    print(f"Render error: {self.last_error}")
                self._last_render_exception = self.last_error
                encoder.clear_texture_float(image, clear_value=[0.0, 0.0, 0.0, 1.0])
        else:
            encoder.clear_texture_float(image, clear_value=[0.1, 0.1, 0.12, 1.0])

        self._update_ui_text(dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Slangpy Gaussian Splat viewer.")
    parser.add_argument("--ply", type=Path, default=None, help="Optional initial PLY file.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=0, help="Run a fixed frame count for smoke tests.")
    parser.add_argument("--debug-layers", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = create_default_device(enable_debug_layers=args.debug_layers)
    app = spy.App(device=device)
    viewer = SplatViewer(
        app,
        width=args.width,
        height=args.height,
    )
    if args.ply is not None:
        viewer.load_scene(args.ply)

    if args.frames > 0:
        for _ in range(args.frames):
            app.run_frame()
        app.terminate()
    else:
        app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

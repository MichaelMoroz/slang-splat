from __future__ import annotations

from pathlib import Path

import numpy as np
import slangpy as spy

from ..app.shared import renderer_kwargs, estimate_point_bounds, estimate_scene_bounds
from ..common import SHADER_ROOT
from ..renderer import GaussianRenderer
from ..scene import (
    GaussianScene,
    build_training_frames,
    load_colmap_reconstruction,
    load_gaussian_ply,
    suggest_colmap_init_hparams,
)
from ..training import GaussianTrainer
from .state import SceneCountProxy


def _clear_renderer(viewer: object, attr: str) -> None:
    renderer = getattr(viewer.s, attr)
    if renderer is None:
        return
    setattr(viewer.s, attr, None)
    del renderer


def _invalidate_synced_steps(viewer: object, *targets: str) -> None:
    for target in targets or ("main", "debug"):
        setattr(viewer.s, f"synced_step_{target}", -1)


def _point_tables(recon: object) -> tuple[np.ndarray, np.ndarray]:
    xyz = getattr(recon, "point_xyz_table", None)
    rgb = getattr(recon, "point_rgb_table", None)
    if xyz is None or rgb is None:
        points = list(recon.points3d.values())
        if not points:
            raise RuntimeError("COLMAP point tables are missing and points3d is empty.")
        xyz = np.stack([point.xyz for point in points], axis=0).astype(np.float32)
        rgb = np.stack([point.rgb for point in points], axis=0).astype(np.float32)
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    if xyz.shape[0] != rgb.shape[0] or xyz.shape[0] == 0:
        raise RuntimeError("COLMAP point tables are empty or mismatched.")
    return xyz, rgb


def reset_loss_debug_state(viewer: object) -> None:
    viewer.s.loss_debug_texture = None
    viewer.s.debug_present_texture = None
    _clear_renderer(viewer, "debug_renderer")
    _invalidate_synced_steps(viewer, "debug")


def reset_suggested_init_defaults(viewer: object) -> None:
    viewer.s.suggested_init_hparams = None
    viewer.s.suggested_init_count = None


def create_debug_shaders(viewer: object) -> None:
    shader_path = str(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang")
    viewer.s.debug_abs_diff_kernel = viewer.device.create_compute_kernel(
        viewer.device.load_program(shader_path, ["csComposeAbsDiffDebug"])
    )
    viewer.s.debug_letterbox_kernel = viewer.device.create_compute_kernel(
        viewer.device.load_program(shader_path, ["csComposeLetterboxDebug"])
    )


def update_debug_frame_slider_range(viewer: object) -> None:
    slider = viewer.c("loss_debug_frame")
    max_index = max(len(viewer.s.training_frames) - 1, 0)
    try:
        slider.min = 0
        slider.max = int(max_index)
    except Exception:
        pass
    slider.value = int(np.clip(int(slider.value), 0, max_index))


def ensure_renderer(viewer: object, attr: str, width: int, height: int, allow_debug_overlays: bool) -> GaussianRenderer:
    size = (int(width), int(height))
    renderer = getattr(viewer.s, attr)
    if renderer is not None and (renderer.width, renderer.height) == size:
        return renderer
    _clear_renderer(viewer, attr)
    renderer = GaussianRenderer(
        viewer.device,
        width=size[0],
        height=size[1],
        **renderer_kwargs(viewer.renderer_params(allow_debug_overlays)),
    )
    if isinstance(viewer.s.scene, GaussianScene):
        renderer.set_scene(viewer.s.scene)
    setattr(viewer.s, attr, renderer)
    _invalidate_synced_steps(viewer, "debug" if attr == "debug_renderer" else "main", "debug")
    return renderer


def recreate_renderer(viewer: object, width: int, height: int) -> None:
    ensure_renderer(viewer, "renderer", width, height, allow_debug_overlays=True)
    reset_loss_debug_state(viewer)


def apply_render_params(viewer: object) -> None:
    for renderer, allow_debug_overlays in (
        (viewer.s.renderer, True),
        (viewer.s.training_renderer, False),
        (viewer.s.debug_renderer, False),
    ):
        if renderer is not None:
            for key, value in renderer_kwargs(viewer.renderer_params(allow_debug_overlays)).items():
                setattr(renderer, key, value)


def apply_training_params(viewer: object) -> None:
    if viewer.s.trainer is None:
        return
    params = viewer.training_params()
    viewer.s.trainer.update_hyperparams(params.adam, params.stability, params.training)


def _reset_loaded_runtime(viewer: object) -> None:
    viewer.s.scene_init_signature = None
    viewer.s.training_active = False
    viewer.s.trainer = None
    viewer.s.colmap_point_positions_buffer = None
    viewer.s.colmap_point_colors_buffer = None
    viewer.s.colmap_point_count = 0
    reset_suggested_init_defaults(viewer)
    update_debug_frame_slider_range(viewer)
    reset_loss_debug_state(viewer)
    _clear_renderer(viewer, "training_renderer")


def upload_colmap_pointcloud_buffers(viewer: object, recon: object) -> None:
    xyz, rgb = _point_tables(recon)
    pos4 = np.pad(xyz, ((0, 0), (0, 1)))
    col4 = np.pad(rgb, ((0, 0), (0, 1)))
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    viewer.s.colmap_point_positions_buffer = viewer.device.create_buffer(size=xyz.shape[0] * 16, usage=usage)
    viewer.s.colmap_point_colors_buffer = viewer.device.create_buffer(size=rgb.shape[0] * 16, usage=usage)
    viewer.s.colmap_point_positions_buffer.copy_from_numpy(pos4.astype(np.float32))
    viewer.s.colmap_point_colors_buffer.copy_from_numpy(col4.astype(np.float32))
    viewer.s.colmap_point_count = int(xyz.shape[0])


def apply_dataset_init_defaults(viewer: object, force: bool = False) -> None:
    if viewer.s.colmap_recon is None:
        return
    gaussian_count = int(np.clip(int(viewer.c("gaussian_count").value), 1, 10_000_000))
    if not force and viewer.s.suggested_init_count == gaussian_count and viewer.s.suggested_init_hparams is not None:
        return
    suggested = suggest_colmap_init_hparams(viewer.s.colmap_recon, gaussian_count)
    previous = viewer.s.suggested_init_hparams

    def update_if_close(control_key: str, value: float, previous_value: float | None) -> None:
        control = viewer.c(control_key)
        if force or previous_value is None or abs(float(control.value) - float(previous_value)) <= max(1e-8, 1e-4 * abs(float(previous_value))):
            control.value = float(value)

    update_if_close("init_pos_jitter", suggested.position_jitter_std, None if previous is None else previous.position_jitter_std)
    update_if_close("init_scale", suggested.base_scale, None if previous is None else previous.base_scale)
    update_if_close("init_scale_jitter", suggested.scale_jitter_ratio, None if previous is None else previous.scale_jitter_ratio)
    update_if_close("init_opacity", suggested.initial_opacity, None if previous is None else previous.initial_opacity)
    viewer.s.suggested_init_hparams = suggested
    viewer.s.suggested_init_count = gaussian_count


def current_scene_init_signature(viewer: object) -> tuple[object, ...] | None:
    if viewer.s.colmap_root is None or viewer.s.colmap_recon is None or not viewer.s.training_frames:
        return None
    init = viewer.init_params()
    return (
        str(viewer.s.colmap_root.resolve()),
        len(viewer.s.training_frames),
        init.gaussian_count,
        init.seed,
        *(round(float(v), 8) for v in (
            init.hparams.position_jitter_std,
            init.hparams.base_scale,
            init.hparams.scale_jitter_ratio,
            init.hparams.initial_opacity,
        )),
    )


def sync_scene_from_training_renderer(viewer: object, dst_renderer: GaussianRenderer, target: str, force: bool = False) -> None:
    if viewer.s.training_renderer is None or viewer.s.trainer is None:
        return
    step = int(viewer.s.trainer.state.step)
    synced = getattr(viewer.s, f"synced_step_{target}")
    if not force and synced == step:
        return
    enc = viewer.device.create_command_encoder()
    viewer.s.training_renderer.copy_scene_state_to(enc, dst_renderer)
    viewer.device.submit_command_buffer(enc.finish())
    setattr(viewer.s, f"synced_step_{target}", step)


def load_scene(viewer: object, path: Path) -> None:
    try:
        scene = load_gaussian_ply(path)
        viewer.s.scene = scene
        viewer.s.scene_path = path
        viewer.s.colmap_root = None
        viewer.s.colmap_recon = None
        viewer.s.training_frames = []
        _reset_loaded_runtime(viewer)
        viewer.s.renderer.set_scene(scene)
        viewer.apply_camera_fit(estimate_scene_bounds(scene))
        viewer.s.last_error = ""
        print(f"Loaded scene: {path} ({scene.count:,} splats)")
    except Exception as exc:
        viewer.s.last_error = str(exc)
        print(f"Failed to load scene {path}: {exc}")


def load_colmap_dataset(viewer: object, root: Path, images_subdir: str) -> None:
    try:
        _reset_loaded_runtime(viewer)
        recon = load_colmap_reconstruction(root)
        viewer.s.colmap_root = Path(root)
        viewer.s.colmap_recon = recon
        viewer.s.training_frames = build_training_frames(recon, images_subdir=images_subdir)
        upload_colmap_pointcloud_buffers(viewer, recon)
        viewer.s.scene_path = None
        apply_dataset_init_defaults(viewer, force=True)
        points, _ = _point_tables(recon)
        viewer.apply_camera_fit(estimate_point_bounds(points))
        initialize_training_scene(viewer)
        viewer.s.last_error = ""
        print(f"Loaded COLMAP: {root} frames={len(viewer.s.training_frames)} images={images_subdir}")
    except Exception as exc:
        viewer.s.last_error = str(exc)
        print(f"Failed to load COLMAP {root}: {exc}")


def initialize_training_scene(viewer: object) -> None:
    if viewer.s.colmap_recon is None or not viewer.s.training_frames:
        viewer.s.last_error = "Load COLMAP dataset first."
        return
    try:
        apply_dataset_init_defaults(viewer)
        init = viewer.init_params()
        if viewer.s.colmap_point_positions_buffer is None or viewer.s.colmap_point_colors_buffer is None or viewer.s.colmap_point_count <= 0:
            raise RuntimeError("COLMAP pointcloud buffers are not initialized.")
        width, height = int(viewer.s.training_frames[0].width), int(viewer.s.training_frames[0].height)
        renderer = ensure_renderer(viewer, "training_renderer", width, height, allow_debug_overlays=False)
        params = viewer.training_params()
        if viewer.s.trainer is None:
            viewer.s.trainer = GaussianTrainer(
                device=viewer.device,
                renderer=renderer,
                scene=None,
                scene_count=init.gaussian_count,
                upload_initial_scene=False,
                frames=viewer.s.training_frames,
                adam_hparams=params.adam,
                stability_hparams=params.stability,
                training_hparams=params.training,
                seed=init.seed,
                init_point_positions_buffer=viewer.s.colmap_point_positions_buffer,
                init_point_colors_buffer=viewer.s.colmap_point_colors_buffer,
                init_point_count=viewer.s.colmap_point_count,
            )
        else:
            viewer.s.trainer.renderer = renderer
            viewer.s.trainer.update_hyperparams(params.adam, params.stability, params.training)
        viewer.s.trainer.initialize_scene_from_pointcloud(splat_count=init.gaussian_count, init_hparams=init.hparams, seed=init.seed)
        viewer.s.scene = SceneCountProxy(init.gaussian_count)
        enc = viewer.device.create_command_encoder()
        renderer.copy_scene_state_to(enc, viewer.s.renderer)
        viewer.device.submit_command_buffer(enc.finish())
        viewer.s.training_active = False
        _invalidate_synced_steps(viewer)
        viewer.s.scene_init_signature = current_scene_init_signature(viewer)
        update_debug_frame_slider_range(viewer)
        reset_loss_debug_state(viewer)
        viewer.s.last_error = ""
        print(f"Initialized training scene ({init.gaussian_count:,} gaussians)")
    except Exception as exc:
        viewer.s.last_error = str(exc)
        viewer.s.training_active = False
        viewer.s.trainer = None
        print(f"Training scene init failed: {exc}")


def start_training(viewer: object) -> None:
    signature = current_scene_init_signature(viewer)
    if viewer.s.trainer is None or (signature is not None and signature != viewer.s.scene_init_signature):
        initialize_training_scene(viewer)
    viewer.s.training_active = viewer.s.trainer is not None


def stop_training(viewer: object) -> None:
    viewer.s.training_active = False

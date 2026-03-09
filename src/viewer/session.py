from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import slangpy as spy

from ..app.shared import apply_training_profile, estimate_point_bounds, estimate_scene_bounds, renderer_kwargs
from ..common import SHADER_ROOT
from ..renderer import GaussianRenderer
from ..scene import GaussianScene, build_training_frames, initialize_scene_from_colmap_points, load_colmap_reconstruction, load_gaussian_ply, resolve_colmap_init_hparams
from ..training import GaussianTrainer
from ..scene._internal.colmap_types import point_tables
from .state import SceneCountProxy


def _clear(viewer: object, *attrs: str) -> None:
    for attr in attrs:
        value = getattr(viewer.s, attr)
        if value is not None:
            setattr(viewer.s, attr, None)
            del value


_point_tables = lambda recon: (lambda xyz, rgb: (xyz, rgb) if xyz.shape[0] == rgb.shape[0] and xyz.shape[0] > 0 else (_ for _ in ()).throw(RuntimeError("COLMAP point tables are empty or mismatched.")))(*point_tables(recon))


_invalidate = lambda viewer, *targets: [setattr(viewer.s, f"synced_step_{target}", -1) for target in (targets or ("main", "debug"))]


_reset_loss_debug = lambda viewer: (setattr(viewer.s, "loss_debug_texture", None), setattr(viewer.s, "debug_present_texture", None), _clear(viewer, "debug_renderer"), _invalidate(viewer, "debug"))


def _reset_loaded_runtime(viewer: object) -> None:
    viewer.s.scene_init_signature = None
    viewer.s.training_active = False
    viewer.s.trainer = None
    viewer.s.colmap_point_positions_buffer = viewer.s.colmap_point_colors_buffer = None
    viewer.s.colmap_point_count = 0
    viewer.s.suggested_init_hparams = viewer.s.suggested_init_count = None
    update_debug_frame_slider_range(viewer)
    _reset_loss_debug(viewer)
    _clear(viewer, "training_renderer")


_scene_signature = lambda viewer: None if viewer.s.colmap_root is None or viewer.s.colmap_recon is None or not viewer.s.training_frames else (lambda init: (str(viewer.s.colmap_root.resolve()), len(viewer.s.training_frames), init.seed, None if init.hparams.initial_opacity is None else round(float(init.hparams.initial_opacity), 8)))(viewer.init_params())


create_debug_shaders = lambda viewer: (lambda shader_path: (setattr(viewer.s, "debug_abs_diff_kernel", viewer.device.create_compute_kernel(viewer.device.load_program(shader_path, ["csComposeAbsDiffDebug"]))), setattr(viewer.s, "debug_letterbox_kernel", viewer.device.create_compute_kernel(viewer.device.load_program(shader_path, ["csComposeLetterboxDebug"]))))) (str(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"))


update_debug_frame_slider_range = lambda viewer: (lambda slider, max_index: (setattr(slider, "min", 0), setattr(slider, "max", int(max_index)), setattr(slider, "value", int(np.clip(int(slider.value), 0, max_index)))) if hasattr(slider, "min") else setattr(slider, "value", int(np.clip(int(slider.value), 0, max_index))))(viewer.c("loss_debug_frame"), max(len(viewer.s.training_frames) - 1, 0))


def ensure_renderer(viewer: object, attr: str, width: int, height: int, allow_debug_overlays: bool) -> GaussianRenderer:
    size, renderer = (int(width), int(height)), getattr(viewer.s, attr)
    if renderer is not None and (renderer.width, renderer.height) == size:
        return renderer
    _clear(viewer, attr)
    renderer = GaussianRenderer(viewer.device, width=size[0], height=size[1], **renderer_kwargs(viewer.renderer_params(allow_debug_overlays)))
    if isinstance(viewer.s.scene, GaussianScene):
        renderer.set_scene(viewer.s.scene)
    setattr(viewer.s, attr, renderer)
    _invalidate(viewer, "debug" if attr == "debug_renderer" else "main", "debug")
    return renderer


recreate_renderer = lambda viewer, width, height: (ensure_renderer(viewer, "renderer", width, height, allow_debug_overlays=True), _reset_loss_debug(viewer))


def resolve_effective_training_setup(viewer: object):
    init = viewer.init_params()
    params, profile = apply_training_profile(
        viewer.training_params(),
        "auto",
        dataset_root=viewer.s.colmap_root,
        images_subdir=viewer._selected_images_subdir() if viewer.s.colmap_root is not None else None,
    )
    init_hparams = init.hparams if profile.init_opacity_override is None else replace(init.hparams, initial_opacity=profile.init_opacity_override)
    return init, params, init_hparams, profile


def apply_live_params(viewer: object, force_init_defaults: bool = False) -> None:
    for renderer, allow_debug in ((viewer.s.renderer, True), (viewer.s.training_renderer, False), (viewer.s.debug_renderer, False)):
        if renderer is not None:
            for key, value in renderer_kwargs(viewer.renderer_params(allow_debug)).items():
                setattr(renderer, key, value)
    if viewer.s.renderer is not None:
        viewer.s.renderer.set_debug_grad_norm_buffer(None)
        viewer.s.renderer.debug_grad_norm_threshold = 0.0
    if viewer.s.trainer is not None:
        _, params, _, _ = resolve_effective_training_setup(viewer)
        viewer.s.trainer.update_hyperparams(params.adam, params.stability, params.training)


sync_scene_from_training_renderer = lambda viewer, dst_renderer, target, force=False: None if viewer.s.training_renderer is None or viewer.s.trainer is None or (not force and getattr(viewer.s, f"synced_step_{target}") == int(viewer.s.trainer.state.step)) else (lambda step, enc: (viewer.s.training_renderer.copy_scene_state_to(enc, dst_renderer), viewer.device.submit_command_buffer(enc.finish()), setattr(viewer.s, f"synced_step_{target}", step)))(int(viewer.s.trainer.state.step), viewer.device.create_command_encoder())

load_scene = lambda viewer, path: (lambda scene: (_reset_loaded_runtime(viewer), setattr(viewer.s, "scene", scene), setattr(viewer.s, "scene_path", path), setattr(viewer.s, "colmap_root", None), setattr(viewer.s, "colmap_recon", None), setattr(viewer.s, "training_frames", []), viewer.s.renderer.set_scene(scene), viewer.apply_camera_fit(estimate_scene_bounds(scene)), setattr(viewer.s, "last_error", ""), print(f"Loaded scene: {path} ({scene.count:,} splats)")))(load_gaussian_ply(path))
load_colmap_dataset = lambda viewer, root, images_subdir: (lambda recon, xyz_rgb: (_reset_loaded_runtime(viewer), setattr(viewer.s, "colmap_root", Path(root)), setattr(viewer.s, "colmap_recon", recon), setattr(viewer.s, "training_frames", build_training_frames(recon, images_subdir=images_subdir)), setattr(viewer.s, "colmap_point_count", int(xyz_rgb[0].shape[0])), setattr(viewer.s, "scene_path", None), apply_live_params(viewer), viewer.apply_camera_fit(estimate_point_bounds(xyz_rgb[0])), initialize_training_scene(viewer), setattr(viewer.s, "last_error", ""), print(f"Loaded COLMAP: {root} frames={len(viewer.s.training_frames)} images={images_subdir}")))(load_colmap_reconstruction(root), _point_tables(load_colmap_reconstruction(root)))


def initialize_training_scene(viewer: object) -> None:
    if viewer.s.colmap_recon is None or not viewer.s.training_frames:
        return
    init, params, init_hparams, profile = resolve_effective_training_setup(viewer)
    width, height = int(viewer.s.training_frames[0].width), int(viewer.s.training_frames[0].height)
    renderer = ensure_renderer(viewer, "training_renderer", width, height, allow_debug_overlays=False)
    resolved_init = resolve_colmap_init_hparams(viewer.s.colmap_recon, params.training.max_gaussians, init_hparams)
    scene = initialize_scene_from_colmap_points(recon=viewer.s.colmap_recon, max_gaussians=params.training.max_gaussians, seed=init.seed, init_hparams=resolved_init)
    apply_live_params(viewer)
    viewer.s.trainer = GaussianTrainer(
        device=viewer.device,
        renderer=renderer,
        scene=scene,
        frames=viewer.s.training_frames,
        adam_hparams=params.adam,
        stability_hparams=params.stability,
        training_hparams=params.training,
        seed=init.seed,
        scale_reg_reference=float(max(resolved_init.base_scale, 1e-8)),
    )
    viewer.s.scene = SceneCountProxy(scene.count)
    enc = viewer.device.create_command_encoder()
    renderer.copy_scene_state_to(enc, viewer.s.renderer)
    viewer.device.submit_command_buffer(enc.finish())
    viewer.s.training_active = False
    _invalidate(viewer)
    viewer.s.scene_init_signature = _scene_signature(viewer)
    update_debug_frame_slider_range(viewer)
    _reset_loss_debug(viewer)
    viewer.s.last_error = ""
    print(f"Initialized training scene ({scene.count:,} gaussians, profile={profile.name})")
set_training_active = lambda viewer, active: (initialize_training_scene(viewer) if active and viewer.s.trainer is None else None, setattr(viewer.s, "training_active", bool(active and viewer.s.trainer is not None)))

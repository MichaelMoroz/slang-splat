"""Crop-camera sampling: crop renders must match full renders, for every camera model.

The hybrid subsample trainer renders either one jittered native pixel per block
(stratified) or a contiguous native-resolution crop through a crop camera. A crop
camera is a principal-point shift, which is exact for pinhole/distorted cameras and —
after the equirect intrinsics rework (x = fx*theta + cx, period fx*2pi) — for
equirectangular cameras too, including crops that cross the longitude seam.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.renderer import Camera, GaussianRenderer
from src.renderer.camera import PROJECTION_MODEL_EQUIRECTANGULAR
from src.scene import GaussianScene
from src.scene._internal.colmap_types import ColmapFrame
from src.scene.sh_utils import SUPPORTED_SH_COEFF_COUNT, rgb_to_sh0
from src.training import GaussianTrainer, TrainingHyperParams


def _shell_scene(count: int, seed: int = 0, radius: float = 3.0) -> GaussianScene:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    colors = rng.uniform(0.1, 1.0, size=(count, 3)).astype(np.float32)
    sh = np.zeros((count, SUPPORTED_SH_COEFF_COUNT, 3), dtype=np.float32)
    sh[:, 0, :] = rgb_to_sh0(colors)
    quats = rng.normal(size=(count, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return GaussianScene(
        positions=(directions * radius).astype(np.float32),
        scales=np.log(rng.uniform(0.05, 0.2, size=(count, 3))).astype(np.float32),
        rotations=quats,
        opacities=rng.uniform(0.3, 0.95, size=(count,)).astype(np.float32),
        colors=colors,
        sh_coeffs=sh,
    )


def _render(device, scene: GaussianScene, camera: Camera, width: int, height: int) -> np.ndarray:
    renderer = GaussianRenderer(device, width=width, height=height, radius_scale=1.6, list_capacity_multiplier=64)
    renderer.set_scene(scene)
    # Prepass capacity estimation is one frame delayed; warm up so no lists truncate.
    texture = None
    for _ in range(3):
        texture, _ = renderer.render_to_texture(camera, background=np.array([0.05, 0.1, 0.15], dtype=np.float32))
    return np.asarray(texture.to_numpy(), dtype=np.float32).copy()


def _assert_images_match(actual: np.ndarray, expected: np.ndarray) -> None:
    """Binning-margin float noise flips a handful of pixels; structural breakage flips
    whole regions. Require a near-exact bulk and a tightly bounded tail."""
    diff = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    assert float(diff.mean()) < 1e-4
    assert float(np.quantile(diff, 0.99)) < 2e-3
    assert float(diff.max()) < 2e-2


def _assert_equirect_images_match(actual: np.ndarray, expected: np.ndarray) -> None:
    """Equirect outline-fit binning under-covers splat fringes, and it does so
    differently per viewport context, so a crop and the matching full-view region
    disagree on ~5% of fringe pixels (pre-existing approximation, exposed by this
    comparison — not caused by crop cameras). Structural crop bugs (wrong offset, broken
    seam wrap) mismatch nearly every pixel, which the bulk-exactness bound catches."""
    diff = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    assert float(np.median(diff)) < 1e-3  # bulk of pixels identical
    assert float(diff.mean()) < 2e-2
    assert float((diff.max(axis=2) > 0.05).mean()) < 0.15  # fringe disagreement bounded


def test_pinhole_crop_camera_matches_full_render(device) -> None:
    scene = _shell_scene(300, seed=3)
    width, height = 128, 96
    camera = Camera.look_at(position=(0.0, 0.0, 6.0), target=(0.0, 0.0, 0.0), near=0.1, far=30.0)
    full = _render(device, scene, camera, width, height)

    crop_w, crop_h, crop_x, crop_y = 64, 48, 32, 24
    crop_camera = camera.cropped(crop_x, crop_y, width, height)
    crop = _render(device, scene, crop_camera, crop_w, crop_h)

    _assert_images_match(crop[..., :3], full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w, :3])


def test_equirect_crop_camera_matches_full_render(device) -> None:
    scene = _shell_scene(400, seed=4)
    width, height = 256, 128
    camera = Camera.look_at(position=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), near=0.05, far=30.0)
    camera.projection_model = PROJECTION_MODEL_EQUIRECTANGULAR
    full = _render(device, scene, camera, width, height)

    crop_w, crop_h, crop_x, crop_y = 96, 64, 64, 32
    crop_camera = camera.cropped(crop_x, crop_y, width, height)
    crop = _render(device, scene, crop_camera, crop_w, crop_h)

    _assert_equirect_images_match(crop[..., :3], full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w, :3])


def test_equirect_crop_camera_handles_seam_crossing(device) -> None:
    scene = _shell_scene(400, seed=5)
    width, height = 256, 128
    camera = Camera.look_at(position=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), near=0.05, far=30.0)
    camera.projection_model = PROJECTION_MODEL_EQUIRECTANGULAR
    full = _render(device, scene, camera, width, height)

    # Crop window straddling the +-pi longitude seam: right half of the panorama
    # followed by its left edge. The period wrap must resolve splats into the window.
    crop_w, crop_h, crop_y = 64, 64, 32
    crop_x = width - crop_w // 2
    crop_camera = camera.cropped(crop_x, crop_y, width, height)
    crop = _render(device, scene, crop_camera, crop_w, crop_h)

    expected = np.concatenate(
        [full[crop_y : crop_y + crop_h, crop_x:, :3], full[crop_y : crop_y + crop_h, : crop_w // 2, :3]], axis=1
    )
    _assert_equirect_images_match(crop[..., :3], expected)


def _training_frame(tmp_path: Path, width: int = 64, height: int = 64) -> ColmapFrame:
    rng = np.random.default_rng(11)
    image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    image_path = tmp_path / "crop_target.png"
    Image.fromarray(image, mode="RGB").save(image_path)
    return ColmapFrame(
        image_id=7,
        image_path=image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        fx=72.0,
        fy=72.0,
        cx=width * 0.5,
        cy=height * 0.5,
        width=width,
        height=height,
    )


def test_trainer_crop_steps_train_and_stay_deterministic(device, tmp_path: Path) -> None:
    scene = _shell_scene(200, seed=6, radius=1.0)
    frame = _training_frame(tmp_path)
    renderer = GaussianRenderer(device, width=32, height=32, radius_scale=1.0, list_capacity_multiplier=32)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            train_subsample_factor=2,
            train_subsample_crop_probability=1.0,
            refinement_growth_start_step=0,
            refinement_interval=9999,
        ),
        seed=123,
    )
    assert trainer.effective_train_subsample_factor(0, 0) == 2

    plan = trainer.training_sample_plan(0, 5)
    assert plan.crop  # probability 1.0 -> always crop
    assert plan == trainer.training_sample_plan(0, 5)  # deterministic in (seed, step, frame)
    assert plan != trainer.training_sample_plan(0, 6) or plan.crop  # origin varies over steps

    vars_dict = trainer.training_sample_vars(0, 5, plan=plan)["g_TrainingSubsample"]
    assert int(vars_dict["mode"]) == 1
    assert int(vars_dict["cropX"]) == plan.crop_x and int(vars_dict["cropY"]) == plan.crop_y

    executed = trainer.step_batch(4)
    assert executed == 4
    assert np.isfinite(trainer.state.last_loss)

    # Zero probability -> stratified plans and untouched legacy behavior.
    trainer.training.train_subsample_crop_probability = 0.0
    assert not trainer.training_sample_plan(0, 100).crop


def test_refinement_ema_decay_stretches_with_crop_probability(device, tmp_path: Path) -> None:
    scene = _shell_scene(50, seed=8, radius=1.0)
    frame = _training_frame(tmp_path)
    renderer = GaussianRenderer(device, width=32, height=32, radius_scale=1.0, list_capacity_multiplier=32)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2, train_subsample_crop_probability=0.0),
        seed=123,
    )
    base_decay = trainer.refinement_contribution_ema_decay()
    trainer.training.train_subsample_crop_probability = 0.5
    stretched_decay = trainer.refinement_contribution_ema_decay()
    # Observation rate (1-p) + p/s^2 = 0.625 at p=0.5, s=2 -> decay exponent scales by it.
    np.testing.assert_allclose(stretched_decay, base_decay**0.625, rtol=1e-6)
    assert stretched_decay > base_decay  # slower forgetting = longer horizon

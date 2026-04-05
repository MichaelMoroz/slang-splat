from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import slangpy as spy

from src.common import buffer_to_numpy
from src.filter import SeparableGaussianBlur
from src.renderer import GaussianRenderer
from src.scene import ColmapFrame, GaussianInitHyperParams, GaussianScene
from src.training import gaussian_trainer as gaussian_trainer_module
from src.training import AdamHyperParams, GaussianTrainer, StabilityHyperParams, TRAIN_BACKGROUND_MODE_CUSTOM, TRAIN_BACKGROUND_MODE_RANDOM, TrainingHyperParams, contribution_fixed_count_from_percent, contribution_percent_from_fixed_count, resolve_clone_probability_threshold, resolve_cosine_base_learning_rate, resolve_effective_maintenance_interval, resolve_maintenance_contribution_cull_threshold, resolve_maintenance_growth_ratio, resolve_max_allowed_density, resolve_training_resolution, should_run_maintenance_step

_ADAM_BUFFER_NAMES = ("adam_moments",)
_OPACITY_EPS = 1e-6
_raw_opacity = lambda alpha: (np.log(np.clip(np.asarray(alpha, dtype=np.float32), _OPACITY_EPS, 1.0 - _OPACITY_EPS)) - np.log1p(-np.clip(np.asarray(alpha, dtype=np.float32), _OPACITY_EPS, 1.0 - _OPACITY_EPS))).astype(np.float32, copy=False)
_actual_opacity = lambda raw: (1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float32)))).astype(np.float32, copy=False)
_actual_scale = lambda log_scale: np.exp(np.asarray(log_scale, dtype=np.float32))
_GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0
_MAINTENANCE_MIN_SCREEN_RADIUS_PX = 1.0
_TRAINING_MAX_SCREEN_FRACTION = 0.1
_log_sigma = lambda sigma: np.log(np.asarray(sigma, dtype=np.float32))
_stored_from_support_scale = lambda support_scale: np.log(np.asarray(support_scale, dtype=np.float32) / _GAUSSIAN_SUPPORT_SIGMA_RADIUS)
_SCALE_GRAD_MULS = (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 0.9375, 0.96875, 1.0, 1.05)
_TARGET_MULS = (2.0, 3.0, 4.0, 6.0, 7.5)


def _make_scene(count: int = 24, seed: int = 7) -> GaussianScene:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32)
    positions[:, 2] += 2.0
    scales = _log_sigma(np.full((count, 3), 0.03, dtype=np.float32))
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.full((count,), 0.5, dtype=np.float32)
    colors = rng.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32)
    sh_coeffs = np.zeros((count, 1, 3), dtype=np.float32)
    return GaussianScene(positions=positions, scales=scales, rotations=rotations, opacities=opacities, colors=colors, sh_coeffs=sh_coeffs)


def _make_frame(tmp_path: Path, width: int = 64, height: int = 64, *, image_name: str = "target.png", image_id: int = 0, green_value: int = 180) -> ColmapFrame:
    image_path = tmp_path / image_name
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = int(green_value)
    Image.fromarray(image, mode="RGB").save(image_path)
    return ColmapFrame(
        image_id=image_id,
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


def _read_scene_groups(renderer: GaussianRenderer, count: int) -> dict[str, np.ndarray]:
    return renderer.read_scene_groups(count)


def _read_grad_groups(renderer: GaussianRenderer, count: int) -> dict[str, np.ndarray]:
    return renderer.read_grad_groups(count)


def _write_grad_groups(
    renderer: GaussianRenderer,
    count: int,
    *,
    grad_positions: np.ndarray | None = None,
    grad_scales: np.ndarray | None = None,
    grad_rotations: np.ndarray | None = None,
    grad_sh_coeffs: np.ndarray | None = None,
    grad_color_alpha: np.ndarray | None = None,
) -> None:
    renderer.write_grad_groups(
        count,
        grad_positions=grad_positions,
        grad_scales=grad_scales,
        grad_rotations=grad_rotations,
        grad_sh_coeffs=grad_sh_coeffs,
        grad_color_alpha=grad_color_alpha,
    )


def _read_output_grads(renderer: GaussianRenderer) -> np.ndarray:
    flat = buffer_to_numpy(renderer.output_grad_buffer, np.float32)
    return flat[: max(renderer.width * renderer.height, 1) * 4].reshape(renderer.height, renderer.width, 4)


def _read_optimizer_lrs(trainer: GaussianTrainer) -> np.ndarray:
    flat = buffer_to_numpy(trainer.optimizer.param_settings, np.uint32)
    return flat.reshape(trainer.renderer.TRAINABLE_PARAM_COUNT, 8)[:, 0].view(np.float32).copy()


def _read_adam_moments(trainer: GaussianTrainer, splat_count: int) -> np.ndarray:
    flat = buffer_to_numpy(trainer.adam_optimizer.buffers["adam_moments"], np.float32)
    count = max(int(splat_count), 1) * trainer.renderer.TRAINABLE_PARAM_COUNT * 2
    return flat[:count].reshape(trainer.renderer.TRAINABLE_PARAM_COUNT, max(int(splat_count), 1), 2).copy()


def _circle_bound_support_radius(camera, position: np.ndarray, width: int, height: int, radius_px: float) -> float | None:
    center_px, visible = camera.project_world_to_screen(position, width, height)
    if not visible:
        return None
    if center_px[0] < 0.0 or center_px[0] >= float(width) or center_px[1] < 0.0 or center_px[1] >= float(height):
        return None
    viewport_center = np.array([0.5 * float(width), 0.5 * float(height)], dtype=np.float32)
    to_viewport_center = viewport_center - np.asarray(center_px, dtype=np.float32)
    distance = float(np.linalg.norm(to_viewport_center))
    if distance <= 1e-6:
        sample_px = np.asarray(center_px, dtype=np.float32) + np.array([radius_px, 0.0], dtype=np.float32)
    else:
        sample_px = np.asarray(center_px, dtype=np.float32) + to_viewport_center * np.float32(radius_px / distance)
    ray_dir = camera.screen_to_world_ray(sample_px, width, height)
    to_center = np.asarray(position, dtype=np.float32) - np.asarray(camera.position, dtype=np.float32)
    return float(np.linalg.norm(np.cross(to_center, ray_dir)))


class _ScaleGradProbe:
    def __init__(self, device, tmp_path: Path, *, image_name: str):
        self.device = device
        self.background = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.frame = _make_frame(tmp_path, image_name=image_name)
        self.camera = self.frame.make_camera(near=0.1, far=20.0)
        self.target_renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
        self.pixel_floor_scale = self.camera.pixel_world_size_max(3.0, self.target_renderer.width, self.target_renderer.height)
        self.target_scene = GaussianScene(
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            scales=np.full((1, 3), _stored_from_support_scale(self.pixel_floor_scale), dtype=np.float32),
            rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            opacities=np.array([0.75], dtype=np.float32),
            colors=np.array([[0.8, 0.6, 0.2]], dtype=np.float32),
            sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
        )
        self.renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
        self.trainer = GaussianTrainer(
            device=device,
            renderer=self.renderer,
            scene=GaussianScene(
                positions=self.target_scene.positions.copy(),
                scales=np.full((1, 3), _stored_from_support_scale(self.pixel_floor_scale), dtype=np.float32),
                rotations=self.target_scene.rotations.copy(),
                opacities=self.target_scene.opacities.copy(),
                colors=self.target_scene.colors.copy(),
                sh_coeffs=self.target_scene.sh_coeffs.copy(),
            ),
            frames=[self.frame],
            adam_hparams=AdamHyperParams(position_lr=0.0, scale_lr=0.1, rotation_lr=0.0, color_lr=0.0, opacity_lr=0.0),
            stability_hparams=StabilityHyperParams(max_update=0.5, min_scale=1e-5, max_scale=16.0 * self.pixel_floor_scale),
            training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0),
            seed=123,
        )
        self.scene_groups = _read_scene_groups(self.renderer, 1)
        self.target_texture = self.trainer.get_frame_target_texture(0, native_resolution=True)
        self._last_target_scale = float("nan")

    def upload_target_scale(self, target_scale: float) -> None:
        if float(target_scale) == self._last_target_scale: return
        self.target_scene.scales[...] = _stored_from_support_scale(float(target_scale))
        rgb = np.clip(
            np.asarray(
                self.target_renderer.render(self.target_scene, self.camera, background=self.background).image,
                dtype=np.float32,
            )[..., :3],
            0.0,
            1.0,
        )
        rgba = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = (255.0 * rgb + 0.5).astype(np.uint8)
        rgba[..., 3] = 255
        self.target_texture.copy_from_numpy(np.ascontiguousarray(rgba, dtype=np.uint8))
        self.trainer._invalidate_downscaled_target()
        self._last_target_scale = float(target_scale)

    def upload_train_scale(self, scale: float) -> None:
        self.scene_groups["scales"][0, :3] = _stored_from_support_scale(float(scale))
        self.renderer.write_scene_groups(
            1,
            positions=self.scene_groups["positions"],
            scales=self.scene_groups["scales"],
            rotations=self.scene_groups["rotations"],
            color_alpha=self.scene_groups["color_alpha"],
        )

    def dispatch_gradient(self) -> None:
        self.renderer.execute_prepass_for_current_scene(self.camera, sync_counts=False)
        enc = self.device.create_command_encoder()
        self.trainer._dispatch_raster_training_forward(enc, self.camera, self.background)
        target_texture = self.trainer.get_frame_target_texture(0, native_resolution=False, encoder=enc)
        self.trainer._dispatch_loss_forward(enc, target_texture)
        self.trainer._dispatch_loss_backward(enc, target_texture)
        self.trainer._dispatch_raster_backward(enc, self.camera, self.background)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

    def read_scale_grad_mean(self) -> float:
        return float(np.mean(_read_grad_groups(self.renderer, 1)["grad_scales"][0, :3]))

    def measure_scale_grad_mean(self, scale: float, target_scale: float) -> float:
        self.upload_target_scale(target_scale)
        self.upload_train_scale(scale)
        self.dispatch_gradient()
        return self.read_scale_grad_mean()


def _scale_target_pairs(scale_muls: tuple[float, ...], target_muls: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    return tuple((scale_mul, target_mul) for target_mul in target_muls for scale_mul in scale_muls if scale_mul < target_mul)


def test_training_step_smoke_updates_params_without_changing_count(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    before = _read_scene_groups(renderer, scene.count)["positions"].copy()
    loss = trainer.step()
    after = _read_scene_groups(renderer, scene.count)["positions"]

    assert np.isfinite(loss)
    assert trainer.state.step == 1
    assert np.isfinite(trainer.state.last_mse)
    assert np.isfinite(trainer.state.avg_loss)
    assert trainer.scene.count == scene.count
    assert np.all(np.isfinite(after))
    assert np.any(np.abs(after - before) > 0.0)


def test_training_step_smoke_updates_params_in_fixed_atomic_mode(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path, image_name="fixed_mode_target.png", image_id=7)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32, cached_raster_grad_atomic_mode="fixed")
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=321)

    before = _read_scene_groups(renderer, scene.count)["positions"].copy()
    loss = trainer.step()
    after = _read_scene_groups(renderer, scene.count)["positions"]

    assert renderer.cached_raster_grad_atomic_mode == "fixed"
    assert np.isfinite(loss)
    assert np.all(np.isfinite(after))
    assert np.any(np.abs(after - before) > 0.0)


def test_random_training_backgrounds_are_seeded_and_vary(device, tmp_path: Path):
    scene = _make_scene(count=4, seed=17)
    frame = _make_frame(tmp_path, image_name="random_background_target.png", image_id=11)
    renderer_a = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    renderer_b = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer_a = GaussianTrainer(device=device, renderer=renderer_a, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(background_mode=TRAIN_BACKGROUND_MODE_RANDOM), seed=123)
    trainer_b = GaussianTrainer(device=device, renderer=renderer_b, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(background_mode=TRAIN_BACKGROUND_MODE_RANDOM), seed=123)

    bg_a0 = trainer_a._training_background()
    bg_a1 = trainer_a._training_background()
    bg_b0 = trainer_b._training_background()
    bg_b1 = trainer_b._training_background()

    assert np.all(bg_a0 >= 0.0) and np.all(bg_a0 <= 1.0)
    assert np.all(bg_a1 >= 0.0) and np.all(bg_a1 <= 1.0)
    assert not np.allclose(bg_a0, bg_a1)
    assert np.allclose(bg_a0, bg_b0)
    assert np.allclose(bg_a1, bg_b1)


def test_custom_training_background_returns_configured_color(device, tmp_path: Path):
    scene = _make_scene(count=4, seed=18)
    frame = _make_frame(tmp_path, image_name="custom_background_target.png", image_id=12)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    color = (1.0, 1.0, 1.0)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(background_mode=TRAIN_BACKGROUND_MODE_CUSTOM, background=color),
        seed=123,
    )

    np.testing.assert_allclose(trainer._training_background(), np.asarray(color, dtype=np.float32), rtol=0.0, atol=0.0)


def test_training_step_batch_updates_params_without_changing_count(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path, image_name="batch_target.png", image_id=1)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    before = _read_scene_groups(renderer, scene.count)["positions"].copy()
    executed = trainer.step_batch(3)
    after = _read_scene_groups(renderer, scene.count)["positions"]

    assert executed == 3
    assert trainer.state.step == 3
    assert np.isfinite(trainer.state.last_loss)
    assert np.isfinite(trainer.state.last_mse)
    assert np.isfinite(trainer.state.avg_loss)
    assert trainer.scene.count == scene.count
    assert np.all(np.isfinite(after))
    assert np.any(np.abs(after - before) > 0.0)


def test_training_step_batch_matches_two_single_steps(device, tmp_path: Path):
    scene = _make_scene(count=8, seed=31)
    frame = _make_frame(tmp_path, image_name="batch_match_target.png", image_id=2)
    renderer_seq = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    renderer_batch = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer_seq = GaussianTrainer(device=device, renderer=renderer_seq, scene=scene, frames=[frame], seed=123)
    trainer_batch = GaussianTrainer(device=device, renderer=renderer_batch, scene=scene, frames=[frame], seed=123)

    trainer_seq.step()
    trainer_seq.step()
    executed = trainer_batch.step_batch(2)

    assert executed == 2
    np.testing.assert_allclose(
        _read_scene_groups(renderer_batch, scene.count)["positions"],
        _read_scene_groups(renderer_seq, scene.count)["positions"],
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        _read_scene_groups(renderer_batch, scene.count)["scales"],
        _read_scene_groups(renderer_seq, scene.count)["scales"],
        rtol=1e-5,
        atol=1e-7,
    )


def test_position_random_steps_move_low_opacity_splats(device, tmp_path: Path):
    scene = _make_scene(count=4, seed=41)
    scene.opacities[:] = np.full((scene.count,), 1e-4, dtype=np.float32)
    scene.scales[:] = _log_sigma(np.full((scene.count, 3), 0.1, dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="position_random_step_low_opacity.png", image_id=13)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            position_random_step_noise_lr=5e4,
        ),
        seed=123,
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    zero_sh = np.zeros((scene.count, 4, 3), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_sh_coeffs=zero_sh, grad_color_alpha=zeros)
    before = _read_scene_groups(renderer, scene.count)["positions"].copy()

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    trainer._dispatch_position_random_steps(enc, 1)
    device.submit_command_buffer(enc.finish())
    device.wait()

    after = _read_scene_groups(renderer, scene.count)["positions"]
    assert np.all(np.isfinite(after))
    assert np.any(np.abs(after - before) > 0.0)


def test_position_random_steps_are_gated_for_high_opacity_splats(device, tmp_path: Path):
    scene = _make_scene(count=4, seed=43)
    scene.opacities[:] = np.full((scene.count,), 0.5, dtype=np.float32)
    scene.scales[:] = _log_sigma(np.full((scene.count, 3), 0.1, dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="position_random_step_high_opacity.png", image_id=14)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            position_random_step_noise_lr=5e4,
        ),
        seed=123,
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    zero_sh = np.zeros((scene.count, 4, 3), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_sh_coeffs=zero_sh, grad_color_alpha=zeros)
    before = _read_scene_groups(renderer, scene.count)["positions"].copy()

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    trainer._dispatch_position_random_steps(enc, 1)
    device.submit_command_buffer(enc.finish())
    device.wait()

    after = _read_scene_groups(renderer, scene.count)["positions"]
    np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-9)


def test_position_random_steps_clamp_anisotropy_for_thin_splats(device, tmp_path: Path):
    scene = _make_scene(count=8, seed=45)
    scene.opacities[:] = np.full((scene.count,), 1e-4, dtype=np.float32)
    scene.scales[:] = _log_sigma(np.array([[1.0, 1e-3, 1e-3]], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="position_random_step_anisotropy_clamp.png", image_id=15)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            position_random_step_noise_lr=5e4,
        ),
        seed=123,
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    zero_sh = np.zeros((scene.count, 4, 3), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_sh_coeffs=zero_sh, grad_color_alpha=zeros)
    before = _read_scene_groups(renderer, scene.count)["positions"].copy()

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    trainer._dispatch_position_random_steps(enc, 1)
    device.submit_command_buffer(enc.finish())
    device.wait()

    after = _read_scene_groups(renderer, scene.count)["positions"]
    displacement = np.abs(after - before)
    assert float(np.mean(displacement[:, 1:3])) > 0.1


def test_split_loss_forward_backward_separates_metrics_from_output_grads(device, tmp_path: Path):
    scene = _make_scene(count=8, seed=19)
    frame = _make_frame(tmp_path, image_name="split_loss_target.png")
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=17)
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer.training.background, dtype=np.float32)

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    renderer.output_grad_buffer.copy_from_numpy(np.zeros((renderer.width * renderer.height, 4), dtype=np.float32))
    enc = device.create_command_encoder()
    trainer._dispatch_raster_training_forward(enc, camera, background)
    target_texture = trainer.get_frame_target_texture(0, native_resolution=False, encoder=enc)
    trainer._dispatch_loss_forward(enc, target_texture)
    device.submit_command_buffer(enc.finish())
    device.wait()

    loss, mse, density_loss = trainer._read_loss_metrics()
    grads_after_forward = _read_output_grads(renderer).copy()

    enc = device.create_command_encoder()
    trainer._dispatch_loss_backward(enc, target_texture)
    device.submit_command_buffer(enc.finish())
    device.wait()

    grads_after_backward = _read_output_grads(renderer)
    np.testing.assert_allclose(trainer._read_loss_metrics(), (loss, mse, density_loss), rtol=0.0, atol=0.0)
    assert np.isfinite(loss)
    assert np.isfinite(mse)
    assert density_loss == 0.0
    assert np.allclose(grads_after_forward, 0.0)
    assert np.any(np.abs(grads_after_backward[..., :3]) > 0.0)
    assert np.all(np.isfinite(grads_after_backward))


def test_split_raster_backward_consumes_forward_cache_only(device, tmp_path: Path):
    scene = _make_scene(count=10, seed=23)
    frame = _make_frame(tmp_path, image_name="split_raster_target.png")
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=29)
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer.training.background, dtype=np.float32)

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    target_texture = trainer.get_frame_target_texture(0, native_resolution=False)

    enc = device.create_command_encoder()
    trainer._dispatch_raster_training_forward(enc, camera, background)
    trainer._dispatch_loss_backward(enc, target_texture)
    device.submit_command_buffer(enc.finish())
    device.wait()

    grads_after_forward = _read_grad_groups(renderer, scene.count)
    for name in grads_after_forward:
        assert np.allclose(grads_after_forward[name], 0.0)

    enc = device.create_command_encoder()
    trainer._dispatch_raster_backward(enc, camera, background)
    device.submit_command_buffer(enc.finish())
    device.wait()

    grads_after_backward = _read_grad_groups(renderer, scene.count)
    assert any(np.any(np.abs(grads_after_backward[name]) > 0.0) for name in grads_after_backward)
    for name in grads_after_backward:
        assert np.all(np.isfinite(grads_after_backward[name]))


def test_scale_gradient_stays_growth_directed_for_large_target_scales(device, tmp_path: Path):
    probe = _ScaleGradProbe(device, tmp_path, image_name="tiny_grad_target.png")
    max_growth_eps = 4e-6
    pairs = _scale_target_pairs(_SCALE_GRAD_MULS, _TARGET_MULS)
    grad_samples = np.asarray(
        [
            probe.measure_scale_grad_mean(scale_mul * probe.pixel_floor_scale, target_mul * probe.pixel_floor_scale)
            for scale_mul, target_mul in pairs
        ],
        dtype=np.float32,
    )

    assert np.all(np.isfinite(grad_samples))
    assert np.all(grad_samples <= max_growth_eps)


def test_training_targets_use_srgb_textures(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    native_target = trainer.get_frame_target_texture(0, native_resolution=True)
    train_target = trainer.get_frame_target_texture(0, native_resolution=False)

    assert native_target.format == spy.Format.rgba8_unorm_srgb
    assert train_target.format == spy.Format.rgba32_float
    assert native_target is not train_target


def test_resolve_training_resolution_uses_ceil_division() -> None:
    assert resolve_training_resolution(64, 32, 1) == (64, 32)
    assert resolve_training_resolution(63, 65, 2) == (32, 33)
    assert resolve_training_resolution(1, 1, 16) == (1, 1)


def test_cosine_base_lr_clamps_after_schedule_end() -> None:
    hparams = TrainingHyperParams(lr_schedule_start_lr=1e-2, lr_schedule_end_lr=1e-3, lr_schedule_steps=4)

    np.testing.assert_allclose(resolve_cosine_base_learning_rate(hparams, 0), 1e-2, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_cosine_base_learning_rate(hparams, 4), 1e-3, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_cosine_base_learning_rate(hparams, 40), 1e-3, rtol=0.0, atol=1e-10)


def test_training_step_updates_optimizer_lrs_from_cosine_schedule(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=77)
    frame = _make_frame(tmp_path, image_name="lr_schedule_target.png", image_id=5)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    adam = AdamHyperParams(position_lr=1e-2, scale_lr=2e-2, rotation_lr=3e-2, color_lr=4e-2, opacity_lr=5e-2)
    training = TrainingHyperParams(
        lr_schedule_start_lr=1e-2,
        lr_schedule_end_lr=1e-3,
        lr_schedule_steps=2,
        scale_l2_weight=0.0,
        scale_abs_reg_weight=0.0,
        opacity_reg_weight=0.0,
    )
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], adam_hparams=adam, training_hparams=training, seed=123)

    before = _read_optimizer_lrs(trainer)
    trainer.step()
    after = _read_optimizer_lrs(trainer)
    expected_scale = resolve_cosine_base_learning_rate(training, 1) / training.lr_schedule_start_lr

    np.testing.assert_allclose(before[[0, 3, 6, 10, 22]], np.array([1e-2, 2e-2, 3e-2, 4e-2, 5e-2], dtype=np.float32), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(after[[0, 3, 6, 10, 22]], expected_scale * np.array([1e-2, 2e-2, 3e-2, 4e-2, 5e-2], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(trainer.state.last_base_lr, resolve_cosine_base_learning_rate(training, 1), rtol=0.0, atol=1e-7)


def test_loss_vars_use_density_schedule(device, tmp_path: Path) -> None:
    scene = _make_scene(count=4, seed=93)
    frame = _make_frame(tmp_path, image_name="density_schedule_target.png", image_id=19)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    training = TrainingHyperParams(lr_schedule_steps=2)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=training, seed=123)

    np.testing.assert_allclose(trainer._loss_vars(0)["g_DensityRegularizer"], 0.05, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(trainer._loss_vars(0)["g_MaxAllowedDensity"], 5.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(trainer._loss_vars(1)["g_MaxAllowedDensity"], 8.5, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(trainer._loss_vars(2)["g_MaxAllowedDensity"], 12.0, rtol=0.0, atol=1e-10)


def test_max_allowed_density_schedule_clamps_to_end_value() -> None:
    hparams = TrainingHyperParams(max_allowed_density_start=5.0, max_allowed_density=12.0, lr_schedule_steps=4)

    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 0), 5.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 2), 8.5, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 4), 12.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 40), 12.0, rtol=0.0, atol=1e-10)


def test_maintenance_cadence_and_clone_probability_follow_growth_budget() -> None:
    hparams = TrainingHyperParams(maintenance_interval=200, maintenance_growth_ratio=0.05, maintenance_growth_start_step=0)

    assert not should_run_maintenance_step(hparams, 199)
    assert should_run_maintenance_step(hparams, 200)
    np.testing.assert_allclose(
        resolve_clone_probability_threshold(hparams, splat_count=1000, pixel_count=64 * 32, step=200),
        1000.0 * 0.05 / 200.0 / float(64 * 32),
        rtol=0.0,
        atol=1e-12,
    )


def test_maintenance_interval_is_floored_by_frame_count() -> None:
    hparams = TrainingHyperParams(maintenance_interval=2, maintenance_growth_ratio=0.05, maintenance_growth_start_step=0)

    assert resolve_effective_maintenance_interval(hparams, frame_count=5) == 5
    assert not should_run_maintenance_step(hparams, 4, frame_count=5)
    assert should_run_maintenance_step(hparams, 5, frame_count=5)
    np.testing.assert_allclose(
        resolve_clone_probability_threshold(hparams, splat_count=1000, pixel_count=64 * 32, step=5, frame_count=5),
        1000.0 * 0.05 / 5.0 / float(64 * 32),
        rtol=0.0,
        atol=1e-12,
    )


def test_maintenance_growth_stays_zero_until_start_step() -> None:
    hparams = TrainingHyperParams(maintenance_interval=200, maintenance_growth_ratio=0.02, maintenance_growth_start_step=2000)

    np.testing.assert_allclose(resolve_maintenance_growth_ratio(hparams, 1999), 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_clone_probability_threshold(hparams, splat_count=1000, pixel_count=100, step=1999), 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_maintenance_growth_ratio(hparams, 2000), 0.02, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_clone_probability_threshold(hparams, splat_count=1000, pixel_count=100, step=2000), 1000.0 * 0.02 / 200.0 / 100.0, rtol=0.0, atol=1e-12)


def test_maintenance_contribution_cull_threshold_decays_by_15_percent_per_maintenance_step() -> None:
    hparams = TrainingHyperParams(maintenance_interval=200, maintenance_contribution_cull_threshold=0.001)

    np.testing.assert_allclose(resolve_maintenance_contribution_cull_threshold(hparams, 0), 0.001, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_maintenance_contribution_cull_threshold(hparams, 199), 0.001, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_maintenance_contribution_cull_threshold(hparams, 200), 0.00085, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_maintenance_contribution_cull_threshold(hparams, 400), 0.0007225, rtol=0.0, atol=1e-12)


def test_clone_probability_threshold_respects_max_gaussians_cap() -> None:
    hparams = TrainingHyperParams(maintenance_interval=200, maintenance_growth_ratio=0.05, maintenance_growth_start_step=0, max_gaussians=1024)

    np.testing.assert_allclose(
        resolve_clone_probability_threshold(hparams, splat_count=1000, pixel_count=100, step=200),
        24.0 / 200.0 / 100.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert resolve_clone_probability_threshold(hparams, splat_count=1024, pixel_count=100, step=200) == 0.0


def test_trainer_allocates_minimal_maintenance_buffers(device, tmp_path: Path) -> None:
    scene = _make_scene(count=32, seed=81)
    frame = _make_frame(tmp_path, image_name="maintenance_buffers_target.png", image_id=9)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(maintenance_growth_ratio=0.05, maintenance_growth_start_step=0),
        seed=123,
    )

    assert set(trainer.maintenance_buffers) == {"total_clone_counter", "clone_counts", "splat_contribution", "append_counter", "append_params", "dst_splat_params", "dst_adam_moments", "camera_rows"}
    assert trainer.clone_probability_threshold() > 0.0


def test_trainer_maintenance_due_uses_dataset_frame_floor(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=82)
    frames = [
        _make_frame(tmp_path, image_name="maintenance_floor_a.png", image_id=90),
        _make_frame(tmp_path, image_name="maintenance_floor_b.png", image_id=91),
        _make_frame(tmp_path, image_name="maintenance_floor_c.png", image_id=92),
    ]
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=frames,
        training_hparams=TrainingHyperParams(maintenance_interval=1, maintenance_growth_ratio=0.05, maintenance_growth_start_step=0),
        seed=123,
    )

    assert trainer.effective_maintenance_interval() == 3
    assert not trainer.maintenance_due(1)
    assert not trainer.maintenance_due(2)
    assert trainer.maintenance_due(3)


def test_training_forward_accumulates_clone_counts(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=83)
    frame = _make_frame(tmp_path, image_name="clone_counts_target.png", image_id=10)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(maintenance_growth_ratio=0.05, maintenance_growth_start_step=0, maintenance_interval=9999),
        seed=123,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer.training.background, dtype=np.float32)

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    enc = device.create_command_encoder()
    renderer.rasterize_training_forward_current_scene(
        enc,
        camera,
        background,
        clone_counts_buffer=trainer.maintenance_buffers["clone_counts"],
        splat_contribution_buffer=trainer.maintenance_buffers["splat_contribution"],
        clone_select_probability=1.0,
        clone_seed=123,
    )
    device.submit_command_buffer(enc.finish())
    device.wait()

    clone_counts = buffer_to_numpy(trainer.maintenance_buffers["clone_counts"], np.uint32)[: scene.count]
    contributions = buffer_to_numpy(trainer.maintenance_buffers["splat_contribution"], np.uint32)[: scene.count]
    assert np.any(clone_counts > 0)
    assert np.any(contributions > 0)


def test_density_loss_respects_threshold_and_changes_gradients(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=89)
    frame = _make_frame(tmp_path, image_name="density_enabled_target.png", image_id=16)
    renderer_off = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    renderer_on = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer_off = GaussianTrainer(device=device, renderer=renderer_off, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(density_regularizer=0.0), seed=123)
    trainer_on = GaussianTrainer(device=device, renderer=renderer_on, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(density_regularizer=0.5, max_allowed_density_start=0.0, max_allowed_density=0.0), seed=123)
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer_on.training.background, dtype=np.float32)

    def run_pass(trainer: GaussianTrainer, renderer: GaussianRenderer):
        renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
        enc = device.create_command_encoder()
        trainer._dispatch_raster_training_forward(enc, camera, background)
        target_texture = trainer.get_frame_target_texture(0, native_resolution=False, encoder=enc)
        trainer._dispatch_loss_forward(enc, target_texture)
        trainer._dispatch_loss_backward(enc, target_texture)
        trainer._dispatch_raster_backward(enc, camera, background)
        device.submit_command_buffer(enc.finish())
        device.wait()
        return trainer._read_loss_metrics(), _read_grad_groups(renderer, scene.count)

    (total_off, mse_off, density_off), grads_off = run_pass(trainer_off, renderer_off)
    (total_on, mse_on, density_on), grads_on = run_pass(trainer_on, renderer_on)

    assert np.isfinite(total_off)
    assert np.isfinite(total_on)
    np.testing.assert_allclose(mse_on, mse_off, rtol=1e-5, atol=1e-7)
    assert density_off == 0.0
    assert density_on > 0.0
    assert total_on >= total_off - 1e-7
    grad_delta = sum(float(np.max(np.abs(grads_on[name] - grads_off[name]))) for name in grads_on)
    assert grad_delta > 0.0


def test_maintenance_rewrite_culls_and_splits_families(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=89)
    scene.opacities[:] = np.array([0.6, 1e-5], dtype=np.float32)
    scene.scales[:] = _log_sigma(np.array([[0.09, 0.06, 0.03], [0.05, 0.05, 0.05]], dtype=np.float32))
    source_position = scene.positions[0].copy()
    frame = _make_frame(tmp_path, image_name="maintenance_rewrite_target.png", image_id=11)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(maintenance_alpha_cull_threshold=1e-3, maintenance_contribution_cull_threshold=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    trainer.maintenance_buffers["clone_counts"].copy_from_numpy(np.array([2, 5], dtype=np.uint32))
    trainer.maintenance_buffers["splat_contribution"].copy_from_numpy(np.array([200, 0], dtype=np.uint32))
    trainer._run_maintenance()

    groups = _read_scene_groups(renderer, trainer.scene.count)
    actual_scales = _actual_scale(groups["scales"][:, :3])
    actual_opacity = _actual_opacity(groups["color_alpha"][:, 3])

    assert trainer.scene.count == 3
    np.testing.assert_allclose(
        actual_scales,
        np.repeat((np.array([[0.09, 0.06, 0.03]], dtype=np.float32) * (3.0 ** (-1.0 / 3.0))), 3, axis=0),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(actual_opacity, np.full((3,), 0.6, dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.mean(groups["positions"][:, :3], axis=0), source_position, rtol=0.0, atol=1e-6)
    offsets = groups["positions"][:, :3] - source_position[None, :]
    np.testing.assert_allclose(np.sum(offsets, axis=0), np.zeros((3,), dtype=np.float32), rtol=0.0, atol=1e-6)
    assert float(np.max(np.linalg.norm(offsets, axis=1))) > 1e-3
    clone_counts_after = buffer_to_numpy(trainer.maintenance_buffers["clone_counts"], np.uint32)[: trainer.scene.count]
    contribution_after = buffer_to_numpy(trainer.maintenance_buffers["splat_contribution"], np.uint32)[: trainer.scene.count]
    assert np.all(clone_counts_after == 0)
    assert np.all(contribution_after == 0)


def test_maintenance_runtime_uses_base_threshold_for_first_maintenance_then_decays(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=93)
    frame = _make_frame(tmp_path, image_name="maintenance_threshold_decay_target.png", image_id=212)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(maintenance_interval=200, maintenance_contribution_cull_threshold=0.001),
        seed=123,
    )
    observed_pixels = renderer.width * renderer.height
    trainer._observed_contribution_pixel_count = observed_pixels

    trainer.state.step = 200
    first_threshold = int(trainer._maintenance_vars()["g_MaintenanceContributionCullThreshold"])
    trainer.state.step = 400
    second_threshold = int(trainer._maintenance_vars()["g_MaintenanceContributionCullThreshold"])

    assert first_threshold == contribution_fixed_count_from_percent(0.001, observed_pixels)
    assert second_threshold == contribution_fixed_count_from_percent(0.00085, observed_pixels)


def test_maintenance_rewrite_culls_low_contribution_splats(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=95)
    scene.opacities[:] = np.array([0.6, 0.7], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="maintenance_contribution_cull_target.png", image_id=111)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(maintenance_alpha_cull_threshold=1e-6, maintenance_contribution_cull_threshold=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    trainer.maintenance_buffers["clone_counts"].copy_from_numpy(np.array([0, 0], dtype=np.uint32))
    trainer.maintenance_buffers["splat_contribution"].copy_from_numpy(np.array([200, 49], dtype=np.uint32))
    trainer._run_maintenance()

    assert trainer.scene.count == 1
    groups = _read_scene_groups(renderer, trainer.scene.count)
    np.testing.assert_allclose(groups["positions"][0, :3], scene.positions[0], rtol=0.0, atol=1e-6)


def test_maintenance_rewrite_migrates_adam_moments(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=97)
    scene.opacities[:] = np.array([0.6, 1e-5], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="maintenance_adam_target.png", image_id=12)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(maintenance_alpha_cull_threshold=1e-3, maintenance_contribution_cull_threshold=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    src_moments = np.zeros((renderer.TRAINABLE_PARAM_COUNT, scene.count, 2), dtype=np.float32)
    src_moments[:, 0, 0] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 1.0
    src_moments[:, 0, 1] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 101.0
    src_moments[:, 1, 0] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 1001.0
    src_moments[:, 1, 1] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 1101.0
    trainer.adam_optimizer.buffers["adam_moments"].copy_from_numpy(src_moments.reshape(-1, 2))

    trainer._observed_contribution_pixel_count = observed_pixels
    trainer.maintenance_buffers["clone_counts"].copy_from_numpy(np.array([2, 5], dtype=np.uint32))
    trainer.maintenance_buffers["splat_contribution"].copy_from_numpy(np.array([200, 0], dtype=np.uint32))
    trainer._run_maintenance()

    expected = np.repeat(src_moments[:, 0:1, :], 3, axis=1)
    np.testing.assert_allclose(_read_adam_moments(trainer, trainer.scene.count), expected, rtol=0.0, atol=1e-7)


def test_maintenance_respects_max_gaussians_cap(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=101)
    scene.opacities[:] = np.array([0.6, 0.7], dtype=np.float32)
    scene.scales[:] = _log_sigma(np.array([[0.09, 0.06, 0.03], [0.04, 0.04, 0.04]], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="maintenance_max_count_target.png", image_id=13)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(max_gaussians=3, maintenance_alpha_cull_threshold=1e-3, maintenance_contribution_cull_threshold=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    trainer.maintenance_buffers["clone_counts"].copy_from_numpy(np.array([4, 4], dtype=np.uint32))
    trainer.maintenance_buffers["splat_contribution"].copy_from_numpy(np.array([200, 200], dtype=np.uint32))
    trainer._run_maintenance()

    assert trainer.scene.count == 3
    groups = _read_scene_groups(renderer, trainer.scene.count)
    actual_scales = _actual_scale(groups["scales"][:, :3])
    split_scale = np.array([0.09, 0.06, 0.03], dtype=np.float32) * (2.0 ** (-1.0 / 3.0))
    split_mask = np.all(np.isclose(actual_scales, split_scale[None, :], rtol=0.0, atol=1e-6), axis=1)
    unsplit_mask = np.all(np.isclose(actual_scales, np.full((1, 3), 0.04, dtype=np.float32), rtol=0.0, atol=1e-6), axis=1)
    assert int(np.count_nonzero(split_mask)) == 2
    assert int(np.count_nonzero(unsplit_mask)) == 1
    split_positions = groups["positions"][split_mask, :3]
    np.testing.assert_allclose(np.mean(split_positions, axis=0), scene.positions[0], rtol=0.0, atol=1e-6)


def test_maintenance_rewrite_sampling_depends_on_frame_hash(device, tmp_path: Path) -> None:
    scene_a = _make_scene(count=1, seed=151)
    scene_b = _make_scene(count=1, seed=151)
    scene_a.opacities[:] = np.array([0.6], dtype=np.float32)
    scene_b.opacities[:] = np.array([0.6], dtype=np.float32)
    scene_a.scales[:] = _log_sigma(np.array([[0.08, 0.05, 0.03]], dtype=np.float32))
    scene_b.scales[:] = scene_a.scales.copy()
    frame_a = _make_frame(tmp_path, image_name="maintenance_frame_hash_a.png", image_id=41)
    frame_b = _make_frame(tmp_path, image_name="maintenance_frame_hash_b.png", image_id=42)
    renderer_a = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    renderer_b = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer_a.width * renderer_a.height
    trainer_a = GaussianTrainer(
        device=device,
        renderer=renderer_a,
        scene=scene_a,
        frames=[frame_a],
        training_hparams=TrainingHyperParams(maintenance_alpha_cull_threshold=1e-6, maintenance_contribution_cull_threshold=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )
    trainer_b = GaussianTrainer(
        device=device,
        renderer=renderer_b,
        scene=scene_b,
        frames=[frame_b],
        training_hparams=TrainingHyperParams(maintenance_alpha_cull_threshold=1e-6, maintenance_contribution_cull_threshold=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    trainer_a._observed_contribution_pixel_count = observed_pixels
    trainer_b._observed_contribution_pixel_count = observed_pixels
    trainer_a.maintenance_buffers["clone_counts"].copy_from_numpy(np.array([1], dtype=np.uint32))
    trainer_b.maintenance_buffers["clone_counts"].copy_from_numpy(np.array([1], dtype=np.uint32))
    trainer_a.maintenance_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer_b.maintenance_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer_a._run_maintenance()
    trainer_b._run_maintenance()

    positions_a = _read_scene_groups(renderer_a, trainer_a.scene.count)["positions"][:, :3]
    positions_b = _read_scene_groups(renderer_b, trainer_b.scene.count)["positions"][:, :3]
    np.testing.assert_allclose(np.mean(positions_a, axis=0), scene_a.positions[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.mean(positions_b, axis=0), scene_b.positions[0], rtol=0.0, atol=1e-6)
    assert not np.allclose(positions_a, positions_b, rtol=0.0, atol=1e-6)


def test_maintenance_min_screen_size_raises_small_splats(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=103)
    scene.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([1e-3, 1e-3, 1e-3], dtype=np.float32))
    near_frame = _make_frame(tmp_path, image_name="maintenance_min_near.png", image_id=21)
    tight_image_path = tmp_path / "maintenance_min_tight.png"
    offscreen_image_path = tmp_path / "maintenance_min_offscreen.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(tight_image_path)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(offscreen_image_path)
    tight_frame = ColmapFrame(
        image_id=22,
        image_path=tight_image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        fx=144.0,
        fy=144.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
    )
    offscreen_frame = ColmapFrame(
        image_id=23,
        image_path=offscreen_image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([10.0, 0.0, 3.0], dtype=np.float32),
        fx=512.0,
        fy=512.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
    )
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[near_frame, tight_frame, offscreen_frame],
        training_hparams=TrainingHyperParams(maintenance_alpha_cull_threshold=1e-3),
        seed=123,
    )

    expected_support = min(
        value
        for value in (
            _circle_bound_support_radius(trainer.make_frame_camera(frame_index, renderer.width, renderer.height), scene.positions[0], renderer.width, renderer.height, _MAINTENANCE_MIN_SCREEN_RADIUS_PX)
            for frame_index in range(3)
        )
        if value is not None
    )

    trainer.maintenance_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_maintenance()

    scales = _actual_scale(_read_scene_groups(renderer, trainer.scene.count)["scales"][0, :3])
    initial_scale = _actual_scale(scene.scales[0, :3])
    assert np.all(scales > initial_scale)
    np.testing.assert_allclose(scales, np.full((3,), scales[0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert float(scales[0]) <= float(expected_support / _GAUSSIAN_SUPPORT_SIGMA_RADIUS) + 1e-6


def test_training_max_screen_size_clamps_large_splats(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=107)
    scene.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([2.0, 2.0, 2.0], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="max_screen_clamp_target.png", image_id=24)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0),
        seed=123,
    )
    camera = trainer.make_frame_camera(0, renderer.width, renderer.height)
    max_radius_px = float(np.sqrt(_TRAINING_MAX_SCREEN_FRACTION * renderer.width * renderer.height / np.pi))
    expected_support = _circle_bound_support_radius(camera, scene.positions[0], renderer.width, renderer.height, max_radius_px)
    assert expected_support is not None

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, 1)["scales"][0, :3])
    np.testing.assert_allclose(scales, np.full((3,), expected_support / _GAUSSIAN_SUPPORT_SIGMA_RADIUS, dtype=np.float32), rtol=0.0, atol=1e-6)


def test_training_max_screen_size_ignores_offscreen_centers(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=109)
    scene.positions[0] = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([2.0, 1.5, 1.0], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="max_screen_offscreen_target.png", image_id=25)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0),
        seed=123,
    )
    camera = trainer.make_frame_camera(0, renderer.width, renderer.height)

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, 1)["scales"][0, :3])
    np.testing.assert_allclose(scales, np.array([2.0, 1.5, 1.0], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_box_downscale_matches_expected_mean(device, tmp_path: Path) -> None:
    image_path = tmp_path / "downscale_target.png"
    image = np.array(
        [
            [[0, 0, 0], [255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255], [255, 255, 0]],
            [[255, 0, 255], [0, 255, 255], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(image, mode="RGB").save(image_path)
    frame = ColmapFrame(
        image_id=0,
        image_path=image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        fx=72.0,
        fy=72.0,
        cx=1.5,
        cy=1.5,
        width=3,
        height=3,
    )
    scene = _make_scene(count=4, seed=123)
    renderer = GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(train_downscale_factor=2),
        seed=5,
    )

    target = trainer.get_frame_target_texture(0, native_resolution=False)
    target_np = np.asarray(target.to_numpy(), dtype=np.float32)
    srgb = image.astype(np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4))
    expected = np.array(
        [
            [np.mean(linear[0:2, 0:2], axis=(0, 1)), np.mean(linear[0:2, 2:3], axis=(0, 1))],
            [np.mean(linear[2:3, 0:2], axis=(0, 1)), np.mean(linear[2:3, 2:3], axis=(0, 1))],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(target_np[:, :, :3], expected, rtol=0.0, atol=5e-5)
    np.testing.assert_allclose(target_np[:, :, 3], np.ones((2, 2), dtype=np.float32), rtol=0.0, atol=1e-6)


def test_separable_gaussian_blur_preserves_impulse_energy_for_n_channels(device):
    width = height = 17
    channel_count = 6
    blur = SeparableGaussianBlur(device, width=width, height=height)
    input_buffer = blur.make_buffer(channel_count)
    output_buffer = blur.make_buffer(channel_count)
    image = np.zeros((height, width, channel_count), dtype=np.float32)
    center = width // 2
    image[center, center, 0] = 1.0
    image[center, center, 5] = 0.5
    input_buffer.copy_from_numpy(np.ascontiguousarray(image.reshape(-1), dtype=np.float32))

    enc = device.create_command_encoder()
    blur.blur(enc, input_buffer, output_buffer, channel_count)
    device.submit_command_buffer(enc.finish())
    device.wait()

    out = np.frombuffer(output_buffer.to_numpy().tobytes(), dtype=np.float32).reshape(height, width, channel_count)
    expected_weights = np.array(
        [
            0.00102838008447911,
            0.00759875813523919,
            0.03600077212843083,
            0.10936068950970002,
            0.21300553771125369,
            0.26601172486179436,
            0.21300553771125369,
            0.10936068950970002,
            0.03600077212843083,
            0.00759875813523919,
            0.00102838008447911,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out[:, :, 0].sum(), 1.0, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(out[:, :, 5].sum(), 0.5, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(out[:, :, 3], 0.0, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(out[center, center - 5 : center + 6, 0], expected_weights * expected_weights[5], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(out[center, center - 5 : center + 6, 5], 0.5 * expected_weights * expected_weights[5], rtol=0.0, atol=1e-6)


def test_fused_adam_handles_nan_grads(device, tmp_path: Path):
    scene = _make_scene(count=16)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=9)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    nan_grads = np.full((scene.count, 4), np.nan, dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=nan_grads, grad_scales=nan_grads, grad_rotations=nan_grads, grad_color_alpha=nan_grads)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scene_groups = _read_scene_groups(renderer, scene.count)
    for name in ("positions", "scales", "rotations", "color_alpha"):
        assert np.all(np.isfinite(scene_groups[name]))


def test_rotation_grad_updates_quaternion(device, tmp_path: Path):
    scene = _make_scene(count=16)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=11)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    grad_rot = np.zeros((scene.count, 4), dtype=np.float32)
    grad_rot[:, 0] = 0.5
    grad_rot[:, 1] = -0.25
    grad_rot[:, 2] = 0.125
    grad_rot[:, 3] = -0.375
    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=grad_rot, grad_color_alpha=zeros)

    before = _read_scene_groups(renderer, scene.count)["rotations"].copy()
    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()
    after = _read_scene_groups(renderer, scene.count)["rotations"]

    assert np.any(np.abs(after - before) > 0.0)
    norms = np.linalg.norm(after, axis=1)
    assert np.all(np.isfinite(norms))
    assert np.all(np.abs(norms - 1.0) < 1e-3)


def test_synthetic_base_grads_update_and_respect_constraints(device, tmp_path: Path):
    scene = _make_scene(count=16)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=21)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    base_grad = np.full((scene.count, 4), 0.25, dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=base_grad, grad_scales=base_grad, grad_rotations=base_grad, grad_color_alpha=base_grad)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scene_groups = _read_scene_groups(renderer, scene.count)
    positions = scene_groups["positions"]
    scales = scene_groups["scales"]
    rotations = scene_groups["rotations"]
    color_alpha = scene_groups["color_alpha"]

    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(scales))
    assert np.all(np.isfinite(rotations))
    assert np.all(np.isfinite(color_alpha))
    assert np.all(np.abs(positions[:, :3]) <= trainer.stability.position_abs_max + 1e-5)
    actual_scales = _actual_scale(scales[:, :3])
    assert np.all(actual_scales >= trainer.stability.min_scale - 1e-6)
    assert np.all(actual_scales <= trainer.stability.max_scale + 1e-6)
    assert np.all(color_alpha[:, :3] >= -1e-6)
    assert np.all(color_alpha[:, :3] <= 1.0 + 1e-6)
    assert np.all(_actual_opacity(color_alpha[:, 3]) >= 0.0)
    assert np.all(_actual_opacity(color_alpha[:, 3]) <= 1.0)


def test_adam_step_clamps_anisotropy(device, tmp_path: Path):
    scene = _make_scene(count=1, seed=25)
    scene.scales[0] = _log_sigma(np.array([0.9, 0.05, 0.05], dtype=np.float32))
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        stability_hparams=StabilityHyperParams(max_anisotropy=10.0),
        training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0),
        seed=27,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _read_scene_groups(renderer, 1)["scales"]
    np.testing.assert_allclose(_actual_scale(scales[0, :3]), np.array([0.9, 0.09, 0.09], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_log_scale_regularizer_pulls_scales_toward_reference(device, tmp_path: Path):
    scene = _make_scene(count=2, seed=27)
    scene.scales[:] = _log_sigma(np.array([[0.02, 0.02, 0.02], [0.08, 0.08, 0.08]], dtype=np.float32))
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(scale_l2_weight=1.0), scale_reg_reference=0.04, seed=29)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales_after = _read_scene_groups(renderer, scene.count)["scales"]
    assert scales_after[0, 0] > scene.scales[0, 0]
    assert scales_after[1, 0] < scene.scales[1, 0]


def test_opacity_regularizer_pushes_true_opacity_down(device, tmp_path: Path):
    scene = _make_scene(count=2, seed=31)
    scene.opacities[:] = np.array([0.2, 0.8], dtype=np.float32)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(scale_l2_weight=0.0, opacity_reg_weight=1.0), seed=41)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    before = _actual_opacity(_read_scene_groups(renderer, 2)["color_alpha"][:, 3])
    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()
    after = _actual_opacity(_read_scene_groups(renderer, 2)["color_alpha"][:, 3])

    assert np.all(after < before)


def test_sh1_regularizer_pushes_sh1_coeffs_toward_zero(device, tmp_path: Path):
    scene = _make_scene(count=2, seed=37)
    scene.sh_coeffs = np.zeros((scene.count, 4, 3), dtype=np.float32)
    scene.sh_coeffs[:, 1:, :] = np.array(
        [
            [[0.6, -0.3, 0.2], [0.1, -0.5, 0.4], [-0.2, 0.25, -0.35]],
            [[-0.4, 0.2, -0.1], [0.3, -0.2, 0.5], [0.15, -0.45, 0.25]],
        ],
        dtype=np.float32,
    )
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0, sh1_reg_weight=1.0), seed=43)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    zero_sh = np.zeros((scene.count, 4, 3), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_sh_coeffs=zero_sh, grad_color_alpha=zeros)

    before = np.linalg.norm(_read_scene_groups(renderer, scene.count)["sh_coeffs"][:, 1:, :].reshape(scene.count, -1), axis=1)
    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()
    after = np.linalg.norm(_read_scene_groups(renderer, scene.count)["sh_coeffs"][:, 1:, :].reshape(scene.count, -1), axis=1)

    assert np.all(after < before)


def test_cpu_pointcloud_initializer_rebuilds_scene_with_nn_scales(device, tmp_path: Path):
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    point_pos = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [-1.0, -1.0, 2.0]], dtype=np.float32)
    point_col = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=None, scene_count=16, upload_initial_scene=False, frames=[frame], init_point_positions=point_pos, init_point_colors=point_col, seed=101)

    init_params = GaussianInitHyperParams(position_jitter_std=0.0, base_scale=0.02, scale_jitter_ratio=0.0, initial_opacity=0.5, color_jitter_std=0.0)
    trainer.initialize_scene_from_pointcloud(splat_count=16, init_hparams=init_params, seed=123)

    scene_groups = _read_scene_groups(renderer, 4)
    positions = scene_groups["positions"]
    scales = scene_groups["scales"]
    rotations = scene_groups["rotations"]
    color_alpha = scene_groups["color_alpha"]
    expected_scales = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)]], dtype=np.float32)

    assert np.all(np.isfinite(positions))
    np.testing.assert_allclose(_actual_scale(scales[:, :3]), expected_scales, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(_actual_opacity(color_alpha[:, 3]), np.full((4,), 0.5, dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.all(np.abs(np.linalg.norm(rotations, axis=1) - 1.0) < 1e-3)
    for name in _ADAM_BUFFER_NAMES:
        np.testing.assert_allclose(
            np.frombuffer(trainer.adam_optimizer.buffers[name].to_numpy().tobytes(), dtype=np.float32)[: 8 * renderer.TRAINABLE_PARAM_COUNT],
            np.zeros((8 * renderer.TRAINABLE_PARAM_COUNT,), dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )


def test_training_frame_order_covers_each_view_once_per_epoch(device, tmp_path: Path):
    scene = _make_scene(count=12, seed=71)
    frames = [_make_frame(tmp_path, image_name=f"target_{idx}.png", image_id=idx, green_value=64 + idx) for idx in range(4)]
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=frames, seed=91)

    seen = []
    for _ in range(8):
        trainer.step()
        seen.append(int(trainer.state.last_frame_index))

    assert sorted(seen[:4]) == [0, 1, 2, 3]
    assert sorted(seen[4:]) == [0, 1, 2, 3]


def test_training_prepass_capacity_sync_runs_every_32_steps(device, tmp_path: Path, monkeypatch) -> None:
    frame = _make_frame(tmp_path, width=32, height=32, image_name="capacity_sync_target.png")
    trainer = object.__new__(GaussianTrainer)
    calls: list[object] = []
    trainer.renderer = SimpleNamespace(sync_prepass_capacity_for_current_scene=lambda sync_camera: calls.append(sync_camera) or False)
    trainer._PREPASS_CAPACITY_SYNC_INTERVAL = 32
    camera = frame.make_camera(near=0.1, far=20.0)

    for step in range(65):
        trainer._maybe_sync_prepass_capacity(camera, step)

    assert len(calls) == 3


def test_maintenance_camera_buffer_refreshes_when_frame_pose_changes(device, tmp_path: Path) -> None:
    scene = _make_scene(count=4, seed=131)
    frame = _make_frame(tmp_path, image_name="maintenance_pose_refresh.png", image_id=31)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    trainer._refresh_maintenance_camera_buffer()
    before = buffer_to_numpy(trainer.maintenance_buffers["camera_rows"], np.float32).copy()
    frame.t_xyz = np.array([0.25, -0.5, 3.0], dtype=np.float32)

    trainer._refresh_maintenance_camera_buffer()
    after = buffer_to_numpy(trainer.maintenance_buffers["camera_rows"], np.float32).copy()

    assert not np.array_equal(before, after)


def test_frame_target_rgba8_resizes_to_frame_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "frame_target_resize.png"
    Image.fromarray(np.full((6, 12, 3), 127, dtype=np.uint8), mode="RGB").save(image_path)
    frame = ColmapFrame(
        image_id=1,
        image_path=image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        fx=40.0,
        fy=20.0,
        cx=2.0,
        cy=1.0,
        width=4,
        height=2,
    )
    trainer = object.__new__(GaussianTrainer)

    rgba8 = trainer._frame_target_rgba8(frame)

    assert rgba8.shape == (2, 4, 4)


def test_create_dataset_textures_threads_cpu_image_loading(monkeypatch) -> None:
    trainer = object.__new__(GaussianTrainer)
    trainer.frames = [SimpleNamespace(image_path=Path("a.png")), SimpleNamespace(image_path=Path("b.png"))]
    calls: list[object] = []

    class _Executor:
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            calls.append(("workers", max_workers, thread_name_prefix))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def map(self, fn, items):
            return map(fn, items)

    monkeypatch.setattr(gaussian_trainer_module, "ThreadPoolExecutor", _Executor)
    monkeypatch.setattr(gaussian_trainer_module, "load_training_frame_rgba8", lambda frame: f"rgba:{frame.image_path.name}")
    trainer._create_gpu_texture = lambda rgba8: f"tex:{rgba8}"

    trainer._create_dataset_textures()

    assert calls == [("workers", 8, "trainer-target")]
    assert trainer._frame_targets_native == ["tex:rgba:a.png", "tex:rgba:b.png"]

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from src.common import buffer_to_numpy
from src.filter import SeparableGaussianBlur
from src.renderer import GaussianRenderer
from src.scene import ColmapFrame, GaussianInitHyperParams, GaussianScene
from src.training import AdamHyperParams, GaussianTrainer, StabilityHyperParams, TrainingHyperParams, resolve_training_resolution

_ADAM_BUFFER_NAMES = ("adam_moments",)
_OPACITY_EPS = 1e-6
_raw_opacity = lambda alpha: (np.log(np.clip(np.asarray(alpha, dtype=np.float32), _OPACITY_EPS, 1.0 - _OPACITY_EPS)) - np.log1p(-np.clip(np.asarray(alpha, dtype=np.float32), _OPACITY_EPS, 1.0 - _OPACITY_EPS))).astype(np.float32, copy=False)
_actual_opacity = lambda raw: (1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float32)))).astype(np.float32, copy=False)
_SCALE_GRAD_MULS = (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 0.9375, 0.96875, 1.0, 1.05)
_TARGET_MULS = (2.0, 3.0, 4.0, 6.0, 7.5)


def _make_scene(count: int = 24, seed: int = 7) -> GaussianScene:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32)
    positions[:, 2] += 2.0
    scales = np.full((count, 3), 0.03, dtype=np.float32)
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
    grad_color_alpha: np.ndarray | None = None,
) -> None:
    renderer.write_grad_groups(
        count,
        grad_positions=grad_positions,
        grad_scales=grad_scales,
        grad_rotations=grad_rotations,
        grad_color_alpha=grad_color_alpha,
    )


def _read_output_grads(renderer: GaussianRenderer) -> np.ndarray:
    flat = buffer_to_numpy(renderer.output_grad_buffer, np.float32)
    return flat[: max(renderer.width * renderer.height, 1) * 4].reshape(renderer.height, renderer.width, 4)


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
            scales=np.full((1, 3), self.pixel_floor_scale, dtype=np.float32),
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
                scales=np.full((1, 3), self.pixel_floor_scale, dtype=np.float32),
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
        self.target_scene.scales[...] = float(target_scale)
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
        self.scene_groups["scales"][0, :3] = float(scale)
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

    loss, mse = trainer._read_loss_metrics()
    grads_after_forward = _read_output_grads(renderer).copy()

    enc = device.create_command_encoder()
    trainer._dispatch_loss_backward(enc, target_texture)
    device.submit_command_buffer(enc.finish())
    device.wait()

    grads_after_backward = _read_output_grads(renderer)
    np.testing.assert_allclose(trainer._read_loss_metrics(), (loss, mse), rtol=0.0, atol=0.0)
    assert np.isfinite(loss)
    assert np.isfinite(mse)
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
    max_growth_eps = 3e-6
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
    assert np.all(scales[:, :3] >= trainer.stability.min_scale - 1e-6)
    assert np.all(scales[:, :3] <= trainer.stability.max_scale + 1e-6)
    assert np.all(color_alpha[:, :3] >= -1e-6)
    assert np.all(color_alpha[:, :3] <= 1.0 + 1e-6)
    assert np.all(_actual_opacity(color_alpha[:, 3]) >= 0.0)
    assert np.all(_actual_opacity(color_alpha[:, 3]) <= 1.0)


def test_adam_step_clamps_anisotropy(device, tmp_path: Path):
    scene = _make_scene(count=1, seed=25)
    scene.scales[0] = np.array([0.9, 0.05, 0.05], dtype=np.float32)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0), seed=27)
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
    np.testing.assert_allclose(scales[0, :3], np.array([0.9, 0.09, 0.09], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_log_scale_regularizer_pulls_scales_toward_reference(device, tmp_path: Path):
    scene = _make_scene(count=2, seed=27)
    scene.scales[:] = np.array([[0.02, 0.02, 0.02], [0.08, 0.08, 0.08]], dtype=np.float32)
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
    np.testing.assert_allclose(scales[:, :3], expected_scales, rtol=0.0, atol=1e-6)
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

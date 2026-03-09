from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from src.filter import SeparableGaussianBlur
from src.renderer import GaussianRenderer
from src.scene import ColmapFrame, GaussianInitHyperParams, GaussianScene
from src.training import AdamHyperParams, GaussianTrainer, StabilityHyperParams, TrainingHyperParams

_ADAM_BUFFER_NAMES = ("adam_m_pos", "adam_v_pos", "adam_m_scale", "adam_v_scale", "adam_m_quat", "adam_v_quat", "adam_m_color_alpha", "adam_v_color_alpha")
_OPACITY_EPS = 1e-6
_raw_opacity = lambda alpha: (np.log(np.clip(np.asarray(alpha, dtype=np.float32), _OPACITY_EPS, 1.0 - _OPACITY_EPS)) - np.log1p(-np.clip(np.asarray(alpha, dtype=np.float32), _OPACITY_EPS, 1.0 - _OPACITY_EPS))).astype(np.float32, copy=False)
_actual_opacity = lambda raw: (1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float32)))).astype(np.float32, copy=False)


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


def _read_f32x4(buffer, count: int) -> np.ndarray:
    values = np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.float32)
    return values[: count * 4].reshape(count, 4).copy()


def _save_rgb(path: Path, image: np.ndarray) -> None:
    rgb = np.clip(np.asarray(image, dtype=np.float32)[..., :3], 0.0, 1.0)
    Image.fromarray((255.0 * rgb + 0.5).astype(np.uint8), mode="RGB").save(path)


def _fit_rendered_gaussian_radius(image: np.ndarray) -> float:
    rgb = np.asarray(image, dtype=np.float32)[..., :3]
    weights = np.mean(np.clip(rgb, 0.0, None), axis=2, dtype=np.float32)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if weight_sum <= 1e-8:
        return 0.0
    h, w = weights.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = float(np.sum(weights * xx, dtype=np.float64) / weight_sum)
    cy = float(np.sum(weights * yy, dtype=np.float64) / weight_sum)
    dx = xx - cx
    dy = yy - cy
    mean_r2 = float(np.sum(weights * (dx * dx + dy * dy), dtype=np.float64) / weight_sum)
    sigma = np.sqrt(max(0.5 * mean_r2, 0.0))
    return float(3.0 * sigma)


def test_training_step_smoke_updates_params_without_changing_count(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    before = _read_f32x4(renderer.scene_buffers["positions"], scene.count).copy()
    loss = trainer.step()
    after = _read_f32x4(renderer.scene_buffers["positions"], scene.count)

    assert np.isfinite(loss)
    assert trainer.state.step == 1
    assert np.isfinite(trainer.state.last_mse)
    assert np.isfinite(trainer.state.avg_loss)
    assert trainer.scene.count == scene.count
    assert np.all(np.isfinite(after))
    assert np.any(np.abs(after - before) > 0.0)


def test_tiny_splat_optimizer_recovers_large_target_scale(device, tmp_path: Path):
    frame = _make_frame(tmp_path, image_name="tiny_target.png")
    target_camera = frame.make_camera(near=0.1, far=20.0)
    target_renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    pixel_floor_scale = target_camera.pixel_world_size_max(3.0, target_renderer.width, target_renderer.height)
    target_scale = 7.5 * pixel_floor_scale
    initial_scale = 0.375 * pixel_floor_scale
    target_scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((1, 3), target_scale, dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.75], dtype=np.float32),
        colors=np.array([[0.8, 0.6, 0.2]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    target_image = target_renderer.render(target_scene, target_camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32)).image
    _save_rgb(frame.image_path, target_image)
    target_radius = _fit_rendered_gaussian_radius(target_image)

    train_scene = GaussianScene(
        positions=target_scene.positions.copy(),
        scales=np.full((1, 3), initial_scale, dtype=np.float32),
        rotations=target_scene.rotations.copy(),
        opacities=target_scene.opacities.copy(),
        colors=target_scene.colors.copy(),
        sh_coeffs=target_scene.sh_coeffs.copy(),
    )
    trainer_renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=trainer_renderer,
        scene=train_scene,
        frames=[frame],
        adam_hparams=AdamHyperParams(position_lr=0.0, scale_lr=0.1, rotation_lr=0.0, color_lr=0.0, opacity_lr=0.0),
        stability_hparams=StabilityHyperParams(max_update=0.5, min_scale=1e-5, max_scale=target_scale * 2.0),
        training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0),
        seed=123,
    )

    initial_image = np.asarray(trainer_renderer.render_to_texture(target_camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))[0].to_numpy(), dtype=np.float32)
    initial_radius = _fit_rendered_gaussian_radius(initial_image)
    losses = []
    radius_history = [initial_radius]
    for _ in range(256):
        losses.append(trainer.step())
        image = np.asarray(trainer_renderer.render_to_texture(target_camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))[0].to_numpy(), dtype=np.float32)
        radius_history.append(_fit_rendered_gaussian_radius(image))
    losses = np.asarray(losses, dtype=np.float32)
    radius_history = np.asarray(radius_history, dtype=np.float32)
    scales = _read_f32x4(trainer_renderer.scene_buffers["scales"], 1)[0, :3]
    final_radius = float(radius_history[-1])
    best_radius = float(np.max(radius_history))
    target_gap = max(target_radius - initial_radius, 1e-6)

    assert np.all(np.isfinite(losses))
    assert np.all(np.isfinite(radius_history))
    assert target_radius > initial_radius
    assert final_radius > initial_radius
    assert best_radius > initial_radius + 0.2 * target_gap
    assert float(np.mean(scales)) > initial_scale


def test_training_targets_use_srgb_textures(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    native_target = trainer.get_frame_target_texture(0, native_resolution=True)
    train_target = trainer.get_frame_target_texture(0, native_resolution=False)

    assert native_target.format == spy.Format.rgba8_unorm_srgb
    assert train_target.format == spy.Format.rgba8_unorm_srgb
    assert native_target is train_target


def test_separable_gaussian_blur_preserves_impulse_energy(device):
    width = height = 17
    blur = SeparableGaussianBlur(device, width=width, height=height)
    input_tex = blur.make_texture()
    output_tex = blur.make_texture()
    image = np.zeros((height, width, 4), dtype=np.float32)
    center = width // 2
    image[center, center, 0] = 1.0
    input_tex.copy_from_numpy(image)

    enc = device.create_command_encoder()
    blur.blur(enc, input_tex, output_tex)
    device.submit_command_buffer(enc.finish())
    device.wait()

    out = np.asarray(output_tex.to_numpy(), dtype=np.float32)
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
    np.testing.assert_allclose(out[center, center - 5 : center + 6, 0], expected_weights * expected_weights[5], rtol=0.0, atol=1e-6)


def test_fused_adam_handles_nan_grads(device, tmp_path: Path):
    scene = _make_scene(count=16)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=9)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    nan_grads = np.full((scene.count * 4,), np.nan, dtype=np.float32)
    renderer.work_buffers["grad_positions"].copy_from_numpy(nan_grads)
    renderer.work_buffers["grad_scales"].copy_from_numpy(nan_grads)
    renderer.work_buffers["grad_rotations"].copy_from_numpy(nan_grads)
    renderer.work_buffers["grad_color_alpha"].copy_from_numpy(nan_grads)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    for name in ("positions", "scales", "rotations", "color_alpha"):
        values = np.frombuffer(renderer.scene_buffers[name].to_numpy().tobytes(), dtype=np.float32)
        assert np.all(np.isfinite(values))


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
    zeros = np.zeros((scene.count * 4,), dtype=np.float32)
    renderer.work_buffers["grad_positions"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_scales"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_rotations"].copy_from_numpy(grad_rot.reshape(-1))
    renderer.work_buffers["grad_color_alpha"].copy_from_numpy(zeros)

    before = _read_f32x4(renderer.scene_buffers["rotations"], scene.count).copy()
    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()
    after = _read_f32x4(renderer.scene_buffers["rotations"], scene.count)

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

    base_grad = np.full((scene.count * 4,), 0.25, dtype=np.float32)
    renderer.work_buffers["grad_positions"].copy_from_numpy(base_grad)
    renderer.work_buffers["grad_scales"].copy_from_numpy(base_grad)
    renderer.work_buffers["grad_rotations"].copy_from_numpy(base_grad)
    renderer.work_buffers["grad_color_alpha"].copy_from_numpy(base_grad)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    positions = _read_f32x4(renderer.scene_buffers["positions"], scene.count)
    scales = _read_f32x4(renderer.scene_buffers["scales"], scene.count)
    rotations = _read_f32x4(renderer.scene_buffers["rotations"], scene.count)
    color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], scene.count)

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

    zeros = np.zeros((scene.count * 4,), dtype=np.float32)
    renderer.work_buffers["grad_positions"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_scales"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_rotations"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_color_alpha"].copy_from_numpy(zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _read_f32x4(renderer.scene_buffers["scales"], 1)
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

    zeros = np.zeros((scene.count * 4,), dtype=np.float32)
    renderer.work_buffers["grad_positions"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_scales"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_rotations"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_color_alpha"].copy_from_numpy(zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales_after = _read_f32x4(renderer.scene_buffers["scales"], scene.count)
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

    zeros = np.zeros((scene.count * 4,), dtype=np.float32)
    renderer.work_buffers["grad_positions"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_scales"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_rotations"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_color_alpha"].copy_from_numpy(zeros)

    before = _actual_opacity(_read_f32x4(renderer.scene_buffers["color_alpha"], 2)[:, 3])
    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()
    after = _actual_opacity(_read_f32x4(renderer.scene_buffers["color_alpha"], 2)[:, 3])

    assert np.all(after < before)


def test_cpu_pointcloud_initializer_rebuilds_scene_with_nn_scales(device, tmp_path: Path):
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    point_pos = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [-1.0, -1.0, 2.0]], dtype=np.float32)
    point_col = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=None, scene_count=16, upload_initial_scene=False, frames=[frame], init_point_positions=point_pos, init_point_colors=point_col, seed=101)

    init_params = GaussianInitHyperParams(position_jitter_std=0.0, base_scale=0.02, scale_jitter_ratio=0.0, initial_opacity=0.5, color_jitter_std=0.0)
    trainer.initialize_scene_from_pointcloud(splat_count=16, init_hparams=init_params, seed=123)

    positions = _read_f32x4(renderer.scene_buffers["positions"], 4)
    scales = _read_f32x4(renderer.scene_buffers["scales"], 4)
    rotations = _read_f32x4(renderer.scene_buffers["rotations"], 4)
    color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], 4)
    expected_scales = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)]], dtype=np.float32)

    assert np.all(np.isfinite(positions))
    np.testing.assert_allclose(scales[:, :3], expected_scales, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(_actual_opacity(color_alpha[:, 3]), np.full((4,), 0.5, dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.all(np.abs(np.linalg.norm(rotations, axis=1) - 1.0) < 1e-3)
    for name in _ADAM_BUFFER_NAMES:
        np.testing.assert_allclose(_read_f32x4(trainer._buffers[name], 4), np.zeros((4, 4), dtype=np.float32), rtol=0.0, atol=1e-7)


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

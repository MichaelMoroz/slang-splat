from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from src.filter import SeparableGaussianBlur
from src.renderer import GaussianRenderer
from src.scene import ColmapFrame, GaussianInitHyperParams, GaussianScene
from src.training import GaussianTrainer, StabilityHyperParams, TrainingHyperParams

_ADAM_BUFFER_NAMES = ("adam_m_pos", "adam_v_pos", "adam_m_scale", "adam_v_scale", "adam_m_quat", "adam_v_quat", "adam_m_color_alpha", "adam_v_color_alpha")


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
    return GaussianScene(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        sh_coeffs=sh_coeffs,
    )


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


def _read_u32(buffer, count: int) -> np.ndarray:
    values = np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.uint32)
    return values[:count].copy()


def _read_f32(buffer, count: int) -> np.ndarray:
    values = np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.float32)
    return values[:count].copy()


def test_training_step_smoke_updates_params(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    loss = trainer.step()
    after = np.concatenate(
        [
            np.frombuffer(renderer.scene_buffers["positions"].to_numpy().tobytes(), dtype=np.float32),
            np.frombuffer(renderer.scene_buffers["scales"].to_numpy().tobytes(), dtype=np.float32),
            np.frombuffer(renderer.scene_buffers["rotations"].to_numpy().tobytes(), dtype=np.float32),
            np.frombuffer(renderer.scene_buffers["color_alpha"].to_numpy().tobytes(), dtype=np.float32),
        ]
    ).copy()

    assert np.isfinite(loss)
    assert trainer.state.step == 1
    assert np.isfinite(trainer.state.last_mse)
    assert np.isfinite(trainer.state.last_psnr)
    assert np.isfinite(trainer.state.avg_psnr)
    assert np.isclose(trainer.state.avg_psnr, trainer.state.last_psnr)
    assert np.all(np.isfinite(after))


def test_training_targets_use_srgb_textures(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    native_target = trainer.get_frame_target_texture(0, native_resolution=True)
    train_target = trainer.get_frame_target_texture(0, native_resolution=False)

    assert native_target.format == spy.Format.rgba8_unorm_srgb
    assert train_target.format == spy.Format.rgba8_unorm_srgb


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
    grad_flat = grad_rot.reshape(-1)
    zeros = np.zeros((scene.count * 4,), dtype=np.float32)
    renderer.work_buffers["grad_positions"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_scales"].copy_from_numpy(zeros)
    renderer.work_buffers["grad_rotations"].copy_from_numpy(grad_flat)
    renderer.work_buffers["grad_color_alpha"].copy_from_numpy(zeros)

    before = np.frombuffer(renderer.scene_buffers["rotations"].to_numpy().tobytes(), dtype=np.float32).reshape(-1, 4).copy()
    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()
    after = np.frombuffer(renderer.scene_buffers["rotations"].to_numpy().tobytes(), dtype=np.float32).reshape(-1, 4).copy()

    delta = np.abs(after - before)
    assert np.any(delta > 0.0)
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

    positions = np.frombuffer(renderer.scene_buffers["positions"].to_numpy().tobytes(), dtype=np.float32).reshape(-1, 4)
    scales = np.frombuffer(renderer.scene_buffers["scales"].to_numpy().tobytes(), dtype=np.float32).reshape(-1, 4)
    rotations = np.frombuffer(renderer.scene_buffers["rotations"].to_numpy().tobytes(), dtype=np.float32).reshape(-1, 4)
    color_alpha = np.frombuffer(renderer.scene_buffers["color_alpha"].to_numpy().tobytes(), dtype=np.float32).reshape(-1, 4)

    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(scales))
    assert np.all(np.isfinite(rotations))
    assert np.all(np.isfinite(color_alpha))
    assert np.all(np.abs(positions[:, :3]) <= trainer.stability.position_abs_max + 1e-5)
    assert np.all(scales[:, :3] >= trainer.stability.min_scale - 1e-6)
    assert np.all(scales[:, :3] <= trainer.stability.max_scale + 1e-6)
    assert np.all(color_alpha[:, :3] >= -1e-6)
    assert np.all(color_alpha[:, :3] <= 1.0 + 1e-6)
    assert np.all(color_alpha[:, 3] >= trainer.stability.min_opacity - 1e-6)
    assert np.all(color_alpha[:, 3] <= trainer.stability.max_opacity + 1e-6)
    norms = np.linalg.norm(rotations, axis=1)
    assert np.all(np.isfinite(norms))
    assert np.all(np.abs(norms - 1.0) < 1e-3)


def test_log_scale_regularizer_pulls_scales_toward_reference(device, tmp_path: Path):
    scene = _make_scene(count=2, seed=27)
    scene.scales[:] = np.array([[0.02, 0.02, 0.02], [0.08, 0.08, 0.08]], dtype=np.float32)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    training = TrainingHyperParams(scale_l2_weight=1.0, mcmc_position_noise_enabled=False, low_quality_reinit_enabled=False)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=training,
        scale_reg_reference=0.04,
        seed=29,
    )
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

    scales_after = np.frombuffer(renderer.scene_buffers["scales"].to_numpy().tobytes(), dtype=np.float32).reshape(-1, 4)
    assert scales_after[0, 0] > scene.scales[0, 0]
    assert scales_after[1, 0] < scene.scales[1, 0]


def test_update_densification_stats_tracks_ema_and_screen_radius(device, tmp_path: Path):
    scene = _make_scene(count=3, seed=41)
    frames = [_make_frame(tmp_path, image_name=f"target_stats_{idx}.png", image_id=idx) for idx in range(4)]
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=frames, seed=23)

    trainer._buffers["grad_ema"].copy_from_numpy(np.array([2.0, 1.0, 9.0], dtype=np.float32))
    trainer._buffers["max_screen_radius"].copy_from_numpy(np.array([1.0, 6.0, 4.0], dtype=np.float32))
    grad_positions = np.array(
        [
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    ).reshape(-1)
    screen = np.array(
        [
            [0.0, 0.0, 5.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 7.0, 1.0],
        ],
        dtype=np.float32,
    )
    renderer.work_buffers["grad_positions"].copy_from_numpy(grad_positions)
    renderer.work_buffers["screen_center_radius_depth"].copy_from_numpy(screen)

    enc = device.create_command_encoder()
    trainer._dispatch_update_densification_stats(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    np.testing.assert_allclose(_read_f32(trainer._buffers["grad_ema"], 3), np.array([2.75, 1.0, 7.5], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(_read_f32(trainer._buffers["max_screen_radius"], 3), np.array([5.0, 6.0, 7.0], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_regenerate_scene_clones_small_high_gradient_gaussians(device, tmp_path: Path):
    scene = _make_scene(count=2, seed=51)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    training = TrainingHyperParams(densify_grad_threshold=0.5, percent_dense=0.5, world_size_prune_ratio=10.0)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=training, seed=31)

    positions = _read_f32x4(renderer.scene_buffers["positions"], 2)
    scales = _read_f32x4(renderer.scene_buffers["scales"], 2)
    scales[0, :3] = 0.05
    scales[1, :3] = 0.2
    renderer.scene_buffers["scales"].copy_from_numpy(scales)
    trainer._buffers["grad_ema"].copy_from_numpy(np.array([1.0, 0.1], dtype=np.float32))
    trainer._buffers["max_screen_radius"].copy_from_numpy(np.zeros((2,), dtype=np.float32))
    for index, name in enumerate(_ADAM_BUFFER_NAMES):
        trainer._buffers[name].copy_from_numpy(np.full((2, 4), float(index + 1), dtype=np.float32))
    trainer._ensure_regen_buffers(2)

    enc = device.create_command_encoder()
    trainer._dispatch_regenerate_scene(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    assert trainer._read_output_count() == 3
    out_pos = _read_f32x4(trainer._regen_buffers["positions"], 3)
    out_scale = _read_f32x4(trainer._regen_buffers["scales"], 3)
    np.testing.assert_allclose(out_pos[0], positions[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(out_pos[1], positions[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(out_pos[2], positions[1], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(out_scale[:2], np.repeat(scales[:1], 2, axis=0), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(_read_f32x4(trainer._regen_buffers["adam_m_pos"], 3)[1], np.zeros((4,), dtype=np.float32), rtol=0.0, atol=1e-7)


def test_regenerate_scene_splits_large_high_gradient_gaussians(device, tmp_path: Path):
    scene = _make_scene(count=1, seed=61)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    training = TrainingHyperParams(densify_grad_threshold=0.5, percent_dense=0.01, split_child_count=2, world_size_prune_ratio=1000.0)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=training, seed=55)

    scales = _read_f32x4(renderer.scene_buffers["scales"], 1)
    scales[0, :3] = 0.5
    renderer.scene_buffers["scales"].copy_from_numpy(scales)
    trainer._buffers["grad_ema"].copy_from_numpy(np.array([1.0], dtype=np.float32))
    trainer._buffers["max_screen_radius"].copy_from_numpy(np.zeros((1,), dtype=np.float32))
    trainer._ensure_regen_buffers(1)

    enc = device.create_command_encoder()
    trainer._dispatch_regenerate_scene(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    assert trainer._read_output_count() == 2
    out_pos = _read_f32x4(trainer._regen_buffers["positions"], 2)
    out_scale = _read_f32x4(trainer._regen_buffers["scales"], 2)
    assert np.all(np.isfinite(out_pos))
    np.testing.assert_allclose(out_scale[:, :3], np.full((2, 3), 0.5 / 1.6, dtype=np.float32), rtol=0.0, atol=1e-5)
    assert np.any(np.abs(out_pos[:, :3] - scene.positions[0]) > 1e-5)


def test_regenerate_scene_prunes_low_opacity_screen_and_world_outliers(device, tmp_path: Path):
    scene = _make_scene(count=3, seed=71)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    training = TrainingHyperParams(densify_grad_threshold=100.0, prune_min_opacity=0.2, screen_size_prune_threshold=20.0, world_size_prune_ratio=0.1, opacity_reset_interval=1)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=training, seed=91)
    trainer.state.step = 2

    scales = _read_f32x4(renderer.scene_buffers["scales"], 3)
    color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], 3)
    scales[0, :3] = 0.05
    scales[1, :3] = 0.05
    scales[2, :3] = trainer.training.world_size_prune_ratio * trainer._scene_extent * 2.0
    color_alpha[0, 3] = 0.1
    color_alpha[1, 3] = 0.8
    color_alpha[2, 3] = 0.8
    renderer.scene_buffers["scales"].copy_from_numpy(scales)
    renderer.scene_buffers["color_alpha"].copy_from_numpy(color_alpha)
    trainer._buffers["grad_ema"].copy_from_numpy(np.zeros((3,), dtype=np.float32))
    trainer._buffers["max_screen_radius"].copy_from_numpy(np.array([0.0, 25.0, 0.0], dtype=np.float32))
    trainer._ensure_regen_buffers(3)

    enc = device.create_command_encoder()
    trainer._dispatch_regenerate_scene(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    assert trainer._read_output_count() == 0


def test_reset_opacity_clamps_alpha_and_clears_color_moments(device, tmp_path: Path):
    scene = _make_scene(count=2, seed=81)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=17)

    color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], 2)
    color_alpha[:, 3] = np.array([0.8, 0.005], dtype=np.float32)
    renderer.scene_buffers["color_alpha"].copy_from_numpy(color_alpha)
    trainer._buffers["adam_m_color_alpha"].copy_from_numpy(np.full((2, 4), 5.0, dtype=np.float32))
    trainer._buffers["adam_v_color_alpha"].copy_from_numpy(np.full((2, 4), 7.0, dtype=np.float32))

    enc = device.create_command_encoder()
    trainer._dispatch_reset_opacity(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    np.testing.assert_allclose(_read_f32x4(renderer.scene_buffers["color_alpha"], 2)[:, 3], np.array([0.01, 0.005], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(_read_f32x4(trainer._buffers["adam_m_color_alpha"], 2), np.zeros((2, 4), dtype=np.float32), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(_read_f32x4(trainer._buffers["adam_v_color_alpha"], 2), np.zeros((2, 4), dtype=np.float32), rtol=0.0, atol=1e-7)


def test_cpu_pointcloud_initializer_rebuilds_scene_with_nn_scales(device, tmp_path: Path):
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    point_pos = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [-1.0, -1.0, 2.0],
        ],
        dtype=np.float32,
    )
    point_col = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=None,
        scene_count=16,
        upload_initial_scene=False,
        frames=[frame],
        init_point_positions=point_pos,
        init_point_colors=point_col,
        seed=101,
    )

    init_params = GaussianInitHyperParams(
        position_jitter_std=0.0,
        base_scale=0.02,
        scale_jitter_ratio=0.0,
        initial_opacity=0.5,
        color_jitter_std=0.0,
    )
    trainer.initialize_scene_from_pointcloud(splat_count=16, init_hparams=init_params, seed=123)

    positions = _read_f32x4(renderer.scene_buffers["positions"], 4)
    scales = _read_f32x4(renderer.scene_buffers["scales"], 4)
    rotations = _read_f32x4(renderer.scene_buffers["rotations"], 4)
    color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], 4)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(scales))
    assert np.all(np.isfinite(rotations))
    assert np.all(np.isfinite(color_alpha))
    expected_scales = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)]], dtype=np.float32)
    np.testing.assert_allclose(scales[:, :3], expected_scales, rtol=0.0, atol=1e-6)
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

    first_epoch, second_epoch = seen[:4], seen[4:]
    assert sorted(first_epoch) == [0, 1, 2, 3]
    assert sorted(second_epoch) == [0, 1, 2, 3]

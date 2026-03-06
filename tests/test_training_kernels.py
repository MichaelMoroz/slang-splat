from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.renderer import GaussianRenderer
from src.scene import ColmapFrame, GaussianInitHyperParams, GaussianScene
from src.training import GaussianTrainer, StabilityHyperParams, TrainingHyperParams


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


def _make_frame(tmp_path: Path, width: int = 64, height: int = 64) -> ColmapFrame:
    image_path = tmp_path / "target.png"
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    Image.fromarray(image, mode="RGB").save(image_path)
    return ColmapFrame(
        image_id=0,
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


def _hash_u32(value: int) -> int:
    x = int(value) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def _rng_next(state: int) -> int:
    return (1664525 * int(state) + 1013904223) & 0xFFFFFFFF


def _expected_donor_id(splat_id: int, step_index: int, splat_count: int) -> int:
    seed = ((int(splat_id) * 0x9E3779B9) ^ (int(step_index) * 0x85EBCA6B) ^ 0x27D4EB2D) & 0xFFFFFFFF
    state = _hash_u32(seed)
    if state == 0:
        state = 1
    return int(_rng_next(state) % int(splat_count))


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
    assert np.isfinite(trainer.state.ema_psnr)
    assert np.all(np.isfinite(after))


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


def test_mark_low_quality_flags_from_stability_thresholds(device, tmp_path: Path):
    scene = _make_scene(count=5, seed=33)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    stability = StabilityHyperParams(min_scale=0.2, max_scale=3.0, min_opacity=0.3, max_opacity=0.9999)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], stability_hparams=stability, seed=17)

    scales = _read_f32x4(renderer.scene_buffers["scales"], scene.count)
    color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], scene.count)
    scales[:, :3] = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1],
            [0.4, 0.2, 0.3],
            [0.8, 0.8, 0.8],
            [0.2, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    color_alpha[:, 3] = np.array([0.8, 0.8, 0.4, 0.2, 0.8], dtype=np.float32)
    renderer.scene_buffers["scales"].copy_from_numpy(scales)
    renderer.scene_buffers["color_alpha"].copy_from_numpy(color_alpha)

    enc = device.create_command_encoder()
    trainer._dispatch_mark_low_quality_splats(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    flags = _read_u32(trainer._buffers["low_quality_flags"], scene.count)
    expected = np.array([0, 1, 0, 1, 1], dtype=np.uint32)
    np.testing.assert_array_equal(flags, expected)


def test_resample_low_quality_from_valid_random_donor(device, tmp_path: Path):
    scene = _make_scene(count=8, seed=41)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    stability = StabilityHyperParams(min_scale=0.2, max_scale=3.0, min_opacity=0.3, max_opacity=0.9999)
    training = TrainingHyperParams(low_quality_reinit_enabled=True, mcmc_position_noise_scale=0.0)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        stability_hparams=stability,
        training_hparams=training,
        seed=23,
    )

    positions = _read_f32x4(renderer.scene_buffers["positions"], scene.count)
    scales = _read_f32x4(renderer.scene_buffers["scales"], scene.count)
    rotations = _read_f32x4(renderer.scene_buffers["rotations"], scene.count)
    color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], scene.count)
    for idx in range(scene.count):
        positions[idx, :3] = np.array([float(idx), float(idx) + 0.1, float(idx) + 0.2], dtype=np.float32)
        scales[idx, :3] = np.array([0.6, 0.5, 0.4], dtype=np.float32)
        rotations[idx] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        color_alpha[idx] = np.array([0.1 * idx, 0.05 * idx, 0.02 * idx, 0.8], dtype=np.float32)

    step_index = int(trainer.state.step + 1)
    target_id, donor_id = -1, -1
    for splat_id in range(scene.count):
        candidate = _expected_donor_id(splat_id, step_index, scene.count)
        if candidate != splat_id:
            target_id = splat_id
            donor_id = candidate
            break
    assert target_id >= 0 and donor_id >= 0
    color_alpha[target_id, 3] = 0.05

    renderer.scene_buffers["positions"].copy_from_numpy(positions)
    renderer.scene_buffers["scales"].copy_from_numpy(scales)
    renderer.scene_buffers["rotations"].copy_from_numpy(rotations)
    renderer.scene_buffers["color_alpha"].copy_from_numpy(color_alpha)

    for name in (
        "adam_m_pos",
        "adam_v_pos",
        "adam_m_scale",
        "adam_v_scale",
        "adam_m_quat",
        "adam_v_quat",
        "adam_m_color_alpha",
        "adam_v_color_alpha",
    ):
        moments = np.full((scene.count, 4), 3.25, dtype=np.float32)
        trainer._buffers[name].copy_from_numpy(moments)

    expected_pos = positions[donor_id].copy()
    expected_scale = scales[donor_id].copy()
    expected_rot = rotations[donor_id].copy()
    expected_color_alpha = color_alpha[donor_id].copy()

    enc = device.create_command_encoder()
    trainer._dispatch_mark_low_quality_splats(enc)
    trainer._dispatch_resample_low_quality_splats(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    flags = _read_u32(trainer._buffers["low_quality_flags"], scene.count)
    assert int(flags[target_id]) == 1
    assert int(flags[donor_id]) == 0

    after_pos = _read_f32x4(renderer.scene_buffers["positions"], scene.count)
    after_scale = _read_f32x4(renderer.scene_buffers["scales"], scene.count)
    after_rot = _read_f32x4(renderer.scene_buffers["rotations"], scene.count)
    after_color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], scene.count)
    np.testing.assert_allclose(after_pos[target_id], expected_pos, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(after_scale[target_id], expected_scale, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(after_rot[target_id], expected_rot, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(after_color_alpha[target_id], expected_color_alpha, rtol=0.0, atol=1e-6)

    for name in (
        "adam_m_pos",
        "adam_v_pos",
        "adam_m_scale",
        "adam_v_scale",
        "adam_m_quat",
        "adam_v_quat",
        "adam_m_color_alpha",
        "adam_v_color_alpha",
    ):
        moment_values = _read_f32x4(trainer._buffers[name], scene.count)
        np.testing.assert_allclose(moment_values[target_id], np.zeros((4,), dtype=np.float32), rtol=0.0, atol=1e-7)


def test_resample_skips_when_random_donor_is_low_quality(device, tmp_path: Path):
    scene = _make_scene(count=8, seed=51)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    stability = StabilityHyperParams(min_scale=0.2, max_scale=3.0, min_opacity=0.3, max_opacity=0.9999)
    training = TrainingHyperParams(low_quality_reinit_enabled=True, mcmc_position_noise_scale=0.0)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        stability_hparams=stability,
        training_hparams=training,
        seed=31,
    )

    step_index = int(trainer.state.step + 1)
    target_id, donor_id = -1, -1
    for splat_id in range(scene.count):
        candidate = _expected_donor_id(splat_id, step_index, scene.count)
        if candidate != splat_id:
            target_id = splat_id
            donor_id = candidate
            break
    assert target_id >= 0 and donor_id >= 0

    positions = _read_f32x4(renderer.scene_buffers["positions"], scene.count)
    scales = _read_f32x4(renderer.scene_buffers["scales"], scene.count)
    rotations = _read_f32x4(renderer.scene_buffers["rotations"], scene.count)
    color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], scene.count)
    scales[:, :3] = 0.6
    color_alpha[:, 3] = 0.8
    color_alpha[target_id, 3] = 0.05
    color_alpha[donor_id, 3] = 0.05

    renderer.scene_buffers["positions"].copy_from_numpy(positions)
    renderer.scene_buffers["scales"].copy_from_numpy(scales)
    renderer.scene_buffers["rotations"].copy_from_numpy(rotations)
    renderer.scene_buffers["color_alpha"].copy_from_numpy(color_alpha)

    sentinel = np.full((scene.count, 4), 1.75, dtype=np.float32)
    for name in (
        "adam_m_pos",
        "adam_v_pos",
        "adam_m_scale",
        "adam_v_scale",
        "adam_m_quat",
        "adam_v_quat",
        "adam_m_color_alpha",
        "adam_v_color_alpha",
    ):
        trainer._buffers[name].copy_from_numpy(sentinel)

    before_target_pos = positions[target_id].copy()
    before_target_scale = scales[target_id].copy()
    before_target_rot = rotations[target_id].copy()
    before_target_color_alpha = color_alpha[target_id].copy()

    enc = device.create_command_encoder()
    trainer._dispatch_mark_low_quality_splats(enc)
    trainer._dispatch_resample_low_quality_splats(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    after_pos = _read_f32x4(renderer.scene_buffers["positions"], scene.count)
    after_scale = _read_f32x4(renderer.scene_buffers["scales"], scene.count)
    after_rot = _read_f32x4(renderer.scene_buffers["rotations"], scene.count)
    after_color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], scene.count)
    np.testing.assert_allclose(after_pos[target_id], before_target_pos, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(after_scale[target_id], before_target_scale, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(after_rot[target_id], before_target_rot, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(after_color_alpha[target_id], before_target_color_alpha, rtol=0.0, atol=1e-7)

    for name in (
        "adam_m_pos",
        "adam_v_pos",
        "adam_m_scale",
        "adam_v_scale",
        "adam_m_quat",
        "adam_v_quat",
        "adam_m_color_alpha",
        "adam_v_color_alpha",
    ):
        moment_values = _read_f32x4(trainer._buffers[name], scene.count)
        np.testing.assert_allclose(moment_values[target_id], sentinel[target_id], rtol=0.0, atol=1e-7)


def test_training_step_smoke_with_low_quality_reinit_enabled(device, tmp_path: Path):
    scene = _make_scene(count=24, seed=61)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    stability = StabilityHyperParams(min_scale=0.05, max_scale=3.0, min_opacity=0.6, max_opacity=0.9999)
    training = TrainingHyperParams(low_quality_reinit_enabled=True, mcmc_position_noise_scale=0.5)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        stability_hparams=stability,
        training_hparams=training,
        seed=55,
    )

    loss = trainer.step()
    positions = _read_f32x4(renderer.scene_buffers["positions"], scene.count)
    scales = _read_f32x4(renderer.scene_buffers["scales"], scene.count)
    rotations = _read_f32x4(renderer.scene_buffers["rotations"], scene.count)
    color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], scene.count)

    assert np.isfinite(loss)
    assert trainer.state.step == 1
    assert np.isfinite(trainer.state.last_mse)
    assert np.isfinite(trainer.state.last_psnr)
    assert np.isfinite(trainer.state.ema_psnr)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(scales))
    assert np.all(np.isfinite(rotations))
    assert np.all(np.isfinite(color_alpha))


def test_gpu_pointcloud_initializer_rebuilds_scene_without_cpu_upload(device, tmp_path: Path):
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

    positions = _read_f32x4(renderer.scene_buffers["positions"], 16)
    scales = _read_f32x4(renderer.scene_buffers["scales"], 16)
    rotations = _read_f32x4(renderer.scene_buffers["rotations"], 16)
    color_alpha = _read_f32x4(renderer.scene_buffers["color_alpha"], 16)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(scales))
    assert np.all(np.isfinite(rotations))
    assert np.all(np.isfinite(color_alpha))
    assert np.all(scales[:, :3] >= 0.02 - 1e-6)
    assert np.all(np.abs(np.linalg.norm(rotations, axis=1) - 1.0) < 1e-3)

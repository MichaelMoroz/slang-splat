from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.renderer import GaussianRenderer
from src.scene import ColmapFrame, GaussianScene
from src.training import GaussianTrainer


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

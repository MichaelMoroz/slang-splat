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

    before = np.frombuffer(renderer.scene_buffers["positions"].to_numpy().tobytes(), dtype=np.float32).copy()
    loss = trainer.step()
    after = np.frombuffer(renderer.scene_buffers["positions"].to_numpy().tobytes(), dtype=np.float32).copy()

    assert np.isfinite(loss)
    assert trainer.state.step == 1
    assert np.any(np.abs(after - before) > 0.0)
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
    renderer.work_buffers["grad_splat_pos_local"].copy_from_numpy(nan_grads)
    renderer.work_buffers["grad_splat_inv_scale"].copy_from_numpy(nan_grads)
    renderer.work_buffers["grad_splat_quat"].copy_from_numpy(nan_grads)
    renderer.work_buffers["grad_screen_color_alpha"].copy_from_numpy(nan_grads)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc, camera.position)
    device.submit_command_buffer(enc.finish())
    device.wait()

    for name in ("positions", "scales", "rotations", "color_alpha"):
        values = np.frombuffer(renderer.scene_buffers[name].to_numpy().tobytes(), dtype=np.float32)
        assert np.all(np.isfinite(values))

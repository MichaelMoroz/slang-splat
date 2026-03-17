from __future__ import annotations

from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.scene import ColmapFrame, GaussianScene
from torch_examples.train_colmap_garden_torch import (
    FrameMetricTracker,
    FrameOrderState,
    TorchGardenTrainConfig,
    compute_rgb_loss_metrics,
    frame_to_camera_tensor,
    linear_rgb_to_srgb8,
    next_frame_index,
    pack_torch_splats,
    project_scene_params_,
    rgb8_to_linear_rgb,
    run_training,
    scene_to_torch_params,
)


def _make_scene(count: int = 3) -> GaussianScene:
    positions = np.arange(count * 3, dtype=np.float32).reshape(count, 3) * 0.1
    scales = np.log(np.full((count, 3), 0.05, dtype=np.float32))
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.linspace(0.2, 0.8, count, dtype=np.float32)
    colors = np.linspace(0.1, 0.9, count * 3, dtype=np.float32).reshape(count, 3)
    sh_coeffs = np.zeros((count, 1, 3), dtype=np.float32)
    return GaussianScene(positions=positions, scales=scales, rotations=rotations, opacities=opacities, colors=colors, sh_coeffs=sh_coeffs)


def _make_frame() -> ColmapFrame:
    return ColmapFrame(
        image_id=7,
        image_path=Path("frame.png"),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        fx=100.0,
        fy=110.0,
        cx=50.0,
        cy=55.0,
        width=100,
        height=80,
    )


def test_scene_to_torch_params_and_pack_roundtrip(torch_cuda_or_cpu_device):
    scene = _make_scene(count=2)
    params = scene_to_torch_params(scene, torch_cuda_or_cpu_device)
    packed = pack_torch_splats(params).detach().cpu().numpy()

    assert packed.shape == (2, 14)
    np.testing.assert_allclose(packed[:, 0:3], scene.positions)
    np.testing.assert_allclose(packed[:, 3:6], scene.scales)
    np.testing.assert_allclose(packed[:, 6:10], scene.rotations)
    np.testing.assert_allclose(packed[:, 10:13], scene.colors)
    np.testing.assert_allclose(packed[:, 13], scene.opacities)


def test_frame_to_camera_tensor_uses_colmap_layout(torch_cuda_or_cpu_device):
    frame = _make_frame()
    tensor = frame_to_camera_tensor(frame, torch_cuda_or_cpu_device, near=0.25, far=90.0).detach().cpu().numpy()

    np.testing.assert_allclose(
        tensor,
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 100.0, 110.0, 50.0, 55.0, 0.25, 90.0, 0.0, 0.0], dtype=np.float32),
    )


def test_frame_order_matches_trainer_permutation_behavior() -> None:
    state = FrameOrderState.create(frame_count=4, seed=123)
    rng = np.random.default_rng(123)
    expected0 = rng.permutation(4).astype(np.int32)
    expected1 = rng.permutation(4).astype(np.int32)

    actual0 = np.array([next_frame_index(state) for _ in range(4)], dtype=np.int32)
    actual1 = np.array([next_frame_index(state) for _ in range(4)], dtype=np.int32)

    np.testing.assert_array_equal(actual0, expected0)
    np.testing.assert_array_equal(actual1, expected1)


def test_project_scene_params_normalizes_and_clamps(torch_cuda_or_cpu_device):
    params = {
        "positions": torch.nn.Parameter(torch.zeros((2, 3), device=torch_cuda_or_cpu_device)),
        "log_scales": torch.nn.Parameter(torch.tensor([[100.0, -100.0, 0.0], [0.0, 0.0, 0.0]], device=torch_cuda_or_cpu_device)),
        "rotations": torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], device=torch_cuda_or_cpu_device)),
        "colors": torch.nn.Parameter(torch.tensor([[-1.0, 2.0, 0.5], [0.5, 0.5, 0.5]], device=torch_cuda_or_cpu_device)),
        "alpha": torch.nn.Parameter(torch.tensor([[-1.0], [2.0]], device=torch_cuda_or_cpu_device)),
    }

    project_scene_params_(params)
    rotations = params["rotations"].detach().cpu().numpy()
    colors = params["colors"].detach().cpu().numpy()
    alpha = params["alpha"].detach().cpu().numpy()
    log_scales = params["log_scales"].detach().cpu().numpy()

    np.testing.assert_allclose(rotations[0], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(rotations[1], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.all((colors >= 0.0) & (colors <= 1.0))
    assert np.all((alpha >= 1e-4) & (alpha <= 0.9999))
    assert np.all(log_scales >= np.float32(np.log(1e-4)))
    assert np.all(log_scales <= np.float32(np.log(10.0)))


def test_rgb_conversion_roundtrip_is_reasonable(torch_cuda_or_cpu_device):
    rgb8 = torch.tensor([[[0, 128, 255]]], dtype=torch.uint8, device=torch_cuda_or_cpu_device)
    linear = rgb8_to_linear_rgb(rgb8)
    roundtrip = linear_rgb_to_srgb8(linear)

    assert linear.dtype == torch.float32
    assert roundtrip.dtype == np.uint8
    np.testing.assert_allclose(roundtrip, np.array([[[0, 128, 255]]], dtype=np.uint8), atol=1)


def test_compute_rgb_loss_metrics_handles_zero_mse(torch_cuda_or_cpu_device):
    image = torch.ones((2, 2, 3), dtype=torch.float32, device=torch_cuda_or_cpu_device) * 0.5
    loss, mse, psnr = compute_rgb_loss_metrics(image, image)

    assert np.isclose(float(loss.detach().cpu().item()), 0.0)
    assert np.isclose(mse, 0.0)
    assert np.isinf(psnr)


def test_frame_metric_tracker_replaces_existing_frame_value() -> None:
    tracker = FrameMetricTracker.create(2)
    tracker.update(0, loss=1.0, mse=0.1, psnr=10.0)
    tracker.update(1, loss=5.0, mse=0.5, psnr=30.0)
    tracker.update(0, loss=9.0, mse=0.9, psnr=50.0)

    assert np.isclose(tracker.mean("loss"), 7.0)
    assert np.isclose(tracker.mean("psnr"), 40.0)


def test_example_script_bootstraps_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "torch_examples" / "train_colmap_garden_torch.py"
    sanitized_path = [entry for entry in sys.path if Path(entry or ".").resolve() != repo_root]

    monkeypatch.setattr(sys, "path", sanitized_path)
    module_globals = runpy.run_path(str(script_path))

    assert "run_training" in module_globals


@pytest.fixture(scope="module")
def torch_cuda_or_cpu_device():
    return torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")


def test_torch_example_dataset_smoke(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch is required for torch example smoke test.")
    dataset_root = Path("dataset/garden").resolve()
    if not dataset_root.exists():
        pytest.skip("dataset/garden is unavailable.")

    config = TorchGardenTrainConfig(
        colmap_root=dataset_root,
        images_subdir="images_4",
        iters=1,
        max_gaussians=128,
        seed=3,
        save_every=0,
        output_dir=tmp_path / "torch_example_smoke",
        enable_saves=False,
        torch_device=f"cuda:{torch.cuda.current_device()}",
        max_frames=2,
    )
    summary = run_training(config)

    assert np.isfinite(summary["last_loss"])
    assert np.isfinite(summary["last_psnr"])
    assert Path(summary["checkpoint_path"]).exists()

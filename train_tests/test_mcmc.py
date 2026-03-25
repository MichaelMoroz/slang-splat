from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.dataset import CameraSample
from train.mcmc import MCMCConfig, RGBMCMCTrainer
from train.losses import training_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_render_normalizes_rectangular_image_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = RGBMCMCTrainer(MCMCConfig(iterations=1), device="cuda")
    trainer.model = SimpleNamespace(splats=lambda: torch.zeros((14, 1), dtype=torch.float32, device="cuda"))
    trainer.iteration = 1

    width, height = 1297, 840
    fake_image = torch.zeros((width, height, 5), dtype=torch.float32, device="cuda")

    def fake_render_gaussian_splats(*args, **kwargs):
        return fake_image

    monkeypatch.setattr("train.mcmc.render_gaussian_splats", fake_render_gaussian_splats)

    camera = CameraSample(
        image_name="frame.png",
        image_path=Path("frame.png"),
        image_size=(width, height),
        camera_params=torch.zeros((15,), dtype=torch.float32, device="cuda"),
        image=torch.zeros((height, width, 3), dtype=torch.uint8, device="cuda"),
    )

    pred = trainer._render(camera, (0.0, 0.0, 0.0))

    assert tuple(pred.shape) == (height, width, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_training_loss_includes_depth_ratio_weight() -> None:
    pred = torch.zeros((4, 5, 3), dtype=torch.float32, device="cuda")
    target = torch.zeros_like(pred)
    depth_ratio = torch.full((4, 5), 2.0, dtype=torch.float32, device="cuda")
    opacity = torch.zeros((3, 1), dtype=torch.float32, device="cuda")
    scaling = torch.zeros((3, 3), dtype=torch.float32, device="cuda")

    total, l1, mse, train_psnr = training_loss(pred, target, depth_ratio, opacity, scaling, 0.0, 0.01, 0.0, 0.0)

    assert float(l1.item()) == 0.0
    assert float(mse.item()) == 0.0
    assert float(total.item()) == pytest.approx(0.02)
    assert torch.isfinite(train_psnr)
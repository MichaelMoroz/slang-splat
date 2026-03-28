from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.dataset import CameraSample
from train.mcmc import MCMCConfig, RGBMCMCTrainer
from train.losses import training_loss, inverse_sigmoid


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
    alpha = torch.full((4, 5), 0.25, dtype=torch.float32, device="cuda")
    depth_ratio = torch.full((4, 5), 2.0, dtype=torch.float32, device="cuda")
    opacity = torch.zeros((3, 1), dtype=torch.float32, device="cuda")
    scaling = torch.zeros((3, 3), dtype=torch.float32, device="cuda")

    total, l1, mse, train_psnr = training_loss(pred, target, alpha, depth_ratio, opacity, scaling, 0.0, 0.01, 0.0, 0.0)

    assert float(l1.item()) == 0.0
    assert float(mse.item()) == 0.0
    assert float(total.item()) == pytest.approx(0.02375)
    assert torch.isfinite(train_psnr)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_densify_step_appends_clones_rescales_parents_and_removes_low_opacity() -> None:
    cfg = MCMCConfig(
        iterations=1,
        densify_enabled=True,
        densify_interval=1,
        densify_clone_opacity=0.33,
        remove_opacity_threshold=0.001,
    )
    trainer = RGBMCMCTrainer(cfg, device="cuda")
    trainer.iteration = 1
    trainer.model._xyz = nn.Parameter(torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=torch.float32, device="cuda"))
    trainer.model._color = nn.Parameter(torch.tensor([[0.2, 0.4, 0.6], [0.9, 0.3, 0.1]], dtype=torch.float32, device="cuda"))
    trainer.model._log_scale = nn.Parameter(torch.log(torch.tensor([[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]], dtype=torch.float32, device="cuda")))
    trainer.model._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"))
    trainer.model._opacity = nn.Parameter(
        torch.cat(
            (
                inverse_sigmoid(torch.tensor([[0.5]], dtype=torch.float32, device="cuda")),
                inverse_sigmoid(torch.tensor([[5e-4]], dtype=torch.float32, device="cuda")),
            ),
            dim=0,
        )
    )
    trainer.model.setup_training(cfg, spatial_lr_scale=1.0)
    for param in (trainer.model._xyz, trainer.model._color, trainer.model._log_scale, trainer.model._rotation, trainer.model._opacity):
        param.grad = torch.zeros_like(param)
    trainer.model.optimizer.step()
    trainer.model.optimizer.zero_grad(set_to_none=True)

    clone_counts = torch.tensor([2, 0], device="cuda", dtype=torch.long)
    trainer.context = SimpleNamespace(
        clone_candidates_current=lambda *args, **kwargs: {
            "count": torch.tensor(2, device="cuda", dtype=torch.int64),
            "positions": torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=torch.float32, device="cuda"),
            "target_colors": torch.tensor([[0.8, 0.2, 0.4], [0.6, 0.1, 0.9]], dtype=torch.float32, device="cuda"),
            "ids": torch.tensor([0, 0], dtype=torch.long, device="cuda"),
            "clone_counts": clone_counts,
        }
    )

    trainer._densify_current_step(torch.zeros((4, 4, 3), dtype=torch.float32, device="cuda"))

    assert trainer.model.count == 3
    expected_scale = torch.full((3, 3), 1.0, dtype=torch.float32, device="cuda")
    assert torch.allclose(trainer.model.scaling, expected_scale, atol=1e-5)
    assert torch.allclose(trainer.model._xyz[1:], torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=torch.float32, device="cuda"))
    expected_clone_colors = torch.tensor([[0.5, 0.3, 0.5], [0.4, 0.25, 0.75]], dtype=torch.float32, device="cuda")
    assert torch.allclose(trainer.model._color[1:], expected_clone_colors, atol=1e-5)
    assert torch.all(trainer.model.opacity[:, 0] >= 0.001)
    for group in trainer.model.optimizer.param_groups:
        param = group["params"][0]
        state = trainer.model.optimizer.state[param]
        assert state["exp_avg"].shape[0] == trainer.model.count
        assert state["exp_avg_sq"].shape[0] == trainer.model.count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_non_densify_step_skips_clone_pass() -> None:
    trainer = RGBMCMCTrainer(MCMCConfig(iterations=1, densify_enabled=True, densify_interval=50), device="cuda")
    trainer.iteration = 1
    trainer.model._xyz = nn.Parameter(torch.zeros((1, 3), dtype=torch.float32, device="cuda"))
    trainer.model._color = nn.Parameter(torch.zeros((1, 3), dtype=torch.float32, device="cuda"))
    trainer.model._log_scale = nn.Parameter(torch.zeros((1, 3), dtype=torch.float32, device="cuda"))
    trainer.model._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"))
    trainer.model._opacity = nn.Parameter(inverse_sigmoid(torch.tensor([[0.5]], dtype=torch.float32, device="cuda")))
    trainer.context = SimpleNamespace(clone_candidates_current=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("clone pass should not run")))

    trainer._densify_current_step(torch.zeros((4, 4, 3), dtype=torch.float32, device="cuda"))

    assert trainer.model.count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_densify_respects_max_splats_cap() -> None:
    cfg = MCMCConfig(iterations=1, densify_enabled=True, densify_interval=1, max_splats=3)
    trainer = RGBMCMCTrainer(cfg, device="cuda")
    trainer.iteration = 1
    trainer.model._xyz = nn.Parameter(torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=torch.float32, device="cuda"))
    trainer.model._color = nn.Parameter(torch.tensor([[0.2, 0.4, 0.6], [0.9, 0.3, 0.1]], dtype=torch.float32, device="cuda"))
    trainer.model._log_scale = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32, device="cuda"))
    trainer.model._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"))
    trainer.model._opacity = nn.Parameter(inverse_sigmoid(torch.full((2, 1), 0.5, dtype=torch.float32, device="cuda")))
    trainer.model.setup_training(cfg, spatial_lr_scale=1.0)
    trainer.context = SimpleNamespace(
        clone_candidates_current=lambda *args, **kwargs: {
            "count": torch.tensor(3, device="cuda", dtype=torch.int64),
            "positions": torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]], dtype=torch.float32, device="cuda"),
            "target_colors": torch.tensor([[0.8, 0.2, 0.4], [0.6, 0.1, 0.9], [0.3, 0.7, 0.2]], dtype=torch.float32, device="cuda"),
            "ids": torch.tensor([0, 0, 0], dtype=torch.long, device="cuda"),
            "clone_counts": torch.tensor([3, 0], device="cuda", dtype=torch.long),
        }
    )

    trainer._densify_current_step(torch.zeros((4, 4, 3), dtype=torch.float32, device="cuda"))

    assert trainer.model.count == 3

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module import SplattingContext, render_gaussian_splats


def _make_splats(count: int = 64, seed: int = 29) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    splats = torch.zeros((14, count), dtype=torch.float32)
    splats[0:2] = torch.rand((2, count), generator=gen) * 0.36 - 0.18
    splats[2] = torch.rand((count,), generator=gen) * 1.5 + 4.5
    splats[3:6] = torch.rand((3, count), generator=gen) * (torch.log(torch.tensor(0.16)) - torch.log(torch.tensor(0.07))) + torch.log(torch.tensor(0.07))
    splats[6] = 1.0
    splats[10:13] = torch.rand((3, count), generator=gen) * 0.8 + 0.2
    splats[13] = torch.rand((count,), generator=gen) * 0.23 + 0.75
    return splats.cuda().requires_grad_(True)


def _make_camera(image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    return torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 72.0, 72.0, width * 0.5, height * 0.5, 0.1, 20.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_torch_wrapper_smoke() -> None:
    image_size = (64, 64)
    splats = _make_splats()
    camera = _make_camera(image_size)
    context = SplattingContext()
    image = render_gaussian_splats(splats, camera, image_size, context=context)
    assert tuple(image.shape) == (image_size[0], image_size[1], 4)
    assert torch.isfinite(image).all()
    loss = image.square().mean()
    loss.backward()
    assert splats.grad is not None
    assert torch.isfinite(splats.grad).all()
    assert float(splats.grad.abs().sum().item()) > 0.0

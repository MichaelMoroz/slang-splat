from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module import SplattingContext, render_gaussian_splats
from src.renderer import TorchGaussianRenderSettings, TorchGaussianRendererContext, render_gaussian_splats_torch


_SMOKE_IMAGE_SIZE = (64, 64)
_PARITY_IMAGE_SIZE = (128, 128)
_GRADIENT_PARITY_SPLAT_COUNT = 16_384


def _make_splats(device: torch.device, count: int = 8, seed: int = 7) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    splats = np.zeros((count, 14), dtype=np.float32)
    splats[:, 0:3] = rng.uniform(-0.35, 0.35, size=(count, 3)).astype(np.float32)
    splats[:, 2] += 3.0
    splats[:, 3:6] = np.log(np.full((count, 3), 0.05, dtype=np.float32))
    splats[:, 6] = 1.0
    splats[:, 10:13] = rng.uniform(0.15, 0.95, size=(count, 3)).astype(np.float32)
    splats[:, 13] = rng.uniform(0.2, 0.8, size=(count,)).astype(np.float32)
    return torch.tensor(splats, device=device, dtype=torch.float32)


def _make_camera(device: torch.device, distortion: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 72.0, 72.0, 32.0, 32.0, 0.1, 20.0, *distortion], device=device, dtype=torch.float32)


def _module_gradients(splats: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    grads = splats.detach().clone().requires_grad_(True)
    (0.5 * render_gaussian_splats(grads, camera, _PARITY_IMAGE_SIZE, context=SplattingContext()).square().sum()).backward()
    return grads.grad.detach().clone()


def _old_renderer_gradients(splats: torch.Tensor, camera: torch.Tensor, cuda_device: torch.device) -> torch.Tensor:
    grads = splats.detach().clone().requires_grad_(True)
    settings = TorchGaussianRenderSettings(width=_PARITY_IMAGE_SIZE[0], height=_PARITY_IMAGE_SIZE[1], list_capacity_multiplier=16, cached_raster_grad_atomic_mode="float")
    (0.5 * render_gaussian_splats_torch(grads, camera, settings, TorchGaussianRendererContext(torch_device=cuda_device)).square().sum()).backward()
    return grads.grad.detach().clone()


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def test_forward_smoke(cuda_device: torch.device) -> None:
    image = render_gaussian_splats(_make_splats(cuda_device), _make_camera(cuda_device), _SMOKE_IMAGE_SIZE, context=SplattingContext())
    assert tuple(image.shape) == (_SMOKE_IMAGE_SIZE[0], _SMOKE_IMAGE_SIZE[1], 4)
    assert torch.isfinite(image).all()


def test_backward_smoke(cuda_device: torch.device) -> None:
    splats = _make_splats(cuda_device, count=6).requires_grad_(True)
    loss = render_gaussian_splats(splats, _make_camera(cuda_device), _SMOKE_IMAGE_SIZE, context=SplattingContext()).sum()
    loss.backward()
    assert splats.grad is not None
    assert torch.isfinite(splats.grad).all()


def test_forward_matches_old_renderer_closely(cuda_device: torch.device) -> None:
    splats = _make_splats(cuda_device, count=6)
    camera = _make_camera(cuda_device)
    new_ctx = SplattingContext()
    old_ctx = TorchGaussianRendererContext(torch_device=cuda_device)
    settings = TorchGaussianRenderSettings(width=_PARITY_IMAGE_SIZE[0], height=_PARITY_IMAGE_SIZE[1], list_capacity_multiplier=16, cached_raster_grad_atomic_mode="float")
    image = render_gaussian_splats(splats, camera, _PARITY_IMAGE_SIZE, context=new_ctx)
    ref = render_gaussian_splats_torch(splats, camera, settings, old_ctx)
    torch.testing.assert_close(image, ref, rtol=1e-4, atol=1e-4)


def test_gradient_matches_old_renderer_closely(cuda_device: torch.device) -> None:
    splats = _make_splats(cuda_device, count=_GRADIENT_PARITY_SPLAT_COUNT)
    camera = _make_camera(cuda_device)
    # Float atomics are non-deterministic, so measure each path's self-variance before comparing them.
    new_grad_a = _module_gradients(splats, camera)
    new_grad_b = _module_gradients(splats, camera)
    old_grad_a = _old_renderer_gradients(splats, camera, cuda_device)
    old_grad_b = _old_renderer_gradients(splats, camera, cuda_device)
    new_self_max = float((new_grad_a - new_grad_b).abs().max().item())
    old_self_max = float((old_grad_a - old_grad_b).abs().max().item())
    cross_max = float((new_grad_a - old_grad_a).abs().max().item())
    try:
        torch.testing.assert_close(new_grad_a, old_grad_a, rtol=5e-4, atol=5e-4)
    except AssertionError as exc:
        raise AssertionError(
            f"{exc}\nMeasured float-atomic self-variance: module={new_self_max:.6g}, old={old_self_max:.6g}, cross={cross_max:.6g}"
        ) from exc


def test_distortion_case(cuda_device: torch.device) -> None:
    image = render_gaussian_splats(_make_splats(cuda_device), _make_camera(cuda_device, (0.05, -0.02)), _SMOKE_IMAGE_SIZE, context=SplattingContext())
    assert torch.isfinite(image).all()


def test_stable_sorting_case(cuda_device: torch.device) -> None:
    splats = _make_splats(cuda_device, count=2)
    splats[:, 0:3] = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]], device=cuda_device)
    splats[0, 10:13] = torch.tensor([1.0, 0.0, 0.0], device=cuda_device)
    splats[1, 10:13] = torch.tensor([0.0, 1.0, 0.0], device=cuda_device)
    image = render_gaussian_splats(splats, _make_camera(cuda_device), _SMOKE_IMAGE_SIZE, context=SplattingContext())
    center = image[32, 32, :3]
    assert center[0] >= center[1]


def test_alpha_gradient_is_public_alpha_space(cuda_device: torch.device) -> None:
    splats = _make_splats(cuda_device, count=4).requires_grad_(True)
    render_gaussian_splats(splats, _make_camera(cuda_device), _SMOKE_IMAGE_SIZE, context=SplattingContext()).sum().backward()
    assert torch.isfinite(splats.grad[:, 13]).all()
    assert torch.count_nonzero(splats.grad[:, 13]).item() > 0


def test_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    py_lines = sum(1 for line in (root / "module" / "splatting.py").read_text().splitlines() if line.strip() and not line.strip().startswith("#"))
    slang_lines = 0
    for path in (root / "module" / "shaders").glob("*.slang"):
        slang_lines += sum(1 for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("//"))
    assert py_lines <= 300
    assert slang_lines <= 1000

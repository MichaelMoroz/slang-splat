from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module import SplattingContext, render_gaussian_splats
from src.renderer import TorchGaussianRenderSettings, TorchGaussianRendererContext, render_gaussian_splats_torch


_SMOKE_IMAGE_SIZE = (64, 64)
_PARITY_IMAGE_SIZE = (128, 128)
_GRADIENT_PARITY_SPLAT_COUNT = 16_384
_GRADIENT_REFERENCE_IMAGE_SIZE = (32, 32)
_GRADIENT_REFERENCE_SPLAT_COUNT = 128
_GRADIENT_REFERENCE_PARAM_SAMPLES = 32
_GRADIENT_FINITE_DIFF_EPS = 1e-3
_GRADIENT_REFERENCE_MAX_MS = 4000.0


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


def _module_loss(splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int], context: SplattingContext | None = None) -> torch.Tensor:
    return 0.5 * render_gaussian_splats(splats, camera, image_size, context=context or SplattingContext()).square().sum()


def _module_gradients(splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int], context: SplattingContext | None = None) -> torch.Tensor:
    grads = splats.detach().clone().requires_grad_(True)
    _module_loss(grads, camera, image_size, context).backward()
    return grads.grad.detach().clone()


def _sample_param_indices(splats: torch.Tensor, sample_count: int, seed: int = 123) -> torch.Tensor:
    return torch.randperm(splats.numel(), generator=torch.Generator(device="cpu").manual_seed(seed))[:sample_count]


def _finite_difference_samples(
    splats: torch.Tensor,
    camera: torch.Tensor,
    image_size: tuple[int, int],
    eps: float,
    sample_indices: torch.Tensor,
    context: SplattingContext,
) -> torch.Tensor:
    work = splats.detach().clone()
    reference = torch.empty((int(sample_indices.shape[0]),), device=splats.device, dtype=splats.dtype)
    param_count = int(splats.shape[1])
    for sample_idx, flat_idx in enumerate(sample_indices.tolist()):
        splat_idx, param_idx = divmod(int(flat_idx), param_count)
        original = float(work[splat_idx, param_idx].item())
        work[splat_idx, param_idx] = original + eps
        loss_plus = float(_module_loss(work, camera, image_size, context).item())
        work[splat_idx, param_idx] = original - eps
        loss_minus = float(_module_loss(work, camera, image_size, context).item())
        work[splat_idx, param_idx] = original
        reference[sample_idx] = (loss_plus - loss_minus) / (2.0 * eps)
    return reference


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


def test_gradient_matches_finite_difference_reference(cuda_device: torch.device) -> None:
    splats = _make_splats(cuda_device, count=_GRADIENT_REFERENCE_SPLAT_COUNT)
    camera = _make_camera(cuda_device)
    sample_indices = _sample_param_indices(splats, _GRADIENT_REFERENCE_PARAM_SAMPLES)
    autograd_context = SplattingContext()
    finite_diff_context = SplattingContext()
    _module_gradients(splats, camera, _GRADIENT_REFERENCE_IMAGE_SIZE, autograd_context)
    _finite_difference_samples(splats, camera, _GRADIENT_REFERENCE_IMAGE_SIZE, _GRADIENT_FINITE_DIFF_EPS, sample_indices[:1], finite_diff_context)
    torch.cuda.synchronize()
    start = time.perf_counter()
    autograd = _module_gradients(splats, camera, _GRADIENT_REFERENCE_IMAGE_SIZE, autograd_context).reshape(-1)
    finite_diff = _finite_difference_samples(
        splats,
        camera,
        _GRADIENT_REFERENCE_IMAGE_SIZE,
        _GRADIENT_FINITE_DIFF_EPS,
        sample_indices,
        finite_diff_context,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    torch.testing.assert_close(autograd.index_select(0, sample_indices.to(device=autograd.device)), finite_diff, rtol=5e-2, atol=2e-1)
    assert elapsed_ms <= _GRADIENT_REFERENCE_MAX_MS, f"finite-difference check took {elapsed_ms:.2f} ms"


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

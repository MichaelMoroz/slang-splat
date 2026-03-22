from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module import SplattingContext, render_gaussian_splats


_SMOKE_IMAGE_SIZE = (64, 64)
_PARITY_IMAGE_SIZE = (128, 128)
_GRADIENT_PARITY_SPLAT_COUNT = 16_384
_GRADIENT_REFERENCE_IMAGE_SIZE = (32, 32)
_GRADIENT_REFERENCE_SPLAT_COUNT = 128
_GRADIENT_REFERENCE_PARAM_SAMPLES = 32
_GRADIENT_FINITE_DIFF_EPS = 1e-3
_GRADIENT_REFERENCE_MAX_MS = 4000.0
_GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0
_RAY_DENOMINATOR_FLOOR = 1e-10
_ALPHA_CUTOFF = 1 / 255
_TRANS_THRESHOLD = 0.005
_OUTPUT_GAMMA = 2.2
_SMALL_VALUE = 1e-6


def _make_splats(device: torch.device, count: int = 8, seed: int = 7) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    splats = np.zeros((count, 14), dtype=np.float32)
    splats[:, 0:3] = rng.uniform(-0.35, 0.35, size=(count, 3)).astype(np.float32)
    splats[:, 2] += 3.0
    splats[:, 3:6] = np.log(np.full((count, 3), 0.05, dtype=np.float32))
    splats[:, 6] = 1.0
    splats[:, 10:13] = rng.uniform(0.15, 0.95, size=(count, 3)).astype(np.float32)
    splats[:, 13] = rng.uniform(0.2, 0.8, size=(count,)).astype(np.float32)
    return torch.tensor(splats, device=device, dtype=torch.float32).T.contiguous()


def _make_camera(device: torch.device, distortion: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 72.0, 72.0, 32.0, 32.0, 0.1, 20.0, *distortion], device=device, dtype=torch.float32)


def _camera_basis(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q / torch.clamp(torch.linalg.norm(q), min=1e-12)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        )
    )


def _quat_rotate(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    qv = q[1:].expand_as(v)
    return v + 2.0 * torch.cross(torch.cross(v, qv, dim=-1) + q[0] * v, qv, dim=-1)


def _undistort_normalized(uv_distorted: torch.Tensor, k1: float, k2: float) -> torch.Tensor:
    if abs(k1) <= _SMALL_VALUE and abs(k2) <= _SMALL_VALUE:
        return uv_distorted
    radius_distorted = torch.linalg.norm(uv_distorted, dim=-1, keepdim=True)
    radius = radius_distorted.clone()
    for _ in range(6):
        r2 = radius.square()
        r4 = r2.square()
        deriv = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4
        safe = deriv.abs() > _SMALL_VALUE
        next_radius = radius - (radius * (1.0 + k1 * r2 + k2 * r4) - radius_distorted) / torch.where(safe, deriv, torch.ones_like(deriv))
        radius = torch.where(safe & torch.isfinite(next_radius) & (next_radius >= 0.0), next_radius, radius)
    return uv_distorted * (radius / torch.clamp(radius_distorted, min=_SMALL_VALUE))


def _screen_to_world_rays(camera: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    ys, xs = torch.meshgrid(
        torch.arange(height, device=camera.device, dtype=torch.float32),
        torch.arange(width, device=camera.device, dtype=torch.float32),
        indexing="ij",
    )
    focal = camera[7:9]
    principal = camera[9:11]
    uv_distorted = (torch.stack((xs + 0.5, ys + 0.5), dim=-1) - principal) / torch.clamp(focal, min=_SMALL_VALUE)
    uv = _undistort_normalized(uv_distorted, float(camera[13].item()), float(camera[14].item()))
    camera_rays = torch.nn.functional.normalize(torch.cat((uv, torch.ones_like(uv[..., :1])), dim=-1), dim=-1)
    return camera_rays @ _camera_basis(camera[0:4]).T


def _ground_truth_render(splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    cam_basis = _camera_basis(camera[0:4])
    cam_pos = -cam_basis.T @ camera[4:7]
    projected = SplattingContext().project(splats, camera, image_size)
    sorted_splats = projected["sorted_splats"][:, projected["tile_counts"].to(dtype=torch.int32) > 0]
    rays = _screen_to_world_rays(camera, image_size).reshape(-1, 3)
    accum = torch.zeros_like(rays)
    trans = torch.ones((rays.shape[0],), device=splats.device, dtype=splats.dtype)
    for i in range(int(sorted_splats.shape[1])):
        gaussian = sorted_splats[:, i]
        scale = torch.clamp(torch.exp(gaussian[3:6]) * _GAUSSIAN_SUPPORT_SIGMA_RADIUS, min=_SMALL_VALUE)
        ro_local = _quat_rotate((cam_pos - gaussian[0:3]).expand_as(rays), gaussian[6:10]) / scale
        ray_local = _quat_rotate(rays, gaussian[6:10]) / scale
        denom = (ray_local * ray_local).sum(dim=-1)
        t_closest = -(ray_local * ro_local).sum(dim=-1) / torch.clamp(denom, min=_RAY_DENOMINATOR_FLOOR)
        closest = ro_local + ray_local * t_closest.unsqueeze(-1)
        rho2 = (closest * closest).sum(dim=-1)
        alpha = torch.where(
            (denom > _RAY_DENOMINATOR_FLOOR) & (t_closest > 0.0),
            torch.clamp(gaussian[13], min=1e-6, max=1.0 - 1e-6) * torch.exp(-0.5 * _GAUSSIAN_SUPPORT_SIGMA_RADIUS * _GAUSSIAN_SUPPORT_SIGMA_RADIUS * rho2),
            torch.zeros_like(trans),
        )
        alpha = torch.where(alpha >= _ALPHA_CUTOFF, alpha, torch.zeros_like(alpha))
        accum = accum + (trans * alpha).unsqueeze(-1) * gaussian[10:13]
        trans = trans * (1.0 - alpha)
        if bool(torch.all(trans < _TRANS_THRESHOLD).item()):
            break
    color = torch.pow(torch.clamp(accum, min=0.0), _OUTPUT_GAMMA)
    return torch.cat((color, (1.0 - trans).unsqueeze(-1)), dim=-1).reshape(image_size[1], image_size[0], 4).transpose(0, 1)


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
    splat_count = int(splats.shape[1])
    for sample_idx, flat_idx in enumerate(sample_indices.tolist()):
        param_idx, splat_idx = divmod(int(flat_idx), splat_count)
        original = float(work[param_idx, splat_idx].item())
        work[param_idx, splat_idx] = original + eps
        loss_plus = float(_module_loss(work, camera, image_size, context).item())
        work[param_idx, splat_idx] = original - eps
        loss_minus = float(_module_loss(work, camera, image_size, context).item())
        work[param_idx, splat_idx] = original
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


def test_forward_matches_ground_truth_closely(cuda_device: torch.device) -> None:
    splats = _make_splats(cuda_device, count=6)
    camera = _make_camera(cuda_device)
    image = render_gaussian_splats(splats, camera, _PARITY_IMAGE_SIZE, context=SplattingContext())
    ref = _ground_truth_render(splats, camera, _PARITY_IMAGE_SIZE)
    torch.testing.assert_close(image[..., :3], ref[..., :3], rtol=2e-2, atol=3e-2)
    assert float((image[..., 3] - ref[..., 3]).abs().max().item()) <= 3e-2


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
    splats[0:3, :] = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]], device=cuda_device)
    splats[10:13, 0] = torch.tensor([1.0, 0.0, 0.0], device=cuda_device)
    splats[10:13, 1] = torch.tensor([0.0, 1.0, 0.0], device=cuda_device)
    image = render_gaussian_splats(splats, _make_camera(cuda_device), _SMOKE_IMAGE_SIZE, context=SplattingContext())
    center = image[32, 32, :3]
    assert center[0] >= center[1]


def test_alpha_gradient_is_public_alpha_space(cuda_device: torch.device) -> None:
    splats = _make_splats(cuda_device, count=6).requires_grad_(True)
    render_gaussian_splats(splats, _make_camera(cuda_device), _PARITY_IMAGE_SIZE, context=SplattingContext()).square().sum().backward()
    assert torch.isfinite(splats.grad[13]).all()
    assert torch.count_nonzero(splats.grad[13]).item() > 0


def test_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    py_lines = sum(1 for line in (root / "module" / "splatting.py").read_text().splitlines() if line.strip() and not line.strip().startswith("#"))
    slang_lines = 0
    for path in (root / "module" / "shaders").glob("*.slang"):
        slang_lines += sum(1 for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("//"))
    assert py_lines <= 300
    assert slang_lines <= 1000

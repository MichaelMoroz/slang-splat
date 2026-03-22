from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module import SplattingContext, render_gaussian_splats

_SMALL_VALUE = 1e-6
_ALPHA_CUTOFF = 1.0 / 255.0
_TRANS_THRESHOLD = 0.005
_GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0
_OUTPUT_GAMMA = 2.2


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


def _make_camera(image_size: tuple[int, int], focal_pixels: tuple[float, float] = (72.0, 72.0)) -> torch.Tensor:
    width, height = image_size
    return torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, focal_pixels[0], focal_pixels[1], width * 0.5, height * 0.5, 0.1, 20.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )


def _image_hw(image: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    if tuple(image.shape) == (height, width, 4):
        return image
    if tuple(image.shape) == (width, height, 4):
        return image.permute(1, 0, 2).contiguous()
    raise AssertionError(f"Unexpected image shape {tuple(image.shape)} for image_size={image_size}.")


def _camera_basis(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.clamp(torch.linalg.norm(q), min=1e-12)
    w, x, y, z = q.unbind()
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        )
    ).to(dtype=torch.float32)


def _undistort_normalized(uv_distorted: torch.Tensor, k1: float, k2: float) -> torch.Tensor:
    if abs(k1) <= _SMALL_VALUE and abs(k2) <= _SMALL_VALUE:
        return uv_distorted
    radius_distorted = torch.linalg.norm(uv_distorted, dim=-1, keepdim=True)
    radius = radius_distorted.clone()
    for _ in range(6):
        r2 = radius * radius
        r4 = r2 * r2
        deriv = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4
        safe = deriv.abs() > _SMALL_VALUE
        next_radius = radius - (radius * (1.0 + k1 * r2 + k2 * r4) - radius_distorted) / torch.where(safe, deriv, torch.ones_like(deriv))
        radius = torch.where(safe & torch.isfinite(next_radius) & (next_radius >= 0.0), next_radius, radius)
    return uv_distorted * (radius / torch.clamp(radius_distorted, min=_SMALL_VALUE))


def _screen_to_world_rays(camera: torch.Tensor, image_size: tuple[int, int], y0: int, y1: int) -> torch.Tensor:
    width, height = image_size
    ys = torch.arange(y0, y1, device=camera.device, dtype=torch.float32)
    xs = torch.arange(width, device=camera.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv_distorted = (torch.stack((xx + 0.5, yy + 0.5), dim=-1) - camera[9:11]) / torch.clamp(camera[7:9], min=_SMALL_VALUE)
    uv = _undistort_normalized(uv_distorted, float(camera[13].item()), float(camera[14].item()))
    camera_rays = torch.cat((uv, torch.ones_like(uv[..., :1])), dim=-1)
    camera_rays = camera_rays / torch.clamp(torch.linalg.norm(camera_rays, dim=-1, keepdim=True), min=_SMALL_VALUE)
    return camera_rays @ _camera_basis(camera[0:4]).T


def _ground_truth_render_torch(splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int], chunk_rows: int = 96) -> torch.Tensor:
    width, height = image_size
    cam_basis = _camera_basis(camera[0:4])
    cam_pos = -cam_basis.T @ camera[4:7]
    distances = torch.linalg.norm(splats[0:3].T - cam_pos[None, :], dim=1)
    sorted_splats = splats.index_select(1, torch.argsort(distances, stable=True))
    rows: list[torch.Tensor] = []
    for y0 in range(0, height, chunk_rows):
        y1 = min(y0 + chunk_rows, height)
        rays = _screen_to_world_rays(camera, image_size, y0, y1).reshape(-1, 3)
        accum = torch.zeros((rays.shape[0], 3), device=splats.device, dtype=torch.float32)
        trans = torch.ones((rays.shape[0],), device=splats.device, dtype=torch.float32)
        for i in range(sorted_splats.shape[1]):
            gaussian = sorted_splats[:, i]
            scale = torch.clamp(torch.exp(gaussian[3:6]) * _GAUSSIAN_SUPPORT_SIGMA_RADIUS, min=_SMALL_VALUE)
            q = gaussian[6:10]
            qv = q[1:]
            ro = cam_pos - gaussian[0:3]
            ro_local = (ro + 2.0 * torch.cross(torch.cross(ro, qv, dim=0) + q[0] * ro, qv, dim=0)) / scale
            cross1 = torch.cross(rays, qv.expand_as(rays), dim=1)
            cross2 = torch.cross(cross1 + q[0] * rays, qv.expand_as(rays), dim=1)
            ray_local = (rays + 2.0 * cross2) / scale
            denom = torch.sum(ray_local * ray_local, dim=1)
            t_closest = -torch.sum(ray_local * ro_local[None, :], dim=1) / torch.clamp(denom, min=1e-10)
            closest = ro_local[None, :] + ray_local * t_closest[:, None]
            rho2 = torch.sum(closest * closest, dim=1)
            alpha = torch.where(
                (denom > 1e-10) & (t_closest > 0.0),
                torch.clamp(gaussian[13], min=_SMALL_VALUE, max=1.0 - _SMALL_VALUE) * torch.exp(-0.5 * _GAUSSIAN_SUPPORT_SIGMA_RADIUS * _GAUSSIAN_SUPPORT_SIGMA_RADIUS * rho2),
                torch.zeros_like(rho2),
            )
            alpha = torch.where(alpha >= _ALPHA_CUTOFF, alpha, torch.zeros_like(alpha))
            accum = accum + (trans * alpha)[:, None] * gaussian[10:13][None, :]
            trans = trans * (1.0 - alpha)
            if bool(torch.all(trans < _TRANS_THRESHOLD)):
                break
        color = torch.pow(torch.clamp(accum, min=0.0), _OUTPUT_GAMMA)
        rows.append(torch.cat((color, (1.0 - trans)[:, None]), dim=1).reshape(y1 - y0, width, 4))
    return torch.cat(rows, dim=0)


def _make_large_resolution_splats(image_size: tuple[int, int], count: int = 8, seed: int = 7) -> torch.Tensor:
    width, height = image_size
    focal = 0.72 * width
    depth = 6.0
    support_sigma_radius = float(torch.sqrt(-2.0 * torch.log(torch.tensor(_ALPHA_CUTOFF / 0.98))).item())
    gen = torch.Generator(device="cpu").manual_seed(seed)
    splats = torch.zeros((14, count), dtype=torch.float32, device="cuda")
    screen_x = (torch.rand((count,), generator=gen) * 0.6 + 0.2) * width
    screen_y = (torch.rand((count,), generator=gen) * 0.6 + 0.2) * height
    radius_px = (torch.rand((count,), generator=gen) * 0.03 + 0.09) * min(width, height)
    splats[0] = (screen_x - width * 0.5) * depth / focal
    splats[1] = (screen_y - height * 0.5) * depth / focal
    splats[2] = depth
    scale = radius_px * depth / (focal * support_sigma_radius)
    splats[3] = torch.log(scale * 0.9)
    splats[4] = torch.log(scale * 1.1)
    splats[5] = torch.log(torch.full((count,), 0.20, device="cuda"))
    splats[6] = 1.0
    splats[10:13] = torch.rand((3, count), generator=gen).to(device="cuda") * 0.7 + 0.3
    splats[13] = torch.rand((count,), generator=gen).to(device="cuda") * 0.07 + 0.91
    return splats


def _make_large_resolution_stretched_splats(image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    focal = 0.72 * width
    depth = 6.0
    support_sigma_radius = float(torch.sqrt(-2.0 * torch.log(torch.tensor(_ALPHA_CUTOFF / 0.98))).item())
    splats = torch.zeros((14, 2), dtype=torch.float32, device="cuda")
    splats[0:3, 0] = torch.tensor([0.0, 0.0, depth], device="cuda")
    radii0 = torch.tensor([0.06 * min(width, height), 0.48 * min(width, height)], dtype=torch.float32, device="cuda")
    scale0 = radii0 * depth / (focal * support_sigma_radius)
    splats[3, 0] = torch.log(scale0[0])
    splats[4, 0] = torch.log(scale0[1])
    splats[5, 0] = torch.log(torch.tensor(0.20, dtype=torch.float32, device="cuda"))
    half_angle = torch.tensor(0.78539816339 * 0.5, dtype=torch.float32, device="cuda")
    splats[6, 0] = torch.cos(half_angle)
    splats[9, 0] = torch.sin(half_angle)
    splats[10:13, 0] = torch.tensor([0.95, 0.40, 0.20], dtype=torch.float32, device="cuda")
    splats[13, 0] = 0.98

    screen_y = 0.28 * height
    splats[0:3, 1] = torch.tensor([0.0, (screen_y - height * 0.5) * depth / focal, depth], dtype=torch.float32, device="cuda")
    radii1 = torch.tensor([0.52 * width, 0.05 * height], dtype=torch.float32, device="cuda")
    scale1 = radii1 * depth / (focal * support_sigma_radius)
    splats[3, 1] = torch.log(scale1[0])
    splats[4, 1] = torch.log(scale1[1])
    splats[5, 1] = torch.log(torch.tensor(0.20, dtype=torch.float32, device="cuda"))
    splats[6, 1] = 1.0
    splats[10:13, 1] = torch.tensor([0.20, 0.60, 0.95], dtype=torch.float32, device="cuda")
    splats[13, 1] = 0.97
    return splats


def _make_large_resolution_offcenter_stretched_splat(image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    focal = 0.72 * width
    depth = 5.5
    support_sigma_radius = float(torch.sqrt(-2.0 * torch.log(torch.tensor(_ALPHA_CUTOFF / 0.98))).item())
    splat = torch.zeros((14, 1), dtype=torch.float32, device="cuda")
    splat[0, 0] = (2600.0 - width * 0.5) * depth / focal
    splat[1, 0] = (500.0 - height * 0.5) * depth / focal
    splat[2, 0] = depth
    radii = torch.tensor([140.0, 1700.0], dtype=torch.float32, device="cuda")
    scale = radii * depth / (focal * support_sigma_radius)
    splat[3, 0] = torch.log(scale[0])
    splat[4, 0] = torch.log(scale[1])
    splat[5, 0] = torch.log(torch.tensor(0.12, dtype=torch.float32, device="cuda"))
    half_angle = torch.tensor(1.0471975512 * 0.5, dtype=torch.float32, device="cuda")
    splat[6, 0] = torch.cos(half_angle)
    splat[9, 0] = torch.sin(half_angle)
    splat[10:13, 0] = torch.tensor([0.90, 0.50, 0.20], dtype=torch.float32, device="cuda")
    splat[13, 0] = 0.98
    return splat


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_torch_wrapper_zero_splats_smoke() -> None:
    image_size = (64, 64)
    splats = torch.zeros((14, 0), dtype=torch.float32, device="cuda", requires_grad=True)
    camera = _make_camera(image_size)
    context = SplattingContext()
    image = render_gaussian_splats(splats, camera, image_size, context=context)
    assert tuple(image.shape) == (image_size[0], image_size[1], 4)
    assert torch.isfinite(image).all()
    assert float(image.abs().max().item()) == 0.0
    loss = image.square().sum()
    loss.backward()
    assert splats.grad is not None
    assert tuple(splats.grad.shape) == (14, 0)
    assert torch.isfinite(splats.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("image_size", [(1920, 1080), (3840, 2160)])
def test_large_resolution_ground_truth_image_matches(image_size: tuple[int, int]) -> None:
    splats = _make_large_resolution_splats(image_size)
    focal = 0.72 * image_size[0]
    camera = _make_camera(image_size, focal_pixels=(focal, focal))
    context = SplattingContext()
    image = _image_hw(render_gaussian_splats(splats, camera, image_size, context=context), image_size)
    ref = _ground_truth_render_torch(splats, camera, image_size)

    alpha = image[..., 3]
    top = alpha[: image_size[1] // 2]
    bottom = alpha[image_size[1] // 2 :]
    assert int(torch.count_nonzero(top > 1e-4).item()) > 10000
    assert int(torch.count_nonzero(bottom > 1e-4).item()) > 10000

    rgb_error = (image[..., :3] - ref[..., :3]).abs()
    alpha_error = (image[..., 3] - ref[..., 3]).abs()
    assert float(rgb_error.mean().item()) <= 3e-3
    assert float(alpha_error.mean().item()) <= 3e-3
    assert float(torch.quantile(alpha_error.reshape(-1), 0.999).item()) <= 3e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("image_size", [(1920, 1080), (3840, 2160)])
def test_large_resolution_stretched_ground_truth_image_matches(image_size: tuple[int, int]) -> None:
    splats = _make_large_resolution_stretched_splats(image_size)
    focal = 0.72 * image_size[0]
    camera = _make_camera(image_size, focal_pixels=(focal, focal))
    context = SplattingContext()
    image = _image_hw(render_gaussian_splats(splats, camera, image_size, context=context), image_size)
    ref = _ground_truth_render_torch(splats, camera, image_size)
    alpha = image[..., 3]
    assert float((alpha > 1e-4).float().mean().item()) >= 0.15
    rgb_error = (image[..., :3] - ref[..., :3]).abs()
    alpha_error = (image[..., 3] - ref[..., 3]).abs()
    assert float(rgb_error.mean().item()) <= 5e-3
    assert float(alpha_error.mean().item()) <= 5e-3
    assert float(torch.quantile(alpha_error.reshape(-1), 0.999).item()) <= 5e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_large_resolution_offcenter_stretched_ground_truth_image_matches() -> None:
    image_size = (3840, 2160)
    splats = _make_large_resolution_offcenter_stretched_splat(image_size)
    camera = _make_camera(image_size, focal_pixels=(0.72 * image_size[0], 0.72 * image_size[0]))
    context = SplattingContext()
    image = _image_hw(render_gaussian_splats(splats, camera, image_size, context=context), image_size)
    ref = _ground_truth_render_torch(splats, camera, image_size)
    alpha_error = (image[..., 3] - ref[..., 3]).abs()
    assert float(alpha_error.mean().item()) <= 5e-4
    assert float(torch.quantile(alpha_error.reshape(-1), 0.999).item()) <= 1e-2

from __future__ import annotations

import math

import numpy as np
import pytest
import slangpy as spy

torch = pytest.importorskip("torch")

from src.common import SHADER_INCLUDE_PATHS
from src.renderer import Camera, GaussianRenderer, TorchGaussianRenderSettings, TorchGaussianRendererContext, render_gaussian_splats_torch
from src.scene import GaussianScene

_SCENE_LAYOUT_IMAGE_SIZE = (160, 160)
_SCENE_SPLAT_COUNT = 61
_SCENE_SEED = 17
_DEFAULT_CAMERA_POSITION_Z = 3.0
_GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0
_TORCH_GT_SMALL_VALUE = 1e-6
_PIXEL_SUBSET_COUNT = 8192
_PIXEL_SUBSET_SEED = 12345
_SCENE_CENTER_SPREAD = 0.08
_SCENE_MINOR_AXIS_PIXELS = (7.0, 11.0)
_SCENE_DEPTH_RANGE = (4.4, 7.4)
_SCENE_ALPHA_RANGE = (0.55, 0.85)
_SCENE_COLOR_RANGE = (0.2, 1.0)
_SCENE_MAX_ANISOTROPY = 6.0
_RGB_MEAN_ABS_TOL = 8e-3
_ALPHA_MEAN_ABS_TOL = 1.2e-2
_ALPHA_P995_TOL = 4e-2
_GRADIENT_COSINE_MIN = 0.992
_GRADIENT_REL_L2_MAX = 0.11
_GRAD_CHANNEL_WEIGHTS = np.array([0.7, -0.25, 0.5, 1.1], dtype=np.float32)
_CAMERA_SPECS = (
    {"image_size": (96, 96), "fov_y_degrees": 28.0, "focal_scale": (1.0, 1.0)},
    {"image_size": (67, 191), "fov_y_degrees": 24.0, "focal_scale": (0.2, 0.7)},
    {"image_size": (171, 83), "fov_y_degrees": 30.0, "focal_scale": (1.45, 0.9)},
)


def _make_splats(device: torch.device, count: int = 8, seed: int = 7) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    splats = np.zeros((count, 14), dtype=np.float32)
    splats[:, 0:3] = rng.uniform(-0.35, 0.35, size=(count, 3)).astype(np.float32)
    splats[:, 2] += 0.2
    splats[:, 3:6] = np.log(np.full((count, 3), 0.05, dtype=np.float32))
    splats[:, 6] = 1.0
    splats[:, 10:13] = rng.uniform(0.15, 0.95, size=(count, 3)).astype(np.float32)
    splats[:, 13] = rng.uniform(0.2, 0.8, size=(count,)).astype(np.float32)
    return torch.tensor(splats, device=device, dtype=torch.float32)


def _make_camera_params(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 72.0, 72.0, 32.0, 32.0, 0.1, 20.0, 0.0, 0.0],
        device=device,
        dtype=torch.float32,
    )


def _make_forward_camera(
    image_size: tuple[int, int],
    camera_position_z: float = _DEFAULT_CAMERA_POSITION_Z,
    fov_y_degrees: float = 28.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    width, height = image_size
    focal = 0.5 * float(height) / math.tan(0.5 * math.radians(fov_y_degrees))
    return torch.tensor(
        (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            camera_position_z,
            focal,
            focal,
            width * 0.5,
            height * 0.5,
            0.1,
            40.0,
            0.0,
            0.0,
        ),
        device=device,
        dtype=torch.float32,
    )


def _random_quaternions(count: int, generator: torch.Generator) -> torch.Tensor:
    quat = torch.randn((count, 4), generator=generator, dtype=torch.float32)
    return quat / torch.clamp(torch.linalg.norm(quat, dim=1, keepdim=True), min=1e-8)


def _make_camera_spec_tensor(spec: dict[str, object], device: torch.device) -> torch.Tensor:
    camera = _make_forward_camera(tuple(spec["image_size"]), fov_y_degrees=float(spec["fov_y_degrees"]), device=device)
    fx_scale, fy_scale = spec["focal_scale"]
    camera[7] *= np.float32(fx_scale)
    camera[8] *= np.float32(fy_scale)
    return camera


def _make_scene_splats(device: torch.device) -> torch.Tensor:
    scene_camera = _make_forward_camera(_SCENE_LAYOUT_IMAGE_SIZE, device="cpu").detach().cpu().numpy().astype(np.float32, copy=False)
    width, height = map(float, _SCENE_LAYOUT_IMAGE_SIZE)
    focal = float(scene_camera[7])
    cam_z = float(scene_camera[6])
    layout_rng = np.random.default_rng(_SCENE_SEED)
    attribute_rng = np.random.default_rng(_SCENE_SEED)
    depths = np.linspace(_SCENE_DEPTH_RANGE[0], _SCENE_DEPTH_RANGE[1], _SCENE_SPLAT_COUNT, dtype=np.float32)
    depth_from_camera = depths + cam_z
    screen_x = layout_rng.uniform(0.5 - _SCENE_CENTER_SPREAD, 0.5 + _SCENE_CENTER_SPREAD, size=_SCENE_SPLAT_COUNT).astype(np.float32) * width
    screen_y = layout_rng.uniform(0.5 - _SCENE_CENTER_SPREAD, 0.5 + _SCENE_CENTER_SPREAD, size=_SCENE_SPLAT_COUNT).astype(np.float32) * height
    ratios = np.exp(attribute_rng.uniform(0.0, math.log(_SCENE_MAX_ANISOTROPY), size=_SCENE_SPLAT_COUNT)).astype(np.float32)
    ratios[0] = np.float32(1.0)
    ratios[1] = np.float32(_SCENE_MAX_ANISOTROPY)
    colors = attribute_rng.uniform(_SCENE_COLOR_RANGE[0], _SCENE_COLOR_RANGE[1], size=(_SCENE_SPLAT_COUNT, 3)).astype(np.float32)
    alpha = attribute_rng.uniform(_SCENE_ALPHA_RANGE[0], _SCENE_ALPHA_RANGE[1], size=_SCENE_SPLAT_COUNT).astype(np.float32)
    minor_px = layout_rng.uniform(_SCENE_MINOR_AXIS_PIXELS[0], _SCENE_MINOR_AXIS_PIXELS[1], size=_SCENE_SPLAT_COUNT).astype(np.float32)
    world_minor = minor_px * depth_from_camera / focal
    world_major = world_minor * ratios
    axis_shifts = attribute_rng.permutation(_SCENE_SPLAT_COUNT) % 3
    quats = _random_quaternions(_SCENE_SPLAT_COUNT, torch.Generator(device="cpu").manual_seed(_SCENE_SEED)).numpy().astype(np.float32, copy=False)

    splats = np.zeros((_SCENE_SPLAT_COUNT, 14), dtype=np.float32)
    splats[:, 0] = (screen_x - 0.5 * width) * depth_from_camera / focal
    splats[:, 1] = (screen_y - 0.5 * height) * depth_from_camera / focal
    splats[:, 2] = depths
    for splat_index in range(_SCENE_SPLAT_COUNT):
        splats[splat_index, 6:10] = quats[splat_index]
        axis_order = [0, 1, 2]
        shift = int(axis_shifts[splat_index])
        axis_order = axis_order[shift:] + axis_order[:shift]
        scales = np.array((world_major[splat_index], world_minor[splat_index], world_minor[splat_index]), dtype=np.float32)[axis_order]
        splats[splat_index, 3:6] = np.log(scales)
    splats[:, 10:13] = colors
    splats[:, 13] = alpha
    return torch.tensor(splats, device=device, dtype=torch.float32)


def _select_pixel_subset(image_size: tuple[int, int]) -> np.ndarray:
    width, height = map(int, image_size)
    pixel_count = width * height
    sample_count = min(_PIXEL_SUBSET_COUNT, pixel_count)
    seed = _PIXEL_SUBSET_SEED + width * 1009 + height * 9176
    return np.random.default_rng(seed).permutation(pixel_count)[:sample_count].astype(np.int64, copy=False)


def _make_sparse_grad_output(image_size: tuple[int, int], pixel_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x = np.broadcast_to(xs[None, :], (height, width))
    grid_y = np.broadcast_to(ys[:, None], (height, width))
    weight_map = 1.0 + 0.35 * grid_x - 0.2 * grid_y + 0.15 * grid_x * grid_y + 0.1 * (grid_x * grid_x + grid_y * grid_y)
    sampled_grad = np.ascontiguousarray(weight_map.reshape(-1)[pixel_indices, None] * _GRAD_CHANNEL_WEIGHTS[None, :])
    grad_output = np.zeros((height * width, 4), dtype=np.float32)
    grad_output[pixel_indices] = sampled_grad
    return grad_output.reshape(height, width, 4), sampled_grad


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


def _quat_rotate_torch(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    qv = q[..., 1:4]
    return v + 2.0 * torch.cross(torch.cross(v, qv.expand_as(v), dim=-1) + q[..., 0:1] * v, qv.expand_as(v), dim=-1)


def _undistort_normalized(uv_distorted: torch.Tensor, k1: float, k2: float) -> torch.Tensor:
    if abs(k1) <= _TORCH_GT_SMALL_VALUE and abs(k2) <= _TORCH_GT_SMALL_VALUE:
        return uv_distorted
    radius_distorted = torch.linalg.norm(uv_distorted, dim=-1, keepdim=True)
    radius = radius_distorted.clone()
    for _ in range(6):
        r2 = radius * radius
        r4 = r2 * r2
        deriv = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4
        safe = deriv.abs() > _TORCH_GT_SMALL_VALUE
        next_radius = radius - (radius * (1.0 + k1 * r2 + k2 * r4) - radius_distorted) / torch.where(safe, deriv, torch.ones_like(deriv))
        radius = torch.where(safe & torch.isfinite(next_radius) & (next_radius >= 0.0), next_radius, radius)
    return uv_distorted * (radius / torch.clamp(radius_distorted, min=_TORCH_GT_SMALL_VALUE))


def _screen_to_world_rays(camera: torch.Tensor, image_size: tuple[int, int], pixel_indices: torch.Tensor) -> torch.Tensor:
    width, height = image_size
    xs = torch.remainder(pixel_indices, width).to(dtype=torch.float32)
    ys = torch.div(pixel_indices, width, rounding_mode="floor").to(dtype=torch.float32)
    sample_positions = torch.stack((xs + 0.5, ys + 0.5), dim=-1)
    uv_distorted = (sample_positions - camera[9:11]) / torch.clamp(camera[7:9], min=_TORCH_GT_SMALL_VALUE)
    uv = _undistort_normalized(uv_distorted, float(camera[13].item()), float(camera[14].item()))
    camera_rays = torch.cat((uv, torch.ones_like(uv[..., :1])), dim=-1)
    camera_rays = camera_rays / torch.clamp(torch.linalg.norm(camera_rays, dim=-1, keepdim=True), min=_TORCH_GT_SMALL_VALUE)
    return camera_rays @ _camera_basis(camera[0:4]).T


def _ground_truth_render_torch(
    splats: torch.Tensor,
    camera: torch.Tensor,
    image_size: tuple[int, int],
    pixel_indices: torch.Tensor,
    has_tile_range: torch.Tensor,
    *,
    background: tuple[float, float, float],
    radius_scale: float,
    alpha_cutoff: float,
    transmittance_threshold: float,
) -> torch.Tensor:
    cam_basis = _camera_basis(camera[0:4])
    cam_pos = -cam_basis.T @ camera[4:7]
    depths = (splats[:, 0:3] - cam_pos[None, :]) @ cam_basis[2]
    sorted_splats = splats.index_select(0, torch.argsort(depths.detach(), stable=True))
    rays = _screen_to_world_rays(camera, image_size, pixel_indices).reshape(-1, 3)
    accum = torch.zeros((rays.shape[0], 3), device=splats.device, dtype=torch.float32)
    trans = torch.ones((rays.shape[0],), device=splats.device, dtype=torch.float32)
    background_t = torch.tensor(background, device=splats.device, dtype=torch.float32)
    exponent_scale = 0.5 * _GAUSSIAN_SUPPORT_SIGMA_RADIUS * _GAUSSIAN_SUPPORT_SIGMA_RADIUS
    for splat in sorted_splats:
        scale = torch.clamp(torch.exp(splat[3:6]) * float(radius_scale * _GAUSSIAN_SUPPORT_SIGMA_RADIUS), min=_TORCH_GT_SMALL_VALUE)
        opacity = torch.clamp(splat[13], GaussianRenderer._OPACITY_EPS, 1.0 - GaussianRenderer._OPACITY_EPS)
        ro_local = _quat_rotate_torch((cam_pos - splat[0:3]).unsqueeze(0), splat[6:10].unsqueeze(0))[0] / scale
        ray_local = _quat_rotate_torch(rays, splat[6:10].unsqueeze(0)) / scale
        denom = torch.sum(ray_local * ray_local, dim=1)
        numer = torch.sum(ray_local * ro_local[None, :], dim=1)
        t_closest = -numer / torch.clamp(denom, min=1e-10)
        closest = ro_local[None, :] + ray_local * t_closest[:, None]
        rho2 = torch.clamp(torch.sum(closest * closest, dim=1), min=0.0)
        alpha = torch.where(
            (denom > 1e-10) & (t_closest > 0.0),
            opacity * torch.exp(-exponent_scale * rho2),
            torch.zeros_like(rho2),
        )
        alpha = torch.where(alpha >= alpha_cutoff, alpha, torch.zeros_like(alpha))
        contribution = trans * alpha
        accum = accum + contribution[:, None] * splat[10:13][None, :]
        trans = trans * (1.0 - alpha)
        if bool(torch.all(trans < transmittance_threshold)):
            break
    final_color = accum + trans[:, None] * background_t[None, :]
    output_alpha = torch.where(has_tile_range, 1.0 - trans, torch.ones_like(trans))
    return torch.cat((torch.pow(torch.clamp(final_color, min=0.0), 2.2), output_alpha[:, None]), dim=1)


def _reference_render_and_grad(
    splats: torch.Tensor,
    camera_params: torch.Tensor,
    settings: TorchGaussianRenderSettings,
    pixel_indices: np.ndarray,
    sampled_grad_output: np.ndarray,
    has_tile_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    splats_t = splats.detach().clone().requires_grad_(True)
    pixel_indices_t = torch.tensor(pixel_indices, dtype=torch.long, device=splats.device)
    sampled_grad_t = torch.tensor(sampled_grad_output, dtype=torch.float32, device=splats.device)
    has_tile_range_t = torch.tensor(has_tile_range, dtype=torch.bool, device=splats.device)
    image = _ground_truth_render_torch(
        splats_t,
        camera_params.detach(),
        (settings.width, settings.height),
        pixel_indices_t,
        has_tile_range_t,
        background=settings.background,
        radius_scale=settings.radius_scale,
        alpha_cutoff=settings.alpha_cutoff,
        transmittance_threshold=settings.transmittance_threshold,
    )
    torch.autograd.backward(image, grad_tensors=sampled_grad_t)
    return image.detach().cpu().numpy().astype(np.float32, copy=False), splats_t.grad.detach().cpu().numpy().astype(np.float32, copy=False)


def _gradient_metrics(renderer_grad: np.ndarray, reference_grad: np.ndarray) -> tuple[float, float]:
    renderer_flat = renderer_grad.reshape(-1).astype(np.float64, copy=False)
    reference_flat = reference_grad.reshape(-1).astype(np.float64, copy=False)
    renderer_norm = float(np.linalg.norm(renderer_flat))
    reference_norm = float(np.linalg.norm(reference_flat))
    cosine = float(np.dot(renderer_flat, reference_flat) / max(renderer_norm * reference_norm, 1e-12))
    relative_l2 = float(np.linalg.norm(renderer_flat - reference_flat) / max(reference_norm, 1e-12))
    return cosine, relative_l2


def _sampled_tile_activity(
    splats: torch.Tensor,
    camera_params: torch.Tensor,
    settings: TorchGaussianRenderSettings,
    renderer: GaussianRenderer,
    pixel_indices: np.ndarray,
) -> np.ndarray:
    scene = _scene_from_splats(splats)
    camera = _camera_from_tensor(camera_params)
    tile_ranges = np.asarray(renderer.debug_pipeline_data(scene, camera)["tile_ranges"], dtype=np.uint32)
    pixel_indices = np.asarray(pixel_indices, dtype=np.int64)
    x = pixel_indices % settings.width
    y = pixel_indices // settings.width
    tile_x = x // renderer.tile_size
    tile_y = y // renderer.tile_size
    tile_id = tile_y * renderer.tile_width + tile_x
    return np.asarray(tile_ranges[tile_id, 1] > tile_ranges[tile_id, 0], dtype=np.bool_)


def _camera_from_tensor(camera_params: torch.Tensor) -> Camera:
    values = camera_params.detach().cpu().numpy()
    return Camera.from_colmap(
        q_wxyz=values[0:4],
        t_xyz=values[4:7],
        fx=float(values[7]),
        fy=float(values[8]),
        cx=float(values[9]),
        cy=float(values[10]),
        distortion_k1=float(values[13]),
        distortion_k2=float(values[14]),
        near=float(values[11]),
        far=float(values[12]),
    )


def _scene_from_splats(splats: torch.Tensor) -> GaussianScene:
    values = splats.detach().cpu().numpy()
    return GaussianScene(
        positions=np.asarray(values[:, 0:3], dtype=np.float32),
        scales=np.asarray(values[:, 3:6], dtype=np.float32),
        rotations=np.asarray(values[:, 6:10], dtype=np.float32),
        opacities=np.asarray(values[:, 13], dtype=np.float32),
        colors=np.asarray(values[:, 10:13], dtype=np.float32),
        sh_coeffs=np.zeros((values.shape[0], 1, 3), dtype=np.float32),
    )


def _public_reference_grads(splats: torch.Tensor, grad_groups: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(splats.detach().cpu().numpy(), dtype=np.float32)
    count = values.shape[0]
    packed = np.zeros((count, 14), dtype=np.float32)
    packed[:, 0:3] = np.asarray(grad_groups["grad_positions"][:, 0:3], dtype=np.float32)
    packed[:, 3:6] = np.asarray(grad_groups["grad_scales"][:, 0:3], dtype=np.float32)
    packed[:, 6:10] = np.asarray(grad_groups["grad_rotations"][:, 0:4], dtype=np.float32)
    packed[:, 10:13] = np.asarray(grad_groups["grad_color_alpha"][:, 0:3], dtype=np.float32)
    alpha = np.clip(values[:, 13], GaussianRenderer._OPACITY_EPS, 1.0 - GaussianRenderer._OPACITY_EPS)
    packed[:, 13] = np.asarray(grad_groups["grad_color_alpha"][:, 3], dtype=np.float32) / (alpha * (1.0 - alpha))
    return packed


@pytest.fixture(scope="module")
def torch_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch is required for torch renderer tests.")
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def test_torch_renderer_forward_smoke(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=64, height=64, list_capacity_multiplier=16)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=12)
    camera_params = _make_camera_params(torch_cuda_device)

    image = render_gaussian_splats_torch(splats, camera_params, settings, context)

    assert image.is_cuda
    assert image.dtype == torch.float32
    assert tuple(image.shape) == (64, 64, 4)
    assert torch.isfinite(image).all()


def test_torch_renderer_backward_smoke(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=64, height=64, list_capacity_multiplier=16)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=10).requires_grad_(True)
    camera_params = _make_camera_params(torch_cuda_device)

    loss = render_gaussian_splats_torch(splats, camera_params, settings, context).sum()
    loss.backward()

    assert splats.grad is not None
    assert splats.grad.shape == splats.shape
    assert torch.isfinite(splats.grad).all()
    assert torch.count_nonzero(splats.grad).item() > 0


def test_torch_renderer_output_matches_renderer_texture(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=64, height=64, list_capacity_multiplier=16)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=9)
    camera_params = _make_camera_params(torch_cuda_device)

    image = render_gaussian_splats_torch(splats, camera_params, settings, context)
    reference = np.asarray(context.renderer.output_texture.to_numpy(), dtype=np.float32)

    np.testing.assert_allclose(image.detach().cpu().numpy(), reference, rtol=0.0, atol=0.0)


def test_torch_renderer_gradients_match_reference_backward(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=64, height=64, list_capacity_multiplier=16)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=7).requires_grad_(True)
    camera_params = _make_camera_params(torch_cuda_device)

    image = render_gaussian_splats_torch(splats, camera_params, settings, context)
    loss = 0.5 * torch.square(image).sum()
    loss.backward()

    scene = _scene_from_splats(splats.detach())
    camera = _camera_from_tensor(camera_params)
    reference = context.renderer.debug_raster_backward_grads(scene, camera, background=settings.background)
    public_reference = _public_reference_grads(splats.detach(), reference)

    np.testing.assert_allclose(
        splats.grad.detach().cpu().numpy(),
        public_reference,
        rtol=2e-5,
        atol=3e-5,
    )


def test_torch_renderer_alpha_gradient_is_public_alpha_space(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=64, height=64, list_capacity_multiplier=16)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=6).requires_grad_(True)
    camera_params = _make_camera_params(torch_cuda_device)

    image = render_gaussian_splats_torch(splats, camera_params, settings, context)
    (0.5 * torch.square(image).sum()).backward()

    scene = _scene_from_splats(splats.detach())
    camera = _camera_from_tensor(camera_params)
    reference = context.renderer.debug_raster_backward_grads(scene, camera, background=settings.background)
    raw_opacity_grad = np.asarray(reference["grad_color_alpha"][:, 3], dtype=np.float32)
    alpha_grad = splats.grad.detach().cpu().numpy()[:, 13]

    assert not np.allclose(alpha_grad, raw_opacity_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(alpha_grad, _public_reference_grads(splats.detach(), reference)[:, 13], rtol=2e-5, atol=3e-5)


def test_torch_renderer_float_cached_grad_mode_preserves_small_high_res_l1_gradients(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(
        width=1024,
        height=1024,
        list_capacity_multiplier=16,
        cached_raster_grad_atomic_mode=GaussianRenderer.CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT,
    )
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=8).requires_grad_(True)
    camera_params = _make_camera_params(torch_cuda_device)

    loss = torch.mean(torch.abs(render_gaussian_splats_torch(splats, camera_params, settings, context)[..., :3]))
    loss.backward()
    grad = splats.grad.detach()

    assert float(torch.max(torch.abs(grad[:, 0:3])).item()) > 0.0
    assert float(torch.max(torch.abs(grad[:, 3:6])).item()) > 0.0
    assert float(torch.max(torch.abs(grad[:, 10:13])).item()) > 0.0
    assert float(torch.max(torch.abs(grad[:, 13])).item()) > 0.0


def test_torch_renderer_rejects_invalid_inputs(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=32, height=32)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    camera_params = _make_camera_params(torch_cuda_device)

    with pytest.raises(ValueError, match="CUDA"):
        render_gaussian_splats_torch(torch.zeros((1, 14), dtype=torch.float32), camera_params, settings, context)

    with pytest.raises(TypeError, match="float32"):
        render_gaussian_splats_torch(torch.zeros((1, 14), device=torch_cuda_device, dtype=torch.float64), camera_params, settings, context)

    with pytest.raises(ValueError, match="shape"):
        render_gaussian_splats_torch(torch.zeros((1, 13), device=torch_cuda_device, dtype=torch.float32), camera_params, settings, context)

    with pytest.raises(ValueError, match="shape"):
        render_gaussian_splats_torch(_make_splats(torch_cuda_device, count=1), torch.zeros((14,), device=torch_cuda_device, dtype=torch.float32), settings, context)

    with pytest.raises(ValueError, match="not supported"):
        render_gaussian_splats_torch(
            _make_splats(torch_cuda_device, count=1),
            _make_camera_params(torch_cuda_device).requires_grad_(True),
            settings,
            context,
        )


def test_torch_renderer_context_reuses_renderer_across_scene_sizes(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=48, height=48, list_capacity_multiplier=8)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    camera_params = _make_camera_params(torch_cuda_device)

    image0 = render_gaussian_splats_torch(_make_splats(torch_cuda_device, count=2), camera_params, settings, context)
    renderer_id = id(context.renderer)
    image1 = render_gaussian_splats_torch(_make_splats(torch_cuda_device, count=11), camera_params, settings, context)
    image2 = render_gaussian_splats_torch(_make_splats(torch_cuda_device, count=5), camera_params, settings, context)

    assert id(context.renderer) == renderer_id
    assert torch.isfinite(image0).all()
    assert torch.isfinite(image1).all()
    assert torch.isfinite(image2).all()


def test_torch_renderer_cuda_device_smoke(torch_cuda_device: torch.device):
    torch.cuda.init()
    torch.cuda.current_device()
    torch.cuda.current_stream()
    with torch.device(torch_cuda_device):
        handles = spy.get_cuda_current_context_native_handles()
    device = spy.Device(
        type=spy.DeviceType.cuda,
        compiler_options={"include_paths": [str(path) for path in SHADER_INCLUDE_PATHS]},
        enable_debug_layers=False,
        enable_rhi_validation=False,
        enable_cuda_interop=False,
        enable_print=False,
        enable_hot_reload=False,
        enable_compilation_reports=False,
        existing_device_handles=handles,
    )
    buffer = device.create_buffer(size=16, usage=spy.BufferUsage.shared | spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)
    src = torch.tensor([1.0, 2.0, 3.0, 4.0], device=torch_cuda_device, dtype=torch.float32)
    dst = torch.zeros_like(src)

    spy.copy_torch_tensor_to_buffer(src, buffer)
    device.sync_to_cuda()
    spy.copy_buffer_to_torch_tensor(buffer, dst)

    assert torch.allclose(src, dst)


def test_torch_renderer_matches_torch_ground_truth_and_gradients(torch_cuda_device: torch.device):
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_scene_splats(torch_cuda_device)

    for spec in _CAMERA_SPECS:
        image_size = tuple(spec["image_size"])
        settings = TorchGaussianRenderSettings(
            width=image_size[0],
            height=image_size[1],
            list_capacity_multiplier=64,
            cached_raster_grad_atomic_mode=GaussianRenderer.CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT,
        )
        camera_params = _make_camera_spec_tensor(spec, torch_cuda_device)
        pixel_indices = _select_pixel_subset(image_size)
        grad_output, sampled_grad_output = _make_sparse_grad_output(image_size, pixel_indices)

        renderer_splats = splats.detach().clone().requires_grad_(True)
        image = render_gaussian_splats_torch(renderer_splats, camera_params, settings, context)
        image.backward(torch.tensor(grad_output, dtype=torch.float32, device=torch_cuda_device))
        has_tile_range = _sampled_tile_activity(splats, camera_params, settings, context.renderer, pixel_indices)
        ref_image, ref_grads = _reference_render_and_grad(splats, camera_params, settings, pixel_indices, sampled_grad_output, has_tile_range)

        sampled_image = image.detach().cpu().numpy().reshape(-1, 4)[pixel_indices]
        renderer_grads = renderer_splats.grad.detach().cpu().numpy()
        assert np.isfinite(sampled_image).all()
        assert np.isfinite(ref_image).all()
        assert np.isfinite(renderer_grads).all()
        assert np.isfinite(ref_grads).all()

        rgb_mae = float(np.abs(sampled_image[:, :3] - ref_image[:, :3]).mean())
        alpha_error = np.abs(sampled_image[:, 3] - ref_image[:, 3])
        cosine, relative_l2 = _gradient_metrics(renderer_grads, ref_grads)

        assert rgb_mae <= _RGB_MEAN_ABS_TOL, f"{image_size} rgb mean abs error {rgb_mae}"
        assert float(alpha_error.mean()) <= _ALPHA_MEAN_ABS_TOL, f"{image_size} alpha mean abs error {float(alpha_error.mean())}"
        assert float(np.quantile(alpha_error.reshape(-1), 0.995)) <= _ALPHA_P995_TOL, f"{image_size} alpha p99.5 error too high"
        assert cosine >= _GRADIENT_COSINE_MIN, f"{image_size} gradient cosine {cosine}"
        assert relative_l2 <= _GRADIENT_REL_L2_MAX, f"{image_size} gradient relative l2 {relative_l2}"

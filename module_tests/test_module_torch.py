from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import slangpy as spy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module import SplattingContext as TorchSplattingContext, render_gaussian_splats
from module.splatting import SplattingContext as CoreSplattingContext

ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / "module" / "shaders"
TEST_SCENE_IMAGE_SIZE = (512, 512)
DEFAULT_CAMERA_POSITION_Z = 3.0
LINE_DEPTH_START = 4.2
LINE_DEPTH_END = 18.0
LINE_SCREEN_SCALE_MIN = 0.005
LINE_SCREEN_SCALE_MAX = 0.08
LINE_SCREEN_MARGIN_PAD = 6.0
LINE_OPACITY_MAX = 0.95

SCENE_SPLAT_COUNT = 97
SCENE_SEED = 17
ALPHA_CUTOFF = 1.0 / 255.0
TRANS_THRESHOLD = 0.005
MAX_ANISOTROPY = 12.0
SMALL_VALUE = 1e-6
ALPHA_EPS = 1e-6
GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0
PIXEL_SUBSET_COUNT = 65_536
PIXEL_SUBSET_SEED = 12345
SCENE_CENTER_SPREAD = 0.06
SCENE_MINOR_AXIS_PIXELS = (16.0, 22.0)
SCENE_DEPTH_RANGE = (4.8, 8.2)
SCENE_ALPHA_RANGE = (0.55, 0.85)
SCENE_COLOR_RANGE = (0.2, 1.0)
MIN_VISIBLE_FRACTION = 0.9
MAX_BBOX_CUTOFF = 0.5
MIN_SCREEN_OVERLAP = 0.5
MIN_MEAN_ELLIPSE_AREA = 0.05
MAX_MEAN_ELLIPSE_AREA = 0.2
RENDER_MEAN_ABS_TOL = 6e-3
ALPHA_MEAN_ABS_TOL = 9e-3
DEPTH_MEAN_ABS_TOL = 7e-3
ALPHA_P999_TOL = 4.5e-2
GRADIENT_COSINE_MIN = 0.995
GRADIENT_REL_L2_MAX = 0.08
PROJECTION_ALPHA_TOL = 5e-4
PROJECTION_SAMPLE_COUNT = 12
GRAD_CHANNEL_WEIGHTS = np.array([0.7, -0.25, 0.5, 1.1, -0.4], dtype=np.float32)
EXTREME_IMAGE_SIZE = (11, 109)
TALL_IMAGE_SIZE = (53, 1447)
EXTREME_RENDER_MEAN_ABS_TOL = 1.2e-2
EXTREME_ALPHA_MEAN_ABS_TOL = 4.0e-2
EXTREME_DEPTH_MEAN_ABS_TOL = 3.0e-2
EXTREME_ALPHA_P999_TOL = 4.3e-1
EXTREME_GRADIENT_COSINE_MIN = 0.98
EXTREME_GRADIENT_REL_L2_MAX = 0.2
TALL_ALPHA_MEAN_ABS_TOL = 9e-3
TALL_ALPHA_P999_TOL = 1e-1
LOCKED_LINE_SPLAT_COUNT = 64
LOCKED_LINE_SCALE_MULTIPLIER = 2.460375
MIN_LOCKED_LINE_ALPHA_MEAN = 0.7
DESCENT_ITERATION_COUNT = 15
DESCENT_STEP_CANDIDATES = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
DESCENT_MIN_IMPROVEMENT = 1e-9
DESCENT_PASSING_RELATIVE_IMPROVEMENT = 0.75

CAMERA_SPECS = (
    {"image_size": (11, 109), "fov_y_degrees": 20.0, "focal_scale": (0.06, 0.4), "distortion": (0.0, 0.0)},
    {"image_size": (53, 1447), "fov_y_degrees": 34.0, "focal_scale": (0.04, 0.8), "distortion": (0.0, 0.0)},
    {"image_size": (3191, 271), "fov_y_degrees": 28.0, "focal_scale": (8.0, 0.65), "distortion": (0.0, 0.0)},
    {"image_size": (4157, 3011), "fov_y_degrees": 18.0, "focal_scale": (1.0, 0.45), "distortion": (0.0, 0.0)},
)


def _make_device(backend_name: str) -> spy.Device:
    return spy.create_device(type=getattr(spy.DeviceType, backend_name), include_paths=[SHADERS], enable_cuda_interop=False, enable_hot_reload=False)


def _make_context(backend_name: str) -> CoreSplattingContext:
    context = CoreSplattingContext(device=_make_device(backend_name))
    context.alpha_cutoff = ALPHA_CUTOFF
    context.trans_threshold = TRANS_THRESHOLD
    context.max_anisotropy = MAX_ANISOTROPY
    return context


def _make_torch_context(alpha_cutoff: float) -> TorchSplattingContext:
    context = TorchSplattingContext()
    context.alpha_cutoff = float(alpha_cutoff)
    return context


def _make_reference_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_forward_camera(
    image_size: tuple[int, int],
    camera_position_z: float = DEFAULT_CAMERA_POSITION_Z,
    fov_y_degrees: float = 28.0,
    near: float = 0.1,
    far: float = 40.0,
    device: str = "cpu",
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
            -camera_position_z,
            focal,
            focal,
            width * 0.5,
            height * 0.5,
            near,
            far,
            0.0,
            0.0,
        ),
        dtype=torch.float32,
        device=device,
    )


def _random_quaternions(count: int, generator: torch.Generator) -> torch.Tensor:
    quat = torch.randn((4, count), generator=generator, dtype=torch.float32)
    return quat / torch.clamp(torch.linalg.norm(quat, dim=0, keepdim=True), min=1e-8)


def _select_pixel_subset(image_size: tuple[int, int]) -> np.ndarray:
    width, height = map(int, image_size)
    pixel_count = width * height
    sample_count = min(PIXEL_SUBSET_COUNT, pixel_count)
    seed = PIXEL_SUBSET_SEED + width * 1009 + height * 9176
    return np.random.default_rng(seed).permutation(pixel_count)[:sample_count].astype(np.int64, copy=False)


def _make_scene_splats() -> np.ndarray:
    scene_camera = _make_forward_camera(TEST_SCENE_IMAGE_SIZE, device="cpu").detach().cpu().numpy().astype(np.float32, copy=False)
    width, height = map(float, TEST_SCENE_IMAGE_SIZE)
    focal = float(scene_camera[7])
    cam_z = -float(scene_camera[6])
    layout_rng = np.random.default_rng(SCENE_SEED)
    attribute_rng = np.random.default_rng(SCENE_SEED)
    depths = np.linspace(SCENE_DEPTH_RANGE[0], SCENE_DEPTH_RANGE[1], SCENE_SPLAT_COUNT, dtype=np.float32)
    depth_from_camera = depths - cam_z
    screen_x = layout_rng.uniform(0.5 - SCENE_CENTER_SPREAD, 0.5 + SCENE_CENTER_SPREAD, size=SCENE_SPLAT_COUNT).astype(np.float32) * width
    screen_y = layout_rng.uniform(0.5 - SCENE_CENTER_SPREAD, 0.5 + SCENE_CENTER_SPREAD, size=SCENE_SPLAT_COUNT).astype(np.float32) * height
    ratios = np.exp(attribute_rng.uniform(0.0, math.log(MAX_ANISOTROPY), size=SCENE_SPLAT_COUNT)).astype(np.float32)
    ratios[0] = np.float32(1.0)
    ratios[1] = np.float32(MAX_ANISOTROPY)
    angles = attribute_rng.uniform(0.0, 2.0 * math.pi, size=SCENE_SPLAT_COUNT).astype(np.float32)
    colors = attribute_rng.uniform(SCENE_COLOR_RANGE[0], SCENE_COLOR_RANGE[1], size=(3, SCENE_SPLAT_COUNT)).astype(np.float32)
    alpha = attribute_rng.uniform(SCENE_ALPHA_RANGE[0], SCENE_ALPHA_RANGE[1], size=SCENE_SPLAT_COUNT).astype(np.float32)
    minor_px = layout_rng.uniform(SCENE_MINOR_AXIS_PIXELS[0], SCENE_MINOR_AXIS_PIXELS[1], size=SCENE_SPLAT_COUNT).astype(np.float32)
    world_minor = minor_px * depth_from_camera / focal
    world_major = world_minor * ratios
    axis_shifts = attribute_rng.permutation(SCENE_SPLAT_COUNT) % 3

    splats = np.zeros((14, SCENE_SPLAT_COUNT), dtype=np.float32)
    splats[0] = (screen_x - 0.5 * width) * depth_from_camera / focal
    splats[1] = (screen_y - 0.5 * height) * depth_from_camera / focal
    splats[2] = depths
    for splat_index in range(SCENE_SPLAT_COUNT):
        half_angle = 0.5 * float(angles[splat_index])
        splats[6:10, splat_index] = np.array((math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)), dtype=np.float32)
        axis_order = [0, 1, 2]
        shift = int(axis_shifts[splat_index])
        axis_order = axis_order[shift:] + axis_order[:shift]
        scales = np.array((world_major[splat_index], world_minor[splat_index], world_minor[splat_index]), dtype=np.float32)[axis_order]
        splats[3:6, splat_index] = np.log(scales)
    splats[10:13] = colors
    splats[13] = alpha
    return np.ascontiguousarray(splats)


def _make_locked_line_scene() -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], float]:
    generator = torch.Generator(device="cpu").manual_seed(SCENE_SEED)
    camera = _make_forward_camera(TEST_SCENE_IMAGE_SIZE, device="cpu")
    width, height = TEST_SCENE_IMAGE_SIZE
    min_dim = float(min(width, height))
    focal = float(camera[7].item())
    camera_z = -float(camera[6].item())
    depths = torch.linspace(LINE_DEPTH_START, LINE_DEPTH_END, LOCKED_LINE_SPLAT_COUNT, dtype=torch.float32)
    camera_depths = depths - camera_z

    screen_fraction = torch.rand((LOCKED_LINE_SPLAT_COUNT,), generator=generator, dtype=torch.float32)
    screen_fraction = screen_fraction * (LINE_SCREEN_SCALE_MAX - LINE_SCREEN_SCALE_MIN) + LINE_SCREEN_SCALE_MIN
    screen_radius_px = screen_fraction * min_dim
    world_radius = screen_radius_px * camera_depths / focal
    anisotropy = torch.rand((LOCKED_LINE_SPLAT_COUNT, 3), generator=generator, dtype=torch.float32) * 0.35 + 0.825
    base_scale = world_radius[:, None] * anisotropy

    xs = torch.empty((LOCKED_LINE_SPLAT_COUNT,), dtype=torch.float32)
    ys = torch.empty((LOCKED_LINE_SPLAT_COUNT,), dtype=torch.float32)
    for idx in range(LOCKED_LINE_SPLAT_COUNT):
        radius = float(screen_radius_px[idx].item())
        margin_x = min(radius + LINE_SCREEN_MARGIN_PAD, 0.5 * float(width) - 1.0)
        margin_y = min(radius + LINE_SCREEN_MARGIN_PAD, 0.5 * float(height) - 1.0)
        x_min, x_max = margin_x, float(width) - margin_x
        y_min, y_max = margin_y, float(height) - margin_y
        xs[idx] = 0.5 * float(width) if x_max <= x_min else torch.rand((1,), generator=generator, dtype=torch.float32).item() * (x_max - x_min) + x_min
        ys[idx] = 0.5 * float(height) if y_max <= y_min else torch.rand((1,), generator=generator, dtype=torch.float32).item() * (y_max - y_min) + y_min

    splats = torch.zeros((LOCKED_LINE_SPLAT_COUNT, 14), dtype=torch.float32)
    splats[:, 0] = (xs - 0.5 * float(width)) * camera_depths / focal
    splats[:, 1] = (ys - 0.5 * float(height)) * camera_depths / focal
    splats[:, 2] = depths
    splats[:, 3:6] = torch.log(torch.clamp(base_scale, min=1e-4))
    splats[:, 6:10] = _random_quaternions(LOCKED_LINE_SPLAT_COUNT, generator).mT
    splats[:, 10:13] = torch.rand((LOCKED_LINE_SPLAT_COUNT, 3), generator=generator, dtype=torch.float32) * 0.75 + 0.15
    splats[:, 13] = torch.rand((LOCKED_LINE_SPLAT_COUNT,), generator=generator, dtype=torch.float32) * LINE_OPACITY_MAX
    splats = splats.mT.contiguous().cuda()
    splats[3:6] += math.log(LOCKED_LINE_SCALE_MULTIPLIER)
    return splats, camera.to(device=splats.device), TEST_SCENE_IMAGE_SIZE, ALPHA_CUTOFF


def _make_camera(spec: dict[str, object], distortion_override: tuple[float, float] | None = None) -> np.ndarray:
    image_size = tuple(spec["image_size"])
    camera = _make_forward_camera(image_size, fov_y_degrees=float(spec["fov_y_degrees"]), device="cpu").detach().cpu().numpy().astype(np.float32, copy=True)
    fx_scale, fy_scale = spec["focal_scale"]
    camera[7] *= np.float32(fx_scale)
    camera[8] *= np.float32(fy_scale)
    distortion = distortion_override if distortion_override is not None else spec["distortion"]
    camera[13] = np.float32(distortion[0])
    camera[14] = np.float32(distortion[1])
    return camera


def _camera_basis_numpy(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


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


def _camera_dict(camera: np.ndarray, image_size: tuple[int, int]) -> dict[str, object]:
    rotation = _camera_basis_numpy(camera[0:4])
    cam_pos = -rotation.T @ camera[4:7]
    return {
        "camPos": spy.float3(*cam_pos.tolist()),
        "camBasis": spy.float3x3(rotation),
        "focalPixels": spy.float2(float(camera[7]), float(camera[8])),
        "principalPoint": spy.float2(float(camera[9]), float(camera[10])),
        "viewport": spy.float2(*map(float, image_size)),
        "nearDepth": float(camera[11]),
        "farDepth": float(camera[12]),
        "k1": float(camera[13]),
        "k2": float(camera[14]),
    }


def _pack_params(splats: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(splats.copy())
    packed[13] = np.log(np.clip(packed[13], ALPHA_EPS, 1.0 - ALPHA_EPS) / np.clip(1.0 - packed[13], ALPHA_EPS, 1.0))
    return packed.reshape(-1)


def _load_public_splats(context: CoreSplattingContext, splats: np.ndarray, image_size: tuple[int, int], background: tuple[float, float, float]) -> None:
    context.prepare(splats.shape[1], image_size, background)
    context.scene["g_Params"].copy_from_numpy(_pack_params(splats))
    context.device.sync_to_cuda()


def _render_scene(context: CoreSplattingContext, splats: np.ndarray, camera: np.ndarray, image_size: tuple[int, int], background: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    _load_public_splats(context, splats, image_size, background)
    context.render(_camera_dict(camera, image_size), splats.shape[1])
    color = np.asarray(context.frame["g_Output"].to_numpy(), dtype=np.float32).copy()
    depth_ratio = np.asarray(context.frame["g_OutputDepth"].to_numpy(), dtype=np.float32).copy()[..., None]
    return np.concatenate((color, depth_ratio), axis=-1)


def _backward_scene(
    context: CoreSplattingContext,
    splats: np.ndarray,
    camera: np.ndarray,
    image_size: tuple[int, int],
    grad_output: np.ndarray,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    _load_public_splats(context, splats, image_size, background)
    context.render(_camera_dict(camera, image_size), splats.shape[1])
    context.frame["g_OutputGrad"].copy_from_numpy(np.ascontiguousarray(grad_output[..., :4].astype(np.float32, copy=False)))
    context.frame["g_OutputDepthGrad"].copy_from_numpy(np.ascontiguousarray(grad_output[..., 4].astype(np.float32, copy=False)))
    context.device.sync_to_cuda()
    grads = np.asarray(context.backward(_camera_dict(camera, image_size), splats.shape[1]).to_numpy(), dtype=np.float32)[: splats.shape[1] * 14].copy()
    grads = grads.reshape(splats.shape[1], 14).T
    grads[13] /= np.clip(splats[13] * (1.0 - splats[13]), ALPHA_EPS, None)
    return grads


def _project_scene(context: CoreSplattingContext, splats: np.ndarray, camera: np.ndarray, image_size: tuple[int, int]) -> dict[str, np.ndarray]:
    _load_public_splats(context, splats, image_size, (0.0, 0.0, 0.0))
    total_scanlines = context.project(_camera_dict(camera, image_size), splats.shape[1])
    projection = np.asarray(context._view(context.scene_views["g_ProjectionState"].tensor, spy.float4, (splats.shape[1], 2)).to_numpy(), dtype=np.float32)[: splats.shape[1]].copy()
    tile_counts = np.zeros((splats.shape[1],), dtype=np.int64)
    if total_scanlines > 0:
        scanline_counts = np.asarray(context.scanlines["g_ScanlineTileCounts"].to_numpy(), dtype=np.uint32).reshape(-1)[:total_scanlines].astype(np.int64, copy=False)
        scanline_splats = np.asarray(context.scanlines["g_ScanlineEntryData"].to_numpy(), dtype=np.uint32)[:total_scanlines, 0].astype(np.int64, copy=False)
        np.add.at(tile_counts, scanline_splats, scanline_counts)
    return {"projection": projection, "tile_counts": tile_counts}


def _undistort_normalized(uv_distorted: torch.Tensor, k1: float, k2: float) -> torch.Tensor:
    if abs(k1) <= SMALL_VALUE and abs(k2) <= SMALL_VALUE:
        return uv_distorted
    radius_distorted = torch.linalg.norm(uv_distorted, dim=-1, keepdim=True)
    radius = radius_distorted.clone()
    for _ in range(6):
        r2 = radius * radius
        r4 = r2 * r2
        deriv = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4
        safe = deriv.abs() > SMALL_VALUE
        next_radius = radius - (radius * (1.0 + k1 * r2 + k2 * r4) - radius_distorted) / torch.where(safe, deriv, torch.ones_like(deriv))
        radius = torch.where(safe & torch.isfinite(next_radius) & (next_radius >= 0.0), next_radius, radius)
    return uv_distorted * (radius / torch.clamp(radius_distorted, min=SMALL_VALUE))


def _screen_to_world_rays(camera: torch.Tensor, image_size: tuple[int, int], pixel_indices: torch.Tensor | None = None) -> torch.Tensor:
    width, height = image_size
    if pixel_indices is None:
        ys = torch.arange(height, device=camera.device, dtype=torch.float32)
        xs = torch.arange(width, device=camera.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        sample_positions = torch.stack((xx + 0.5, yy + 0.5), dim=-1)
    else:
        pixel_indices = pixel_indices.to(device=camera.device, dtype=torch.long)
        xs = torch.remainder(pixel_indices, width).to(dtype=torch.float32)
        ys = torch.div(pixel_indices, width, rounding_mode="floor").to(dtype=torch.float32)
        sample_positions = torch.stack((xs + 0.5, ys + 0.5), dim=-1)
    uv_distorted = (sample_positions - camera[9:11]) / torch.clamp(camera[7:9], min=SMALL_VALUE)
    uv = _undistort_normalized(uv_distorted, float(camera[13].item()), float(camera[14].item()))
    camera_rays = torch.cat((uv, torch.ones_like(uv[..., :1])), dim=-1)
    camera_rays = camera_rays / torch.clamp(torch.linalg.norm(camera_rays, dim=-1, keepdim=True), min=SMALL_VALUE)
    return camera_rays @ _camera_basis(camera[0:4]).T


def _ground_truth_render_torch(
    splats: torch.Tensor,
    camera: torch.Tensor,
    image_size: tuple[int, int],
    pixel_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    width, height = image_size
    cam_basis = _camera_basis(camera[0:4])
    cam_pos = -cam_basis.T @ camera[4:7]
    distances = torch.linalg.norm(splats[0:3].T - cam_pos[None, :], dim=1)
    sorted_splats = splats.index_select(1, torch.argsort(distances, stable=True))
    rays = _screen_to_world_rays(camera, image_size, pixel_indices).reshape(-1, 3)
    accum = torch.zeros((rays.shape[0], 3), device=splats.device, dtype=torch.float32)
    trans = torch.ones((rays.shape[0],), device=splats.device, dtype=torch.float32)
    depth_weight = torch.zeros((rays.shape[0],), device=splats.device, dtype=torch.float32)
    depth_mean = torch.zeros((rays.shape[0],), device=splats.device, dtype=torch.float32)
    depth_m2 = torch.zeros((rays.shape[0],), device=splats.device, dtype=torch.float32)
    for i in range(sorted_splats.shape[1]):
        gaussian = sorted_splats[:, i]
        scale = torch.clamp(torch.exp(gaussian[3:6]) * GAUSSIAN_SUPPORT_SIGMA_RADIUS, min=SMALL_VALUE)
        scale = torch.maximum(scale, torch.max(scale) / MAX_ANISOTROPY)
        q = gaussian[6:10]
        qv = q[1:]
        ro = cam_pos - gaussian[0:3]
        ro_local = (ro + 2.0 * torch.cross(torch.cross(ro, qv, dim=0) + q[0] * ro, qv, dim=0)) / scale
        cross1 = torch.cross(rays, qv.expand_as(rays), dim=1)
        cross2 = torch.cross(cross1 + q[0] * rays, qv.expand_as(rays), dim=1)
        ray_local = (rays + 2.0 * cross2) / scale
        denom = torch.sum(ray_local * ray_local, dim=1)
        t_closest = -torch.sum(ray_local * ro_local[None, :], dim=1) / torch.clamp(denom, min=1e-10)
        rho2 = torch.clamp(torch.sum(ro_local[None, :] * ro_local[None, :], dim=1) - torch.square(torch.sum(ray_local * ro_local[None, :], dim=1)) / torch.clamp(denom, min=1e-10), min=0.0)
        alpha = torch.where(
            (denom > 1e-10) & (t_closest > 0.0),
            torch.clamp(gaussian[13], min=SMALL_VALUE, max=1.0 - SMALL_VALUE) * torch.exp(-0.5 * GAUSSIAN_SUPPORT_SIGMA_RADIUS * GAUSSIAN_SUPPORT_SIGMA_RADIUS * rho2),
            torch.zeros_like(rho2),
        )
        alpha = torch.where(alpha >= ALPHA_CUTOFF, alpha, torch.zeros_like(alpha))
        contribution = trans * alpha
        total_weight = depth_weight + contribution
        safe = contribution > SMALL_VALUE
        delta = t_closest - depth_mean
        depth_mean = torch.where(safe, depth_mean + torch.where(total_weight > SMALL_VALUE, (contribution / torch.clamp(total_weight, min=SMALL_VALUE)) * delta, torch.zeros_like(delta)), depth_mean)
        depth_m2 = torch.where(safe, depth_m2 + torch.where(total_weight > SMALL_VALUE, depth_weight * contribution * delta.square() / torch.clamp(total_weight, min=SMALL_VALUE), torch.zeros_like(delta)), depth_m2)
        depth_weight = torch.where(safe, total_weight, depth_weight)
        accum = accum + contribution[:, None] * gaussian[10:13][None, :]
        trans = trans * (1.0 - alpha)
        if bool(torch.all(trans < TRANS_THRESHOLD)):
            break
    color = torch.pow(torch.clamp(accum, min=0.0), 2.2)
    stable_variance = torch.clamp(depth_m2 / torch.clamp(depth_weight, min=SMALL_VALUE), min=SMALL_VALUE)
    depth_std = torch.where(depth_weight > SMALL_VALUE, torch.sqrt(stable_variance), torch.zeros_like(depth_weight))
    depth_ratio = torch.where(depth_weight > SMALL_VALUE, depth_std / torch.clamp(depth_mean.abs(), min=SMALL_VALUE), torch.zeros_like(depth_std))
    image = torch.cat((color, (1.0 - trans)[:, None], depth_ratio[:, None]), dim=1)
    if pixel_indices is None:
        return image.reshape(height, width, 5)
    return image


def _make_sparse_grad_output(image_size: tuple[int, int], pixel_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x = np.broadcast_to(xs[None, :], (height, width))
    grid_y = np.broadcast_to(ys[:, None], (height, width))
    weight_map = 1.0 + 0.35 * grid_x - 0.2 * grid_y + 0.15 * grid_x * grid_y + 0.1 * (grid_x * grid_x + grid_y * grid_y)
    sampled_grad = np.ascontiguousarray(weight_map.reshape(-1)[pixel_indices, None] * GRAD_CHANNEL_WEIGHTS[None, :])
    grad_output = np.zeros((height * width, 5), dtype=np.float32)
    grad_output[pixel_indices] = sampled_grad
    return grad_output.reshape(height, width, 5), sampled_grad


def _reference_render_and_grad(
    splats: np.ndarray,
    camera: np.ndarray,
    image_size: tuple[int, int],
    pixel_indices: np.ndarray,
    sampled_grad_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ref_device = _make_reference_device()
    splats_t = torch.tensor(splats, dtype=torch.float32, device=ref_device, requires_grad=True)
    camera_t = torch.tensor(camera, dtype=torch.float32, device=ref_device)
    pixel_indices_t = torch.tensor(pixel_indices, dtype=torch.long, device=ref_device)
    sampled_grad_t = torch.tensor(sampled_grad_output, dtype=torch.float32, device=ref_device)
    image = _ground_truth_render_torch(splats_t, camera_t, image_size, pixel_indices=pixel_indices_t)
    torch.autograd.backward(image, grad_tensors=sampled_grad_t)
    image_np = image.detach().cpu().numpy().astype(np.float32, copy=False)
    grad_np = splats_t.grad.detach().cpu().numpy().astype(np.float32, copy=False)
    del sampled_grad_t, pixel_indices_t, image, camera_t, splats_t
    if ref_device.type == "cuda":
        torch.cuda.empty_cache()
    return image_np, grad_np


def _gradient_metrics(renderer_grad: np.ndarray, reference_grad: np.ndarray) -> tuple[float, float]:
    renderer_flat = renderer_grad.reshape(-1).astype(np.float64, copy=False)
    reference_flat = reference_grad.reshape(-1).astype(np.float64, copy=False)
    renderer_norm = float(np.linalg.norm(renderer_flat))
    reference_norm = float(np.linalg.norm(reference_flat))
    cosine = float(np.dot(renderer_flat, reference_flat) / max(renderer_norm * reference_norm, 1e-12))
    relative_l2 = float(np.linalg.norm(renderer_flat - reference_flat) / max(reference_norm, 1e-12))
    return cosine, relative_l2


def _undistort_normalized_numpy(uv_distorted: np.ndarray, k1: float, k2: float) -> np.ndarray:
    if abs(k1) <= SMALL_VALUE and abs(k2) <= SMALL_VALUE:
        return uv_distorted
    radius_distorted = np.linalg.norm(uv_distorted, axis=-1, keepdims=True)
    radius = radius_distorted.copy()
    for _ in range(6):
        r2 = radius * radius
        r4 = r2 * r2
        deriv = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4
        safe = np.abs(deriv) > SMALL_VALUE
        next_radius = radius - (radius * (1.0 + k1 * r2 + k2 * r4) - radius_distorted) / np.where(safe, deriv, 1.0)
        radius = np.where(safe & np.isfinite(next_radius) & (next_radius >= 0.0), next_radius, radius)
    return uv_distorted * (radius / np.clip(radius_distorted, SMALL_VALUE, None))


def _screen_point_to_world_ray(camera: np.ndarray, screen_point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = _camera_basis_numpy(camera[0:4])
    cam_pos = -rotation.T @ camera[4:7]
    uv_distorted = (screen_point - camera[9:11]) / np.clip(camera[7:9], SMALL_VALUE, None)
    uv = _undistort_normalized_numpy(uv_distorted[None, :], float(camera[13]), float(camera[14]))[0]
    camera_ray = np.array([uv[0], uv[1], 1.0], dtype=np.float32)
    camera_ray /= max(float(np.linalg.norm(camera_ray)), SMALL_VALUE)
    return cam_pos.astype(np.float32, copy=False), (camera_ray @ rotation.T).astype(np.float32, copy=False)


def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    qv = q[1:]
    cross1 = np.cross(v, qv)
    cross2 = np.cross(cross1 + q[0] * v, qv)
    return v + 2.0 * cross2


def _outline_screen_point(projection_row: np.ndarray, theta: float) -> np.ndarray:
    center = projection_row[0, :2]
    a, b, c = projection_row[1, :3]
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    denom = a * direction[0] * direction[0] + 2.0 * b * direction[0] * direction[1] + c * direction[1] * direction[1]
    return center + direction / np.float32(math.sqrt(max(float(denom), 1e-12)))


def _ray_splat_intersection_alpha(ray_origin: np.ndarray, ray_direction: np.ndarray, splat: np.ndarray) -> float:
    raw_scale = np.maximum(np.exp(splat[3:6]).astype(np.float32), np.max(np.exp(splat[3:6])) / MAX_ANISOTROPY)
    ro = ray_origin - splat[0:3]
    ro_local = _quat_rotate(ro[None, :], splat[6:10])[0] / raw_scale
    ray_local = _quat_rotate(ray_direction[None, :], splat[6:10])[0] / raw_scale
    denom = float(np.dot(ray_local, ray_local))
    if denom <= 1e-10:
        return 0.0
    t_closest = -float(np.dot(ray_local, ro_local)) / denom
    if t_closest <= 0.0:
        return 0.0
    rho2 = max(float(np.dot(ro_local, ro_local) - np.dot(ray_local, ro_local) ** 2 / denom), 0.0)
    return float(np.clip(splat[13], ALPHA_EPS, 1.0 - ALPHA_EPS) * math.exp(-0.5 * rho2))


def _bbox_half_extent(projection_row: np.ndarray) -> np.ndarray:
    a, b, c = projection_row[1, :3]
    det = a * c - b * b
    if det <= 1e-12 or a <= 1e-12 or c <= 1e-12:
        return np.zeros((2,), dtype=np.float32)
    return np.array((math.sqrt(max(float(c / det), 0.0)), math.sqrt(max(float(a / det), 0.0))), dtype=np.float32)


def _ellipse_union_fraction(projection: np.ndarray, image_size: tuple[int, int]) -> float:
    width, height = image_size
    grid_w = min(width, 256)
    grid_h = min(height, 256)
    xs = (np.arange(grid_w, dtype=np.float32) + 0.5) * (width / grid_w)
    ys = (np.arange(grid_h, dtype=np.float32) + 0.5) * (height / grid_h)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    covered = np.zeros((grid_h, grid_w), dtype=bool)
    for row in projection:
        center = row[0, :2]
        a, b, c = row[1, :3]
        dx = xx - center[0]
        dy = yy - center[1]
        covered |= (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy) <= 1.0
    return float(covered.mean())


def _projection_geometry_metrics(projected: dict[str, np.ndarray], image_size: tuple[int, int]) -> dict[str, float]:
    visible = projected["tile_counts"] > 0
    visible_fraction = float(np.mean(visible))
    if not np.any(visible):
        return {"visible_fraction": visible_fraction, "max_bbox_cutoff": 1.0, "union_fraction": 0.0, "mean_ellipse_area": 0.0}
    projection = projected["projection"][visible]
    extents = np.stack([_bbox_half_extent(row) for row in projection], axis=0)
    centers = projection[:, 0, :2]
    mins = centers - extents
    maxs = centers + extents
    width, height = image_size
    image_area = float(width * height)
    clipped = np.clip(maxs, (0.0, 0.0), (float(width), float(height))) - np.clip(mins, (0.0, 0.0), (float(width), float(height)))
    clipped_area = np.prod(np.maximum(clipped, 0.0), axis=1)
    bbox_area = np.prod(np.maximum(maxs - mins, 0.0), axis=1)
    cutoff = np.zeros_like(clipped_area)
    valid = bbox_area > 1e-6
    cutoff[valid] = 1.0 - clipped_area[valid] / bbox_area[valid]
    det = np.maximum(projection[:, 1, 0] * projection[:, 1, 2] - projection[:, 1, 1] * projection[:, 1, 1], 1e-12)
    ellipse_area = np.pi / np.sqrt(det) / image_area
    return {
        "visible_fraction": visible_fraction,
        "max_bbox_cutoff": float(np.max(cutoff)),
        "union_fraction": _ellipse_union_fraction(projection, image_size),
        "mean_ellipse_area": float(np.mean(ellipse_area)),
    }


def _comparison_tolerances(image_size: tuple[int, int]) -> dict[str, float]:
    if image_size == EXTREME_IMAGE_SIZE:
        return {
            "rgb_mae": EXTREME_RENDER_MEAN_ABS_TOL,
            "alpha_mean": EXTREME_ALPHA_MEAN_ABS_TOL,
            "depth_mean": EXTREME_DEPTH_MEAN_ABS_TOL,
            "alpha_p999": EXTREME_ALPHA_P999_TOL,
            "grad_cosine": EXTREME_GRADIENT_COSINE_MIN,
            "grad_rel_l2": EXTREME_GRADIENT_REL_L2_MAX,
        }
    if image_size == TALL_IMAGE_SIZE:
        return {
            "rgb_mae": RENDER_MEAN_ABS_TOL,
            "alpha_mean": TALL_ALPHA_MEAN_ABS_TOL,
            "depth_mean": DEPTH_MEAN_ABS_TOL,
            "alpha_p999": TALL_ALPHA_P999_TOL,
            "grad_cosine": GRADIENT_COSINE_MIN,
            "grad_rel_l2": GRADIENT_REL_L2_MAX,
        }
    return {
        "rgb_mae": RENDER_MEAN_ABS_TOL,
        "alpha_mean": ALPHA_MEAN_ABS_TOL,
        "depth_mean": DEPTH_MEAN_ABS_TOL,
        "alpha_p999": ALPHA_P999_TOL,
        "grad_cosine": GRADIENT_COSINE_MIN,
        "grad_rel_l2": GRADIENT_REL_L2_MAX,
    }


def _make_weight_map(image_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    width, height = image_size
    xs = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=device)
    ys = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=device)
    grid_x = xs[None, :].expand(height, width)
    grid_y = ys[:, None].expand(height, width)
    radial = grid_x.square() + grid_y.square()
    return 1.0 + 0.35 * grid_x - 0.2 * grid_y + 0.15 * grid_x * grid_y + 0.1 * radial


def _evaluate_losses(
    splats: torch.Tensor,
    camera: torch.Tensor,
    image_size: tuple[int, int],
    context: TorchSplattingContext,
    weight_map: torch.Tensor,
) -> dict[str, torch.Tensor]:
    image = render_gaussian_splats(splats, camera, image_size, background=(0.0, 0.0, 0.0), context=context)
    return {
        "red_weighted_mean": (weight_map * image[..., 0]).mean(),
        "green_weighted_mean": (weight_map * image[..., 1]).mean(),
        "blue_weighted_mean": (weight_map * image[..., 2]).mean(),
        "alpha_weighted_mean": (weight_map * image[..., 3]).mean(),
        "depth_weighted_mean": (weight_map * image[..., 4]).mean(),
    }


def _alpha_mean(splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int], context: TorchSplattingContext) -> float:
    image = render_gaussian_splats(splats, camera, image_size, background=(0.0, 0.0, 0.0), context=context)
    return float(image[..., 3].mean().item())


def _identity_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return gradient


def _rolled_splat_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return torch.roll(gradient, shifts=1, dims=1)


def _run_descent_for_loss(
    base_splats: torch.Tensor,
    camera: torch.Tensor,
    image_size: tuple[int, int],
    context: TorchSplattingContext,
    weight_map: torch.Tensor,
    loss_name: str,
    gradient_transform,
) -> dict[str, float | bool | int]:
    current = base_splats.detach().clone()
    final_loss = 0.0

    for _ in range(DESCENT_ITERATION_COUNT):
        work = current.detach().clone().requires_grad_(True)
        losses = _evaluate_losses(work, camera, image_size, context, weight_map)
        loss = losses[loss_name]
        loss.backward()
        gradient = gradient_transform(work.grad.detach().clone())
        gradient_norm = float(torch.linalg.norm(gradient).item())
        loss_value = float(loss.item())
        final_loss = loss_value
        if gradient_norm <= 1e-12:
            break

        direction = gradient / gradient_norm
        accepted = False
        for step_scale in DESCENT_STEP_CANDIDATES:
            candidate = current - float(step_scale) * direction
            with torch.no_grad():
                candidate_losses = _evaluate_losses(candidate, camera, image_size, context, weight_map)
            candidate_loss = float(candidate_losses[loss_name].item())
            if candidate_loss < loss_value - DESCENT_MIN_IMPROVEMENT:
                current = candidate.detach().clone()
                final_loss = candidate_loss
                accepted = True
                break
        if not accepted:
            break

    initial_loss = float(_evaluate_losses(base_splats, camera, image_size, context, weight_map)[loss_name].item())
    relative_improvement = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    return {
        "relative_improvement": relative_improvement,
        "passed": relative_improvement >= DESCENT_PASSING_RELATIVE_IMPROVEMENT,
    }


def test_rendering_matches_torch_ground_truth_and_gradients(backend_name: str) -> None:
    try:
        context = _make_context(backend_name)
    except Exception as exc:
        pytest.skip(f"{backend_name} unavailable: {exc}")

    # Use one shared synthetic scene across all camera resolutions so forward and backward
    # comparisons exercise identical splat content under different projection regimes.
    splats = _make_scene_splats()
    for spec in CAMERA_SPECS:
        image_size = tuple(spec["image_size"])
        camera = _make_camera(spec)

        # Compare renderer and reference on a deterministic sparse pixel subset. This keeps
        # the torch ground truth affordable on large images while still sampling the full frame.
        pixel_indices = _select_pixel_subset(image_size)
        grad_output, sampled_grad_output = _make_sparse_grad_output(image_size, pixel_indices)

        # Run the backend renderer, its backward pass, and projection in the same configuration.
        # The projection result is also used to enforce the scene framing constraints directly.
        image = _render_scene(context, splats, camera, image_size)
        grads = _backward_scene(context, splats, camera, image_size, grad_output)
        projected = _project_scene(context, splats, camera, image_size)

        # Evaluate the torch reference on exactly the sampled pixels and with the same sparse
        # gradient output, so the backward comparison measures the same loss functional.
        ref_image, ref_grads = _reference_render_and_grad(splats, camera, image_size, pixel_indices, sampled_grad_output)
        sampled_image = image.reshape(-1, 5)[pixel_indices]
        tolerances = _comparison_tolerances(image_size)

        # Guard the deliberately tuned scene layout. These checks ensure that most splats stay
        # visible, projected ellipses overlap enough of the screen, and the fitted bbox clipping
        # does not exceed the requested cutoff budget.
        geometry = _projection_geometry_metrics(projected, image_size)
        assert geometry["visible_fraction"] >= MIN_VISIBLE_FRACTION, (
            f"{backend_name} {image_size} visible fraction {geometry['visible_fraction']} below {MIN_VISIBLE_FRACTION}"
        )
        assert geometry["max_bbox_cutoff"] <= MAX_BBOX_CUTOFF, (
            f"{backend_name} {image_size} max bbox cutoff {geometry['max_bbox_cutoff']} exceeds {MAX_BBOX_CUTOFF}"
        )
        assert geometry["union_fraction"] >= MIN_SCREEN_OVERLAP, (
            f"{backend_name} {image_size} union fraction {geometry['union_fraction']} below {MIN_SCREEN_OVERLAP}"
        )
        assert MIN_MEAN_ELLIPSE_AREA <= geometry["mean_ellipse_area"] <= MAX_MEAN_ELLIPSE_AREA, (
            f"{backend_name} {image_size} mean ellipse area {geometry['mean_ellipse_area']} outside [{MIN_MEAN_ELLIPSE_AREA}, {MAX_MEAN_ELLIPSE_AREA}]"
        )

        # Finite outputs are a hard requirement before any numeric comparison. If these fail,
        # subsequent error metrics become misleading or silently mask backend instability.
        assert np.isfinite(image).all()
        assert np.isfinite(grads).all()
        assert np.isfinite(ref_image).all()
        assert np.isfinite(ref_grads).all()

        # Forward checks compare renderer output against the torch reference on the sampled pixels.
        # RGB uses mean absolute error, alpha also tracks a heavy tail via p99.9, and depth ratio
        # is checked separately because it is numerically more sensitive than color.
        rgb_mae = float(np.abs(sampled_image[:, :3] - ref_image[:, :3]).mean())
        alpha_error = np.abs(sampled_image[:, 3] - ref_image[:, 3])
        depth_error = np.abs(sampled_image[:, 4] - ref_image[:, 4])
        assert rgb_mae <= tolerances["rgb_mae"], f"{backend_name} {image_size} rgb mean abs error {rgb_mae}"
        assert float(alpha_error.mean()) <= tolerances["alpha_mean"], f"{backend_name} {image_size} alpha mean abs error {float(alpha_error.mean())}"
        assert float(depth_error.mean()) <= tolerances["depth_mean"], f"{backend_name} {image_size} depth mean abs error {float(depth_error.mean())}"
        assert float(np.quantile(alpha_error.reshape(-1), 0.999)) <= tolerances["alpha_p999"], f"{backend_name} {image_size} alpha p99.9 error too high"

        # Backward checks require both directional agreement and bounded magnitude error. Cosine
        # similarity catches sign or alignment failures; relative L2 catches globally wrong scale.
        cosine, relative_l2 = _gradient_metrics(grads, ref_grads)
        assert cosine >= tolerances["grad_cosine"], f"{backend_name} {image_size} gradient cosine {cosine}"
        assert relative_l2 <= tolerances["grad_rel_l2"], f"{backend_name} {image_size} gradient relative l2 {relative_l2}"


def test_projection_outline_hits_alpha_cutoff(backend_name: str) -> None:
    try:
        context = _make_context(backend_name)
    except Exception as exc:
        pytest.skip(f"{backend_name} unavailable: {exc}")

    # Reuse the same scene as the render/gradient test so projection validity is checked on the
    # exact framing configuration that the main regression depends on.
    splats = _make_scene_splats()
    rng = np.random.default_rng(123)
    for spec in CAMERA_SPECS:
        image_size = tuple(spec["image_size"])

        # Disable distortion for this test so the projected conic boundary can be compared
        # exactly, as distorted cameras do not produce conic projections
        camera = _make_camera(spec, distortion_override=(0.0, 0.0))
        projected = _project_scene(context, splats, camera, image_size)
        visible = np.flatnonzero(projected["tile_counts"] > 0)
        assert visible.size > 0, f"{backend_name} {image_size} produced no visible splats"

        # Sample a small set of visible splats, choose random angles on each fitted outline,
        # convert those outline points back into world-space rays, and verify that the implied
        # Gaussian alpha lands at the configured renderer cutoff.
        sample_count = min(PROJECTION_SAMPLE_COUNT, visible.size)
        sampled_indices = rng.choice(visible, size=sample_count, replace=False)
        for splat_index in sampled_indices.tolist():
            outline_point = _outline_screen_point(projected["projection"][splat_index], float(rng.uniform(0.0, 2.0 * math.pi)))
            ray_origin, ray_direction = _screen_point_to_world_ray(camera, outline_point)
            alpha = _ray_splat_intersection_alpha(ray_origin, ray_direction, splats[:, splat_index])
            assert abs(alpha - context.alpha_cutoff) <= PROJECTION_ALPHA_TOL, (
                f"{backend_name} {image_size} splat {splat_index} outline alpha {alpha} differs from cutoff {context.alpha_cutoff}"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_channel_descent_reference_passes_and_broken_control_fails() -> None:
    # Keep the original locked line scene for this regression. It is intentionally denser than
    # the main module scene so the descent signal is strong and step acceptance is easy to judge.
    splats, camera, image_size, alpha_cutoff = _make_locked_line_scene()
    context = _make_torch_context(alpha_cutoff)

    # Sanity-check the fixed scene density before running optimization. If this fails, the scene
    # no longer exercises the same regime that originally exposed broken gradient channels.
    alpha_mean = _alpha_mean(splats, camera, image_size, context)
    assert alpha_mean >= MIN_LOCKED_LINE_ALPHA_MEAN, (
        f"Expected mean alpha >= {MIN_LOCKED_LINE_ALPHA_MEAN}, got {alpha_mean}"
    )

    # Use the same weighted scalar objectives across all five output channels so each one gets a
    # descent test, and compare the true gradient against a negative control that rolls splats.
    weight_map = _make_weight_map(image_size, splats.device)
    loss_names = list(_evaluate_losses(splats, camera, image_size, context, weight_map).keys())
    reference_results = {
        loss_name: _run_descent_for_loss(splats, camera, image_size, context, weight_map, loss_name, _identity_gradient)
        for loss_name in loss_names
    }
    broken_results = {
        loss_name: _run_descent_for_loss(splats, camera, image_size, context, weight_map, loss_name, _rolled_splat_gradient)
        for loss_name in loss_names
    }

    # Every reference channel should make substantial progress, while the broken control should
    # fail the same threshold. This turns the test into a behavioral check instead of a fragile
    # pointwise finite-difference comparison.
    reference_failures = {
        loss_name: result["relative_improvement"]
        for loss_name, result in reference_results.items()
        if not bool(result["passed"])
    }
    broken_passes = {
        loss_name: result["relative_improvement"]
        for loss_name, result in broken_results.items()
        if bool(result["passed"])
    }

    assert not reference_failures, (
        f"Reference gradient descent failed threshold {DESCENT_PASSING_RELATIVE_IMPROVEMENT}: {reference_failures}"
    )
    assert not broken_passes, (
        f"Broken gradient negative control unexpectedly passed threshold {DESCENT_PASSING_RELATIVE_IMPROVEMENT}: {broken_passes}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_clone_candidate_pass_yields_expected_record_count() -> None:
    context = _make_torch_context(ALPHA_CUTOFF)
    context.trans_threshold = TRANS_THRESHOLD
    context.max_anisotropy = MAX_ANISOTROPY
    splats = torch.tensor(_make_scene_splats(), dtype=torch.float32, device="cuda")
    camera = _make_forward_camera(TEST_SCENE_IMAGE_SIZE, device="cuda")
    render_gaussian_splats(splats, camera, TEST_SCENE_IMAGE_SIZE, context=context, render_seed=123)
    target = torch.zeros((TEST_SCENE_IMAGE_SIZE[1], TEST_SCENE_IMAGE_SIZE[0], 3), dtype=torch.float32, device="cuda")

    clones = context.clone_candidates_current(target, select_probability=0.01, max_clone_candidates=4096, clone_seed=123)

    assert int(clones["count"].item()) == 89
    assert int(clones["clone_counts"].sum().item()) == 89
    assert torch.all(torch.isfinite(clones["positions"]))
    assert torch.all((clones["ids"] >= 0) & (clones["ids"] < splats.shape[1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_clone_candidate_pass_truncates_at_capacity() -> None:
    context = _make_torch_context(ALPHA_CUTOFF)
    context.trans_threshold = TRANS_THRESHOLD
    context.max_anisotropy = MAX_ANISOTROPY
    splats = torch.tensor(_make_scene_splats(), dtype=torch.float32, device="cuda")
    camera = _make_forward_camera(TEST_SCENE_IMAGE_SIZE, device="cuda")
    render_gaussian_splats(splats, camera, TEST_SCENE_IMAGE_SIZE, context=context, render_seed=123)
    target = torch.zeros((TEST_SCENE_IMAGE_SIZE[1], TEST_SCENE_IMAGE_SIZE[0], 3), dtype=torch.float32, device="cuda")

    clones = context.clone_candidates_current(target, select_probability=1.0, max_clone_candidates=32, clone_seed=123)

    assert int(clones["count"].item()) == 32
    assert int(clones["clone_counts"].sum().item()) == 32

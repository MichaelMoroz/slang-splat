from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import slangpy as spy
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module.splatting import SplattingContext


_SMOKE_IMAGE_SIZE = (64, 64)
_PARITY_IMAGE_SIZE = (128, 128)
_GRADIENT_IMAGE_SIZE = (32, 32)
_GRADIENT_REFERENCE_SPLAT_COUNT = 64
_GRADIENT_REFERENCE_PARAM_SAMPLES = 12
_GRADIENT_FINITE_DIFF_EPS = 1e-3
_GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0
_RAY_DENOMINATOR_FLOOR = 1e-10
_ALPHA_CUTOFF = 1 / 255
_TRANS_THRESHOLD = 0.005
_OUTPUT_GAMMA = 2.2
_SMALL_VALUE = 1e-6
_ALPHA_EPS = 1e-6
_PROJECTION_OUTLINE_MAX_ERROR = 1e-3
_SHADERS = Path(__file__).resolve().parents[1] / "module" / "shaders"
_GARDEN_PLY = Path(__file__).resolve().parents[1] / "test_data" / "garden" / "garden.ply"


def _make_device(backend_name: str) -> spy.Device:
    return spy.create_device(type=getattr(spy.DeviceType, backend_name), include_paths=[_SHADERS], enable_cuda_interop=False)


def _make_splats(count: int = 64, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    splats = np.zeros((14, count), dtype=np.float32)
    splats[0:2] = rng.uniform(-0.18, 0.18, size=(2, count)).astype(np.float32)
    splats[2] = rng.uniform(4.5, 6.0, size=(count,)).astype(np.float32)
    splats[3:6] = rng.uniform(np.log(0.07), np.log(0.16), size=(3, count)).astype(np.float32)
    splats[6] = 1.0
    splats[10:13] = rng.uniform(0.2, 1.0, size=(3, count)).astype(np.float32)
    splats[13] = rng.uniform(0.75, 0.98, size=(count,)).astype(np.float32)
    return splats


def _make_gradient_splats(count: int = 64, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    splats = np.zeros((14, count), dtype=np.float32)
    splats[0:2] = rng.uniform(-0.08, 0.08, size=(2, count)).astype(np.float32)
    splats[2] = rng.uniform(5.2, 6.4, size=(count,)).astype(np.float32)
    splats[3:6] = rng.uniform(np.log(0.05), np.log(0.09), size=(3, count)).astype(np.float32)
    splats[6] = 1.0
    splats[10:13] = rng.uniform(0.2, 1.0, size=(3, count)).astype(np.float32)
    splats[13] = rng.uniform(0.7, 0.95, size=(count,)).astype(np.float32)
    return splats


def _make_parity_splats(count: int = 192, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    splats = np.zeros((14, count), dtype=np.float32)
    splats[0:2] = rng.uniform(-0.28, 0.28, size=(2, count)).astype(np.float32)
    splats[2] = rng.uniform(4.0, 5.5, size=(count,)).astype(np.float32)
    splats[3:6] = rng.uniform(np.log(0.14), np.log(0.28), size=(3, count)).astype(np.float32)
    splats[6] = 1.0
    splats[10:13] = rng.uniform(0.25, 1.0, size=(3, count)).astype(np.float32)
    splats[13] = rng.uniform(0.82, 0.99, size=(count,)).astype(np.float32)
    return splats


def _make_camera(
    image_size: tuple[int, int],
    focal_pixels: tuple[float, float] = (72.0, 72.0),
    distortion: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    width, height = image_size
    return np.array(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, focal_pixels[0], focal_pixels[1], width * 0.5, height * 0.5, 0.1, 20.0, *distortion],
        dtype=np.float32,
    )


def _camera_basis(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _camera_dict(camera: np.ndarray, image_size: tuple[int, int]) -> dict[str, object]:
    rot = _camera_basis(camera[0:4])
    cam_pos = -rot.T @ camera[4:7]
    return {
        "camPos": spy.float3(*cam_pos.tolist()),
        "camBasis": spy.float3x3(rot),
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
    alpha = np.clip(packed[13], _ALPHA_EPS, 1.0 - _ALPHA_EPS)
    packed[13] = np.log(alpha / np.clip(1.0 - alpha, _ALPHA_EPS, 1.0))
    return packed


def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    qv = q[1:]
    cross1 = np.cross(v, qv)
    cross2 = np.cross(cross1 + q[0] * v, qv)
    return v + 2.0 * cross2


def _undistort_normalized(uv_distorted: np.ndarray, k1: float, k2: float) -> np.ndarray:
    if abs(k1) <= _SMALL_VALUE and abs(k2) <= _SMALL_VALUE:
        return uv_distorted
    radius_distorted = np.linalg.norm(uv_distorted, axis=-1, keepdims=True)
    radius = radius_distorted.copy()
    for _ in range(6):
        r2 = radius * radius
        r4 = r2 * r2
        deriv = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4
        safe = np.abs(deriv) > _SMALL_VALUE
        next_radius = radius - (radius * (1.0 + k1 * r2 + k2 * r4) - radius_distorted) / np.where(safe, deriv, 1.0)
        radius = np.where(safe & np.isfinite(next_radius) & (next_radius >= 0.0), next_radius, radius)
    return uv_distorted * (radius / np.clip(radius_distorted, _SMALL_VALUE, None))


def _screen_to_world_rays(camera: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    uv_distorted = (np.stack((xs + 0.5, ys + 0.5), axis=-1) - camera[9:11]) / np.clip(camera[7:9], _SMALL_VALUE, None)
    uv = _undistort_normalized(uv_distorted, float(camera[13]), float(camera[14]))
    camera_rays = np.concatenate((uv, np.ones_like(uv[..., :1])), axis=-1)
    camera_rays /= np.clip(np.linalg.norm(camera_rays, axis=-1, keepdims=True), _SMALL_VALUE, None)
    return camera_rays @ _camera_basis(camera[0:4]).T


def _project_world_to_screen(camera: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    rot = _camera_basis(camera[0:4])
    cam_pos = -rot.T @ camera[4:7]
    cam_points = (world_points - cam_pos) @ rot.T
    uv = cam_points[..., :2] / np.clip(cam_points[..., 2:], _SMALL_VALUE, None)
    radius2 = np.sum(uv * uv, axis=-1, keepdims=True)
    distortion = 1.0 + camera[13] * radius2 + camera[14] * radius2 * radius2
    return uv * distortion * camera[7:9] + camera[9:11]


def _quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    half_angle = 0.5 * angle_rad
    return np.concatenate((np.array([np.cos(half_angle)], dtype=np.float32), axis * np.sin(half_angle))).astype(np.float32)


def _gaussian_outline_points(camera: np.ndarray, gaussian: np.ndarray) -> np.ndarray:
    return _gaussian_outline_points_sampled(camera, gaussian, 5)


def _gaussian_outline_points_sampled(camera: np.ndarray, gaussian: np.ndarray, sample_count: int) -> np.ndarray:
    fused_opacity = gaussian[13]
    support_sigma_radius = np.sqrt(max(-2.0 * np.log(_ALPHA_CUTOFF / fused_opacity), 0.0))
    support_scale = np.clip(np.exp(gaussian[3:6]) * support_sigma_radius, _SMALL_VALUE, None)
    cam_pos = -_camera_basis(camera[0:4]).T @ camera[4:7]
    view_origin_local = _quat_rotate((cam_pos - gaussian[0:3])[None, :], gaussian[6:10])[0] / support_scale
    view_distance = np.linalg.norm(view_origin_local)
    view_dir_local = view_origin_local / max(view_distance, _SMALL_VALUE)
    tangent_circle_center = view_dir_local / view_distance
    tangent_circle_radius = np.sqrt(max(1.0 - 1.0 / (view_distance * view_distance), 0.0))
    if float(view_dir_local[2]) < -0.999999:
        tangent_basis_u = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        tangent_basis_v = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    else:
        inv_south_pole_distance = 1.0 / (1.0 + view_dir_local[2])
        xy_mix = -view_dir_local[0] * view_dir_local[1] * inv_south_pole_distance
        tangent_basis_u = np.array([1.0 - view_dir_local[0] * view_dir_local[0] * inv_south_pole_distance, xy_mix, -view_dir_local[0]], dtype=np.float32)
        tangent_basis_v = np.array([xy_mix, 1.0 - view_dir_local[1] * view_dir_local[1] * inv_south_pole_distance, -view_dir_local[1]], dtype=np.float32)
    world_points = []
    inv_rotation = np.concatenate((gaussian[6:7], -gaussian[7:10])).astype(np.float32)
    for i in range(sample_count):
        theta = 2.0 * np.pi * (i / float(sample_count))
        support_point_local = tangent_circle_center + tangent_circle_radius * (np.cos(theta) * tangent_basis_u + np.sin(theta) * tangent_basis_v)
        world_points.append(gaussian[0:3] + _quat_rotate((support_point_local * support_scale)[None, :], inv_rotation)[0])
    return _project_world_to_screen(camera, np.stack(world_points))


def _gaussian_outline_points_from_camera_dict(camera: dict[str, object], gaussian: np.ndarray, sample_count: int) -> np.ndarray:
    cam_pos = np.array([camera["camPos"].x, camera["camPos"].y, camera["camPos"].z], dtype=np.float32)
    cam_basis = np.array(camera["camBasis"].to_numpy(), dtype=np.float32)
    focal = np.array([camera["focalPixels"].x, camera["focalPixels"].y], dtype=np.float32)
    principal = np.array([camera["principalPoint"].x, camera["principalPoint"].y], dtype=np.float32)
    fused_opacity = gaussian[13]
    support_sigma_radius = np.sqrt(max(-2.0 * np.log(_ALPHA_CUTOFF / fused_opacity), 0.0))
    support_scale = np.clip(np.exp(gaussian[3:6]) * support_sigma_radius, _SMALL_VALUE, None)
    view_origin_local = _quat_rotate((cam_pos - gaussian[0:3])[None, :], gaussian[6:10])[0] / support_scale
    view_distance = np.linalg.norm(view_origin_local)
    view_dir_local = view_origin_local / max(view_distance, _SMALL_VALUE)
    tangent_circle_center = view_dir_local / view_distance
    tangent_circle_radius = np.sqrt(max(1.0 - 1.0 / (view_distance * view_distance), 0.0))
    tangent_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(float(view_dir_local[2])) < 0.999 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_basis_u = np.cross(tangent_axis, view_dir_local)
    tangent_basis_u /= max(float(np.linalg.norm(tangent_basis_u)), _SMALL_VALUE)
    tangent_basis_v = np.cross(view_dir_local, tangent_basis_u)
    inv_rotation = np.concatenate((gaussian[6:7], -gaussian[7:10])).astype(np.float32)
    world_points = []
    for i in range(sample_count):
        theta = 2.0 * np.pi * (i / float(sample_count))
        support_point_local = tangent_circle_center + tangent_circle_radius * (np.cos(theta) * tangent_basis_u + np.sin(theta) * tangent_basis_v)
        world_points.append(gaussian[0:3] + _quat_rotate((support_point_local * support_scale)[None, :], inv_rotation)[0])
    world_points = np.stack(world_points)
    cam_points = (world_points - cam_pos) @ cam_basis.T
    uv = cam_points[:, :2] / np.clip(cam_points[:, 2:], _SMALL_VALUE, None)
    return uv * focal + principal


def _load_ply_splat(path: Path, index: int) -> np.ndarray:
    ply = PlyData.read(str(path), mmap=True)
    vertex = ply.elements[0]
    names = [prop.name for prop in vertex.properties]
    row = vertex.data[index]
    splat = np.zeros((14, 1), dtype=np.float32)
    splat[0, 0] = np.float32(row["x"])
    splat[1, 0] = np.float32(row["y"])
    splat[2, 0] = np.float32(row["z"])
    scale_names = sorted((name for name in names if name.startswith("scale_")), key=lambda name: int(name.split("_")[-1]))
    if scale_names:
        splat[3:6, 0] = np.array([row[name] for name in scale_names], dtype=np.float32)
    rot_names = sorted((name for name in names if name.startswith("rot")), key=lambda name: int("".join(ch for ch in name if ch.isdigit()) or "0"))
    if rot_names:
        rotation = np.array([row[name] for name in rot_names], dtype=np.float32)
        splat[6:10, 0] = rotation / max(float(np.linalg.norm(rotation)), 1e-8)
    else:
        splat[6, 0] = 1.0
    sh_c0 = 0.28209479177387814
    splat[10:13, 0] = np.clip(0.5 + sh_c0 * np.array([row["f_dc_0"], row["f_dc_1"], row["f_dc_2"]], dtype=np.float32), 0.0, 1.0)
    splat[13, 0] = 1.0 / (1.0 + np.exp(-np.float32(row["opacity"])))
    return splat


def _projection_conic_error(projection_row: np.ndarray, outline_points: np.ndarray) -> np.ndarray:
    center = projection_row[0, :2]
    conic = projection_row[1, :3]
    delta = outline_points - center
    return conic[0] * delta[:, 0] * delta[:, 0] + 2.0 * conic[1] * delta[:, 0] * delta[:, 1] + conic[2] * delta[:, 1] * delta[:, 1] - 1.0


def _load_public_splats(context: SplattingContext, splats: np.ndarray, image_size: tuple[int, int], background: tuple[float, float, float]) -> None:
    context.prepare(splats.shape[1], image_size, background)
    context.scene["g_Params"].copy_from_numpy(_pack_params(splats).reshape(-1))
    context.device.sync_to_cuda()


def _tile_counts_from_scanlines(context: SplattingContext, total_scanlines: int, splat_count: int) -> np.ndarray:
    tile_counts = np.zeros((splat_count,), dtype=np.int64)
    if total_scanlines == 0:
        return tile_counts
    scanline_counts = np.asarray(context.scanlines["g_ScanlineTileCounts"].to_numpy()).reshape(-1)[:total_scanlines].astype(np.int64, copy=False)
    scanline_splats = np.asarray(context.scanlines["g_ScanlineEntryData"].to_numpy())[:total_scanlines, 0].astype(np.int64, copy=False)
    np.add.at(tile_counts, scanline_splats, scanline_counts)
    return tile_counts


def _project_scene(context: SplattingContext, splats: np.ndarray, camera: np.ndarray, image_size: tuple[int, int]) -> dict[str, np.ndarray]:
    _load_public_splats(context, splats, image_size, (0.0, 0.0, 0.0))
    total_scanlines = context.project(_camera_dict(camera, image_size), splats.shape[1])
    order = np.asarray(context._sorted_splat_order_tensor.to_numpy()).reshape(-1)[: splats.shape[1]].astype(np.int64, copy=True)
    projection = np.asarray(context._view(context.scene_views["g_ProjectionState"].tensor, spy.float4, (splats.shape[1], 2)).to_numpy())[: splats.shape[1]].copy()
    raster = np.asarray(context._view(context.scene_views["g_RasterState"].tensor, spy.float4, (splats.shape[1], 4)).to_numpy())[: splats.shape[1]].copy()
    return {
        "order": order,
        "projection": projection,
        "raster": raster,
        "tile_counts": _tile_counts_from_scanlines(context, total_scanlines, splats.shape[1]),
    }


def _render_scene(context: SplattingContext, splats: np.ndarray, camera: np.ndarray, image_size: tuple[int, int], background: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    _load_public_splats(context, splats, image_size, background)
    return np.asarray(context.render(_camera_dict(camera, image_size), splats.shape[1]).to_numpy()).copy()


def _backward_scene(
    context: SplattingContext,
    splats: np.ndarray,
    camera: np.ndarray,
    image_size: tuple[int, int],
    grad_output: np.ndarray,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    _load_public_splats(context, splats, image_size, background)
    context.render(_camera_dict(camera, image_size), splats.shape[1])
    context.frame["g_OutputGrad"].copy_from_numpy(np.ascontiguousarray(grad_output.astype(np.float32, copy=False)))
    context.device.sync_to_cuda()
    grads = np.asarray(context.backward(_camera_dict(camera, image_size), splats.shape[1]).to_numpy())[: splats.shape[1] * 14].copy()
    return grads.reshape(splats.shape[1], 14).T


def _loss_and_grad(context: SplattingContext, splats: np.ndarray, camera: np.ndarray, image_size: tuple[int, int]) -> tuple[float, np.ndarray]:
    image = _render_scene(context, splats, camera, image_size)
    grad_output = image.astype(np.float32, copy=False)
    grads = _backward_scene(context, splats, camera, image_size, grad_output)
    return float(0.5 * np.square(image).sum()), grads


def _finite_difference_samples(splats: np.ndarray, camera: np.ndarray, image_size: tuple[int, int], eps: float, sample_indices: np.ndarray, context: SplattingContext) -> np.ndarray:
    work = splats.copy()
    reference = np.empty((sample_indices.shape[0],), dtype=np.float32)
    splat_count = splats.shape[1]
    for sample_idx, flat_idx in enumerate(sample_indices.tolist()):
        param_idx, splat_idx = divmod(int(flat_idx), splat_count)
        original = float(work[param_idx, splat_idx])
        work[param_idx, splat_idx] = original + eps
        loss_plus = _loss_and_grad(context, work, camera, image_size)[0]
        work[param_idx, splat_idx] = original - eps
        loss_minus = _loss_and_grad(context, work, camera, image_size)[0]
        work[param_idx, splat_idx] = original
        reference[sample_idx] = (loss_plus - loss_minus) / (2.0 * eps)
    return reference


def _render_signature(context: SplattingContext, splats: np.ndarray, camera: np.ndarray, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _render_scene(context, splats, camera, image_size)
    order = np.asarray(context._sorted_splat_order_tensor.to_numpy()).reshape(-1)[: splats.shape[1]].astype(np.int64, copy=True)
    entries = np.asarray(
        context.raw["g_SortedEntryData"].to_numpy() if context._sorted_entries_tensor is context.raw["g_SortedEntryData"] else context.raw["g_TileEntryData"].to_numpy()
    )[: context._last_total * 2].reshape(-1, 2).copy()
    ranges = np.asarray(context.tiles["g_TileRanges"].to_numpy()).reshape(-1).copy()
    return order, entries, ranges


def _select_stable_param_samples(
    splats: np.ndarray,
    camera: np.ndarray,
    image_size: tuple[int, int],
    sample_count: int,
    eps: float,
    context: SplattingContext,
    seed: int = 123,
    min_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    reference: list[float] = []
    min_count = sample_count if min_count is None else min_count
    candidates = np.random.default_rng(seed).permutation(splats.size)
    for flat_idx in candidates.tolist():
        param_idx, splat_idx = divmod(int(flat_idx), splats.shape[1])
        minus = splats.copy()
        plus = splats.copy()
        minus[param_idx, splat_idx] -= eps
        plus[param_idx, splat_idx] += eps
        minus_sig = _render_signature(context, minus, camera, image_size)
        base_sig = _render_signature(context, splats, camera, image_size)
        plus_sig = _render_signature(context, plus, camera, image_size)
        if not (
            np.array_equal(minus_sig[0], base_sig[0]) and np.array_equal(plus_sig[0], base_sig[0]) and
            np.array_equal(minus_sig[1], base_sig[1]) and np.array_equal(plus_sig[1], base_sig[1]) and
            np.array_equal(minus_sig[2], base_sig[2]) and np.array_equal(plus_sig[2], base_sig[2])
        ):
            continue
        sample = np.array([flat_idx], dtype=np.int64)
        fd_half = _finite_difference_samples(splats, camera, image_size, 0.5 * eps, sample, context)[0]
        fd_eps = _finite_difference_samples(splats, camera, image_size, eps, sample, context)[0]
        fd_2eps = _finite_difference_samples(splats, camera, image_size, 2.0 * eps, sample, context)[0]
        spread = max(abs(float(fd_half) - float(fd_eps)), abs(float(fd_eps) - float(fd_2eps)), abs(float(fd_half) - float(fd_2eps)))
        tol = max(0.2, 0.15 * max(abs(float(fd_half)), abs(float(fd_eps)), abs(float(fd_2eps))))
        if spread > tol:
            continue
        selected.append(flat_idx)
        reference.append(float(fd_eps))
        if len(selected) == sample_count:
            break
    if len(selected) < min_count:
        raise AssertionError(f"Only found {len(selected)} stable finite-difference samples, expected at least {min_count}.")
    return np.array(selected, dtype=np.int64), np.array(reference, dtype=np.float32)


def _ground_truth_render(splats: np.ndarray, camera: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    cam_basis = _camera_basis(camera[0:4])
    cam_pos = -cam_basis.T @ camera[4:7]
    rays = _screen_to_world_rays(camera, image_size).reshape(-1, 3)
    accum = np.zeros((rays.shape[0], 3), dtype=np.float32)
    trans = np.ones((rays.shape[0],), dtype=np.float32)
    for i in range(splats.shape[1]):
        gaussian = splats[:, i]
        scale = np.clip(np.exp(gaussian[3:6]) * _GAUSSIAN_SUPPORT_SIGMA_RADIUS, _SMALL_VALUE, None)
        ro_local = _quat_rotate(np.broadcast_to(cam_pos - gaussian[0:3], rays.shape), gaussian[6:10]) / scale
        ray_local = _quat_rotate(rays, gaussian[6:10]) / scale
        denom = np.sum(ray_local * ray_local, axis=-1)
        t_closest = -np.sum(ray_local * ro_local, axis=-1) / np.clip(denom, _RAY_DENOMINATOR_FLOOR, None)
        closest = ro_local + ray_local * t_closest[:, None]
        rho2 = np.sum(closest * closest, axis=-1)
        alpha = np.where(
            (denom > _RAY_DENOMINATOR_FLOOR) & (t_closest > 0.0),
            np.clip(gaussian[13], _ALPHA_EPS, 1.0 - _ALPHA_EPS) * np.exp(-0.5 * _GAUSSIAN_SUPPORT_SIGMA_RADIUS * _GAUSSIAN_SUPPORT_SIGMA_RADIUS * rho2),
            0.0,
        ).astype(np.float32)
        alpha = np.where(alpha >= _ALPHA_CUTOFF, alpha, 0.0)
        accum = accum + (trans * alpha)[:, None] * gaussian[10:13]
        trans = trans * (1.0 - alpha)
        if np.all(trans < _TRANS_THRESHOLD):
            break
    color = np.power(np.clip(accum, 0.0, None), _OUTPUT_GAMMA)
    return np.concatenate((color, (1.0 - trans)[:, None]), axis=-1).reshape(image_size[1], image_size[0], 4)


@pytest.fixture
def backend_context(backend_name: str) -> SplattingContext:
    try:
        return SplattingContext(device=_make_device(backend_name))
    except Exception as exc:
        pytest.skip(f"{backend_name} unavailable: {exc}")


def test_forward_smoke(backend_context: SplattingContext) -> None:
    image = _render_scene(backend_context, _make_splats(), _make_camera(_SMOKE_IMAGE_SIZE), _SMOKE_IMAGE_SIZE)
    assert tuple(image.shape) == (_SMOKE_IMAGE_SIZE[1], _SMOKE_IMAGE_SIZE[0], 4)
    assert np.isfinite(image).all()


def test_zero_splats_smoke(backend_context: SplattingContext) -> None:
    image = _render_scene(backend_context, np.zeros((14, 0), dtype=np.float32), _make_camera(_SMOKE_IMAGE_SIZE), _SMOKE_IMAGE_SIZE)
    assert tuple(image.shape) == (_SMOKE_IMAGE_SIZE[1], _SMOKE_IMAGE_SIZE[0], 4)
    assert np.isfinite(image).all()
    np.testing.assert_allclose(image, 0.0, atol=0.0, rtol=0.0)
    grads = _backward_scene(
        backend_context,
        np.zeros((14, 0), dtype=np.float32),
        _make_camera(_SMOKE_IMAGE_SIZE),
        _SMOKE_IMAGE_SIZE,
        np.ones_like(image, dtype=np.float32),
    )
    assert grads.shape == (14, 0)
    assert np.isfinite(grads).all()


def test_backward_smoke(backend_context: SplattingContext) -> None:
    image_size = _SMOKE_IMAGE_SIZE
    splats = _make_splats()
    image = _render_scene(backend_context, splats, _make_camera(image_size), image_size)
    grads = _backward_scene(backend_context, splats, _make_camera(image_size), image_size, np.ones_like(image, dtype=np.float32))
    assert np.isfinite(grads).all()
    assert float(np.abs(grads).sum()) > 0.0


def test_forward_matches_ground_truth_closely(backend_context: SplattingContext) -> None:
    splats = _make_parity_splats()
    camera = _make_camera(_PARITY_IMAGE_SIZE)
    image = _render_scene(backend_context, splats, camera, _PARITY_IMAGE_SIZE)
    projected = _project_scene(backend_context, splats, camera, _PARITY_IMAGE_SIZE)
    sorted_splats = splats[:, projected["order"]]
    ref = _ground_truth_render(sorted_splats[:, projected["tile_counts"] > 0], camera, _PARITY_IMAGE_SIZE)
    nonzero_fraction = np.mean(np.any(image[..., :3] > 1e-6, axis=-1) | (image[..., 3] > 1e-6))
    assert float(nonzero_fraction) >= 0.5
    np.testing.assert_allclose(image[..., :3], ref[..., :3], rtol=2e-2, atol=3e-2)
    alpha_error = np.abs(image[..., 3] - ref[..., 3])
    assert float(alpha_error.mean()) <= 4e-3
    assert float(np.quantile(alpha_error, 0.99)) <= 3e-2


def test_gradient_matches_finite_difference_reference(backend_context: SplattingContext) -> None:
    splats = _make_gradient_splats(count=_GRADIENT_REFERENCE_SPLAT_COUNT)
    camera = _make_camera(_GRADIENT_IMAGE_SIZE)
    sample_indices, finite_diff = _select_stable_param_samples(
        splats,
        camera,
        _GRADIENT_IMAGE_SIZE,
        _GRADIENT_REFERENCE_PARAM_SAMPLES,
        _GRADIENT_FINITE_DIFF_EPS,
        backend_context,
        min_count=max(1, _GRADIENT_REFERENCE_PARAM_SAMPLES // 2),
    )
    assert int(sample_indices.shape[0]) >= max(1, _GRADIENT_REFERENCE_PARAM_SAMPLES // 2)
    _, grads = _loss_and_grad(backend_context, splats, camera, _GRADIENT_IMAGE_SIZE)
    np.testing.assert_allclose(grads.reshape(-1)[sample_indices], finite_diff, rtol=5e-2, atol=2e-1)


def test_distortion_case(backend_context: SplattingContext) -> None:
    image = _render_scene(backend_context, _make_gradient_splats(), _make_camera(_SMOKE_IMAGE_SIZE, distortion=(0.05, -0.02)), _SMOKE_IMAGE_SIZE)
    assert np.isfinite(image).all()


def test_stable_sorting_case(backend_context: SplattingContext) -> None:
    splats = _make_splats(count=2)
    splats[0:3, :] = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]], dtype=np.float32)
    splats[10:13, 0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    splats[10:13, 1] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    image = _render_scene(backend_context, splats, _make_camera(_SMOKE_IMAGE_SIZE), _SMOKE_IMAGE_SIZE)
    center = image[32, 32, :3]
    assert center[0] >= center[1]


def test_alpha_logit_gradient_is_finite_and_nonzero(backend_context: SplattingContext) -> None:
    splats = _make_gradient_splats()
    _, grads = _loss_and_grad(backend_context, splats, _make_camera(_PARITY_IMAGE_SIZE), _PARITY_IMAGE_SIZE)
    assert np.isfinite(grads[13]).all()
    assert np.count_nonzero(grads[13]) > 0


@pytest.mark.parametrize(
    ("name", "log_scale", "position_xy", "rotation"),
    (
        ("tiny", (-4.6, -4.7, -4.8), (0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ("elongated", (-1.4, -4.8, -4.8), (0.18, -0.12), None),
    ),
)
def test_projection_matches_outline_for_stress_cases(
    backend_context: SplattingContext,
    name: str,
    log_scale: tuple[float, float, float],
    position_xy: tuple[float, float],
    rotation: tuple[float, float, float, float] | None,
) -> None:
    image_size = (1024, 1024)
    camera = _make_camera(image_size, focal_pixels=(768.0, 768.0))
    splats = _make_splats(count=1)
    splats[0:3, 0] = np.array([position_xy[0], position_xy[1], 6.0], dtype=np.float32)
    splats[3:6, 0] = np.array(log_scale, dtype=np.float32)
    splats[13, 0] = 0.8
    splats[6:10, 0] = _quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), np.deg2rad(32.0)) if rotation is None else np.array(rotation, dtype=np.float32)
    projected = _project_scene(backend_context, splats, camera, image_size)
    assert int(projected["tile_counts"][0]) > 0, f"{name} splat was culled"
    outline_points = _gaussian_outline_points(camera, splats[:, 0])
    errors = np.abs(_projection_conic_error(projected["projection"][0], outline_points))
    assert float(errors.max()) <= _PROJECTION_OUTLINE_MAX_ERROR, f"{name} projection max error {float(errors.max()):.4e}"


@pytest.mark.skipif(not _GARDEN_PLY.exists(), reason="garden.ply unavailable")
def test_projection_is_stable_for_garden_far_large_splat(backend_context: SplattingContext) -> None:
    image_size = (1600, 900)
    target = np.array([0.47749576, 2.1900983, 0.893687], dtype=np.float32)
    splats = _load_ply_splat(_GARDEN_PLY, 1374831)
    for camera_x in (8.0, 8.01):
        eye = np.array([camera_x, 2.0, -10.0], dtype=np.float32)
        forward = target - eye
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        right = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), forward)
        right /= max(float(np.linalg.norm(right)), 1e-8)
        up = np.cross(forward, right)
        focal = 0.5 * image_size[1] / np.tan(0.5 * np.deg2rad(60.0))
        camera_dict = {
            "camPos": spy.float3(*eye.tolist()),
            "camBasis": spy.float3x3(np.stack((right, up, forward), axis=0)),
            "focalPixels": spy.float2(float(focal), float(focal)),
            "principalPoint": spy.float2(image_size[0] * 0.5, image_size[1] * 0.5),
            "viewport": spy.float2(*map(float, image_size)),
            "nearDepth": 0.0,
            "farDepth": 1000.0,
            "k1": 0.0,
            "k2": 0.0,
        }
        _load_public_splats(backend_context, splats, image_size, (0.0, 0.0, 0.0))
        total_scanlines = backend_context.project(camera_dict, splats.shape[1])
        assert total_scanlines > 0
        projection = np.asarray(backend_context._view(backend_context.scene_views["g_ProjectionState"].tensor, spy.float4, (1, 2)).to_numpy())[0]
        outline_points = _gaussian_outline_points_from_camera_dict(camera_dict, splats[:, 0], 128)
        errors = np.abs(_projection_conic_error(projection, outline_points))
        assert float(errors.max()) <= 5e-3


def test_core_buffer_reuse_grows_only(backend_context: SplattingContext) -> None:
    image_size = _SMOKE_IMAGE_SIZE
    backend_context.prepare(32, image_size, (0.0, 0.0, 0.0))
    scene_params_small = backend_context.scene["g_Params"].storage
    projection_small = backend_context.scene["g_ProjectionStateData"].storage
    raster_small = backend_context.scene["g_RasterStateData"].storage
    scanline_small = backend_context.scanlines["g_ScanlineEntryData"].storage
    entry_small = backend_context.raw["g_TileEntryData"].storage

    backend_context.prepare(16, image_size, (0.0, 0.0, 0.0))
    assert backend_context.scene["g_Params"].storage is scene_params_small
    assert backend_context.scene["g_ProjectionStateData"].storage is projection_small
    assert backend_context.scene["g_RasterStateData"].storage is raster_small
    assert backend_context.scanlines["g_ScanlineEntryData"].storage is scanline_small
    assert backend_context.raw["g_TileEntryData"].storage is entry_small

    backend_context.prepare(96, image_size, (0.0, 0.0, 0.0))
    assert backend_context.scene["g_Params"].storage is not scene_params_small
    assert backend_context.scene["g_ProjectionStateData"].storage is not projection_small
    assert backend_context.scene["g_RasterStateData"].storage is not raster_small


def test_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    py_lines = sum(1 for line in (root / "module" / "splatting.py").read_text().splitlines() if line.strip() and not line.strip().startswith("#"))
    slang_lines = 0
    for path in (root / "module" / "shaders").glob("*.slang"):
        slang_lines += sum(1 for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("//"))
    assert py_lines <= 300
    assert slang_lines <= 1000

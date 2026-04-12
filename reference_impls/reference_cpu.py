from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.scene.gaussian_scene import GaussianScene
from src.scene.sh_utils import evaluate_sh0_sh1, resolve_supported_sh_coeffs
from src.renderer.camera import Camera

ALPHA_CUTOFF_DEFAULT = np.float32(1.0 / 255.0)
ELLIPSE_EPS = 1e-6
MIN_CONIC_DET = 1e-12
GAUSSIAN_SUPPORT_SIGMA_RADIUS = np.float32(3.0)


@dataclass(slots=True)
class ProjectedSplats:
    center_radius_depth: np.ndarray
    ellipse_conic: np.ndarray
    color_alpha: np.ndarray
    opacity_scale: np.ndarray
    valid: np.ndarray
    pos_local: np.ndarray
    inv_scale: np.ndarray
    quat: np.ndarray


def _linear_to_gamma_exact_np(value: np.ndarray) -> np.ndarray:
    tensor = np.asarray(value, dtype=np.float32)
    positive = np.where(
        tensor <= 0.0031308,
        12.92 * tensor,
        np.where(
            tensor < 1.0,
            1.055 * np.power(np.maximum(tensor, 0.0), 0.4166667) - 0.055,
            np.power(np.maximum(tensor, 0.0), 0.45454545),
        ),
    )
    return np.where(tensor <= 0.0, 0.0, positive).astype(np.float32, copy=False)


def quantize_depth(depth: float, near_depth: float, far_depth: float, depth_bits: int) -> np.uint32:
    max_value = (1 << depth_bits) - 1
    t = np.float32(np.clip((depth - near_depth) / max(far_depth - near_depth, 1e-6), 0.0, 1.0))
    return np.uint32(np.floor(np.float32(t * max_value) + np.float32(0.5)))


def _quat_rotate(v: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
    qv = q[1:4]
    vec = np.asarray(v, dtype=np.float32).reshape(3)
    return np.asarray(vec + 2.0 * np.cross(np.cross(vec, qv) + q[0] * vec, qv), dtype=np.float32)


def _quat_conj(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _solve_conic_renorm(points: np.ndarray, eps: float) -> np.ndarray | None:
    pts = np.asarray(points, dtype=np.float64).reshape(5, 2)
    sx = float(pts[0, 0] - pts[1, 0])
    sy = float(pts[1, 1] - pts[0, 1])
    if abs(sx) <= eps or abs(sy) <= eps:
        return None

    inv_sx = 1.0 / sx
    inv_sy = 1.0 / sy
    offset_x = float(pts[1, 0])
    offset_y = float(pts[0, 1])
    uv = np.empty((3, 2), dtype=np.float64)
    uv[:, 0] = (pts[2:, 0] - offset_x) * inv_sx
    uv[:, 1] = (pts[2:, 1] - offset_y) * inv_sy

    m00 = uv[0, 0] * uv[0, 0] - uv[0, 0]
    m01 = 2.0 * uv[0, 0] * uv[0, 1]
    m02 = uv[0, 1] * uv[0, 1] - uv[0, 1]
    r0 = uv[0, 0] + uv[0, 1] - 1.0
    m10 = uv[1, 0] * uv[1, 0] - uv[1, 0]
    m11 = 2.0 * uv[1, 0] * uv[1, 1]
    m12 = uv[1, 1] * uv[1, 1] - uv[1, 1]
    r1 = uv[1, 0] + uv[1, 1] - 1.0
    m20 = uv[2, 0] * uv[2, 0] - uv[2, 0]
    m21 = 2.0 * uv[2, 0] * uv[2, 1]
    m22 = uv[2, 1] * uv[2, 1] - uv[2, 1]
    r2 = uv[2, 0] + uv[2, 1] - 1.0

    if abs(m00) <= eps:
        return None
    inv_m00 = 1.0 / m00
    m01 *= inv_m00
    m02 *= inv_m00
    r0 *= inv_m00
    factor = m10
    m11 -= factor * m01
    m12 -= factor * m02
    r1 -= factor * r0
    factor = m20
    m21 -= factor * m01
    m22 -= factor * m02
    r2 -= factor * r0
    if abs(m11) <= eps:
        return None
    inv_m11 = 1.0 / m11
    m12 *= inv_m11
    r1 *= inv_m11
    factor = m21
    m22 -= factor * m12
    r2 -= factor * r1
    if abs(m22) <= eps:
        return None
    conic_c = r2 / m22
    conic_b = r1 - m12 * conic_c
    conic_a = r0 - m01 * conic_b - m02 * conic_c
    conic_d = -(conic_a + 1.0)
    conic_e = -(conic_c + 1.0)

    inv_sx2 = inv_sx * inv_sx
    inv_sy2 = inv_sy * inv_sy
    inv_sx_sy = inv_sx * inv_sy
    coeff_a = conic_a * inv_sx2
    coeff_b = conic_b * inv_sx_sy
    coeff_c = conic_c * inv_sy2
    coeff_d = -2.0 * coeff_a * offset_x - 2.0 * coeff_b * offset_y + conic_d * inv_sx
    coeff_e = -2.0 * coeff_b * offset_x - 2.0 * coeff_c * offset_y + conic_e * inv_sy
    coeff_f = coeff_a * offset_x * offset_x + 2.0 * coeff_b * offset_x * offset_y + coeff_c * offset_y * offset_y - conic_d * inv_sx * offset_x - conic_e * inv_sy * offset_y + 1.0
    if abs(coeff_f) <= eps:
        return None

    inv_f = -1.0 / coeff_f
    return np.array((coeff_a * inv_f, coeff_b * inv_f, coeff_c * inv_f, coeff_d * inv_f, coeff_e * inv_f), dtype=np.float32)


def _compute_outline_ellipse(
    world_pos: np.ndarray,
    inv_scale: np.ndarray,
    rotation: np.ndarray,
    camera: Camera,
    width: int,
    height: int,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    screen_center, screen_ok = camera.project_world_to_screen(world_pos, width, height)
    if not screen_ok:
        return None

    view_origin_local = _quat_rotate(camera.position - world_pos, rotation) * inv_scale
    view_distance = float(np.linalg.norm(view_origin_local))
    if view_distance <= 1.0 + ELLIPSE_EPS:
        return None

    view_dir_local = view_origin_local / view_distance
    tangent_circle_center = view_dir_local / view_distance
    tangent_circle_radius = math.sqrt(max(1.0 - 1.0 / (view_distance * view_distance), 0.0))
    tangent_axis = np.array((0.0, 0.0, 1.0), dtype=np.float32) if abs(float(view_dir_local[2])) < 0.999 else np.array((0.0, 1.0, 0.0), dtype=np.float32)
    tangent_basis_u = np.cross(tangent_axis, view_dir_local).astype(np.float32, copy=False)
    tangent_basis_u_norm = float(np.linalg.norm(tangent_basis_u))
    if tangent_basis_u_norm <= ELLIPSE_EPS or not np.isfinite(tangent_basis_u_norm):
        return None
    tangent_basis_u /= tangent_basis_u_norm
    tangent_basis_v = np.cross(view_dir_local, tangent_basis_u).astype(np.float32, copy=False)
    tangent_basis_v_norm = float(np.linalg.norm(tangent_basis_v))
    if tangent_basis_v_norm <= ELLIPSE_EPS or not np.isfinite(tangent_basis_v_norm):
        return None
    tangent_basis_v /= tangent_basis_v_norm

    outline_points = np.zeros((5, 2), dtype=np.float32)
    outline_min = np.full((2,), 1e30, dtype=np.float32)
    outline_max = np.full((2,), -1e30, dtype=np.float32)
    q_inv = _quat_conj(rotation)
    scale = 1.0 / np.maximum(inv_scale, np.float32(1e-12))
    for index in range(5):
        theta = np.float32(2.0 * np.pi * index / 5.0)
        local_point = tangent_circle_center + tangent_circle_radius * (np.cos(theta) * tangent_basis_u + np.sin(theta) * tangent_basis_v)
        world_point = world_pos + _quat_rotate(local_point * scale, q_inv)
        screen_point, point_ok = camera.project_world_to_screen(world_point, width, height)
        if not point_ok:
            return None
        outline_points[index] = screen_point
        outline_min = np.minimum(outline_min, screen_point)
        outline_max = np.maximum(outline_max, screen_point)

    outline_min = np.minimum(outline_min, screen_center.astype(np.float32, copy=False))
    outline_max = np.maximum(outline_max, screen_center.astype(np.float32, copy=False))
    if outline_max[0] < 0.0 or outline_min[0] >= width or outline_max[1] < 0.0 or outline_min[1] >= height:
        return None

    bbox_center = 0.5 * (outline_min + outline_max)
    bbox_half_extent = np.maximum(0.5 * (outline_max - outline_min), np.full((2,), ELLIPSE_EPS, dtype=np.float32))
    norm_points = (outline_points - bbox_center[None, :]) / bbox_half_extent[None, :]
    solution = _solve_conic_renorm(norm_points, ELLIPSE_EPS)
    if solution is None:
        return None

    conic_norm = solution[:3].astype(np.float64, copy=False)
    linear = solution[3:5].astype(np.float64, copy=False)
    det_a = float(conic_norm[0] * conic_norm[2] - conic_norm[1] * conic_norm[1])
    if det_a <= ELLIPSE_EPS:
        return None
    ellipse_center = 0.5 * np.array(
        (
            (conic_norm[1] * linear[1] - conic_norm[2] * linear[0]) / det_a,
            (conic_norm[1] * linear[0] - conic_norm[0] * linear[1]) / det_a,
        ),
        dtype=np.float64,
    )
    center_times_a = np.array(
        (
            conic_norm[0] * ellipse_center[0] + conic_norm[1] * ellipse_center[1],
            conic_norm[1] * ellipse_center[0] + conic_norm[2] * ellipse_center[1],
        ),
        dtype=np.float64,
    )
    center_scale = float(1.0 + np.dot(ellipse_center, center_times_a))
    if center_scale <= ELLIPSE_EPS:
        return None
    conic_norm /= center_scale
    trace = float(conic_norm[0] + conic_norm[2])
    det_a = float(conic_norm[0] * conic_norm[2] - conic_norm[1] * conic_norm[1])
    if det_a <= ELLIPSE_EPS or conic_norm[0] <= ELLIPSE_EPS or conic_norm[2] <= ELLIPSE_EPS:
        return None
    disc = math.sqrt(max(0.25 * trace * trace - det_a, 0.0))
    axis0 = 1.0 / math.sqrt(max(0.5 * trace + disc, ELLIPSE_EPS))
    axis1 = 1.0 / math.sqrt(max(0.5 * trace - disc, ELLIPSE_EPS))
    center_px = (bbox_center + ellipse_center.astype(np.float32, copy=False) * bbox_half_extent).astype(np.float32, copy=False)
    radius_px = float(max(axis0 * float(bbox_half_extent[0]), axis1 * float(bbox_half_extent[1])))
    conic = np.array(
        (
            conic_norm[0] / max(float(bbox_half_extent[0] * bbox_half_extent[0]), ELLIPSE_EPS),
            conic_norm[1] / max(float(bbox_half_extent[0] * bbox_half_extent[1]), ELLIPSE_EPS),
            conic_norm[2] / max(float(bbox_half_extent[1] * bbox_half_extent[1]), ELLIPSE_EPS),
        ),
        dtype=np.float32,
    )
    if not (np.isfinite(center_px).all() and np.isfinite(radius_px) and np.isfinite(conic).all() and radius_px > 0.0):
        return None
    return center_px, radius_px, conic


def project_splats(
    scene: GaussianScene,
    camera: Camera,
    width: int,
    height: int,
    radius_scale: float,
    alpha_cutoff: float = float(ALPHA_CUTOFF_DEFAULT),
) -> ProjectedSplats:
    count = scene.count
    center_radius_depth = np.zeros((count, 4), dtype=np.float32)
    ellipse_conic = np.zeros((count, 3), dtype=np.float32)
    view_dirs = np.asarray(camera.position, dtype=np.float32).reshape(1, 3) - np.asarray(scene.positions, dtype=np.float32)
    colors = evaluate_sh0_sh1(resolve_supported_sh_coeffs(scene.sh_coeffs, scene.colors), view_dirs)
    color_alpha = np.concatenate([colors, scene.opacities[:, None]], axis=1).astype(np.float32, copy=False)
    opacity_scale = np.ones((count,), dtype=np.float32)
    valid = np.zeros((count,), dtype=np.uint32)
    pos_local = np.zeros((count, 3), dtype=np.float32)
    inv_scale = np.zeros((count, 3), dtype=np.float32)
    quat = scene.rotations.astype(np.float32, copy=True)

    radius_scale = float(radius_scale)
    alpha_cutoff = float(alpha_cutoff)
    for index in range(count):
        world_pos = scene.positions[index].astype(np.float32, copy=False)
        rotation = quat[index]
        sigma = np.exp(scene.scales[index].astype(np.float32, copy=False))
        raster_scale = np.maximum(sigma * np.float32(radius_scale * float(GAUSSIAN_SUPPORT_SIGMA_RADIUS)), np.float32(1e-6))
        inv_scale[index] = 1.0 / raster_scale
        pos_local[index] = _quat_rotate(camera.position - world_pos, rotation) * inv_scale[index]

        camera_pos = camera.world_point_to_camera(world_pos)
        cam_distance = float(np.linalg.norm(world_pos - camera.position))
        depth_value = float(camera_pos[2])
        opacity = float(np.clip(scene.opacities[index], 0.0, 1.0))
        if opacity < alpha_cutoff:
            continue
        support_sigma_radius = math.sqrt(max(-2.0 * math.log(alpha_cutoff / max(opacity, alpha_cutoff)), 0.0))
        outline_scale = np.maximum(sigma * np.float32(radius_scale * support_sigma_radius), np.float32(1e-6))
        outline_inv_scale = 1.0 / outline_scale
        fitted = _compute_outline_ellipse(world_pos, outline_inv_scale, rotation, camera, width, height)
        if fitted is None:
            continue

        center_px, radius_px, conic = fitted
        radius_px = float(max(radius_px + 1.0, 1.0))
        center_radius_depth[index] = np.array((center_px[0], center_px[1], radius_px, cam_distance), dtype=np.float32)
        ellipse_conic[index] = conic
        visible = (
            depth_value > 1e-4
            and center_px[0] + radius_px >= 0.0
            and center_px[0] - radius_px < float(width)
            and center_px[1] + radius_px >= 0.0
            and center_px[1] - radius_px < float(height)
        )
        valid[index] = np.uint32(1 if visible else 0)

    return ProjectedSplats(
        center_radius_depth=center_radius_depth,
        ellipse_conic=ellipse_conic,
        color_alpha=color_alpha,
        opacity_scale=opacity_scale,
        valid=valid,
        pos_local=pos_local,
        inv_scale=inv_scale,
        quat=quat,
    )


def _try_prepare_conic_for_binning(conic: np.ndarray, radius: float, half_tile: float) -> tuple[bool, np.ndarray]:
    det = float(conic[0] * conic[2] - conic[1] * conic[1])
    bbox = np.array([radius, radius], dtype=np.float32)
    if not (np.isfinite(conic).all() and float(conic[0]) > 1e-10 and float(conic[2]) > 1e-10 and det > MIN_CONIC_DET):
        return False, bbox
    trace = float(conic[0] + conic[2])
    disc = float(np.sqrt(max(0.25 * trace * trace - det, 0.0)))
    eig_min, eig_max = 0.5 * trace - disc, 0.5 * trace + disc
    max_axis = float(1.0 / np.sqrt(max(eig_min, 1e-20)))
    min_axis = float(1.0 / np.sqrt(max(eig_max, 1e-20)))
    axis_limit = max(radius, 1.0) * 2.0 + half_tile
    if not (np.isfinite(max_axis) and np.isfinite(min_axis) and eig_min > 0.0 and eig_max > 0.0 and max_axis <= axis_limit and min_axis >= 0.125):
        return False, bbox
    extent = np.array([np.sqrt(max(float(conic[2]) / det, 0.0)), np.sqrt(max(float(conic[0]) / det, 0.0))], dtype=np.float32)
    return (False, bbox) if not np.isfinite(extent).all() else (True, np.clip(extent, 1e-4, radius).astype(np.float32))


def _eval_conic(conic: np.ndarray, x: float, y: float) -> float:
    return float(conic[0]) * x * x + 2.0 * float(conic[1]) * x * y + float(conic[2]) * y * y


def _min_conic_over_tile_box(conic: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float:
    values = [_eval_conic(conic, x, y) for x in (x0, x1) for y in (y0, y1)]
    if x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1:
        return 0.0
    a, b, c = map(float, conic)
    if c > 1e-12:
        values.extend(_eval_conic(conic, x, float(np.clip(-b * x / c, y0, y1))) for x in (x0, x1))
    if a > 1e-12:
        values.extend(_eval_conic(conic, float(np.clip(-b * y / a, x0, x1)), y) for y in (y0, y1))
    return float(min(values))


def _tile_box(center: tuple[float, float], scan_along_x: bool, tile_size: int, line_tile: int, minor_tile: int) -> tuple[float, float, float, float]:
    major = np.array([minor_tile, line_tile], dtype=np.float32) if scan_along_x else np.array([line_tile, minor_tile], dtype=np.float32)
    lo = major * float(tile_size) - np.asarray(center, dtype=np.float32)
    hi = (major + 1.0) * float(tile_size) - np.asarray(center, dtype=np.float32)
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _tile_intersects_ellipse(center: tuple[float, float], conic: np.ndarray, scan_along_x: bool, tile_size: int, line_tile: int, minor_tile: int) -> bool:
    return _min_conic_over_tile_box(conic, *_tile_box(center, scan_along_x, tile_size, line_tile, minor_tile)) <= 1.0 + ELLIPSE_EPS


def _compute_scanline_tile_span_universal(center: tuple[float, float], conic: np.ndarray, scan_along_x: bool, tile_size: int, line_coord_tile: int, min_minor_tile: int, max_minor_tile: int) -> tuple[bool, int, int]:
    hits = [minor for minor in range(max(min_minor_tile, 0), max_minor_tile + 1) if _tile_intersects_ellipse(center, conic, scan_along_x, tile_size, line_coord_tile, minor)]
    return (False, 0, 0) if not hits else (True, hits[0], hits[-1] - hits[0] + 1)


def _iter_spans(center: tuple[float, float], conic: np.ndarray, scan_along_x: bool, tile_size: int, primary_lo: int, primary_hi: int, minor_lo: int, minor_hi: int):
    for primary in range(primary_lo, primary_hi + 1):
        has_span, minor_start, count = _compute_scanline_tile_span_universal(center, conic, scan_along_x, tile_size, primary, minor_lo, minor_hi)
        if has_span:
            yield primary, minor_start, count


def _write_span(keys: np.ndarray, values: np.ndarray, write_index: int, count: int, tile_width: int, splat_id: int, scan_along_x: bool, primary: int, minor_start: int) -> int:
    for offset in range(count):
        tile_x, tile_y = (minor_start + offset, primary) if scan_along_x else (primary, minor_start + offset)
        tile_id = tile_y * tile_width + tile_x
        keys[write_index] = np.uint32(tile_id)
        values[write_index] = np.uint32(splat_id)
        write_index += 1
    return write_index


def build_tile_key_value_pairs(projected: ProjectedSplats, tile_width: int, tile_height: int, tile_size: int, max_list_entries: int) -> tuple[np.ndarray, np.ndarray, int]:
    keys, values, counter = np.zeros((max_list_entries,), dtype=np.uint32), np.zeros((max_list_entries,), dtype=np.uint32), 0
    visible_ids = np.flatnonzero(projected.valid != 0)
    if visible_ids.size == 0:
        return keys, values, counter
    depths = projected.center_radius_depth[visible_ids, 3]
    ordered_ids = visible_ids[np.argsort(depths, kind="stable")]
    for splat_id in ordered_ids.tolist():
        cx, cy, radius, _ = projected.center_radius_depth[splat_id]
        if projected.valid[splat_id] == 0:
            continue
        use_conic, bbox_extent = _try_prepare_conic_for_binning(projected.ellipse_conic[splat_id], float(radius), 0.5 * float(tile_size))
        if not use_conic:
            continue
        extent = bbox_extent + 0.5 * float(tile_size)
        min_x = max(int(np.floor((cx - float(extent[0])) / float(tile_size))), 0)
        max_x = min(int(np.ceil((cx + float(extent[0])) / float(tile_size))), tile_width - 1)
        min_y = max(int(np.floor((cy - float(extent[1])) / float(tile_size))), 0)
        max_y = min(int(np.ceil((cy + float(extent[1])) / float(tile_size))), tile_height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        scan_along_x = bool(float(bbox_extent[0]) > float(bbox_extent[1]) or (abs(float(bbox_extent[0]) - float(bbox_extent[1])) < 1e-6 and (max_x - min_x) >= (max_y - min_y)))
        primary_lo, primary_hi = (min_y, max_y) if scan_along_x else (min_x, max_x)
        minor_lo, minor_hi = (min_x, max_x) if scan_along_x else (min_y, max_y)
        spans = tuple(_iter_spans((float(cx), float(cy)), projected.ellipse_conic[splat_id], scan_along_x, tile_size, primary_lo, primary_hi, minor_lo, minor_hi))
        total_count = sum(count for _, _, count in spans)
        if total_count <= 0:
            continue
        base_index, counter = counter, counter + total_count
        if base_index >= max_list_entries:
            continue
        write_limit, write_index, written = min(total_count, max_list_entries - base_index), base_index, 0
        for primary, minor_start, count in spans:
            if written >= write_limit:
                break
            count = min(count, write_limit - written)
            write_index = _write_span(keys, values, write_index, count, tile_width, splat_id, scan_along_x, primary, minor_start)
            written += count
    return keys, values, counter


def sort_key_values(keys: np.ndarray, values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(keys[:count], kind="stable")
    return keys[:count][order].copy(), values[:count][order].copy()


def build_tile_ranges(sorted_keys: np.ndarray, sorted_count: int, tile_count: int) -> np.ndarray:
    ranges = np.full((tile_count, 2), fill_value=np.uint32(0xFFFFFFFF), dtype=np.uint32)
    ranges[:, 1] = 0
    if sorted_count == 0:
        return ranges
    tiles = sorted_keys[:sorted_count]
    boundaries = np.flatnonzero(np.r_[True, tiles[1:] != tiles[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranges[int(tiles[start])] = (np.uint32(start), np.uint32(end))
    return ranges


def rasterize(projected: ProjectedSplats, sorted_values: np.ndarray, tile_ranges: np.ndarray, camera: Camera, width: int, height: int, tile_size: int, tile_width: int, background: np.ndarray, alpha_cutoff: float, max_splat_steps: int, transmittance_threshold: float) -> np.ndarray:
    output = np.zeros((height, width, 4), dtype=np.float32)
    bg_display = _linear_to_gamma_exact_np(background.reshape(3))

    def quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        return v + 2.0 * np.cross(np.cross(v, q[1:4]) + q[0] * v, q[1:4])

    for py in range(height):
        tile_y = py // tile_size
        for px in range(width):
            tile = tile_y * tile_width + px // tile_size
            start, end = tile_ranges[tile]
            if start == np.uint32(0xFFFFFFFF) or int(end) <= int(start):
                output[py, px, :3], output[py, px, 3] = bg_display, 1.0
                continue
            ray = camera.screen_to_world_ray(np.array([float(px) + 0.5, float(py) + 0.5], dtype=np.float32), width, height)
            accum, trans = np.zeros((3,), dtype=np.float32), 1.0
            for splat_id in map(int, sorted_values[int(start): int(end)][:max_splat_steps]):
                if projected.valid[splat_id] == 0:
                    continue
                rd_local = quat_rotate(ray, projected.quat[splat_id]) * projected.inv_scale[splat_id]
                denom = float(np.dot(rd_local, rd_local))
                if denom <= 1e-10:
                    continue
                closest = projected.pos_local[splat_id] + rd_local * float(np.dot(rd_local, -projected.pos_local[splat_id]) / denom)
                rho = float(np.linalg.norm(closest))
                if rho >= 1.0:
                    continue
                coverage = float((1.0 - rho) ** 2 * (1.0 + 2.0 * rho))
                alpha = float(np.clip(projected.color_alpha[splat_id, 3] * coverage, 0.0, 1.0))
                if alpha < alpha_cutoff:
                    continue
                accum += trans * alpha * projected.color_alpha[splat_id, :3]
                trans *= 1.0 - alpha
                if trans < transmittance_threshold:
                    break
            output[py, px, :3] = _linear_to_gamma_exact_np(accum + trans * background)
            output[py, px, 3] = 1.0 - trans
    return output

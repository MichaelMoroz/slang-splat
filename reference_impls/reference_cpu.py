from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.scene.gaussian_scene import GaussianScene
from src.renderer.camera import Camera
from .projection_sampled5_mvee_reference import project_splats_sampled5_mvee

ELLIPSE_EPS = 1e-5
MIN_CONIC_DET = 1e-12
SCALE_AREA_PROXY_POWER = np.float32(2.0 / 3.0)
SCALE_FLOOR_SMOOTH_RATIO = np.float32(1.0)
GAUSSIAN_SUPPORT_SIGMA_RADIUS = np.float32(2.6)


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


def quantize_depth(depth: float, near_depth: float, far_depth: float, depth_bits: int) -> np.uint32:
    max_value = (1 << depth_bits) - 1
    t = np.float32(np.clip((depth - near_depth) / max(far_depth - near_depth, 1e-6), 0.0, 1.0))
    return np.uint32(np.floor(np.float32(t * max_value) + np.float32(0.5)))


def _smooth_max_scale(raw_scale: np.ndarray, min_scale: np.ndarray) -> np.ndarray:
    smoothness = np.maximum(np.asarray(min_scale, dtype=np.float32) * SCALE_FLOOR_SMOOTH_RATIO, np.float32(1e-6))
    delta = np.asarray(raw_scale, dtype=np.float32) - np.asarray(min_scale, dtype=np.float32)
    return (0.5 * (np.asarray(raw_scale, dtype=np.float32) + np.asarray(min_scale, dtype=np.float32) + np.sqrt(delta * delta + smoothness * smoothness))).astype(np.float32)


def project_splats(scene: GaussianScene, camera: Camera, width: int, height: int, radius_scale: float) -> ProjectedSplats:
    projected = project_splats_sampled5_mvee(
        scene=scene,
        camera=camera,
        width=width,
        height=height,
        radius_scale=radius_scale,
        mvee_iters=6,
        safety_scale=1.0,
        radius_pad_px=1.0,
        mvee_eps=1e-6,
        distortion_k1=0.0,
        distortion_k2=0.0,
    )
    radius = np.maximum(projected.center_radius_depth[:, 2].astype(np.float32), 1e-3)
    inv_r2 = 1.0 / np.maximum(radius * radius, 1e-6)
    conic = np.stack((inv_r2, np.zeros_like(inv_r2), inv_r2), axis=1).astype(np.float32)
    axes = projected.ellipse_center_axes.astype(np.float32)
    major, minor, angle = axes[:, 2], axes[:, 3], axes[:, 4]
    valid = (major > 1e-6) & (minor > 1e-6) & np.isfinite(major) & np.isfinite(minor) & np.isfinite(angle)
    if np.any(valid):
        major = major[valid] * (radius[valid] / np.maximum(major[valid], 1e-6))
        minor = minor[valid] * (radius[valid] / np.maximum(axes[valid, 2], 1e-6))
        c, s = np.cos(angle[valid]), np.sin(angle[valid])
        inv_a2, inv_b2 = 1.0 / (major * major), 1.0 / (minor * minor)
        conic[valid] = np.stack((c * c * inv_a2 + s * s * inv_b2, c * s * (inv_a2 - inv_b2), s * s * inv_a2 + c * c * inv_b2), axis=1).astype(np.float32)
    _, _, forward = camera.basis()
    position_delta = scene.positions.astype(np.float32) - camera.position.astype(np.float32)
    depth = np.sum(position_delta * forward[None, :].astype(np.float32), axis=1, dtype=np.float32)
    min_scale_world = np.asarray([camera.pixel_world_size_max(float(value), width, height) for value in depth], dtype=np.float32)
    raw_scale = np.maximum(np.exp(scene.scales.astype(np.float32)) * (np.float32(radius_scale) * GAUSSIAN_SUPPORT_SIGMA_RADIUS), np.float32(1e-6))
    clamped_scale = _smooth_max_scale(raw_scale, np.repeat(min_scale_world[:, None], 3, axis=1))
    raw_area = np.power(np.maximum(np.prod(raw_scale, axis=1, dtype=np.float32), np.float32(1e-6)), SCALE_AREA_PROXY_POWER).astype(np.float32)
    clamped_area = np.power(np.maximum(np.prod(clamped_scale, axis=1, dtype=np.float32), np.float32(1e-6)), SCALE_AREA_PROXY_POWER).astype(np.float32)
    opacity_scale = np.minimum(raw_area / np.maximum(clamped_area, np.float32(1e-6)), np.float32(1.0)).astype(np.float32)
    return ProjectedSplats(
        center_radius_depth=projected.center_radius_depth,
        ellipse_conic=conic,
        color_alpha=projected.color_alpha,
        opacity_scale=opacity_scale,
        valid=projected.valid,
        pos_local=projected.pos_local,
        inv_scale=projected.inv_scale,
        quat=projected.quat,
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


def _write_span(keys: np.ndarray, values: np.ndarray, write_index: int, count: int, depth_bits: int, tile_width: int, splat_id: int, depth_key: np.uint32, scan_along_x: bool, primary: int, minor_start: int) -> int:
    for offset in range(count):
        tile_x, tile_y = (minor_start + offset, primary) if scan_along_x else (primary, minor_start + offset)
        tile_id = tile_y * tile_width + tile_x
        keys[write_index] = np.uint32((tile_id << depth_bits) | int(depth_key))
        values[write_index] = np.uint32(splat_id)
        write_index += 1
    return write_index


def build_tile_key_value_pairs(projected: ProjectedSplats, tile_width: int, tile_height: int, tile_size: int, depth_bits: int, near_depth: float, far_depth: float, max_list_entries: int) -> tuple[np.ndarray, np.ndarray, int]:
    keys, values, counter = np.zeros((max_list_entries,), dtype=np.uint32), np.zeros((max_list_entries,), dtype=np.uint32), 0
    for splat_id, (cx, cy, radius, depth) in enumerate(projected.center_radius_depth):
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
        depth_key = quantize_depth(float(depth), near_depth, far_depth, depth_bits)
        for primary, minor_start, count in spans:
            if written >= write_limit:
                break
            count = min(count, write_limit - written)
            write_index = _write_span(keys, values, write_index, count, depth_bits, tile_width, splat_id, depth_key, scan_along_x, primary, minor_start)
            written += count
    return keys, values, counter


def sort_key_values(keys: np.ndarray, values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(keys[:count], kind="stable")
    return keys[:count][order].copy(), values[:count][order].copy()


def build_tile_ranges(sorted_keys: np.ndarray, sorted_count: int, tile_count: int, depth_bits: int) -> np.ndarray:
    ranges = np.full((tile_count, 2), fill_value=np.uint32(0xFFFFFFFF), dtype=np.uint32)
    ranges[:, 1] = 0
    if sorted_count == 0:
        return ranges
    tiles = sorted_keys[:sorted_count] >> np.uint32(depth_bits)
    boundaries = np.flatnonzero(np.r_[True, tiles[1:] != tiles[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranges[int(tiles[start])] = (np.uint32(start), np.uint32(end))
    return ranges


def rasterize(projected: ProjectedSplats, sorted_values: np.ndarray, tile_ranges: np.ndarray, camera: Camera, width: int, height: int, tile_size: int, tile_width: int, background: np.ndarray, alpha_cutoff: float, max_splat_steps: int, transmittance_threshold: float) -> np.ndarray:
    output = np.zeros((height, width, 4), dtype=np.float32)
    right, up, forward = camera.basis()
    fx, fy = camera.focal_pixels_xy(width, height)
    cx, cy = camera.principal_point(width, height)
    bg_linear = np.power(np.clip(background.reshape(3), 0.0, None), 2.2).astype(np.float32)

    def quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        return v + 2.0 * np.cross(np.cross(v, q[1:4]) + q[0] * v, q[1:4])

    for py in range(height):
        tile_y = py // tile_size
        uv_y = (float(py) + 0.5 - float(cy)) / float(fy)
        for px in range(width):
            tile = tile_y * tile_width + px // tile_size
            start, end = tile_ranges[tile]
            if start == np.uint32(0xFFFFFFFF) or int(end) <= int(start):
                output[py, px, :3], output[py, px, 3] = bg_linear, 1.0
                continue
            ray = forward + ((float(px) + 0.5 - float(cx)) / float(fx)) * right + uv_y * up
            ray = ray / max(np.linalg.norm(ray), 1e-8)
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
                base_alpha = float(np.clip(projected.color_alpha[splat_id, 3] * coverage, 0.0, 1.0))
                if base_alpha < alpha_cutoff:
                    continue
                alpha = float(np.clip(base_alpha * projected.opacity_scale[splat_id], 0.0, 1.0))
                accum += trans * alpha * projected.color_alpha[splat_id, :3]
                trans *= 1.0 - alpha
                if trans < transmittance_threshold:
                    break
            output[py, px, :3] = np.power(np.clip(accum + trans * background, 0.0, None), 2.2).astype(np.float32)
            output[py, px, 3] = 1.0 - trans
    return output

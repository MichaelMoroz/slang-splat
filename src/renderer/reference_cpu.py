from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .projection_sampled5_mvee_reference import project_splats_sampled5_mvee
from ..scene.gaussian_scene import GaussianScene


@dataclass(slots=True)
class ProjectedSplats:
    center_radius_depth: np.ndarray
    ellipse_conic: np.ndarray
    color_alpha: np.ndarray
    valid: np.ndarray
    pos_local: np.ndarray
    inv_scale: np.ndarray
    quat: np.ndarray


def quantize_depth(depth: float, near_depth: float, far_depth: float, depth_bits: int) -> np.uint32:
    max_value = (1 << depth_bits) - 1
    t = np.float32(np.clip((depth - near_depth) / max(far_depth - near_depth, 1e-6), 0.0, 1.0))
    value = np.float32(t * np.float32(max_value))
    return np.uint32(np.floor(value + np.float32(0.5)))


def project_splats(
    scene: GaussianScene,
    camera: Camera,
    width: int,
    height: int,
    radius_scale: float,
    max_splat_radius_px: float = 512.0,
) -> ProjectedSplats:
    projected = project_splats_sampled5_mvee(
        scene=scene,
        camera=camera,
        width=width,
        height=height,
        radius_scale=radius_scale,
        max_splat_radius_px=max_splat_radius_px,
        mvee_iters=6,
        safety_scale=1.0,
        radius_pad_px=1.0,
        mvee_eps=1e-6,
        distortion_k1=0.0,
        distortion_k2=0.0,
    )
    ellipse_conic = np.zeros((scene.count, 3), dtype=np.float32)
    for i in range(scene.count):
        radius = float(projected.center_radius_depth[i, 2])
        inv_r2 = 1.0 / max(radius * radius, 1e-6)
        conic = np.array([inv_r2, 0.0, inv_r2], dtype=np.float32)
        axes_angle = projected.ellipse_center_axes[i]
        axis_major = float(axes_angle[2])
        axis_minor = float(axes_angle[3])
        angle = float(axes_angle[4])
        if axis_major > 1e-6 and axis_minor > 1e-6 and np.isfinite(axis_major) and np.isfinite(axis_minor):
            conic_scale = radius / max(axis_major, 1e-6)
            axis_major = axis_major * conic_scale
            axis_minor = axis_minor * conic_scale
            c = float(np.cos(angle))
            s = float(np.sin(angle))
            inv_a2 = 1.0 / (axis_major * axis_major)
            inv_b2 = 1.0 / (axis_minor * axis_minor)
            q00 = c * c * inv_a2 + s * s * inv_b2
            q01 = c * s * (inv_a2 - inv_b2)
            q11 = s * s * inv_a2 + c * c * inv_b2
            conic = np.array([q00, q01, q11], dtype=np.float32)
        ellipse_conic[i, :] = conic
    return ProjectedSplats(
        center_radius_depth=projected.center_radius_depth,
        ellipse_conic=ellipse_conic,
        color_alpha=projected.color_alpha,
        valid=projected.valid,
        pos_local=projected.pos_local,
        inv_scale=projected.inv_scale,
        quat=projected.quat,
    )


def _try_prepare_conic_for_binning(conic: np.ndarray, radius: float, half_tile: float) -> tuple[bool, np.ndarray]:
    bbox_extent = np.array([radius, radius], dtype=np.float32)
    det_conic = float(conic[0] * conic[2] - conic[1] * conic[1])
    use_conic = (
        np.isfinite(conic).all()
        and float(conic[0]) > 1e-10
        and float(conic[2]) > 1e-10
        and det_conic > 1e-12
    )
    if not use_conic:
        return False, bbox_extent

    trace = float(conic[0] + conic[2])
    disc = float(np.sqrt(max(0.25 * trace * trace - det_conic, 0.0)))
    eig_min = 0.5 * trace - disc
    eig_max = 0.5 * trace + disc
    max_axis = float(1.0 / np.sqrt(max(eig_min, 1e-20)))
    min_axis = float(1.0 / np.sqrt(max(eig_max, 1e-20)))
    radius_safe = max(radius, 1.0)
    axis_limit = radius_safe * 2.0 + half_tile
    use_conic = (
        np.isfinite(max_axis)
        and np.isfinite(min_axis)
        and eig_min > 0.0
        and eig_max > 0.0
        and max_axis <= axis_limit
        and min_axis >= 0.125
    )
    if not use_conic:
        return False, bbox_extent

    x_extent = float(np.sqrt(max(float(conic[2]) / det_conic, 0.0)))
    y_extent = float(np.sqrt(max(float(conic[0]) / det_conic, 0.0)))
    if not np.isfinite(x_extent) or not np.isfinite(y_extent):
        return False, bbox_extent
    bbox_extent = np.clip(np.array([x_extent, y_extent], dtype=np.float32), 1e-4, radius).astype(np.float32)
    return True, bbox_extent


def _eval_conic(conic: np.ndarray, x: float, y: float) -> float:
    return float(conic[0]) * x * x + 2.0 * float(conic[1]) * x * y + float(conic[2]) * y * y


def _min_conic_over_tile_box(conic: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float:
    min_val = min(
        _eval_conic(conic, x0, y0),
        _eval_conic(conic, x1, y0),
        _eval_conic(conic, x0, y1),
        _eval_conic(conic, x1, y1),
    )
    if x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1:
        return 0.0

    a = float(conic[0])
    b = float(conic[1])
    c = float(conic[2])
    if c > 1e-12:
        y_at_x0 = float(np.clip(-b * x0 / c, y0, y1))
        y_at_x1 = float(np.clip(-b * x1 / c, y0, y1))
        min_val = min(min_val, _eval_conic(conic, x0, y_at_x0), _eval_conic(conic, x1, y_at_x1))
    if a > 1e-12:
        x_at_y0 = float(np.clip(-b * y0 / a, x0, x1))
        x_at_y1 = float(np.clip(-b * y1 / a, x0, x1))
        min_val = min(min_val, _eval_conic(conic, x_at_y0, y0), _eval_conic(conic, x_at_y1, y1))
    return float(min_val)


def _tile_intersects_ellipse(
    center: tuple[float, float],
    radius: float,
    conic: np.ndarray,
    use_conic: bool,
    scan_along_x: bool,
    tile_size: int,
    line_coord_tile: int,
    minor_coord_tile: int,
) -> bool:
    cx, cy = center
    ts = float(tile_size)
    if scan_along_x:
        x0 = float(minor_coord_tile) * ts - cx
        x1 = float(minor_coord_tile + 1) * ts - cx
        y0 = float(line_coord_tile) * ts - cy
        y1 = float(line_coord_tile + 1) * ts - cy
    else:
        x0 = float(line_coord_tile) * ts - cx
        x1 = float(line_coord_tile + 1) * ts - cx
        y0 = float(minor_coord_tile) * ts - cy
        y1 = float(minor_coord_tile + 1) * ts - cy

    if use_conic:
        return _min_conic_over_tile_box(conic, x0, x1, y0, y1) <= 1.0 + 1e-5

    qx = float(np.clip(0.0, x0, x1))
    qy = float(np.clip(0.0, y0, y1))
    return (qx * qx + qy * qy) <= (radius * radius)


def _compute_scanline_tile_span_universal(
    center: tuple[float, float],
    radius: float,
    bbox_extent: np.ndarray,
    conic: np.ndarray,
    use_conic: bool,
    scan_along_x: bool,
    tile_size: int,
    line_coord_tile: int,
    min_minor_tile: int,
    max_minor_tile: int,
) -> tuple[bool, int, int]:
    _ = bbox_extent
    first = -1
    last = -1
    for minor in range(min_minor_tile, max_minor_tile + 1):
        if minor < 0:
            continue
        if not _tile_intersects_ellipse(
            center=center,
            radius=radius,
            conic=conic,
            use_conic=use_conic,
            scan_along_x=scan_along_x,
            tile_size=tile_size,
            line_coord_tile=line_coord_tile,
            minor_coord_tile=minor,
        ):
            continue
        if first < 0:
            first = minor
        last = minor
    if first < 0 or last < first:
        return False, 0, 0
    return True, first, last - first + 1


def build_tile_key_value_pairs(
    projected: ProjectedSplats,
    tile_width: int,
    tile_height: int,
    tile_size: int,
    depth_bits: int,
    near_depth: float,
    far_depth: float,
    max_list_entries: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    keys = np.zeros((max_list_entries,), dtype=np.uint32)
    values = np.zeros((max_list_entries,), dtype=np.uint32)
    counter = 0
    count = projected.center_radius_depth.shape[0]
    for splat_id in range(count):
        if projected.valid[splat_id] == 0:
            continue
        cx, cy, radius, depth = projected.center_radius_depth[splat_id]
        conic = projected.ellipse_conic[splat_id]
        tile_size_f = float(tile_size)
        half_tile = 0.5 * tile_size_f
        use_conic, bbox_extent = _try_prepare_conic_for_binning(conic, float(radius), half_tile)
        bounds_extent = bbox_extent + half_tile
        min_x = max(int(np.floor((cx - float(bounds_extent[0])) / tile_size_f)), 0)
        max_x = min(int(np.ceil((cx + float(bounds_extent[0])) / tile_size_f)), tile_width - 1)
        min_y = max(int(np.floor((cy - float(bounds_extent[1])) / tile_size_f)), 0)
        max_y = min(int(np.ceil((cy + float(bounds_extent[1])) / tile_size_f)), tile_height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        span_x = max_x - min_x + 1
        span_y = max_y - min_y + 1
        scan_along_x = float(bbox_extent[0]) >= float(bbox_extent[1])
        if abs(float(bbox_extent[0]) - float(bbox_extent[1])) < 1e-6:
            scan_along_x = span_x >= span_y
        depth_key = quantize_depth(float(depth), near_depth, far_depth, depth_bits)
        total_count = 0
        if scan_along_x:
            for tile_y in range(min_y, max_y + 1):
                has_span, _, line_count = _compute_scanline_tile_span_universal(
                    center=(float(cx), float(cy)),
                    radius=float(radius),
                    bbox_extent=bbox_extent,
                    conic=conic,
                    use_conic=use_conic,
                    scan_along_x=True,
                    tile_size=tile_size,
                    line_coord_tile=tile_y,
                    min_minor_tile=min_x,
                    max_minor_tile=max_x,
                )
                if has_span:
                    total_count += line_count
        else:
            for tile_x in range(min_x, max_x + 1):
                has_span, _, line_count = _compute_scanline_tile_span_universal(
                    center=(float(cx), float(cy)),
                    radius=float(radius),
                    bbox_extent=bbox_extent,
                    conic=conic,
                    use_conic=use_conic,
                    scan_along_x=False,
                    tile_size=tile_size,
                    line_coord_tile=tile_x,
                    min_minor_tile=min_y,
                    max_minor_tile=max_y,
                )
                if has_span:
                    total_count += line_count
        if total_count <= 0:
            continue

        base_index = counter
        counter += total_count
        if base_index >= max_list_entries:
            continue

        write_count = min(total_count, max_list_entries - base_index)
        written = 0
        write_index = base_index
        if scan_along_x:
            for tile_y in range(min_y, max_y + 1):
                if written >= write_count:
                    break
                has_span, line_min_x, line_count = _compute_scanline_tile_span_universal(
                    center=(float(cx), float(cy)),
                    radius=float(radius),
                    bbox_extent=bbox_extent,
                    conic=conic,
                    use_conic=use_conic,
                    scan_along_x=True,
                    tile_size=tile_size,
                    line_coord_tile=tile_y,
                    min_minor_tile=min_x,
                    max_minor_tile=max_x,
                )
                if not has_span or line_min_x >= tile_width or tile_y >= tile_height:
                    continue
                line_write_count = min(line_count, write_count - written)
                row_base = tile_y * tile_width + line_min_x
                for tile_offset_x in range(line_write_count):
                    tile_id = row_base + tile_offset_x
                    if tile_id >= tile_width * tile_height:
                        continue
                    key = np.uint32((tile_id << depth_bits) | int(depth_key))
                    keys[write_index] = key
                    values[write_index] = np.uint32(splat_id)
                    write_index += 1
                written += line_write_count
        else:
            for tile_x in range(min_x, max_x + 1):
                if written >= write_count:
                    break
                has_span, line_min_y, line_count = _compute_scanline_tile_span_universal(
                    center=(float(cx), float(cy)),
                    radius=float(radius),
                    bbox_extent=bbox_extent,
                    conic=conic,
                    use_conic=use_conic,
                    scan_along_x=False,
                    tile_size=tile_size,
                    line_coord_tile=tile_x,
                    min_minor_tile=min_y,
                    max_minor_tile=max_y,
                )
                if not has_span or line_min_y >= tile_height or tile_x >= tile_width:
                    continue
                line_write_count = min(line_count, write_count - written)
                for tile_offset_y in range(line_write_count):
                    tile_y = line_min_y + tile_offset_y
                    if tile_y >= tile_height:
                        continue
                    tile_id = tile_y * tile_width + tile_x
                    if tile_id >= tile_width * tile_height:
                        continue
                    key = np.uint32((tile_id << depth_bits) | int(depth_key))
                    keys[write_index] = key
                    values[write_index] = np.uint32(splat_id)
                    write_index += 1
                written += line_write_count
    return keys, values, counter


def sort_key_values(keys: np.ndarray, values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(keys[:count], kind="stable")
    return keys[:count][order].copy(), values[:count][order].copy()


def build_tile_ranges(
    sorted_keys: np.ndarray,
    sorted_count: int,
    tile_count: int,
    depth_bits: int,
) -> np.ndarray:
    ranges = np.full((tile_count, 2), fill_value=np.uint32(0xFFFFFFFF), dtype=np.uint32)
    ranges[:, 1] = 0
    if sorted_count == 0:
        return ranges
    tiles = sorted_keys[:sorted_count] >> np.uint32(depth_bits)
    start = 0
    while start < sorted_count:
        tile = int(tiles[start])
        end = start + 1
        while end < sorted_count and int(tiles[end]) == tile:
            end += 1
        ranges[tile, 0] = np.uint32(start)
        ranges[tile, 1] = np.uint32(end)
        start = end
    return ranges


def rasterize(
    projected: ProjectedSplats,
    sorted_values: np.ndarray,
    tile_ranges: np.ndarray,
    camera: Camera,
    width: int,
    height: int,
    tile_size: int,
    tile_width: int,
    background: np.ndarray,
    alpha_cutoff: float,
    max_splat_steps: int,
    transmittance_threshold: float,
) -> np.ndarray:
    output = np.zeros((height, width, 4), dtype=np.float32)
    output[:, :, :3] = background.reshape(1, 1, 3)
    right, up, forward = camera.basis()
    focal = camera.focal_pixels(height)

    def quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        qv = q[1:4]
        return v + 2.0 * np.cross(np.cross(v, qv) + q[0] * v, qv)

    for py in range(height):
        tile_y = py // tile_size
        for px in range(width):
            tile_x = px // tile_size
            tile = tile_y * tile_width + tile_x
            start, end = tile_ranges[tile]
            if start == np.uint32(0xFFFFFFFF) or int(end) <= int(start):
                bg_linear = np.power(np.clip(background, 0.0, None), 2.2).astype(np.float32)
                output[py, px, :3] = bg_linear
                output[py, px, 3] = 1.0
                continue
            accum = np.zeros((3,), dtype=np.float32)
            trans = 1.0
            uv_x = (float(px) + 0.5 - 0.5 * float(width)) / float(focal)
            uv_y = (float(py) + 0.5 - 0.5 * float(height)) / float(focal)
            ray = forward + uv_x * right + uv_y * up
            ray = ray / max(np.linalg.norm(ray), 1e-8)
            steps = 0
            for item in range(int(start), int(end)):
                if steps >= max_splat_steps:
                    break
                steps += 1
                splat_id = int(sorted_values[item])
                if projected.valid[splat_id] == 0:
                    continue
                ro = projected.pos_local[splat_id]
                invs = projected.inv_scale[splat_id]
                q = projected.quat[splat_id]
                rd_local = quat_rotate(ray, q) * invs
                denom = float(np.dot(rd_local, rd_local))
                if denom <= 1e-10:
                    continue
                t_closest = float(np.dot(rd_local, -ro) / denom)
                if t_closest <= 0.0:
                    continue
                closest = ro + rd_local * t_closest
                rho = float(np.linalg.norm(closest))
                if rho >= 1.0:
                    continue
                t = float(np.clip(1.0 - rho, 0.0, 1.0))
                coverage = t * t * (3.0 - 2.0 * t)
                alpha = float(projected.color_alpha[splat_id, 3]) * coverage
                if alpha < alpha_cutoff:
                    continue
                alpha = float(np.clip(alpha, 0.0, 1.0))
                accum += trans * alpha * projected.color_alpha[splat_id, :3]
                trans *= 1.0 - alpha
                if trans < transmittance_threshold:
                    break
            final_color = accum + trans * background
            output[py, px, :3] = np.power(np.clip(final_color, 0.0, None), 2.2).astype(np.float32)
            output[py, px, 3] = 1.0 - trans
    return output

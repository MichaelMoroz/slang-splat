from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from ..scene.gaussian_scene import GaussianScene


@dataclass(slots=True)
class ProjectedSplats:
    center_radius_depth: np.ndarray
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
    max_splat_radius_px: float = 64.0,
) -> ProjectedSplats:
    right, up, forward = camera.basis()
    focal = float(camera.focal_pixels(height))
    center_radius_depth = np.zeros((scene.count, 4), dtype=np.float32)
    pos_local = np.zeros((scene.count, 3), dtype=np.float32)
    inv_scale = np.zeros((scene.count, 3), dtype=np.float32)
    quat = scene.rotations.astype(np.float32).copy()
    valid = np.zeros((scene.count,), dtype=np.uint32)

    def quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
        qv = q[1:4]
        return v + 2.0 * np.cross(np.cross(v, qv) + q[0] * v, qv)

    for i in range(scene.count):
        rel = scene.positions[i] - camera.position
        cam_dist = float(np.linalg.norm(rel))
        cam_pos = np.array(
            [np.dot(rel, right), np.dot(rel, up), np.dot(rel, forward)],
            dtype=np.float32,
        )
        x, y, z = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])
        if z <= 1e-6:
            center_radius_depth[i, 3] = z
            continue

        inv_depth = 1.0 / z
        px = x * focal * inv_depth + 0.5 * float(width)
        py = -y * focal * inv_depth + 0.5 * float(height)

        q = scene.rotations[i]
        scale = scene.scales[i].astype(np.float32).copy()
        max_scale = float(np.max(scale))
        scale = np.maximum(scale, max_scale * 0.05)
        scale = scale * float(radius_scale)

        axis0_world = quat_rotate(np.array([1.0, 0.0, 0.0], dtype=np.float32), q) * scale[0]
        axis1_world = quat_rotate(np.array([0.0, 1.0, 0.0], dtype=np.float32), q) * scale[1]
        axis2_world = quat_rotate(np.array([0.0, 0.0, 1.0], dtype=np.float32), q) * scale[2]

        axis0 = np.array(
            [np.dot(axis0_world, right), np.dot(axis0_world, up), np.dot(axis0_world, forward)],
            dtype=np.float32,
        )
        axis1 = np.array(
            [np.dot(axis1_world, right), np.dot(axis1_world, up), np.dot(axis1_world, forward)],
            dtype=np.float32,
        )
        axis2 = np.array(
            [np.dot(axis2_world, right), np.dot(axis2_world, up), np.dot(axis2_world, forward)],
            dtype=np.float32,
        )

        def proj_cam(p: np.ndarray) -> np.ndarray:
            zc = max(float(p[2]), 1e-6)
            return np.array(
                [p[0] * focal / zc + 0.5 * float(width), -p[1] * focal / zc + 0.5 * float(height)],
                dtype=np.float32,
            )

        c0p = proj_cam(cam_pos + axis0)
        c0m = proj_cam(cam_pos - axis0)
        c1p = proj_cam(cam_pos + axis1)
        c1m = proj_cam(cam_pos - axis1)
        c2p = proj_cam(cam_pos + axis2)
        c2m = proj_cam(cam_pos - axis2)
        center = np.array([px, py], dtype=np.float32)
        extent = np.zeros((2,), dtype=np.float32)
        for s in (c0p, c0m, c1p, c1m, c2p, c2m):
            extent = np.maximum(extent, np.abs(s - center))
        radius = float(np.clip(np.max(extent) + 1.0, 1.0, max_splat_radius_px))

        center_radius_depth[i, :] = np.array([px, py, radius, cam_dist], dtype=np.float32)
        invs = 1.0 / np.maximum(scale, 1e-6)
        inv_scale[i, :] = invs.astype(np.float32)
        pos_local[i, :] = (quat_rotate(camera.position - scene.positions[i], q) * invs).astype(np.float32)
        is_visible = (
            z > 1e-4
            and cam_dist > camera.near
            and cam_dist < camera.far
            and (px + radius) >= 0.0
            and (px - radius) < float(width)
            and (py + radius) >= 0.0
            and (py - radius) < float(height)
        )
        valid[i] = 1 if is_visible else 0

    color_alpha = np.concatenate([scene.colors, scene.opacities[:, None]], axis=1).astype(np.float32)
    return ProjectedSplats(
        center_radius_depth=center_radius_depth,
        color_alpha=color_alpha,
        valid=valid,
        pos_local=pos_local,
        inv_scale=inv_scale,
        quat=quat,
    )


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
        min_x = max(int(np.floor((cx - radius) / float(tile_size))), 0)
        max_x = min(int(np.floor((cx + radius) / float(tile_size))), tile_width - 1)
        min_y = max(int(np.floor((cy - radius) / float(tile_size))), 0)
        max_y = min(int(np.floor((cy + radius) / float(tile_size))), tile_height - 1)
        depth_key = quantize_depth(float(depth), near_depth, far_depth, depth_bits)
        for tile_y in range(min_y, max_y + 1):
            for tile_x in range(min_x, max_x + 1):
                tile_id = tile_y * tile_width + tile_x
                key = np.uint32((tile_id << depth_bits) | int(depth_key))
                if counter < max_list_entries:
                    keys[counter] = key
                    values[counter] = np.uint32(splat_id)
                counter += 1
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
                output[py, px, 3] = 1.0
                continue
            accum = np.zeros((3,), dtype=np.float32)
            trans = 1.0
            uv_x = (float(px) + 0.5 - 0.5 * float(width)) / float(focal)
            uv_y = (float(py) + 0.5 - 0.5 * float(height)) / float(focal)
            ray = forward + uv_x * right - uv_y * up
            ray = ray / max(np.linalg.norm(ray), 1e-8)
            for item in range(int(start), int(end)):
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
                if trans < 0.01:
                    break
            output[py, px, :3] = accum + trans * background
            output[py, px, 3] = 1.0 - trans
    return output

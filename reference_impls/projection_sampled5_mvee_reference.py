from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.scene.gaussian_scene import GaussianScene
from src.renderer.camera import Camera

STATUS_OK = np.uint32(0)
STATUS_NEAR_CAMERA_UNSTABLE = np.uint32(1 << 0)
STATUS_SILHOUETTE_UNSTABLE = np.uint32(1 << 1)
STATUS_MVEE_REGULARIZED = np.uint32(1 << 2)
STATUS_MVEE_DEGENERATE = np.uint32(1 << 3)
STATUS_HARD_FALLBACK = np.uint32(1 << 4)
_TAU = np.float32(2.0 * np.pi)
_GAUSSIAN_SUPPORT_SIGMA_RADIUS = np.float32(3.0)


@dataclass(slots=True)
class Sampled5MVEEProjectedSplats:
    center_radius_depth: np.ndarray
    color_alpha: np.ndarray
    valid: np.ndarray
    pos_local: np.ndarray
    inv_scale: np.ndarray
    quat: np.ndarray
    status_bits: np.ndarray
    sample_points_screen: np.ndarray
    ellipse_center_axes: np.ndarray


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _quat_rotate(v: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    qv = q_wxyz[1:4]
    return v + 2.0 * np.cross(np.cross(v, qv) + q_wxyz[0] * v, qv)


def _quat_conj(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]], dtype=np.float32)


def _silhouette_points_local5(ro_local: np.ndarray, eps: float) -> tuple[np.ndarray, bool]:
    d = float(np.linalg.norm(ro_local))
    if d <= 1.0 + eps:
        return np.zeros((5, 3), dtype=np.float32), False
    n = ro_local / d
    center = n / d
    r = float(np.sqrt(max(1.0 - 1.0 / (d * d), 0.0)))
    if float(n[2]) < -0.9999999:
        u = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        v = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    else:
        a = np.float32(1.0 / (1.0 + float(n[2])))
        b = np.float32(-float(n[0]) * float(n[1]) * float(a))
        u = np.array([1.0 - float(n[0]) * float(n[0]) * float(a), float(b), -float(n[0])], dtype=np.float32)
        v = np.array([float(b), 1.0 - float(n[1]) * float(n[1]) * float(a), -float(n[1])], dtype=np.float32)
    if np.linalg.norm(u) <= 0.0 or np.linalg.norm(v) <= 0.0:
        return np.zeros((5, 3), dtype=np.float32), False
    pts = np.zeros((5, 3), dtype=np.float32)
    for i in range(5):
        t = _TAU * np.float32(i) / np.float32(5.0)
        pts[i, :] = center + r * (np.cos(t) * u + np.sin(t) * v)
    return pts, True


def _project_world_to_screen(
    world_pos: np.ndarray,
    cam_pos: np.ndarray,
    cam_right: np.ndarray,
    cam_up: np.ndarray,
    cam_forward: np.ndarray,
    focal_pixels: np.ndarray,
    principal_point: np.ndarray,
    width: int,
    height: int,
    distortion_k1: float = 0.0,
    distortion_k2: float = 0.0,
) -> tuple[np.ndarray, float, bool]:
    rel = world_pos - cam_pos
    cam = np.array([np.dot(rel, cam_right), np.dot(rel, cam_up), np.dot(rel, cam_forward)], dtype=np.float32)
    z = float(cam[2])
    if z <= 1e-6:
        return np.zeros((2,), dtype=np.float32), z, False
    uv = cam[:2] / z
    r2 = float(np.dot(uv, uv))
    distort = 1.0 + distortion_k1 * r2 + distortion_k2 * r2 * r2
    uv = uv * distort
    px = float(uv[0] * float(focal_pixels[0]) + float(principal_point[0]))
    py = float(uv[1] * float(focal_pixels[1]) + float(principal_point[1]))
    ok = np.isfinite(px) and np.isfinite(py)
    return np.array([px, py], dtype=np.float32), z, bool(ok)


def _fit_mvee_5pt(points: np.ndarray, mvee_iters: int, eps: float) -> tuple[np.ndarray, np.ndarray, float, np.uint32]:
    d = 2.0
    points64 = points.astype(np.float64)
    norm_center = np.mean(points64, axis=0)
    norm_scale = float(np.max(np.linalg.norm(points64 - norm_center[None, :], axis=1)))
    status = STATUS_OK
    if norm_scale <= float(eps):
        return np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32), 0.0, np.uint32(status | STATUS_MVEE_DEGENERATE)
    points_n = (points64 - norm_center[None, :]) / norm_scale
    q = np.concatenate([points_n, np.ones((5, 1), dtype=np.float64)], axis=1)
    u = np.full((5,), 1.0 / 5.0, dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)
    for _ in range(max(int(mvee_iters), 0)):
        x_mat = q.T @ (u[:, None] * q)
        try:
            x_inv = np.linalg.inv(x_mat)
        except np.linalg.LinAlgError:
            x_inv = np.linalg.inv(x_mat + float(eps) * eye3)
            status |= STATUS_MVEE_REGULARIZED
        m = np.einsum("ni,ij,nj->n", q, x_inv, q)
        j = int(np.argmax(m))
        mj = float(m[j])
        denom = (d + 1.0) * (mj - 1.0)
        if denom <= float(eps):
            status |= STATUS_MVEE_DEGENERATE
            break
        alpha = float(np.clip((mj - (d + 1.0)) / denom, 0.0, 1.0))
        u *= 1.0 - alpha
        u[j] += alpha

    center_n = np.sum(u[:, None] * points_n, axis=0)
    diffs = points_n - center_n[None, :]
    s_mat = np.einsum("n,ni,nj->ij", u, diffs, diffs)
    try:
        a_mat = (1.0 / d) * np.linalg.inv(s_mat)
    except np.linalg.LinAlgError:
        a_mat = (1.0 / d) * np.linalg.inv(s_mat + float(eps) * np.eye(2, dtype=np.float64))
        status |= STATUS_MVEE_REGULARIZED
    eigvals, eigvecs = np.linalg.eigh(a_mat)
    eigvals = np.maximum(eigvals, float(eps))
    axes = (1.0 / np.sqrt(eigvals)).astype(np.float64)
    order = np.argsort(axes)[::-1]
    axes, eigvecs = axes[order], eigvecs[:, order]
    rel = (points_n - center_n[None, :]) @ eigvecs
    denom2 = np.maximum(axes * axes, float(eps))
    qmax = float(np.max(np.sum((rel * rel) / denom2[None, :], axis=1)))
    if qmax > 1.0:
        axes *= np.sqrt(qmax)
        status |= STATUS_MVEE_REGULARIZED
    angle = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    center = norm_center + center_n * norm_scale
    axes = axes * norm_scale
    return center.astype(np.float32), axes.astype(np.float32), angle, status


def project_splats_sampled5_mvee(
    scene: GaussianScene,
    camera: Camera,
    width: int,
    height: int,
    radius_scale: float,
    *,
    mvee_iters: int = 6,
    safety_scale: float = 1.05,
    radius_pad_px: float = 1.0,
    mvee_eps: float = 1e-6,
    distortion_k1: float = 0.0,
    distortion_k2: float = 0.0,
) -> Sampled5MVEEProjectedSplats:
    right, up, forward = camera.basis()
    fx, fy = camera.focal_pixels_xy(width, height)
    cx, cy = camera.principal_point(width, height)
    focal, principal = np.array([fx, fy], dtype=np.float32), np.array([cx, cy], dtype=np.float32)
    fallback_focal, radius_cap, n = float(max(fx, fy)), float(max(width, height, 1)), scene.count
    center_radius_depth = np.zeros((n, 4), dtype=np.float32)
    pos_local, inv_scale = np.zeros((n, 3), dtype=np.float32), np.zeros((n, 3), dtype=np.float32)
    quat = scene.rotations.astype(np.float32).copy()
    valid, status_bits = np.zeros((n,), dtype=np.uint32), np.zeros((n,), dtype=np.uint32)
    sample_points_screen = np.full((n, 5, 2), np.nan, dtype=np.float32)
    ellipse_center_axes = np.zeros((n, 5), dtype=np.float32)

    for i in range(n):
        world_pos = scene.positions[i].astype(np.float32)
        rel = world_pos - camera.position
        cam_distance = float(np.linalg.norm(rel))
        cam_pos = np.array([np.dot(rel, right), np.dot(rel, up), np.dot(rel, forward)], dtype=np.float32)
        depth_value = float(cam_pos[2])
        center = np.array(
            [
                float(cam_pos[0] * float(focal[0]) / max(depth_value, 1e-6) + float(principal[0])),
                float(cam_pos[1] * float(focal[1]) / max(depth_value, 1e-6) + float(principal[1])),
            ],
            dtype=np.float32,
        )
        q = quat[i]
        sigma = np.exp(scene.scales[i].astype(np.float32).copy())
        raw_scale = np.maximum(sigma * (np.float32(radius_scale) * _GAUSSIAN_SUPPORT_SIGMA_RADIUS), np.float32(1e-6))
        fallback_radius = float(np.clip((fallback_focal * float(np.max(raw_scale)) / max(depth_value, 1e-6)) * float(safety_scale) + float(radius_pad_px), 1.0, radius_cap))
        invs = 1.0 / np.maximum(raw_scale, np.float32(1e-6))
        ro_local = (_quat_rotate(camera.position - world_pos, q) * invs).astype(np.float32)
        pos_local[i, :], inv_scale[i, :] = ro_local, invs
        local_pts, ok_silhouette = _silhouette_points_local5(ro_local, float(mvee_eps))
        radius_px = fallback_radius
        if not ok_silhouette:
            status_bits[i] |= STATUS_SILHOUETTE_UNSTABLE
        else:
            q_inv = _quat_conj(q)
            screen_pts = np.zeros((5, 2), dtype=np.float32)
            ok_project = True
            for j in range(5):
                world_sample = world_pos + _quat_rotate(local_pts[j] / np.maximum(invs, 1e-6), q_inv)
                screen_p, _, valid_p = _project_world_to_screen(
                    world_sample,
                    camera.position,
                    right,
                    up,
                    forward,
                    focal,
                    principal,
                    width,
                    height,
                    distortion_k1=distortion_k1,
                    distortion_k2=distortion_k2,
                )
                if not valid_p:
                    ok_project = False
                    break
                screen_pts[j, :] = screen_p
            sample_points_screen[i, :, :] = screen_pts
            if not ok_project:
                status_bits[i] |= STATUS_NEAR_CAMERA_UNSTABLE
            else:
                center_fit, axes_fit, angle_fit, st = _fit_mvee_5pt(screen_pts, mvee_iters=max(int(mvee_iters), 0), eps=float(mvee_eps))
                status_bits[i] |= st
                ellipse_center_axes[i, :2], ellipse_center_axes[i, 2:4], ellipse_center_axes[i, 4] = center_fit, axes_fit, np.float32(angle_fit)
                if np.all(np.isfinite(center_fit)) and np.all(np.isfinite(axes_fit)):
                    center = center_fit
                    radius_px = float(np.clip(float(np.max(axes_fit)) * safety_scale + radius_pad_px, 1.0, radius_cap))
                else:
                    status_bits[i] |= STATUS_HARD_FALLBACK
        center_radius_depth[i, :] = np.array([center[0], center[1], radius_px, cam_distance], dtype=np.float32)
        valid[i] = np.uint32(
            1
            if depth_value > 1e-4
            and cam_distance > camera.near
            and cam_distance < camera.far
            and (center[0] + radius_px) >= 0.0
            and (center[0] - radius_px) < float(width)
            and (center[1] + radius_px) >= 0.0
            and (center[1] - radius_px) < float(height)
            else 0
        )

    return Sampled5MVEEProjectedSplats(
        center_radius_depth=center_radius_depth,
        color_alpha=np.concatenate([scene.colors, scene.opacities[:, None]], axis=1).astype(np.float32),
        valid=valid,
        pos_local=pos_local,
        inv_scale=inv_scale,
        quat=quat,
        status_bits=status_bits,
        sample_points_screen=sample_points_screen,
        ellipse_center_axes=ellipse_center_axes,
    )

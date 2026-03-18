from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.renderer import Camera
from src.scene import load_colmap_reconstruction


def _project_point(camera: Camera, point_world: np.ndarray, width: int, height: int) -> tuple[float, float, bool]:
    projected, ok = camera.project_world_to_screen(point_world, width, height)
    return float(projected[0]), float(projected[1]), bool(ok)


def test_colmap_projection_recovers_observed_pixels():
    dataset_root = Path("dataset/garden")
    if not dataset_root.exists():
        pytest.skip("dataset/garden is unavailable for COLMAP projection recovery test.")

    recon = load_colmap_reconstruction(dataset_root)
    rng = np.random.default_rng(20260217)
    errors: list[float] = []

    for image_id in sorted(recon.images.keys()):
        image = recon.images[image_id]
        colmap_camera = recon.cameras[image.camera_id]
        camera = Camera.from_colmap(
            q_wxyz=image.q_wxyz,
            t_xyz=image.t_xyz,
            fx=float(colmap_camera.fx),
            fy=float(colmap_camera.fy),
            cx=float(colmap_camera.cx),
            cy=float(colmap_camera.cy),
            distortion_k1=float(colmap_camera.k1),
            distortion_k2=float(colmap_camera.k2),
            near=0.1,
            far=1000.0,
        )
        point_ids = image.points2d_point3d_ids
        valid_obs = np.flatnonzero(point_ids >= 0)
        if valid_obs.size == 0:
            continue
        rng.shuffle(valid_obs)
        sampled = valid_obs[: min(valid_obs.size, 256)]
        for obs_idx in sampled:
            point_id = int(point_ids[int(obs_idx)])
            point3d = recon.points3d.get(point_id)
            if point3d is None:
                continue
            px, py, ok = _project_point(
                camera=camera,
                point_world=point3d.xyz.astype(np.float32),
                width=int(colmap_camera.width),
                height=int(colmap_camera.height),
            )
            if not ok:
                continue
            target = image.points2d_xy[int(obs_idx)]
            err = float(np.linalg.norm(np.array([px, py], dtype=np.float32) - target.astype(np.float32)))
            if np.isfinite(err):
                errors.append(err)
        if len(errors) >= 3000:
            break

    assert len(errors) >= 500, f"Not enough matched observations were projected: got {len(errors)}."
    errs = np.asarray(errors, dtype=np.float32)
    mean_err = float(np.mean(errs))
    p95_err = float(np.percentile(errs, 95.0))
    assert mean_err < 2.5, f"Mean reprojection error too high: {mean_err:.4f}px"
    assert p95_err < 8.0, f"95th percentile reprojection error too high: {p95_err:.4f}px"

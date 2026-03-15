from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.scene import (
    GaussianInitHyperParams,
    build_training_frames,
    initialize_scene_from_colmap_points,
    load_colmap_reconstruction,
    resolve_colmap_init_hparams,
    suggest_colmap_init_hparams,
)

_actual_scale = lambda log_scale: np.exp(np.asarray(log_scale, dtype=np.float32))


def _write_cameras_bin(path: Path, model_id: int = 1) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<i", 7))
        handle.write(struct.pack("<i", model_id))
        handle.write(struct.pack("<Q", 400))
        handle.write(struct.pack("<Q", 200))
        if model_id == 0:
            handle.write(struct.pack("<ddd", 420.0, 200.0, 100.0))
        elif model_id == 1:
            handle.write(struct.pack("<dddd", 400.0, 420.0, 200.0, 100.0))
        else:
            handle.write(struct.pack("<dddddddd", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))


def _write_images_bin(path: Path, image_name: str) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<i", 3))
        handle.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))
        handle.write(struct.pack("<ddd", 0.0, 0.0, -2.0))
        handle.write(struct.pack("<i", 7))
        handle.write(image_name.encode("utf-8"))
        handle.write(b"\x00")
        handle.write(struct.pack("<Q", 0))


def _write_points3d_bin(path: Path) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 2))
        for point_id, xyz, rgb in (
            (11, (1.0, 2.0, 3.0), (255, 128, 64)),
            (12, (-1.0, 0.0, 2.0), (12, 34, 56)),
        ):
            handle.write(struct.pack("<Q", point_id))
            handle.write(struct.pack("<ddd", *xyz))
            handle.write(struct.pack("<BBB", *rgb))
            handle.write(struct.pack("<d", 0.5))
            handle.write(struct.pack("<Q", 0))


def _build_tiny_colmap_tree(tmp_path: Path, model_id: int = 1) -> Path:
    root = tmp_path / "scene"
    sparse = root / "sparse" / "0"
    images = root / "images_4"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    _write_cameras_bin(sparse / "cameras.bin", model_id=model_id)
    _write_images_bin(sparse / "images.bin", "frame.png")
    _write_points3d_bin(sparse / "points3D.bin")
    Image.fromarray(np.full((100, 200, 3), 127, dtype=np.uint8), mode="RGB").save(images / "frame.png")
    return root


def test_colmap_loader_and_frame_scaling(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=1)
    recon = load_colmap_reconstruction(root)
    assert len(recon.cameras) == 1
    assert len(recon.images) == 1
    assert len(recon.points3d) == 2

    frames = build_training_frames(recon, images_subdir="images_4")
    assert len(frames) == 1
    frame = frames[0]
    assert frame.width == 200
    assert frame.height == 100
    assert np.isclose(frame.fx, 200.0)
    assert np.isclose(frame.fy, 210.0)
    assert np.isclose(frame.cx, 100.0)
    assert np.isclose(frame.cy, 50.0)

    init_hparams = GaussianInitHyperParams(base_scale=0.02, initial_opacity=0.4)
    scene = initialize_scene_from_colmap_points(recon, max_gaussians=1, seed=42, init_hparams=init_hparams)
    assert scene.count == 1
    assert scene.positions.shape == (1, 3)
    assert scene.scales.shape == (1, 3)
    assert scene.rotations.shape == (1, 4)
    assert scene.colors.shape == (1, 3)
    assert np.all(np.isfinite(scene.positions))


def test_colmap_loader_rejects_unsupported_camera_model(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=4)
    with pytest.raises(ValueError):
        _ = load_colmap_reconstruction(root)


def test_colmap_init_uses_direct_pointcloud_when_requested_count_exceeds_points(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=1)
    recon = load_colmap_reconstruction(root)
    requested_count = 7
    init_hparams = GaussianInitHyperParams(position_jitter_std=0.0, scale_jitter_ratio=0.0)
    scene = initialize_scene_from_colmap_points(
        recon,
        max_gaussians=requested_count,
        seed=123,
        init_hparams=init_hparams,
    )

    assert scene.count == 2
    assert scene.positions.shape == (2, 3)
    assert scene.colors.shape == (2, 3)
    np.testing.assert_allclose(_actual_scale(scene.scales), np.full((2, 3), 3.0, dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(scene.rotations, np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_colmap_init_suggestions_scale_with_requested_density(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=1)
    recon = load_colmap_reconstruction(root)

    coarse = suggest_colmap_init_hparams(recon, max_gaussians=1)
    direct = suggest_colmap_init_hparams(recon, max_gaussians=2)
    dense = suggest_colmap_init_hparams(recon, max_gaussians=8)

    assert coarse.base_scale is not None
    assert coarse.position_jitter_std is not None
    assert coarse.initial_opacity is not None
    assert direct.base_scale is not None
    assert dense.base_scale is not None
    assert dense.position_jitter_std is not None
    assert dense.initial_opacity is not None
    assert np.isclose(direct.base_scale, 0.75)
    assert coarse.base_scale > dense.base_scale
    assert coarse.position_jitter_std > dense.position_jitter_std
    assert coarse.initial_opacity >= dense.initial_opacity


def test_colmap_init_resolver_preserves_manual_overrides(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=1)
    recon = load_colmap_reconstruction(root)

    resolved = resolve_colmap_init_hparams(
        recon,
        max_gaussians=4,
        init_hparams=GaussianInitHyperParams(
            position_jitter_std=0.0,
            base_scale=None,
            scale_jitter_ratio=0.0,
            initial_opacity=None,
            color_jitter_std=0.05,
        ),
    )

    assert resolved.position_jitter_std == 0.0
    assert resolved.scale_jitter_ratio == 0.0
    assert resolved.color_jitter_std == 0.05
    assert resolved.base_scale is not None and resolved.base_scale > 0.0
    assert resolved.initial_opacity is not None and 0.0 < resolved.initial_opacity < 1.0


def test_colmap_init_resolver_uses_nn_spacing_as_default_base_scale(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=1)
    recon = load_colmap_reconstruction(root)

    resolved = resolve_colmap_init_hparams(recon, max_gaussians=2)
    scene = initialize_scene_from_colmap_points(
        recon,
        max_gaussians=2,
        seed=7,
        init_hparams=GaussianInitHyperParams(
            position_jitter_std=0.0,
            base_scale=resolved.base_scale,
            scale_jitter_ratio=0.0,
            initial_opacity=resolved.initial_opacity,
            color_jitter_std=0.0,
        ),
    )

    assert resolved.base_scale is not None
    assert np.isclose(resolved.base_scale, 0.75)
    np.testing.assert_allclose(_actual_scale(scene.scales), np.full((2, 3), 0.75, dtype=np.float32), rtol=0.0, atol=1e-6)

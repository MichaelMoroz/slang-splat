from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.scene import (
    GaussianInitHyperParams,
    build_training_frames,
    initialize_scene_from_colmap_diffused_points,
    initialize_scene_from_colmap_points,
    load_colmap_reconstruction,
    resolve_colmap_init_hparams,
    sample_colmap_diffused_points,
    suggest_colmap_init_hparams,
    transform_colmap_reconstruction_pca,
    transform_poses_pca,
)
from src.scene._internal import colmap_ops
from src.scene._internal.colmap_types import ColmapCamera, ColmapFrame, ColmapImage, ColmapPoint3D, ColmapReconstruction

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
        elif model_id == 2:
            handle.write(struct.pack("<dddd", 420.0, 200.0, 100.0, 0.07))
        elif model_id == 3:
            handle.write(struct.pack("<ddddd", 420.0, 200.0, 100.0, 0.07, -0.02))
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
        for point_id, xyz, rgb, track_length in (
            (11, (1.0, 2.0, 3.0), (255, 128, 64), 3),
            (12, (-1.0, 0.0, 2.0), (12, 34, 56), 3),
        ):
            handle.write(struct.pack("<Q", point_id))
            handle.write(struct.pack("<ddd", *xyz))
            handle.write(struct.pack("<BBB", *rgb))
            handle.write(struct.pack("<d", 0.5))
            handle.write(struct.pack("<Q", track_length))
            handle.write(struct.pack("<ii", 1, 0) * track_length)


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


def _write_cameras_txt(path: Path, model_name: str = "PINHOLE") -> None:
    params = {
        "SIMPLE_PINHOLE": "420 200 100",
        "PINHOLE": "400 420 200 100",
        "SIMPLE_RADIAL": "420 200 100 0.07",
        "RADIAL": "420 200 100 0.07 -0.02",
        "OPENCV": "400 420 200 100 0.07 -0.02 0.001 -0.002",
        "FULL_OPENCV": "400 420 200 100 0.07 -0.02 0.001 -0.002 0.0 0.0 0.0 0.0",
    }[model_name]
    path.write_text(
        "# Camera list\n"
        f"7 {model_name} 400 200 {params}\n",
        encoding="utf-8",
    )


def _write_images_txt(path: Path, image_name: str) -> None:
    path.write_text(
        "# Image list\n"
        f"3 1 0 0 0 0 0 -2 7 {image_name}\n"
        "10 20 11 30 40 -1\n",
        encoding="utf-8",
    )


def _write_points3d_txt(path: Path) -> None:
    path.write_text(
        "# 3D point list\n"
        "11 1 2 3 255 128 64 0.5 3 0 5 1 9 2\n"
        "12 -1 0 2 12 34 56 0.5 3 0 5 1 9 2\n",
        encoding="utf-8",
    )


def _build_tiny_colmap_text_tree(tmp_path: Path, model_name: str = "PINHOLE") -> Path:
    root = tmp_path / "scene_text"
    sparse = root / "sparse" / "0"
    images = root / "images_4"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    _write_cameras_txt(sparse / "cameras.txt", model_name=model_name)
    _write_images_txt(sparse / "images.txt", "frame.png")
    _write_points3d_txt(sparse / "points3D.txt")
    Image.fromarray(np.full((100, 200, 3), 127, dtype=np.uint8), mode="RGB").save(images / "frame.png")
    return root


def test_colmap_loader_and_frame_scaling(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=1)
    recon = load_colmap_reconstruction(root)
    assert len(recon.cameras) == 1
    assert len(recon.images) == 1
    assert len(recon.points3d) == 2
    assert recon.points3d[11].track_length == 3

    frames = build_training_frames(recon, images_subdir="images_4")
    assert len(frames) == 1
    frame = frames[0]
    assert frame.width == 200
    assert frame.height == 100
    assert np.isclose(frame.fx, 200.0)
    assert np.isclose(frame.fy, 210.0)
    assert np.isclose(frame.cx, 100.0)
    assert np.isclose(frame.cy, 50.0)
    assert np.isclose(frame.k1, 0.0)
    assert np.isclose(frame.k2, 0.0)

    init_hparams = GaussianInitHyperParams(base_scale=0.02, initial_opacity=0.4)
    scene = initialize_scene_from_colmap_points(recon, max_gaussians=1, seed=42, init_hparams=init_hparams)
    assert scene.count == 1
    assert scene.positions.shape == (1, 3)
    assert scene.scales.shape == (1, 3)
    assert scene.rotations.shape == (1, 4)
    assert scene.colors.shape == (1, 3)
    assert np.all(np.isfinite(scene.positions))


def test_colmap_loader_supports_text_sparse_exports(tmp_path: Path) -> None:
    root = _build_tiny_colmap_text_tree(tmp_path, model_name="PINHOLE")

    recon = load_colmap_reconstruction(root)
    frames = build_training_frames(recon, images_subdir="images_4")

    assert len(recon.cameras) == 1
    assert len(recon.images) == 1
    assert len(recon.points3d) == 2
    assert recon.points3d[11].track_length == 3
    assert len(frames) == 1
    assert frames[0].width == 200
    assert frames[0].height == 100
    assert np.isclose(frames[0].fx, 200.0)
    assert np.isclose(frames[0].fy, 210.0)


def test_build_training_frames_uses_sixteen_loader_threads(tmp_path: Path, monkeypatch) -> None:
    root = _build_tiny_colmap_tree(tmp_path, model_id=1)
    Image.fromarray(np.full((60, 120, 3), 80, dtype=np.uint8), mode="RGB").save(root / "images_4" / "frame_b.png")
    sparse = root / "sparse" / "0"
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 2))
        for image_id, image_name, tx in ((3, "frame.png", -2.0), (5, "frame_b.png", -4.0)):
            handle.write(struct.pack("<i", image_id))
            handle.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))
            handle.write(struct.pack("<ddd", 0.0, 0.0, tx))
            handle.write(struct.pack("<i", 7))
            handle.write(image_name.encode("utf-8"))
            handle.write(b"\x00")
            handle.write(struct.pack("<Q", 0))
    recon = load_colmap_reconstruction(root)
    calls: list[int] = []

    class _Executor:
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            calls.append(int(max_workers))

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> bool:
            return False

        def map(self, fn, items):
            return map(fn, items)

    monkeypatch.setattr(colmap_ops, "ThreadPoolExecutor", _Executor)

    frames = colmap_ops.build_training_frames_from_root(recon, root / "images_4")

    assert calls == [16]
    assert [frame.image_id for frame in frames] == [3, 5]
    assert [frame.width for frame in frames] == [200, 120]
    assert [frame.height for frame in frames] == [100, 60]


def test_resolve_training_frame_image_size_max_size_clamps_longer_side() -> None:
    assert colmap_ops.resolve_training_frame_image_size(200, 100, downscale_mode="max_size", downscale_max_size=120) == (120, 60)
    assert colmap_ops.resolve_training_frame_image_size(100, 200, downscale_mode="max_size", downscale_max_size=120) == (60, 120)
    assert colmap_ops.resolve_training_frame_image_size(100, 80, downscale_mode="max_size", downscale_max_size=120) == (100, 80)


def test_colmap_loader_rejects_unsupported_camera_model(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=4)
    with pytest.raises(ValueError):
        _ = load_colmap_reconstruction(root)


def test_colmap_loader_supports_simple_radial_camera_model(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=2)
    recon = load_colmap_reconstruction(root)
    camera = recon.cameras[7]
    frame = build_training_frames(recon, images_subdir="images_4")[0]

    assert np.isclose(camera.fx, 420.0)
    assert np.isclose(camera.fy, 420.0)
    assert np.isclose(camera.k1, 0.07)
    assert np.isclose(camera.k2, 0.0)
    assert np.isclose(frame.fx, 210.0)
    assert np.isclose(frame.fy, 210.0)
    assert np.isclose(frame.k1, 0.07)
    assert np.isclose(frame.k2, 0.0)


def test_colmap_loader_supports_radial_camera_model(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=3)
    recon = load_colmap_reconstruction(root)
    camera = recon.cameras[7]
    frame = build_training_frames(recon, images_subdir="images_4")[0]

    assert np.isclose(camera.fx, 420.0)
    assert np.isclose(camera.fy, 420.0)
    assert np.isclose(camera.k1, 0.07)
    assert np.isclose(camera.k2, -0.02)
    assert np.isclose(frame.k1, 0.07)
    assert np.isclose(frame.k2, -0.02)


@pytest.mark.parametrize("model_name", ["OPENCV", "FULL_OPENCV"])
def test_colmap_loader_supports_opencv_camera_models(tmp_path: Path, model_name: str) -> None:
    root = _build_tiny_colmap_text_tree(tmp_path, model_name=model_name)

    recon = load_colmap_reconstruction(root)
    camera = recon.cameras[7]
    frame = build_training_frames(recon, images_subdir="images_4")[0]

    assert np.isclose(camera.fx, 400.0)
    assert np.isclose(camera.fy, 420.0)
    assert np.isclose(camera.k1, 0.07)
    assert np.isclose(camera.k2, -0.02)
    assert np.isclose(frame.fx, 200.0)
    assert np.isclose(frame.fy, 210.0)
    assert np.isclose(frame.k1, 0.07)
    assert np.isclose(frame.k2, -0.02)


def test_colmap_loader_preserves_full_opencv_higher_order_terms(tmp_path: Path) -> None:
    root = _build_tiny_colmap_text_tree(tmp_path, model_name="FULL_OPENCV")

    recon = load_colmap_reconstruction(root)
    camera = recon.cameras[7]
    frame = build_training_frames(recon, images_subdir="images_4")[0]

    assert np.isclose(camera.p1, 0.001)
    assert np.isclose(camera.p2, -0.002)
    assert np.isclose(camera.k3, 0.0)
    assert np.isclose(camera.k4, 0.0)
    assert np.isclose(camera.k5, 0.0)
    assert np.isclose(camera.k6, 0.0)
    assert np.isclose(frame.p1, 0.001)
    assert np.isclose(frame.p2, -0.002)
    assert np.isclose(frame.k3, 0.0)
    assert np.isclose(frame.k4, 0.0)
    assert np.isclose(frame.k5, 0.0)
    assert np.isclose(frame.k6, 0.0)


def test_transform_poses_pca_defaults_to_no_rescale_and_wrapper_matches() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
    poses[:, 0, 3] = np.array([0.0, 10.0, 20.0], dtype=np.float32)

    transformed_default, transform_default = transform_poses_pca(poses)
    transformed_no_rescale, transform_no_rescale = transform_poses_pca(poses, rescale=False)
    transformed_rescaled, _ = transform_poses_pca(poses, rescale=True)

    np.testing.assert_allclose(transformed_default, transformed_no_rescale, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(transform_default, transform_no_rescale, rtol=0.0, atol=1e-6)
    assert float(np.max(np.abs(transformed_default[:, :3, 3]))) > 1.0
    assert float(np.max(np.abs(transformed_rescaled[:, :3, 3]))) <= 1.0 + 1e-6

    camera = ColmapCamera(camera_id=1, model_id=1, width=400, height=200, fx=400.0, fy=400.0, cx=200.0, cy=100.0)
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    recon = ColmapReconstruction(
        root=Path("synthetic"),
        sparse_dir=Path("synthetic") / "sparse" / "0",
        cameras={1: camera},
        images={
            1: ColmapImage(1, q_wxyz, np.array([0.0, 0.0, 0.0], dtype=np.float32), 1, "a.png", np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
            2: ColmapImage(2, q_wxyz, np.array([-10.0, 0.0, 0.0], dtype=np.float32), 1, "b.png", np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
            3: ColmapImage(3, q_wxyz, np.array([-20.0, 0.0, 0.0], dtype=np.float32), 1, "c.png", np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
        },
        points3d={1: ColmapPoint3D(1, np.array([20.0, 0.0, 0.0], dtype=np.float32), np.array([255.0, 255.0, 255.0], dtype=np.float32), 0.0)},
    )

    recon_default, wrapper_transform_default = transform_colmap_reconstruction_pca(recon)
    recon_no_rescale, wrapper_transform_no_rescale = transform_colmap_reconstruction_pca(recon, rescale=False)
    recon_rescaled, _ = transform_colmap_reconstruction_pca(recon, rescale=True)

    np.testing.assert_allclose(recon_default.points3d[1].xyz, recon_no_rescale.points3d[1].xyz, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(wrapper_transform_default, wrapper_transform_no_rescale, rtol=0.0, atol=1e-6)
    assert float(np.max(np.abs(recon_default.points3d[1].xyz))) > 1.0
    assert float(np.max(np.abs(recon_rescaled.points3d[1].xyz))) <= 1.0 + 1e-6


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


def test_colmap_fibonacci_sphere_points_use_camera_pose_mean() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=400, height=200, fx=400.0, fy=400.0, cx=200.0, cy=100.0)
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    recon = ColmapReconstruction(
        root=Path("synthetic"),
        sparse_dir=Path("synthetic") / "sparse" / "0",
        cameras={1: camera},
        images={
            1: ColmapImage(1, q_wxyz, np.array([0.0, 0.0, 0.0], dtype=np.float32), 1, "a.png", np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
            2: ColmapImage(2, q_wxyz, np.array([-4.0, 0.0, 0.0], dtype=np.float32), 1, "b.png", np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
        },
        points3d={},
    )

    positions, colors = colmap_ops.sample_colmap_fibonacci_sphere_points(recon, point_count=8, radius=2.5)

    assert positions.shape == (8, 3)
    assert colors.shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(positions - np.array([[2.0, 0.0, 0.0]], dtype=np.float32), axis=1), np.full((8,), 2.5, dtype=np.float32), rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(colors, np.full((8, 3), 0.8, dtype=np.float32), rtol=0.0, atol=1e-6)


def test_colmap_pointcloud_init_filters_points_below_selected_camera_observations() -> None:
    recon = ColmapReconstruction(
        root=Path("synthetic"),
        sparse_dir=Path("synthetic") / "sparse" / "0",
        cameras={},
        images={},
        points3d={
            1: ColmapPoint3D(1, np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0, track_length=2),
            2: ColmapPoint3D(2, np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0, track_length=3),
            3: ColmapPoint3D(3, np.array([2.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0, track_length=5),
        },
    )

    scene = initialize_scene_from_colmap_points(
        recon,
        max_gaussians=8,
        seed=123,
        init_hparams=GaussianInitHyperParams(position_jitter_std=0.0, scale_jitter_ratio=0.0),
        min_track_length=3,
    )

    assert scene.count == 2
    np.testing.assert_allclose(scene.positions, np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_colmap_diffused_sampling_filters_points_below_selected_camera_observations() -> None:
    recon = ColmapReconstruction(
        root=Path("synthetic"),
        sparse_dir=Path("synthetic") / "sparse" / "0",
        cameras={},
        images={},
        points3d={
            1: ColmapPoint3D(1, np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0, track_length=2),
            2: ColmapPoint3D(2, np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0, track_length=3),
            3: ColmapPoint3D(3, np.array([2.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0, track_length=4),
        },
    )

    positions, colors = sample_colmap_diffused_points(recon, point_count=6, diffusion_radius=0.0, seed=7, min_track_length=3)

    assert positions.shape == (6, 3)
    assert colors.shape == (6, 3)
    assert not np.any(np.all(np.isclose(positions, np.array([0.0, 0.0, 0.0], dtype=np.float32)), axis=1))


def test_colmap_pointcloud_init_allows_disabling_track_length_filter() -> None:
    recon = ColmapReconstruction(
        root=Path("synthetic"),
        sparse_dir=Path("synthetic") / "sparse" / "0",
        cameras={},
        images={},
        points3d={
            1: ColmapPoint3D(1, np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0, track_length=1),
            2: ColmapPoint3D(2, np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0, track_length=2),
        },
    )

    scene = initialize_scene_from_colmap_points(
        recon,
        max_gaussians=8,
        seed=3,
        init_hparams=GaussianInitHyperParams(position_jitter_std=0.0, scale_jitter_ratio=0.0),
        min_track_length=0,
    )

    assert scene.count == 2
    np.testing.assert_allclose(scene.positions, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32), rtol=0.0, atol=1e-6)


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


def test_colmap_diffused_sampling_scales_radius_by_original_nn_distance(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=1)
    recon = load_colmap_reconstruction(root)

    positions, colors = sample_colmap_diffused_points(recon, point_count=4, diffusion_radius=0.5, seed=9)

    base_xyz = np.stack([point.xyz for point in recon.points3d.values()], axis=0).astype(np.float32)
    base_rgb = np.stack([point.rgb for point in recon.points3d.values()], axis=0).astype(np.float32)
    rng = np.random.default_rng(9)
    base_indices = rng.integers(0, base_xyz.shape[0], size=4, dtype=np.int64)
    expected = base_xyz[base_indices] + rng.normal(0.0, 1.0, size=(4, 3)).astype(np.float32) * np.float32(1.5)

    np.testing.assert_allclose(positions, expected, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(colors, base_rgb[base_indices], rtol=0.0, atol=0.0)


def test_colmap_diffused_init_uses_requested_count_and_nn_scales(tmp_path: Path):
    root = _build_tiny_colmap_tree(tmp_path, model_id=1)
    recon = load_colmap_reconstruction(root)
    scene = initialize_scene_from_colmap_diffused_points(
        recon,
        point_count=5,
        diffusion_radius=0.0,
        seed=4,
        init_hparams=GaussianInitHyperParams(position_jitter_std=0.0, scale_jitter_ratio=0.0),
    )

    assert scene.count == 5
    assert scene.positions.shape == (5, 3)
    assert scene.colors.shape == (5, 3)
    assert np.all(np.isfinite(scene.positions))
    assert np.all(np.isfinite(scene.scales))


def test_depth_path_matching_is_relative_stem_and_extension_agnostic(tmp_path: Path) -> None:
    images_root = (tmp_path / "images").resolve()
    depth_root = (tmp_path / "depth").resolve()
    (images_root / "nested").mkdir(parents=True)
    (depth_root / "nested").mkdir(parents=True)
    (images_root / "nested" / "frame_0001.jpg").write_bytes(b"rgb")
    (depth_root / "nested" / "frame_0001.png").write_bytes(b"depth")

    depth_index = colmap_ops.build_depth_path_index(depth_root)
    matched = colmap_ops.match_depth_path(images_root, images_root / "nested" / "frame_0001.jpg", depth_index)

    assert matched == (depth_root / "nested" / "frame_0001.png").resolve()


def test_fit_depth_distance_remap_for_payload_uses_only_pose_local_correspondences() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=8, height=8, fx=8.0, fy=8.0, cx=4.0, cy=4.0)
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    frames = (
        ColmapFrame(1, Path("frame_a.png"), q_wxyz, np.zeros((3,), dtype=np.float32), 8.0, 8.0, 4.0, 4.0, 8, 8),
        ColmapFrame(2, Path("frame_b.png"), q_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32), 8.0, 8.0, 4.0, 4.0, 8, 8),
    )
    xy = np.array(
        [
            [1.0, 1.0], [3.0, 1.0], [5.0, 1.0], [6.0, 1.0],
            [1.0, 3.0], [3.0, 3.0], [5.0, 3.0], [6.0, 3.0],
            [1.0, 5.0], [3.0, 5.0], [5.0, 5.0], [6.0, 5.0],
            [1.0, 6.0], [3.0, 6.0], [5.0, 6.0], [6.0, 6.0],
        ],
        dtype=np.float32,
    )
    raw_depths_a = np.linspace(100.0, 240.0, num=xy.shape[0], dtype=np.float32)
    raw_depths_b = np.linspace(260.0, 400.0, num=xy.shape[0], dtype=np.float32)
    image_a_point_ids = np.arange(1, xy.shape[0] + 1, dtype=np.int64)
    image_b_point_ids = np.arange(xy.shape[0] + 1, 2 * xy.shape[0] + 1, dtype=np.int64)
    images = (
        ColmapImage(1, q_wxyz, np.zeros((3,), dtype=np.float32), 1, "frame_a.png", xy, image_a_point_ids),
        ColmapImage(2, q_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32), 1, "frame_b.png", xy, image_b_point_ids),
    )
    depth_maps = (np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32))
    points3d: dict[int, ColmapPoint3D] = {}
    scales = (2.5, 4.0)
    offsets = (3.0, 7.0)
    for frame, image, raw_depths, depth_map, scale, offset in zip(frames, images, (raw_depths_a, raw_depths_b), depth_maps, scales, offsets, strict=False):
        camera_obj = frame.make_camera()
        for point_id, screen_xy, raw_depth in zip(np.asarray(image.points2d_point3d_ids, dtype=np.int64), xy, raw_depths, strict=False):
            depth_map[int(screen_xy[1]), int(screen_xy[0])] = raw_depth
            ray = camera_obj.screen_to_world_ray(screen_xy, frame.width, frame.height)
            distance = offset + scale * float(raw_depth)
            world = np.asarray(camera_obj.position, dtype=np.float32) + np.asarray(ray, dtype=np.float32) * np.float32(distance)
            points3d[int(point_id)] = ColmapPoint3D(int(point_id), world.astype(np.float32), np.array([255.0, 255.0, 255.0], dtype=np.float32), 0.0)
    recon = ColmapReconstruction(root=Path("synthetic"), sparse_dir=Path("synthetic") / "sparse" / "0", cameras={1: camera}, images={1: images[0], 2: images[1]}, points3d=points3d)
    payloads: list[colmap_ops.DepthInitFramePayload] = []
    for frame, image, depth_map in zip(frames, images, depth_maps, strict=False):
        features, targets = colmap_ops.collect_depth_distance_remap_samples(recon, image, frame, camera, depth_map, colmap_ops.DEPTH_INIT_VALUE_DISTANCE)
        payloads.append(
            colmap_ops.DepthInitFramePayload(
                frame=frame,
                rgba8=np.zeros((8, 8, 4), dtype=np.uint8),
                depth_map=depth_map,
                camera_id=1,
                fit_features=features,
                fit_targets=targets,
            )
        )

    coeffs_a = colmap_ops.fit_depth_distance_remap_for_payload(payloads[0])
    coeffs_b = colmap_ops.fit_depth_distance_remap_for_payload(payloads[1])

    assert coeffs_a is not None
    assert coeffs_b is not None
    predicted_a = coeffs_a[0] + coeffs_a[1] * raw_depths_a
    predicted_b = coeffs_b[0] + coeffs_b[1] * raw_depths_b
    np.testing.assert_allclose(predicted_a, offsets[0] + scales[0] * raw_depths_a, rtol=0.0, atol=1e-2)
    np.testing.assert_allclose(predicted_b, offsets[1] + scales[1] * raw_depths_b, rtol=0.0, atol=1e-2)


def test_collect_depth_distance_remap_samples_uses_reprojected_pose_points_not_stored_keypoints() -> None:
    camera = ColmapCamera(camera_id=1, model_id=3, width=16, height=16, fx=16.0, fy=16.0, cx=8.0, cy=8.0, k1=0.1, k2=-0.02)
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    frame = ColmapFrame(1, Path("frame.png"), q_wxyz, np.zeros((3,), dtype=np.float32), 16.0, 16.0, 8.0, 8.0, 16, 16, 0.1, -0.02)
    image = ColmapImage(
        1,
        q_wxyz,
        np.zeros((3,), dtype=np.float32),
        1,
        "frame.png",
        np.zeros((16, 2), dtype=np.float32),
        np.arange(1, 17, dtype=np.int64),
    )
    depth_map = np.zeros((16, 16), dtype=np.float32)
    camera_obj = frame.make_camera()
    points3d: dict[int, ColmapPoint3D] = {}
    for point_id, screen_xy in enumerate(
        (
            (2.0, 2.0), (5.0, 2.0), (8.0, 2.0), (11.0, 2.0),
            (2.0, 5.0), (5.0, 5.0), (8.0, 5.0), (11.0, 5.0),
            (2.0, 8.0), (5.0, 8.0), (8.0, 8.0), (11.0, 8.0),
            (2.0, 11.0), (5.0, 11.0), (8.0, 11.0), (11.0, 11.0),
        ),
        start=1,
    ):
        world_pos = np.asarray(camera_obj.screen_to_world(np.asarray(screen_xy, dtype=np.float32), 8.0, frame.width, frame.height), dtype=np.float32)
        projected_xy, valid = camera_obj.project_world_to_screen(world_pos, frame.width, frame.height)
        assert valid
        depth_map[int(screen_xy[1]), int(screen_xy[0])] = 0.25 + 0.5 * float(np.linalg.norm(world_pos))
        np.testing.assert_allclose(projected_xy, np.asarray(screen_xy, dtype=np.float32), rtol=0.0, atol=1e-4)
        points3d[point_id] = ColmapPoint3D(point_id, world_pos, np.array([255.0, 255.0, 255.0], dtype=np.float32), 0.0)
    recon = ColmapReconstruction(root=Path("synthetic"), sparse_dir=Path("synthetic") / "sparse" / "0", cameras={1: camera}, images={1: image}, points3d=points3d)

    features, targets = colmap_ops.collect_depth_distance_remap_samples(recon, image, frame, camera, depth_map, colmap_ops.DEPTH_INIT_VALUE_DISTANCE)

    assert features.shape == (16, 2)
    coeffs = colmap_ops._robust_ridge_fit(features, targets)
    assert coeffs is not None
    predicted = features @ coeffs
    np.testing.assert_allclose(predicted, targets, rtol=0.0, atol=5e-2)


def test_collect_depth_distance_remap_samples_uses_all_visible_pose_points() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=64, height=64, fx=64.0, fy=64.0, cx=32.0, cy=32.0)
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    frame = ColmapFrame(1, Path("frame.png"), q_wxyz, np.zeros((3,), dtype=np.float32), 64.0, 64.0, 32.0, 32.0, 64, 64)
    image = ColmapImage(1, q_wxyz, np.zeros((3,), dtype=np.float32), 1, "frame.png", np.zeros((40, 2), dtype=np.float32), np.arange(1, 41, dtype=np.int64))
    camera_obj = frame.make_camera()
    depth_map = np.zeros((64, 64), dtype=np.float32)
    points3d: dict[int, ColmapPoint3D] = {}
    depths = np.arange(4.0, 44.0, dtype=np.float32)
    for point_id, depth in enumerate(depths, start=1):
        screen_xy = np.asarray((16.0 + float((point_id - 1) % 5) * 8.0, 16.0 + float((point_id - 1) // 5) * 4.0), dtype=np.float32)
        world_pos = np.asarray(camera_obj.screen_to_world(screen_xy, float(depth), frame.width, frame.height), dtype=np.float32)
        projected_xy, valid = camera_obj.project_world_to_screen(world_pos, frame.width, frame.height)
        assert valid
        depth_map[int(round(float(projected_xy[1]))), int(round(float(projected_xy[0])))] = depth
        points3d[point_id] = ColmapPoint3D(point_id, world_pos, np.array([255.0, 255.0, 255.0], dtype=np.float32), 0.0)
    recon = ColmapReconstruction(root=Path("synthetic"), sparse_dir=Path("synthetic") / "sparse" / "0", cameras={1: camera}, images={1: image}, points3d=points3d)

    features, targets = colmap_ops.collect_depth_distance_remap_samples(recon, image, frame, camera, depth_map, colmap_ops.DEPTH_INIT_VALUE_Z_DEPTH)

    assert features.shape == (40, 2)
    np.testing.assert_allclose(features[:, 1], depths, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(targets, depths, rtol=0.0, atol=1e-6)


def test_collect_depth_distance_remap_samples_skips_depth_discontinuities() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=8, height=8, fx=8.0, fy=8.0, cx=4.0, cy=4.0)
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    frame = ColmapFrame(1, Path("frame.png"), q_wxyz, np.zeros((3,), dtype=np.float32), 8.0, 8.0, 4.0, 4.0, 8, 8)
    image = ColmapImage(1, q_wxyz, np.zeros((3,), dtype=np.float32), 1, "frame.png", np.zeros((16, 2), dtype=np.float32), np.arange(1, 17, dtype=np.int64))
    camera_obj = frame.make_camera()
    depth_map = np.full((8, 8), 10.0, dtype=np.float32)
    points3d: dict[int, ColmapPoint3D] = {}
    for point_id, screen_xy in enumerate(((1.0, 1.0), (3.0, 1.0), (5.0, 1.0), (6.0, 1.0), (1.0, 3.0), (3.0, 3.0), (5.0, 3.0), (6.0, 3.0), (1.0, 5.0), (3.0, 5.0), (5.0, 5.0), (6.0, 5.0), (1.0, 6.0), (3.0, 6.0), (5.0, 6.0), (6.0, 6.0)), start=1):
        world_pos = np.asarray(camera_obj.screen_to_world(np.asarray(screen_xy, dtype=np.float32), 10.0, frame.width, frame.height), dtype=np.float32)
        points3d[point_id] = ColmapPoint3D(point_id, world_pos, np.array([255.0, 255.0, 255.0], dtype=np.float32), 0.0)
    depth_map[3, 3] = 2.0
    depth_map[3, 4] = 12.0
    depth_map[4, 3] = 14.0
    depth_map[4, 4] = 16.0
    points3d[1] = ColmapPoint3D(1, np.asarray(camera_obj.screen_to_world(np.asarray((3.5, 3.5), dtype=np.float32), 10.0, frame.width, frame.height), dtype=np.float32), np.array([255.0, 255.0, 255.0], dtype=np.float32), 0.0)
    recon = ColmapReconstruction(root=Path("synthetic"), sparse_dir=Path("synthetic") / "sparse" / "0", cameras={1: camera}, images={1: image}, points3d=points3d)

    features, targets = colmap_ops.collect_depth_distance_remap_samples(recon, image, frame, camera, depth_map, colmap_ops.DEPTH_INIT_VALUE_Z_DEPTH)

    assert features.shape == (12, 2)
    assert targets.shape == (12,)


def test_robust_ridge_fit_downweights_strong_outliers() -> None:
    raw_depth = np.linspace(50.0, 200.0, num=48, dtype=np.float32)
    features = np.stack((np.ones_like(raw_depth), raw_depth), axis=1)
    targets = 2.0 + 3.0 * raw_depth
    targets[-12:] += np.linspace(500.0, 1500.0, num=12, dtype=np.float32)

    coeffs = colmap_ops._robust_ridge_fit(features, targets)

    assert coeffs is not None
    np.testing.assert_allclose(coeffs, np.array([2.0, 3.0], dtype=np.float32), rtol=0.0, atol=5e-2)


def test_generate_depth_init_points_respects_budget_and_uniqueness() -> None:
    frame = ColmapFrame(
        image_id=1,
        image_path=Path("frame.png"),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.zeros((3,), dtype=np.float32),
        fx=8.0,
        fy=8.0,
        cx=4.0,
        cy=4.0,
        width=4,
        height=4,
    )
    rgba8 = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba8[..., 0] = np.arange(16, dtype=np.uint8).reshape(4, 4)
    depth_map = np.full((4, 4), 10.0, dtype=np.float32)
    payload = colmap_ops.DepthInitFramePayload(
        frame=frame,
        rgba8=rgba8,
        depth_map=depth_map,
        camera_id=7,
        fit_features=np.repeat(np.array([[1.0, 10.0]], dtype=np.float32), 16, axis=0),
        fit_targets=np.array([32.0] * 16, dtype=np.float32),
    )

    positions, colors = colmap_ops.generate_depth_init_points([payload], total_point_count=4, seed=7, depth_value_mode=colmap_ops.DEPTH_INIT_VALUE_DISTANCE)

    assert positions.shape == (4, 3)
    assert colors.shape == (4, 3)
    assert np.unique(positions, axis=0).shape[0] == 4
    assert np.all(colors[:, 0] >= 0.0)


def test_generate_depth_init_points_skips_frames_without_a_pose_local_fit() -> None:
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    frame_a = ColmapFrame(1, Path("frame_a.png"), q_wxyz, np.zeros((3,), dtype=np.float32), 8.0, 8.0, 4.0, 4.0, 4, 4)
    frame_b = ColmapFrame(2, Path("frame_b.png"), q_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32), 8.0, 8.0, 4.0, 4.0, 4, 4)
    rgba8_a = np.full((4, 4, 4), [255, 0, 0, 255], dtype=np.uint8)
    rgba8_b = np.full((4, 4, 4), [0, 255, 0, 255], dtype=np.uint8)
    payload_a = colmap_ops.DepthInitFramePayload(
        frame=frame_a,
        rgba8=rgba8_a,
        depth_map=np.full((4, 4), 10.0, dtype=np.float32),
        camera_id=3,
        fit_features=np.repeat(np.array([[1.0, 10.0]], dtype=np.float32), 16, axis=0),
        fit_targets=np.array([23.0] * 16, dtype=np.float32),
    )
    payload_b = colmap_ops.DepthInitFramePayload(
        frame=frame_b,
        rgba8=rgba8_b,
        depth_map=np.full((4, 4), 10.0, dtype=np.float32),
        camera_id=3,
        fit_features=np.zeros((0, 2), dtype=np.float32),
        fit_targets=np.zeros((0,), dtype=np.float32),
    )

    positions, colors = colmap_ops.generate_depth_init_points([payload_a, payload_b], total_point_count=6, seed=5, depth_value_mode=colmap_ops.DEPTH_INIT_VALUE_DISTANCE)

    assert positions.shape == (6, 3)
    np.testing.assert_allclose(colors, np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (6, 1)), rtol=0.0, atol=0.0)


def test_generate_depth_init_points_supports_z_depth_calibration() -> None:
    frame = ColmapFrame(
        image_id=1,
        image_path=Path("frame.png"),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.zeros((3,), dtype=np.float32),
        fx=8.0,
        fy=8.0,
        cx=4.0,
        cy=4.0,
        width=4,
        height=4,
    )
    rgba8 = np.full((4, 4, 4), 255, dtype=np.uint8)
    depth_map = np.full((4, 4), 10.0, dtype=np.float32)
    payload = colmap_ops.DepthInitFramePayload(
        frame=frame,
        rgba8=rgba8,
        depth_map=depth_map,
        camera_id=5,
        fit_features=np.repeat(np.array([[1.0, 10.0]], dtype=np.float32), 16, axis=0),
        fit_targets=np.array([43.0] * 16, dtype=np.float32),
    )

    positions, _ = colmap_ops.generate_depth_init_points([payload], total_point_count=4, seed=3, depth_value_mode=colmap_ops.DEPTH_INIT_VALUE_Z_DEPTH)

    camera = frame.make_camera()
    camera_positions = np.asarray([camera.world_point_to_camera(position) for position in positions], dtype=np.float32)
    np.testing.assert_allclose(camera_positions[:, 2], np.full((4,), 43.0, dtype=np.float32), rtol=0.0, atol=1e-5)

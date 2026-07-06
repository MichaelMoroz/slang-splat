from __future__ import annotations

import numpy as np

from reference_impls.reference_cpu import project_splats
from src.renderer import Camera, GaussianRenderer, PROJECTION_MODEL_EQUIRECTANGULAR
from src.scene import GaussianScene, rgb_to_sh0

_log_sigma = lambda sigma: np.log(np.asarray(sigma, dtype=np.float32))
_EQUIRECT_TEST_WIDTH = 256
_EQUIRECT_TEST_HEIGHT = 128
_EQUIRECT_LONGITUDE_COUNT = 32
_EQUIRECT_LATITUDE_COUNT = 15
_EQUIRECT_SPHERE_RADIUS = np.float32(3.0)
_EQUIRECT_STRESS_EDGE_PX = np.float32(0.1)
_EQUIRECT_SPLAT_SIGMA = 0.006
_EQUIRECT_RADIUS_SCALE = 1.6
_EQUIRECT_LIST_CAPACITY_MULTIPLIER = 64
_EQUIRECT_SPLAT_COLOR = np.array([[0.8, 0.7, 0.6]], dtype=np.float32)
_EQUIRECT_SPLAT_OPACITY = 0.8
_EQUIRECT_ALPHA_CUTOFF = 1.0 / 255.0


def _equirectangular_camera() -> Camera:
    return Camera.look_at(
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        near=0.1,
        far=50.0,
        projection_model=PROJECTION_MODEL_EQUIRECTANGULAR,
    )


def _equirectangular_rays(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    cos_phi = np.cos(phi)
    return np.stack((cos_phi * np.sin(theta), -np.sin(phi), cos_phi * np.cos(theta)), axis=-1).astype(np.float32)


def _latlong_sphere_shell() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_ids, lat_ids = np.meshgrid(
        np.arange(_EQUIRECT_LONGITUDE_COUNT, dtype=np.float32),
        np.arange(_EQUIRECT_LATITUDE_COUNT, dtype=np.float32),
    )
    theta = np.float32(2.0 * np.pi) * ((lon_ids + np.float32(0.5)) / np.float32(_EQUIRECT_LONGITUDE_COUNT)) - np.float32(np.pi)
    phi = np.float32(np.pi) * (np.float32(0.5) - (lat_ids + np.float32(0.5)) / np.float32(_EQUIRECT_LATITUDE_COUNT))
    positions = _equirectangular_rays(theta, phi).reshape(-1, 3) * _EQUIRECT_SPHERE_RADIUS
    expected_screen = np.stack(
        (
            (lon_ids + np.float32(0.5)) * np.float32(_EQUIRECT_TEST_WIDTH) / np.float32(_EQUIRECT_LONGITUDE_COUNT),
            (lat_ids + np.float32(0.5)) * np.float32(_EQUIRECT_TEST_HEIGHT) / np.float32(_EQUIRECT_LATITUDE_COUNT),
        ),
        axis=-1,
    ).reshape(-1, 2)
    expected_bins = np.stack((lon_ids, lat_ids), axis=-1).reshape(-1, 2).astype(np.int32)
    return positions.astype(np.float32), expected_screen.astype(np.float32), expected_bins


def _equirectangular_edge_stress_positions() -> np.ndarray:
    seam_eps = np.float32(2.0 * np.pi) * _EQUIRECT_STRESS_EDGE_PX / np.float32(_EQUIRECT_TEST_WIDTH)
    pole_eps = np.float32(np.pi) * _EQUIRECT_STRESS_EDGE_PX / np.float32(_EQUIRECT_TEST_HEIGHT)
    theta = np.array((-np.pi + seam_eps, np.pi - seam_eps, 0.0, 0.0), dtype=np.float32)
    phi = np.array((0.0, 0.0, 0.5 * np.pi - pole_eps, -0.5 * np.pi + pole_eps), dtype=np.float32)
    return _equirectangular_rays(theta, phi) * _EQUIRECT_SPHERE_RADIUS


def _scene_from_positions(positions: np.ndarray, sigma: float = _EQUIRECT_SPLAT_SIGMA) -> GaussianScene:
    count = int(positions.shape[0])
    return GaussianScene(
        positions=np.asarray(positions, dtype=np.float32).reshape(count, 3),
        scales=np.full((count, 3), _log_sigma(sigma), dtype=np.float32),
        rotations=np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), count, axis=0),
        opacities=np.full((count,), _EQUIRECT_SPLAT_OPACITY, dtype=np.float32),
        colors=np.repeat(_EQUIRECT_SPLAT_COLOR, count, axis=0),
        sh_coeffs=np.zeros((count, 1, 3), dtype=np.float32),
    )


def test_projection_keeps_in_view_splat_for_highly_distorted_camera(device):
    scene = GaussianScene(
        positions=np.array([[-11.5, 0.0, 19.80808]], dtype=np.float32),
        scales=np.full((1, 3), _log_sigma(0.01), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.8], dtype=np.float32),
        colors=np.array([[0.8, 0.7, 0.6]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        near=0.1,
        far=50.0,
        distortion_k1=-0.6,
        distortion_k2=0.15,
    )
    renderer = GaussianRenderer(device, width=256, height=256, radius_scale=1.6, list_capacity_multiplier=16)

    screen_point, ok = camera.project_world_to_screen(scene.positions[0], renderer.width, renderer.height)
    projected = project_splats(scene, camera, renderer.width, renderer.height, renderer.radius_scale)
    debug = renderer.debug_pipeline_data(scene, camera)

    assert ok
    assert 0.0 <= float(screen_point[0]) < float(renderer.width)
    assert 0.0 <= float(screen_point[1]) < float(renderer.height)
    assert int(projected.valid[0]) == 1
    assert int(np.asarray(debug["splat_visible"], dtype=np.uint32)[0]) == 1
    assert int(debug["generated_entries"]) > 0


def test_projection_keeps_side_splat_for_equirectangular_camera(device):
    scene = GaussianScene(
        positions=np.array([[3.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((1, 3), _log_sigma(0.01), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.8], dtype=np.float32),
        colors=np.array([[0.8, 0.7, 0.6]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        near=0.1,
        far=50.0,
        projection_model=PROJECTION_MODEL_EQUIRECTANGULAR,
    )
    renderer = GaussianRenderer(device, width=256, height=128, radius_scale=1.6, list_capacity_multiplier=16)

    screen_point, ok = camera.project_world_to_screen(scene.positions[0], renderer.width, renderer.height)
    projected = project_splats(scene, camera, renderer.width, renderer.height, renderer.radius_scale)
    debug = renderer.debug_pipeline_data(scene, camera)

    assert ok
    assert 0.0 <= float(screen_point[0]) < float(renderer.width)
    assert 0.0 <= float(screen_point[1]) < float(renderer.height)
    assert int(projected.valid[0]) == 1
    assert int(np.asarray(debug["splat_visible"], dtype=np.uint32)[0]) == 1
    assert int(debug["generated_entries"]) > 0


def test_equirectangular_camera_projects_latlong_sphere_uniformly():
    camera = _equirectangular_camera()
    positions, expected_screen, expected_bins = _latlong_sphere_shell()
    projected = np.zeros_like(expected_screen)
    valid = np.zeros((positions.shape[0],), dtype=np.bool_)
    for index, position in enumerate(positions):
        projected[index], valid[index] = camera.project_world_to_screen(position, _EQUIRECT_TEST_WIDTH, _EQUIRECT_TEST_HEIGHT)

    lon_bins = np.clip(
        np.floor(projected[:, 0] * np.float32(_EQUIRECT_LONGITUDE_COUNT) / np.float32(_EQUIRECT_TEST_WIDTH)).astype(np.int32),
        0,
        _EQUIRECT_LONGITUDE_COUNT - 1,
    )
    lat_bins = np.clip(
        np.floor(projected[:, 1] * np.float32(_EQUIRECT_LATITUDE_COUNT) / np.float32(_EQUIRECT_TEST_HEIGHT)).astype(np.int32),
        0,
        _EQUIRECT_LATITUDE_COUNT - 1,
    )
    occupancy = np.zeros((_EQUIRECT_LATITUDE_COUNT, _EQUIRECT_LONGITUDE_COUNT), dtype=np.int32)
    np.add.at(occupancy, (lat_bins, lon_bins), 1)

    assert np.all(valid)
    np.testing.assert_allclose(projected, expected_screen, rtol=0.0, atol=1e-4)
    np.testing.assert_array_equal(np.stack((lon_bins, lat_bins), axis=1), expected_bins)
    np.testing.assert_array_equal(occupancy, np.ones_like(occupancy))


def test_equirectangular_renderer_keeps_full_sphere_shell_visible(device):
    shell_positions, _, _ = _latlong_sphere_shell()
    positions = np.concatenate((shell_positions, _equirectangular_edge_stress_positions()), axis=0)
    scene = _scene_from_positions(positions)
    camera = _equirectangular_camera()
    renderer = GaussianRenderer(
        device,
        width=_EQUIRECT_TEST_WIDTH,
        height=_EQUIRECT_TEST_HEIGHT,
        radius_scale=_EQUIRECT_RADIUS_SCALE,
        alpha_cutoff=_EQUIRECT_ALPHA_CUTOFF,
        list_capacity_multiplier=_EQUIRECT_LIST_CAPACITY_MULTIPLIER,
    )

    projected = project_splats(scene, camera, renderer.width, renderer.height, renderer.radius_scale, alpha_cutoff=renderer.alpha_cutoff)
    debug = renderer.debug_pipeline_data(scene, camera)
    gpu_visible = np.asarray(debug["splat_visible"], dtype=np.uint32)
    gpu_center_radius_depth = np.asarray(debug["screen_center_radius_depth"], dtype=np.float32)
    gpu_ellipse_conic = np.asarray(debug["screen_ellipse_conic"], dtype=np.float32)[:, :3]

    assert int(np.sum(projected.valid, dtype=np.uint32)) == scene.count
    assert int(np.sum(gpu_visible, dtype=np.uint32)) == scene.count
    assert int(debug["generated_entries"]) > scene.count
    assert np.all(np.isfinite(projected.center_radius_depth))
    assert np.all(np.isfinite(projected.ellipse_conic))
    np.testing.assert_allclose(gpu_center_radius_depth, projected.center_radius_depth, rtol=0.0, atol=3e-3)
    np.testing.assert_allclose(gpu_ellipse_conic, projected.ellipse_conic, rtol=5e-3, atol=2e-4)


def test_equirectangular_renderer_samples_seam_column_at_pixel_center(device):
    width, height = 64, 32
    theta = np.array((-np.pi + np.pi / np.float32(width),), dtype=np.float32)
    phi = np.array((np.pi * (0.5 - (0.5 * np.float32(height) + 0.5) / np.float32(height)),), dtype=np.float32)
    scene = _scene_from_positions(_equirectangular_rays(theta, phi) * _EQUIRECT_SPHERE_RADIUS, sigma=0.025)
    scene.sh_coeffs = rgb_to_sh0(scene.colors)[:, None, :].astype(np.float32)
    camera = _equirectangular_camera()
    renderer = GaussianRenderer(
        device,
        width=width,
        height=height,
        radius_scale=_EQUIRECT_RADIUS_SCALE,
        alpha_cutoff=_EQUIRECT_ALPHA_CUTOFF,
        list_capacity_multiplier=16,
        allocate_training_work_buffers=False,
        allocate_grad_work_buffers=False,
    )

    debug = renderer.debug_pipeline_data(scene, camera)
    tile_ranges = np.asarray(debug["tile_ranges"], dtype=np.uint32)
    seam_tile_y = (height // 2) // renderer.tile_size
    left_tile = seam_tile_y * renderer.tile_width
    right_tile = left_tile + renderer.tile_width - 1
    image = renderer.render(scene, camera, background=(0.0, 0.0, 0.0)).image[:, :, :3]

    assert float(np.max(image[:, 0, :])) > 0.05
    assert int(tile_ranges[left_tile, 1]) > int(tile_ranges[left_tile, 0])
    assert int(tile_ranges[right_tile, 1]) > int(tile_ranges[right_tile, 0])

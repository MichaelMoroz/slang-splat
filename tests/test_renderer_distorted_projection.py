from __future__ import annotations

import numpy as np

from reference_impls.reference_cpu import project_splats
from src.renderer import Camera, GaussianRenderer, PROJECTION_MODEL_EQUIRECTANGULAR
from src.scene import GaussianScene

_log_sigma = lambda sigma: np.log(np.asarray(sigma, dtype=np.float32))


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

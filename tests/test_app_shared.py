from __future__ import annotations

import numpy as np

from src.app.shared import build_training_params, estimate_scene_bounds
from src.scene import GaussianScene


def _scene() -> GaussianScene:
    return GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [50.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]], dtype=np.float32),
        rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (3, 1)),
        opacities=np.array([0.9, 0.8, 0.01], dtype=np.float32),
        colors=np.ones((3, 3), dtype=np.float32),
        sh_coeffs=np.zeros((3, 1, 3), dtype=np.float32),
    )


def test_estimate_scene_bounds_prefers_weighted_core():
    bounds = estimate_scene_bounds(_scene())
    assert 0.0 <= float(bounds.center[0]) < 5.0
    assert np.allclose(bounds.center[1:], np.zeros((2,), dtype=np.float32))
    assert float(bounds.radius) > 1.0


def test_build_training_params_clamps_ranges():
    params = build_training_params(
        background=(0.1, 0.2, 0.3),
        base_lr=10.0,
        lr_pos_mul=20.0,
        lr_scale_mul=0.01,
        lr_rot_mul=1.0,
        lr_color_mul=1.0,
        lr_opacity_mul=1.0,
        beta1=2.0,
        beta2=-1.0,
        epsilon=1e-20,
        grad_clip=0.0,
        grad_norm_clip=1e9,
        max_update=0.0,
        min_scale=2.0,
        max_scale=1.0,
        max_anisotropy=0.5,
        min_opacity=0.8,
        max_opacity=0.2,
        position_abs_max=0.0,
        near=5.0,
        far=1.0,
        scale_l2_weight=-1.0,
        lambda_dssim=2.0,
        mcmc_position_noise_enabled=True,
        mcmc_position_noise_scale=-5.0,
        mcmc_opacity_gate_sharpness=-1.0,
        mcmc_opacity_gate_center=2.0,
        low_quality_reinit_enabled=False,
    )
    assert params.adam.position_lr == 10.0
    assert params.adam.scale_lr == 0.1
    assert params.stability.max_scale == params.stability.min_scale == 2.0
    assert params.stability.max_opacity == params.stability.min_opacity == 0.8
    assert params.training.far > params.training.near
    assert params.training.scale_l2_weight == 0.0
    assert params.training.lambda_dssim == 1.0

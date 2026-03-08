from __future__ import annotations

import numpy as np

from pathlib import Path

from src.app.shared import apply_training_profile, build_training_params, estimate_scene_bounds
from src.scene import GaussianScene
from src.viewer.app import default_training_params
from src.viewer.ui import default_control_values, default_prune_small_threshold


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
        opacity_reg_weight=-1.0,
        lambda_dssim=2.0,
        mcmc_position_noise_enabled=True,
        mcmc_position_noise_scale=-5.0,
        mcmc_opacity_gate_sharpness=-1.0,
        mcmc_opacity_gate_center=2.0,
        max_gaussians=-1,
        densify_from_iter=-5,
        densify_until_iter=1,
        densification_interval=0,
        densify_grad_threshold=-1.0,
        percent_dense=2.0,
        prune_min_opacity=2.0,
        screen_size_prune_threshold=-1.0,
        world_size_prune_ratio=-1.0,
        opacity_reset_interval=-1,
    )
    assert params.adam.position_lr == 10.0
    assert params.adam.scale_lr == 0.1
    assert params.stability.max_scale == params.stability.min_scale == 2.0
    assert params.stability.max_opacity == params.stability.min_opacity == 0.8
    assert params.training.far > params.training.near
    assert params.training.scale_l2_weight == 0.0
    assert params.training.opacity_reg_weight == 0.0
    assert params.training.lambda_dssim == 1.0
    assert params.training.max_gaussians == 0
    assert params.training.densify_from_iter == 0
    assert params.training.densify_until_iter == 1
    assert params.training.densification_interval == 1
    assert params.training.densify_grad_threshold == 0.0
    assert params.training.percent_dense == 1.0
    assert params.training.prune_min_opacity == 1.0
    assert params.training.screen_size_prune_threshold == 0.0
    assert params.training.world_size_prune_ratio == 0.0
    assert params.training.opacity_reset_interval == 0


def test_default_training_params_match_mcmc_reference_defaults():
    params = default_training_params()
    assert params.training.mcmc_position_noise_enabled is True
    assert params.training.mcmc_position_noise_scale == 5e5
    assert params.training.mcmc_opacity_gate_sharpness == 100.0
    assert params.training.mcmc_opacity_gate_center == 0.995
    assert params.training.opacity_reg_weight == 1e-3
    assert params.training.max_gaussians == 200000
    assert params.training.densify_from_iter == 500
    assert params.training.densify_until_iter == 15000
    assert params.training.densification_interval == 100
    assert params.training.opacity_reset_interval == 3000


def test_bicycle_images4_profile_applies_psnr_overrides():
    params, profile = apply_training_profile(default_training_params(), "auto", dataset_root=Path("dataset/bicycle"), images_subdir="images_4")
    assert profile.name == "bicycle-images4-psnr"
    assert params.adam.position_lr == 1.6e-4
    assert params.adam.scale_lr == 5e-3
    assert params.adam.opacity_lr == 5e-2
    assert params.training.mcmc_position_noise_enabled is False
    assert params.training.background == (1.0, 1.0, 1.0)
    assert params.training.lambda_dssim == 0.0
    assert params.training.max_gaussians == 200000
    assert params.training.densify_from_iter == 10000
    assert params.training.densify_until_iter == 10000
    assert params.training.screen_size_prune_threshold == 0.0
    assert params.training.world_size_prune_ratio == 0.0
    assert params.training.opacity_reset_interval == 0


def test_default_prune_small_threshold_tracks_min_scale_default():
    assert default_prune_small_threshold() == 4.0 * float(default_control_values("Train Stability")["min_scale"])

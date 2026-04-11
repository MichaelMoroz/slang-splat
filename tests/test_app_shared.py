from __future__ import annotations

from pathlib import Path

import numpy as np

from src.app.shared import apply_training_profile, build_training_params, estimate_scene_bounds
from src.scene import GaussianInitHyperParams, GaussianScene
from src.training import DEPTH_RATIO_GRAD_MIN_BAND_WIDTH, TRAIN_BACKGROUND_MODE_RANDOM, TrainingHyperParams
from src.viewer.app import default_training_params
from src.viewer.session import resolve_effective_training_setup
from src.viewer.ui import default_control_values


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
        lr_pos_mul=200.0,
        lr_pos_stage1_mul=123.0,
        lr_pos_stage2_mul=234.0,
        lr_pos_stage3_mul=345.0,
        lr_scale_mul=0.01,
        lr_rot_mul=30.0,
        lr_color_mul=40.0,
        lr_opacity_mul=50.0,
        lr_sh_mul=67.0,
        lr_sh_stage1_mul=78.0,
        lr_sh_stage2_mul=89.0,
        lr_sh_stage3_mul=90.0,
        beta1=2.0,
        beta2=-1.0,
        grad_clip=0.0,
        grad_norm_clip=1e9,
        max_update=0.0,
        max_scale=1.0,
        max_anisotropy=0.5,
        min_opacity=0.8,
        max_opacity=0.2,
        position_abs_max=0.0,
        near=5.0,
        far=1.0,
        background_mode=7,
        use_sh=0,
        scale_l2_weight=-1.0,
        scale_abs_reg_weight=-1.0,
        sh1_reg_weight=-1.0,
        opacity_reg_weight=-1.0,
        refinement_loss_weight=-1.0,
        refinement_target_edge_weight=-2.0,
        depth_ratio_grad_min=0.2,
        depth_ratio_grad_max=0.1,
        max_gaussians=-1,
    )
    assert params.adam.position_lr == 200.0
    assert params.adam.scale_lr == 0.01
    assert params.adam.rotation_lr == 30.0
    assert params.adam.color_lr == 40.0
    assert params.adam.opacity_lr == 50.0
    assert params.stability.max_scale == 1.0
    assert params.stability.max_opacity == params.stability.min_opacity == 0.8
    assert params.training.far > params.training.near
    assert params.training.background_mode == TRAIN_BACKGROUND_MODE_RANDOM
    assert params.training.use_target_alpha_mask is False
    assert params.training.use_sh is False
    assert params.training.sh_band == 0
    assert params.training.scale_l2_weight == 0.0
    assert params.training.scale_abs_reg_weight == 0.0
    assert params.training.sh1_reg_weight == 0.0
    assert params.training.opacity_reg_weight == 0.0
    assert params.training.refinement_loss_weight == 0.0
    assert params.training.refinement_target_edge_weight == 0.0
    assert params.training.density_regularizer == 0.02
    assert params.training.color_non_negative_reg == 0.01
    assert params.training.depth_ratio_weight == 0.5
    assert params.training.max_screen_fraction == 0.25
    assert params.training.ssim_weight == 0.05
    assert params.training.ssim_c2 == 9e-4
    assert params.training.depth_ratio_grad_min == 0.2
    assert params.training.depth_ratio_grad_max == 0.2 + DEPTH_RATIO_GRAD_MIN_BAND_WIDTH
    assert params.training.max_allowed_density_start == 5.0
    assert params.training.max_allowed_density == 12.0
    assert params.training.lr_pos_mul == 200.0
    assert params.training.lr_pos_stage1_mul == 123.0
    assert params.training.lr_pos_stage2_mul == 234.0
    assert params.training.lr_pos_stage3_mul == 345.0
    assert params.training.lr_sh_mul == 67.0
    assert params.training.lr_sh_stage1_mul == 78.0
    assert params.training.lr_sh_stage2_mul == 89.0
    assert params.training.lr_sh_stage3_mul == 90.0
    assert params.training.max_screen_fraction_stage1 == 0.05
    assert params.training.max_screen_fraction_stage2 == 0.04
    assert params.training.max_screen_fraction_stage3 == 0.03
    assert params.training.position_random_step_noise_lr == 5e5
    assert np.isclose(params.training.position_random_step_noise_stage1_lr, 466666.6666666667)
    assert np.isclose(params.training.position_random_step_noise_stage2_lr, 416666.6666666667)
    assert params.training.position_random_step_noise_stage3_lr == 0.0
    assert params.training.position_random_step_opacity_gate_center == 0.005
    assert params.training.position_random_step_opacity_gate_sharpness == 100.0
    assert params.training.lr_schedule_stage1_lr == 0.002
    assert params.training.lr_schedule_stage2_lr == 0.001
    assert params.training.lr_schedule_end_lr == 1.5e-4
    assert params.training.lr_schedule_stage1_step == 3000
    assert params.training.lr_schedule_stage2_step == 14000
    assert params.training.depth_ratio_stage1_weight == 0.03
    assert params.training.depth_ratio_stage2_weight == 0.01
    assert params.training.depth_ratio_stage3_weight == 0.001
    assert params.training.ssim_weight_stage1 == 0.1
    assert params.training.ssim_weight_stage2 == 0.3
    assert params.training.ssim_weight_stage3 == 0.4
    assert params.training.use_sh_stage1 is False
    assert params.training.use_sh_stage2 is True
    assert params.training.use_sh_stage3 is True
    assert params.training.sh_band_stage1 == 0
    assert params.training.sh_band_stage2 == 2
    assert params.training.sh_band_stage3 == 3
    assert params.training.refinement_min_contribution_percent == 1e-05
    assert params.training.refinement_min_contribution_decay == 0.995
    assert params.training.refinement_opacity_mul == 1.0
    assert params.training.max_gaussians == 0
    assert params.training.train_subsample_factor == 0


def test_default_training_params_match_fixed_count_defaults():
    params = default_training_params()
    assert params.training.background_mode == TRAIN_BACKGROUND_MODE_RANDOM
    assert params.training.background == (1.0, 1.0, 1.0)
    assert params.training.use_target_alpha_mask is False
    assert params.training.use_sh is False
    assert params.training.sh_band == 0
    assert params.training.scale_l2_weight == 0.0
    assert params.training.scale_abs_reg_weight == 0.01
    assert params.training.sh1_reg_weight == 0.01
    assert params.training.opacity_reg_weight == 0.01
    assert params.training.refinement_loss_weight == 0.25
    assert params.training.refinement_target_edge_weight == 0.75
    assert params.training.density_regularizer == 0.02
    assert params.training.color_non_negative_reg == 0.01
    assert params.training.depth_ratio_weight == 0.5
    assert params.training.max_screen_fraction == 0.25
    assert params.training.ssim_weight == 0.05
    assert params.training.ssim_c2 == 9e-4
    assert params.training.depth_ratio_grad_min == 0.0
    assert params.training.depth_ratio_grad_max == 0.1
    assert params.training.max_allowed_density_start == 5.0
    assert params.training.max_allowed_density == 12.0
    assert params.training.lr_pos_mul == 0.5
    assert params.training.lr_pos_stage1_mul == 0.1
    assert params.training.lr_pos_stage2_mul == 0.05
    assert params.training.lr_pos_stage3_mul == 0.02
    assert params.training.lr_sh_mul == 0.1
    assert params.training.lr_sh_stage1_mul == 0.1
    assert params.training.lr_sh_stage2_mul == 0.1
    assert params.training.lr_sh_stage3_mul == 0.1
    assert params.training.max_screen_fraction_stage1 == 0.07
    assert params.training.max_screen_fraction_stage2 == 0.02
    assert params.training.max_screen_fraction_stage3 == 0.007
    assert params.training.position_random_step_noise_lr == 5e5
    assert np.isclose(params.training.position_random_step_noise_stage1_lr, 466666.6666666667)
    assert np.isclose(params.training.position_random_step_noise_stage2_lr, 416666.6666666667)
    assert params.training.position_random_step_noise_stage3_lr == 0.0
    assert params.training.position_random_step_opacity_gate_center == 0.005
    assert params.training.position_random_step_opacity_gate_sharpness == 100.0
    assert params.training.lr_schedule_stage1_lr == 0.002
    assert params.training.lr_schedule_stage2_lr == 0.001
    assert params.training.lr_schedule_end_lr == 1.5e-4
    assert params.training.lr_schedule_stage1_step == 3000
    assert params.training.lr_schedule_stage2_step == 14000
    assert params.training.depth_ratio_stage1_weight == 0.03
    assert params.training.depth_ratio_stage2_weight == 0.01
    assert params.training.depth_ratio_stage3_weight == 0.001
    assert params.training.ssim_weight_stage1 == 0.1
    assert params.training.ssim_weight_stage2 == 0.3
    assert params.training.ssim_weight_stage3 == 0.4
    assert params.training.use_sh_stage1 is False
    assert params.training.use_sh_stage2 is True
    assert params.training.use_sh_stage3 is True
    assert params.training.sh_band_stage1 == 0
    assert params.training.sh_band_stage2 == 2
    assert params.training.sh_band_stage3 == 3
    assert params.training.refinement_growth_ratio == 0.04
    assert params.training.refinement_growth_start_step == 500
    assert params.training.refinement_alpha_cull_threshold == 1e-2
    assert params.training.refinement_min_contribution_percent == 1e-05
    assert params.training.refinement_min_contribution_decay == 0.995
    assert params.training.refinement_opacity_mul == 1.0
    assert params.training.max_gaussians == 1_000_000
    assert params.training.train_subsample_factor == 0


def test_auto_profile_resolves_to_legacy_defaults():
    params, profile = apply_training_profile(default_training_params(), "auto", dataset_root=Path("dataset/bicycle"), images_subdir="images_4")
    assert profile.name == "legacy"
    assert params.training.scale_abs_reg_weight == 0.01
    assert params.training.sh1_reg_weight == 0.01
    assert params.training.depth_ratio_weight == 0.5
    assert params.training.refinement_loss_weight == 0.25
    assert params.training.refinement_target_edge_weight == 0.75
    assert params.training.depth_ratio_grad_min == 0.0
    assert params.training.depth_ratio_grad_max == 0.1
    assert params.training.opacity_reg_weight == 0.01


def test_viewer_effective_training_setup_keeps_requested_init_opacity():
    class _StubViewer:
        def __init__(self) -> None:
            self.s = type(
                "_State",
                (),
                {
                    "colmap_root": Path("dataset/bicycle"),
                    "cached_training_setup_signature": None,
                    "cached_training_setup": None,
                    "colmap_import": None,
                },
            )()

        def training_params(self):
            return default_training_params()

        def init_params(self):
            return type("_Init", (), {"seed": 1234, "hparams": GaussianInitHyperParams(initial_opacity=0.5)})()

    init, params, init_hparams, profile = resolve_effective_training_setup(_StubViewer())
    assert init.seed == 1234
    assert profile.name == "legacy"
    assert params.training.scale_abs_reg_weight == 0.01
    assert params.training.sh1_reg_weight == 0.01
    assert params.training.depth_ratio_weight == 0.5
    assert params.training.refinement_loss_weight == 0.25
    assert params.training.refinement_target_edge_weight == 0.75
    assert params.training.depth_ratio_grad_min == 0.0
    assert params.training.depth_ratio_grad_max == 0.1
    assert params.training.opacity_reg_weight == 0.01
    assert init_hparams.initial_opacity == 0.5


def test_training_hparams_clamp_depth_ratio_grad_band() -> None:
    params = TrainingHyperParams(depth_ratio_grad_min=0.05, depth_ratio_grad_max=0.01)

    assert params.depth_ratio_grad_min == 0.05
    assert params.depth_ratio_grad_max == 0.05 + DEPTH_RATIO_GRAD_MIN_BAND_WIDTH


def test_training_hparams_clamp_schedule_breakpoints() -> None:
    params = TrainingHyperParams(
        lr_schedule_steps=3000,
        lr_schedule_stage1_step=4000,
        lr_schedule_stage2_step=1000,
    )

    assert params.lr_schedule_stage1_step == 3000
    assert params.lr_schedule_stage2_step == 3000


def test_viewer_defaults_expose_only_fixed_count_training_controls():
    defaults = default_control_values("Train Optimizer", "Train Stability")
    assert "scale_l2" in defaults
    assert "scale_abs_reg" in defaults
    assert "sh1_reg" in defaults
    assert "opacity_reg" in defaults
    assert defaults["color_non_negative_reg"] == 0.01
    assert defaults["max_screen_fraction"] == 0.25
    assert defaults["max_screen_fraction_stage1"] == 0.07
    assert defaults["max_screen_fraction_stage2"] == 0.02
    assert defaults["max_screen_fraction_stage3"] == 0.007
    assert defaults["ssim_weight"] == 0.05
    assert defaults["ssim_weight_stage1"] == 0.1
    assert defaults["ssim_weight_stage2"] == 0.3
    assert defaults["ssim_weight_stage3"] == 0.4
    assert defaults["ssim_c2"] == 9e-4
    assert "lambda_dssim" not in defaults
    assert "mcmc_growth_ratio" not in defaults

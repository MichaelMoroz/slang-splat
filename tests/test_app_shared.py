from __future__ import annotations

from pathlib import Path

import numpy as np

from src.app.shared import apply_training_profile, build_training_params, estimate_scene_bounds
from src.scene import GaussianInitHyperParams, GaussianScene
from src.training import TRAIN_BACKGROUND_MODE_RANDOM
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
        lr_pos_mul=20.0,
        lr_scale_mul=0.01,
        lr_rot_mul=1.0,
        lr_color_mul=1.0,
        lr_opacity_mul=1.0,
        beta1=2.0,
        beta2=-1.0,
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
        background_mode=7,
        use_sh=0,
        scale_l2_weight=-1.0,
        scale_abs_reg_weight=-1.0,
        sh1_reg_weight=-1.0,
        opacity_reg_weight=-1.0,
        max_gaussians=-1,
    )
    assert params.adam.position_lr == 10.0
    assert params.adam.scale_lr == 0.1
    assert params.stability.max_scale == params.stability.min_scale == 2.0
    assert params.stability.max_opacity == params.stability.min_opacity == 0.8
    assert params.training.far > params.training.near
    assert params.training.background_mode == TRAIN_BACKGROUND_MODE_RANDOM
    assert params.training.use_sh is False
    assert params.training.scale_l2_weight == 0.0
    assert params.training.scale_abs_reg_weight == 0.0
    assert params.training.sh1_reg_weight == 0.0
    assert params.training.opacity_reg_weight == 0.0
    assert params.training.density_regularizer == 0.05
    assert params.training.max_allowed_density_start == 5.0
    assert params.training.max_allowed_density == 12.0
    assert params.training.position_random_step_noise_lr == 5e5
    assert params.training.position_random_step_opacity_gate_center == 0.005
    assert params.training.position_random_step_opacity_gate_sharpness == 100.0
    assert params.training.refinement_contribution_cull_threshold == 0.001
    assert params.training.refinement_contribution_cull_decay == 0.95
    assert params.training.max_gaussians == 0


def test_default_training_params_match_fixed_count_defaults():
    params = default_training_params()
    assert params.training.background_mode == TRAIN_BACKGROUND_MODE_RANDOM
    assert params.training.background == (1.0, 1.0, 1.0)
    assert params.training.use_sh is True
    assert params.training.scale_l2_weight == 0.0
    assert params.training.scale_abs_reg_weight == 0.01
    assert params.training.sh1_reg_weight == 0.01
    assert params.training.opacity_reg_weight == 0.01
    assert params.training.density_regularizer == 0.05
    assert params.training.max_allowed_density_start == 5.0
    assert params.training.max_allowed_density == 12.0
    assert params.training.position_random_step_noise_lr == 5e5
    assert params.training.position_random_step_opacity_gate_center == 0.005
    assert params.training.position_random_step_opacity_gate_sharpness == 100.0
    assert params.training.refinement_growth_ratio == 0.02
    assert params.training.refinement_growth_start_step == 500
    assert params.training.refinement_alpha_cull_threshold == 1e-2
    assert params.training.refinement_contribution_cull_threshold == 0.001
    assert params.training.refinement_contribution_cull_decay == 0.95
    assert params.training.max_gaussians == 1_000_000


def test_auto_profile_resolves_to_legacy_defaults():
    params, profile = apply_training_profile(default_training_params(), "auto", dataset_root=Path("dataset/bicycle"), images_subdir="images_4")
    assert profile.name == "legacy"
    assert params.training.scale_abs_reg_weight == 0.01
    assert params.training.sh1_reg_weight == 0.01
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

        def _selected_images_subdir(self) -> str:
            return "images_4"

        def training_params(self):
            return default_training_params()

        def init_params(self):
            return type("_Init", (), {"seed": 1234, "hparams": GaussianInitHyperParams(initial_opacity=0.5)})()

    init, params, init_hparams, profile = resolve_effective_training_setup(_StubViewer())
    assert init.seed == 1234
    assert profile.name == "legacy"
    assert params.training.scale_abs_reg_weight == 0.01
    assert params.training.sh1_reg_weight == 0.01
    assert params.training.opacity_reg_weight == 0.01
    assert init_hparams.initial_opacity == 0.5


def test_viewer_defaults_expose_only_fixed_count_training_controls():
    defaults = default_control_values("Train Optimizer", "Train Stability")
    assert "scale_l2" in defaults
    assert "scale_abs_reg" in defaults
    assert "sh1_reg" in defaults
    assert "opacity_reg" in defaults
    assert "lambda_dssim" not in defaults
    assert "mcmc_growth_ratio" not in defaults

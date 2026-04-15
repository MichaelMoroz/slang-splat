from __future__ import annotations

from ..repo_defaults import training_build_arg_defaults

TRAINING_BUILD_ARG_DEFAULTS: dict[str, object] = training_build_arg_defaults()
DEFAULT_DEBUG_CONTRIBUTION_RANGE_PERCENT = (0.001, 1.0)
DEFAULT_REFINEMENT_MIN_CONTRIBUTION_PERCENT = float(TRAINING_BUILD_ARG_DEFAULTS["refinement_min_contribution_percent"])
DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY = float(TRAINING_BUILD_ARG_DEFAULTS["refinement_min_contribution_decay"])
DEFAULT_REFINEMENT_CLONE_SCALE_MUL = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_clone_scale_mul", 1.0))
DEPTH_RATIO_GRAD_MIN_BAND_WIDTH = 1e-4
DEFAULT_DEPTH_RATIO_GRAD_MIN = float(TRAINING_BUILD_ARG_DEFAULTS["depth_ratio_grad_min"])
DEFAULT_DEPTH_RATIO_GRAD_MAX = float(TRAINING_BUILD_ARG_DEFAULTS["depth_ratio_grad_max"])
DEFAULT_SSIM_WEIGHT = float(TRAINING_BUILD_ARG_DEFAULTS["ssim_weight"])
DEFAULT_SSIM_C2 = float(TRAINING_BUILD_ARG_DEFAULTS["ssim_c2"])
DEFAULT_MAX_SCREEN_FRACTION = float(TRAINING_BUILD_ARG_DEFAULTS["max_screen_fraction"])
DEFAULT_LR_SCHEDULE_STEPS = int(TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_steps"])
DEFAULT_LR_STAGE1_STEP = int(TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage1_step"])
DEFAULT_LR_STAGE2_STEP = int(TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage2_step"])

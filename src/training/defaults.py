from __future__ import annotations

from ..repo_defaults import training_build_arg_defaults

TRAINING_BUILD_ARG_DEFAULTS: dict[str, object] = training_build_arg_defaults()
DEFAULT_DEBUG_CONTRIBUTION_RANGE = (0.001, 1.0)
DEFAULT_REFINEMENT_MIN_CONTRIBUTION = int(TRAINING_BUILD_ARG_DEFAULTS["refinement_min_contribution"])
DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY = float(TRAINING_BUILD_ARG_DEFAULTS["refinement_min_contribution_decay"])
DEFAULT_REFINEMENT_CLONE_SCALE_MUL = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_clone_scale_mul", 1.0))
DEFAULT_REFINEMENT_SPLIT_BETA = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_split_beta", 0.28))
DEFAULT_REFINEMENT_GRAD_VARIANCE_WEIGHT_EXPONENT = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_grad_variance_weight_exponent", 0.1))
DEFAULT_REFINEMENT_CONTRIBUTION_WEIGHT_EXPONENT = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_contribution_weight_exponent", 0.1))
DEFAULT_SSIM_WEIGHT = float(TRAINING_BUILD_ARG_DEFAULTS["ssim_weight"])
DEFAULT_SSIM_C2 = float(TRAINING_BUILD_ARG_DEFAULTS["ssim_c2"])
DEFAULT_MAX_VISIBLE_ANGLE_DEG = float(TRAINING_BUILD_ARG_DEFAULTS["max_visible_angle_deg"])
DEFAULT_LR_SCHEDULE_STEPS = int(TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_steps"])
DEFAULT_LR_STAGE1_STEP = int(TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage1_step"])
DEFAULT_LR_STAGE2_STEP = int(TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage2_step"])

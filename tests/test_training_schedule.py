from src.app.shared import build_training_params
from src.training.gaussian_trainer import TrainingHyperParams
from src.training.schedule import resolve_max_opacity, resolve_sh_band


def test_resolve_sh_band_respects_global_cap_across_schedule() -> None:
    hparams = TrainingHyperParams(
        sh_band=3,
        max_sh_band=1,
        use_sh_stage1=True,
        use_sh_stage2=True,
        use_sh_stage3=True,
        use_sh_stage4=True,
        sh_band_stage1=3,
        sh_band_stage2=2,
        sh_band_stage3=3,
        sh_band_stage4=3,
        lr_schedule_steps=100,
        lr_schedule_stage1_step=20,
        lr_schedule_stage2_step=40,
        lr_schedule_stage3_step=60,
    )

    assert resolve_sh_band(hparams, 0) == 1
    assert resolve_sh_band(hparams, 25) == 1
    assert resolve_sh_band(hparams, 45) == 1
    assert resolve_sh_band(hparams, 75) == 1
    assert resolve_sh_band(hparams, 100) == 1


def test_training_params_threads_global_sh_cap() -> None:
    params = build_training_params(
        background=(1.0, 1.0, 1.0),
        sh_band=3,
        max_sh_band=2,
        sh_band_stage1=3,
        sh_band_stage2=3,
        sh_band_stage3=3,
        sh_band_stage4=3,
    )

    assert params.training.max_sh_band == 2
    assert resolve_sh_band(params.training, 0) == 2


def test_resolve_max_opacity_uses_staged_schedule() -> None:
    params = TrainingHyperParams(
        max_opacity=0.23,
        max_opacity_stage0=0.5,
        max_opacity_stage1=0.8,
        max_opacity_stage2=1.0,
        max_opacity_stage3=1.0,
        max_opacity_stage4=1.0,
        lr_schedule_steps=100,
        lr_schedule_stage1_step=20,
        lr_schedule_stage2_step=60,
        lr_schedule_stage3_step=80,
    )

    assert resolve_max_opacity(params, 0) == 0.5
    assert resolve_max_opacity(params, 20) == 0.8
    assert resolve_max_opacity(params, 60) == 1.0
    assert resolve_max_opacity(params, 80) == 1.0
    assert resolve_max_opacity(params, 100) == 1.0
    assert abs(resolve_max_opacity(params, 10) - 0.65) < 1e-6

    disabled = TrainingHyperParams(
        lr_schedule_enabled=False,
        max_opacity=0.23,
        max_opacity_stage0=0.5,
        max_opacity_stage1=0.8,
        max_opacity_stage2=1.0,
        max_opacity_stage3=1.0,
        max_opacity_stage4=1.0,
    )

    assert resolve_max_opacity(disabled, 100) == 0.23
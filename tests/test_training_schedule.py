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

def test_suggest_schedule_stretch_matches_frame_count_anchors() -> None:
    from src.training.schedule import suggest_schedule_stretch

    # Each doubling of the dataset over the 100-frame reference adds one unit.
    assert suggest_schedule_stretch(100) == 1.0
    assert suggest_schedule_stretch(200) == 2.0
    assert suggest_schedule_stretch(400) == 3.0
    assert suggest_schedule_stretch(800) == 4.0
    # Small datasets never shorten the authored schedule automatically.
    assert suggest_schedule_stretch(50) == 1.0
    assert suggest_schedule_stretch(1) == 1.0


def test_schedule_stretch_dilates_stage_boundaries_and_refinement() -> None:
    from src.training.schedule import (
        resolve_base_learning_rate,
        resolve_effective_refinement_interval,
        resolve_max_allowed_density,
        resolve_schedule_stretch,
        should_run_refinement_step,
    )

    base = TrainingHyperParams(lr_schedule_steps=30_000, refinement_growth_start_step=500, refinement_interval=200)
    stretched = TrainingHyperParams(lr_schedule_steps=30_000, refinement_growth_start_step=500, refinement_interval=200, schedule_stretch=3.0)

    assert resolve_schedule_stretch(base) == 1.0
    assert resolve_schedule_stretch(stretched) == 3.0

    # Every schedule value at step S under the authored timeline appears at 3*S under
    # the stretched one; the 30k end lands at 90k.
    for step in (0, 500, 5_000, 15_000, 30_000):
        assert resolve_base_learning_rate(stretched, step * 3) == resolve_base_learning_rate(base, step)
        assert resolve_sh_band(stretched, step * 3) == resolve_sh_band(base, step)
        assert abs(resolve_max_allowed_density(stretched, step * 3) - resolve_max_allowed_density(base, step)) < 1e-6
    assert resolve_base_learning_rate(stretched, 30_000) != resolve_base_learning_rate(base, 30_000)

    # Refinement start and cadence dilate: 500 -> 1500, interval 200 -> 600.
    assert resolve_effective_refinement_interval(stretched) == 600
    assert not should_run_refinement_step(stretched, 1_400)
    assert should_run_refinement_step(stretched, 1_800)
    assert should_run_refinement_step(base, 600)
    assert not should_run_refinement_step(stretched, 601)


def test_schedule_stretch_dilates_auto_downscale_ladder() -> None:
    from src.training.gaussian_trainer import TRAIN_DOWNSCALE_MODE_AUTO, resolve_effective_train_downscale_factor

    kwargs = dict(
        train_downscale_mode=TRAIN_DOWNSCALE_MODE_AUTO,
        train_auto_start_downscale=2,
        train_downscale_base_iters=1_000,
        train_downscale_iter_step=0,
    )
    base = TrainingHyperParams(**kwargs)
    stretched = TrainingHyperParams(**kwargs, schedule_stretch=2.0)

    assert resolve_effective_train_downscale_factor(base, 999) == 2
    assert resolve_effective_train_downscale_factor(base, 1_000) == 1
    assert resolve_effective_train_downscale_factor(stretched, 1_999) == 2
    assert resolve_effective_train_downscale_factor(stretched, 2_000) == 1


def test_build_training_params_threads_schedule_stretch() -> None:
    params = build_training_params(background=(0.0, 0.0, 0.0), schedule_stretch=2.5)
    assert params.training.schedule_stretch == 2.5
    assert TrainingHyperParams().schedule_stretch == 1.0

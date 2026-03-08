from __future__ import annotations

from dataclasses import replace
import time
from pathlib import Path

import pytest

from src.app.shared import renderer_kwargs
from src.renderer import GaussianRenderer
from src.scene import build_training_frames, initialize_scene_from_colmap_points, load_colmap_reconstruction, resolve_colmap_init_hparams
from src.training import GaussianTrainer
from src.viewer.app import default_images_subdir, default_init_params, default_renderer_params, default_training_params

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "garden"
IMAGES_SUBDIR = default_images_subdir()
SPARSE_SUBDIR = Path("sparse/0")
TRAIN_STEPS = 2000
REGRESSION_OPACITY_RESET_INTERVAL = 1000
PSNR_THRESHOLD_DB = 25.0
TRAIN_WIDTH = 256
TRAIN_HEIGHT = 168


def _require_dataset() -> Path:
    images_root = DATASET_ROOT / IMAGES_SUBDIR
    sparse_root = DATASET_ROOT / SPARSE_SUBDIR
    if not images_root.exists() or not sparse_root.exists():
        pytest.skip(f"Garden regression dataset subset is unavailable: expected {images_root} and {sparse_root}.")
    return DATASET_ROOT


def _build_trainer(device) -> GaussianTrainer:
    dataset_root = _require_dataset()
    init = default_init_params()
    params = default_training_params()
    training = replace(params.training, opacity_reset_interval=REGRESSION_OPACITY_RESET_INTERVAL)
    recon = load_colmap_reconstruction(dataset_root, sparse_subdir=str(SPARSE_SUBDIR))
    frames = build_training_frames(recon, images_subdir=IMAGES_SUBDIR)
    init_hparams = resolve_colmap_init_hparams(recon, 0, init.hparams)
    scene = initialize_scene_from_colmap_points(recon=recon, max_gaussians=0, seed=init.seed, init_hparams=init.hparams)
    renderer = GaussianRenderer(device, width=TRAIN_WIDTH, height=TRAIN_HEIGHT, **renderer_kwargs(default_renderer_params()))
    return GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=frames,
        adam_hparams=params.adam,
        stability_hparams=params.stability,
        training_hparams=training,
        seed=init.seed,
        scale_reg_reference=float(max(init_hparams.base_scale, 1e-8)),
    )


def test_garden_images4_training_reaches_25db_after_2000_steps_with_1k_reinit(device):
    trainer = _build_trainer(device)
    start = time.perf_counter()
    for _ in range(TRAIN_STEPS):
        trainer.step()
    elapsed = time.perf_counter() - start
    assert trainer.state.step == TRAIN_STEPS
    assert trainer.state.avg_psnr >= PSNR_THRESHOLD_DB, (
        f"Expected garden/{IMAGES_SUBDIR} training to stay above {PSNR_THRESHOLD_DB:.1f} dB after {TRAIN_STEPS} steps "
        f"with opacity reset every {REGRESSION_OPACITY_RESET_INTERVAL} steps; "
        f"final_psnr={trainer.state.last_psnr:.3f} dB final_avg_psnr={trainer.state.avg_psnr:.3f} dB "
        f"steps={trainer.state.step} elapsed={elapsed:.2f}s."
    )

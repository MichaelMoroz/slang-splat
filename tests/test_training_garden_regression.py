from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.app.shared import renderer_kwargs
from src.renderer import GaussianRenderer
from src.scene import build_training_frames, initialize_scene_from_colmap_points, load_colmap_reconstruction, resolve_colmap_init_hparams
from src.training import GaussianTrainer
from src.viewer.app import default_images_subdir, default_init_params, default_renderer_params, default_training_params

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "garden"
IMAGES_SUBDIR = default_images_subdir()
SPARSE_SUBDIR = Path("sparse/0")
TRAIN_TIMEOUT_SECONDS = 60.0
PSNR_THRESHOLD_DB = 25.0
TRAIN_WIDTH = 256
TRAIN_HEIGHT = 168


def _require_dataset() -> Path:
    images_root = DATASET_ROOT / IMAGES_SUBDIR
    sparse_root = DATASET_ROOT / SPARSE_SUBDIR
    if not images_root.exists() or not sparse_root.exists():
        pytest.skip(f"Garden regression dataset subset is unavailable: expected {images_root} and {sparse_root}.")
    return DATASET_ROOT


def _finite_peak(current: float, value: float) -> float:
    return max(current, float(value)) if np.isfinite(value) else current


def _build_trainer(device) -> GaussianTrainer:
    dataset_root = _require_dataset()
    init = default_init_params()
    params = default_training_params()
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
        training_hparams=params.training,
        seed=init.seed,
        scale_reg_reference=float(max(init_hparams.base_scale, 1e-8)),
    )


def test_garden_images4_training_reaches_25db_within_60_seconds(device):
    trainer = _build_trainer(device)
    start = time.perf_counter()
    deadline = start + TRAIN_TIMEOUT_SECONDS
    best_avg_psnr = float("-inf")
    best_last_psnr = float("-inf")
    while time.perf_counter() < deadline:
        trainer.step()
        best_avg_psnr = _finite_peak(best_avg_psnr, trainer.state.avg_psnr)
        best_last_psnr = _finite_peak(best_last_psnr, trainer.state.last_psnr)
    elapsed = time.perf_counter() - start
    assert trainer.state.step > 0
    assert best_avg_psnr >= PSNR_THRESHOLD_DB, (
        f"Expected garden/{IMAGES_SUBDIR} training to reach {PSNR_THRESHOLD_DB:.1f} dB within {TRAIN_TIMEOUT_SECONDS:.0f}s; "
        f"best_avg_psnr={best_avg_psnr:.3f} dB best_last_psnr={best_last_psnr:.3f} dB "
        f"final_psnr={trainer.state.last_psnr:.3f} dB final_avg_psnr={trainer.state.avg_psnr:.3f} dB "
        f"steps={trainer.state.step} elapsed={elapsed:.2f}s."
    )

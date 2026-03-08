from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import create_default_device
from src.app.shared import apply_training_profile, renderer_kwargs
from src.renderer import GaussianRenderer
from src.scene import GaussianInitHyperParams, build_training_frames, initialize_scene_from_colmap_points, load_colmap_reconstruction, resolve_colmap_init_hparams
from src.training import GaussianTrainer
from src.viewer.app import default_renderer_params, default_training_params

DATASET_ROOT = Path("dataset/bicycle")
IMAGES_SUBDIR = "images_4"
SPARSE_SUBDIR = "sparse/0"
TARGET_PSNR_DB = 23.18
DEFAULT_STEPS = 5_000
DEFAULT_SEED = 1_234


def build_trainer(seed: int) -> GaussianTrainer:
    params, profile = apply_training_profile(default_training_params(), "auto", dataset_root=DATASET_ROOT, images_subdir=IMAGES_SUBDIR)
    recon = load_colmap_reconstruction(DATASET_ROOT, sparse_subdir=SPARSE_SUBDIR)
    frames = build_training_frames(recon, images_subdir=IMAGES_SUBDIR)
    init = GaussianInitHyperParams(initial_opacity=profile.init_opacity_override)
    init_hparams = resolve_colmap_init_hparams(recon, params.training.max_gaussians, init)
    scene = initialize_scene_from_colmap_points(recon=recon, max_gaussians=params.training.max_gaussians, seed=seed, init_hparams=init)
    renderer = GaussianRenderer(create_default_device(enable_debug_layers=False), width=int(frames[0].width), height=int(frames[0].height), **renderer_kwargs(default_renderer_params()))
    return GaussianTrainer(
        device=renderer.device,
        renderer=renderer,
        scene=scene,
        frames=frames,
        adam_hparams=params.adam,
        stability_hparams=params.stability,
        training_hparams=params.training,
        seed=seed,
        scale_reg_reference=float(max(init_hparams.base_scale, 1e-8)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the bicycle/images_4 training profile against the paper PSNR target.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    trainer = build_trainer(seed=int(args.seed))
    start = time.perf_counter()
    for step in range(int(args.steps)):
        trainer.step()
        if step == 0 or (step + 1) % 1000 == 0:
            print(
                f"step={step + 1:5d} avg_psnr={trainer.state.avg_psnr:.3f}dB last_psnr={trainer.state.last_psnr:.3f}dB "
                f"avg_loss={trainer.state.avg_loss:.6f} splats={trainer.scene.count:,}"
            )
    elapsed = time.perf_counter() - start
    gap = TARGET_PSNR_DB - trainer.state.avg_psnr
    print(
        f"final_avg_psnr={trainer.state.avg_psnr:.3f}dB target={TARGET_PSNR_DB:.2f}dB "
        f"gap={gap:.3f}dB elapsed={elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

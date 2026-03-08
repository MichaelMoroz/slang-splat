from __future__ import annotations

import argparse
from dataclasses import replace
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

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
DEFAULT_HOLD_EVERY = 8
DEFAULT_TRAIN_SPLIT = "eval"


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def split_holdout_frames(frames: list, hold_every: int) -> tuple[list, list]:
    hold = max(int(hold_every), 2)
    ordered = sorted(frames, key=lambda frame: frame.image_path.name.lower())
    return [frame for i, frame in enumerate(ordered) if i % hold != 0], [frame for i, frame in enumerate(ordered) if i % hold == 0]


def estimate_border_background(frames: list) -> tuple[float, float, float]:
    samples = []
    for frame in frames:
        with Image.open(frame.image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
        samples.append(srgb_to_linear(border).mean(axis=0))
    mean = np.mean(np.stack(samples, axis=0), axis=0)
    return tuple(float(v) for v in mean.tolist())


def load_frame_target(frame) -> np.ndarray:
    with Image.open(frame.image_path) as image:
        return srgb_to_linear(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)


def frame_psnr(rendered: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((np.asarray(rendered, dtype=np.float32) - np.asarray(target, dtype=np.float32)) ** 2, dtype=np.float64)
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def _override_training_params(params, args):
    updates = {
        name: value
        for name, value in {
            "max_gaussians": args.max_gaussians,
            "densify_from_iter": args.densify_from_iter,
            "densify_until_iter": args.densify_until_iter,
            "densification_interval": args.densification_interval,
            "densify_grad_threshold": args.densify_grad_threshold,
            "percent_dense": args.percent_dense,
            "prune_min_opacity": args.prune_min_opacity,
            "screen_size_prune_threshold": args.screen_size_prune_threshold,
            "world_size_prune_ratio": args.world_size_prune_ratio,
            "opacity_reset_interval": args.opacity_reset_interval,
        }.items()
        if value is not None
    }
    return params if not updates else replace(params, training=replace(params.training, **updates))


def build_trainer(seed: int, hold_every: int, train_split: str, args) -> tuple[GaussianTrainer, list, list, tuple[float, float, float]]:
    params, profile = apply_training_profile(default_training_params(), "auto", dataset_root=DATASET_ROOT, images_subdir=IMAGES_SUBDIR)
    params = _override_training_params(params, args)
    recon = load_colmap_reconstruction(DATASET_ROOT, sparse_subdir=SPARSE_SUBDIR)
    frames = build_training_frames(recon, images_subdir=IMAGES_SUBDIR)
    train_eval_frames, test_eval_frames = split_holdout_frames(frames, hold_every)
    train_frames = frames if train_split == "full" else train_eval_frames
    background = estimate_border_background(train_frames)
    init = GaussianInitHyperParams(initial_opacity=profile.init_opacity_override)
    init_hparams = resolve_colmap_init_hparams(recon, params.training.max_gaussians, init)
    scene = initialize_scene_from_colmap_points(recon=recon, max_gaussians=params.training.max_gaussians, seed=seed, init_hparams=init)
    renderer = GaussianRenderer(create_default_device(enable_debug_layers=False), width=int(frames[0].width), height=int(frames[0].height), **renderer_kwargs(default_renderer_params()))
    trainer = GaussianTrainer(
        device=renderer.device,
        renderer=renderer,
        scene=scene,
        frames=train_frames,
        adam_hparams=params.adam,
        stability_hparams=params.stability,
        training_hparams=replace(params.training, background=background),
        seed=seed,
        scale_reg_reference=float(max(init_hparams.base_scale, 1e-8)),
    )
    return trainer, frames, test_eval_frames, background


def evaluate_psnr(trainer: GaussianTrainer, frames: list, background: tuple[float, float, float]) -> float:
    values = []
    bg = np.asarray(background, dtype=np.float32)
    for frame in frames:
        texture, _ = trainer.renderer.render_to_texture(frame.make_camera(near=float(trainer.training.near), far=float(trainer.training.far)), background=bg, read_stats=False)
        values.append(frame_psnr(np.asarray(texture.to_numpy(), dtype=np.float32)[..., :3], load_frame_target(frame)))
    return float(np.mean(values, dtype=np.float64))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the bicycle/images_4 training profile against the paper PSNR target.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hold-every", type=int, default=DEFAULT_HOLD_EVERY)
    parser.add_argument("--train-split", choices=("eval", "full"), default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--densify-from-iter", type=int, default=None)
    parser.add_argument("--densify-until-iter", type=int, default=None)
    parser.add_argument("--densification-interval", type=int, default=None)
    parser.add_argument("--densify-grad-threshold", type=float, default=None)
    parser.add_argument("--percent-dense", type=float, default=None)
    parser.add_argument("--prune-min-opacity", type=float, default=None)
    parser.add_argument("--screen-size-prune-threshold", type=float, default=None)
    parser.add_argument("--world-size-prune-ratio", type=float, default=None)
    parser.add_argument("--opacity-reset-interval", type=int, default=None)
    args = parser.parse_args()
    trainer, all_frames, test_frames, background = build_trainer(seed=int(args.seed), hold_every=int(args.hold_every), train_split=str(args.train_split), args=args)
    print(f"train_split={args.train_split} train_frames={len(trainer.frames)} eval_test_frames={len(test_frames)} total_frames={len(all_frames)} background={background}")
    start = time.perf_counter()
    for step in range(int(args.steps)):
        trainer.step()
        if step == 0 or (step + 1) % 1000 == 0:
            print(
                f"step={step + 1:5d} avg_psnr={trainer.state.avg_psnr:.3f}dB last_psnr={trainer.state.last_psnr:.3f}dB "
                f"avg_loss={trainer.state.avg_loss:.6f} splats={trainer.scene.count:,}"
            )
    elapsed = time.perf_counter() - start
    full_psnr = evaluate_psnr(trainer, all_frames, background)
    eval_psnr = evaluate_psnr(trainer, test_frames, background)
    gap = TARGET_PSNR_DB - eval_psnr
    print(
        f"full_dataset_psnr={full_psnr:.3f}dB eval_holdout_psnr={eval_psnr:.3f}dB target={TARGET_PSNR_DB:.2f}dB "
        f"gap={gap:.3f}dB elapsed={elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

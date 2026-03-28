from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import load_colmap_scene
from .mcmc import MCMCConfig, RGBMCMCTrainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RGB MCMC Gaussians with the local renderer.")
    for name, default, typ in (
        ("--images", "images_8", str),
        ("--iterations", 3000, int),
        ("--eval-interval", 250, int),
        ("--init-points", 50000, int),
        ("--init-scale-spacing-ratio", 0.25, float),
        ("--init-scale-multiplier", 1.0, float),
        ("--init-opacity", 0.5, float),
        ("--opacity-lr", 0.001, float),
        ("--device", "cuda", str),
        ("--llff-hold", 8, int),
        ("--opacity-reg", 0.01, float),
        ("--scale-reg", 0.01, float),
        ("--noise-lr", 5e5, float),
        ("--lambda-dssim", 0.2, float),
        ("--depth-ratio-weight", 0.1, float),
        ("--dither-strength", 1.0, float),
        ("--dither-decay-until-iter", 0, int),
        ("--seed", 0, int),
    ):
        p.add_argument(name, default=default, type=typ)
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-preload-cuda", action="store_true")
    p.add_argument("--no-eval-split", action="store_true")
    p.add_argument("--random-background", action="store_true")
    return p


def main() -> int:
    a = build_parser().parse_args()
    cfg = MCMCConfig(
        iterations=a.iterations,
        eval_interval=a.eval_interval,
        init_points=a.init_points,
        init_scale_spacing_ratio=a.init_scale_spacing_ratio,
        init_scale_multiplier=a.init_scale_multiplier,
        init_opacity=a.init_opacity,
        opacity_lr=a.opacity_lr,
        random_background=a.random_background,
        opacity_reg=a.opacity_reg,
        scale_reg=a.scale_reg,
        noise_lr=a.noise_lr,
        lambda_dssim=a.lambda_dssim,
        depth_ratio_weight=a.depth_ratio_weight,
        dither_strength=a.dither_strength,
        dither_decay_until_iter=a.dither_decay_until_iter,
        seed=a.seed,
    )
    scene = load_colmap_scene(a.scene, image_dir=a.images, eval_split=not a.no_eval_split, llff_hold=a.llff_hold, preload_cuda=not a.no_preload_cuda, device=a.device)
    metrics = RGBMCMCTrainer(cfg, device=a.device).train(scene, a.output)
    summary = {k: getattr(metrics[-1], k) for k in ("iteration", "loss", "l1", "test_psnr", "test_ssim", "point_count")}
    summary = {f"final_{k}": v for k, v in summary.items()}
    Path(a.output).mkdir(parents=True, exist_ok=True)
    (Path(a.output) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

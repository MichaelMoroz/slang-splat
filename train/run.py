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
        ("--base-lr-init", 0.001, float),
        ("--base-lr-final", 0.001, float),
        ("--base-lr-delay-mult", 1.0, float),
        ("--base-lr-max-steps", 30000, int),
        ("--position-lr-mult", 0.1, float),
        ("--feature-lr-mult", 1.0, float),
        ("--opacity-lr-mult", 1.0, float),
        ("--scaling-lr-mult", 20.0, float),
        ("--rotation-lr-mult", 1.0, float),
        ("--device", "cuda", str),
        ("--llff-hold", 8, int),
        ("--opacity-reg", 0.01, float),
        ("--scale-reg", 0.01, float),
        ("--noise-lr", 5e5, float),
        ("--densify-interval", 50, int),
        ("--densify-until-iter", 500, int),
        ("--densify-interval-after", 50, int),
        ("--densify-target-ratio", 0.05, float),
        ("--densify-append-multiplier", 2.0, float),
        ("--densify-clone-opacity", 0.33, float),
        ("--remove-opacity-threshold", 0.001, float),
        ("--max-splats", 2_000_000, int),
        ("--lambda-dssim", 0.2, float),
        ("--depth-ratio-weight", 0.05, float),
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
    p.add_argument("--disable-densify", action="store_true")
    return p


def main() -> int:
    a = build_parser().parse_args()
    cfg = MCMCConfig(
        iterations=a.iterations,
        eval_interval=a.eval_interval,
        base_lr_init=a.base_lr_init,
        base_lr_final=a.base_lr_final,
        base_lr_delay_mult=a.base_lr_delay_mult,
        base_lr_max_steps=a.base_lr_max_steps,
        position_lr_mult=a.position_lr_mult,
        feature_lr_mult=a.feature_lr_mult,
        opacity_lr_mult=a.opacity_lr_mult,
        scaling_lr_mult=a.scaling_lr_mult,
        rotation_lr_mult=a.rotation_lr_mult,
        init_points=a.init_points,
        init_scale_spacing_ratio=a.init_scale_spacing_ratio,
        init_scale_multiplier=a.init_scale_multiplier,
        init_opacity=a.init_opacity,
        random_background=a.random_background,
        opacity_reg=a.opacity_reg,
        scale_reg=a.scale_reg,
        noise_lr=a.noise_lr,
        densify_enabled=not a.disable_densify,
        densify_interval=a.densify_interval,
        densify_until_iter=a.densify_until_iter,
        densify_interval_after=a.densify_interval_after,
        densify_target_ratio=a.densify_target_ratio,
        densify_append_multiplier=a.densify_append_multiplier,
        densify_clone_opacity=a.densify_clone_opacity,
        remove_opacity_threshold=a.remove_opacity_threshold,
        max_splats=a.max_splats,
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

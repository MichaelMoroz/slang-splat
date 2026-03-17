from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import create_default_device
from src.app.shared import RendererParams, apply_training_profile, build_training_params, renderer_kwargs
from src.renderer import GaussianRenderer
from src.scene import build_training_frames, initialize_scene_from_colmap_points, load_colmap_reconstruction, resolve_colmap_init_hparams
from src.training import GaussianTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep cached raster fixed-point scale on a real COLMAP training run.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/room"))
    parser.add_argument("--images-subdir", type=str, default="images_4")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--max-gaussians", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("outputs/cached_raster_fixed_scale_sweep_room_images4_512.json"))
    parser.add_argument("--scales", type=float, nargs="+", default=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0))
    return parser.parse_args()


def _training_setup(dataset_root: Path, images_subdir: str, max_gaussians: int, seed: int):
    recon = load_colmap_reconstruction(dataset_root, sparse_subdir="sparse/0")
    frames = build_training_frames(recon, images_subdir=images_subdir)
    width, height = int(frames[0].width), int(frames[0].height)
    params, _ = apply_training_profile(
        build_training_params(
            background=(0.0, 0.0, 0.0),
            base_lr=1e-3,
            lr_pos_mul=1.0,
            lr_scale_mul=1.0,
            lr_rot_mul=1.0,
            lr_color_mul=1.0,
            lr_opacity_mul=1.0,
            beta1=0.9,
            beta2=0.999,
            grad_clip=10.0,
            grad_norm_clip=10.0,
            max_update=0.05,
            min_scale=1e-3,
            max_scale=3.0,
            max_anisotropy=10.0,
            min_opacity=1e-4,
            max_opacity=0.9999,
            position_abs_max=1e4,
            near=0.1,
            far=120.0,
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.01,
            opacity_reg_weight=0.01,
            max_gaussians=max_gaussians,
            train_downscale_mode=1,
        ),
        "auto",
        dataset_root=dataset_root,
        images_subdir=images_subdir,
    )
    init_hparams = resolve_colmap_init_hparams(recon, params.training.max_gaussians)
    scene = initialize_scene_from_colmap_points(recon=recon, max_gaussians=params.training.max_gaussians, seed=seed, init_hparams=init_hparams)
    return recon, frames, width, height, params, init_hparams, scene


def _train_run(
    device,
    scene,
    frames,
    width: int,
    height: int,
    params,
    init_hparams,
    seed: int,
    steps: int,
    *,
    atomic_mode: str,
    fixed_scale: float,
) -> dict[str, float]:
    renderer = GaussianRenderer(
        device,
        width=width,
        height=height,
        **renderer_kwargs(RendererParams(cached_raster_grad_atomic_mode=atomic_mode, cached_raster_grad_fixed_scale=fixed_scale)),
    )
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=copy.deepcopy(scene),
        frames=frames,
        adam_hparams=params.adam,
        stability_hparams=params.stability,
        training_hparams=replace(params.training, train_downscale_mode=1, train_downscale_factor=1),
        seed=seed,
        scale_reg_reference=float(max(init_hparams.base_scale, 1e-8)),
    )
    start = time.perf_counter()
    for _ in range(steps):
        trainer.step()
    return {
        "elapsed_sec": float(time.perf_counter() - start),
        "final_loss": float(trainer.state.last_loss),
        "final_mse": float(trainer.state.last_mse),
        "final_psnr": float(trainer.state.last_psnr),
        "avg_psnr": float(trainer.state.avg_psnr),
    }


def _load_target_image(frame, width: int, height: int) -> np.ndarray:
    with Image.open(frame.image_path) as pil_image:
        rgb = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    if rgb.shape[0] != height or rgb.shape[1] != width:
        raise RuntimeError(f"Expected frame size {(height, width)}, got {rgb.shape[:2]} for {frame.image_path}")
    return np.concatenate((rgb, np.ones((height, width, 1), dtype=np.float32)), axis=2)


def _initial_occupancy(device, scene, frames, width: int, height: int, fixed_scale: float) -> dict[str, float]:
    frame = frames[0]
    camera = frame.make_camera(near=0.1, far=120.0)
    target_image = _load_target_image(frame, width, height)
    renderer_fixed = GaussianRenderer(
        device,
        width=width,
        height=height,
        cached_raster_grad_atomic_mode="fixed",
        cached_raster_grad_fixed_scale=fixed_scale,
    )
    grads = renderer_fixed.debug_raster_backward_grads_against_target(scene, camera, target_image, background=np.zeros((3,), dtype=np.float32))
    raw = np.asarray(grads["cached_raster_grads_fixed"], dtype=np.int32)
    usage = np.max(np.abs(raw), axis=0) / np.float32(np.iinfo(np.int32).max)
    labels = renderer_fixed.CACHED_RASTER_GRAD_COMPONENT_LABELS
    return {
        "max_component_occupancy": float(np.max(usage)) if usage.size else 0.0,
        "max_component_label": labels[int(np.argmax(usage))] if usage.size else "",
    }


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    recon, frames, width, height, params, init_hparams, scene = _training_setup(dataset_root, args.images_subdir, int(args.max_gaussians), int(args.seed))
    del recon
    device = create_default_device(enable_debug_layers=True)

    results: list[dict[str, object]] = []
    float_reference = _train_run(device, scene, frames, width, height, params, init_hparams, int(args.seed), int(args.steps), atomic_mode="float", fixed_scale=1.0)
    for scale in (float(value) for value in args.scales):
        occupancy = _initial_occupancy(device, scene, frames, width, height, scale)
        fixed_run = _train_run(device, scene, frames, width, height, params, init_hparams, int(args.seed), int(args.steps), atomic_mode="fixed", fixed_scale=scale)
        results.append(
            {
                "fixed_scale": scale,
                "occupancy": occupancy,
                "fixed": fixed_run,
                "psnr_gap_to_float": float_reference["final_psnr"] - fixed_run["final_psnr"],
            }
        )

    best = max(results, key=lambda row: float(row["fixed"]["final_psnr"]))
    payload = {
        "dataset_root": str(dataset_root),
        "images_subdir": args.images_subdir,
        "width": width,
        "height": height,
        "steps": int(args.steps),
        "max_gaussians": int(args.max_gaussians),
        "seed": int(args.seed),
        "float_reference": float_reference,
        "results": results,
        "best_fixed_scale": float(best["fixed_scale"]),
        "best_fixed_final_psnr": float(best["fixed"]["final_psnr"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

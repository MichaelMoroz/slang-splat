from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from src import create_default_device
from src.renderer import GaussianRenderer
from src.scene import (
    GaussianInitHyperParams,
    build_training_frames,
    initialize_scene_from_colmap_points,
    load_colmap_reconstruction,
)
from src.training import AdamHyperParams, GaussianTrainer, StabilityHyperParams, TrainingHyperParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COLMAP-driven Gaussian splat trainer + GUI launcher.")
    parser.add_argument("--mode", type=str, choices=("gui", "cli"), default="gui", help="Run mode.")
    parser.add_argument("--colmap-root", type=Path, default=None, help="COLMAP scene root folder.")
    parser.add_argument("--sparse-subdir", type=str, default="sparse/0", help="Sparse model relative folder.")
    parser.add_argument("--images-subdir", type=str, default="images_4", help="Training image relative folder.")
    parser.add_argument("--iters", type=int, default=1000, help="Number of optimization iterations.")
    parser.add_argument("--max-gaussians", type=int, default=50000, help="Maximum initialized Gaussians.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--width", type=int, default=0, help="Render width. 0 uses frame width.")
    parser.add_argument("--height", type=int, default=0, help="Render height. 0 uses frame height.")
    parser.add_argument("--prepass-memory-mb", type=int, default=4096, help="Renderer prepass memory cap.")
    parser.add_argument("--radius-scale", type=float, default=2.6)
    parser.add_argument("--max-splat-radius-px", type=float, default=512.0)
    parser.add_argument("--alpha-cutoff", type=float, default=1.0 / 255.0)
    parser.add_argument("--max-splat-steps", type=int, default=32768)
    parser.add_argument("--trans-threshold", type=float, default=0.005)
    parser.add_argument("--sampled5-safety", type=float, default=1.0)
    parser.add_argument("--lr-pos", type=float, default=1e-3)
    parser.add_argument("--lr-scale", type=float, default=2.5e-4)
    parser.add_argument("--lr-rot", type=float, default=1e-3)
    parser.add_argument("--lr-color", type=float, default=1e-3)
    parser.add_argument("--lr-opacity", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--grad-norm-clip", type=float, default=10.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--min-scale", type=float, default=1e-3)
    parser.add_argument("--max-scale", type=float, default=3.0)
    parser.add_argument("--min-opacity", type=float, default=1e-4)
    parser.add_argument("--max-opacity", type=float, default=0.9999)
    parser.add_argument("--position-abs-max", type=float, default=1e4)
    parser.add_argument("--loss-grad-clip", type=float, default=10.0)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=120.0)
    parser.add_argument("--bg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--target-flip-y", action="store_true", help="Enable target Y flip before loss.")
    parser.add_argument("--init-position-jitter", type=float, default=0.01)
    parser.add_argument("--init-base-scale", type=float, default=0.03)
    parser.add_argument("--init-scale-jitter", type=float, default=0.2)
    parser.add_argument("--init-opacity", type=float, default=0.5)
    parser.add_argument("--init-color-jitter", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--snapshot-interval", type=int, default=0, help="Write PNG snapshot every N steps.")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("outputs/train_snapshots"))
    parser.add_argument("--frames", type=int, default=0, help="GUI mode: run fixed frame count and exit.")
    parser.add_argument("--debug-layers", action="store_true")
    return parser.parse_args()


def _save_snapshot(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(rgba[:, :, :3], 0.0, 1.0)
    rgb = np.flipud(rgb)
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _run_cli(args: argparse.Namespace) -> int:
    if args.colmap_root is None:
        raise SystemExit("--colmap-root is required in --mode cli.")
    colmap_root = Path(args.colmap_root).resolve()
    recon = load_colmap_reconstruction(colmap_root, sparse_subdir=args.sparse_subdir)
    frames = build_training_frames(recon, images_subdir=args.images_subdir)
    width = int(args.width) if int(args.width) > 0 else int(frames[0].width)
    height = int(args.height) if int(args.height) > 0 else int(frames[0].height)

    init_hparams = GaussianInitHyperParams(
        position_jitter_std=float(args.init_position_jitter),
        base_scale=float(args.init_base_scale),
        scale_jitter_ratio=float(args.init_scale_jitter),
        initial_opacity=float(args.init_opacity),
        color_jitter_std=float(args.init_color_jitter),
    )
    scene = initialize_scene_from_colmap_points(
        recon=recon,
        max_gaussians=int(args.max_gaussians),
        seed=int(args.seed),
        init_hparams=init_hparams,
    )
    device = create_default_device(enable_debug_layers=args.debug_layers)
    renderer = GaussianRenderer(
        device=device,
        width=width,
        height=height,
        radius_scale=float(args.radius_scale),
        max_splat_radius_px=float(args.max_splat_radius_px),
        alpha_cutoff=float(args.alpha_cutoff),
        max_splat_steps=int(args.max_splat_steps),
        transmittance_threshold=float(args.trans_threshold),
        sampled5_safety_scale=float(args.sampled5_safety),
        max_prepass_memory_mb=int(args.prepass_memory_mb),
    )
    adam = AdamHyperParams(
        position_lr=float(args.lr_pos),
        scale_lr=float(args.lr_scale),
        rotation_lr=float(args.lr_rot),
        color_lr=float(args.lr_color),
        opacity_lr=float(args.lr_opacity),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        epsilon=float(args.eps),
    )
    stability = StabilityHyperParams(
        grad_component_clip=float(args.grad_clip),
        grad_norm_clip=float(args.grad_norm_clip),
        max_update=float(args.max_update),
        min_scale=float(args.min_scale),
        max_scale=float(args.max_scale),
        min_opacity=float(args.min_opacity),
        max_opacity=float(args.max_opacity),
        position_abs_max=float(args.position_abs_max),
        loss_grad_clip=float(args.loss_grad_clip),
    )
    training = TrainingHyperParams(
        background=tuple(float(v) for v in args.bg),
        near=float(args.near),
        far=float(args.far),
        target_flip_y=bool(args.target_flip_y),
        ema_decay=float(args.ema_decay),
    )
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=frames,
        adam_hparams=adam,
        stability_hparams=stability,
        training_hparams=training,
        seed=int(args.seed),
    )

    print(
        f"Training start: scene={colmap_root} images={args.images_subdir} "
        f"frames={len(frames)} gaussians={scene.count} size={width}x{height}"
    )
    t_start = time.perf_counter()
    for step in range(int(args.iters)):
        loss = trainer.step()
        if (step + 1) % max(int(args.log_interval), 1) == 0 or step == 0:
            elapsed = max(time.perf_counter() - t_start, 1e-6)
            ips = float(step + 1) / elapsed
            print(
                f"step={step + 1:6d} loss={loss:.6e} ema={trainer.state.ema_loss:.6e} "
                f"iter/s={ips:.2f} instability='{trainer.state.last_instability}'"
            )
        if int(args.snapshot_interval) > 0 and (step + 1) % int(args.snapshot_interval) == 0:
            frame = frames[max(trainer.state.last_frame_index, 0)]
            camera = frame.make_camera(near=float(args.near), far=float(args.far))
            tex, _ = renderer.render_to_texture(camera, background=np.asarray(args.bg, dtype=np.float32))
            _save_snapshot(args.snapshot_dir / f"step_{step + 1:06d}.png", np.asarray(tex.to_numpy(), dtype=np.float32))
    return 0


def _run_gui(args: argparse.Namespace) -> int:
    from viewer import SplatViewer

    width = int(args.width) if int(args.width) > 0 else 1280
    height = int(args.height) if int(args.height) > 0 else 720
    device = create_default_device(enable_debug_layers=args.debug_layers)
    app = spy.App(device=device)
    viewer = SplatViewer(
        app,
        width=width,
        height=height,
        max_prepass_memory_mb=int(args.prepass_memory_mb),
    )
    if args.colmap_root is not None:
        viewer.load_colmap_dataset(Path(args.colmap_root), args.images_subdir)
    if int(args.frames) > 0:
        for _ in range(int(args.frames)):
            app.run_frame()
        app.terminate()
    else:
        app.run()
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "gui":
        return _run_gui(args)
    return _run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())

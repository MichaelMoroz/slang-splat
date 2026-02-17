from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src import create_default_device
from src.renderer import Camera, GaussianRenderer
from src.scene import (
    GaussianInitHyperParams,
    GaussianScene,
    build_training_frames,
    initialize_scene_from_colmap_points,
    load_colmap_reconstruction,
    load_gaussian_ply,
)
from src.training import AdamHyperParams, GaussianTrainer, StabilityHyperParams, TrainingHyperParams


def _save_snapshot(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(rgba[:, :, :3], 0.0, 1.0)
    rgb = np.flipud(rgb)
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _estimate_scene_bounds(scene: GaussianScene) -> tuple[np.ndarray, float]:
    pos = np.asarray(scene.positions, dtype=np.float32)
    scales = np.asarray(scene.scales, dtype=np.float32)
    opac = np.asarray(scene.opacities, dtype=np.float32).reshape(-1)
    finite = np.isfinite(pos).all(axis=1)
    if not np.any(finite):
        return np.zeros((3,), dtype=np.float32), 1.0

    pos_f = pos[finite]
    scales_f = scales[finite]
    opac_f = np.clip(opac[finite], 1e-3, 1.0)
    core_mask = opac_f > np.quantile(opac_f, 0.7)
    if np.count_nonzero(core_mask) > 2048:
        pos_c = pos_f[core_mask]
        scales_c = scales_f[core_mask]
        opac_c = opac_f[core_mask]
    else:
        pos_c = pos_f
        scales_c = scales_f
        opac_c = opac_f

    weight_sum = float(np.sum(opac_c))
    if weight_sum > 1e-6:
        center = (np.sum(pos_c * opac_c[:, None], axis=0) / weight_sum).astype(np.float32)
    else:
        center = np.mean(pos_c, axis=0).astype(np.float32)

    rel = pos_c - center[None, :]
    dist = np.linalg.norm(rel, axis=1)
    splat_extent = np.max(scales_c, axis=1)
    effective = dist + 2.0 * splat_extent
    core_radius = max(float(np.percentile(effective, 90.0)), 1.0)
    q_lo = np.percentile(pos_c, 5.0, axis=0)
    q_hi = np.percentile(pos_c, 95.0, axis=0)
    quant_extent = 0.5 * np.linalg.norm((q_hi - q_lo).astype(np.float32))
    return center, max(core_radius, float(quant_extent), 1.0)


def _run_train_colmap(args: argparse.Namespace) -> int:
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
    device = create_default_device(enable_debug_layers=bool(args.debug_layers))
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


def _run_render_ply(args: argparse.Namespace) -> int:
    ply_path = Path(args.ply).resolve()
    scene = load_gaussian_ply(ply_path)
    device = create_default_device(enable_debug_layers=bool(args.debug_layers))
    renderer = GaussianRenderer(
        device=device,
        width=int(args.width),
        height=int(args.height),
        radius_scale=float(args.radius_scale),
        max_splat_radius_px=float(args.max_splat_radius_px),
        alpha_cutoff=float(args.alpha_cutoff),
        max_splat_steps=int(args.max_splat_steps),
        transmittance_threshold=float(args.trans_threshold),
        sampled5_safety_scale=float(args.sampled5_safety),
        max_prepass_memory_mb=int(args.prepass_memory_mb),
    )
    renderer.set_scene(scene)

    center, radius = _estimate_scene_bounds(scene)
    distance = max(float(args.distance_multiplier) * float(radius), 1.0)
    cam_y = float(center[1] + float(args.elevation_ratio) * float(radius))
    near = max(float(args.near), 1e-4)
    far = max(float(args.far), distance + 4.0 * float(radius))
    bg = np.asarray(args.bg, dtype=np.float32).reshape(3)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = max(int(args.views), 1)
    print(
        f"Render start: ply={ply_path} views={views} size={int(args.width)}x{int(args.height)} "
        f"output={out_dir}"
    )
    for idx in range(views):
        theta = (2.0 * np.pi * float(idx)) / float(views)
        cam_pos = np.array(
            [
                float(center[0] + np.sin(theta) * distance),
                cam_y,
                float(center[2] - np.cos(theta) * distance),
            ],
            dtype=np.float32,
        )
        camera = Camera.look_at(
            position=cam_pos,
            target=center,
            up=(0.0, 1.0, 0.0),
            fov_y_degrees=float(args.fov_y),
            near=near,
            far=far,
        )
        tex, _ = renderer.render_to_texture(camera, background=bg, read_stats=False)
        _save_snapshot(out_dir / f"view_{idx:04d}.png", np.asarray(tex.to_numpy(), dtype=np.float32))
        if (idx + 1) % max(int(args.log_interval), 1) == 0 or idx == 0 or idx + 1 == views:
            print(f"rendered {idx + 1}/{views}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for PLY rendering and COLMAP training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-colmap", help="Train gaussians on a COLMAP reconstruction.")
    train.add_argument("--colmap-root", type=Path, required=True, help="COLMAP scene root folder.")
    train.add_argument("--sparse-subdir", type=str, default="sparse/0", help="Sparse model relative folder.")
    train.add_argument("--images-subdir", type=str, default="images_4", help="Training image relative folder.")
    train.add_argument("--iters", type=int, default=1000, help="Number of optimization iterations.")
    train.add_argument("--max-gaussians", type=int, default=50000, help="Maximum initialized Gaussians.")
    train.add_argument("--seed", type=int, default=1234, help="Random seed.")
    train.add_argument("--width", type=int, default=0, help="Render width. 0 uses frame width.")
    train.add_argument("--height", type=int, default=0, help="Render height. 0 uses frame height.")
    train.add_argument("--prepass-memory-mb", type=int, default=4096, help="Renderer prepass memory cap.")
    train.add_argument("--radius-scale", type=float, default=2.6)
    train.add_argument("--max-splat-radius-px", type=float, default=512.0)
    train.add_argument("--alpha-cutoff", type=float, default=1.0 / 255.0)
    train.add_argument("--max-splat-steps", type=int, default=32768)
    train.add_argument("--trans-threshold", type=float, default=0.005)
    train.add_argument("--sampled5-safety", type=float, default=1.0)
    train.add_argument("--lr-pos", type=float, default=1e-3)
    train.add_argument("--lr-scale", type=float, default=2.5e-4)
    train.add_argument("--lr-rot", type=float, default=1e-3)
    train.add_argument("--lr-color", type=float, default=1e-3)
    train.add_argument("--lr-opacity", type=float, default=1e-3)
    train.add_argument("--beta1", type=float, default=0.9)
    train.add_argument("--beta2", type=float, default=0.999)
    train.add_argument("--eps", type=float, default=1e-8)
    train.add_argument("--grad-clip", type=float, default=10.0)
    train.add_argument("--grad-norm-clip", type=float, default=10.0)
    train.add_argument("--max-update", type=float, default=0.05)
    train.add_argument("--min-scale", type=float, default=1e-3)
    train.add_argument("--max-scale", type=float, default=3.0)
    train.add_argument("--min-opacity", type=float, default=1e-4)
    train.add_argument("--max-opacity", type=float, default=0.9999)
    train.add_argument("--position-abs-max", type=float, default=1e4)
    train.add_argument("--loss-grad-clip", type=float, default=10.0)
    train.add_argument("--near", type=float, default=0.1)
    train.add_argument("--far", type=float, default=120.0)
    train.add_argument("--bg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    train.add_argument("--ema-decay", type=float, default=0.95)
    train.add_argument("--target-flip-y", action="store_true", help="Enable target Y flip before loss.")
    train.add_argument("--init-position-jitter", type=float, default=0.01)
    train.add_argument("--init-base-scale", type=float, default=0.03)
    train.add_argument("--init-scale-jitter", type=float, default=0.2)
    train.add_argument("--init-opacity", type=float, default=0.5)
    train.add_argument("--init-color-jitter", type=float, default=0.0)
    train.add_argument("--log-interval", type=int, default=10)
    train.add_argument("--snapshot-interval", type=int, default=0, help="Write PNG snapshot every N steps.")
    train.add_argument("--snapshot-dir", type=Path, default=Path("outputs/train_snapshots"))
    train.add_argument("--debug-layers", action="store_true")

    render = subparsers.add_parser("render-ply", help="Render a set of views for a PLY scene.")
    render.add_argument("--ply", type=Path, required=True, help="Input gaussian PLY.")
    render.add_argument("--output-dir", type=Path, default=Path("outputs/ply_views"))
    render.add_argument("--views", type=int, default=24)
    render.add_argument("--log-interval", type=int, default=5)
    render.add_argument("--width", type=int, default=1280)
    render.add_argument("--height", type=int, default=720)
    render.add_argument("--fov-y", type=float, default=60.0)
    render.add_argument("--near", type=float, default=0.1)
    render.add_argument("--far", type=float, default=120.0)
    render.add_argument("--distance-multiplier", type=float, default=1.35)
    render.add_argument("--elevation-ratio", type=float, default=0.0)
    render.add_argument("--bg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    render.add_argument("--prepass-memory-mb", type=int, default=4096)
    render.add_argument("--radius-scale", type=float, default=2.6)
    render.add_argument("--max-splat-radius-px", type=float, default=512.0)
    render.add_argument("--alpha-cutoff", type=float, default=1.0 / 255.0)
    render.add_argument("--max-splat-steps", type=int, default=32768)
    render.add_argument("--trans-threshold", type=float, default=0.005)
    render.add_argument("--sampled5-safety", type=float, default=1.0)
    render.add_argument("--debug-layers", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "train-colmap":
        return _run_train_colmap(args)
    if args.command == "render-ply":
        return _run_render_ply(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

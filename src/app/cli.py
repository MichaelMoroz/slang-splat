from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import create_default_device
from ..renderer import Camera, GaussianRenderer
from ..scene import (
    GaussianInitHyperParams,
    build_training_frames,
    initialize_scene_from_colmap_points,
    load_colmap_reconstruction,
    load_gaussian_ply,
    resolve_colmap_init_hparams,
)
from ..training import GaussianTrainer
from .shared import (
    RendererParams,
    build_training_params,
    estimate_scene_bounds,
    renderer_kwargs,
    save_snapshot,
)


@dataclass(frozen=True, slots=True)
class ArgSpec:
    flags: tuple[str, ...]
    kwargs: dict[str, object]


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    help: str
    args: tuple[ArgSpec, ...]
    handler_name: str


def _format_metric(value: float, fmt: str) -> str:
    return format(value, fmt) if np.isfinite(value) else "n/a"


def _renderer(args: argparse.Namespace, width: int, height: int) -> GaussianRenderer:
    params = RendererParams(
        radius_scale=float(args.radius_scale),
        alpha_cutoff=float(args.alpha_cutoff),
        max_splat_steps=int(args.max_splat_steps),
        transmittance_threshold=float(args.trans_threshold),
        sampled5_safety_scale=float(args.sampled5_safety),
        max_prepass_memory_mb=int(args.prepass_memory_mb),
        list_capacity_multiplier=int(getattr(args, "list_capacity_multiplier", 64)),
    )
    return GaussianRenderer(
        device=create_default_device(enable_debug_layers=bool(args.debug_layers)),
        width=int(width),
        height=int(height),
        **renderer_kwargs(params),
    )


def _init_hparams(args: argparse.Namespace) -> GaussianInitHyperParams:
    return GaussianInitHyperParams(
        position_jitter_std=None if args.init_position_jitter is None else float(args.init_position_jitter),
        base_scale=None if args.init_base_scale is None else float(args.init_base_scale),
        scale_jitter_ratio=None if args.init_scale_jitter is None else float(args.init_scale_jitter),
        initial_opacity=None if args.init_opacity is None else float(args.init_opacity),
        color_jitter_std=None if args.init_color_jitter is None else float(args.init_color_jitter),
    )


def run_train_colmap(args: argparse.Namespace) -> int:
    root = Path(args.colmap_root).resolve()
    recon = load_colmap_reconstruction(root, sparse_subdir=args.sparse_subdir)
    frames = build_training_frames(recon, images_subdir=args.images_subdir)
    width = int(args.width) if int(args.width) > 0 else int(frames[0].width)
    height = int(args.height) if int(args.height) > 0 else int(frames[0].height)
    init_hparams = _init_hparams(args)
    resolved_init = resolve_colmap_init_hparams(recon, int(args.max_gaussians), init_hparams)
    scene = initialize_scene_from_colmap_points(
        recon=recon,
        max_gaussians=int(args.max_gaussians),
        seed=int(args.seed),
        init_hparams=init_hparams,
    )
    params = build_training_params(
        background=args.bg,
        base_lr=args.lr_base,
        lr_pos_mul=args.lr_mul_pos,
        lr_scale_mul=args.lr_mul_scale,
        lr_rot_mul=args.lr_mul_rot,
        lr_color_mul=args.lr_mul_color,
        lr_opacity_mul=args.lr_mul_opacity,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.eps,
        grad_clip=args.grad_clip,
        grad_norm_clip=args.grad_norm_clip,
        max_update=args.max_update,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        max_anisotropy=args.max_anisotropy,
        min_opacity=args.min_opacity,
        max_opacity=args.max_opacity,
        position_abs_max=args.position_abs_max,
        near=args.near,
        far=args.far,
        scale_l2_weight=args.scale_l2,
        mcmc_position_noise_enabled=True,
        mcmc_position_noise_scale=1.0,
        mcmc_opacity_gate_sharpness=100.0,
        mcmc_opacity_gate_center=0.995,
        low_quality_reinit_enabled=bool(args.low_quality_reinit),
        ema_decay=args.ema_decay,
    )
    renderer = _renderer(args, width, height)
    trainer = GaussianTrainer(
        device=renderer.device,
        renderer=renderer,
        scene=scene,
        frames=frames,
        adam_hparams=params.adam,
        stability_hparams=params.stability,
        training_hparams=params.training,
        seed=int(args.seed),
        scale_reg_reference=float(max(resolved_init.base_scale, 1e-8)),
    )
    print(
        f"Training start: scene={root} images={args.images_subdir} "
        f"frames={len(frames)} gaussians={scene.count} size={width}x{height}"
    )
    start = time.perf_counter()
    background = np.asarray(args.bg, dtype=np.float32)
    for step in range(int(args.iters)):
        loss = trainer.step()
        if step == 0 or (step + 1) % max(int(args.log_interval), 1) == 0:
            elapsed = max(time.perf_counter() - start, 1e-6)
            print(
                f"step={step + 1:6d} loss={loss:.6e} ema={trainer.state.ema_loss:.6e} "
                f"psnr={_format_metric(trainer.state.ema_psnr, '.2f')}dB "
                f"iter/s={(step + 1) / elapsed:.2f} instability='{trainer.state.last_instability}'"
            )
        if int(args.snapshot_interval) > 0 and (step + 1) % int(args.snapshot_interval) == 0:
            frame = frames[max(trainer.state.last_frame_index, 0)]
            tex, _ = renderer.render_to_texture(
                frame.make_camera(near=float(args.near), far=float(args.far)),
                background=background,
            )
            save_snapshot(args.snapshot_dir / f"step_{step + 1:06d}.png", tex.to_numpy())
    return 0


def run_render_ply(args: argparse.Namespace) -> int:
    scene = load_gaussian_ply(Path(args.ply).resolve())
    renderer = _renderer(args, int(args.width), int(args.height))
    renderer.set_scene(scene)
    bounds = estimate_scene_bounds(scene)
    distance = max(float(args.distance_multiplier) * float(bounds.radius), 1.0)
    cam_y = float(bounds.center[1] + float(args.elevation_ratio) * float(bounds.radius))
    near = max(float(args.near), 1e-4)
    far = max(float(args.far), distance + 4.0 * float(bounds.radius))
    background = np.asarray(args.bg, dtype=np.float32).reshape(3)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    views = max(int(args.views), 1)
    print(f"Render start: ply={args.ply} views={views} size={int(args.width)}x{int(args.height)} output={out_dir}")
    for idx in range(views):
        theta = (2.0 * np.pi * float(idx)) / float(views)
        tex, _ = renderer.render_to_texture(
            Camera.look_at(
                position=np.array(
                    [
                        float(bounds.center[0] + np.sin(theta) * distance),
                        cam_y,
                        float(bounds.center[2] - np.cos(theta) * distance),
                    ],
                    dtype=np.float32,
                ),
                target=bounds.center,
                up=(0.0, 1.0, 0.0),
                fov_y_degrees=float(args.fov_y),
                near=near,
                far=far,
            ),
            background=background,
            read_stats=False,
        )
        save_snapshot(out_dir / f"view_{idx:04d}.png", tex.to_numpy())
        if idx == 0 or idx + 1 == views or (idx + 1) % max(int(args.log_interval), 1) == 0:
            print(f"rendered {idx + 1}/{views}")
    return 0


def run_render_single(args: argparse.Namespace) -> int:
    scene = load_gaussian_ply(Path(args.ply).resolve())
    if int(args.max_splats) > 0:
        scene = scene.subset(int(args.max_splats))
    output = _renderer(args, int(args.width), int(args.height)).render(
        scene,
        Camera.look_at(
            position=np.asarray(args.cam_pos, dtype=np.float32),
            target=np.asarray(args.cam_target, dtype=np.float32),
            fov_y_degrees=float(args.fov),
            near=float(args.near),
            far=float(args.far),
        ),
        background=np.asarray(args.bg, dtype=np.float32),
    )
    print(output.stats)
    save_snapshot(Path(args.output), output.image, flip_y=not bool(args.no_flip_y))
    print(f"Saved {args.output}")
    return 0


COMMON_RENDER_ARGS = (
    ArgSpec(("--prepass-memory-mb",), {"type": int, "default": 4096}),
    ArgSpec(("--radius-scale",), {"type": float, "default": 2.6}),
    ArgSpec(("--alpha-cutoff",), {"type": float, "default": 1.0 / 255.0}),
    ArgSpec(("--max-splat-steps",), {"type": int, "default": 32768}),
    ArgSpec(("--trans-threshold",), {"type": float, "default": 0.005}),
    ArgSpec(("--sampled5-safety",), {"type": float, "default": 1.0}),
    ArgSpec(("--debug-layers",), {"action": "store_true"}),
)

COMMANDS = (
    CommandSpec(
        "train-colmap",
        "Train gaussians on a COLMAP reconstruction.",
        (
            ArgSpec(("--colmap-root",), {"type": Path, "required": True}),
            ArgSpec(("--sparse-subdir",), {"type": str, "default": "sparse/0"}),
            ArgSpec(("--images-subdir",), {"type": str, "default": "images_4"}),
            ArgSpec(("--iters",), {"type": int, "default": 1000}),
            ArgSpec(("--max-gaussians",), {"type": int, "default": 50000}),
            ArgSpec(("--seed",), {"type": int, "default": 1234}),
            ArgSpec(("--width",), {"type": int, "default": 0}),
            ArgSpec(("--height",), {"type": int, "default": 0}),
            *COMMON_RENDER_ARGS,
            ArgSpec(("--lr-base",), {"type": float, "default": 1e-3}),
            ArgSpec(("--lr-mul-pos",), {"type": float, "default": 1.0}),
            ArgSpec(("--lr-mul-scale",), {"type": float, "default": 1.0}),
            ArgSpec(("--lr-mul-rot",), {"type": float, "default": 1.0}),
            ArgSpec(("--lr-mul-color",), {"type": float, "default": 1.0}),
            ArgSpec(("--lr-mul-opacity",), {"type": float, "default": 1.0}),
            ArgSpec(("--beta1",), {"type": float, "default": 0.9}),
            ArgSpec(("--beta2",), {"type": float, "default": 0.999}),
            ArgSpec(("--eps",), {"type": float, "default": 1e-8}),
            ArgSpec(("--grad-clip",), {"type": float, "default": 10.0}),
            ArgSpec(("--grad-norm-clip",), {"type": float, "default": 10.0}),
            ArgSpec(("--max-update",), {"type": float, "default": 0.05}),
            ArgSpec(("--min-scale",), {"type": float, "default": 1e-3}),
            ArgSpec(("--max-scale",), {"type": float, "default": 3.0}),
            ArgSpec(("--min-opacity",), {"type": float, "default": 1e-4}),
            ArgSpec(("--max-opacity",), {"type": float, "default": 0.9999}),
            ArgSpec(("--position-abs-max",), {"type": float, "default": 1e4}),
            ArgSpec(("--loss-grad-clip",), {"type": float, "default": 10.0}),
            ArgSpec(("--near",), {"type": float, "default": 0.1}),
            ArgSpec(("--far",), {"type": float, "default": 120.0}),
            ArgSpec(("--bg",), {"type": float, "nargs": 3, "default": (0.0, 0.0, 0.0)}),
            ArgSpec(("--ema-decay",), {"type": float, "default": 0.95}),
            ArgSpec(("--scale-l2",), {"type": float, "default": 1e-3}),
            ArgSpec(("--max-anisotropy",), {"type": float, "default": 3.0}),
            ArgSpec(("--low-quality-reinit",), {"action": argparse.BooleanOptionalAction, "default": True}),
            ArgSpec(("--init-position-jitter",), {"type": float, "default": None}),
            ArgSpec(("--init-base-scale",), {"type": float, "default": None}),
            ArgSpec(("--init-scale-jitter",), {"type": float, "default": None}),
            ArgSpec(("--init-opacity",), {"type": float, "default": None}),
            ArgSpec(("--init-color-jitter",), {"type": float, "default": None}),
            ArgSpec(("--log-interval",), {"type": int, "default": 10}),
            ArgSpec(("--snapshot-interval",), {"type": int, "default": 0}),
            ArgSpec(("--snapshot-dir",), {"type": Path, "default": Path("outputs/train_snapshots")}),
        ),
        "run_train_colmap",
    ),
    CommandSpec(
        "render-ply",
        "Render a set of views for a PLY scene.",
        (
            ArgSpec(("--ply",), {"type": Path, "required": True}),
            ArgSpec(("--output-dir",), {"type": Path, "default": Path("outputs/ply_views")}),
            ArgSpec(("--views",), {"type": int, "default": 24}),
            ArgSpec(("--log-interval",), {"type": int, "default": 5}),
            ArgSpec(("--width",), {"type": int, "default": 1280}),
            ArgSpec(("--height",), {"type": int, "default": 720}),
            ArgSpec(("--fov-y",), {"type": float, "default": 60.0}),
            ArgSpec(("--near",), {"type": float, "default": 0.1}),
            ArgSpec(("--far",), {"type": float, "default": 120.0}),
            ArgSpec(("--distance-multiplier",), {"type": float, "default": 1.35}),
            ArgSpec(("--elevation-ratio",), {"type": float, "default": 0.0}),
            ArgSpec(("--bg",), {"type": float, "nargs": 3, "default": (0.0, 0.0, 0.0)}),
            *COMMON_RENDER_ARGS,
        ),
        "run_render_ply",
    ),
)

SINGLE_RENDER_ARGS = (
    ArgSpec(("--ply",), {"type": Path, "required": True}),
    ArgSpec(("--output",), {"type": Path, "default": Path("render.png")}),
    ArgSpec(("--width",), {"type": int, "default": 1280}),
    ArgSpec(("--height",), {"type": int, "default": 720}),
    ArgSpec(("--max-splats",), {"type": int, "default": 0}),
    ArgSpec(("--max-splat-radius-px",), {"type": float, "default": 64.0}),
    ArgSpec(("--cam-pos",), {"type": float, "nargs": 3, "default": (0.0, 0.0, 3.0)}),
    ArgSpec(("--cam-target",), {"type": float, "nargs": 3, "default": (0.0, 0.0, 0.0)}),
    ArgSpec(("--fov",), {"type": float, "default": 60.0}),
    ArgSpec(("--near",), {"type": float, "default": 0.1}),
    ArgSpec(("--far",), {"type": float, "default": 120.0}),
    ArgSpec(("--bg",), {"type": float, "nargs": 3, "default": (0.0, 0.0, 0.0)}),
    ArgSpec(("--no-flip-y",), {"action": "store_true"}),
    *COMMON_RENDER_ARGS,
)


def _add_arguments(parser: argparse.ArgumentParser, specs: tuple[ArgSpec, ...]) -> None:
    for spec in specs:
        parser.add_argument(*spec.flags, **spec.kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for PLY rendering and COLMAP training.")
    subs = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        sub = subs.add_parser(command.name, help=command.help)
        _add_arguments(sub, command.args)
        sub.set_defaults(handler=globals()[command.handler_name])
    return parser


def build_single_render_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Basic Slangpy Gaussian splat renderer.")
    _add_arguments(parser, SINGLE_RENDER_ARGS)
    parser.set_defaults(handler=run_render_single)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def parse_single_render_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_single_render_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.handler(args))


def render_main(argv: list[str] | None = None) -> int:
    args = parse_single_render_args(argv)
    return int(args.handler(args))

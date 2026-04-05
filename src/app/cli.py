from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .. import create_default_device
from ..renderer import Camera, GaussianRenderSettings, GaussianRenderer
from ..scene import GaussianInitHyperParams, build_training_frames, initialize_scene_from_colmap_points, load_colmap_reconstruction, load_gaussian_ply, resolve_colmap_init_hparams
from ..training import GaussianTrainer
from .shared import RendererParams, apply_training_profile, build_training_params, estimate_scene_bounds, renderer_kwargs, save_snapshot
from ..training import TRAINING_PROFILE_CHOICES


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


def A(*flags: str, **kwargs: object) -> ArgSpec:
    return ArgSpec(flags=tuple(flags), kwargs=dict(kwargs))


def _format_metric(value: float, fmt: str) -> str:
    return format(value, fmt) if np.isfinite(value) else "n/a"


def _optional_float_arg(args: argparse.Namespace, name: str) -> float | None:
    value = getattr(args, name, None)
    return None if value is None else float(value)


def _renderer(args: argparse.Namespace, width: int, height: int) -> GaussianRenderer:
    params = RendererParams(
        radius_scale=float(args.radius_scale),
        alpha_cutoff=float(args.alpha_cutoff),
        max_anisotropy=float(getattr(args, "max_anisotropy", 32.0)),
        transmittance_threshold=float(args.trans_threshold),
        max_prepass_memory_mb=int(args.prepass_memory_mb),
        list_capacity_multiplier=int(getattr(args, "list_capacity_multiplier", 64)),
    )
    settings = GaussianRenderSettings(width=int(width), height=int(height), **renderer_kwargs(params))
    return settings.create_renderer(create_default_device(enable_debug_layers=False))


def _init_hparams(args: argparse.Namespace) -> GaussianInitHyperParams:
    return GaussianInitHyperParams(
        position_jitter_std=_optional_float_arg(args, "init_position_jitter"),
        base_scale=_optional_float_arg(args, "init_base_scale"),
        scale_jitter_ratio=_optional_float_arg(args, "init_scale_jitter"),
        initial_opacity=_optional_float_arg(args, "init_opacity"),
        color_jitter_std=_optional_float_arg(args, "init_color_jitter"),
    )


def _training_params(args: argparse.Namespace):
    return build_training_params(
        background=args.bg,
        base_lr=args.lr_base,
        lr_pos_mul=args.lr_mul_pos,
        lr_scale_mul=args.lr_mul_scale,
        lr_rot_mul=args.lr_mul_rot,
        lr_color_mul=args.lr_mul_color,
        lr_opacity_mul=args.lr_mul_opacity,
        beta1=args.beta1,
        beta2=args.beta2,
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
        scale_abs_reg_weight=args.scale_abs_reg,
        sh1_reg_weight=args.sh1_reg,
        opacity_reg_weight=args.opacity_reg,
        density_regularizer=args.density_reg,
        depth_ratio_weight=args.depth_ratio_weight,
        depth_ratio_grad_min=args.depth_ratio_grad_min,
        depth_ratio_grad_max=args.depth_ratio_grad_max,
        max_allowed_density_start=args.max_allowed_density_start,
        max_allowed_density=args.max_allowed_density,
        max_gaussians=args.max_gaussians,
        use_sh=args.use_sh,
        refinement_min_contribution_percent=args.refinement_min_contribution_percent,
    )


def run_train_colmap(args: argparse.Namespace) -> int:
    root = Path(args.colmap_root).resolve()
    recon = load_colmap_reconstruction(root, sparse_subdir=args.sparse_subdir)
    frames = build_training_frames(recon, images_subdir=args.images_subdir)
    width, height = (int(args.width), int(args.height))
    width, height = (width if width > 0 else int(frames[0].width), height if height > 0 else int(frames[0].height))
    init_hparams = _init_hparams(args)
    params, profile = apply_training_profile(_training_params(args), args.training_profile, dataset_root=root, images_subdir=args.images_subdir)
    init_hparams = replace(init_hparams, initial_opacity=profile.init_opacity_override) if init_hparams.initial_opacity is None and profile.init_opacity_override is not None else init_hparams
    resolved_init = resolve_colmap_init_hparams(recon, params.training.max_gaussians, init_hparams)
    scene = initialize_scene_from_colmap_points(recon=recon, max_gaussians=params.training.max_gaussians, seed=int(args.seed), init_hparams=resolved_init)
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
    print(f"Training start: scene={root} images={args.images_subdir} profile={profile.name} frames={len(frames)} gaussians={scene.count} size={width}x{height}")
    background, start = np.asarray(args.bg, dtype=np.float32), time.perf_counter()
    for step in range(int(args.iters)):
        loss = trainer.step()
        if step == 0 or (step + 1) % max(int(args.log_interval), 1) == 0:
            elapsed = max(time.perf_counter() - start, 1e-6)
            print(
                f"step={step + 1:6d} loss={loss:.6e} avg={trainer.state.avg_loss:.6e} "
                f"mse={_format_metric(trainer.state.last_mse, '.6e')} psnr={_format_metric(trainer.state.last_psnr, '.3f')}dB iter/s={(step + 1) / elapsed:.2f} "
                f"instability='{trainer.state.last_instability}'"
            )
        if int(args.snapshot_interval) > 0 and (step + 1) % int(args.snapshot_interval) == 0:
            frame = frames[max(trainer.state.last_frame_index, 0)]
            tex, _ = renderer.render_to_texture(frame.make_camera(near=float(args.near), far=float(args.far)), background=background)
            save_snapshot(args.snapshot_dir / f"step_{step + 1:06d}.png", tex.to_numpy())
    return 0


def run_render_ply(args: argparse.Namespace) -> int:
    scene = load_gaussian_ply(Path(args.ply).resolve())
    renderer = _renderer(args, int(args.width), int(args.height))
    renderer.set_scene(scene)
    bounds = estimate_scene_bounds(scene)
    distance = max(float(args.distance_multiplier) * float(bounds.radius), 1.0)
    cam_y = float(bounds.center[1] + float(args.elevation_ratio) * float(bounds.radius))
    near, far = max(float(args.near), 1e-4), max(float(args.far), distance + 4.0 * float(bounds.radius))
    background, out_dir, views = np.asarray(args.bg, dtype=np.float32).reshape(3), Path(args.output_dir), max(int(args.views), 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Render start: ply={args.ply} views={views} size={int(args.width)}x{int(args.height)} output={out_dir}")
    for idx in range(views):
        theta = (2.0 * np.pi * float(idx)) / float(views)
        tex, _ = renderer.render_to_texture(
            Camera.look_at(
                position=np.array([float(bounds.center[0] + np.sin(theta) * distance), cam_y, float(bounds.center[2] - np.cos(theta) * distance)], dtype=np.float32),
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
    A("--prepass-memory-mb", type=int, default=4096),
    A("--radius-scale", type=float, default=1.0),
    A("--alpha-cutoff", type=float, default=1.0 / 255.0),
    A("--trans-threshold", type=float, default=0.005),
    A("--debug-layers", action="store_true"),
)
TRAIN_RENDER_ARGS = tuple(
    A(flag, type=float, default=default)
    for flag, default in (
        ("--lr-base", 1e-3),
        ("--lr-mul-pos", 1.0),
        ("--lr-mul-scale", 5.0),
        ("--lr-mul-rot", 1.0),
        ("--lr-mul-color", 5.0),
        ("--lr-mul-opacity", 5.0),
        ("--beta1", 0.9),
        ("--beta2", 0.999),
        ("--grad-clip", 10.0),
        ("--grad-norm-clip", 10.0),
        ("--max-update", 0.05),
        ("--min-scale", 1e-3),
        ("--max-scale", 3.0),
        ("--min-opacity", 1e-4),
        ("--max-opacity", 0.9999),
        ("--position-abs-max", 1e4),
        ("--loss-grad-clip", 10.0),
        ("--near", 0.1),
        ("--far", 120.0),
        ("--scale-l2", 0.0),
        ("--scale-abs-reg", 0.01),
        ("--sh1-reg", 0.01),
        ("--opacity-reg", 0.01),
        ("--density-reg", 0.05),
        ("--depth-ratio-weight", 0.05),
        ("--depth-ratio-grad-min", 0.01),
        ("--depth-ratio-grad-max", 0.05),
        ("--max-allowed-density-start", 5.0),
        ("--max-allowed-density", 12.0),
        ("--max-anisotropy", 32.0),
    )
)
TRAIN_INIT_ARGS = tuple(
    A(flag, type=float, default=None)
    for flag in ("--init-opacity",)
)
COMMANDS = (
    CommandSpec(
        "train-colmap",
        "Train gaussians on a COLMAP reconstruction.",
        (
            A("--colmap-root", type=Path, required=True),
            A("--sparse-subdir", type=str, default="sparse/0"),
            A("--images-subdir", type=str, default="images_4"),
            A("--iters", type=int, default=1000),
            A("--max-gaussians", type=int, default=1000000),
            A("--refinement-min-contribution-percent", type=float, default=1e-05),
            A("--no-use-sh", action="store_false", dest="use_sh", default=True),
            A("--training-profile", type=str, default="auto", choices=TRAINING_PROFILE_CHOICES),
            A("--seed", type=int, default=1234),
            A("--width", type=int, default=0),
            A("--height", type=int, default=0),
            *COMMON_RENDER_ARGS,
            *TRAIN_RENDER_ARGS,
            A("--bg", type=float, nargs=3, default=(0.0, 0.0, 0.0)),
            *TRAIN_INIT_ARGS,
            A("--log-interval", type=int, default=10),
            A("--snapshot-interval", type=int, default=0),
            A("--snapshot-dir", type=Path, default=Path("outputs/train_snapshots")),
        ),
        "run_train_colmap",
    ),
    CommandSpec(
        "render-ply",
        "Render a set of views for a PLY scene.",
        (
            A("--ply", type=Path, required=True),
            A("--output-dir", type=Path, default=Path("outputs/ply_views")),
            A("--views", type=int, default=24),
            A("--log-interval", type=int, default=5),
            A("--width", type=int, default=1280),
            A("--height", type=int, default=720),
            A("--fov-y", type=float, default=60.0),
            A("--near", type=float, default=0.1),
            A("--far", type=float, default=120.0),
            A("--distance-multiplier", type=float, default=1.35),
            A("--elevation-ratio", type=float, default=0.0),
            A("--bg", type=float, nargs=3, default=(0.0, 0.0, 0.0)),
            *COMMON_RENDER_ARGS,
        ),
        "run_render_ply",
    ),
)
SINGLE_RENDER_ARGS = (
    A("--ply", type=Path, required=True),
    A("--output", type=Path, default=Path("render.png")),
    A("--width", type=int, default=1280),
    A("--height", type=int, default=720),
    A("--max-splats", type=int, default=0),
    A("--max-splat-radius-px", type=float, default=64.0),
    A("--cam-pos", type=float, nargs=3, default=(0.0, 0.0, 3.0)),
    A("--cam-target", type=float, nargs=3, default=(0.0, 0.0, 0.0)),
    A("--fov", type=float, default=60.0),
    A("--near", type=float, default=0.1),
    A("--far", type=float, default=120.0),
    A("--bg", type=float, nargs=3, default=(0.0, 0.0, 0.0)),
    A("--no-flip-y", action="store_true"),
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


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def parse_single_render_args(argv=None):
    return build_single_render_parser().parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return int(args.handler(args))


def render_main(argv=None) -> int:
    args = parse_single_render_args(argv)
    return int(args.handler(args))

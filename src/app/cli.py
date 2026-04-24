from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .. import create_default_device
from ..repo_defaults import cli_defaults
from .training_controls import TRAINING_CLI_ARG_DEFS, training_cli_build_kwargs
from ..renderer import Camera, GaussianRenderSettings, GaussianRenderer
from ..scene import GaussianInitHyperParams, build_training_frames, initialize_scene_from_colmap_points, load_colmap_reconstruction, load_gaussian_ply, resolve_colmap_init_hparams
from ..training import GaussianTrainer
from ..training import resolve_effective_train_render_factor, resolve_training_resolution
from ..training.defaults import DEFAULT_REFINEMENT_MIN_CONTRIBUTION, TRAINING_BUILD_ARG_DEFAULTS
from .shared import RendererParams, apply_training_profile, build_training_params, estimate_scene_bounds, renderer_kwargs, save_snapshot
from ..training import TRAINING_PROFILE_CHOICES
_CLI_DEFAULTS = cli_defaults()
_CLI_COMMON_RENDER_DEFAULTS = _CLI_DEFAULTS["common_render"]
_CLI_TRAIN_COLMAP_DEFAULTS = _CLI_DEFAULTS["train_colmap"]
_CLI_RENDER_PLY_DEFAULTS = _CLI_DEFAULTS["render_ply"]
_CLI_RENDER_SINGLE_DEFAULTS = _CLI_DEFAULTS["render_single"]


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
        list_capacity_multiplier=int(getattr(args, "list_capacity_multiplier", _CLI_COMMON_RENDER_DEFAULTS["list_capacity_multiplier"])),
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
        max_gaussians=args.max_gaussians,
        use_sh=args.use_sh,
        refinement_min_contribution=args.refinement_min_contribution,
        **training_cli_build_kwargs(args),
    )


def run_train_colmap(args: argparse.Namespace) -> int:
    root = Path(args.colmap_root).resolve()
    recon = load_colmap_reconstruction(root, sparse_subdir=args.sparse_subdir)
    frames = build_training_frames(recon, images_subdir=args.images_subdir)
    init_hparams = _init_hparams(args)
    params, profile = apply_training_profile(_training_params(args), args.training_profile, dataset_root=root, images_subdir=args.images_subdir)
    width, height = int(args.width), int(args.height)
    if width <= 0 and height <= 0:
        resolutions = [resolve_training_resolution(int(frame.width), int(frame.height), resolve_effective_train_render_factor(params.training, 0, int(frame.width), int(frame.height))) for frame in frames]
        width, height = max(w for w, _ in resolutions), max(h for _, h in resolutions)
    else:
        width, height = width if width > 0 else int(frames[0].width), height if height > 0 else int(frames[0].height)
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
    A("--prepass-memory-mb", type=int, default=int(_CLI_COMMON_RENDER_DEFAULTS["prepass_memory_mb"])),
    A("--radius-scale", type=float, default=float(_CLI_COMMON_RENDER_DEFAULTS["radius_scale"])),
    A("--alpha-cutoff", type=float, default=float(_CLI_COMMON_RENDER_DEFAULTS["alpha_cutoff"])),
    A("--trans-threshold", type=float, default=float(_CLI_COMMON_RENDER_DEFAULTS["trans_threshold"])),
    A("--debug-layers", action="store_true", default=bool(_CLI_COMMON_RENDER_DEFAULTS["debug_layers"])),
)
TRAIN_RENDER_ARGS = tuple(A(*spec.flags, **spec.kwargs) for spec in TRAINING_CLI_ARG_DEFS)
TRAIN_INIT_ARGS = tuple(
    A(flag, type=float, default=_CLI_TRAIN_COLMAP_DEFAULTS["init_opacity"])
    for flag in ("--init-opacity",)
)
COMMANDS = (
    CommandSpec(
        "train-colmap",
        "Train gaussians on a COLMAP reconstruction.",
        (
            A("--colmap-root", type=Path, required=True),
            A("--sparse-subdir", type=str, default=str(_CLI_TRAIN_COLMAP_DEFAULTS["sparse_subdir"])),
            A("--images-subdir", type=str, default=str(_CLI_TRAIN_COLMAP_DEFAULTS["images_subdir"])),
            A("--iters", type=int, default=int(_CLI_TRAIN_COLMAP_DEFAULTS["iters"])),
            A("--max-gaussians", type=int, default=int(_CLI_TRAIN_COLMAP_DEFAULTS["max_gaussians"])),
            A("--refinement-min-contribution", type=int, default=int(_CLI_TRAIN_COLMAP_DEFAULTS["refinement_min_contribution"])),
            A("--no-use-sh", action="store_false", dest="use_sh", default=bool(_CLI_TRAIN_COLMAP_DEFAULTS["use_sh"])),
            A("--training-profile", type=str, default=str(_CLI_TRAIN_COLMAP_DEFAULTS["training_profile"]), choices=TRAINING_PROFILE_CHOICES),
            A("--seed", type=int, default=int(_CLI_TRAIN_COLMAP_DEFAULTS["seed"])),
            A("--width", type=int, default=int(_CLI_TRAIN_COLMAP_DEFAULTS["width"])),
            A("--height", type=int, default=int(_CLI_TRAIN_COLMAP_DEFAULTS["height"])),
            *COMMON_RENDER_ARGS,
            *TRAIN_RENDER_ARGS,
            A("--bg", type=float, nargs=3, default=tuple(float(v) for v in _CLI_TRAIN_COLMAP_DEFAULTS["bg"])),
            *TRAIN_INIT_ARGS,
            A("--log-interval", type=int, default=int(_CLI_TRAIN_COLMAP_DEFAULTS["log_interval"])),
            A("--snapshot-interval", type=int, default=int(_CLI_TRAIN_COLMAP_DEFAULTS["snapshot_interval"])),
            A("--snapshot-dir", type=Path, default=Path(str(_CLI_TRAIN_COLMAP_DEFAULTS["snapshot_dir"]))),
        ),
        "run_train_colmap",
    ),
    CommandSpec(
        "render-ply",
        "Render a set of views for a PLY scene.",
        (
            A("--ply", type=Path, required=True),
            A("--output-dir", type=Path, default=Path(str(_CLI_RENDER_PLY_DEFAULTS["output_dir"]))),
            A("--views", type=int, default=int(_CLI_RENDER_PLY_DEFAULTS["views"])),
            A("--log-interval", type=int, default=int(_CLI_RENDER_PLY_DEFAULTS["log_interval"])),
            A("--width", type=int, default=int(_CLI_RENDER_PLY_DEFAULTS["width"])),
            A("--height", type=int, default=int(_CLI_RENDER_PLY_DEFAULTS["height"])),
            A("--fov-y", type=float, default=float(_CLI_RENDER_PLY_DEFAULTS["fov_y"])),
            A("--near", type=float, default=float(_CLI_RENDER_PLY_DEFAULTS["near"])),
            A("--far", type=float, default=float(_CLI_RENDER_PLY_DEFAULTS["far"])),
            A("--distance-multiplier", type=float, default=float(_CLI_RENDER_PLY_DEFAULTS["distance_multiplier"])),
            A("--elevation-ratio", type=float, default=float(_CLI_RENDER_PLY_DEFAULTS["elevation_ratio"])),
            A("--bg", type=float, nargs=3, default=tuple(float(v) for v in _CLI_RENDER_PLY_DEFAULTS["bg"])),
            *COMMON_RENDER_ARGS,
        ),
        "run_render_ply",
    ),
)
SINGLE_RENDER_ARGS = (
    A("--ply", type=Path, required=True),
    A("--output", type=Path, default=Path(str(_CLI_RENDER_SINGLE_DEFAULTS["output"]))),
    A("--width", type=int, default=int(_CLI_RENDER_SINGLE_DEFAULTS["width"])),
    A("--height", type=int, default=int(_CLI_RENDER_SINGLE_DEFAULTS["height"])),
    A("--max-splats", type=int, default=int(_CLI_RENDER_SINGLE_DEFAULTS["max_splats"])),
    A("--max-splat-radius-px", type=float, default=float(_CLI_RENDER_SINGLE_DEFAULTS["max_splat_radius_px"])),
    A("--cam-pos", type=float, nargs=3, default=tuple(float(v) for v in _CLI_RENDER_SINGLE_DEFAULTS["cam_pos"])),
    A("--cam-target", type=float, nargs=3, default=tuple(float(v) for v in _CLI_RENDER_SINGLE_DEFAULTS["cam_target"])),
    A("--fov", type=float, default=float(_CLI_RENDER_SINGLE_DEFAULTS["fov"])),
    A("--near", type=float, default=float(_CLI_RENDER_SINGLE_DEFAULTS["near"])),
    A("--far", type=float, default=float(_CLI_RENDER_SINGLE_DEFAULTS["far"])),
    A("--bg", type=float, nargs=3, default=tuple(float(v) for v in _CLI_RENDER_SINGLE_DEFAULTS["bg"])),
    A("--no-flip-y", action="store_true", default=bool(_CLI_RENDER_SINGLE_DEFAULTS["no_flip_y"])),
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

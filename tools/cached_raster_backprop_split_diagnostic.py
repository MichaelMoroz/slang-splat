from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import create_default_device
from src.renderer import GaussianRenderer
from src.scene import build_training_frames, initialize_scene_from_colmap_points, load_colmap_reconstruction, resolve_colmap_init_hparams


@dataclass(frozen=True, slots=True)
class ContributionStats:
    label: str
    float_scale_norm: float
    fixed_scale_norm: float
    scale_norm_ratio: float
    scale_cosine: float
    float_rotation_norm: float
    fixed_rotation_norm: float
    rotation_norm_ratio: float
    rotation_cosine: float
    scale_max_abs_diff: float
    rotation_max_abs_diff: float


def _cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_flat = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs_flat = np.asarray(rhs, dtype=np.float64).reshape(-1)
    lhs_norm = np.linalg.norm(lhs_flat)
    rhs_norm = np.linalg.norm(rhs_flat)
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 1.0 if lhs_norm <= 0.0 and rhs_norm <= 0.0 else 0.0
    return float(np.dot(lhs_flat, rhs_flat) / (lhs_norm * rhs_norm))


def _norm_ratio(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_norm = float(np.linalg.norm(np.asarray(lhs, dtype=np.float64).reshape(-1)))
    rhs_norm = float(np.linalg.norm(np.asarray(rhs, dtype=np.float64).reshape(-1)))
    return lhs_norm / rhs_norm if rhs_norm > 0.0 else (1.0 if lhs_norm <= 0.0 else float("inf"))


def _load_target(frame, width: int, height: int) -> np.ndarray:
    with Image.open(frame.image_path) as pil_image:
        rgb = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    if rgb.shape[:2] != (height, width):
        raise RuntimeError(f"Expected {(height, width)}, got {rgb.shape[:2]} for {frame.image_path}")
    return np.concatenate((rgb, np.ones((height, width, 1), dtype=np.float32)), axis=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare per-cached-field backprop contributions to raw scale and rotation grads.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/room"))
    parser.add_argument("--images-subdir", type=str, default="images_4")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--max-gaussians", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fixed-scale", type=float, default=0.125)
    parser.add_argument("--output", type=Path, default=Path("outputs/cached_raster_backprop_split_room_images4.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    recon = load_colmap_reconstruction(dataset_root, sparse_subdir="sparse/0")
    frames = build_training_frames(recon, images_subdir=args.images_subdir)
    frame = frames[int(args.frame_index) % len(frames)]
    width, height = int(frame.width), int(frame.height)
    scene = initialize_scene_from_colmap_points(
        recon=recon,
        max_gaussians=int(args.max_gaussians),
        seed=int(args.seed),
        init_hparams=resolve_colmap_init_hparams(recon, int(args.max_gaussians)),
    )
    camera = frame.make_camera(near=0.1, far=120.0)
    target = _load_target(frame, width, height)
    device = create_default_device(enable_debug_layers=True)

    renderer_float = GaussianRenderer(device, width=width, height=height, cached_raster_grad_atomic_mode="float")
    renderer_fixed = GaussianRenderer(
        device,
        width=width,
        height=height,
        cached_raster_grad_atomic_mode="fixed",
        cached_raster_grad_fixed_scale=float(args.fixed_scale),
    )
    split_float = renderer_float.debug_cached_raster_backprop_contributions_against_target(scene, camera, target)
    split_fixed = renderer_fixed.debug_cached_raster_backprop_contributions_against_target(scene, camera, target)

    rows: list[ContributionStats] = []
    for label, _ in GaussianRenderer.CACHED_RASTER_GRAD_COMPONENT_SLICES:
        float_scale = np.asarray(split_float["backprop_contributions"][label]["grad_scales"], dtype=np.float32)[:, :3]
        fixed_scale = np.asarray(split_fixed["backprop_contributions"][label]["grad_scales"], dtype=np.float32)[:, :3]
        float_rotation = np.asarray(split_float["backprop_contributions"][label]["grad_rotations"], dtype=np.float32)
        fixed_rotation = np.asarray(split_fixed["backprop_contributions"][label]["grad_rotations"], dtype=np.float32)
        rows.append(
            ContributionStats(
                label=label,
                float_scale_norm=float(np.linalg.norm(float_scale.reshape(-1))),
                fixed_scale_norm=float(np.linalg.norm(fixed_scale.reshape(-1))),
                scale_norm_ratio=_norm_ratio(fixed_scale, float_scale),
                scale_cosine=_cosine(fixed_scale, float_scale),
                float_rotation_norm=float(np.linalg.norm(float_rotation.reshape(-1))),
                fixed_rotation_norm=float(np.linalg.norm(fixed_rotation.reshape(-1))),
                rotation_norm_ratio=_norm_ratio(fixed_rotation, float_rotation),
                rotation_cosine=_cosine(fixed_rotation, float_rotation),
                scale_max_abs_diff=float(np.max(np.abs(fixed_scale - float_scale))),
                rotation_max_abs_diff=float(np.max(np.abs(fixed_rotation - float_rotation))),
            )
        )

    payload = {
        "dataset_root": str(dataset_root),
        "images_subdir": args.images_subdir,
        "frame_index": int(args.frame_index) % len(frames),
        "frame_name": Path(frame.image_path).name,
        "width": width,
        "height": height,
        "scene_count": int(scene.count),
        "fixed_scale": float(args.fixed_scale),
        "full_scale_cosine": _cosine(split_fixed["backprop_full"]["grad_scales"][:, :3], split_float["backprop_full"]["grad_scales"][:, :3]),
        "full_rotation_cosine": _cosine(split_fixed["backprop_full"]["grad_rotations"], split_float["backprop_full"]["grad_rotations"]),
        "contributions": [asdict(row) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

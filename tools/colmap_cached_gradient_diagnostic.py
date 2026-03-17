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
from src.scene import GaussianScene, build_training_frames, initialize_scene_from_colmap_points, load_colmap_reconstruction, resolve_colmap_init_hparams

_INT32_MAX = np.iinfo(np.int32).max
_HEADROOM_USAGE = 0.5
_TARGET_MODE_DATASET_IMAGE = "dataset-image"
_TARGET_MODE_DISTORTED_SCENE = "distorted-scene"
_TARGET_MODES = (_TARGET_MODE_DATASET_IMAGE, _TARGET_MODE_DISTORTED_SCENE)


@dataclass(frozen=True, slots=True)
class ComponentStats:
    label: str
    fixed_scale: float
    float_abs_max: float
    float_abs_p99: float
    fixed_abs_max: float
    raw_int_abs_max: int
    raw_int_usage: float
    max_abs_err: float
    mean_abs_err: float
    p99_abs_err: float
    max_rel_err: float
    mean_rel_err: float
    suggested_scale_headroom_50: float


@dataclass(frozen=True, slots=True)
class GroupStats:
    label: str
    float_abs_max: float
    raw_int_usage: float
    max_abs_err: float
    mean_abs_err: float
    max_rel_err: float
    mean_rel_err: float
    suggested_scale_headroom_50: float


class ColmapCachedGradientDiagnostic:
    def __init__(
        self,
        dataset_root: Path,
        output_dir: Path,
        *,
        images_subdir: str,
        frame_index: int,
        max_gaussians: int,
        seed: int,
        target_mode: str,
        position_jitter: float,
        scale_jitter: float,
        color_jitter: float,
        opacity_jitter: float,
        list_capacity_multiplier: int,
        background: tuple[float, float, float],
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.images_subdir = images_subdir
        self.frame_index = int(frame_index)
        self.max_gaussians = int(max_gaussians)
        self.seed = int(seed)
        self.target_mode = target_mode
        self.position_jitter = float(position_jitter)
        self.scale_jitter = float(scale_jitter)
        self.color_jitter = float(color_jitter)
        self.opacity_jitter = float(opacity_jitter)
        self.list_capacity_multiplier = int(list_capacity_multiplier)
        self.background = np.asarray(background, dtype=np.float32)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = create_default_device(enable_debug_layers=False)

    def run(self) -> dict[str, object]:
        recon = load_colmap_reconstruction(self.dataset_root)
        frames = build_training_frames(recon, images_subdir=self.images_subdir)
        frame = frames[self.frame_index % len(frames)]
        camera = frame.make_camera(near=0.1, far=1000.0)
        scene = self._build_scene(recon)
        width, height = int(frame.width), int(frame.height)
        target_image = self._build_target_image(scene, camera, frame, width, height)

        renderer_float = self._make_renderer(width, height, "float")
        renderer_fixed = self._make_renderer(width, height, "fixed")
        grads_float = renderer_float.debug_raster_backward_grads_against_target(scene, camera, target_image, background=self.background)
        grads_fixed = renderer_fixed.debug_raster_backward_grads_against_target(scene, camera, target_image, background=self.background)

        cached_float = np.asarray(grads_float["cached_raster_grads_float"], dtype=np.float32)
        cached_fixed = np.asarray(renderer_fixed.read_cached_raster_grads_fixed_decoded(scene.count), dtype=np.float32)
        cached_raw = np.asarray(grads_fixed["cached_raster_grads_fixed"], dtype=np.int32)
        encode_scales = 1.0 / np.asarray(renderer_fixed._RASTER_GRAD_FIXED_DECODE_SCALES, dtype=np.float32)

        component_stats = self._component_stats(cached_float, cached_fixed, cached_raw, encode_scales)
        group_stats = self._group_stats(component_stats)
        summary = {
            "dataset_root": str(self.dataset_root),
            "images_subdir": self.images_subdir,
            "frame_index": int(self.frame_index % len(frames)),
            "frame_name": Path(frame.image_path).name,
            "width": width,
            "height": height,
            "scene_count": int(scene.count),
            "target_mode": self.target_mode,
            "background": self.background.tolist(),
            "component_stats": [asdict(row) for row in component_stats],
            "group_stats": [asdict(row) for row in group_stats],
        }
        self._write_summary(summary, target_image, grads_float["output_grad"])
        return summary

    def _build_scene(self, recon) -> object:
        init_hparams = resolve_colmap_init_hparams(recon, self.max_gaussians)
        return initialize_scene_from_colmap_points(recon, max_gaussians=self.max_gaussians, seed=self.seed, init_hparams=init_hparams)

    def _build_target_image(self, scene, camera, frame, width: int, height: int) -> np.ndarray:
        if self.target_mode == _TARGET_MODE_DATASET_IMAGE:
            return self._load_frame_target_image(frame.image_path, width, height)
        target_renderer = self._make_renderer(width, height, "float")
        target_scene = self._make_distorted_scene(scene)
        return target_renderer.render(target_scene, camera, background=self.background).image

    def _make_renderer(self, width: int, height: int, atomic_mode: str) -> GaussianRenderer:
        return GaussianRenderer(
            self.device,
            width=width,
            height=height,
            radius_scale=1.0,
            list_capacity_multiplier=self.list_capacity_multiplier,
            cached_raster_grad_atomic_mode=atomic_mode,
        )

    def _load_frame_target_image(self, path: Path, width: int, height: int) -> np.ndarray:
        with Image.open(path) as pil_image:
            rgb = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
        if rgb.shape[0] != height or rgb.shape[1] != width:
            raise RuntimeError(f"Target frame size mismatch for {path}: expected {(height, width)}, got {rgb.shape[:2]}")
        alpha = np.ones((height, width, 1), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2), dtype=np.float32)

    def _make_distorted_scene(self, scene) -> object:
        rng = np.random.default_rng(self.seed + 17)
        return GaussianScene(
            positions=np.asarray(scene.positions + rng.normal(0.0, self.position_jitter, size=scene.positions.shape), dtype=np.float32),
            scales=np.asarray(scene.scales + rng.normal(0.0, self.scale_jitter, size=scene.scales.shape), dtype=np.float32),
            rotations=np.asarray(scene.rotations, dtype=np.float32),
            opacities=np.clip(np.asarray(scene.opacities + rng.normal(0.0, self.opacity_jitter, size=scene.opacities.shape), dtype=np.float32), 1e-3, 0.999),
            colors=np.clip(np.asarray(scene.colors + rng.normal(0.0, self.color_jitter, size=scene.colors.shape), dtype=np.float32), 0.0, 1.0),
            sh_coeffs=np.asarray(scene.sh_coeffs, dtype=np.float32),
        )

    def _component_stats(
        self,
        cached_float: np.ndarray,
        cached_fixed: np.ndarray,
        cached_raw: np.ndarray,
        encode_scales: np.ndarray,
    ) -> list[ComponentStats]:
        rows: list[ComponentStats] = []
        magnitude_floor = 1e-6
        labels = GaussianRenderer.CACHED_RASTER_GRAD_COMPONENT_LABELS
        for param_id, label in enumerate(labels):
            ref = cached_float[:, param_id]
            actual = cached_fixed[:, param_id]
            raw = cached_raw[:, param_id]
            abs_ref = np.abs(ref)
            abs_err = np.abs(actual - ref)
            rel_mask = abs_ref > magnitude_floor
            rel_err = abs_err[rel_mask] / abs_ref[rel_mask] if np.any(rel_mask) else np.zeros((0,), dtype=np.float32)
            float_abs_max = float(np.max(abs_ref)) if abs_ref.size else 0.0
            rows.append(
                ComponentStats(
                    label=label,
                    fixed_scale=float(encode_scales[param_id]),
                    float_abs_max=float_abs_max,
                    float_abs_p99=float(np.percentile(abs_ref, 99.0)) if abs_ref.size else 0.0,
                    fixed_abs_max=float(np.max(np.abs(actual))) if actual.size else 0.0,
                    raw_int_abs_max=int(np.max(np.abs(raw))) if raw.size else 0,
                    raw_int_usage=float(np.max(np.abs(raw)) / _INT32_MAX) if raw.size else 0.0,
                    max_abs_err=float(np.max(abs_err)) if abs_err.size else 0.0,
                    mean_abs_err=float(np.mean(abs_err)) if abs_err.size else 0.0,
                    p99_abs_err=float(np.percentile(abs_err, 99.0)) if abs_err.size else 0.0,
                    max_rel_err=float(np.max(rel_err)) if rel_err.size else 0.0,
                    mean_rel_err=float(np.mean(rel_err)) if rel_err.size else 0.0,
                    suggested_scale_headroom_50=float(_HEADROOM_USAGE * _INT32_MAX / max(float_abs_max, 1e-12)),
                )
            )
        return rows

    def _group_stats(self, rows: list[ComponentStats]) -> list[GroupStats]:
        groups = (
            ("roLocal", slice(0, 3)),
            ("logInvScale", slice(3, 6)),
            ("quat", slice(6, 10)),
            ("color", slice(10, 13)),
            ("opacity", slice(13, 14)),
        )
        grouped: list[GroupStats] = []
        for label, group_slice in groups:
            group_rows = rows[group_slice]
            grouped.append(
                GroupStats(
                    label=label,
                    float_abs_max=max(row.float_abs_max for row in group_rows),
                    raw_int_usage=max(row.raw_int_usage for row in group_rows),
                    max_abs_err=max(row.max_abs_err for row in group_rows),
                    mean_abs_err=float(np.mean([row.mean_abs_err for row in group_rows])),
                    max_rel_err=max(row.max_rel_err for row in group_rows),
                    mean_rel_err=float(np.mean([row.mean_rel_err for row in group_rows])),
                    suggested_scale_headroom_50=min(row.suggested_scale_headroom_50 for row in group_rows),
                )
            )
        return grouped

    def _write_summary(self, summary: dict[str, object], target_image: np.ndarray, output_grad: np.ndarray) -> None:
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._save_rgb(self.output_dir / "target.png", target_image)
        self._save_rgb(self.output_dir / "output_grad_abs.png", np.clip(np.abs(output_grad[..., :3]) * 64.0, 0.0, 1.0))

    @staticmethod
    def _save_rgb(path: Path, image: np.ndarray) -> None:
        rgb = np.clip(np.asarray(image, dtype=np.float32)[..., :3], 0.0, 1.0)
        Image.fromarray((255.0 * rgb + 0.5).astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed and float cached raster gradients on a real COLMAP scene.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/garden"))
    parser.add_argument("--images-subdir", type=str, default="images_8")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/colmap_cached_gradient_diagnostic"))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--max-gaussians", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-mode", type=str, choices=_TARGET_MODES, default=_TARGET_MODE_DATASET_IMAGE)
    parser.add_argument("--position-jitter", type=float, default=0.01)
    parser.add_argument("--scale-jitter", type=float, default=0.08)
    parser.add_argument("--color-jitter", type=float, default=0.05)
    parser.add_argument("--opacity-jitter", type=float, default=0.04)
    parser.add_argument("--list-capacity-multiplier", type=int, default=64)
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = ColmapCachedGradientDiagnostic(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        images_subdir=args.images_subdir,
        frame_index=args.frame_index,
        max_gaussians=args.max_gaussians,
        seed=args.seed,
        target_mode=args.target_mode,
        position_jitter=args.position_jitter,
        scale_jitter=args.scale_jitter,
        color_jitter=args.color_jitter,
        opacity_jitter=args.opacity_jitter,
        list_capacity_multiplier=args.list_capacity_multiplier,
        background=tuple(float(x) for x in args.background),
    ).run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

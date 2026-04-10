from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src import create_default_device
from src.renderer import GaussianRenderer
from src.scene import ColmapFrame, GaussianScene
from src.training import AdamHyperParams, GaussianTrainer, StabilityHyperParams, TrainingHyperParams

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass(frozen=True, slots=True)
class SweepRow:
    scale: float
    scale_over_floor: float
    fitted_radius: float
    loss: float
    grad_scale_x: float
    grad_scale_mean: float
    grad_scale_norm: float


class ScaleGradientDiagnostic:
    def __init__(self, output_dir: Path, target_scale_mul: float, init_opacity: float) -> None:
        self.output_dir = output_dir
        self.target_scale_mul = float(target_scale_mul)
        self.init_opacity = float(init_opacity)
        self.background = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = create_default_device(enable_debug_layers=False)
        self.width = 64
        self.height = 64
        self.radius_scale = 1.0
        self.frame = self._make_frame(self.output_dir / "target.png")
        self.camera = self.frame.make_camera(near=0.1, far=20.0)

    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target_renderer = self._make_renderer()
        pixel_floor_scale = self.camera.pixel_world_size_max(3.0, self.width, self.height)
        target_scale = self.target_scale_mul * pixel_floor_scale
        target_image = target_renderer.render(self._make_scene(target_scale), self.camera, background=self.background).image
        self._save_rgb(self.frame.image_path, target_image)

        rows = [self._measure_row(scale, target_scale) for scale in self._sweep_scales(pixel_floor_scale)]
        self._write_csv(rows)
        self._write_plot(rows, pixel_floor_scale)
        summary = {
            "output_dir": str(self.output_dir),
            "pixel_floor_scale": float(pixel_floor_scale),
            "target_scale": float(target_scale),
            "target_radius": float(self._fit_rendered_gaussian_radius(target_image)),
            "row_count": len(rows),
            "grad_min": float(min(row.grad_scale_mean for row in rows)),
            "grad_max": float(max(row.grad_scale_mean for row in rows)),
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _make_renderer(self) -> GaussianRenderer:
        return GaussianRenderer(self.device, width=self.width, height=self.height, radius_scale=self.radius_scale, list_capacity_multiplier=16)

    def _make_scene(self, scale: float) -> GaussianScene:
        return GaussianScene(
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            scales=np.full((1, 3), float(scale), dtype=np.float32),
            rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            opacities=np.array([self.init_opacity], dtype=np.float32),
            colors=np.array([[0.8, 0.6, 0.2]], dtype=np.float32),
            sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
        )

    def _make_frame(self, path: Path) -> ColmapFrame:
        Image.fromarray(np.zeros((self.height, self.width, 3), dtype=np.uint8)).save(path)
        return ColmapFrame(
            image_id=0,
            image_path=path,
            q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
            fx=72.0,
            fy=72.0,
            cx=self.width * 0.5,
            cy=self.height * 0.5,
            width=self.width,
            height=self.height,
        )

    def _sweep_scales(self, pixel_floor_scale: float) -> np.ndarray:
        return np.unique(
            np.concatenate(
                [
                    np.geomspace(0.05 * pixel_floor_scale, 0.95 * pixel_floor_scale, 18),
                    np.linspace(0.95 * pixel_floor_scale, 1.05 * pixel_floor_scale, 21),
                    np.geomspace(1.05 * pixel_floor_scale, self.target_scale_mul * 1.6 * pixel_floor_scale, 24),
                ]
            ).astype(np.float32)
        )

    def _render_image(self, renderer: GaussianRenderer) -> np.ndarray:
        tex, _ = renderer.render_to_texture(self.camera, background=self.background)
        return np.asarray(tex.to_numpy(), dtype=np.float32).copy()

    def _fit_rendered_gaussian_radius(self, image: np.ndarray) -> float:
        rgb = np.asarray(image, dtype=np.float32)[..., :3]
        weights = np.mean(np.clip(rgb, 0.0, None), axis=2, dtype=np.float32)
        weight_sum = float(np.sum(weights, dtype=np.float64))
        if weight_sum <= 1e-8:
            return 0.0
        height, width = weights.shape
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cx = float(np.sum(weights * xx, dtype=np.float64) / weight_sum)
        cy = float(np.sum(weights * yy, dtype=np.float64) / weight_sum)
        dx = xx - cx
        dy = yy - cy
        mean_r2 = float(np.sum(weights * (dx * dx + dy * dy), dtype=np.float64) / weight_sum)
        sigma = np.sqrt(max(0.5 * mean_r2, 0.0))
        return float(3.0 * sigma)

    def _make_trainer(self, renderer: GaussianRenderer, scene: GaussianScene, target_scale: float) -> GaussianTrainer:
        return GaussianTrainer(
            device=self.device,
            renderer=renderer,
            scene=scene,
            frames=[self.frame],
            adam_hparams=AdamHyperParams(position_lr=0.0, scale_lr=0.1, rotation_lr=0.0, color_lr=0.0, opacity_lr=0.0),
            stability_hparams=StabilityHyperParams(max_update=0.5, max_scale=target_scale * 2.0),
            training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0),
            seed=123,
        )

    def _measure_row(self, scale: float, target_scale: float) -> SweepRow:
        renderer = self._make_renderer()
        trainer = self._make_trainer(renderer, self._make_scene(scale), target_scale)
        renderer.execute_prepass_for_current_scene(self.camera, sync_counts=False)
        enc = self.device.create_command_encoder()
        renderer.rasterize_current_scene(enc, self.camera, self.background)
        trainer._dispatch_loss_grad(enc, trainer.get_frame_target_texture(0, native_resolution=False))
        trainer._dispatch_raster_forward_backward(enc, self.camera, self.background)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()
        grad = np.frombuffer(renderer.work_buffers["grad_scales"].to_numpy().tobytes(), dtype=np.float32)[:4][:3].copy()
        image = self._render_image(renderer)
        return SweepRow(
            scale=float(scale),
            scale_over_floor=float(scale / self.camera.pixel_world_size_max(3.0, self.width, self.height)),
            fitted_radius=self._fit_rendered_gaussian_radius(image),
            loss=float(trainer._read_loss_metrics()[0]),
            grad_scale_x=float(grad[0]),
            grad_scale_mean=float(np.mean(grad)),
            grad_scale_norm=float(np.linalg.norm(grad)),
        )

    def _write_csv(self, rows: list[SweepRow]) -> None:
        with (self.output_dir / "scale_gradient_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("scale", "scale_over_floor", "fitted_radius", "loss", "grad_scale_x", "grad_scale_mean", "grad_scale_norm"))
            for row in rows:
                writer.writerow((row.scale, row.scale_over_floor, row.fitted_radius, row.loss, row.grad_scale_x, row.grad_scale_mean, row.grad_scale_norm))

    def _write_plot(self, rows: list[SweepRow], pixel_floor_scale: float) -> None:
        if plt is None:
            return
        scale = np.asarray([row.scale_over_floor for row in rows], dtype=np.float32)
        radius = np.asarray([row.fitted_radius for row in rows], dtype=np.float32)
        grad = np.asarray([row.grad_scale_mean for row in rows], dtype=np.float32)
        loss = np.asarray([row.loss for row in rows], dtype=np.float32)

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=160, sharex=True)
        axes[0].plot(scale, radius, color="#1f77b4", linewidth=2.0)
        axes[0].axvline(1.0, color="#888888", linestyle="--", linewidth=1.5)
        axes[0].set_ylabel("Fitted Radius (px)")
        axes[0].set_title("Rendered Radius and Training Gradient vs Scale")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(scale, grad, color="#d62728", linewidth=2.0, label="dL/dscale (mean xyz)")
        axes[1].plot(scale, loss, color="#2ca02c", linewidth=1.5, label="loss")
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].axvline(1.0, color="#888888", linestyle="--", linewidth=1.5, label="pixel floor")
        axes[1].set_xlabel("Scale / Pixel-Floor Scale")
        axes[1].set_ylabel("Gradient / Loss")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "scale_gradient_sweep.png")
        plt.close(fig)

    @staticmethod
    def _save_rgb(path: Path, image: np.ndarray) -> None:
        rgb = np.clip(np.asarray(image, dtype=np.float32)[..., :3], 0.0, 1.0)
        Image.fromarray((255.0 * rgb + 0.5).astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep one-splat scale across a wide range and record rendered radius plus training gradients.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scale_gradient_diagnostic"))
    parser.add_argument("--target-scale-mul", type=float, default=7.5)
    parser.add_argument("--init-opacity", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = ScaleGradientDiagnostic(
        output_dir=args.output_dir,
        target_scale_mul=args.target_scale_mul,
        init_opacity=args.init_opacity,
    ).run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

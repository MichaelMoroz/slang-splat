from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import create_default_device
from src.app.shared import RendererParams, apply_training_profile, build_training_params, renderer_kwargs
from src.renderer import GaussianRenderer
from src.scene import build_training_frames, initialize_scene_from_colmap_points, load_colmap_reconstruction, resolve_colmap_init_hparams
from src.training import GaussianTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture per-fragment cached-raster gradient contributions on the last full-resolution training iteration and compare offline quantization hypotheses.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/room"))
    parser.add_argument("--images-subdir", type=str, default="images_4")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--max-gaussians", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fixed-scale", type=float, default=0.125)
    parser.add_argument("--sample-count", type=int, default=768)
    parser.add_argument("--capture-capacity", type=int, default=1_000_000)
    parser.add_argument("--modes", nargs="+", default=("float", "fixed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cached_raster_grad_convergence_room_images4"))
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


def _make_trainer(device, scene, frames, width: int, height: int, params, init_hparams, seed: int, *, atomic_mode: str, fixed_scale: float) -> GaussianTrainer:
    renderer = GaussianRenderer(
        device,
        width=width,
        height=height,
        **renderer_kwargs(RendererParams(cached_raster_grad_atomic_mode=atomic_mode, cached_raster_grad_fixed_scale=fixed_scale)),
    )
    return GaussianTrainer(
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


def _sym_round(values: np.ndarray) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64)
    return np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5)).astype(np.int64)


def _quantize_stream_current(contribs: np.ndarray, encode_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ints = np.zeros((contribs.shape[1],), dtype=np.int64)
    prefix = np.zeros_like(contribs, dtype=np.float64)
    for index, value in enumerate(np.asarray(contribs, dtype=np.float64)):
        ints += _sym_round(value * encode_scales)
        prefix[index] = ints / encode_scales
    return ints / encode_scales, prefix


def _quantize_stream_stochastic(contribs: np.ndarray, encode_scales: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ints = np.zeros((contribs.shape[1],), dtype=np.int64)
    prefix = np.zeros_like(contribs, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for index, value in enumerate(np.asarray(contribs, dtype=np.float64)):
        scaled = value * encode_scales
        lower = np.floor(scaled)
        frac = scaled - lower
        ints += (lower + (rng.random(scaled.shape) < frac).astype(np.float64)).astype(np.int64)
        prefix[index] = ints / encode_scales
    return ints / encode_scales, prefix


def _quantize_stream_residual(contribs: np.ndarray, encode_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ints = np.zeros((contribs.shape[1],), dtype=np.int64)
    residual = np.zeros((contribs.shape[1],), dtype=np.float64)
    prefix = np.zeros_like(contribs, dtype=np.float64)
    for index, value in enumerate(np.asarray(contribs, dtype=np.float64)):
        quantized = _sym_round((value + residual) * encode_scales)
        ints += quantized
        residual += value - quantized / encode_scales
        prefix[index] = ints / encode_scales
    return ints / encode_scales, prefix


def _quantize_stream_chunked(contribs: np.ndarray, encode_scales: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    ints = np.zeros((contribs.shape[1],), dtype=np.int64)
    prefix = np.zeros_like(contribs, dtype=np.float64)
    for chunk_start in range(0, contribs.shape[0], max(int(chunk_size), 1)):
        chunk_end = min(chunk_start + max(int(chunk_size), 1), contribs.shape[0])
        ints += _sym_round(np.sum(contribs[chunk_start:chunk_end], axis=0, dtype=np.float64) * encode_scales)
        prefix[chunk_start:chunk_end] = ints / encode_scales
    return ints / encode_scales, prefix


def _cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_flat = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs_flat = np.asarray(rhs, dtype=np.float64).reshape(-1)
    lhs_norm = np.linalg.norm(lhs_flat)
    rhs_norm = np.linalg.norm(rhs_flat)
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 1.0 if lhs_norm <= 0.0 and rhs_norm <= 0.0 else 0.0
    return float(np.dot(lhs_flat, rhs_flat) / (lhs_norm * rhs_norm))


def _relative_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
    rhs_norm = float(np.linalg.norm(np.asarray(rhs, dtype=np.float64).reshape(-1)))
    if rhs_norm <= 1e-20:
        return 0.0
    return float(np.linalg.norm(np.asarray(lhs, dtype=np.float64).reshape(-1) - np.asarray(rhs, dtype=np.float64).reshape(-1)) / rhs_norm)


def _analyze_quantization(records: dict[str, np.ndarray], encode_scales: np.ndarray, labels: tuple[str, ...], seed: int) -> dict[str, object]:
    splat_ids = np.asarray(records["splat_ids"], dtype=np.int32)
    gradients = np.asarray(records["gradients"], dtype=np.float32)
    selected = np.unique(splat_ids)
    hypotheses = {
        "current": lambda contribs, splat_id: _quantize_stream_current(contribs, encode_scales),
        "stochastic": lambda contribs, splat_id: _quantize_stream_stochastic(contribs, encode_scales, seed + int(splat_id) * 17),
        "residual_feedback": lambda contribs, splat_id: _quantize_stream_residual(contribs, encode_scales),
        "chunk8": lambda contribs, splat_id: _quantize_stream_chunked(contribs, encode_scales, 8),
        "chunk32": lambda contribs, splat_id: _quantize_stream_chunked(contribs, encode_scales, 32),
        "finer_x4": lambda contribs, splat_id: _quantize_stream_current(contribs, encode_scales * 4.0),
        "coarser_div4": lambda contribs, splat_id: _quantize_stream_current(contribs, encode_scales * 0.25),
    }
    hypothesis_stats: dict[str, dict[str, float]] = {}
    per_splat_counts: list[dict[str, object]] = []
    final_reference = np.zeros((selected.shape[0], len(labels)), dtype=np.float64)
    for row, splat_id in enumerate(selected):
        mask = splat_ids == splat_id
        contribs = gradients[mask].astype(np.float64, copy=False)
        true_prefix = np.cumsum(contribs, axis=0, dtype=np.float64)
        true_final = true_prefix[-1]
        final_reference[row] = true_final
        per_splat_counts.append({"splat_id": int(splat_id), "contribution_count": int(contribs.shape[0]), "float_final_norm": float(np.linalg.norm(true_final))})
        final_norm = max(float(np.linalg.norm(true_final)), 1e-20)
        for name, quantizer in hypotheses.items():
            final_sum, prefix = quantizer(contribs, int(splat_id))
            stats = hypothesis_stats.setdefault(name, {"final_rel_error_sum": 0.0, "final_abs_error_max": 0.0, "prefix_rel_error_sum": 0.0, "prefix_rel_error_p95_accum": 0.0, "cosine_sum": 0.0, "count": 0.0})
            prefix_error = np.linalg.norm(prefix - true_prefix, axis=1) / final_norm
            final_error = final_sum - true_final
            stats["final_rel_error_sum"] += float(np.linalg.norm(final_error) / final_norm)
            stats["final_abs_error_max"] = max(stats["final_abs_error_max"], float(np.max(np.abs(final_error))))
            stats["prefix_rel_error_sum"] += float(np.mean(prefix_error))
            stats["prefix_rel_error_p95_accum"] += float(np.percentile(prefix_error, 95))
            stats["cosine_sum"] += _cosine(final_sum, true_final)
            stats["count"] += 1.0
    ranked: list[dict[str, object]] = []
    for name, stats in hypothesis_stats.items():
        count = max(stats.pop("count"), 1.0)
        row = {
            "name": name,
            "mean_final_rel_error": stats["final_rel_error_sum"] / count,
            "mean_prefix_rel_error": stats["prefix_rel_error_sum"] / count,
            "mean_prefix_rel_error_p95": stats["prefix_rel_error_p95_accum"] / count,
            "mean_final_cosine": stats["cosine_sum"] / count,
            "max_final_abs_error": stats["final_abs_error_max"],
        }
        ranked.append(row)
    ranked.sort(key=lambda row: (row["mean_prefix_rel_error"], row["mean_final_rel_error"]))
    component_abs_sum = np.sum(np.abs(final_reference), axis=0, dtype=np.float64)
    return {
        "selected_splat_count": int(selected.shape[0]),
        "captured_record_count": int(gradients.shape[0]),
        "per_splat_counts": per_splat_counts,
        "component_abs_sum": [{"label": label, "abs_sum": float(component_abs_sum[index])} for index, label in enumerate(labels)],
        "hypotheses": ranked,
        "best_hypothesis": ranked[0]["name"] if ranked else "",
    }


def _select_splats(candidate_grads: np.ndarray, sample_count: int, seed: int) -> np.ndarray:
    norms = np.linalg.norm(np.asarray(candidate_grads, dtype=np.float64), axis=1)
    candidates = np.flatnonzero(norms > 0.0)
    if candidates.size == 0:
        return np.zeros((0,), dtype=np.int32)
    rng = np.random.default_rng(int(seed))
    if candidates.size <= int(sample_count):
        return candidates.astype(np.int32)
    return np.sort(rng.choice(candidates, size=int(sample_count), replace=False).astype(np.int32))


def _execute_last_training_backward(trainer: GaussianTrainer, frame_index: int, *, selected_splats: np.ndarray | None = None, capture_capacity: int = 0) -> dict[str, object]:
    camera = trainer.make_frame_camera(frame_index, trainer.renderer.width, trainer.renderer.height)
    background = np.asarray(trainer.training.background, dtype=np.float32).reshape(3)
    if selected_splats is None:
        trainer.renderer.disable_cached_raster_grad_contribution_capture()
    else:
        trainer.renderer.configure_cached_raster_grad_contribution_capture(trainer.scene.count, np.asarray(selected_splats, dtype=np.int32), int(capture_capacity))
    try:
        enc = trainer.device.create_command_encoder()
        trainer.renderer.record_prepass_for_current_scene(enc, camera)
        target_texture = trainer.get_frame_target_texture(frame_index, native_resolution=False, encoder=enc)
        trainer._dispatch_training_forward(enc, camera, background, target_texture)
        trainer._dispatch_training_backward(enc, camera, background, target_texture)
        trainer.device.submit_command_buffer(enc.finish())
        trainer.device.wait()
        result = {
            "frame_index": int(frame_index),
            "frame_name": Path(trainer._frame(frame_index).image_path).name,
            "camera": camera,
            "cached_raster_grads_float": trainer.renderer.read_cached_raster_grads_float(trainer.scene.count),
            "cached_raster_grads_fixed_decoded": trainer.renderer.read_cached_raster_grads_fixed_decoded(trainer.scene.count),
        }
        if selected_splats is not None:
            result["capture"] = trainer.renderer.read_cached_raster_grad_contribution_capture()
        return result
    finally:
        trainer.renderer.disable_cached_raster_grad_contribution_capture()


def _run_mode(device, scene, frames, width: int, height: int, params, init_hparams, args: argparse.Namespace, atomic_mode: str) -> dict[str, object]:
    trainer = _make_trainer(device, scene, frames, width, height, params, init_hparams, int(args.seed), atomic_mode=atomic_mode, fixed_scale=float(args.fixed_scale))
    if int(args.steps) > 1:
        trainer.step_batch(int(args.steps) - 1)
    frame_index = int(trainer._next_frame_index())
    dry = _execute_last_training_backward(trainer, frame_index)
    candidate = np.asarray(dry["cached_raster_grads_fixed_decoded"] if atomic_mode == "fixed" else dry["cached_raster_grads_float"], dtype=np.float32)
    selected_splats = _select_splats(candidate, int(args.sample_count), int(args.seed) + (0 if atomic_mode == "float" else 1))
    captured = _execute_last_training_backward(trainer, frame_index, selected_splats=selected_splats, capture_capacity=int(args.capture_capacity))
    capture = trainer.renderer.decode_cached_raster_grad_contribution_records(np.asarray(captured["capture"]["records"], dtype=np.float32))
    capture["record_count"] = int(captured["capture"]["record_count"])
    capture["attempted_count"] = int(captured["capture"]["attempted_count"])
    capture["overflow_count"] = int(captured["capture"]["overflow_count"])
    encode_scales = 1.0 / trainer.renderer.cached_raster_grad_fixed_decode_scales.astype(np.float64)
    analysis = _analyze_quantization(capture, encode_scales, trainer.renderer.CACHED_RASTER_GRAD_COMPONENT_LABELS, int(args.seed))
    selected_active = np.asarray(
        captured["cached_raster_grads_fixed_decoded"] if atomic_mode == "fixed" else captured["cached_raster_grads_float"],
        dtype=np.float32,
    )[selected_splats]
    return {
        "mode": atomic_mode,
        "step": int(args.steps),
        "frame_index": frame_index,
        "frame_name": captured["frame_name"],
        "selected_splats": selected_splats,
        "capture": capture,
        "analysis": analysis,
        "selected_active_cached_grads": selected_active,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).resolve()
    recon, frames, width, height, params, init_hparams, scene = _training_setup(dataset_root, args.images_subdir, int(args.max_gaussians), int(args.seed))
    del recon
    device = create_default_device(enable_debug_layers=True)

    summaries: list[dict[str, object]] = []
    for mode in (str(value).strip().lower() for value in args.modes):
        result = _run_mode(device, scene, frames, width, height, params, init_hparams, args, mode)
        capture = result["capture"]
        np.savez_compressed(
            output_dir / f"{mode}_capture_records.npz",
            splat_ids=np.asarray(capture["splat_ids"], dtype=np.int32),
            pixel_x=np.asarray(capture["pixel_x"], dtype=np.int32),
            pixel_y=np.asarray(capture["pixel_y"], dtype=np.int32),
            global_indices=np.asarray(capture["global_indices"], dtype=np.int32),
            gradients=np.asarray(capture["gradients"], dtype=np.float32),
            selected_splats=np.asarray(result["selected_splats"], dtype=np.int32),
            selected_active_cached_grads=np.asarray(result["selected_active_cached_grads"], dtype=np.float32),
        )
        summary = {
            "dataset_root": str(dataset_root),
            "images_subdir": args.images_subdir,
            "width": width,
            "height": height,
            "steps": int(args.steps),
            "max_gaussians": int(args.max_gaussians),
            "seed": int(args.seed),
            "fixed_scale": float(args.fixed_scale),
            "sample_count": int(args.sample_count),
            "capture_capacity": int(args.capture_capacity),
            "mode": mode,
            "frame_index": result["frame_index"],
            "frame_name": result["frame_name"],
            "record_count": int(result["capture"]["record_count"]),
            "attempted_count": int(result["capture"]["attempted_count"]),
            "overflow_count": int(result["capture"]["overflow_count"]),
            "analysis": result["analysis"],
        }
        summaries.append(summary)
        (output_dir / f"{mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    payload = {
        "dataset_root": str(dataset_root),
        "images_subdir": args.images_subdir,
        "width": width,
        "height": height,
        "steps": int(args.steps),
        "max_gaussians": int(args.max_gaussians),
        "seed": int(args.seed),
        "fixed_scale": float(args.fixed_scale),
        "sample_count": int(args.sample_count),
        "capture_capacity": int(args.capture_capacity),
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

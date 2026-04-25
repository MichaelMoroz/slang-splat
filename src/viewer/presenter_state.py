from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..training import (
    TRAIN_SUBSAMPLE_MAX_FACTOR,
    resolve_auto_train_subsample_factor,
    resolve_base_learning_rate,
    resolve_colorspace_mod,
    resolve_position_lr_mul,
    resolve_position_random_step_noise_lr,
    resolve_refinement_growth_ratio,
    resolve_refinement_min_contribution,
    resolve_sh_band,
    resolve_sh_lr_mul,
    resolve_sorting_order_dithering,
)
from ..utility import clamp_index

_DEFAULT_TRAINING_STEPS_PER_FRAME = 3
_MAX_TRAINING_STEPS_PER_FRAME = 8
_TRAIN_DOWNSCALE_MODE_AUTO = 0
_CAMERA_OVERLAY_LENGTH_FRACTION = 0.08
_CAMERA_OVERLAY_MIN_LENGTH = 0.05
_CAMERA_OVERLAY_NEAR_FRACTION = 0.35
_CAMERA_OVERLAY_COLOR = (0.18, 0.70, 0.98, 0.72)
_CAMERA_OVERLAY_ACTIVE_COLOR = (1.00, 0.78, 0.18, 0.96)
_CAMERA_MIN_DIST_RING_SAMPLES = 40


def _control_value(viewer: object, key: str, default: object) -> object:
    control = getattr(getattr(viewer, "ui", None), "controls", {}).get(key)
    return default if control is None else control.value


def _schedule_state_from_controls(viewer: object) -> object:
    return SimpleNamespace(
        lr_schedule_enabled=bool(_control_value(viewer, "lr_schedule_enabled", True)),
        lr_schedule_start_lr=float(_control_value(viewer, "lr_schedule_start_lr", 0.005)),
        lr_schedule_stage1_lr=float(_control_value(viewer, "lr_schedule_stage1_lr", 0.002)),
        lr_schedule_stage2_lr=float(_control_value(viewer, "lr_schedule_stage2_lr", 0.001)),
        lr_schedule_end_lr=float(_control_value(viewer, "lr_schedule_end_lr", 1.5e-4)),
        lr_schedule_steps=int(_control_value(viewer, "lr_schedule_steps", 30000)),
        lr_schedule_stage1_step=int(_control_value(viewer, "lr_schedule_stage1_step", 3000)),
        lr_schedule_stage2_step=int(_control_value(viewer, "lr_schedule_stage2_step", 14000)),
        lr_pos_mul=float(_control_value(viewer, "lr_pos_mul", 1.0)),
        lr_pos_stage1_mul=float(_control_value(viewer, "lr_pos_stage1_mul", 0.75)),
        lr_pos_stage2_mul=float(_control_value(viewer, "lr_pos_stage2_mul", 0.2)),
        lr_pos_stage3_mul=float(_control_value(viewer, "lr_pos_stage3_mul", 0.2)),
        lr_sh_mul=float(_control_value(viewer, "lr_sh_mul", 0.05)),
        lr_sh_stage1_mul=float(_control_value(viewer, "lr_sh_stage1_mul", 0.05)),
        lr_sh_stage2_mul=float(_control_value(viewer, "lr_sh_stage2_mul", 0.05)),
        lr_sh_stage3_mul=float(_control_value(viewer, "lr_sh_stage3_mul", 0.05)),
        colorspace_mod=float(_control_value(viewer, "colorspace_mod", 1.0)),
        colorspace_mod_stage1=float(_control_value(viewer, "colorspace_mod_stage1", 1.0)),
        colorspace_mod_stage2=float(_control_value(viewer, "colorspace_mod_stage2", 1.0)),
        colorspace_mod_stage3=float(_control_value(viewer, "colorspace_mod_stage3", 1.0)),
        sorting_order_dithering=float(_control_value(viewer, "sorting_order_dithering", 0.5)),
        sorting_order_dithering_stage1=float(_control_value(viewer, "sorting_order_dithering_stage1", 0.2)),
        sorting_order_dithering_stage2=float(_control_value(viewer, "sorting_order_dithering_stage2", 0.05)),
        sorting_order_dithering_stage3=float(_control_value(viewer, "sorting_order_dithering_stage3", 0.01)),
        position_random_step_noise_lr=float(_control_value(viewer, "position_random_step_noise_lr", 5e5)),
        position_random_step_noise_stage1_lr=float(_control_value(viewer, "position_random_step_noise_stage1_lr", 466666.6666666667)),
        position_random_step_noise_stage2_lr=float(_control_value(viewer, "position_random_step_noise_stage2_lr", 416666.6666666667)),
        position_random_step_noise_stage3_lr=float(_control_value(viewer, "position_random_step_noise_stage3_lr", 0.0)),
        sh_band=int(_control_value(viewer, "sh_band", 0)),
        sh_band_stage1=int(_control_value(viewer, "sh_band_stage1", 1)),
        sh_band_stage2=int(_control_value(viewer, "sh_band_stage2", 1)),
        sh_band_stage3=int(_control_value(viewer, "sh_band_stage3", 1)),
    )


def _schedule_steps(training: object) -> tuple[int, int, int]:
    stage1 = max(int(getattr(training, "lr_schedule_stage1_step", 0)), 0)
    stage2 = max(int(getattr(training, "lr_schedule_stage2_step", stage1)), stage1)
    return stage1, stage2, max(int(getattr(training, "lr_schedule_steps", stage2)), stage2)


def _schedule_runtime(viewer: object) -> tuple[object, int]:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is None:
        return _schedule_state_from_controls(viewer), 0
    return trainer.training, max(int(trainer.state.step), 0)


def _schedule_summary_text(training: object, current_lr: float) -> str:
    lr0 = max(float(training.lr_schedule_start_lr), 1e-8)
    lr1 = max(float(training.lr_schedule_stage1_lr), 1e-8)
    lr2 = max(float(training.lr_schedule_stage2_lr), 1e-8)
    lr3 = max(float(training.lr_schedule_end_lr), 1e-8)
    stage1, stage2, stage3 = _schedule_steps(training)
    return f"LR Schedule: {lr0:.2e}@0 -> {lr1:.2e}@{stage1:,} -> {lr2:.2e}@{stage2:,} -> {lr3:.2e}@{stage3:,} | current={current_lr:.2e}"


def _refinement_summary_values(viewer: object) -> tuple[int, float, float, int, float, int, float, float, float, int]:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is not None:
        training = trainer.training
        current_step = max(int(trainer.state.step), 0)
        return (
            int(trainer.effective_refinement_interval()) if hasattr(trainer, "effective_refinement_interval") else int(training.refinement_interval),
            resolve_refinement_growth_ratio(training, current_step) * 100.0,
            max(float(training.refinement_growth_ratio), 0.0) * 100.0,
            max(int(getattr(training, "refinement_growth_start_step", 0)), 0),
            float(training.refinement_alpha_cull_threshold),
            int(resolve_refinement_min_contribution(training, current_step, len(getattr(trainer, "frames", getattr(viewer.s, "training_frames", ())))),),
            min(max(float(getattr(training, "refinement_min_contribution_decay", 0.995)), 0.0), 1.0) * 100.0,
            min(max(float(getattr(training, "refinement_opacity_mul", 1.0)), 0.0), 1.0),
            max(float(getattr(training, "refinement_clone_scale_mul", 1.0)), 0.0),
            int(training.max_gaussians),
        )
    return (
        max(int(viewer.c("refinement_interval").value), 1),
        0.0,
        max(float(viewer.c("refinement_growth_ratio").value), 0.0) * 100.0,
        max(int(viewer.c("refinement_growth_start_step").value), 0),
        max(float(viewer.c("refinement_alpha_cull_threshold").value), 1e-8),
        max(int(viewer.c("refinement_min_contribution").value), 0),
        min(max(float(viewer.c("refinement_min_contribution_decay").value), 0.0), 1.0) * 100.0,
        min(max(float(viewer.c("refinement_opacity_mul").value), 0.0), 1.0),
        max(float(viewer.c("refinement_clone_scale_mul").value), 0.0),
        max(int(viewer.c("max_gaussians").value), 0),
    )


def _schedule_stage_label(training: object, step: int) -> str:
    if not bool(getattr(training, "lr_schedule_enabled", True)): return "Stage 0"
    stage1, stage2, stage3 = _schedule_steps(training)
    current_step = max(int(step), 0)
    if current_step < stage1: return "Stage 0"
    if current_step < stage2: return "Stage 1"
    if current_step < stage3: return "Stage 2"
    return "Stage 3"


def _active_sh_band_control_key(training: object, step: int) -> str:
    if not bool(getattr(training, "lr_schedule_enabled", True)): return "sh_band"
    stage1, stage2, stage3 = _schedule_steps(training)
    current_step = max(int(step), 0)
    if current_step < stage1: return "sh_band"
    if current_step < stage2: return "sh_band_stage1"
    if current_step < stage3: return "sh_band_stage2"
    return "sh_band_stage3"


def _viewport_sh_state(viewer: object) -> tuple[int, str, str]:
    training, step = _schedule_runtime(viewer)
    return int(resolve_sh_band(training, step)), _active_sh_band_control_key(training, step), _schedule_stage_label(training, step)


def _current_schedule_values_text(viewer: object) -> str:
    training, step = _schedule_runtime(viewer)
    return (
        f"Current Values: step={step:,} | { _schedule_stage_label(training, step) } | "
        f"lr={resolve_base_learning_rate(training, step):.2e} | "
        f"pos={resolve_position_lr_mul(training, step):.2f}x | "
        f"shlr={resolve_sh_lr_mul(training, step):.2f}x | "
        f"cspace={resolve_colorspace_mod(training, step):.3g} | "
        f"dither={resolve_sorting_order_dithering(training, step):.3g} | "
        f"noise={resolve_position_random_step_noise_lr(training, step):.2e} | "
        f"sh=SH{resolve_sh_band(training, step)}"
    )


def _training_schedule_text(viewer: object) -> str:
    training, _ = _schedule_runtime(viewer)
    if not bool(training.lr_schedule_enabled):
        return "LR Schedule: disabled"
    trainer = getattr(viewer.s, "trainer", None)
    current = float(trainer.current_base_lr()) if trainer is not None and hasattr(trainer, "current_base_lr") else float(training.lr_schedule_start_lr)
    return _schedule_summary_text(training, current)


def _training_refinement_text(viewer: object) -> str:
    interval, current_growth, target_growth, start_step, alpha_threshold, contribution_cull, decay, alpha_mul, clone_scale_mul, max_gaussians = _refinement_summary_values(viewer)
    return (
        f"Refinement: every {interval:,} | growth={current_growth:.2f}% now | target={target_growth:.2f}% after {start_step:,} | "
        f"alpha<{alpha_threshold:.2e} or min contrib<{contribution_cull:,} | decay={decay:.2f}%/pass | alpha mul={alpha_mul:.2f}x | clone scale={clone_scale_mul:.2f}x | max={max_gaussians:,}"
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"


def _debug_psnr_text(debug_metrics: dict[str, np.ndarray], frame_idx: int) -> str:
    if frame_idx < debug_metrics["psnr"].size and np.isfinite(float(debug_metrics["psnr"][frame_idx])):
        return f"PSNR: {float(debug_metrics['psnr'][frame_idx]):.2f} dB"
    if frame_idx < debug_metrics["psnr"].size and float(debug_metrics["psnr"][frame_idx]) == float("inf"):
        return "PSNR: inf"
    return "PSNR: n/a"


def _scene_path_text(viewer: object) -> str:
    if viewer.s.scene_path is not None:
        return f"Scene: {viewer.s.scene_path.name} [PLY]"
    if viewer.s.colmap_root is not None:
        return f"Scene: {viewer.s.colmap_root.name} [COLMAP]"
    return "Scene: <none>"


def _colmap_import_status_text(import_progress: object) -> str:
    if import_progress is None:
        return ""
    if import_progress.phase == "prepare":
        return "Preparing COLMAP import..."
    if import_progress.phase == "scan_frames":
        return f"Scanning image metadata: {import_progress.current}/{import_progress.total}"
    if import_progress.phase == "load_textures":
        return f"Loading images: {import_progress.current}/{import_progress.total}"
    return "Finalizing import..."


def _render_stats_text(stats: object) -> str:
    if not stats:
        return "Generated: 0 | Written: 0"
    return (
        f"Generated: {int(stats['generated_entries']):,} | Written: {int(stats['written_entries']):,} | "
        f"Overflow: {bool(stats['overflow'])}"
        f"{' [cap]' if bool(stats.get('capacity_limited', False)) else ''}"
        f"{' (delayed)' if bool(stats.get('stats_latency_frames', 0)) else ''}"
        f"{'' if bool(stats.get('stats_valid', True)) else ' [warming]'}"
    )


def _training_status_texts(viewer: object, current_splat_count: int, training_elapsed_s: float) -> dict[str, str]:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is None:
        return {
            "training": "Training: not initialized",
            "training_time": "Time: n/a",
            "training_iters_avg": "Avg it/s: n/a",
            "training_loss": "Loss Avg: n/a",
            "training_mse": "MSE Avg: n/a",
            "training_density": "Density Avg: n/a",
            "training_psnr": "PSNR Avg: n/a",
            "training_instability": "",
        }
    state = trainer.state
    batch_steps = int(getattr(viewer.s, "last_training_batch_steps", 0))
    batch_text = f" | batch={batch_steps}" if viewer.s.training_active else ""
    avg_iters_s = float(state.step) / training_elapsed_s if training_elapsed_s > 1e-6 else 0.0
    return {
        "training": f"Training: {'running' if viewer.s.training_active else 'paused'} | step={state.step:,} | frame={state.last_frame_index} | splats={int(current_splat_count):,}{batch_text}",
        "training_time": f"Time: {_format_duration(training_elapsed_s)}",
        "training_iters_avg": f"Avg it/s: {avg_iters_s:.2f}" if training_elapsed_s > 1e-6 else "Avg it/s: n/a",
        "training_loss": f"Loss Avg: {state.avg_loss:.6e}",
        "training_mse": f"MSE Avg: {state.avg_mse:.6e}" if np.isfinite(state.avg_mse) else "MSE Avg: n/a",
        "training_density": f"Density Avg: {state.avg_density_loss:.6e}" if np.isfinite(state.avg_density_loss) else "Density Avg: n/a",
        "training_psnr": f"PSNR Avg: {state.avg_psnr:.3f} dB" if np.isfinite(state.avg_psnr) else "PSNR Avg: inf" if state.avg_psnr == float("inf") else "PSNR Avg: n/a",
        "training_instability": state.last_instability,
    }


def _ui_header_state(viewer: object, debug_metrics: dict[str, np.ndarray], frame_idx: int, debug_idx: int) -> dict[str, object]:
    import_progress = getattr(viewer.s, "colmap_import_progress", None)
    current_splat_count = viewer.s.trainer.scene.count if viewer.s.trainer is not None else (viewer.s.scene.count if viewer.s.scene is not None else 0)
    return {
        "loss_debug_view": f"View: {viewer.loss_debug_view_options[debug_idx][1]}",
        "loss_debug_frame": f"Frame[{frame_idx}]: {Path(viewer.s.training_frames[frame_idx].image_path).name}" if viewer.s.training_frames else "Frame: <none>",
        "loss_debug_psnr": _debug_psnr_text(debug_metrics, frame_idx),
        "path": _scene_path_text(viewer),
        "colmap_import_active": bool(import_progress is not None),
        "colmap_import_fraction": 0.0 if import_progress is None else float(import_progress.fraction),
        "can_export_ply": bool(viewer.s.trainer is not None or hasattr(viewer.s.scene, "positions")),
        "colmap_import_status": _colmap_import_status_text(import_progress),
        "colmap_import_current": "" if import_progress is None else import_progress.current_name,
        "scene_stats": f"Splats: {int(current_splat_count):,}",
        "current_splat_count": int(current_splat_count),
    }


def _viewer_panel_state(viewer: object) -> dict[str, object]:
    _, viewport_sh_control_key, viewport_sh_stage_label = _viewport_sh_state(viewer)
    return {
        "training_resolution": _training_resolution_text(viewer),
        "training_downscale": _training_downscale_text(viewer),
        "training_schedule": _training_schedule_text(viewer),
        "training_schedule_values": _current_schedule_values_text(viewer),
        "training_refinement": _training_refinement_text(viewer),
        "viewport_sh_control_key": str(viewport_sh_control_key),
        "viewport_sh_stage_label": str(viewport_sh_stage_label),
        "histogram_status": str(getattr(viewer.s, "cached_raster_grad_histogram_status", "")),
        "histogram_payload": getattr(viewer.s, "cached_raster_grad_histograms", None),
        "histogram_range_payload": getattr(viewer.s, "cached_raster_grad_ranges", None),
        "training_views_rows": _training_view_rows(viewer),
        "training_view_overlay_segments": _camera_overlay_segments(viewer),
    }


def _debug_frame_idx(viewer: object) -> int:
    return clamp_index(int(viewer.c("loss_debug_frame").value), len(viewer.s.training_frames))


def _debug_view_key(viewer: object) -> str:
    return viewer.loss_debug_view_options[clamp_index(int(viewer.c("loss_debug_view").value), len(viewer.loss_debug_view_options))][0]


def _training_steps_per_frame(viewer: object) -> int:
    try:
        value = int(viewer.c("training_steps_per_frame").value)
    except Exception:
        return _DEFAULT_TRAINING_STEPS_PER_FRAME
    return max(1, min(value, _MAX_TRAINING_STEPS_PER_FRAME))


def _training_camera_debug_active(viewer: object) -> bool:
    try:
        return bool(viewer.ui._values.get("show_training_cameras", False))
    except Exception:
        return False


def _viewport_target_size(viewer: object, fallback_width: int, fallback_height: int) -> tuple[int, int]:
    toolkit = getattr(viewer, "toolkit", None)
    viewport_size = None if toolkit is None else getattr(toolkit, "viewport_size", None)
    if callable(viewport_size):
        width, height = viewport_size()
        if int(width) > 0 and int(height) > 0:
            return int(width), int(height)
    return max(int(fallback_width), 1), max(int(fallback_height), 1)


def _coerce_metric_array(values: object, count: int, fill: float = float("nan")) -> np.ndarray:
    array = np.full((max(int(count), 0),), fill, dtype=np.float64)
    source = np.asarray(values, dtype=np.float64).reshape(-1) if values is not None else np.zeros((0,), dtype=np.float64)
    limit = min(array.size, source.size)
    if limit > 0:
        array[:limit] = source[:limit]
    return array


def _frame_metrics_snapshot(viewer: object, frame_count: int) -> dict[str, np.ndarray]:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is None or not hasattr(trainer, "frame_metrics_snapshot"):
        return {
            "loss": np.full((frame_count,), np.nan, dtype=np.float64),
            "mse": np.full((frame_count,), np.nan, dtype=np.float64),
            "psnr": np.full((frame_count,), np.nan, dtype=np.float64),
            "visited": np.zeros((frame_count,), dtype=bool),
        }
    try:
        snapshot = trainer.frame_metrics_snapshot()
    except Exception:
        return {
            "loss": np.full((frame_count,), np.nan, dtype=np.float64),
            "mse": np.full((frame_count,), np.nan, dtype=np.float64),
            "psnr": np.full((frame_count,), np.nan, dtype=np.float64),
            "visited": np.zeros((frame_count,), dtype=bool),
        }
    visited = np.zeros((frame_count,), dtype=bool)
    source_visited = np.asarray(snapshot.get("visited", visited), dtype=bool).reshape(-1)
    visited[: min(frame_count, source_visited.size)] = source_visited[: min(frame_count, source_visited.size)]
    return {
        "loss": _coerce_metric_array(snapshot.get("loss"), frame_count),
        "mse": _coerce_metric_array(snapshot.get("mse"), frame_count),
        "psnr": _coerce_metric_array(snapshot.get("psnr"), frame_count),
        "visited": visited,
    }


def _training_view_rows(viewer: object) -> tuple[dict[str, object], ...]:
    frames = tuple(getattr(viewer.s, "training_frames", ()))
    if len(frames) == 0:
        return ()
    metrics = _frame_metrics_snapshot(viewer, len(frames))
    trainer = getattr(viewer.s, "trainer", None)
    training = getattr(trainer, "training", None)
    camera_min_dist = float(getattr(training, "camera_min_dist", float("nan")))
    last_frame_index = int(getattr(getattr(trainer, "state", None), "last_frame_index", -1))
    rows: list[dict[str, object]] = []
    for frame_index, frame in enumerate(frames):
        rows.append(
            {
                "frame_index": int(frame_index),
                "image_name": Path(getattr(frame, "image_path", f"frame_{frame_index}")).name,
                "resolution": f"{max(int(getattr(frame, 'width', 0)), 0)}x{max(int(getattr(frame, 'height', 0)), 0)}",
                "fx": float(getattr(frame, "fx", float("nan"))),
                "fy": float(getattr(frame, "fy", float("nan"))),
                "cx": float(getattr(frame, "cx", float("nan"))),
                "cy": float(getattr(frame, "cy", float("nan"))),
                "camera_min_dist": camera_min_dist,
                "loss": float(metrics["loss"][frame_index]) if frame_index < metrics["loss"].size else float("nan"),
                "psnr": float(metrics["psnr"][frame_index]) if frame_index < metrics["psnr"].size else float("nan"),
                "visited": bool(metrics["visited"][frame_index]) if frame_index < metrics["visited"].size else False,
                "is_last": int(frame_index) == last_frame_index,
            }
        )
    return tuple(rows)


def _camera_overlay_scale(viewer: object) -> float:
    trainer = getattr(viewer.s, "trainer", None)
    frames = tuple(getattr(viewer.s, "training_frames", ()))
    if trainer is None or len(frames) == 0 or not hasattr(trainer, "make_frame_camera"):
        return _CAMERA_OVERLAY_MIN_LENGTH
    positions: list[np.ndarray] = []
    for frame_index, frame in enumerate(frames):
        try:
            camera = trainer.make_frame_camera(frame_index, int(getattr(frame, "width", 1)), int(getattr(frame, "height", 1)))
        except Exception:
            continue
        position = getattr(camera, "position", None)
        if position is None:
            continue
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        if np.all(np.isfinite(pos)):
            positions.append(pos)
    if len(positions) == 0:
        return _CAMERA_OVERLAY_MIN_LENGTH
    centers = np.stack(positions, axis=0)
    centroid = np.mean(centers, axis=0, dtype=np.float32)
    radius = float(np.median(np.linalg.norm(centers - centroid[None, :], axis=1)))
    return max(radius * _CAMERA_OVERLAY_LENGTH_FRACTION, _CAMERA_OVERLAY_MIN_LENGTH)


def _camera_overlay_signature(viewer: object) -> tuple[object, ...]:
    frames = tuple(getattr(viewer.s, "training_frames", ()))
    trainer = getattr(viewer.s, "trainer", None)
    return (id(trainer), tuple(id(frame) for frame in frames), round(float(getattr(getattr(trainer, "training", None), "camera_min_dist", 0.0)), 8))


def _build_camera_overlay_geometry(viewer: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = tuple(getattr(viewer.s, "training_frames", ()))
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is None or len(frames) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=np.float32)
    overlay_far = _camera_overlay_scale(viewer)
    overlay_near = overlay_far * _CAMERA_OVERLAY_NEAR_FRACTION
    cameras: list[np.ndarray] = []
    frame_indices: list[int] = []
    camera_positions: list[np.ndarray] = []
    for frame_index, frame in enumerate(frames):
        try:
            camera = trainer.make_frame_camera(frame_index, int(getattr(frame, "width", 1)), int(getattr(frame, "height", 1)))
        except Exception:
            continue
        if not hasattr(camera, "basis") or not hasattr(camera, "focal_pixels_xy"):
            continue
        try:
            right, up, forward = (np.asarray(axis, dtype=np.float32).reshape(3) for axis in camera.basis())
            position = np.asarray(getattr(camera, "position"), dtype=np.float32).reshape(3)
            fx, fy = camera.focal_pixels_xy(int(getattr(frame, "width", 1)), int(getattr(frame, "height", 1)))
        except Exception:
            continue
        if not (np.all(np.isfinite(position)) and np.all(np.isfinite(right)) and np.all(np.isfinite(up)) and np.all(np.isfinite(forward))):
            continue
        if not (np.isfinite(fx) and np.isfinite(fy) and fx > 1e-8 and fy > 1e-8):
            continue
        frame_width = float(max(int(getattr(frame, "width", 1)), 1))
        frame_height = float(max(int(getattr(frame, "height", 1)), 1))
        near_half = np.array([0.5 * frame_width * overlay_near / float(fx), 0.5 * frame_height * overlay_near / float(fy)], dtype=np.float32)
        far_half = np.array([0.5 * frame_width * overlay_far / float(fx), 0.5 * frame_height * overlay_far / float(fy)], dtype=np.float32)
        near_center = position + forward * np.float32(overlay_near)
        far_center = position + forward * np.float32(overlay_far)
        corners = np.stack(
            (
                near_center - right * near_half[0] - up * near_half[1],
                near_center + right * near_half[0] - up * near_half[1],
                near_center + right * near_half[0] + up * near_half[1],
                near_center - right * near_half[0] + up * near_half[1],
                far_center - right * far_half[0] - up * far_half[1],
                far_center + right * far_half[0] - up * far_half[1],
                far_center + right * far_half[0] + up * far_half[1],
                far_center - right * far_half[0] + up * far_half[1],
            ),
            axis=0,
        ).astype(np.float32, copy=False)
        cameras.append(corners)
        frame_indices.append(int(frame_index))
        camera_positions.append(position.astype(np.float32, copy=False))
    if len(cameras) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=np.float32)
    return np.stack(cameras, axis=0), np.asarray(frame_indices, dtype=np.int32), np.stack(camera_positions, axis=0)


def _camera_overlay_geometry(viewer: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signature = _camera_overlay_signature(viewer)
    cached_signature = getattr(viewer.s, "camera_overlay_signature", None)
    cached_segments = getattr(viewer.s, "camera_overlay_world_segments", None)
    cached_indices = getattr(viewer.s, "camera_overlay_frame_indices", None)
    cached_positions = getattr(viewer.s, "camera_overlay_world_positions", None)
    if cached_signature == signature and cached_segments is not None and cached_indices is not None and cached_positions is not None:
        return cached_segments, cached_indices, cached_positions
    world_segments, frame_indices, camera_positions = _build_camera_overlay_geometry(viewer)
    viewer.s.camera_overlay_signature = signature
    viewer.s.camera_overlay_world_segments = world_segments
    viewer.s.camera_overlay_frame_indices = frame_indices
    viewer.s.camera_overlay_world_positions = camera_positions
    return world_segments, frame_indices, camera_positions


def _camera_min_dist_rings(position: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray], radius: float) -> tuple[np.ndarray, ...]:
    if not (np.isfinite(radius) and radius > 1e-8):
        return ()
    angles = np.linspace(0.0, 2.0 * np.pi, _CAMERA_MIN_DIST_RING_SAMPLES, endpoint=False, dtype=np.float32)
    cos_theta = np.cos(angles)[:, None].astype(np.float32, copy=False)
    sin_theta = np.sin(angles)[:, None].astype(np.float32, copy=False)
    right, up, forward = (np.asarray(axis, dtype=np.float32).reshape(3) for axis in basis)
    origin = np.asarray(position, dtype=np.float32).reshape(1, 3)
    radius32 = np.float32(radius)
    return (
        origin + radius32 * (cos_theta * right[None, :] + sin_theta * up[None, :]),
        origin + radius32 * (cos_theta * up[None, :] + sin_theta * forward[None, :]),
        origin + radius32 * (cos_theta * forward[None, :] + sin_theta * right[None, :]),
    )


def _project_overlay_points(viewer_camera: object, points_world: np.ndarray, viewport_width: int, viewport_height: int) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(viewer_camera, "basis") or not hasattr(viewer_camera, "focal_pixels_xy") or not hasattr(viewer_camera, "principal_point"):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    try:
        basis = np.stack(tuple(np.asarray(axis, dtype=np.float32).reshape(3) for axis in viewer_camera.basis()), axis=0)
        position = np.asarray(getattr(viewer_camera, "position"), dtype=np.float32).reshape(3)
        fx, fy = viewer_camera.focal_pixels_xy(int(viewport_width), int(viewport_height))
        cx, cy = viewer_camera.principal_point(int(viewport_width), int(viewport_height))
        k1, k2 = viewer_camera.distortion_coeffs() if hasattr(viewer_camera, "distortion_coeffs") else (0.0, 0.0)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    rel = points_world.astype(np.float32, copy=False) - position[None, :]
    camera_points = rel @ basis.T
    depth = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & np.isfinite(depth) & (depth > 1e-12)
    screen = np.zeros((points_world.shape[0], 2), dtype=np.float32)
    if not np.any(valid):
        return screen, valid
    uv = camera_points[valid, :2] / depth[valid, None]
    if abs(float(k1)) > 1e-12 or abs(float(k2)) > 1e-12:
        r2 = np.sum(uv * uv, axis=1)
        uv *= (1.0 + float(k1) * r2 + float(k2) * r2 * r2)[:, None]
    screen_valid = uv * np.asarray((float(fx), float(fy)), dtype=np.float32)[None, :] + np.asarray((float(cx), float(cy)), dtype=np.float32)[None, :]
    valid_indices = np.flatnonzero(valid)
    screen[valid_indices] = screen_valid
    valid[valid_indices] &= np.isfinite(screen_valid).all(axis=1)
    return screen, valid


def _camera_overlay_segments(
    viewer: object,
) -> tuple[
    tuple[
        tuple[tuple[float, float], ...],
        tuple[tuple[float, float], ...],
        tuple[tuple[float, float, float, float], ...],
        tuple[tuple[tuple[float, float], ...], ...],
        tuple[float, float],
        str,
        tuple[float, float, float, float],
        float,
    ],
    ...,
]:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is None or _training_camera_debug_active(viewer):
        return ()
    viewport_camera = viewer.camera()
    if not hasattr(viewport_camera, "basis"):
        return ()
    fallback_width = int(getattr(getattr(viewer.s, "renderer", None), "width", 1))
    fallback_height = int(getattr(getattr(viewer.s, "renderer", None), "height", 1))
    viewport_width, viewport_height = _viewport_target_size(viewer, fallback_width, fallback_height)
    world_corners, frame_indices, camera_positions = _camera_overlay_geometry(viewer)
    if world_corners.size == 0:
        return ()
    last_frame_index = int(getattr(getattr(trainer, "state", None), "last_frame_index", -1))
    metrics = _frame_metrics_snapshot(viewer, len(getattr(viewer.s, "training_frames", ())))
    points_world = world_corners.reshape(-1, 3)
    screen_points, valid_points = _project_overlay_points(viewport_camera, points_world, viewport_width, viewport_height)
    if screen_points.size == 0:
        return ()
    screen_corners = screen_points.reshape(-1, 8, 2)
    valid_corners = valid_points.reshape(-1, 8).all(axis=1)
    if not np.any(valid_corners):
        return ()
    active_cameras = frame_indices == last_frame_index
    min_dist = max(float(getattr(getattr(trainer, "training", None), "camera_min_dist", 0.0)), 0.0)
    show_min_dist_spheres = bool(getattr(getattr(viewer, "ui", None), "_values", {}).get("show_camera_min_dist_spheres", True))
    sphere_rings_by_camera: list[tuple[tuple[float, float], ...]] = []
    if show_min_dist_spheres and min_dist > 1e-8:
        for frame_index, position in enumerate(camera_positions):
            frame = viewer.s.training_frames[int(frame_index)]
            try:
                frame_camera = trainer.make_frame_camera(int(frame_index), int(getattr(frame, "width", 1)), int(getattr(frame, "height", 1)))
            except Exception:
                sphere_rings_by_camera.append(())
                continue
            rings: list[tuple[tuple[float, float], ...]] = []
            for ring_world in _camera_min_dist_rings(position, frame_camera.basis(), min_dist):
                ring_screen, ring_valid = _project_overlay_points(viewport_camera, ring_world, viewport_width, viewport_height)
                if ring_screen.size == 0 or not bool(np.all(ring_valid)):
                    continue
                rings.append(tuple((float(point[0]), float(point[1])) for point in ring_screen))
            sphere_rings_by_camera.append(tuple(rings))
    else:
        sphere_rings_by_camera = [()] * int(frame_indices.shape[0])
    return tuple(
        (
            tuple((float(point[0]), float(point[1])) for point in corners[:4]),
            tuple((float(point[0]), float(point[1])) for point in corners[4:]),
            tuple(
                (
                    float(corners[i0, 0]),
                    float(corners[i0, 1]),
                    float(corners[i1, 0]),
                    float(corners[i1, 1]),
                )
                for i0, i1 in ((0, 4), (1, 5), (2, 6), (3, 7))
            ),
            sphere_rings,
            (float(corners[6, 0]), float(corners[6, 1])),
            (
                f"{Path(getattr(viewer.s.training_frames[int(frame_index)], 'image_path', f'frame_{int(frame_index)}')).name}"
                f" | {float(metrics['psnr'][int(frame_index)]):.2f} dB"
                if int(frame_index) < metrics["psnr"].size and np.isfinite(float(metrics["psnr"][int(frame_index)]))
                else Path(getattr(viewer.s.training_frames[int(frame_index)], "image_path", f"frame_{int(frame_index)}")).name
            ),
            _CAMERA_OVERLAY_ACTIVE_COLOR if bool(is_active) else _CAMERA_OVERLAY_COLOR,
            2.0 if bool(is_active) else 1.25,
        )
        for corners, frame_index, is_valid, is_active, sphere_rings in zip(screen_corners, frame_indices, valid_corners, active_cameras, sphere_rings_by_camera, strict=False)
        if bool(is_valid)
    )


def _run_training_batch(viewer: object) -> int:
    if not viewer.s.training_active or viewer.s.trainer is None:
        viewer.s.training_runtime_factor_changed = False
        viewer.s.last_training_batch_steps = 0
        return 0
    factor_before = int(viewer.s.trainer.effective_train_render_factor()) if hasattr(viewer.s.trainer, "effective_train_render_factor") else int(viewer.s.trainer.effective_train_downscale_factor())
    steps = _training_steps_per_frame(viewer)
    if hasattr(viewer.s.trainer, "step_batch"):
        steps = int(viewer.s.trainer.step_batch(steps))
    else:
        for _ in range(steps):
            viewer.s.trainer.step()
    factor_after = int(viewer.s.trainer.effective_train_render_factor()) if hasattr(viewer.s.trainer, "effective_train_render_factor") else int(viewer.s.trainer.effective_train_downscale_factor())
    viewer.s.training_runtime_factor_changed = factor_after != factor_before
    viewer.s.last_training_batch_steps = steps
    return steps


def _preview_train_downscale_factor(viewer: object) -> int:
    mode = int(viewer.c("train_downscale_mode").value)
    return max(int(viewer.c("train_auto_start_downscale").value), 1) if mode == _TRAIN_DOWNSCALE_MODE_AUTO else max(mode, 1)


def _preview_train_subsample_factor(viewer: object) -> int:
    try:
        mode = int(viewer.c("train_subsample_factor").value)
    except Exception:
        return 1
    if mode != 0:
        return max(min(mode, TRAIN_SUBSAMPLE_MAX_FACTOR), 1)
    if not getattr(viewer.s, "training_frames", None):
        return 1
    native_width = max(int(viewer.s.training_frames[0].width), 1)
    native_height = max(int(viewer.s.training_frames[0].height), 1)
    return resolve_auto_train_subsample_factor(native_width, native_height, _preview_train_downscale_factor(viewer))


def _preview_train_render_factor(viewer: object) -> int:
    return _preview_train_downscale_factor(viewer) * _preview_train_subsample_factor(viewer)


def _training_resolution_text(viewer: object) -> str:
    if viewer.s.training_renderer is not None and viewer.s.trainer is not None:
        factor = max(int(viewer.s.trainer.effective_train_render_factor()) if hasattr(viewer.s.trainer, "effective_train_render_factor") else int(viewer.s.trainer.effective_train_downscale_factor()), 1)
        return f"Train Res: {int(viewer.s.training_renderer.width)}x{int(viewer.s.training_renderer.height)} (N={factor})"
    if viewer.s.training_frames:
        factor = _preview_train_render_factor(viewer)
        native_width = max(int(viewer.s.training_frames[0].width), 1)
        native_height = max(int(viewer.s.training_frames[0].height), 1)
        width = (native_width + factor - 1) // factor
        height = (native_height + factor - 1) // factor
        return f"Train Res: {width}x{height} (N={factor})"
    return "Train Res: n/a"


def _training_downscale_text(viewer: object) -> str:
    if viewer.s.trainer is not None:
        training = viewer.s.trainer.training
        current = int(viewer.s.trainer.effective_train_downscale_factor())
        subsample = int(viewer.s.trainer.effective_train_subsample_factor()) if hasattr(viewer.s.trainer, "effective_train_subsample_factor") else max(int(getattr(training, "train_subsample_factor", 1)), 1)
        combined = int(viewer.s.trainer.effective_train_render_factor()) if hasattr(viewer.s.trainer, "effective_train_render_factor") else current
        auto = int(getattr(training, "train_subsample_factor", 1)) == 0
        subsample_text = f"Auto (1/{subsample})" if auto and subsample > 1 else "Auto (Off)" if auto else "Off" if subsample <= 1 else f"1/{subsample}"
        if int(training.train_downscale_mode) == _TRAIN_DOWNSCALE_MODE_AUTO:
            return (
                f"Downscale: Auto | start={int(training.train_auto_start_downscale)}x | "
                f"current={current}x | subsampling={subsample_text} | effective={combined}x | step {int(viewer.s.trainer.state.step)}/{int(training.train_downscale_max_iters)}"
            )
        return f"Downscale: Manual {current}x | subsampling={subsample_text} | effective={combined}x"
    mode = int(viewer.c("train_downscale_mode").value)
    subsample = _preview_train_subsample_factor(viewer)
    combined = _preview_train_render_factor(viewer)
    auto = int(viewer.c("train_subsample_factor").value) == 0
    subsample_text = f"Auto (1/{subsample})" if auto and subsample > 1 else "Auto (Off)" if auto else "Off" if subsample <= 1 else f"1/{subsample}"
    if mode == _TRAIN_DOWNSCALE_MODE_AUTO:
        return (
            f"Downscale: Auto | start={max(int(viewer.c('train_auto_start_downscale').value), 1)}x | "
            f"current={_preview_train_downscale_factor(viewer)}x | subsampling={subsample_text} | effective={combined}x | step 0/{max(int(viewer.c('train_downscale_max_iters').value), 1)}"
        )
    return f"Downscale: Manual {max(mode, 1)}x | subsampling={subsample_text} | effective={combined}x"
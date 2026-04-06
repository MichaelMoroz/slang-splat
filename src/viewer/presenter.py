from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import slangpy as spy

from ..common import clamp_index, debug_region, require_not_none
from ..training import resolve_auto_train_subsample_factor, resolve_base_learning_rate, resolve_depth_ratio_weight, resolve_position_lr_mul, resolve_position_random_step_noise_lr, resolve_refinement_growth_ratio, resolve_refinement_min_contribution_percent, resolve_sh_lr_mul, resolve_use_sh
from . import session

_DEBUG_HUGE_VALUE = 1e8
_DEBUG_TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_destination
_DEBUG_ABS_DIFF_SCALE_DEFAULT = 1.0
_DEBUG_ABS_DIFF_SCALE_MIN = 0.125
_DEBUG_ABS_DIFF_SCALE_MAX = 64.0
_DEFAULT_TRAINING_STEPS_PER_FRAME = 3
_MAX_TRAINING_STEPS_PER_FRAME = 8
_TRAIN_DOWNSCALE_MODE_AUTO = 0
_VIEWER_CLEAR_COLOR = [0.08, 0.09, 0.11, 1.0]
_CAMERA_OVERLAY_LENGTH_FRACTION = 0.08
_CAMERA_OVERLAY_MIN_LENGTH = 0.05
_CAMERA_OVERLAY_NEAR_FRACTION = 0.35
_CAMERA_OVERLAY_COLOR = (0.18, 0.70, 0.98, 0.72)
_CAMERA_OVERLAY_ACTIVE_COLOR = (1.00, 0.78, 0.18, 0.96)
_CAMERA_OVERLAY_EDGE_INDICES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _schedule_state_from_controls(viewer: object) -> object:
    return SimpleNamespace(
        lr_schedule_enabled=bool(viewer.c("lr_schedule_enabled").value),
        lr_schedule_start_lr=float(viewer.c("lr_schedule_start_lr").value),
        lr_schedule_stage1_lr=float(viewer.c("lr_schedule_stage1_lr").value),
        lr_schedule_stage2_lr=float(viewer.c("lr_schedule_stage2_lr").value),
        lr_schedule_end_lr=float(viewer.c("lr_schedule_end_lr").value),
        lr_schedule_steps=int(viewer.c("lr_schedule_steps").value),
        lr_schedule_stage1_step=int(viewer.c("lr_schedule_stage1_step").value),
        lr_schedule_stage2_step=int(viewer.c("lr_schedule_stage2_step").value),
        lr_pos_mul=float(viewer.c("lr_pos_mul").value),
        lr_pos_stage1_mul=float(viewer.c("lr_pos_stage1_mul").value),
        lr_pos_stage2_mul=float(viewer.c("lr_pos_stage2_mul").value),
        lr_pos_stage3_mul=float(viewer.c("lr_pos_stage3_mul").value),
        lr_sh_mul=float(viewer.c("lr_sh_mul").value),
        lr_sh_stage1_mul=float(viewer.c("lr_sh_stage1_mul").value),
        lr_sh_stage2_mul=float(viewer.c("lr_sh_stage2_mul").value),
        lr_sh_stage3_mul=float(viewer.c("lr_sh_stage3_mul").value),
        depth_ratio_weight=float(viewer.c("depth_ratio_weight").value),
        depth_ratio_stage1_weight=float(viewer.c("depth_ratio_stage1_weight").value),
        depth_ratio_stage2_weight=float(viewer.c("depth_ratio_stage2_weight").value),
        depth_ratio_stage3_weight=float(viewer.c("depth_ratio_stage3_weight").value),
        position_random_step_noise_lr=float(viewer.c("position_random_step_noise_lr").value),
        position_random_step_noise_stage1_lr=float(viewer.c("position_random_step_noise_stage1_lr").value),
        position_random_step_noise_stage2_lr=float(viewer.c("position_random_step_noise_stage2_lr").value),
        position_random_step_noise_stage3_lr=float(viewer.c("position_random_step_noise_stage3_lr").value),
        use_sh=bool(viewer.c("use_sh").value),
        use_sh_stage1=bool(viewer.c("use_sh_stage1").value),
        use_sh_stage2=bool(viewer.c("use_sh_stage2").value),
        use_sh_stage3=bool(viewer.c("use_sh_stage3").value),
    )


def _schedule_stage_label(training: object, step: int) -> str:
    if not bool(getattr(training, "lr_schedule_enabled", True)): return "Stage 0"
    stage1 = max(int(getattr(training, "lr_schedule_stage1_step", 0)), 0)
    stage2 = max(int(getattr(training, "lr_schedule_stage2_step", stage1)), stage1)
    current_step = max(int(step), 0)
    if current_step < stage1: return "Stage 0"
    if current_step < stage2: return "Stage 1"
    if current_step < max(int(getattr(training, "lr_schedule_steps", stage2)), stage2): return "Stage 2"
    return "Stage 3"


def _current_schedule_values_text(viewer: object) -> str:
    if viewer.s.trainer is not None:
        training = viewer.s.trainer.training
        step = max(int(viewer.s.trainer.state.step), 0)
    else:
        training = _schedule_state_from_controls(viewer)
        step = 0
    return (
        f"Current Values: step={step:,} | { _schedule_stage_label(training, step) } | "
        f"lr={resolve_base_learning_rate(training, step):.2e} | "
        f"pos={resolve_position_lr_mul(training, step):.2f}x | "
        f"shlr={resolve_sh_lr_mul(training, step):.2f}x | "
        f"depth={resolve_depth_ratio_weight(training, step):.2e} | "
        f"noise={resolve_position_random_step_noise_lr(training, step):.2e} | "
        f"sh={'on' if resolve_use_sh(training, step) else 'off'}"
    )


def _training_schedule_text(viewer: object) -> str:
    if viewer.s.trainer is not None:
        training = viewer.s.trainer.training
        if not bool(training.lr_schedule_enabled):
            return "LR Schedule: disabled"
        current = float(viewer.s.trainer.current_base_lr()) if hasattr(viewer.s.trainer, "current_base_lr") else float(training.lr_schedule_start_lr)
        lr0 = max(float(training.lr_schedule_start_lr), 1e-8)
        lr1 = max(float(training.lr_schedule_stage1_lr), 1e-8)
        lr2 = max(float(training.lr_schedule_stage2_lr), 1e-8)
        lr3 = max(float(training.lr_schedule_end_lr), 1e-8)
        stage1 = max(int(training.lr_schedule_stage1_step), 0)
        stage2 = max(int(training.lr_schedule_stage2_step), stage1)
        max_step = max(int(training.lr_schedule_steps), stage2, 1)
        return (
            f"LR Schedule: {lr0:.2e}@0 -> {lr1:.2e}@{stage1:,} -> {lr2:.2e}@{stage2:,} -> {lr3:.2e}@{max_step:,} | current={current:.2e}"
        )
    if not bool(viewer.c("lr_schedule_enabled").value):
        return "LR Schedule: disabled"
    lr0 = max(float(viewer.c("lr_schedule_start_lr").value), 1e-8)
    lr1 = max(float(viewer.c("lr_schedule_stage1_lr").value), 1e-8)
    lr2 = max(float(viewer.c("lr_schedule_stage2_lr").value), 1e-8)
    lr3 = max(float(viewer.c("lr_schedule_end_lr").value), 1e-8)
    stage1 = min(max(int(viewer.c("lr_schedule_stage1_step").value), 0), max(int(viewer.c("lr_schedule_steps").value), 1))
    stage2 = min(max(int(viewer.c("lr_schedule_stage2_step").value), stage1), max(int(viewer.c("lr_schedule_steps").value), 1))
    max_step = max(int(viewer.c("lr_schedule_steps").value), 1)
    return (
        f"LR Schedule: {lr0:.2e}@0 -> {lr1:.2e}@{stage1:,} -> {lr2:.2e}@{stage2:,} -> {lr3:.2e}@{max_step:,} | current={lr0:.2e}"
    )


def _training_refinement_text(viewer: object) -> str:
    if viewer.s.trainer is not None:
        training = viewer.s.trainer.training
        current_step = max(int(viewer.s.trainer.state.step), 0)
        target_growth = max(float(training.refinement_growth_ratio), 0.0)
        current_growth = resolve_refinement_growth_ratio(training, current_step)
        start_step = max(int(getattr(training, "refinement_growth_start_step", 0)), 0)
        interval = int(viewer.s.trainer.effective_refinement_interval()) if hasattr(viewer.s.trainer, "effective_refinement_interval") else int(training.refinement_interval)
        frame_count = len(getattr(viewer.s.trainer, "frames", getattr(viewer.s, "training_frames", ())))
        contribution_cull = resolve_refinement_min_contribution_percent(training, current_step, frame_count)
        decay = min(max(float(getattr(training, "refinement_min_contribution_decay", 0.995)), 0.0), 1.0)
        alpha_mul = min(max(float(getattr(training, "refinement_opacity_mul", 1.0)), 0.0), 1.0)
        return (
            f"Refinement: every {interval:,} | growth={current_growth * 100.0:.2f}% now | target={target_growth * 100.0:.2f}% after {start_step:,} | "
            f"alpha<{float(training.refinement_alpha_cull_threshold):.2e} or min contrib<{contribution_cull:.6g}% | decay={decay * 100.0:.2f}%/pass | alpha mul={alpha_mul:.2f}x | max={int(training.max_gaussians):,}"
        )
    target_growth = max(float(viewer.c("refinement_growth_ratio").value), 0.0)
    start_step = max(int(viewer.c("refinement_growth_start_step").value), 0)
    contribution_cull = max(float(viewer.c("refinement_min_contribution_percent").value), 0.0)
    decay = min(max(float(viewer.c("refinement_min_contribution_decay").value), 0.0), 1.0)
    alpha_mul = min(max(float(viewer.c("refinement_opacity_mul").value), 0.0), 1.0)
    return (
        f"Refinement: every {max(int(viewer.c('refinement_interval').value), 1):,} | growth=0.00% now | target={target_growth * 100.0:.2f}% after {start_step:,} | "
        f"alpha<{max(float(viewer.c('refinement_alpha_cull_threshold').value), 1e-8):.2e} or min contrib<{contribution_cull:.6g}% | decay={decay * 100.0:.2f}%/pass | alpha mul={alpha_mul:.2f}x | max={max(int(viewer.c('max_gaussians').value), 0):,}"
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"


def _debug_frame_idx(viewer: object) -> int:
    return clamp_index(int(viewer.c("loss_debug_frame").value), len(viewer.s.training_frames))


def _debug_view_key(viewer: object) -> str:
    return viewer.loss_debug_view_options[clamp_index(int(viewer.c("loss_debug_view").value), len(viewer.loss_debug_view_options))][0]


def _ensure_texture(viewer: object, attr: str, width: int, height: int) -> spy.Texture:
    texture = getattr(viewer.s, attr)
    if texture is not None and int(texture.width) == int(width) and int(texture.height) == int(height):
        return texture
    created = viewer.device.create_texture(format=spy.Format.rgba32_float, width=int(width), height=int(height), usage=_DEBUG_TEXTURE_USAGE)
    setattr(viewer.s, attr, created)
    return created


def _training_steps_per_frame(viewer: object) -> int:
    try:
        value = int(viewer.c("training_steps_per_frame").value)
    except Exception:
        return _DEFAULT_TRAINING_STEPS_PER_FRAME
    return max(1, min(value, _MAX_TRAINING_STEPS_PER_FRAME))


def _debug_abs_diff_scale(viewer: object) -> float:
    try:
        value = float(viewer.c("loss_debug_abs_scale").value)
    except Exception:
        return _DEBUG_ABS_DIFF_SCALE_DEFAULT
    return max(_DEBUG_ABS_DIFF_SCALE_MIN, min(value, _DEBUG_ABS_DIFF_SCALE_MAX))


def _training_camera_debug_active(viewer: object) -> bool:
    try:
        return int(viewer.c("debug_mode").value) == 0
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
    near_value = float(getattr(training, "near", float("nan")))
    far_value = float(getattr(training, "far", float("nan")))
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
                "near": near_value,
                "far": far_value,
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
    return (id(trainer), tuple(id(frame) for frame in frames))


def _build_camera_overlay_geometry(viewer: object) -> tuple[np.ndarray, np.ndarray]:
    frames = tuple(getattr(viewer.s, "training_frames", ()))
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is None or len(frames) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    overlay_far = _camera_overlay_scale(viewer)
    overlay_near = overlay_far * _CAMERA_OVERLAY_NEAR_FRACTION
    cameras: list[np.ndarray] = []
    frame_indices: list[int] = []
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
    if len(cameras) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.stack(cameras, axis=0), np.asarray(frame_indices, dtype=np.int32)


def _camera_overlay_geometry(viewer: object) -> tuple[np.ndarray, np.ndarray]:
    signature = _camera_overlay_signature(viewer)
    cached_signature = getattr(viewer.s, "camera_overlay_signature", None)
    cached_segments = getattr(viewer.s, "camera_overlay_world_segments", None)
    cached_indices = getattr(viewer.s, "camera_overlay_frame_indices", None)
    if cached_signature == signature and cached_segments is not None and cached_indices is not None:
        return cached_segments, cached_indices
    world_segments, frame_indices = _build_camera_overlay_geometry(viewer)
    viewer.s.camera_overlay_signature = signature
    viewer.s.camera_overlay_world_segments = world_segments
    viewer.s.camera_overlay_frame_indices = frame_indices
    return world_segments, frame_indices


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
    world_corners, frame_indices = _camera_overlay_geometry(viewer)
    if world_corners.size == 0:
        return ()
    last_frame_index = int(getattr(getattr(trainer, "state", None), "last_frame_index", -1))
    points_world = world_corners.reshape(-1, 3)
    screen_points, valid_points = _project_overlay_points(viewport_camera, points_world, viewport_width, viewport_height)
    if screen_points.size == 0:
        return ()
    screen_corners = screen_points.reshape(-1, 8, 2)
    valid_corners = valid_points.reshape(-1, 8).all(axis=1)
    if not np.any(valid_corners):
        return ()
    active_cameras = frame_indices == last_frame_index
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
            _CAMERA_OVERLAY_ACTIVE_COLOR if bool(is_active) else _CAMERA_OVERLAY_COLOR,
            2.0 if bool(is_active) else 1.25,
        )
        for corners, is_valid, is_active in zip(screen_corners, valid_corners, active_cameras, strict=False)
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
        return max(min(mode, 4), 1)
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


def _dispatch_debug_abs_diff(viewer: object, encoder: spy.CommandEncoder, rendered_tex: spy.Texture, target_tex: spy.Texture, width: int, height: int) -> spy.Texture:
    output = _ensure_texture(viewer, "loss_debug_texture", width, height)
    with debug_region(encoder, "Viewer Debug Abs Diff", 150):
        require_not_none(viewer.s.debug_abs_diff_kernel, "Debug abs-diff kernel is not initialized.").dispatch(
            thread_count=spy.uint3(int(width), int(height), 1),
            vars={
                "g_DebugRendered": rendered_tex,
                "g_DebugTarget": target_tex,
                "g_DebugOutput": output,
                "g_DebugWidth": int(width),
                "g_DebugHeight": int(height),
                "g_DebugDiffScale": _debug_abs_diff_scale(viewer),
                "g_HugeValue": _DEBUG_HUGE_VALUE,
            },
            command_encoder=encoder,
        )
    return output


def _dispatch_viewport_present(viewer: object, encoder: spy.CommandEncoder, source_tex: spy.Texture, source_width: int, source_height: int, output_width: int, output_height: int) -> spy.Texture:
    output = _ensure_texture(viewer, "debug_present_texture", output_width, output_height)
    with debug_region(encoder, "Viewer Present", 151):
        require_not_none(viewer.s.debug_letterbox_kernel, "Debug letterbox kernel is not initialized.").dispatch(
            thread_count=spy.uint3(int(output_width), int(output_height), 1),
            vars={
                "g_LetterboxSource": source_tex,
                "g_LetterboxOutput": output,
                "g_LetterboxSourceWidth": int(source_width),
                "g_LetterboxSourceHeight": int(source_height),
                "g_LetterboxOutputWidth": int(output_width),
                "g_LetterboxOutputHeight": int(output_height),
            },
            command_encoder=encoder,
        )
    return output


def update_ui_text(viewer: object, dt: float) -> None:
    viewer.s.fps_smooth += (1.0 / max(dt, 1e-5) - viewer.s.fps_smooth) * min(dt * 5.0, 1.0)
    session.update_debug_frame_slider_range(viewer)
    frame_idx = _debug_frame_idx(viewer)
    debug_idx = clamp_index(int(viewer.c("loss_debug_view").value), len(viewer.loss_debug_view_options))
    stats = viewer.s.stats
    viewer.t("fps").text = f"FPS: {viewer.s.fps_smooth:.1f}"
    viewer.t("loss_debug_view").text = f"View: {viewer.loss_debug_view_options[debug_idx][1]}"
    viewer.t("loss_debug_frame").text = f"Frame[{frame_idx}]: {Path(viewer.s.training_frames[frame_idx].image_path).name}" if viewer.s.training_frames else "Frame: <none>"
    viewer.t("path").text = f"Scene: {viewer.s.scene_path.name} [PLY]" if viewer.s.scene_path is not None else f"Scene: {viewer.s.colmap_root.name} [COLMAP]" if viewer.s.colmap_root is not None else "Scene: <none>"
    import_progress = getattr(viewer.s, "colmap_import_progress", None)
    viewer.ui._values["_colmap_import_active"] = bool(import_progress is not None)
    viewer.ui._values["_colmap_import_fraction"] = 0.0 if import_progress is None else float(import_progress.fraction)
    viewer.ui._values["_can_export_ply"] = bool(viewer.s.trainer is not None or hasattr(viewer.s.scene, "positions"))
    viewer.t("colmap_import_status").text = "" if import_progress is None else (
        "Preparing COLMAP import..."
        if import_progress.phase == "prepare"
        else f"Scanning image metadata: {import_progress.current}/{import_progress.total}"
        if import_progress.phase == "scan_frames"
        else f"Loading images: {import_progress.current}/{import_progress.total}"
        if import_progress.phase == "load_textures"
        else "Finalizing import..."
    )
    viewer.t("colmap_import_current").text = "" if import_progress is None else import_progress.current_name
    current_splat_count = viewer.s.trainer.scene.count if viewer.s.trainer is not None else (viewer.s.scene.count if viewer.s.scene is not None else 0)
    viewer.t("scene_stats").text = f"Splats: {int(current_splat_count):,}"
    viewer.t("training_resolution").text = _training_resolution_text(viewer)
    viewer.t("training_downscale").text = _training_downscale_text(viewer)
    viewer.t("training_schedule").text = _training_schedule_text(viewer)
    viewer.t("training_schedule_values").text = _current_schedule_values_text(viewer)
    viewer.t("training_refinement").text = _training_refinement_text(viewer)
    viewer.t("histogram_status").text = str(getattr(viewer.s, "cached_raster_grad_histogram_status", ""))
    viewer.ui._values["_histogram_payload"] = getattr(viewer.s, "cached_raster_grad_histograms", None)
    viewer.ui._values["_histogram_range_payload"] = getattr(viewer.s, "cached_raster_grad_ranges", None)
    viewer.ui._values["_training_views_rows"] = _training_view_rows(viewer)
    viewer.ui._values["_training_view_overlay_segments"] = _camera_overlay_segments(viewer)
    viewer.t("render_stats").text = "Generated: 0 | Written: 0" if not stats else f"Generated: {int(stats['generated_entries']):,} | Written: {int(stats['written_entries']):,} | Overflow: {bool(stats['overflow'])}{' [cap]' if bool(stats.get('capacity_limited', False)) else ''}{' (delayed)' if bool(stats.get('stats_latency_frames', 0)) else ''}{'' if bool(stats.get('stats_valid', True)) else ' [warming]'}"
    if viewer.s.trainer is None:
        viewer.t("training").text = "Training: not initialized"
        viewer.t("training_time").text = "Time: n/a"
        viewer.t("training_iters_avg").text = "Avg it/s: n/a"
        viewer.t("training_loss").text = "Loss Avg: n/a"
        viewer.t("training_mse").text = "MSE Avg: n/a"
        viewer.t("training_density").text = "Density Avg: n/a"
        viewer.t("training_psnr").text = "PSNR Avg: n/a"
        viewer.t("training_instability").text = ""
    else:
        state = viewer.s.trainer.state
        batch_steps = int(getattr(viewer.s, "last_training_batch_steps", 0))
        batch_text = f" | batch={batch_steps}" if viewer.s.training_active else ""
        training_elapsed_s = float(session.training_elapsed_seconds(viewer, now=viewer.s.last_time))
        avg_iters_s = float(state.step) / training_elapsed_s if training_elapsed_s > 1e-6 else 0.0
        viewer.t("training").text = f"Training: {'running' if viewer.s.training_active else 'paused'} | step={state.step:,} | frame={state.last_frame_index} | splats={int(current_splat_count):,}{batch_text}"
        viewer.t("training_time").text = f"Time: {_format_duration(training_elapsed_s)}"
        viewer.t("training_iters_avg").text = f"Avg it/s: {avg_iters_s:.2f}" if training_elapsed_s > 1e-6 else "Avg it/s: n/a"
        viewer.t("training_loss").text = f"Loss Avg: {state.avg_loss:.6e}"
        viewer.t("training_mse").text = f"MSE Avg: {state.avg_mse:.6e}" if np.isfinite(state.avg_mse) else "MSE Avg: n/a"
        viewer.t("training_density").text = f"Density Avg: {state.avg_density_loss:.6e}" if np.isfinite(state.avg_density_loss) else "Density Avg: n/a"
        viewer.t("training_psnr").text = f"PSNR Avg: {state.avg_psnr:.3f} dB" if np.isfinite(state.avg_psnr) else "PSNR Avg: inf" if state.avg_psnr == float("inf") else "PSNR Avg: n/a"
        viewer.t("training_instability").text = state.last_instability
    viewer.t("error").text = f"Error: {viewer.s.last_error}" if viewer.s.last_error else ""
    _update_toolkit_history(viewer, dt)


def _update_toolkit_history(viewer: object, dt: float) -> None:
    tk = getattr(viewer, "toolkit", None)
    if tk is None or not hasattr(tk, "tk"):
        return
    tk.tk.fps_history.append(viewer.s.fps_smooth)
    viewer.ui._values["_loss_debug_frame_max"] = max(len(viewer.s.training_frames) - 1, 0)
    if viewer.s.trainer is not None:
        state = viewer.s.trainer.state
        step = int(state.step)
        if step > 0 and (not tk.tk.step_history or step != tk.tk.step_history[-1]):
            tk.tk.step_history.append(step)
            tk.tk.step_time_history.append(float(viewer.s.last_time))
            if np.isfinite(state.avg_loss) and state.avg_loss > 0:
                tk.tk.loss_history.append(float(state.avg_loss))
            elif tk.tk.loss_history:
                tk.tk.loss_history.append(tk.tk.loss_history[-1])
            if np.isfinite(state.avg_psnr):
                tk.tk.psnr_history.append(float(state.avg_psnr))
            elif tk.tk.psnr_history:
                tk.tk.psnr_history.append(tk.tk.psnr_history[-1])


def _render_debug_source(viewer: object, encoder: spy.CommandEncoder, frame_idx: int) -> tuple[spy.Texture, dict[str, int | bool | float], int, int]:
    frame = viewer.s.training_frames[frame_idx]
    debug_width, debug_height = max(int(frame.width), 1), max(int(frame.height), 1)
    frame_camera = viewer.s.trainer.make_frame_camera(frame_idx, debug_width, debug_height)
    debug_renderer = session.ensure_renderer(viewer, "debug_renderer", debug_width, debug_height, allow_debug_overlays=True)
    session.sync_scene_from_training_renderer(viewer, debug_renderer, target="debug")
    source_tex, stats = debug_renderer.render_to_texture(frame_camera, background=viewer.s.background, read_stats=True, command_encoder=encoder)
    return source_tex, stats, debug_width, debug_height


def _render_debug_view(viewer: object, encoder: spy.CommandEncoder, output_width: int, output_height: int) -> spy.Texture:
    frame_idx = _debug_frame_idx(viewer)
    debug_render_tex, viewer.s.stats, debug_width, debug_height = _render_debug_source(viewer, encoder, frame_idx)
    target_tex = viewer.s.trainer.get_frame_target_texture(frame_idx, native_resolution=True, encoder=encoder)
    source_tex = debug_render_tex if _debug_view_key(viewer) == "rendered" else target_tex if _debug_view_key(viewer) == "target" else _dispatch_debug_abs_diff(viewer, encoder, debug_render_tex, target_tex, debug_width, debug_height)
    return _dispatch_viewport_present(viewer, encoder, source_tex, debug_width, debug_height, output_width, output_height)




def _render_main_view(viewer: object, encoder: spy.CommandEncoder) -> spy.Texture:
    if viewer.s.trainer is not None and viewer.s.training_renderer is not None:
        session.sync_scene_from_training_renderer(viewer, viewer.s.renderer, target="main")
    out_tex, stats = viewer.s.renderer.render_to_texture(viewer.camera(), background=viewer.s.background, read_stats=True, command_encoder=encoder)
    viewer.s.stats = stats
    return _dispatch_viewport_present(viewer, encoder, out_tex, int(viewer.s.renderer.width), int(viewer.s.renderer.height), int(viewer.s.renderer.width), int(viewer.s.renderer.height))


def render_frame(viewer: object, render_context: spy.AppWindow.RenderContext) -> None:
    image, encoder = render_context.surface_texture, render_context.command_encoder
    now = spy.time.perf_counter() if hasattr(spy, "time") else time.perf_counter()
    dt = max(now - viewer.s.last_time, 1e-5)
    viewer.s.last_time = now
    iw, ih = int(image.width), int(image.height)
    render_width, render_height = _viewport_target_size(viewer, iw, ih)
    try:
        viewer.update_camera(dt)
        runtime_reconfigured = False
        if bool(getattr(viewer.s, "pending_training_reinitialize", False)):
            viewer.s.pending_training_reinitialize = False
            session.initialize_training_scene(viewer)
        session.apply_live_params(viewer)
        session.advance_colmap_import(viewer)
        if bool(getattr(viewer.s, "pending_training_runtime_resize", False)):
            runtime_reconfigured = bool(session.ensure_training_runtime_resolution(viewer))
        if viewer.s.renderer is None:
            session.recreate_renderer(viewer, render_width, render_height)
        elif (viewer.s.renderer.width, viewer.s.renderer.height) != (render_width, render_height):
            session.recreate_renderer(viewer, render_width, render_height)
        encoder.clear_texture_float(image, clear_value=_VIEWER_CLEAR_COLOR)
        if viewer.s.scene is None:
            viewer.s.viewport_texture = None
            viewer.s.last_render_exception = ""
            update_ui_text(viewer, dt)
            return
        if runtime_reconfigured:
            viewer.s.training_runtime_factor_changed = False
            viewer.s.last_training_batch_steps = 0
        else:
            _run_training_batch(viewer)
        if bool(getattr(viewer.s, "training_runtime_factor_changed", False)):
            session.ensure_training_runtime_resolution(viewer)
        viewer.s.training_runtime_factor_changed = False
        if _training_camera_debug_active(viewer) and viewer.s.trainer is not None and viewer.s.training_frames:
            viewer.s.viewport_texture = _render_debug_view(viewer, encoder, render_width, render_height)
        else:
            viewer.s.viewport_texture = _render_main_view(viewer, encoder)
        if bool(viewer.ui._values.get("_histograms_refresh_requested", False)):
            session.refresh_cached_raster_grad_histograms(viewer)
        viewer.s.last_render_exception = ""
    except Exception as exc:
        viewer.s.training_active = False
        viewer.s.last_training_batch_steps = 0
        viewer.s.viewport_texture = None
        viewer.s.last_error = str(exc)
        if viewer.s.last_render_exception != viewer.s.last_error:
            print(f"Render/training error: {viewer.s.last_error}")
        viewer.s.last_render_exception = viewer.s.last_error
        encoder.clear_texture_float(image, clear_value=[0.0, 0.0, 0.0, 1.0])
    update_ui_text(viewer, dt)

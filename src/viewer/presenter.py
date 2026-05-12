from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import slangpy as spy

from ..utility import alloc_texture_2d, clamp_index, debug_region, require_not_none
from ..filter import SeparableGaussianBlur
from ..training import PPISPTonemapParams, resolve_colorspace_mod, resolve_sh_band
from . import session
from .buffer_debug import ResourceDebugSnapshot, collect_resource_debug_snapshot, query_total_device_vram_capacity, query_total_device_vram_used_cached, split_resource_usage
from .presenter_state import (
    _debug_frame_idx,
    _debug_view_key,
    _frame_metrics_snapshot,
    _render_stats_text,
    _run_training_batch,
    _training_camera_debug_active,
    _training_status_texts,
    _viewer_panel_state,
    _ui_header_state,
    _viewport_target_size,
)
from .ui_schema import PPISP_DEBUG_MODE, _DEBUG_MODE_VALUES

_DEBUG_TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_destination
_DEBUG_ABS_DIFF_SCALE_DEFAULT = 1.0
_DEBUG_ABS_DIFF_SCALE_MIN = 0.125
_DEBUG_ABS_DIFF_SCALE_MAX = 64.0
_DEBUG_DSSIM_FEATURE_CHANNELS = 15
_DEBUG_TARGET_SAMPLE_REGION = 155
_VIEWER_CLEAR_COLOR = [0.08, 0.09, 0.11, 1.0]
_RESOURCE_DEBUG_REFRESH_SECONDS = 5.0
_MENU_BAR_RESOURCE_REFRESH_SECONDS = 1.0
_HISTOGRAM_REALTIME_REFRESH_SECONDS = 1.0
_TRAINING_CAMERA_COLMAP_POINT_LIMIT = 4096


def _training_camera_full_resolution(viewer: object) -> bool:
    try:
        return bool(viewer.ui._values.get("training_camera_full_resolution", False))
    except Exception:
        return False


def _ensure_texture(viewer: object, attr: str, width: int, height: int) -> spy.Texture:
    texture = getattr(viewer.s, attr)
    if texture is not None and int(texture.width) == int(width) and int(texture.height) == int(height):
        return texture
    created = alloc_texture_2d(
        viewer.device,
        name=f"viewer.{attr}",
        format=spy.Format.rgba32_float,
        width=int(width),
        height=int(height),
        usage=_DEBUG_TEXTURE_USAGE,
    )
    setattr(viewer.s, attr, created)
    return created


def _debug_abs_diff_scale(viewer: object) -> float:
    try:
        value = float(viewer.c("loss_debug_abs_scale").value)
    except Exception:
        return _DEBUG_ABS_DIFF_SCALE_DEFAULT
    return max(_DEBUG_ABS_DIFF_SCALE_MIN, min(value, _DEBUG_ABS_DIFF_SCALE_MAX))


def _dispatch_debug_abs_diff(viewer: object, encoder: spy.CommandEncoder, rendered_tex: spy.Texture, target_tex: spy.Texture, width: int, height: int, *, rendered_is_linear: bool = True) -> spy.Texture:
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
                "g_DebugRenderedIsLinear": int(rendered_is_linear),
            },
            command_encoder=encoder,
        )
    return output


def _dispatch_debug_edge_filter(viewer: object, encoder: spy.CommandEncoder, source_tex: spy.Texture, width: int, height: int, *, source_is_linear: bool = False) -> spy.Texture:
    output = _ensure_texture(viewer, "loss_debug_texture", width, height)
    with debug_region(encoder, "Viewer Debug Edge", 152):
        require_not_none(viewer.s.debug_edge_kernel, "Debug edge kernel is not initialized.").dispatch(
            thread_count=spy.uint3(int(width), int(height), 1),
            vars={
                "g_DebugRendered": source_tex,
                "g_DebugOutput": output,
                "g_DebugWidth": int(width),
                "g_DebugHeight": int(height),
                "g_DebugSourceIsLinear": int(source_is_linear),
            },
            command_encoder=encoder,
        )
    return output


def _debug_ssim_c2(viewer: object) -> float:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is not None and hasattr(trainer, "training") and hasattr(trainer.training, "ssim_c2"):
        return float(trainer.training.ssim_c2)
    try:
        return float(viewer.c("ssim_c2").value)
    except Exception:
        return 9e-4


def _debug_target_alpha_mask_enabled(viewer: object) -> int:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is not None and hasattr(trainer, "training") and hasattr(trainer.training, "use_target_alpha_mask"):
        return int(bool(trainer.training.use_target_alpha_mask))
    return 0


def _ensure_debug_dssim_runtime(viewer: object, width: int, height: int) -> None:
    resolution = (int(width), int(height))
    if (
        getattr(viewer.s, "debug_dssim_resolution", None) == resolution
        and getattr(viewer.s, "debug_dssim_blur", None) is not None
        and getattr(viewer.s, "debug_dssim_moments", None) is not None
        and getattr(viewer.s, "debug_dssim_blurred_moments", None) is not None
    ):
        return
    blur = SeparableGaussianBlur(viewer.device, width=resolution[0], height=resolution[1])
    viewer.s.debug_dssim_blur = blur
    viewer.s.debug_dssim_resolution = resolution
    viewer.s.debug_dssim_moments = blur.make_buffer(_DEBUG_DSSIM_FEATURE_CHANNELS)
    viewer.s.debug_dssim_blurred_moments = blur.make_buffer(_DEBUG_DSSIM_FEATURE_CHANNELS)


def _dispatch_debug_dssim_features(viewer: object, encoder: spy.CommandEncoder, rendered_tex: spy.Texture, target_tex: spy.Texture, width: int, height: int) -> None:
    _ensure_debug_dssim_runtime(viewer, width, height)
    with debug_region(encoder, "Viewer DSSIM Features", 153):
        require_not_none(viewer.s.debug_dssim_features_kernel, "Debug DSSIM feature kernel is not initialized.").dispatch(
            thread_count=spy.uint3(int(width), int(height), 1),
            vars={
                "g_DebugRendered": rendered_tex,
                "g_DebugTarget": target_tex,
                "g_SSIMMoments": require_not_none(viewer.s.debug_dssim_moments, "Debug DSSIM moments buffer is not initialized."),
                "g_Width": int(width),
                "g_Height": int(height),
                "g_DebugWidth": int(width),
                "g_DebugHeight": int(height),
            },
            command_encoder=encoder,
        )


def _dispatch_debug_dssim_compose(viewer: object, encoder: spy.CommandEncoder, target_tex: spy.Texture, width: int, height: int) -> spy.Texture:
    output = _ensure_texture(viewer, "loss_debug_texture", width, height)
    with debug_region(encoder, "Viewer DSSIM Compose", 154):
        require_not_none(viewer.s.debug_dssim_compose_kernel, "Debug DSSIM compose kernel is not initialized.").dispatch(
            thread_count=spy.uint3(int(width), int(height), 1),
            vars={
                "g_DebugTarget": target_tex,
                "g_DebugOutput": output,
                "g_SSIMBlurredMoments": require_not_none(viewer.s.debug_dssim_blurred_moments, "Debug DSSIM blurred moments buffer is not initialized."),
                "g_DebugWidth": int(width),
                "g_DebugHeight": int(height),
                "g_Width": int(width),
                "g_Height": int(height),
                "g_SSIMC2": _debug_ssim_c2(viewer),
                "g_UseTargetAlphaMask": _debug_target_alpha_mask_enabled(viewer),
            },
            command_encoder=encoder,
        )
    return output


def _dispatch_debug_dssim(viewer: object, encoder: spy.CommandEncoder, rendered_tex: spy.Texture, target_tex: spy.Texture, width: int, height: int) -> spy.Texture:
    _dispatch_debug_dssim_features(viewer, encoder, rendered_tex, target_tex, width, height)
    require_not_none(viewer.s.debug_dssim_blur, "Debug DSSIM blur is not initialized.").blur(
        encoder,
        require_not_none(viewer.s.debug_dssim_moments, "Debug DSSIM moments buffer is not initialized."),
        require_not_none(viewer.s.debug_dssim_blurred_moments, "Debug DSSIM blurred moments buffer is not initialized."),
        _DEBUG_DSSIM_FEATURE_CHANNELS,
    )
    return _dispatch_debug_dssim_compose(viewer, encoder, target_tex, width, height)


def _training_debug_colorspace_mod(viewer: object) -> float:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is None or not hasattr(trainer, "training"):
        return 1.0
    return float(resolve_colorspace_mod(trainer.training, _training_debug_step(viewer)))


def _dispatch_training_debug_present(
    viewer: object,
    encoder: spy.CommandEncoder,
    source_tex: spy.Texture,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    *,
    source_is_linear: bool = False,
    apply_loss_colorspace: bool = False,
) -> spy.Texture:
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
                "g_LetterboxSourceIsLinear": int(source_is_linear),
                "g_LetterboxApplyLossColorspace": int(apply_loss_colorspace),
                "g_ColorspaceMod": _training_debug_colorspace_mod(viewer),
            },
            command_encoder=encoder,
        )
    return output


def _dispatch_viewport_present(viewer: object, encoder: spy.CommandEncoder, source_tex: spy.Texture, source_width: int, source_height: int, output_width: int, output_height: int, *, source_is_linear: bool = False) -> spy.Texture:
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
                "g_LetterboxSourceIsLinear": int(source_is_linear),
            },
            command_encoder=encoder,
        )
    return output


def _ppisp_preview_enabled(viewer: object) -> bool:
    renderer = getattr(getattr(viewer, "s", None), "renderer", None)
    values = getattr(getattr(viewer, "ui", None), "_values", {})
    mode_index = min(max(int(values.get("debug_mode", 0)), 0), len(_DEBUG_MODE_VALUES) - 1)
    return _DEBUG_MODE_VALUES[mode_index] == PPISP_DEBUG_MODE and str(getattr(renderer, "debug_mode", "normal")) == "normal"


def _refresh_resource_debug_snapshot(viewer: object) -> None:
    values = viewer.ui._values
    if not bool(values.get("show_resource_debug", False)):
        return
    now = float(getattr(viewer.s, "last_time", time.perf_counter()))
    snapshot = values.get("_resource_debug_snapshot")
    next_update = float(values.get("_resource_debug_next_update", 0.0) or 0.0)
    refresh_requested = bool(values.get("_resource_debug_refresh_requested", False))
    include_process_vram = bool(values.get("_resource_debug_process_vram_requested", False))
    if snapshot is not None and not refresh_requested and not include_process_vram and now < next_update:
        return
    values["_resource_debug_snapshot"] = collect_resource_debug_snapshot(viewer, include_process_vram=include_process_vram)
    values["_resource_debug_next_update"] = now + _RESOURCE_DEBUG_REFRESH_SECONDS
    values["_resource_debug_refresh_requested"] = False
    values["_resource_debug_process_vram_requested"] = False


def _refresh_menu_bar_device_vram(viewer: object) -> None:
    values = viewer.ui._values
    previous_value = values.get("_menu_bar_device_vram_bytes")
    try:
        used_bytes, source = query_total_device_vram_used_cached(getattr(viewer, "device", None))
        total_bytes, total_source = query_total_device_vram_capacity(getattr(viewer, "device", None))
    except Exception:
        return
    if used_bytes is None and previous_value is not None:
        used_bytes = previous_value
    values["_menu_bar_device_vram_bytes"] = None if used_bytes is None else int(used_bytes)
    values["_menu_bar_device_vram_source"] = str(source)
    values["_menu_bar_device_vram_total_bytes"] = None if total_bytes is None else int(total_bytes)
    values["_menu_bar_device_vram_total_source"] = str(total_source)


def _refresh_menu_bar_resource_totals(viewer: object) -> None:
    values = viewer.ui._values
    now = float(getattr(viewer.s, "last_time", time.perf_counter()))
    snapshot = values.get("_resource_debug_snapshot")
    if isinstance(snapshot, ResourceDebugSnapshot):
        usage = split_resource_usage(snapshot)
        values["_menu_bar_dataset_vram_bytes"] = int(usage.dataset_bytes)
        values["_menu_bar_app_vram_bytes"] = int(usage.app_bytes)
        values["_menu_bar_total_vram_bytes"] = int(usage.total_bytes)
        values["_menu_bar_resource_next_update"] = now + _MENU_BAR_RESOURCE_REFRESH_SECONDS
        return
    next_update = float(values.get("_menu_bar_resource_next_update", 0.0) or 0.0)
    if now < next_update:
        return
    try:
        snapshot = collect_resource_debug_snapshot(viewer, include_process_vram=False)
    except Exception:
        return
    if not isinstance(snapshot, ResourceDebugSnapshot):
        return
    usage = split_resource_usage(snapshot)
    values["_menu_bar_dataset_vram_bytes"] = int(usage.dataset_bytes)
    values["_menu_bar_app_vram_bytes"] = int(usage.app_bytes)
    values["_menu_bar_total_vram_bytes"] = int(usage.total_bytes)
    values["_menu_bar_resource_next_update"] = now + _MENU_BAR_RESOURCE_REFRESH_SECONDS


def _set_text(viewer: object, key: str, value: object) -> None:
    viewer.t(key).text = str(value)


def _set_ui_value(viewer: object, key: str, value: object) -> None:
    viewer.ui._values[key] = value


def update_ui_text(viewer: object, dt: float) -> None:
    viewer.s.fps_smooth += (1.0 / max(dt, 1e-5) - viewer.s.fps_smooth) * min(dt * 5.0, 1.0)
    session.update_debug_frame_slider_range(viewer)
    frame_idx = _debug_frame_idx(viewer)
    debug_metrics = _frame_metrics_snapshot(viewer, len(getattr(viewer.s, "training_frames", ())))
    stats = viewer.s.stats
    header_state = _ui_header_state(viewer, debug_metrics, frame_idx)
    _set_text(viewer, "fps", f"FPS: {viewer.s.fps_smooth:.1f}")
    _set_text(viewer, "loss_debug_frame", header_state["loss_debug_frame"])
    _set_text(viewer, "loss_debug_psnr", header_state["loss_debug_psnr"])
    training_camera_sections, pose_available = _training_camera_debug_panel_sections(viewer)
    _set_ui_value(viewer, "_training_camera_struct_sections", training_camera_sections)
    _set_ui_value(viewer, "_training_camera_pose_available", pose_available)
    _set_ui_value(viewer, "_training_camera_colmap_points_payload", _training_camera_colmap_points_payload(viewer))
    _set_text(viewer, "path", header_state["path"])
    _set_ui_value(viewer, "_colmap_import_active", bool(header_state["colmap_import_active"]))
    _set_ui_value(viewer, "_colmap_import_fraction", float(header_state["colmap_import_fraction"]))
    _set_ui_value(viewer, "_can_export_ply", bool(header_state["can_export_ply"]))
    _set_text(viewer, "colmap_import_status", header_state["colmap_import_status"])
    _set_text(viewer, "colmap_import_current", header_state["colmap_import_current"])
    _set_text(viewer, "scene_stats", header_state["scene_stats"])
    current_splat_count = int(header_state["current_splat_count"])
    panel_state = _viewer_panel_state(viewer, debug_metrics=debug_metrics)
    _set_ui_value(viewer, "_training_resolution_sections", panel_state["training_resolution_sections"])
    _set_ui_value(viewer, "_training_downscale_sections", panel_state["training_downscale_sections"])
    _set_text(viewer, "training_schedule", panel_state["training_schedule"])
    _set_ui_value(viewer, "_training_schedule_sections", panel_state["training_schedule_sections"])
    _set_ui_value(viewer, "_training_refinement_sections", panel_state["training_refinement_sections"])
    _set_ui_value(viewer, "_viewport_sh_control_key", str(panel_state["viewport_sh_control_key"]))
    _set_ui_value(viewer, "_viewport_sh_stage_label", str(panel_state["viewport_sh_stage_label"]))
    _set_text(viewer, "histogram_status", panel_state["histogram_status"])
    _set_ui_value(viewer, "_histogram_payload", panel_state["histogram_payload"])
    _set_ui_value(viewer, "_histogram_range_payload", panel_state["histogram_range_payload"])
    _set_ui_value(viewer, "_training_views_rows", panel_state["training_views_rows"])
    _set_ui_value(viewer, "_training_view_overlay_segments", panel_state["training_view_overlay_segments"])
    _refresh_resource_debug_snapshot(viewer)
    _refresh_menu_bar_device_vram(viewer)
    _refresh_menu_bar_resource_totals(viewer)
    _set_text(viewer, "render_stats", _render_stats_text(stats))
    training_elapsed_s = 0.0 if viewer.s.trainer is None else float(session.training_elapsed_seconds(viewer, now=viewer.s.last_time))
    for key, text in _training_status_texts(viewer, current_splat_count, training_elapsed_s).items():
        _set_text(viewer, key, text)
    _set_text(viewer, "error", f"Error: {viewer.s.last_error}" if viewer.s.last_error else "")
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
            loss_value = float(state.avg_loss) if np.isfinite(state.avg_loss) and state.avg_loss > 0 else (float(tk.tk.loss_history[-1]) if tk.tk.loss_history else float("nan"))
            ssim_value = float(np.clip(state.avg_ssim, 0.0, 1.0)) if np.isfinite(getattr(state, "avg_ssim", float("nan"))) else (float(tk.tk.ssim_history[-1]) if tk.tk.ssim_history else float("nan"))
            psnr_value = float(state.avg_psnr) if np.isfinite(state.avg_psnr) else (float(tk.tk.psnr_history[-1]) if tk.tk.psnr_history else float("nan"))
            tk.tk.append_training_plot_sample(step, float(viewer.s.last_time), loss_value, ssim_value, psnr_value)


def _training_debug_step(viewer: object) -> int:
    return max(int(getattr(getattr(viewer.s.trainer, "state", None), "step", 0)), 0)


def _training_debug_resolution(viewer: object, frame_idx: int, step: int) -> tuple[int, int]:
    if _training_camera_full_resolution(viewer):
        return _training_debug_frame_size(viewer, frame_idx)
    trainer = viewer.s.trainer
    if hasattr(trainer, "training_resolution"):
        width, height = trainer.training_resolution(frame_idx, step)
        return max(int(width), 1), max(int(height), 1)
    frame = viewer.s.training_frames[frame_idx]
    return max(int(frame.width), 1), max(int(frame.height), 1)


def _training_debug_frame_size(viewer: object, frame_idx: int) -> tuple[int, int]:
    trainer = viewer.s.trainer
    if hasattr(trainer, "frame_size"):
        width, height = trainer.frame_size(frame_idx)
        return max(int(width), 1), max(int(height), 1)
    frame = viewer.s.training_frames[frame_idx]
    return max(int(frame.width), 1), max(int(frame.height), 1)


def _training_debug_sample_vars(viewer: object, frame_idx: int, step: int, render_frame_index: int) -> dict[str, object] | None:
    if _training_camera_full_resolution(viewer):
        return None
    trainer = viewer.s.trainer
    if hasattr(trainer, "training_sample_vars"):
        return trainer.training_sample_vars(frame_idx, step, sample_seed_step=render_frame_index)
    return {}


def _training_debug_background(viewer: object) -> np.ndarray:
    trainer = viewer.s.trainer
    if hasattr(trainer, "training_background"):
        return np.asarray(trainer.training_background(), dtype=np.float32).reshape(3)
    return np.asarray(getattr(viewer.s, "background", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(3)


def _training_debug_background_seed(viewer: object, render_frame_index: int) -> int:
    trainer = viewer.s.trainer
    if hasattr(trainer, "training_background_seed"):
        return int(trainer.training_background_seed(render_frame_index))
    return int(render_frame_index)


def _training_debug_pose_available(camera: object) -> bool:
    if camera is None:
        return False
    try:
        position = np.asarray(getattr(camera, "position"), dtype=np.float32).reshape(-1)
        target = np.asarray(getattr(camera, "target"), dtype=np.float32).reshape(-1)
        up = np.asarray(getattr(camera, "up"), dtype=np.float32).reshape(-1)
    except Exception:
        return False
    return position.size >= 3 and target.size >= 3 and up.size >= 3 and np.all(np.isfinite(position[:3])) and np.all(np.isfinite(target[:3])) and np.all(np.isfinite(up[:3]))


def _training_debug_pose_camera(viewer: object, frame_idx: int, native_width: int, native_height: int):
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is not None and hasattr(trainer, "make_frame_camera"):
        try:
            camera = trainer.make_frame_camera(frame_idx, int(native_width), int(native_height))
        except Exception:
            camera = None
        if _training_debug_pose_available(camera):
            return camera
    frame = viewer.s.training_frames[frame_idx]
    make_camera = getattr(frame, "make_camera", None)
    if not callable(make_camera):
        return None
    try:
        camera = make_camera(near=float(getattr(viewer.s, "near", 0.1)), far=float(getattr(viewer.s, "far", 120.0)))
    except TypeError:
        camera = make_camera()
    except Exception:
        camera = None
    return camera if _training_debug_pose_available(camera) else None


def _scalar_or_none(value: object) -> float | None:
    try:
        scalar = float(value)
    except Exception:
        return None
    return None if not np.isfinite(scalar) else scalar


def _vec3_or_none(value: object) -> tuple[float, float, float] | None:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if vector.size < 3 or not np.all(np.isfinite(vector[:3])):
        return None
    return float(vector[0]), float(vector[1]), float(vector[2])


def _training_camera_debug_panel_sections(viewer: object) -> tuple[tuple, bool]:
    frames = tuple(getattr(viewer.s, "training_frames", ()))
    if len(frames) == 0:
        return (), False
    frame_idx = _debug_frame_idx(viewer)
    step = _training_debug_step(viewer)
    frame = frames[frame_idx]
    target_width, target_height = _training_debug_resolution(viewer, frame_idx, step)
    native_width, native_height = _training_debug_frame_size(viewer, frame_idx)
    camera = _training_debug_pose_camera(viewer, frame_idx, native_width, native_height)
    camera_id = getattr(frame, "camera_id", None)
    pose_section: tuple = ()
    if camera is not None:
        pose_section = (("Pose", (
            ("pos", _vec3_or_none(getattr(camera, "position", None))),
            ("target", _vec3_or_none(getattr(camera, "target", None))),
            ("up", _vec3_or_none(getattr(camera, "up", None))),
            ("near", _scalar_or_none(getattr(camera, "near", None))),
            ("far", _scalar_or_none(getattr(camera, "far", None))),
        )),)
    sections = (
        ("Resolution", (
            ("target", f"{int(target_width)}x{int(target_height)}"),
            ("source", f"{int(native_width)}x{int(native_height)}"),
            ("full_res", bool(_training_camera_full_resolution(viewer))),
        )),
        ("Ids", (
            ("image", int(getattr(frame, "image_id", -1))),
            ("camera", int(camera_id) if camera_id is not None else None),
        )),
        *pose_section,
        ("Projection", (
            ("fx", _scalar_or_none(getattr(frame, "fx", None))),
            ("fy", _scalar_or_none(getattr(frame, "fy", None))),
            ("cx", _scalar_or_none(getattr(frame, "cx", None))),
            ("cy", _scalar_or_none(getattr(frame, "cy", None))),
        )),
        ("Distortion A", (
            ("k1", _scalar_or_none(getattr(frame, "k1", None))),
            ("k2", _scalar_or_none(getattr(frame, "k2", None))),
            ("p1", _scalar_or_none(getattr(frame, "p1", None))),
            ("p2", _scalar_or_none(getattr(frame, "p2", None))),
        )),
        ("Distortion B", (
            ("k3", _scalar_or_none(getattr(frame, "k3", None))),
            ("k4", _scalar_or_none(getattr(frame, "k4", None))),
            ("k5", _scalar_or_none(getattr(frame, "k5", None))),
            ("k6", _scalar_or_none(getattr(frame, "k6", None))),
        )),
    )
    return sections, camera is not None


def _training_camera_colmap_observation_index(viewer: object) -> dict[int, tuple[tuple[int, str], ...]]:
    recon = getattr(viewer.s, "colmap_recon", None)
    if recon is None:
        return {}
    signature = (id(recon),)
    cached_signature = getattr(viewer.s, "training_camera_colmap_observation_signature", None)
    cached_index = getattr(viewer.s, "training_camera_colmap_observation_index", None)
    if cached_signature == signature and isinstance(cached_index, dict):
        return cached_index
    observations: dict[int, list[tuple[int, str]]] = {}
    for image_id, image in sorted(getattr(recon, "images", {}).items()):
        point_ids = np.asarray(getattr(image, "points2d_point3d_ids", ()), dtype=np.int64).reshape(-1)
        if point_ids.size == 0:
            continue
        valid_point_ids = point_ids[point_ids > 0]
        if valid_point_ids.size == 0:
            continue
        image_name = Path(str(getattr(image, "name", f"image_{int(image_id)}"))).name
        view = (int(image_id), image_name)
        for point_id in np.unique(valid_point_ids):
            observations.setdefault(int(point_id), []).append(view)
    cached = {point_id: tuple(views) for point_id, views in observations.items()}
    viewer.s.training_camera_colmap_observation_signature = signature
    viewer.s.training_camera_colmap_observation_index = cached
    return cached


def _training_camera_colmap_points_payload(viewer: object) -> dict[str, object] | None:
    frames = tuple(getattr(viewer.s, "training_frames", ()))
    recon = getattr(viewer.s, "colmap_recon", None)
    if recon is None or len(frames) == 0:
        return None
    frame = frames[_debug_frame_idx(viewer)]
    image_id = getattr(frame, "image_id", None)
    if image_id is None:
        return None
    image = getattr(recon, "images", {}).get(int(image_id))
    if image is None:
        return None
    point_xy = np.asarray(getattr(image, "points2d_xy", ()), dtype=np.float32).reshape(-1, 2)
    point_ids = np.asarray(getattr(image, "points2d_point3d_ids", ()), dtype=np.int64).reshape(-1)
    count = min(int(point_xy.shape[0]), int(point_ids.size))
    source_width = max(int(getattr(frame, "width", 0)), 1)
    source_height = max(int(getattr(frame, "height", 0)), 1)
    image_name = Path(str(getattr(image, "name", getattr(frame, "image_path", image_id)))).name
    if count <= 0:
        return {
            "image_id": int(image_id),
            "image_name": image_name,
            "source_size": (source_width, source_height),
            "total_count": 0,
            "render_count": 0,
            "point_ids": np.zeros((0,), dtype=np.int64),
            "xy": np.zeros((0, 2), dtype=np.float32),
            "uv": np.zeros((0, 2), dtype=np.float32),
            "track_lengths": np.zeros((0,), dtype=np.int32),
            "errors": np.zeros((0,), dtype=np.float32),
            "other_views": (),
        }
    point_xy = np.ascontiguousarray(point_xy[:count], dtype=np.float32)
    point_ids = np.ascontiguousarray(point_ids[:count], dtype=np.int64)
    valid = np.isfinite(point_xy).all(axis=1) & (point_ids > 0)
    valid &= (point_xy[:, 0] >= 0.0) & (point_xy[:, 0] <= float(source_width))
    valid &= (point_xy[:, 1] >= 0.0) & (point_xy[:, 1] <= float(source_height))
    total_count = int(np.count_nonzero(valid))
    point_xy = np.ascontiguousarray(point_xy[valid], dtype=np.float32)
    point_ids = np.ascontiguousarray(point_ids[valid], dtype=np.int64)
    if point_xy.shape[0] > _TRAINING_CAMERA_COLMAP_POINT_LIMIT:
        point_xy = np.ascontiguousarray(point_xy[:_TRAINING_CAMERA_COLMAP_POINT_LIMIT], dtype=np.float32)
        point_ids = np.ascontiguousarray(point_ids[:_TRAINING_CAMERA_COLMAP_POINT_LIMIT], dtype=np.int64)
    uv = np.zeros((int(point_xy.shape[0]), 2), dtype=np.float32)
    if point_xy.shape[0] > 0:
        uv[:, 0] = np.clip(point_xy[:, 0] / float(source_width), 0.0, 1.0)
        uv[:, 1] = np.clip(point_xy[:, 1] / float(source_height), 0.0, 1.0)
    point_lookup = getattr(recon, "points3d", {})
    observation_index = _training_camera_colmap_observation_index(viewer)
    track_lengths = np.zeros((int(point_ids.size),), dtype=np.int32)
    errors = np.full((int(point_ids.size),), np.nan, dtype=np.float32)
    other_views: list[tuple[tuple[int, str], ...]] = []
    for point_index, point_id in enumerate(point_ids.tolist()):
        point = point_lookup.get(int(point_id)) if isinstance(point_lookup, dict) else None
        if point is not None:
            track_lengths[point_index] = int(getattr(point, "track_length", 0))
            errors[point_index] = float(getattr(point, "error", float("nan")))
        other_views.append(tuple(view for view in observation_index.get(int(point_id), ()) if int(view[0]) != int(image_id)))
    return {
        "image_id": int(image_id),
        "image_name": image_name,
        "source_size": (source_width, source_height),
        "total_count": total_count,
        "render_count": int(point_ids.size),
        "point_ids": point_ids,
        "xy": point_xy,
        "uv": uv,
        "track_lengths": track_lengths,
        "errors": errors,
        "other_views": tuple(other_views),
    }


def _apply_training_debug_renderer_hparams(viewer: object, debug_renderer: object, step: int) -> None:
    trainer = viewer.s.trainer
    if hasattr(trainer, "apply_renderer_training_hparams"):
        trainer.apply_renderer_training_hparams(step, renderer=debug_renderer)
    else:
        debug_renderer.sh_band = resolve_sh_band(trainer.training, step)


def _render_debug_source(viewer: object, encoder: spy.CommandEncoder, frame_idx: int, render_frame_index: int) -> tuple[spy.Texture, dict[str, int | bool | float], int, int, dict[str, object] | None]:
    step = _training_debug_step(viewer)
    debug_width, debug_height = _training_debug_resolution(viewer, frame_idx, step)
    native_width, native_height = _training_debug_frame_size(viewer, frame_idx)
    frame_camera = viewer.s.trainer.make_frame_camera(frame_idx, debug_width, debug_height)
    native_camera = viewer.s.trainer.make_frame_camera(frame_idx, native_width, native_height)
    debug_renderer = session.ensure_renderer(viewer, "debug_renderer", debug_width, debug_height, allow_debug_overlays=True)
    session.sync_scene_from_training_renderer(viewer, debug_renderer, target="debug")
    _apply_training_debug_renderer_hparams(viewer, debug_renderer, step)
    sample_vars = _training_debug_sample_vars(viewer, frame_idx, step, render_frame_index)
    sort_dither = viewer.s.trainer.sorting_dither(frame_idx, step, frame_camera) if hasattr(viewer.s.trainer, "sorting_dither") else None
    training = viewer.s.trainer.training
    source_tex, stats = debug_renderer.render_training_forward_to_texture(
        frame_camera,
        background=_training_debug_background(viewer),
        read_stats=True,
        command_encoder=encoder,
        sort_camera_position=None if sort_dither is None else sort_dither.position,
        sort_camera_dither_sigma=0.0 if sort_dither is None else sort_dither.sigma,
        sort_camera_dither_seed=0 if sort_dither is None else sort_dither.seed,
        training_background_mode=int(getattr(training, "background_mode", 0)),
        training_background_seed=_training_debug_background_seed(viewer, render_frame_index),
        training_native_camera=native_camera,
        training_sample_vars=sample_vars,
    )
    return source_tex, stats, debug_width, debug_height, sample_vars


def _sample_training_debug_target(viewer: object, encoder: spy.CommandEncoder, source_tex: spy.Texture, width: int, height: int, sample_vars: dict[str, object] | None, frame_idx: int) -> spy.Texture:
    output = _ensure_texture(viewer, "debug_target_texture", width, height)
    frame = viewer.s.training_frames[frame_idx]
    resolved_sample_vars = {} if sample_vars is None else sample_vars
    with debug_region(encoder, "Viewer Debug Target Sample", _DEBUG_TARGET_SAMPLE_REGION):
        require_not_none(viewer.s.debug_target_sample_kernel, "Debug target sample kernel is not initialized.").dispatch(
            thread_count=spy.uint3(int(width), int(height), 1),
            vars={
                "g_SourceTarget": source_tex,
                "g_DownscaledTarget": output,
                "g_SourceWidth": max(int(frame.width), 1),
                "g_SourceHeight": max(int(frame.height), 1),
                "g_TargetWidth": int(width),
                "g_TargetHeight": int(height),
                "g_DownscaleFactor": int(resolved_sample_vars.get("g_TrainingSubsample", {}).get("factor", 1)),
                **resolved_sample_vars,
            },
            command_encoder=encoder,
        )
    return output


def _render_debug_target(viewer: object, encoder: spy.CommandEncoder, frame_idx: int, width: int, height: int, step: int, sample_vars: dict[str, object] | None) -> spy.Texture:
    trainer = viewer.s.trainer
    if hasattr(trainer, "ensure_frame_render_resolution"):
        trainer.ensure_frame_render_resolution(frame_idx, step)
    if _training_camera_full_resolution(viewer):
        try:
            return trainer.get_frame_target_texture(frame_idx, native_resolution=True, encoder=encoder, step=step)
        except TypeError:
            return trainer.get_frame_target_texture(frame_idx, native_resolution=True, encoder=encoder)
    subsample = int(trainer.effective_train_subsample_factor(frame_idx, step)) if hasattr(trainer, "effective_train_subsample_factor") else 1
    if subsample > 1:
        try:
            native_target = trainer.get_frame_target_texture(frame_idx, native_resolution=True, encoder=encoder, step=step)
        except TypeError:
            native_target = trainer.get_frame_target_texture(frame_idx, native_resolution=True, encoder=encoder)
        return _sample_training_debug_target(viewer, encoder, native_target, width, height, sample_vars, frame_idx)
    try:
        return trainer.get_frame_target_texture(frame_idx, native_resolution=False, encoder=encoder, step=step)
    except TypeError:
        return trainer.get_frame_target_texture(frame_idx, native_resolution=False, encoder=encoder)


def _render_debug_view(viewer: object, encoder: spy.CommandEncoder, output_width: int, output_height: int, render_frame_index: int) -> spy.Texture:
    frame_idx = _debug_frame_idx(viewer)
    debug_render_tex, viewer.s.stats, debug_width, debug_height, sample_vars = _render_debug_source(viewer, encoder, frame_idx, render_frame_index)
    debug_view = _debug_view_key(viewer)
    if debug_view == "rendered":
        source_tex = debug_render_tex
        source_is_linear = True
        apply_loss_colorspace = True
    elif debug_view == "rendered_edges":
        source_tex = _dispatch_debug_edge_filter(viewer, encoder, debug_render_tex, debug_width, debug_height, source_is_linear=True)
        source_is_linear = False
        apply_loss_colorspace = False
    else:
        target_tex = _render_debug_target(viewer, encoder, frame_idx, debug_width, debug_height, _training_debug_step(viewer), sample_vars)
        source_is_linear = False
        apply_loss_colorspace = debug_view == "target"
        source_tex = (
            target_tex if debug_view == "target"
            else _dispatch_debug_abs_diff(viewer, encoder, debug_render_tex, target_tex, debug_width, debug_height, rendered_is_linear=True) if debug_view == "abs_diff"
            else _dispatch_debug_dssim(viewer, encoder, debug_render_tex, target_tex, debug_width, debug_height) if debug_view == "dssim"
            else _dispatch_debug_edge_filter(viewer, encoder, target_tex, debug_width, debug_height, source_is_linear=False)
        )
    return _dispatch_training_debug_present(viewer, encoder, source_tex, debug_width, debug_height, debug_width, debug_height, source_is_linear=source_is_linear, apply_loss_colorspace=apply_loss_colorspace)
def _render_main_view(viewer: object, encoder: spy.CommandEncoder) -> spy.Texture:
    if viewer.s.trainer is not None and viewer.s.training_renderer is not None:
        session.sync_scene_from_training_renderer(viewer, viewer.s.renderer, target="main")
    camera = viewer.camera()
    width, height = int(viewer.s.renderer.width), int(viewer.s.renderer.height)
    if _ppisp_preview_enabled(viewer):
        out_tex, stats = viewer.s.renderer.render_ppisp_to_texture(
            camera,
            PPISPTonemapParams.from_viewer_values(viewer.ui._values).to_shader_dict(),
            background=viewer.s.background,
            read_stats=True,
            command_encoder=encoder,
        )
        viewer.s.stats = stats
        return _dispatch_viewport_present(viewer, encoder, out_tex, width, height, width, height, source_is_linear=False)
    out_tex, stats = viewer.s.renderer.render_to_texture(camera, background=viewer.s.background, read_stats=True, command_encoder=encoder)
    viewer.s.stats = stats
    return _dispatch_viewport_present(viewer, encoder, out_tex, width, height, width, height, source_is_linear=False)


def render_frame(viewer: object, render_context: spy.AppWindow.RenderContext) -> None:
    image, encoder = render_context.surface_texture, render_context.command_encoder
    now = spy.time.perf_counter() if hasattr(spy, "time") else time.perf_counter()
    dt = max(now - viewer.s.last_time, 1e-5)
    viewer.s.last_time = now
    iw, ih = int(image.width), int(image.height)
    render_width, render_height = _viewport_target_size(viewer, iw, ih)
    render_frame_index = int(getattr(viewer.s, "render_frame_index", 0))
    viewer.s.render_frame_index = render_frame_index + 1
    try:
        viewer.update_camera(dt)
        runtime_reconfigured = False
        if bool(getattr(viewer.s, "pending_training_reinitialize", False)):
            viewer.s.pending_training_reinitialize = False
            session.reinitialize_training_scene(viewer)
        session.apply_live_params(viewer)
        session.advance_colmap_import(viewer)
        if bool(getattr(viewer.s, "pending_training_runtime_resize", False)):
            runtime_reconfigured = bool(session.ensure_training_runtime_resolution(viewer))
        if viewer.s.renderer is None:
            session.recreate_renderer(viewer, render_width, render_height)
        elif (viewer.s.renderer.width, viewer.s.renderer.height) != (render_width, render_height):
            session.recreate_renderer(viewer, render_width, render_height)
        else:
            session.maybe_reallocate_renderers(viewer, render_width, render_height, now)
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
            viewer.s.viewport_texture = _render_debug_view(viewer, encoder, render_width, render_height, render_frame_index)
        else:
            viewer.s.viewport_texture = _render_main_view(viewer, encoder)
        realtime_enabled = bool(viewer.ui._values.get("show_histograms", False)) and bool(viewer.ui._values.get("_histograms_update_realtime", False))
        realtime_refresh_due = realtime_enabled and now >= float(viewer.ui._values.get("_histograms_realtime_next_refresh_time", 0.0))
        if bool(viewer.ui._values.get("_histograms_refresh_requested", False)) or realtime_refresh_due:
            session.refresh_cached_raster_grad_histograms(viewer, force=realtime_refresh_due)
            viewer.ui._values["_histograms_realtime_next_refresh_time"] = now + _HISTOGRAM_REALTIME_REFRESH_SECONDS if realtime_enabled else 0.0
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

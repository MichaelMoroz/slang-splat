from __future__ import annotations

import time

import numpy as np
import slangpy as spy

from ..utility import alloc_texture_2d, clamp_index, debug_region, require_not_none
from ..filter import SeparableGaussianBlur
from ..training import resolve_colorspace_mod, resolve_sh_band
from . import session
from .buffer_debug import collect_resource_debug_snapshot
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

_DEBUG_TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_destination
_DEBUG_ABS_DIFF_SCALE_DEFAULT = 1.0
_DEBUG_ABS_DIFF_SCALE_MIN = 0.125
_DEBUG_ABS_DIFF_SCALE_MAX = 64.0
_DEBUG_DSSIM_FEATURE_CHANNELS = 15
_DEBUG_TARGET_SAMPLE_REGION = 155
_VIEWER_CLEAR_COLOR = [0.08, 0.09, 0.11, 1.0]
_RESOURCE_DEBUG_REFRESH_SECONDS = 5.0


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


def _set_text(viewer: object, key: str, value: object) -> None:
    viewer.t(key).text = str(value)


def _set_ui_value(viewer: object, key: str, value: object) -> None:
    viewer.ui._values[key] = value


def update_ui_text(viewer: object, dt: float) -> None:
    viewer.s.fps_smooth += (1.0 / max(dt, 1e-5) - viewer.s.fps_smooth) * min(dt * 5.0, 1.0)
    session.update_debug_frame_slider_range(viewer)
    frame_idx = _debug_frame_idx(viewer)
    debug_idx = clamp_index(int(viewer.c("loss_debug_view").value), len(viewer.loss_debug_view_options))
    debug_metrics = _frame_metrics_snapshot(viewer, len(getattr(viewer.s, "training_frames", ())))
    stats = viewer.s.stats
    header_state = _ui_header_state(viewer, debug_metrics, frame_idx, debug_idx)
    _set_text(viewer, "fps", f"FPS: {viewer.s.fps_smooth:.1f}")
    _set_text(viewer, "loss_debug_view", header_state["loss_debug_view"])
    _set_text(viewer, "loss_debug_frame", header_state["loss_debug_frame"])
    _set_text(viewer, "loss_debug_psnr", header_state["loss_debug_psnr"])
    _set_text(viewer, "path", header_state["path"])
    _set_ui_value(viewer, "_colmap_import_active", bool(header_state["colmap_import_active"]))
    _set_ui_value(viewer, "_colmap_import_fraction", float(header_state["colmap_import_fraction"]))
    _set_ui_value(viewer, "_can_export_ply", bool(header_state["can_export_ply"]))
    _set_text(viewer, "colmap_import_status", header_state["colmap_import_status"])
    _set_text(viewer, "colmap_import_current", header_state["colmap_import_current"])
    _set_text(viewer, "scene_stats", header_state["scene_stats"])
    current_splat_count = int(header_state["current_splat_count"])
    panel_state = _viewer_panel_state(viewer)
    _set_text(viewer, "training_resolution", panel_state["training_resolution"])
    _set_text(viewer, "training_downscale", panel_state["training_downscale"])
    _set_text(viewer, "training_schedule", panel_state["training_schedule"])
    _set_text(viewer, "training_schedule_values", panel_state["training_schedule_values"])
    _set_text(viewer, "training_refinement", panel_state["training_refinement"])
    _set_ui_value(viewer, "_viewport_sh_control_key", str(panel_state["viewport_sh_control_key"]))
    _set_ui_value(viewer, "_viewport_sh_stage_label", str(panel_state["viewport_sh_stage_label"]))
    _set_text(viewer, "histogram_status", panel_state["histogram_status"])
    _set_ui_value(viewer, "_histogram_payload", panel_state["histogram_payload"])
    _set_ui_value(viewer, "_histogram_range_payload", panel_state["histogram_range_payload"])
    _set_ui_value(viewer, "_training_views_rows", panel_state["training_views_rows"])
    _set_ui_value(viewer, "_training_view_overlay_segments", panel_state["training_view_overlay_segments"])
    _refresh_resource_debug_snapshot(viewer)
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


def _training_debug_step(viewer: object) -> int:
    return max(int(getattr(getattr(viewer.s.trainer, "state", None), "step", 0)), 0)


def _training_debug_resolution(viewer: object, frame_idx: int, step: int) -> tuple[int, int]:
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


def _training_debug_sample_vars(viewer: object, frame_idx: int, step: int, render_frame_index: int) -> dict[str, object]:
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


def _apply_training_debug_renderer_hparams(viewer: object, debug_renderer: object, step: int) -> None:
    trainer = viewer.s.trainer
    if hasattr(trainer, "apply_renderer_training_hparams"):
        trainer.apply_renderer_training_hparams(step, renderer=debug_renderer)
    else:
        debug_renderer.sh_band = resolve_sh_band(trainer.training, step)


def _render_debug_source(viewer: object, encoder: spy.CommandEncoder, frame_idx: int, render_frame_index: int) -> tuple[spy.Texture, dict[str, int | bool | float], int, int, dict[str, object]]:
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


def _sample_training_debug_target(viewer: object, encoder: spy.CommandEncoder, source_tex: spy.Texture, width: int, height: int, sample_vars: dict[str, object], frame_idx: int) -> spy.Texture:
    output = _ensure_texture(viewer, "debug_target_texture", width, height)
    frame = viewer.s.training_frames[frame_idx]
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
                "g_DownscaleFactor": int(sample_vars.get("g_TrainingSubsample", {}).get("factor", 1)),
                **sample_vars,
            },
            command_encoder=encoder,
        )
    return output


def _render_debug_target(viewer: object, encoder: spy.CommandEncoder, frame_idx: int, width: int, height: int, step: int, sample_vars: dict[str, object]) -> spy.Texture:
    trainer = viewer.s.trainer
    if hasattr(trainer, "ensure_frame_render_resolution"):
        trainer.ensure_frame_render_resolution(frame_idx, step)
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
    return _dispatch_training_debug_present(viewer, encoder, source_tex, debug_width, debug_height, output_width, output_height, source_is_linear=source_is_linear, apply_loss_colorspace=apply_loss_colorspace)
def _render_main_view(viewer: object, encoder: spy.CommandEncoder) -> spy.Texture:
    if viewer.s.trainer is not None and viewer.s.training_renderer is not None:
        session.sync_scene_from_training_renderer(viewer, viewer.s.renderer, target="main")
    out_tex, stats = viewer.s.renderer.render_to_texture(viewer.camera(), background=viewer.s.background, read_stats=True, command_encoder=encoder)
    viewer.s.stats = stats
    return _dispatch_viewport_present(viewer, encoder, out_tex, int(viewer.s.renderer.width), int(viewer.s.renderer.height), int(viewer.s.renderer.width), int(viewer.s.renderer.height), source_is_linear=False)


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

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import slangpy as spy

from ..common import clamp_index, debug_region, require_not_none
from ..training import resolve_maintenance_growth_ratio
from . import session

_DEBUG_HUGE_VALUE = 1e8
_DEBUG_TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_destination
_DEBUG_ABS_DIFF_SCALE_DEFAULT = 1.0
_DEBUG_ABS_DIFF_SCALE_MIN = 0.125
_DEBUG_ABS_DIFF_SCALE_MAX = 64.0
_DEFAULT_TRAINING_STEPS_PER_FRAME = 3
_MAX_TRAINING_STEPS_PER_FRAME = 8
_TRAIN_DOWNSCALE_MODE_AUTO = 0


def _training_schedule_text(viewer: object) -> str:
    if viewer.s.trainer is not None:
        training = viewer.s.trainer.training
        if not bool(training.lr_schedule_enabled):
            return "LR Schedule: disabled"
        current = float(viewer.s.trainer.current_base_lr()) if hasattr(viewer.s.trainer, "current_base_lr") else float(training.lr_schedule_start_lr)
        return (
            f"LR Schedule: cosine {float(training.lr_schedule_start_lr):.2e} -> {float(training.lr_schedule_end_lr):.2e} | "
            f"steps={int(training.lr_schedule_steps):,} | current={current:.2e}"
        )
    if not bool(viewer.c("lr_schedule_enabled").value):
        return "LR Schedule: disabled"
    return (
        f"LR Schedule: cosine {float(viewer.c('lr_schedule_start_lr').value):.2e} -> {float(viewer.c('lr_schedule_end_lr').value):.2e} | "
        f"steps={max(int(viewer.c('lr_schedule_steps').value), 1):,} | current={float(viewer.c('lr_schedule_start_lr').value):.2e}"
    )


def _training_maintenance_text(viewer: object) -> str:
    if viewer.s.trainer is not None:
        training = viewer.s.trainer.training
        current_step = max(int(viewer.s.trainer.state.step), 0)
        target_growth = max(float(training.maintenance_growth_ratio), 0.0)
        current_growth = resolve_maintenance_growth_ratio(training, current_step)
        start_step = max(int(getattr(training, "maintenance_growth_start_step", 0)), 0)
        interval = int(viewer.s.trainer.effective_maintenance_interval()) if hasattr(viewer.s.trainer, "effective_maintenance_interval") else int(training.maintenance_interval)
        return (
            f"Maintenance: every {interval:,} | growth={current_growth * 100.0:.2f}% now | target={target_growth * 100.0:.2f}% after {start_step:,} | "
            f"alpha<{float(training.maintenance_alpha_cull_threshold):.2e} or contrib<{int(training.maintenance_contribution_cull_threshold)} culled | max={int(training.max_gaussians):,}"
        )
    target_growth = max(float(viewer.c("maintenance_growth_ratio").value), 0.0)
    start_step = max(int(viewer.c("maintenance_growth_start_step").value), 0)
    return (
        f"Maintenance: every {max(int(viewer.c('maintenance_interval').value), 1):,} | growth=0.00% now | target={target_growth * 100.0:.2f}% after {start_step:,} | "
        f"alpha<{max(float(viewer.c('maintenance_alpha_cull_threshold').value), 1e-8):.2e} or contrib<{max(int(viewer.c('maintenance_contribution_cull_threshold').value), 0)} culled | max={max(int(viewer.c('max_gaussians').value), 0):,}"
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


def _run_training_batch(viewer: object) -> int:
    if not viewer.s.training_active or viewer.s.trainer is None:
        viewer.s.training_runtime_factor_changed = False
        viewer.s.last_training_batch_steps = 0
        return 0
    factor_before = int(viewer.s.trainer.effective_train_downscale_factor())
    steps = _training_steps_per_frame(viewer)
    if hasattr(viewer.s.trainer, "step_batch"):
        steps = int(viewer.s.trainer.step_batch(steps))
    else:
        for _ in range(steps):
            viewer.s.trainer.step()
    viewer.s.training_runtime_factor_changed = int(viewer.s.trainer.effective_train_downscale_factor()) != factor_before
    viewer.s.last_training_batch_steps = steps
    return steps


def _preview_train_downscale_factor(viewer: object) -> int:
    mode = int(viewer.c("train_downscale_mode").value)
    return max(int(viewer.c("train_auto_start_downscale").value), 1) if mode == _TRAIN_DOWNSCALE_MODE_AUTO else max(mode, 1)


def _training_resolution_text(viewer: object) -> str:
    if viewer.s.training_renderer is not None and viewer.s.trainer is not None:
        factor = max(int(viewer.s.trainer.effective_train_downscale_factor()), 1)
        return f"Train Res: {int(viewer.s.training_renderer.width)}x{int(viewer.s.training_renderer.height)} (N={factor})"
    if viewer.s.training_frames:
        factor = _preview_train_downscale_factor(viewer)
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
        if int(training.train_downscale_mode) == _TRAIN_DOWNSCALE_MODE_AUTO:
            return (
                f"Downscale: Auto | start={int(training.train_auto_start_downscale)}x | "
                f"current={current}x | step {int(viewer.s.trainer.state.step)}/{int(training.train_downscale_max_iters)}"
            )
        return f"Downscale: Manual {current}x"
    mode = int(viewer.c("train_downscale_mode").value)
    if mode == _TRAIN_DOWNSCALE_MODE_AUTO:
        return (
            f"Downscale: Auto | start={max(int(viewer.c('train_auto_start_downscale').value), 1)}x | "
            f"current={_preview_train_downscale_factor(viewer)}x | step 0/{max(int(viewer.c('train_downscale_max_iters').value), 1)}"
        )
    return f"Downscale: Manual {max(mode, 1)}x"


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


def _dispatch_debug_letterbox(viewer: object, encoder: spy.CommandEncoder, source_tex: spy.Texture, source_width: int, source_height: int, output_width: int, output_height: int) -> spy.Texture:
    output = _ensure_texture(viewer, "debug_present_texture", output_width, output_height)
    with debug_region(encoder, "Viewer Debug Letterbox", 151):
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
    viewer.t("training_maintenance").text = _training_maintenance_text(viewer)
    viewer.t("histogram_status").text = str(getattr(viewer.s, "cached_raster_grad_histogram_status", ""))
    viewer.ui._values["_histogram_payload"] = getattr(viewer.s, "cached_raster_grad_histograms", None)
    viewer.ui._values["_histogram_range_payload"] = getattr(viewer.s, "cached_raster_grad_ranges", None)
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
    if tk is None:
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
    training_renderer = require_not_none(viewer.s.training_renderer, "Training renderer is not initialized.")
    debug_width, debug_height = int(training_renderer.width), int(training_renderer.height)
    frame_camera = viewer.s.trainer.make_frame_camera(frame_idx, debug_width, debug_height)
    if _debug_view_key(viewer) != "rendered":
        source_tex, stats = training_renderer.render_to_texture(frame_camera, background=viewer.s.background, read_stats=True, command_encoder=encoder)
        return source_tex, stats, debug_width, debug_height

    debug_renderer = session.ensure_renderer(viewer, "debug_renderer", debug_width, debug_height, allow_debug_overlays=True)
    session.sync_scene_from_training_renderer(viewer, debug_renderer, target="debug")
    source_tex, stats = debug_renderer.render_to_texture(frame_camera, background=viewer.s.background, read_stats=True, command_encoder=encoder)
    return source_tex, stats, debug_width, debug_height


def _render_debug_view(viewer: object, image: spy.Texture, encoder: spy.CommandEncoder, output_width: int, output_height: int) -> None:
    frame_idx = _debug_frame_idx(viewer)
    debug_render_tex, viewer.s.stats, debug_width, debug_height = _render_debug_source(viewer, encoder, frame_idx)
    target_tex = viewer.s.trainer.get_frame_target_texture(frame_idx, native_resolution=False, encoder=encoder)
    source_tex = debug_render_tex if _debug_view_key(viewer) == "rendered" else target_tex if _debug_view_key(viewer) == "target" else _dispatch_debug_abs_diff(viewer, encoder, debug_render_tex, target_tex, debug_width, debug_height)
    encoder.blit(image, _dispatch_debug_letterbox(viewer, encoder, source_tex, debug_width, debug_height, output_width, output_height))




def _render_main_view(viewer: object, image: spy.Texture, encoder: spy.CommandEncoder) -> None:
    if viewer.s.trainer is not None and viewer.s.training_renderer is not None:
        session.sync_scene_from_training_renderer(viewer, viewer.s.renderer, target="main")
    out_tex, stats = viewer.s.renderer.render_to_texture(viewer.camera(), background=viewer.s.background, read_stats=True, command_encoder=encoder)
    viewer.s.stats = stats
    encoder.blit(image, out_tex)


def render_frame(viewer: object, render_context: spy.AppWindow.RenderContext) -> None:
    image, encoder = render_context.surface_texture, render_context.command_encoder
    now = spy.time.perf_counter() if hasattr(spy, "time") else time.perf_counter()
    dt = max(now - viewer.s.last_time, 1e-5)
    viewer.s.last_time = now
    iw, ih = int(image.width), int(image.height)
    try:
        viewer.update_camera(dt)
        if bool(getattr(viewer.s, "pending_training_reinitialize", False)):
            viewer.s.pending_training_reinitialize = False
            session.initialize_training_scene(viewer)
        session.apply_live_params(viewer)
        session.advance_colmap_import(viewer)
        if bool(getattr(viewer.s, "pending_training_runtime_resize", False)):
            session.ensure_training_runtime_resolution(viewer)
        if viewer.s.renderer is None:
            session.recreate_renderer(viewer, iw, ih)
        elif (viewer.s.renderer.width, viewer.s.renderer.height) != (iw, ih):
            session.recreate_renderer(viewer, iw, ih)
        if viewer.s.scene is None:
            encoder.clear_texture_float(image, clear_value=[0.1, 0.1, 0.12, 1.0])
            viewer.s.last_render_exception = ""
            update_ui_text(viewer, dt)
            return
        _run_training_batch(viewer)
        if bool(getattr(viewer.s, "training_runtime_factor_changed", False)):
            session.ensure_training_runtime_resolution(viewer)
        viewer.s.training_runtime_factor_changed = False
        if bool(viewer.c("loss_debug").value) and viewer.s.trainer is not None and viewer.s.training_frames:
            _render_debug_view(viewer, image, encoder, iw, ih)
        else:
            _render_main_view(viewer, image, encoder)
        if bool(viewer.ui._values.get("show_histograms", False)):
            session.refresh_cached_raster_grad_histograms(viewer)
        viewer.s.last_render_exception = ""
    except Exception as exc:
        viewer.s.training_active = False
        viewer.s.last_training_batch_steps = 0
        viewer.s.last_error = str(exc)
        if viewer.s.last_render_exception != viewer.s.last_error:
            print(f"Render/training error: {viewer.s.last_error}")
        viewer.s.last_render_exception = viewer.s.last_error
        encoder.clear_texture_float(image, clear_value=[0.0, 0.0, 0.0, 1.0])
    update_ui_text(viewer, dt)

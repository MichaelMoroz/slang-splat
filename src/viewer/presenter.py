from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import slangpy as spy

from ..common import clamp_index, require_not_none
from . import session

_DEBUG_HUGE_VALUE = 1e8
_DEBUG_TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_destination
_DEFAULT_TRAINING_STEPS_PER_FRAME = 1
_MAX_TRAINING_STEPS_PER_FRAME = 8


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


def _run_training_batch(viewer: object) -> int:
    if not viewer.s.training_active or viewer.s.trainer is None:
        viewer.s.last_training_batch_steps = 0
        return 0
    steps = _training_steps_per_frame(viewer)
    for _ in range(steps):
        viewer.s.trainer.step()
    viewer.s.last_training_batch_steps = steps
    return steps


def _dispatch_debug_abs_diff(viewer: object, encoder: spy.CommandEncoder, rendered_tex: spy.Texture, target_tex: spy.Texture, width: int, height: int) -> spy.Texture:
    output = _ensure_texture(viewer, "loss_debug_texture", width, height)
    require_not_none(viewer.s.debug_abs_diff_kernel, "Debug abs-diff kernel is not initialized.").dispatch(
        thread_count=spy.uint3(int(width), int(height), 1),
        vars={
            "g_DebugRendered": rendered_tex,
            "g_DebugTarget": target_tex,
            "g_DebugOutput": output,
            "g_DebugWidth": int(width),
            "g_DebugHeight": int(height),
            "g_HugeValue": _DEBUG_HUGE_VALUE,
        },
        command_encoder=encoder,
    )
    return output


def _dispatch_debug_letterbox(viewer: object, encoder: spy.CommandEncoder, source_tex: spy.Texture, source_width: int, source_height: int, output_width: int, output_height: int) -> spy.Texture:
    output = _ensure_texture(viewer, "debug_present_texture", output_width, output_height)
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
    viewer.t("images_subdir").text = f"Train images: {viewer.image_subdir_options[clamp_index(int(viewer.c('images_subdir').value), len(viewer.image_subdir_options))]}"
    viewer.t("loss_debug_view").text = f"View: {viewer.loss_debug_view_options[debug_idx][1]}"
    viewer.t("loss_debug_frame").text = f"Frame[{frame_idx}]: {Path(viewer.s.training_frames[frame_idx].image_path).name}" if viewer.s.training_frames else "Frame: <none>"
    viewer.t("path").text = f"Scene: {viewer.s.scene_path.name} [PLY]" if viewer.s.scene_path is not None else f"Scene: {viewer.s.colmap_root.name} [COLMAP]" if viewer.s.colmap_root is not None else "Scene: <none>"
    current_splat_count = viewer.s.trainer.scene.count if viewer.s.trainer is not None else (viewer.s.scene.count if viewer.s.scene is not None else 0)
    viewer.t("scene_stats").text = f"Splats: {int(current_splat_count):,}"
    viewer.t("render_stats").text = "Generated: 0 | Written: 0" if not stats else f"Generated: {int(stats['generated_entries']):,} | Written: {int(stats['written_entries']):,} | Overflow: {bool(stats['overflow'])}{' [cap]' if bool(stats.get('capacity_limited', False)) else ''}{' (delayed)' if bool(stats.get('stats_latency_frames', 0)) else ''}{'' if bool(stats.get('stats_valid', True)) else ' [warming]'}"
    if viewer.s.trainer is None:
        viewer.t("training").text = "Training: not initialized"
        viewer.t("training_loss").text = "Loss Avg: n/a"
        viewer.t("training_mse").text = "MSE: n/a"
        viewer.t("training_instability").text = ""
    else:
        state = viewer.s.trainer.state
        batch_steps = int(getattr(viewer.s, "last_training_batch_steps", 0))
        batch_text = f" | batch={batch_steps}" if viewer.s.training_active else ""
        viewer.t("training").text = f"Training: {'running' if viewer.s.training_active else 'paused'} | step={state.step:,} | frame={state.last_frame_index} | splats={int(current_splat_count):,}{batch_text}"
        viewer.t("training_loss").text = f"Loss Avg: {state.avg_loss:.6e}"
        viewer.t("training_mse").text = f"MSE: {state.last_mse:.6e}" if np.isfinite(state.last_mse) else "MSE: n/a"
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
            if np.isfinite(state.last_mse) and state.last_mse > 0:
                tk.tk.mse_history.append(float(state.last_mse))
            elif tk.tk.mse_history:
                tk.tk.mse_history.append(tk.tk.mse_history[-1])


def _render_debug_view(viewer: object, image: spy.Texture, encoder: spy.CommandEncoder, output_width: int, output_height: int) -> None:
    frame_idx = _debug_frame_idx(viewer)
    debug_renderer = require_not_none(viewer.s.training_renderer, "Training renderer is not initialized.")
    debug_width, debug_height = int(debug_renderer.width), int(debug_renderer.height)
    debug_render_tex, viewer.s.stats = debug_renderer.render_to_texture(viewer.s.trainer.make_frame_camera(frame_idx, debug_width, debug_height), background=viewer.s.background, read_stats=True, command_encoder=encoder)
    target_tex = viewer.s.trainer.get_frame_target_texture(frame_idx, native_resolution=False)
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
    viewer.update_camera(dt)
    session.apply_live_params(viewer)
    iw, ih = int(image.width), int(image.height)
    if (viewer.s.renderer.width, viewer.s.renderer.height) != (iw, ih):
        session.recreate_renderer(viewer, iw, ih)
    if viewer.s.scene is None:
        encoder.clear_texture_float(image, clear_value=[0.1, 0.1, 0.12, 1.0])
        update_ui_text(viewer, dt)
        return
    try:
        _run_training_batch(viewer)
        if bool(viewer.c("loss_debug").value) and viewer.s.trainer is not None and viewer.s.training_frames:
            _render_debug_view(viewer, image, encoder, iw, ih)
        else:
            _render_main_view(viewer, image, encoder)
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

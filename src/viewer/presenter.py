from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import slangpy as spy

from . import session

_DEBUG_HUGE_VALUE = 1e8
_DEBUG_TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_destination
_debug_frame_idx = lambda viewer: int(np.clip(int(viewer.c("loss_debug_frame").value), 0, max(len(viewer.s.training_frames) - 1, 0)))
_debug_view_key = lambda viewer: viewer.loss_debug_view_options[int(np.clip(int(viewer.c("loss_debug_view").value), 0, len(viewer.loss_debug_view_options) - 1))][0]
_require = lambda value, message: value if value is not None else (_ for _ in ()).throw(RuntimeError(message))
_ensure_texture = lambda viewer, attr, width, height: (lambda texture: texture if texture is not None and int(texture.width) == int(width) and int(texture.height) == int(height) else (lambda created: (setattr(viewer.s, attr, created), created)[1])(viewer.device.create_texture(format=spy.Format.rgba32_float, width=int(width), height=int(height), usage=_DEBUG_TEXTURE_USAGE)))(getattr(viewer.s, attr))
_dispatch_debug_abs_diff = lambda viewer, encoder, rendered_tex, target_tex, width, height: (lambda output: (_require(viewer.s.debug_abs_diff_kernel, "Debug abs-diff kernel is not initialized.").dispatch(thread_count=spy.uint3(int(width), int(height), 1), vars={"g_DebugRendered": rendered_tex, "g_DebugTarget": target_tex, "g_DebugOutput": output, "g_DebugWidth": int(width), "g_DebugHeight": int(height), "g_Stability": {"gradComponentClip": 0.0, "gradNormClip": 0.0, "maxUpdate": 0.0, "minScale": 0.0, "maxScale": 0.0, "minOpacity": 0.0, "maxOpacity": 0.0, "positionAbsMax": 0.0, "hugeValue": _DEBUG_HUGE_VALUE}}, command_encoder=encoder), output)[1])(_ensure_texture(viewer, "loss_debug_texture", width, height))
_dispatch_debug_letterbox = lambda viewer, encoder, source_tex, source_width, source_height, output_width, output_height: (lambda output: (_require(viewer.s.debug_letterbox_kernel, "Debug letterbox kernel is not initialized.").dispatch(thread_count=spy.uint3(int(output_width), int(output_height), 1), vars={"g_LetterboxSource": source_tex, "g_LetterboxOutput": output, "g_LetterboxSourceWidth": int(source_width), "g_LetterboxSourceHeight": int(source_height), "g_LetterboxOutputWidth": int(output_width), "g_LetterboxOutputHeight": int(output_height)}, command_encoder=encoder), output)[1])(_ensure_texture(viewer, "debug_present_texture", output_width, output_height))


def update_ui_text(viewer: object, dt: float) -> None:
    viewer.s.fps_smooth += (1.0 / max(dt, 1e-5) - viewer.s.fps_smooth) * min(dt * 5.0, 1.0)
    session.update_debug_frame_slider_range(viewer)
    frame_idx = _debug_frame_idx(viewer)
    debug_idx = int(np.clip(int(viewer.c("loss_debug_view").value), 0, len(viewer.loss_debug_view_options) - 1))
    stats = viewer.s.stats
    viewer.t("fps").text = f"FPS: {viewer.s.fps_smooth:.1f}"
    viewer.t("images_subdir").text = f"Train images: {viewer.image_subdir_options[int(np.clip(int(viewer.c('images_subdir').value), 0, len(viewer.image_subdir_options) - 1))]}"
    viewer.t("loss_debug_view").text = f"View: {viewer.loss_debug_view_options[debug_idx][1]}"
    viewer.t("loss_debug_frame").text = f"Frame[{frame_idx}]: {Path(viewer.s.training_frames[frame_idx].image_path).name}" if viewer.s.training_frames else "Frame: <none>"
    viewer.t("path").text = f"Scene: {viewer.s.scene_path.name} [PLY]" if viewer.s.scene_path is not None else f"Scene: {viewer.s.colmap_root.name} [COLMAP]" if viewer.s.colmap_root is not None else "Scene: <none>"
    current_splat_count = viewer.s.trainer.scene.count if viewer.s.trainer is not None else (viewer.s.scene.count if viewer.s.scene is not None else 0)
    viewer.t("scene_stats").text = f"Splats: {int(current_splat_count):,}"
    viewer.t("render_stats").text = "Generated: 0 | Written: 0" if not stats else f"Generated: {int(stats['generated_entries']):,} | Written: {int(stats['written_entries']):,} | Overflow: {bool(stats['overflow'])}{' [cap]' if bool(stats.get('capacity_limited', False)) else ''}{' (delayed)' if bool(stats.get('stats_latency_frames', 0)) else ''}{'' if bool(stats.get('stats_valid', True)) else ' [warming]'}"
    if viewer.s.trainer is None:
        viewer.t("training").text = "Training: not initialized"
        viewer.t("training_ssim").text = "SSIM Avg: n/a"
        viewer.t("training_psnr").text = "PSNR Avg: n/a"
        viewer.t("training_loss").text = "Loss Avg: n/a"
        viewer.t("training_instability").text = ""
    else:
        state = viewer.s.trainer.state
        avg_ssim = f"{state.avg_ssim:.4f}" if np.isfinite(state.avg_ssim) else "n/a"
        avg_psnr = f"{state.avg_psnr:.2f} dB" if np.isfinite(state.avg_psnr) else "n/a"
        viewer.t("training").text = f"Training: {'running' if viewer.s.training_active else 'paused'} | step={state.step:,} | frame={state.last_frame_index} | splats={int(current_splat_count):,}"
        viewer.t("training_ssim").text = f"SSIM Avg: {avg_ssim}"
        viewer.t("training_psnr").text = f"PSNR Avg: {avg_psnr}"
        viewer.t("training_loss").text = f"Loss Avg: {state.avg_loss:.6e}"
        viewer.t("training_instability").text = state.last_instability
    viewer.t("error").text = f"Error: {viewer.s.last_error}" if viewer.s.last_error else ""


def _render_debug_view(viewer: object, image: spy.Texture, encoder: spy.CommandEncoder, output_width: int, output_height: int) -> None:
    frame_idx = _debug_frame_idx(viewer)
    debug_width, debug_height = viewer.s.trainer.frame_size(frame_idx)
    debug_renderer = session.ensure_renderer(viewer, "debug_renderer", debug_width, debug_height, allow_debug_overlays=False)
    session.sync_scene_from_training_renderer(viewer, debug_renderer, target="debug")
    debug_render_tex, viewer.s.stats = debug_renderer.render_to_texture(viewer.s.trainer.make_frame_camera(frame_idx, debug_width, debug_height), background=viewer.s.background, read_stats=True)
    target_tex = viewer.s.trainer.get_frame_target_texture(frame_idx, native_resolution=True)
    source_tex = debug_render_tex if _debug_view_key(viewer) == "rendered" else target_tex if _debug_view_key(viewer) == "target" else _dispatch_debug_abs_diff(viewer, encoder, debug_render_tex, target_tex, debug_width, debug_height)
    encoder.blit(image, _dispatch_debug_letterbox(viewer, encoder, source_tex, debug_width, debug_height, output_width, output_height))


_render_main_view = lambda viewer, image, encoder: ((session.sync_scene_from_training_renderer(viewer, viewer.s.renderer, target="main") if viewer.s.trainer is not None and viewer.s.training_renderer is not None else None), (lambda out_tex, stats: (setattr(viewer.s, "stats", stats), encoder.blit(image, out_tex)))(*viewer.s.renderer.render_to_texture(viewer.camera(), background=viewer.s.background, read_stats=True)))


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
        if viewer.s.training_active and viewer.s.trainer is not None:
            viewer.s.trainer.step()
        if bool(viewer.c("loss_debug").value) and viewer.s.trainer is not None and viewer.s.training_frames:
            _render_debug_view(viewer, image, encoder, iw, ih)
        else:
            _render_main_view(viewer, image, encoder)
        viewer.s.last_render_exception = ""
    except Exception as exc:
        viewer.s.training_active = False
        viewer.s.last_error = str(exc)
        if viewer.s.last_render_exception != viewer.s.last_error:
            print(f"Render/training error: {viewer.s.last_error}")
        viewer.s.last_render_exception = viewer.s.last_error
        encoder.clear_texture_float(image, clear_value=[0.0, 0.0, 0.0, 1.0])
    update_ui_text(viewer, dt)

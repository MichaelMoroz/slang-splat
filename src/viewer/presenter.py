from __future__ import annotations

from pathlib import Path

import numpy as np
import slangpy as spy

from . import session


def _ensure_texture(viewer: object, attr: str, width: int, height: int) -> spy.Texture:
    texture = getattr(viewer.s, attr)
    if texture is not None and int(texture.width) == int(width) and int(texture.height) == int(height):
        return texture
    texture = viewer.device.create_texture(
        format=spy.Format.rgba32_float,
        width=int(width),
        height=int(height),
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_destination,
    )
    setattr(viewer.s, attr, texture)
    return texture


def dispatch_debug_abs_diff(viewer: object, encoder: spy.CommandEncoder, rendered_tex: spy.Texture, target_tex: spy.Texture, width: int, height: int) -> spy.Texture:
    if viewer.s.debug_abs_diff_kernel is None:
        raise RuntimeError("Debug abs-diff kernel is not initialized.")
    out_tex = _ensure_texture(viewer, "loss_debug_texture", width, height)
    viewer.s.debug_abs_diff_kernel.dispatch(
        thread_count=spy.uint3(int(width), int(height), 1),
        vars={
            "g_DebugRendered": rendered_tex,
            "g_DebugTarget": target_tex,
            "g_DebugOutput": out_tex,
            "g_DebugWidth": int(width),
            "g_DebugHeight": int(height),
            "g_Stability": {
                "gradComponentClip": 0.0,
                "gradNormClip": 0.0,
                "maxUpdate": 0.0,
                "minScale": 0.0,
                "maxScale": 0.0,
                "minOpacity": 0.0,
                "maxOpacity": 0.0,
                "positionAbsMax": 0.0,
                "hugeValue": 1e8,
            },
        },
        command_encoder=encoder,
    )
    return out_tex


def dispatch_debug_letterbox(viewer: object, encoder: spy.CommandEncoder, source_tex: spy.Texture, source_width: int, source_height: int, output_width: int, output_height: int) -> spy.Texture:
    if viewer.s.debug_letterbox_kernel is None:
        raise RuntimeError("Debug letterbox kernel is not initialized.")
    out_tex = _ensure_texture(viewer, "debug_present_texture", output_width, output_height)
    viewer.s.debug_letterbox_kernel.dispatch(
        thread_count=spy.uint3(int(output_width), int(output_height), 1),
        vars={
            "g_LetterboxSource": source_tex,
            "g_LetterboxOutput": out_tex,
            "g_LetterboxSourceWidth": int(source_width),
            "g_LetterboxSourceHeight": int(source_height),
            "g_LetterboxOutputWidth": int(output_width),
            "g_LetterboxOutputHeight": int(output_height),
        },
        command_encoder=encoder,
    )
    return out_tex


def update_ui_text(viewer: object, dt: float) -> None:
    viewer.s.fps_smooth += (1.0 / max(dt, 1e-5) - viewer.s.fps_smooth) * min(dt * 5.0, 1.0)
    viewer.t("fps").text = f"FPS: {viewer.s.fps_smooth:.1f}"
    image_subdir_idx = int(np.clip(int(viewer.c("images_subdir").value), 0, len(viewer.image_subdir_options) - 1))
    viewer.t("images_subdir").text = f"Train images: {viewer.image_subdir_options[image_subdir_idx]}"
    session.update_debug_frame_slider_range(viewer)
    debug_view_idx = int(np.clip(int(viewer.c("loss_debug_view").value), 0, len(viewer.loss_debug_view_options) - 1))
    _, debug_label = viewer.loss_debug_view_options[debug_view_idx]
    viewer.t("loss_debug_view").text = f"View: {debug_label}"
    if viewer.s.training_frames:
        frame_idx = int(np.clip(int(viewer.c("loss_debug_frame").value), 0, len(viewer.s.training_frames) - 1))
        viewer.t("loss_debug_frame").text = f"Frame[{frame_idx}]: {Path(viewer.s.training_frames[frame_idx].image_path).name}"
    else:
        viewer.t("loss_debug_frame").text = "Frame: <none>"
    viewer.t("path").text = (
        f"Scene: {viewer.s.scene_path.name} [PLY]" if viewer.s.scene_path is not None else
        f"Scene: {viewer.s.colmap_root.name} [COLMAP]" if viewer.s.colmap_root is not None else
        "Scene: <none>"
    )
    viewer.t("scene_stats").text = f"Splats: {(viewer.s.scene.count if viewer.s.scene is not None else 0):,}"
    viewer.t("render_stats").text = (
        "Generated: 0 | Written: 0"
        if not viewer.s.stats else
        f"Generated: {int(viewer.s.stats['generated_entries']):,} | "
        f"Written: {int(viewer.s.stats['written_entries']):,} | "
        f"Overflow: {bool(viewer.s.stats['overflow'])}"
        f"{' [cap]' if bool(viewer.s.stats.get('capacity_limited', False)) else ''}"
        f"{' (delayed)' if bool(viewer.s.stats.get('stats_latency_frames', 0)) else ''}"
        f"{'' if bool(viewer.s.stats.get('stats_valid', True)) else ' [warming]'}"
    )
    if viewer.s.trainer is None:
        viewer.t("training").text = "Training: not initialized"
        viewer.t("training_loss").text = "Loss: n/a"
    else:
        state = viewer.s.trainer.state
        viewer.t("training").text = f"Training: {'running' if viewer.s.training_active else 'paused'} | step={state.step:,} | frame={state.last_frame_index}"
        psnr_text = f"{state.ema_psnr:.2f} dB" if np.isfinite(state.ema_psnr) else "n/a"
        viewer.t("training_loss").text = f"Loss: {state.last_loss:.6e} | EMA: {state.ema_loss:.6e} | PSNR: {psnr_text} | {state.last_instability}"
    viewer.t("error").text = f"Error: {viewer.s.last_error}" if viewer.s.last_error else ""


def _render_debug_view(viewer: object, image: spy.Texture, encoder: spy.CommandEncoder, output_width: int, output_height: int) -> None:
    frame_idx = int(np.clip(int(viewer.c("loss_debug_frame").value), 0, len(viewer.s.training_frames) - 1))
    debug_width, debug_height = viewer.s.trainer.frame_size(frame_idx)
    debug_renderer = session.ensure_renderer(viewer, "debug_renderer", debug_width, debug_height, allow_debug_overlays=False)
    session.sync_scene_from_training_renderer(viewer, debug_renderer, target="debug")
    debug_render_tex, stats = debug_renderer.render_to_texture(
        viewer.s.trainer.make_frame_camera(frame_idx, debug_width, debug_height),
        background=viewer.s.background,
        read_stats=True,
    )
    target_tex = viewer.s.trainer.get_frame_target_texture(frame_idx, native_resolution=True)
    viewer.s.stats = stats
    debug_view_idx = int(np.clip(int(viewer.c("loss_debug_view").value), 0, len(viewer.loss_debug_view_options) - 1))
    debug_view_key, _ = viewer.loss_debug_view_options[debug_view_idx]
    source_tex = (
        debug_render_tex if debug_view_key == "rendered" else
        target_tex if debug_view_key == "target" else
        dispatch_debug_abs_diff(viewer, encoder, debug_render_tex, target_tex, debug_width, debug_height)
    )
    encoder.blit(
        image,
        dispatch_debug_letterbox(viewer, encoder, source_tex, debug_width, debug_height, output_width, output_height),
    )


def _render_main_view(viewer: object, image: spy.Texture, encoder: spy.CommandEncoder) -> None:
    if viewer.s.trainer is not None and viewer.s.training_renderer is not None:
        session.sync_scene_from_training_renderer(viewer, viewer.s.renderer, target="main")
    out_tex, viewer.s.stats = viewer.s.renderer.render_to_texture(viewer.camera(), background=viewer.s.background, read_stats=True)
    encoder.blit(image, out_tex)


def render_frame(viewer: object, render_context: spy.AppWindow.RenderContext) -> None:
    image = render_context.surface_texture
    encoder = render_context.command_encoder
    now = spy.time.perf_counter() if hasattr(spy, "time") else None
    now = now or __import__("time").perf_counter()
    dt = max(now - viewer.s.last_time, 1e-5)
    viewer.s.last_time = now
    viewer.update_camera(dt)
    session.apply_render_params(viewer)
    session.apply_training_params(viewer)
    session.apply_dataset_init_defaults(viewer)
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

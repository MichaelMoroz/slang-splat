from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import slangpy as spy

from . import session


def update_ui_text(viewer: object, dt: float) -> None:
    viewer.s.fps_smooth += (1.0 / max(dt, 1e-5) - viewer.s.fps_smooth) * min(dt * 5.0, 1.0)
    session.update_debug_frame_slider_range(viewer)
    frame_idx = int(np.clip(int(viewer.c("loss_debug_frame").value), 0, max(len(viewer.s.training_frames) - 1, 0)))
    debug_idx = int(np.clip(int(viewer.c("loss_debug_view").value), 0, len(viewer.loss_debug_view_options) - 1))
    stats = viewer.s.stats
    viewer.t("fps").text = f"FPS: {viewer.s.fps_smooth:.1f}"
    viewer.t("images_subdir").text = f"Train images: {viewer.image_subdir_options[int(np.clip(int(viewer.c('images_subdir').value), 0, len(viewer.image_subdir_options) - 1))]}"
    viewer.t("loss_debug_view").text = f"View: {viewer.loss_debug_view_options[debug_idx][1]}"
    viewer.t("loss_debug_frame").text = f"Frame[{frame_idx}]: {Path(viewer.s.training_frames[frame_idx].image_path).name}" if viewer.s.training_frames else "Frame: <none>"
    viewer.t("path").text = f"Scene: {viewer.s.scene_path.name} [PLY]" if viewer.s.scene_path is not None else f"Scene: {viewer.s.colmap_root.name} [COLMAP]" if viewer.s.colmap_root is not None else "Scene: <none>"
    viewer.t("scene_stats").text = f"Splats: {(viewer.s.scene.count if viewer.s.scene is not None else 0):,}"
    viewer.t("render_stats").text = "Generated: 0 | Written: 0" if not stats else f"Generated: {int(stats['generated_entries']):,} | Written: {int(stats['written_entries']):,} | Overflow: {bool(stats['overflow'])}{' [cap]' if bool(stats.get('capacity_limited', False)) else ''}{' (delayed)' if bool(stats.get('stats_latency_frames', 0)) else ''}{'' if bool(stats.get('stats_valid', True)) else ' [warming]'}"
    if viewer.s.trainer is None:
        viewer.t("training").text, viewer.t("training_loss").text = "Training: not initialized", "Loss: n/a"
    else:
        state = viewer.s.trainer.state
        psnr_text = f"{state.ema_psnr:.2f} dB" if np.isfinite(state.ema_psnr) else "n/a"
        viewer.t("training").text = f"Training: {'running' if viewer.s.training_active else 'paused'} | step={state.step:,} | frame={state.last_frame_index}"
        viewer.t("training_loss").text = f"Loss: {state.last_loss:.6e} | EMA: {state.ema_loss:.6e} | PSNR: {psnr_text} | {state.last_instability}"
    viewer.t("error").text = f"Error: {viewer.s.last_error}" if viewer.s.last_error else ""


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
        if viewer.s.trainer is not None and viewer.s.training_renderer is not None:
            session.sync_scene_from_training_renderer(viewer, viewer.s.renderer, target="main")
        out_tex, viewer.s.stats = viewer.s.renderer.render_to_texture(viewer.camera(), background=viewer.s.background, read_stats=True)
        encoder.blit(image, out_tex)
        viewer.s.last_render_exception = ""
    except Exception as exc:
        viewer.s.training_active = False
        viewer.s.last_error = str(exc)
        if viewer.s.last_render_exception != viewer.s.last_error:
            print(f"Render/training error: {viewer.s.last_error}")
        viewer.s.last_render_exception = viewer.s.last_error
        encoder.clear_texture_float(image, clear_value=[0.0, 0.0, 0.0, 1.0])
    update_ui_text(viewer, dt)

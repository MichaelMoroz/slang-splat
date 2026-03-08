from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import slangpy as spy

from . import session
from .state import DEFAULT_IMAGE_SUBDIR_INDEX, IMAGE_SUBDIR_OPTIONS, LOSS_DEBUG_OPTIONS


@dataclass(frozen=True, slots=True)
class ControlSpec:
    key: str
    kind: str
    label: str
    kwargs: dict[str, object]


@dataclass(slots=True)
class ViewerUI:
    texts: dict[str, object]
    controls: dict[str, object]
    __getitem__ = lambda self, key: self.controls[key]
    text = lambda self, key: self.texts[key]


WIDGETS = {
    "slider_float": spy.ui.SliderFloat,
    "slider_int": spy.ui.SliderInt,
    "input_float": spy.ui.InputFloat,
    "checkbox": spy.ui.CheckBox,
    "text": spy.ui.Text,
}


GROUP_SPECS = {
    "Main": (
        ControlSpec("images_subdir", "slider_int", "Image Dir", {"value": DEFAULT_IMAGE_SUBDIR_INDEX, "min": 0, "max": len(IMAGE_SUBDIR_OPTIONS) - 1}),
        ControlSpec("loss_debug", "checkbox", "Visual Loss Debug", {"value": False}),
        ControlSpec("loss_debug_view", "slider_int", "Debug View", {"value": 2, "min": 0, "max": len(LOSS_DEBUG_OPTIONS) - 1}),
        ControlSpec("loss_debug_frame", "slider_int", "Debug Frame", {"value": 0, "min": 0, "max": 10000}),
    ),
    "Camera": (
        ControlSpec("move_speed", "slider_float", "Move Speed", {"value": 2.0, "min": 0.1, "max": 20.0, "flags": spy.ui.SliderFlags.logarithmic, "format": "%.3g"}),
        ControlSpec("fov", "slider_float", "FOV", {"value": 60.0, "min": 25.0, "max": 100.0}),
    ),
    "Train Setup": (
        ControlSpec("max_gaussians", "slider_int", "Max Gaussians", {"value": 50000, "min": 1000, "max": 2000000, "flags": spy.ui.SliderFlags.logarithmic}),
        ControlSpec("seed", "slider_int", "Shuffle Seed", {"value": 1234, "min": 0, "max": 1000000}),
        ControlSpec("init_opacity", "input_float", "Init Opacity", {"value": 0.5, "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
    ),
    "Train Optimizer": (
        ControlSpec("lr_base", "input_float", "Base LR", {"value": 1e-3, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("lr_pos_mul", "input_float", "LR Mul Position", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_scale_mul", "input_float", "LR Mul Scale", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_rot_mul", "input_float", "LR Mul Rotation", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_color_mul", "input_float", "LR Mul Color", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_opacity_mul", "input_float", "LR Mul Opacity", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("beta1", "input_float", "Beta1", {"value": 0.9, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ControlSpec("beta2", "input_float", "Beta2", {"value": 0.999, "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ControlSpec("eps", "input_float", "Adam Eps", {"value": 1e-8, "step": 1e-9, "step_fast": 1e-8, "format": "%.10f"}),
        ControlSpec("scale_l2", "input_float", "Scale L2", {"value": 1e-3, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("lambda_dssim", "input_float", "Lambda DSSIM", {"value": 0.2, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ControlSpec("max_anisotropy", "input_float", "Max Anisotropy", {"value": 10.0, "step": 0.1, "step_fast": 0.5, "format": "%.6f"}),
        ControlSpec("mcmc_pos_noise_enabled", "checkbox", "MCMC Pos Noise", {"value": False}),
        ControlSpec("mcmc_pos_noise_scale", "input_float", "MCMC Noise Scale", {"value": 1.0, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ControlSpec("mcmc_opacity_k", "input_float", "MCMC Opacity K", {"value": 100.0, "step": 0.5, "step_fast": 5.0, "format": "%.4f"}),
        ControlSpec("mcmc_opacity_t", "input_float", "MCMC Opacity T", {"value": 0.995, "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ControlSpec("grad_clip", "input_float", "Grad Clip", {"value": 10.0, "step": 0.1, "step_fast": 1.0, "format": "%.4f"}),
        ControlSpec("grad_norm_clip", "input_float", "Grad Norm Clip", {"value": 10.0, "step": 0.1, "step_fast": 1.0, "format": "%.4f"}),
        ControlSpec("max_update", "input_float", "Max Update", {"value": 0.05, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
    ),
    "Train Stability": (
        ControlSpec("min_scale", "input_float", "Min Scale", {"value": 1e-3, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("max_scale", "input_float", "Max Scale", {"value": 3.0, "step": 1e-2, "step_fast": 0.1, "format": "%.5f"}),
        ControlSpec("min_opacity", "input_float", "Min Opacity", {"value": 1e-4, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("max_opacity", "input_float", "Max Opacity", {"value": 0.9999, "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ControlSpec("position_abs_max", "input_float", "Pos Abs Max", {"value": 1e4, "step": 10.0, "step_fast": 100.0, "format": "%.3f"}),
        ControlSpec("train_near", "input_float", "Train Near", {"value": 0.1, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ControlSpec("train_far", "input_float", "Train Far", {"value": 120.0, "step": 1.0, "step_fast": 10.0, "format": "%.3f"}),
    ),
    "Train Density": (
        ControlSpec("densify_from_iter", "input_float", "Densify From", {"value": 500.0, "step": 1.0, "step_fast": 10.0, "format": "%.0f"}),
        ControlSpec("densify_until_iter", "input_float", "Densify Until", {"value": 15000.0, "step": 10.0, "step_fast": 100.0, "format": "%.0f"}),
        ControlSpec("densification_interval", "input_float", "Densify Every", {"value": 100.0, "step": 1.0, "step_fast": 10.0, "format": "%.0f"}),
        ControlSpec("densify_grad_threshold", "input_float", "Grad Threshold", {"value": 2e-4, "step": 1e-5, "step_fast": 1e-4, "format": "%.6f"}),
        ControlSpec("percent_dense", "input_float", "Percent Dense", {"value": 0.01, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ControlSpec("prune_min_opacity", "input_float", "Prune Min Opacity", {"value": 0.005, "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ControlSpec("screen_size_prune_threshold", "input_float", "Screen Prune Px", {"value": 20.0, "step": 0.5, "step_fast": 5.0, "format": "%.3f"}),
        ControlSpec("world_size_prune_ratio", "input_float", "World Prune Ratio", {"value": 0.1, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ControlSpec("opacity_reset_interval", "input_float", "Opacity Reset", {"value": 3000.0, "step": 10.0, "step_fast": 100.0, "format": "%.0f"}),
    ),
}


def default_control_values(*group_names: str) -> dict[str, object]:
    groups = GROUP_SPECS.values() if not group_names else (GROUP_SPECS[name] for name in group_names)
    return {spec.key: spec.kwargs["value"] for specs in groups for spec in specs if "value" in spec.kwargs}


def _build_group(panel: object, title: str, specs: tuple[ControlSpec, ...], controls: dict[str, object]) -> object:
    group = spy.ui.Group(panel, title)
    for spec in specs:
        controls[spec.key] = WIDGETS[spec.kind](group, spec.label, **spec.kwargs)
    return group


def build_ui(screen: object, app: object, renderer: object) -> ViewerUI:
    texts: dict[str, object] = {}
    controls: dict[str, object] = {}
    panel = spy.ui.Window(screen, "Splat Viewer + Trainer", size=spy.float2(520, 760))
    for key, value in {
        "fps": "FPS: 0.0",
        "path": "Scene: <none>",
        "scene_stats": "Splats: 0",
        "render_stats": "Generated: 0 | Written: 0",
        "training": "Training: idle",
        "training_ssim": "SSIM: n/a",
        "training_psnr": "PSNR: n/a",
        "training_loss": "Loss: n/a",
        "training_instability": "",
        "error": "",
    }.items():
        texts[key] = spy.ui.Text(panel, value)
    main_group = _build_group(panel, "Main", GROUP_SPECS["Main"], controls)
    for label, callback in (
        ("Load PLY...", lambda: (lambda path: session.load_scene(app, Path(path)) if path else None)(spy.platform.open_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")]))),
        ("Load COLMAP...", lambda: (lambda path: session.load_colmap_dataset(app, Path(path), app._selected_images_subdir()) if path else None)(spy.platform.choose_folder_dialog())),
        ("Reload", lambda: session.load_scene(app, app.s.scene_path) if app.s.scene_path is not None else session.load_colmap_dataset(app, app.s.colmap_root, app._selected_images_subdir()) if app.s.colmap_root is not None else None),
        ("Reinitialize Gaussians", lambda: session.initialize_training_scene(app)),
        ("Start Training", lambda: session.set_training_active(app, True)),
        ("Stop Training", lambda: session.set_training_active(app, False)),
    ):
        spy.ui.Button(main_group, label, callback=callback)
    texts["images_subdir"] = spy.ui.Text(main_group, f"Train images: {IMAGE_SUBDIR_OPTIONS[DEFAULT_IMAGE_SUBDIR_INDEX]}")
    texts["loss_debug_view"] = spy.ui.Text(main_group, "View: Abs Diff")
    texts["loss_debug_frame"] = spy.ui.Text(main_group, "Frame: <none>")
    _build_group(panel, "Camera", GROUP_SPECS["Camera"], controls)
    setup_group = _build_group(panel, "Train Setup", GROUP_SPECS["Train Setup"], controls)
    texts["setup_hint"] = spy.ui.Text(setup_group, "COLMAP init uses direct points + NN scales")
    _build_group(panel, "Train Optimizer", GROUP_SPECS["Train Optimizer"], controls)
    stab_group = _build_group(panel, "Train Stability", GROUP_SPECS["Train Stability"], controls)
    texts["stability_hint"] = spy.ui.Text(stab_group, "Hard clamps are applied after every ADAM step")
    density_group = _build_group(panel, "Train Density", GROUP_SPECS["Train Density"], controls)
    texts["density_hint"] = spy.ui.Text(density_group, "Clone/split/prune + opacity reset schedule")
    params_group = spy.ui.Group(panel, "Render Params")
    for spec in (
        ControlSpec("radius_scale", "slider_float", "Radius Scale", {"value": float(renderer.radius_scale), "min": 0.5, "max": 4.0, "flags": spy.ui.SliderFlags.logarithmic, "format": "%.3g"}),
        ControlSpec("alpha_cutoff", "slider_float", "Alpha Cutoff", {"value": float(renderer.alpha_cutoff), "min": 0.0001, "max": 0.1, "flags": spy.ui.SliderFlags.logarithmic, "format": "%.2e"}),
        ControlSpec("max_splat_steps", "slider_int", "Max Splat Steps", {"value": int(renderer.max_splat_steps), "min": 16, "max": 32768}),
        ControlSpec("trans_threshold", "slider_float", "Trans Threshold", {"value": float(renderer.transmittance_threshold), "min": 0.001, "max": 0.2, "flags": spy.ui.SliderFlags.logarithmic, "format": "%.2e"}),
        ControlSpec("sampled5_safety", "slider_float", "MVEE Safety", {"value": float(renderer.sampled5_safety_scale), "min": 1.0, "max": 1.2}),
        ControlSpec("debug_ellipse", "checkbox", "Debug Ellipse Outlines", {"value": bool(renderer.debug_show_ellipses)}),
        ControlSpec("debug_processed_count", "checkbox", "Debug Processed Count", {"value": bool(renderer.debug_show_processed_count)}),
    ):
        controls[spec.key] = WIDGETS[spec.kind](params_group, spec.label, **spec.kwargs)
    spy.ui.Text(panel, "Controls: LMB drag=look | WASDQE=move | Wheel=speed")
    return ViewerUI(texts=texts, controls=controls)

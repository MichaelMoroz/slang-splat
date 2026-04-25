from __future__ import annotations

from src.renderer import renderer_context


def test_render_settings_forward_debug_overlays_to_renderer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _RendererStub:
        CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED = "fixed"

        @staticmethod
        def _validate_cached_raster_grad_atomic_mode(mode: str) -> str:
            return str(mode).strip().lower()

        def __init__(self, device, width: int, height: int, **kwargs) -> None:
            captured["device"] = device
            captured["width"] = width
            captured["height"] = height
            captured["kwargs"] = kwargs

    monkeypatch.setattr(renderer_context, "GaussianRenderer", _RendererStub)

    settings = renderer_context.GaussianRenderSettings(
        width=64,
        height=32,
        debug_show_ellipses=True,
        debug_show_processed_count=True,
        debug_show_grad_norm=True,
    )
    renderer = settings.create_renderer(device="stub-device")

    assert isinstance(renderer, _RendererStub)
    assert captured["device"] == "stub-device"
    assert captured["width"] == 64
    assert captured["height"] == 32
    assert captured["kwargs"] == {
        "radius_scale": 1.0,
        "alpha_cutoff": 1.0 / 255.0,
        "max_anisotropy": 32.0,
        "transmittance_threshold": 0.005,
        "list_capacity_multiplier": 64,
        "max_prepass_memory_mb": 4096,
        "cached_raster_grad_atomic_mode": "fixed",
        "cached_raster_grad_fixed_ro_local_range": 2.0,
        "cached_raster_grad_fixed_scale_range": 256.0,
        "cached_raster_grad_fixed_color_range": 8.0,
        "cached_raster_grad_fixed_opacity_range": 8.0,
        "debug_grad_norm_threshold": 2e-4,
        "debug_ellipse_thickness_px": 4.0,
        "debug_gaussian_scale_multiplier": 1.0,
        "debug_min_opacity": 0.0,
        "debug_opacity_multiplier": 1.0,
        "debug_ellipse_scale_multiplier": 1.0,
        "debug_splat_age_range": (0.0, 1.0),
        "debug_density_range": (0.0, 20.0),
        "debug_contribution_range": (0.001, 1.0),
        "debug_adam_momentum_range": (0.0, 0.1),
        "debug_depth_mean_range": (0.0, 10.0),
        "debug_depth_std_range": (0.0, 0.5),
        "debug_depth_local_mismatch_range": (0.0, 0.5),
        "debug_depth_local_mismatch_smooth_radius": 2.0,
        "debug_depth_local_mismatch_reject_radius": 4.0,
        "debug_sh_coeff_index": 0,
        "debug_show_ellipses": True,
        "debug_show_processed_count": True,
        "debug_show_grad_norm": True,
    }

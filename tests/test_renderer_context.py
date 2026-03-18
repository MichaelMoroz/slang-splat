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
    assert captured["kwargs"]["debug_show_ellipses"] is True
    assert captured["kwargs"]["debug_show_processed_count"] is True
    assert captured["kwargs"]["debug_show_grad_norm"] is True
    assert "sampled5_mvee_iters" not in captured["kwargs"]
    assert "sampled5_safety_scale" not in captured["kwargs"]
    assert "sampled5_radius_pad_px" not in captured["kwargs"]
    assert "sampled5_eps" not in captured["kwargs"]

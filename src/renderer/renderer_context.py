from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import slangpy as spy

from .render_params import RendererParams, runtime_renderer_params
from .gaussian_renderer import GaussianRenderer


@dataclass(frozen=True, slots=True)
class GaussianRenderSettings:
    width: int
    height: int
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    params: RendererParams = field(default_factory=runtime_renderer_params)

    def __post_init__(self) -> None:
        width = max(int(self.width), 1)
        height = max(int(self.height), 1)
        background = tuple(float(x) for x in self.background)
        if len(background) != 3:
            raise ValueError("background must contain exactly 3 floats.")
        GaussianRenderer._validate_cached_raster_grad_atomic_mode(self.params.cached_raster_grad.atomic_mode)
        if self.params.debug_mode is not None:
            GaussianRenderer._validate_debug_mode(self.params.debug_mode)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "background", background)

    def background_array(self) -> np.ndarray:
        return np.asarray(self.background, dtype=np.float32)

    def renderer_kwargs(self) -> dict[str, object]:
        kwargs = self.params.renderer_kwargs()
        if kwargs["debug_mode"] is None:
            del kwargs["debug_mode"]
        return kwargs

    @classmethod
    def from_renderer_params(cls, width: int, height: int, params: object) -> GaussianRenderSettings:
        return cls(width=max(int(width), 1), height=max(int(height), 1), params=params)

    def renderer_key(self) -> tuple[object, ...]:
        kwargs = self.renderer_kwargs()
        return (int(self.width), int(self.height), *(kwargs[name] for name in sorted(kwargs)))

    def create_renderer(self, device: spy.Device) -> GaussianRenderer:
        return GaussianRenderer(device, width=self.width, height=self.height, **self.renderer_kwargs())


class GaussianRendererContext:
    def __init__(self, device: spy.Device) -> None:
        self._device = device
        self._renderer: GaussianRenderer | None = None
        self._renderer_key: tuple[object, ...] | None = None

    @property
    def device(self) -> spy.Device:
        return self._device

    @property
    def renderer(self) -> GaussianRenderer:
        if self._renderer is None:
            raise RuntimeError("Renderer is not initialized yet.")
        return self._renderer

    def ensure_renderer(self, settings: GaussianRenderSettings) -> GaussianRenderer:
        key = settings.renderer_key()
        if self._renderer is not None and self._renderer_key == key:
            return self._renderer
        self._renderer = settings.create_renderer(self._device)
        self._renderer_key = key
        return self._renderer

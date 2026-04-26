from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy

from .gaussian_renderer import GaussianRenderer


@dataclass(frozen=True, slots=True)
class GaussianRenderSettings:
    width: int
    height: int
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_scale: float = 1.0
    alpha_cutoff: float = 1.0 / 255.0
    max_anisotropy: float = 32.0
    transmittance_threshold: float = 0.005
    list_capacity_multiplier: int = 64
    max_prepass_memory_mb: int = 4096
    cached_raster_grad_atomic_mode: str = GaussianRenderer.CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT
    cached_raster_grad_fixed_ro_local_range: float = 1.0
    cached_raster_grad_fixed_scale_range: float = 15.0
    cached_raster_grad_fixed_color_range: float = 8.0
    cached_raster_grad_fixed_opacity_range: float = 8.0
    debug_mode: str | None = None
    debug_grad_norm_threshold: float = 2e-4
    debug_ellipse_thickness_px: float = 4.0
    debug_gaussian_scale_multiplier: float = 1.0
    debug_min_opacity: float = 0.0
    debug_opacity_multiplier: float = 1.0
    debug_ellipse_scale_multiplier: float = 1.0
    debug_splat_age_range: tuple[float, float] = (0.0, 1.0)
    debug_density_range: tuple[float, float] = (0.0, 20.0)
    debug_contribution_range: tuple[float, float] = (0.001, 1.0)
    debug_refinement_distribution_range: tuple[float, float] = (0.0, 1.0)
    debug_adam_momentum_range: tuple[float, float] = (0.0, 0.1)
    debug_depth_mean_range: tuple[float, float] = (0.0, 10.0)
    debug_depth_std_range: tuple[float, float] = (0.0, 0.5)
    debug_depth_local_mismatch_range: tuple[float, float] = (0.0, 0.5)
    debug_depth_local_mismatch_smooth_radius: float = 2.0
    debug_depth_local_mismatch_reject_radius: float = 4.0
    debug_sh_coeff_index: int = 0
    debug_show_ellipses: bool = False
    debug_show_processed_count: bool = False
    debug_show_grad_norm: bool = False

    def __post_init__(self) -> None:
        width = max(int(self.width), 1)
        height = max(int(self.height), 1)
        background = tuple(float(x) for x in self.background)
        if len(background) != 3:
            raise ValueError("background must contain exactly 3 floats.")
        GaussianRenderer._validate_cached_raster_grad_atomic_mode(self.cached_raster_grad_atomic_mode)
        if self.debug_mode is not None:
            GaussianRenderer._validate_debug_mode(self.debug_mode)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "background", background)

    def background_array(self) -> np.ndarray:
        return np.asarray(self.background, dtype=np.float32)

    def renderer_kwargs(self) -> dict[str, object]:
        kwargs = {
            "radius_scale": float(self.radius_scale),
            "alpha_cutoff": float(self.alpha_cutoff),
            "max_anisotropy": float(self.max_anisotropy),
            "transmittance_threshold": float(self.transmittance_threshold),
            "list_capacity_multiplier": int(self.list_capacity_multiplier),
            "max_prepass_memory_mb": int(self.max_prepass_memory_mb),
            "cached_raster_grad_atomic_mode": str(self.cached_raster_grad_atomic_mode),
            "cached_raster_grad_fixed_ro_local_range": float(self.cached_raster_grad_fixed_ro_local_range),
            "cached_raster_grad_fixed_scale_range": float(self.cached_raster_grad_fixed_scale_range),
            "cached_raster_grad_fixed_color_range": float(self.cached_raster_grad_fixed_color_range),
            "cached_raster_grad_fixed_opacity_range": float(self.cached_raster_grad_fixed_opacity_range),
            "debug_mode": None if self.debug_mode is None else str(self.debug_mode),
            "debug_grad_norm_threshold": float(self.debug_grad_norm_threshold),
            "debug_ellipse_thickness_px": float(self.debug_ellipse_thickness_px),
            "debug_gaussian_scale_multiplier": float(self.debug_gaussian_scale_multiplier),
            "debug_min_opacity": float(self.debug_min_opacity),
            "debug_opacity_multiplier": float(self.debug_opacity_multiplier),
            "debug_ellipse_scale_multiplier": float(self.debug_ellipse_scale_multiplier),
            "debug_splat_age_range": tuple(float(x) for x in self.debug_splat_age_range),
            "debug_density_range": tuple(float(x) for x in self.debug_density_range),
            "debug_contribution_range": tuple(float(x) for x in self.debug_contribution_range),
            "debug_refinement_distribution_range": tuple(float(x) for x in self.debug_refinement_distribution_range),
            "debug_adam_momentum_range": tuple(float(x) for x in self.debug_adam_momentum_range),
            "debug_depth_mean_range": tuple(float(x) for x in self.debug_depth_mean_range),
            "debug_depth_std_range": tuple(float(x) for x in self.debug_depth_std_range),
            "debug_depth_local_mismatch_range": tuple(float(x) for x in self.debug_depth_local_mismatch_range),
            "debug_depth_local_mismatch_smooth_radius": float(self.debug_depth_local_mismatch_smooth_radius),
            "debug_depth_local_mismatch_reject_radius": float(self.debug_depth_local_mismatch_reject_radius),
            "debug_sh_coeff_index": int(self.debug_sh_coeff_index),
            "debug_show_ellipses": bool(self.debug_show_ellipses),
            "debug_show_processed_count": bool(self.debug_show_processed_count),
            "debug_show_grad_norm": bool(self.debug_show_grad_norm),
        }
        if kwargs["debug_mode"] is None:
            del kwargs["debug_mode"]
        return kwargs

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

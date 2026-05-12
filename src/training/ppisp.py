from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import slangpy as spy


@dataclass(frozen=True, slots=True)
class PPISPFieldSpec:
    attr: str
    key: str
    label: str
    size: int
    default: float | tuple[float, ...]
    step: float
    step_fast: float
    fmt: str

    @property
    def cast(self) -> Callable[[object], object]:
        return float if self.size == 1 else tuple


def _vec3(value: float) -> tuple[float, float, float]:
    scalar = float(value)
    return (scalar, scalar, scalar)


PPISP_FIELD_SPECS: tuple[PPISPFieldSpec, ...] = (
    PPISPFieldSpec("exposureEv", "ppisp_exposure_ev", "Exposure EV", 1, 0.0, 0.01, 0.1, "%.4f"),
    PPISPFieldSpec("vignetteCenterX", "ppisp_vignette_center_x", "Vignette Center X", 3, _vec3(0.5), 0.001, 0.01, "%.4f"),
    PPISPFieldSpec("vignetteCenterY", "ppisp_vignette_center_y", "Vignette Center Y", 3, _vec3(0.5), 0.001, 0.01, "%.4f"),
    PPISPFieldSpec("vignetteCoeffR2", "ppisp_vignette_coeff_r2", "Vignette R2", 3, _vec3(0.0), 0.001, 0.01, "%.5f"),
    PPISPFieldSpec("vignetteCoeffR4", "ppisp_vignette_coeff_r4", "Vignette R4", 3, _vec3(0.0), 0.001, 0.01, "%.5f"),
    PPISPFieldSpec("vignetteCoeffR6", "ppisp_vignette_coeff_r6", "Vignette R6", 3, _vec3(0.0), 0.001, 0.01, "%.5f"),
    PPISPFieldSpec("chromaOffsetR", "ppisp_chroma_offset_r", "Chroma Offset R", 2, (0.0, 0.0), 0.001, 0.01, "%.5f"),
    PPISPFieldSpec("chromaOffsetG", "ppisp_chroma_offset_g", "Chroma Offset G", 2, (0.0, 0.0), 0.001, 0.01, "%.5f"),
    PPISPFieldSpec("chromaOffsetB", "ppisp_chroma_offset_b", "Chroma Offset B", 2, (0.0, 0.0), 0.001, 0.01, "%.5f"),
    PPISPFieldSpec("chromaOffsetW", "ppisp_chroma_offset_w", "Chroma Offset W", 2, (0.0, 0.0), 0.001, 0.01, "%.5f"),
    PPISPFieldSpec("crfTau", "ppisp_crf_tau", "CRF Tau", 3, _vec3(1.0), 0.001, 0.01, "%.5f"),
    PPISPFieldSpec("crfEta", "ppisp_crf_eta", "CRF Eta", 3, _vec3(1.0), 0.001, 0.01, "%.5f"),
    PPISPFieldSpec("crfXi", "ppisp_crf_xi", "CRF Xi", 3, _vec3(0.5), 0.001, 0.01, "%.5f"),
    PPISPFieldSpec("crfGamma", "ppisp_crf_gamma", "CRF Gamma", 3, _vec3(2.2), 0.001, 0.01, "%.5f"),
)

PPISP_PACKED_PARAM_COUNT = sum(int(spec.size) for spec in PPISP_FIELD_SPECS)


def _coerce_tuple(value: object, size: int) -> tuple[float, ...]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        arr = np.zeros((size,), dtype=np.float32)
    if arr.size < size:
        arr = np.pad(arr, (0, size - arr.size), mode="edge")
    return tuple(float(v) for v in arr[:size])


@dataclass(slots=True)
class PPISPTonemapParams:
    exposureEv: float = 0.0
    vignetteCenterX: tuple[float, float, float] = _vec3(0.5)
    vignetteCenterY: tuple[float, float, float] = _vec3(0.5)
    vignetteCoeffR2: tuple[float, float, float] = _vec3(0.0)
    vignetteCoeffR4: tuple[float, float, float] = _vec3(0.0)
    vignetteCoeffR6: tuple[float, float, float] = _vec3(0.0)
    chromaOffsetR: tuple[float, float] = (0.0, 0.0)
    chromaOffsetG: tuple[float, float] = (0.0, 0.0)
    chromaOffsetB: tuple[float, float] = (0.0, 0.0)
    chromaOffsetW: tuple[float, float] = (0.0, 0.0)
    crfTau: tuple[float, float, float] = _vec3(1.0)
    crfEta: tuple[float, float, float] = _vec3(1.0)
    crfXi: tuple[float, float, float] = _vec3(0.5)
    crfGamma: tuple[float, float, float] = _vec3(2.2)

    def __post_init__(self) -> None:
        for spec in PPISP_FIELD_SPECS:
            value = getattr(self, spec.attr)
            if spec.size == 1:
                setattr(self, spec.attr, float(value))
            else:
                setattr(self, spec.attr, _coerce_tuple(value, spec.size))

    @classmethod
    def from_viewer_values(cls, values: dict[str, object]) -> "PPISPTonemapParams":
        defaults = ppisp_viewer_defaults()
        return cls(**{spec.attr: values.get(spec.key, defaults[spec.key]) for spec in PPISP_FIELD_SPECS})

    def to_shader_dict(self) -> dict[str, object]:
        return {
            "exposureEv": float(self.exposureEv),
            "vignetteCenterX": spy.float3(*self.vignetteCenterX),
            "vignetteCenterY": spy.float3(*self.vignetteCenterY),
            "vignetteCoeffR2": spy.float3(*self.vignetteCoeffR2),
            "vignetteCoeffR4": spy.float3(*self.vignetteCoeffR4),
            "vignetteCoeffR6": spy.float3(*self.vignetteCoeffR6),
            "chromaOffsetR": spy.float2(*self.chromaOffsetR),
            "chromaOffsetG": spy.float2(*self.chromaOffsetG),
            "chromaOffsetB": spy.float2(*self.chromaOffsetB),
            "chromaOffsetW": spy.float2(*self.chromaOffsetW),
            "crfTau": spy.float3(*self.crfTau),
            "crfEta": spy.float3(*self.crfEta),
            "crfXi": spy.float3(*self.crfXi),
            "crfGamma": spy.float3(*self.crfGamma),
        }


@dataclass(slots=True)
class PPISPStaticTonemapProvider:
    params: PPISPTonemapParams
    version: int = 0

    def params_for_frame(self, frame_index: int) -> PPISPTonemapParams:
        return self.params


class PPISPTonemapProvider(Protocol):
    @property
    def version(self) -> int: ...

    def params_for_frame(self, frame_index: int) -> PPISPTonemapParams: ...


def ppisp_viewer_defaults() -> dict[str, object]:
    return {spec.key: spec.default for spec in PPISP_FIELD_SPECS}


def ppisp_viewer_export_fields() -> tuple[tuple[str, Callable[[object], object]], ...]:
    return tuple((spec.key, spec.cast) for spec in PPISP_FIELD_SPECS)

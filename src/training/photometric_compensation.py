from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import slangpy as spy

from ..scene import ColmapFrame, ColmapReconstruction
from ..scene._internal.colmap_ops import load_training_frame_rgba8
from ..utility import RO_BUFFER_USAGE, RW_BUFFER_USAGE, SHADER_ROOT, alloc_buffer, buffer_to_numpy, defer_resource_release, dispatch, load_compute_items, thread_count_1d
from .adam import AdamOptimizer, AdamRuntimeHyperParams
from .ppisp import PPISP_FIELD_SPECS, PPISP_PACKED_PARAM_COUNT, PPISPTonemapParams, PPISPTonemapProvider

_PARAM_SETTINGS_U32_WIDTH = 10
_PAIR_GRAD_THREADS = 64
_PHOTOMETRIC_SHADER_PATH = SHADER_ROOT / "utility" / "photometric_compensation.slang"
_PPISP_IDENTITY_VALUES = np.concatenate(
    [
        np.asarray(spec.default if spec.size > 1 else (spec.default,), dtype=np.float32).reshape(spec.size)
        for spec in PPISP_FIELD_SPECS
    ],
    axis=0,
).astype(np.float32, copy=False)


def _srgb_to_linear_rgb(rgb: np.ndarray) -> np.ndarray:
    srgb = np.asarray(rgb, dtype=np.float32)
    return np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4)).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class _PackedFieldLayout:
    attr: str
    start: int
    stop: int
    size: int


_FIELD_LAYOUTS: tuple[_PackedFieldLayout, ...] = ()
_field_layouts: list[_PackedFieldLayout] = []
_field_offset = 0
for _spec in PPISP_FIELD_SPECS:
    _field_layouts.append(_PackedFieldLayout(_spec.attr, _field_offset, _field_offset + int(_spec.size), int(_spec.size)))
    _field_offset += int(_spec.size)
_FIELD_LAYOUTS = tuple(_field_layouts)
del _field_layouts, _field_offset, _spec


@dataclass(slots=True)
class PhotometricCompensationAdamHyperParams:
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass(slots=True)
class PhotometricCompensationHyperParams:
    batch_pair_count: int = 2048
    frame_window_size: int = 0
    neighborhood_size: int = 3
    min_track_length: int = 2
    learning_rate: float = 0.05
    exposure_lr_mul: float = 0.5
    vignette_lr_mul: float = 0.25
    chroma_lr_mul: float = 0.25
    crf_lr_mul: float = 0.125
    exposure_regularize_weight: float = 0.05
    vignette_regularize_weight: float = 0.1
    chroma_regularize_weight: float = 0.2
    crf_regularize_weight: float = 0.2
    exposure_l1_weight: float = 0.0
    vignette_l1_weight: float = 0.01
    chroma_l1_weight: float = 0.02
    crf_l1_weight: float = 0.02
    grad_component_clip: float = 10.0
    grad_norm_clip: float = 10.0
    max_update: float = 0.05
    huge_value: float = 1e8
    ema_decay: float = 0.95

    def __post_init__(self) -> None:
        self.batch_pair_count = max(int(self.batch_pair_count), 1)
        self.frame_window_size = max(int(self.frame_window_size), 0)
        self.neighborhood_size = max(int(self.neighborhood_size), 1)
        if self.neighborhood_size % 2 == 0:
            self.neighborhood_size += 1
        self.neighborhood_size = min(self.neighborhood_size, 15)
        self.min_track_length = max(int(self.min_track_length), 2)
        self.learning_rate = float(max(self.learning_rate, 0.0))
        self.exposure_lr_mul = float(max(self.exposure_lr_mul, 0.0))
        self.vignette_lr_mul = float(max(self.vignette_lr_mul, 0.0))
        self.chroma_lr_mul = float(max(self.chroma_lr_mul, 0.0))
        self.crf_lr_mul = float(max(self.crf_lr_mul, 0.0))
        self.exposure_regularize_weight = float(max(self.exposure_regularize_weight, 0.0))
        self.vignette_regularize_weight = float(max(self.vignette_regularize_weight, 0.0))
        self.chroma_regularize_weight = float(max(self.chroma_regularize_weight, 0.0))
        self.crf_regularize_weight = float(max(self.crf_regularize_weight, 0.0))
        self.exposure_l1_weight = float(max(self.exposure_l1_weight, 0.0))
        self.vignette_l1_weight = float(max(self.vignette_l1_weight, 0.0))
        self.chroma_l1_weight = float(max(self.chroma_l1_weight, 0.0))
        self.crf_l1_weight = float(max(self.crf_l1_weight, 0.0))
        self.grad_component_clip = float(max(self.grad_component_clip, 1e-6))
        self.grad_norm_clip = float(max(self.grad_norm_clip, 1e-6))
        self.max_update = float(max(self.max_update, 1e-6))
        self.huge_value = float(max(self.huge_value, 1.0))
        self.ema_decay = float(np.clip(self.ema_decay, 0.0, 0.9999))


@dataclass(slots=True)
class PhotometricCompensationState:
    step: int = 0
    last_loss: float = float("nan")
    ema_loss: float = float("nan")
    last_regularization_loss: float = float("nan")
    last_pair_count: int = 0


@dataclass(frozen=True, slots=True)
class PhotometricObservationPairBatch:
    pair_indices: np.ndarray
    point_ids: np.ndarray
    track_lengths: np.ndarray
    frame_indices_a: np.ndarray
    frame_indices_b: np.ndarray
    xy_a: np.ndarray
    xy_b: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(self.pair_indices.size)


@dataclass(frozen=True, slots=True)
class PhotometricObservationPairPool:
    point_ids: np.ndarray
    track_lengths: np.ndarray
    frame_indices_a: np.ndarray
    frame_indices_b: np.ndarray
    xy_a: np.ndarray
    xy_b: np.ndarray

    def __len__(self) -> int:
        return int(self.point_ids.size)

    def sample(self, rng: np.random.Generator, pair_count: int) -> PhotometricObservationPairBatch:
        requested = max(int(pair_count), 0)
        total = len(self)
        if requested <= 0 or total <= 0:
            empty_i64 = np.zeros((0,), dtype=np.int64)
            empty_i32 = np.zeros((0,), dtype=np.int32)
            empty_xy = np.zeros((0, 2), dtype=np.float32)
            return PhotometricObservationPairBatch(empty_i64, empty_i64, empty_i32, empty_i32, empty_i32, empty_xy, empty_xy)
        indices = np.asarray(rng.integers(0, total, size=requested, endpoint=False), dtype=np.int64)
        return PhotometricObservationPairBatch(
            pair_indices=indices,
            point_ids=np.ascontiguousarray(self.point_ids[indices], dtype=np.int64),
            track_lengths=np.ascontiguousarray(self.track_lengths[indices], dtype=np.int32),
            frame_indices_a=np.ascontiguousarray(self.frame_indices_a[indices], dtype=np.int32),
            frame_indices_b=np.ascontiguousarray(self.frame_indices_b[indices], dtype=np.int32),
            xy_a=np.ascontiguousarray(self.xy_a[indices], dtype=np.float32),
            xy_b=np.ascontiguousarray(self.xy_b[indices], dtype=np.float32),
        )


@dataclass(frozen=True, slots=True)
class _PhotometricDispatchBatch:
    pair_batch: PhotometricObservationPairBatch
    frame_pair_ranges: np.ndarray
    frame_pair_entries: np.ndarray


def _float_bits(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def pack_ppisp_tonemap_params(params: PPISPTonemapParams) -> np.ndarray:
    packed = np.empty((PPISP_PACKED_PARAM_COUNT,), dtype=np.float32)
    offset = 0
    for spec in PPISP_FIELD_SPECS:
        value = getattr(params, spec.attr)
        if spec.size == 1:
            packed[offset] = float(value)
        else:
            packed[offset : offset + spec.size] = np.asarray(value, dtype=np.float32).reshape(spec.size)
        offset += int(spec.size)
    return packed


def unpack_ppisp_tonemap_params(values: np.ndarray) -> PPISPTonemapParams:
    packed = np.asarray(values, dtype=np.float32).reshape(-1)
    if packed.size != PPISP_PACKED_PARAM_COUNT:
        raise ValueError(f"Expected {PPISP_PACKED_PARAM_COUNT} packed PPISP params, got {packed.size}.")
    kwargs: dict[str, object] = {}
    for layout in _FIELD_LAYOUTS:
        field_values = packed[layout.start : layout.stop]
        kwargs[layout.attr] = float(field_values[0]) if layout.size == 1 else tuple(float(v) for v in field_values)
    return PPISPTonemapParams(**kwargs)


def identity_packed_ppisp_params(frame_count: int, *, flatten: bool = False) -> np.ndarray:
    count = max(int(frame_count), 0)
    packed = np.repeat(_PPISP_IDENTITY_VALUES[:, None], count, axis=1).astype(np.float32, copy=False)
    return packed.reshape(-1).copy() if flatten else np.ascontiguousarray(packed, dtype=np.float32)


def _reshape_packed_params(values: np.ndarray, frame_count: int) -> np.ndarray:
    resolved_frame_count = max(int(frame_count), 0)
    reshaped = np.asarray(values, dtype=np.float32).reshape(-1)
    expected = PPISP_PACKED_PARAM_COUNT * resolved_frame_count
    if reshaped.size != expected:
        raise ValueError(f"Expected {expected} packed PPISP values for {resolved_frame_count} frames, got {reshaped.size}.")
    return np.ascontiguousarray(reshaped.reshape(PPISP_PACKED_PARAM_COUNT, resolved_frame_count), dtype=np.float32)


class PackedPPISPTonemapProvider(PPISPTonemapProvider):
    def __init__(self, frame_count: int, packed_params: np.ndarray | None = None, *, version: int = 0) -> None:
        self._frame_count = max(int(frame_count), 0)
        self._packed_params = identity_packed_ppisp_params(self._frame_count) if packed_params is None else _reshape_packed_params(packed_params, self._frame_count)
        self._version = int(version)

    @property
    def version(self) -> int:
        return int(self._version)

    @property
    def frame_count(self) -> int:
        return int(self._frame_count)

    def params_for_frame(self, frame_index: int) -> PPISPTonemapParams:
        if self._frame_count <= 0:
            raise IndexError("No PPISP frame parameters are available.")
        resolved = min(max(int(frame_index), 0), self._frame_count - 1)
        return unpack_ppisp_tonemap_params(self._packed_params[:, resolved])

    def replace_packed_params(self, packed_params: np.ndarray) -> None:
        self._packed_params = _reshape_packed_params(packed_params, self._frame_count)
        self._version += 1

    def snapshot_packed_params(self, *, flatten: bool = False) -> np.ndarray:
        return self._packed_params.reshape(-1).copy() if flatten else self._packed_params.copy()


def build_photometric_observation_pair_pool(
    reconstruction: ColmapReconstruction,
    frames: list[ColmapFrame],
    *,
    min_track_length: int = 2,
) -> PhotometricObservationPairPool:
    frame_lookup = {int(frame.image_id): (frame_index, frame) for frame_index, frame in enumerate(frames)}
    observation_map: dict[int, list[tuple[int, float, float, int]]] = {}
    min_track = max(int(min_track_length), 2)
    points = getattr(reconstruction, "points3d", {})
    images = getattr(reconstruction, "images", {})
    for image_id, image in sorted(images.items()):
        frame_entry = frame_lookup.get(int(image_id))
        if frame_entry is None:
            continue
        frame_index, frame = frame_entry
        point_xy = np.asarray(getattr(image, "points2d_xy", ()), dtype=np.float32).reshape(-1, 2)
        point_ids = np.asarray(getattr(image, "points2d_point3d_ids", ()), dtype=np.int64).reshape(-1)
        count = min(int(point_xy.shape[0]), int(point_ids.size))
        if count <= 0:
            continue
        seen_point_ids: set[int] = set()
        for offset in range(count):
            point_id = int(point_ids[offset])
            if point_id <= 0 or point_id in seen_point_ids:
                continue
            point = points.get(point_id) if isinstance(points, dict) else None
            track_length = int(getattr(point, "track_length", 0)) if point is not None else 0
            if track_length < min_track:
                continue
            xy = point_xy[offset]
            if not np.all(np.isfinite(xy)):
                continue
            if float(xy[0]) < 0.0 or float(xy[0]) > float(frame.width) or float(xy[1]) < 0.0 or float(xy[1]) > float(frame.height):
                continue
            observation_map.setdefault(point_id, []).append((int(frame_index), float(xy[0]), float(xy[1]), track_length))
            seen_point_ids.add(point_id)

    point_ids_out: list[int] = []
    track_lengths_out: list[int] = []
    frame_indices_a: list[int] = []
    frame_indices_b: list[int] = []
    xy_a: list[tuple[float, float]] = []
    xy_b: list[tuple[float, float]] = []
    for point_id in sorted(observation_map):
        observations = sorted(observation_map[point_id], key=lambda item: (item[0], item[1], item[2]))
        if len(observations) < 2:
            continue
        track_length = int(observations[0][3])
        for left_index in range(len(observations) - 1):
            left = observations[left_index]
            for right in observations[left_index + 1 :]:
                point_ids_out.append(int(point_id))
                track_lengths_out.append(track_length)
                frame_indices_a.append(int(left[0]))
                frame_indices_b.append(int(right[0]))
                xy_a.append((float(left[1]), float(left[2])))
                xy_b.append((float(right[1]), float(right[2])))

    return PhotometricObservationPairPool(
        point_ids=np.ascontiguousarray(point_ids_out, dtype=np.int64),
        track_lengths=np.ascontiguousarray(track_lengths_out, dtype=np.int32),
        frame_indices_a=np.ascontiguousarray(frame_indices_a, dtype=np.int32),
        frame_indices_b=np.ascontiguousarray(frame_indices_b, dtype=np.int32),
        xy_a=np.ascontiguousarray(xy_a, dtype=np.float32).reshape(-1, 2),
        xy_b=np.ascontiguousarray(xy_b, dtype=np.float32).reshape(-1, 2),
    )


def _field_group_name(attr: str) -> str:
    if attr == "exposureEv":
        return "exposure"
    if attr.startswith("vignette"):
        return "vignette"
    if attr.startswith("chroma"):
        return "chroma"
    return "crf"


def _field_lr_mul(attr: str, hparams: PhotometricCompensationHyperParams) -> float:
    group = _field_group_name(attr)
    if group == "exposure":
        return float(hparams.exposure_lr_mul)
    if group == "vignette":
        return float(hparams.vignette_lr_mul)
    if group == "chroma":
        return float(hparams.chroma_lr_mul)
    return float(hparams.crf_lr_mul)


def _field_regularize_weight(attr: str, hparams: PhotometricCompensationHyperParams) -> float:
    group = _field_group_name(attr)
    if group == "exposure":
        return float(hparams.exposure_regularize_weight)
    if group == "vignette":
        return float(hparams.vignette_regularize_weight)
    if group == "chroma":
        return float(hparams.chroma_regularize_weight)
    return float(hparams.crf_regularize_weight)


def _field_l1_weight(attr: str, hparams: PhotometricCompensationHyperParams) -> float:
    group = _field_group_name(attr)
    if group == "exposure":
        return float(hparams.exposure_l1_weight)
    if group == "vignette":
        return float(hparams.vignette_l1_weight)
    if group == "chroma":
        return float(hparams.chroma_l1_weight)
    return float(hparams.crf_l1_weight)


def _field_value_bounds(attr: str) -> tuple[float, float]:
    if attr == "exposureEv":
        return (-6.0, 6.0)
    if attr in ("vignetteCenterX", "vignetteCenterY"):
        return (0.0, 1.0)
    if attr.startswith("vignetteCoeff"):
        return (-4.0, 4.0)
    if attr.startswith("chromaOffset"):
        return (-0.35, 0.35)
    if attr in ("crfTau", "crfEta"):
        return (0.25, 4.0)
    if attr == "crfXi":
        return (0.05, 0.95)
    if attr == "crfGamma":
        return (0.5, 4.0)
    return (-1e6, 1e6)


def build_ppisp_param_settings(hparams: PhotometricCompensationHyperParams) -> np.ndarray:
    settings = np.zeros((PPISP_PACKED_PARAM_COUNT, _PARAM_SETTINGS_U32_WIDTH), dtype=np.uint32)
    group_starts = np.arange(PPISP_PACKED_PARAM_COUNT, dtype=np.uint32)
    group_sizes = np.ones((PPISP_PACKED_PARAM_COUNT,), dtype=np.uint32)
    lrs = np.zeros((PPISP_PACKED_PARAM_COUNT,), dtype=np.float32)
    value_mins = np.zeros((PPISP_PACKED_PARAM_COUNT,), dtype=np.float32)
    value_maxs = np.zeros((PPISP_PACKED_PARAM_COUNT,), dtype=np.float32)
    regularize_toward = _PPISP_IDENTITY_VALUES.copy()
    regularize_weight = np.zeros((PPISP_PACKED_PARAM_COUNT,), dtype=np.float32)
    regularize_l1 = np.zeros((PPISP_PACKED_PARAM_COUNT,), dtype=np.float32)

    offset = 0
    for spec in PPISP_FIELD_SPECS:
        group_starts[offset : offset + spec.size] = np.uint32(offset)
        group_sizes[offset : offset + spec.size] = np.uint32(spec.size)
        value_min, value_max = _field_value_bounds(spec.attr)
        lrs[offset : offset + spec.size] = float(hparams.learning_rate) * _field_lr_mul(spec.attr, hparams)
        value_mins[offset : offset + spec.size] = float(value_min)
        value_maxs[offset : offset + spec.size] = float(value_max)
        regularize_weight[offset : offset + spec.size] = _field_regularize_weight(spec.attr, hparams)
        regularize_l1[offset : offset + spec.size] = _field_l1_weight(spec.attr, hparams)
        offset += int(spec.size)

    settings[:, 0] = _float_bits(lrs)
    settings[:, 1] = _float_bits(float(hparams.grad_component_clip))
    settings[:, 2] = _float_bits(float(hparams.grad_norm_clip))
    settings[:, 3] = _float_bits(value_mins)
    settings[:, 4] = _float_bits(value_maxs)
    settings[:, 5] = group_starts
    settings[:, 6] = group_sizes
    settings[:, 7] = _float_bits(regularize_toward)
    settings[:, 8] = _float_bits(regularize_weight)
    settings[:, 9] = _float_bits(regularize_l1)
    return settings


def _coerce_frame_rgba_linear(frame: ColmapFrame, value: np.ndarray) -> np.ndarray:
    image = np.asarray(value, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != int(frame.height) or image.shape[1] != int(frame.width):
        raise ValueError(f"Expected frame override for image {frame.image_id} with shape ({frame.height}, {frame.width}, C), got {image.shape}.")
    if image.shape[2] == 3:
        alpha = np.ones((int(frame.height), int(frame.width), 1), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate((image, alpha), axis=2), dtype=np.float32)
    if image.shape[2] != 4:
        raise ValueError(f"Expected 3 or 4 channels for frame override {frame.image_id}, got {image.shape[2]}.")
    return np.ascontiguousarray(image, dtype=np.float32)


def _load_frame_rgba_linear(frame: ColmapFrame) -> np.ndarray:
    rgba8 = np.asarray(load_training_frame_rgba8(frame), dtype=np.uint8)
    rgb = rgba8[:, :, :3].astype(np.float32) / 255.0
    alpha = rgba8[:, :, 3:4].astype(np.float32) / 255.0
    linear = _srgb_to_linear_rgb(rgb)
    return np.ascontiguousarray(np.concatenate((linear, alpha), axis=2), dtype=np.float32)


def _build_frame_pixel_data(frames: list[ColmapFrame], frame_rgba_linear: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    frame_info = np.zeros((len(frames), 4), dtype=np.uint32)
    pixels: list[np.ndarray] = []
    offset = 0
    for frame_index, (frame, rgba) in enumerate(zip(frames, frame_rgba_linear, strict=True)):
        image = _coerce_frame_rgba_linear(frame, rgba)
        pixel_count = int(frame.width) * int(frame.height)
        frame_info[frame_index, 0] = np.uint32(offset)
        frame_info[frame_index, 1] = np.uint32(int(frame.width))
        frame_info[frame_index, 2] = np.uint32(int(frame.height))
        pixels.append(np.ascontiguousarray(image.reshape(pixel_count, 4), dtype=np.float32))
        offset += pixel_count
    flat_pixels = np.ascontiguousarray(np.concatenate(pixels, axis=0) if pixels else np.zeros((0, 4), dtype=np.float32), dtype=np.float32)
    return flat_pixels, frame_info


def _build_sparse_frame_pixel_data(frames: list[ColmapFrame], frame_rgba_linear: list[np.ndarray | None], active_frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_info = np.zeros((len(frames), 4), dtype=np.uint32)
    if active_frame_indices.size <= 0:
        return np.zeros((0, 4), dtype=np.float32), frame_info
    pixels: list[np.ndarray] = []
    offset = 0
    for frame_index in np.asarray(active_frame_indices, dtype=np.int32):
        resolved_frame_index = int(frame_index)
        frame = frames[resolved_frame_index]
        rgba = frame_rgba_linear[resolved_frame_index]
        if rgba is None:
            raise ValueError(f"Frame {resolved_frame_index} was not loaded before sparse pixel upload.")
        image = _coerce_frame_rgba_linear(frame, rgba)
        pixel_count = int(frame.width) * int(frame.height)
        frame_info[resolved_frame_index, 0] = np.uint32(offset)
        frame_info[resolved_frame_index, 1] = np.uint32(int(frame.width))
        frame_info[resolved_frame_index, 2] = np.uint32(int(frame.height))
        pixels.append(np.ascontiguousarray(image.reshape(pixel_count, 4), dtype=np.float32))
        offset += pixel_count
    flat_pixels = np.ascontiguousarray(np.concatenate(pixels, axis=0) if pixels else np.zeros((0, 4), dtype=np.float32), dtype=np.float32)
    return flat_pixels, frame_info


def _sample_pair_batch_from_indices(pool: PhotometricObservationPairPool, rng: np.random.Generator, pair_indices: np.ndarray, pair_count: int) -> PhotometricObservationPairBatch:
    candidates = np.asarray(pair_indices, dtype=np.int64).reshape(-1)
    requested = max(int(pair_count), 0)
    if requested <= 0 or candidates.size <= 0:
        empty_i64 = np.zeros((0,), dtype=np.int64)
        empty_i32 = np.zeros((0,), dtype=np.int32)
        empty_xy = np.zeros((0, 2), dtype=np.float32)
        return PhotometricObservationPairBatch(empty_i64, empty_i64, empty_i32, empty_i32, empty_i32, empty_xy, empty_xy)
    sampled = np.ascontiguousarray(candidates[np.asarray(rng.integers(0, candidates.size, size=requested, endpoint=False), dtype=np.int64)], dtype=np.int64)
    return PhotometricObservationPairBatch(
        pair_indices=sampled,
        point_ids=np.ascontiguousarray(pool.point_ids[sampled], dtype=np.int64),
        track_lengths=np.ascontiguousarray(pool.track_lengths[sampled], dtype=np.int32),
        frame_indices_a=np.ascontiguousarray(pool.frame_indices_a[sampled], dtype=np.int32),
        frame_indices_b=np.ascontiguousarray(pool.frame_indices_b[sampled], dtype=np.int32),
        xy_a=np.ascontiguousarray(pool.xy_a[sampled], dtype=np.float32),
        xy_b=np.ascontiguousarray(pool.xy_b[sampled], dtype=np.float32),
    )


def _build_frame_pair_batch(frame_count: int, batch: PhotometricObservationPairBatch) -> _PhotometricDispatchBatch:
    entry_count = int(batch.pair_count) * 2
    ranges = np.zeros((max(int(frame_count), 0) + 1,), dtype=np.uint32)
    if entry_count <= 0 or int(frame_count) <= 0:
        return _PhotometricDispatchBatch(batch, ranges, np.zeros((0, 2), dtype=np.uint32))
    for frame_index in np.asarray(batch.frame_indices_a, dtype=np.int32):
        ranges[min(max(int(frame_index), 0), int(frame_count) - 1) + 1] += np.uint32(1)
    for frame_index in np.asarray(batch.frame_indices_b, dtype=np.int32):
        ranges[min(max(int(frame_index), 0), int(frame_count) - 1) + 1] += np.uint32(1)
    np.cumsum(ranges, out=ranges)
    entries = np.zeros((entry_count, 2), dtype=np.uint32)
    cursor = ranges[:-1].copy()
    for pair_index in range(int(batch.pair_count)):
        frame_a = min(max(int(batch.frame_indices_a[pair_index]), 0), int(frame_count) - 1)
        frame_b = min(max(int(batch.frame_indices_b[pair_index]), 0), int(frame_count) - 1)
        write_a = int(cursor[frame_a])
        entries[write_a, 0] = np.uint32(pair_index)
        entries[write_a, 1] = np.uint32(0)
        cursor[frame_a] += np.uint32(1)
        write_b = int(cursor[frame_b])
        entries[write_b, 0] = np.uint32(pair_index)
        entries[write_b, 1] = np.uint32(1)
        cursor[frame_b] += np.uint32(1)
    return _PhotometricDispatchBatch(batch, np.ascontiguousarray(ranges, dtype=np.uint32), np.ascontiguousarray(entries, dtype=np.uint32))


def _photometric_regularization_loss(params: np.ndarray, hparams: PhotometricCompensationHyperParams) -> float:
    packed = np.asarray(params, dtype=np.float32).reshape(PPISP_PACKED_PARAM_COUNT, -1)
    total = 0.0
    for spec in PPISP_FIELD_SPECS:
        layout = next(layout for layout in _FIELD_LAYOUTS if layout.attr == spec.attr)
        values = packed[layout.start : layout.stop]
        identity = _PPISP_IDENTITY_VALUES[layout.start : layout.stop].reshape(layout.size, 1)
        regularize_weight = _field_regularize_weight(spec.attr, hparams)
        l1_weight = _field_l1_weight(spec.attr, hparams)
        if regularize_weight > 0.0:
            total += 0.5 * regularize_weight * float(np.mean(np.square(values - identity), dtype=np.float64))
        if l1_weight > 0.0:
            total += l1_weight * float(np.mean(np.abs(values - identity), dtype=np.float64))
    return float(total)


class PhotometricCompensationTrainer:
    def __init__(
        self,
        device: spy.Device,
        reconstruction: ColmapReconstruction,
        frames: list[ColmapFrame],
        *,
        hparams: PhotometricCompensationHyperParams | None = None,
        adam_hparams: PhotometricCompensationAdamHyperParams | None = None,
        seed: int = 0,
        frame_rgba_linear: list[np.ndarray] | None = None,
    ) -> None:
        if not frames:
            raise ValueError("Photometric compensation requires at least one frame.")
        self.device = device
        self.reconstruction = reconstruction
        self.frames = list(frames)
        self.hparams = PhotometricCompensationHyperParams() if hparams is None else hparams
        self.adam = PhotometricCompensationAdamHyperParams() if adam_hparams is None else adam_hparams
        self.state = PhotometricCompensationState()
        self._frame_count = len(self.frames)
        self._packed_param_count = self._frame_count * PPISP_PACKED_PARAM_COUNT
        self._rng = np.random.default_rng(int(seed))
        self._buffers: dict[str, spy.Buffer] = {}
        self._kernels = load_compute_items(self.device, {"pair_loss_backward": ("kernel", _PHOTOMETRIC_SHADER_PATH, "csPhotometricPairLossBackward")})
        self._provider = PackedPPISPTonemapProvider(self._frame_count)
        self._frame_rgba_linear: list[np.ndarray | None] = (
            [None] * self._frame_count if frame_rgba_linear is None else [
                _coerce_frame_rgba_linear(frame, frame_rgba_linear[index])
                for index, frame in enumerate(self.frames)
            ]
        )
        self._frame_pixel_capacity = 0
        self._batch_pair_capacity = 0
        self._frame_pair_entry_capacity = 0
        self.pair_pool = build_photometric_observation_pair_pool(
            self.reconstruction,
            self.frames,
            min_track_length=self.hparams.min_track_length,
        )
        if len(self.pair_pool) <= 0:
            raise ValueError("Photometric compensation requires at least one valid cross-view sparse-track pair.")
        self._pair_frame_mins = np.minimum(self.pair_pool.frame_indices_a, self.pair_pool.frame_indices_b).astype(np.int32, copy=False)
        self._pair_frame_maxs = np.maximum(self.pair_pool.frame_indices_a, self.pair_pool.frame_indices_b).astype(np.int32, copy=False)
        self.adam_optimizer = AdamOptimizer(self.device, self.adam, self._runtime_hparams())
        self._ensure_buffers()
        self._upload_param_settings()
        self.reset()

    def _runtime_hparams(self) -> AdamRuntimeHyperParams:
        return AdamRuntimeHyperParams(
            grad_component_clip=float(self.hparams.grad_component_clip),
            grad_norm_clip=float(self.hparams.grad_norm_clip),
            max_update=float(self.hparams.max_update),
            huge_value=float(self.hparams.huge_value),
        )

    def _ensure_buffers(self) -> None:
        if "params" not in self._buffers:
            self._buffers["params"] = alloc_buffer(self.device, name="photometric_compensation.params", size=max(self._packed_param_count, 1) * 4, usage=RW_BUFFER_USAGE)
        if "grads" not in self._buffers:
            self._buffers["grads"] = alloc_buffer(self.device, name="photometric_compensation.grads", size=max(self._packed_param_count, 1) * 4, usage=RW_BUFFER_USAGE)
        if "param_settings" not in self._buffers:
            self._buffers["param_settings"] = alloc_buffer(
                self.device,
                name="photometric_compensation.param_settings",
                size=max(PPISP_PACKED_PARAM_COUNT, 1) * _PARAM_SETTINGS_U32_WIDTH * 4,
                usage=RO_BUFFER_USAGE,
            )
        if "loss_sum" not in self._buffers:
            self._buffers["loss_sum"] = alloc_buffer(self.device, name="photometric_compensation.loss_sum", size=4, usage=RW_BUFFER_USAGE)
        if "frame_info" not in self._buffers:
            self._buffers["frame_info"] = alloc_buffer(self.device, name="photometric_compensation.frame_info", size=max(self._frame_count, 1) * 16, usage=RO_BUFFER_USAGE)

    def _ensure_frame_pixel_buffer(self, pixel_count: int) -> None:
        required = max(int(pixel_count), 1)
        if self._frame_pixel_capacity >= required and "frame_pixels" in self._buffers:
            return
        defer_resource_release(self._buffers.get("frame_pixels"))
        self._buffers["frame_pixels"] = alloc_buffer(self.device, name="photometric_compensation.frame_pixels", size=required * 16, usage=RO_BUFFER_USAGE)
        self._frame_pixel_capacity = required

    def _ensure_batch_buffers(self, pair_count: int) -> None:
        required_pairs = max(int(pair_count), 1)
        required_entries = max(int(pair_count) * 2, 1)
        if self._batch_pair_capacity < required_pairs or "pair_frame_indices_a" not in self._buffers:
            for name in ("pair_frame_indices_a", "pair_frame_indices_b", "pair_xy_a", "pair_xy_b"):
                defer_resource_release(self._buffers.get(name))
            self._buffers["pair_frame_indices_a"] = alloc_buffer(self.device, name="photometric_compensation.pair_frame_indices_a", size=required_pairs * 4, usage=RO_BUFFER_USAGE)
            self._buffers["pair_frame_indices_b"] = alloc_buffer(self.device, name="photometric_compensation.pair_frame_indices_b", size=required_pairs * 4, usage=RO_BUFFER_USAGE)
            self._buffers["pair_xy_a"] = alloc_buffer(self.device, name="photometric_compensation.pair_xy_a", size=required_pairs * 8, usage=RO_BUFFER_USAGE)
            self._buffers["pair_xy_b"] = alloc_buffer(self.device, name="photometric_compensation.pair_xy_b", size=required_pairs * 8, usage=RO_BUFFER_USAGE)
            self._batch_pair_capacity = required_pairs
        if self._frame_pair_entry_capacity < required_entries or "frame_pair_entries" not in self._buffers:
            for name in ("frame_pair_ranges", "frame_pair_entries"):
                defer_resource_release(self._buffers.get(name))
            self._buffers["frame_pair_ranges"] = alloc_buffer(self.device, name="photometric_compensation.frame_pair_ranges", size=max(self._frame_count + 1, 1) * 4, usage=RO_BUFFER_USAGE)
            self._buffers["frame_pair_entries"] = alloc_buffer(self.device, name="photometric_compensation.frame_pair_entries", size=required_entries * 8, usage=RO_BUFFER_USAGE)
            self._frame_pair_entry_capacity = required_entries

    def _resolve_frame_rgba_linear(self, frame_index: int) -> np.ndarray:
        resolved_frame_index = min(max(int(frame_index), 0), self._frame_count - 1)
        rgba = self._frame_rgba_linear[resolved_frame_index]
        if rgba is None:
            rgba = _load_frame_rgba_linear(self.frames[resolved_frame_index])
            self._frame_rgba_linear[resolved_frame_index] = rgba
        return rgba

    def _upload_frame_images(self, dispatch_batch: _PhotometricDispatchBatch) -> None:
        pair_batch = dispatch_batch.pair_batch
        active_frame_indices = np.unique(
            np.concatenate(
                (
                    np.asarray(pair_batch.frame_indices_a, dtype=np.int32),
                    np.asarray(pair_batch.frame_indices_b, dtype=np.int32),
                ),
                axis=0,
            )
        ).astype(np.int32, copy=False)
        for frame_index in active_frame_indices:
            self._resolve_frame_rgba_linear(int(frame_index))
        pixels, frame_info = _build_sparse_frame_pixel_data(self.frames, self._frame_rgba_linear, active_frame_indices)
        self._ensure_frame_pixel_buffer(int(pixels.shape[0]))
        self._buffers["frame_pixels"].copy_from_numpy(pixels.reshape(-1, 4))
        self._buffers["frame_info"].copy_from_numpy(frame_info)

    def _sample_windowed_pair_batch(self, pair_count: int) -> PhotometricObservationPairBatch:
        window_size = min(max(int(self.hparams.frame_window_size), 1), self._frame_count)
        if window_size >= self._frame_count:
            return self.pair_pool.sample(self._rng, pair_count)
        pair_total = len(self.pair_pool)
        search_attempts = min(max(pair_total, 1), 16)
        for _ in range(search_attempts):
            anchor_pair_index = int(self._rng.integers(0, pair_total, endpoint=False))
            min_frame_index = int(self._pair_frame_mins[anchor_pair_index])
            max_frame_index = int(self._pair_frame_maxs[anchor_pair_index])
            span = max_frame_index - min_frame_index + 1
            if span > window_size:
                continue
            start_min = max(0, max_frame_index - window_size + 1)
            start_max = min(min_frame_index, self._frame_count - window_size)
            start = start_min if start_max < start_min else int(self._rng.integers(start_min, start_max + 1, endpoint=False))
            stop = start + window_size
            candidate_indices = np.flatnonzero((self._pair_frame_mins >= start) & (self._pair_frame_maxs < stop))
            if candidate_indices.size > 0:
                return _sample_pair_batch_from_indices(self.pair_pool, self._rng, candidate_indices, pair_count)
        best_candidate_indices = np.zeros((0,), dtype=np.int64)
        best_candidate_count = 0
        for start in range(0, self._frame_count - window_size + 1):
            stop = start + window_size
            candidate_indices = np.flatnonzero((self._pair_frame_mins >= start) & (self._pair_frame_maxs < stop))
            candidate_count = int(candidate_indices.size)
            if candidate_count <= best_candidate_count:
                continue
            best_candidate_indices = np.ascontiguousarray(candidate_indices, dtype=np.int64)
            best_candidate_count = candidate_count
            if best_candidate_count >= pair_count:
                break
        if best_candidate_count > 0:
            return _sample_pair_batch_from_indices(self.pair_pool, self._rng, best_candidate_indices, pair_count)
        return self.pair_pool.sample(self._rng, pair_count)

    def _upload_pair_batch(self, dispatch_batch: _PhotometricDispatchBatch) -> None:
        pair_batch = dispatch_batch.pair_batch
        self._ensure_batch_buffers(pair_batch.pair_count)
        self._buffers["pair_frame_indices_a"].copy_from_numpy(np.ascontiguousarray(pair_batch.frame_indices_a, dtype=np.uint32))
        self._buffers["pair_frame_indices_b"].copy_from_numpy(np.ascontiguousarray(pair_batch.frame_indices_b, dtype=np.uint32))
        self._buffers["pair_xy_a"].copy_from_numpy(np.ascontiguousarray(pair_batch.xy_a, dtype=np.float32))
        self._buffers["pair_xy_b"].copy_from_numpy(np.ascontiguousarray(pair_batch.xy_b, dtype=np.float32))
        self._buffers["frame_pair_ranges"].copy_from_numpy(dispatch_batch.frame_pair_ranges)
        self._buffers["frame_pair_entries"].copy_from_numpy(dispatch_batch.frame_pair_entries)

    def _dispatch_pair_loss_backward(self, encoder: spy.CommandEncoder, pair_count: int) -> None:
        self._buffers["loss_sum"].copy_from_numpy(np.zeros((1,), dtype=np.float32))
        dispatch(
            kernel=self._kernels["pair_loss_backward"],
            thread_count=thread_count_1d(self._frame_count * _PAIR_GRAD_THREADS),
            vars={
                "g_Params": self._buffers["params"],
                "g_Grads": self._buffers["grads"],
                "g_LossSum": self._buffers["loss_sum"],
                "g_FramePixels": self._buffers["frame_pixels"],
                "g_FrameInfo": self._buffers["frame_info"],
                "g_PairFrameIndicesA": self._buffers["pair_frame_indices_a"],
                "g_PairFrameIndicesB": self._buffers["pair_frame_indices_b"],
                "g_PairXYA": self._buffers["pair_xy_a"],
                "g_PairXYB": self._buffers["pair_xy_b"],
                "g_FramePairRanges": self._buffers["frame_pair_ranges"],
                "g_FramePairEntries": self._buffers["frame_pair_entries"],
                "g_FrameCount": int(self._frame_count),
                "g_PairCount": int(pair_count),
                "g_NeighborhoodRadius": int(self.hparams.neighborhood_size // 2),
            },
            command_encoder=encoder,
            debug_label="Photometric::pair_loss_backward",
            debug_color_index=93,
        )

    def _update_state_after_step(self, pair_count: int) -> None:
        self.state.last_pair_count = int(pair_count)
        self.state.last_loss = float(buffer_to_numpy(self._buffers["loss_sum"], np.float32)[0])
        if np.isfinite(self.state.ema_loss):
            self.state.ema_loss = float(self.hparams.ema_decay * self.state.ema_loss + (1.0 - self.hparams.ema_decay) * self.state.last_loss)
        else:
            self.state.ema_loss = float(self.state.last_loss)
        packed = self.read_packed_params()
        self.state.last_regularization_loss = _photometric_regularization_loss(packed, self.hparams)
        self._provider.replace_packed_params(packed)

    def _upload_param_settings(self) -> None:
        self._buffers["param_settings"].copy_from_numpy(build_ppisp_param_settings(self.hparams))

    @property
    def buffers(self) -> dict[str, spy.Buffer]:
        return self._buffers

    @property
    def frame_count(self) -> int:
        return int(self._frame_count)

    @property
    def packed_param_count(self) -> int:
        return int(self._packed_param_count)

    @property
    def param_settings_count(self) -> int:
        return int(PPISP_PACKED_PARAM_COUNT)

    @property
    def provider(self) -> PackedPPISPTonemapProvider:
        return self._provider

    def sample_pair_batch(self, pair_count: int | None = None) -> PhotometricObservationPairBatch:
        requested = self.hparams.batch_pair_count if pair_count is None else int(pair_count)
        if 0 < int(self.hparams.frame_window_size) < self._frame_count:
            return self._sample_windowed_pair_batch(requested)
        return self.pair_pool.sample(self._rng, requested)

    def build_dispatch_batch(self, pair_count: int | None = None) -> _PhotometricDispatchBatch:
        return _build_frame_pair_batch(self._frame_count, self.sample_pair_batch(pair_count))

    def replace_packed_params(self, packed_params: np.ndarray) -> None:
        reshaped = _reshape_packed_params(packed_params, self._frame_count)
        self._buffers["params"].copy_from_numpy(reshaped.reshape(-1))
        self._provider.replace_packed_params(reshaped)

    def read_packed_params(self) -> np.ndarray:
        flat = buffer_to_numpy(self._buffers["params"], np.float32)[: self._packed_param_count]
        return np.ascontiguousarray(flat.reshape(PPISP_PACKED_PARAM_COUNT, self._frame_count), dtype=np.float32)

    def snapshot_provider(self) -> PackedPPISPTonemapProvider:
        return PackedPPISPTonemapProvider(self._frame_count, self.read_packed_params(), version=self._provider.version)

    def zero_grads(self) -> None:
        self._buffers["grads"].copy_from_numpy(np.zeros((max(self._packed_param_count, 1),), dtype=np.float32))

    def reset(self) -> None:
        self.state = PhotometricCompensationState()
        self.replace_packed_params(identity_packed_ppisp_params(self._frame_count))
        self._buffers["loss_sum"].copy_from_numpy(np.zeros((1,), dtype=np.float32))
        self.adam_optimizer.update_hyperparams(self.adam, self._runtime_hparams())
        self.adam_optimizer.zero_moments(self._packed_param_count)

    def dispatch_optimizer_step(self, encoder: spy.CommandEncoder, step_index: int) -> None:
        self.adam_optimizer.update_hyperparams(self.adam, self._runtime_hparams())
        self._upload_param_settings()
        self.adam_optimizer.dispatch_step(
            encoder,
            params_buffer=self._buffers["params"],
            grads_buffer=self._buffers["grads"],
            element_count=self._frame_count,
            packed_param_count=self._packed_param_count,
            param_group_size=self._frame_count,
            param_settings=self._buffers["param_settings"],
            param_settings_count=PPISP_PACKED_PARAM_COUNT,
            step_index=int(step_index),
        )

    def step_optimizer(self, step_index: int | None = None) -> None:
        resolved_step = self.state.step + 1 if step_index is None else int(step_index)
        encoder = self.device.create_command_encoder()
        self.dispatch_optimizer_step(encoder, resolved_step)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        self.state.step = int(resolved_step)
        packed = self.read_packed_params()
        self._provider.replace_packed_params(packed)

    def train_step(self, pair_count: int | None = None, *, step_index: int | None = None) -> float:
        dispatch_batch = self.build_dispatch_batch(pair_count)
        resolved_step = self.state.step + 1 if step_index is None else int(step_index)
        self._upload_frame_images(dispatch_batch)
        self._upload_param_settings()
        self._upload_pair_batch(dispatch_batch)
        encoder = self.device.create_command_encoder()
        self._dispatch_pair_loss_backward(encoder, dispatch_batch.pair_batch.pair_count)
        self.dispatch_optimizer_step(encoder, resolved_step)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        self.state.step = int(resolved_step)
        self._update_state_after_step(dispatch_batch.pair_batch.pair_count)
        return float(self.state.last_loss)

    def release_resources(self) -> None:
        for buffer in self._buffers.values():
            defer_resource_release(buffer)
        self._buffers = {}
        if getattr(self.adam_optimizer, "_buffers", None) is not None:
            for buffer in self.adam_optimizer._buffers.values():
                defer_resource_release(buffer)
            self.adam_optimizer._buffers = {}
            self.adam_optimizer._capacity = 0

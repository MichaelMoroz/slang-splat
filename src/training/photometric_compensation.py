from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from ..scene import ColmapFrame, ColmapReconstruction
from ..scene._internal.colmap_ops import load_training_frame_rgba8
from ..utility import (
    RO_BUFFER_USAGE,
    RW_BUFFER_USAGE,
    SRV_TEXTURE_USAGE,
    SHADER_ROOT,
    alloc_buffer,
    alloc_texture_2d,
    buffer_to_numpy,
    defer_resource_release,
    dispatch,
    drain_deferred_resource_releases,
    load_compute_items,
    thread_count_1d,
)
from .adam import AdamOptimizer, AdamRuntimeHyperParams
from .ppisp import PPISP_FIELD_SPECS, PPISP_PACKED_PARAM_COUNT, PPISPTonemapParams, PPISPTonemapProvider

_PARAM_SETTINGS_U32_WIDTH = 10
_PAIR_GRAD_THREADS = 64
_PHOTOMETRIC_SHADER_PATH = SHADER_ROOT / "utility" / "photometric_compensation.slang"
_PHOTOMETRIC_FULL_PAIR_DATASET_MAX_PAIRS = 1_000_000
_PPISP_IDENTITY_VALUES = np.concatenate(
    [
        np.asarray(spec.default if spec.size > 1 else (spec.default,), dtype=np.float32).reshape(spec.size)
        for spec in PPISP_FIELD_SPECS
    ],
    axis=0,
).astype(np.float32, copy=False)


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
    gamma_regularize_weight: float | None = None
    exposure_l1_weight: float = 0.0
    vignette_l1_weight: float = 0.01
    chroma_l1_weight: float = 0.02
    crf_l1_weight: float = 0.02
    gamma_l1_weight: float | None = None
    grad_component_clip: float = 10.0
    grad_norm_clip: float = 10.0
    max_update: float = 0.05
    huge_value: float = 1e8
    ema_decay: float = 0.95

    def __post_init__(self) -> None:
        self.batch_pair_count = max(int(self.batch_pair_count), 1)
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
        if self.gamma_regularize_weight is None:
            self.gamma_regularize_weight = float(self.crf_regularize_weight)
        self.gamma_regularize_weight = float(max(self.gamma_regularize_weight, 0.0))
        self.exposure_l1_weight = float(max(self.exposure_l1_weight, 0.0))
        self.vignette_l1_weight = float(max(self.vignette_l1_weight, 0.0))
        self.chroma_l1_weight = float(max(self.chroma_l1_weight, 0.0))
        self.crf_l1_weight = float(max(self.crf_l1_weight, 0.0))
        if self.gamma_l1_weight is None:
            self.gamma_l1_weight = float(self.crf_l1_weight)
        self.gamma_l1_weight = float(max(self.gamma_l1_weight, 0.0))
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
class PhotometricObservationTrackPool:
    point_ids: np.ndarray
    track_lengths: np.ndarray
    observation_ranges: np.ndarray
    observation_frame_indices: np.ndarray
    observation_xy: np.ndarray
    pair_ranges: np.ndarray

    def __len__(self) -> int:
        return int(self.pair_ranges[-1]) if self.pair_ranges.size > 0 else 0

    def sample(self, rng: np.random.Generator, pair_count: int) -> PhotometricObservationPairBatch:
        requested = max(int(pair_count), 0)
        total = len(self)
        if requested <= 0 or total <= 0:
            empty_i64 = np.zeros((0,), dtype=np.int64)
            empty_i32 = np.zeros((0,), dtype=np.int32)
            empty_xy = np.zeros((0, 2), dtype=np.float32)
            return PhotometricObservationPairBatch(empty_i64, empty_i64, empty_i32, empty_i32, empty_i32, empty_xy, empty_xy)
        indices = np.asarray(rng.integers(0, total, size=requested, endpoint=False), dtype=np.int64)
        track_indices = np.searchsorted(np.asarray(self.pair_ranges[1:], dtype=np.int64), indices, side="right")
        local_pair_indices = indices - np.asarray(self.pair_ranges[track_indices], dtype=np.int64)
        point_ids = np.empty((requested,), dtype=np.int64)
        track_lengths = np.empty((requested,), dtype=np.int32)
        frame_indices_a = np.empty((requested,), dtype=np.int32)
        frame_indices_b = np.empty((requested,), dtype=np.int32)
        xy_a = np.empty((requested, 2), dtype=np.float32)
        xy_b = np.empty((requested, 2), dtype=np.float32)
        for batch_index in range(requested):
            track_index = int(track_indices[batch_index])
            obs_start = int(self.observation_ranges[track_index])
            obs_stop = int(self.observation_ranges[track_index + 1])
            left_index, right_index = _decode_observation_pair_index(int(local_pair_indices[batch_index]), obs_stop - obs_start)
            left_obs_index = obs_start + left_index
            right_obs_index = obs_start + right_index
            point_ids[batch_index] = int(self.point_ids[track_index])
            track_lengths[batch_index] = int(self.track_lengths[track_index])
            frame_indices_a[batch_index] = int(self.observation_frame_indices[left_obs_index])
            frame_indices_b[batch_index] = int(self.observation_frame_indices[right_obs_index])
            xy_a[batch_index] = np.asarray(self.observation_xy[left_obs_index], dtype=np.float32)
            xy_b[batch_index] = np.asarray(self.observation_xy[right_obs_index], dtype=np.float32)
        return PhotometricObservationPairBatch(
            pair_indices=np.ascontiguousarray(indices, dtype=np.int64),
            point_ids=np.ascontiguousarray(point_ids, dtype=np.int64),
            track_lengths=np.ascontiguousarray(track_lengths, dtype=np.int32),
            frame_indices_a=np.ascontiguousarray(frame_indices_a, dtype=np.int32),
            frame_indices_b=np.ascontiguousarray(frame_indices_b, dtype=np.int32),
            xy_a=np.ascontiguousarray(xy_a, dtype=np.float32),
            xy_b=np.ascontiguousarray(xy_b, dtype=np.float32),
        )


@dataclass(frozen=True, slots=True)
class _PhotometricDispatchBatch:
    pair_batch: PhotometricObservationPairBatch
    frame_pair_ranges: np.ndarray
    frame_pair_entries: np.ndarray


def _decode_observation_pair_index(pair_index: int, observation_count: int) -> tuple[int, int]:
    remaining = max(int(observation_count) - 1, 0)
    left = 0
    rank = max(int(pair_index), 0)
    while remaining > 0 and rank >= remaining:
        rank -= remaining
        left += 1
        remaining -= 1
    return left, left + 1 + rank


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


def build_photometric_observation_track_pool(
    reconstruction: ColmapReconstruction,
    frames: list[ColmapFrame],
    *,
    min_track_length: int = 2,
) -> PhotometricObservationTrackPool:
    frame_lookup = {int(frame.image_id): (frame_index, frame) for frame_index, frame in enumerate(frames)}
    min_track = max(int(min_track_length), 2)
    points = getattr(reconstruction, "points3d", {})
    images = getattr(reconstruction, "images", {})
    if not frame_lookup or not images or not points:
        empty_i64 = np.zeros((0,), dtype=np.int64)
        empty_i32 = np.zeros((0,), dtype=np.int32)
        empty_xy = np.zeros((0, 2), dtype=np.float32)
        empty_ranges = np.zeros((1,), dtype=np.int64)
        return PhotometricObservationTrackPool(empty_i64, empty_i32, empty_ranges, empty_i32, empty_xy, empty_ranges.copy())

    max_point_id = max((int(point_id) for point_id in points), default=0)
    track_length_lookup = np.zeros((max(max_point_id, 0) + 1,), dtype=np.int32)
    for point_id, point in points.items():
        track_length = int(getattr(point, "track_length", 0))
        if track_length >= min_track:
            track_length_lookup[int(point_id)] = np.int32(track_length)

    point_id_chunks: list[np.ndarray] = []
    track_length_chunks: list[np.ndarray] = []
    frame_index_chunks: list[np.ndarray] = []
    xy_chunks: list[np.ndarray] = []
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
        point_ids = point_ids[:count]
        point_xy = point_xy[:count]
        valid = point_ids > 0
        valid &= point_ids < track_length_lookup.size
        if not np.any(valid):
            continue
        valid_ids = point_ids[valid]
        valid_xy = point_xy[valid]
        valid_track_lengths = track_length_lookup[valid_ids]
        valid &= False
        keep = valid_track_lengths >= min_track
        if not np.any(keep):
            continue
        valid_ids = valid_ids[keep]
        valid_xy = valid_xy[keep]
        valid_track_lengths = valid_track_lengths[keep]
        finite = np.isfinite(valid_xy).all(axis=1)
        finite &= valid_xy[:, 0] >= 0.0
        finite &= valid_xy[:, 0] <= float(frame.width)
        finite &= valid_xy[:, 1] >= 0.0
        finite &= valid_xy[:, 1] <= float(frame.height)
        if not np.any(finite):
            continue
        valid_ids = valid_ids[finite]
        valid_xy = valid_xy[finite]
        valid_track_lengths = valid_track_lengths[finite]
        _, first_positions = np.unique(valid_ids, return_index=True)
        if first_positions.size <= 0:
            continue
        first_positions = np.sort(first_positions)
        valid_ids = np.ascontiguousarray(valid_ids[first_positions], dtype=np.int64)
        valid_xy = np.ascontiguousarray(valid_xy[first_positions], dtype=np.float32)
        valid_track_lengths = np.ascontiguousarray(valid_track_lengths[first_positions], dtype=np.int32)
        point_id_chunks.append(valid_ids)
        track_length_chunks.append(valid_track_lengths)
        frame_index_chunks.append(np.full((valid_ids.size,), int(frame_index), dtype=np.int32))
        xy_chunks.append(valid_xy)

    if not point_id_chunks:
        empty_i64 = np.zeros((0,), dtype=np.int64)
        empty_i32 = np.zeros((0,), dtype=np.int32)
        empty_xy = np.zeros((0, 2), dtype=np.float32)
        empty_ranges = np.zeros((1,), dtype=np.int64)
        return PhotometricObservationTrackPool(empty_i64, empty_i32, empty_ranges, empty_i32, empty_xy, empty_ranges.copy())

    point_ids = np.ascontiguousarray(np.concatenate(point_id_chunks, axis=0), dtype=np.int64)
    track_lengths = np.ascontiguousarray(np.concatenate(track_length_chunks, axis=0), dtype=np.int32)
    frame_indices = np.ascontiguousarray(np.concatenate(frame_index_chunks, axis=0), dtype=np.int32)
    xy = np.ascontiguousarray(np.concatenate(xy_chunks, axis=0), dtype=np.float32)
    order = np.lexsort((xy[:, 1], xy[:, 0], frame_indices, point_ids))
    point_ids = np.ascontiguousarray(point_ids[order], dtype=np.int64)
    track_lengths = np.ascontiguousarray(track_lengths[order], dtype=np.int32)
    frame_indices = np.ascontiguousarray(frame_indices[order], dtype=np.int32)
    xy = np.ascontiguousarray(xy[order], dtype=np.float32)
    unique_point_ids, starts, counts = np.unique(point_ids, return_index=True, return_counts=True)
    valid_tracks = counts >= 2
    if not np.any(valid_tracks):
        empty_i64 = np.zeros((0,), dtype=np.int64)
        empty_i32 = np.zeros((0,), dtype=np.int32)
        empty_xy = np.zeros((0, 2), dtype=np.float32)
        empty_ranges = np.zeros((1,), dtype=np.int64)
        return PhotometricObservationTrackPool(empty_i64, empty_i32, empty_ranges, empty_i32, empty_xy, empty_ranges.copy())

    keep_observations = np.repeat(valid_tracks, counts)
    filtered_counts = np.ascontiguousarray(counts[valid_tracks], dtype=np.int64)
    observation_ranges = np.zeros((int(filtered_counts.size) + 1,), dtype=np.int64)
    observation_ranges[1:] = np.cumsum(filtered_counts, dtype=np.int64)
    pair_ranges = np.zeros((int(filtered_counts.size) + 1,), dtype=np.int64)
    pair_ranges[1:] = np.cumsum(filtered_counts * (filtered_counts - 1) // 2, dtype=np.int64)
    return PhotometricObservationTrackPool(
        point_ids=np.ascontiguousarray(unique_point_ids[valid_tracks], dtype=np.int64),
        track_lengths=np.ascontiguousarray(track_lengths[starts[valid_tracks]], dtype=np.int32),
        observation_ranges=np.ascontiguousarray(observation_ranges, dtype=np.int64),
        observation_frame_indices=np.ascontiguousarray(frame_indices[keep_observations], dtype=np.int32),
        observation_xy=np.ascontiguousarray(xy[keep_observations], dtype=np.float32),
        pair_ranges=np.ascontiguousarray(pair_ranges, dtype=np.int64),
    )


def materialize_photometric_observation_pair_pool(track_pool: PhotometricObservationTrackPool) -> PhotometricObservationPairPool:
    total_pairs = len(track_pool)
    if total_pairs <= 0:
        empty_i64 = np.zeros((0,), dtype=np.int64)
        empty_i32 = np.zeros((0,), dtype=np.int32)
        empty_xy = np.zeros((0, 2), dtype=np.float32)
        return PhotometricObservationPairPool(empty_i64, empty_i32, empty_i32, empty_i32, empty_xy, empty_xy)
    point_ids = np.empty((total_pairs,), dtype=np.int64)
    track_lengths = np.empty((total_pairs,), dtype=np.int32)
    frame_indices_a = np.empty((total_pairs,), dtype=np.int32)
    frame_indices_b = np.empty((total_pairs,), dtype=np.int32)
    xy_a = np.empty((total_pairs, 2), dtype=np.float32)
    xy_b = np.empty((total_pairs, 2), dtype=np.float32)
    write_index = 0
    for track_index in range(track_pool.point_ids.size):
        obs_start = int(track_pool.observation_ranges[track_index])
        obs_stop = int(track_pool.observation_ranges[track_index + 1])
        obs_count = obs_stop - obs_start
        if obs_count < 2:
            continue
        frames = np.asarray(track_pool.observation_frame_indices[obs_start:obs_stop], dtype=np.int32)
        xy = np.asarray(track_pool.observation_xy[obs_start:obs_stop], dtype=np.float32)
        point_id = int(track_pool.point_ids[track_index])
        track_length = int(track_pool.track_lengths[track_index])
        for left_index in range(obs_count - 1):
            count = obs_count - left_index - 1
            next_write = write_index + count
            point_ids[write_index:next_write] = point_id
            track_lengths[write_index:next_write] = track_length
            frame_indices_a[write_index:next_write] = frames[left_index]
            frame_indices_b[write_index:next_write] = frames[left_index + 1 :]
            xy_a[write_index:next_write] = xy[left_index]
            xy_b[write_index:next_write] = xy[left_index + 1 :]
            write_index = next_write
    return PhotometricObservationPairPool(
        point_ids=np.ascontiguousarray(point_ids[:write_index], dtype=np.int64),
        track_lengths=np.ascontiguousarray(track_lengths[:write_index], dtype=np.int32),
        frame_indices_a=np.ascontiguousarray(frame_indices_a[:write_index], dtype=np.int32),
        frame_indices_b=np.ascontiguousarray(frame_indices_b[:write_index], dtype=np.int32),
        xy_a=np.ascontiguousarray(xy_a[:write_index], dtype=np.float32),
        xy_b=np.ascontiguousarray(xy_b[:write_index], dtype=np.float32),
    )


def build_photometric_observation_pair_pool(
    reconstruction: ColmapReconstruction,
    frames: list[ColmapFrame],
    *,
    min_track_length: int = 2,
) -> PhotometricObservationPairPool:
    return materialize_photometric_observation_pair_pool(
        build_photometric_observation_track_pool(
            reconstruction,
            frames,
            min_track_length=min_track_length,
        )
    )


def _field_group_name(attr: str) -> str:
    if attr == "exposureEv":
        return "exposure"
    if attr.startswith("vignette"):
        return "vignette"
    if attr.startswith("chroma"):
        return "chroma"
    if attr == "crfGamma":
        return "gamma"
    return "crf"


def _field_lr_mul(attr: str, hparams: PhotometricCompensationHyperParams) -> float:
    group = _field_group_name(attr)
    if group == "exposure":
        return float(hparams.exposure_lr_mul)
    if group == "vignette":
        return float(hparams.vignette_lr_mul)
    if group == "chroma":
        return float(hparams.chroma_lr_mul)
    if group == "gamma":
        return float(hparams.crf_lr_mul)
    return float(hparams.crf_lr_mul)


def _field_regularize_weight(attr: str, hparams: PhotometricCompensationHyperParams) -> float:
    group = _field_group_name(attr)
    if group == "exposure":
        return float(hparams.exposure_regularize_weight)
    if group == "vignette":
        return float(hparams.vignette_regularize_weight)
    if group == "chroma":
        return float(hparams.chroma_regularize_weight)
    if group == "gamma":
        return float(hparams.gamma_regularize_weight)
    return float(hparams.crf_regularize_weight)


def _field_l1_weight(attr: str, hparams: PhotometricCompensationHyperParams) -> float:
    group = _field_group_name(attr)
    if group == "exposure":
        return float(hparams.exposure_l1_weight)
    if group == "vignette":
        return float(hparams.vignette_l1_weight)
    if group == "chroma":
        return float(hparams.chroma_l1_weight)
    if group == "gamma":
        return float(hparams.gamma_l1_weight)
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


def _pair_dataset_sample_count(neighborhood_size: int) -> int:
    size = max(int(neighborhood_size), 1)
    return size * size


def _pair_pool_batch(pair_pool: PhotometricObservationPairPool) -> PhotometricObservationPairBatch:
    pair_count = len(pair_pool)
    return PhotometricObservationPairBatch(
        pair_indices=np.arange(pair_count, dtype=np.int64),
        point_ids=np.ascontiguousarray(pair_pool.point_ids, dtype=np.int64),
        track_lengths=np.ascontiguousarray(pair_pool.track_lengths, dtype=np.int32),
        frame_indices_a=np.ascontiguousarray(pair_pool.frame_indices_a, dtype=np.int32),
        frame_indices_b=np.ascontiguousarray(pair_pool.frame_indices_b, dtype=np.int32),
        xy_a=np.ascontiguousarray(pair_pool.xy_a, dtype=np.float32),
        xy_b=np.ascontiguousarray(pair_pool.xy_b, dtype=np.float32),
    )


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
        frame_source_textures: list[spy.Texture] | None = None,
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
        self._kernels = load_compute_items(
            self.device,
            {
                "build_pair_dataset": ("kernel", _PHOTOMETRIC_SHADER_PATH, "csBuildPhotometricPairDataset"),
                "pair_loss_backward": ("kernel", _PHOTOMETRIC_SHADER_PATH, "csPhotometricPairLossBackward"),
            },
        )
        self._provider = PackedPPISPTonemapProvider(self._frame_count)
        self._frame_rgba_linear = None if frame_rgba_linear is None else [
            _coerce_frame_rgba_linear(frame, frame_rgba_linear[index])
            for index, frame in enumerate(self.frames)
        ]
        if frame_source_textures is not None and len(frame_source_textures) != self._frame_count:
            raise ValueError(f"Expected {self._frame_count} frame source textures, got {len(frame_source_textures)}.")
        self._frame_source_textures = None if frame_source_textures is None else tuple(frame_source_textures)
        self._pair_dataset_uploaded = False
        self._pair_dataset_metadata_uploaded = False
        self._pair_dataset_sample_capacity = 0
        self._pair_dataset_pair_capacity = 0
        self._pair_dataset_entry_capacity = 0
        self._pair_dataset_prepare_started = False
        self._pair_dataset_prepare_frame_indices: tuple[int, ...] = ()
        self._pair_dataset_prepare_index = 0
        self._batch_pair_capacity = 0
        self._frame_pair_entry_capacity = 0
        self.track_pool = build_photometric_observation_track_pool(
            self.reconstruction,
            self.frames,
            min_track_length=self.hparams.min_track_length,
        )
        if len(self.track_pool) <= 0:
            raise ValueError("Photometric compensation requires at least one valid cross-view sparse-track pair.")
        self._pair_dataset_full_precompute_enabled = len(self.track_pool) <= _PHOTOMETRIC_FULL_PAIR_DATASET_MAX_PAIRS
        self.pair_pool = materialize_photometric_observation_pair_pool(self.track_pool) if self._pair_dataset_full_precompute_enabled else self.track_pool
        self._pair_dataset_sample_count = _pair_dataset_sample_count(self.hparams.neighborhood_size)
        self._pair_dataset_dispatch = _build_frame_pair_batch(self._frame_count, _pair_pool_batch(self.pair_pool)) if self._pair_dataset_full_precompute_enabled else None
        if not self._pair_dataset_full_precompute_enabled:
            self._pair_dataset_uploaded = True
            self._pair_dataset_metadata_uploaded = True
        self.adam_optimizer = AdamOptimizer(self.device, self.adam, self._runtime_hparams())
        self._ensure_buffers()
        self._upload_param_settings()
        self.reset()

    @staticmethod
    def _reference_frame_index(frame_count: int) -> int | None:
        return 0 if int(frame_count) > 0 else None

    def _anchor_reference_frame_params(self, packed: np.ndarray) -> bool:
        anchor_frame = self._reference_frame_index(self._frame_count)
        if anchor_frame is None:
            return False
        reference = _PPISP_IDENTITY_VALUES.reshape(PPISP_PACKED_PARAM_COUNT)
        if np.array_equal(np.asarray(packed[:, anchor_frame], dtype=np.float32), reference):
            return False
        packed[:, anchor_frame] = reference
        return True

    def _anchor_reference_frame_buffer(self, packed: np.ndarray | None = None) -> np.ndarray:
        anchored = self.read_packed_params().copy() if packed is None else np.ascontiguousarray(packed, dtype=np.float32).copy()
        if self._anchor_reference_frame_params(anchored):
            self._buffers["params"].copy_from_numpy(anchored.reshape(-1))
        return anchored

    def _runtime_hparams(self) -> AdamRuntimeHyperParams:
        return AdamRuntimeHyperParams(
            grad_component_clip=float(self.hparams.grad_component_clip),
            grad_norm_clip=float(self.hparams.grad_norm_clip),
            max_update=float(self.hparams.max_update),
            huge_value=float(self.hparams.huge_value),
        )

    @property
    def uses_full_pair_dataset(self) -> bool:
        return bool(self._pair_dataset_full_precompute_enabled)

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

    def _ensure_pair_dataset_buffers(self, sample_count: int) -> None:
        required = max(int(sample_count), 1)
        if self._pair_dataset_sample_capacity >= required and all(name in self._buffers for name in ("pair_samples_a", "pair_samples_b", "pair_sensor_coords_a", "pair_sensor_coords_b")):
            return
        for name in ("pair_samples_a", "pair_samples_b", "pair_sensor_coords_a", "pair_sensor_coords_b"):
            defer_resource_release(self._buffers.get(name))
        self._buffers["pair_samples_a"] = alloc_buffer(self.device, name="photometric_compensation.pair_samples_a", size=required * 16, usage=RW_BUFFER_USAGE)
        self._buffers["pair_samples_b"] = alloc_buffer(self.device, name="photometric_compensation.pair_samples_b", size=required * 16, usage=RW_BUFFER_USAGE)
        self._buffers["pair_sensor_coords_a"] = alloc_buffer(self.device, name="photometric_compensation.pair_sensor_coords_a", size=required * 8, usage=RW_BUFFER_USAGE)
        self._buffers["pair_sensor_coords_b"] = alloc_buffer(self.device, name="photometric_compensation.pair_sensor_coords_b", size=required * 8, usage=RW_BUFFER_USAGE)
        self._pair_dataset_sample_capacity = required

    def _ensure_pair_dataset_metadata_buffers(self, pair_count: int, entry_count: int) -> None:
        required_pairs = max(int(pair_count), 1)
        required_entries = max(int(entry_count), 1)
        if self._pair_dataset_pair_capacity < required_pairs or "pair_xy_a" not in self._buffers:
            for name in ("pair_xy_a", "pair_xy_b"):
                defer_resource_release(self._buffers.get(name))
            self._buffers["pair_xy_a"] = alloc_buffer(self.device, name="photometric_compensation.pair_xy_a", size=required_pairs * 8, usage=RO_BUFFER_USAGE)
            self._buffers["pair_xy_b"] = alloc_buffer(self.device, name="photometric_compensation.pair_xy_b", size=required_pairs * 8, usage=RO_BUFFER_USAGE)
            self._pair_dataset_pair_capacity = required_pairs
        if self._pair_dataset_entry_capacity < required_entries or "dataset_frame_pair_entries" not in self._buffers:
            for name in ("dataset_frame_pair_ranges", "dataset_frame_pair_entries"):
                defer_resource_release(self._buffers.get(name))
            self._buffers["dataset_frame_pair_ranges"] = alloc_buffer(
                self.device,
                name="photometric_compensation.dataset_frame_pair_ranges",
                size=max(self._frame_count + 1, 1) * 4,
                usage=RO_BUFFER_USAGE,
            )
            self._buffers["dataset_frame_pair_entries"] = alloc_buffer(
                self.device,
                name="photometric_compensation.dataset_frame_pair_entries",
                size=required_entries * 8,
                usage=RO_BUFFER_USAGE,
            )
            self._pair_dataset_entry_capacity = required_entries

    def _ensure_batch_buffers(self, pair_count: int) -> None:
        required_pairs = max(int(pair_count), 1)
        required_entries = max(int(pair_count) * 2, 1)
        if self._batch_pair_capacity < required_pairs or "pair_indices" not in self._buffers:
            for name in ("pair_indices", "pair_frame_indices_a", "pair_frame_indices_b"):
                defer_resource_release(self._buffers.get(name))
            self._buffers["pair_indices"] = alloc_buffer(self.device, name="photometric_compensation.pair_indices", size=required_pairs * 4, usage=RO_BUFFER_USAGE)
            self._buffers["pair_frame_indices_a"] = alloc_buffer(self.device, name="photometric_compensation.pair_frame_indices_a", size=required_pairs * 4, usage=RO_BUFFER_USAGE)
            self._buffers["pair_frame_indices_b"] = alloc_buffer(self.device, name="photometric_compensation.pair_frame_indices_b", size=required_pairs * 4, usage=RO_BUFFER_USAGE)
            self._batch_pair_capacity = required_pairs
        if self._frame_pair_entry_capacity < required_entries or "frame_pair_entries" not in self._buffers:
            for name in ("frame_pair_ranges", "frame_pair_entries"):
                defer_resource_release(self._buffers.get(name))
            self._buffers["frame_pair_ranges"] = alloc_buffer(self.device, name="photometric_compensation.frame_pair_ranges", size=max(self._frame_count + 1, 1) * 4, usage=RO_BUFFER_USAGE)
            self._buffers["frame_pair_entries"] = alloc_buffer(self.device, name="photometric_compensation.frame_pair_entries", size=required_entries * 8, usage=RO_BUFFER_USAGE)
            self._frame_pair_entry_capacity = required_entries

    def _upload_pair_dataset_metadata(self) -> None:
        if not self._pair_dataset_full_precompute_enabled or self._pair_dataset_metadata_uploaded or self._pair_dataset_dispatch is None:
            return
        entry_count = int(self._pair_dataset_dispatch.frame_pair_entries.shape[0])
        self._ensure_pair_dataset_metadata_buffers(len(self.pair_pool), entry_count)
        self._buffers["pair_xy_a"].copy_from_numpy(np.ascontiguousarray(self.pair_pool.xy_a, dtype=np.float32))
        self._buffers["pair_xy_b"].copy_from_numpy(np.ascontiguousarray(self.pair_pool.xy_b, dtype=np.float32))
        self._buffers["dataset_frame_pair_ranges"].copy_from_numpy(self._pair_dataset_dispatch.frame_pair_ranges)
        self._buffers["dataset_frame_pair_entries"].copy_from_numpy(self._pair_dataset_dispatch.frame_pair_entries)
        self._pair_dataset_metadata_uploaded = True

    @staticmethod
    def _pair_dataset_frame_indices_from_ranges(frame_pair_ranges: np.ndarray) -> tuple[int, ...]:
        frame_ranges = np.asarray(frame_pair_ranges, dtype=np.uint32).reshape(-1)
        if frame_ranges.size <= 1:
            return ()
        active = np.flatnonzero(frame_ranges[1:] > frame_ranges[:-1])
        return tuple(int(frame_index) for frame_index in active)

    def _pair_dataset_work_frame_indices(self) -> tuple[int, ...]:
        if self._pair_dataset_dispatch is None:
            return ()
        return self._pair_dataset_frame_indices_from_ranges(self._pair_dataset_dispatch.frame_pair_ranges)

    def _pair_dataset_source_texture(self, frame_index: int) -> tuple[spy.Texture, bool]:
        if self._frame_source_textures is not None:
            return self._frame_source_textures[frame_index], False
        frame = self.frames[frame_index]
        if self._frame_rgba_linear is not None:
            rgba = _coerce_frame_rgba_linear(frame, self._frame_rgba_linear[frame_index])
            texture = alloc_texture_2d(
                self.device,
                name="photometric_compensation.frame_source",
                format=spy.Format.rgba32_float,
                width=int(frame.width),
                height=int(frame.height),
                usage=SRV_TEXTURE_USAGE,
            )
            texture.copy_from_numpy(rgba)
            return texture, True
        rgba8 = np.ascontiguousarray(load_training_frame_rgba8(frame), dtype=np.uint8)
        texture = alloc_texture_2d(
            self.device,
            name="photometric_compensation.frame_source",
            format=spy.Format.rgba8_unorm_srgb,
            width=int(frame.width),
            height=int(frame.height),
            usage=SRV_TEXTURE_USAGE,
        )
        texture.copy_from_numpy(rgba8)
        return texture, True

    def _dispatch_build_pair_dataset(self, encoder: spy.CommandEncoder, frame_pair_ranges: np.ndarray, frame_index: int, source_texture: spy.Texture) -> None:
        start = int(frame_pair_ranges[frame_index])
        stop = int(frame_pair_ranges[frame_index + 1])
        if stop <= start:
            return
        frame = self.frames[frame_index]
        dispatch(
            kernel=self._kernels["build_pair_dataset"],
            thread_count=thread_count_1d((stop - start) * self._pair_dataset_sample_count),
            vars={
                "g_PairDatasetSourceFrame": source_texture,
                "g_PairSamplesA": self._buffers["pair_samples_a"],
                "g_PairSamplesB": self._buffers["pair_samples_b"],
                "g_PairSensorCoordsA": self._buffers["pair_sensor_coords_a"],
                "g_PairSensorCoordsB": self._buffers["pair_sensor_coords_b"],
                "g_PairXYA": self._buffers["pair_xy_a"],
                "g_PairXYB": self._buffers["pair_xy_b"],
                "g_DatasetFramePairRanges": self._buffers["dataset_frame_pair_ranges"],
                "g_DatasetFramePairEntries": self._buffers["dataset_frame_pair_entries"],
                "g_DatasetFrameIndex": int(frame_index),
                "g_DatasetFrameWidth": int(frame.width),
                "g_DatasetFrameHeight": int(frame.height),
                "g_NeighborhoodSize": int(self.hparams.neighborhood_size),
                "g_NeighborhoodSampleCount": int(self._pair_dataset_sample_count),
            },
            command_encoder=encoder,
            debug_label="Photometric::build_pair_dataset",
            debug_color_index=92,
        )

    def _reset_pair_dataset_prepare_state(self) -> None:
        self._pair_dataset_prepare_started = False
        self._pair_dataset_prepare_frame_indices = ()
        self._pair_dataset_prepare_index = 0

    def _finalize_pair_dataset_prepare(self) -> None:
        self._frame_rgba_linear = None
        self._pair_dataset_uploaded = True
        self._reset_pair_dataset_prepare_state()

    @property
    def pair_dataset_prepare_active(self) -> bool:
        return bool(self._pair_dataset_prepare_started) and not bool(self._pair_dataset_uploaded)

    @property
    def pair_dataset_prepare_total_frames(self) -> int:
        if self._pair_dataset_prepare_started:
            return int(len(self._pair_dataset_prepare_frame_indices))
        if self._pair_dataset_uploaded:
            return int(len(self._pair_dataset_work_frame_indices()))
        return 0

    @property
    def pair_dataset_prepare_completed_frames(self) -> int:
        if self._pair_dataset_prepare_started:
            return min(int(self._pair_dataset_prepare_index), int(len(self._pair_dataset_prepare_frame_indices)))
        return int(self.pair_dataset_prepare_total_frames) if self._pair_dataset_uploaded else 0

    @property
    def pair_dataset_prepare_fraction(self) -> float:
        total = int(self.pair_dataset_prepare_total_frames)
        if total <= 0:
            return 1.0 if self._pair_dataset_uploaded else 0.0
        return min(max(float(self.pair_dataset_prepare_completed_frames) / float(total), 0.0), 1.0)

    @property
    def pair_dataset_prepare_current_name(self) -> str:
        if not self._pair_dataset_prepare_started:
            return ""
        if self._pair_dataset_prepare_index >= len(self._pair_dataset_prepare_frame_indices):
            return ""
        frame_index = int(self._pair_dataset_prepare_frame_indices[self._pair_dataset_prepare_index])
        image_path = getattr(self.frames[frame_index], "image_path", f"frame_{frame_index}")
        return Path(image_path).name

    def begin_prepare_pair_dataset(self) -> None:
        if self._pair_dataset_uploaded or self._pair_dataset_prepare_started:
            return
        self._upload_pair_dataset_metadata()
        self._ensure_pair_dataset_buffers(len(self.pair_pool) * self._pair_dataset_sample_count)
        self._pair_dataset_prepare_frame_indices = self._pair_dataset_work_frame_indices()
        self._pair_dataset_prepare_index = 0
        self._pair_dataset_prepare_started = True
        if not self._pair_dataset_prepare_frame_indices:
            self._finalize_pair_dataset_prepare()

    def advance_prepare_pair_dataset(self, frame_budget: int = 1) -> bool:
        if self._pair_dataset_uploaded:
            return True
        self.begin_prepare_pair_dataset()
        if self._pair_dataset_uploaded:
            return True
        budget = max(int(frame_budget), 1)
        while budget > 0 and self._pair_dataset_prepare_index < len(self._pair_dataset_prepare_frame_indices):
            frame_index = int(self._pair_dataset_prepare_frame_indices[self._pair_dataset_prepare_index])
            source_texture, owns_texture = self._pair_dataset_source_texture(frame_index)
            encoder = self.device.create_command_encoder()
            self._dispatch_build_pair_dataset(encoder, self._pair_dataset_dispatch.frame_pair_ranges, frame_index, source_texture)
            self.device.submit_command_buffer(encoder.finish())
            self.device.wait()
            if owns_texture:
                defer_resource_release(source_texture)
                drain_deferred_resource_releases(min_age=0)
            self._pair_dataset_prepare_index += 1
            budget -= 1
        if self._pair_dataset_prepare_index >= len(self._pair_dataset_prepare_frame_indices):
            self._finalize_pair_dataset_prepare()
        return bool(self._pair_dataset_uploaded)

    def prepare_pair_dataset(self) -> None:
        if self._pair_dataset_uploaded:
            return
        self.begin_prepare_pair_dataset()
        self.advance_prepare_pair_dataset(frame_budget=max(self.pair_dataset_prepare_total_frames, 1))

    def _upload_pair_dataset(self) -> None:
        self.prepare_pair_dataset()

    def _upload_pair_dataset_metadata_for_batch(self, dispatch_batch: _PhotometricDispatchBatch) -> None:
        pair_batch = dispatch_batch.pair_batch
        entry_count = int(dispatch_batch.frame_pair_entries.shape[0])
        self._ensure_pair_dataset_metadata_buffers(pair_batch.pair_count, entry_count)
        self._buffers["pair_xy_a"].copy_from_numpy(np.ascontiguousarray(pair_batch.xy_a, dtype=np.float32))
        self._buffers["pair_xy_b"].copy_from_numpy(np.ascontiguousarray(pair_batch.xy_b, dtype=np.float32))
        self._buffers["dataset_frame_pair_ranges"].copy_from_numpy(np.ascontiguousarray(dispatch_batch.frame_pair_ranges, dtype=np.uint32))
        self._buffers["dataset_frame_pair_entries"].copy_from_numpy(np.ascontiguousarray(dispatch_batch.frame_pair_entries, dtype=np.uint32))

    def _prepare_pair_dataset_for_batch(self, dispatch_batch: _PhotometricDispatchBatch) -> None:
        pair_batch = dispatch_batch.pair_batch
        self._ensure_pair_dataset_buffers(pair_batch.pair_count * self._pair_dataset_sample_count)
        self._upload_pair_dataset_metadata_for_batch(dispatch_batch)
        active_frames = self._pair_dataset_frame_indices_from_ranges(dispatch_batch.frame_pair_ranges)
        if not active_frames:
            return
        encoder = self.device.create_command_encoder()
        owned_textures: list[spy.Texture] = []
        for frame_index in active_frames:
            source_texture, owns_texture = self._pair_dataset_source_texture(frame_index)
            self._dispatch_build_pair_dataset(encoder, dispatch_batch.frame_pair_ranges, frame_index, source_texture)
            if owns_texture:
                owned_textures.append(source_texture)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        for texture in owned_textures:
            defer_resource_release(texture)
        if owned_textures:
            drain_deferred_resource_releases(min_age=0)

    def _upload_pair_batch(self, dispatch_batch: _PhotometricDispatchBatch) -> None:
        pair_batch = dispatch_batch.pair_batch
        self._ensure_batch_buffers(pair_batch.pair_count)
        if self._pair_dataset_full_precompute_enabled:
            pair_indices = np.ascontiguousarray(pair_batch.pair_indices, dtype=np.uint32)
        else:
            pair_indices = np.arange(pair_batch.pair_count, dtype=np.uint32)
        self._buffers["pair_indices"].copy_from_numpy(pair_indices)
        self._buffers["pair_frame_indices_a"].copy_from_numpy(np.ascontiguousarray(pair_batch.frame_indices_a, dtype=np.uint32))
        self._buffers["pair_frame_indices_b"].copy_from_numpy(np.ascontiguousarray(pair_batch.frame_indices_b, dtype=np.uint32))
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
                "g_PairIndices": self._buffers["pair_indices"],
                "g_PairFrameIndicesA": self._buffers["pair_frame_indices_a"],
                "g_PairFrameIndicesB": self._buffers["pair_frame_indices_b"],
                "g_PairSamplesA": self._buffers["pair_samples_a"],
                "g_PairSamplesB": self._buffers["pair_samples_b"],
                "g_PairSensorCoordsA": self._buffers["pair_sensor_coords_a"],
                "g_PairSensorCoordsB": self._buffers["pair_sensor_coords_b"],
                "g_FramePairRanges": self._buffers["frame_pair_ranges"],
                "g_FramePairEntries": self._buffers["frame_pair_entries"],
                "g_FrameCount": int(self._frame_count),
                "g_PairCount": int(pair_count),
                "g_NeighborhoodSampleCount": int(self.hparams.neighborhood_size * self.hparams.neighborhood_size),
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
        packed = self._anchor_reference_frame_buffer()
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
        return self.pair_pool.sample(self._rng, requested)

    def build_dispatch_batch(self, pair_count: int | None = None) -> _PhotometricDispatchBatch:
        return _build_frame_pair_batch(self._frame_count, self.sample_pair_batch(pair_count))

    def replace_packed_params(self, packed_params: np.ndarray) -> None:
        reshaped = _reshape_packed_params(packed_params, self._frame_count)
        self._anchor_reference_frame_params(reshaped)
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
        packed = self._anchor_reference_frame_buffer()
        self._provider.replace_packed_params(packed)

    def train_step(self, pair_count: int | None = None, *, step_index: int | None = None) -> float:
        dispatch_batch = self.build_dispatch_batch(pair_count)
        resolved_step = self.state.step + 1 if step_index is None else int(step_index)
        if self._pair_dataset_full_precompute_enabled:
            self.prepare_pair_dataset()
        else:
            self._prepare_pair_dataset_for_batch(dispatch_batch)
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

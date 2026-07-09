"""A-priori VRAM estimation for the COLMAP importer.

Given a target splat count, max SH band, training resolution and the dataset
frame sizes, estimate how much GPU memory training will need so the importer can
(a) decide whether the whole dataset can live in VRAM or must be streamed and
(b) warn when a splat-count / SH-band choice won't fit on the current GPU.

The per-buffer multipliers mirror the real allocations:
  - renderer scene + optimizer: src/renderer/gaussian_renderer.py, src/training/adam.py
  - trainer render workspace + refinement: src/training/gaussian_trainer.py
  - renderer work buffers: src/renderer/gaussian_renderer.py (_ensure_work_buffers)
  - dataset textures: src/training/dataset_texture_pool.py
Buffers scale with the *target* splat count (the high-water capacity training
will grow into), the training-render pixel count, or the SSIM pixel count.

Everything here is pure arithmetic so it can be unit-tested without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

# --- SH / parameter sizing (mirrors gaussian_renderer.sh_coeff_count_for_band) --
_SH_COEFF_COUNT = (1, 4, 9, 16)  # bands 0..3
# packed trainable params = pos(3)+scale(3)+rot(4)+opacity(1) + 3*sh_coeffs
_BASE_PARAMS = 11


def sh_coeff_count_for_band(max_sh_band: int) -> int:
    return _SH_COEFF_COUNT[min(max(int(max_sh_band), 0), 3)]


def packed_param_count_for_band(max_sh_band: int) -> int:
    return _BASE_PARAMS + 3 * sh_coeff_count_for_band(max_sh_band)


# --- per-splat byte multipliers (bytes per splat at capacity) -------------------
# Trainer/renderer buffers grow via _grow_trainer_buffer_capacity (+base//16), so
# the live high-water allocation overshoots the splat count by up to ~1/16.
_CAPACITY_OVERGROW = 1.0625
# Refinement dst_*/append_* buffers size to splats + clone budget; the clone
# budget is capped at refinement_max_growth_per_step (default 0.30) of survivors
# (resolve_refinement_clone_budget in src/training/schedule.py).
_REFINE_GROWTH = 0.30

# Constant (SH-independent) per-splat bytes, summed from the allocation sites.
# renderer.work: screen*3(48) + visible/area/fallback(12) + debug age/norm/viewed(12)
#   + debug_grad_stats(8) + visible_keys/values(8) + scanline_counts/offsets(8)
#   + raster_cache(13*4=52) = 148
_WORK_CONST = 148
# trainer render workspace: splat_contribution(16) + 3*raster_grads(3*13*4=156) = 172
_WORKSPACE_CONST = 172
# trainer refinement, splat-capacity buffers: gradient_stats(8) + contribution(16)
#   + 11 misc u32 buffers(44) + splat_age(4) = 72
_REFINEMENT_SPLAT_CONST = 72
# trainer refinement, output-capacity (splats+append) u32 buffers:
#   dst_splat_age(4) + dst_contribution_history(4) + dst_viewed_fraction_history(4) = 12
_REFINEMENT_OUTPUT_CONST = 12
# trainer refinement, append-capacity u32 buffer: append_splat_age(4)
_REFINEMENT_APPEND_CONST = 4
# renderer scene splat_highlight(4)
_SCENE_CONST = 4
# Per-splat bytes that scale with packed param count P:
#   scene splat_params(4P) + adam_moments(8P) + workspace param_grads(4P) = 16P,
#   refinement dst_splat_params(4P, sized to splats+append) and append_params(4P).
_PERSPLAT_PARAM_BASE = 16


def _refine_growth(value: float) -> float:
    return max(float(value), 0.0)


def _persplat_const(refinement_growth_per_step: float = _REFINE_GROWTH) -> float:
    growth = _refine_growth(refinement_growth_per_step)
    return (
        _WORK_CONST
        + _WORKSPACE_CONST
        + _SCENE_CONST
        + _REFINEMENT_SPLAT_CONST
        + _REFINEMENT_OUTPUT_CONST * (1.0 + growth)
        + _REFINEMENT_APPEND_CONST * growth
    )


_PERSPLAT_CONST = _persplat_const()


def _persplat_param_factor(refinement_growth_per_step: float = _REFINE_GROWTH) -> float:
    growth = _refine_growth(refinement_growth_per_step)
    return _PERSPLAT_PARAM_BASE + 4.0 * (1.0 + growth) + 4.0 * growth


def per_splat_bytes(max_sh_band: int, refinement_growth_per_step: float = _REFINE_GROWTH) -> int:
    raw = _persplat_param_factor(refinement_growth_per_step) * packed_param_count_for_band(max_sh_band) + _persplat_const(refinement_growth_per_step)
    return int(round(raw * _CAPACITY_OVERGROW))


# --- per-pixel byte multipliers ------------------------------------------------
# trainer workspace per training-render pixel:
#   forward_state(16)+density(4)+rgb_loss(4)+target_edge(4)+regularizer(8)
#   +processed_end(4)+output_grad(16) = 56 ; depth_stats texture rgba32f = 16
_PERPIXEL_TRAIN = 56 + 16
# SSIM: 4 feature buffers + 1 blur scratch, each SSIM_FEATURE_CHANNELS(16) floats/px
_SSIM_FEATURE_CHANNELS = 16
_PERPIXEL_SSIM = 5 * _SSIM_FEATURE_CHANNELS * 4

# --- prepass work lists (visibility/budget dependent; modelled, not exact) ------
# Per visible (tile,splat) list entry: keys+values(2*4) + radix double-buffer
# keys/values(4*4) = 24 B. Conservatively assume this many list entries per splat.
_PREPASS_BYTES_PER_LIST_ENTRY = 24
_DEFAULT_LIST_ENTRIES_PER_SPLAT = 8
# Scanline work items + tile counts/offsets also scale with visible splats; the
# captures show ~64 B/splat on top of the list buffers. Folded in conservatively.
_PREPASS_SCANLINE_BYTES_PER_SPLAT = 64

# Fixed driver/runtime overhead not tracked per-resource (sort histograms,
# metrics, command buffers, driver reserve). Seen ~300-560 MiB in captures.
_FIXED_OVERHEAD_BYTES = 400 * MIB

# Driver texture alignment: BC7 textures round the block-row count up (observed
# 1326 -> 1408 block rows, i.e. a multiple of 128) so the resident texture is
# larger than the BC7 payload/staging buffer. rgba8 textures align the row pitch.
_BC7_BLOCK_ROW_ALIGN = 128
_TEXTURE_ROW_PITCH_ALIGN = 256


def _align_up(value: int, alignment: int) -> int:
    return ((max(int(value), 0) + alignment - 1) // alignment) * alignment


def _bc7_block_count(value: int) -> int:
    return (max(int(value), 1) + 3) // 4


def bc7_payload_bytes(width: int, height: int) -> int:
    """BC7 payload / staging-buffer bytes (16 bytes per 4x4 block)."""
    return _bc7_block_count(width) * _bc7_block_count(height) * 16


def bc7_texture_bytes(width: int, height: int) -> int:
    """Resident BC7 texture bytes, including driver block-row alignment.

    Matches the captured footprint (7952x5304 -> 44,785,664 bytes) where the
    payload alone is 42,177,408; using the texture size avoids underestimating
    full-resident datasets near the fit threshold.
    """
    return _bc7_block_count(width) * _align_up(_bc7_block_count(height), _BC7_BLOCK_ROW_ALIGN) * 16


def rgba8_texture_bytes(width: int, height: int) -> int:
    return _align_up(max(int(width), 1) * 4, _TEXTURE_ROW_PITCH_ALIGN) * max(int(height), 1)


def frame_texture_bytes(width: int, height: int, compress_bc7: bool) -> int:
    """Resident texture bytes for one dataset frame."""
    return bc7_texture_bytes(width, height) if compress_bc7 else rgba8_texture_bytes(width, height)


def frame_staging_bytes(width: int, height: int, compress_bc7: bool) -> int:
    """Per-slot upload staging bytes (BC7 only; rgba8 uploads directly)."""
    return bc7_payload_bytes(width, height) if compress_bc7 else 0


@dataclass(frozen=True, slots=True)
class DatasetVramEstimate:
    full_bytes: int        # all frames resident (no staging)
    streaming_bytes: int   # pool_size slots sized to the largest frame (+ BC7 staging)
    pool_size: int


def estimate_dataset_vram(
    frame_sizes: list[tuple[int, int]],
    *,
    compress_bc7: bool,
    pool_size: int,
) -> DatasetVramEstimate:
    """Full-load vs streaming dataset VRAM.

    Full-load keeps one texture per frame (no staging buffers). Streaming keeps
    `pool_size` slots, each sized to the largest frame, plus an equal-sized BC7
    staging buffer per slot (see DatasetTexturePool).
    """
    if not frame_sizes:
        return DatasetVramEstimate(0, 0, max(int(pool_size), 0))
    full = sum(frame_texture_bytes(w, h, compress_bc7) for w, h in frame_sizes)
    max_w = max(w for w, _ in frame_sizes)
    max_h = max(h for _, h in frame_sizes)
    slot_bytes = frame_texture_bytes(max_w, max_h, compress_bc7) + frame_staging_bytes(max_w, max_h, compress_bc7)
    slots = max(min(int(pool_size), len(frame_sizes)), 1)
    return DatasetVramEstimate(full_bytes=full, streaming_bytes=slot_bytes * slots, pool_size=slots)


@dataclass(frozen=True, slots=True)
class TrainingVramEstimate:
    splat_bytes: int
    pixel_bytes: int
    ssim_bytes: int
    prepass_bytes: int
    overhead_bytes: int

    @property
    def total(self) -> int:
        return self.splat_bytes + self.pixel_bytes + self.ssim_bytes + self.prepass_bytes + self.overhead_bytes


def estimate_training_vram(
    splat_count: int,
    *,
    max_sh_band: int,
    train_width: int,
    train_height: int,
    ssim_width: int | None = None,
    ssim_height: int | None = None,
    list_entries_per_splat: int = _DEFAULT_LIST_ENTRIES_PER_SPLAT,
    refinement_growth_per_step: float = _REFINE_GROWTH,
) -> TrainingVramEstimate:
    """GPU memory for training a scene of `splat_count` splats (excludes dataset)."""
    splats = max(int(splat_count), 1)
    train_px = max(int(train_width), 1) * max(int(train_height), 1)
    ssim_px = max(int(ssim_width or train_width), 1) * max(int(ssim_height or train_height), 1)
    prepass_per_splat = max(int(list_entries_per_splat), 0) * _PREPASS_BYTES_PER_LIST_ENTRY + _PREPASS_SCANLINE_BYTES_PER_SPLAT
    return TrainingVramEstimate(
        splat_bytes=splats * per_splat_bytes(max_sh_band, refinement_growth_per_step),
        pixel_bytes=train_px * _PERPIXEL_TRAIN,
        ssim_bytes=ssim_px * _PERPIXEL_SSIM,
        prepass_bytes=splats * prepass_per_splat,
        overhead_bytes=_FIXED_OVERHEAD_BYTES,
    )


RESIDENCY_AUTO = "auto"
RESIDENCY_FULL = "full"
RESIDENCY_STREAM = "stream"


def representative_train_resolution(frame_sizes: list[tuple[int, int]], max_long_side: int = 1920) -> tuple[int, int]:
    """A training-render resolution proxy for the estimate's pixel/SSIM terms.

    Training renders downscaled frames, not full-resolution images. Clamp the
    largest frame's long side to `max_long_side` so the (secondary) pixel/SSIM
    terms stay realistic; the residency decision is dominated by dataset size.
    """
    if not frame_sizes:
        return (max_long_side, max_long_side)
    w = max(w for w, _ in frame_sizes)
    h = max(h for _, h in frame_sizes)
    long_side = max(w, h, 1)
    if long_side <= max_long_side:
        return (w, h)
    scale = max_long_side / long_side
    return (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))


@dataclass(frozen=True, slots=True)
class VramFitReport:
    training: TrainingVramEstimate
    dataset: DatasetVramEstimate
    capacity_bytes: int | None          # total GPU VRAM, or None if unknown
    recommend_streaming: bool
    full_total_bytes: int               # training + full dataset
    streaming_total_bytes: int          # training + streaming dataset
    dataset_fits_full: bool             # full dataset (with training) fits capacity
    config_fits_streaming: bool         # training + streaming dataset fits capacity

    @property
    def recommended_total_bytes(self) -> int:
        return self.streaming_total_bytes if self.recommend_streaming else self.full_total_bytes


def evaluate_fit(
    *,
    splat_count: int,
    max_sh_band: int,
    train_width: int,
    train_height: int,
    frame_sizes: list[tuple[int, int]],
    compress_bc7: bool,
    pool_size: int,
    capacity_bytes: int | None,
    ssim_width: int | None = None,
    ssim_height: int | None = None,
    safety_fraction: float = 0.85,
    refinement_growth_per_step: float = _REFINE_GROWTH,
) -> VramFitReport:
    """Estimate training+dataset VRAM and recommend full-load vs streaming.

    Auto recommends full-load only when the whole dataset plus training fits
    under `safety_fraction` of the GPU's capacity; otherwise it recommends
    streaming. `safety_fraction` leaves headroom for fragmentation/refinement
    growth. When capacity is unknown, defaults to streaming for big datasets.
    """
    training = estimate_training_vram(
        splat_count,
        max_sh_band=max_sh_band,
        train_width=train_width,
        train_height=train_height,
        ssim_width=ssim_width,
        ssim_height=ssim_height,
        refinement_growth_per_step=refinement_growth_per_step,
    )
    dataset = estimate_dataset_vram(frame_sizes, compress_bc7=compress_bc7, pool_size=pool_size)
    full_total = training.total + dataset.full_bytes
    streaming_total = training.total + dataset.streaming_bytes
    if capacity_bytes is None or int(capacity_bytes) <= 0:
        budget = None
        dataset_fits_full = dataset.full_bytes <= dataset.streaming_bytes  # tiny datasets only
        config_fits_streaming = True
    else:
        budget = int(capacity_bytes) * float(safety_fraction)
        dataset_fits_full = full_total <= budget
        config_fits_streaming = streaming_total <= budget
    recommend_streaming = not dataset_fits_full
    return VramFitReport(
        training=training,
        dataset=dataset,
        capacity_bytes=None if capacity_bytes is None else int(capacity_bytes),
        recommend_streaming=recommend_streaming,
        full_total_bytes=full_total,
        streaming_total_bytes=streaming_total,
        dataset_fits_full=dataset_fits_full,
        config_fits_streaming=config_fits_streaming,
    )


def resolve_import_residency(
    *,
    residency: str,
    frame_sizes: list[tuple[int, int]],
    splat_count: int,
    max_sh_band: int,
    compress_bc7: bool,
    requested_pool_size: int,
    capacity_bytes: int | None,
    train_width: int | None = None,
    train_height: int | None = None,
    refinement_growth_per_step: float = _REFINE_GROWTH,
) -> tuple[int, VramFitReport]:
    """Resolve the dataset pool size for an import given the residency mode.

    Returns ``(dataset_pool_size, report)`` where pool size 0 (or >= frame count)
    means full-load and a smaller positive value means streaming with that many
    slots. ``auto`` picks full-load only when the whole dataset + training fits
    the GPU (per evaluate_fit), else streams. ``full``/``stream`` force the mode.
    """
    total_frames = len(frame_sizes)
    requested = int(requested_pool_size)
    stream_pool = max(min(max(requested, 1), max(total_frames - 1, 1)), 1)
    if train_width is None or train_height is None:
        train_width, train_height = representative_train_resolution(frame_sizes)
    report = evaluate_fit(
        splat_count=splat_count,
        max_sh_band=max_sh_band,
        train_width=train_width,
        train_height=train_height,
        frame_sizes=frame_sizes,
        compress_bc7=compress_bc7,
        pool_size=stream_pool,
        capacity_bytes=capacity_bytes,
        refinement_growth_per_step=refinement_growth_per_step,
    )
    mode = str(residency or RESIDENCY_AUTO).lower()
    if mode == RESIDENCY_FULL:
        pool_size = 0  # 0 => all frames resident
    elif mode == RESIDENCY_STREAM:
        pool_size = stream_pool
    elif capacity_bytes is None or int(capacity_bytes) <= 0:
        # Auto with unknown capacity: keep the caller's requested pool (status quo).
        pool_size = requested
    else:  # auto with a known GPU budget
        pool_size = 0 if report.dataset_fits_full else stream_pool
    return pool_size, report

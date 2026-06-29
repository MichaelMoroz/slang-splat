from __future__ import annotations

from src.viewer import vram_estimate as ve


def test_sh_param_counts_match_renderer():
    # packed_trainable_param_count = 11 + 3*coeffs; coeffs = 1/4/9/16 for band 0/1/2/3
    assert [ve.packed_param_count_for_band(b) for b in range(4)] == [14, 23, 38, 59]


def test_per_splat_bytes_band3_includes_refinement_growth_and_overgrow():
    # param factor now accounts for dst_splat_params (splats+append) + append_params.
    expected = int(round((ve._persplat_param_factor() * 59 + ve._PERSPLAT_CONST) * ve._CAPACITY_OVERGROW))
    assert ve.per_splat_bytes(3) == expected
    assert ve.per_splat_bytes(3, refinement_growth_per_step=0.30) > ve.per_splat_bytes(3, refinement_growth_per_step=0.15)
    # param factor is above the bare 20P because refinement has dst and append params.
    assert ve._persplat_param_factor(0.15) > 20.0
    # Order of magnitude: ~1.7-1.9 KiB/splat at band 3 (captures showed ~1-1.6 KiB live).
    assert 1400 <= ve.per_splat_bytes(3) <= 2200


def test_bc7_payload_and_texture_bytes_match_capture():
    # Payload/staging is exactly the captured bc7_staging buffer size.
    assert ve.bc7_payload_bytes(7952, 5304) == 42_177_408
    # Resident texture includes block-row alignment -> exactly the captured texture.
    assert ve.bc7_texture_bytes(7952, 5304) == 44_785_664
    assert ve.bc7_texture_bytes(7952, 5304) > ve.bc7_payload_bytes(7952, 5304)


def test_rscan_dataset_full_vs_streaming():
    frames = [(7952, 5304)] * 339
    est = ve.estimate_dataset_vram(frames, compress_bc7=True, pool_size=16)
    # Full load uses resident texture bytes: 339 * 44.78 MiB ~= 14.1 GiB.
    assert est.full_bytes == 339 * ve.bc7_texture_bytes(7952, 5304)
    assert est.full_bytes > 14 * ve.GIB  # exceeds a 12-14 GiB GPU -> must stream
    # Streaming slot = resident texture + staging payload.
    slot = ve.bc7_texture_bytes(7952, 5304) + ve.bc7_payload_bytes(7952, 5304)
    assert est.streaming_bytes == 16 * slot
    assert est.streaming_bytes < 1.6 * ve.GIB


def test_rscan_auto_recommends_streaming_on_typical_gpu():
    frames = [(7952, 5304)] * 339
    report = ve.evaluate_fit(
        splat_count=1_500_000,
        max_sh_band=3,
        train_width=1740,
        train_height=1410,
        frame_sizes=frames,
        compress_bc7=True,
        pool_size=16,
        capacity_bytes=12 * ve.GIB,  # full dataset is 14.5 GiB -> cannot fit
    )
    assert report.recommend_streaming is True
    assert report.dataset_fits_full is False
    assert report.config_fits_streaming is True


def test_small_dataset_fits_full_on_big_gpu():
    # A small dataset that comfortably fits should recommend full-load.
    frames = [(1024, 768)] * 50
    report = ve.evaluate_fit(
        splat_count=500_000,
        max_sh_band=1,
        train_width=1024,
        train_height=768,
        frame_sizes=frames,
        compress_bc7=True,
        pool_size=16,
        capacity_bytes=24 * ve.GIB,
    )
    assert report.recommend_streaming is False
    assert report.dataset_fits_full is True


def test_total_estimate_sane_vs_capture_magnitude():
    # capture frame020984: ~1.02M splats, band 3, train ~1136x758, 16 BC7 slots.
    # Tracked total consumption was ~3.47 GiB; estimate should be same ballpark.
    report = ve.evaluate_fit(
        splat_count=1_024_000,
        max_sh_band=3,
        train_width=1136,
        train_height=758,
        frame_sizes=[(7952, 5304)] * 339,
        compress_bc7=True,
        pool_size=16,
        capacity_bytes=None,
    )
    gib = report.streaming_total_bytes / ve.GIB
    assert 2.5 <= gib <= 5.0, f"estimate {gib:.2f} GiB out of expected band"


def test_resolve_residency_auto_streams_rscan_on_12gib():
    frames = [(7952, 5304)] * 339
    pool, report = ve.resolve_import_residency(
        residency="auto", frame_sizes=frames, splat_count=1_500_000, max_sh_band=3,
        compress_bc7=True, requested_pool_size=16, capacity_bytes=12 * ve.GIB,
    )
    assert report.recommend_streaming is True
    assert pool == 16  # streaming with the requested slot count


def test_resolve_residency_auto_full_on_big_gpu():
    frames = [(1024, 768)] * 50
    pool, report = ve.resolve_import_residency(
        residency="auto", frame_sizes=frames, splat_count=500_000, max_sh_band=1,
        compress_bc7=True, requested_pool_size=16, capacity_bytes=24 * ve.GIB,
    )
    assert report.dataset_fits_full is True
    assert pool == 0  # 0 => full-load (all frames resident)


def test_resolve_residency_forced_modes_override_estimate():
    frames = [(1024, 768)] * 50
    # Force STREAM even though it would fit fully:
    pool_stream, _ = ve.resolve_import_residency(
        residency="stream", frame_sizes=frames, splat_count=500_000, max_sh_band=1,
        compress_bc7=True, requested_pool_size=16, capacity_bytes=24 * ve.GIB,
    )
    assert pool_stream == 16
    # Force FULL even on a tiny GPU where it won't fit:
    big = [(7952, 5304)] * 339
    pool_full, rep = ve.resolve_import_residency(
        residency="full", frame_sizes=big, splat_count=1_500_000, max_sh_band=3,
        compress_bc7=True, requested_pool_size=16, capacity_bytes=8 * ve.GIB,
    )
    assert pool_full == 0
    assert rep.dataset_fits_full is False  # readout can warn it won't fit


def test_resolve_residency_auto_unknown_capacity_keeps_requested_pool():
    # No GPU capacity (headless/stub): Auto must not change the requested pool.
    frames = [(7952, 5304)] * 339
    pool, _ = ve.resolve_import_residency(
        residency="auto", frame_sizes=frames, splat_count=1_000_000, max_sh_band=3,
        compress_bc7=True, requested_pool_size=16, capacity_bytes=None,
    )
    assert pool == 16
    pool_full, _ = ve.resolve_import_residency(
        residency="auto", frame_sizes=frames, splat_count=1_000_000, max_sh_band=3,
        compress_bc7=True, requested_pool_size=0, capacity_bytes=None,
    )
    assert pool_full == 0


def test_representative_train_resolution_clamps_long_side():
    assert ve.representative_train_resolution([(7952, 5304)], max_long_side=1920) == (1920, 1281)
    assert ve.representative_train_resolution([(1024, 768)], max_long_side=1920) == (1024, 768)


def test_capacity_unknown_defaults_to_streaming_for_big_dataset():
    frames = [(7952, 5304)] * 339
    report = ve.evaluate_fit(
        splat_count=1_000_000,
        max_sh_band=3,
        train_width=1740,
        train_height=1410,
        frame_sizes=frames,
        compress_bc7=True,
        pool_size=16,
        capacity_bytes=None,
    )
    assert report.recommend_streaming is True

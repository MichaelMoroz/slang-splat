from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
from PIL import Image

from src.renderer import GaussianRenderer
from src.scene import ColmapFrame, ColmapReconstruction, GaussianScene, build_training_frames, load_colmap_reconstruction
from src.scene._internal.colmap_ops import load_training_frame_rgba8
from src.scene._internal.colmap_types import ColmapImage, ColmapPoint3D
from src.training import GaussianTrainer, TrainingHyperParams
from src.training.photometric_compensation import (
    PPISP_PACKED_PARAM_COUNT,
    PackedPPISPTonemapProvider,
    PhotometricCompensationHyperParams,
    PhotometricCompensationTrainer,
    build_photometric_observation_pair_pool,
    identity_packed_ppisp_params,
    pack_ppisp_tonemap_params,
    unpack_ppisp_tonemap_params,
)
from src.training.ppisp import PPISPStaticTonemapProvider, PPISPTonemapParams
from src.utility import buffer_to_numpy

_TRAIN_SUBSAMPLE_HASH_STEP = 0x9E3779B9
_TRAIN_SUBSAMPLE_HASH_FRAME = 0x85EBCA6B
_TRAIN_SUBSAMPLE_HASH_X = 0xC2B2AE35
_TRAIN_SUBSAMPLE_HASH_Y = 0x27D4EB2F
_TRAIN_SUBSAMPLE_HASH_FACTOR = 0x165667B1
_RUN_SLOW_PHOTOMETRIC_REGRESSIONS = os.environ.get("RUN_SLOW_TRAINING_REGRESSIONS") == "1"


def _make_frame(image_id: int, width: int = 64, height: int = 48) -> ColmapFrame:
    return ColmapFrame(
        image_id=image_id,
        image_path=Path(f"frame_{image_id}.png"),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        fx=72.0,
        fy=72.0,
        cx=width * 0.5,
        cy=height * 0.5,
        width=width,
        height=height,
    )


def _make_reconstruction() -> tuple[ColmapReconstruction, list[ColmapFrame]]:
    frames = [_make_frame(0), _make_frame(1), _make_frame(2)]
    images = {
        0: ColmapImage(
            image_id=0,
            q_wxyz=frames[0].q_wxyz,
            t_xyz=frames[0].t_xyz,
            camera_id=0,
            name="frame_0.png",
            points2d_xy=np.array([[10.0, 20.0], [40.0, 18.0], [7.0, 6.0]], dtype=np.float32),
            points2d_point3d_ids=np.array([11, 12, -1], dtype=np.int64),
        ),
        1: ColmapImage(
            image_id=1,
            q_wxyz=frames[1].q_wxyz,
            t_xyz=frames[1].t_xyz,
            camera_id=0,
            name="frame_1.png",
            points2d_xy=np.array([[12.0, 22.0], [42.0, 17.0], [8.0, 5.0]], dtype=np.float32),
            points2d_point3d_ids=np.array([11, 12, 13], dtype=np.int64),
        ),
        2: ColmapImage(
            image_id=2,
            q_wxyz=frames[2].q_wxyz,
            t_xyz=frames[2].t_xyz,
            camera_id=0,
            name="frame_2.png",
            points2d_xy=np.array([[14.0, 24.0], [60.0, 30.0]], dtype=np.float32),
            points2d_point3d_ids=np.array([11, 13], dtype=np.int64),
        ),
    }
    points3d = {
        11: ColmapPoint3D(11, np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.1, track_length=3),
        12: ColmapPoint3D(12, np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.1, track_length=2),
        13: ColmapPoint3D(13, np.array([0.3, 0.2, 0.1], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.1, track_length=1),
    }
    recon = ColmapReconstruction(root=Path("."), sparse_dir=Path("sparse/0"), cameras={}, images=images, points3d=points3d)
    return recon, frames


def _make_scene(count: int = 4, seed: int = 7) -> GaussianScene:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32)
    positions[:, 2] += 2.0
    scales = np.log(np.full((count, 3), 0.03, dtype=np.float32)).astype(np.float32)
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.full((count,), 0.5, dtype=np.float32)
    colors = rng.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32)
    sh_coeffs = np.zeros((count, 1, 3), dtype=np.float32)
    return GaussianScene(positions=positions, scales=scales, rotations=rotations, opacities=opacities, colors=colors, sh_coeffs=sh_coeffs)


def _make_rgb_frame(tmp_path: Path, image: np.ndarray, *, image_name: str, image_id: int = 0) -> ColmapFrame:
    rgb = np.asarray(image, dtype=np.uint8)
    height, width = rgb.shape[:2]
    Image.fromarray(rgb, mode="RGB").save(tmp_path / image_name)
    return ColmapFrame(
        image_id=image_id,
        image_path=tmp_path / image_name,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        fx=72.0,
        fy=72.0,
        cx=width * 0.5,
        cy=height * 0.5,
        width=width,
        height=height,
    )


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    srgb = np.asarray(image, dtype=np.float32) / 255.0
    return np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4))


def _hash_u32(value: int) -> int:
    x = int(value) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0xFFFFFFFF


def _training_sample_native_pixel(low_res_x: int, low_res_y: int, *, factor: int, width: int, height: int, step_index: int, frame_index: int) -> tuple[int, int]:
    base_x = int(low_res_x) * int(factor)
    base_y = int(low_res_y) * int(factor)
    if factor <= 1:
        return base_x, base_y
    block_width = max(min(int(factor), int(width) - min(base_x, int(width) - 1)), 1)
    block_height = max(min(int(factor), int(height) - min(base_y, int(height) - 1)), 1)
    seed = _hash_u32(int(step_index) + _TRAIN_SUBSAMPLE_HASH_STEP)
    seed = _hash_u32(seed ^ (((int(frame_index) + 1) * _TRAIN_SUBSAMPLE_HASH_FRAME) & 0xFFFFFFFF))
    seed = _hash_u32(seed ^ (((int(low_res_x) + 1) * _TRAIN_SUBSAMPLE_HASH_X) & 0xFFFFFFFF))
    seed = _hash_u32(seed ^ (((int(low_res_y) + 1) * _TRAIN_SUBSAMPLE_HASH_Y) & 0xFFFFFFFF))
    seed = _hash_u32(seed ^ ((max(int(factor), 1) * _TRAIN_SUBSAMPLE_HASH_FACTOR) & 0xFFFFFFFF))
    offset_x = seed % block_width
    offset_y = _hash_u32(seed ^ _TRAIN_SUBSAMPLE_HASH_FACTOR) % block_height
    return min(base_x + offset_x, int(width) - 1), min(base_y + offset_y, int(height) - 1)


def _make_linear_rgba(height: int, width: int, rgb_scale: float = 1.0) -> np.ndarray:
    x = np.linspace(0.06, 0.28, num=width, dtype=np.float32)
    y = np.linspace(0.05, 0.22, num=height, dtype=np.float32)[:, None]
    red = np.clip((x[None, :] + y) * rgb_scale, 0.0, 1.0)
    green = np.clip((0.5 * x[None, :] + 0.75 * y + 0.03) * rgb_scale, 0.0, 1.0)
    blue = np.clip((0.2 * x[None, :] + 1.1 * y + 0.01) * rgb_scale, 0.0, 1.0)
    alpha = np.ones((height, width), dtype=np.float32)
    return np.stack((red, green, blue, alpha), axis=2).astype(np.float32)


def _load_frame_linear_rgba(frame: ColmapFrame) -> np.ndarray:
    rgba8 = np.asarray(load_training_frame_rgba8(frame), dtype=np.uint8)
    rgb = _srgb_to_linear(rgba8[:, :, :3])
    alpha = rgba8[:, :, 3:4].astype(np.float32) / 255.0
    return np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2), dtype=np.float32)


def _reference_pair_dataset(
    frames: list[ColmapFrame],
    pair_pool,
    frame_rgba_linear: list[np.ndarray],
    neighborhood_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_count = max(int(neighborhood_size), 1) * max(int(neighborhood_size), 1)
    pair_count = len(pair_pool)
    total_samples = pair_count * sample_count
    samples_a = np.zeros((total_samples, 4), dtype=np.float32)
    samples_b = np.zeros((total_samples, 4), dtype=np.float32)
    sensor_coords_a = np.zeros((total_samples, 2), dtype=np.float32)
    sensor_coords_b = np.zeros((total_samples, 2), dtype=np.float32)
    radius = max(int(neighborhood_size), 1) // 2

    for pair_index in range(pair_count):
        frame_index_a = int(pair_pool.frame_indices_a[pair_index])
        frame_index_b = int(pair_pool.frame_indices_b[pair_index])
        frame_a = frames[frame_index_a]
        frame_b = frames[frame_index_b]
        image_a = np.asarray(frame_rgba_linear[frame_index_a], dtype=np.float32)
        image_b = np.asarray(frame_rgba_linear[frame_index_b], dtype=np.float32)
        center_a = np.floor(np.asarray(pair_pool.xy_a[pair_index], dtype=np.float32)).astype(np.int32, copy=False)
        center_b = np.floor(np.asarray(pair_pool.xy_b[pair_index], dtype=np.float32)).astype(np.int32, copy=False)
        width_a = max(int(frame_a.width), 1)
        height_a = max(int(frame_a.height), 1)
        width_b = max(int(frame_b.width), 1)
        height_b = max(int(frame_b.height), 1)
        for sample_y in range(max(int(neighborhood_size), 1)):
            for sample_x in range(max(int(neighborhood_size), 1)):
                sample_index = sample_y * max(int(neighborhood_size), 1) + sample_x
                write_index = pair_index * sample_count + sample_index
                dx = sample_x - radius
                dy = sample_y - radius
                pixel_x_a = int(np.clip(center_a[0] + dx, 0, width_a - 1))
                pixel_y_a = int(np.clip(center_a[1] + dy, 0, height_a - 1))
                pixel_x_b = int(np.clip(center_b[0] + dx, 0, width_b - 1))
                pixel_y_b = int(np.clip(center_b[1] + dy, 0, height_b - 1))
                samples_a[write_index] = image_a[pixel_y_a, pixel_x_a]
                samples_b[write_index] = image_b[pixel_y_b, pixel_x_b]
                sensor_coords_a[write_index] = ((pixel_x_a + 0.5) / width_a, (pixel_y_a + 0.5) / height_a)
                sensor_coords_b[write_index] = ((pixel_x_b + 0.5) / width_b, (pixel_y_b + 0.5) / height_b)
    return samples_a, samples_b, sensor_coords_a, sensor_coords_b


def test_packed_ppisp_round_trip_and_provider_versioning() -> None:
    params = PPISPTonemapParams(
        exposureEv=0.125,
        vignetteCenterX=(0.51, 0.49, 0.5),
        vignetteCenterY=(0.48, 0.52, 0.5),
        vignetteCoeffR2=(-0.02, -0.01, -0.015),
        vignetteCoeffR4=(0.001, 0.002, 0.003),
        vignetteCoeffR6=(0.0, 0.0, 0.0),
        chromaOffsetR=(0.01, -0.02),
        chromaOffsetG=(0.02, -0.01),
        chromaOffsetB=(-0.03, 0.01),
        chromaOffsetW=(0.0, 0.01),
        crfTau=(1.1, 1.0, 0.95),
        crfEta=(0.9, 1.05, 1.0),
        crfXi=(0.48, 0.5, 0.52),
        crfGamma=(2.15, 2.2, 2.25),
    )

    packed = pack_ppisp_tonemap_params(params)
    round_trip = unpack_ppisp_tonemap_params(packed)
    np.testing.assert_allclose(pack_ppisp_tonemap_params(round_trip), packed, rtol=0.0, atol=1e-6)
    assert packed.size == PPISP_PACKED_PARAM_COUNT

    provider = PackedPPISPTonemapProvider(2)
    assert provider.version == 0
    replaced = np.repeat(packed[:, None], 2, axis=1)
    provider.replace_packed_params(replaced)
    assert provider.version == 1
    np.testing.assert_allclose(pack_ppisp_tonemap_params(provider.params_for_frame(1)), packed, rtol=0.0, atol=1e-6)


def test_build_photometric_observation_pair_pool_is_deterministic() -> None:
    recon, frames = _make_reconstruction()
    pool = build_photometric_observation_pair_pool(recon, frames, min_track_length=2)

    assert len(pool) == 4
    np.testing.assert_array_equal(pool.point_ids, np.array([11, 11, 11, 12], dtype=np.int64))
    np.testing.assert_array_equal(pool.track_lengths, np.array([3, 3, 3, 2], dtype=np.int32))
    np.testing.assert_array_equal(pool.frame_indices_a, np.array([0, 0, 1, 0], dtype=np.int32))
    np.testing.assert_array_equal(pool.frame_indices_b, np.array([1, 2, 2, 1], dtype=np.int32))
    np.testing.assert_allclose(pool.xy_a, np.array([[10.0, 20.0], [10.0, 20.0], [12.0, 22.0], [40.0, 18.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(pool.xy_b, np.array([[12.0, 22.0], [14.0, 24.0], [14.0, 24.0], [42.0, 17.0]], dtype=np.float32), rtol=0.0, atol=1e-6)

    batch = pool.sample(np.random.default_rng(9), 3)
    assert batch.pair_count == 3
    assert np.all(batch.frame_indices_a <= batch.frame_indices_b)


def test_photometric_packed_adam_converges_per_frame_params(device) -> None:
    grad_shader_path = Path(__file__).with_name("optimizer_test_stage.slang")
    grad_kernel = device.create_compute_kernel(device.load_program(str(grad_shader_path), ["csComputeQuadraticGrad"]))
    recon, frames = _make_reconstruction()
    hparams = PhotometricCompensationHyperParams(
        learning_rate=0.1,
        exposure_regularize_weight=0.0,
        vignette_regularize_weight=0.0,
        chroma_regularize_weight=0.0,
        crf_regularize_weight=0.0,
        vignette_l1_weight=0.0,
        chroma_l1_weight=0.0,
        crf_l1_weight=0.0,
    )
    trainer = PhotometricCompensationTrainer(device, recon, frames, hparams=hparams, seed=7)
    rng = np.random.default_rng(17)
    identity = identity_packed_ppisp_params(len(frames))
    targets = identity + rng.normal(0.0, 0.04, size=(PPISP_PACKED_PARAM_COUNT, len(frames))).astype(np.float32)
    targets[:, 0] = identity[:, 0]
    params_init = targets + rng.normal(0.0, 0.15, size=targets.shape).astype(np.float32)
    params_init[:, 0] = identity[:, 0]
    trainer.replace_packed_params(params_init)

    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    target_buffer = device.create_buffer(size=trainer.packed_param_count * 4, usage=usage)
    target_buffer.copy_from_numpy(np.ascontiguousarray(targets.reshape(-1), dtype=np.float32))

    start_error = float(np.mean(np.abs(params_init - targets), dtype=np.float64))
    for step in range(1, 201):
        encoder = device.create_command_encoder()
        grad_kernel.dispatch(
            thread_count=spy.uint3(trainer.packed_param_count, 1, 1),
            vars={
                "g_ParamCount": int(trainer.packed_param_count),
                "g_Targets": target_buffer,
                "g_Params": trainer.buffers["params"],
                "g_Grads": trainer.buffers["grads"],
                "g_Stability": {
                    "gradComponentClip": float(hparams.grad_component_clip),
                    "gradNormClip": float(hparams.grad_norm_clip),
                    "maxUpdate": float(hparams.max_update),
                    "hugeValue": float(hparams.huge_value),
                },
            },
            command_encoder=encoder,
        )
        trainer.dispatch_optimizer_step(encoder, step)
        device.submit_command_buffer(encoder.finish())
    device.wait()

    final_params = trainer.read_packed_params()
    final_error = float(np.mean(np.abs(final_params - targets), dtype=np.float64))

    assert np.all(np.isfinite(final_params))
    np.testing.assert_allclose(final_params[:, 0], identity[:, 0], rtol=0.0, atol=1e-6)
    assert final_error < start_error * 0.02


def test_photometric_param_settings_regularize_toward_identity(device) -> None:
    recon, frames = _make_reconstruction()
    hparams = PhotometricCompensationHyperParams(
        learning_rate=0.2,
        exposure_regularize_weight=0.5,
        vignette_regularize_weight=0.5,
        chroma_regularize_weight=0.5,
        crf_regularize_weight=0.5,
        vignette_l1_weight=0.05,
        chroma_l1_weight=0.05,
        crf_l1_weight=0.05,
    )
    trainer = PhotometricCompensationTrainer(device, recon, frames, hparams=hparams, seed=3)
    identity = identity_packed_ppisp_params(len(frames))
    params = identity.copy()
    params[0, :] += 0.75
    params[6:10, :] += 0.1
    params[27:36, :] += 0.2
    trainer.replace_packed_params(params)

    start_distance = float(np.mean(np.abs(params - identity), dtype=np.float64))
    for step in range(1, 33):
        trainer.zero_grads()
        trainer.step_optimizer(step)

    final_params = trainer.read_packed_params()
    final_distance = float(np.mean(np.abs(final_params - identity), dtype=np.float64))

    assert final_distance < start_distance * 0.5


def test_photometric_precomputed_pair_dataset_avoids_full_frame_upload(device) -> None:
    dataset_root = Path("dataset/garden")
    if not dataset_root.exists():
        pytest.skip("dataset/garden is unavailable.")
    recon = load_colmap_reconstruction(dataset_root)
    frames = build_training_frames(recon, images_subdir="images_4")
    if len(frames) < 16:
        pytest.skip("dataset/garden/images_4 does not provide enough frames for the photometric dataset regression.")

    trainer = PhotometricCompensationTrainer(
        device,
        recon,
        frames,
        hparams=PhotometricCompensationHyperParams(
            batch_pair_count=512,
        ),
        seed=29,
    )

    trainer.prepare_pair_dataset()

    total_frame_pixels = sum(int(frame.width) * int(frame.height) for frame in frames)
    total_dataset_samples = len(trainer.pair_pool) * trainer.hparams.neighborhood_size * trainer.hparams.neighborhood_size

    assert trainer._pair_dataset_uploaded is True
    assert "pair_samples_a" in trainer.buffers
    assert "pair_samples_b" in trainer.buffers
    assert "pair_sensor_coords_a" in trainer.buffers
    assert "pair_sensor_coords_b" in trainer.buffers
    assert "frame_pixels" not in trainer.buffers
    assert total_dataset_samples < total_frame_pixels
    assert trainer.buffers["pair_samples_a"].size >= total_dataset_samples * 16


def test_photometric_gpu_pair_dataset_matches_reference(device) -> None:
    recon, frames = _make_reconstruction()
    frame_rgba_linear = [
        _make_linear_rgba(frame.height, frame.width, rgb_scale=1.0 + 0.1 * frame_index)
        for frame_index, frame in enumerate(frames)
    ]
    trainer = PhotometricCompensationTrainer(
        device,
        recon,
        frames,
        hparams=PhotometricCompensationHyperParams(batch_pair_count=4, neighborhood_size=3),
        seed=17,
        frame_rgba_linear=frame_rgba_linear,
    )

    trainer.prepare_pair_dataset()

    total_dataset_samples = len(trainer.pair_pool) * trainer.hparams.neighborhood_size * trainer.hparams.neighborhood_size
    expected_samples_a, expected_samples_b, expected_sensor_coords_a, expected_sensor_coords_b = _reference_pair_dataset(
        frames,
        trainer.pair_pool,
        frame_rgba_linear,
        trainer.hparams.neighborhood_size,
    )
    pair_samples_a = buffer_to_numpy(trainer.buffers["pair_samples_a"], np.float32)[: total_dataset_samples * 4].reshape(total_dataset_samples, 4)
    pair_samples_b = buffer_to_numpy(trainer.buffers["pair_samples_b"], np.float32)[: total_dataset_samples * 4].reshape(total_dataset_samples, 4)
    pair_sensor_coords_a = buffer_to_numpy(trainer.buffers["pair_sensor_coords_a"], np.float32)[: total_dataset_samples * 2].reshape(total_dataset_samples, 2)
    pair_sensor_coords_b = buffer_to_numpy(trainer.buffers["pair_sensor_coords_b"], np.float32)[: total_dataset_samples * 2].reshape(total_dataset_samples, 2)

    np.testing.assert_allclose(pair_samples_a, expected_samples_a, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(pair_samples_b, expected_samples_b, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(pair_sensor_coords_a, expected_sensor_coords_a, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(pair_sensor_coords_b, expected_sensor_coords_b, rtol=0.0, atol=1e-6)


def test_photometric_reference_frame_stays_identity(device) -> None:
    recon, frames = _make_reconstruction()
    frame_rgba_linear = [
        _make_linear_rgba(frame.height, frame.width, rgb_scale=1.0 + 0.05 * frame_index)
        for frame_index, frame in enumerate(frames)
    ]
    trainer = PhotometricCompensationTrainer(
        device,
        recon,
        frames,
        hparams=PhotometricCompensationHyperParams(batch_pair_count=4, neighborhood_size=3),
        seed=23,
        frame_rgba_linear=frame_rgba_linear,
    )

    identity = identity_packed_ppisp_params(len(frames))
    params = identity.copy()
    params[:, 0] += np.float32(0.25)
    params[:, 1] += np.float32(0.10)
    trainer.replace_packed_params(params)
    np.testing.assert_allclose(trainer.read_packed_params()[:, 0], identity[:, 0], rtol=0.0, atol=1e-6)

    trainer.train_step(step_index=1)

    np.testing.assert_allclose(trainer.read_packed_params()[:, 0], identity[:, 0], rtol=0.0, atol=1e-6)


def test_gaussian_trainer_applies_target_tonemap_to_downscaled_targets(device, tmp_path: Path) -> None:
    image = np.array(
        [
            [[16, 32, 48], [32, 48, 64]],
            [[48, 64, 80], [64, 80, 96]],
        ],
        dtype=np.uint8,
    )
    frame = _make_rgb_frame(tmp_path, image, image_name="photometric_downscaled_target.png")
    scene = _make_scene(count=2, seed=13)
    renderer = GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16)
    provider = PPISPStaticTonemapProvider(PPISPTonemapParams(exposureEv=1.0))
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=5, target_tonemap_provider=provider)

    target = trainer.get_frame_target_texture(0, native_resolution=False)
    target_np = np.asarray(target.to_numpy(), dtype=np.float32)
    expected = np.minimum(_srgb_to_linear(image) * 2.0, 1.0)

    np.testing.assert_allclose(target_np[:, :, :3], expected, rtol=0.0, atol=5e-4)
    np.testing.assert_allclose(target_np[:, :, 3], np.ones((2, 2), dtype=np.float32), rtol=0.0, atol=1e-6)


def test_gaussian_trainer_applies_target_tonemap_to_native_subsample_targets(device, tmp_path: Path) -> None:
    image = np.array(
        [
            [[16, 24, 32], [24, 32, 40], [32, 40, 48], [40, 48, 56]],
            [[24, 32, 40], [32, 40, 48], [40, 48, 56], [48, 56, 64]],
            [[32, 40, 48], [40, 48, 56], [48, 56, 64], [56, 64, 72]],
            [[40, 48, 56], [48, 56, 64], [56, 64, 72], [64, 72, 80]],
        ],
        dtype=np.uint8,
    )
    frame = _make_rgb_frame(tmp_path, image, image_name="photometric_native_target.png")
    scene = _make_scene(count=1, seed=19)
    renderer = GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16)
    provider = PPISPStaticTonemapProvider(PPISPTonemapParams(exposureEv=1.0))
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=11,
        target_tonemap_provider=provider,
    )

    renderer.output_texture.copy_from_numpy(np.zeros((renderer.height, renderer.width, 4), dtype=np.float32))
    target_texture = trainer.get_frame_target_texture(0, native_resolution=True)
    encoder = device.create_command_encoder()
    trainer._dispatch_ssim_feature_extraction(encoder, target_texture, step=0, frame_index=0)
    device.submit_command_buffer(encoder.finish())
    device.wait()

    ssim = np.frombuffer(trainer._buffers["ssim_moments"].to_numpy().tobytes(), dtype=np.float32).reshape(renderer.height, renderer.width, 15)
    linear = np.minimum(_srgb_to_linear(image) * 2.0, 1.0)
    expected = np.zeros((renderer.height, renderer.width, 3), dtype=np.float32)
    for low_res_y in range(renderer.height):
        for low_res_x in range(renderer.width):
            src_x, src_y = _training_sample_native_pixel(low_res_x, low_res_y, factor=2, width=frame.width, height=frame.height, step_index=0, frame_index=0)
            expected[low_res_y, low_res_x] = linear[src_y, src_x]

    np.testing.assert_allclose(ssim[:, :, [1, 6, 11]], expected, rtol=0.0, atol=5e-4)


def test_photometric_trainer_pair_loss_step_reduces_synthetic_exposure_error(device) -> None:
    recon, frames = _make_reconstruction()
    base = _make_linear_rgba(frames[0].height, frames[0].width)
    frame_rgba_linear = [base.copy(), np.ascontiguousarray(base.copy(), dtype=np.float32), base.copy()]
    frame_rgba_linear[1][:, :, :3] *= 0.5
    hparams = PhotometricCompensationHyperParams(
        batch_pair_count=512,
        neighborhood_size=3,
        learning_rate=0.2,
        exposure_lr_mul=1.0,
        vignette_lr_mul=0.0,
        chroma_lr_mul=0.0,
        crf_lr_mul=0.0,
        exposure_regularize_weight=0.01,
        vignette_regularize_weight=0.5,
        chroma_regularize_weight=0.5,
        crf_regularize_weight=0.5,
    )
    trainer = PhotometricCompensationTrainer(device, recon, frames, hparams=hparams, seed=13, frame_rgba_linear=frame_rgba_linear)

    first_loss = float("nan")
    for step in range(1, 161):
        loss = trainer.train_step(step_index=step)
        if step == 1:
            first_loss = float(loss)

    exposure_values = [trainer.provider.params_for_frame(index).exposureEv for index in range(len(frames))]

    assert np.isfinite(first_loss)
    assert np.isfinite(trainer.state.ema_loss)
    assert trainer.state.ema_loss < first_loss * 0.3
    assert exposure_values[0] == pytest.approx(0.0, abs=1e-6)
    assert exposure_values[1] > exposure_values[0] + 0.25
    assert exposure_values[1] > exposure_values[2] + 0.25


@pytest.mark.skipif(not _RUN_SLOW_PHOTOMETRIC_REGRESSIONS, reason="set RUN_SLOW_TRAINING_REGRESSIONS=1 to run the garden photometric regression")
def test_photometric_trainer_garden_subset_loss_converges(device) -> None:
    dataset_root = Path("dataset/garden")
    if not dataset_root.exists():
        pytest.skip("dataset/garden is unavailable.")
    recon = load_colmap_reconstruction(dataset_root)
    frames = build_training_frames(recon, images_subdir="images_4")[:4]
    if len(frames) < 4:
        pytest.skip("dataset/garden/images_4 does not provide enough frames for the photometric regression.")

    exposure_scales = (1.0, 0.62, 0.78, 0.55)
    frame_rgba_linear = []
    for frame, scale in zip(frames, exposure_scales, strict=True):
        image = _load_frame_linear_rgba(frame)
        image[:, :, :3] = np.clip(image[:, :, :3] * float(scale), 0.0, 1.0)
        frame_rgba_linear.append(np.ascontiguousarray(image, dtype=np.float32))

    trainer = PhotometricCompensationTrainer(
        device,
        recon,
        frames,
        hparams=PhotometricCompensationHyperParams(
            batch_pair_count=2048,
            neighborhood_size=3,
            learning_rate=0.15,
            exposure_lr_mul=1.0,
            vignette_lr_mul=0.0,
            chroma_lr_mul=0.0,
            crf_lr_mul=0.0,
            exposure_regularize_weight=0.01,
            vignette_regularize_weight=0.5,
            chroma_regularize_weight=0.5,
            crf_regularize_weight=0.5,
        ),
        seed=29,
        frame_rgba_linear=frame_rgba_linear,
    )

    first_loss = float("nan")
    for step in range(1, 501):
        loss = trainer.train_step(step_index=step)
        if step == 1:
            first_loss = float(loss)

    assert np.isfinite(first_loss)
    assert np.isfinite(trainer.state.ema_loss)
    assert trainer.state.ema_loss < first_loss * 0.4
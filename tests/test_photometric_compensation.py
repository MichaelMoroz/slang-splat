from __future__ import annotations

import os
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import slangpy as spy
from PIL import Image
import src.training.photometric_compensation as photometric_compensation_module

from src.renderer import GaussianRenderer
from src.scene import ColmapFrame, ColmapReconstruction, GaussianScene, build_training_frames, load_colmap_reconstruction
from src.scene._internal.colmap_ops import load_training_frame_rgba8
from src.scene._internal.colmap_types import ColmapCamera, ColmapImage, ColmapPoint3D
from src.training import GaussianTrainer, TrainingHyperParams
from src.training.photometric_compensation import (
    PPISP_PACKED_PARAM_COUNT,
    PackedPPISPTonemapProvider,
    PhotometricObservationPairPool,
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


def _colmap_pair_id(image_id_a: int, image_id_b: int) -> int:
    first, second = sorted((int(image_id_a), int(image_id_b)))
    return int(first * photometric_compensation_module._COLMAP_PAIR_ID_PRIME + second)


def _write_match_database(
    path: Path,
    *,
    image_names: dict[int, str] | None = None,
    keypoints: dict[int, np.ndarray] | None = None,
    matches: dict[tuple[int, int], np.ndarray] | None = None,
) -> Path:
    resolved_keypoints = {
        1: np.array([[10.0, 20.0], [40.0, 18.0]], dtype=np.float32),
        2: np.array([[12.0, 22.0], [42.0, 17.0], [8.0, 5.0]], dtype=np.float32),
        3: np.array([[14.0, 24.0], [60.0, 30.0]], dtype=np.float32),
    } if keypoints is None else {int(image_id): np.asarray(xy, dtype=np.float32).reshape(-1, 2) for image_id, xy in keypoints.items()}
    resolved_matches = {
        (1, 2): np.array([[0, 0], [1, 1]], dtype=np.uint32),
        (1, 3): np.array([[0, 0]], dtype=np.uint32),
    } if matches is None else {tuple(int(v) for v in pair): np.asarray(pair_matches, dtype=np.uint32).reshape(-1, 2) for pair, pair_matches in matches.items()}
    resolved_image_names = {int(image_id): f"frame_{int(image_id)}.png" for image_id in resolved_keypoints} if image_names is None else {int(image_id): str(name) for image_id, name in image_names.items()}
    with sqlite3.connect(str(path)) as conn:
        conn.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB)")
        conn.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB)")
        conn.executemany(
            "INSERT INTO images(image_id, name) VALUES (?, ?)",
            [(image_id, resolved_image_names[image_id]) for image_id in sorted(resolved_keypoints)],
        )
        conn.executemany(
            "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            [
                (int(image_id), int(xy.shape[0]), int(xy.shape[1]), np.ascontiguousarray(xy, dtype=np.float32).tobytes())
                for image_id, xy in resolved_keypoints.items()
            ],
        )
        conn.executemany(
            "INSERT INTO two_view_geometries(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            [
                (_colmap_pair_id(image_id_a, image_id_b), int(pair_matches.shape[0]), int(pair_matches.shape[1]), np.ascontiguousarray(pair_matches, dtype=np.uint32).tobytes())
                for (image_id_a, image_id_b), pair_matches in resolved_matches.items()
            ],
        )
        conn.commit()
    return path


def _make_scaled_reconstruction() -> tuple[ColmapReconstruction, list[ColmapFrame]]:
    frames = [_make_frame(1), _make_frame(2), _make_frame(3)]
    cameras = {
        0: ColmapCamera(camera_id=0, model_id=1, width=128, height=96, fx=144.0, fy=144.0, cx=64.0, cy=48.0),
    }
    images = {
        1: ColmapImage(
            image_id=1,
            q_wxyz=frames[0].q_wxyz,
            t_xyz=frames[0].t_xyz,
            camera_id=0,
            name="frame_1.png",
            points2d_xy=np.array([[20.0, 40.0], [80.0, 36.0], [14.0, 12.0]], dtype=np.float32),
            points2d_point3d_ids=np.array([11, 12, -1], dtype=np.int64),
        ),
        2: ColmapImage(
            image_id=2,
            q_wxyz=frames[1].q_wxyz,
            t_xyz=frames[1].t_xyz,
            camera_id=0,
            name="frame_2.png",
            points2d_xy=np.array([[24.0, 44.0], [84.0, 34.0], [16.0, 10.0]], dtype=np.float32),
            points2d_point3d_ids=np.array([11, 12, 13], dtype=np.int64),
        ),
        3: ColmapImage(
            image_id=3,
            q_wxyz=frames[2].q_wxyz,
            t_xyz=frames[2].t_xyz,
            camera_id=0,
            name="frame_3.png",
            points2d_xy=np.array([[28.0, 48.0], [120.0, 60.0]], dtype=np.float32),
            points2d_point3d_ids=np.array([11, 13], dtype=np.int64),
        ),
    }
    points3d = {
        11: ColmapPoint3D(11, np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.1, track_length=3),
        12: ColmapPoint3D(12, np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.1, track_length=2),
        13: ColmapPoint3D(13, np.array([0.3, 0.2, 0.1], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.1, track_length=1),
    }
    recon = ColmapReconstruction(root=Path("."), sparse_dir=Path("sparse/0"), cameras=cameras, images=images, points3d=points3d)
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


def _native_subsample_target_ssim_rgb(trainer: GaussianTrainer, device: spy.Device, *, frame_index: int = 0, step: int = 0) -> tuple[spy.Texture, np.ndarray]:
    trainer.renderer.output_texture.copy_from_numpy(np.zeros((trainer.renderer.height, trainer.renderer.width, 4), dtype=np.float32))
    target_texture = trainer.get_frame_target_texture(frame_index, native_resolution=True)
    encoder = device.create_command_encoder()
    trainer._dispatch_ssim_feature_extraction(encoder, target_texture, step=step, frame_index=frame_index)
    device.submit_command_buffer(encoder.finish())
    device.wait()
    ssim = np.frombuffer(trainer._buffers["ssim_moments"].to_numpy().tobytes(), dtype=np.float32).reshape(trainer.renderer.height, trainer.renderer.width, 15)
    return target_texture, ssim[:, :, [1, 6, 11]]


def _reference_observation_dataset(
    frames: list[ColmapFrame],
    track_pool,
    frame_rgba_linear: list[np.ndarray],
    neighborhood_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_neighborhood = max(int(neighborhood_size), 1)
    observation_count = int(track_pool.observation_frame_indices.size)
    mean_samples = np.zeros((observation_count, 4), dtype=np.float32)
    mean_sensor_coords = np.zeros((observation_count, 2), dtype=np.float32)
    radius = resolved_neighborhood // 2
    inv_sample_count = 1.0 / float(resolved_neighborhood * resolved_neighborhood)

    for observation_index in range(observation_count):
        frame_index = int(track_pool.observation_frame_indices[observation_index])
        frame = frames[frame_index]
        image = np.asarray(frame_rgba_linear[frame_index], dtype=np.float32)
        center = np.floor(np.asarray(track_pool.observation_xy[observation_index], dtype=np.float32)).astype(np.int32, copy=False)
        width = max(int(frame.width), 1)
        height = max(int(frame.height), 1)
        accum_sample = np.zeros((4,), dtype=np.float64)
        accum_sensor_coord = np.zeros((2,), dtype=np.float64)
        for sample_y in range(resolved_neighborhood):
            for sample_x in range(resolved_neighborhood):
                dx = sample_x - radius
                dy = sample_y - radius
                pixel_x = int(np.clip(center[0] + dx, 0, width - 1))
                pixel_y = int(np.clip(center[1] + dy, 0, height - 1))
                accum_sample += np.asarray(image[pixel_y, pixel_x], dtype=np.float64)
                accum_sensor_coord += np.asarray(((pixel_x + 0.5) / width, (pixel_y + 0.5) / height), dtype=np.float64)
        mean_samples[observation_index] = np.asarray(accum_sample * inv_sample_count, dtype=np.float32)
        mean_sensor_coords[observation_index] = np.asarray(accum_sensor_coord * inv_sample_count, dtype=np.float32)
    return mean_samples, mean_sensor_coords


def _dispatch_ppisp_round_trip(
    device: spy.Device,
    params: PPISPTonemapParams,
    radiance: np.ndarray,
    sensor_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shader_path = Path(__file__).with_name("ppisp_inverse_test.slang")
    kernel = device.create_compute_kernel(device.load_program(str(shader_path), ["csRoundTripPPISP"]))
    colors = np.asarray(radiance, dtype=np.float32).reshape(-1, 3)
    coords = np.asarray(sensor_coords, dtype=np.float32).reshape(-1, 2)
    count = max(int(colors.shape[0]), 0)
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    input_buffer = device.create_buffer(size=max(count, 1) * 16, usage=usage)
    sensor_buffer = device.create_buffer(size=max(count, 1) * 8, usage=usage)
    forward_buffer = device.create_buffer(size=max(count, 1) * 16, usage=usage)
    recovered_buffer = device.create_buffer(size=max(count, 1) * 16, usage=usage)
    input_colors = np.ones((max(count, 1), 4), dtype=np.float32)
    input_colors[:count, :3] = colors
    input_buffer.copy_from_numpy(np.ascontiguousarray(input_colors.reshape(-1), dtype=np.float32))
    input_coords = np.zeros((max(count, 1), 2), dtype=np.float32)
    input_coords[:count] = coords
    sensor_buffer.copy_from_numpy(np.ascontiguousarray(input_coords.reshape(-1), dtype=np.float32))

    encoder = device.create_command_encoder()
    kernel.dispatch(
        thread_count=spy.uint3(max(count, 1), 1, 1),
        vars={
            "g_InputColors": input_buffer,
            "g_SensorCoords": sensor_buffer,
            "g_OutputForward": forward_buffer,
            "g_OutputRecovered": recovered_buffer,
            "g_Count": int(count),
            "g_PPISP": params.to_shader_dict(),
        },
        command_encoder=encoder,
    )
    device.submit_command_buffer(encoder.finish())
    device.wait()

    forward = buffer_to_numpy(forward_buffer, np.float32)[: max(count, 1) * 4].reshape(max(count, 1), 4)[:count, :3]
    recovered = buffer_to_numpy(recovered_buffer, np.float32)[: max(count, 1) * 4].reshape(max(count, 1), 4)[:count, :3]
    return np.ascontiguousarray(forward, dtype=np.float32), np.ascontiguousarray(recovered, dtype=np.float32)


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


def test_photometric_param_settings_disable_selected_aspects() -> None:
    hparams = PhotometricCompensationHyperParams(
        learning_rate=0.2,
        enable_exposure=False,
        enable_color=False,
        enable_vignette=True,
        enable_gamma=False,
        exposure_lr_mul=1.0,
        vignette_lr_mul=1.0,
        chroma_lr_mul=1.0,
        crf_lr_mul=1.0,
        exposure_regularize_weight=0.5,
        vignette_regularize_weight=0.5,
        chroma_regularize_weight=0.5,
        crf_regularize_weight=0.5,
        gamma_regularize_weight=0.5,
        exposure_l1_weight=0.05,
        vignette_l1_weight=0.05,
        chroma_l1_weight=0.05,
        crf_l1_weight=0.05,
        gamma_l1_weight=0.05,
    )
    settings = photometric_compensation_module.build_ppisp_param_settings(hparams).view(np.float32)
    lrs = np.asarray(settings[:, 0], dtype=np.float32)
    regularize = np.asarray(settings[:, 5], dtype=np.float32)
    regularize_l1 = np.asarray(settings[:, 6], dtype=np.float32)

    offset = 0
    for spec in photometric_compensation_module.PPISP_FIELD_SPECS:
        field_slice = slice(offset, offset + spec.size)
        group = photometric_compensation_module._field_group_name(spec.attr)
        if group == "vignette":
            assert np.all(lrs[field_slice] > 0.0)
            assert np.all(regularize[field_slice] > 0.0)
            assert np.all(regularize_l1[field_slice] > 0.0)
        else:
            assert np.all(lrs[field_slice] == 0.0)
            assert np.all(regularize[field_slice] == 0.0)
            assert np.all(regularize_l1[field_slice] == 0.0)
        offset += int(spec.size)


def test_ppisp_shader_inverse_round_trips_moderate_colors(device) -> None:
    params = PPISPTonemapParams(
        exposureEv=0.35,
        vignetteCenterX=(0.48, 0.52, 0.50),
        vignetteCenterY=(0.51, 0.49, 0.50),
        vignetteCoeffR2=(-0.15, -0.10, -0.12),
        vignetteCoeffR4=(0.02, 0.015, 0.01),
        vignetteCoeffR6=(0.0, 0.0, 0.0),
        chromaOffsetR=(0.015, -0.010),
        chromaOffsetG=(-0.012, 0.008),
        chromaOffsetB=(0.006, -0.005),
        chromaOffsetW=(0.004, 0.003),
        crfTau=(1.15, 0.95, 1.05),
        crfEta=(0.85, 1.10, 0.90),
        crfXi=(0.46, 0.52, 0.49),
        crfGamma=(1.80, 2.00, 1.60),
    )
    radiance = np.asarray(
        (
            (0.05, 0.08, 0.10),
            (0.09, 0.12, 0.07),
            (0.14, 0.11, 0.16),
            (0.18, 0.16, 0.12),
            (0.22, 0.19, 0.15),
            (0.26, 0.17, 0.13),
        ),
        dtype=np.float32,
    )
    sensor_coords = np.asarray(
        (
            (0.18, 0.22),
            (0.33, 0.41),
            (0.47, 0.63),
            (0.58, 0.27),
            (0.71, 0.54),
            (0.82, 0.76),
        ),
        dtype=np.float32,
    )

    forward, recovered = _dispatch_ppisp_round_trip(device, params, radiance, sensor_coords)

    assert np.all(np.isfinite(forward))
    assert np.all(np.isfinite(recovered))
    assert np.all(forward >= 0.0)
    assert np.all(forward <= 1.0)
    np.testing.assert_allclose(recovered, radiance, rtol=0.0, atol=1.5e-3)


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


def test_pair_pool_dispatch_sampling_matches_full_sampling() -> None:
    recon, frames = _make_reconstruction()
    pool = build_photometric_observation_pair_pool(recon, frames, min_track_length=2)

    full = pool.sample(np.random.default_rng(9), 3)
    dispatch = pool.sample_dispatch_batch(np.random.default_rng(9), 3)

    np.testing.assert_array_equal(dispatch.frame_indices_a, full.frame_indices_a)
    np.testing.assert_array_equal(dispatch.frame_indices_b, full.frame_indices_b)
    np.testing.assert_array_equal(dispatch.observation_indices_a, full.observation_indices_a)
    np.testing.assert_array_equal(dispatch.observation_indices_b, full.observation_indices_b)


def test_track_pool_dispatch_sampling_matches_full_sampling() -> None:
    recon, frames = _make_reconstruction()
    pool = photometric_compensation_module.build_photometric_observation_track_pool(recon, frames, min_track_length=2)

    full = pool.sample(np.random.default_rng(9), 3)
    dispatch = pool.sample_dispatch_batch(np.random.default_rng(9), 3)

    np.testing.assert_array_equal(dispatch.frame_indices_a, full.frame_indices_a)
    np.testing.assert_array_equal(dispatch.frame_indices_b, full.frame_indices_b)
    np.testing.assert_array_equal(dispatch.observation_indices_a, full.observation_indices_a)
    np.testing.assert_array_equal(dispatch.observation_indices_b, full.observation_indices_b)


def test_build_photometric_observation_pair_pool_scales_sparse_tracks_to_training_frame_resolution() -> None:
    recon, frames = _make_scaled_reconstruction()

    pool = build_photometric_observation_pair_pool(recon, frames, min_track_length=2)

    assert len(pool) == 4
    np.testing.assert_array_equal(pool.frame_indices_a, np.array([0, 0, 1, 0], dtype=np.int32))
    np.testing.assert_array_equal(pool.frame_indices_b, np.array([1, 2, 2, 1], dtype=np.int32))
    np.testing.assert_allclose(pool.xy_a, np.array([[10.0, 20.0], [10.0, 20.0], [12.0, 22.0], [40.0, 18.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(pool.xy_b, np.array([[12.0, 22.0], [14.0, 24.0], [14.0, 24.0], [42.0, 17.0]], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_build_photometric_observation_track_pool_falls_back_to_database_matches(tmp_path: Path) -> None:
    frames = [_make_frame(1), _make_frame(2), _make_frame(3)]
    cameras = {
        0: ColmapCamera(camera_id=0, model_id=1, width=128, height=96, fx=144.0, fy=144.0, cx=64.0, cy=48.0),
    }
    images = {
        1: ColmapImage(1, frames[0].q_wxyz, frames[0].t_xyz, 0, "frame_1.png", np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
        2: ColmapImage(2, frames[1].q_wxyz, frames[1].t_xyz, 0, "frame_2.png", np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
        3: ColmapImage(3, frames[2].q_wxyz, frames[2].t_xyz, 0, "frame_3.png", np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
    }
    recon = ColmapReconstruction(root=tmp_path, sparse_dir=tmp_path / "sparse" / "0", cameras=cameras, images=images, points3d={})
    database_path = _write_match_database(
        tmp_path / "database.db",
        image_names={101: "frame_1.png", 202: "frame_2.png", 303: "frame_3.png"},
        keypoints={
            101: np.array([[20.0, 40.0], [80.0, 36.0]], dtype=np.float32),
            202: np.array([[24.0, 44.0], [84.0, 34.0], [16.0, 10.0]], dtype=np.float32),
            303: np.array([[28.0, 48.0], [120.0, 60.0]], dtype=np.float32),
        },
        matches={
            (101, 202): np.array([[0, 0], [1, 1]], dtype=np.uint32),
            (101, 303): np.array([[0, 0]], dtype=np.uint32),
        },
    )

    track_pool = photometric_compensation_module.build_photometric_observation_track_pool(
        recon,
        frames,
        min_track_length=2,
        database_path=database_path,
    )
    pair_pool = build_photometric_observation_pair_pool(
        recon,
        frames,
        min_track_length=2,
        database_path=database_path,
    )

    np.testing.assert_array_equal(track_pool.point_ids, np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(track_pool.track_lengths, np.array([3, 2], dtype=np.int32))
    np.testing.assert_array_equal(track_pool.observation_frame_indices, np.array([0, 1, 2, 0, 1], dtype=np.int32))
    np.testing.assert_allclose(track_pool.observation_xy, np.array([[10.0, 20.0], [12.0, 22.0], [14.0, 24.0], [40.0, 18.0], [42.0, 17.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(pair_pool.frame_indices_a, np.array([0, 0, 1, 0], dtype=np.int32))
    np.testing.assert_array_equal(pair_pool.frame_indices_b, np.array([1, 2, 2, 1], dtype=np.int32))
    np.testing.assert_allclose(pair_pool.xy_a, np.array([[10.0, 20.0], [10.0, 20.0], [12.0, 22.0], [40.0, 18.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(pair_pool.xy_b, np.array([[12.0, 22.0], [14.0, 24.0], [14.0, 24.0], [42.0, 17.0]], dtype=np.float32), rtol=0.0, atol=1e-6)


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


def test_photometric_gamma_regularization_is_independent_from_crf_weights(device) -> None:
    recon, frames = _make_reconstruction()
    hparams = PhotometricCompensationHyperParams(
        learning_rate=0.2,
        exposure_regularize_weight=0.0,
        vignette_regularize_weight=0.0,
        chroma_regularize_weight=0.0,
        crf_regularize_weight=0.0,
        gamma_regularize_weight=0.5,
        exposure_l1_weight=0.0,
        vignette_l1_weight=0.0,
        chroma_l1_weight=0.0,
        crf_l1_weight=0.0,
        gamma_l1_weight=0.05,
    )
    trainer = PhotometricCompensationTrainer(device, recon, frames, hparams=hparams, seed=11)
    identity = identity_packed_ppisp_params(len(frames))
    params = identity.copy()
    params[24:33, :] += 0.25
    params[33:36, :] += 0.6
    trainer.replace_packed_params(params)

    start_gamma_distance = float(np.mean(np.abs(params[33:36, :] - identity[33:36, :]), dtype=np.float64))
    start_other_crf_distance = float(np.mean(np.abs(params[24:33, :] - identity[24:33, :]), dtype=np.float64))
    for step in range(1, 65):
        trainer.zero_grads()
        trainer.step_optimizer(step)

    final_params = trainer.read_packed_params()
    final_gamma_distance = float(np.mean(np.abs(final_params[33:36, :] - identity[33:36, :]), dtype=np.float64))
    final_other_crf_distance = float(np.mean(np.abs(final_params[24:33, :] - identity[24:33, :]), dtype=np.float64))

    assert final_gamma_distance < start_gamma_distance * 0.5
    assert final_other_crf_distance == pytest.approx(start_other_crf_distance, rel=0.0, abs=5e-5)


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
    total_observations = int(trainer.track_pool.observation_frame_indices.size)

    assert trainer._pair_dataset_uploaded is True
    assert "observation_mean_samples" in trainer.buffers
    assert "observation_sensor_coords" in trainer.buffers
    assert "frame_pixels" not in trainer.buffers
    assert total_observations < total_frame_pixels
    assert trainer.buffers["observation_mean_samples"].size >= total_observations * 16


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

    observation_count = int(trainer.track_pool.observation_frame_indices.size)
    expected_mean_samples, expected_sensor_coords = _reference_observation_dataset(
        frames,
        trainer.track_pool,
        frame_rgba_linear,
        trainer.hparams.neighborhood_size,
    )
    observation_mean_samples = buffer_to_numpy(trainer.buffers["observation_mean_samples"], np.float32)[: observation_count * 4].reshape(observation_count, 4)
    observation_sensor_coords = buffer_to_numpy(trainer.buffers["observation_sensor_coords"], np.float32)[: observation_count * 2].reshape(observation_count, 2)

    np.testing.assert_allclose(observation_mean_samples, expected_mean_samples, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(observation_sensor_coords, expected_sensor_coords, rtol=0.0, atol=1e-6)


def test_photometric_incremental_pair_dataset_preparation_reports_progress(device) -> None:
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

    trainer.begin_prepare_pair_dataset()

    assert trainer.pair_dataset_prepare_active is True
    assert trainer.pair_dataset_prepare_total_frames > 0
    assert trainer.pair_dataset_prepare_completed_frames == 0
    assert trainer.pair_dataset_prepare_fraction == pytest.approx(0.0)
    assert trainer.pair_dataset_prepare_current_name != ""

    completed = trainer.advance_prepare_pair_dataset(frame_budget=1)

    assert completed is False
    assert trainer.pair_dataset_prepare_active is True
    assert trainer.pair_dataset_prepare_completed_frames == 1
    assert trainer.pair_dataset_prepare_fraction > 0.0

    trainer.prepare_pair_dataset()

    assert trainer.pair_dataset_prepare_active is False
    assert trainer._pair_dataset_uploaded is True
    assert trainer.pair_dataset_prepare_fraction == pytest.approx(1.0)


def test_photometric_trainer_large_pair_pool_uses_precomputed_observation_dataset(device, monkeypatch) -> None:
    recon, frames = _make_reconstruction()
    frame_rgba_linear = [
        _make_linear_rgba(frame.height, frame.width, rgb_scale=1.0 + 0.1 * frame_index)
        for frame_index, frame in enumerate(frames)
    ]
    monkeypatch.setattr(photometric_compensation_module, "_PHOTOMETRIC_FULL_PAIR_DATASET_MAX_PAIRS", 0)
    trainer = PhotometricCompensationTrainer(
        device,
        recon,
        frames,
        hparams=PhotometricCompensationHyperParams(batch_pair_count=4, neighborhood_size=3),
        seed=17,
        frame_rgba_linear=frame_rgba_linear,
    )

    assert trainer.uses_full_pair_dataset is False
    assert trainer._pair_dataset_uploaded is False
    assert trainer._pair_dataset_observation_capacity == 0

    trainer.prepare_pair_dataset()

    observation_count = int(trainer.track_pool.observation_frame_indices.size)
    expected_mean_samples, expected_sensor_coords = _reference_observation_dataset(
        frames,
        trainer.track_pool,
        frame_rgba_linear,
        trainer.hparams.neighborhood_size,
    )
    observation_mean_samples = buffer_to_numpy(trainer.buffers["observation_mean_samples"], np.float32)[: observation_count * 4].reshape(observation_count, 4)
    observation_sensor_coords = buffer_to_numpy(trainer.buffers["observation_sensor_coords"], np.float32)[: observation_count * 2].reshape(observation_count, 2)

    np.testing.assert_allclose(observation_mean_samples, expected_mean_samples, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(observation_sensor_coords, expected_sensor_coords, rtol=0.0, atol=1e-6)
    assert trainer._pair_dataset_observation_capacity == observation_count

    monkeypatch.setattr(trainer, "_prepare_pair_dataset_for_batch", lambda dispatch_batch: (_ for _ in ()).throw(AssertionError("batch-local pair dataset rebuild should not run")))
    trainer.train_step(pair_count=2, step_index=1)
    assert trainer.state.step == 1


def test_photometric_train_step_avoids_cpu_pair_batch_sampling(device, monkeypatch) -> None:
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

    monkeypatch.setattr(trainer, "build_dispatch_batch", lambda pair_count=None: (_ for _ in ()).throw(AssertionError("CPU dispatch batch builder should not run")))
    monkeypatch.setattr(trainer, "_upload_pair_batch", lambda dispatch_batch: (_ for _ in ()).throw(AssertionError("CPU pair batch upload should not run")))

    trainer.train_step(pair_count=2, step_index=1)

    assert trainer.state.step == 1


def test_photometric_target_average_exposure_controls_exposure_regularization_target(device) -> None:
    recon, frames = _make_reconstruction()
    trainer = PhotometricCompensationTrainer(
        device,
        recon,
        frames,
        hparams=PhotometricCompensationHyperParams(
            batch_pair_count=4,
            neighborhood_size=3,
            learning_rate=0.2,
            target_average_exposure=0.5,
            exposure_regularize_weight=0.5,
            vignette_regularize_weight=0.0,
            chroma_regularize_weight=0.0,
            crf_regularize_weight=0.0,
            exposure_l1_weight=0.0,
            vignette_l1_weight=0.0,
            chroma_l1_weight=0.0,
            crf_l1_weight=0.0,
        ),
        seed=23,
    )

    params = identity_packed_ppisp_params(len(frames))
    params[0, :] = np.array((-0.75, 1.25, -0.25), dtype=np.float32)
    trainer.replace_packed_params(params)

    start_distance = float(np.mean(np.abs(trainer.read_packed_params()[0, :] - np.float32(0.5)), dtype=np.float64))
    for step in range(1, 33):
        trainer.zero_grads()
        trainer.step_optimizer(step)

    final_exposure = trainer.read_packed_params()[0, :]
    final_distance = float(np.mean(np.abs(final_exposure - np.float32(0.5)), dtype=np.float64))

    assert final_distance < start_distance * 0.5
    assert float(np.mean(final_exposure, dtype=np.float64)) == pytest.approx(0.5, abs=0.1)


def test_photometric_exposure_l1_regularization_uses_target_average_exposure(device) -> None:
    recon, frames = _make_reconstruction()
    trainer = PhotometricCompensationTrainer(
        device,
        recon,
        frames,
        hparams=PhotometricCompensationHyperParams(
            batch_pair_count=4,
            neighborhood_size=3,
            learning_rate=0.2,
            target_average_exposure=0.5,
            exposure_lr_mul=1.0,
            exposure_regularize_weight=0.0,
            vignette_regularize_weight=0.0,
            chroma_regularize_weight=0.0,
            crf_regularize_weight=0.0,
            exposure_l1_weight=0.2,
            vignette_l1_weight=0.0,
            chroma_l1_weight=0.0,
            crf_l1_weight=0.0,
        ),
        seed=37,
    )

    params = identity_packed_ppisp_params(len(frames))
    params[0, :] = np.array((-0.5, -0.25, -0.75), dtype=np.float32)
    trainer.replace_packed_params(params)

    start_distance = float(np.mean(np.abs(trainer.read_packed_params()[0, :] - np.float32(0.5)), dtype=np.float64))
    for step in range(1, 33):
        trainer.zero_grads()
        trainer.step_optimizer(step)

    final_exposure = trainer.read_packed_params()[0, :]
    final_distance = float(np.mean(np.abs(final_exposure - np.float32(0.5)), dtype=np.float64))

    assert final_distance < start_distance * 0.5
    assert float(np.mean(final_exposure, dtype=np.float64)) == pytest.approx(0.5, abs=0.1)

def test_photometric_gamma_l1_regularization_shrinks_toward_gamma_identity(device) -> None:
    recon, frames = _make_reconstruction()
    trainer = PhotometricCompensationTrainer(
        device,
        recon,
        frames,
        hparams=PhotometricCompensationHyperParams(
            batch_pair_count=4,
            neighborhood_size=3,
            learning_rate=0.2,
            crf_lr_mul=1.0,
            exposure_regularize_weight=0.0,
            vignette_regularize_weight=0.0,
            chroma_regularize_weight=0.0,
            crf_regularize_weight=0.0,
            gamma_regularize_weight=0.0,
            exposure_l1_weight=0.0,
            vignette_l1_weight=0.0,
            chroma_l1_weight=0.0,
            crf_l1_weight=0.0,
            gamma_l1_weight=0.2,
            enable_exposure=False,
            enable_color=False,
            enable_vignette=False,
            enable_gamma=True,
        ),
        seed=41,
    )

    gamma_layout = next(layout for layout in photometric_compensation_module._FIELD_LAYOUTS if layout.attr == "crfGamma")
    params = identity_packed_ppisp_params(len(frames))
    params[gamma_layout.start : gamma_layout.stop, :] = np.array([[3.25], [3.1], [3.4]], dtype=np.float32)
    trainer.replace_packed_params(params)

    gamma_target = photometric_compensation_module._PPISP_IDENTITY_VALUES[gamma_layout.start : gamma_layout.stop].reshape(gamma_layout.size, 1)
    start_distance = float(np.mean(np.abs(trainer.read_packed_params()[gamma_layout.start : gamma_layout.stop, :] - gamma_target), dtype=np.float64))
    for step in range(1, 49):
        trainer.zero_grads()
        trainer.step_optimizer(step)

    final_gamma = trainer.read_packed_params()[gamma_layout.start : gamma_layout.stop, :]
    final_distance = float(np.mean(np.abs(final_gamma - gamma_target), dtype=np.float64))

    assert final_distance < start_distance * 0.5


def test_photometric_replace_packed_params_keeps_all_frames_trainable(device) -> None:
    recon, frames = _make_reconstruction()
    trainer = PhotometricCompensationTrainer(
        device,
        recon,
        frames,
        hparams=PhotometricCompensationHyperParams(batch_pair_count=4, neighborhood_size=3),
        seed=23,
    )

    params = identity_packed_ppisp_params(len(frames))
    params[:, 0] += np.float32(0.25)
    params[:, 1] += np.float32(0.10)
    trainer.replace_packed_params(params)

    np.testing.assert_allclose(trainer.read_packed_params(), params, rtol=0.0, atol=1e-6)


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
    identity = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=2, seed=13),
        frames=[frame],
        seed=5,
        target_tonemap_provider=PPISPStaticTonemapProvider(PPISPTonemapParams()),
    )
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=5, target_tonemap_provider=provider)

    identity_target = np.asarray(identity.get_frame_target_texture(0, native_resolution=False).to_numpy(), dtype=np.float32)
    target_np = np.asarray(trainer.get_frame_target_texture(0, native_resolution=False).to_numpy(), dtype=np.float32)

    np.testing.assert_allclose(target_np[:, :, :3], identity_target[:, :, :3] * 0.5, rtol=0.0, atol=5e-4)
    np.testing.assert_allclose(target_np[:, :, 3], identity_target[:, :, 3], rtol=0.0, atol=1e-6)


def test_gaussian_trainer_identity_target_tonemap_matches_inverse_target_contract_for_native_subsample(device, tmp_path: Path) -> None:
    image = np.array(
        [
            [[16, 24, 32], [24, 32, 40], [32, 40, 48], [40, 48, 56]],
            [[24, 32, 40], [32, 40, 48], [40, 48, 56], [48, 56, 64]],
            [[32, 40, 48], [40, 48, 56], [48, 56, 64], [56, 64, 72]],
            [[40, 48, 56], [48, 56, 64], [56, 64, 72], [64, 72, 80]],
        ],
        dtype=np.uint8,
    )
    frame = _make_rgb_frame(tmp_path, image, image_name="photometric_native_identity_target.png")
    identity = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=19),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=11,
        target_tonemap_provider=PPISPStaticTonemapProvider(PPISPTonemapParams()),
    )
    exposed = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=19),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=11,
        target_tonemap_provider=PPISPStaticTonemapProvider(PPISPTonemapParams(exposureEv=1.0)),
    )

    target_texture = identity.get_frame_target_texture(0, native_resolution=True)
    target_np = np.asarray(target_texture.to_numpy(), dtype=np.float32)
    exposed_np = np.asarray(exposed.get_frame_target_texture(0, native_resolution=True).to_numpy(), dtype=np.float32)

    assert identity._loss_vars(0, step=0, target_texture=target_texture)["g_TargetTextureIsLinear"] == np.uint32(1)
    np.testing.assert_allclose(exposed_np[:, :, :3], target_np[:, :, :3] * 0.5, rtol=0.0, atol=5e-4)
    np.testing.assert_allclose(exposed_np[:, :, 3], target_np[:, :, 3], rtol=0.0, atol=1e-6)


def test_gaussian_trainer_identity_target_tonemap_matches_no_provider_baseline(device, tmp_path: Path) -> None:
    image = np.array(
        [
            [[16, 24, 32], [24, 32, 40], [32, 40, 48], [40, 48, 56]],
            [[24, 32, 40], [32, 40, 48], [40, 48, 56], [48, 56, 64]],
            [[32, 40, 48], [40, 48, 56], [48, 56, 64], [56, 64, 72]],
            [[40, 48, 56], [48, 56, 64], [56, 64, 72], [64, 72, 80]],
        ],
        dtype=np.uint8,
    )
    frame = _make_rgb_frame(tmp_path, image, image_name="photometric_native_identity_baseline.png")
    baseline = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=29),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=17,
    )
    identity = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=29),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=17,
        target_tonemap_provider=PPISPStaticTonemapProvider(PPISPTonemapParams()),
    )

    _, baseline_ssim = _native_subsample_target_ssim_rgb(baseline, device)
    _, identity_ssim = _native_subsample_target_ssim_rgb(identity, device)

    np.testing.assert_allclose(identity_ssim, baseline_ssim, rtol=0.0, atol=5e-4)


def test_gaussian_trainer_can_bypass_target_tonemap_for_debug_targets(device, tmp_path: Path) -> None:
    image = np.array(
        [
            [[16, 24, 32], [24, 32, 40], [32, 40, 48], [40, 48, 56]],
            [[24, 32, 40], [32, 40, 48], [40, 48, 56], [48, 56, 64]],
            [[32, 40, 48], [40, 48, 56], [48, 56, 64], [56, 64, 72]],
            [[40, 48, 56], [48, 56, 64], [56, 64, 72], [64, 72, 80]],
        ],
        dtype=np.uint8,
    )
    frame = _make_rgb_frame(tmp_path, image, image_name="photometric_native_debug_bypass.png")
    baseline = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=29),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=17,
    )
    compensated = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=29),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=17,
        target_tonemap_provider=PPISPStaticTonemapProvider(PPISPTonemapParams(exposureEv=1.0)),
    )

    baseline_np = np.asarray(baseline.get_frame_target_texture(0, native_resolution=True).to_numpy(), dtype=np.float32)
    bypass_texture = compensated.get_frame_target_texture(0, native_resolution=True, apply_target_tonemap=False)
    bypass_np = np.asarray(bypass_texture.to_numpy(), dtype=np.float32)
    compensated_texture = compensated.get_frame_target_texture(0, native_resolution=True)
    compensated_np = np.asarray(compensated_texture.to_numpy(), dtype=np.float32)

    np.testing.assert_allclose(bypass_np, baseline_np, rtol=0.0, atol=1e-6)
    assert compensated.target_texture_is_linear(compensated_texture) is True
    assert compensated.target_texture_is_linear(bypass_texture) is False
    assert float(np.max(np.abs(compensated_np[:, :, :3] - bypass_np[:, :, :3]))) > 1e-3


def test_gaussian_trainer_native_subsample_target_tonemap_approaches_baseline_near_identity(device, tmp_path: Path) -> None:
    image = np.array(
        [
            [[16, 24, 32], [24, 32, 40], [32, 40, 48], [40, 48, 56]],
            [[24, 32, 40], [32, 40, 48], [40, 48, 56], [48, 56, 64]],
            [[32, 40, 48], [40, 48, 56], [48, 56, 64], [56, 64, 72]],
            [[40, 48, 56], [48, 56, 64], [56, 64, 72], [64, 72, 80]],
        ],
        dtype=np.uint8,
    )
    frame = _make_rgb_frame(tmp_path, image, image_name="photometric_native_identity_distance.png")
    baseline = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=23),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=13,
    )
    near_identity = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=23),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=13,
        target_tonemap_provider=PPISPStaticTonemapProvider(PPISPTonemapParams(exposureEv=0.05)),
    )
    farther = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=23),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=13,
        target_tonemap_provider=PPISPStaticTonemapProvider(PPISPTonemapParams(exposureEv=1.0)),
    )

    _, baseline_ssim = _native_subsample_target_ssim_rgb(baseline, device)
    _, near_identity_ssim = _native_subsample_target_ssim_rgb(near_identity, device)
    _, farther_ssim = _native_subsample_target_ssim_rgb(farther, device)

    near_distance = float(np.max(np.abs(near_identity_ssim - baseline_ssim)))
    far_distance = float(np.max(np.abs(farther_ssim - baseline_ssim)))

    assert near_distance > 0.0
    assert near_distance < far_distance


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
    trainer = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=19),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=11,
        target_tonemap_provider=PPISPStaticTonemapProvider(PPISPTonemapParams(exposureEv=1.0)),
    )

    target_texture, ssim_rgb = _native_subsample_target_ssim_rgb(trainer, device)
    assert trainer._loss_vars(0, step=0, target_texture=target_texture)["g_TargetTextureIsLinear"] == np.uint32(1)

    baseline = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=19),
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=11,
    )
    _, baseline_ssim = _native_subsample_target_ssim_rgb(baseline, device)

    assert float(np.max(np.abs(ssim_rgb - baseline_ssim))) > 1e-3


def test_gaussian_trainer_reuses_single_compensated_native_target_across_frames(device, tmp_path: Path) -> None:
    image_a = np.array(
        [
            [[16, 32, 48], [32, 48, 64]],
            [[48, 64, 80], [64, 80, 96]],
        ],
        dtype=np.uint8,
    )
    image_b = np.array(
        [
            [[96, 80, 64], [80, 64, 48]],
            [[64, 48, 32], [48, 32, 16]],
        ],
        dtype=np.uint8,
    )
    frame_a = _make_rgb_frame(tmp_path, image_a, image_name="photometric_native_reuse_a.png")
    frame_b = _make_rgb_frame(tmp_path, image_b, image_name="photometric_native_reuse_b.png", image_id=1)
    scene = _make_scene(count=1, seed=31)
    renderer = GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16)
    provider = PPISPStaticTonemapProvider(PPISPTonemapParams(exposureEv=1.0))
    identity = GaussianTrainer(
        device=device,
        renderer=GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16),
        scene=_make_scene(count=1, seed=31),
        frames=[frame_a, frame_b],
        seed=17,
        target_tonemap_provider=PPISPStaticTonemapProvider(PPISPTonemapParams()),
    )
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame_a, frame_b], seed=17, target_tonemap_provider=provider)

    target_a = trainer.get_frame_target_texture(0, native_resolution=True)
    identity_a = np.asarray(identity.get_frame_target_texture(0, native_resolution=True).to_numpy(), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(target_a.to_numpy(), dtype=np.float32)[:, :, :3], identity_a[:, :, :3] * 0.5, rtol=0.0, atol=5e-4)

    target_b = trainer.get_frame_target_texture(1, native_resolution=True)
    identity_b = np.asarray(identity.get_frame_target_texture(1, native_resolution=True).to_numpy(), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(target_b.to_numpy(), dtype=np.float32)[:, :, :3], identity_b[:, :, :3] * 0.5, rtol=0.0, atol=5e-4)

    assert target_a is target_b


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
    assert exposure_values[0] == pytest.approx(exposure_values[2], abs=0.25)
    assert exposure_values[1] < exposure_values[0] - 0.25
    assert exposure_values[1] < exposure_values[2] - 0.25


def test_photometric_trainer_pair_loss_step_reduces_synthetic_gamma_error(device) -> None:
    recon, frames = _make_reconstruction()
    base = _make_linear_rgba(frames[0].height, frames[0].width)
    target_gamma = np.float32(1.4)
    identity_gamma = np.float32(2.2)
    frame_rgba_linear = [base.copy(), np.ascontiguousarray(base.copy(), dtype=np.float32), base.copy()]
    frame_rgba_linear[1][:, :, :3] = np.power(np.clip(base[:, :, :3], 1e-4, 1.0), identity_gamma / target_gamma).astype(np.float32)
    hparams = PhotometricCompensationHyperParams(
        batch_pair_count=512,
        neighborhood_size=3,
        learning_rate=0.15,
        exposure_lr_mul=0.0,
        vignette_lr_mul=0.0,
        chroma_lr_mul=0.0,
        crf_lr_mul=0.75,
        exposure_regularize_weight=0.0,
        vignette_regularize_weight=0.5,
        chroma_regularize_weight=0.5,
        crf_regularize_weight=0.5,
        gamma_regularize_weight=0.01,
        exposure_l1_weight=0.0,
        vignette_l1_weight=0.0,
        chroma_l1_weight=0.0,
        crf_l1_weight=0.05,
        gamma_l1_weight=0.0,
    )
    trainer = PhotometricCompensationTrainer(device, recon, frames, hparams=hparams, seed=31, frame_rgba_linear=frame_rgba_linear)

    first_loss = float("nan")
    for step in range(1, 241):
        loss = trainer.train_step(step_index=step)
        if step == 1:
            first_loss = float(loss)

    gamma_values = [trainer.provider.params_for_frame(index).crfGamma for index in range(len(frames))]
    frame_one_gamma = float(np.mean(np.asarray(gamma_values[1], dtype=np.float32), dtype=np.float64))
    frame_two_gamma = float(np.mean(np.asarray(gamma_values[2], dtype=np.float32), dtype=np.float64))

    assert np.isfinite(first_loss)
    assert np.isfinite(trainer.state.ema_loss)
    assert trainer.state.ema_loss < first_loss * 0.35
    assert frame_one_gamma > frame_two_gamma + 0.2


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
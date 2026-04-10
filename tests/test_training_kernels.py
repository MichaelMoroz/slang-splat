from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import slangpy as spy

from src.utility import buffer_to_numpy
from src.filter import SeparableGaussianBlur
from src.renderer import GaussianRenderer
from src.scene import ColmapFrame, GaussianInitHyperParams, GaussianScene
from src.scene.sh_utils import SH_C0, evaluate_sh_color
from src.training import gaussian_trainer as gaussian_trainer_module
from src.training import AdamHyperParams, GaussianTrainer, StabilityHyperParams, TRAIN_BACKGROUND_MODE_CUSTOM, TRAIN_BACKGROUND_MODE_RANDOM, TrainingHyperParams, contribution_fixed_count_from_percent, contribution_percent_from_fixed_count, resolve_auto_train_subsample_factor, resolve_base_learning_rate, resolve_clone_probability_threshold, resolve_cosine_base_learning_rate, resolve_depth_ratio_grad_band, resolve_depth_ratio_weight, resolve_effective_refinement_interval, resolve_effective_train_render_factor, resolve_lr_schedule_breakpoints, resolve_position_lr_mul, resolve_position_random_step_noise_lr, resolve_refinement_growth_ratio, resolve_refinement_min_contribution_percent, resolve_max_allowed_density, resolve_sh_band, resolve_sh_lr_mul, resolve_stage_schedule_steps, resolve_training_resolution, resolve_train_subsample_factor, resolve_use_sh, should_run_refinement_step

_ADAM_BUFFER_NAMES = ("adam_moments",)
_OPACITY_EPS = 1e-6
_raw_opacity = lambda alpha: (np.log(np.clip(np.asarray(alpha, dtype=np.float32), _OPACITY_EPS, 1.0 - _OPACITY_EPS)) - np.log1p(-np.clip(np.asarray(alpha, dtype=np.float32), _OPACITY_EPS, 1.0 - _OPACITY_EPS))).astype(np.float32, copy=False)
_actual_opacity = lambda raw: (1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float32)))).astype(np.float32, copy=False)
_actual_scale = lambda log_scale: np.exp(np.asarray(log_scale, dtype=np.float32))
_GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0
_REFINEMENT_MIN_SCREEN_RADIUS_PX = 1.0
_TRAINING_MAX_SCREEN_FRACTION = 0.1
_log_sigma = lambda sigma: np.log(np.asarray(sigma, dtype=np.float32))
_stored_from_support_scale = lambda support_scale: np.log(np.asarray(support_scale, dtype=np.float32) / _GAUSSIAN_SUPPORT_SIGMA_RADIUS)
_SCALE_GRAD_MULS = (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 0.9375, 0.96875, 1.0, 1.05)
_TARGET_MULS = (2.0, 3.0, 4.0, 6.0, 7.5)
_DEPTH_RATIO_GRAD_SOFTNESS_RATIO = 0.1
_DEPTH_RATIO_GRAD_SOFTNESS_FLOOR = 1e-4
_SSIM_BLUR_WEIGHTS = np.array(
    [
        0.00102838008447911,
        0.00759875813523919,
        0.03600077212843083,
        0.10936068950970002,
        0.21300553771125369,
        0.26601172486179436,
        0.21300553771125369,
        0.10936068950970002,
        0.03600077212843083,
        0.00759875813523919,
        0.00102838008447911,
    ],
    dtype=np.float32,
)
_SSIM_C1 = 0.0001
_SSIM_C2 = 0.0009
_SSIM_SMALL_VALUE = 1e-6
_SSIM_YCBCR_WEIGHTS = np.array([4.0, 1.0, 1.0], dtype=np.float32) / 6.0
_SSIM_FEATURE_CHANNELS = 15


def _expected_refinement_child_scale(parent_scale: np.ndarray, family_size: int) -> np.ndarray:
    scale = np.asarray(parent_scale, dtype=np.float32).reshape(3)
    shrink = float(max(int(family_size), 1)) ** (-1.0 / 3.0)
    child_log = np.log(scale) + np.log(shrink)
    order = np.argsort(np.log(scale))
    sorted_parent_log = np.log(scale[order])
    parent_span = float(sorted_parent_log[2] - sorted_parent_log[0])
    if not parent_span > 1e-6:
        return np.exp(child_log).astype(np.float32)
    target_ratio = max(float(scale[order[2]] / scale[order[0]]) * 0.5, 1.0)
    target_span = float(np.log(target_ratio))
    if not target_span > 1e-6:
        return np.full((3,), np.exp(np.mean(child_log, dtype=np.float64)), dtype=np.float32)
    mid_t = float((sorted_parent_log[1] - sorted_parent_log[0]) / parent_span)
    mean_log = float(np.mean(child_log, dtype=np.float64))
    sorted_min = mean_log - ((1.0 + mid_t) * target_span) / 3.0
    sorted_mid = sorted_min + mid_t * target_span
    sorted_max = sorted_min + target_span
    child_scale = np.zeros((3,), dtype=np.float32)
    child_scale[order] = np.exp(np.array([sorted_min, sorted_mid, sorted_max], dtype=np.float64)).astype(np.float32)
    return child_scale


def _make_scene(count: int = 24, seed: int = 7) -> GaussianScene:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32)
    positions[:, 2] += 2.0
    scales = _log_sigma(np.full((count, 3), 0.03, dtype=np.float32))
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.full((count,), 0.5, dtype=np.float32)
    colors = rng.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32)
    sh_coeffs = np.zeros((count, 1, 3), dtype=np.float32)
    return GaussianScene(positions=positions, scales=scales, rotations=rotations, opacities=opacities, colors=colors, sh_coeffs=sh_coeffs)


def _make_frame(tmp_path: Path, width: int = 64, height: int = 64, *, image_name: str = "target.png", image_id: int = 0, green_value: int = 180) -> ColmapFrame:
    image_path = tmp_path / image_name
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = int(green_value)
    Image.fromarray(image, mode="RGB").save(image_path)
    return ColmapFrame(
        image_id=image_id,
        image_path=image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        fx=72.0,
        fy=72.0,
        cx=width * 0.5,
        cy=height * 0.5,
        width=width,
        height=height,
    )


def _make_rgba_frame(tmp_path: Path, rgba: np.ndarray, *, image_name: str = "target_rgba.png", image_id: int = 0) -> ColmapFrame:
    image = np.asarray(rgba, dtype=np.uint8)
    height, width = image.shape[:2]
    Image.fromarray(image).save(tmp_path / image_name)
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


def _make_float_texture(device: spy.Device, image: np.ndarray) -> spy.Texture:
    rgba = np.asarray(image, dtype=np.float32)
    texture = device.create_texture(
        format=spy.Format.rgba32_float,
        width=int(rgba.shape[1]),
        height=int(rgba.shape[0]),
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )
    texture.copy_from_numpy(np.ascontiguousarray(rgba, dtype=np.float32))
    return texture


def _read_scene_groups(renderer: GaussianRenderer, count: int) -> dict[str, np.ndarray]:
    return renderer.read_scene_groups(count)


def _read_grad_groups(renderer: GaussianRenderer, count: int) -> dict[str, np.ndarray]:
    return renderer.read_grad_groups(count)


def _write_grad_groups(
    renderer: GaussianRenderer,
    count: int,
    *,
    grad_positions: np.ndarray | None = None,
    grad_scales: np.ndarray | None = None,
    grad_rotations: np.ndarray | None = None,
    grad_sh_coeffs: np.ndarray | None = None,
    grad_color_alpha: np.ndarray | None = None,
) -> None:
    renderer.write_grad_groups(
        count,
        grad_positions=grad_positions,
        grad_scales=grad_scales,
        grad_rotations=grad_rotations,
        grad_sh_coeffs=grad_sh_coeffs,
        grad_color_alpha=grad_color_alpha,
    )


def _read_output_grads(renderer: GaussianRenderer) -> np.ndarray:
    flat = buffer_to_numpy(renderer.output_grad_buffer, np.float32)
    return flat[: max(renderer.width * renderer.height, 1) * 4].reshape(renderer.height, renderer.width, 4)


def _read_training_rgb_loss(renderer: GaussianRenderer) -> np.ndarray:
    flat = buffer_to_numpy(renderer.work_buffers["training_rgb_loss"], np.float32)
    return flat[: max(renderer.width * renderer.height, 1)].reshape(renderer.height, renderer.width).copy()


def _read_training_rgb_loss_total(renderer: GaussianRenderer) -> float:
    return float(buffer_to_numpy(renderer.work_buffers["training_rgb_loss_total"], np.float32)[0])


def _read_training_depth_ratios(renderer: GaussianRenderer) -> np.ndarray:
    return np.asarray(renderer.training_depth_stats_texture.to_numpy(), dtype=np.float32)[..., 3].reshape(-1).copy()


def _read_ssim_buffer(trainer: GaussianTrainer, name: str) -> np.ndarray:
    width = int(trainer.renderer.width)
    height = int(trainer.renderer.height)
    flat = buffer_to_numpy(trainer._buffers[name], np.float32)
    return np.asarray(flat[: max(width * height, 1) * _SSIM_FEATURE_CHANNELS], dtype=np.float32).reshape(height, width, _SSIM_FEATURE_CHANNELS).copy()


def _rgb_to_ycbcr_bt601_np(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float32)
    y = np.tensordot(value, np.array([0.299, 0.587, 0.114], dtype=np.float32), axes=([-1], [0]))
    cb = np.tensordot(value, np.array([-0.168736, -0.331264, 0.5], dtype=np.float32), axes=([-1], [0])) + 0.5
    cr = np.tensordot(value, np.array([0.5, -0.418688, -0.081312], dtype=np.float32), axes=([-1], [0])) + 0.5
    return np.stack((y, cb, cr), axis=-1).astype(np.float32, copy=False)


def _ssim_feature_moments_np(rendered_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    rendered = _rgb_to_ycbcr_bt601_np(rendered_rgb)
    target = _rgb_to_ycbcr_bt601_np(target_rgb)
    return np.concatenate((rendered, target, rendered * rendered, target * target, rendered * target), axis=2).astype(np.float32, copy=False)


def _blur_axis_clamped_np(values: np.ndarray, axis: int) -> np.ndarray:
    radius = len(_SSIM_BLUR_WEIGHTS) // 2
    pad = [(0, 0)] * values.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(values, pad, mode="edge")
    out = np.zeros_like(values, dtype=np.float32)
    for tap, weight in enumerate(_SSIM_BLUR_WEIGHTS):
        offset = tap
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(offset, offset + values.shape[axis])
        out += np.asarray(weight, dtype=np.float32) * padded[tuple(slices)]
    return out


def _separable_gaussian_blur_np(values: np.ndarray) -> np.ndarray:
    return _blur_axis_clamped_np(_blur_axis_clamped_np(np.asarray(values, dtype=np.float32), 1), 0)


def _training_target_mask_np(alpha: np.ndarray, use_target_alpha_mask: bool) -> np.ndarray:
    return np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0) if bool(use_target_alpha_mask) else np.ones_like(alpha, dtype=np.float32)


def _blended_rgb_metrics_np(
    rendered_rgba: np.ndarray,
    target_rgba: np.ndarray,
    *,
    ssim_weight: float,
    use_target_alpha_mask: bool,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    rendered = np.asarray(rendered_rgba, dtype=np.float32)[..., :3]
    target = np.asarray(target_rgba, dtype=np.float32)[..., :3]
    alpha = np.asarray(target_rgba, dtype=np.float32)[..., 3]
    inv_pixel_count = np.float32(1.0 / max(rendered.shape[0] * rendered.shape[1], 1))
    mask = _training_target_mask_np(alpha, use_target_alpha_mask)
    diff = rendered - target
    l1 = np.mean(np.abs(diff), axis=2).astype(np.float32) * inv_pixel_count * mask
    mse = float(np.sum(np.mean(diff * diff, axis=2).astype(np.float32) * inv_pixel_count * mask, dtype=np.float64))
    blurred = _separable_gaussian_blur_np(_ssim_feature_moments_np(rendered, target))
    x, y, x2, y2, xy = np.split(blurred, 5, axis=2)
    sigma_x2 = np.maximum(x2 - x * x, 0.0)
    sigma_y2 = np.maximum(y2 - y * y, 0.0)
    sigma_xy = xy - x * y
    numer = (2.0 * x * y + _SSIM_C1) * (2.0 * sigma_xy + _SSIM_C2)
    denom = np.maximum((x * x + y * y + _SSIM_C1) * (sigma_x2 + sigma_y2 + _SSIM_C2), _SSIM_SMALL_VALUE)
    ssim = numer / denom
    dssim = 0.5 * (1.0 - np.sum(ssim * _SSIM_YCBCR_WEIGHTS.reshape(1, 1, 3), axis=2, dtype=np.float32))
    dssim = dssim.astype(np.float32, copy=False) * inv_pixel_count * mask
    blended = ((1.0 - float(ssim_weight)) * l1 + float(ssim_weight) * dssim).astype(np.float32, copy=False)
    l1_grad = np.sign(diff).astype(np.float32) * (np.float32(1.0 / 3.0) * inv_pixel_count * mask[..., None] * np.float32(1.0 - float(ssim_weight)))
    return blended, float(np.sum(blended, dtype=np.float64)), mse, l1_grad


def _make_loss_only_trainer(
    device: spy.Device,
    tmp_path: Path,
    *,
    width: int,
    height: int,
    training_hparams: TrainingHyperParams,
    image_name: str,
    image_id: int,
) -> GaussianTrainer:
    renderer = GaussianRenderer(device, width=width, height=height, list_capacity_multiplier=4)
    return GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=_make_scene(count=1, seed=97),
        frames=[_make_frame(tmp_path, width=width, height=height, image_name=image_name, image_id=image_id)],
        training_hparams=training_hparams,
        seed=123,
    )


def _dispatch_manual_loss(
    trainer: GaussianTrainer,
    rendered_rgba: np.ndarray,
    target_rgba: np.ndarray,
    *,
    run_backward: bool = True,
) -> spy.Texture:
    renderer = trainer.renderer
    rendered = np.asarray(rendered_rgba, dtype=np.float32)
    target = np.asarray(target_rgba, dtype=np.float32)
    renderer.output_texture.copy_from_numpy(np.ascontiguousarray(rendered, dtype=np.float32))
    renderer.training_depth_stats_texture.copy_from_numpy(np.zeros((renderer.height, renderer.width, 4), dtype=np.float32))
    renderer.work_buffers["training_density"].copy_from_numpy(np.zeros((renderer.width * renderer.height,), dtype=np.float32))
    target_texture = _make_float_texture(trainer.device, target)
    enc = trainer.device.create_command_encoder()
    trainer._dispatch_loss_forward(enc, target_texture)
    if run_backward:
        trainer._dispatch_loss_backward(enc, target_texture)
    trainer.device.submit_command_buffer(enc.finish())
    trainer.device.wait()
    return target_texture


def _depth_ratio_window_softplus(x: float) -> float:
    if x >= 30.0: return float(x)
    if x <= -30.0: return float(np.exp(x))
    return float(np.log1p(np.exp(x)))


def _depth_ratio_window_sigmoid(x: float) -> float:
    if x >= 0.0:
        z = float(np.exp(-x))
        return 1.0 / (1.0 + z)
    z = float(np.exp(x))
    return z / (1.0 + z)


def _depth_ratio_window_params(grad_min: float, grad_max: float) -> tuple[float, float, float]:
    band_min, band_max = resolve_depth_ratio_grad_band(grad_min, grad_max)
    softness = max((band_max - band_min) * _DEPTH_RATIO_GRAD_SOFTNESS_RATIO, _DEPTH_RATIO_GRAD_SOFTNESS_FLOOR)
    return band_min, band_max, softness


def _depth_ratio_window_loss(depth_ratio: float, weight: float, grad_min: float, grad_max: float) -> float:
    band_min, band_max, softness = _depth_ratio_window_params(grad_min, grad_max)
    x = max(float(depth_ratio), 0.0)
    return float(weight) * softness * (
        _depth_ratio_window_softplus((x - band_min) / softness) - _depth_ratio_window_softplus((x - band_max) / softness)
    )


def _depth_ratio_window_grad(depth_ratio: float, weight: float, grad_min: float, grad_max: float, inv_pixel_count: float) -> float:
    band_min, band_max, softness = _depth_ratio_window_params(grad_min, grad_max)
    x = max(float(depth_ratio), 0.0)
    return float(weight) * (
        _depth_ratio_window_sigmoid((x - band_min) / softness) - _depth_ratio_window_sigmoid((x - band_max) / softness)
    ) * float(inv_pixel_count)


def _read_optimizer_lrs(trainer: GaussianTrainer) -> np.ndarray:
    flat = buffer_to_numpy(trainer.optimizer.param_settings, np.uint32)
    return flat.reshape(trainer.renderer.TRAINABLE_PARAM_COUNT, 8)[:, 0].view(np.float32).copy()


def _read_adam_moments(trainer: GaussianTrainer, splat_count: int) -> np.ndarray:
    flat = buffer_to_numpy(trainer.adam_optimizer.buffers["adam_moments"], np.float32)
    count = max(int(splat_count), 1) * trainer.renderer.TRAINABLE_PARAM_COUNT * 2
    return flat[:count].reshape(trainer.renderer.TRAINABLE_PARAM_COUNT, max(int(splat_count), 1), 2).copy()


def _circle_bound_support_radius(camera, position: np.ndarray, width: int, height: int, radius_px: float) -> float | None:
    center_px, visible = camera.project_world_to_screen(position, width, height)
    if not visible:
        return None
    if center_px[0] < 0.0 or center_px[0] >= float(width) or center_px[1] < 0.0 or center_px[1] >= float(height):
        return None
    viewport_center = np.array([0.5 * float(width), 0.5 * float(height)], dtype=np.float32)
    to_viewport_center = viewport_center - np.asarray(center_px, dtype=np.float32)
    distance = float(np.linalg.norm(to_viewport_center))
    if distance <= 1e-6:
        sample_px = np.asarray(center_px, dtype=np.float32) + np.array([radius_px, 0.0], dtype=np.float32)
    else:
        sample_px = np.asarray(center_px, dtype=np.float32) + to_viewport_center * np.float32(radius_px / distance)
    ray_dir = camera.screen_to_world_ray(sample_px, width, height)
    to_center = np.asarray(position, dtype=np.float32) - np.asarray(camera.position, dtype=np.float32)
    return float(np.linalg.norm(np.cross(to_center, ray_dir)))


class _ScaleGradProbe:
    def __init__(self, device, tmp_path: Path, *, image_name: str):
        self.device = device
        self.background = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.frame = _make_frame(tmp_path, image_name=image_name)
        self.camera = self.frame.make_camera(near=0.1, far=20.0)
        self.target_renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
        self.pixel_floor_scale = self.camera.pixel_world_size_max(3.0, self.target_renderer.width, self.target_renderer.height)
        self.target_scene = GaussianScene(
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            scales=np.full((1, 3), _stored_from_support_scale(self.pixel_floor_scale), dtype=np.float32),
            rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            opacities=np.array([0.75], dtype=np.float32),
            colors=np.array([[0.8, 0.6, 0.2]], dtype=np.float32),
            sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
        )
        self.renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
        self.trainer = GaussianTrainer(
            device=device,
            renderer=self.renderer,
            scene=GaussianScene(
                positions=self.target_scene.positions.copy(),
                scales=np.full((1, 3), _stored_from_support_scale(self.pixel_floor_scale), dtype=np.float32),
                rotations=self.target_scene.rotations.copy(),
                opacities=self.target_scene.opacities.copy(),
                colors=self.target_scene.colors.copy(),
                sh_coeffs=self.target_scene.sh_coeffs.copy(),
            ),
            frames=[self.frame],
            adam_hparams=AdamHyperParams(position_lr=0.0, scale_lr=0.1, rotation_lr=0.0, color_lr=0.0, opacity_lr=0.0),
            stability_hparams=StabilityHyperParams(max_update=0.5, max_scale=16.0 * self.pixel_floor_scale),
            training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0),
            seed=123,
        )
        self.scene_groups = _read_scene_groups(self.renderer, 1)
        self.target_texture = self.trainer.get_frame_target_texture(0, native_resolution=True)
        self._last_target_scale = float("nan")

    def upload_target_scale(self, target_scale: float) -> None:
        if float(target_scale) == self._last_target_scale: return
        self.target_scene.scales[...] = _stored_from_support_scale(float(target_scale))
        rgb = np.clip(
            np.asarray(
                self.target_renderer.render(self.target_scene, self.camera, background=self.background).image,
                dtype=np.float32,
            )[..., :3],
            0.0,
            1.0,
        )
        rgba = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = (255.0 * rgb + 0.5).astype(np.uint8)
        rgba[..., 3] = 255
        self.target_texture.copy_from_numpy(np.ascontiguousarray(rgba, dtype=np.uint8))
        self.trainer._invalidate_downscaled_target()
        self._last_target_scale = float(target_scale)

    def upload_train_scale(self, scale: float) -> None:
        self.scene_groups["scales"][0, :3] = _stored_from_support_scale(float(scale))
        self.renderer.write_scene_groups(
            1,
            positions=self.scene_groups["positions"],
            scales=self.scene_groups["scales"],
            rotations=self.scene_groups["rotations"],
            color_alpha=self.scene_groups["color_alpha"],
        )

    def dispatch_gradient(self) -> None:
        self.renderer.execute_prepass_for_current_scene(self.camera, sync_counts=False)
        enc = self.device.create_command_encoder()
        self.trainer._dispatch_raster_training_forward(enc, self.camera, self.background)
        target_texture = self.trainer.get_frame_target_texture(0, native_resolution=False, encoder=enc)
        self.trainer._dispatch_loss_forward(enc, target_texture)
        self.trainer._dispatch_loss_backward(enc, target_texture)
        self.trainer._dispatch_raster_backward(enc, self.camera, self.background)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

    def read_scale_grad_mean(self) -> float:
        return float(np.mean(_read_grad_groups(self.renderer, 1)["grad_scales"][0, :3]))

    def measure_scale_grad_mean(self, scale: float, target_scale: float) -> float:
        self.upload_target_scale(target_scale)
        self.upload_train_scale(scale)
        self.dispatch_gradient()
        return self.read_scale_grad_mean()


def _scale_target_pairs(scale_muls: tuple[float, ...], target_muls: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    return tuple((scale_mul, target_mul) for target_mul in target_muls for scale_mul in scale_muls if scale_mul < target_mul)


def test_training_step_smoke_updates_params_without_changing_count(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    before = _read_scene_groups(renderer, scene.count)["positions"].copy()
    loss = trainer.step()
    after = _read_scene_groups(renderer, scene.count)["positions"]

    assert np.isfinite(loss)
    assert trainer.state.step == 1
    assert np.isfinite(trainer.state.last_mse)
    assert np.isfinite(trainer.state.avg_loss)
    assert trainer.scene.count == scene.count
    assert np.all(np.isfinite(after))
    assert np.any(np.abs(after - before) > 0.0)


def test_training_step_smoke_updates_params_in_fixed_atomic_mode(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path, image_name="fixed_mode_target.png", image_id=7)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32, cached_raster_grad_atomic_mode="fixed")
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=321)

    before = _read_scene_groups(renderer, scene.count)["positions"].copy()
    loss = trainer.step()
    after = _read_scene_groups(renderer, scene.count)["positions"]

    assert renderer.cached_raster_grad_atomic_mode == "fixed"
    assert np.isfinite(loss)
    assert np.all(np.isfinite(after))
    assert np.any(np.abs(after - before) > 0.0)


def test_training_step_smoke_with_subsampling_produces_finite_updates(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path, image_name="subsample_target.png", image_id=17)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=32)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=123,
    )

    before = _read_scene_groups(renderer, scene.count)["positions"].copy()
    loss = trainer.step()
    after = _read_scene_groups(renderer, scene.count)["positions"]

    assert np.isfinite(loss)
    assert trainer.state.step == 1
    assert np.isfinite(trainer.state.last_mse)
    assert np.all(np.isfinite(after))
    assert np.any(np.abs(after - before) > 0.0)


def test_random_training_background_seeds_are_deterministic(device, tmp_path: Path):
    scene = _make_scene(count=4, seed=17)
    frame = _make_frame(tmp_path, image_name="random_background_target.png", image_id=11)
    renderer_a = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    renderer_b = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer_a = GaussianTrainer(device=device, renderer=renderer_a, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(background_mode=TRAIN_BACKGROUND_MODE_RANDOM), seed=123)
    trainer_b = GaussianTrainer(device=device, renderer=renderer_b, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(background_mode=TRAIN_BACKGROUND_MODE_RANDOM), seed=123)

    assert trainer_a._training_background_seed(0) == trainer_b._training_background_seed(0)
    assert trainer_a._training_background_seed(1) == trainer_b._training_background_seed(1)
    assert trainer_a._training_background_seed(0) != trainer_a._training_background_seed(1)


def test_random_training_background_rasterizes_seeded_white_noise(device, tmp_path: Path):
    scene = _make_scene(count=1, seed=17)
    scene.opacities[:] = 0.0
    frame = _make_frame(tmp_path, width=24, height=16, image_name="random_background_noise_target.png", image_id=13)
    renderer = GaussianRenderer(device, width=24, height=16, list_capacity_multiplier=16)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(background_mode=TRAIN_BACKGROUND_MODE_RANDOM), seed=123)
    camera = frame.make_camera(near=0.1, far=20.0)
    background = trainer._training_background()

    def _render(step: int) -> np.ndarray:
        renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
        enc = device.create_command_encoder()
        trainer._dispatch_raster_training_forward(enc, camera, background, step=step, frame_index=0, native_camera=camera)
        device.submit_command_buffer(enc.finish())
        device.wait()
        return np.asarray(renderer.output_texture.to_numpy(), dtype=np.float32).copy()

    image0 = _render(0)
    image0_repeat = _render(0)
    image1 = _render(1)

    assert np.all(np.isfinite(image0))
    assert np.var(image0[..., :3]) > 0.0
    assert np.allclose(image0, image0_repeat)
    assert not np.allclose(image0, image1)


def test_custom_training_background_returns_configured_color(device, tmp_path: Path):
    scene = _make_scene(count=4, seed=18)
    frame = _make_frame(tmp_path, image_name="custom_background_target.png", image_id=12)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    color = (1.0, 1.0, 1.0)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(background_mode=TRAIN_BACKGROUND_MODE_CUSTOM, background=color),
        seed=123,
    )

    np.testing.assert_allclose(trainer._training_background(), np.asarray(color, dtype=np.float32), rtol=0.0, atol=0.0)


def test_training_step_batch_updates_params_without_changing_count(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path, image_name="batch_target.png", image_id=1)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    before = _read_scene_groups(renderer, scene.count)["positions"].copy()
    executed = trainer.step_batch(3)
    after = _read_scene_groups(renderer, scene.count)["positions"]

    assert executed == 3
    assert trainer.state.step == 3
    assert np.isfinite(trainer.state.last_loss)
    assert np.isfinite(trainer.state.last_mse)
    assert np.isfinite(trainer.state.avg_loss)
    assert trainer.scene.count == scene.count
    assert np.all(np.isfinite(after))
    assert np.any(np.abs(after - before) > 0.0)


def test_training_step_batch_matches_two_single_steps(device, tmp_path: Path):
    scene = _make_scene(count=8, seed=31)
    frame = _make_frame(tmp_path, image_name="batch_match_target.png", image_id=2)
    renderer_seq = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    renderer_batch = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer_seq = GaussianTrainer(device=device, renderer=renderer_seq, scene=scene, frames=[frame], seed=123)
    trainer_batch = GaussianTrainer(device=device, renderer=renderer_batch, scene=scene, frames=[frame], seed=123)

    trainer_seq.step()
    trainer_seq.step()
    executed = trainer_batch.step_batch(2)

    assert executed == 2
    np.testing.assert_allclose(
        _read_scene_groups(renderer_batch, scene.count)["positions"],
        _read_scene_groups(renderer_seq, scene.count)["positions"],
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        _read_scene_groups(renderer_batch, scene.count)["scales"],
        _read_scene_groups(renderer_seq, scene.count)["scales"],
        rtol=1e-5,
        atol=1e-7,
    )


def test_position_random_steps_move_low_opacity_splats(device, tmp_path: Path):
    scene = _make_scene(count=4, seed=41)
    scene.opacities[:] = np.full((scene.count,), 1e-4, dtype=np.float32)
    scene.scales[:] = _log_sigma(np.full((scene.count, 3), 0.1, dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="position_random_step_low_opacity.png", image_id=13)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            position_random_step_noise_lr=5e4,
        ),
        seed=123,
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    zero_sh = np.zeros((scene.count, 4, 3), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_sh_coeffs=zero_sh, grad_color_alpha=zeros)
    before = _read_scene_groups(renderer, scene.count)["positions"].copy()

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    trainer._dispatch_position_random_steps(enc, 1)
    device.submit_command_buffer(enc.finish())
    device.wait()

    after = _read_scene_groups(renderer, scene.count)["positions"]
    assert np.all(np.isfinite(after))
    assert np.any(np.abs(after - before) > 0.0)


def test_position_random_steps_are_gated_for_high_opacity_splats(device, tmp_path: Path):
    scene = _make_scene(count=4, seed=43)
    scene.opacities[:] = np.full((scene.count,), 0.5, dtype=np.float32)
    scene.scales[:] = _log_sigma(np.full((scene.count, 3), 0.1, dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="position_random_step_high_opacity.png", image_id=14)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            position_random_step_noise_lr=5e4,
        ),
        seed=123,
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    zero_sh = np.zeros((scene.count, 4, 3), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_sh_coeffs=zero_sh, grad_color_alpha=zeros)
    before = _read_scene_groups(renderer, scene.count)["positions"].copy()

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    trainer._dispatch_position_random_steps(enc, 1)
    device.submit_command_buffer(enc.finish())
    device.wait()

    after = _read_scene_groups(renderer, scene.count)["positions"]
    np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-9)


def test_position_random_steps_clamp_anisotropy_for_thin_splats(device, tmp_path: Path):
    scene = _make_scene(count=8, seed=45)
    scene.opacities[:] = np.full((scene.count,), 1e-4, dtype=np.float32)
    scene.scales[:] = _log_sigma(np.array([[1.0, 1e-3, 1e-3]], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="position_random_step_anisotropy_clamp.png", image_id=15)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            position_random_step_noise_lr=5e4,
        ),
        seed=123,
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    zero_sh = np.zeros((scene.count, 4, 3), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_sh_coeffs=zero_sh, grad_color_alpha=zeros)
    before = _read_scene_groups(renderer, scene.count)["positions"].copy()

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    trainer._dispatch_position_random_steps(enc, 1)
    device.submit_command_buffer(enc.finish())
    device.wait()

    after = _read_scene_groups(renderer, scene.count)["positions"]
    displacement = np.abs(after - before)
    assert float(np.mean(displacement[:, 1:3])) > 0.1


def test_split_loss_forward_backward_separates_metrics_from_output_grads(device, tmp_path: Path):
    scene = _make_scene(count=8, seed=19)
    frame = _make_frame(tmp_path, image_name="split_loss_target.png")
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=17)
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer.training.background, dtype=np.float32)

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    renderer.output_grad_buffer.copy_from_numpy(np.zeros((renderer.width * renderer.height, 4), dtype=np.float32))
    enc = device.create_command_encoder()
    trainer._dispatch_raster_training_forward(enc, camera, background)
    target_texture = trainer.get_frame_target_texture(0, native_resolution=False, encoder=enc)
    trainer._dispatch_loss_forward(enc, target_texture)
    device.submit_command_buffer(enc.finish())
    device.wait()

    loss, mse, density_loss = trainer._read_loss_metrics()
    grads_after_forward = _read_output_grads(renderer).copy()

    enc = device.create_command_encoder()
    trainer._dispatch_loss_backward(enc, target_texture)
    device.submit_command_buffer(enc.finish())
    device.wait()

    grads_after_backward = _read_output_grads(renderer)
    np.testing.assert_allclose(trainer._read_loss_metrics(), (loss, mse, density_loss), rtol=0.0, atol=0.0)
    assert np.isfinite(loss)
    assert np.isfinite(mse)
    assert density_loss == 0.0
    assert np.allclose(grads_after_forward, 0.0)
    assert np.any(np.abs(grads_after_backward[..., :3]) > 0.0)
    assert np.all(np.isfinite(grads_after_backward))


def test_ssim_weight_zero_matches_l1_metrics_and_gradients(device, tmp_path: Path):
    trainer = _make_loss_only_trainer(
        device,
        tmp_path,
        width=4,
        height=3,
        training_hparams=TrainingHyperParams(
            background_mode=TRAIN_BACKGROUND_MODE_CUSTOM,
            background=(0.0, 0.0, 0.0),
            density_regularizer=0.0,
            depth_ratio_weight=0.0,
            ssim_weight=0.0,
        ),
        image_name="ssim_weight_zero_target.png",
        image_id=31,
    )
    rendered = np.array(
        [
            [[0.1, 0.2, 0.3, 1.0], [0.7, 0.2, 0.4, 1.0], [0.9, 0.6, 0.2, 1.0], [0.0, 0.1, 0.2, 1.0]],
            [[0.3, 0.8, 0.4, 1.0], [0.4, 0.3, 0.9, 1.0], [0.5, 0.5, 0.5, 1.0], [0.2, 0.9, 0.1, 1.0]],
            [[0.8, 0.1, 0.7, 1.0], [0.6, 0.4, 0.2, 1.0], [0.1, 0.7, 0.8, 1.0], [0.2, 0.2, 0.9, 1.0]],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [[0.2, 0.1, 0.4, 1.0], [0.6, 0.4, 0.3, 1.0], [0.7, 0.6, 0.1, 1.0], [0.1, 0.2, 0.0, 1.0]],
            [[0.4, 0.7, 0.2, 1.0], [0.2, 0.5, 0.8, 1.0], [0.3, 0.6, 0.4, 1.0], [0.5, 0.7, 0.2, 1.0]],
            [[0.6, 0.2, 0.9, 1.0], [0.5, 0.2, 0.3, 1.0], [0.3, 0.6, 0.7, 1.0], [0.1, 0.4, 0.8, 1.0]],
        ],
        dtype=np.float32,
    )

    _dispatch_manual_loss(trainer, rendered, target)

    expected_rgb_loss, expected_total, expected_mse, expected_grad = _blended_rgb_metrics_np(
        rendered,
        target,
        ssim_weight=0.0,
        use_target_alpha_mask=False,
    )
    total, mse, density = trainer._read_loss_metrics()
    np.testing.assert_allclose(_read_training_rgb_loss(trainer.renderer), expected_rgb_loss, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(_read_training_rgb_loss_total(trainer.renderer), expected_total, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(total, expected_total, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(mse, expected_mse, rtol=0.0, atol=1e-7)
    assert density == 0.0
    np.testing.assert_allclose(_read_output_grads(trainer.renderer)[..., :3], expected_grad, rtol=0.0, atol=1e-7)


def test_ssim_feature_blur_matches_cpu_reference(device, tmp_path: Path):
    trainer = _make_loss_only_trainer(
        device,
        tmp_path,
        width=5,
        height=4,
        training_hparams=TrainingHyperParams(
            background_mode=TRAIN_BACKGROUND_MODE_CUSTOM,
            background=(0.0, 0.0, 0.0),
            density_regularizer=0.0,
            depth_ratio_weight=0.0,
            ssim_weight=1.0,
        ),
        image_name="ssim_feature_blur_target.png",
        image_id=32,
    )
    rendered = np.linspace(0.05, 0.95, num=5 * 4 * 4, dtype=np.float32).reshape(4, 5, 4)
    target = np.linspace(0.95, 0.05, num=5 * 4 * 4, dtype=np.float32).reshape(4, 5, 4)
    rendered[..., 3] = 1.0
    target[..., 3] = 1.0

    trainer.renderer.output_texture.copy_from_numpy(np.ascontiguousarray(rendered, dtype=np.float32))
    target_texture = _make_float_texture(device, target)
    enc = device.create_command_encoder()
    trainer._dispatch_ssim_feature_extraction(enc, target_texture)
    trainer._dispatch_ssim_blur(enc, "ssim_moments", "ssim_blurred_moments")
    device.submit_command_buffer(enc.finish())
    device.wait()

    expected_moments = _ssim_feature_moments_np(rendered[..., :3], target[..., :3])
    expected_blurred = _separable_gaussian_blur_np(expected_moments)
    np.testing.assert_allclose(_read_ssim_buffer(trainer, "ssim_moments"), expected_moments, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(_read_ssim_buffer(trainer, "ssim_blurred_moments"), expected_blurred, rtol=0.0, atol=2e-6)


def test_identical_images_zero_blended_ssim_loss(device, tmp_path: Path):
    trainer = _make_loss_only_trainer(
        device,
        tmp_path,
        width=5,
        height=5,
        training_hparams=TrainingHyperParams(
            background_mode=TRAIN_BACKGROUND_MODE_CUSTOM,
            background=(0.0, 0.0, 0.0),
            density_regularizer=0.0,
            depth_ratio_weight=0.0,
            ssim_weight=1.0,
        ),
        image_name="ssim_identical_target.png",
        image_id=33,
    )
    image = np.linspace(0.0, 1.0, num=5 * 5 * 4, dtype=np.float32).reshape(5, 5, 4)
    image[..., 3] = 1.0

    _dispatch_manual_loss(trainer, image, image)

    total, mse, density = trainer._read_loss_metrics()
    np.testing.assert_allclose(total, 0.0, rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(mse, 0.0, rtol=0.0, atol=5e-6)
    assert density == 0.0
    np.testing.assert_allclose(_read_training_rgb_loss_total(trainer.renderer), 0.0, rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(_read_output_grads(trainer.renderer)[..., :3], 0.0, rtol=0.0, atol=2e-5)


def test_ssim_backward_matches_torch_image_gradients(device, tmp_path: Path):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    rng = np.random.default_rng(1234)
    trainer = _make_loss_only_trainer(
        device,
        tmp_path,
        width=12,
        height=12,
        training_hparams=TrainingHyperParams(
            background_mode=TRAIN_BACKGROUND_MODE_CUSTOM,
            background=(0.0, 0.0, 0.0),
            density_regularizer=0.0,
            depth_ratio_weight=0.0,
            ssim_weight=0.2,
        ),
        image_name="ssim_torch_grad_target.png",
        image_id=34,
    )
    rendered = rng.uniform(0.05, 0.95, size=(12, 12, 4)).astype(np.float32)
    target = rng.uniform(0.05, 0.95, size=(12, 12, 4)).astype(np.float32)
    rendered[..., 3] = 1.0
    target[..., 3] = 1.0

    _dispatch_manual_loss(trainer, rendered, target)
    gpu_grad = _read_output_grads(trainer.renderer)[..., :3]

    kernel = torch.tensor(_SSIM_BLUR_WEIGHTS, dtype=torch.float32)
    weight_h = kernel.view(1, 1, 1, -1).repeat(_SSIM_FEATURE_CHANNELS, 1, 1, 1)
    weight_v = kernel.view(1, 1, -1, 1).repeat(_SSIM_FEATURE_CHANNELS, 1, 1, 1)
    rendered_t = torch.tensor(rendered[..., :3].transpose(2, 0, 1)[None], dtype=torch.float32, requires_grad=True)
    target_t = torch.tensor(target[..., :4].transpose(2, 0, 1)[None], dtype=torch.float32)
    target_rgb = target_t[:, :3]
    target_mask = target_t[:, 3:4].clamp(0.0, 1.0)
    inv_pixel_count = 1.0 / float(rendered.shape[0] * rendered.shape[1])

    def rgb_to_ycbcr_bt601_torch(value):
        y = (value * torch.tensor([0.299, 0.587, 0.114], dtype=value.dtype).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        cb = (value * torch.tensor([-0.168736, -0.331264, 0.5], dtype=value.dtype).view(1, 3, 1, 1)).sum(dim=1, keepdim=True) + 0.5
        cr = (value * torch.tensor([0.5, -0.418688, -0.081312], dtype=value.dtype).view(1, 3, 1, 1)).sum(dim=1, keepdim=True) + 0.5
        return torch.cat((y, cb, cr), dim=1)

    rendered_ycc = rgb_to_ycbcr_bt601_torch(rendered_t)
    target_ycc = rgb_to_ycbcr_bt601_torch(target_rgb)
    moments = torch.cat((rendered_ycc, target_ycc, rendered_ycc * rendered_ycc, target_ycc * target_ycc, rendered_ycc * target_ycc), dim=1)
    blurred = F.conv2d(F.pad(moments, (5, 5, 0, 0), mode="replicate"), weight_h, groups=_SSIM_FEATURE_CHANNELS)
    blurred = F.conv2d(F.pad(blurred, (0, 0, 5, 5), mode="replicate"), weight_v, groups=_SSIM_FEATURE_CHANNELS)
    x, y, x2, y2, xy = torch.split(blurred, 3, dim=1)
    sigma_x2 = torch.clamp(x2 - x * x, min=0.0)
    sigma_y2 = torch.clamp(y2 - y * y, min=0.0)
    sigma_xy = xy - x * y
    numer = (2.0 * x * y + _SSIM_C1) * (2.0 * sigma_xy + _SSIM_C2)
    denom = torch.clamp((x * x + y * y + _SSIM_C1) * (sigma_x2 + sigma_y2 + _SSIM_C2), min=_SSIM_SMALL_VALUE)
    ssim = numer / denom
    dssim = 0.5 * (1.0 - (ssim * torch.tensor(_SSIM_YCBCR_WEIGHTS, dtype=torch.float32).view(1, 3, 1, 1)).sum(dim=1, keepdim=True))
    l1 = (rendered_t - target_rgb).abs().mean(dim=1, keepdim=True)
    loss = ((((1.0 - trainer.training.ssim_weight) * l1) + (trainer.training.ssim_weight * dssim)) * inv_pixel_count * target_mask).sum()
    loss.backward()
    torch_grad = rendered_t.grad.detach().cpu().numpy()[0].transpose(1, 2, 0)

    np.testing.assert_allclose(gpu_grad, torch_grad, rtol=0.0, atol=1.25e-3)


def test_target_alpha_mask_skips_masked_pixel_loss_and_output_grads(device, tmp_path: Path):
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[..., 1] = 255
    frame = _make_rgba_frame(tmp_path, image, image_name="alpha_mask_target.png", image_id=21)
    scene = _make_scene(count=1, seed=111)
    scene.opacities[:] = 0.0
    scene.colors[:] = 0.0
    renderer_off = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    renderer_on = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer_off = GaussianTrainer(
        device=device,
        renderer=renderer_off,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            background_mode=TRAIN_BACKGROUND_MODE_CUSTOM,
            background=(0.0, 0.0, 0.0),
            density_regularizer=0.0,
            depth_ratio_weight=0.0,
            use_target_alpha_mask=False,
        ),
        seed=123,
    )
    trainer_on = GaussianTrainer(
        device=device,
        renderer=renderer_on,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            background_mode=TRAIN_BACKGROUND_MODE_CUSTOM,
            background=(0.0, 0.0, 0.0),
            density_regularizer=0.0,
            depth_ratio_weight=0.0,
            use_target_alpha_mask=True,
        ),
        seed=123,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.zeros((3,), dtype=np.float32)

    def run_pass(trainer: GaussianTrainer, renderer: GaussianRenderer) -> tuple[tuple[float, float, float], np.ndarray]:
        renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
        enc = device.create_command_encoder()
        trainer._dispatch_raster_training_forward(enc, camera, background)
        target_texture = trainer.get_frame_target_texture(0, native_resolution=False, encoder=enc)
        trainer._dispatch_loss_forward(enc, target_texture)
        trainer._dispatch_loss_backward(enc, target_texture)
        device.submit_command_buffer(enc.finish())
        device.wait()
        return trainer._read_loss_metrics(), _read_output_grads(renderer).copy()

    (loss_off, mse_off, density_off), grads_off = run_pass(trainer_off, renderer_off)
    (loss_on, mse_on, density_on), grads_on = run_pass(trainer_on, renderer_on)

    assert loss_off > 0.0
    assert mse_off > 0.0
    assert density_off == 0.0
    assert np.any(np.abs(grads_off[..., :3]) > 0.0)
    np.testing.assert_allclose((loss_on, mse_on, density_on), (0.0, 0.0, 0.0), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(grads_on[..., :3], 0.0, rtol=0.0, atol=1e-7)


def test_split_raster_backward_consumes_forward_cache_only(device, tmp_path: Path):
    scene = _make_scene(count=10, seed=23)
    frame = _make_frame(tmp_path, image_name="split_raster_target.png")
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=29)
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer.training.background, dtype=np.float32)

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    target_texture = trainer.get_frame_target_texture(0, native_resolution=False)

    enc = device.create_command_encoder()
    trainer._dispatch_raster_training_forward(enc, camera, background)
    trainer._dispatch_loss_backward(enc, target_texture)
    device.submit_command_buffer(enc.finish())
    device.wait()

    grads_after_forward = _read_grad_groups(renderer, scene.count)
    for name in grads_after_forward:
        assert np.allclose(grads_after_forward[name], 0.0)

    enc = device.create_command_encoder()
    trainer._dispatch_raster_backward(enc, camera, background)
    device.submit_command_buffer(enc.finish())
    device.wait()

    grads_after_backward = _read_grad_groups(renderer, scene.count)
    assert any(np.any(np.abs(grads_after_backward[name]) > 0.0) for name in grads_after_backward)
    for name in grads_after_backward:
        assert np.all(np.isfinite(grads_after_backward[name]))


def test_scale_gradient_stays_growth_directed_for_large_target_scales(device, tmp_path: Path):
    probe = _ScaleGradProbe(device, tmp_path, image_name="tiny_grad_target.png")
    max_growth_eps = 4e-6
    pairs = _scale_target_pairs(_SCALE_GRAD_MULS, _TARGET_MULS)
    grad_samples = np.asarray(
        [
            probe.measure_scale_grad_mean(scale_mul * probe.pixel_floor_scale, target_mul * probe.pixel_floor_scale)
            for scale_mul, target_mul in pairs
        ],
        dtype=np.float32,
    )

    assert np.all(np.isfinite(grad_samples))
    assert np.all(grad_samples <= max_growth_eps)


def test_training_targets_use_srgb_textures(device, tmp_path: Path):
    scene = _make_scene()
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    native_target = trainer.get_frame_target_texture(0, native_resolution=True)
    train_target = trainer.get_frame_target_texture(0, native_resolution=False)

    assert native_target.format == spy.Format.rgba8_unorm_srgb
    assert train_target.format == spy.Format.rgba32_float
    assert native_target is not train_target


def test_resolve_training_resolution_uses_ceil_division() -> None:
    assert resolve_training_resolution(64, 32, 1) == (64, 32)
    assert resolve_training_resolution(63, 65, 2) == (32, 33)
    assert resolve_training_resolution(1, 1, 16) == (1, 1)


def test_effective_train_render_factor_multiplies_downscale_and_subsample() -> None:
    hparams = TrainingHyperParams(train_downscale_factor=2, train_subsample_factor=3)

    assert resolve_train_subsample_factor(hparams) == 3
    assert resolve_effective_train_render_factor(hparams, 0) == 6


def test_auto_train_subsample_targets_nearest_strictly_above_1k_max_side() -> None:
    hparams = TrainingHyperParams(train_subsample_factor=0, train_downscale_mode=1)

    assert resolve_auto_train_subsample_factor(640, 360, 1) == 1
    assert resolve_auto_train_subsample_factor(2048, 1024, 1) == 2
    assert resolve_auto_train_subsample_factor(3000, 1600, 1) == 2
    assert resolve_auto_train_subsample_factor(4000, 2000, 1) == 3
    assert resolve_auto_train_subsample_factor(1000, 800, 1) == 1
    assert resolve_train_subsample_factor(hparams, 2048, 1024, 0) == 2
    assert resolve_effective_train_render_factor(hparams, 0, 2048, 1024) == 2


def test_base_lr_uses_requested_piecewise_schedule() -> None:
    hparams = TrainingHyperParams(lr_schedule_start_lr=0.005, lr_schedule_end_lr=1.5e-4, lr_schedule_steps=30_000)

    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 0), 0.005, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 1500), 0.0035, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 3000), 0.002, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 8500), 0.0015, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 14000), 0.001, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 30_000), 1.5e-4, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 40_000), 1.5e-4, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_cosine_base_learning_rate(hparams, 8250), resolve_base_learning_rate(hparams, 8250), rtol=0.0, atol=1e-12)
    assert resolve_lr_schedule_breakpoints(hparams) == (3000, 14000, 30_000)


def test_piecewise_schedule_uses_configured_viewer_breakpoints() -> None:
    hparams = TrainingHyperParams(
        lr_schedule_start_lr=0.005,
        lr_schedule_stage1_lr=0.002,
        lr_schedule_stage2_lr=0.001,
        lr_schedule_end_lr=1e-4,
        lr_schedule_steps=20_000,
        lr_schedule_stage1_step=1000,
        lr_schedule_stage2_step=4000,
        lr_pos_mul=1.5,
        lr_pos_stage1_mul=1.25,
        lr_pos_stage2_mul=0.75,
        lr_pos_stage3_mul=0.5,
        lr_sh_mul=1.1,
        lr_sh_stage1_mul=0.9,
        lr_sh_stage2_mul=0.6,
        lr_sh_stage3_mul=0.3,
        depth_ratio_weight=1.0,
        depth_ratio_stage1_weight=0.25,
        depth_ratio_stage2_weight=0.05,
        depth_ratio_stage3_weight=0.01,
        position_random_step_noise_lr=5e5,
        position_random_step_noise_stage1_lr=250000.0,
        position_random_step_noise_stage2_lr=100000.0,
        position_random_step_noise_stage3_lr=0.0,
        sh_band_stage1=0,
        sh_band_stage2=2,
        sh_band_stage3=3,
    )

    assert resolve_lr_schedule_breakpoints(hparams) == (1000, 4000, 20_000)
    assert resolve_stage_schedule_steps(hparams) == (1000, 4000, 20_000)
    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 1000), 0.002, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 4000), 0.001, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_position_lr_mul(hparams, 0), 1.5, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_lr_mul(hparams, 1000), 1.25, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_lr_mul(hparams, 4000), 0.75, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_lr_mul(hparams, 20_000), 0.5, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_sh_lr_mul(hparams, 0), 1.1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_sh_lr_mul(hparams, 1000), 0.9, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_sh_lr_mul(hparams, 4000), 0.6, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_sh_lr_mul(hparams, 20_000), 0.3, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_depth_ratio_weight(hparams, 1000), 0.25, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_depth_ratio_weight(hparams, 4000), 0.05, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_depth_ratio_weight(hparams, 20_000), 0.01, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 1000), 250000.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 4000), 100000.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 20_000), 0.0, rtol=0.0, atol=1e-6)
    assert resolve_sh_band(hparams, 0) == 0
    assert resolve_sh_band(hparams, 999) == 0
    assert resolve_sh_band(hparams, 1000) == 0
    assert resolve_sh_band(hparams, 3999) == 0
    assert resolve_sh_band(hparams, 4000) == 2
    assert resolve_use_sh(hparams, 0) is False
    assert resolve_use_sh(hparams, 999) is False
    assert resolve_use_sh(hparams, 1000) is False
    assert resolve_use_sh(hparams, 3999) is False
    assert resolve_use_sh(hparams, 4000) is True


def test_training_step_updates_optimizer_lrs_from_piecewise_schedule(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=77)
    frame = _make_frame(tmp_path, image_name="lr_schedule_target.png", image_id=5)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    adam = AdamHyperParams(position_lr=1e-2, scale_lr=2e-2, rotation_lr=3e-2, color_lr=4e-2, opacity_lr=5e-2)
    training = TrainingHyperParams(
        lr_schedule_start_lr=0.005,
        lr_schedule_end_lr=1e-3,
        lr_schedule_steps=30_000,
        scale_l2_weight=0.0,
        scale_abs_reg_weight=0.0,
        opacity_reg_weight=0.0,
    )
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], adam_hparams=adam, training_hparams=training, seed=123)

    before = _read_optimizer_lrs(trainer)
    trainer.step()
    after = _read_optimizer_lrs(trainer)
    expected_scale = resolve_base_learning_rate(training, 1) / training.lr_schedule_start_lr

    param_ids = np.array([0, 3, 6, 10, trainer.renderer.PARAM_RAW_OPACITY_ID], dtype=np.int32)
    np.testing.assert_allclose(before[param_ids], np.array([1e-2, 2e-2, 3e-2, 4e-2, 5e-2], dtype=np.float32), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(after[param_ids], expected_scale * np.array([1e-2, 2e-2, 3e-2, 4e-2, 5e-2], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(trainer.state.last_base_lr, resolve_base_learning_rate(training, 1), rtol=0.0, atol=1e-7)


def test_training_step_updates_position_lr_from_staged_position_multiplier(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=78)
    frame = _make_frame(tmp_path, image_name="lr_pos_schedule_target.png", image_id=6)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    adam = AdamHyperParams(position_lr=1e-2, scale_lr=2e-2, rotation_lr=3e-2, color_lr=4e-2, opacity_lr=5e-2)
    training = TrainingHyperParams(
        lr_schedule_start_lr=0.005,
        lr_schedule_stage1_step=1,
        lr_schedule_stage2_step=2,
        lr_schedule_steps=3,
        lr_pos_mul=1.0,
        lr_pos_stage1_mul=0.5,
        lr_pos_stage2_mul=0.25,
        lr_pos_stage3_mul=0.125,
        scale_l2_weight=0.0,
        scale_abs_reg_weight=0.0,
        opacity_reg_weight=0.0,
    )
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], adam_hparams=adam, training_hparams=training, seed=123)

    trainer.step()
    lrs_after_step_1 = _read_optimizer_lrs(trainer)
    expected_scale = resolve_base_learning_rate(training, 1) / training.lr_schedule_start_lr

    np.testing.assert_allclose(lrs_after_step_1[0], 1e-2 * expected_scale * 0.5, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(lrs_after_step_1[3], 2e-2 * expected_scale, rtol=0.0, atol=1e-6)


def test_training_step_updates_non_dc_sh_lr_without_affecting_dc(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=79)
    frame = _make_frame(tmp_path, image_name="lr_sh_schedule_target.png", image_id=16)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    adam = AdamHyperParams(position_lr=1e-2, scale_lr=2e-2, rotation_lr=3e-2, color_lr=4e-2, opacity_lr=5e-2)
    training = TrainingHyperParams(
        lr_schedule_start_lr=0.005,
        lr_schedule_stage1_step=1,
        lr_schedule_stage2_step=2,
        lr_schedule_steps=3,
        lr_sh_mul=1.0,
        lr_sh_stage1_mul=0.25,
        lr_sh_stage2_mul=0.125,
        lr_sh_stage3_mul=0.0625,
        scale_l2_weight=0.0,
        scale_abs_reg_weight=0.0,
        opacity_reg_weight=0.0,
    )
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], adam_hparams=adam, training_hparams=training, seed=123)

    trainer.step()
    lrs_after_step_1 = _read_optimizer_lrs(trainer)
    expected_scale = resolve_base_learning_rate(training, 1) / training.lr_schedule_start_lr
    sh_dc_id = trainer.renderer.PARAM_SH0_IDS[0]
    sh_non_dc_id = trainer.renderer.PARAM_SH_COEFF_IDS[1][0]

    np.testing.assert_allclose(lrs_after_step_1[sh_dc_id], 4e-2 * expected_scale, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(lrs_after_step_1[sh_non_dc_id], 4e-2 * expected_scale * 0.25, rtol=0.0, atol=1e-6)


def test_sh_lr_multiplier_scales_non_dc_coeffs_absolutely(device, tmp_path: Path) -> None:
    scene = _make_scene(count=4, seed=80)
    frame = _make_frame(tmp_path, image_name="lr_sh_absolute_target.png", image_id=17)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    adam = AdamHyperParams(position_lr=1e-2, scale_lr=2e-2, rotation_lr=3e-2, color_lr=4e-2, opacity_lr=5e-2)
    training = TrainingHyperParams(
        lr_schedule_enabled=False,
        lr_schedule_start_lr=0.005,
        lr_sh_mul=0.2,
        scale_l2_weight=0.0,
        scale_abs_reg_weight=0.0,
        opacity_reg_weight=0.0,
    )
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], adam_hparams=adam, training_hparams=training, seed=123)

    trainer.optimizer.update_step(0, training)
    lrs = _read_optimizer_lrs(trainer)
    sh_dc_id = trainer.renderer.PARAM_SH0_IDS[0]
    sh_non_dc_id = trainer.renderer.PARAM_SH_COEFF_IDS[1][0]

    np.testing.assert_allclose(lrs[sh_dc_id], 4e-2, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(lrs[sh_non_dc_id], 4e-2 * 0.2, rtol=0.0, atol=1e-7)


def test_loss_vars_use_density_schedule(device, tmp_path: Path) -> None:
    scene = _make_scene(count=4, seed=93)
    frame = _make_frame(tmp_path, image_name="density_schedule_target.png", image_id=19)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    training = TrainingHyperParams(lr_schedule_steps=30_000)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=training, seed=123)

    np.testing.assert_allclose(trainer._loss_vars(0, 0)["g_DensityRegularizer"], 0.02, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(trainer._loss_vars(0, 0)["g_DepthRatioWeight"], 1.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(trainer._loss_vars(0, 0)["g_DepthRatioGradMin"], 0.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(trainer._loss_vars(0, 0)["g_DepthRatioGradMax"], 0.1, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(trainer._loss_vars(0, 0)["g_MaxAllowedDensity"], 5.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(trainer._loss_vars(0, 15_000)["g_MaxAllowedDensity"], 8.5, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(trainer._loss_vars(0, 30_000)["g_MaxAllowedDensity"], 12.0, rtol=0.0, atol=1e-10)


def test_depth_ratio_noise_and_sh_schedules_follow_requested_defaults() -> None:
    hparams = TrainingHyperParams(
        depth_ratio_weight=1.0,
        lr_pos_mul=1.0,
        position_random_step_noise_lr=5e5,
        sh_band=1,
        lr_schedule_steps=30_000,
    )

    np.testing.assert_allclose(resolve_position_lr_mul(hparams, 0), 1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_lr_mul(hparams, 3000), 0.75, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_lr_mul(hparams, 14000), 0.2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_lr_mul(hparams, 30_000), 0.2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_sh_lr_mul(hparams, 0), 0.05, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_sh_lr_mul(hparams, 3000), 0.05, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_sh_lr_mul(hparams, 14000), 0.05, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_sh_lr_mul(hparams, 30_000), 0.05, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_depth_ratio_weight(hparams, 0), 1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_depth_ratio_weight(hparams, 3000), 0.05, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_depth_ratio_weight(hparams, 14000), 0.01, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_depth_ratio_weight(hparams, 30_000), 0.001, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 0), 5e5, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 3000), 466666.6666666667, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 14000), 416666.6666666667, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 30_000), 0.0, rtol=0.0, atol=1e-6)
    assert resolve_sh_band(hparams, 0) == 1
    assert resolve_sh_band(hparams, 13999) == 1
    assert resolve_sh_band(hparams, 14000) == 1
    assert resolve_use_sh(hparams, 13999) is True
    assert resolve_use_sh(hparams, 14000) is True


def test_schedule_disabled_uses_stage0_only_for_scheduled_values() -> None:
    hparams = TrainingHyperParams(
        lr_schedule_enabled=False,
        lr_schedule_start_lr=0.006,
        lr_schedule_stage1_lr=0.002,
        lr_schedule_stage2_lr=0.001,
        lr_schedule_end_lr=1.5e-4,
        lr_pos_mul=1.5,
        lr_pos_stage1_mul=1.25,
        lr_pos_stage2_mul=0.75,
        lr_pos_stage3_mul=0.5,
        lr_sh_mul=1.3,
        lr_sh_stage1_mul=0.9,
        lr_sh_stage2_mul=0.6,
        lr_sh_stage3_mul=0.4,
        depth_ratio_weight=0.8,
        depth_ratio_stage1_weight=0.05,
        depth_ratio_stage2_weight=0.01,
        depth_ratio_stage3_weight=0.001,
        position_random_step_noise_lr=1234.0,
        position_random_step_noise_stage1_lr=250.0,
        position_random_step_noise_stage2_lr=100.0,
        position_random_step_noise_stage3_lr=0.0,
        sh_band=2,
        sh_band_stage1=0,
        sh_band_stage2=1,
        sh_band_stage3=0,
    )

    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 0), 0.006, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_base_learning_rate(hparams, 30_000), 0.006, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_lr_mul(hparams, 0), 1.5, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_lr_mul(hparams, 30_000), 1.5, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_sh_lr_mul(hparams, 0), 1.3, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_sh_lr_mul(hparams, 30_000), 1.3, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_depth_ratio_weight(hparams, 0), 0.8, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_depth_ratio_weight(hparams, 30_000), 0.8, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 0), 1234.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 30_000), 1234.0, rtol=0.0, atol=1e-12)
    assert resolve_sh_band(hparams, 0) == 2
    assert resolve_sh_band(hparams, 30_000) == 2
    assert resolve_use_sh(hparams, 0) is True
    assert resolve_use_sh(hparams, 30_000) is True


def test_max_allowed_density_schedule_clamps_to_end_value() -> None:
    hparams = TrainingHyperParams(max_allowed_density_start=5.0, max_allowed_density=12.0, lr_schedule_steps=4)

    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 0), 5.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 2), 8.5, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 4), 12.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 40), 12.0, rtol=0.0, atol=1e-10)


def test_refinement_cadence_and_clone_probability_follow_growth_budget() -> None:
    hparams = TrainingHyperParams(refinement_interval=200, refinement_growth_ratio=0.05, refinement_growth_start_step=0)

    assert not should_run_refinement_step(hparams, 199)
    assert should_run_refinement_step(hparams, 200)
    np.testing.assert_allclose(
        resolve_clone_probability_threshold(hparams, splat_count=1000, pixel_count=64 * 32, step=200),
        1000.0 * 0.05 / 200.0 / float(64 * 32),
        rtol=0.0,
        atol=1e-12,
    )


def test_refinement_interval_is_floored_by_frame_count() -> None:
    hparams = TrainingHyperParams(refinement_interval=2, refinement_growth_ratio=0.05, refinement_growth_start_step=0)

    assert resolve_effective_refinement_interval(hparams, frame_count=5) == 5
    assert not should_run_refinement_step(hparams, 4, frame_count=5)
    assert should_run_refinement_step(hparams, 5, frame_count=5)
    np.testing.assert_allclose(
        resolve_clone_probability_threshold(hparams, splat_count=1000, pixel_count=64 * 32, step=5, frame_count=5),
        1000.0 * 0.05 / 5.0 / float(64 * 32),
        rtol=0.0,
        atol=1e-12,
    )


def test_refinement_growth_stays_zero_until_start_step() -> None:
    hparams = TrainingHyperParams(refinement_interval=200, refinement_growth_ratio=0.02, refinement_growth_start_step=2000)

    np.testing.assert_allclose(resolve_refinement_growth_ratio(hparams, 1999), 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_clone_probability_threshold(hparams, splat_count=1000, pixel_count=100, step=1999), 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_refinement_growth_ratio(hparams, 2000), 0.02, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_clone_probability_threshold(hparams, splat_count=1000, pixel_count=100, step=2000), 1000.0 * 0.02 / 200.0 / 100.0, rtol=0.0, atol=1e-12)


def test_refinement_min_contribution_percent_uses_configured_decay() -> None:
    hparams = TrainingHyperParams(refinement_interval=200, refinement_min_contribution_percent=1e-05, refinement_min_contribution_decay=0.95)

    np.testing.assert_allclose(resolve_refinement_min_contribution_percent(hparams, 0), 1e-05, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_refinement_min_contribution_percent(hparams, 199), 1e-05, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_refinement_min_contribution_percent(hparams, 200), 9.5e-06, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_refinement_min_contribution_percent(hparams, 400), 9.025e-06, rtol=0.0, atol=1e-12)


def test_clone_probability_threshold_respects_max_gaussians_cap() -> None:
    hparams = TrainingHyperParams(refinement_interval=200, refinement_growth_ratio=0.05, refinement_growth_start_step=0, max_gaussians=1024)

    np.testing.assert_allclose(
        resolve_clone_probability_threshold(hparams, splat_count=1000, pixel_count=100, step=200),
        24.0 / 200.0 / 100.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert resolve_clone_probability_threshold(hparams, splat_count=1024, pixel_count=100, step=200) == 0.0


def test_trainer_allocates_minimal_refinement_buffers(device, tmp_path: Path) -> None:
    scene = _make_scene(count=32, seed=81)
    frame = _make_frame(tmp_path, image_name="refinement_buffers_target.png", image_id=9)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_growth_ratio=0.05, refinement_growth_start_step=0),
        seed=123,
    )

    assert set(trainer.refinement_buffers) == {"total_clone_counter", "clone_counts", "splat_contribution", "append_counter", "append_params", "dst_splat_params", "dst_adam_moments", "camera_rows"}
    assert trainer.clone_probability_threshold() > 0.0


def test_trainer_refinement_due_uses_dataset_frame_floor(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=82)
    frames = [
        _make_frame(tmp_path, image_name="refinement_floor_a.png", image_id=90),
        _make_frame(tmp_path, image_name="refinement_floor_b.png", image_id=91),
        _make_frame(tmp_path, image_name="refinement_floor_c.png", image_id=92),
    ]
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=frames,
        training_hparams=TrainingHyperParams(refinement_interval=1, refinement_growth_ratio=0.05, refinement_growth_start_step=0),
        seed=123,
    )

    assert trainer.effective_refinement_interval() == 3
    assert not trainer.refinement_due(1)
    assert not trainer.refinement_due(2)
    assert trainer.refinement_due(3)


def test_training_forward_keeps_clone_counts_zero_until_backward(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=83)
    frame = _make_frame(tmp_path, image_name="clone_counts_target.png", image_id=10)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_growth_ratio=0.05, refinement_growth_start_step=0, refinement_interval=9999),
        seed=123,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer.training.background, dtype=np.float32)

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    enc = device.create_command_encoder()
    renderer.rasterize_training_forward_current_scene(
        enc,
        camera,
        background,
        clone_counts_buffer=trainer.refinement_buffers["clone_counts"],
        splat_contribution_buffer=trainer.refinement_buffers["splat_contribution"],
        clone_select_probability=1.0,
        clone_seed=123,
    )
    device.submit_command_buffer(enc.finish())
    device.wait()

    clone_counts = buffer_to_numpy(trainer.refinement_buffers["clone_counts"], np.uint32)[: scene.count]
    contributions = buffer_to_numpy(trainer.refinement_buffers["splat_contribution"], np.uint32)[: scene.count]
    assert np.all(clone_counts == 0)
    assert np.any(contributions > 0)

    enc = device.create_command_encoder()
    target_texture = trainer.get_frame_target_texture(0, native_resolution=False, encoder=enc)
    trainer._dispatch_loss_forward(enc, target_texture)
    trainer._dispatch_loss_backward(enc, target_texture)
    renderer.clear_raster_grads_current_scene(enc)
    renderer.rasterize_backward_current_scene(
        enc,
        camera,
        background,
        renderer.output_grad_buffer,
        regularizer_grad=renderer.work_buffers["training_regularizer_grad"],
        clone_counts_buffer=trainer.refinement_buffers["clone_counts"],
        clone_select_probability=1.0,
        clone_seed=123,
    )
    device.submit_command_buffer(enc.finish())
    device.wait()

    clone_counts = buffer_to_numpy(trainer.refinement_buffers["clone_counts"], np.uint32)[: scene.count]
    assert np.any(clone_counts > 0)


def test_loss_weighted_backward_clone_counts_follow_high_loss_region(device, tmp_path: Path) -> None:
    frame = _make_frame(tmp_path, image_name="weighted_clone_counts_target.png", image_id=17)
    rotations = np.zeros((2, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    scene = GaussianScene(
        positions=np.array([[-0.9, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32),
        scales=np.log(np.full((2, 3), 0.12, dtype=np.float32)),
        rotations=rotations,
        opacities=np.full((2,), 0.95, dtype=np.float32),
        colors=np.full((2, 3), 1.0, dtype=np.float32),
        sh_coeffs=np.zeros((2, 1, 3), dtype=np.float32),
    )
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(background_mode=TRAIN_BACKGROUND_MODE_CUSTOM, background=(0.0, 0.0, 0.0), density_regularizer=0.0, depth_ratio_weight=0.0, refinement_interval=9999, refinement_loss_weight=1.0, refinement_target_edge_weight=0.0),
        seed=123,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.zeros((3,), dtype=np.float32)
    usage = spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination

    def make_target_texture(image: np.ndarray) -> spy.Texture:
        texture = device.create_texture(format=spy.Format.rgba32_float, width=int(image.shape[1]), height=int(image.shape[0]), usage=usage)
        texture.copy_from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        return texture

    def run_target(target_image: np.ndarray) -> np.ndarray:
        trainer.refinement_buffers["clone_counts"].copy_from_numpy(np.zeros((scene.count,), dtype=np.uint32))
        trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.zeros((scene.count,), dtype=np.uint32))
        renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
        enc = device.create_command_encoder()
        trainer._dispatch_raster_training_forward(enc, camera, background)
        target_texture = make_target_texture(target_image)
        trainer._dispatch_loss_forward(enc, target_texture)
        trainer._dispatch_loss_backward(enc, target_texture)
        renderer.clear_raster_grads_current_scene(enc)
        renderer.rasterize_backward_current_scene(
            enc,
            camera,
            background,
            renderer.output_grad_buffer,
            regularizer_grad=renderer.work_buffers["training_regularizer_grad"],
            clone_counts_buffer=trainer.refinement_buffers["clone_counts"],
            clone_select_probability=1.0,
            clone_seed=123,
            refinement_loss_weight=float(trainer.training.refinement_loss_weight),
            refinement_target_edge_weight=float(trainer.training.refinement_target_edge_weight),
        )
        device.submit_command_buffer(enc.finish())
        device.wait()
        return buffer_to_numpy(trainer.refinement_buffers["clone_counts"], np.uint32)[: scene.count].copy()

    def side_dominant_splat() -> tuple[int, int]:
        side_energy: list[tuple[float, float]] = []
        for splat_index in range(scene.count):
            single_scene = GaussianScene(
                positions=scene.positions[[splat_index]].copy(),
                scales=scene.scales[[splat_index]].copy(),
                rotations=scene.rotations[[splat_index]].copy(),
                opacities=scene.opacities[[splat_index]].copy(),
                colors=scene.colors[[splat_index]].copy(),
                sh_coeffs=scene.sh_coeffs[[splat_index]].copy(),
            )
            single_renderer = GaussianRenderer(device, width=renderer.width, height=renderer.height, list_capacity_multiplier=16)
            image = np.asarray(single_renderer.render(single_scene, camera, background=background).image, dtype=np.float32)[..., :3]
            side_energy.append((float(np.sum(image[:, : renderer.width // 2], dtype=np.float64)), float(np.sum(image[:, renderer.width // 2 :], dtype=np.float64))))
        left_index = int(np.argmax([energy[0] for energy in side_energy]))
        right_index = int(np.argmax([energy[1] for energy in side_energy]))
        assert left_index != right_index
        return left_index, right_index

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    enc = device.create_command_encoder()
    trainer._dispatch_raster_training_forward(enc, camera, background)
    device.submit_command_buffer(enc.finish())
    device.wait()
    rendered = np.asarray(renderer.output_texture.to_numpy(), dtype=np.float32).copy()
    left_splat, right_splat = side_dominant_splat()
    left_target = rendered.copy()
    right_target = rendered.copy()
    left_target[:, : renderer.width // 2, :3] = 0.0
    right_target[:, renderer.width // 2 :, :3] = 0.0

    left_counts = run_target(left_target)
    right_counts = run_target(right_target)

    assert left_counts[left_splat] > left_counts[right_splat]
    assert right_counts[right_splat] > right_counts[left_splat]


def test_edge_weighted_backward_clone_counts_follow_target_edges(device, tmp_path: Path) -> None:
    frame = _make_frame(tmp_path, image_name="edge_weighted_clone_counts_target.png", image_id=18)
    rotations = np.zeros((2, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    scene = GaussianScene(
        positions=np.array([[-0.9, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32),
        scales=np.log(np.full((2, 3), 0.12, dtype=np.float32)),
        rotations=rotations,
        opacities=np.full((2,), 0.95, dtype=np.float32),
        colors=np.full((2, 3), 1.0, dtype=np.float32),
        sh_coeffs=np.zeros((2, 1, 3), dtype=np.float32),
    )
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(background_mode=TRAIN_BACKGROUND_MODE_CUSTOM, background=(0.0, 0.0, 0.0), density_regularizer=0.0, depth_ratio_weight=0.0, refinement_interval=9999, refinement_loss_weight=0.0, refinement_target_edge_weight=1.0),
        seed=123,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.zeros((3,), dtype=np.float32)
    usage = spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination

    def make_target_texture(image: np.ndarray) -> spy.Texture:
        texture = device.create_texture(format=spy.Format.rgba32_float, width=int(image.shape[1]), height=int(image.shape[0]), usage=usage)
        texture.copy_from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        return texture

    def run_target(target_image: np.ndarray) -> np.ndarray:
        trainer.refinement_buffers["clone_counts"].copy_from_numpy(np.zeros((scene.count,), dtype=np.uint32))
        trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.zeros((scene.count,), dtype=np.uint32))
        renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
        enc = device.create_command_encoder()
        trainer._dispatch_raster_training_forward(enc, camera, background)
        target_texture = make_target_texture(target_image)
        trainer._dispatch_loss_forward(enc, target_texture)
        trainer._dispatch_loss_backward(enc, target_texture)
        renderer.clear_raster_grads_current_scene(enc)
        renderer.rasterize_backward_current_scene(
            enc,
            camera,
            background,
            renderer.output_grad_buffer,
            regularizer_grad=renderer.work_buffers["training_regularizer_grad"],
            clone_counts_buffer=trainer.refinement_buffers["clone_counts"],
            clone_select_probability=1.0,
            clone_seed=123,
            refinement_loss_weight=float(trainer.training.refinement_loss_weight),
            refinement_target_edge_weight=float(trainer.training.refinement_target_edge_weight),
        )
        device.submit_command_buffer(enc.finish())
        device.wait()
        return buffer_to_numpy(trainer.refinement_buffers["clone_counts"], np.uint32)[: scene.count].copy()

    def single_splat_mask(splat_index: int) -> np.ndarray:
        single_scene = GaussianScene(
            positions=scene.positions[[splat_index]].copy(),
            scales=scene.scales[[splat_index]].copy(),
            rotations=scene.rotations[[splat_index]].copy(),
            opacities=scene.opacities[[splat_index]].copy(),
            colors=scene.colors[[splat_index]].copy(),
            sh_coeffs=scene.sh_coeffs[[splat_index]].copy(),
        )
        single_renderer = GaussianRenderer(device, width=renderer.width, height=renderer.height, list_capacity_multiplier=16)
        image = np.asarray(single_renderer.render(single_scene, camera, background=background).image, dtype=np.float32)[..., :3]
        return np.any(image > 1e-4, axis=2)

    left_mask = single_splat_mask(0)
    right_mask = single_splat_mask(1)
    left_target = np.zeros((renderer.height, renderer.width, 4), dtype=np.float32)
    right_target = np.zeros((renderer.height, renderer.width, 4), dtype=np.float32)
    left_target[left_mask, :3] = 1.0
    left_target[..., 3] = 1.0
    right_target[right_mask, :3] = 1.0
    right_target[..., 3] = 1.0

    left_counts = run_target(left_target)
    right_counts = run_target(right_target)

    assert left_counts[0] > left_counts[1]
    assert right_counts[1] > right_counts[0]


def test_loss_weighted_backward_clone_counts_disable_when_rgb_loss_is_zero(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=91)
    frame = _make_frame(tmp_path, image_name="zero_rgb_loss_clone_counts_target.png", image_id=18)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(density_regularizer=0.0, depth_ratio_weight=0.0, refinement_interval=9999, refinement_loss_weight=1.0, refinement_target_edge_weight=0.0),
        seed=123,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer.training.background, dtype=np.float32)
    usage = spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    enc = device.create_command_encoder()
    trainer._dispatch_raster_training_forward(enc, camera, background)
    device.submit_command_buffer(enc.finish())
    device.wait()

    target_texture = device.create_texture(format=spy.Format.rgba32_float, width=renderer.width, height=renderer.height, usage=usage)
    target_texture.copy_from_numpy(np.ascontiguousarray(np.asarray(renderer.output_texture.to_numpy(), dtype=np.float32), dtype=np.float32))
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(np.zeros((scene.count,), dtype=np.uint32))
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.zeros((scene.count,), dtype=np.uint32))

    enc = device.create_command_encoder()
    trainer._dispatch_loss_forward(enc, target_texture)
    trainer._dispatch_loss_backward(enc, target_texture)
    renderer.clear_raster_grads_current_scene(enc)
    renderer.rasterize_backward_current_scene(
        enc,
        camera,
        background,
        renderer.output_grad_buffer,
        regularizer_grad=renderer.work_buffers["training_regularizer_grad"],
        clone_counts_buffer=trainer.refinement_buffers["clone_counts"],
        clone_select_probability=1.0,
        clone_seed=123,
        refinement_loss_weight=float(trainer.training.refinement_loss_weight),
        refinement_target_edge_weight=float(trainer.training.refinement_target_edge_weight),
    )
    device.submit_command_buffer(enc.finish())
    device.wait()

    clone_counts = buffer_to_numpy(trainer.refinement_buffers["clone_counts"], np.uint32)[: scene.count]
    assert np.all(clone_counts == 0)


def test_hybrid_clone_weighting_disables_when_both_weights_are_zero(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=92)
    frame = _make_frame(tmp_path, image_name="zero_hybrid_clone_counts_target.png", image_id=19)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(density_regularizer=0.0, depth_ratio_weight=0.0, refinement_interval=9999, refinement_loss_weight=0.0, refinement_target_edge_weight=0.0),
        seed=123,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer.training.background, dtype=np.float32)

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    enc = device.create_command_encoder()
    trainer._dispatch_raster_training_forward(enc, camera, background)
    target_texture = trainer.get_frame_target_texture(0, native_resolution=False, encoder=enc)
    trainer._dispatch_loss_forward(enc, target_texture)
    trainer._dispatch_loss_backward(enc, target_texture)
    renderer.clear_raster_grads_current_scene(enc)
    renderer.rasterize_backward_current_scene(
        enc,
        camera,
        background,
        renderer.output_grad_buffer,
        regularizer_grad=renderer.work_buffers["training_regularizer_grad"],
        clone_counts_buffer=trainer.refinement_buffers["clone_counts"],
        clone_select_probability=1.0,
        clone_seed=123,
        refinement_loss_weight=float(trainer.training.refinement_loss_weight),
        refinement_target_edge_weight=float(trainer.training.refinement_target_edge_weight),
    )
    device.submit_command_buffer(enc.finish())
    device.wait()

    clone_counts = buffer_to_numpy(trainer.refinement_buffers["clone_counts"], np.uint32)[: scene.count]
    assert np.all(clone_counts == 0)


def test_density_loss_respects_threshold_and_changes_gradients(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=89)
    frame = _make_frame(tmp_path, image_name="density_enabled_target.png", image_id=16)
    renderer_off = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    renderer_on = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer_off = GaussianTrainer(device=device, renderer=renderer_off, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(density_regularizer=0.0), seed=123)
    trainer_on = GaussianTrainer(device=device, renderer=renderer_on, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(density_regularizer=0.5, max_allowed_density_start=0.0, max_allowed_density=0.0), seed=123)
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer_on.training.background, dtype=np.float32)

    def run_pass(trainer: GaussianTrainer, renderer: GaussianRenderer):
        renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
        enc = device.create_command_encoder()
        trainer._dispatch_raster_training_forward(enc, camera, background)
        target_texture = trainer.get_frame_target_texture(0, native_resolution=False, encoder=enc)
        trainer._dispatch_loss_forward(enc, target_texture)
        trainer._dispatch_loss_backward(enc, target_texture)
        trainer._dispatch_raster_backward(enc, camera, background)
        device.submit_command_buffer(enc.finish())
        device.wait()
        return trainer._read_loss_metrics(), _read_grad_groups(renderer, scene.count)

    (total_off, mse_off, density_off), grads_off = run_pass(trainer_off, renderer_off)
    (total_on, mse_on, density_on), grads_on = run_pass(trainer_on, renderer_on)

    assert np.isfinite(total_off)
    assert np.isfinite(total_on)
    np.testing.assert_allclose(mse_on, mse_off, rtol=1e-5, atol=1e-7)
    assert density_off == 0.0
    assert density_on > 0.0
    assert total_on >= total_off - 1e-7
    grad_delta = sum(float(np.max(np.abs(grads_on[name] - grads_off[name]))) for name in grads_on)
    assert grad_delta > 0.0


def test_depth_ratio_loss_changes_total_loss_and_gradients(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=90)
    frame = _make_frame(tmp_path, image_name="depth_ratio_target.png", image_id=17)
    grad_min = 0.01
    grad_max = 0.05
    weight = 0.5
    renderer_off = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    renderer_on = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer_off = GaussianTrainer(device=device, renderer=renderer_off, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(density_regularizer=0.0, depth_ratio_weight=0.0, depth_ratio_grad_min=grad_min, depth_ratio_grad_max=grad_max), seed=123)
    trainer_on = GaussianTrainer(device=device, renderer=renderer_on, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(density_regularizer=0.0, depth_ratio_weight=weight, depth_ratio_grad_min=grad_min, depth_ratio_grad_max=grad_max), seed=123)

    def run_pass(trainer: GaussianTrainer, renderer: GaussianRenderer, depth_ratio: float):
        packed_depth_stats = np.zeros((renderer.height, renderer.width, 4), dtype=np.float32)
        packed_depth_stats[:, :, :] = np.array([1.0, 2.0, 1.0, depth_ratio], dtype=np.float32)
        renderer.training_depth_stats_texture.copy_from_numpy(packed_depth_stats)
        target_texture = trainer.get_frame_target_texture(0, native_resolution=False)
        enc = device.create_command_encoder()
        trainer._dispatch_loss_forward(enc, target_texture)
        trainer._dispatch_loss_backward(enc, target_texture)
        device.submit_command_buffer(enc.finish())
        device.wait()
        regularizer_grad = buffer_to_numpy(renderer.work_buffers["training_regularizer_grad"], np.float32).reshape(-1, 2)
        return trainer._read_loss_metrics(), regularizer_grad

    inv_pixel_count = 1.0 / float(renderer_on.width * renderer_on.height)
    samples = (0.005, 0.03, 0.2)
    losses_on: list[float] = []
    grads_on: list[float] = []
    for depth_ratio in samples:
        (total_off, mse_off, density_off), regularizer_grad_off = run_pass(trainer_off, renderer_off, depth_ratio)
        (total_on, mse_on, density_on), regularizer_grad_on = run_pass(trainer_on, renderer_on, depth_ratio)

        expected_loss = _depth_ratio_window_loss(depth_ratio, weight, grad_min, grad_max)
        expected_grad = _depth_ratio_window_grad(depth_ratio, weight, grad_min, grad_max, inv_pixel_count)

        assert np.isfinite(total_off)
        assert np.isfinite(total_on)
        np.testing.assert_allclose(mse_on, mse_off, rtol=1e-5, atol=1e-7)
        assert density_off == 0.0
        assert density_on == 0.0
        np.testing.assert_allclose(total_on - total_off, expected_loss, rtol=1e-5, atol=1e-7)
        assert np.allclose(regularizer_grad_off[:, 1], 0.0)
        np.testing.assert_allclose(regularizer_grad_on[:, 1], np.full_like(regularizer_grad_on[:, 1], expected_grad), rtol=1e-5, atol=1e-7)
        losses_on.append(float(total_on - total_off))
        grads_on.append(float(regularizer_grad_on[0, 1]))

    assert losses_on[0] < losses_on[1] < losses_on[2]
    assert grads_on[1] > grads_on[0] > grads_on[2]


def test_depth_ratio_regularizer_affects_training_raster_replay(device, tmp_path: Path) -> None:
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.03, -0.02, 0.1]], dtype=np.float32),
        scales=_log_sigma(np.full((3, 3), 0.12, dtype=np.float32)),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.full((3,), 0.8, dtype=np.float32),
        colors=np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]], dtype=np.float32),
        sh_coeffs=np.zeros((3, 1, 3), dtype=np.float32),
    )
    frame = _make_frame(tmp_path, image_name="depth_ratio_raster_target.png", image_id=18)
    renderer_off = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    renderer_on = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer_off = GaussianTrainer(device=device, renderer=renderer_off, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(density_regularizer=0.0, depth_ratio_weight=0.0), seed=123)
    trainer_on = GaussianTrainer(device=device, renderer=renderer_on, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(density_regularizer=0.0, depth_ratio_weight=0.5), seed=123)
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.asarray(trainer_on.training.background, dtype=np.float32)

    def configure_band(trainer: GaussianTrainer, renderer: GaussianRenderer) -> tuple[float, float]:
        renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
        enc = device.create_command_encoder()
        trainer._dispatch_raster_training_forward(enc, camera, background)
        device.submit_command_buffer(enc.finish())
        device.wait()
        ratios = _read_training_depth_ratios(renderer)
        positive = ratios[np.isfinite(ratios) & (ratios > 0.0)]
        assert positive.size >= 4
        grad_min = float(np.quantile(positive, 0.25))
        grad_max = float(np.quantile(positive, 0.75))
        grad_min, grad_max = resolve_depth_ratio_grad_band(grad_min, grad_max)
        trainer.training.depth_ratio_grad_min = grad_min
        trainer.training.depth_ratio_grad_max = grad_max
        return grad_min, grad_max

    def run_pass(trainer: GaussianTrainer, renderer: GaussianRenderer):
        renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
        enc = device.create_command_encoder()
        trainer._dispatch_raster_training_forward(enc, camera, background)
        target_texture = trainer.get_frame_target_texture(0, native_resolution=False, encoder=enc)
        trainer._dispatch_loss_forward(enc, target_texture)
        trainer._dispatch_loss_backward(enc, target_texture)
        trainer._dispatch_raster_backward(enc, camera, background)
        device.submit_command_buffer(enc.finish())
        device.wait()
        regularizer_grad = buffer_to_numpy(renderer.work_buffers["training_regularizer_grad"], np.float32).reshape(-1, 2)
        return trainer._read_loss_metrics(), _read_grad_groups(renderer, scene.count), regularizer_grad, _read_training_depth_ratios(renderer)

    grad_min, grad_max = configure_band(trainer_on, renderer_on)
    trainer_off.training.depth_ratio_grad_min = grad_min
    trainer_off.training.depth_ratio_grad_max = grad_max
    (total_off, mse_off, density_off), grads_off, regularizer_grad_off, depth_ratio_off = run_pass(trainer_off, renderer_off)
    (total_on, mse_on, density_on), grads_on, regularizer_grad_on, depth_ratio_on = run_pass(trainer_on, renderer_on)

    assert np.isfinite(total_off)
    assert np.isfinite(total_on)
    np.testing.assert_allclose(mse_on, mse_off, rtol=1e-5, atol=1e-7)
    assert density_off == 0.0
    assert density_on == 0.0
    assert total_on > total_off
    assert np.allclose(regularizer_grad_off[:, 1], 0.0)
    assert float(np.max(np.abs(regularizer_grad_on[:, 1]))) > 0.0
    inside_mask = np.isfinite(depth_ratio_on) & (depth_ratio_on >= grad_min) & (depth_ratio_on <= grad_max)
    outside_mask = np.isfinite(depth_ratio_on) & (depth_ratio_on > 0.0) & ~inside_mask
    assert np.any(inside_mask)
    assert np.any(outside_mask)
    assert float(np.max(np.abs(regularizer_grad_on[inside_mask, 1]))) > float(np.max(np.abs(regularizer_grad_on[outside_mask, 1])))
    np.testing.assert_allclose(depth_ratio_off, depth_ratio_on, rtol=1e-5, atol=1e-7)
    grad_delta = sum(float(np.max(np.abs(grads_on[name] - grads_off[name]))) for name in grads_on)
    assert grad_delta > 0.0


def test_refinement_rewrite_culls_and_splits_families(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=89)
    scene.opacities[:] = np.array([0.6, 1e-5], dtype=np.float32)
    scene.scales[:] = _log_sigma(np.array([[0.09, 0.06, 0.03], [0.05, 0.05, 0.05]], dtype=np.float32))
    source_position = scene.positions[0].copy()
    frame = _make_frame(tmp_path, image_name="refinement_rewrite_target.png", image_id=11)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-3, refinement_min_contribution_percent=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(np.array([2, 5], dtype=np.uint32))
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200, 0], dtype=np.uint32))
    trainer._run_refinement()

    groups = _read_scene_groups(renderer, trainer.scene.count)
    actual_scales = _actual_scale(groups["scales"][:, :3])
    actual_opacity = _actual_opacity(groups["color_alpha"][:, 3])
    expected_scale = _expected_refinement_child_scale(np.array([0.09, 0.06, 0.03], dtype=np.float32), trainer.scene.count)

    assert trainer.scene.count == 3
    np.testing.assert_allclose(
        actual_scales,
        np.repeat(expected_scale[None, :], 3, axis=0),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.max(actual_scales, axis=1) / np.min(actual_scales, axis=1), np.full((3,), 1.5, dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(actual_opacity, np.full((3,), 0.6, dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.mean(groups["positions"][:, :3], axis=0), source_position, rtol=0.0, atol=1e-6)
    offsets = groups["positions"][:, :3] - source_position[None, :]
    np.testing.assert_allclose(np.sum(offsets, axis=0), np.zeros((3,), dtype=np.float32), rtol=0.0, atol=1e-6)
    assert float(np.max(np.linalg.norm(offsets, axis=1))) > 1e-3
    clone_counts_after = buffer_to_numpy(trainer.refinement_buffers["clone_counts"], np.uint32)[: trainer.scene.count]
    contribution_after = buffer_to_numpy(trainer.refinement_buffers["splat_contribution"], np.uint32)[: trainer.scene.count]
    assert np.all(clone_counts_after == 0)
    assert np.all(contribution_after == 0)


def test_refinement_runtime_uses_base_min_contribution_for_first_refinement_then_decays(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=93)
    frame = _make_frame(tmp_path, image_name="refinement_threshold_decay_target.png", image_id=212)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_interval=200, refinement_min_contribution_percent=1e-05, refinement_min_contribution_decay=0.95),
        seed=123,
    )
    observed_pixels = renderer.width * renderer.height
    trainer._observed_contribution_pixel_count = observed_pixels

    trainer.state.step = 200
    first_threshold = int(trainer._refinement_vars()["g_RefinementMinContributionThreshold"])
    trainer.state.step = 400
    second_threshold = int(trainer._refinement_vars()["g_RefinementMinContributionThreshold"])

    assert first_threshold == contribution_fixed_count_from_percent(1e-05, observed_pixels)
    assert second_threshold == contribution_fixed_count_from_percent(9.5e-06, observed_pixels)


def test_refinement_rewrite_culls_low_contribution_splats(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=95)
    scene.opacities[:] = np.array([0.6, 0.7], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="refinement_contribution_cull_target.png", image_id=111)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-6, refinement_min_contribution_percent=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(np.array([0, 0], dtype=np.uint32))
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200, 49], dtype=np.uint32))
    trainer._run_refinement()

    assert trainer.scene.count == 1
    groups = _read_scene_groups(renderer, trainer.scene.count)
    np.testing.assert_allclose(groups["positions"][0, :3], scene.positions[0], rtol=0.0, atol=1e-6)


def test_refinement_opacity_mul_respects_half_opacity_floor(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=191)
    scene.opacities[:] = np.array([0.6], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="refinement_opacity_floor_target.png", image_id=191)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            refinement_alpha_cull_threshold=1e-6,
            refinement_min_contribution_percent=contribution_percent_from_fixed_count(50, observed_pixels),
            refinement_opacity_mul=0.25,
        ),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(np.array([0], dtype=np.uint32))
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement()

    groups = _read_scene_groups(renderer, trainer.scene.count)
    np.testing.assert_allclose(_actual_opacity(groups["color_alpha"][:, 3]), np.array([0.5], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_refinement_rewrite_migrates_adam_moments(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=97)
    scene.opacities[:] = np.array([0.6, 1e-5], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="refinement_adam_target.png", image_id=12)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-3, refinement_min_contribution_percent=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    src_moments = np.zeros((renderer.TRAINABLE_PARAM_COUNT, scene.count, 2), dtype=np.float32)
    src_moments[:, 0, 0] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 1.0
    src_moments[:, 0, 1] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 101.0
    src_moments[:, 1, 0] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 1001.0
    src_moments[:, 1, 1] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 1101.0
    trainer.adam_optimizer.buffers["adam_moments"].copy_from_numpy(src_moments.reshape(-1, 2))

    trainer._observed_contribution_pixel_count = observed_pixels
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(np.array([2, 5], dtype=np.uint32))
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200, 0], dtype=np.uint32))
    trainer._run_refinement()

    expected = np.repeat(src_moments[:, 0:1, :], 3, axis=1)
    np.testing.assert_allclose(_read_adam_moments(trainer, trainer.scene.count), expected, rtol=0.0, atol=1e-7)


def test_refinement_respects_max_gaussians_cap(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=101)
    scene.opacities[:] = np.array([0.6, 0.7], dtype=np.float32)
    scene.scales[:] = _log_sigma(np.array([[0.09, 0.06, 0.03], [0.04, 0.04, 0.04]], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="refinement_max_count_target.png", image_id=13)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(max_gaussians=3, refinement_alpha_cull_threshold=1e-3, refinement_min_contribution_percent=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(np.array([4, 4], dtype=np.uint32))
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200, 200], dtype=np.uint32))
    trainer._run_refinement()

    assert trainer.scene.count == 3
    groups = _read_scene_groups(renderer, trainer.scene.count)
    actual_scales = _actual_scale(groups["scales"][:, :3])
    split_scale = _expected_refinement_child_scale(np.array([0.09, 0.06, 0.03], dtype=np.float32), 2)
    split_mask = np.all(np.isclose(actual_scales, split_scale[None, :], rtol=0.0, atol=1e-6), axis=1)
    unsplit_mask = np.all(np.isclose(actual_scales, np.full((1, 3), 0.04, dtype=np.float32), rtol=0.0, atol=1e-6), axis=1)
    assert int(np.count_nonzero(split_mask)) == 2
    assert int(np.count_nonzero(unsplit_mask)) == 1
    split_positions = groups["positions"][split_mask, :3]
    np.testing.assert_allclose(np.mean(split_positions, axis=0), scene.positions[0], rtol=0.0, atol=1e-6)


def test_refinement_rewrite_sampling_depends_on_frame_hash(device, tmp_path: Path) -> None:
    scene_a = _make_scene(count=1, seed=151)
    scene_b = _make_scene(count=1, seed=151)
    scene_a.opacities[:] = np.array([0.6], dtype=np.float32)
    scene_b.opacities[:] = np.array([0.6], dtype=np.float32)
    scene_a.scales[:] = _log_sigma(np.array([[0.08, 0.05, 0.03]], dtype=np.float32))
    scene_b.scales[:] = scene_a.scales.copy()
    frame_a = _make_frame(tmp_path, image_name="refinement_frame_hash_a.png", image_id=41)
    frame_b = _make_frame(tmp_path, image_name="refinement_frame_hash_b.png", image_id=42)
    renderer_a = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    renderer_b = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer_a.width * renderer_a.height
    trainer_a = GaussianTrainer(
        device=device,
        renderer=renderer_a,
        scene=scene_a,
        frames=[frame_a],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-6, refinement_min_contribution_percent=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )
    trainer_b = GaussianTrainer(
        device=device,
        renderer=renderer_b,
        scene=scene_b,
        frames=[frame_b],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-6, refinement_min_contribution_percent=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    trainer_a._observed_contribution_pixel_count = observed_pixels
    trainer_b._observed_contribution_pixel_count = observed_pixels
    trainer_a.refinement_buffers["clone_counts"].copy_from_numpy(np.array([1], dtype=np.uint32))
    trainer_b.refinement_buffers["clone_counts"].copy_from_numpy(np.array([1], dtype=np.uint32))
    trainer_a.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer_b.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer_a._run_refinement()
    trainer_b._run_refinement()

    positions_a = _read_scene_groups(renderer_a, trainer_a.scene.count)["positions"][:, :3]
    positions_b = _read_scene_groups(renderer_b, trainer_b.scene.count)["positions"][:, :3]
    np.testing.assert_allclose(np.mean(positions_a, axis=0), scene_a.positions[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.mean(positions_b, axis=0), scene_b.positions[0], rtol=0.0, atol=1e-6)
    assert not np.allclose(positions_a, positions_b, rtol=0.0, atol=1e-6)


def test_refinement_rewrite_keeps_sampled_family_offsets_within_fibonacci_volume(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=173)
    scene.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([0.7, 0.5, 0.3], dtype=np.float32))
    scene.rotations[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scene.opacities[0] = np.float32(0.6)
    frame = _make_frame(tmp_path, image_name="refinement_sigma_clamp_target.png", image_id=61)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-6, refinement_min_contribution_percent=contribution_percent_from_fixed_count(50, observed_pixels)),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(np.array([31], dtype=np.uint32))
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement()

    groups = _read_scene_groups(renderer, trainer.scene.count)
    family_positions = groups["positions"][:, :3]
    parent_scale = np.array([0.7, 0.5, 0.3], dtype=np.float32)
    family_size = trainer.scene.count
    shrink = family_size ** (-1.0 / 3.0)
    residual_sigma = parent_scale * np.sqrt(max(1.0 - shrink * shrink, 0.0))
    normalized_lengths = np.linalg.norm((family_positions - scene.positions[0][None, :]) / residual_sigma[None, :], axis=1)

    assert trainer.scene.count == 32
    assert float(np.max(normalized_lengths)) <= 2.5 + 1e-5


def test_refinement_min_screen_size_raises_small_splats(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=103)
    scene.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([1e-5, 1e-5, 1e-5], dtype=np.float32))
    near_frame = _make_frame(tmp_path, image_name="refinement_min_near.png", image_id=21)
    tight_image_path = tmp_path / "refinement_min_tight.png"
    offscreen_image_path = tmp_path / "refinement_min_offscreen.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(tight_image_path)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(offscreen_image_path)
    tight_frame = ColmapFrame(
        image_id=22,
        image_path=tight_image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        fx=144.0,
        fy=144.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
    )
    offscreen_frame = ColmapFrame(
        image_id=23,
        image_path=offscreen_image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([10.0, 0.0, 3.0], dtype=np.float32),
        fx=512.0,
        fy=512.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
    )
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[near_frame, tight_frame, offscreen_frame],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-3),
        seed=123,
    )

    expected_support = min(
        value
        for value in (
            _circle_bound_support_radius(trainer.make_frame_camera(frame_index, renderer.width, renderer.height), scene.positions[0], renderer.width, renderer.height, _REFINEMENT_MIN_SCREEN_RADIUS_PX)
            for frame_index in range(3)
        )
        if value is not None
    )

    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement()

    scales = _actual_scale(_read_scene_groups(renderer, trainer.scene.count)["scales"][0, :3])
    initial_scale = _actual_scale(scene.scales[0, :3])
    assert np.all(scales > initial_scale)
    np.testing.assert_allclose(scales, np.full((3,), scales[0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert float(scales[0]) <= float(expected_support / _GAUSSIAN_SUPPORT_SIGMA_RADIUS) + 1e-6


def test_training_max_screen_size_clamps_large_splats(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=107)
    scene.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([2.0, 2.0, 2.0], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="max_screen_clamp_target.png", image_id=24)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0),
        seed=123,
    )
    camera = trainer.make_frame_camera(0, renderer.width, renderer.height)
    max_radius_px = float(np.sqrt(_TRAINING_MAX_SCREEN_FRACTION * renderer.width * renderer.height / np.pi))
    expected_support = _circle_bound_support_radius(camera, scene.positions[0], renderer.width, renderer.height, max_radius_px)
    assert expected_support is not None

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, 1)["scales"][0, :3])
    np.testing.assert_allclose(scales, np.full((3,), expected_support / _GAUSSIAN_SUPPORT_SIGMA_RADIUS, dtype=np.float32), rtol=0.0, atol=1e-6)


def test_training_max_screen_size_ignores_offscreen_centers(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=109)
    scene.positions[0] = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([2.0, 1.5, 1.0], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="max_screen_offscreen_target.png", image_id=25)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0),
        seed=123,
    )
    camera = trainer.make_frame_camera(0, renderer.width, renderer.height)

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, 1)["scales"][0, :3])
    np.testing.assert_allclose(scales, np.array([2.0, 1.5, 1.0], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_optimizer_projection_clamps_sh_coefficients(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=111)
    frame = _make_frame(tmp_path, image_name="sh_projection_clamp_target.png", image_id=26)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0, sh1_reg_weight=0.0),
        seed=123,
    )
    camera = trainer.make_frame_camera(0, renderer.width, renderer.height)
    initial_groups = _read_scene_groups(renderer, 1)
    renderer.write_scene_groups(
        1,
        positions=initial_groups["positions"],
        scales=initial_groups["scales"],
        rotations=initial_groups["rotations"],
        sh_coeffs=np.array(
            [[[3.0, -3.0, 0.25], [1.5, -1.25, 0.5], [-1.6, 0.25, 1.2], [0.0, 2.0, -3.0]]],
            dtype=np.float32,
        ),
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    zero_sh = np.zeros((scene.count, 4, 3), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_sh_coeffs=zero_sh, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    sh_coeffs = _read_scene_groups(renderer, 1)["sh_coeffs"][0]
    sh0_limit = np.float32(0.5 / SH_C0)
    np.testing.assert_allclose(sh_coeffs[0], np.array([sh0_limit, -sh0_limit, 0.25], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        sh_coeffs[1:4],
        np.array([[1.0, -1.0, 0.5], [-1.0, 0.25, 1.0], [0.0, 1.0, -1.0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(sh_coeffs[4:], 0.0, rtol=0.0, atol=1e-6)


def test_box_downscale_matches_expected_mean(device, tmp_path: Path) -> None:
    image_path = tmp_path / "downscale_target.png"
    image = np.array(
        [
            [[0, 0, 0], [255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255], [255, 255, 0]],
            [[255, 0, 255], [0, 255, 255], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(image, mode="RGB").save(image_path)
    frame = ColmapFrame(
        image_id=0,
        image_path=image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        fx=72.0,
        fy=72.0,
        cx=1.5,
        cy=1.5,
        width=3,
        height=3,
    )
    scene = _make_scene(count=4, seed=123)
    renderer = GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(train_downscale_factor=2),
        seed=5,
    )

    target = trainer.get_frame_target_texture(0, native_resolution=False)
    target_np = np.asarray(target.to_numpy(), dtype=np.float32)
    srgb = image.astype(np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4))
    expected = np.array(
        [
            [np.mean(linear[0:2, 0:2], axis=(0, 1)), np.mean(linear[0:2, 2:3], axis=(0, 1))],
            [np.mean(linear[2:3, 0:2], axis=(0, 1)), np.mean(linear[2:3, 2:3], axis=(0, 1))],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(target_np[:, :, :3], expected, rtol=0.0, atol=5e-5)
    np.testing.assert_allclose(target_np[:, :, 3], np.ones((2, 2), dtype=np.float32), rtol=0.0, atol=1e-6)


def test_subsample_factor_contributes_to_downscaled_target_resolution(device, tmp_path: Path) -> None:
    image_path = tmp_path / "subsample_downscale_target.png"
    image = np.array(
        [
            [[0, 0, 0], [255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255], [255, 255, 0]],
            [[255, 0, 255], [0, 255, 255], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(image, mode="RGB").save(image_path)
    frame = ColmapFrame(
        image_id=0,
        image_path=image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        fx=72.0,
        fy=72.0,
        cx=1.5,
        cy=1.5,
        width=3,
        height=3,
    )
    scene = _make_scene(count=4, seed=123)
    renderer = GaussianRenderer(device, width=2, height=2, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(train_subsample_factor=2),
        seed=5,
    )

    target = trainer.get_frame_target_texture(0, native_resolution=False)
    target_np = np.asarray(target.to_numpy(), dtype=np.float32)
    srgb = image.astype(np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4))
    expected = np.array(
        [
            [np.mean(linear[0:2, 0:2], axis=(0, 1)), np.mean(linear[0:2, 2:3], axis=(0, 1))],
            [np.mean(linear[2:3, 0:2], axis=(0, 1)), np.mean(linear[2:3, 2:3], axis=(0, 1))],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(target_np[:, :, :3], expected, rtol=0.0, atol=5e-5)
    np.testing.assert_allclose(target_np[:, :, 3], np.ones((2, 2), dtype=np.float32), rtol=0.0, atol=1e-6)


def test_separable_gaussian_blur_preserves_impulse_energy_for_n_channels(device):
    width = height = 17
    channel_count = 6
    blur = SeparableGaussianBlur(device, width=width, height=height)
    input_buffer = blur.make_buffer(channel_count)
    output_buffer = blur.make_buffer(channel_count)
    image = np.zeros((height, width, channel_count), dtype=np.float32)
    center = width // 2
    image[center, center, 0] = 1.0
    image[center, center, 5] = 0.5
    input_buffer.copy_from_numpy(np.ascontiguousarray(image.reshape(-1), dtype=np.float32))

    enc = device.create_command_encoder()
    blur.blur(enc, input_buffer, output_buffer, channel_count)
    device.submit_command_buffer(enc.finish())
    device.wait()

    out = np.frombuffer(output_buffer.to_numpy().tobytes(), dtype=np.float32).reshape(height, width, channel_count)
    expected_weights = np.array(
        [
            0.00102838008447911,
            0.00759875813523919,
            0.03600077212843083,
            0.10936068950970002,
            0.21300553771125369,
            0.26601172486179436,
            0.21300553771125369,
            0.10936068950970002,
            0.03600077212843083,
            0.00759875813523919,
            0.00102838008447911,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out[:, :, 0].sum(), 1.0, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(out[:, :, 5].sum(), 0.5, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(out[:, :, 3], 0.0, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(out[center, center - 5 : center + 6, 0], expected_weights * expected_weights[5], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(out[center, center - 5 : center + 6, 5], 0.5 * expected_weights * expected_weights[5], rtol=0.0, atol=1e-6)


def test_fused_adam_handles_nan_grads(device, tmp_path: Path):
    scene = _make_scene(count=16)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=9)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    nan_grads = np.full((scene.count, 4), np.nan, dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=nan_grads, grad_scales=nan_grads, grad_rotations=nan_grads, grad_color_alpha=nan_grads)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scene_groups = _read_scene_groups(renderer, scene.count)
    for name in ("positions", "scales", "rotations", "color_alpha"):
        assert np.all(np.isfinite(scene_groups[name]))


def test_rotation_grad_updates_quaternion(device, tmp_path: Path):
    scene = _make_scene(count=16)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=11)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    grad_rot = np.zeros((scene.count, 4), dtype=np.float32)
    grad_rot[:, 0] = 0.5
    grad_rot[:, 1] = -0.25
    grad_rot[:, 2] = 0.125
    grad_rot[:, 3] = -0.375
    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=grad_rot, grad_color_alpha=zeros)

    before = _read_scene_groups(renderer, scene.count)["rotations"].copy()
    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()
    after = _read_scene_groups(renderer, scene.count)["rotations"]

    assert np.any(np.abs(after - before) > 0.0)
    norms = np.linalg.norm(after, axis=1)
    assert np.all(np.isfinite(norms))
    assert np.all(np.abs(norms - 1.0) < 1e-3)


def test_synthetic_base_grads_update_and_respect_constraints(device, tmp_path: Path):
    scene = _make_scene(count=16)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=21)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    base_grad = np.full((scene.count, 4), 0.25, dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=base_grad, grad_scales=base_grad, grad_rotations=base_grad, grad_color_alpha=base_grad)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scene_groups = _read_scene_groups(renderer, scene.count)
    positions = scene_groups["positions"]
    scales = scene_groups["scales"]
    rotations = scene_groups["rotations"]
    color_alpha = scene_groups["color_alpha"]

    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(scales))
    assert np.all(np.isfinite(rotations))
    assert np.all(np.isfinite(color_alpha))
    assert np.all(np.abs(positions[:, :3]) <= trainer.stability.position_abs_max + 1e-5)
    actual_scales = _actual_scale(scales[:, :3])
    assert np.all(actual_scales <= trainer.stability.max_scale + 1e-6)
    assert np.all(color_alpha[:, :3] >= -1e-6)
    assert np.all(color_alpha[:, :3] <= 1.0 + 1e-6)
    assert np.all(_actual_opacity(color_alpha[:, 3]) >= 0.0)
    assert np.all(_actual_opacity(color_alpha[:, 3]) <= 1.0)


def test_adam_step_clamps_anisotropy(device, tmp_path: Path):
    scene = _make_scene(count=1, seed=25)
    scene.scales[0] = _log_sigma(np.array([0.9, 0.05, 0.05], dtype=np.float32))
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        stability_hparams=StabilityHyperParams(max_anisotropy=10.0),
        training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0),
        seed=27,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _read_scene_groups(renderer, 1)["scales"]
    np.testing.assert_allclose(_actual_scale(scales[0, :3]), np.array([0.9, 0.09, 0.09], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_log_scale_regularizer_pulls_scales_toward_reference(device, tmp_path: Path):
    scene = _make_scene(count=2, seed=27)
    scene.scales[:] = _log_sigma(np.array([[0.02, 0.02, 0.02], [0.08, 0.08, 0.08]], dtype=np.float32))
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(scale_l2_weight=1.0), scale_reg_reference=0.04, seed=29)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales_after = _read_scene_groups(renderer, scene.count)["scales"]
    assert scales_after[0, 0] > scene.scales[0, 0]
    assert scales_after[1, 0] < scene.scales[1, 0]


def test_opacity_regularizer_pushes_true_opacity_down(device, tmp_path: Path):
    scene = _make_scene(count=2, seed=31)
    scene.opacities[:] = np.array([0.2, 0.8], dtype=np.float32)
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(scale_l2_weight=0.0, opacity_reg_weight=1.0), seed=41)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    before = _actual_opacity(_read_scene_groups(renderer, 2)["color_alpha"][:, 3])
    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()
    after = _actual_opacity(_read_scene_groups(renderer, 2)["color_alpha"][:, 3])

    assert np.all(after < before)


def test_sh1_regularizer_pushes_all_non_dc_sh_coeffs_toward_zero(device, tmp_path: Path):
    scene = _make_scene(count=2, seed=37)
    scene.sh_coeffs = np.zeros((scene.count, 16, 3), dtype=np.float32)
    scene.sh_coeffs[0, 1:, :] = np.array(
        [
            [0.60, -0.30, 0.20],
            [0.10, -0.50, 0.40],
            [-0.20, 0.25, -0.35],
            [0.30, -0.10, 0.15],
            [-0.45, 0.35, -0.20],
            [0.25, -0.40, 0.30],
            [-0.15, 0.20, -0.10],
            [0.40, -0.25, 0.05],
            [-0.35, 0.15, -0.45],
            [0.22, -0.18, 0.12],
            [-0.28, 0.32, -0.08],
            [0.18, -0.12, 0.27],
            [-0.24, 0.14, -0.16],
            [0.08, -0.22, 0.34],
            [-0.11, 0.09, -0.19],
        ],
        dtype=np.float32,
    )
    scene.sh_coeffs[1, 1:, :] = np.array(
        [
            [-0.40, 0.20, -0.10],
            [0.30, -0.20, 0.50],
            [0.15, -0.45, 0.25],
            [-0.18, 0.12, -0.28],
            [0.42, -0.16, 0.18],
            [-0.26, 0.31, -0.22],
            [0.12, -0.08, 0.14],
            [-0.33, 0.27, -0.05],
            [0.29, -0.11, 0.38],
            [-0.21, 0.17, -0.13],
            [0.24, -0.34, 0.07],
            [-0.16, 0.10, -0.26],
            [0.19, -0.15, 0.21],
            [-0.09, 0.23, -0.31],
            [0.13, -0.07, 0.17],
        ],
        dtype=np.float32,
    )
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=TrainingHyperParams(scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0, sh1_reg_weight=1.0), seed=43)
    camera = frame.make_camera(near=0.1, far=20.0)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    device.wait()

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    zero_sh = np.zeros((scene.count, 16, 3), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_sh_coeffs=zero_sh, grad_color_alpha=zeros)

    before = np.linalg.norm(_read_scene_groups(renderer, scene.count)["sh_coeffs"][:, 1:, :].reshape(scene.count, -1), axis=1)
    enc = device.create_command_encoder()
    trainer._dispatch_adam_step(enc)
    device.submit_command_buffer(enc.finish())
    device.wait()
    after = np.linalg.norm(_read_scene_groups(renderer, scene.count)["sh_coeffs"][:, 1:, :].reshape(scene.count, -1), axis=1)

    assert np.all(after < before)


def test_color_non_negative_regularizer_pushes_negative_sh_color_toward_zero(device, tmp_path: Path):
    color_non_negative_hash_splat = 0x9E3779B9
    color_non_negative_hash_phi = 0x85EBCA6B
    color_non_negative_hash_z = 0xC2B2AE35

    def hash_u32(value: int) -> int:
        value &= 0xFFFFFFFF
        value ^= value >> 16
        value = (value * 0x7FEB352D) & 0xFFFFFFFF
        value ^= value >> 15
        value = (value * 0x846CA68B) & 0xFFFFFFFF
        value ^= value >> 16
        return value & 0xFFFFFFFF

    def random01(seed: int) -> float:
        return float(hash_u32(seed) + 1) / 4294967297.0

    def sample_dir(seed: int, splat_id: int) -> np.ndarray:
        splat_seed = hash_u32(seed ^ (((splat_id + 1) * color_non_negative_hash_splat) & 0xFFFFFFFF))
        z = 1.0 - 2.0 * random01(splat_seed ^ color_non_negative_hash_z)
        phi = 2.0 * np.pi * random01(splat_seed ^ color_non_negative_hash_phi)
        radial = np.sqrt(max(1.0 - z * z, 0.0))
        return np.array([radial * np.cos(phi), radial * np.sin(phi), z], dtype=np.float32)

    scene = _make_scene(count=1, seed=44)
    scene.sh_coeffs = np.zeros((1, 16, 3), dtype=np.float32)
    view_dir = sample_dir(45, 0)
    strength = np.float32(2.0)
    scene.sh_coeffs[0, 1, :] = view_dir[1] * strength
    scene.sh_coeffs[0, 2, :] = -view_dir[2] * strength
    scene.sh_coeffs[0, 3, :] = view_dir[0] * strength
    frame = _make_frame(tmp_path, image_name="non_negative_color_reg_target.png", image_id=23)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            sh_band=1,
            lr_schedule_enabled=False,
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            color_non_negative_reg=1.0,
            depth_ratio_weight=0.0,
            refinement_interval=9999,
        ),
        seed=44,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    zero_sh = np.zeros((scene.count, 16, 3), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_sh_coeffs=zero_sh, grad_color_alpha=zeros)

    before_coeffs = _read_scene_groups(renderer, 1)["sh_coeffs"].copy()
    before_color = evaluate_sh_color(before_coeffs, view_dir[None, :])[0]
    encoder = device.create_command_encoder()
    trainer._dispatch_optimizer_step(encoder, 1, camera)
    device.submit_command_buffer(encoder.finish())
    device.wait()
    after_coeffs = _read_scene_groups(renderer, 1)["sh_coeffs"].copy()
    after_color = evaluate_sh_color(after_coeffs, view_dir[None, :])[0]

    assert np.all(before_color < 0.0)
    assert np.all(after_color > before_color)
    assert np.all(after_coeffs[0, 0, :] > before_coeffs[0, 0, :])


def test_training_with_sh_enabled_updates_non_dc_sh_coeffs(device, tmp_path: Path):
    scene = _make_scene(count=4, seed=123)
    frame = _make_frame(tmp_path, image_name="sh_enabled_target.png", image_id=22)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            sh_band=3,
            lr_schedule_enabled=False,
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            depth_ratio_weight=0.0,
            refinement_interval=9999,
        ),
        seed=123,
    )

    before = _read_scene_groups(renderer, scene.count)["sh_coeffs"][:, 1:, :].copy()
    frame_index = 0
    trainer._refresh_train_target(None, frame_index)
    camera = trainer.make_frame_camera(frame_index, renderer.width, renderer.height)
    target = trainer.get_frame_target_texture(frame_index, native_resolution=False)
    background = trainer._training_background()
    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    encoder = device.create_command_encoder()
    trainer._dispatch_training_forward(encoder, camera, background, target, step=0, frame_index=frame_index)
    trainer._dispatch_training_backward(encoder, camera, background, target, step=0, frame_index=frame_index)
    device.submit_command_buffer(encoder.finish())
    device.wait()
    grad_groups = _read_grad_groups(renderer, scene.count)["grad_sh_coeffs"]
    grad_sh0 = grad_groups[:, 0, :]
    grad_sh = grad_groups[:, 1:, :]
    trainer.step()
    after = _read_scene_groups(renderer, scene.count)["sh_coeffs"][:, 1:, :]

    assert trainer.renderer.sh_band == 3
    assert float(np.max(np.abs(before))) == 0.0
    assert float(np.max(np.abs(grad_sh0))) > 0.0
    assert float(np.max(np.abs(grad_sh))) > 0.0
    assert float(np.max(np.abs(after))) > 0.0


def test_cpu_pointcloud_initializer_rebuilds_scene_with_nn_scales(device, tmp_path: Path):
    frame = _make_frame(tmp_path)
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=32)
    point_pos = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [-1.0, -1.0, 2.0]], dtype=np.float32)
    point_col = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=None, scene_count=16, upload_initial_scene=False, frames=[frame], init_point_positions=point_pos, init_point_colors=point_col, seed=101)

    init_params = GaussianInitHyperParams(position_jitter_std=0.0, base_scale=0.02, scale_jitter_ratio=0.0, initial_opacity=0.5, color_jitter_std=0.0)
    trainer.initialize_scene_from_pointcloud(splat_count=16, init_hparams=init_params, seed=123)

    scene_groups = _read_scene_groups(renderer, 4)
    positions = scene_groups["positions"]
    scales = scene_groups["scales"]
    rotations = scene_groups["rotations"]
    color_alpha = scene_groups["color_alpha"]
    expected_scales = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)]], dtype=np.float32)

    assert np.all(np.isfinite(positions))
    np.testing.assert_allclose(_actual_scale(scales[:, :3]), expected_scales, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(_actual_opacity(color_alpha[:, 3]), np.full((4,), 0.5, dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.all(np.abs(np.linalg.norm(rotations, axis=1) - 1.0) < 1e-3)
    for name in _ADAM_BUFFER_NAMES:
        np.testing.assert_allclose(
            np.frombuffer(trainer.adam_optimizer.buffers[name].to_numpy().tobytes(), dtype=np.float32)[: 8 * renderer.TRAINABLE_PARAM_COUNT],
            np.zeros((8 * renderer.TRAINABLE_PARAM_COUNT,), dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )


def test_training_frame_order_covers_each_view_once_per_epoch(device, tmp_path: Path):
    scene = _make_scene(count=12, seed=71)
    frames = [_make_frame(tmp_path, image_name=f"target_{idx}.png", image_id=idx, green_value=64 + idx) for idx in range(4)]
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=frames, seed=91)

    seen = []
    for _ in range(8):
        trainer.step()
        seen.append(int(trainer.state.last_frame_index))

    assert sorted(seen[:4]) == [0, 1, 2, 3]
    assert sorted(seen[4:]) == [0, 1, 2, 3]


def test_training_prepass_capacity_sync_runs_every_32_steps(device, tmp_path: Path, monkeypatch) -> None:
    frame = _make_frame(tmp_path, width=32, height=32, image_name="capacity_sync_target.png")
    trainer = object.__new__(GaussianTrainer)
    calls: list[object] = []
    trainer.renderer = SimpleNamespace(sync_prepass_capacity_for_current_scene=lambda sync_camera: calls.append(sync_camera) or False)
    trainer._PREPASS_CAPACITY_SYNC_INTERVAL = 32
    camera = frame.make_camera(near=0.1, far=20.0)

    for step in range(65):
        trainer._maybe_sync_prepass_capacity(camera, step)

    assert len(calls) == 3


def test_refinement_camera_buffer_refreshes_when_frame_pose_changes(device, tmp_path: Path) -> None:
    scene = _make_scene(count=4, seed=131)
    frame = _make_frame(tmp_path, image_name="refinement_pose_refresh.png", image_id=31)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    trainer._refresh_refinement_camera_buffer()
    before = buffer_to_numpy(trainer.refinement_buffers["camera_rows"], np.float32).copy()
    frame.t_xyz = np.array([0.25, -0.5, 3.0], dtype=np.float32)

    trainer._refresh_refinement_camera_buffer()
    after = buffer_to_numpy(trainer.refinement_buffers["camera_rows"], np.float32).copy()

    assert not np.array_equal(before, after)


def test_frame_target_rgba8_resizes_to_frame_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "frame_target_resize.png"
    Image.fromarray(np.full((6, 12, 3), 127, dtype=np.uint8), mode="RGB").save(image_path)
    frame = ColmapFrame(
        image_id=1,
        image_path=image_path,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        fx=40.0,
        fy=20.0,
        cx=2.0,
        cy=1.0,
        width=4,
        height=2,
    )
    trainer = object.__new__(GaussianTrainer)

    rgba8 = trainer._frame_target_rgba8(frame)

    assert rgba8.shape == (2, 4, 4)


def test_create_dataset_textures_threads_cpu_image_loading(monkeypatch) -> None:
    trainer = object.__new__(GaussianTrainer)
    trainer.frames = [SimpleNamespace(image_path=Path("a.png")), SimpleNamespace(image_path=Path("b.png"))]
    calls: list[object] = []

    class _Executor:
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            calls.append(("workers", max_workers, thread_name_prefix))

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> bool:
            return False

        def map(self, fn, items):
            return map(fn, items)

    monkeypatch.setattr(gaussian_trainer_module, "ThreadPoolExecutor", _Executor)
    monkeypatch.setattr(gaussian_trainer_module, "load_training_frame_rgba8", lambda frame: f"rgba:{frame.image_path.name}")
    trainer._create_gpu_texture = lambda rgba8: f"tex:{rgba8}"

    trainer._create_dataset_textures()

    assert calls == [("workers", 16, "trainer-target")]
    assert trainer._frame_targets_native == ["tex:rgba:a.png", "tex:rgba:b.png"]

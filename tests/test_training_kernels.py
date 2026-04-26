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
from src.training import AdamHyperParams, GaussianTrainer, SPLAT_CONTRIBUTION_FIXED_SCALE, StabilityHyperParams, TRAIN_BACKGROUND_MODE_CUSTOM, TRAIN_BACKGROUND_MODE_RANDOM, TrainingHyperParams, resolve_auto_train_subsample_factor, resolve_base_learning_rate, resolve_colorspace_mod, resolve_cosine_base_learning_rate, resolve_effective_refinement_interval, resolve_effective_train_render_factor, resolve_lr_schedule_breakpoints, resolve_max_allowed_density, resolve_max_visible_angle_deg, resolve_position_lr_mul, resolve_position_random_step_noise_lr, resolve_refinement_clone_budget, resolve_refinement_growth_ratio, resolve_refinement_min_contribution, resolve_sh_band, resolve_sh_lr_mul, resolve_sorting_order_dithering, resolve_ssim_weight, resolve_stage_schedule_steps, resolve_training_resolution, resolve_train_subsample_factor, resolve_use_sh, should_run_refinement_step

_ADAM_BUFFER_NAMES = ("adam_moments",)
_OPACITY_EPS = 1e-6
_raw_opacity = lambda alpha: (np.log(np.clip(np.asarray(alpha, dtype=np.float32), _OPACITY_EPS, 1.0 - _OPACITY_EPS)) - np.log1p(-np.clip(np.asarray(alpha, dtype=np.float32), _OPACITY_EPS, 1.0 - _OPACITY_EPS))).astype(np.float32, copy=False)
_actual_opacity = lambda raw: (1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float32)))).astype(np.float32, copy=False)
_actual_scale = lambda log_scale: np.exp(np.asarray(log_scale, dtype=np.float32))
_GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0
_REFINEMENT_MIN_SCREEN_RADIUS_PX = 0.05
_log_sigma = lambda sigma: np.log(np.asarray(sigma, dtype=np.float32))
_stored_from_support_scale = lambda support_scale: np.log(np.asarray(support_scale, dtype=np.float32) / _GAUSSIAN_SUPPORT_SIGMA_RADIUS)
_SCALE_GRAD_MULS = (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 0.9375, 0.96875, 1.0, 1.05)
_TARGET_MULS = (2.0, 3.0, 4.0, 6.0, 7.5)
_RNG_OPEN01_SCALE = np.float32(1.0 / 4294967297.0)
_SH_CLAMP_SAMPLE_COUNT = 8
_SH_CLAMP_HASH_SPLAT = np.uint32(0x68E31DA4)
_SH_CLAMP_HASH_SAMPLE = np.uint32(0xB5297A4D)
_SH_CLAMP_HASH_PHI = np.uint32(0x1B56C4E9)
_SH_CLAMP_HASH_Z = np.uint32(0x7F4A7C15)
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
_SSIM_C2 = 0.0009
_SSIM_C1 = _SSIM_C2 / 9.0
_SSIM_SMALL_VALUE = 1e-6
_SSIM_FEATURES_PER_COLOR = 5
_SSIM_FEATURE_CHANNELS = 3 * _SSIM_FEATURES_PER_COLOR
_OUTPUT_GAMMA = 2.2


def _hash_u32(x: int) -> np.uint32:
    value = int(x) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return np.uint32(value)


def _sh_clamp_random01(seed: int) -> np.float32:
    return np.float32((float(_hash_u32(seed)) + 1.0) * float(_RNG_OPEN01_SCALE))


def _sh_clamp_sample_dir(splat_id: int, sample_id: int) -> np.ndarray:
    splat_seed = _hash_u32((((splat_id + 1) * int(_SH_CLAMP_HASH_SPLAT)) & 0xFFFFFFFF) ^ (((sample_id + 1) * int(_SH_CLAMP_HASH_SAMPLE)) & 0xFFFFFFFF))
    z = np.float32(1.0 - 2.0 * float(_sh_clamp_random01(int(splat_seed ^ _SH_CLAMP_HASH_Z))))
    phi = np.float32(2.0 * np.pi * float(_sh_clamp_random01(int(splat_seed ^ _SH_CLAMP_HASH_PHI))))
    radial = np.float32(np.sqrt(max(1.0 - float(z * z), 0.0)))
    return np.array([radial * np.cos(phi), radial * np.sin(phi), z], dtype=np.float32)


def _training_target_to_linear_np(target_rgb: np.ndarray) -> np.ndarray:
    return np.power(np.maximum(np.asarray(target_rgb, dtype=np.float32), 0.0), np.float32(1.0 / _OUTPUT_GAMMA)).astype(np.float32, copy=False)


def _training_rendered_to_display_np(rendered_rgb: np.ndarray) -> np.ndarray:
    return np.power(np.maximum(np.asarray(rendered_rgb, dtype=np.float32), 0.0), np.float32(_OUTPUT_GAMMA)).astype(np.float32, copy=False)


def _training_loss_colorspace_np(rgb: np.ndarray, colorspace_mod: float) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    return (np.sign(values) * np.power(np.abs(values), np.float32(colorspace_mod))).astype(np.float32, copy=False)


def _expected_refinement_child_scale(parent_scale: np.ndarray, family_size: int, scale_mul: float = 1.0) -> np.ndarray:
    scale = np.asarray(parent_scale, dtype=np.float32).reshape(3)
    shrink = float(max(int(family_size), 1)) ** (-1.0 / 3.0) * float(scale_mul)
    return (scale * np.float32(shrink)).astype(np.float32, copy=False)


def _expected_refinement_child_scale_beta(parent_scale: np.ndarray, family_size: int, beta: float, scale_mul: float = 1.0) -> np.ndarray:
    scale = np.asarray(parent_scale, dtype=np.float32).reshape(3)
    shrink = float(max(int(family_size), 1)) ** (-float(beta)) * float(scale_mul)
    return (scale * np.float32(shrink)).astype(np.float32, copy=False)


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


def _make_frame_at_position(position: tuple[float, float, float], image_id: int) -> ColmapFrame:
    camera_position = np.asarray(position, dtype=np.float32)
    return ColmapFrame(
        image_id=image_id,
        image_path=Path(f"frame_{image_id}.png"),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=-camera_position,
        fx=72.0,
        fy=72.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
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


def _read_ssim_buffer(trainer: GaussianTrainer, name: str) -> np.ndarray:
    width = int(trainer.renderer.width)
    height = int(trainer.renderer.height)
    flat = buffer_to_numpy(trainer._buffers[name], np.float32)
    return np.asarray(flat[: max(width * height, 1) * _SSIM_FEATURE_CHANNELS], dtype=np.float32).reshape(height, width, _SSIM_FEATURE_CHANNELS).copy()


def _ssim_feature_moments_np(rendered_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    rendered = np.asarray(rendered_rgb, dtype=np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)
    channel_moments = [
        np.stack(
            (
                rendered[..., channel],
                target[..., channel],
                rendered[..., channel] * rendered[..., channel],
                target[..., channel] * target[..., channel],
                rendered[..., channel] * target[..., channel],
            ),
            axis=2,
        )
        for channel in range(3)
    ]
    return np.concatenate(channel_moments, axis=2).astype(np.float32, copy=False)


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
    colorspace_mod: float,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    rendered = np.asarray(rendered_rgba, dtype=np.float32)[..., :3]
    target_linear = _training_target_to_linear_np(np.asarray(target_rgba, dtype=np.float32)[..., :3])
    target_loss = _training_loss_colorspace_np(
        _training_target_to_linear_np(np.asarray(target_rgba, dtype=np.float32)[..., :3]),
        colorspace_mod,
    )
    alpha = np.asarray(target_rgba, dtype=np.float32)[..., 3]
    inv_pixel_count = np.float32(1.0 / max(rendered.shape[0] * rendered.shape[1], 1))
    mask = _training_target_mask_np(alpha, use_target_alpha_mask)
    loss_diff = rendered - target_loss
    mse_diff = rendered - target_linear
    l1 = np.mean(np.abs(loss_diff), axis=2).astype(np.float32) * inv_pixel_count * mask
    mse = float(np.sum(np.mean(mse_diff * mse_diff, axis=2).astype(np.float32) * inv_pixel_count * mask, dtype=np.float64))
    blurred = _separable_gaussian_blur_np(_ssim_feature_moments_np(rendered, target_loss))
    channel_dssim: list[np.ndarray] = []
    for channel in range(3):
        offset = channel * _SSIM_FEATURES_PER_COLOR
        x = blurred[..., offset + 0]
        y = blurred[..., offset + 1]
        x2 = blurred[..., offset + 2]
        y2 = blurred[..., offset + 3]
        xy = blurred[..., offset + 4]
        sigma_x2 = np.maximum(x2 - x * x, 0.0)
        sigma_y2 = np.maximum(y2 - y * y, 0.0)
        sigma_xy = xy - x * y
        numer = (2.0 * x * y + _SSIM_C1) * (2.0 * sigma_xy + _SSIM_C2)
        denom = np.maximum((x * x + y * y + _SSIM_C1) * (sigma_x2 + sigma_y2 + _SSIM_C2), _SSIM_SMALL_VALUE)
        channel_dssim.append(1.0 - numer / denom)
    dssim = np.mean(np.stack(channel_dssim, axis=2), axis=2, dtype=np.float32)
    dssim = dssim.astype(np.float32, copy=False) * inv_pixel_count * mask
    blended = ((1.0 - float(ssim_weight)) * l1 + float(ssim_weight) * dssim).astype(np.float32, copy=False)
    l1_grad = np.sign(loss_diff).astype(np.float32) * (np.float32(1.0 / 3.0) * inv_pixel_count * mask[..., None] * np.float32(1.0 - float(ssim_weight)))
    return blended, float(np.sum(blended, dtype=np.float64)), mse, l1_grad


def _display_mse_np(rendered_rgba: np.ndarray, target_rgba: np.ndarray, *, use_target_alpha_mask: bool, colorspace_mod: float) -> float:
    rendered = _training_rendered_to_display_np(np.asarray(rendered_rgba, dtype=np.float32)[..., :3])
    target = _training_rendered_to_display_np(
        _training_loss_colorspace_np(
            _training_target_to_linear_np(np.asarray(target_rgba, dtype=np.float32)[..., :3]),
            colorspace_mod,
        )
    )
    alpha = np.asarray(target_rgba, dtype=np.float32)[..., 3]
    inv_pixel_count = np.float32(1.0 / max(rendered.shape[0] * rendered.shape[1], 1))
    mask = _training_target_mask_np(alpha, use_target_alpha_mask)
    diff = rendered - target
    return float(np.sum(np.mean(diff * diff, axis=2).astype(np.float32) * inv_pixel_count * mask, dtype=np.float64))


def _torch_blended_rgb_grad_np(rendered_rgba: np.ndarray, target_rgba: np.ndarray, *, ssim_weight: float) -> np.ndarray:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    rendered = np.asarray(rendered_rgba, dtype=np.float32)
    target = np.asarray(target_rgba, dtype=np.float32)
    kernel = torch.tensor(_SSIM_BLUR_WEIGHTS, dtype=torch.float32)
    weight_h = kernel.view(1, 1, 1, -1).repeat(_SSIM_FEATURE_CHANNELS, 1, 1, 1)
    weight_v = kernel.view(1, 1, -1, 1).repeat(_SSIM_FEATURE_CHANNELS, 1, 1, 1)
    rendered_t = torch.tensor(rendered[..., :3].transpose(2, 0, 1)[None], dtype=torch.float32, requires_grad=True)
    target_t = torch.tensor(target[..., :4].transpose(2, 0, 1)[None], dtype=torch.float32)
    target_rgb = target_t[:, :3]
    target_mask = target_t[:, 3:4].clamp(0.0, 1.0)
    target_rgb = torch.pow(torch.clamp(target_rgb, min=0.0), 1.0 / _OUTPUT_GAMMA)
    inv_pixel_count = 1.0 / float(rendered.shape[0] * rendered.shape[1])
    moments = torch.cat(
        tuple(
            component
            for channel in range(3)
            for component in (
                rendered_t[:, channel : channel + 1],
                target_rgb[:, channel : channel + 1],
                rendered_t[:, channel : channel + 1] * rendered_t[:, channel : channel + 1],
                target_rgb[:, channel : channel + 1] * target_rgb[:, channel : channel + 1],
                rendered_t[:, channel : channel + 1] * target_rgb[:, channel : channel + 1],
            )
        ),
        dim=1,
    )
    blurred = F.conv2d(F.pad(moments, (5, 5, 0, 0), mode="replicate"), weight_h, groups=_SSIM_FEATURE_CHANNELS)
    blurred = F.conv2d(F.pad(blurred, (0, 0, 5, 5), mode="replicate"), weight_v, groups=_SSIM_FEATURE_CHANNELS)
    channel_dssim = []
    for channel in range(3):
        offset = channel * _SSIM_FEATURES_PER_COLOR
        x = blurred[:, offset + 0 : offset + 1]
        y = blurred[:, offset + 1 : offset + 2]
        x2 = blurred[:, offset + 2 : offset + 3]
        y2 = blurred[:, offset + 3 : offset + 4]
        xy = blurred[:, offset + 4 : offset + 5]
        sigma_x2 = torch.clamp(x2 - x * x, min=0.0)
        sigma_y2 = torch.clamp(y2 - y * y, min=0.0)
        sigma_xy = xy - x * y
        numer = (2.0 * x * y + _SSIM_C1) * (2.0 * sigma_xy + _SSIM_C2)
        denom = torch.clamp((x * x + y * y + _SSIM_C1) * (sigma_x2 + sigma_y2 + _SSIM_C2), min=_SSIM_SMALL_VALUE)
        channel_dssim.append(1.0 - numer / denom)
    dssim = torch.stack(channel_dssim, dim=1).mean(dim=1, keepdim=False)
    l1 = (rendered_t - target_rgb).abs().mean(dim=1, keepdim=True)
    loss = ((((1.0 - float(ssim_weight)) * l1) + (float(ssim_weight) * dssim)) * inv_pixel_count * target_mask).sum()
    loss.backward()
    return rendered_t.grad.detach().cpu().numpy()[0].transpose(1, 2, 0)


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
    renderer.work_buffers["training_density"].copy_from_numpy(np.zeros((renderer.width * renderer.height,), dtype=np.float32))
    target_texture = _make_float_texture(trainer.device, target)
    enc = trainer.device.create_command_encoder()
    trainer._dispatch_loss_forward(enc, target_texture)
    if run_backward:
        trainer._dispatch_loss_backward(enc, target_texture)
    trainer.device.submit_command_buffer(enc.finish())
    trainer.device.wait()
    return target_texture


def _read_optimizer_lrs(trainer: GaussianTrainer) -> np.ndarray:
    flat = buffer_to_numpy(trainer.optimizer.param_settings, np.uint32)
    return flat.reshape(trainer.renderer.TRAINABLE_PARAM_COUNT, 8)[:, 0].view(np.float32).copy()


def _read_adam_moments(trainer: GaussianTrainer, splat_count: int) -> np.ndarray:
    flat = buffer_to_numpy(trainer.adam_optimizer.buffers["adam_moments"], np.float32)
    count = max(int(splat_count), 1) * trainer.renderer.TRAINABLE_PARAM_COUNT * 2
    return flat[:count].reshape(trainer.renderer.TRAINABLE_PARAM_COUNT, max(int(splat_count), 1), 2).copy()


def _circle_bound_support_radius(camera, position: np.ndarray, width: int, height: int, radius_px: float) -> float:
    del width, height
    focal = np.asarray(camera.focal_pixels_xy(1, 1), dtype=np.float32)
    min_focal = max(float(min(focal[0], focal[1])), 1e-8)
    camera_distance = float(np.linalg.norm(np.asarray(position, dtype=np.float32) - np.asarray(camera.position, dtype=np.float32)))
    return float(camera_distance * float(radius_px) / min_focal)


def _legacy_screen_fraction_to_visible_angle_deg(screen_fraction: float) -> float:
    fraction = max(float(screen_fraction), 1e-8)
    return float(np.degrees(np.arctan(2.0 * np.sqrt(fraction / np.pi))))


def _view_angle_cap_expected(
    *,
    scale: np.ndarray,
    position: np.ndarray,
    camera,
    renderer: GaussianRenderer,
    max_visible_angle_deg: float,
) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float32).reshape(3)
    camera_distance = float(np.linalg.norm(np.asarray(position, dtype=np.float32) - np.asarray(camera.position, dtype=np.float32)))
    max_support_radius = camera_distance * float(np.tan(np.radians(float(max_visible_angle_deg))))
    max_sigma = max_support_radius / max(float(renderer.radius_scale) * _GAUSSIAN_SUPPORT_SIGMA_RADIUS, 1e-8)
    return np.minimum(scale, np.full((3,), max_sigma, dtype=np.float32)).astype(np.float32, copy=False)


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


def test_optimizer_screen_scale_cap_skips_clamp_inside_camera_min_distance(device, tmp_path: Path):
    initial_sigma = np.array([1.5, 1.5, 1.5], dtype=np.float32)

    def _make_single_scene() -> GaussianScene:
        return GaussianScene(
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            scales=_log_sigma(initial_sigma[None, :]),
            rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            opacities=np.array([0.8], dtype=np.float32),
            colors=np.array([[0.8, 0.7, 0.6]], dtype=np.float32),
            sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
        )

    frame = _make_frame(tmp_path, image_name="screen_scale_cap_target.png", image_id=23)

    def _project_once(camera_min_dist: float) -> np.ndarray:
        scene = _make_single_scene()
        renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=16)
        trainer = GaussianTrainer(
            device=device,
            renderer=renderer,
            scene=scene,
            frames=[frame],
            training_hparams=TrainingHyperParams(camera_min_dist=camera_min_dist, max_visible_angle_deg=0.5),
            seed=123,
        )
        camera = trainer.make_frame_camera(0, renderer.width, renderer.height)
        encoder = device.create_command_encoder()
        trainer.optimizer.dispatch_projection(
            encoder,
            scene_buffers=renderer.scene_buffers,
            work_buffers=renderer.work_buffers,
            splat_count=scene.count,
            training_hparams=trainer.training,
            scale_reg_reference=trainer._scale_reg_reference,
            frame_camera=camera,
            width=renderer.width,
            height=renderer.height,
            step_index=0,
        )
        device.submit_command_buffer(encoder.finish())
        device.wait()
        return _actual_scale(_read_scene_groups(renderer, scene.count)["scales"][:, :3])[0]

    clamped_scale = _project_once(0.0)
    skipped_scale = _project_once(4.0)

    assert np.all(clamped_scale < initial_sigma * 0.1)
    np.testing.assert_allclose(skipped_scale, initial_sigma, rtol=1e-6, atol=1e-6)


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
        colorspace_mod=resolve_colorspace_mod(trainer.training, 0),
    )
    total, mse, density = trainer._read_loss_metrics()
    np.testing.assert_allclose(_read_training_rgb_loss(trainer.renderer), expected_rgb_loss, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(_read_training_rgb_loss_total(trainer.renderer), expected_total, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(total, expected_total, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(mse, expected_mse, rtol=0.0, atol=1e-7)
    assert density == 0.0
    np.testing.assert_allclose(_read_output_grads(trainer.renderer)[..., :3], expected_grad, rtol=0.0, atol=1e-7)

    enc = device.create_command_encoder()
    trainer._dispatch_cache_step_info(enc, 0)
    device.submit_command_buffer(enc.finish())
    device.wait()

    batch_metrics = trainer._read_batch_step_metrics(1)
    np.testing.assert_allclose(batch_metrics[0, trainer._LOSS_SLOT_TOTAL], expected_total, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(batch_metrics[0, trainer._LOSS_SLOT_MSE], expected_mse, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        batch_metrics[0, trainer._BATCH_STEP_INFO_DISPLAY_MSE],
        _display_mse_np(
            rendered,
            target,
            use_target_alpha_mask=False,
            colorspace_mod=resolve_colorspace_mod(trainer.training, 0),
        ),
        rtol=0.0,
        atol=1e-7,
    )


def test_display_mse_uses_loss_colorspace_target(device, tmp_path: Path):
    trainer = _make_loss_only_trainer(
        device,
        tmp_path,
        width=2,
        height=1,
        training_hparams=TrainingHyperParams(
            background_mode=TRAIN_BACKGROUND_MODE_CUSTOM,
            background=(0.0, 0.0, 0.0),
            density_regularizer=0.0,
            ssim_weight=0.0,
            colorspace_mod=0.5,
            colorspace_mod_stage1=0.5,
            colorspace_mod_stage2=0.5,
            colorspace_mod_stage3=0.5,
        ),
        image_name="display_mse_colorspace_target.png",
        image_id=32,
    )
    rendered = np.array(
        [[[0.25, 0.49, 0.81, 1.0], [0.36, 0.64, 0.16, 1.0]]],
        dtype=np.float32,
    )
    target = np.array(rendered, copy=True)
    target[..., :3] = _training_rendered_to_display_np(rendered[..., :3])

    _dispatch_manual_loss(trainer, rendered, target)

    enc = device.create_command_encoder()
    trainer._dispatch_cache_step_info(enc, 0)
    device.submit_command_buffer(enc.finish())
    device.wait()

    batch_metrics = trainer._read_batch_step_metrics(1)
    expected_display_mse = _display_mse_np(
        rendered,
        target,
        use_target_alpha_mask=False,
        colorspace_mod=resolve_colorspace_mod(trainer.training, 0),
    )

    assert expected_display_mse > 0.0
    np.testing.assert_allclose(batch_metrics[0, trainer._BATCH_STEP_INFO_DISPLAY_MSE], expected_display_mse, rtol=0.0, atol=1e-7)


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
            ssim_weight=1.0,
            colorspace_mod=1.0,
            colorspace_mod_stage1=1.0,
            colorspace_mod_stage2=1.0,
            colorspace_mod_stage3=1.0,
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

    expected_moments = _ssim_feature_moments_np(rendered[..., :3], _training_target_to_linear_np(target[..., :3]))
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
            ssim_weight=1.0,
            colorspace_mod=1.0,
            colorspace_mod_stage1=1.0,
            colorspace_mod_stage2=1.0,
            colorspace_mod_stage3=1.0,
        ),
        image_name="ssim_identical_target.png",
        image_id=33,
    )
    rendered = np.linspace(0.0, 1.0, num=5 * 5 * 4, dtype=np.float32).reshape(5, 5, 4)
    rendered[..., 3] = 1.0
    target = rendered.copy()
    target[..., :3] = _training_rendered_to_display_np(rendered[..., :3])

    _dispatch_manual_loss(trainer, rendered, target)

    total, mse, density = trainer._read_loss_metrics()
    np.testing.assert_allclose(total, 0.0, rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(mse, 0.0, rtol=0.0, atol=5e-6)
    assert density == 0.0
    np.testing.assert_allclose(_read_training_rgb_loss_total(trainer.renderer), 0.0, rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(_read_output_grads(trainer.renderer)[..., :3], 0.0, rtol=0.0, atol=2e-5)


def test_ssim_backward_matches_torch_image_gradients(device, tmp_path: Path):
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
            ssim_weight=0.2,
            colorspace_mod=1.0,
            colorspace_mod_stage1=1.0,
            colorspace_mod_stage2=1.0,
            colorspace_mod_stage3=1.0,
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
    torch_grad = _torch_blended_rgb_grad_np(rendered, target, ssim_weight=trainer.training.ssim_weight)

    np.testing.assert_allclose(gpu_grad, torch_grad, rtol=0.0, atol=1.6e-3)


def test_ssim_backward_matches_torch_garden_blur_distortion(device, tmp_path: Path):
    image_path = Path(__file__).resolve().parent.parent / "dataset" / "garden" / "images" / "DSC07956.JPG"
    if not image_path.exists():
        pytest.skip("garden dataset image unavailable")

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    crop = rgb[height // 2 - 32 : height // 2 + 32, width // 2 - 32 : width // 2 + 32].copy()
    target = np.ones((crop.shape[0], crop.shape[1], 4), dtype=np.float32)
    target[..., :3] = crop
    rendered = target.copy()
    rendered[..., :3] = _separable_gaussian_blur_np(crop)

    trainer = _make_loss_only_trainer(
        device,
        tmp_path,
        width=target.shape[1],
        height=target.shape[0],
        training_hparams=TrainingHyperParams(
            background_mode=TRAIN_BACKGROUND_MODE_CUSTOM,
            background=(0.0, 0.0, 0.0),
            density_regularizer=0.0,
            ssim_weight=1.0,
            colorspace_mod=1.0,
            colorspace_mod_stage1=1.0,
            colorspace_mod_stage2=1.0,
            colorspace_mod_stage3=1.0,
        ),
        image_name="ssim_garden_blur_target.png",
        image_id=38,
    )

    _dispatch_manual_loss(trainer, rendered, target)
    gpu_grad = _read_output_grads(trainer.renderer)[..., :3]
    torch_grad = _torch_blended_rgb_grad_np(rendered, target, ssim_weight=1.0)

    np.testing.assert_allclose(gpu_grad, torch_grad, rtol=0.0, atol=2e-5)


def test_ssim_blurred_gradients_leave_target_side_moments_zero(device, tmp_path: Path):
    trainer = _make_loss_only_trainer(
        device,
        tmp_path,
        width=8,
        height=8,
        training_hparams=TrainingHyperParams(
            ssim_weight=1.0,
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            colorspace_mod=1.0,
            colorspace_mod_stage1=1.0,
            colorspace_mod_stage2=1.0,
            colorspace_mod_stage3=1.0,
        ),
        image_name="ssim_luma_only_target.png",
        image_id=35,
    )
    rng = np.random.default_rng(35)
    rendered = rng.uniform(0.02, 0.98, size=(8, 8, 4)).astype(np.float32)
    target = rng.uniform(0.02, 0.98, size=(8, 8, 4)).astype(np.float32)
    rendered[..., 3] = 1.0
    target[..., 3] = 1.0

    _dispatch_manual_loss(trainer, rendered, target)
    blurred_grads = _read_ssim_buffer(trainer, "ssim_blurred_feature_grads")
    assert blurred_grads.shape[-1] == _SSIM_FEATURE_CHANNELS
    for channel in range(3):
        offset = channel * _SSIM_FEATURES_PER_COLOR
        np.testing.assert_allclose(blurred_grads[..., offset + 1], 0.0, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(blurred_grads[..., offset + 3], 0.0, rtol=0.0, atol=1e-7)


def test_ssim_keeps_neutral_black_gradients_neutral(device, tmp_path: Path):
    trainer = _make_loss_only_trainer(
        device,
        tmp_path,
        width=8,
        height=8,
        training_hparams=TrainingHyperParams(
            ssim_weight=0.4,
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            colorspace_mod=1.0,
            colorspace_mod_stage1=1.0,
            colorspace_mod_stage2=1.0,
            colorspace_mod_stage3=1.0,
        ),
        image_name="ssim_neutral_black_target.png",
        image_id=36,
    )
    rendered = np.full((8, 8, 4), 0.1, dtype=np.float32)
    target = np.zeros((8, 8, 4), dtype=np.float32)
    rendered[..., 3] = 1.0
    target[..., 3] = 1.0

    _dispatch_manual_loss(trainer, rendered, target)
    gpu_grad = _read_output_grads(trainer.renderer)[..., :3]

    assert np.any(np.abs(gpu_grad) > 1e-7)
    np.testing.assert_allclose(gpu_grad[..., 0], gpu_grad[..., 1], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(gpu_grad[..., 1], gpu_grad[..., 2], rtol=0.0, atol=1e-7)


def test_full_ssim_penalizes_flat_mean_only_chroma_shift(device, tmp_path: Path):
    trainer = _make_loss_only_trainer(
        device,
        tmp_path,
        width=8,
        height=8,
        training_hparams=TrainingHyperParams(
            ssim_weight=1.0,
            scale_l2_weight=0.0,
            scale_abs_reg_weight=0.0,
            opacity_reg_weight=0.0,
            sh1_reg_weight=0.0,
            density_regularizer=0.0,
            colorspace_mod=1.0,
            colorspace_mod_stage1=1.0,
            colorspace_mod_stage2=1.0,
            colorspace_mod_stage3=1.0,
        ),
        image_name="ssim_equal_mean_chroma_shift.png",
        image_id=37,
    )
    rendered = np.zeros((8, 8, 4), dtype=np.float32)
    target = np.zeros((8, 8, 4), dtype=np.float32)
    rendered[..., :3] = np.array([0.0, 0.3, 0.0], dtype=np.float32)
    target[..., :3] = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    rendered[..., 3] = 1.0
    target[..., 3] = 1.0

    _dispatch_manual_loss(trainer, rendered, target)

    total, mse, density = trainer._read_loss_metrics()
    grads = _read_output_grads(trainer.renderer)[..., :3]

    expected_loss, expected_total, _, _ = _blended_rgb_metrics_np(
        rendered,
        target,
        ssim_weight=1.0,
        use_target_alpha_mask=False,
        colorspace_mod=resolve_colorspace_mod(trainer.training, 0),
    )

    assert total > 0.0
    assert mse > 0.0
    assert density == 0.0
    np.testing.assert_allclose(total, expected_total, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(_read_training_rgb_loss(trainer.renderer), expected_loss, rtol=1e-5, atol=1e-7)
    assert float(np.max(np.abs(grads))) > 0.0


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


def test_training_raster_output_stays_linear_while_display_render_uses_gamma(device, tmp_path: Path):
    scene = _make_scene(count=1, seed=124)
    scene.opacities[:] = 0.0
    frame = _make_frame(tmp_path, image_name="linear_training_output_target.png", image_id=39)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            background_mode=TRAIN_BACKGROUND_MODE_CUSTOM,
            background=(0.25, 0.5, 0.75),
            density_regularizer=0.0,
        ),
        seed=123,
    )
    camera = frame.make_camera(near=0.1, far=20.0)
    background = np.array([0.25, 0.5, 0.75], dtype=np.float32)

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    enc = device.create_command_encoder()
    trainer._dispatch_raster_training_forward(enc, camera, background)
    device.submit_command_buffer(enc.finish())
    device.wait()

    training_output = np.asarray(renderer.output_texture.to_numpy(), dtype=np.float32)
    expected_training = np.broadcast_to(np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32), training_output.shape)
    np.testing.assert_allclose(training_output, expected_training, rtol=0.0, atol=1e-6)

    display_output = np.asarray(renderer.render(scene, camera, background=background).image, dtype=np.float32)
    expected_display = np.broadcast_to(np.array([0.25**2.2, 0.5**2.2, 0.75**2.2, 1.0], dtype=np.float32), display_output.shape)
    np.testing.assert_allclose(display_output, expected_display, rtol=0.0, atol=1e-6)


def test_colorspace_noise_and_sh_schedules_follow_requested_defaults() -> None:
    hparams = TrainingHyperParams(
        colorspace_mod=0.5,
        ssim_weight=0.2,
        max_visible_angle_deg=_legacy_screen_fraction_to_visible_angle_deg(0.3),
        position_random_step_noise_lr=1234.0,
        sh_band=2,
        use_sh=True,
        lr_schedule_enabled=False,
        lr_schedule_steps=30_000,
    )

    np.testing.assert_allclose(resolve_ssim_weight(hparams, 30_000), 0.2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_colorspace_mod(hparams, 0), 0.5, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_colorspace_mod(hparams, 30_000), 0.5, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_max_visible_angle_deg(hparams, 0), _legacy_screen_fraction_to_visible_angle_deg(0.3), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_max_visible_angle_deg(hparams, 30_000), _legacy_screen_fraction_to_visible_angle_deg(0.3), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 0), 1234.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_position_random_step_noise_lr(hparams, 30_000), 1234.0, rtol=0.0, atol=1e-12)
    assert resolve_sh_band(hparams, 0) == 2
    assert resolve_sh_band(hparams, 30_000) == 2
    assert resolve_use_sh(hparams, 0) is True
    assert resolve_use_sh(hparams, 30_000) is True


def test_colorspace_mod_schedule_interpolates_across_stages() -> None:
    hparams = TrainingHyperParams(
        colorspace_mod=0.5,
        colorspace_mod_stage1=0.75,
        colorspace_mod_stage2=0.9,
        colorspace_mod_stage3=1.0,
        lr_schedule_enabled=True,
        lr_schedule_steps=100,
        lr_schedule_stage1_step=25,
        lr_schedule_stage2_step=75,
    )

    np.testing.assert_allclose(resolve_colorspace_mod(hparams, 0), 0.5, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_colorspace_mod(hparams, 25), 0.75, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_colorspace_mod(hparams, 50), 0.825, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_colorspace_mod(hparams, 75), 0.9, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resolve_colorspace_mod(hparams, 100), 1.0, rtol=0.0, atol=1e-12)


def test_max_allowed_density_schedule_clamps_to_end_value() -> None:
    hparams = TrainingHyperParams(max_allowed_density_start=5.0, max_allowed_density=12.0, lr_schedule_steps=4)

    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 0), 5.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 2), 8.5, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 4), 12.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(resolve_max_allowed_density(hparams, 40), 12.0, rtol=0.0, atol=1e-10)


def test_refinement_cadence_and_clone_budget_follow_growth_budget() -> None:
    hparams = TrainingHyperParams(refinement_interval=200, refinement_growth_ratio=0.035, refinement_growth_start_step=0)

    assert not should_run_refinement_step(hparams, 199)
    assert should_run_refinement_step(hparams, 200)
    assert resolve_refinement_clone_budget(hparams, splat_count=1000, step=200) == 35


def test_refinement_interval_is_floored_by_frame_count() -> None:
    hparams = TrainingHyperParams(refinement_interval=2, refinement_growth_ratio=0.035, refinement_growth_start_step=0)

    assert resolve_effective_refinement_interval(hparams, frame_count=5) == 5
    assert not should_run_refinement_step(hparams, 4, frame_count=5)
    assert should_run_refinement_step(hparams, 5, frame_count=5)
    assert resolve_refinement_clone_budget(hparams, splat_count=1000, step=5, frame_count=5) == 35


def test_refinement_growth_stays_zero_until_start_step() -> None:
    hparams = TrainingHyperParams(refinement_interval=200, refinement_growth_ratio=0.02, refinement_growth_start_step=2000)

    np.testing.assert_allclose(resolve_refinement_growth_ratio(hparams, 1999), 0.0, rtol=0.0, atol=1e-12)
    assert resolve_refinement_clone_budget(hparams, splat_count=1000, step=1999) == 0
    np.testing.assert_allclose(resolve_refinement_growth_ratio(hparams, 2000), 0.02, rtol=0.0, atol=1e-12)
    assert resolve_refinement_clone_budget(hparams, splat_count=1000, step=2000) == 20


def test_refinement_min_contribution_uses_configured_decay() -> None:
    hparams = TrainingHyperParams(refinement_interval=200, refinement_min_contribution=512, refinement_min_contribution_decay=0.95)

    assert resolve_refinement_min_contribution(hparams, 0) == 512
    assert resolve_refinement_min_contribution(hparams, 199) == 512
    assert resolve_refinement_min_contribution(hparams, 200) == 486
    assert resolve_refinement_min_contribution(hparams, 400) == 462


def test_refinement_clone_budget_respects_max_gaussians_cap() -> None:
    hparams = TrainingHyperParams(refinement_interval=200, refinement_growth_ratio=0.05, refinement_growth_start_step=0, max_gaussians=1024)

    assert resolve_refinement_clone_budget(hparams, splat_count=1000, step=200) == 24
    assert resolve_refinement_clone_budget(hparams, splat_count=1024, step=200) == 0


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

    assert set(trainer.refinement_buffers) == {
        "total_clone_counter",
        "clone_counts",
        "splat_contribution",
        "splat_age",
        "refinement_eligible_mask",
        "refinement_weights",
        "refinement_weight_prefix",
        "refinement_weight_total",
        "append_counter",
        "append_params",
        "append_splat_age",
        "dst_splat_params",
        "dst_splat_age",
        "camera_rows",
    }
    assert trainer.refinement_clone_budget() > 0


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


def test_training_raster_path_preserves_clone_counts(device, tmp_path: Path) -> None:
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
    seed_counts = np.arange(scene.count, dtype=np.uint32)
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(seed_counts)

    renderer.execute_prepass_for_current_scene(camera, sync_counts=False)
    enc = device.create_command_encoder()
    renderer.rasterize_training_forward_current_scene(
        enc,
        camera,
        background,
        clone_counts_buffer=trainer.refinement_buffers["clone_counts"],
        splat_contribution_buffer=trainer.refinement_buffers["splat_contribution"],
    )
    device.submit_command_buffer(enc.finish())
    device.wait()

    clone_counts = buffer_to_numpy(trainer.refinement_buffers["clone_counts"], np.uint32)[: scene.count]
    contributions = buffer_to_numpy(trainer.refinement_buffers["splat_contribution"], np.uint32)[: scene.count]
    np.testing.assert_array_equal(clone_counts, seed_counts)
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
    )
    device.submit_command_buffer(enc.finish())
    device.wait()

    clone_counts = buffer_to_numpy(trainer.refinement_buffers["clone_counts"], np.uint32)[: scene.count]
    np.testing.assert_array_equal(clone_counts, seed_counts)


def _write_refinement_distribution_inputs(trainer: GaussianTrainer, variances: np.ndarray, contributions: np.ndarray | None = None) -> None:
    variances = np.ascontiguousarray(variances, dtype=np.float32).reshape(-1)
    contributions = np.ones_like(variances) if contributions is None else np.ascontiguousarray(contributions, dtype=np.float32).reshape(-1)
    samples = np.sqrt(np.maximum(variances, 0.0)).astype(np.float32, copy=False)
    stats = np.zeros((variances.shape[0], 2), dtype=np.float32)
    stats[:, 0] = samples * 2.0
    stats[:, 1] = samples * samples * 4.0
    trainer._observed_contribution_pixel_count = 2
    trainer.refinement_buffers["gradient_stats"].copy_from_numpy(stats)
    contribution_fixed = np.maximum(
        np.rint(np.maximum(contributions, 0.0) * trainer.observed_contribution_pixel_count * SPLAT_CONTRIBUTION_FIXED_SCALE),
        0.0,
    ).astype(np.uint32)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(contribution_fixed)


def test_refinement_sampling_prefers_higher_gradient_variance(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=91)
    scene.opacities[:] = np.array([0.9, 0.9], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="variance_sampling_target.png", image_id=18)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_growth_ratio=0.5, refinement_growth_start_step=0, refinement_interval=9999, refinement_min_contribution=50),
        seed=123,
    )
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.full((scene.count,), 500, dtype=np.uint32))
    _write_refinement_distribution_inputs(trainer, np.array([1.0, 0.05 * 0.05], dtype=np.float32))

    selections = np.zeros((scene.count,), dtype=np.int32)
    for seed in range(64):
        trainer._seed = seed
        clone_counts, _ = trainer._sample_refinement_clone_counts()
        selections += clone_counts.astype(np.int32)

    assert selections[0] > selections[1]


def test_gradient_stats_accumulate_raster_contribution_squares_and_clear(device, tmp_path: Path) -> None:
    scene = _make_scene(count=4, seed=190)
    frame = _make_frame(tmp_path, image_name="gradient_stats_target.png", image_id=190)
    renderer = GaussianRenderer(device, width=24, height=24, radius_scale=1.6, list_capacity_multiplier=16)
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], seed=123)

    trainer.step()

    stats = buffer_to_numpy(trainer.refinement_buffers["gradient_stats"], np.float32).reshape(-1, 2)[: scene.count]
    assert np.any(stats[:, 0] > 0.0)
    assert np.any(stats[:, 1] > 0.0)
    assert trainer.observed_contribution_pixel_count == renderer.width * renderer.height

    trainer._clear_clone_counts()
    cleared = buffer_to_numpy(trainer.refinement_buffers["gradient_stats"], np.float32).reshape(-1, 2)[: scene.count]
    np.testing.assert_allclose(cleared, np.zeros((scene.count, 2), dtype=np.float32), rtol=0.0, atol=0.0)
    assert trainer.observed_contribution_pixel_count == 0


def test_refinement_sampling_exponent_controls_variance_spikiness(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=196)
    scene.opacities[:] = np.array([0.9, 0.9], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="variance_exponent_target.png", image_id=196)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            refinement_growth_ratio=5.0,
            refinement_growth_start_step=0,
            refinement_interval=9999,
            refinement_grad_variance_weight_exponent=2.0,
            refinement_min_contribution=50,
        ),
        seed=123,
    )
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.full((scene.count,), 500, dtype=np.uint32))
    _write_refinement_distribution_inputs(trainer, np.array([1.0, 2.0], dtype=np.float32))

    selections = np.zeros((scene.count,), dtype=np.int32)
    for seed in range(64):
        trainer._seed = seed
        clone_counts, survivor_mask = trainer._sample_refinement_clone_counts()
        assert int(np.count_nonzero(survivor_mask)) == scene.count
        selections += clone_counts.astype(np.int32)

    ratio = float(selections[1]) / max(float(selections[0]), 1.0)
    assert 3.0 < ratio < 5.5


def test_refinement_sampling_exponent_controls_contribution_spikiness(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=198)
    scene.opacities[:] = np.array([0.9, 0.9], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="contribution_exponent_target.png", image_id=198)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            refinement_growth_ratio=5.0,
            refinement_growth_start_step=0,
            refinement_interval=9999,
            refinement_grad_variance_weight_exponent=0.0,
            refinement_contribution_weight_exponent=2.0,
            refinement_min_contribution=50,
        ),
        seed=123,
    )
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.full((scene.count,), 500, dtype=np.uint32))
    _write_refinement_distribution_inputs(trainer, np.ones((scene.count,), dtype=np.float32), np.array([1.0, 2.0], dtype=np.float32))

    selections = np.zeros((scene.count,), dtype=np.int32)
    for seed in range(64):
        trainer._seed = seed
        clone_counts, survivor_mask = trainer._sample_refinement_clone_counts()
        assert int(np.count_nonzero(survivor_mask)) == scene.count
        selections += clone_counts.astype(np.int32)

    ratio = float(selections[1]) / max(float(selections[0]), 1.0)
    assert 3.0 < ratio < 5.5


def test_refinement_sampling_is_seed_reproducible(device, tmp_path: Path) -> None:
    scene = _make_scene(count=3, seed=92)
    scene.opacities[:] = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="variance_seed_target.png", image_id=19)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_growth_ratio=1.0, refinement_growth_start_step=0, refinement_interval=9999, refinement_min_contribution=50),
        seed=123,
    )
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.full((scene.count,), 500, dtype=np.uint32))
    _write_refinement_distribution_inputs(trainer, np.array([1.0, 0.25, 0.0625], dtype=np.float32))

    trainer._seed = 77
    first, _ = trainer._sample_refinement_clone_counts()
    trainer._seed = 77
    second, _ = trainer._sample_refinement_clone_counts()
    np.testing.assert_array_equal(first, second)


def test_refinement_sampling_respects_budget_and_clone_cap(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=93)
    scene.opacities[:] = np.array([0.9, 0.9], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="variance_cap_target.png", image_id=20)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_growth_ratio=5.0, refinement_growth_start_step=0, refinement_interval=9999, max_gaussians=64, refinement_min_contribution=50),
        seed=123,
    )
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.full((scene.count,), 500, dtype=np.uint32))
    _write_refinement_distribution_inputs(trainer, np.ones((scene.count,), dtype=np.float32))

    clone_counts, survivor_mask = trainer._sample_refinement_clone_counts()

    assert int(np.count_nonzero(survivor_mask)) == 2
    assert int(np.sum(clone_counts, dtype=np.int64)) == trainer.refinement_clone_budget()
    assert np.all(clone_counts <= 8)


def test_refinement_sampling_zero_variance_yields_zero_clone_counts(device, tmp_path: Path) -> None:
    scene = _make_scene(count=4, seed=94)
    scene.opacities[:] = np.full((scene.count,), 0.9, dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="variance_zero_target.png", image_id=21)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_growth_ratio=1.0, refinement_growth_start_step=0, refinement_interval=9999, refinement_min_contribution=50),
        seed=123,
    )
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.full((scene.count,), 500, dtype=np.uint32))
    _write_refinement_distribution_inputs(trainer, np.zeros((scene.count,), dtype=np.float32))

    clone_counts, survivor_mask = trainer._sample_refinement_clone_counts()

    assert int(np.count_nonzero(survivor_mask)) == scene.count
    assert np.all(clone_counts == 0)


def test_refinement_sampling_single_nonzero_weight_always_selects_that_splat(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=96)
    scene.opacities[:] = np.full((scene.count,), 0.9, dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="variance_single_nonzero_target.png", image_id=23)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_growth_ratio=1.0, refinement_growth_start_step=0, refinement_interval=9999, refinement_min_contribution=50),
        seed=123,
    )
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.full((scene.count,), 500, dtype=np.uint32))

    selected_splat = 5
    variances = np.zeros((scene.count,), dtype=np.float32)
    variances[selected_splat] = 1.0
    _write_refinement_distribution_inputs(trainer, variances)

    expected = np.zeros((scene.count,), dtype=np.uint32)
    expected[selected_splat] = 8
    for seed in range(16):
        trainer._seed = seed
        clone_counts, survivor_mask = trainer._sample_refinement_clone_counts()
        assert int(np.count_nonzero(survivor_mask)) == scene.count
        np.testing.assert_array_equal(clone_counts, expected)


def test_refinement_sampling_routes_all_mass_to_single_positive_weight(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=95)
    scene.opacities[:] = np.array([0.9, 0.9], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="variance_single_mass_target.png", image_id=22)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_growth_ratio=1.5, refinement_growth_start_step=0, refinement_interval=9999, refinement_min_contribution=50),
        seed=123,
    )
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.full((scene.count,), 500, dtype=np.uint32))
    _write_refinement_distribution_inputs(trainer, np.array([1.0, 0.0], dtype=np.float32))

    clone_counts, survivor_mask = trainer._sample_refinement_clone_counts()

    assert int(np.count_nonzero(survivor_mask)) == 2
    assert clone_counts[0] == trainer.refinement_clone_budget()
    assert clone_counts[1] == 0


def test_refinement_growth_ratio_realizes_full_budget_when_mass_concentrates(device, tmp_path: Path) -> None:
    scene = _make_scene(count=8, seed=197)
    scene.opacities[:] = np.full((scene.count,), 0.9, dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="refinement_growth_budget_target.png", image_id=197)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_growth_ratio=1.0, refinement_growth_start_step=0, refinement_interval=9999, refinement_min_contribution=50),
        seed=123,
    )
    trainer._observed_contribution_pixel_count = renderer.width * renderer.height
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.full((scene.count,), 500, dtype=np.uint32))

    selected_splat = 3
    variances = np.zeros((scene.count,), dtype=np.float32)
    variances[selected_splat] = 1.0
    _write_refinement_distribution_inputs(trainer, variances)

    budget = trainer.refinement_clone_budget()
    trainer._run_refinement()

    assert trainer.scene.count == scene.count + min(budget, 8)


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
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-3, refinement_min_contribution=50),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([2, 5], dtype=np.uint32)
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200, 0], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

    groups = _read_scene_groups(renderer, trainer.scene.count)
    actual_scales = _actual_scale(groups["scales"][:, :3])
    actual_opacity = _actual_opacity(groups["color_alpha"][:, 3])
    parent_scale = np.array([0.09, 0.06, 0.03], dtype=np.float32)

    assert trainer.scene.count == 3
    np.testing.assert_allclose(actual_scales, np.repeat(actual_scales[:1], 3, axis=0), rtol=0.0, atol=1e-6)
    assert np.all(actual_scales < parent_scale[None, :])
    np.testing.assert_allclose(
        np.max(actual_scales, axis=1) / np.min(actual_scales, axis=1),
        np.full((3,), 3.0, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(actual_opacity, np.full((3,), actual_opacity[0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert 0.25 < float(actual_opacity[0]) <= 0.600001
    np.testing.assert_allclose(np.mean(groups["positions"][:, :3], axis=0), source_position, rtol=0.0, atol=2e-3)
    offsets = groups["positions"][:, :3] - source_position[None, :]
    np.testing.assert_allclose(np.sum(offsets, axis=0), np.zeros((3,), dtype=np.float32), rtol=0.0, atol=6e-3)
    assert float(np.max(np.linalg.norm(offsets, axis=1))) > 1e-3
    clone_counts_after = buffer_to_numpy(trainer.refinement_buffers["clone_counts"], np.uint32)[: trainer.scene.count]
    contribution_after = buffer_to_numpy(trainer.refinement_buffers["splat_contribution"], np.uint32)[: trainer.scene.count]
    assert np.all(clone_counts_after == 0)
    assert np.all(contribution_after == 0)


def test_refinement_clone_scale_mul_scales_split_family_sigma(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=90)
    scene.opacities[:] = np.array([0.6], dtype=np.float32)
    scene.scales[:] = _log_sigma(np.array([[0.09, 0.06, 0.03]], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="refinement_clone_scale_mul_target.png", image_id=111)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            refinement_alpha_cull_threshold=1e-6,
            refinement_min_contribution=50,
            refinement_clone_scale_mul=1.5,
        ),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([1], dtype=np.uint32)
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

    assert trainer.scene.count == 2
    groups = _read_scene_groups(renderer, trainer.scene.count)
    actual_scales = _actual_scale(groups["scales"][:, :3])
    expected_scale = _expected_refinement_child_scale(np.array([0.09, 0.06, 0.03], dtype=np.float32), 2, scale_mul=1.5)
    np.testing.assert_allclose(actual_scales, np.repeat(expected_scale[None, :], 2, axis=0), rtol=0.0, atol=1e-6)


def test_refinement_rewrite_halves_unsplit_splat_age(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=92)
    scene.opacities[:] = np.array([0.6], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="refinement_unsplit_splat_age_target.png", image_id=113)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-6, refinement_min_contribution=50),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([0], dtype=np.uint32)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

    splat_age = buffer_to_numpy(trainer.refinement_buffers["splat_age"], np.float32)[: trainer.scene.count]
    np.testing.assert_allclose(splat_age, np.array([0.5], dtype=np.float32), rtol=0.0, atol=1e-7)


def test_refinement_rewrite_resets_split_family_splat_age(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=94)
    scene.opacities[:] = np.array([0.6], dtype=np.float32)
    frame = _make_frame(tmp_path, image_name="refinement_split_splat_age_target.png", image_id=114)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-6, refinement_min_contribution=50),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([1], dtype=np.uint32)
    trainer.refinement_buffers["splat_age"].copy_from_numpy(np.array([0.25], dtype=np.float32))
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

    assert trainer.scene.count == 2
    splat_age = buffer_to_numpy(trainer.refinement_buffers["splat_age"], np.float32)[: trainer.scene.count]
    np.testing.assert_allclose(splat_age, np.ones((2,), dtype=np.float32), rtol=0.0, atol=1e-7)


def test_refinement_compact_split_beta_scales_split_family_sigma(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=91)
    scene.opacities[:] = np.array([0.6], dtype=np.float32)
    scene.scales[:] = _log_sigma(np.array([[0.09, 0.06, 0.03]], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="refinement_compact_split_beta_target.png", image_id=112)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            refinement_alpha_cull_threshold=1e-6,
            refinement_min_contribution=50,
            refinement_use_compact_split=True,
            refinement_split_beta=0.28,
        ),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([1], dtype=np.uint32)
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

    groups = _read_scene_groups(renderer, trainer.scene.count)
    actual_scales = _actual_scale(groups["scales"][:, :3])
    expected_scale = _expected_refinement_child_scale_beta(np.array([0.09, 0.06, 0.03], dtype=np.float32), 2, beta=0.28)
    np.testing.assert_allclose(actual_scales, np.repeat(expected_scale[None, :], 2, axis=0), rtol=0.0, atol=1e-6)


def test_refinement_runtime_uses_base_min_contribution_for_first_refinement_then_decays(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=93)
    frame = _make_frame(tmp_path, image_name="refinement_threshold_decay_target.png", image_id=212)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_interval=200, refinement_min_contribution=512, refinement_min_contribution_decay=0.95),
        seed=123,
    )
    observed_pixels = renderer.width * renderer.height
    trainer._observed_contribution_pixel_count = observed_pixels

    trainer.state.step = 200
    first_threshold = int(trainer._refinement_vars()["g_RefinementMinContributionThreshold"])
    trainer.state.step = 400
    second_threshold = int(trainer._refinement_vars()["g_RefinementMinContributionThreshold"])

    assert first_threshold == 512
    assert second_threshold == 486


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
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-6, refinement_min_contribution=50),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([0, 0], dtype=np.uint32)
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200, 49], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

    assert trainer.scene.count == 1
    groups = _read_scene_groups(renderer, trainer.scene.count)
    np.testing.assert_allclose(groups["positions"][0, :3], scene.positions[0], rtol=0.0, atol=1e-6)


def test_refinement_opacity_mul_rewrites_unsplit_survivor_alpha(device, tmp_path: Path) -> None:
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
            refinement_min_contribution=50,
            refinement_opacity_mul=0.25,
        ),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([0], dtype=np.uint32)
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

    groups = _read_scene_groups(renderer, trainer.scene.count)
    np.testing.assert_allclose(_actual_opacity(groups["color_alpha"][:, 3]), np.array([0.25], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_refinement_solve_opacity_shares_family_alpha(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=192)
    scene.opacities[:] = np.array([0.6], dtype=np.float32)
    scene.scales[:] = _log_sigma(np.array([[0.09, 0.06, 0.03]], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="refinement_solve_opacity_target.png", image_id=192)
    renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
    observed_pixels = renderer.width * renderer.height
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(
            refinement_alpha_cull_threshold=1e-6,
            refinement_min_contribution=50,
            refinement_use_compact_split=True,
            refinement_solve_opacity=True,
            refinement_split_beta=0.28,
            refinement_sample_radius=1.35,
        ),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([1], dtype=np.uint32)
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

    groups = _read_scene_groups(renderer, trainer.scene.count)
    actual_opacity = _actual_opacity(groups["color_alpha"][:, 3])
    assert trainer.scene.count == 2
    np.testing.assert_allclose(actual_opacity, np.full((2,), actual_opacity[0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert 0.25 < float(actual_opacity[0]) < 0.6


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
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-3, refinement_min_contribution=50),
        seed=123,
    )

    src_moments = np.zeros((renderer.TRAINABLE_PARAM_COUNT, scene.count, 2), dtype=np.float32)
    src_moments[:, 0, 0] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 1.0
    src_moments[:, 0, 1] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 101.0
    src_moments[:, 1, 0] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 1001.0
    src_moments[:, 1, 1] = np.arange(renderer.TRAINABLE_PARAM_COUNT, dtype=np.float32) + 1101.0
    trainer.adam_optimizer.buffers["adam_moments"].copy_from_numpy(src_moments.reshape(-1, 2))

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([2, 5], dtype=np.uint32)
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200, 0], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

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
        training_hparams=TrainingHyperParams(max_gaussians=3, refinement_alpha_cull_threshold=1e-3, refinement_min_contribution=50),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([4, 4], dtype=np.uint32)
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200, 200], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

    assert trainer.scene.count == 3
    groups = _read_scene_groups(renderer, trainer.scene.count)
    actual_scales = _actual_scale(groups["scales"][:, :3])
    positions = groups["positions"][:, :3]
    unsplit_mask = np.linalg.norm(positions - scene.positions[1][None, :], axis=1) < 1e-6
    split_mask = ~unsplit_mask
    assert int(np.count_nonzero(split_mask)) == 2
    assert int(np.count_nonzero(unsplit_mask)) == 1
    np.testing.assert_allclose(actual_scales[unsplit_mask], np.full((1, 3), 0.04, dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(actual_scales[split_mask], np.repeat(actual_scales[split_mask][:1], 2, axis=0), rtol=0.0, atol=1e-6)
    assert np.all(actual_scales[split_mask] < np.array([[0.09, 0.06, 0.03]], dtype=np.float32))
    split_positions = positions[split_mask, :3]
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
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-6, refinement_min_contribution=50),
        seed=123,
    )
    trainer_b = GaussianTrainer(
        device=device,
        renderer=renderer_b,
        scene=scene_b,
        frames=[frame_b],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-6, refinement_min_contribution=50),
        seed=123,
    )

    trainer_a._observed_contribution_pixel_count = observed_pixels
    trainer_b._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([1], dtype=np.uint32)
    trainer_a.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer_b.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer_a.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer_b.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer_a._run_refinement(clone_counts_override=clone_counts)
    trainer_b._run_refinement(clone_counts_override=clone_counts)

    positions_a = _read_scene_groups(renderer_a, trainer_a.scene.count)["positions"][:, :3]
    positions_b = _read_scene_groups(renderer_b, trainer_b.scene.count)["positions"][:, :3]
    np.testing.assert_allclose(np.mean(positions_a, axis=0), scene_a.positions[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.mean(positions_b, axis=0), scene_b.positions[0], rtol=0.0, atol=1e-6)
    assert not np.allclose(positions_a, positions_b, rtol=0.0, atol=1e-6)


def test_refinement_rewrite_samples_family_offsets_on_largest_area_plane(device, tmp_path: Path) -> None:
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
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-6, refinement_min_contribution=50),
        seed=123,
    )

    trainer._observed_contribution_pixel_count = observed_pixels
    clone_counts = np.array([3], dtype=np.uint32)
    trainer.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement(clone_counts_override=clone_counts)

    groups = _read_scene_groups(renderer, trainer.scene.count)
    family_positions = groups["positions"][:, :3]
    parent_scale = np.array([0.7, 0.5, 0.3], dtype=np.float32)
    family_size = trainer.scene.count
    shrink = family_size ** (-0.28)
    residual_sigma = parent_scale * np.sqrt(max(1.0 - shrink * shrink, 0.0))
    normalized_lengths = np.linalg.norm((family_positions - scene.positions[0][None, :]) / residual_sigma[None, :], axis=1)

    assert trainer.scene.count == 4
    np.testing.assert_allclose(family_positions[:, 2], scene.positions[0, 2], rtol=0.0, atol=1e-6)
    assert float(np.max(normalized_lengths)) <= 3.0 + 1e-5


def test_refinement_rewrite_sample_radius_scales_child_offsets(device, tmp_path: Path) -> None:
    base_scene = _make_scene(count=1, seed=177)
    base_scene.positions[0] = np.array([0.25, -0.4, 0.75], dtype=np.float32)
    base_scene.scales[0] = _log_sigma(np.array([0.3, 0.2, 0.1], dtype=np.float32))
    base_scene.rotations[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    base_scene.opacities[0] = np.float32(0.6)
    frame = _make_frame(tmp_path, image_name="refinement_sample_radius_target.png", image_id=71)
    observed_pixels = 32 * 32

    def run_radius(sample_radius: float) -> np.ndarray:
        scene = GaussianScene(
            positions=base_scene.positions.copy(),
            scales=base_scene.scales.copy(),
            rotations=base_scene.rotations.copy(),
            opacities=base_scene.opacities.copy(),
            colors=base_scene.colors.copy(),
            sh_coeffs=base_scene.sh_coeffs.copy(),
        )
        renderer = GaussianRenderer(device, width=32, height=32, list_capacity_multiplier=16)
        trainer = GaussianTrainer(
            device=device,
            renderer=renderer,
            scene=scene,
            frames=[frame],
            training_hparams=TrainingHyperParams(
                refinement_alpha_cull_threshold=1e-6,
                refinement_min_contribution=50,
                refinement_sample_radius=sample_radius,
            ),
            seed=123,
        )

        trainer._observed_contribution_pixel_count = observed_pixels
        clone_counts = np.array([1], dtype=np.uint32)
        trainer.refinement_buffers["clone_counts"].copy_from_numpy(clone_counts)
        trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
        trainer._run_refinement(clone_counts_override=clone_counts)

        positions = _read_scene_groups(renderer, trainer.scene.count)["positions"][:, :3]
        np.testing.assert_allclose(np.mean(positions, axis=0), base_scene.positions[0], rtol=0.0, atol=1e-6)
        return positions

    positions_small = run_radius(2.0)
    positions_large = run_radius(4.0)
    offsets_small = positions_small - base_scene.positions[0][None, :]
    offsets_large = positions_large - base_scene.positions[0][None, :]
    lengths_small = np.linalg.norm(offsets_small, axis=1)
    lengths_large = np.linalg.norm(offsets_large, axis=1)
    length_ratio = lengths_large / lengths_small

    assert positions_small.shape == positions_large.shape == (2, 3)
    assert float(np.max(lengths_small)) > 1e-4
    assert np.all(length_ratio > 1.5)
    assert np.all(length_ratio < 2.1)


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
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16, sh_band=3)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[near_frame, tight_frame, offscreen_frame],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-3),
        seed=123,
    )

    expected_support = min(
        _circle_bound_support_radius(trainer.make_frame_camera(frame_index, renderer.width, renderer.height), scene.positions[0], renderer.width, renderer.height, _REFINEMENT_MIN_SCREEN_RADIUS_PX)
        for frame_index in range(3)
    )
    expected_sigma = expected_support / (renderer.radius_scale * _GAUSSIAN_SUPPORT_SIGMA_RADIUS)

    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200], dtype=np.uint32))
    trainer._run_refinement()

    scales = _actual_scale(_read_scene_groups(renderer, trainer.scene.count)["scales"][0, :3])
    initial_scale = _actual_scale(scene.scales[0, :3])
    assert np.all(scales > initial_scale)
    np.testing.assert_allclose(scales, np.full((3,), scales[0], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(scales, np.full((3,), expected_sigma, dtype=np.float32), rtol=0.0, atol=1e-6)


def test_refinement_min_screen_size_scales_with_distance(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=130)
    scene.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.positions[1] = np.array([0.0, 0.0, 6.0], dtype=np.float32)
    scene.scales[:, :3] = _log_sigma(np.full((2, 3), 1e-5, dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="refinement_min_distance.png", image_id=131)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(refinement_alpha_cull_threshold=1e-3),
        seed=123,
    )

    camera = trainer.make_frame_camera(0, renderer.width, renderer.height)
    expected_near = _circle_bound_support_radius(camera, scene.positions[0], renderer.width, renderer.height, _REFINEMENT_MIN_SCREEN_RADIUS_PX) / (
        renderer.radius_scale * _GAUSSIAN_SUPPORT_SIGMA_RADIUS
    )
    expected_far = _circle_bound_support_radius(camera, scene.positions[1], renderer.width, renderer.height, _REFINEMENT_MIN_SCREEN_RADIUS_PX) / (
        renderer.radius_scale * _GAUSSIAN_SUPPORT_SIGMA_RADIUS
    )

    trainer.refinement_buffers["splat_contribution"].copy_from_numpy(np.array([200, 200], dtype=np.uint32))
    trainer._run_refinement()

    scales = _actual_scale(_read_scene_groups(renderer, trainer.scene.count)["scales"][:, :3])
    np.testing.assert_allclose(scales[0], np.full((3,), expected_near, dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(scales[1], np.full((3,), expected_far, dtype=np.float32), rtol=0.0, atol=1e-6)
    assert float(scales[1, 0]) > float(scales[0, 0])


def test_training_max_screen_size_clamps_large_splats(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=107)
    scene.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([0.5, 0.5, 0.5], dtype=np.float32))
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
    expected_scales = _view_angle_cap_expected(
        scale=_actual_scale(scene.scales[0, :3]),
        position=scene.positions[0],
        camera=camera,
        renderer=renderer,
        max_visible_angle_deg=resolve_max_visible_angle_deg(trainer.training, 1),
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, 1)["scales"][0, :3])
    np.testing.assert_allclose(scales, expected_scales, rtol=0.0, atol=1e-6)


def test_training_max_screen_size_clamps_scale_components_with_shared_scalar_cap(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=108)
    scene.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([1.0, 0.2, 0.1], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="max_screen_component_clamp_target.png", image_id=28)
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
    expected_scales = _view_angle_cap_expected(
        scale=_actual_scale(scene.scales[0, :3]),
        position=scene.positions[0],
        camera=camera,
        renderer=renderer,
        max_visible_angle_deg=resolve_max_visible_angle_deg(trainer.training, 1),
    )
    max_sigma = float(expected_scales[0])
    assert expected_scales[0] < _actual_scale(scene.scales[0, :3])[0]
    np.testing.assert_allclose(expected_scales, np.minimum(_actual_scale(scene.scales[0, :3]), np.full((3,), max_sigma, dtype=np.float32)), rtol=0.0, atol=1e-6)

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, 1)["scales"][0, :3])
    np.testing.assert_allclose(scales, expected_scales, rtol=0.0, atol=1e-6)


def test_training_max_screen_size_clamps_offscreen_splats_by_distance(device, tmp_path: Path) -> None:
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
    expected_scales = _view_angle_cap_expected(
        scale=_actual_scale(scene.scales[0, :3]),
        position=scene.positions[0],
        camera=camera,
        renderer=renderer,
        max_visible_angle_deg=resolve_max_visible_angle_deg(trainer.training, 1),
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, 1)["scales"][0, :3])
    np.testing.assert_allclose(scales, expected_scales, rtol=0.0, atol=1e-6)


def test_training_max_screen_size_clamps_previously_invisible_splats_by_distance(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=111)
    scene.positions[0] = np.array([100.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([2.0, 1.5, 1.0], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="max_screen_invisible_target.png", image_id=27)
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
    expected_scales = _view_angle_cap_expected(
        scale=_actual_scale(scene.scales[0, :3]),
        position=scene.positions[0],
        camera=camera,
        renderer=renderer,
        max_visible_angle_deg=resolve_max_visible_angle_deg(trainer.training, 1),
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, 1)["scales"][0, :3])
    np.testing.assert_allclose(scales, expected_scales, rtol=0.0, atol=1e-6)


def test_training_max_screen_size_uses_scheduled_value(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=110)
    scene.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[0] = _log_sigma(np.array([0.5, 0.5, 0.5], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="max_screen_schedule_target.png", image_id=26)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    training = TrainingHyperParams(
        lr_schedule_stage1_step=1,
        lr_schedule_stage2_step=2,
        lr_schedule_steps=3,
        max_visible_angle_deg=_legacy_screen_fraction_to_visible_angle_deg(0.2),
        max_visible_angle_deg_stage1=_legacy_screen_fraction_to_visible_angle_deg(0.05),
        max_visible_angle_deg_stage2=_legacy_screen_fraction_to_visible_angle_deg(0.025),
        max_visible_angle_deg_stage3=_legacy_screen_fraction_to_visible_angle_deg(0.0125),
        scale_l2_weight=0.0,
        scale_abs_reg_weight=0.0,
        opacity_reg_weight=0.0,
    )
    trainer = GaussianTrainer(device=device, renderer=renderer, scene=scene, frames=[frame], training_hparams=training, seed=123)
    camera = trainer.make_frame_camera(0, renderer.width, renderer.height)
    scheduled_angle = resolve_max_visible_angle_deg(training, 2)
    expected_scales = _view_angle_cap_expected(
        scale=_actual_scale(scene.scales[0, :3]),
        position=scene.positions[0],
        camera=camera,
        renderer=renderer,
        max_visible_angle_deg=scheduled_angle,
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 2, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, 1)["scales"][0, :3])
    np.testing.assert_allclose(scales, expected_scales, rtol=0.0, atol=1e-6)


def test_training_max_screen_size_matches_same_distance_for_offscreen_splats(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=132)
    scene.positions[0] = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    scene.positions[1] = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    scene.scales[:, :3] = _log_sigma(np.array([[2.0, 1.5, 1.0], [2.0, 1.5, 1.0]], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="max_screen_same_distance.png", image_id=132)
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
    expected_scales = np.stack(
        [
            _view_angle_cap_expected(
                scale=_actual_scale(scene.scales[index, :3]),
                position=scene.positions[index],
                camera=camera,
                renderer=renderer,
                max_visible_angle_deg=resolve_max_visible_angle_deg(trainer.training, 1),
            )
            for index in range(2)
        ],
        axis=0,
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, scene.count)["scales"][:, :3])
    np.testing.assert_allclose(scales, expected_scales, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(scales[0], scales[1], rtol=0.0, atol=1e-6)


def test_training_max_screen_size_gets_looser_with_distance(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=133)
    scene.positions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scene.positions[1] = np.array([0.0, 0.0, 6.0], dtype=np.float32)
    scene.scales[:, :3] = _log_sigma(np.array([[2.0, 1.5, 1.0], [2.0, 1.5, 1.0]], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="max_screen_distance.png", image_id=133)
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
    expected_scales = np.stack(
        [
            _view_angle_cap_expected(
                scale=_actual_scale(scene.scales[index, :3]),
                position=scene.positions[index],
                camera=camera,
                renderer=renderer,
                max_visible_angle_deg=resolve_max_visible_angle_deg(trainer.training, 1),
            )
            for index in range(2)
        ],
        axis=0,
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, scene.count)["scales"][:, :3])
    np.testing.assert_allclose(scales, expected_scales, rtol=0.0, atol=1e-6)
    assert float(scales[1, 0]) > float(scales[0, 0])


def test_training_max_screen_size_is_independent_of_opacity(device, tmp_path: Path) -> None:
    scene = _make_scene(count=2, seed=134)
    scene.positions[:] = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    scene.scales[:, :3] = _log_sigma(np.array([[2.0, 1.5, 1.0], [2.0, 1.5, 1.0]], dtype=np.float32))
    scene.opacities[:] = _raw_opacity(np.array([0.05, 0.95], dtype=np.float32))
    frame = _make_frame(tmp_path, image_name="max_screen_opacity.png", image_id=134)
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
    expected_scales = np.stack(
        [
            _view_angle_cap_expected(
                scale=_actual_scale(scene.scales[index, :3]),
                position=scene.positions[index],
                camera=camera,
                renderer=renderer,
                max_visible_angle_deg=resolve_max_visible_angle_deg(trainer.training, 1),
            )
            for index in range(2)
        ],
        axis=0,
    )

    zeros = np.zeros((scene.count, 4), dtype=np.float32)
    _write_grad_groups(renderer, scene.count, grad_positions=zeros, grad_scales=zeros, grad_rotations=zeros, grad_color_alpha=zeros)

    enc = device.create_command_encoder()
    trainer._dispatch_optimizer_step(enc, 1, camera)
    device.submit_command_buffer(enc.finish())
    device.wait()

    scales = _actual_scale(_read_scene_groups(renderer, scene.count)["scales"][:, :3])
    np.testing.assert_allclose(scales, expected_scales, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(scales[0], scales[1], rtol=0.0, atol=1e-6)


def test_optimizer_projection_clamps_sh_coefficients(device, tmp_path: Path) -> None:
    scene = _make_scene(count=1, seed=111)
    frame = _make_frame(tmp_path, image_name="sh_projection_clamp_target.png", image_id=26)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=16)
    trainer = GaussianTrainer(
        device=device,
        renderer=renderer,
        scene=scene,
        frames=[frame],
        training_hparams=TrainingHyperParams(use_sh=True, sh_band=3, scale_l2_weight=0.0, scale_abs_reg_weight=0.0, opacity_reg_weight=0.0, sh1_reg_weight=0.0),
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
    per_component_clamped = np.array([[1.0, -1.0, 0.5], [-1.0, 0.25, 1.0], [0.0, 1.0, -1.0]], dtype=np.float32)
    assert np.all(np.abs(sh_coeffs[1:4]) <= np.abs(per_component_clamped) + 1e-6)
    sampled_dirs = np.stack([_sh_clamp_sample_dir(0, sample_id) for sample_id in range(_SH_CLAMP_SAMPLE_COUNT)], axis=0)
    sampled_sh = np.repeat(sh_coeffs[None, :, :], _SH_CLAMP_SAMPLE_COUNT, axis=0)
    sampled_colors = evaluate_sh_color(sampled_sh, sampled_dirs)
    assert np.all(np.isfinite(sampled_colors))
    assert np.all(sampled_colors >= -1e-5)
    assert np.all(sampled_colors <= 1.0 + 1e-5)
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

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

torch = pytest.importorskip("torch")

from src.renderer import Camera, GaussianRenderer, TorchGaussianRenderSettings, TorchGaussianRendererContext, render_gaussian_splats_torch
from src.scene import GaussianScene


def _make_splats(device: torch.device, count: int = 8, seed: int = 7) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    splats = np.zeros((count, 14), dtype=np.float32)
    splats[:, 0:3] = rng.uniform(-0.35, 0.35, size=(count, 3)).astype(np.float32)
    splats[:, 2] += 0.2
    splats[:, 3:6] = np.log(np.full((count, 3), 0.05, dtype=np.float32))
    splats[:, 6] = 1.0
    splats[:, 10:13] = rng.uniform(0.15, 0.95, size=(count, 3)).astype(np.float32)
    splats[:, 13] = rng.uniform(0.2, 0.8, size=(count,)).astype(np.float32)
    return torch.tensor(splats, device=device, dtype=torch.float32)


def _make_camera_params(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 72.0, 72.0, 32.0, 32.0, 0.1, 20.0, 0.0, 0.0],
        device=device,
        dtype=torch.float32,
    )


def _camera_from_tensor(camera_params: torch.Tensor) -> Camera:
    values = camera_params.detach().cpu().numpy()
    return Camera.from_colmap(
        q_wxyz=values[0:4],
        t_xyz=values[4:7],
        fx=float(values[7]),
        fy=float(values[8]),
        cx=float(values[9]),
        cy=float(values[10]),
        near=float(values[11]),
        far=float(values[12]),
    )


def _scene_from_splats(splats: torch.Tensor) -> GaussianScene:
    values = splats.detach().cpu().numpy()
    return GaussianScene(
        positions=np.asarray(values[:, 0:3], dtype=np.float32),
        scales=np.asarray(values[:, 3:6], dtype=np.float32),
        rotations=np.asarray(values[:, 6:10], dtype=np.float32),
        opacities=np.asarray(values[:, 13], dtype=np.float32),
        colors=np.asarray(values[:, 10:13], dtype=np.float32),
        sh_coeffs=np.zeros((values.shape[0], 1, 3), dtype=np.float32),
    )


def _public_reference_grads(splats: torch.Tensor, grad_groups: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(splats.detach().cpu().numpy(), dtype=np.float32)
    count = values.shape[0]
    packed = np.zeros((count, 14), dtype=np.float32)
    packed[:, 0:3] = np.asarray(grad_groups["grad_positions"][:, 0:3], dtype=np.float32)
    packed[:, 3:6] = np.asarray(grad_groups["grad_scales"][:, 0:3], dtype=np.float32)
    packed[:, 6:10] = np.asarray(grad_groups["grad_rotations"][:, 0:4], dtype=np.float32)
    packed[:, 10:13] = np.asarray(grad_groups["grad_color_alpha"][:, 0:3], dtype=np.float32)
    alpha = np.clip(values[:, 13], GaussianRenderer._OPACITY_EPS, 1.0 - GaussianRenderer._OPACITY_EPS)
    packed[:, 13] = np.asarray(grad_groups["grad_color_alpha"][:, 3], dtype=np.float32) / (alpha * (1.0 - alpha))
    return packed


@pytest.fixture(scope="module")
def torch_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch is required for torch renderer tests.")
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def test_torch_renderer_forward_smoke(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=64, height=64, list_capacity_multiplier=16)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=12)
    camera_params = _make_camera_params(torch_cuda_device)

    image = render_gaussian_splats_torch(splats, camera_params, settings, context)

    assert image.is_cuda
    assert image.dtype == torch.float32
    assert tuple(image.shape) == (64, 64, 4)
    assert torch.isfinite(image).all()


def test_torch_renderer_backward_smoke(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=64, height=64, list_capacity_multiplier=16)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=10).requires_grad_(True)
    camera_params = _make_camera_params(torch_cuda_device)

    loss = render_gaussian_splats_torch(splats, camera_params, settings, context).sum()
    loss.backward()

    assert splats.grad is not None
    assert splats.grad.shape == splats.shape
    assert torch.isfinite(splats.grad).all()
    assert torch.count_nonzero(splats.grad).item() > 0


def test_torch_renderer_output_matches_renderer_texture(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=64, height=64, list_capacity_multiplier=16)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=9)
    camera_params = _make_camera_params(torch_cuda_device)

    image = render_gaussian_splats_torch(splats, camera_params, settings, context)
    reference = np.asarray(context.renderer.output_texture.to_numpy(), dtype=np.float32)

    np.testing.assert_allclose(image.detach().cpu().numpy(), reference, rtol=0.0, atol=0.0)


def test_torch_renderer_gradients_match_reference_backward(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=64, height=64, list_capacity_multiplier=16)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=7).requires_grad_(True)
    camera_params = _make_camera_params(torch_cuda_device)

    image = render_gaussian_splats_torch(splats, camera_params, settings, context)
    loss = 0.5 * torch.square(image).sum()
    loss.backward()

    scene = _scene_from_splats(splats.detach())
    camera = _camera_from_tensor(camera_params)
    reference = context.renderer.debug_raster_backward_grads(scene, camera, background=settings.background)
    public_reference = _public_reference_grads(splats.detach(), reference)

    np.testing.assert_allclose(
        splats.grad.detach().cpu().numpy(),
        public_reference,
        rtol=2e-5,
        atol=3e-5,
    )


def test_torch_renderer_alpha_gradient_is_public_alpha_space(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=64, height=64, list_capacity_multiplier=16)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    splats = _make_splats(torch_cuda_device, count=6).requires_grad_(True)
    camera_params = _make_camera_params(torch_cuda_device)

    image = render_gaussian_splats_torch(splats, camera_params, settings, context)
    (0.5 * torch.square(image).sum()).backward()

    scene = _scene_from_splats(splats.detach())
    camera = _camera_from_tensor(camera_params)
    reference = context.renderer.debug_raster_backward_grads(scene, camera, background=settings.background)
    raw_opacity_grad = np.asarray(reference["grad_color_alpha"][:, 3], dtype=np.float32)
    alpha_grad = splats.grad.detach().cpu().numpy()[:, 13]

    assert not np.allclose(alpha_grad, raw_opacity_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(alpha_grad, _public_reference_grads(splats.detach(), reference)[:, 13], rtol=2e-5, atol=3e-5)


def test_torch_renderer_rejects_invalid_inputs(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=32, height=32)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    camera_params = _make_camera_params(torch_cuda_device)

    with pytest.raises(ValueError, match="CUDA"):
        render_gaussian_splats_torch(torch.zeros((1, 14), dtype=torch.float32), camera_params, settings, context)

    with pytest.raises(TypeError, match="float32"):
        render_gaussian_splats_torch(torch.zeros((1, 14), device=torch_cuda_device, dtype=torch.float64), camera_params, settings, context)

    with pytest.raises(ValueError, match="shape"):
        render_gaussian_splats_torch(torch.zeros((1, 13), device=torch_cuda_device, dtype=torch.float32), camera_params, settings, context)

    with pytest.raises(ValueError, match="shape"):
        render_gaussian_splats_torch(_make_splats(torch_cuda_device, count=1), torch.zeros((14,), device=torch_cuda_device, dtype=torch.float32), settings, context)

    with pytest.raises(ValueError, match="not supported"):
        render_gaussian_splats_torch(
            _make_splats(torch_cuda_device, count=1),
            _make_camera_params(torch_cuda_device).requires_grad_(True),
            settings,
            context,
        )


def test_torch_renderer_context_reuses_renderer_across_scene_sizes(torch_cuda_device: torch.device):
    settings = TorchGaussianRenderSettings(width=48, height=48, list_capacity_multiplier=8)
    context = TorchGaussianRendererContext(torch_device=torch_cuda_device)
    camera_params = _make_camera_params(torch_cuda_device)

    image0 = render_gaussian_splats_torch(_make_splats(torch_cuda_device, count=2), camera_params, settings, context)
    renderer_id = id(context.renderer)
    image1 = render_gaussian_splats_torch(_make_splats(torch_cuda_device, count=11), camera_params, settings, context)
    image2 = render_gaussian_splats_torch(_make_splats(torch_cuda_device, count=5), camera_params, settings, context)

    assert id(context.renderer) == renderer_id
    assert torch.isfinite(image0).all()
    assert torch.isfinite(image1).all()
    assert torch.isfinite(image2).all()


def test_torch_renderer_cuda_device_smoke(torch_cuda_device: torch.device):
    device = spy.create_torch_device(
        type=spy.DeviceType.cuda,
        torch_device=torch_cuda_device,
        include_paths=("c:/Development/slang-splat/shaders", "c:/Development/slang-splat/shaders/renderer", "c:/Development/slang-splat/shaders/utility"),
        enable_debug_layers=False,
        enable_print=False,
        enable_hot_reload=False,
        enable_compilation_reports=False,
    )
    buffer = device.create_buffer(size=16, usage=spy.BufferUsage.shared | spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)
    src = torch.tensor([1.0, 2.0, 3.0, 4.0], device=torch_cuda_device, dtype=torch.float32)
    dst = torch.zeros_like(src)

    spy.copy_torch_tensor_to_buffer(src, buffer)
    device.sync_to_cuda()
    spy.copy_buffer_to_torch_tensor(buffer, dst)

    assert torch.allclose(src, dst)

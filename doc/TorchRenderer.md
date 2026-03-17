# PyTorch CUDA Renderer

`src/renderer/torch_renderer.py` exposes a CUDA-only PyTorch interface over the existing Slang raster training path.

## Public API
- `TorchGaussianRenderSettings`: non-differentiable render configuration.
- `TorchGaussianRendererContext`: owns the Slang CUDA device, one cached `GaussianRenderer`, the output-pack shader, and the shared torch bridge buffers.
- `render_gaussian_splats_torch(splats, camera_params, settings, context=None)`: differentiable render entrypoint.

## Tensor Contracts
- `splats`: CUDA `torch.float32`, shape `[N, 14]`
  - layout: `[x, y, z, log_sigma_x, log_sigma_y, log_sigma_z, q_w, q_x, q_y, q_z, r, g, b, alpha]`
- `camera_params`: CUDA `torch.float32`, shape `[15]`
  - layout: `[q_w, q_x, q_y, q_z, t_x, t_y, t_z, fx, fy, cx, cy, near, far, k1, k2]`
- output: CUDA `torch.float32`, shape `[H, W, 4]`

## Forward Path
- Public splats are packed on CUDA into the renderer's internal param-major layout.
- Alpha is clamped with the renderer's opacity epsilon and converted to raw opacity with `torch.logit`.
- The context copies the packed scene tensor into a shared bridge buffer, synchronizes CUDA writes to the Slang device, copies into `g_SplatParams`, runs prepass, then runs `csRasterizeTrainingForward`.
- `csPackOutputTexture` writes the rendered `RWTexture2D<float4>` into a tightly packed shared `RWStructuredBuffer<float4>`, and the wrapper copies that buffer into a fresh CUDA tensor for the PyTorch result.

## Backward Path
- Camera gradients are not supported in v1.
- PyTorch's upstream image gradient is copied into a shared `float4` output-grad buffer, synchronized to the Slang device, then consumed by `csRasterizeBackward` and `csBackpropCachedRasterGrads`.
- Final param-major gradients are copied into a shared bridge buffer and then into a CUDA tensor.
- The public opacity gradient is converted from internal raw-opacity space back to alpha space with `1 / (alpha * (1 - alpha))`.

## Notes
- The interface is pure CUDA through `slangpy.create_torch_device(type=spy.DeviceType.cuda, ...)`.
- Viewer and CLI remain Vulkan-first; this module is an optional CUDA/PyTorch path.
- PyTorch is intentionally not part of `requirements.txt`; install a CUDA-enabled PyTorch build separately before using this interface.

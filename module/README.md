# Compact Gaussian Splatting Module

`module/splatting.py` exposes a compact CUDA+PyTorch renderer:

`render_gaussian_splats(splats, camera_params, image_size, background=(0, 0, 0), render_seed=0, context=None)`

Tensor contracts:
- `splats`: CUDA `torch.float32`, shape `[14, N]`
- `camera_params`: CUDA `torch.float32`, shape `[15]`
- output: CUDA `torch.float32`, shape `[H, W, 4]`

Pipeline:
1. Stable PyTorch sort by camera distance.
2. Raw projection/count dispatch.
3. PyTorch prefix sum.
4. Raw projection/fill dispatch.
5. Stable PyTorch sort by tile id.
6. SlangPy module tile-range build.
7. Raw forward raster.
8. PyTorch loss outside the module.
9. Raw backward raster with float atomics to packed gradients.

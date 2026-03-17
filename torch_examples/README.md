# Torch Examples

This folder contains small torch-side examples built on top of the CUDA torch renderer.

## Garden COLMAP Trainer

Run from an elevated PowerShell session:

```powershell
python torch_examples/train_colmap_garden_torch.py
```

Default behavior:
- loads `dataset/garden`
- uses `images_4`
- initializes the scene from the COLMAP point cloud
- runs `30000` Adam steps
- uses native image resolution
- uses float cached raster gradients so native-resolution mean L1 gradients do not underflow
- shows `tqdm` progress with loss and PSNR

Prerequisites:
- CUDA-enabled PyTorch
- `slangpy`
- `dataset/garden`

This is a compact torch example over the existing renderer, not the main project training path.

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import slangpy as spy

ROOT = Path(__file__).resolve().parent.parent.parent
SHADER_ROOT = ROOT / "shaders"
SLANGPY_SHADER_ROOT = Path(spy.__file__).resolve().parent / "slang"
SHADER_INCLUDE_PATHS = (SLANGPY_SHADER_ROOT, SHADER_ROOT, SHADER_ROOT / "renderer", SHADER_ROOT / "utility")


def device_type_from_name(name: str) -> spy.DeviceType:
    normalized = str(name).strip().lower()
    if normalized in {"vulkan", "vk"}:
        return spy.DeviceType.vulkan
    raise ValueError(f"Unsupported device type '{name}'. Use 'vulkan'.")


def create_default_device(device_type: spy.DeviceType = spy.DeviceType.vulkan, enable_debug_layers: bool = False) -> spy.Device:
    return spy.Device(
        type=device_type,
        compiler_options={"include_paths": [str(path) for path in SHADER_INCLUDE_PATHS]},
        enable_debug_layers=bool(enable_debug_layers),
        enable_rhi_validation=False,
        enable_print=False,
        enable_hot_reload=True,
        enable_compilation_reports=True,
    )


def load_compute_kernel(device: spy.Device, shader_path: str | Path, entry_point: str) -> spy.ComputeKernel:
    return device.create_compute_kernel(device.load_program(str(shader_path), [str(entry_point)]))


def load_compute_pipeline(device: spy.Device, shader_path: str | Path, entry_point: str) -> spy.ComputePipeline:
    return device.create_compute_pipeline(device.load_program(str(shader_path), [str(entry_point)]))


def load_compute_items(
    device: spy.Device,
    specs: Mapping[str, tuple[str, str | Path, str]],
) -> dict[str, spy.ComputeKernel | spy.ComputePipeline]:
    loaded: dict[str, spy.ComputeKernel | spy.ComputePipeline] = {}
    for name, (kind, shader_path, entry_point) in specs.items():
        if str(kind) == "kernel":
            loaded[name] = load_compute_kernel(device, shader_path, entry_point)
        elif str(kind) == "pipeline":
            loaded[name] = load_compute_pipeline(device, shader_path, entry_point)
        else:
            raise ValueError(f"Unsupported compute item kind '{kind}' for '{name}'.")
    return loaded


def load_compute_kernels(
    device: spy.Device,
    shader_path: str | Path,
    entries: Mapping[str, str],
) -> dict[str, spy.ComputeKernel]:
    return {name: load_compute_kernel(device, shader_path, entry_point) for name, entry_point in entries.items()}


def load_compute_pipelines(
    device: spy.Device,
    shader_path: str | Path,
    entries: Mapping[str, str],
) -> dict[str, spy.ComputePipeline]:
    return {name: load_compute_pipeline(device, shader_path, entry_point) for name, entry_point in entries.items()}

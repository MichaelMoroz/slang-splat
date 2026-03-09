from __future__ import annotations

from pathlib import Path

import slangpy as spy

from ..common import SHADER_ROOT


class SeparableGaussianBlur:
    _TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.copy_source | spy.TextureUsage.copy_destination
    _KERNEL_ENTRIES = {"horizontal": "csGaussianBlurHorizontal", "vertical": "csGaussianBlurVertical"}
    _dispatch = lambda self, kernel, encoder, vars: self._kernels[kernel].dispatch(thread_count=spy.uint3(self.width, self.height, 1), vars=vars, command_encoder=encoder)

    def __init__(self, device: spy.Device, width: int, height: int) -> None:
        self.device, self.width, self.height = device, int(width), int(height)
        self._shader_path = Path(SHADER_ROOT / "utility" / "blur" / "separable_gaussian_blur.slang")
        self._kernels = {name: self.device.create_compute_kernel(self.device.load_program(str(self._shader_path), [entry])) for name, entry in self._KERNEL_ENTRIES.items()}
        self._scratch_texture: spy.Texture | None = None

    def make_texture(self) -> spy.Texture:
        return self.device.create_texture(format=spy.Format.rgba32_float, width=self.width, height=self.height, usage=self._TEXTURE_USAGE)

    def _ensure_scratch_texture(self) -> spy.Texture:
        if self._scratch_texture is None:
            self._scratch_texture = self.make_texture()
        return self._scratch_texture

    def blur(self, encoder: spy.CommandEncoder, input_texture: spy.Texture, output_texture: spy.Texture) -> spy.Texture:
        if int(output_texture.width) != self.width or int(output_texture.height) != self.height:
            raise ValueError("Blur output texture size does not match blur utility size.")
        shared = {"g_BlurWidth": self.width, "g_BlurHeight": self.height}
        scratch = self._ensure_scratch_texture()
        self._dispatch("horizontal", encoder, {"g_BlurInput": input_texture, "g_BlurOutput": scratch, **shared})
        self._dispatch("vertical", encoder, {"g_BlurInput": scratch, "g_BlurOutput": output_texture, **shared})
        return output_texture

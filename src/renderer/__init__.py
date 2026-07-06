from .camera import PROJECTION_MODEL_EQUIRECTANGULAR, PROJECTION_MODEL_PINHOLE, Camera
from .gaussian_renderer import GaussianRenderer, RenderOutput
from .renderer_context import GaussianRenderSettings, GaussianRendererContext

__all__ = [
    "Camera",
    "GaussianRenderer",
    "GaussianRenderSettings",
    "GaussianRendererContext",
    "PROJECTION_MODEL_EQUIRECTANGULAR",
    "PROJECTION_MODEL_PINHOLE",
    "RenderOutput",
]

try:
    from .torch_renderer import (
        TorchGaussianRenderSettings as TorchGaussianRenderSettings,
        TorchGaussianRendererContext as TorchGaussianRendererContext,
        render_gaussian_splats_torch as render_gaussian_splats_torch,
    )
except ImportError:
    pass
else:
    __all__.extend(["TorchGaussianRenderSettings", "TorchGaussianRendererContext", "render_gaussian_splats_torch"])

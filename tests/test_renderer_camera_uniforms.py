from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.renderer import Camera, GaussianRenderer, PROJECTION_MODEL_EQUIRECTANGULAR


def _renderer_stub(width: int = 256, height: int = 128) -> GaussianRenderer:
    renderer = object.__new__(GaussianRenderer)
    renderer.width = int(width)
    renderer.height = int(height)
    renderer._scene_count = 1
    renderer._sorted_values = lambda: "sorted_values"
    renderer._work_buffers = {"tile_ranges": "tile_ranges", "fallback_clone_counts": "fallback_clone_counts"}
    renderer.proj_distortion_k1 = 0.0
    renderer.proj_distortion_k2 = 0.0
    renderer._scene_vars = lambda: {}
    renderer._screen_vars = lambda: {}
    renderer._raster_cache_vars = lambda: {}
    renderer._raster_grad_vars = lambda _workspace=None: {}
    renderer._raster_grad_decode_scale_var = lambda _scale: {}
    renderer._raster_grad_fixed_range_vars = lambda: {}
    renderer._prepass_uniforms = lambda _count: {}
    renderer._raster_uniforms = lambda *_args: {}
    renderer._anisotropy_uniforms = lambda: {}
    renderer._resolve_training_workspace_buffer = lambda name, _workspace=None: f"buffer:{name}"
    renderer._resolve_training_workspace_texture = lambda name, _workspace=None: f"texture:{name}"
    renderer._raster_thread_count = lambda: "threads"
    renderer._raster_grad_shader_set = lambda: SimpleNamespace(training_forward="training_forward", backward="backward", resolve_stats="resolve_stats")
    return renderer


def _equirectangular_camera() -> Camera:
    return Camera.look_at(
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        projection_model=PROJECTION_MODEL_EQUIRECTANGULAR,
    )


def _float2_tuple(value: object) -> tuple[float, float]:
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(getattr(value, "x")), float(getattr(value, "y"))
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(array[0]), float(array[1])


def _subsample_vars(native_width: int = 1024, native_height: int = 512) -> dict[str, object]:
    return {
        "g_TrainingSubsample": {
            "enabled": np.uint32(1),
            "factor": np.uint32(4),
            "nativeWidth": np.uint32(native_width),
            "nativeHeight": np.uint32(native_height),
            "frameIndex": np.uint32(3),
            "stepIndex": np.uint32(9),
        }
    }


def test_camera_uniforms_support_explicit_viewport_override() -> None:
    renderer = _renderer_stub()
    uniforms = renderer._camera_uniforms(_equirectangular_camera(), width=1024, height=512)["g_Camera"]

    assert _float2_tuple(uniforms["viewport"]) == (1024.0, 512.0)
    assert uniforms["projectionModel"] == np.uint32(PROJECTION_MODEL_EQUIRECTANGULAR)


def test_training_raster_uses_native_viewport_for_sample_camera() -> None:
    renderer = _renderer_stub(width=256, height=128)
    calls: list[tuple[str, dict[str, object]]] = []
    renderer._dispatch = lambda shader, _encoder, _thread_count, vars, *_rest: calls.append((str(shader), vars))
    camera = _equirectangular_camera()
    native_camera = _equirectangular_camera()
    training_sample_vars = _subsample_vars()
    background = np.zeros((3,), dtype=np.float32)

    renderer._rasterize_training_forward(
        encoder=None,
        camera=camera,
        background=background,
        output="training_output",
        clone_counts_buffer="clone_counts",
        splat_contribution_buffer="splat_contribution",
        training_native_camera=native_camera,
        training_sample_vars=training_sample_vars,
    )
    renderer._rasterize_backward(
        encoder=None,
        camera=camera,
        background=background,
        output_grad="output_grad",
        target_texture="target",
        regularizer_grad="regularizer_grad",
        clone_counts_buffer="clone_counts",
        gradient_stats_buffer="gradient_stats",
        splat_contribution_buffer="splat_contribution",
        training_native_camera=native_camera,
        training_sample_vars=training_sample_vars,
    )

    for shader, vars in (calls[0], calls[1]):
        assert shader in {"training_forward", "backward"}
        assert _float2_tuple(vars["g_Camera"]["viewport"]) == (256.0, 128.0)
        assert _float2_tuple(vars["g_TrainingNativeCamera"]["viewport"]) == (1024.0, 512.0)
        assert vars["g_TrainingNativeCamera"]["projectionModel"] == np.uint32(PROJECTION_MODEL_EQUIRECTANGULAR)

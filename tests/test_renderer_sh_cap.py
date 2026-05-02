import numpy as np
from types import SimpleNamespace

from src.renderer import GaussianRenderer
from src.renderer import gaussian_renderer as gaussian_renderer_module
from src.scene import SUPPORTED_SH_COEFF_COUNT


def _renderer_for_cap(max_sh_band: int) -> GaussianRenderer:
    renderer = object.__new__(GaussianRenderer)
    renderer._max_sh_band = int(max_sh_band)
    renderer._sh_band = int(max_sh_band)
    return renderer


def test_packed_trainable_param_count_tracks_sh_cap() -> None:
    renderer = _renderer_for_cap(0)
    assert renderer.stored_sh_coeff_count == 1
    assert renderer.packed_raw_opacity_param_id == 13
    assert renderer.packed_trainable_param_count == 14

    renderer = _renderer_for_cap(1)
    assert renderer.stored_sh_coeff_count == 4
    assert renderer.packed_raw_opacity_param_id == 22
    assert renderer.packed_trainable_param_count == 23

    renderer = _renderer_for_cap(2)
    assert renderer.stored_sh_coeff_count == 9
    assert renderer.packed_raw_opacity_param_id == 37
    assert renderer.packed_trainable_param_count == 38

    renderer = _renderer_for_cap(3)
    assert renderer.stored_sh_coeff_count == SUPPORTED_SH_COEFF_COUNT
    assert renderer.packed_raw_opacity_param_id == GaussianRenderer.PARAM_RAW_OPACITY_ID
    assert renderer.packed_trainable_param_count == GaussianRenderer.TRAINABLE_PARAM_COUNT


def test_pack_unpack_param_groups_zeroes_coeffs_above_cap() -> None:
    renderer = _renderer_for_cap(1)
    count = 2
    positions = np.arange(count * 4, dtype=np.float32).reshape(count, 4)
    scales = np.arange(count * 4, dtype=np.float32).reshape(count, 4) + 100.0
    rotations = np.arange(count * 4, dtype=np.float32).reshape(count, 4) + 200.0
    sh_coeffs = np.arange(count * SUPPORTED_SH_COEFF_COUNT * 3, dtype=np.float32).reshape(count, SUPPORTED_SH_COEFF_COUNT, 3)
    color_alpha = np.zeros((count, 4), dtype=np.float32)
    color_alpha[:, 3] = np.array([0.25, 0.75], dtype=np.float32)

    packed = renderer._pack_param_groups(
        count,
        positions=positions,
        scales=scales,
        rotations=rotations,
        sh_coeffs=sh_coeffs,
        color_alpha=color_alpha,
    )

    assert packed.shape == (count * renderer.packed_trainable_param_count,)

    unpacked = renderer._unpack_param_groups(packed, count)
    np.testing.assert_allclose(unpacked["positions"][:, :3], positions[:, :3], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(unpacked["positions"][:, 3], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(unpacked["scales"][:, :3], scales[:, :3], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(unpacked["scales"][:, 3], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(unpacked["rotations"], rotations, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(unpacked["sh_coeffs"][:, :4, :], sh_coeffs[:, :4, :], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(unpacked["sh_coeffs"][:, 4:, :], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(unpacked["color_alpha"][:, 3], color_alpha[:, 3], rtol=0.0, atol=0.0)


def test_max_sh_band_reuploads_bound_scene() -> None:
    renderer = object.__new__(GaussianRenderer)
    scene = object()
    uploaded: list[object] = []
    renderer._max_sh_band = 3
    renderer._sh_band = 3
    renderer._current_scene = scene
    renderer._scene_buffers = {}

    renderer.set_scene = lambda value: uploaded.append(value)

    GaussianRenderer.max_sh_band.fset(renderer, 1)

    assert renderer.max_sh_band == 1
    assert renderer.sh_band == 1
    assert uploaded == [scene]


def test_ensure_scene_buffers_reallocates_when_packed_layout_changes(monkeypatch) -> None:
    allocations: list[int] = []
    released: list[object] = []
    old_buffer = object()
    renderer = object.__new__(GaussianRenderer)
    renderer.device = object()
    renderer._max_sh_band = 3
    renderer._sh_band = 3
    renderer._scene_buffers = {"splat_params": old_buffer}
    renderer._scene_capacity = 16
    renderer._scene_count = 16
    renderer._scene_packed_param_count = 23
    renderer._resource_groups = SimpleNamespace(scene={})
    renderer._RW_BUFFER_USAGE = 0

    monkeypatch.setattr(gaussian_renderer_module, "defer_resource_releases", lambda buffers: released.extend(list(buffers)))
    monkeypatch.setattr(
        gaussian_renderer_module,
        "alloc_buffer",
        lambda _device, *, name, size, usage: allocations.append(size) or SimpleNamespace(name=name, size=size, usage=usage),
    )

    GaussianRenderer._ensure_scene_buffers(renderer, 8)

    assert released == [old_buffer]
    assert allocations == [renderer._scene_capacity * renderer.packed_trainable_param_count * renderer._U32_BYTES]
    assert renderer._scene_count == 8
    assert renderer._scene_packed_param_count == renderer.packed_trainable_param_count
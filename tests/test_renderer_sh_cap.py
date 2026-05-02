import numpy as np

from src.renderer import GaussianRenderer
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
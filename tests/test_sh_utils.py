from __future__ import annotations

import numpy as np

from src.scene.sh_utils import SH_C0, SH_C1, evaluate_sh_color, rgb_to_sh0


def test_evaluate_sh_color_uses_camera_to_splat_direction() -> None:
    sh_coeffs = np.zeros((2, 16, 3), dtype=np.float32)
    sh_coeffs[:, 3, 0] = 1.0
    view_dirs = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)

    colors = evaluate_sh_color(sh_coeffs, view_dirs)

    np.testing.assert_allclose(colors[0, 0], 0.5 - SH_C1, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(colors[1, 0], 0.5 + SH_C1, rtol=0.0, atol=1e-6)


def test_rgb_to_sh0_clamps_input_colors_to_display_range() -> None:
    sh0 = rgb_to_sh0(np.array([[-1.0, 0.5, 2.0]], dtype=np.float32))

    np.testing.assert_allclose(sh0[0], np.array([-0.5 / SH_C0, 0.0, 0.5 / SH_C0], dtype=np.float32), rtol=0.0, atol=1e-6)

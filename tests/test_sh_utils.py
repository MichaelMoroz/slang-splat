from __future__ import annotations

import numpy as np

from src.scene.sh_utils import SH_C1, evaluate_sh_color


def test_evaluate_sh_color_uses_camera_to_splat_direction() -> None:
    sh_coeffs = np.zeros((2, 16, 3), dtype=np.float32)
    sh_coeffs[:, 3, 0] = 1.0
    view_dirs = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)

    colors = evaluate_sh_color(sh_coeffs, view_dirs)

    np.testing.assert_allclose(colors[0, 0], 0.5 - SH_C1, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(colors[1, 0], 0.5 + SH_C1, rtol=0.0, atol=1e-6)

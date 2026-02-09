from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GaussianScene:
    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    sh_coeffs: np.ndarray

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float32)
        self.scales = np.ascontiguousarray(self.scales, dtype=np.float32)
        self.rotations = np.ascontiguousarray(self.rotations, dtype=np.float32)
        self.opacities = np.ascontiguousarray(self.opacities, dtype=np.float32).reshape(-1)
        self.colors = np.ascontiguousarray(self.colors, dtype=np.float32)
        self.sh_coeffs = np.ascontiguousarray(self.sh_coeffs, dtype=np.float32)
        n = self.positions.shape[0]
        if (
            self.scales.shape[0] != n
            or self.rotations.shape[0] != n
            or self.opacities.shape[0] != n
            or self.colors.shape[0] != n
            or self.sh_coeffs.shape[0] != n
        ):
            raise ValueError("All GaussianScene arrays must have the same first dimension.")

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def subset(self, max_splats: int) -> "GaussianScene":
        if max_splats <= 0 or max_splats >= self.count:
            return self
        return GaussianScene(
            positions=self.positions[:max_splats],
            scales=self.scales[:max_splats],
            rotations=self.rotations[:max_splats],
            opacities=self.opacities[:max_splats],
            colors=self.colors[:max_splats],
            sh_coeffs=self.sh_coeffs[:max_splats],
        )

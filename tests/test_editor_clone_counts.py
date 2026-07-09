"""Unit tests for the editor densify clone-count distribution.

The splat editor's "resample up" reuses the training refinement split. To do that it
turns the current selection + target ratio into a per-splat clone-count array (the same
override the refinement pass accepts). These tests cover that distribution logic without
needing a GPU device.
"""
from __future__ import annotations

import numpy as np

from src.training.gaussian_trainer import (
    _REFINEMENT_MAX_CLONES_PER_SPLAT,
    _build_editor_clone_counts,
)


def test_clone_counts_hit_exact_target_and_touch_only_parents() -> None:
    scene_count = 50
    parents = np.array([3, 7, 11, 19, 20, 33], dtype=np.intp)
    add_count = 17
    counts = _build_editor_clone_counts(parents, add_count, scene_count, np.random.default_rng(0))

    assert counts.shape == (scene_count,)
    assert counts.dtype == np.uint32
    assert int(counts.sum()) == add_count  # exact target when there is room
    non_zero = np.where(counts > 0)[0]
    assert set(non_zero.tolist()).issubset(set(parents.tolist()))  # never clones outside selection


def test_clone_counts_respect_per_parent_cap() -> None:
    parents = np.array([0, 1], dtype=np.intp)
    # Ask for far more than the two parents can hold; expect saturation at the cap.
    counts = _build_editor_clone_counts(parents, 999, 4, np.random.default_rng(1))
    assert int(counts.sum()) == 2 * _REFINEMENT_MAX_CLONES_PER_SPLAT
    assert counts[0] == _REFINEMENT_MAX_CLONES_PER_SPLAT
    assert counts[1] == _REFINEMENT_MAX_CLONES_PER_SPLAT


def test_clone_counts_empty_when_nothing_to_add() -> None:
    parents = np.array([2, 5], dtype=np.intp)
    assert int(_build_editor_clone_counts(parents, 0, 10, np.random.default_rng(2)).sum()) == 0
    empty = _build_editor_clone_counts(np.array([], dtype=np.intp), 5, 10, np.random.default_rng(3))
    assert int(empty.sum()) == 0

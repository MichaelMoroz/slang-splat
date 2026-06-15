from __future__ import annotations

from pathlib import Path

import pytest

from src.viewer.session_colmap_utils import (
    _sample_colmap_image_names,
    _suggest_images_root_from_dataset_root,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_sample_spreads_across_full_name_list() -> None:
    names = [f"{subset}/{i:03d}.png" for subset in ("A", "B", "C") for i in range(90)]
    sample = _sample_colmap_image_names(names, limit=12)
    subsets = {name.split("/")[0] for name in sample}
    # Every capture subset must be represented, not just the leading one.
    assert subsets == {"A", "B", "C"}


def test_suggest_images_root_with_partial_subset_and_prefixed_names(tmp_path: Path) -> None:
    # Reconstruction references A/, B/, C/ (sorted by id), but only the C subset
    # is present on disk under <root>/C — the turmalin-C dataset shape.
    root = tmp_path / "turmalin C"
    for i in range(190, 269):
        _touch(root / "C" / f"{i:03d}.png")
    names = (
        [f"A/{i:03d}.png" for i in range(90)]
        + [f"B/{i:03d}.png" for i in range(90, 190)]
        + [f"C/{i:03d}.png" for i in range(190, 269)]
    )
    assert _suggest_images_root_from_dataset_root(root, names) == root


def test_suggest_images_root_for_plain_names_in_subfolder(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for i in range(10):
        _touch(root / "images_4" / f"{i:03d}.png")
    names = [f"{i:03d}.png" for i in range(10)]
    assert _suggest_images_root_from_dataset_root(root, names) == root / "images_4"


def test_suggest_images_root_robust_to_extension_mismatch(tmp_path: Path) -> None:
    # COLMAP names use .jpg but the images on disk are .png (handled by the
    # basename/relative-stem fallback resolver).
    root = tmp_path / "dataset"
    for i in range(8):
        _touch(root / "frames" / f"{i:03d}.png")
    names = [f"{i:03d}.jpg" for i in range(8)]
    assert _suggest_images_root_from_dataset_root(root, names) == root / "frames"


def test_suggest_images_root_raises_when_no_images_present(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        _suggest_images_root_from_dataset_root(root, ["A/000.png", "A/001.png"])

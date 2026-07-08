"""Tests for the widest-coverage camera pose subset selection."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.scene._internal.colmap_ops import select_wide_coverage_image_ids
from src.viewer.session import _select_import_image_items


def _image(image_id: int, point_ids, *, camera_id: int = 1, position=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        image_id=image_id,
        camera_id=camera_id,
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        t_xyz=np.asarray(position, dtype=np.float64),
        points2d_point3d_ids=np.asarray(point_ids, dtype=np.int64),
    )


def _recon(images):
    return SimpleNamespace(images={image.image_id: image for image in images})


def test_covisibility_subset_drops_redundant_views_and_maximizes_coverage() -> None:
    # Images 1 and 2 see the SAME points; 3 and 4 each see a disjoint region.
    recon = _recon([
        _image(1, range(1, 11)),
        _image(2, range(1, 11)),
        _image(3, range(11, 21)),
        _image(4, range(21, 31)),
    ])
    selected = select_wide_coverage_image_ids(recon, [1, 2, 3, 4], 3)
    assert 3 in selected and 4 in selected  # the unique regions are always kept
    assert (1 in selected) ^ (2 in selected)  # exactly one of the redundant pair
    covered = set()
    region = {1: set(range(1, 11)), 2: set(range(1, 11)), 3: set(range(11, 21)), 4: set(range(21, 31))}
    for image_id in selected:
        covered |= region[image_id]
    assert len(covered) == 30  # widest possible coverage for a 3-subset


def test_subset_returns_all_when_target_covers_everything() -> None:
    recon = _recon([_image(i, range(i, i + 5)) for i in range(1, 5)])
    assert select_wide_coverage_image_ids(recon, [1, 2, 3, 4], 4) == [1, 2, 3, 4]
    assert select_wide_coverage_image_ids(recon, [1, 2, 3, 4], 99) == [1, 2, 3, 4]
    assert select_wide_coverage_image_ids(recon, [1, 2, 3, 4], 0) == [1, 2, 3, 4]


def test_coverage_subset_avoids_views_with_no_tracked_points() -> None:
    recon = _recon([
        _image(1, range(1, 11)),
        _image(2, range(11, 21)),
        _image(3, []),  # sees nothing -> contributes no coverage
        _image(4, range(21, 31)),
    ])
    assert 3 not in select_wide_coverage_image_ids(recon, [1, 2, 3, 4], 3)


def test_pose_fallback_spreads_selection_when_no_tracks() -> None:
    # No tracked points anywhere -> fall back to camera position/orientation spread.
    recon = _recon([_image(i, [], position=(-float(i), 0.0, 0.0)) for i in range(6)])
    selected = select_wide_coverage_image_ids(recon, list(range(6)), 3)
    assert 0 in selected and 5 in selected  # the two extremes are always chosen


def test_select_import_image_items_filters_camera_and_applies_subset() -> None:
    recon = _recon([
        _image(1, range(1, 11), camera_id=1),
        _image(2, range(1, 11), camera_id=1),  # redundant with 1
        _image(3, range(11, 21), camera_id=1),
        _image(4, range(21, 31), camera_id=2),  # different camera model
    ])
    # Only camera 1 selected, cap to 2 poses: keep the two widest-coverage of {1,2,3}.
    progress = SimpleNamespace(recon=recon, selected_camera_ids=(1,), max_pose_subset=2)
    items = _select_import_image_items(progress)
    ids = [image_id for image_id, _ in items]
    assert 4 not in ids  # camera 2 filtered out
    assert 3 in ids and len(ids) == 2 and (1 in ids) ^ (2 in ids)

    # No cap -> every camera-1 pose, camera 2 excluded.
    progress_all = SimpleNamespace(recon=recon, selected_camera_ids=(1,), max_pose_subset=0)
    assert [image_id for image_id, _ in _select_import_image_items(progress_all)] == [1, 2, 3]

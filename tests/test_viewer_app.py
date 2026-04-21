from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.viewer import app
from src.viewer.app import SplatViewer
from src.scene import GaussianScene
from src.training import TRAIN_BACKGROUND_MODE_RANDOM


def _viewer(keyboard_capture: bool = False, mouse_capture: bool = False) -> SimpleNamespace:
    viewer = SimpleNamespace()
    viewer.toolkit = SimpleNamespace(
        handle_keyboard_event=lambda event: keyboard_capture,
        handle_mouse_event=lambda event: mouse_capture,
        viewport_size=lambda: (0, 0),
    )
    viewer.s = SimpleNamespace(
        keys={},
        mouse_left=False,
        mouse_delta=app.spy.float2(1.0, 2.0),
        scroll_delta=0.0,
        mx=None,
        my=None,
    )
    return viewer


def test_keyboard_capture_blocks_camera_input() -> None:
    viewer = _viewer(keyboard_capture=True)
    event = SimpleNamespace(type=app.spy.KeyboardEventType.key_press, key=app.spy.KeyCode.w)

    app.SplatViewer.on_keyboard_event(viewer, event)

    assert viewer.s.keys == {}


def test_keyboard_capture_clears_released_key() -> None:
    viewer = _viewer(keyboard_capture=True)
    viewer.s.keys[app.spy.KeyCode.w] = True
    event = SimpleNamespace(type=app.spy.KeyboardEventType.key_release, key=app.spy.KeyCode.w)

    app.SplatViewer.on_keyboard_event(viewer, event)

    assert viewer.s.keys[app.spy.KeyCode.w] is False


def test_mouse_capture_resets_camera_drag_state() -> None:
    viewer = _viewer(mouse_capture=True)
    viewer.s.mouse_left = True
    event = SimpleNamespace(
        type=app.spy.MouseEventType.button_down,
        button=app.spy.MouseButton.left,
        pos=app.spy.float2(16.0, 24.0),
        scroll=app.spy.float2(0.0, 0.0),
    )

    app.SplatViewer.on_mouse_event(viewer, event)

    assert viewer.s.mouse_left is False


def test_mouse_capture_updates_reference_cursor_without_motion() -> None:
    viewer = _viewer(mouse_capture=True)
    event = SimpleNamespace(
        type=app.spy.MouseEventType.move,
        button=app.spy.MouseButton.unknown,
        pos=app.spy.float2(32.0, 48.0),
        scroll=app.spy.float2(0.0, 0.0),
    )

    app.SplatViewer.on_mouse_event(viewer, event)

    assert viewer.s.mx == 32.0
    assert viewer.s.my == 48.0
    assert viewer.s.mouse_delta.x == 0.0
    assert viewer.s.mouse_delta.y == 0.0


def test_mouse_passthrough_keeps_existing_camera_behavior() -> None:
    viewer = _viewer(mouse_capture=False)
    viewer.s.mx = 10.0
    viewer.s.my = 20.0
    viewer.s.mouse_delta = app.spy.float2(0.0, 0.0)
    event = SimpleNamespace(
        type=app.spy.MouseEventType.move,
        button=app.spy.MouseButton.unknown,
        pos=app.spy.float2(14.0, 26.0),
        scroll=app.spy.float2(0.0, 0.0),
    )

    app.SplatViewer.on_mouse_event(viewer, event)

    assert viewer.s.mouse_delta.x == 4.0
    assert viewer.s.mouse_delta.y == 6.0


def test_on_resize_records_exception_without_raising() -> None:
    viewer = SimpleNamespace(
        device=SimpleNamespace(wait=lambda: None),
        s=SimpleNamespace(renderer=SimpleNamespace(width=640, height=360), last_resize_exception="", last_error=""),
    )

    def _fail_resize(width: int, height: int) -> None:
        raise RuntimeError(f"resize failed: {width}x{height}")

    viewer._apply_resize = _fail_resize

    SplatViewer.on_resize(viewer, 800, 600)

    assert viewer.s.last_resize_exception == "resize failed: 800x600"
    assert viewer.s.last_error == "resize failed: 800x600"


def test_apply_resize_recreates_renderer_only_for_size_changes(monkeypatch) -> None:
    calls: list[tuple[int, int] | str] = []
    viewer = SimpleNamespace(
        device=SimpleNamespace(wait=lambda: calls.append("wait")),
        s=SimpleNamespace(renderer=SimpleNamespace(width=640, height=360), last_resize_exception="stale resize", last_error="stale error"),
    )

    monkeypatch.setattr("src.viewer.app.session.recreate_renderer", lambda viewer_obj, width, height: calls.append((width, height)))

    SplatViewer._apply_resize(viewer, 640, 360)
    SplatViewer._apply_resize(viewer, 800, 600)

    assert calls == ["wait", "wait", (800, 600)]
    assert viewer.s.last_resize_exception == ""
    assert viewer.s.last_error == ""


def test_apply_resize_prefers_viewport_size_when_available(monkeypatch) -> None:
    calls: list[tuple[int, int] | str] = []
    viewer = SimpleNamespace(
        toolkit=SimpleNamespace(viewport_size=lambda: (512, 288)),
        device=SimpleNamespace(wait=lambda: calls.append("wait")),
        s=SimpleNamespace(renderer=SimpleNamespace(width=640, height=360), last_resize_exception="", last_error=""),
    )

    monkeypatch.setattr("src.viewer.app.session.recreate_renderer", lambda viewer_obj, width, height: calls.append((width, height)))

    SplatViewer._apply_resize(viewer, 1280, 720)

    assert calls == ["wait", (512, 288)]


def test_render_records_toolkit_failure_without_raising() -> None:
    viewer = SimpleNamespace(
        s=SimpleNamespace(training_active=True, last_error="", last_render_exception=""),
    )

    def _fail_render(render_context: object) -> None:
        raise RuntimeError("toolkit boom")

    viewer._render_frame = _fail_render

    SplatViewer.render(viewer, SimpleNamespace())

    assert viewer.s.training_active is False
    assert viewer.s.last_error == "toolkit boom"
    assert viewer.s.last_render_exception == "toolkit boom"


def test_reinitialize_callback_defers_scene_rebuild_to_next_frame() -> None:
    viewer = SimpleNamespace(s=SimpleNamespace(training_active=True, pending_training_reinitialize=False))

    SplatViewer._reinitialize_callback(viewer)

    assert viewer.s.training_active is False
    assert viewer.s.pending_training_reinitialize is True


def test_default_training_params_include_training_control_fields() -> None:
    params = app.default_training_params()

    for key in (
        "background_mode",
        "density_regularizer",
        "depth_ratio_weight",
        "max_screen_fraction",
        "ssim_weight",
        "ssim_c2",
        "color_non_negative_reg",
        "sorting_order_dithering_stage1",
        "lr_schedule_stage1_lr",
        "depth_ratio_stage1_weight",
        "refinement_growth_ratio",
        "refinement_min_contribution",
        "train_subsample_factor",
    ):
        assert hasattr(params.training, key)


def test_viewer_background_defaults_to_custom_black() -> None:
    values = {
        "render_background_mode": 1,
        "render_background_color": (0.0, 0.0, 0.0),
        "background_mode": 1,
        "train_background_color": (1.0, 1.0, 1.0),
    }

    resolved = app._viewer_background_value(values.__getitem__)

    assert resolved == (0.0, 0.0, 0.0)


def test_viewer_background_can_follow_train_background_color() -> None:
    values = {
        "render_background_mode": 0,
        "render_background_color": (0.0, 0.0, 0.0),
        "background_mode": 0,
        "train_background_color": (0.25, 0.5, 0.75),
    }

    resolved = app._viewer_background_value(values.__getitem__)

    assert resolved == (0.25, 0.5, 0.75)


def test_fit_camera_to_training_views_orbits_camera_centroid() -> None:
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    fit = app._fit_camera_to_training_views(positions)

    assert fit is not None
    np.testing.assert_allclose(fit.center, np.array([0.0, 2.0 / 3.0, 0.0], dtype=np.float32), rtol=0.0, atol=1e-6)
    radius = 2.0 * float(np.max(np.linalg.norm(positions - fit.center[None, :], axis=1)))
    np.testing.assert_allclose(fit.position, fit.center + np.array([0.0, 0.0, -radius], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.isclose(fit.yaw, 0.0, rtol=0.0, atol=1e-6)
    assert np.isclose(fit.pitch, 0.0, rtol=0.0, atol=1e-6)


def test_apply_camera_fit_to_training_views_updates_free_fly_state() -> None:
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            near=0.1,
            far=120.0,
            camera_pos=app.spy.float3(0.0, 0.0, -3.0),
            yaw=1.0,
            pitch=1.0,
            move_speed=1.0,
            move_vel=app.spy.float3(1.0, 0.0, 0.0),
            rot_vel=app.spy.float2(1.0, 0.0),
        ),
        c=lambda _key: SimpleNamespace(value=0.0),
    )
    frames = [
        SimpleNamespace(make_camera=lambda near=0.1, far=120.0: SimpleNamespace(position=np.array([1.0, 0.0, 0.0], dtype=np.float32))),
        SimpleNamespace(make_camera=lambda near=0.1, far=120.0: SimpleNamespace(position=np.array([-1.0, 0.0, 0.0], dtype=np.float32))),
    ]

    changed = SplatViewer.apply_camera_fit_to_training_views(viewer, frames)

    assert changed is True
    np.testing.assert_allclose(np.asarray(viewer.s.camera_pos, dtype=np.float32), np.array([0.0, 0.0, -2.0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.yaw, 0.0, rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.pitch, 0.0, rtol=0.0, atol=1e-6)
    assert np.allclose(np.asarray(viewer.s.move_vel, dtype=np.float32), np.zeros((3,), dtype=np.float32))
    assert np.allclose(np.asarray(viewer.s.rot_vel, dtype=np.float32), np.zeros((2,), dtype=np.float32))


def test_renderer_params_maps_adam_momentum_to_grad_norm_log_range() -> None:
    controls = {
        "cached_raster_grad_atomic_mode": SimpleNamespace(value=1),
        "debug_mode": SimpleNamespace(value=app._DEBUG_MODE_VALUES.index("adam_momentum")),
        "radius_scale": SimpleNamespace(value=1.0),
        "alpha_cutoff": SimpleNamespace(value=1.0 / 255.0),
        "max_anisotropy": SimpleNamespace(value=32.0),
        "trans_threshold": SimpleNamespace(value=0.005),
        "cached_raster_grad_fixed_ro_local_range": SimpleNamespace(value=0.01),
        "cached_raster_grad_fixed_scale_range": SimpleNamespace(value=0.01),
        "cached_raster_grad_fixed_quat_range": SimpleNamespace(value=0.01),
        "cached_raster_grad_fixed_color_range": SimpleNamespace(value=0.2),
        "cached_raster_grad_fixed_opacity_range": SimpleNamespace(value=0.2),
        "debug_grad_norm_threshold": SimpleNamespace(value=2e-4),
        "debug_ellipse_thickness_px": SimpleNamespace(value=4.0),
        "debug_gaussian_scale_multiplier": SimpleNamespace(value=1.5),
        "debug_min_opacity": SimpleNamespace(value=0.05),
        "debug_opacity_multiplier": SimpleNamespace(value=2.0),
        "debug_ellipse_scale_multiplier": SimpleNamespace(value=0.75),
        "debug_splat_age_min": SimpleNamespace(value=0.0),
        "debug_splat_age_max": SimpleNamespace(value=1.0),
        "debug_density_min": SimpleNamespace(value=0.0),
        "debug_density_max": SimpleNamespace(value=20.0),
        "debug_contribution_min": SimpleNamespace(value=0.001),
        "debug_contribution_max": SimpleNamespace(value=1.0),
        "debug_sh_coeff_index": SimpleNamespace(value=0),
        "debug_depth_mean_min": SimpleNamespace(value=0.0),
        "debug_depth_mean_max": SimpleNamespace(value=10.0),
        "debug_depth_std_min": SimpleNamespace(value=0.0),
        "debug_depth_std_max": SimpleNamespace(value=0.5),
        "debug_depth_local_mismatch_min": SimpleNamespace(value=0.0),
        "debug_depth_local_mismatch_max": SimpleNamespace(value=0.5),
        "debug_depth_local_mismatch_smooth_radius": SimpleNamespace(value=2.0),
        "debug_depth_local_mismatch_reject_radius": SimpleNamespace(value=4.0),
    }
    viewer = SimpleNamespace(
        ui=SimpleNamespace(controls=controls),
        s=SimpleNamespace(list_capacity_multiplier=64, max_prepass_memory_mb=4096),
        c=lambda key: controls[key],
    )

    params = app.SplatViewer.renderer_params(viewer, allow_debug_overlays=True)

    assert params.debug_mode == "adam_momentum"
    assert params.debug_gaussian_scale_multiplier == 1.5
    assert params.debug_min_opacity == 0.05
    assert params.debug_opacity_multiplier == 2.0
    assert params.debug_ellipse_scale_multiplier == 0.75
    assert np.isclose(params.debug_adam_momentum_range[0], 2e-7, rtol=0.0, atol=1e-15)
    assert np.isclose(params.debug_adam_momentum_range[1], 2e-3, rtol=0.0, atol=1e-15)


def test_export_ply_callback_saves_active_scene(monkeypatch, tmp_path: Path) -> None:
    scene = GaussianScene(
        positions=np.zeros((1, 3), dtype=np.float32),
        scales=np.zeros((1, 3), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.5], dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    saved: dict[str, object] = {}
    viewer = SimpleNamespace(
        s=SimpleNamespace(trainer=None, scene=scene, last_error="stale"),
        _run_action=lambda action: action(),
        training_params=lambda: SimpleNamespace(training=SimpleNamespace(sh_band=0, use_sh=False)),
    )

    monkeypatch.setattr(app.spy.platform, "save_file_dialog", lambda *_args: tmp_path / "exported_scene")
    monkeypatch.setattr(app, "save_gaussian_ply", lambda path, src_scene, include_sh=True: saved.update(path=Path(path), scene=src_scene, include_sh=include_sh) or Path(path).resolve())

    SplatViewer._export_ply_callback(viewer)

    assert saved["scene"] is scene
    assert saved["path"] == tmp_path / "exported_scene.ply"
    assert saved["include_sh"] is False


def test_export_ply_callback_prefers_training_scene(monkeypatch, tmp_path: Path) -> None:
    base_scene = GaussianScene(
        positions=np.zeros((1, 3), dtype=np.float32),
        scales=np.zeros((1, 3), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.5], dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    trained_scene = GaussianScene(
        positions=np.ones((1, 3), dtype=np.float32),
        scales=np.zeros((1, 3), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.5], dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    saved: dict[str, object] = {}
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            trainer=SimpleNamespace(
                read_live_scene=lambda: trained_scene,
                training=SimpleNamespace(lr_schedule_enabled=False, sh_band=3, use_sh=True),
                state=SimpleNamespace(step=123),
            ),
            scene=base_scene,
            last_error="",
        ),
        _run_action=lambda action: action(),
    )

    monkeypatch.setattr(app.spy.platform, "save_file_dialog", lambda *_args: tmp_path / "trained_export.ply")
    monkeypatch.setattr(app, "save_gaussian_ply", lambda path, src_scene, include_sh=True: saved.update(path=Path(path), scene=src_scene, include_sh=include_sh) or Path(path).resolve())

    SplatViewer._export_ply_callback(viewer)

    assert saved["scene"] is trained_scene
    assert saved["include_sh"] is True

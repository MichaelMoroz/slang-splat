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
        mouse_right=False,
        mouse_delta=app.spy.float2(1.0, 2.0),
        scroll_delta=0.0,
        last_interaction_time=0.0,
        last_time=0.0,
        mx=None,
        my=None,
    )
    return viewer


def test_keyboard_capture_blocks_camera_input() -> None:
    viewer = _viewer(keyboard_capture=True)
    event = SimpleNamespace(type=app.spy.KeyboardEventType.key_press, key=app.spy.KeyCode.w)

    app.SplatViewer.on_keyboard_event(viewer, event)

    assert viewer.s.keys == {}


def test_keyboard_capture_marks_recent_interaction() -> None:
    viewer = _viewer(keyboard_capture=True)
    event = SimpleNamespace(type=app.spy.KeyboardEventType.key_press, key=app.spy.KeyCode.w)

    app.SplatViewer.on_keyboard_event(viewer, event)

    assert viewer.s.last_interaction_time > 0.0


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


def test_mouse_capture_resets_camera_pan_drag_state() -> None:
    viewer = _viewer(mouse_capture=True)
    viewer.s.mouse_right = True
    event = SimpleNamespace(
        type=app.spy.MouseEventType.button_down,
        button=app.spy.MouseButton.right,
        pos=app.spy.float2(16.0, 24.0),
        scroll=app.spy.float2(0.0, 0.0),
    )

    app.SplatViewer.on_mouse_event(viewer, event)

    assert viewer.s.mouse_right is False


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


def test_right_mouse_button_updates_pan_drag_state() -> None:
    viewer = _viewer(mouse_capture=False)
    down = SimpleNamespace(
        type=app.spy.MouseEventType.button_down,
        button=app.spy.MouseButton.right,
        pos=app.spy.float2(14.0, 26.0),
        scroll=app.spy.float2(0.0, 0.0),
    )
    up = SimpleNamespace(
        type=app.spy.MouseEventType.button_up,
        button=app.spy.MouseButton.right,
        pos=app.spy.float2(14.0, 26.0),
        scroll=app.spy.float2(0.0, 0.0),
    )

    app.SplatViewer.on_mouse_event(viewer, down)
    app.SplatViewer.on_mouse_event(viewer, up)

    assert viewer.s.mouse_right is False


def test_update_camera_right_drag_pans_along_view_plane() -> None:
    controls = {
        "move_speed": SimpleNamespace(value=2.0),
        "fov": SimpleNamespace(value=60.0),
    }
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            move_speed=2.0,
            fov_y=60.0,
            scroll_delta=0.0,
            mouse_delta=app.spy.float2(10.0, -20.0),
            mouse_left=False,
            mouse_right=True,
            look_speed=0.003,
            rot_vel=app.spy.float2(0.0, 0.0),
            yaw=0.0,
            pitch=0.0,
            up=app.spy.float3(0.0, 1.0, 0.0),
            keys={},
            move_vel=app.spy.float3(0.0, 0.0, 0.0),
            camera_pos=app.spy.float3(0.0, 0.0, -3.0),
        ),
        c=lambda key: controls[key],
        _forward=lambda: app.spy.float3(0.0, 0.0, 1.0),
    )

    app.SplatViewer.update_camera(viewer, 0.1)

    np.testing.assert_allclose(np.asarray(viewer.s.camera_pos, dtype=np.float32), np.array([-0.06, 0.12, -3.0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert viewer.s.mouse_delta.x == 0.0
    assert viewer.s.mouse_delta.y == 0.0


def test_update_camera_marks_recent_interaction_while_keyboard_motion_is_active() -> None:
    controls = {
        "move_speed": SimpleNamespace(value=2.0),
        "fov": SimpleNamespace(value=60.0),
    }
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            move_speed=2.0,
            fov_y=60.0,
            scroll_delta=0.0,
            mouse_delta=app.spy.float2(0.0, 0.0),
            mouse_left=False,
            mouse_right=False,
            look_speed=0.003,
            rot_vel=app.spy.float2(0.0, 0.0),
            yaw=0.0,
            pitch=0.0,
            up=app.spy.float3(0.0, 1.0, 0.0),
            keys={app.spy.KeyCode.w: True},
            move_vel=app.spy.float3(0.0, 0.0, 0.0),
            camera_pos=app.spy.float3(0.0, 0.0, -3.0),
            last_time=123.0,
            last_interaction_time=0.0,
        ),
        c=lambda key: controls[key],
        _forward=lambda: app.spy.float3(0.0, 0.0, 1.0),
    )

    app.SplatViewer.update_camera(viewer, 0.1)

    assert viewer.s.last_interaction_time == 123.0


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

    app.SplatViewer._apply_resize(viewer, 640, 360)
    app.SplatViewer._apply_resize(viewer, 800, 600)

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

    app.SplatViewer._apply_resize(viewer, 1280, 720)

    assert calls == ["wait", (512, 288)]


def test_precompile_runtime_shaders_loads_lazy_runtime_shader_sets(monkeypatch) -> None:
    item_calls: list[dict[str, tuple[str, str, str]]] = []
    kernel_calls: list[tuple[str, dict[str, str]]] = []

    monkeypatch.setattr(app, "load_compute_items", lambda _device, specs: item_calls.append({name: (kind, str(path), entry) for name, (kind, path, entry) in specs.items()}) or {})
    monkeypatch.setattr(app, "load_compute_kernels", lambda _device, path, entries: kernel_calls.append((str(path), dict(entries))) or {})

    app._precompile_runtime_shaders(object())

    assert any(specs.get("scan_blocks_pipeline") == ("pipeline", str(app.SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang"), "csPrefixScanBlocks") for specs in item_calls)
    assert any(specs.get("scatter") == ("pipeline", str(app.SHADER_ROOT / "utility" / "radix_sort" / "scatter.slang"), "csRadixScatter") for specs in item_calls)
    assert any(specs.get("_k_raster_ppisp") == ("kernel", str(app.SHADER_ROOT / "renderer" / "gaussian_raster_stage.slang"), "csRasterizePPISP") for specs in item_calls)
    assert any(path == str(app.SHADER_ROOT / "renderer" / "gaussian_raster_stage.slang") and entries.get("training_forward") == "csRasterizeTrainingForwardFixed" for path, entries in kernel_calls)
    assert any(path == str(app.SHADER_ROOT / "renderer" / "gaussian_raster_stage.slang") and entries.get("training_forward") == "csRasterizeTrainingForwardFloat" for path, entries in kernel_calls)
    assert any(path == str(app.SHADER_ROOT / "renderer" / "gaussian_training_stage.slang") and entries.get("debug_target_sample_kernel") == "csSampleTrainingDebugTarget" for path, entries in kernel_calls)
    assert any(path == str(app.SHADER_ROOT / "utility" / "blur" / "separable_gaussian_blur.slang") and entries.get("horizontal") == "csGaussianBlurHorizontal" for path, entries in kernel_calls)
    assert any(path == str(app.TrainingImageColorInitializer.SHADER_PATH) and entries.get("sample") == "csSampleTrainingImageColorInit" for path, entries in kernel_calls)
    assert any(path == str(app.SHADER_ROOT / "utility" / "optimizer" / "optimizer.slang") and entries.get("adam_step") == "csAdamStepPacked" for path, entries in kernel_calls)


def test_viewer_init_precompiles_runtime_shaders_before_renderer_setup(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(app._ViewerWindowHost, "__init__", lambda self, _app, **_kwargs: (setattr(self, "_device", "device"), setattr(self, "_app", _app), setattr(self, "_window", None), setattr(self, "_surface", None), setattr(self, "_terminated", False)))
    monkeypatch.setattr(app._ViewerWindowHost, "device", property(lambda _self: "device"))
    monkeypatch.setattr(app, "ViewerState", lambda **_kwargs: SimpleNamespace(list_capacity_multiplier=64, max_prepass_memory_mb=4096))
    monkeypatch.setattr(app, "_precompile_runtime_shaders", lambda device: calls.append(("precompile", device)))
    monkeypatch.setattr(app.GaussianRenderSettings, "from_renderer_params", lambda *_args, **_kwargs: SimpleNamespace(create_renderer=lambda device: calls.append(("renderer", device)) or "renderer"))
    monkeypatch.setattr(app, "build_ui", lambda renderer: calls.append(("ui", renderer)) or "ui")
    monkeypatch.setattr(app, "create_toolkit_window", lambda device, width, height: calls.append(("toolkit", (device, width, height))) or SimpleNamespace(callbacks=SimpleNamespace()))
    monkeypatch.setattr(app.SplatViewer, "_bind_toolkit_callbacks", lambda self: calls.append(("bind", None)))
    monkeypatch.setattr(app.session, "create_debug_shaders", lambda viewer: calls.append(("debug", viewer.device)))

    app.SplatViewer(SimpleNamespace(), width=640, height=360)

    assert calls[:2] == [("precompile", "device"), ("renderer", "device")]


def test_save_defaults_callback_updates_cli_common_render(monkeypatch) -> None:
    written: dict[str, object] = {}
    viewer = SimpleNamespace(ui=SimpleNamespace(_values={}))
    exported = {
        "renderer": {"radius_scale": 1.25},
        "cli": {"common_render": {"cached_raster_grad_atomic_mode": "fixed", "cached_raster_grad_fixed_scale_range": 512.0}},
        "viewer": {"controls": {}, "import": {}, "ui": {"graphics_api": "dx12"}},
    }

    monkeypatch.setattr(
        app,
        "load_defaults",
        lambda: {
            "training_build_args": {},
            "renderer": {},
            "cli": {"common_render": {}},
            "viewer": {"controls": {}, "import": {}, "ui": {}},
        },
    )
    monkeypatch.setattr(app, "export_repo_defaults_from_ui_values", lambda values: exported)
    monkeypatch.setattr(app, "write_defaults", lambda defaults: written.setdefault("defaults", defaults))

    app.SplatViewer._save_defaults_callback(viewer)

    assert written["defaults"]["renderer"] == exported["renderer"]
    assert written["defaults"]["cli"]["common_render"] == exported["cli"]["common_render"]
    assert written["defaults"]["viewer"]["ui"]["graphics_api"] == "dx12"


def test_set_graphics_api_callback_updates_viewer_defaults(monkeypatch) -> None:
    written: dict[str, object] = {}
    status = SimpleNamespace(text="")
    viewer = SimpleNamespace(
        device=SimpleNamespace(info=SimpleNamespace(api_name="Vulkan")),
        ui=SimpleNamespace(_values={"graphics_api": "vulkan"}),
        t=lambda _key: status,
        s=SimpleNamespace(last_error="stale"),
    )
    defaults = {"viewer": {"ui": {}}, "cli": {}, "renderer": {}, "training_build_args": {}}

    monkeypatch.setattr(app, "load_defaults", lambda: defaults)
    monkeypatch.setattr(app, "write_defaults", lambda data: written.setdefault("defaults", data))

    app.SplatViewer._set_graphics_api_callback(viewer, "dx12")

    assert written["defaults"]["viewer"]["ui"]["graphics_api"] == "dx12"
    assert viewer.ui._values["graphics_api"] == "dx12"
    assert "restart required" in status.text.lower()
    assert viewer.s.last_error == ""


def test_main_uses_preferred_graphics_api_from_defaults(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    viewer = SimpleNamespace(run=lambda: calls.append(("run", None)), shutdown=lambda: calls.append(("shutdown", None)))

    monkeypatch.setattr(app, "load_defaults", lambda: {"viewer": {"ui": {"graphics_api": "dx12"}}})
    monkeypatch.setattr(app, "_compute_view_geometry", lambda: (640, 360))
    monkeypatch.setattr(app, "create_default_device", lambda **kwargs: calls.append(("device", kwargs["device_type"])) or "device")
    monkeypatch.setattr(app.spy, "App", lambda device: calls.append(("app", device)) or "app")
    monkeypatch.setattr(app, "SplatViewer", lambda _app, **kwargs: calls.append(("viewer", kwargs)) or viewer)

    result = app.main()

    assert result == 0
    assert calls[0] == ("device", app.spy.DeviceType.d3d12)


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

    assert viewer.s.training_active is True
    assert viewer.s.pending_training_reinitialize is True


def test_move_to_training_camera_callback_routes_through_run_action(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    viewer = SimpleNamespace(_run_action=lambda action: calls.append(("run_action", action)))

    monkeypatch.setattr(app.session, "move_main_camera_to_selected_training_frame", lambda viewer_obj: calls.append(("move", viewer_obj)))

    SplatViewer._move_to_training_camera_callback(viewer)

    assert calls[0][0] == "run_action"
    calls[0][1]()
    assert calls[1] == ("move", viewer)


def test_reset_camera_callback_routes_through_run_action(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    viewer = SimpleNamespace(_run_action=lambda action: calls.append(("run_action", action)))

    monkeypatch.setattr(app.session, "reset_main_camera", lambda viewer_obj: calls.append(("reset", viewer_obj)))

    SplatViewer._reset_camera_callback(viewer)

    assert calls[0][0] == "run_action"
    calls[0][1]()
    assert calls[1] == ("reset", viewer)


def test_capture_python_frame_callback_routes_through_run_action() -> None:
    calls: list[tuple[str, object]] = []
    viewer = SimpleNamespace(
        s=SimpleNamespace(pending_python_frame_capture=False),
        _run_action=lambda action: calls.append(("run_action", action)),
    )

    app.SplatViewer._capture_python_frame_callback(viewer)

    assert calls[0][0] == "run_action"
    calls[0][1]()
    assert viewer.s.pending_python_frame_capture is True


def test_capture_renderdoc_frame_callback_routes_through_run_action() -> None:
    calls: list[tuple[str, object]] = []
    viewer = SimpleNamespace(
        s=SimpleNamespace(pending_renderdoc_frame_capture=False),
        _run_action=lambda action: calls.append(("run_action", action)),
    )

    app.SplatViewer._capture_renderdoc_frame_callback(viewer)

    assert calls[0][0] == "run_action"
    calls[0][1]()
    assert viewer.s.pending_renderdoc_frame_capture is True


    params = app.default_training_params()

    for key in (
        "background_mode",
        "density_regularizer",
        "position_push_away_from_camera_step",
        "position_push_away_from_camera_step_stage1",
        "max_visible_angle_deg",
        "ssim_weight",
        "ssim_c2",
        "sorting_order_dithering_stage1",
        "lr_schedule_stage1_lr",
        "refinement_target_splat_ratio",
        "refinement_max_growth_per_step",
        "refinement_max_prune_per_step",
        "refinement_contribution_area_exponent",
        "refinement_contribution_view_count_exponent",
        "refinement_prune_lowest_contribution_ratio_stage1",
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


def test_request_exit_callback_opens_confirmation() -> None:
    viewer = SimpleNamespace(ui=SimpleNamespace(_values={}))

    SplatViewer._request_exit_callback(viewer)

    assert viewer.ui._values["_exit_confirmation_open"] is True
    assert viewer._exit_confirmed is False


def test_cancel_exit_callback_clears_confirmation() -> None:
    viewer = SimpleNamespace(ui=SimpleNamespace(_values={"_exit_confirmation_open": True}))

    SplatViewer._cancel_exit_callback(viewer)

    assert viewer.ui._values["_exit_confirmation_open"] is False


def test_confirm_exit_callback_requests_termination() -> None:
    terminated: list[str] = []
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"_exit_confirmation_open": True}),
        _app=SimpleNamespace(terminate=lambda: terminated.append("terminate")),
    )

    SplatViewer._confirm_exit_callback(viewer)

    assert viewer.ui._values["_exit_confirmation_open"] is False
    assert viewer._exit_confirmed is True
    assert terminated == ["terminate"]


def test_window_host_run_recreates_window_after_close_request(monkeypatch) -> None:
    calls: list[str] = []
    windows: list[object] = []
    close_sequence = iter((True, False, True))

    class _Window:
        def __init__(self, width: int, height: int, title: str, resizable: bool) -> None:
            self.width = width
            self.height = height
            self.title = title
            self.resizable = resizable
            self.position = None
            self.on_resize = None
            self.on_keyboard_event = None
            self.on_mouse_event = None
            windows.append(self)

        def process_events(self) -> None:
            calls.append("events")

        def should_close(self) -> bool:
            return next(close_sequence)

        def close(self) -> None:
            calls.append("window_close")

    class _Surface:
        def configure(self, width: int, height: int, format=app.spy.Format.undefined, vsync: bool = False) -> None:
            calls.append(f"configure:{width}x{height}:{bool(vsync)}")

        def unconfigure(self) -> None:
            calls.append("unconfigure")

        def acquire_next_image(self):
            return SimpleNamespace(width=64, height=64)

        def present(self) -> None:
            calls.append("present")

    device = SimpleNamespace(
        create_surface=lambda _window: _Surface(),
        create_command_encoder=lambda: SimpleNamespace(finish=lambda: "command_buffer"),
        submit_command_buffer=lambda command_buffer: calls.append(str(command_buffer)),
    )
    host = object.__new__(app._ViewerWindowHost)
    host._app = SimpleNamespace(device=device)
    host._device = device
    host._window_width = 64
    host._window_height = 64
    host._window_title = "Viewer"
    host._window_resizable = True
    host._surface_format = app.spy.Format.undefined
    host._enable_vsync = False
    host._window = None
    host._surface = None
    host._window_position = None
    host._terminated = False
    host._exit_confirmed = False
    host.ui = SimpleNamespace(_values={})
    host.render = lambda render_context: (calls.append(f"render:{render_context.surface_texture.width}x{render_context.surface_texture.height}"), setattr(host, "_terminated", True))
    host.on_resize = lambda *_args: None
    host.on_keyboard_event = lambda *_args: None
    host.on_mouse_event = lambda *_args: None

    monkeypatch.setattr(app.spy, "Window", _Window)

    app._ViewerWindowHost._recreate_window(host, open_exit_confirmation=False)
    app._ViewerWindowHost.run(host)

    assert len(windows) == 2
    assert host.ui._values["_exit_confirmation_open"] is True
    assert calls.count("unconfigure") == 1
    assert "render:64x64" in calls
    assert "present" in calls


def test_window_host_run_ignores_reopened_close_until_first_present(monkeypatch) -> None:
    calls: list[str] = []
    windows: list[object] = []

    class _Window:
        def __init__(self, width: int, height: int, title: str, resizable: bool) -> None:
            self.width = width
            self.height = height
            self.title = title
            self.resizable = resizable
            self.position = None
            self.on_resize = None
            self.on_keyboard_event = None
            self.on_mouse_event = None
            self._should_close_calls = 0
            self._index = len(windows)
            windows.append(self)

        def process_events(self) -> None:
            calls.append(f"events:{self._index}")

        def should_close(self) -> bool:
            self._should_close_calls += 1
            if self._index == 0:
                return True
            return self._should_close_calls == 1

        def close(self) -> None:
            calls.append(f"window_close:{self._index}")

    class _Surface:
        def configure(self, width: int, height: int, format=app.spy.Format.undefined, vsync: bool = False) -> None:
            calls.append(f"configure:{width}x{height}:{bool(vsync)}")

        def unconfigure(self) -> None:
            calls.append("unconfigure")

        def acquire_next_image(self):
            return SimpleNamespace(width=64, height=64)

        def present(self) -> None:
            calls.append("present")

    device = SimpleNamespace(
        create_surface=lambda _window: _Surface(),
        create_command_encoder=lambda: SimpleNamespace(finish=lambda: "command_buffer"),
        submit_command_buffer=lambda command_buffer: calls.append(str(command_buffer)),
    )
    host = object.__new__(app._ViewerWindowHost)
    host._app = SimpleNamespace(device=device)
    host._device = device
    host._window_width = 64
    host._window_height = 64
    host._window_title = "Viewer"
    host._window_resizable = True
    host._surface_format = app.spy.Format.undefined
    host._enable_vsync = False
    host._window = None
    host._surface = None
    host._surface_suspended = False
    host._window_position = None
    host._terminated = False
    host._exit_confirmed = False
    host._ignore_close_until_present = False
    host.ui = SimpleNamespace(_values={})
    host.render = lambda render_context: (calls.append(f"render:{render_context.surface_texture.width}x{render_context.surface_texture.height}"), setattr(host, "_terminated", True))
    host.on_resize = lambda *_args: None
    host.on_keyboard_event = lambda *_args: None
    host.on_mouse_event = lambda *_args: None

    monkeypatch.setattr(app.spy, "Window", _Window)

    app._ViewerWindowHost._recreate_window(host, open_exit_confirmation=False)
    app._ViewerWindowHost.run(host)

    assert len(windows) == 2
    assert host.ui._values["_exit_confirmation_open"] is True
    assert "render:64x64" in calls
    assert "present" in calls


def test_window_host_close_issues_native_close_even_when_already_closing() -> None:
    calls: list[str] = []
    host = SimpleNamespace(
        _terminated=False,
        _window=SimpleNamespace(close=lambda: calls.append("window_close")),
    )

    app._ViewerWindowHost.close(host)

    assert host._terminated is True
    assert calls == ["window_close"]


def test_window_host_run_reconfigures_surface_after_acquire_failure(monkeypatch) -> None:
    calls: list[str] = []
    windows: list[object] = []

    class _Window:
        def __init__(self, width: int, height: int, title: str, resizable: bool) -> None:
            self.width = width
            self.height = height
            self.title = title
            self.resizable = resizable
            self.position = None
            self.on_resize = None
            self.on_keyboard_event = None
            self.on_mouse_event = None
            windows.append(self)

        def process_events(self) -> None:
            calls.append("events")

        def should_close(self) -> bool:
            return False

        def close(self) -> None:
            calls.append("window_close")

    class _Surface:
        def __init__(self) -> None:
            self._acquire_calls = 0

        def configure(self, width: int, height: int, format=app.spy.Format.undefined, vsync: bool = False) -> None:
            calls.append(f"configure:{width}x{height}:{bool(vsync)}")

        def unconfigure(self) -> None:
            calls.append("unconfigure")

        def acquire_next_image(self):
            self._acquire_calls += 1
            if self._acquire_calls == 1:
                raise RuntimeError("surface acquire failed")
            return SimpleNamespace(width=64, height=64)

        def present(self) -> None:
            calls.append("present")

    device = SimpleNamespace(
        create_surface=lambda _window: _Surface(),
        create_command_encoder=lambda: SimpleNamespace(finish=lambda: "command_buffer"),
        submit_command_buffer=lambda command_buffer: calls.append(str(command_buffer)),
    )
    host = object.__new__(app._ViewerWindowHost)
    host._app = SimpleNamespace(device=device)
    host._device = device
    host._window_width = 64
    host._window_height = 64
    host._window_title = "Viewer"
    host._window_resizable = True
    host._surface_format = app.spy.Format.undefined
    host._enable_vsync = False
    host._window = None
    host._surface = None
    host._window_position = None
    host._terminated = False
    host._exit_confirmed = False
    host.ui = SimpleNamespace(_values={})
    host.render = lambda render_context: (calls.append(f"render:{render_context.surface_texture.width}x{render_context.surface_texture.height}"), setattr(host, "_terminated", True))
    host.on_resize = lambda *_args: None
    host.on_keyboard_event = lambda *_args: None
    host.on_mouse_event = lambda *_args: None

    monkeypatch.setattr(app.spy, "Window", _Window)

    app._ViewerWindowHost._recreate_window(host, open_exit_confirmation=False)
    app._ViewerWindowHost.run(host)

    assert len(windows) == 1
    assert calls.count("configure:64x64:False") == 2
    assert "render:64x64" in calls
    assert "present" in calls


def test_window_host_run_reconfigures_surface_after_present_failure(monkeypatch) -> None:
    calls: list[str] = []
    windows: list[object] = []
    state: dict[str, object] = {}

    class _Window:
        def __init__(self, width: int, height: int, title: str, resizable: bool) -> None:
            self.width = width
            self.height = height
            self.title = title
            self.resizable = resizable
            self.position = None
            self.on_resize = None
            self.on_keyboard_event = None
            self.on_mouse_event = None
            windows.append(self)

        def process_events(self) -> None:
            calls.append("events")

        def should_close(self) -> bool:
            return False

        def close(self) -> None:
            calls.append("window_close")

    class _Surface:
        def __init__(self) -> None:
            self._present_calls = 0

        def configure(self, width: int, height: int, format=app.spy.Format.undefined, vsync: bool = False) -> None:
            calls.append(f"configure:{width}x{height}:{bool(vsync)}")

        def unconfigure(self) -> None:
            calls.append("unconfigure")

        def acquire_next_image(self):
            return SimpleNamespace(width=64, height=64)

        def present(self) -> None:
            self._present_calls += 1
            if self._present_calls == 1:
                raise RuntimeError("surface present failed")
            calls.append("present")
            state["host"]._terminated = True

    device = SimpleNamespace(
        create_surface=lambda _window: _Surface(),
        create_command_encoder=lambda: SimpleNamespace(finish=lambda: "command_buffer"),
        submit_command_buffer=lambda command_buffer: calls.append(str(command_buffer)),
    )
    host = object.__new__(app._ViewerWindowHost)
    host._app = SimpleNamespace(device=device)
    host._device = device
    host._window_width = 64
    host._window_height = 64
    host._window_title = "Viewer"
    host._window_resizable = True
    host._surface_format = app.spy.Format.undefined
    host._enable_vsync = False
    host._window = None
    host._surface = None
    host._window_position = None
    host._terminated = False
    host._exit_confirmed = False
    host._surface_suspended = False
    host._ignore_close_until_present = False
    host.ui = SimpleNamespace(_values={})
    state["host"] = host

    def _render(render_context) -> None:
        calls.append(f"render:{render_context.surface_texture.width}x{render_context.surface_texture.height}")

    host.render = _render
    host.on_resize = lambda *_args: None
    host.on_keyboard_event = lambda *_args: None
    host.on_mouse_event = lambda *_args: None

    monkeypatch.setattr(app.spy, "Window", _Window)

    app._ViewerWindowHost._recreate_window(host, open_exit_confirmation=False)
    app._ViewerWindowHost.run(host)

    assert len(windows) == 1
    assert calls.count("configure:64x64:False") == 2
    assert calls.count("render:64x64") == 2
    assert calls.count("command_buffer") == 2
    assert calls.count("present") == 1


def test_window_host_run_suspends_surface_while_minimized(monkeypatch) -> None:
    calls: list[str] = []

    class _Window:
        def __init__(self, width: int, height: int, title: str, resizable: bool) -> None:
            self.width = width
            self.height = height
            self.title = title
            self.resizable = resizable
            self.position = None
            self.on_resize = None
            self.on_keyboard_event = None
            self.on_mouse_event = None
            self._event_count = 0

        def process_events(self) -> None:
            calls.append("events")
            self._event_count += 1
            if self._event_count == 1:
                self.width = 0
                self.height = 0
                self.on_resize(0, 0)
            elif self._event_count == 2:
                self.width = 64
                self.height = 64
                self.on_resize(64, 64)

        def should_close(self) -> bool:
            return False

        def close(self) -> None:
            calls.append("window_close")

    class _Surface:
        def configure(self, width: int, height: int, format=app.spy.Format.undefined, vsync: bool = False) -> None:
            calls.append(f"configure:{width}x{height}:{bool(vsync)}")

        def unconfigure(self) -> None:
            calls.append("unconfigure")

        def acquire_next_image(self):
            calls.append("acquire")
            return SimpleNamespace(width=64, height=64)

        def present(self) -> None:
            calls.append("present")

    device = SimpleNamespace(
        create_surface=lambda _window: _Surface(),
        create_command_encoder=lambda: SimpleNamespace(finish=lambda: "command_buffer"),
        submit_command_buffer=lambda command_buffer: calls.append(str(command_buffer)),
    )
    host = object.__new__(app._ViewerWindowHost)
    host._app = SimpleNamespace(device=device)
    host._device = device
    host._window_width = 64
    host._window_height = 64
    host._window_title = "Viewer"
    host._window_resizable = True
    host._surface_format = app.spy.Format.undefined
    host._enable_vsync = False
    host._window = None
    host._surface = None
    host._surface_suspended = False
    host._window_position = None
    host._terminated = False
    host._exit_confirmed = False
    host.ui = SimpleNamespace(_values={})
    host.render = lambda render_context: (calls.append(f"render:{render_context.surface_texture.width}x{render_context.surface_texture.height}"), setattr(host, "_terminated", True))
    host.on_resize = lambda *_args: calls.append("resize")
    host.on_keyboard_event = lambda *_args: None
    host.on_mouse_event = lambda *_args: None

    monkeypatch.setattr(app.spy, "Window", _Window)

    app._ViewerWindowHost._recreate_window(host, open_exit_confirmation=False)
    app._ViewerWindowHost.run(host)

    assert calls.count("acquire") == 1
    assert calls.count("unconfigure") == 1
    assert calls.count("configure:64x64:False") == 2
    assert calls.count("resize") == 1
    assert "render:64x64" in calls
    assert "present" in calls


def test_shutdown_unconfigures_surface_and_drops_window() -> None:
    calls: list[str] = []
    host = SimpleNamespace(
        _surface=SimpleNamespace(unconfigure=lambda: calls.append("unconfigure")),
        _window=object(),
    )

    app._ViewerWindowHost.shutdown(host)

    assert calls == ["unconfigure"]
    assert host._surface is None
    assert host._window is None


def test_apply_camera_pose_updates_free_fly_state() -> None:
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            near=0.1,
            far=120.0,
            camera_pos=app.spy.float3(0.0, 0.0, -3.0),
            yaw=1.0,
            pitch=1.0,
            up=app.spy.float3(0.0, 1.0, 0.0),
            move_speed=1.0,
            move_vel=app.spy.float3(1.0, 0.0, 0.0),
            rot_vel=app.spy.float2(1.0, 0.0),
        ),
        c=lambda _key: SimpleNamespace(value=1.0),
    )
    camera = SimpleNamespace(
        position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        target=np.array([1.0, 2.0, 4.0], dtype=np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        near=0.25,
        far=250.0,
    )

    SplatViewer.apply_camera_pose(viewer, camera, move_speed=3.5)

    np.testing.assert_allclose(np.asarray(viewer.s.camera_pos, dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(viewer.s.up, dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.yaw, 0.0, rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.pitch, 0.0, rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.near, 0.25, rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.far, 250.0, rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.move_speed, 3.5, rtol=0.0, atol=1e-6)
    assert np.allclose(np.asarray(viewer.s.move_vel, dtype=np.float32), np.zeros((3,), dtype=np.float32))
    assert np.allclose(np.asarray(viewer.s.rot_vel, dtype=np.float32), np.zeros((2,), dtype=np.float32))


def test_apply_camera_pose_keeps_pan_controls_in_free_fly_plane() -> None:
    controls = {
        "move_speed": SimpleNamespace(value=2.0),
        "fov": SimpleNamespace(value=60.0),
    }
    viewer = SimpleNamespace()
    viewer.c = lambda key: controls[key]
    viewer.s = SimpleNamespace(
        camera_pos=app.spy.float3(0.0, 0.0, -3.0),
        yaw=0.0,
        pitch=0.0,
        up=app.spy.float3(0.0, 1.0, 0.0),
        near=0.1,
        far=120.0,
        move_speed=2.0,
        move_vel=app.spy.float3(0.0, 0.0, 0.0),
        rot_vel=app.spy.float2(0.0, 0.0),
        mouse_left=False,
        mouse_right=True,
        mouse_delta=app.spy.float2(10.0, -20.0),
        scroll_delta=0.0,
        look_speed=0.003,
        keys={},
        fov_y=60.0,
    )
    viewer._forward = lambda: SplatViewer._forward(viewer)
    rolled_camera = SimpleNamespace(
        position=np.array([0.0, 0.0, -3.0], dtype=np.float32),
        target=np.array([0.0, 0.0, -2.0], dtype=np.float32),
        up=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        near=0.1,
        far=120.0,
    )

    SplatViewer.apply_camera_pose(viewer, rolled_camera, move_speed=2.0)
    SplatViewer.update_camera(viewer, 0.1)

    np.testing.assert_allclose(np.asarray(viewer.s.up, dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(viewer.s.camera_pos, dtype=np.float32), np.array([-0.06, 0.12, -3.0], dtype=np.float32), rtol=0.0, atol=1e-6)


def test_apply_camera_position_preserves_orientation() -> None:
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            near=0.1,
            far=120.0,
            camera_pos=app.spy.float3(0.0, 0.0, -3.0),
            yaw=1.25,
            pitch=-0.5,
            up=app.spy.float3(0.0, 1.0, 0.0),
            move_speed=1.0,
            move_vel=app.spy.float3(1.0, 0.0, 0.0),
            rot_vel=app.spy.float2(1.0, 0.0),
        ),
        c=lambda _key: SimpleNamespace(value=1.0),
    )
    camera = SimpleNamespace(position=np.array([4.0, 5.0, 6.0], dtype=np.float32))

    SplatViewer.apply_camera_position(viewer, camera, near=0.25, far=250.0, move_speed=3.5)

    np.testing.assert_allclose(np.asarray(viewer.s.camera_pos, dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.yaw, 1.25, rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.pitch, -0.5, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(viewer.s.up, dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.near, 0.25, rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.far, 250.0, rtol=0.0, atol=1e-6)
    assert np.isclose(viewer.s.move_speed, 3.5, rtol=0.0, atol=1e-6)
    assert np.allclose(np.asarray(viewer.s.move_vel, dtype=np.float32), np.zeros((3,), dtype=np.float32))
    assert np.allclose(np.asarray(viewer.s.rot_vel, dtype=np.float32), np.zeros((2,), dtype=np.float32))


def test_renderer_params_maps_adam_moment_modes_to_grad_norm_log_range() -> None:
    controls = {
        "cached_raster_grad_atomic_mode": SimpleNamespace(value=1),
        "debug_mode": SimpleNamespace(value=app._DEBUG_MODE_VALUES.index("adam_momentum")),
        "radius_scale": SimpleNamespace(value=1.0),
        "alpha_cutoff": SimpleNamespace(value=1.0 / 255.0),
        "max_anisotropy": SimpleNamespace(value=32.0),
        "trans_threshold": SimpleNamespace(value=0.005),
        "cached_raster_grad_fixed_ro_local_range": SimpleNamespace(value=2.0),
        "cached_raster_grad_fixed_scale_range": SimpleNamespace(value=256.0),
        "cached_raster_grad_fixed_color_range": SimpleNamespace(value=8.0),
        "cached_raster_grad_fixed_opacity_range": SimpleNamespace(value=8.0),
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
        "debug_contribution_min": SimpleNamespace(value=0.0),
        "debug_contribution_max": SimpleNamespace(value=1.0),
        "debug_refinement_distribution_min": SimpleNamespace(value=0.0),
        "debug_refinement_distribution_max": SimpleNamespace(value=1.0),
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

    for mode in ("adam_momentum", "adam_second_moment"):
        controls["debug_mode"].value = app._DEBUG_MODE_VALUES.index(mode)
        params = app.SplatViewer.renderer_params(viewer, allow_debug_overlays=True)

        assert params.debug_mode == mode
        assert params.debug_gaussian_scale_multiplier == 1.5
        assert params.debug_min_opacity == 0.05
        assert params.debug_opacity_multiplier == 2.0
        assert params.debug_ellipse_scale_multiplier == 0.75
        assert np.isclose(params.debug_adam_momentum_range[0], 2e-7, rtol=0.0, atol=1e-15)
        assert np.isclose(params.debug_adam_momentum_range[1], 2e-3, rtol=0.0, atol=1e-15)

    controls["debug_mode"].value = app._DEBUG_MODE_VALUES.index("ppisp_tonemap")
    params = app.SplatViewer.renderer_params(viewer, allow_debug_overlays=True)
    assert params.debug_mode is None


def test_initial_renderer_params_use_repo_defaults_for_cached_grad_depth() -> None:
    base = app.RendererParams()

    params = app._initial_renderer_params(
        SimpleNamespace(list_capacity_multiplier=17, max_prepass_memory_mb=3072),
    )

    assert params.list_capacity_multiplier == 17
    assert params.max_prepass_memory_mb == 3072
    assert params.cached_raster_grad.include_depth is base.cached_raster_grad.include_depth


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

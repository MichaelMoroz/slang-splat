from __future__ import annotations

from types import SimpleNamespace

from src.viewer import app
from src.viewer.app import SplatViewer


def _viewer(keyboard_capture: bool = False, mouse_capture: bool = False) -> SimpleNamespace:
    viewer = SimpleNamespace()
    viewer.toolkit = SimpleNamespace(
        handle_keyboard_event=lambda event: keyboard_capture,
        handle_mouse_event=lambda event: mouse_capture,
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

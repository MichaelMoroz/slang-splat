from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.viewer import frame_capture


def test_capture_python_frame_writes_log_and_profile(tmp_path: Path) -> None:
    calls: list[str] = []

    text_path, profile_path = frame_capture.capture_python_frame(
        lambda: calls.append("frame"),
        frame_index=7,
        directory=tmp_path,
    )

    assert calls == ["frame"]
    assert text_path.parent == tmp_path
    assert profile_path.parent == tmp_path
    assert text_path.exists()
    assert profile_path.exists()
    assert text_path.suffix == ".txt"
    assert profile_path.suffix == ".prof"
    text = text_path.read_text(encoding="utf-8")
    assert "Slang Splat Python Frame Profile" in text
    assert "Frame Index: 7" in text
    assert "function calls" in text
    assert profile_path.stat().st_size > 0


def test_capture_renderdoc_frame_wraps_action_when_available(monkeypatch) -> None:
    calls: list[object] = []

    monkeypatch.setattr(frame_capture, "ensure_qrenderdoc_running", lambda target_control=None: Path("C:/Program Files/RenderDoc/qrenderdoc.exe"))
    monkeypatch.setattr(frame_capture, "_find_process_listener_port", lambda _pid: None)
    monkeypatch.setattr(
        frame_capture,
        "renderdoc",
        SimpleNamespace(
            is_available=lambda: True,
            is_frame_capturing=lambda: False,
            start_frame_capture=lambda device, window=None: calls.append(("start", device, window)) or True,
            end_frame_capture=lambda: calls.append("end") or True,
        ),
    )

    resolved = frame_capture.capture_renderdoc_frame(
        lambda: calls.append("frame"),
        device="device",
        window="window",
    )

    assert resolved == Path("C:/Program Files/RenderDoc/qrenderdoc.exe")
    assert calls == [("start", "device", "window"), "frame", "end"]


def test_capture_renderdoc_frame_runs_frame_and_reports_unavailable(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(frame_capture, "find_renderdoccmd", lambda: Path("C:/Program Files/RenderDoc/renderdoccmd.exe"))
    monkeypatch.setattr(frame_capture, "_inject_renderdoc", lambda _path, _pid: 38920)
    monkeypatch.setattr(frame_capture, "_wait_for_runtime_renderdoc_api", lambda _timeout: None)
    monkeypatch.setattr(
        frame_capture,
        "renderdoc",
        SimpleNamespace(
            is_available=lambda: False,
            is_frame_capturing=lambda: False,
            start_frame_capture=lambda *_args, **_kwargs: True,
            end_frame_capture=lambda: True,
        ),
    )

    with pytest.raises(RuntimeError, match="runtime API never became available"):
        frame_capture.capture_renderdoc_frame(lambda: calls.append("frame"), device="device", window="window")

    assert calls == ["frame"]


def test_capture_renderdoc_frame_runs_frame_and_reports_missing_qrenderdoc(monkeypatch) -> None:
    calls: list[str] = []

    def _missing_qrenderdoc(_target_control=None) -> Path:
        raise RuntimeError("RenderDoc was not found. Install RenderDoc or add qrenderdoc to PATH.")

    monkeypatch.setattr(frame_capture, "ensure_qrenderdoc_running", _missing_qrenderdoc)
    monkeypatch.setattr(frame_capture, "_find_process_listener_port", lambda _pid: None)
    monkeypatch.setattr(
        frame_capture,
        "renderdoc",
        SimpleNamespace(
            is_available=lambda: True,
            is_frame_capturing=lambda: False,
            start_frame_capture=lambda *_args, **_kwargs: True,
            end_frame_capture=lambda: True,
        ),
    )

    with pytest.raises(RuntimeError, match="RenderDoc was not found"):
        frame_capture.capture_renderdoc_frame(lambda: calls.append("frame"), device="device", window="window")

    assert calls == ["frame"]


def test_inject_renderdoc_accepts_target_id_exit_code(monkeypatch) -> None:
    monkeypatch.setattr(frame_capture, "_RENDERDOC_TARGET_PORT", None)
    monkeypatch.setattr(
        frame_capture.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["renderdoccmd", "inject", "--PID=43004"],
            returncode=38921,
            stdout="Injecting into PID 43004\nLaunched as ID 38921\n",
            stderr="",
        ),
    )

    assert frame_capture._inject_renderdoc(Path("C:/Program Files/RenderDoc/renderdoccmd.exe"), 43004) == 38921


def test_prepare_renderdoc_startup_injects_and_records_target_port(monkeypatch) -> None:
    calls: list[object] = []

    class _RuntimeAPI:
        def disable_overlay_and_capture_keys(self) -> None:
            calls.append("disable")

        def trigger_capture(self) -> None:
            calls.append("trigger")

        def is_target_control_connected(self) -> bool:
            return True

    monkeypatch.setattr(frame_capture, "_RENDERDOC_TARGET_PORT", None)
    monkeypatch.setattr(frame_capture, "find_renderdoccmd", lambda: Path("C:/Program Files/RenderDoc/renderdoccmd.exe"))
    monkeypatch.setattr(frame_capture, "_inject_renderdoc", lambda _path, _pid: calls.append(("inject", _pid)) or 38920)
    monkeypatch.setattr(frame_capture, "_get_runtime_renderdoc_api", lambda: None)
    monkeypatch.setattr(frame_capture, "_wait_for_runtime_renderdoc_api", lambda _timeout: _RuntimeAPI())
    monkeypatch.setattr(
        frame_capture,
        "renderdoc",
        SimpleNamespace(
            is_available=lambda: (_ for _ in ()).throw(AssertionError("prepare_renderdoc_startup should not call renderdoc.is_available before injection")),
            is_frame_capturing=lambda: False,
            start_frame_capture=lambda *_args, **_kwargs: False,
            end_frame_capture=lambda: False,
        ),
    )

    resolved = frame_capture.prepare_renderdoc_startup()

    assert resolved == 38920
    assert frame_capture._RENDERDOC_TARGET_PORT == 38920
    assert calls == [("inject", frame_capture.getpid()), "disable"]


def test_capture_renderdoc_frame_prefers_recorded_target_port(monkeypatch) -> None:
    calls: list[object] = []

    monkeypatch.setattr(frame_capture, "_RENDERDOC_TARGET_PORT", 38920)
    monkeypatch.setattr(frame_capture, "_find_process_listener_port", lambda _pid: 45000)

    class _RuntimeAPI:
        def disable_overlay_and_capture_keys(self) -> None:
            calls.append("disable")

        def get_num_captures(self) -> int:
            return 0

        def get_capture_path(self, _index: int) -> Path | None:
            return None

        def trigger_capture(self) -> None:
            calls.append("trigger")

        def is_target_control_connected(self) -> bool:
            return True

    monkeypatch.setattr(frame_capture, "_get_runtime_renderdoc_api", lambda: _RuntimeAPI())
    monkeypatch.setattr(frame_capture, "ensure_qrenderdoc_running", lambda target_control=None: calls.append(("ui", target_control)) or Path("C:/Program Files/RenderDoc/qrenderdoc.exe"))
    monkeypatch.setattr(
        frame_capture,
        "renderdoc",
        SimpleNamespace(
            is_available=lambda: True,
            is_frame_capturing=lambda: False,
            start_frame_capture=lambda device, window=None: calls.append(("start", device, window)) or True,
            end_frame_capture=lambda: calls.append("end") or True,
        ),
    )

    resolved = frame_capture.capture_renderdoc_frame(lambda: calls.append("frame"), device="device", window="window")

    assert resolved == Path("C:/Program Files/RenderDoc/qrenderdoc.exe")
    assert calls == [("start", "device", "window"), "frame", "end", ("ui", "localhost:38920")]


def test_capture_renderdoc_frame_opens_latest_capture_file(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []
    capture_path = tmp_path / "capture.rdc"
    capture_path.write_text("rdc", encoding="utf-8")

    class _RuntimeAPI:
        def disable_overlay_and_capture_keys(self) -> None:
            calls.append("disable")

        def get_num_captures(self) -> int:
            return 1 if "end" in calls else 0

        def get_capture_path(self, _index: int) -> Path | None:
            return capture_path

        def trigger_capture(self) -> None:
            calls.append("trigger")

        def is_target_control_connected(self) -> bool:
            return True

    monkeypatch.setattr(frame_capture, "_RENDERDOC_TARGET_PORT", 38920)
    monkeypatch.setattr(frame_capture, "_get_runtime_renderdoc_api", lambda: _RuntimeAPI())
    monkeypatch.setattr(frame_capture, "_open_qrenderdoc_capture", lambda path, capture, target_control=None: calls.append(("open", path, capture, target_control)) or path)
    monkeypatch.setattr(frame_capture, "find_qrenderdoc", lambda: Path("C:/Program Files/RenderDoc/qrenderdoc.exe"))
    monkeypatch.setattr(
        frame_capture,
        "renderdoc",
        SimpleNamespace(
            is_available=lambda: True,
            is_frame_capturing=lambda: False,
            start_frame_capture=lambda device, window=None: calls.append(("start", device, window)) or True,
            end_frame_capture=lambda: calls.append("end") or True,
        ),
    )

    resolved = frame_capture.capture_renderdoc_frame(lambda: calls.append("frame"), device="device", window="window")

    assert resolved == Path("C:/Program Files/RenderDoc/qrenderdoc.exe")
    assert calls == [("start", "device", "window"), "frame", "end", ("open", Path("C:/Program Files/RenderDoc/qrenderdoc.exe"), capture_path, "localhost:38920")]


def test_capture_renderdoc_frame_uses_runtime_api_after_injection(monkeypatch) -> None:
    calls: list[str] = []

    class _RuntimeAPI:
        def __init__(self) -> None:
            self.triggered = 0

        def trigger_capture(self) -> None:
            self.triggered += 1
            calls.append("trigger")

        def is_target_control_connected(self) -> bool:
            return True

    runtime_api = _RuntimeAPI()
    monkeypatch.setattr(frame_capture, "find_renderdoccmd", lambda: Path("C:/Program Files/RenderDoc/renderdoccmd.exe"))
    monkeypatch.setattr(frame_capture, "_inject_renderdoc", lambda _path, _pid: 38920)
    monkeypatch.setattr(frame_capture, "_get_runtime_renderdoc_api", lambda: None)
    monkeypatch.setattr(frame_capture, "_wait_for_runtime_renderdoc_api", lambda _timeout: runtime_api)
    monkeypatch.setattr(frame_capture, "ensure_qrenderdoc_running", lambda target_control=None: calls.append(f"ui:{target_control}") or Path("C:/Program Files/RenderDoc/qrenderdoc.exe"))
    monkeypatch.setattr(
        frame_capture,
        "renderdoc",
        SimpleNamespace(
            is_available=lambda: False,
            is_frame_capturing=lambda: False,
            start_frame_capture=lambda *_args, **_kwargs: False,
            end_frame_capture=lambda: False,
        ),
    )

    resolved = frame_capture.capture_renderdoc_frame(lambda: calls.append("frame"), device="device", window="window")

    assert resolved == Path("C:/Program Files/RenderDoc/qrenderdoc.exe")
    assert runtime_api.triggered == 1
    assert calls == ["trigger", "frame", "ui:localhost:38920"]
from __future__ import annotations

import cProfile
import ctypes
import io
import pstats
import re
import shutil
import subprocess
import time
from contextlib import suppress
from datetime import datetime
from os import environ, getpid
from pathlib import Path
from typing import Callable

import slangpy as spy

with suppress(Exception):
    from slangpy import renderdoc

if "renderdoc" not in globals():
    renderdoc = None


_RENDERDOC_ATTACH_TIMEOUT_SECONDS = 5.0
_RENDERDOC_ATTACH_POLL_SECONDS = 0.05
_RENDERDOC_API_VERSION = 10101
_RENDERDOC_CALLBACK = ctypes.WINFUNCTYPE if hasattr(ctypes, "WINFUNCTYPE") else ctypes.CFUNCTYPE
_RENDERDOC_TARGET_PORT: int | None = None


class _RenderDocApiStruct(ctypes.Structure):
    _fields_ = [
        ("GetAPIVersion", ctypes.c_void_p),
        ("SetCaptureOptionU32", ctypes.c_void_p),
        ("SetCaptureOptionF32", ctypes.c_void_p),
        ("GetCaptureOptionU32", ctypes.c_void_p),
        ("GetCaptureOptionF32", ctypes.c_void_p),
        ("SetFocusToggleKeys", ctypes.c_void_p),
        ("SetCaptureKeys", ctypes.c_void_p),
        ("GetOverlayBits", ctypes.c_void_p),
        ("MaskOverlayBits", ctypes.c_void_p),
        ("RemoveHooks", ctypes.c_void_p),
        ("GetNumCaptures", ctypes.c_void_p),
        ("GetCapture", ctypes.c_void_p),
        ("TriggerCapture", ctypes.c_void_p),
        ("IsTargetControlConnected", ctypes.c_void_p),
        ("LaunchReplayUI", ctypes.c_void_p),
        ("SetActiveWindow", ctypes.c_void_p),
        ("StartFrameCapture", ctypes.c_void_p),
        ("IsFrameCapturing", ctypes.c_void_p),
        ("EndFrameCapture", ctypes.c_void_p),
    ]


class _RuntimeRenderDocAPI:
    def __init__(self, api_ptr: ctypes.c_void_p) -> None:
        api = ctypes.cast(api_ptr, ctypes.POINTER(_RenderDocApiStruct)).contents
        self._set_focus_toggle_keys = _RENDERDOC_CALLBACK(None, ctypes.c_void_p, ctypes.c_int)(api.SetFocusToggleKeys) if api.SetFocusToggleKeys else None
        self._set_capture_keys = _RENDERDOC_CALLBACK(None, ctypes.c_void_p, ctypes.c_int)(api.SetCaptureKeys) if api.SetCaptureKeys else None
        self._mask_overlay_bits = _RENDERDOC_CALLBACK(None, ctypes.c_uint32, ctypes.c_uint32)(api.MaskOverlayBits) if api.MaskOverlayBits else None
        self._get_num_captures = _RENDERDOC_CALLBACK(ctypes.c_uint32)(api.GetNumCaptures) if api.GetNumCaptures else None
        self._get_capture = _RENDERDOC_CALLBACK(ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64))(api.GetCapture) if api.GetCapture else None
        self._trigger_capture = _RENDERDOC_CALLBACK(None)(api.TriggerCapture) if api.TriggerCapture else None
        self._is_target_control_connected = _RENDERDOC_CALLBACK(ctypes.c_uint32)(api.IsTargetControlConnected) if api.IsTargetControlConnected else None

    def disable_overlay_and_capture_keys(self) -> None:
        if self._set_focus_toggle_keys is not None:
            self._set_focus_toggle_keys(None, 0)
        if self._set_capture_keys is not None:
            self._set_capture_keys(None, 0)
        if self._mask_overlay_bits is not None:
            self._mask_overlay_bits(0, 0)

    def get_num_captures(self) -> int:
        if self._get_num_captures is None:
            return 0
        return int(self._get_num_captures())

    def get_capture_path(self, index: int) -> Path | None:
        if self._get_capture is None:
            return None
        path_length = ctypes.c_uint32(0)
        timestamp = ctypes.c_uint64(0)
        if int(self._get_capture(int(index), None, ctypes.byref(path_length), ctypes.byref(timestamp))) == 0:
            return None
        if path_length.value == 0:
            return None
        buffer = ctypes.create_string_buffer(int(path_length.value))
        if int(self._get_capture(int(index), ctypes.cast(buffer, ctypes.c_void_p), ctypes.byref(path_length), ctypes.byref(timestamp))) == 0:
            return None
        raw_path = buffer.value.decode("utf-8", errors="ignore").strip()
        return Path(raw_path) if raw_path else None

    def trigger_capture(self) -> None:
        if self._trigger_capture is None:
            raise RuntimeError("RenderDoc runtime API does not expose TriggerCapture.")
        self._trigger_capture()

    def is_target_control_connected(self) -> bool:
        return self._is_target_control_connected is not None and bool(self._is_target_control_connected())


def _get_runtime_renderdoc_api() -> _RuntimeRenderDocAPI | None:
    windll = getattr(ctypes, "WinDLL", None)
    if windll is None:
        return None
    try:
        kernel32 = windll("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        module_handle = kernel32.GetModuleHandleW("renderdoc.dll")
        if not module_handle:
            return None
        renderdoc_module = windll("renderdoc.dll")
        get_api_symbol = getattr(renderdoc_module, "RENDERDOC_GetAPI", None)
        if get_api_symbol is None:
            return None
        get_api = _RENDERDOC_CALLBACK(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(("RENDERDOC_GetAPI", renderdoc_module))
        api_ptr = ctypes.c_void_p()
        if int(get_api(int(_RENDERDOC_API_VERSION), ctypes.byref(api_ptr))) != 1 or not api_ptr.value:
            return None
    except Exception:
        return None
    return _RuntimeRenderDocAPI(api_ptr)


def _remember_renderdoc_target_port(target_port: int | None) -> int | None:
    global _RENDERDOC_TARGET_PORT
    if target_port is not None:
        _RENDERDOC_TARGET_PORT = int(target_port)
    return _RENDERDOC_TARGET_PORT


def _wait_for_runtime_renderdoc_api(timeout_seconds: float) -> _RuntimeRenderDocAPI | None:
    deadline = time.monotonic() + float(timeout_seconds)
    runtime_api = _get_runtime_renderdoc_api()
    while runtime_api is None and time.monotonic() < deadline:
        time.sleep(_RENDERDOC_ATTACH_POLL_SECONDS)
        runtime_api = _get_runtime_renderdoc_api()
    return runtime_api


def _wait_for_target_control_connection(runtime_api: _RuntimeRenderDocAPI, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if runtime_api.is_target_control_connected():
            return True
        time.sleep(_RENDERDOC_ATTACH_POLL_SECONDS)
    return runtime_api.is_target_control_connected()


def _resolve_renderdoc_target_port(pid: int) -> int | None:
    if _RENDERDOC_TARGET_PORT is not None:
        return int(_RENDERDOC_TARGET_PORT)
    return _find_process_listener_port(pid)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_capture_stem(frame_index: int | None = None, now: datetime | None = None) -> str:
    timestamp = (datetime.now() if now is None else now).strftime("%Y%m%d_%H%M%S")
    frame_suffix = "" if frame_index is None else f"_frame{int(frame_index):06d}"
    return f"python_frame_capture_{timestamp}{frame_suffix}"


def _active_api_name(device: spy.Device) -> str:
    info = getattr(device, "info", None)
    api_name = getattr(info, "api_name", "")
    return str(api_name).strip().lower()


def capture_python_frame(
    action: Callable[[], object],
    *,
    frame_index: int | None = None,
    directory: Path | str | None = None,
) -> tuple[Path, Path]:
    output_dir = _repo_root() / "temp" if directory is None else Path(directory)
    stem = _python_capture_stem(frame_index)
    profile_path = output_dir / f"{stem}.prof"
    text_path = output_dir / f"{stem}.txt"
    profiler = cProfile.Profile()
    error: Exception | None = None

    profiler.enable()
    try:
        action()
    except Exception as exc:
        error = exc
    finally:
        profiler.disable()

    output_dir.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(profile_path))

    stream = io.StringIO()
    stream.write("Slang Splat Python Frame Profile\n")
    if frame_index is not None:
        stream.write(f"Frame Index: {int(frame_index)}\n")
    stream.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats(pstats.SortKey.CUMULATIVE, pstats.SortKey.TIME)
    stats.print_stats()
    text_path.write_text(stream.getvalue(), encoding="utf-8")

    if error is not None:
        raise error.with_traceback(error.__traceback__)
    return text_path, profile_path


def find_qrenderdoc() -> Path | None:
    candidates: list[Path] = []
    for executable in ("qrenderdoc.exe", "qrenderdoc"):
        resolved = shutil.which(executable)
        if resolved:
            candidates.append(Path(resolved))
    for env_key in ("ProgramFiles", "ProgramFiles(x86)"):
        root = environ.get(env_key)
        if root:
            candidates.append(Path(root) / "RenderDoc" / "qrenderdoc.exe")
    with suppress(Exception):
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\qrenderdoc.exe") as key:
            value, _ = winreg.QueryValueEx(key, None)
            if value:
                candidates.append(Path(value))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_renderdoccmd() -> Path | None:
    candidates: list[Path] = []
    for executable in ("renderdoccmd.exe", "renderdoccmd"):
        resolved = shutil.which(executable)
        if resolved:
            candidates.append(Path(resolved))
    for env_key in ("ProgramFiles", "ProgramFiles(x86)"):
        root = environ.get(env_key)
        if root:
            candidates.append(Path(root) / "RenderDoc" / "renderdoccmd.exe")
    with suppress(Exception):
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\renderdoccmd.exe") as key:
            value, _ = winreg.QueryValueEx(key, None)
            if value:
                candidates.append(Path(value))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _is_qrenderdoc_running() -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq qrenderdoc.exe"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0 and "qrenderdoc.exe" in result.stdout.lower()


def _launch_qrenderdoc(qrenderdoc_path: Path, target_control: str | None = None) -> None:
    launch_args = [str(qrenderdoc_path)]
    if target_control:
        launch_args.extend(["--targetcontrol", target_control])
    try:
        subprocess.Popen(launch_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        raise RuntimeError(f"Failed to launch RenderDoc: {exc}") from exc


def ensure_qrenderdoc_running(target_control: str | None = None) -> Path:
    qrenderdoc_path = find_qrenderdoc()
    if qrenderdoc_path is None:
        raise RuntimeError("RenderDoc was not found. Install RenderDoc or add qrenderdoc to PATH.")
    if target_control is not None or not _is_qrenderdoc_running():
        _launch_qrenderdoc(qrenderdoc_path, target_control=target_control)
    return qrenderdoc_path


def _open_qrenderdoc_capture(qrenderdoc_path: Path, capture_path: Path, *, target_control: str | None = None) -> Path:
    launch_args = [str(qrenderdoc_path)]
    if target_control:
        launch_args.extend(["--targetcontrol", target_control])
    launch_args.append(str(capture_path))
    try:
        subprocess.Popen(launch_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        raise RuntimeError(f"Failed to open RenderDoc capture: {exc}") from exc
    return qrenderdoc_path


def _wait_for_new_capture(runtime_api: _RuntimeRenderDocAPI, previous_count: int, timeout_seconds: float) -> Path | None:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        capture_count = runtime_api.get_num_captures()
        if capture_count > int(previous_count):
            return runtime_api.get_capture_path(capture_count - 1)
        time.sleep(_RENDERDOC_ATTACH_POLL_SECONDS)
    capture_count = runtime_api.get_num_captures()
    if capture_count > int(previous_count):
        return runtime_api.get_capture_path(capture_count - 1)
    return None


def _parse_renderdoc_target_port(output: str) -> int | None:
    match = re.search(r"Launched as ID\s+(\d+)", output)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _find_process_listener_port(pid: int) -> int | None:
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    ports: list[int] = []
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*TCP\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$", line)
        if match is None:
            continue
        try:
            local_port = int(match.group(1))
            owning_pid = int(match.group(2))
        except ValueError:
            continue
        if owning_pid == int(pid):
            ports.append(local_port)
    if not ports:
        return None
    return max(ports)


def _inject_renderdoc(renderdoccmd_path: Path, pid: int) -> int | None:
    try:
        result = subprocess.run(
            [str(renderdoccmd_path), "inject", f"--PID={int(pid)}"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to inject RenderDoc into PID {int(pid)}: {exc}") from exc
    output = f"{result.stdout}\n{result.stderr}".strip()
    target_port = _parse_renderdoc_target_port(output)
    if target_port is not None:
        return _remember_renderdoc_target_port(target_port)
    if result.returncode != 0:
        detail = output if output else f"exit code {int(result.returncode)}"
        raise RuntimeError(f"RenderDoc injection failed for PID {int(pid)}: {detail}")
    return None


def prepare_renderdoc_startup() -> int | None:
    runtime_api = _get_runtime_renderdoc_api()
    if runtime_api is not None:
        runtime_api.disable_overlay_and_capture_keys()
        return _remember_renderdoc_target_port(_resolve_renderdoc_target_port(getpid()))
    renderdoccmd_path = find_renderdoccmd()
    if renderdoccmd_path is None:
        return None
    target_port = _inject_renderdoc(renderdoccmd_path, getpid())
    runtime_api = _wait_for_runtime_renderdoc_api(_RENDERDOC_ATTACH_TIMEOUT_SECONDS)
    if runtime_api is None:
        raise RuntimeError("RenderDoc startup injection succeeded, but the RenderDoc runtime API never became available in this process.")
    runtime_api.disable_overlay_and_capture_keys()
    return _remember_renderdoc_target_port(target_port)


def _wait_for_renderdoc_attach(timeout_seconds: float) -> bool:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if renderdoc is not None and bool(renderdoc.is_available()):
            return True
        time.sleep(_RENDERDOC_ATTACH_POLL_SECONDS)
    return renderdoc is not None and bool(renderdoc.is_available())


def capture_renderdoc_frame(
    action: Callable[[], object],
    *,
    device: spy.Device,
    window: spy.Window | None = None,
) -> Path:
    qrenderdoc_path: Path | None = None
    setup_error: RuntimeError | None = None
    action_error: Exception | None = None
    runtime_api = _get_runtime_renderdoc_api()
    use_runtime_trigger = False
    capture_count_before = runtime_api.get_num_captures() if runtime_api is not None else 0

    try:
        target_port: int | None = None
        slangpy_renderdoc_ready = renderdoc is not None and bool(renderdoc.is_available())
        if not slangpy_renderdoc_ready:
            if runtime_api is None:
                renderdoccmd_path = find_renderdoccmd()
                if renderdoccmd_path is None:
                    raise RuntimeError("RenderDoc CLI was not found. Install RenderDoc or add renderdoccmd to PATH.")
                target_port = _inject_renderdoc(renderdoccmd_path, getpid())
                runtime_api = _wait_for_runtime_renderdoc_api(_RENDERDOC_ATTACH_TIMEOUT_SECONDS)
                if runtime_api is None:
                    raise RuntimeError("RenderDoc injection succeeded, but the RenderDoc runtime API never became available in this process.")
            else:
                target_port = _resolve_renderdoc_target_port(getpid())
        if target_port is None:
            target_port = _resolve_renderdoc_target_port(getpid())
        if slangpy_renderdoc_ready:
            if bool(getattr(renderdoc, "is_frame_capturing", lambda: False)()):
                raise RuntimeError("RenderDoc is already capturing a frame.")
            if not bool(renderdoc.start_frame_capture(device, window)):
                raise RuntimeError("Failed to start the RenderDoc frame capture.")
        else:
            if runtime_api is None:
                raise RuntimeError("RenderDoc runtime API is not available after launch/injection.")
            _wait_for_target_control_connection(runtime_api, _RENDERDOC_ATTACH_TIMEOUT_SECONDS)
            runtime_api.trigger_capture()
            use_runtime_trigger = True
    except RuntimeError as exc:
        setup_error = exc

    if setup_error is not None:
        action()
        raise setup_error

    try:
        action()
    except Exception as exc:
        action_error = exc

    if not use_runtime_trigger:
        end_ok = bool(renderdoc.end_frame_capture())
        if not end_ok:
            raise RuntimeError("Failed to end the RenderDoc frame capture.")
        capture_path = _wait_for_new_capture(runtime_api, capture_count_before, _RENDERDOC_ATTACH_TIMEOUT_SECONDS) if runtime_api is not None else None
        resolved_target_control = None if target_port is None else f"localhost:{int(target_port)}"
        if capture_path is not None:
            qrenderdoc_path = _open_qrenderdoc_capture(find_qrenderdoc() or Path("qrenderdoc.exe"), capture_path, target_control=resolved_target_control)
        else:
            qrenderdoc_path = ensure_qrenderdoc_running(resolved_target_control)
    else:
        qrenderdoc_path = ensure_qrenderdoc_running(None if target_port is None else f"localhost:{int(target_port)}")
    if action_error is not None:
        raise action_error.with_traceback(action_error.__traceback__)
    return qrenderdoc_path if qrenderdoc_path is not None else Path("qrenderdoc.exe")
from __future__ import annotations

import cProfile
import io
import pstats
import shutil
import subprocess
from contextlib import suppress
from datetime import datetime
from os import environ
from pathlib import Path
from typing import Callable

import slangpy as spy

with suppress(Exception):
    from slangpy import renderdoc

if "renderdoc" not in globals():
    renderdoc = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_capture_stem(frame_index: int | None = None, now: datetime | None = None) -> str:
    timestamp = (datetime.now() if now is None else now).strftime("%Y%m%d_%H%M%S")
    frame_suffix = "" if frame_index is None else f"_frame{int(frame_index):06d}"
    return f"python_frame_capture_{timestamp}{frame_suffix}"


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


def ensure_qrenderdoc_running() -> Path:
    qrenderdoc_path = find_qrenderdoc()
    if qrenderdoc_path is None:
        raise RuntimeError("RenderDoc was not found. Install RenderDoc or add qrenderdoc to PATH.")
    if not _is_qrenderdoc_running():
        try:
            subprocess.Popen([str(qrenderdoc_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            raise RuntimeError(f"Failed to launch RenderDoc: {exc}") from exc
    return qrenderdoc_path


def capture_renderdoc_frame(
    action: Callable[[], object],
    *,
    device: spy.Device,
    window: spy.Window | None = None,
) -> Path:
    qrenderdoc_path: Path | None = None
    setup_error: RuntimeError | None = None
    action_error: Exception | None = None

    try:
        qrenderdoc_path = ensure_qrenderdoc_running()
        if renderdoc is None:
            raise RuntimeError("SlangPy RenderDoc integration is unavailable.")
        if not bool(renderdoc.is_available()):
            raise RuntimeError("RenderDoc is open, but SlangPy cannot control it yet. Make sure the viewer is attached to the opened RenderDoc instance.")
        if bool(getattr(renderdoc, "is_frame_capturing", lambda: False)()):
            raise RuntimeError("RenderDoc is already capturing a frame.")
        if not bool(renderdoc.start_frame_capture(device, window)):
            raise RuntimeError("Failed to start the RenderDoc frame capture.")
    except RuntimeError as exc:
        setup_error = exc

    if setup_error is not None:
        action()
        raise setup_error

    try:
        action()
    except Exception as exc:
        action_error = exc

    end_ok = bool(renderdoc.end_frame_capture())
    if not end_ok:
        raise RuntimeError("Failed to end the RenderDoc frame capture.")
    if action_error is not None:
        raise action_error.with_traceback(action_error.__traceback__)
    return qrenderdoc_path if qrenderdoc_path is not None else Path("qrenderdoc.exe")
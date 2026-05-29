from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PROJECT_VENV_DIR_NAME = ".venv"
_REQUIREMENTS_FILE_NAME = "requirements.txt"
_BOOTSTRAP_REEXEC_ENV = "SLANGPY_BOOTSTRAP_PROJECT_PYTHON"
_SLANGPY_VERSION = "0.42.0"
_SLANGPY_PIP_SPEC = f"slangpy=={_SLANGPY_VERSION}"
_SLANGPY_MIN_PYTHON = (3, 9)
_SLANGPY_MAX_PYTHON = (3, 13)
_REEXEC_ENV_DROP_NAMES = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONEXECUTABLE",
    "UV_INTERNAL__PYTHONHOME",
    "VIRTUAL_ENV",
    "__PYVENV_LAUNCHER__",
)
_PACKAGE_IMPORT_NAMES = {
    "imgui-bundle": "imgui_bundle",
    "pillow": "PIL",
}


def _repo_root(repo_root: Path | None = None) -> Path:
    return _ROOT if repo_root is None else Path(repo_root).resolve()


def _requirements_path(repo_root: Path) -> Path:
    return repo_root / _REQUIREMENTS_FILE_NAME


def _project_python_candidates(repo_root: Path) -> tuple[Path, ...]:
    venv_root = repo_root / _PROJECT_VENV_DIR_NAME
    return (venv_root / "Scripts" / "python.exe", venv_root / "bin" / "python")


def find_project_python(repo_root: Path | None = None) -> Path | None:
    resolved_root = _repo_root(repo_root)
    for candidate in _project_python_candidates(resolved_root):
        if candidate.is_file():
            return candidate
    return None


def _resolved_path(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _same_path(first: Path, second: Path) -> bool:
    return _resolved_path(first) == _resolved_path(second)


def _project_dependencies_missing(repo_root: Path) -> bool:
    return bool(_missing_requirements(repo_root)) or not _slangpy_requirement_satisfied(repo_root)


def _running_inside_virtual_environment() -> bool:
    if getattr(sys, "real_prefix", None):
        return True
    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix))
    prefix = Path(sys.prefix)
    if _same_path(prefix, base_prefix):
        return False
    return True


def _current_python_externally_managed() -> bool:
    if _running_inside_virtual_environment():
        return False
    stdlib_path = sysconfig.get_path("stdlib")
    if not stdlib_path:
        return False
    return (Path(stdlib_path) / "EXTERNALLY-MANAGED").is_file()


def _project_python_command(project_python: Path) -> list[str]:
    original_argv = list(getattr(sys, "orig_argv", ()))
    if original_argv:
        return [str(project_python), *original_argv[1:]]
    return [str(project_python), *sys.argv]


def _project_venv_root(project_python: Path) -> Path:
    parent_name = project_python.parent.name.lower()
    if parent_name in {"scripts", "bin"}:
        return project_python.parent.parent
    return project_python.parent


def _project_python_environment(project_python: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in _REEXEC_ENV_DROP_NAMES:
        env.pop(name, None)
    env[_BOOTSTRAP_REEXEC_ENV] = str(project_python)
    env["VIRTUAL_ENV"] = str(_project_venv_root(project_python))
    return env


def _current_interpreter_matches(path: Path) -> bool:
    return _same_path(_resolved_path(path), Path(sys.executable))


def _python_supports_slangpy(major: int, minor: int) -> bool:
    return _SLANGPY_MIN_PYTHON <= (major, minor) <= _SLANGPY_MAX_PYTHON


def _current_python_supports_slangpy() -> bool:
    return _python_supports_slangpy(sys.version_info.major, sys.version_info.minor)


def _interpreter_version(python_executable: Path) -> tuple[int, int] | None:
    if _current_interpreter_matches(python_executable):
        return (sys.version_info.major, sys.version_info.minor)
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.match(r"^(\d+)\.(\d+)$", completed.stdout.strip())
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)))


def _project_python_supports_slangpy(project_python: Path) -> bool:
    version = _interpreter_version(project_python)
    return version is not None and _python_supports_slangpy(*version)


def _supported_venv_python() -> Path | None:
    if _current_python_supports_slangpy():
        return Path(sys.executable)
    if os.name != "nt":
        return None
    for minor in range(_SLANGPY_MAX_PYTHON[1], _SLANGPY_MIN_PYTHON[1] - 1, -1):
        try:
            completed = subprocess.run(
                ["py", f"-3.{minor}", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            continue
        candidate = Path(completed.stdout.strip())
        if candidate.is_file() and _project_python_supports_slangpy(candidate):
            return candidate
    return None


def _create_project_venv(repo_root: Path, *, clear: bool = False) -> Path:
    venv_python = _supported_venv_python()
    if venv_python is None:
        supported = f"{_SLANGPY_MIN_PYTHON[0]}.{_SLANGPY_MIN_PYTHON[1]}-{_SLANGPY_MAX_PYTHON[0]}.{_SLANGPY_MAX_PYTHON[1]}"
        raise RuntimeError(
            f"Slangpy {_SLANGPY_VERSION} requires Python {supported}, but no compatible interpreter was found to create {repo_root / _PROJECT_VENV_DIR_NAME}."
        )
    command = [str(venv_python), "-m", "venv"]
    if clear:
        command.append("--clear")
    command.append(_PROJECT_VENV_DIR_NAME)
    try:
        subprocess.run(command, cwd=repo_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to create project virtual environment at {repo_root / _PROJECT_VENV_DIR_NAME}.") from exc
    project_python = find_project_python(repo_root)
    if project_python is None:
        raise RuntimeError(f"Project virtual environment was created but no Python executable was found under {repo_root / _PROJECT_VENV_DIR_NAME}.")
    return project_python


def _ensure_project_python(repo_root: Path) -> Path | None:
    project_python = find_project_python(repo_root)
    if _running_inside_virtual_environment():
        if project_python is None or not _current_interpreter_matches(project_python):
            return None
        return project_python if _project_python_supports_slangpy(project_python) else None
    project_python_compatible = project_python is not None and _project_python_supports_slangpy(project_python)
    if project_python_compatible:
        return project_python
    if project_python is not None:
        return _create_project_venv(repo_root, clear=True)
    if not _current_python_supports_slangpy() or _current_python_externally_managed():
        return _create_project_venv(repo_root)
    return None


def _maybe_reexec_into_project_python(repo_root: Path) -> None:
    if not _project_dependencies_missing(repo_root):
        return
    project_python = _ensure_project_python(repo_root)
    if project_python is None:
        return
    resolved_project_python = _resolved_path(project_python)
    if _current_interpreter_matches(resolved_project_python):
        return
    if os.environ.get(_BOOTSTRAP_REEXEC_ENV) == str(resolved_project_python):
        return
    env = _project_python_environment(resolved_project_python)
    raise SystemExit(subprocess.run(_project_python_command(resolved_project_python), env=env, check=False).returncode)


def _normalized_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _package_name_from_requirement(line: str) -> str | None:
    stripped = line.split("#", 1)[0].strip()
    if not stripped or stripped.startswith(("-r", "--", "-e", ".", "/", "\\")):
        return None
    match = re.match(r"([A-Za-z0-9_.-]+)", stripped)
    if match is None:
        return None
    return _normalized_package_name(match.group(1))


def _requirements_packages(repo_root: Path) -> tuple[str, ...]:
    packages: list[str] = []
    seen: set[str] = set()
    for line in _requirements_path(repo_root).read_text(encoding="utf-8").splitlines():
        package = _package_name_from_requirement(line)
        if package is None or package in seen:
            continue
        packages.append(package)
        seen.add(package)
    return tuple(packages)


def _package_import_name(package: str) -> str:
    normalized = _normalized_package_name(package)
    return _PACKAGE_IMPORT_NAMES.get(normalized, normalized.replace("-", "_"))


def _package_available(package: str) -> bool:
    return importlib.util.find_spec(_package_import_name(package)) is not None


def _missing_requirements(repo_root: Path) -> tuple[str, ...]:
    return tuple(package for package in _requirements_packages(repo_root) if not _package_available(package))


def _slangpy_available() -> bool:
    return importlib.util.find_spec("slangpy") is not None


def _installed_package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _slangpy_requirement_satisfied(repo_root: Path) -> bool:
    del repo_root
    return _slangpy_available() and _installed_package_version("slangpy") == _SLANGPY_VERSION


def _run_pip_install(args: list[str], repo_root: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *args],
        cwd=repo_root,
        check=True,
    )


def _install_requirements(repo_root: Path) -> None:
    try:
        _run_pip_install(["-r", str(_requirements_path(repo_root))], repo_root)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to install Python requirements from {_requirements_path(repo_root)}.") from exc


def _install_from_pip(repo_root: Path) -> None:
    if not _current_python_supports_slangpy():
        supported = f"{_SLANGPY_MIN_PYTHON[0]}.{_SLANGPY_MIN_PYTHON[1]}-{_SLANGPY_MAX_PYTHON[0]}.{_SLANGPY_MAX_PYTHON[1]}"
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(f"Slangpy {_SLANGPY_VERSION} requires Python {supported}, but the active interpreter is Python {current}.")
    try:
        _run_pip_install(["--upgrade", _SLANGPY_PIP_SPEC], repo_root)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to install {_SLANGPY_PIP_SPEC} from pip.") from exc


def ensure_requirements_available(repo_root: Path | None = None) -> None:
    resolved_root = _repo_root(repo_root)
    missing = _missing_requirements(resolved_root)
    if not missing:
        return
    _install_requirements(resolved_root)
    importlib.invalidate_caches()
    remaining = _missing_requirements(resolved_root)
    if remaining:
        joined = ", ".join(remaining)
        raise RuntimeError(f"Required Python packages are still unavailable after installing requirements.txt: {joined}")


def ensure_slangpy_available(repo_root: Path | None = None) -> None:
    resolved_root = _repo_root(repo_root)
    if _slangpy_requirement_satisfied(resolved_root):
        return
    _install_from_pip(resolved_root)
    installed_version = _installed_package_version("slangpy")
    importlib.invalidate_caches()
    if installed_version != _SLANGPY_VERSION:
        raise RuntimeError(f"slangpy version {installed_version!r} is installed, but repo requires {_SLANGPY_VERSION!r} from pip.")
    if not _slangpy_available():
        raise RuntimeError("slangpy is still unavailable after installation from pip.")


def ensure_project_dependencies_available(repo_root: Path | None = None) -> None:
    resolved_root = _repo_root(repo_root)
    _maybe_reexec_into_project_python(resolved_root)
    ensure_requirements_available(resolved_root)
    ensure_slangpy_available(resolved_root)


__all__ = [
    "ensure_project_dependencies_available",
    "ensure_requirements_available",
    "ensure_slangpy_available",
    "find_project_python",
]
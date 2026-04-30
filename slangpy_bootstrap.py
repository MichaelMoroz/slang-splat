from __future__ import annotations

import importlib
import importlib.util
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_REQUIREMENTS_FILE_NAME = "requirements.txt"
_WHEEL_DIR_NAME = "slangpy_wheels"
_PACKAGE_IMPORT_NAMES = {
    "imgui-bundle": "imgui_bundle",
    "pillow": "PIL",
}


def _repo_root(repo_root: Path | None = None) -> Path:
    return _ROOT if repo_root is None else Path(repo_root).resolve()


def _requirements_path(repo_root: Path) -> Path:
    return repo_root / _REQUIREMENTS_FILE_NAME


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


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}" if sys.implementation.name == "cpython" else ""


def _platform_tag() -> str:
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def _wheel_tags(wheel_path: Path) -> tuple[str, str, str] | None:
    if wheel_path.suffix != ".whl":
        return None
    try:
        _, python_tag, abi_tag, platform_tag = wheel_path.name[:-4].rsplit("-", 3)
    except ValueError:
        return None
    return python_tag, abi_tag, platform_tag


def find_local_slangpy_wheel(repo_root: Path | None = None) -> Path | None:
    wheel_dir = _repo_root(repo_root) / _WHEEL_DIR_NAME
    if not wheel_dir.is_dir():
        return None
    python_tag = _python_tag()
    if not python_tag:
        return None
    platform_tag = _platform_tag()
    matches = []
    for wheel_path in wheel_dir.glob("slangpy-*.whl"):
        tags = _wheel_tags(wheel_path)
        if tags != (python_tag, python_tag, platform_tag):
            continue
        matches.append(wheel_path)
    return sorted(matches)[-1] if matches else None


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


def _install_local_wheel(wheel_path: Path, repo_root: Path) -> None:
    try:
        _run_pip_install([str(wheel_path)], repo_root)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to install slangpy from local wheel: {wheel_path}") from exc


def _install_from_pip(repo_root: Path) -> None:
    try:
        _run_pip_install(["slangpy"], repo_root)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to install slangpy from pip.") from exc


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
    if _slangpy_available():
        return
    resolved_root = _repo_root(repo_root)
    wheel_path = find_local_slangpy_wheel(resolved_root)
    if wheel_path is not None:
        _install_local_wheel(wheel_path, resolved_root)
    else:
        _install_from_pip(resolved_root)
    importlib.invalidate_caches()
    if not _slangpy_available():
        source = str(wheel_path) if wheel_path is not None else "pip"
        raise RuntimeError(f"slangpy is still unavailable after installation from {source}.")


def ensure_project_dependencies_available(repo_root: Path | None = None) -> None:
    resolved_root = _repo_root(repo_root)
    ensure_requirements_available(resolved_root)
    ensure_slangpy_available(resolved_root)


__all__ = ["ensure_project_dependencies_available", "ensure_requirements_available", "ensure_slangpy_available", "find_local_slangpy_wheel"]
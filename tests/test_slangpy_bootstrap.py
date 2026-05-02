from __future__ import annotations

import builtins
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import slangpy_bootstrap as bootstrap


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _clear_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def test_find_local_slangpy_wheel_prefers_matching_interpreter_and_platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel_dir = tmp_path / "slangpy_wheels"
    wheel_dir.mkdir()
    expected = wheel_dir / "slangpy-0.41.0-cp313-cp313-win_amd64.whl"
    expected.touch()
    (wheel_dir / "slangpy-0.41.0-cp312-cp312-win_amd64.whl").touch()
    (wheel_dir / "slangpy-0.41.0-cp313-cp313-manylinux_2_28_x86_64.whl").touch()

    monkeypatch.setattr(bootstrap, "_python_tag", lambda: "cp313")
    monkeypatch.setattr(bootstrap, "_platform_tag", lambda: "win_amd64")

    assert bootstrap.find_local_slangpy_wheel(tmp_path) == expected


def test_find_project_python_prefers_repo_venv(tmp_path: Path) -> None:
    project_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    project_python.parent.mkdir(parents=True)
    project_python.write_text("", encoding="utf-8")

    assert bootstrap.find_project_python(tmp_path) == project_python


def test_missing_requirements_uses_import_name_aliases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pillow>=10.4\nimgui-bundle>=1.92\nnumpy>=1.26\n", encoding="utf-8")
    missing_modules = {"PIL", "imgui_bundle"}

    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda name: None if name in missing_modules else object())

    assert bootstrap._missing_requirements(tmp_path) == ("pillow", "imgui-bundle")


def test_ensure_requirements_available_skips_install_when_declared_packages_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_missing_requirements", lambda repo_root: ())
    monkeypatch.setattr(bootstrap, "_install_requirements", lambda repo_root: (_ for _ in ()).throw(AssertionError("should not install requirements")))

    bootstrap.ensure_requirements_available()


def test_ensure_requirements_available_installs_requirements_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Path] = []
    states = iter((("numpy",), ()))

    monkeypatch.setattr(bootstrap, "_missing_requirements", lambda repo_root: next(states))
    monkeypatch.setattr(bootstrap, "_install_requirements", lambda repo_root: calls.append(Path(repo_root)))

    bootstrap.ensure_requirements_available(tmp_path)

    assert calls == [tmp_path.resolve()]


def test_project_dependencies_available_installs_requirements_before_slangpy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr(bootstrap, "_maybe_reexec_into_project_python", lambda repo_root: None)
    monkeypatch.setattr(bootstrap, "ensure_requirements_available", lambda repo_root=None: calls.append("requirements"))
    monkeypatch.setattr(bootstrap, "ensure_slangpy_available", lambda repo_root=None: calls.append("slangpy"))

    bootstrap.ensure_project_dependencies_available(tmp_path)

    assert calls == ["requirements", "slangpy"]


def test_project_dependencies_available_reexecs_into_repo_venv_when_current_interpreter_is_missing_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    project_python.parent.mkdir(parents=True)
    project_python.write_text("", encoding="utf-8")
    current_python = tmp_path / "external" / "python.exe"
    current_prefix = tmp_path / "external-prefix"
    argv = [str(tmp_path / "viewer.py"), "--dataset", "garden"]
    calls: list[tuple[str, list[str], dict[str, str]]] = []

    monkeypatch.setattr(sys, "executable", str(current_python))
    monkeypatch.setattr(sys, "prefix", str(current_prefix))
    monkeypatch.setattr(sys, "base_prefix", str(current_prefix), raising=False)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "orig_argv", [str(current_python), *argv], raising=False)
    monkeypatch.setattr(bootstrap, "_missing_requirements", lambda repo_root: ("numpy",))
    monkeypatch.setattr(bootstrap, "_slangpy_available", lambda: False)
    monkeypatch.delenv(bootstrap._BOOTSTRAP_REEXEC_ENV, raising=False)
    monkeypatch.setenv("PYTHONHOME", "uv-home")
    monkeypatch.setenv("UV_INTERNAL__PYTHONHOME", "uv-internal-home")
    monkeypatch.setenv("VIRTUAL_ENV", "wrong-venv")

    def _fake_run(arguments: list[str], env: dict[str, str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append((arguments[0], arguments, dict(env)))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", _fake_run)

    with pytest.raises(SystemExit) as exc:
        bootstrap.ensure_project_dependencies_available(tmp_path)

    assert exc.value.code == 0
    assert len(calls) == 1
    executable, arguments, env = calls[0]
    assert executable == str(project_python.resolve())
    assert arguments == [str(project_python.resolve()), *argv]
    assert env[bootstrap._BOOTSTRAP_REEXEC_ENV] == str(project_python.resolve())
    assert env["VIRTUAL_ENV"] == str(project_python.parent.parent)
    assert "PYTHONHOME" not in env
    assert "UV_INTERNAL__PYTHONHOME" not in env


def test_project_dependencies_available_creates_repo_venv_when_missing_for_externally_managed_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    current_python = tmp_path / "external" / "python.exe"
    current_prefix = tmp_path / "external-prefix"
    project_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    argv = [str(tmp_path / "viewer.py")]
    externally_managed_root = tmp_path / "managed"
    externally_managed_root.mkdir()
    (externally_managed_root / "EXTERNALLY-MANAGED").write_text("", encoding="utf-8")
    calls: list[tuple[str, list[str], Path | None, dict[str, str] | None, bool]] = []

    monkeypatch.setattr(sys, "executable", str(current_python))
    monkeypatch.setattr(sys, "prefix", str(current_prefix))
    monkeypatch.setattr(sys, "base_prefix", str(current_prefix), raising=False)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "orig_argv", [str(current_python), *argv], raising=False)
    monkeypatch.setattr(bootstrap, "_missing_requirements", lambda repo_root: ("numpy",))
    monkeypatch.setattr(bootstrap, "_slangpy_available", lambda: False)
    monkeypatch.setattr(bootstrap.sysconfig, "get_path", lambda name: str(externally_managed_root) if name == "stdlib" else None)
    monkeypatch.delenv(bootstrap._BOOTSTRAP_REEXEC_ENV, raising=False)
    monkeypatch.setenv("PYTHONHOME", "uv-home")

    def _fake_run(
        arguments: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if arguments == [str(current_python), "-m", "venv", bootstrap._PROJECT_VENV_DIR_NAME]:
            project_python.parent.mkdir(parents=True)
            project_python.write_text("", encoding="utf-8")
            calls.append(("venv", arguments, cwd, dict(env) if env is not None else None, check))
            return subprocess.CompletedProcess(arguments, 0)
        calls.append(("reexec", arguments, cwd, dict(env) if env is not None else None, check))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", _fake_run)

    with pytest.raises(SystemExit) as exc:
        bootstrap.ensure_project_dependencies_available(tmp_path)

    assert exc.value.code == 0
    assert calls[0] == ("venv", [str(current_python), "-m", "venv", bootstrap._PROJECT_VENV_DIR_NAME], tmp_path.resolve(), None, True)
    assert len(calls) == 2
    relaunch_kind, relaunch_args, relaunch_cwd, relaunch_env, relaunch_check = calls[1]
    assert relaunch_kind == "reexec"
    assert relaunch_args == [str(project_python.resolve()), *argv]
    assert relaunch_cwd is None
    assert relaunch_check is False
    assert relaunch_env is not None
    assert relaunch_env[bootstrap._BOOTSTRAP_REEXEC_ENV] == str(project_python.resolve())
    assert relaunch_env["VIRTUAL_ENV"] == str(project_python.parent.parent)
    assert "PYTHONHOME" not in relaunch_env


def test_ensure_project_python_does_not_create_repo_venv_when_running_inside_other_local_venv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    current_prefix = tmp_path / ".venv2"
    base_prefix = tmp_path / "base-python"
    externally_managed_root = tmp_path / "managed"
    externally_managed_root.mkdir()
    (externally_managed_root / "EXTERNALLY-MANAGED").write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "prefix", str(current_prefix))
    monkeypatch.setattr(sys, "base_prefix", str(base_prefix), raising=False)
    monkeypatch.setattr(bootstrap.sysconfig, "get_path", lambda name: str(externally_managed_root) if name == "stdlib" else None)
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not create repo venv")))

    assert bootstrap._ensure_project_python(tmp_path) is None


def test_project_dependencies_available_does_not_reexec_into_repo_venv_when_running_inside_other_local_venv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    current_prefix = tmp_path / ".venv2"
    current_python = current_prefix / "Scripts" / "python.exe"
    base_prefix = tmp_path / "base-python"
    project_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    calls: list[str] = []

    current_python.parent.mkdir(parents=True)
    current_python.write_text("", encoding="utf-8")
    project_python.parent.mkdir(parents=True)
    project_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "executable", str(current_python))
    monkeypatch.setattr(sys, "prefix", str(current_prefix))
    monkeypatch.setattr(sys, "base_prefix", str(base_prefix), raising=False)
    monkeypatch.setattr(bootstrap, "_missing_requirements", lambda repo_root: ("numpy",))
    monkeypatch.setattr(bootstrap, "_slangpy_available", lambda: False)
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not reexec into repo venv")))
    monkeypatch.setattr(bootstrap, "ensure_requirements_available", lambda repo_root=None: calls.append("requirements"))
    monkeypatch.setattr(bootstrap, "ensure_slangpy_available", lambda repo_root=None: calls.append("slangpy"))

    bootstrap.ensure_project_dependencies_available(tmp_path)

    assert calls == ["requirements", "slangpy"]


def test_ensure_slangpy_available_skips_install_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_slangpy_available", lambda: True)
    monkeypatch.setattr(bootstrap, "_install_local_wheel", lambda wheel_path, repo_root: (_ for _ in ()).throw(AssertionError("should not install local wheel")))
    monkeypatch.setattr(bootstrap, "_install_from_pip", lambda repo_root: (_ for _ in ()).throw(AssertionError("should not install from pip")))

    bootstrap.ensure_slangpy_available()


def test_ensure_slangpy_available_installs_matching_local_wheel_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []
    wheel_path = tmp_path / "slangpy_wheels" / "slangpy-0.41.0-cp313-cp313-win_amd64.whl"
    states = iter((False, True))

    monkeypatch.setattr(bootstrap, "_slangpy_available", lambda: next(states))
    monkeypatch.setattr(bootstrap, "find_local_slangpy_wheel", lambda repo_root=None: wheel_path)
    monkeypatch.setattr(bootstrap, "_install_local_wheel", lambda resolved_wheel, repo_root: calls.append(("wheel", resolved_wheel)))
    monkeypatch.setattr(bootstrap, "_install_from_pip", lambda repo_root: calls.append(("pip", Path(repo_root))))

    bootstrap.ensure_slangpy_available(tmp_path)

    assert calls == [("wheel", wheel_path)]


def test_ensure_slangpy_available_falls_back_to_pip_without_matching_wheel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []
    states = iter((False, True))

    monkeypatch.setattr(bootstrap, "_slangpy_available", lambda: next(states))
    monkeypatch.setattr(bootstrap, "find_local_slangpy_wheel", lambda repo_root=None: None)
    monkeypatch.setattr(bootstrap, "_install_local_wheel", lambda resolved_wheel, repo_root: calls.append(("wheel", resolved_wheel)))
    monkeypatch.setattr(bootstrap, "_install_from_pip", lambda repo_root: calls.append(("pip", Path(repo_root))))

    bootstrap.ensure_slangpy_available(tmp_path)

    assert calls == [("pip", tmp_path.resolve())]


def test_viewer_entrypoint_bootstraps_before_app_import(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sanitized_path = [str(tmp_path), *(entry for entry in sys.path if Path(entry or ".").resolve() != repo_root)]
    _write_file(tmp_path / "src" / "__init__.py", "")
    _write_file(tmp_path / "src" / "viewer" / "__init__.py", "")
    _write_file(
        tmp_path / "src" / "viewer" / "app.py",
        "import builtins\n"
        "builtins._entrypoint_calls.append('import-app')\n"
        "class SplatViewer:\n"
        "    pass\n\n"
        "def main():\n"
        "    return 0\n",
    )
    bootstrap_module = ModuleType("slangpy_bootstrap")
    bootstrap_module.ensure_project_dependencies_available = lambda: builtins._entrypoint_calls.append("ensure")
    _clear_modules("slangpy_bootstrap", "src", "src.viewer", "src.viewer.app")
    monkeypatch.setitem(sys.modules, "slangpy_bootstrap", bootstrap_module)
    monkeypatch.setattr(sys, "path", sanitized_path)
    monkeypatch.setattr(builtins, "_entrypoint_calls", [], raising=False)

    runpy.run_path(str(repo_root / "viewer.py"))

    assert builtins._entrypoint_calls == ["ensure", "import-app"]
    _clear_modules("slangpy_bootstrap", "src", "src.viewer", "src.viewer.app")


def test_cli_entrypoint_bootstraps_before_cli_import(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sanitized_path = [str(tmp_path), *(entry for entry in sys.path if Path(entry or ".").resolve() != repo_root)]
    _write_file(tmp_path / "src" / "__init__.py", "")
    _write_file(tmp_path / "src" / "app" / "__init__.py", "")
    _write_file(
        tmp_path / "src" / "app" / "cli.py",
        "import builtins\n"
        "builtins._entrypoint_calls.append('import-cli')\n"
        "def parse_args():\n"
        "    return None\n\n"
        "def main():\n"
        "    return 0\n",
    )
    bootstrap_module = ModuleType("slangpy_bootstrap")
    bootstrap_module.ensure_project_dependencies_available = lambda: builtins._entrypoint_calls.append("ensure")
    _clear_modules("slangpy_bootstrap", "src", "src.app", "src.app.cli")
    monkeypatch.setitem(sys.modules, "slangpy_bootstrap", bootstrap_module)
    monkeypatch.setattr(sys, "path", sanitized_path)
    monkeypatch.setattr(builtins, "_entrypoint_calls", [], raising=False)

    runpy.run_path(str(repo_root / "cli.py"))

    assert builtins._entrypoint_calls == ["ensure", "import-cli"]
    _clear_modules("slangpy_bootstrap", "src", "src.app", "src.app.cli")


def test_render_entrypoint_bootstraps_before_cli_import(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sanitized_path = [str(tmp_path), *(entry for entry in sys.path if Path(entry or ".").resolve() != repo_root)]
    _write_file(tmp_path / "src" / "__init__.py", "")
    _write_file(tmp_path / "src" / "app" / "__init__.py", "")
    _write_file(
        tmp_path / "src" / "app" / "cli.py",
        "import builtins\n"
        "builtins._entrypoint_calls.append('import-render-cli')\n"
        "def parse_single_render_args():\n"
        "    return None\n\n"
        "def render_main():\n"
        "    return 0\n",
    )
    bootstrap_module = ModuleType("slangpy_bootstrap")
    bootstrap_module.ensure_project_dependencies_available = lambda: builtins._entrypoint_calls.append("ensure")
    _clear_modules("slangpy_bootstrap", "src", "src.app", "src.app.cli")
    monkeypatch.setitem(sys.modules, "slangpy_bootstrap", bootstrap_module)
    monkeypatch.setattr(sys, "path", sanitized_path)
    monkeypatch.setattr(builtins, "_entrypoint_calls", [], raising=False)

    runpy.run_path(str(repo_root / "render.py"))

    assert builtins._entrypoint_calls == ["ensure", "import-render-cli"]
    _clear_modules("slangpy_bootstrap", "src", "src.app", "src.app.cli")
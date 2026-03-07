from __future__ import annotations

from pathlib import Path

from tools.complexity_budget import ROOT, iter_python_files


def test_complexity_budget_scopes_to_entrypoints_and_src():
    files = {path.relative_to(ROOT).as_posix() for path in iter_python_files(ROOT)}
    assert {"cli.py", "render.py", "viewer.py"} <= files
    assert "tools/complexity_budget.py" not in files
    assert all(not path.startswith("tests/") for path in files)
    assert all(path == "cli.py" or path == "render.py" or path == "viewer.py" or path.startswith("src/") for path in files)

from __future__ import annotations

from slangpy_bootstrap import ensure_project_dependencies_available

ensure_project_dependencies_available()

from src.app.cli import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())

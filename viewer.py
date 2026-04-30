from __future__ import annotations

from slangpy_bootstrap import ensure_project_dependencies_available

ensure_project_dependencies_available()

from src.viewer.app import SplatViewer, main

__all__ = ["SplatViewer", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

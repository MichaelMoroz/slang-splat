from __future__ import annotations

from slangpy_bootstrap import ensure_project_dependencies_available

ensure_project_dependencies_available()

from src.app.cli import parse_single_render_args, render_main

__all__ = ["render_main", "parse_single_render_args"]


def main() -> int:
    return int(render_main())


if __name__ == "__main__":
    raise SystemExit(main())

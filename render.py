from __future__ import annotations

from src.app.cli import parse_single_render_args, render_main

__all__ = ["render_main", "parse_single_render_args"]


def main() -> int:
    return int(render_main())


if __name__ == "__main__":
    raise SystemExit(main())

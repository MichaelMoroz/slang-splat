from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import lizard

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINTS = ("cli.py", "render.py", "viewer.py")
EXCLUDED_PARTS = {".venv", "tests", "__pycache__"}
TARGET_NLOC = 2263
TARGET_TOTAL_CCN = 340
TARGET_MAX_CCN = 15


@dataclass(frozen=True, slots=True)
class FileMetrics:
    path: str
    nloc: int
    total_ccn: int
    max_ccn: int
    function_count: int


@dataclass(frozen=True, slots=True)
class BudgetMetrics:
    files: tuple[FileMetrics, ...]
    total_nloc: int
    total_ccn: int
    max_ccn: int
    function_count: int


def iter_python_files(root: Path = ROOT) -> list[Path]:
    src_files = [path for path in (root / "src").rglob("*.py") if not (set(path.relative_to(root).parts) & EXCLUDED_PARTS)]
    return sorted([*(root / name for name in ENTRYPOINTS if (root / name).exists()), *src_files])


def analyze(root: Path = ROOT) -> BudgetMetrics:
    file_metrics: list[FileMetrics] = []
    total_nloc = total_ccn = function_count = max_ccn = 0
    for path in iter_python_files(root):
        analysis = lizard.analyze_file(str(path))
        function_metrics = tuple(analysis.function_list)
        file_total_ccn = sum(int(fn.cyclomatic_complexity) for fn in function_metrics)
        file_max_ccn = max((int(fn.cyclomatic_complexity) for fn in function_metrics), default=0)
        file_metrics.append(
            FileMetrics(
                path=str(path.relative_to(root)),
                nloc=int(analysis.nloc),
                total_ccn=file_total_ccn,
                max_ccn=file_max_ccn,
                function_count=len(function_metrics),
            )
        )
        total_nloc += int(analysis.nloc)
        total_ccn += file_total_ccn
        function_count += len(function_metrics)
        max_ccn = max(max_ccn, file_max_ccn)
    return BudgetMetrics(
        files=tuple(sorted(file_metrics, key=lambda item: (item.nloc, item.total_ccn, item.path), reverse=True)),
        total_nloc=total_nloc,
        total_ccn=total_ccn,
        max_ccn=max_ccn,
        function_count=function_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure production Python complexity budget with lizard.")
    parser.add_argument("--check", action="store_true", help="Fail when the project exceeds the complexity budget.")
    parser.add_argument("--top", type=int, default=10, help="Number of largest files to print.")
    return parser


def _print(metrics: BudgetMetrics, top: int) -> None:
    print(
        f"files={len(metrics.files)} functions={metrics.function_count} "
        f"nloc={metrics.total_nloc} total_ccn={metrics.total_ccn} max_ccn={metrics.max_ccn}"
    )
    for item in metrics.files[: max(int(top), 0)]:
        print(
            f"{item.nloc:4d} nloc | {item.total_ccn:4d} total_ccn | {item.max_ccn:2d} max_ccn | "
            f"{item.function_count:3d} funcs | {item.path}"
        )


def _violations(metrics: BudgetMetrics) -> list[str]:
    failures: list[str] = []
    if metrics.total_nloc > TARGET_NLOC:
        failures.append(f"NLOC {metrics.total_nloc} > {TARGET_NLOC}")
    if metrics.total_ccn > TARGET_TOTAL_CCN:
        failures.append(f"total CCN {metrics.total_ccn} > {TARGET_TOTAL_CCN}")
    if metrics.max_ccn > TARGET_MAX_CCN:
        failures.append(f"max CCN {metrics.max_ccn} > {TARGET_MAX_CCN}")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = analyze()
    _print(metrics, int(args.top))
    failures = _violations(metrics)
    if args.check and failures:
        for failure in failures:
            print(f"budget-fail: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

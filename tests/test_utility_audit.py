from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _ROOT / "src"
_FORBIDDEN_PATTERNS = (
    ("legacy common import", "from .common"),
    ("legacy common import", "from ..common"),
    ("legacy common import", "from src.common"),
    ("manual shader program load", "load_program("),
    ("manual compute kernel creation", "create_compute_kernel("),
    ("manual compute pipeline creation", "create_compute_pipeline("),
    ("manual buffer allocation", "create_buffer("),
    ("manual texture allocation", "create_texture("),
)


def test_src_common_module_removed() -> None:
    assert not (_SRC_ROOT / "common.py").exists()


def test_non_utility_src_modules_use_shared_runtime_helpers() -> None:
    offenders: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(_SRC_ROOT)
        if relative.parts[0] == "utility":
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in _FORBIDDEN_PATTERNS:
            if pattern in text:
                offenders.append(f"{relative}: {label} -> {pattern}")
    assert not offenders, "Found runtime-helper regressions:\n" + "\n".join(offenders)

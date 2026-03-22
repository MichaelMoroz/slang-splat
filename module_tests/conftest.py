from __future__ import annotations

import pytest


_ALL_BACKENDS = ("cuda", "vulkan", "d3d12")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--backend",
        action="append",
        default=[],
        help="Module test backend(s): cuda, vulkan, d3d12. Can be passed multiple times.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "backend_name" not in metafunc.fixturenames:
        return
    selected = metafunc.config.getoption("backend") or list(_ALL_BACKENDS)
    invalid = [name for name in selected if name not in _ALL_BACKENDS]
    if invalid:
        raise pytest.UsageError(f"Unsupported backend(s): {', '.join(invalid)}")
    metafunc.parametrize("backend_name", selected)

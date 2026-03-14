from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def device():
    import slangpy as spy

    from src import create_default_device

    try:
        return create_default_device(device_type=spy.DeviceType.vulkan, enable_debug_layers=True)
    except Exception as exc:
        pytest.skip(f"GPU device unavailable for Slangpy tests (vulkan): {exc}")

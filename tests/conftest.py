from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def device():
    import slangpy as spy

    from src import create_default_device, device_type_from_name

    backend = os.getenv("SLANG_SPLAT_DEVICE", "d3d12")

    try:
        return create_default_device(device_type=device_type_from_name(backend), enable_debug_layers=False)
    except Exception as exc:
        pytest.skip(f"GPU device unavailable for Slangpy tests ({backend}): {exc}")

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_train_cli_smoke():
    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = repo_root / "dataset" / "garden"
    if not dataset_root.exists():
        pytest.skip("Dataset folder is missing for CLI smoke test.")
    cmd = [
        sys.executable,
        "train.py",
        "--mode",
        "cli",
        "--colmap-root",
        str(dataset_root),
        "--images-subdir",
        "images_8",
        "--iters",
        "2",
        "--max-gaussians",
        "512",
        "--width",
        "64",
        "--height",
        "64",
        "--log-interval",
        "1",
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"
    assert "step=" in result.stdout

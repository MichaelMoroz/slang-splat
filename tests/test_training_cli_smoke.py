from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.app import cli
from src.app.training_controls import training_cli_build_kwargs
from src.scene import GaussianInitHyperParams


def test_train_cli_smoke():
    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = repo_root / "dataset" / "garden"
    if not dataset_root.exists():
        pytest.skip("Dataset folder is missing for CLI smoke test.")
    cmd = [
        sys.executable,
        "cli.py",
        "train-colmap",
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
    assert "mse=" in result.stdout
    assert "psnr=" in result.stdout


def test_train_cli_forwards_resolved_init_hparams(monkeypatch, tmp_path: Path):
    resolved_init = GaussianInitHyperParams(
        position_jitter_std=0.0,
        base_scale=1.25,
        scale_jitter_ratio=0.15,
        initial_opacity=0.42,
        color_jitter_std=0.0,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_colmap_reconstruction", lambda root, sparse_subdir="sparse/0": object())
    monkeypatch.setattr(cli, "build_training_frames", lambda recon, images_subdir="images_4": [SimpleNamespace(width=64, height=32)])
    monkeypatch.setattr(cli, "_training_params", lambda args: object())
    monkeypatch.setattr(
        cli,
        "apply_training_profile",
        lambda params, profile_name, dataset_root=None, images_subdir=None: (
            SimpleNamespace(training=SimpleNamespace(max_gaussians=17), adam=object(), stability=object()),
            SimpleNamespace(init_opacity_override=None, name="test"),
        ),
    )
    monkeypatch.setattr(cli, "resolve_colmap_init_hparams", lambda recon, max_gaussians, init_hparams=None: resolved_init)
    monkeypatch.setattr(
        cli,
        "initialize_scene_from_colmap_points",
        lambda recon, max_gaussians, seed, init_hparams=None: (
            captured.setdefault("init_hparams", init_hparams),
            SimpleNamespace(count=max_gaussians),
        )[1],
    )
    monkeypatch.setattr(cli, "_renderer", lambda args, width, height: SimpleNamespace(device=object()))
    monkeypatch.setattr(
        cli,
        "GaussianTrainer",
        lambda **kwargs: SimpleNamespace(state=SimpleNamespace(avg_loss=0.0, last_mse=0.0, last_psnr=float("inf"), last_instability="", last_frame_index=0)),
    )

    args = Namespace(
        colmap_root=tmp_path,
        sparse_subdir="sparse/0",
        images_subdir="images_4",
        width=0,
        height=0,
        seed=123,
        iters=0,
        log_interval=1,
        snapshot_interval=0,
        training_profile="auto",
        init_position_jitter=None,
        init_base_scale=None,
        init_scale_jitter=None,
        init_opacity=None,
        init_color_jitter=None,
        radius_scale=1.0,
        alpha_cutoff=1.0 / 255.0,
        max_splat_steps=32768,
        trans_threshold=0.005,
        prepass_memory_mb=256,
        list_capacity_multiplier=64,
        debug_layers=False,
        bg=(0.0, 0.0, 0.0),
        lr_base=0.005,
        lr_mul_pos=1.0,
        lr_mul_scale=20.0,
        lr_mul_rot=1.0,
        lr_mul_color=1.0,
        lr_mul_opacity=1.0,
        beta1=0.9,
        beta2=0.999,
        grad_clip=10.0,
        grad_norm_clip=10.0,
        max_update=0.05,
        max_scale=3.0,
        max_anisotropy=32.0,
        min_opacity=1e-4,
        max_opacity=0.9999,
        position_abs_max=1e4,
        near=0.1,
        far=120.0,
        scale_l2=0.0,
        scale_abs_reg=0.01,
        opacity_reg=0.01,
        sh1_reg=0.01,
        density_reg=0.02,
        depth_ratio_weight=1.0,
        refinement_loss_weight=0.25,
        refinement_target_edge_weight=0.75,
        depth_ratio_grad_min=0.0,
        depth_ratio_grad_max=0.1,
        max_allowed_density_start=5.0,
        max_allowed_density=12.0,
        max_gaussians=17,
        use_sh=True,
        refinement_min_contribution_percent=1e-05,
        snapshot_dir=tmp_path / "snapshots",
    )

    assert cli.run_train_colmap(args) == 0
    assert captured["init_hparams"] is resolved_init


def test_train_cli_parser_defaults_color_and_opacity_lr_mul_to_five() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["train-colmap", "--colmap-root", "dummy"])

    assert args.lr_mul_color == 5.0
    assert args.lr_mul_opacity == 5.0
    assert args.sh1_reg == 0.01
    assert args.depth_ratio_weight == 0.5
    assert args.lr_mul_scale == 15.0
    assert args.refinement_loss_weight == 0.25
    assert args.refinement_target_edge_weight == 0.75
    assert args.sorting_order_dithering == 0.5
    assert args.sorting_order_dithering_stage1 == 0.2
    assert args.sorting_order_dithering_stage2 == 0.05
    assert args.sorting_order_dithering_stage3 == 0.01
    assert training_cli_build_kwargs(args)["sorting_order_dithering"] == 0.5
    assert training_cli_build_kwargs(args)["sorting_order_dithering_stage1"] == 0.2
    assert training_cli_build_kwargs(args)["sorting_order_dithering_stage2"] == 0.05
    assert training_cli_build_kwargs(args)["sorting_order_dithering_stage3"] == 0.01
    assert args.depth_ratio_grad_min == 0.0
    assert args.depth_ratio_grad_max == 0.1
    assert args.refinement_min_contribution_percent == 1e-05
    assert args.init_opacity is None


def test_train_cli_parser_maps_sorting_order_dithering() -> None:
    parser = cli.build_parser()

    default_args = parser.parse_args(["train-colmap", "--colmap-root", "dummy"])
    explicit_args = parser.parse_args(
        [
            "train-colmap",
            "--colmap-root",
            "dummy",
            "--sorting-order-dithering",
            "0.375",
            "--sorting-order-dithering-stage1",
            "0.25",
            "--sorting-order-dithering-stage2",
            "0.125",
            "--sorting-order-dithering-stage3",
            "0.0625",
        ]
    )

    assert default_args.sorting_order_dithering == 0.5
    assert training_cli_build_kwargs(default_args)["sorting_order_dithering"] == 0.5
    assert training_cli_build_kwargs(default_args)["sorting_order_dithering_stage1"] == 0.2
    assert training_cli_build_kwargs(default_args)["sorting_order_dithering_stage2"] == 0.05
    assert training_cli_build_kwargs(default_args)["sorting_order_dithering_stage3"] == 0.01
    assert explicit_args.sorting_order_dithering == 0.375
    assert explicit_args.sorting_order_dithering_stage1 == 0.25
    assert explicit_args.sorting_order_dithering_stage2 == 0.125
    assert explicit_args.sorting_order_dithering_stage3 == 0.0625
    assert training_cli_build_kwargs(explicit_args)["sorting_order_dithering"] == 0.375
    assert training_cli_build_kwargs(explicit_args)["sorting_order_dithering_stage1"] == 0.25
    assert training_cli_build_kwargs(explicit_args)["sorting_order_dithering_stage2"] == 0.125
    assert training_cli_build_kwargs(explicit_args)["sorting_order_dithering_stage3"] == 0.0625

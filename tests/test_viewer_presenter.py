from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import slangpy as spy

from src.training.ppisp import PPISP_FIELD_SPECS
from src.viewer import presenter
from src.viewer import ui as viewer_ui
from src.viewer import session as viewer_session
from src.viewer.state import ColmapImportProgress
from tests.viewer_test_harness import (
    _buffer_snapshot,
    _CaptureKernel,
    _DummyEncoder,
    _DummyRenderer,
    _patch_vram_queries,
    _patch_render_frame,
    _render_context,
    _resource_debug_snapshot,
    _section_dict,
    _training_camera_colmap_viewer,
    _viewer,
)


def test_refresh_menu_bar_device_vram_skips_live_heap_query(monkeypatch) -> None:
    calls: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        presenter,
        "query_total_device_vram_used_cached",
        lambda device, *, allow_heap_query=True: calls.append((device, bool(allow_heap_query))) or (123, "cached"),
    )
    monkeypatch.setattr(presenter, "query_total_device_vram_capacity", lambda device: (456, "capacity"))
    viewer = SimpleNamespace(
        device="device",
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(last_time=0.0),
    )

    presenter._refresh_menu_bar_device_vram(viewer)

    assert calls == [("device", False)]
    assert viewer.ui._values["_menu_bar_device_vram_bytes"] == 123
    assert viewer.ui._values["_menu_bar_device_vram_total_bytes"] == 456


def test_training_camera_colmap_points_payload_clips_and_lists_other_views() -> None:
    render_limit = presenter._TRAINING_CAMERA_COLMAP_POINT_LIMIT
    viewer = _training_camera_colmap_viewer(
        observations=(
            {
                "image_id": 3,
                "name": "frame.png",
                "width": 64,
                "height": 32,
                "points2d_xy": np.column_stack((np.arange(render_limit + 2, dtype=np.float32) % 64.0, np.full((render_limit + 2,), 12.0, dtype=np.float32))),
                "point_ids": np.full((render_limit + 2,), 101, dtype=np.int64),
            },
            {"image_id": 5, "name": "other.png", "width": 32, "height": 18, "points2d_xy": ((8.0, 9.0),), "point_ids": (101,)},
        ),
        points=({"point_id": 101},),
    )

    payload = presenter._training_camera_colmap_points_payload(viewer)

    assert payload is not None
    assert payload["total_count"] == render_limit + 2
    assert payload["render_count"] == render_limit
    assert payload["point_ids"].shape == (render_limit,)
    assert payload["track_lengths"][0] == 2
    assert np.isclose(payload["errors"][0], 0.125)
    assert payload["other_views"] == ()
    assert payload["other_view_resolver"] is not None
    assert payload["other_view_resolver"](101) == ((1, 5, "other.png", 8.0, 9.0, 0.25, 0.5),)
    assert np.all(payload["uv"] >= 0.0)
    assert np.all(payload["uv"] <= 1.0)


def test_training_camera_colmap_points_payload_reuses_cached_frame_payload() -> None:
    viewer = _training_camera_colmap_viewer(
        observations=({"image_id": 3, "name": "frame.png", "width": 16, "height": 12, "points2d_xy": ((4.0, 6.0),), "point_ids": (101,)},),
        points=({"point_id": 101},),
    )

    payload_first = presenter._training_camera_colmap_points_payload(viewer)
    payload_second = presenter._training_camera_colmap_points_payload(viewer)

    assert payload_first is payload_second
    assert viewer.s.training_camera_colmap_payload is payload_first


def test_training_camera_colmap_points_payload_reuses_cached_payload_after_frame_switch() -> None:
    viewer = _training_camera_colmap_viewer(
        observations=(
            {"image_id": 3, "name": "frame.png", "width": 16, "height": 12, "points2d_xy": ((4.0, 6.0),), "point_ids": (101,)},
            {"image_id": 5, "name": "other.png", "width": 20, "height": 10, "points2d_xy": ((8.0, 9.0),), "point_ids": (202,)},
        ),
        points=(
            {"point_id": 101},
            {"point_id": 202, "xyz": (2.0, 3.0, 4.0), "rgb": (0.5, 1.0, 0.25), "error": 0.25, "track_length": 3},
        ),
    )

    payload_first = presenter._training_camera_colmap_points_payload(viewer)
    viewer.ui.controls["loss_debug_frame"].value = 1
    payload_second = presenter._training_camera_colmap_points_payload(viewer)
    viewer.ui.controls["loss_debug_frame"].value = 0
    payload_first_again = presenter._training_camera_colmap_points_payload(viewer)

    assert payload_first is not payload_second
    assert payload_first_again is payload_first


def test_update_ui_text_skips_training_camera_colmap_payload_when_overlay_inactive(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    viewer.ui._values["show_training_camera_colmap_points"] = True

    monkeypatch.setattr(
        presenter,
        "_training_camera_colmap_points_payload",
        lambda _viewer: (_ for _ in ()).throw(AssertionError("payload should stay inactive")),
    )

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.ui._values["_training_camera_colmap_points_payload"] is None


@pytest.mark.parametrize("case,loss_debug", (("debug", True), ("main", False), ("periodic", False)))
def test_render_frame_branch_paths(monkeypatch, case: str, loss_debug: bool) -> None:
    viewer = _viewer(loss_debug=loss_debug)
    render_context = _render_context()
    calls: list[str] = []
    patch_kwargs: dict[str, object] = {}
    if case == "debug":
        patch_kwargs["render_debug_view"] = lambda viewer_obj, encoder, width, height, render_frame_index: calls.append(f"debug:{render_frame_index}") or "debug_tex"
        expected_calls, expected_texture = ["apply", "debug:0", "ui"], "debug_tex"
    else:
        expected_time = viewer.s.last_time
        if case == "periodic":
            viewer.s.last_periodic_renderer_reallocation_time = 0.0
            expected_time = viewer_session._PERIODIC_RENDERER_REALLOCATION_INTERVAL_S + 1.0
            monkeypatch.setattr(presenter.time, "perf_counter", lambda: expected_time)
        patch_kwargs["maybe_reallocate_renderers"] = lambda viewer_obj, width, height, current_time: calls.append(f"periodic:{current_time:.1f}")
        expected_calls, expected_texture = ["apply", f"periodic:{expected_time:.1f}", "main", "ui"], "main_tex"
    _patch_render_frame(monkeypatch, calls, **patch_kwargs)
    presenter.render_frame(viewer, render_context)
    assert viewer.s.trainer.step_calls == 3
    assert viewer.s.viewport_texture == expected_texture
    assert viewer.s.render_frame_index == 1
    assert calls == expected_calls


@pytest.mark.parametrize("case", ("raster", "ppisp", "skip_ppisp", "shared_training"))
def test_render_main_view_cases(monkeypatch, case: str) -> None:
    viewer = _viewer(loss_debug=False)
    encoder = _DummyEncoder()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(presenter, "_dispatch_viewport_present", lambda viewer_obj, enc, source_tex, source_width, source_height, output_width, output_height, *, source_is_linear=False: calls.append(("present", source_tex, source_is_linear)) or "present_tex")
    if case != "shared_training":
        viewer.s.trainer = None; viewer.s.training_renderer = None
    if case == "ppisp":
        viewer.ui._values["debug_mode"] = viewer_ui._DEBUG_MODE_VALUES.index(viewer_ui.PPISP_DEBUG_MODE)
        viewer.ui._values["ppisp_exposure_ev"] = 1.25
    elif case == "skip_ppisp":
        viewer.s.renderer.debug_mode = "splat_age"
    assert presenter._render_main_view(viewer, encoder) == "present_tex"
    if case == "shared_training":
        assert len(viewer.s.training_renderer.render_calls) == 1
        assert viewer.s.training_renderer.render_calls[0]["camera"] == "camera"
        assert viewer.s.training_renderer.resolution_calls == [(640, 360)]
        assert viewer.s.renderer.render_calls == []
        return
    if case == "ppisp":
        params = viewer.s.renderer.render_ppisp_calls[0]["ppisp_tonemap"]
        assert viewer.s.renderer.render_calls == []
        assert viewer.s.renderer.render_linear_calls == []
        assert viewer.s.renderer.render_ppisp_calls[0]["camera"] == "camera"
        assert params["exposureEv"] == 1.25
        assert set(params) == {spec.attr for spec in PPISP_FIELD_SPECS}
        assert isinstance(params["chromaOffsetR"], type(spy.float2(0.0, 0.0)))
        assert isinstance(params["crfGamma"], type(spy.float3(0.0, 0.0, 0.0)))
        assert viewer.s.stats["generated_entries"] == 5
        assert calls == [("present", "main_ppisp_tex", False)]
    elif case == "skip_ppisp":
        assert len(viewer.s.renderer.render_calls) == 1
        assert viewer.s.renderer.render_linear_calls == []
        assert viewer.s.renderer.render_ppisp_calls == []
    else:
        assert viewer.s.renderer.render_calls[0]["camera"] == "camera"
        assert viewer.s.renderer.render_linear_calls == []
        assert viewer.s.renderer.render_ppisp_calls == []
        assert viewer.s.stats["generated_entries"] == 1
        assert calls == [("present", "main_render_tex", False)]


@pytest.mark.parametrize("case", ("configured", "reduced", "runtime_resize"))
def test_render_frame_training_batch_cases(monkeypatch, case: str) -> None:
    viewer = _viewer(loss_debug=False)
    render_context = _render_context()
    calls: list[str] = []
    patch_kwargs: dict[str, object] = {}
    if case == "reduced":
        viewer.s.last_interaction_time = viewer.s.last_time
    elif case == "runtime_resize":
        patch_kwargs["apply_live_params"] = lambda viewer_obj: setattr(viewer_obj.s, "pending_training_runtime_resize", True) or calls.append("apply")
        patch_kwargs["advance_colmap_import"] = lambda viewer_obj: calls.append("import")
        patch_kwargs["ensure_training_runtime_resolution"] = lambda viewer_obj: calls.append("train_resize") or True
    _patch_render_frame(monkeypatch, calls, **patch_kwargs)
    presenter.render_frame(viewer, render_context)
    expected_steps = 0 if case == "runtime_resize" else (1 if case == "reduced" else 3)
    assert viewer.s.trainer.step_calls == expected_steps
    assert viewer.s.trainer.step_batch_calls == ([] if case == "runtime_resize" else [expected_steps])
    assert viewer.s.last_training_batch_steps == expected_steps
    assert calls == (["apply", "import", "train_resize", "main", "ui"] if case == "runtime_resize" else ["apply", "main", "ui"])


@pytest.mark.parametrize(
    "case,ui_updates,expected_calls",
    (
        ("requested", {"_histograms_refresh_requested": True}, ["apply", "main", "hist", "ui"]),
        ("hidden", {"show_histograms": True}, ["apply", "main", "ui"]),
        ("realtime", {"show_histograms": True, "_histograms_update_realtime": True, "_histograms_realtime_next_refresh_time": 0.0}, ["apply", "main", "hist", "ui"]),
        ("realtime_wait", {"show_histograms": True, "_histograms_update_realtime": True, "_histograms_realtime_next_refresh_time": float("inf")}, ["apply", "main", "ui"]),
    ),
)
def test_render_frame_histogram_cases(monkeypatch, case: str, ui_updates: dict[str, object], expected_calls: list[str]) -> None:
    viewer = _viewer(loss_debug=False)
    render_context = _render_context()
    calls: list[str] = []
    viewer.ui._values.update(ui_updates)
    refresh = (lambda viewer_obj, force=False: calls.append("hist")) if case != "realtime_wait" else (lambda viewer_obj: calls.append("hist"))
    _patch_render_frame(monkeypatch, calls, refresh_cached_raster_grad_histograms=refresh)
    presenter.render_frame(viewer, render_context)
    assert calls == expected_calls


@pytest.mark.parametrize("case,error", (("resize", "resize boom"), ("live_params", "live params boom"), ("import", "import boom")))
def test_render_frame_failure_cases(monkeypatch, case: str, error: str) -> None:
    viewer = _viewer(loss_debug=False)
    render_context = _render_context()
    calls: list[str] = []
    patch_kwargs: dict[str, object] = {}
    if case == "resize":
        viewer.s.renderer.width = 320; viewer.s.renderer.height = 180
        patch_kwargs["recreate_renderer"] = lambda viewer_obj, width, height: (_ for _ in ()).throw(RuntimeError(error))
    elif case == "live_params":
        patch_kwargs["apply_live_params"] = lambda viewer_obj: (_ for _ in ()).throw(RuntimeError(error))
        patch_kwargs["advance_colmap_import"] = lambda viewer_obj: calls.append("import")
    else:
        patch_kwargs["advance_colmap_import"] = lambda viewer_obj: (_ for _ in ()).throw(RuntimeError(error))
    _patch_render_frame(monkeypatch, calls, **patch_kwargs)
    presenter.render_frame(viewer, render_context)
    assert viewer.s.training_active is False
    assert viewer.s.last_error == error
    assert viewer.s.last_render_exception == error
    assert render_context.command_encoder.clear_calls[-1] == (render_context.surface_texture, [0.0, 0.0, 0.0, 1.0])
    assert calls == (["apply", "ui"] if case == "resize" else (["ui"] if case == "live_params" else ["apply", "ui"]))


def test_update_ui_text_reports_training_schedule_and_refinement() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.ui._values["show_training_views"] = True

    presenter.update_ui_text(viewer, 1.0 / 60.0)
    schedule_sections = _section_dict(viewer.ui._values["_training_schedule_sections"])
    refinement = _section_dict(viewer.ui._values["_training_refinement_sections"])["Refinement"]

    assert viewer.ui._values["_training_resolution_sections"] == (("Train Res", (("size", "640x360"), ("factor", 1))),)
    assert viewer.ui._values["_training_downscale_sections"] == (("Downscale", (("mode", "Manual"), ("current", 1), ("subsample", "Off"), ("effective", 1))),)
    assert viewer.t("training_schedule").text == "LR Schedule: 2.00e-03@0 -> 2.00e-03@3,000 -> 1.00e-03@12,225 -> 7.00e-04@30,058 -> 4.00e-04@100,000 | current=5.00e-03"
    assert schedule_sections[""] == {"step": 0, "stage": "Stage 0", "sh": "SH0"}
    assert schedule_sections["Learning Rates"] == pytest.approx({"base": 0.002, "pos": 0.25, "scale": 5.0, "rot": 1.0, "dc": 5.0, "opacity": 5.0, "sh": 0.1})
    assert schedule_sections["Other"] == pytest.approx({"colorspace": 0.6, "dither": 0.01, "target%": 10.0, "prune_floor%": 20.0, "opacity_reg": 1.0, "push": 0.005, "noise": 0.0})
    assert refinement == pytest.approx({"every": 200, "target_now%": 0.0, "target%": 10.0, "after": 1000, "prune_now%": 20.0, "prune_floor%": 20.0, "grow_cap%": 30.0, "prune_cap%": 30.0, "alpha<": 0.01, "min_contrib<": 0.05, "decay%/pass": 99.5, "alpha_mul": 1.0, "clone_scale": 1.0, "max": 1500000})
    assert viewer.t("loss_debug_psnr").text == "PSNR: 32.50 dB"
    assert viewer.ui._values["_training_camera_struct_sections"] == (
        ("Resolution", (("target", "320x180"), ("source", "640x360"), ("full_res", False))),
        ("Ids", (("image", 5), ("camera", 7))),
        ("Pose", (("pos", (1.0, 2.0, 3.0)), ("target", (1.5, 2.0, 4.0)), ("up", (0.0, 1.0, 0.0)), ("near", 0.1), ("far", 120.0))),
        ("Projection", (("fx", 525.0), ("fy", 520.0), ("cx", 320.0), ("cy", 180.0))),
        ("Distortion A", (("k1", 0.01), ("k2", -0.02), ("p1", 0.001), ("p2", -0.002))),
        ("Distortion B", (("k3", 0.003), ("k4", -0.004), ("k5", 0.005), ("k6", -0.006))),
    )
    assert viewer.ui._values["_training_camera_pose_available"] is True
    assert viewer.ui._values["_training_view_overlay_segments"] == ()
    assert viewer.ui._values["_training_views_rows"] == (
        {
            "frame_index": 0,
            "image_name": "frame.png",
            "resolution": "640x360",
            "fx": 525.0,
            "fy": 520.0,
            "cx": 320.0,
            "cy": 180.0,
            "camera_min_dist": 0.1,
            "loss": 0.25,
            "psnr": 32.5,
            "visited": True,
            "is_last": True,
        },
    )


def test_update_ui_text_appends_pose_ppisp_values_when_provider_exists() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.trainer.target_tonemap_provider = SimpleNamespace(
        params_for_frame=lambda _frame_index: presenter.PPISPTonemapParams(exposureEv=1.25, crfGamma=(2.0, 2.1, 2.2))
    )

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    sections = viewer.ui._values["_training_camera_struct_sections"]
    assert sections[0] == ("Resolution", (("target", "320x180"), ("source", "640x360"), ("full_res", False), ("ppisp", True)))
    assert sections[-4] == ("PPISP Exposure", (("Exposure EV", 1.25),))
    assert sections[-1][0] == "PPISP Curve"
    curve_values = dict(sections[-1][1])
    assert curve_values["CRF Gamma"] == pytest.approx((2.0, 2.1, 2.2), abs=1e-6)


def test_update_ui_text_skips_training_view_rows_when_hidden() -> None:
    viewer = _viewer(loss_debug=False)

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.ui._values["_training_views_rows"] == ()


def test_update_ui_text_reuses_single_metrics_snapshot_for_training_rows() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.ui._values["show_training_views"] = True
    calls = 0

    def _snapshot() -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        return {
            "loss": np.asarray([0.25], dtype=np.float64),
            "mse": np.asarray([0.125], dtype=np.float64),
            "psnr": np.asarray([32.5], dtype=np.float64),
            "visited": np.asarray([True], dtype=bool),
        }

    viewer.s.trainer.frame_metrics_snapshot = _snapshot

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert calls == 1


def test_update_ui_text_sorts_training_view_rows_by_selected_column() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.ui._values["show_training_views"] = True
    viewer.ui._values["_training_views_sort_column"] = "loss"
    viewer.ui._values["_training_views_sort_descending"] = True
    first_frame = viewer.s.training_frames[0]
    viewer.s.training_frames = [
        SimpleNamespace(**{**vars(first_frame), "image_path": Path("frame_a.png"), "width": 640, "height": 360, "fx": 525.0, "fy": 520.0, "cx": 320.0, "cy": 180.0}),
        SimpleNamespace(**{**vars(first_frame), "image_id": 6, "camera_id": 8, "image_path": Path("frame_b.png"), "width": 960, "height": 540, "fx": 430.0, "fy": 428.0, "cx": 480.0, "cy": 270.0}),
    ]
    viewer.s.trainer.state.last_frame_index = 1
    viewer.s.trainer.frame_metrics_snapshot = lambda: {
        "loss": np.asarray([0.25, 0.75], dtype=np.float64),
        "mse": np.asarray([0.125, 0.25], dtype=np.float64),
        "psnr": np.asarray([32.5, 28.0], dtype=np.float64),
        "visited": np.asarray([True, False], dtype=bool),
    }

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    rows = viewer.ui._values["_training_views_rows"]
    assert [row["image_name"] for row in rows] == ["frame_b.png", "frame_a.png"]
    assert [row["loss"] for row in rows] == [0.75, 0.25]


def test_update_ui_text_publishes_photometric_prepare_progress(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.photometric_prepare_pending_active = True
    viewer.s.photometric_trainer = SimpleNamespace(
        pair_dataset_prepare_active=True,
        pair_dataset_prepare_fraction=0.5,
        pair_dataset_prepare_current_name="frame_001.png",
        pair_dataset_prepare_total_frames=4,
        pair_dataset_prepare_completed_frames=2,
        pair_pool=(0, 1, 2),
        provider=SimpleNamespace(params_for_frame=lambda _frame_index: presenter.PPISPTonemapParams()),
        state=SimpleNamespace(step=0, last_loss=0.0, ema_loss=0.0, last_pair_count=0),
    )
    monkeypatch.setattr(presenter, "_photometric_param_sections", lambda _viewer: ())

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.ui._values["_photometric_prepare_active"] is True
    assert viewer.ui._values["_photometric_prepare_fraction"] == pytest.approx(0.5)
    assert viewer.ui._texts["photometric_prepare_current"] == "frame_001.png"
    assert viewer.ui._texts["photometric_status"] == "Photometric: preparing dataset | auto-start | frames=2/4"


def test_update_toolkit_history_averages_samples_within_bucket() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.toolkit = SimpleNamespace(tk=viewer_ui.ToolkitState())
    viewer.s.last_time = 10.0
    viewer.s.trainer.state.step = 1
    viewer.s.trainer.state.avg_loss = 3.0
    viewer.s.trainer.state.avg_ssim = 0.3
    viewer.s.trainer.state.avg_psnr = 20.0

    presenter._update_toolkit_history(viewer, 1.0 / 60.0)

    viewer.s.last_time = 11.0
    viewer.s.trainer.state.step = 10
    viewer.s.trainer.state.avg_loss = 6.0
    viewer.s.trainer.state.avg_ssim = 0.6
    viewer.s.trainer.state.avg_psnr = 22.0
    presenter._update_toolkit_history(viewer, 1.0 / 60.0)

    viewer.s.last_time = 12.0
    viewer.s.trainer.state.step = 20
    viewer.s.trainer.state.avg_loss = 9.0
    viewer.s.trainer.state.avg_ssim = 0.9
    viewer.s.trainer.state.avg_psnr = 24.0
    presenter._update_toolkit_history(viewer, 1.0 / 60.0)

    tk = viewer.toolkit.tk
    assert list(tk.frame_time_history) == [pytest.approx(1.0 / 60.0), pytest.approx(1.0 / 60.0), pytest.approx(1.0 / 60.0)]
    assert list(tk.step_history) == [pytest.approx((1.0 + 10.0 + 20.0) / 3.0)]
    assert list(tk.step_time_history) == [pytest.approx(11.0)]
    assert list(tk.loss_history) == [pytest.approx(6.0)]
    assert list(tk.ssim_history) == [pytest.approx(0.6)]
    assert list(tk.psnr_history) == [pytest.approx(22.0)]

    viewer.s.last_time = 13.0
    viewer.s.trainer.state.step = 31
    viewer.s.trainer.state.avg_loss = 12.0
    viewer.s.trainer.state.avg_ssim = 0.95
    viewer.s.trainer.state.avg_psnr = 26.0
    presenter._update_toolkit_history(viewer, 1.0 / 60.0)

    assert list(tk.frame_time_history) == [
        pytest.approx(1.0 / 60.0),
        pytest.approx(1.0 / 60.0),
        pytest.approx(1.0 / 60.0),
        pytest.approx(1.0 / 60.0),
    ]
    assert list(tk.step_history) == [pytest.approx((1.0 + 10.0 + 20.0) / 3.0), pytest.approx(31.0)]
    assert list(tk.loss_history) == [pytest.approx(6.0), pytest.approx(12.0)]


def test_python_frame_capture_summary_uses_recent_frame_times_and_process_vram(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    viewer.toolkit = SimpleNamespace(tk=viewer_ui.ToolkitState())
    viewer.toolkit.tk.frame_time_history.extend([1.0 / 60.0, 1.0 / 58.0])
    viewer.s.fps_smooth = 57.5
    snapshot = _buffer_snapshot(64, process_vram=96, process_vram_delta=32, process_vram_source="nvidia-smi")
    calls: list[bool] = []

    monkeypatch.setattr(presenter, "collect_resource_debug_snapshot", lambda _viewer, include_process_vram=False: calls.append(bool(include_process_vram)) or snapshot)

    summary = presenter._python_frame_capture_summary(viewer)

    assert calls == [True]
    assert summary.resource_snapshot is snapshot
    assert summary.recent_frame_times_s == (pytest.approx(1.0 / 60.0), pytest.approx(1.0 / 58.0))
    assert summary.smoothed_fps == pytest.approx(57.5)


def test_update_ui_text_skips_resource_debug_snapshot_when_closed(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    calls: list[bool] = []

    monkeypatch.setattr(presenter, "collect_resource_debug_snapshot", lambda _viewer_obj, *, include_process_vram=False: calls.append(bool(include_process_vram)) or _buffer_snapshot(128))
    _patch_vram_queries(monkeypatch)

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert calls == [False]
    assert "_resource_debug_snapshot" not in viewer.ui._values
    assert viewer.ui._values["_menu_bar_dataset_vram_bytes"] == 0
    assert viewer.ui._values["_menu_bar_app_vram_bytes"] == 128
    assert viewer.ui._values["_menu_bar_total_vram_bytes"] == 128


def test_update_ui_text_throttles_resource_debug_snapshot(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    viewer.ui._values.update({"show_resource_debug": True, "_resource_debug_snapshot": None, "_resource_debug_next_update": 0.0})
    snapshots = [_buffer_snapshot(64), _buffer_snapshot(96)]
    calls: list[bool] = []

    monkeypatch.setattr(presenter, "collect_resource_debug_snapshot", lambda _viewer_obj, *, include_process_vram=False: calls.append(bool(include_process_vram)) or snapshots[min(len(calls) - 1, len(snapshots) - 1)])

    presenter.update_ui_text(viewer, 1.0 / 60.0)
    presenter.update_ui_text(viewer, 1.0 / 60.0)
    viewer.ui._values["_resource_debug_refresh_requested"] = True
    viewer.ui._values["_resource_debug_process_vram_requested"] = True
    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert calls == [False, True]
    assert viewer.ui._values["_resource_debug_snapshot"] is snapshots[1]
    assert viewer.ui._values["_resource_debug_refresh_requested"] is False
    assert viewer.ui._values["_resource_debug_process_vram_requested"] is False


def test_update_ui_text_refreshes_menu_bar_device_vram(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    _patch_vram_queries(monkeypatch, used=3 * 1024**3, used_source="Windows GPU Adapter Memory", capacity=24 * 1024**3, capacity_source="DXGI Adapter Desc")
    monkeypatch.setattr(
        presenter,
        "collect_resource_debug_snapshot",
        lambda _viewer, include_process_vram=False: _resource_debug_snapshot(
            ("Texture", "viewer.dataset_texture", "viewer.trainer.frame_targets_native[0]", 256 * 1024**2, "rgba8", "srv", 1),
            ("Buffer", "renderer.buf", "viewer.main_renderer.buf", 512 * 1024**2, "buf", "rw", 2),
        ),
    )

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.ui._values["_menu_bar_device_vram_bytes"] == 3 * 1024**3
    assert viewer.ui._values["_menu_bar_device_vram_source"] == "Windows GPU Adapter Memory"
    assert viewer.ui._values["_menu_bar_device_vram_total_bytes"] == 24 * 1024**3
    assert viewer.ui._values["_menu_bar_device_vram_total_source"] == "DXGI Adapter Desc"
    assert viewer.ui._values["_menu_bar_dataset_vram_bytes"] == 256 * 1024**2
    assert viewer.ui._values["_menu_bar_app_vram_bytes"] == 512 * 1024**2
    assert viewer.ui._values["_menu_bar_total_vram_bytes"] == 768 * 1024**2


def test_update_ui_text_reuses_resource_debug_snapshot_for_menu_bar_dataset_vram(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    snapshot = _resource_debug_snapshot(
        ("Texture", "viewer.dataset_texture_bc7", "viewer.state.colmap_import_textures[0]", 256, "bc7", "srv", 1),
        ("Buffer", "renderer.buf", "viewer.main_renderer.buf", 768, "buf", "rw", 2),
    )
    viewer.ui._values["show_resource_debug"] = True
    viewer.ui._values["_resource_debug_snapshot"] = snapshot
    viewer.ui._values["_resource_debug_next_update"] = viewer.s.last_time + 60.0
    _patch_vram_queries(monkeypatch)
    monkeypatch.setattr(presenter, "collect_resource_debug_snapshot", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not recollect snapshot")))

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.ui._values["_menu_bar_dataset_vram_bytes"] == 256
    assert viewer.ui._values["_menu_bar_app_vram_bytes"] == 768
    assert viewer.ui._values["_menu_bar_total_vram_bytes"] == 1024


def test_update_ui_text_previews_current_schedule_values_without_trainer() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.trainer = None
    viewer.s.training_active = False

    presenter.update_ui_text(viewer, 1.0 / 60.0)
    schedule_sections = _section_dict(viewer.ui._values["_training_schedule_sections"])

    assert schedule_sections[""] == {"step": 0, "stage": "Stage 0", "sh": "SH0"}
    assert schedule_sections["Learning Rates"] == pytest.approx({"base": 0.005, "pos": 1.0, "scale": 5.0, "rot": 1.0, "dc": 5.0, "opacity": 5.0, "sh": 0.05})
    assert schedule_sections["Other"] == pytest.approx({"colorspace": 1.0, "dither": 0.5, "target%": 10.0, "prune_floor%": 10.0, "opacity_reg": 3.0, "push": 0.001, "noise": 500000.0})


def test_render_frame_recovers_missing_main_renderer_by_recreating_it(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.s.renderer = None
    render_context = _render_context()
    calls: list[object] = []
    replacement_renderer = _DummyRenderer()

    _patch_render_frame(
        monkeypatch,
        calls,
        advance_colmap_import=lambda viewer_obj: calls.append("import"),
        recreate_renderer=lambda viewer_obj, width, height: calls.append(("resize", width, height)) or setattr(viewer_obj.s, "renderer", replacement_renderer),
    )

    presenter.render_frame(viewer, render_context)

    assert viewer.s.renderer is replacement_renderer
    assert viewer.s.viewport_texture == "main_tex"
    assert calls == ["apply", "import", ("resize", 640, 360), "main", "ui"]


def test_render_frame_consumes_pending_reinitialize_before_live_params(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.s.training_active = False
    viewer.s.pending_training_reinitialize = True
    render_context = _render_context()
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "reinitialize_training_scene", lambda viewer_obj: calls.append("init"))
    _patch_render_frame(monkeypatch, calls, advance_colmap_import=lambda viewer_obj: calls.append("import"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.pending_training_reinitialize is False
    assert viewer.s.trainer.step_calls == 0
    assert calls == ["init", "apply", "import", "main", "ui"]


def test_render_frame_consumes_pending_python_capture(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.s.pending_python_frame_capture = True
    render_context = _render_context()
    calls: list[object] = []
    summary = presenter.frame_capture.PythonFrameCaptureSummary(recent_frame_times_s=(1.0 / 60.0,), smoothed_fps=60.0)

    def _capture(action, *, frame_index=None, directory=None, summary_provider=None):
        calls.append(("capture", frame_index, directory, callable(summary_provider)))
        action()
        calls.append(("summary", summary_provider() if callable(summary_provider) else None))
        return Path("temp/python_frame_capture_000.txt"), Path("temp/python_frame_capture_000.prof")

    monkeypatch.setattr(presenter.frame_capture, "capture_python_frame", _capture)
    monkeypatch.setattr(presenter, "_python_frame_capture_summary", lambda _viewer: summary)
    _patch_render_frame(
        monkeypatch,
        calls,
        maybe_reallocate_renderers=lambda viewer_obj, width, height, current_time: calls.append(("periodic", width, height)),
    )

    presenter.render_frame(viewer, render_context)

    assert viewer.s.pending_python_frame_capture is False
    assert viewer.s.viewport_texture == "main_tex"
    assert calls == [("capture", 0, None, True), "apply", ("periodic", 640, 360), "main", "ui", ("summary", summary)]


def test_render_frame_leaves_pending_renderdoc_capture_for_app(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.s.pending_renderdoc_frame_capture = True
    render_context = _render_context()
    calls: list[object] = []

    _patch_render_frame(
        monkeypatch,
        calls,
        maybe_reallocate_renderers=lambda viewer_obj, width, height, current_time: calls.append(("periodic", width, height)),
    )

    presenter.render_frame(viewer, render_context)

    assert viewer.s.pending_renderdoc_frame_capture is True
    assert viewer.s.viewport_texture == "main_tex"
    assert calls == ["apply", ("periodic", 640, 360), "main", "ui"]


def test_render_frame_skips_training_batch_when_runtime_resize_is_applied(monkeypatch):
    viewer = _viewer(loss_debug=False)
    render_context = _render_context()
    calls: list[str] = []

    def _apply_live_params(viewer_obj) -> None:
        viewer_obj.s.pending_training_runtime_resize = True
        calls.append("apply")

    _patch_render_frame(
        monkeypatch,
        calls,
        apply_live_params=_apply_live_params,
        advance_colmap_import=lambda viewer_obj: calls.append("import"),
        ensure_training_runtime_resolution=lambda viewer_obj: calls.append("train_resize") or True,
    )

    presenter.render_frame(viewer, render_context)

    assert viewer.s.trainer.step_calls == 0
    assert viewer.s.trainer.step_batch_calls == []
    assert viewer.s.last_training_batch_steps == 0
    assert calls == ["apply", "import", "train_resize", "main", "ui"]


def test_render_frame_resizes_main_renderer_from_viewport_size(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.toolkit.viewport_size = lambda: (480, 270)
    viewer.s.renderer.width = 640
    viewer.s.renderer.height = 360
    render_context = _render_context(1280, 720)
    calls: list[object] = []

    _patch_render_frame(
        monkeypatch,
        calls,
        advance_colmap_import=lambda viewer_obj: calls.append("import"),
        recreate_renderer=lambda viewer_obj, width, height: calls.append(("resize", width, height)) or setattr(viewer_obj.s.renderer, "width", width) or setattr(viewer_obj.s.renderer, "height", height),
    )

    presenter.render_frame(viewer, render_context)

    assert viewer.s.viewport_texture == "main_tex"
    assert ("resize", 480, 270) in calls


def test_update_ui_text_uses_permutation_averages() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.trainer.state.avg_loss = 1.25
    viewer.s.trainer.state.avg_ssim = 0.9975
    viewer.s.trainer.state.avg_density_loss = 6.5e-3
    viewer.s.trainer.state.avg_psnr = 26.75
    viewer.s.trainer.state.step = 120
    viewer.s.training_elapsed_s = 30.0

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.t("training_time").text == "Time: 00:30"
    assert viewer.t("training_iters_avg").text == "Avg it/s: 4.00"
    assert viewer.t("training_loss").text == "Loss Avg: 1.250000e+00"
    assert viewer.t("training_ssim").text == "SSIM Avg: 0.9975"
    assert viewer.t("training_density").text == "Density Avg: 6.500000e-03"
    assert viewer.t("training_psnr").text == "PSNR Avg: 26.750 dB"
    assert viewer.t("loss_debug_psnr").text == "PSNR: 32.50 dB"
    assert viewer.ui._values["_training_resolution_sections"] == (("Train Res", (("size", "640x360"), ("factor", 1))),)


def test_update_ui_text_populates_colmap_import_progress_fields() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.colmap_import_progress = ColmapImportProgress(
        dataset_root=Path("dataset"),
        colmap_root=Path("dataset"),
        database_path=None,
        images_root=Path("dataset/images"),
        init_mode="pointcloud",
        compress_dataset_using_bc7=False,
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=1600,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.25,
        phase="load_textures",
        current=3,
        total=8,
        current_name="frame_003.png",
    )

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.ui._values["_colmap_import_active"] is True
    assert viewer.ui._values["_colmap_import_fraction"] == 3.0 / 8.0
    assert viewer.t("colmap_import_status").text == "Loading images: 3/8"
    assert viewer.t("colmap_import_current").text == "frame_003.png"
    
def test_update_ui_text_publishes_photometric_colmap_import_progress() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.colmap_import_progress = ColmapImportProgress(
        dataset_root=Path("dataset"),
        colmap_root=Path("dataset"),
        database_path=None,
        images_root=Path("dataset/images"),
        init_mode="pointcloud",
        compress_dataset_using_bc7=False,
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=1600,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.25,
        phase="photometric_optimize",
        current=24,
        total=1000,
        current_name="Loss: last=1.250000e-03 | ema=9.500000e-04",
    )

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.ui._values["_colmap_import_active"] is True
    assert viewer.ui._values["_colmap_import_fraction"] == 24.0 / 1000.0
    assert viewer.t("colmap_import_status").text == "Optimizing photometric compensation: 24/1000"
    assert viewer.t("colmap_import_current").text == "Loss: last=1.250000e-03 | ema=9.500000e-04"


@pytest.mark.parametrize("case,loss_debug_view,step,render_frame_index", (("rendered", 0, 17, 42), ("abs_diff", 2, 0, 7)))
def test_render_debug_source_cases(monkeypatch, case: str, loss_debug_view: int, step: int, render_frame_index: int) -> None:
    viewer = _viewer(loss_debug=True)
    overlay_renderer = _DummyRenderer()
    viewer.c("loss_debug_view").value = loss_debug_view
    viewer.s.trainer.state.step = step
    viewer.s.training_renderer = overlay_renderer
    source_tex, stats, width, height, sample_vars = presenter._render_debug_source(viewer, _DummyEncoder(), 0, render_frame_index)
    assert source_tex == "training_preview_tex"
    assert width == 320 and height == 180
    assert overlay_renderer.resolution_calls == [(320, 180)]
    if case == "abs_diff":
        assert stats["written_entries"] == 2
        return
    render_call = overlay_renderer.training_forward_calls[0]
    assert stats["generated_entries"] == 1
    assert viewer.s.trainer.training_resolution_calls == [(0, 17)]
    assert viewer.s.trainer.sample_vars_calls == [(0, 17, 42)]
    assert viewer.s.trainer.background_seed_calls == [42]
    assert viewer.s.trainer.hparam_calls == [(17, overlay_renderer)]
    assert viewer.s.trainer.sort_calls == [(0, 17, (0, 320, 180))]
    assert sample_vars["g_TrainingSubsample"]["stepIndex"] == np.uint32(42)
    assert render_call["camera"] == (0, 320, 180)
    assert render_call["training_native_camera"] == (0, 640, 360)
    assert render_call["training_background_seed"] == 1042
    assert render_call["clone_counts_buffer"] == "clone_counts"
    assert render_call["splat_contribution_buffer"] == "splat_contribution"
    assert render_call["training_workspace"] == {"workspace": True}
    np.testing.assert_allclose(render_call["sort_camera_position"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert render_call["sort_camera_dither_sigma"] == 0.125
    assert render_call["sort_camera_dither_seed"] == 555


@pytest.mark.parametrize("case", ("downscaled", "sampled_native", "full_res", "full_res_linear", "tonemap_off"))
def test_render_debug_target_cases(monkeypatch, case: str) -> None:
    viewer = _viewer(loss_debug=True)
    width, height, step, sample_vars = 320, 180, 9, None
    if case in {"sampled_native", "tonemap_off"}:
        viewer.s.debug_target_sample_kernel = _CaptureKernel()
        output = SimpleNamespace(width=320, height=180)
        sample_vars = viewer.s.trainer.training_sample_vars(0, step, sample_seed_step=77)
        monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)
    if case == "downscaled":
        sample_vars = {"g_TrainingSubsample": {"enabled": np.uint32(0), "factor": np.uint32(1)}}; step = 5
    elif case == "sampled_native":
        viewer.s.trainer.subsample_factor = 2
    elif case == "full_res":
        viewer.ui._values["training_camera_full_resolution"] = True; viewer.s.trainer.subsample_factor = 2; width, height = 640, 360
    elif case == "full_res_linear":
        viewer.ui._values["training_camera_full_resolution"] = True; viewer.s.trainer.native_target_is_linear = True; width, height = 640, 360
    elif case == "tonemap_off":
        viewer.ui._values["training_camera_ppisp_tonemap"] = False
    target, is_linear = presenter._render_debug_target(viewer, _DummyEncoder(), 0, width, height, step, sample_vars)
    if case == "downscaled":
        assert (target, is_linear, viewer.s.trainer.target_calls) == ("target_tex_0_False", True, [(0, False, True)])
    elif case == "sampled_native":
        vars = viewer.s.debug_target_sample_kernel.calls[0]["vars"]
        assert target is output and is_linear is False and viewer.s.trainer.target_calls == [(0, True, True)]
        assert vars["g_SourceTarget"] == "target_tex_0_True" and vars["g_DownscaledTarget"] is output
        assert vars["g_TargetWidth"] == 320 and vars["g_TargetHeight"] == 180
        assert vars["g_TrainingSubsample"]["stepIndex"] == np.uint32(77)
    elif case == "full_res":
        assert (target, is_linear, viewer.s.trainer.target_calls) == ("target_tex_0_True", False, [(0, True, True)])
    elif case == "full_res_linear":
        assert (target, is_linear, viewer.s.trainer.target_calls) == ("target_tex_0_True", True, [(0, True, True)])
    else:
        vars = viewer.s.debug_target_sample_kernel.calls[0]["vars"]
        assert target is output and is_linear is False and viewer.s.trainer.target_calls == [(0, True, False)]
        assert vars["g_SourceTarget"] == "target_tex_0_True" and vars["g_DownscaledTarget"] is output


def test_dispatch_debug_abs_diff_uses_runtime_ui_scale(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.c("loss_debug_abs_scale").value = 3.5
    viewer.s.debug_abs_diff_kernel = _CaptureKernel()
    encoder = _DummyEncoder()
    output = SimpleNamespace(width=640, height=360)

    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)

    result = presenter._dispatch_debug_abs_diff(viewer, encoder, "rendered_tex", "target_tex", 640, 360, target_is_linear=True)

    assert result is output
    assert len(viewer.s.debug_abs_diff_kernel.calls) == 1
    vars = viewer.s.debug_abs_diff_kernel.calls[0]["vars"]
    assert vars["g_DebugDiffScale"] == 3.5
    assert vars["g_DebugRenderedIsLinear"] == 1
    assert vars["g_DebugTargetIsLinear"] == 1


def test_dispatch_debug_edge_filter_uses_source_texture(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.s.debug_edge_kernel = _CaptureKernel()
    encoder = _DummyEncoder()
    output = SimpleNamespace(width=640, height=360)

    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)

    result = presenter._dispatch_debug_edge_filter(viewer, encoder, "source_tex", 640, 360)

    assert result is output
    assert len(viewer.s.debug_edge_kernel.calls) == 1
    vars = viewer.s.debug_edge_kernel.calls[0]["vars"]
    assert vars["g_DebugRendered"] == "source_tex"
    assert vars["g_DebugWidth"] == 640
    assert vars["g_DebugHeight"] == 360
    assert vars["g_DebugSourceIsLinear"] == 0


def test_dispatch_debug_dssim_runs_features_blur_and_compose(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.s.trainer.training.target_alpha_mode = 2
    viewer.s.trainer.training.target_alpha_threshold = 0.25
    encoder = _DummyEncoder()
    blur_calls: list[tuple[object, object, int]] = []

    class _DummyBlur:
        def blur(self, encoder_obj, input_buffer, output_buffer, channel_count):
            blur_calls.append((input_buffer, output_buffer, channel_count))

    output = SimpleNamespace(width=640, height=360)
    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)
    monkeypatch.setattr(presenter, "_ensure_debug_dssim_runtime", lambda viewer_obj, width, height: setattr(viewer_obj.s, "debug_dssim_blur", _DummyBlur()) or setattr(viewer_obj.s, "debug_dssim_moments", "moments") or setattr(viewer_obj.s, "debug_dssim_blurred_moments", "blurred"))
    viewer.s.debug_dssim_features_kernel = _CaptureKernel()
    viewer.s.debug_dssim_compose_kernel = _CaptureKernel()

    result = presenter._dispatch_debug_dssim(viewer, encoder, "rendered_tex", "target_tex", 640, 360, target_is_linear=True)

    assert result is output
    assert blur_calls == [("moments", "blurred", 16)]
    assert viewer.s.debug_dssim_features_kernel.calls[0]["vars"]["g_DebugRendered"] == "rendered_tex"
    assert viewer.s.debug_dssim_features_kernel.calls[0]["vars"]["g_DebugTarget"] == "target_tex"
    assert viewer.s.debug_dssim_features_kernel.calls[0]["vars"]["g_DebugTargetIsLinear"] == 1
    assert viewer.s.debug_dssim_compose_kernel.calls[0]["vars"]["g_DebugTarget"] == "target_tex"
    assert viewer.s.debug_dssim_compose_kernel.calls[0]["vars"]["g_SSIMC2"] == 9e-4
    assert viewer.s.debug_dssim_compose_kernel.calls[0]["vars"]["g_TargetAlphaMode"] == 2
    assert viewer.s.debug_dssim_compose_kernel.calls[0]["vars"]["g_TargetAlphaThreshold"] == pytest.approx(0.25)


def test_ensure_debug_dssim_runtime_releases_stale_resources_before_recreate(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.s.debug_dssim_resolution = (4946, 3286)
    viewer.s.debug_dssim_blur = SimpleNamespace(_scratch_buffers={16: "old_scratch"})
    viewer.s.debug_dssim_moments = "old_moments"
    viewer.s.debug_dssim_blurred_moments = "old_blurred"
    released: list[tuple[object, object, object, tuple[object, ...]]] = []

    class _Blur:
        def __init__(self, device, width: int, height: int) -> None:
            del device
            self.width = width
            self.height = height

        def make_buffer(self, channel_count: int, name: str | None = None):
            del name
            return f"buffer:{self.width}x{self.height}:{channel_count}"

    def _release_runtime(viewer_obj) -> None:
        blur = viewer_obj.s.debug_dssim_blur
        released.append(
            (
                viewer_obj.s.debug_dssim_resolution,
                viewer_obj.s.debug_dssim_moments,
                viewer_obj.s.debug_dssim_blurred_moments,
                tuple(getattr(blur, "_scratch_buffers", {}).values()),
            )
        )
        viewer_obj.s.debug_dssim_blur = None
        viewer_obj.s.debug_dssim_resolution = None
        viewer_obj.s.debug_dssim_moments = None
        viewer_obj.s.debug_dssim_blurred_moments = None

    monkeypatch.setattr(viewer_session, "_release_debug_dssim_runtime", _release_runtime)
    monkeypatch.setattr(presenter, "SeparableGaussianBlur", _Blur)

    presenter._ensure_debug_dssim_runtime(viewer, 640, 360)

    assert released == [
        ((4946, 3286), "old_moments", "old_blurred", ("old_scratch",)),
    ]
    assert viewer.s.debug_dssim_resolution == (640, 360)
    assert isinstance(viewer.s.debug_dssim_blur, _Blur)
    assert viewer.s.debug_dssim_moments == "buffer:640x360:16"
    assert viewer.s.debug_dssim_blurred_moments == "buffer:640x360:16"


def test_render_debug_view_routes_edge_modes(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    encoder = _DummyEncoder()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(presenter, "_render_debug_source", lambda viewer_obj, enc, frame_idx, render_frame_index: ("rendered_tex", {"generated_entries": 1}, 640, 360, {"g_TrainingSubsample": {"enabled": np.uint32(0)}}))
    monkeypatch.setattr(presenter, "_render_debug_target", lambda viewer_obj, enc, frame_idx, width, height, step, sample_vars: ("target_tex", True))
    monkeypatch.setattr(presenter, "_dispatch_debug_abs_diff", lambda viewer_obj, enc, rendered_tex, target_tex, width, height, *, rendered_is_linear=True, target_is_linear=False: calls.append(("abs_diff", rendered_tex, target_tex, width, height, rendered_is_linear, target_is_linear)) or "abs_diff_tex")
    monkeypatch.setattr(presenter, "_dispatch_debug_dssim", lambda viewer_obj, enc, rendered_tex, target_tex, width, height, *, target_is_linear=False: calls.append(("dssim", rendered_tex, target_tex, width, height, target_is_linear)) or "dssim_tex")
    monkeypatch.setattr(presenter, "_dispatch_debug_edge_filter", lambda viewer_obj, enc, source_tex, width, height, *, source_is_linear=False: calls.append(("edge", source_tex, width, height, source_is_linear)) or f"edge_{source_tex}")
    monkeypatch.setattr(presenter, "_dispatch_training_debug_present", lambda viewer_obj, enc, source_tex, source_width, source_height, output_width, output_height, *, source_is_linear=False, apply_loss_colorspace=False, source_uses_target_loss_colorspace=False: calls.append(("present", source_tex, source_width, source_height, output_width, output_height, source_is_linear, apply_loss_colorspace, source_uses_target_loss_colorspace)) or "present_tex")

    viewer.c("loss_debug_view").value = 2
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"
    viewer.c("loss_debug_view").value = 3
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"
    viewer.c("loss_debug_view").value = 4
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"
    viewer.c("loss_debug_view").value = 5
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"

    assert calls == [
        ("abs_diff", "rendered_tex", "target_tex", 640, 360, True, True),
        ("present", "abs_diff_tex", 640, 360, 640, 360, False, False, False),
        ("dssim", "rendered_tex", "target_tex", 640, 360, True),
        ("present", "dssim_tex", 640, 360, 640, 360, False, False, False),
        ("edge", "rendered_tex", 640, 360, True),
        ("present", "edge_rendered_tex", 640, 360, 640, 360, False, False, False),
        ("edge", "target_tex", 640, 360, True),
        ("present", "edge_target_tex", 640, 360, 640, 360, False, False, False),
    ]


def test_render_debug_view_presents_target_using_reported_linearity(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    encoder = _DummyEncoder()
    calls: list[tuple[str, object, bool, bool, bool]] = []
    target_results = iter([("linear_target_tex", True), ("raw_target_tex", False)])

    monkeypatch.setattr(presenter, "_render_debug_source", lambda viewer_obj, enc, frame_idx, render_frame_index: ("rendered_tex", {"generated_entries": 1}, 640, 360, {"g_TrainingSubsample": {"enabled": np.uint32(0)}}))
    monkeypatch.setattr(presenter, "_render_debug_target", lambda viewer_obj, enc, frame_idx, width, height, step, sample_vars: next(target_results))
    monkeypatch.setattr(presenter, "_dispatch_training_debug_present", lambda viewer_obj, enc, source_tex, source_width, source_height, output_width, output_height, *, source_is_linear=False, apply_loss_colorspace=False, source_uses_target_loss_colorspace=False: calls.append(("present", source_tex, source_is_linear, apply_loss_colorspace, source_uses_target_loss_colorspace)) or "present_tex")

    viewer.c("loss_debug_view").value = 0
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"
    viewer.c("loss_debug_view").value = 1
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"

    assert calls == [
        ("present", "rendered_tex", True, True, False),
        ("present", "linear_target_tex", True, True, True),
        ("present", "raw_target_tex", False, True, True),
    ]


def test_render_debug_view_skips_target_work_for_rendered_mode(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    encoder = _DummyEncoder()
    calls: list[tuple[str, object, bool, bool, bool]] = []

    monkeypatch.setattr(presenter, "_render_debug_source", lambda viewer_obj, enc, frame_idx, render_frame_index: ("rendered_tex", {"generated_entries": 1}, 640, 360, {"g_TrainingSubsample": {"enabled": np.uint32(0)}}))
    monkeypatch.setattr(presenter, "_render_debug_target", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rendered view should not render the target path")))
    monkeypatch.setattr(presenter, "_dispatch_training_debug_present", lambda viewer_obj, enc, source_tex, source_width, source_height, output_width, output_height, *, source_is_linear=False, apply_loss_colorspace=False, source_uses_target_loss_colorspace=False: calls.append(("present", source_tex, source_is_linear, apply_loss_colorspace, source_uses_target_loss_colorspace)) or "present_tex")

    viewer.c("loss_debug_view").value = 0
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"

    assert calls == [("present", "rendered_tex", True, True, False)]


def test_dispatch_training_debug_present_passes_colorspace_mod(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.s.debug_letterbox_kernel = _CaptureKernel()
    viewer.s.trainer.state.step = 17
    viewer.s.trainer.training.target_alpha_mode = 1
    viewer.s.trainer.training.target_alpha_threshold = 0.375
    encoder = _DummyEncoder()
    output = SimpleNamespace(width=640, height=360)

    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)

    result = presenter._dispatch_training_debug_present(viewer, encoder, "source_tex", 320, 180, 640, 360, source_is_linear=True, apply_loss_colorspace=True, source_uses_target_loss_colorspace=True)

    assert result is output
    vars = viewer.s.debug_letterbox_kernel.calls[0]["vars"]
    assert vars["g_LetterboxSource"] == "source_tex"
    assert vars["g_LetterboxSourceIsLinear"] == 1
    assert vars["g_LetterboxApplyLossColorspace"] == 1
    assert vars["g_TargetAlphaMode"] == 1
    assert vars["g_TargetAlphaThreshold"] == pytest.approx(0.375)
    assert np.isclose(vars["g_ColorspaceMod"], presenter.resolve_colorspace_mod(viewer.s.trainer.training, 17))


def test_render_debug_view_routes_target_view_through_target_loss_present(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    encoder = _DummyEncoder()
    calls: list[tuple[object, bool, bool, bool]] = []

    monkeypatch.setattr(presenter, "_render_debug_source", lambda viewer_obj, enc, frame_idx, render_frame_index: ("rendered_tex", {"generated_entries": 1}, 640, 360, {"g_TrainingSubsample": {"enabled": np.uint32(0)}}))
    monkeypatch.setattr(presenter, "_render_debug_target", lambda viewer_obj, enc, frame_idx, width, height, step, sample_vars: ("target_tex", True))
    monkeypatch.setattr(
        presenter,
        "_dispatch_training_debug_present",
        lambda viewer_obj, enc, source_tex, source_width, source_height, output_width, output_height, *, source_is_linear=False, apply_loss_colorspace=False, source_uses_target_loss_colorspace=False: calls.append((source_tex, source_is_linear, apply_loss_colorspace, source_uses_target_loss_colorspace)) or "present_tex",
    )

    viewer.c("loss_debug_view").value = 1

    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"
    assert calls == [("target_tex", True, True, True)]


def test_dispatch_viewport_present_uses_present_kernel(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.debug_letterbox_kernel = _CaptureKernel()
    encoder = _DummyEncoder()
    output = SimpleNamespace(width=640, height=360)

    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)

    result = presenter._dispatch_viewport_present(viewer, encoder, "source_tex", 320, 180, 640, 360)

    assert result is output
    assert len(viewer.s.debug_letterbox_kernel.calls) == 1
    vars = viewer.s.debug_letterbox_kernel.calls[0]["vars"]
    assert vars["g_LetterboxSource"] == "source_tex"
    assert vars["g_LetterboxSourceWidth"] == 320
    assert vars["g_LetterboxSourceHeight"] == 180
    assert vars["g_LetterboxOutputWidth"] == 640
    assert vars["g_LetterboxOutputHeight"] == 360
    assert vars["g_LetterboxSourceIsLinear"] == 0

    assert vars["g_LetterboxSourceUsesTargetLossColorspace"] == 1
    presenter._dispatch_viewport_present(viewer, encoder, "linear_source_tex", 320, 180, 640, 360, source_is_linear=True)
    assert viewer.s.debug_letterbox_kernel.calls[1]["vars"]["g_LetterboxSourceIsLinear"] == 1


def test_render_main_view_wraps_viewport_present_in_main_view_group(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.debug_letterbox_kernel = _CaptureKernel()
    encoder = _DummyEncoder()
    output = SimpleNamespace(width=640, height=360)

    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)
    monkeypatch.setattr(presenter, "_ppisp_preview_enabled", lambda _viewer: False)

    result = presenter._render_main_view(viewer, encoder)

    assert result is output
    assert encoder.groups == [
        ("push", "Viewer Main View"),
        ("push", "Viewer Viewport Present"),
        ("pop", None),
        ("pop", None),
    ]


def test_dispatch_viewport_present_zero_stays_zero_and_output_is_finite(device) -> None:
    viewer = SimpleNamespace(
        device=device,
        s=SimpleNamespace(debug_present_texture=None, debug_letterbox_kernel=None),
    )
    viewer_session.create_debug_shaders(viewer)
    source = device.create_texture(
        format=spy.Format.rgba32_float,
        width=2,
        height=2,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )
    source.copy_from_numpy(
        np.array(
            [
                [[0.0, 0.0, 0.0, 1.0], [np.nan, 0.0, -0.5, 1.0]],
                [[0.25, 0.5, 1.0, 1.0], [4.0, 2.0, 0.125, 1.0]],
            ],
            dtype=np.float32,
        )
    )

    encoder = device.create_command_encoder()
    output = presenter._dispatch_viewport_present(viewer, encoder, source, 2, 2, 2, 2)
    device.submit_command_buffer(encoder.finish())
    device.wait()

    image = np.asarray(output.to_numpy(), dtype=np.float32)
    assert np.all(np.isfinite(image[..., :3]))
    np.testing.assert_allclose(image[0, 0, :3], np.zeros((3,), dtype=np.float32), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(image[0, 1, :3], np.zeros((3,), dtype=np.float32), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(image[..., 3], np.ones((2, 2), dtype=np.float32), rtol=0.0, atol=1e-7)


def test_dispatch_viewport_present_does_not_apply_extra_srgb_transform(device) -> None:
    viewer = SimpleNamespace(
        device=device,
        s=SimpleNamespace(debug_present_texture=None, debug_letterbox_kernel=None),
    )
    viewer_session.create_debug_shaders(viewer)
    source = device.create_texture(
        format=spy.Format.rgba32_float,
        width=1,
        height=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )
    source.copy_from_numpy(np.array([[[0.25, 0.5, 1.0, 1.0]]], dtype=np.float32))

    encoder = device.create_command_encoder()
    output = presenter._dispatch_viewport_present(viewer, encoder, source, 1, 1, 1, 1, source_is_linear=False)
    device.submit_command_buffer(encoder.finish())
    device.wait()

    image = np.asarray(output.to_numpy(), dtype=np.float32)
    expected = np.array([0.25, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_allclose(image[0, 0, :3], expected, rtol=0.0, atol=1e-5)

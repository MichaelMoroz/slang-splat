from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from train import CameraSample, MCMCConfig, RGBMCMCTrainer, TrainingStepStats, load_colmap_scene

_HISTORY_LIMIT = 512


@dataclass
class TrainingUiConfig:
    scene_path: str = ""
    image_dir: str = "images_8"
    eval_split: bool = True
    llff_hold: int = 8
    preload_cuda: bool = True
    white_background: bool = False
    near: float = 0.0
    far: float = 1000.0
    radius_scale: float = 1.0
    max_anisotropy: float = 12.0
    alpha_cutoff: float = 0.01
    trans_threshold: float = 0.005
    iterations: int = 3000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    lambda_dssim: float = 0.2
    noise_lr: float = 5e5
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    densification_interval: int = 100
    densify_from_iter: int = 500
    densify_until_iter: int = 25000
    cap_max: int = 500000
    random_background: bool = False
    init_points: int = 50000
    init_scale_spacing_ratio: float = 0.25
    init_scale_multiplier: float = 1.0
    init_opacity: float = 0.5
    eval_interval: int = 250
    update_period: int = 1
    seed: int = 0

    def to_mcmc_config(self) -> MCMCConfig:
        return MCMCConfig(
            iterations=int(self.iterations),
            position_lr_init=float(self.position_lr_init),
            position_lr_final=float(self.position_lr_final),
            position_lr_delay_mult=float(self.position_lr_delay_mult),
            position_lr_max_steps=int(self.position_lr_max_steps),
            feature_lr=float(self.feature_lr),
            opacity_lr=float(self.opacity_lr),
            scaling_lr=float(self.scaling_lr),
            rotation_lr=float(self.rotation_lr),
            lambda_dssim=float(self.lambda_dssim),
            noise_lr=float(self.noise_lr),
            opacity_reg=float(self.opacity_reg),
            scale_reg=float(self.scale_reg),
            densification_interval=int(self.densification_interval),
            densify_from_iter=int(self.densify_from_iter),
            densify_until_iter=int(self.densify_until_iter),
            cap_max=int(self.cap_max),
            random_background=bool(self.random_background),
            init_points=int(self.init_points),
            init_scale_spacing_ratio=float(self.init_scale_spacing_ratio),
            init_scale_multiplier=float(self.init_scale_multiplier),
            init_opacity=float(self.init_opacity),
            eval_interval=int(self.eval_interval),
            seed=int(self.seed),
        )


@dataclass(slots=True)
class _FrameMetricBookkeeper:
    loss: np.ndarray
    mse: np.ndarray
    psnr: np.ndarray
    visited: np.ndarray

    @classmethod
    def create(cls, frame_count: int) -> _FrameMetricBookkeeper:
        count = max(int(frame_count), 1)
        return cls(
            loss=np.full((count,), np.nan, dtype=np.float64),
            mse=np.full((count,), np.nan, dtype=np.float64),
            psnr=np.full((count,), np.nan, dtype=np.float64),
            visited=np.zeros((count,), dtype=bool),
        )

    def update(self, frame_index: int, loss: float, mse: float, psnr: float) -> None:
        if frame_index < 0 or frame_index >= int(self.loss.shape[0]):
            return
        self.loss[frame_index] = float(loss)
        self.mse[frame_index] = float(mse)
        self.psnr[frame_index] = float(psnr)
        self.visited[frame_index] = True

    def mean(self, name: str) -> float:
        values = getattr(self, name)
        valid = self.visited & np.isfinite(values)
        return float(np.mean(values[valid], dtype=np.float64)) if np.any(valid) else float("nan")


@dataclass(frozen=True)
class TrainingSnapshot:
    status: str
    error: str
    running: bool
    paused: bool
    done: bool
    scene_path: str
    train_camera_count: int
    test_camera_count: int
    heartbeat: int
    iteration: int
    point_count: int
    preview_count: int
    latest: TrainingStepStats | None
    history: dict[str, list[float]]
    last_mse: float
    last_psnr: float
    avg_loss: float
    avg_mse: float
    avg_psnr: float
    elapsed_seconds: float


def _history_series(history: deque[TrainingStepStats], key: str) -> list[float]:
    values: list[float] = []
    for item in history:
        value = getattr(item, key)
        values.append(float("nan") if value is None else float(value))
    return values


def _safe_rel_path(root: Path, value: Path) -> str:
    try:
        return str(value.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(value.resolve())


class TrainingController:
    def __init__(self) -> None:
        self.config = TrainingUiConfig()
        self._running = False
        self._paused = False
        self._done = False
        self._status = "Idle"
        self._error = ""
        self._train_camera_count = 0
        self._test_camera_count = 0
        self._heartbeat = 0
        self._latest_step: TrainingStepStats | None = None
        self._history: deque[TrainingStepStats] = deque(maxlen=_HISTORY_LIMIT)
        self._latest_preview: torch.Tensor | None = None
        self._preview_dirty = False
        self._pending_fit_points: np.ndarray | None = None
        self._target_image_cache: dict[Path, torch.Tensor] = {}
        self._scene = None
        self._trainer: RGBMCMCTrainer | None = None
        self._pending_start = False
        self._frame_metrics = _FrameMetricBookkeeper.create(1)
        self._last_mse = float("nan")
        self._last_psnr = float("nan")
        self._avg_loss = float("nan")
        self._avg_mse = float("nan")
        self._avg_psnr = float("nan")
        self._elapsed_seconds = 0.0
        self._resume_time: float | None = None

    def _has_colmap_sparse(self, root: Path) -> bool:
        sparse = root / "sparse" / "0"
        return all((sparse / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin"))

    def _resolve_colmap_root(self, selected: Path) -> Path:
        selected = selected.resolve()
        candidates = [selected]
        candidates.extend(path.resolve() for path in sorted(selected.rglob("*")) if path.is_dir())
        for candidate in candidates:
            if self._has_colmap_sparse(candidate):
                return candidate
        raise FileNotFoundError(f"Could not find COLMAP sparse reconstruction under {selected}")

    def _suggest_images_root(self, dataset_root: Path) -> Path:
        dataset_root = dataset_root.resolve()
        candidates = [dataset_root / name for name in ("images_8", "images_4", "images_2", "images", "input")]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()
        for candidate in sorted(dataset_root.rglob("*")):
            if candidate.is_dir() and candidate.name.lower().startswith("images"):
                return candidate.resolve()
        return dataset_root

    def _store_images_root(self, dataset_root: Path, images_root: Path) -> None:
        self.config.image_dir = _safe_rel_path(dataset_root, images_root)

    def _training_elapsed_seconds(self) -> float:
        elapsed = float(self._elapsed_seconds)
        if self._resume_time is not None:
            elapsed += max(time.perf_counter() - self._resume_time, 0.0)
        return elapsed

    def _reset_metrics(self) -> None:
        self._latest_step = None
        self._history.clear()
        self._frame_metrics = _FrameMetricBookkeeper.create(max(self._train_camera_count, 1))
        self._last_mse = float("nan")
        self._last_psnr = float("nan")
        self._avg_loss = float("nan")
        self._avg_mse = float("nan")
        self._avg_psnr = float("nan")

    def _record_step(self, step: TrainingStepStats) -> None:
        self._latest_step = step
        self._history.append(step)
        self._last_mse = float(step.mse)
        self._last_psnr = float(step.train_psnr)
        self._frame_metrics.update(int(step.frame_index), float(step.loss), float(step.mse), float(step.train_psnr))
        self._avg_loss = self._frame_metrics.mean("loss")
        self._avg_mse = self._frame_metrics.mean("mse")
        self._avg_psnr = self._frame_metrics.mean("psnr")

    def _frame(self, frame_index: int) -> CameraSample | None:
        if self._scene is None or not self._scene.train_cameras:
            return None
        idx = min(max(int(frame_index), 0), len(self._scene.train_cameras) - 1)
        return self._scene.train_cameras[idx]

    def set_dataset_folder(self, path: str | Path) -> None:
        root = self._resolve_colmap_root(Path(path))
        self.config.scene_path = str(root)
        self._store_images_root(root, self._suggest_images_root(root))
        self._target_image_cache.clear()

    def set_images_folder(self, path: str | Path) -> None:
        root = Path(self.config.scene_path).resolve() if self.config.scene_path.strip() else Path(path).resolve().parent
        self._store_images_root(root, Path(path))
        self._target_image_cache.clear()

    def shutdown(self) -> None:
        self.stop(wait=False)

    def start(self) -> None:
        if self._running and not self._paused:
            return
        if self._paused:
            self.toggle_pause()
            return
        if self._trainer is not None and self._scene is not None and not self._pending_start:
            self._running = True
            self._paused = False
            self._done = False
            self._error = ""
            self._resume_time = time.perf_counter()
            self._status = "Running"
            return
        self.reinitialize()

    def reinitialize(self) -> None:
        scene_path = self.config.scene_path.strip()
        if not scene_path:
            self._error = "Training scene path is empty."
            self._status = "Idle"
            return
        self.stop(wait=False)
        self._running = True
        self._paused = False
        self._done = False
        self._pending_start = True
        self._status = "Loading scene"
        self._error = ""
        self._train_camera_count = 0
        self._test_camera_count = 0
        self._heartbeat = 0
        self._latest_preview = None
        self._preview_dirty = False
        self._pending_fit_points = None
        self._scene = None
        self._trainer = None
        self._elapsed_seconds = 0.0
        self._resume_time = time.perf_counter()
        self._reset_metrics()
        self._set_status("Loading scene")

    def restart(self) -> None:
        self.reinitialize()

    def stop(self, wait: bool = False) -> None:
        if self._resume_time is not None:
            self._elapsed_seconds += max(time.perf_counter() - self._resume_time, 0.0)
            self._resume_time = None
        self._running = False
        self._paused = False
        self._pending_start = False
        self._trainer = None
        self._target_image_cache.clear()
        self._scene = None
        if not self._done and self._status not in {"Idle", "Error"}:
            self._status = "Stopped"

    def toggle_pause(self) -> None:
        if self._running and not self._done:
            self._paused = not self._paused
            if self._paused:
                if self._resume_time is not None:
                    self._elapsed_seconds += max(time.perf_counter() - self._resume_time, 0.0)
                    self._resume_time = None
            elif self._resume_time is None:
                self._resume_time = time.perf_counter()
            self._status = "Paused" if self._paused else "Running"

    def consume_preview(self) -> torch.Tensor | None:
        if not self._preview_dirty or self._latest_preview is None:
            return None
        self._preview_dirty = False
        return self._latest_preview

    def consume_fit_points(self) -> np.ndarray | None:
        points = self._pending_fit_points
        self._pending_fit_points = None
        return points

    def frame_count(self) -> int:
        return 0 if self._scene is None else len(self._scene.train_cameras)

    def frame_name(self, frame_index: int) -> str:
        frame = self._frame(frame_index)
        return "<none>" if frame is None else frame.image_path.name

    def camera_sample(self, frame_index: int) -> CameraSample | None:
        return self._frame(frame_index)

    def target_image(self, frame_index: int) -> torch.Tensor | None:
        frame = self._frame(frame_index)
        if frame is None:
            return None
        if self._trainer is not None:
            return self._trainer._target_image(frame)
        image = frame.image
        if image is None:
            image = self._target_image_cache.get(frame.image_path)
            if image is None:
                image = _load_image_tensor(frame.image_path, frame.camera_params.device, preload_cuda=True)
                self._target_image_cache[frame.image_path] = image
        return _image_to_linear_float(image) if image.dtype == torch.uint8 else image

    def snapshot(self) -> TrainingSnapshot:
        history = {
            "loss": [item.loss for item in self._history],
            "l1": [item.l1 for item in self._history],
            "psnr": [item.train_psnr for item in self._history],
            "mse": [item.mse for item in self._history],
            "eval_psnr": _history_series(self._history, "test_psnr"),
            "ssim": _history_series(self._history, "test_ssim"),
            "points": [float(item.point_count) for item in self._history],
            "elapsed_ms": [item.elapsed_ms for item in self._history],
            "xyz_lr": [item.xyz_lr for item in self._history],
        }
        return TrainingSnapshot(
            status=self._status,
            error=self._error,
            running=self._running,
            paused=self._paused,
            done=self._done,
            scene_path=self.config.scene_path,
            train_camera_count=self._train_camera_count,
            test_camera_count=self._test_camera_count,
            heartbeat=self._heartbeat,
            iteration=0 if self._latest_step is None else self._latest_step.iteration,
            point_count=0 if self._latest_step is None else self._latest_step.point_count,
            preview_count=0 if self._latest_preview is None else int(self._latest_preview.shape[1]),
            latest=self._latest_step,
            history=history,
            last_mse=float(self._last_mse),
            last_psnr=float(self._last_psnr),
            avg_loss=float(self._avg_loss),
            avg_mse=float(self._avg_mse),
            avg_psnr=float(self._avg_psnr),
            elapsed_seconds=self._training_elapsed_seconds(),
        )

    def _apply_context_settings(self, trainer: RGBMCMCTrainer) -> None:
        trainer.context.radius_scale = float(self.config.radius_scale)
        trainer.context.max_anisotropy = float(self.config.max_anisotropy)
        trainer.context.alpha_cutoff = float(self.config.alpha_cutoff)
        trainer.context.trans_threshold = float(self.config.trans_threshold)

    def _set_status(self, value: str) -> None:
        self._status = value
        self._heartbeat += 1

    def _capture_preview(self) -> None:
        if self._trainer is None:
            return
        self._latest_preview = self._trainer.snapshot_splats().detach().contiguous()
        self._preview_dirty = True

    def update(self) -> None:
        if not self._running or self._paused or self._done:
            return
        try:
            if self._pending_start:
                scene = load_colmap_scene(
                    self.config.scene_path,
                    image_dir=self.config.image_dir,
                    eval_split=bool(self.config.eval_split),
                    llff_hold=int(self.config.llff_hold),
                    preload_cuda=bool(self.config.preload_cuda),
                    device="cuda",
                    near=float(self.config.near),
                    far=float(self.config.far),
                    white_background=bool(self.config.white_background),
                )
                self._scene = scene
                self._train_camera_count = len(scene.train_cameras)
                self._test_camera_count = len(scene.test_cameras)
                self._frame_metrics = _FrameMetricBookkeeper.create(max(self._train_camera_count, 1))
                self._set_status("Initializing trainer")
                trainer = RGBMCMCTrainer(self.config.to_mcmc_config(), device="cuda")
                self._apply_context_settings(trainer)
                trainer.start(scene, status=self._set_status)
                self._trainer = trainer
                self._pending_fit_points = scene.point_xyz.copy()
                self._capture_preview()
                self._pending_start = False
                self._status = "Running"
                return
            if self._trainer is None:
                self._running = False
                self._done = True
                self._status = "Completed"
                if self._resume_time is not None:
                    self._elapsed_seconds += max(time.perf_counter() - self._resume_time, 0.0)
                    self._resume_time = None
                return
            self._apply_context_settings(self._trainer)
            self._set_status(f"Running step {self._trainer.iteration + 1}")
            step = self._trainer.step()
            self._record_step(step)
            update_period = max(int(self.config.update_period), 1)
            if step.iteration == 1 or step.iteration % update_period == 0:
                self._capture_preview()
            self._status = "Running"
        except Exception as exc:
            if self._resume_time is not None:
                self._elapsed_seconds += max(time.perf_counter() - self._resume_time, 0.0)
                self._resume_time = None
            self._running = False
            self._done = False
            self._status = "Error"
            self._error = str(exc)
            self._heartbeat += 1

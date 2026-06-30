# Headless Automation

Run the full import / train / evaluate / render / export pipeline without a window:

```powershell
python viewer.py --headless --config configs\garden_train.json
```

`--headless` requires `--config`. The headless run and the interactive viewer share the same `session`/`presenter` code and the same configuration tree, so behavior matches the GUI. Implementation lives in [`src/viewer/headless.py`](../src/viewer/headless.py).

## Configuration and layering

- The base configuration `config/defaults.json` is **always loaded first**.
- `--config <path>` is deep-merged on top, overriding only the keys it defines (`load_config` in [`src/repo_defaults.py`](../src/repo_defaults.py)).
- So a run config usually contains **only a `viewer.run` block** and inherits every training / renderer / import setting from the base. Everything the GUI's importer and training panels expose (init sources, rotation, residency, downscale, photometric on/off, training hyperparameters, …) stays in its normal section (`viewer.import`, `training_build_args`, `renderer`) and is read exactly as in the GUI.
- Named configs live in `configs/*.json`. See [`configs/my_run.json`](../configs/my_run.json) and [`configs/garden_train.json`](../configs/garden_train.json).

## The run pipeline

`viewer.run` drives an ordered pipeline; each stage runs only when its inputs are present. Stages execute in this order:

1. **source** — load the scene/dataset
2. **photometric** — optional pre-training PPISP calibration
3. **reinitialize** — optional re-seed of the gaussians
4. **train** — optional, `train_iters` steps
5. **metrics** — optional dataset evaluation report
6. **render** — optional snapshot images
7. **export** — optional PLY write

Training is optional: `train_iters: 0` runs everything else. Examples: `colmap → metrics` (evaluate the initialized scene), `colmap → photometric → metrics` (no gaussian training), `colmap → train → metrics → export`, `ply → render`.

## `viewer.run` keys

| Key | Type | Meaning |
|---|---|---|
| `source` | `"colmap"` \| `"ply"` | Scene source. `colmap` builds dataset + trainer; `ply` loads a scene only (render/export only — no frames for metrics). |
| `ply_path` | path | PLY to load when `source: "ply"`. |
| `colmap_root_path` | path | COLMAP dataset root. |
| `colmap_images_root` | path | RGB image folder. |
| `colmap_depth_root` / `colmap_alpha_mask_root` | path | Optional depth / alpha-mask folders. |
| `colmap_custom_ply_path` / `colmap_custom_mesh_path` | path | Optional custom init seeds. |
| `colmap_selected_camera_ids` | list[int] | Camera models to include (empty = all). |
| `seed` | int | Training / init seed. |
| `photometric` | `{ "steps": N }` | Standalone PPISP calibration before training; runs only if `steps > 0`. (In-import photometric is controlled by `viewer.import.colmap_photometric_compensation_enabled`.) Requires a COLMAP source. |
| `reinitialize` | bool | Re-seed gaussians from the configured sources before training. |
| `train_iters` | int | Training steps. `0` (or omitted) skips the training stage. |
| `metrics` | `{ "output": path, "scene_ply": path }` | Present (non-null) runs the dataset-metrics stage. **Requires `source: "colmap"`** (needs frames + trainer, with training/photometric/import idle). `output` is the report path; `scene_ply` scores an external PLY by swapping it into the trainer. |
| `render` | `{ "views": [...], "width": N, "height": N, "output_dir": path }` | Final render snapshots (see Views below). Defaults: `views: ["rendered"]`, `1280x720`, `outputs/headless/renders`. |
| `output_ply` | path | Export the trained/loaded scene. SH inclusion follows the training `use_sh` setting. |
| `stats` | `{ "log_interval": N, "output": csv, "per_view_output": csv }` | Training stats. `log_interval` controls stdout cadence; `output` writes a per-step CSV; `per_view_output` writes one row per training view at the end. |
| `schedule` | list | Point-in-time actions during training (see below). |

## Scheduled actions (`viewer.run.schedule`)

Recurring or point-in-time actions are a list, each keyed by training step:

```jsonc
"schedule": [
  { "at": 1000,        "do": "capture_renderdoc", "output_dir": "outputs/headless/rdc" },
  { "at": [500, 2500], "do": "capture_python" },
  { "at": 2000,        "do": "capture_buffers" },
  { "every": 5000,     "do": "render", "views": ["rendered", "abs_diff"], "output_dir": "outputs/headless/snaps" }
]
```

- `at`: a step or list of steps. `every`: an interval (expands to every N steps up to `train_iters`). Steps are clamped to `[0, train_iters]`, and the training loop splits batches so it lands exactly on each scheduled step.
- `do` values:
  - `capture_python` / `capture_renderdoc` — wrap that step's GPU submission in a Python / RenderDoc capture.
  - `capture_buffers` — write a resource/allocation log (via `collect_resource_debug_snapshot`); not a raw GPU-memory dump.
  - `render` — render snapshots at that step (same options as `viewer.run.render`).
- Output directory resolves from the action's `output_dir`, else `viewer.run.capture_output_dir`, else `outputs/headless/<kind>`.

## Views

`render.views` (and scheduled `render` actions) accept:

- `render` / `main` / `plain` — plain rasterized view (needs a renderer + camera).
- Loss-debug views — `rendered`, `target`, `abs_diff`, `dssim`, `rendered_edges`, `target_edges` (need a trainer + training frames).

## Notes and current limitations

- The metrics stage requires a COLMAP source; `source: "ply"` clears the dataset frames, so it supports render/export only.
- `viewer.run.edit` exists in the schema but is **not yet executed** by the headless pipeline.
- There is no in-GUI config dropdown / "Save Config As"; author configs as files. The GUI's **Update Defaults** writes the current control state (including the `viewer.run` value keys) back into `config/defaults.json`.
- Graphics API follows `viewer.ui.graphics_api` (`vulkan` default; `dx12`/`d3d12` supported), or the `--graphics-api` command-line flag when given.

See [`doc/HeadlessAutomation.md`](HeadlessAutomation.md) for the original design rationale and the full Phase-0 audit.

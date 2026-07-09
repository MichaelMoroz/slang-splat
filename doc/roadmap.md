# Roadmap: Reconstruction, Projects, and Scene Hierarchy

Three larger features that build on the current viewer/trainer architecture. Each
section states the goal, the concrete code it touches, a proposed data model, and a
staged implementation path. Nothing here is committed to yet — this is a design plan.

Shared context for all three:

- The renderer holds the scene on the GPU as a set of double-buffered structured
  buffers keyed by name (`splat_params`, `splat_age`, `splat_init`, Adam moments).
  See `GaussianRenderer.scene_buffers` and the role machinery in
  [session.py](../src/viewer/session.py) (`_MAIN_RENDERER_ROLE`,
  `_TRAINING_RENDERER_ROLE`, `_DEBUG_RENDERER_ROLE`). `trainer.renderer` **is**
  `viewer.s.training_renderer` — they share one scene buffer set.
- `GaussianScene` ([src/scene/gaussian_scene.py](../src/scene/gaussian_scene.py)) is
  the CPU-side container: `positions, scales, rotations, opacities, colors,
  sh_coeffs, refinable`, with `.subset(mask)`.
- `GaussianTrainer` ([src/training/gaussian_trainer.py](../src/training/gaussian_trainer.py))
  optimizes exactly one scene. `TrainingHyperParams` (line 285) already carries
  per-parameter LR multipliers (`lr_pos_mul`, `lr_scale_mul`, `lr_rot_mul`,
  `lr_color_mul`, `lr_opacity_mul`, `lr_sh_mul`, plus per-stage variants).
- Per-splat "refinable" is already encoded: the trainer packs it into the **sign**
  of the init radius in `splat_init[:, 3]` (positive = refinable, negative = frozen
  from clone/prune) — see `gaussian_trainer.py:1450-1460` and the readback at
  `:2374-2385`. Non-refinable splats are still optimized but never cloned/pruned.
- COLMAP data enters through `load_colmap_reconstruction`
  ([colmap_binary.py:239](../src/scene/_internal/colmap_binary.py)) into
  `ColmapReconstruction` ([colmap_types.py](../src/scene/_internal/colmap_types.py)),
  then `build_training_frames_from_root`
  ([colmap_ops.py](../src/scene/_internal/colmap_ops.py)) turns images into training
  frames. Import is a streaming state machine in `session.py`
  (`advance_colmap_import`: prepare → scan_frames → load_textures).

---

## 1. pyCOLMAP integration — import video/images and reconstruct poses in-viewer

### Goal

Let the user point the importer at a folder of images **or a video file** with no
precomputed COLMAP output, and run Structure-from-Motion inside the viewer to produce
camera poses + a sparse point cloud, which then flows into the existing training path.
Today the importer requires a `sparse/` reconstruction to already exist on disk.

### What exists to build on

- The whole downstream pipeline already consumes a `ColmapReconstruction`
  (`cameras`, `images` with `q_wxyz`/`t_xyz`/`points2d_*`, `points3d`). If the new SfM
  step emits that structure (or a standard COLMAP `sparse/0` directory), **everything
  after it is unchanged**: `build_training_frames_from_root`,
  `transform_colmap_reconstruction_pca`, `point_nn_scales`, camera-subset selection
  (`select_wide_coverage_image_ids`), and the VRAM estimate all keep working.
- The import UI + streaming state machine (`advance_colmap_import`) already shows a
  multi-stage progress bar. A "reconstructing" stage slots in ahead of `scan_frames`.

### Proposed data model / flow

```
video.mp4 ─┐
           ├─► frame extraction (ffmpeg) ─► images/*.jpg ─┐
images/*  ─┘                                              │
                                                          ▼
                                        pycolmap SfM (feature → match → mapper)
                                                          │
                                                          ▼
                                   sparse/0  (cameras.bin, images.bin, points3D.bin)
                                                          │
                                                          ▼
                                   existing load_colmap_reconstruction(...) path
```

### Implementation details

1. **Dependency**: add `pycolmap` (optional extra, e.g. `[reconstruct]`) plus a
   runtime check. `pycolmap` bundles COLMAP's SfM; GPU SIFT needs CUDA but there is a
   CPU fallback. Also require `ffmpeg` on PATH for video (shell out — do not add a
   heavy python video dep). Guard all imports so the base viewer still runs without
   the extra installed (mirror how optional paths are already handled).

2. **New module** `src/scene/_internal/colmap_reconstruct.py`:
   - `extract_frames_from_video(video_path, out_dir, fps=None, max_frames=None) ->
     list[Path]` — ffmpeg wrapper. Expose an FPS / max-frame cap so long clips don't
     explode into thousands of frames. Optionally a blur-rejection pass (variance of
     Laplacian) to drop motion-blurred frames before matching.
   - `run_sfm(image_dir, work_dir, *, camera_model="OPENCV", matcher="sequential",
     progress=cb) -> Path` — drive `pycolmap.extract_features`,
     `pycolmap.match_*` (sequential matcher is right for video; exhaustive/vocab-tree
     for unordered photos), then `pycolmap.incremental_mapping`. Returns the
     `sparse/0` path. Emit progress through a callback so the UI stage bar can move.
   - Reuse `load_colmap_reconstruction` on the produced directory rather than
     converting `pycolmap` objects directly — keeps one canonical loader and one set
     of coordinate conventions.

3. **State machine**: add a `reconstruct` stage to `ColmapImportProgress` /
   `advance_colmap_import` in [session.py](../src/viewer/session.py) and
   [state.py](../src/viewer/state.py), sequenced **before** `scan_frames`. Because SfM
   is long-running and blocking, run it on the existing worker/thread the importer
   already uses for streaming, and surface coarse sub-progress (feature extraction %,
   pairs matched, images registered). Cache results: if `work_dir/sparse/0` already
   exists and inputs are unchanged, skip straight to loading.

4. **UI** ([ui.py](../src/viewer/ui.py) import window): add a source toggle
   `Precomputed COLMAP | Images | Video`. For images/video, expose matcher mode,
   camera model, target FPS (video), and max frames. Persist these as new
   `viewer.import` config keys (see repo_defaults defaults machinery). The existing
   camera-model dropdown and pose-subset slider apply unchanged once the recon exists.

5. **Coordinate + intrinsics sanity**: SfM output uses COLMAP's world convention, which
   the current loader/`transform_colmap_reconstruction_pca` already normalizes, so no
   new alignment code — but add a regression test that a tiny synthetic image set
   produces a reconstruction the existing frame-builder accepts.

### Risks / open questions

- Reconstruction quality is dataset-dependent; failed/partial maps need a clear error
  path (e.g. "only 12 of 200 images registered — try exhaustive matching").
- GPU contention: SfM SIFT + the live renderer both want the GPU. Consider forcing CPU
  features while a training/render context is active, or pausing rendering.
- Packaging size: `pycolmap` wheels are large — keep it an opt-in extra.

### Suggested staging

1. ffmpeg frame extraction + images→SfM→`sparse/0`, loaded through the existing path
   (no UI yet; drive from a headless flag).
2. Import-window UI + progress stage.
3. Matcher/camera-model options, caching, blur rejection, error surfaces.

---

## 2. Splat project support — save/load a project (settings + dataset + result)

### Goal

Persist everything needed to reopen a working session: current settings, the dataset
reference, the trained splats, and edit state — as a single project the user can save
and reload, instead of re-importing and re-configuring each time.

### What exists to build on

The config system is already 80% of a project format:

- [config.py](../src/viewer/config.py): `RUN_UI_VALUE_FIELDS` (lines 9-23) already
  captures `source`, all COLMAP/PLY paths, `colmap_selected_camera_ids`, `seed`,
  `train_iters`, `output_ply`. `apply_config_overlay(values, overlay)` restores a saved
  config into the live UI, and `exported_run_values(values, existing_run)` writes the
  current UI back out — **preserving** the `schedule`, `stats`, `photometric`,
  `metrics`, `render`, and `edit` sub-sections it doesn't own.
- [repo_defaults.py](../src/repo_defaults.py): `load_json`, `deep_merge_config`,
  `defaults_path`, `configs_path`, `list_configs` — the layered load/merge machinery.
  A project is "defaults ⊕ project overlay".
- Result export already exists: `save_gaussian_ply`
  ([ply_loader.py:159](../src/scene/ply_loader.py)) and
  `_export_source_scene` ([app.py:577](../src/viewer/app.py)).

So a "project" = the existing run/config JSON + a pointer to (or copy of) the dataset +
the trained PLY + a manifest. The main new work is **bundling** and **relinking**.

### Proposed format

A project directory (or zip) `name.splatproj/`:

```
name.splatproj/
  project.json        # manifest: schema version, created/modified, app version
                      #   ├─ config: full exported config overlay (reuse exported_run_values)
                      #   ├─ dataset: { mode: "reference"|"embedded", root, hash }
                      #   └─ result:  { ply: "scene.ply", iterations, splat_count }
  scene.ply           # trained splats (save_gaussian_ply, include_sh=True)
  edit_state.json     # optional: selection/edit history if worth persisting
  thumbnail.png       # optional: last viewport for a project browser
```

Two dataset modes:

- **reference** (default): store absolute + project-relative paths and a content hash;
  on load, verify and offer relink if the dataset moved. Keeps projects small.
- **embedded**: copy the dataset (or just the selected/subsampled frames) inside the
  bundle for portability. Large — opt-in.

### Implementation details

1. **New module** `src/viewer/project.py`:
   - `save_project(path, values, scene, *, dataset_mode, edit_state=None)`:
     - `config = exported_run_values(values, existing_run)` (reuse as-is — do not
       reinvent field capture).
     - Write `save_gaussian_ply(path/"scene.ply", scene)`.
     - Compute a dataset hash (cheap: file list + sizes + mtimes, or sampled content).
     - Emit `project.json`.
   - `load_project(path) -> LoadedProject(config_overlay, ply_path, dataset_info,
     edit_state)`. Caller then runs `apply_config_overlay(values, overlay)`, points the
     PLY source at `scene.ply`, and restores edit state.
   - `migrate_project(manifest)` keyed on `schema_version` for forward-compat.

2. **Relinking**: if `dataset.mode == reference` and the root is missing, fall back to
   the project-relative path, then prompt. Reuse the existing path-field normalization
   in `config.py` (`_PATH_VALUE_FIELDS`, lines 25-37) so saved paths round-trip.

3. **UI** ([ui.py](../src/viewer/ui.py) / [app.py](../src/viewer/app.py)):
   File menu `Save Project` / `Save Project As…` / `Open Project…` / recent projects.
   `Save` overwrites the current `.splatproj`; keep a dirty flag so quitting with
   unsaved changes warns. A later "project browser" can use the thumbnails.

4. **Headless parity**: the headless runner already accepts `--config`. Let it accept a
   `.splatproj` (load its `config` section) so a project trained in the GUI can be
   re-run/continued headless. This keeps one config schema across GUI and CLI.

5. **Versioning**: stamp `app_version` + `schema_version`. `deep_merge_config` over
   current defaults means projects saved before a new setting was added still load
   (the new key takes its default) — verify this holds and add a round-trip test.

### Interaction with feature 3

When the scene becomes a hierarchy (below), `project.json` grows a `scene_graph`
section and `result` becomes a list of per-object PLYs + their per-object training
settings. Design `project.json` with that in mind now (a `scene` object rather than a
flat single-PLY field) so the format doesn't break later.

### Suggested staging

1. Save/load config + single trained PLY, reference dataset mode (covers the common
   case).
2. Relink flow + dirty/unsaved warnings + recent projects.
3. Embedded dataset mode, thumbnails, headless `.splatproj`, edit-state persistence.

---

## 3. Scene hierarchy — splat objects, camera-pose objects, per-object training

### Goal

Turn the implicit single-scene model into an explicit scene graph:

- **Splat objects**: multiple named splat sets in one scene, each with its own
  transform and its own training settings.
- **Camera-pose objects**: camera poses become first-class, selectable/inspectable
  objects rather than an opaque frame list.
- **Per-object training config**: e.g. a **frozen background** object that only trains
  color (SH), plus a **foreground** object trained fully. Enables
  low-VRAM/composited workflows (a static captured background + an editable subject).

### What exists to build on

- Per-splat `refinable` already gates clone/prune per splat (sign of
  `splat_init[:, 3]`). Per-object "what is trainable" is a **generalization** of this:
  instead of one bit, a small per-splat bitmask over {position, scale, rotation,
  opacity, color/SH}.
- `TrainingHyperParams` already has per-parameter LR multipliers. "Train color only"
  is conceptually "position/scale/rotation/opacity LR = 0 for these splats."
- The renderer already supports multiple **roles** (main/training/debug) over shared
  buffers — the plumbing for "more than one logical scene view" exists.
- `GaussianScene.subset()` already splits/rejoins splat arrays.

### Proposed data model

Introduce a lightweight scene graph on the CPU side; the GPU stays a single flat buffer
for render/optimize performance, tagged by object id.

```
SceneGraph
 ├─ objects: list[SceneObject]
 │    ├─ SplatObject { id, name, transform(SRT), visible,
 │    │                scene: GaussianScene subset,
 │    │                training: ObjectTrainingConfig }
 │    └─ CameraObject { id, name, pose(q,t), intrinsics, frame_ref }
 └─ active_object_id

ObjectTrainingConfig
 ├─ enabled: bool                     # participates in training at all
 ├─ trainable_mask: {pos, scale, rot, opacity, sh}  # per-param freeze
 ├─ allow_refine: bool                # clone/prune allowed (maps to existing refinable)
 └─ hparams_override: partial TrainingHyperParams  # optional per-object LR/reg
```

**Frozen-background example**: `trainable_mask = {sh: True, everything else: False}`,
`allow_refine = False`. **Foreground**: all-True, `allow_refine = True`.

### GPU representation

Add a per-splat `object_id` (uint) and a per-splat `trainable_mask` (packed bits),
uploaded alongside `splat_params`. Two viable layouts:

- **Single concatenated buffer** (recommended): all objects live in one
  `splat_params` buffer with an `object_id` channel. Refinement (clone/prune) must keep
  splats grouped or carry `object_id` through compaction so a cloned splat inherits its
  parent's object. The mask multiplies the gradient/step per param — the optimizer
  already computes per-param settings in
  [optimizer.py](../src/training/optimizer.py) `_param_settings`; extend the Adam apply
  kernel to read the per-splat mask and zero frozen-param updates (cleaner than LR=0
  because it also skips momentum). This is the smallest change to the hot loop.
- **Separate buffers per object**: cleaner isolation, but the sort/raster/refine
  passes would need to iterate objects or merge before rendering — more invasive. Only
  worth it if objects need independent refinement cadence.

### Implementation details

1. **Extend `GaussianScene`** with optional `object_id` and `trainable_mask` arrays
   (default: single object, all-trainable — preserves current behavior). Update
   `.subset()` to carry them.

2. **Trainer** ([gaussian_trainer.py](../src/training/gaussian_trainer.py)):
   - Upload `trainable_mask` as a scene buffer next to the existing `splat_init`.
   - In the Adam/step kernel, gate each parameter block by the mask bit (extends the
     existing per-param-settings path; frozen params get zero step **and** are skipped
     for momentum/regularizers like `init_position_reg_weight`).
   - Generalize the refinable handling: `allow_refine=False` reuses the existing
     negative-radius encoding; clone/prune already respects it.
   - Per-object `hparams_override`: fold into the per-param settings upload so a splat's
     effective LR is `base * object_override`.

3. **Renderer**: add `object_id` channel; unchanged for basic compositing (one sort
   over all splats). Per-object visibility toggles by writing opacity→0 or excluding by
   id at sort time.

4. **Camera objects**: wrap the existing training `frames` as `CameraObject`s so the UI
   can list/select/preview them (feeds naturally from feature 1's reconstruction and
   from `select_wide_coverage_image_ids`). Mostly a view over existing data — no change
   to how frames drive training.

5. **UI**: a scene-hierarchy panel (tree of objects) with per-object visibility, active
   selection, transform, and a training sub-panel exposing `trainable_mask` /
   `allow_refine` / LR overrides. The existing splat editor's selection tooling becomes
   "operate on the active object."

6. **Ingest / merge**: "add splat object from PLY/COLMAP" imports into a new object
   rather than replacing the scene — `GaussianScene` concat with a fresh `object_id`.
   "Freeze as background" is a one-click preset that sets the frozen-color mask.

### Risks / open questions

- **Refinement + object_id bookkeeping**: clone/prune compaction must preserve
  `object_id` and `trainable_mask` per splat. This is the highest-risk piece — the
  refinement kernels currently assume a homogeneous scene. Land the mask plumbing and
  test it on a single object before enabling multiple.
- **Transforms**: if `SplatObject.transform` is non-identity, either bake it into splat
  positions/rotations at edit time (simpler, matches current world-space model) or
  apply per-object transforms in the render/optimize path (more flexible, more work).
  Recommend bake-on-commit first.
- **Memory**: two extra per-splat channels (`object_id` u32, `trainable_mask` u8) are
  cheap relative to `splat_params`.

### Suggested staging

1. Per-splat `trainable_mask` plumbed through scene → trainer → Adam kernel, single
   object, verified against current behavior when mask is all-true.
2. Multiple objects in one concatenated buffer with `object_id`; "freeze as background
   (color-only)" preset; hierarchy panel with visibility + active selection.
3. Camera-pose objects in the panel; per-object LR/reg overrides; PLY/COLMAP "add as
   object" ingest; per-object transforms.

---

## Cross-feature ordering

The three features reinforce each other, and there is a natural build order:

1. **pyCOLMAP (1)** removes the external-tool prerequisite and produces camera-pose
   objects that feature 3 will surface.
2. **Projects (2)** is the smallest lift (the config system already captures most of a
   run) and immediately improves daily workflow — do it early, but design
   `project.json` with a `scene` object so it survives feature 3.
3. **Scene hierarchy (3)** is the largest and riskiest (touches the refinement hot
   loop); land the per-splat `trainable_mask` foundation first, single-object, before
   multi-object.

# Splat Project Deep Dive — Save/Load a Project (Settings + Dataset + Result)

Design investigation expanding [roadmap.md §2](roadmap.md). Sequenced **before** the
scene hierarchy ([SceneHierarchyDesign.md](SceneHierarchyDesign.md)); the project
format is designed so feature 3 extends it without a schema break. File/line anchors
refer to the `streaming` branch as of July 2026.

> **Status (July 2026): stage 1 is implemented** in
> [`src/viewer/project.py`](../src/viewer/project.py) (manifest capture/save/open,
> mandatory creation on import, step-based autosave, headless `.splatproj` support,
> relink + collision modals), with hooks in `session._finish_import_colmap_dataset`
> (import completion), `session.set_training_active` (stop-boundary save), and
> `presenter_state._run_training_batch` (autosave safe point). Tests:
> [`tests/test_viewer_project.py`](../tests/test_viewer_project.py). `Start fresh`
> archives a stale `scene.ply` to `scene.ply.bak` (resolves the §12 open question
> as archive-by-rename). Stage 2/3 items remain open.

1. [What the code actually provides (findings vs. the roadmap sketch)](#1-what-the-code-actually-provides)
2. [Project format](#2-project-format)
3. [What the config snapshot contains — and what it deliberately omits](#3-the-config-snapshot)
4. [Save flow, creation on import, and autosave](#4-save-flow-creation-on-import-and-autosave)
5. [Open flow (GUI and headless)](#5-open-flow)
6. [Resume semantics](#6-resume-semantics)
7. [Relinking and dataset modes](#7-relinking-and-dataset-modes)
8. [Dirty tracking and UI](#8-dirty-tracking-and-ui)
9. [Versioning and migration](#9-versioning-and-migration)
10. [Forward compatibility with the scene hierarchy](#10-forward-compatibility-with-the-scene-hierarchy)
11. [Staging plan](#11-staging-plan)
12. [Risks and open questions](#12-risks-and-open-questions)

---

## 1. What the code actually provides

The roadmap says the config system is "80 % of a project format". The investigation
supports that — and found the missing 20 % is smaller than sketched, because several
pieces the roadmap listed as new work already exist:

**a) Full-state capture already exists — it's the `Update Defaults` path, not just
`RUN_UI_VALUE_FIELDS`.** `_save_defaults_callback`
([app.py:895-915](../src/viewer/app.py)) assembles the complete stable UI state from
three existing exporters:

- `training_build_args` via the `_TRAINING_PARAM_KEYS` control-key mapping (all 164
  training knobs in `config/defaults.json`),
- `renderer` + `viewer.controls` / `viewer.import` / `viewer.ui` via
  `export_repo_defaults_from_ui_values` ([ui.py:653](../src/viewer/ui.py)),
- `viewer.run` via `exported_run_values` ([config.py:159](../src/viewer/config.py)),
  which additionally **preserves** the headless automation keys (`schedule`, `stats`,
  `photometric`, `metrics`, `render`, `edit`) from an existing run section.

A project's config snapshot is these three calls writing to `project.json` instead of
`config/defaults.json`. No new capture code.

**b) Full-state restore already exists — but only headless uses it.**
`apply_config_overlay(values, overlay)` ([config.py:140](../src/viewer/config.py))
restores training args, renderer settings, controls, import settings, and run fields
into a live `ui._values` dict; `HeadlessViewer.__init__` calls it
([headless.py:118](../src/viewer/headless.py)). The GUI currently has **no**
config-apply path at all — `--config` in GUI mode is parsed but unused ("reserved for
config preselection", [app.py:1025](../src/viewer/app.py)). `Open Project` is that
missing GUI apply path plus a source trigger (§5).

**c) Resume machinery already exists, with documented semantics.**
`_apply_training_resume` ([headless.py](../src/viewer/headless.py), `resume_ply` +
`train_start_step`) swaps a previously exported scene into the trainer
(`trainer.replace_scene`) and fast-forwards `trainer.state.step` so every staged
hyperparameter resumes mid-schedule. Its docstring already states the approximation
contract: *not bit-identical — Adam moments, splat ages, contribution history, and
init anchors restart from the checkpoint scene*. Project reopen adopts exactly this
mechanism and this contract (§6).

**d) The source stage is already factored for both entry points.** Headless
`_run_source_stage` drives the same `session.import_colmap_from_ui` /
`advance_colmap_import` streaming state machine the GUI import window uses, and
`import_colmap_from_ui` reads **everything from `ui._values`**
([session.py](../src/viewer/session.py), `import_colmap_from_ui`) — so once the
overlay is applied to the UI values, triggering a faithful re-import is one call. The
GUI additionally gets the progress window for free because the state machine is
already frame-pumped.

**e) Native file dialogs exist.** `spy.platform.open_file_dialog` /
`save_file_dialog` / `choose_folder_dialog` are used throughout
([app.py:716-766](../src/viewer/app.py)).

**f) Path handling today is repo-root-relative.** Run configs like
[configs/my_run.json](../configs/my_run.json) store `dataset/garden` resolved against
the working directory; `_PATH_VALUE_FIELDS` round-trips them as plain strings and
`import_colmap_from_ui` only applies `expanduser`. Projects introduce the first
*portable* path requirement → the relink design in §7.

**g) `Reload` already reconstructs an import from stored state** — the
`_reload_callback` ([app.py:770](../src/viewer/app.py)) re-imports from
`viewer.s.colmap_import` (`ColmapImportSettings`,
[state.py:46](../src/viewer/state.py)). Project open is deliberately **not** built on
this path: the UI-values path (d) is the one both entry points share and the one the
overlay already targets.

---

## 2. Project format

A project is a **directory**. Every COLMAP import creates one automatically at
`<colmap_root>/slang_splat/` (§4); user-placed projects (`Save As…`) use the
`name.splatproj/` naming convention. The directory name is convention only — the
Open dialog targets `project.json`:

```
<colmap_root>/slang_splat/    # default (§4); or name.splatproj/ anywhere via Save As…
  project.json        # manifest (below) — the file the Open dialog targets
  scene.ply           # latest saved splats (save_gaussian_ply, include_sh honoring SH band)
                      # — also the single autosave slot, overwritten every 15 000 steps (§4)
  thumbnail.png       # last viewport frame, for recents/browser (stage 2)
  edit_state.json     # optional: splat-editor box/ranges + selection bitmask ref (stage 3)
  selection.bin       # optional: 1 bit/splat packed selection (stage 3)
```

`project.json`:

```jsonc
{
  "format": "splatproj",
  "schema_version": 1,
  "app_version": "<git describe>",
  "created": "2026-07-09T12:00:00Z",
  "modified": "2026-07-09T14:32:11Z",

  "config": {
    // exactly the defaults.json shape, as an overlay:
    "training_build_args": { ... },
    "renderer": { ... },
    "viewer": { "controls": { ... }, "import": { ... }, "run": { ... } }
    // note: viewer.ui and viewer.state are machine-local and NOT saved (§3)
  },

  "dataset": {
    "mode": "reference",              // "reference" | "embedded" (stage 3)
    "colmap_root_abs": "C:/data/garden",
    "colmap_root_rel": "../data/garden",   // relative to the project dir (".." for default-location projects, §4)
    "images_root_abs": "C:/data/garden/images_4",
    "images_root_rel": "../data/garden/images_4",
    "fingerprint": { "files": 312, "bytes": 812345678, "newest_mtime": 1750000000,
                      "sample": "sha1 of sorted (name,size) list" }
  },

  "scene": {                          // object list — single entry until feature 3
    "objects": [
      { "id": 0, "name": "scene", "ply": "scene.ply",
        "training": null }            // feature 3 fills preset/mask here
    ]
  },

  "result": {
    "step": 30000,                    // trainer.state.step at save
    "train_elapsed_s": 5423.0,
    "splat_count": 5214321,
    "sh_band": 3
  }
}
```

Decisions:

- **Directory, not zip.** Incremental saves rewrite only what changed (`project.json`
  is tiny; `scene.ply` is the big rewrite either way), the PLY stays directly usable
  by external tools, and a zip variant can be added later as an export. The Open
  dialog targets `project.json` (filter `project.json;*.splatproj`) because
  `spy.platform.open_file_dialog` picks files; `choose_folder_dialog` is the fallback
  affordance ("Open Project Folder…" is unnecessary if the filter is right).
- **`config` is a plain defaults-shaped overlay**, not a new schema. Loading is
  `deep_merge_config(load_defaults(), project.config)` →
  `apply_config_overlay` — identical to the headless path, so one code path serves
  GUI open, headless open, and forward-compat (§9).
- **`scene.objects` is a list from day one** with a single entry, exactly what
  [SceneHierarchyDesign.md §10](SceneHierarchyDesign.md) expects to extend
  (per-object PLYs + `training` presets). No schema break when feature 3 lands.
- **`dataset` stores both absolute and project-relative roots** plus a cheap
  fingerprint (§7). Only the two roots need relinking; every other path in
  `viewer.run` / `viewer.import` (depth root, alpha masks, custom PLY/mesh) is stored
  as saved but *validated* on open with the same relink prompt when missing.
- **Default location inside the dataset** makes the stored rel roots trivially
  stable (`colmap_root_rel: ".."`), so moving or renaming the whole dataset folder
  never triggers the relink prompt (§7). `slang_splat/` sits next to `sparse/` /
  `images*/` and is touched by neither the dataset fingerprint (which scans the
  images root only) nor embedded-mode copying (which copies the images root and
  `sparse/`), so a project cannot recurse into itself.

## 3. The config snapshot

Saved: `training_build_args`, `renderer`, `viewer.controls`, `viewer.import`,
`viewer.run` (including preserved automation sections so a GUI-saved project can
carry headless schedules).

Deliberately **not** saved (machine/user-local, stays in `config/defaults.json`):

- `viewer.ui` — theme, interface scale, graphics API, window-open toggles, histogram
  ranges, photometric *panel* settings. Rationale: reopening a colleague's project
  must not restyle your viewer or switch your GPU API. (The photometric *training*
  keys that affect results live in `viewer.ui` today — see open question in §12.)
- `viewer.state` — prepass memory budget, list capacity: hardware-sized.
- Recent-projects MRU — lives in `defaults.json → viewer.ui.recent_projects` (it is
  machine state; the existing `write_defaults` merge pattern in
  `_set_graphics_api_callback`, [app.py:927-934](../src/viewer/app.py), is the
  template for updating one key without clobbering the rest).

## 4. Save flow, creation on import, and autosave

New module `src/viewer/project.py`, actions bound like every other callback
(`cb.save_project`, `cb.save_project_as`, `cb.open_project` in
`_bind_toolkit_callbacks`, [app.py:671](../src/viewer/app.py)):

```
save_project(viewer, project_dir):
  1. config   = capture via the three existing exporters (§1a), with viewer.ui/state
                sections dropped and existing_run preserved from the loaded project
  2. scene    = viewer._export_source_scene()          # trainer live scene, or edited
                                                       # GPU buffer, or CPU scene
                save_gaussian_ply(dir/"scene.ply", scene,
                                  include_sh=viewer._export_should_include_sh())
  3. dataset  = fingerprint + abs/rel roots from viewer.s.colmap_root / colmap_import
  4. result   = trainer.state.step, training_elapsed_s, count, sh_band (None if no trainer)
  5. thumbnail (stage 2): renderer._read_image() of the last presented frame → png
  6. write project.json atomically (tmp + replace), bump modified
```

Notes grounded in existing behavior:

- `_export_source_scene` ([app.py:577](../src/viewer/app.py)) already prefers the
  live GPU buffer, so **splat-editor edits are saved** without extra work; this is
  the same guarantee Export PLY gives today.
- Saving mid-training is safe at the same points Export PLY is safe today (between
  batches on the render thread via `_run_action`); no pause required. The saved
  `result.step` and the PLY are captured in the same action so they can't skew.
- `Save` (no dialog) reuses the stored project dir; `Save As…` =
  `save_file_dialog` → create `name.splatproj/`.

### Mandatory project creation on COLMAP import

Importing a COLMAP dataset **always creates a project** — there are no projectless
COLMAP sessions. The location is resolved *before* the import runs (prompts below,
at `Load COLMAP…` confirm time); the initial `project.json` (config snapshot +
dataset block, `result.step = 0`) is written at import completion — the same
trainer-construction point §5 step 5 hooks — and registered in the recents MRU.
`scene.ply` first appears at the first save/autosave, so opening a never-saved
project simply re-imports and re-initializes (§5 step 4 with no resume; the resume
hook checks for `scene.ply`).

- **Collision**: if `<colmap_root>/slang_splat/project.json` already exists, the
  import prompts — `Open existing` (default; becomes a §5 open) / `Start fresh`
  (rewrite the manifest, discarding the previous save) / `Choose location…`
  (`save_file_dialog`, Save As semantics).
- **Unwritable dataset root** (read-only media, share without write access): fall
  back to the same `Choose location…` dialog before the import proceeds — the
  project dir must be known up front because autosave needs it. Cancelling the
  dialog cancels the import (§12).
- **Headless parity**: a headless COLMAP run creates/updates the same default
  project, so long unattended runs get autosave checkpoints and crash-resume for
  free.
- Only COLMAP imports are mandatory-project. `Load PLY…` sessions stay projectless
  until `Save As…` (unchanged).

### Autosave checkpoints

Autosave is **step-driven**: it fires when `trainer.state.step` crosses a multiple
of the interval, plus on training stop/pause (so a run stopped at step 22 000 does
not roll back to 15 000 on reopen):

- **Key** (in `viewer.run`, so it travels with the project and applies headless):
  `autosave_interval_steps` (default 15000; `0` = manual saves only).
- **What it writes**: exactly the manual save above (steps 1–6) into the **single
  latest slot** — `project.json` + `scene.ply`, overwriting the previous autosave.
  No checkpoint history is kept (disk: one scene copy, not N). Autosave is a real
  save: it clears the dirty flag and bumps `modified`.
- **Scheduling**: the step-crossing check and the capture run at the same
  between-batches safe point Export PLY uses, but the capture is the *raw GPU
  readback only* (`capture_live_scene` splits `read_live_scene`); param unpacking,
  SH padding, PLY serialization, the first-save dataset fingerprint walk, and all
  file writes (tmp + atomic replace, so a crash mid-write never corrupts the
  previous good save) run on a background writer thread. **Manual saves and the
  stop-boundary save use the same deferred writer** — no save ever blocks the
  render thread on conversion or IO. Writers chain in submission order (each joins
  its predecessor on the worker, never on the caller); an autosave crossing hit
  while a write is in flight is deferred to the next safe point, not dropped. The
  menu bar shows a `Saving project…` indicator while a write is in flight, and
  shutdown/headless-exit join the pending writer.
- **Restore = reopen.** The autosave slot is the ordinary project state; no separate
  restore UI. `result.step` in the manifest always matches the slot because both are
  written in the same action.
- Autosave only advances with training. Config or splat-editor changes made without
  training steps stay unsaved until a manual save or a stop boundary — the
  dirty-flag exit modal (§8) is the guard for those.

## 5. Open flow

```
open_project(viewer, manifest_path):
  1. manifest = load + migrate (schema_version)                        (§9)
  2. verify/relink dataset roots                                       (§7)
  3. overlay  = deep_merge_config(load_defaults(), manifest.config)
     apply_config_overlay(viewer.ui._values, overlay)                  # existing fn
  4. source stage:
     - source == "ply":    session.load_scene(project scene.ply)       # done
     - source == "colmap": session.import_colmap_from_ui(viewer)
                           → GUI: existing streaming import window pumps
                             advance_colmap_import per frame (progress UI for free)
  5. on import completion (new completion hook): resume (§6)
  6. register in recents; set window-title project name; clear dirty flag
```

Step 4/5 needs the only genuinely new orchestration in the feature: a small
`pending_project_resume` state on `ViewerState` consumed when
`advance_colmap_import` finishes (the import completion point already exists — it is
where the trainer is constructed today). Everything else is existing calls in a new
order.

**No render-thread parse**: the `scene.ply` parse starts on a background thread the
moment the project is opened, so it runs concurrently with the dataset import
(which usually dominates). If the parse outlives the import, the resume stays
pending and a per-frame pump applies it once ready — the render thread never
blocks on the load, and the menu bar shows `Opening project…` while a pending open
is in flight. Scene-only (PLY-source) projects load through the same pump.
Autosave is suppressed while a resume is pending so a pre-resume init scene can
never be captured against the resumed manifest.

**Headless parity**: `run_headless_from_config` accepts a `.splatproj` manifest —
if the config path ends in `project.json`/`.splatproj`, load the manifest, merge its
`config`, and map `result` onto the existing `resume_ply`/`train_start_step` keys.
One schema, both entry points, as the roadmap required.

## 6. Resume semantics

Reopening a project restores: dataset, all hyperparameters, schedule position
(`trainer.state.step = result.step`), elapsed-time counter, the trained scene
(`trainer.replace_scene(load_gaussian_ply(scene.ply))`), and the viewer/session
settings. It does **not** restore Adam moments, splat ages, contribution/viewed
histories, or the original init anchors — `replace_scene` →
`_reset_edited_scene_state` rebuilds anchors from the saved scene and zeroes moments
([gaussian_trainer.py:2632-2643](../src/training/gaussian_trainer.py)).

This adopts the already-shipped `resume_ply` contract verbatim (§1c) rather than
inventing a checkpoint format. Consequences worth stating in the UI/docs:

- Continuing training after reopen is a *warm restart*: expect a brief loss bump as
  moments rebuild (the schedule position, and thus LRs/stage params, are correct).
- The init-position regularizer now pulls toward the *saved* geometry, not the
  original COLMAP cloud — for a mostly-converged scene this is the better anchor
  anyway.
- A full training checkpoint (moments ≈ 472 B/splat ≈ 2.4 GB at 5 M splats, plus
  ages/histories) is explicitly out of scope for v1; if ever needed it slots in as an
  optional `checkpoint.bin` next to `scene.ply` without a schema change.
- `Reinitialize Gaussians` after reopen still works and rebuilds from the original
  import sources (the import config is preserved intact — the project does *not*
  rewrite the init mode to `custom_ply`, precisely so reinitialize keeps meaning
  "from the original sources").

## 7. Relinking and dataset modes

**Reference mode (v1 default).** On open, resolve `colmap_root` in order: absolute
path → project-relative path → prompt. For default-location projects (§4) the rel
path is `..`, so a moved or renamed dataset folder resolves silently — the prompt is
effectively reserved for `Save As…`-relocated projects. The prompt is a small modal ("Dataset not
found at …") with `Locate…` (`choose_folder_dialog`) / `Open anyway` (scene-only:
skip import, load `scene.ply` as a plain PLY — the project degrades to a viewer
session with all settings intact) / `Cancel`. After a successful relink, rewrite the
other stored dataset-derived paths by prefix substitution and re-verify.

**Fingerprint** = file count + total bytes + newest mtime + sha1 over the sorted
(name, size) listing of the images root. Cheap (directory scan only, no content
reads), catches moved/regenerated datasets, and mismatch is a *warning banner* — not
a hard failure — because downscale-regenerated datasets are a legitimate workflow.

**Embedded mode (stage 3).** `dataset.mode == "embedded"` copies the images root and
`sparse/` into `name.splatproj/dataset/` at save. With `Best Pose Subset` active,
embed only the selected frames' images (the subset ids are already in
`colmap_selected_camera_ids` / the pose-subset state). BC7 cache directories are
**not** embedded (they regenerate; they are also large).

## 8. Dirty tracking and UI

**File menu** (extends [ui.py `_draw_file_menu`](../src/viewer/ui.py)):

```
File
  New Project              (reset to defaults + empty scene — optional, stage 3)
  Open Project…            Ctrl+O
  Open Recent              ▸ (from defaults.json MRU, existence-checked)
  ─────
  Save Project             Ctrl+S   (always enabled in COLMAP sessions (§4);
                                     disabled only in projectless PLY sessions)
  Save Project As…         Ctrl+Shift+S
  ─────
  Load PLY… / Export PLY… / Load COLMAP… / Reload / Reinitialize Gaussians
  ─────
  Exit
```

**Dirty flag** — pragmatic definition, no config diffing:
`dirty = (trainer.state.step != saved_step) or scene_topology_changed or
edits_applied or config_snapshot_hash != saved_hash`, where the snapshot hash is the
capture from §1a hashed at save/open time and re-hashed lazily (on menu open /
periodic 1 Hz — capture is a dict build, cheap). Title bar shows
`name.splatproj*`. The existing exit-confirmation modal
([Viewer.md "Input Routing"](Viewer.md)) gains a `Save and Exit` button when a dirty
project is open — it already intercepts both `File → Exit` and native title-bar
close, so unsaved-changes protection needs no new interception machinery. Autosave
(§4) clears the dirty flag each time it fires, so during training the modal only
guards the last < 15 000 steps; because autosave is step-driven, config and editor
changes made without training remain guarded by the modal until a manual save.

**Status**: the Scene I/O section shows the project name/path alongside the existing
root/import summary; `defaults_status`-style toast after save (the `t("…")` text
proxy pattern, [app.py:923](../src/viewer/app.py)).

## 9. Versioning and migration

- `schema_version` (manifest structure) and `app_version` (informational) stamped at
  save. `migrate_project(manifest)` is a chain of `vN → vN+1` dict rewrites; v1 has
  none.
- Config forward-compat is inherited from the defaults merge: a project saved before
  a new training knob existed loads with the knob at its current default
  (`deep_merge_config(load_defaults(), project.config)` — the same guarantee run
  configs rely on today). A round-trip test locks this in: save → delete a key →
  open → assert default applied; save → add unknown key → open → assert ignored
  gracefully.
- Renamed/removed keys are handled where they already are: `apply_config_overlay`
  simply skips unknown control keys, and legacy aliases follow the existing pattern
  (cf. `fibonacci_sphere_radius` → `fibonacci_sphere_radius_multiplier` fallbacks in
  [state.py](../src/viewer/state.py)).

## 10. Forward compatibility with the scene hierarchy

Designed-in touch points so feature 3 extends rather than breaks the format:

- `scene.objects[]` exists from v1 (single entry). Feature 3 adds entries with
  `{ id, name, ply, training: {preset, trainable, allow_refine, lr_mul}, visible }`
  and per-object PLY files (`objects/<name>.ply`), keeping `scene.ply` as the legacy
  single-object case.
- `result` is already per-run, not per-object; per-object counts join the object
  entries, not `result`.
- Auto-objects (e.g. the Fibonacci `sky shell`) are identified by `source` name in
  the object entry so `Reinitialize Gaussians` can recreate them with stable
  names/presets even though ids may change
  ([SceneHierarchyDesign.md §10](SceneHierarchyDesign.md)).
- `edit_state.json` + `selection.bin` are keyed to `splat_count` and object list
  hash; any mismatch silently drops the stale selection (same policy as the editor's
  `sync_selection_to_scene` today, [splat_editor.py:149](../src/viewer/splat_editor.py)).

## 11. Staging plan

**Stage 1 — core save/open (the daily-workflow win).**
`src/viewer/project.py` (`save_project` / `load_project` / `migrate_project`);
capture via the three existing exporters; GUI apply path (`apply_config_overlay` +
source trigger + `pending_project_resume` hook); File menu Save/Save As/Open; window
title; reference-mode paths with abs→rel→prompt relink (prompt = simple modal);
`result.step` resume; mandatory creation on COLMAP import at
`<colmap_root>/slang_splat/` (collision + unwritable-root prompts); autosave (every
`autosave_interval_steps` = 15 000 steps + stop/pause boundary, single latest-slot
overwrite, background write).
*Verify:* round-trip test (save → wipe UI values to defaults → open → values equal);
COLMAP project reopens, trainer resumes at saved step with staged params resolved
for that step (assert LR against `resolve_learning_rate_scale`); PLY-only project
round-trips; config forward-compat test (§9); every COLMAP import leaves a valid
`project.json` at the default path (and in recents); autosave at a short test
interval overwrites the slot with `result.step` in sync with `scene.ply`;
kill-and-reopen resumes at the last autosaved step.

**Stage 2 — polish.**
Recents MRU in `defaults.json`; dirty flag + `Save and Exit` in the existing exit
modal; thumbnail capture; Scene I/O project status; headless `.splatproj` support;
fingerprint warning banner.
*Verify:* native-close with dirty project offers save; headless run from a GUI-saved
project reproduces the GUI configuration (assert resolved training params equal).

**Stage 3 — bigger options.**
Embedded dataset mode (with pose-subset-only embedding); edit-state persistence
(box/ranges/selection bitmask); `New Project`; project browser over thumbnails
(optional).

## 12. Risks and open questions

- **Photometric training keys live in `viewer.ui`** (all `photometric_*` values,
  [defaults.json viewer.ui]) — §3 excludes `viewer.ui` from projects, which would
  drop result-relevant photometric hyperparameters. Recommendation: move the
  photometric *trainer* keys into `viewer.controls` (or capture them explicitly into
  the project's `viewer.controls` section) as part of stage 1 — a small,
  behavior-neutral key reshuffle that `apply_config_overlay` already handles, plus a
  legacy alias for old defaults files.
- **Learned photometric compensation state is not persisted** — reopening a project
  that used PPISP compensation replays `photometric.steps` (if configured in the run
  section) or reopens uncompensated. Acceptable for v1 (compensation retrains in
  ~1000 steps at import, an existing import option); persisting learned PPISP params
  is a candidate `photometric.bin` extra, no schema change needed.
- **Two sources of truth for the import config** (`ui._values` vs.
  `viewer.s.colmap_import`): the project saves the UI-values form (§1d). If a future
  change makes `Reload` semantics diverge from re-import-from-UI, project reopen
  fidelity should be re-checked. A test asserting `import_colmap_from_ui` consumes
  every key the project saves guards this.
- **Windows path portability**: store rel paths POSIX-style (`as_posix()`), resolve
  with `Path`; abs paths keep drive letters and simply fail over to rel/prompt on
  another machine — that is the designed path, not an error case.
- **Autosave write cost and disk footprint**: a save is ~236 B/splat with SH3
  (~1.2 GB at 5 M splats), held once on disk (~2× transiently during the tmp +
  atomic-replace write). At one save per 15 000 steps the write amortizes to noise,
  but the safe-point GPU readback is synchronous with training — if it stalls large
  scenes noticeably, measure during stage 1 and consider staging the readback.
- **Mandatory creation is strict**: cancelling the fallback location dialog cancels
  the import. If that proves hostile to quick dataset triage, the escape hatch is a
  scratch project in the temp dir — not a projectless session — so autosave and
  reopen keep working.
- **Open**: should `Save Project` also auto-export `output_ply` when
  `viewer.run.output_ply` is set (headless habit), or keep project scene.ply and run
  output strictly separate? Proposed: separate — `scene.ply` is project state,
  `output_ply` is an automation artifact.
- **Open**: `train_iters` after resume — headless treats it as the target step count;
  a reopened project at step 30 000 with `train_iters: 30000` has nothing left to do.
  GUI training is unbounded so this only affects headless replay; document that
  continuing a project headless requires raising `train_iters`.
- **Open**: should `Start fresh` on an import collision (§4) archive the previous
  project (rename to `slang_splat_old/`) instead of overwriting the manifest and
  `scene.ply`? Proposed: archive-by-rename — cheap insurance against clobbering a
  trained result with a misclick.

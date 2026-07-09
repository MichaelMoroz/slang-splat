# Initialization Regularization + Non-Refinable Splats — Implementation Plan

## Goal

1. Every splat carries its **source initialization state**: init position, init nearest-neighbor (NN) radius, and a **refinable** flag (a `float4` for `(init_pos.xyz, nn_radius)` plus a bool).
2. An **L2 position regularizer** pulls each splat toward its init position, scaled by a user weight and by the NN radius. It is applied as a **direct post-step displacement** each optimizer step: `pos -= weight · NN_radius · (pos - init_pos)`, default `weight = 0.001`. `weight` is the displacement coefficient itself (not a loss coefficient, not multiplied by the optimizer LR) — matching the stated default `0.001 · NNrad · (pos - init_pos)`. See "Regularizer formula" below for why.
3. Splats marked **non-refinable** are never pruned or subdivided by refinement — only optimized. The refinable flag is set per import source (a toggle in the importer).
4. The refinement **distribution / prune-target / clone accounting** must exclude non-refinable splats correctly, and the existing refinement path should be bug-checked while we are in there.

## Key facts established from the code

- Trainable params live in a planar `splat_params` buffer owned by the renderer (`paramId * count + splatId`). The trainer optimizes that buffer in place; refinement double-buffers it (`dst_splat_params` → copied back).
- Per-splat **static aux data** already rides alongside via the refinement machinery as triple buffers `X` (src) / `dst_X` / `append_X`, allocated in `_ensure_refinement_buffers` / `_ensure_post_refinement_source_buffer_capacity`, written by `csRewriteRefinementSplats`, and copied dst→src after a rewrite (`splat_age` is the reference example: `_ensure_splat_age_capacity`, `_reset_splat_ages`, `_copy_refinement_splat_ages_to_source`).
- The optimizer **post-step** kernel `csProjectGaussianParams` (`shaders/utility/optimizer/gaussian_optimizer_stage.slang`) already applies position/scale/opacity regularizers (camera push-away, `scale_abs_reg`, `opacity_reg`). This is the natural home for the init-position pull. It is dispatched from `GaussianOptimizer.dispatch_projection` (`src/training/optimizer.py`).
- Refinement eligibility/skip points in `shaders/renderer/gaussian_training_stage.slang`:
  - `should_cull_refinement_splat(splatId, alpha)` — prune gate (used by `csPrepareRefinementCounts`, `csRewriteRefinementSplats`).
  - `csPrepareRefinementPruneSortInputs` — sets `refinement_eligible_mask` + prune sort key.
  - `refinement_prune_target_count(candidateCount)` — `candidateCount = g_TotalCloneCounter[0]` = inclusive scan of `refinement_eligible_mask` (so prune target already scales to the *eligible* population).
  - `refinement_sampling_distribution(splatId)` / `csPrepareRefinementSamplingWeights` — clone weight per splat.
  - `csRewriteRefinementSplats` — compacts survivors + emits children.
- Scene sources are built independently and concatenated (`_concat_gaussian_scenes` in `session.py`); each source has per-source UI values `colmap_<source>_*`.

## Design decisions

- **Storage = one `float4` per splat** in a new static aux buffer `splat_init`:
  `xyz = init_pos`, `w = refinable ? +NN_radius : -NN_radius`.
  The refinable bool is encoded in the **sign of w** (NN radius is always `> 0`, clamped to `_MIN_SCALE`). This realizes the requested "float4 + bool" in a single buffer, which matters because every static per-splat buffer must be threaded through the whole refinement double-buffer path; a second buffer would double that surface and its bug budget. Shader helpers: `init_pos(d)=d.xyz`, `init_nn_radius(d)=abs(d.w)`, `init_refinable(d)= d.w >= 0.0`.
- **NN radius source**: computed once at trainer init as `point_nn_scales(scene.positions)` over the final (concatenated) positions — local density in the actual training scene. Works uniformly for COLMAP / PLY / editor scenes.
- **Init position**: `scene.positions.copy()` captured at trainer init (and reset on `replace_scene`).
- **Regularizer formula (resolved)**: a **direct post-step displacement** `pos -= weight · NN · (pos − init_pos)`, with `weight = 0.001` default — exactly the stated default `0.001 · NNrad · (pos − init_pos)`. `weight` is the displacement coefficient; it is **not** multiplied by the optimizer LR and there is **no** factor of 2. This is the plain reading of the goal statement, is LR-schedule-independent (a steady anchoring pull), and mirrors the camera push-away regularizer in `csProjectGaussianParams`, which also applies a direct displacement rather than an LR-scaled loss gradient. (Interpretation as a squared-loss term `weight·NN·‖·‖²` added to the training loss was rejected: it couples the pull to the LR schedule and needs a factor-2/LR convention the user did not ask for.) `NN` is `abs(w)` from the packed init `float4`; the pull is applied to all splats.
- **Regularizer applies to all splats** (refinable and not) — the flag only governs refinement, not optimization.
- **Children inherit the parent's init state** (`init_pos`, `NN_radius`) and are refinable (`w = +NN`). Non-refinable parents never clone, so children are always refinable.

## Changes

### 1. Scene data model — `src/scene/gaussian_scene.py`
- Add optional field `refinable: np.ndarray | None = None` (per-splat bool, default = all True when None). Keep `init_positions` / `init_nn_radius` **out** of the scene: they are derived at trainer init (positions copy + NN), so the scene stays lean and existing constructors don't all need updating. Only `refinable` is authored (by the importer), so only it lives on the scene.
- `__post_init__`: coerce `refinable` to `bool[count]` when provided; validate length.
- `subset` / any scene copy helpers: carry `refinable` through.

### 2. Importer — refinable per source
The `_refinable` flag must ride the **full settings→progress→builder chain**, or the UI checkbox silently reads back as all-true. For each of the five sources (pointcloud, diffused, custom PLY, custom mesh, fibonacci sphere), add `colmap_<source>_refinable` (default `True`) to **every** hop:
- `_VIEWER_IMPORT_DEFAULTS` — default `True` (so `state.py` field defaults resolve).
- `_VIEWER_IMPORT_EXPORT_FIELDS` in `src/viewer/ui_schema.py` (~line 198) — `("colmap_<source>_refinable", bool)` so it round-trips through saved configs, next to the existing `colmap_<source>_enabled` / `_nn_radius_scale_coef`.
- `ColmapImportSettings` (state.py ~line 79) — `<source>_refinable: bool` field.
- `ColmapImportProgress` (state.py ~line 134) — `<source>_refinable: bool` field.
- `ColmapImportSettings(...)` construction (session.py ~line 2924) and `ColmapImportProgress(...)` construction (session.py ~line 3144) — read the UI value / carry the settings field through, mirroring `pointcloud_enabled`.
- `src/viewer/ui.py` import window: a "Refinable" checkbox on each source row (default on), writing `colmap_<source>_refinable`.
- Scene builder: each source builder sets `scene.refinable = full(count, progress.<source>_refinable)`; `_concat_gaussian_scenes` concatenates `refinable` (treat `None` as all-True) so mixed sources compose.
- Non-COLMAP load (PLY via File menu) stays fully refinable (`refinable=None` → all True).

### 3. GPU init buffer in the trainer — `src/training/gaussian_trainer.py`
Mirror the `splat_age` triple-buffer exactly:
- New refinement buffers `splat_init` (src, `float4`), `dst_splat_init`, `append_splat_init`, with capacity tracking `_refinement_splat_init_capacity`.
- **Readiness gate (High finding).** `_ensure_refinement_buffers` early-returns (line ~1323) when `age_buffers_ready and prune_buffers_ready and not grow_*`. That gate does **not** cover the new init buffers, so a steady-state call could return before `dst_splat_init` / `append_splat_init` exist. Add an explicit `init_buffers_ready = all(name in self._refinement_buffers for name in ("splat_init","dst_splat_init","append_splat_init")) and required_splats <= self._refinement_splat_init_capacity` and AND it into the early-return condition (next to `age_buffers_ready`). Allocate dst/append init in the same `grow_splats` / "not present" branches that handle `dst_splat_age` / `append_splat_age`.
- `_build_splat_init_array(scene)`: `float4[count]` from `init_pos = positions`, `w = signed NN` using `point_nn_scales`, sign from `scene.refinable`.
- `_ensure_splat_init_capacity(count, reset=)`, `_reset_splat_init(scene)`, `_copy_refinement_splat_init_to_source(count)` — analogues of the age helpers; alloc dst/append in `_ensure_refinement_buffers` and `_ensure_post_refinement_source_buffer_capacity`.
- Populate at construction and in `replace_scene` (fresh init from the new scene). For the GPU **editor resample** (`resample_selection` → `renderer.edit_resample`, which changes count without a CPU scene), init data must be compacted too — see §6.
- `_refinement_vars`: bind `g_SrcSplatInit`, `g_DstSplatInit`, `g_AppendSplatInit`.
- After `csRewriteRefinementSplats`: `_copy_refinement_splat_init_to_source(next_count)` and extend the dst→src copy block (like contribution history).
- Add hyperparameter `init_position_reg_weight` (default `0.001`) to `TrainingHyperParams` + `TRAINING_BUILD_ARG_DEFAULTS`; thread into the optimizer projection dispatch.

### 4. Optimizer regularizer — `gaussian_optimizer_stage.slang` + `optimizer.py`
- Add uniforms `g_SplatInit : StructuredBuffer<float4>` and `g_InitPositionRegWeight : float` to `csProjectGaussianParams`.
- In `csProjectGaussianParams`, before the value clamps (direct displacement, no LR, no factor 2):
  ```
  if (g_InitPositionRegWeight != 0.0) {
      float4 d = g_SplatInit[splatId];
      float nn = abs(d.w);
      float coeff = saturate(g_InitPositionRegWeight * nn); // >=1 would overshoot init; clamp to [0,1]
      state.position -= coeff * (state.position - d.xyz);
  }
  ```
  Clamping `coeff` to `[0,1]` in the shader means the pull can at most snap a splat exactly onto its init position in one step, never overshoot — robust to a large user `weight` without a separate upload-time clamp.
- `GaussianOptimizer.dispatch_projection`: pass `g_SplatInit` (trainer's `splat_init` src buffer) and `g_InitPositionRegWeight`. Because the optimizer is constructed with the renderer only, the trainer will pass the init buffer through `dispatch_projection` (extend its signature) — the trainer already calls it from `_dispatch_optimizer_step`.

### 5. Refinement skip + accounting — `gaussian_training_stage.slang`
- Add `uniform StructuredBuffer<float4> g_SrcSplatInit;` and helpers `refinement_is_refinable(splatId) = g_SrcSplatInit[splatId].w >= 0.0`.
- `should_cull_refinement_splat`: `if (!refinement_is_refinable(splatId)) return false;` (non-refinable never pruned).
- `csPrepareRefinementPruneSortInputs`: `eligible = refinement_is_refinable(splatId) && alpha>=cull && contribution>=min;` non-eligible get sort key `0xFFFFFFFF`. → non-refinable excluded from the eligible count (so `refinement_prune_target_count` scales to the refinable-eligible population) and never marked for prune.
- `refinement_sampling_distribution`: `if (!refinement_is_refinable(splatId)) return 0.0;` (never cloned). `csPrepareRefinementSamplingWeights` also gates `eligible` on refinable for defense-in-depth.
- **Effective clone count shared by count + rewrite (Medium finding).** The trainer sizes `next_count = survivor_count + capped_clone_total` from `g_TotalCloneCounter`, which `csPrepareRefinementCounts` fills by summing `g_CloneCounts[splatId]` over non-culled splats (line ~1235). Non-refinable splats are non-culled (they always survive), so if one ever had a nonzero clone count the count phase would over-size `next_count` while `csRewriteRefinementSplats` emits fewer children → adopted-count vs. emitted-children mismatch. Introduce one helper `uint refinement_effective_clone_count(uint splatId) { return refinement_is_refinable(splatId) ? min(g_CloneCounts[splatId], REFINEMENT_MAX_CLONES_PER_SPLAT) : 0u; }` and use it in **both** `csPrepareRefinementCounts` (the `g_TotalCloneCounter` add) and `csRewriteRefinementSplats` (the child-emission count), so counting and emission always agree regardless of sampling. (Survivor/append counting is unaffected — non-refinable still survive and count as survivors.)
- `csRewriteRefinementSplats`: use `refinement_effective_clone_count` for `emittedCloneCount`; non-refinable → 0 (kept as-is, no split, no opacity solve). Write `g_DstSplatInit[parentIndex]` for survivors and `g_AppendSplatInit` / `g_DstSplatInit[childIndex]` for children (children inherit parent init with positive w). `csPrepareRefinementCounts` already survives non-refinable via `should_cull`.

### 6. Editor resample compaction — init data
- `renderer.edit_resample` compacts `splat_params` + mask. Init data is a trainer buffer, not a renderer buffer, so the trainer's `resample_selection` must compact `splat_init` with the same keep/clone mapping. Simplest correct approach: after `renderer.edit_resample` returns the new count, the trainer rebuilds `splat_init` from the **new** GPU scene (`init_pos = current positions`, `NN = point_nn_scales(new positions)`, refinable = all True). Rationale: editor resample already resets Adam/ages/refinement bookkeeping (identity change), and edited splats have no meaningful "original init" to preserve, so re-anchoring init to the post-edit positions is the consistent choice. Documented as such.

### 7. Bug check (requested)
While touching refinement, verify and note:
- `total_clone_counter` is reused as (a) the eligible-count for `refinement_prune_target_count` and (b) the running clone total in the rewrite; confirm `clear_refinement_counters` resets it between phases so prune targeting can't see a stale clone total (and vice-versa).
- Confirm the eligible-mask inclusive scan denominator matches what `refinement_prune_target_count` expects after non-refinable exclusion (prune ratio applies to refinable-eligible, not all splats).
- Confirm `csPrepareRefinementSamplingWeights` `eligible = eligible_mask && !prune_mask` still holds with refinable gating and that a fully non-refinable scene yields zero clones / zero prunes without div-by-zero (weight_total = 0 guarded in `csSampleRefinementCloneCounts`).

### 8. Tests — `tests/`
- `test_training_kernels.py` (or a new `test_init_regularization.py`):
  - Init reg pulls a displaced splat toward its init position: a single post-step moves `pos` toward `init_pos` by exactly `weight·NN·Δ` (assert the closed-form displacement); `weight = 0` is a no-op; magnitude scales linearly with NN radius and with `weight`.
  - Non-refinable splat is never pruned even with contribution below threshold, and never cloned even with high clone weight (run a refinement step on a mixed scene; assert non-refinable indices persist and don't spawn children).
  - Prune target scales to the refinable-eligible count (a scene half non-refinable prunes at most from the refinable half).
  - Init data survives a refinement step (survivor init unchanged; child inherits parent init).
- `test_splat_edit.py` / scene: `refinable` concatenation + subset round-trips.

## Out of scope / follow-ups
- Persisting `refinable` / init state into exported PLY (not requested; export currently drops trainer-only aux). Note in docs.
- Per-splat editor toggle of refinable (importer-level only, as requested).

## Order of implementation
1. Scene `refinable` field + concat/subset + tests.
2. Trainer `splat_init` buffers (alloc/reset/copy-back) mirroring `splat_age`; `init_position_reg_weight` hparam.
3. Optimizer shader + `dispatch_projection` regularizer; test the pull.
4. Refinement shader skips + accounting; rewrite carries init; dst→src copy; tests.
5. Importer per-source refinable toggle (session + UI).
6. Editor resample init rebuild.
7. Bug-check notes, docs (`doc/Training.md`/`Rendering.md`), full test sweep, commit.

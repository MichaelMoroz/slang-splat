# Scene Hierarchy Deep Dive — Splat Objects, Camera-Pose Objects, Per-Object Training

Design investigation expanding [roadmap.md §3](roadmap.md). Everything here is grounded
in the current code; file/line anchors refer to the `streaming` branch as of July 2026.
Structure:

1. [Corrections to the roadmap sketch](#1-what-the-code-actually-does-corrections-to-the-roadmap-sketch)
2. [Data model](#2-data-model)
3. [GPU representation: the `splat_meta` channel](#3-gpu-representation-the-splat_meta-channel)
4. [Trainer rework](#4-trainer-rework)
5. [Renderer rework](#5-renderer-rework)
6. [Editing with objects](#6-editing-with-objects)
7. [Gizmos and object transforms](#7-gizmos-and-object-transforms)
8. [Camera-pose objects](#8-camera-pose-objects)
9. [UI design](#9-ui-design)
10. [Ingest, merge, export](#10-ingest-merge-export)
11. [Staging plan](#11-staging-plan)
12. [Risks and open questions](#12-risks-and-open-questions)

---

## 1. What the code actually does (corrections to the roadmap sketch)

The roadmap's §3 sketch is directionally right, but four details of the current
architecture change the design:

**a) There are three independent scene-buffer copies, not one shared set.**
Each renderer role (`renderer`, `training_renderer`, `debug_renderer` —
[session.py:1790-1812](../src/viewer/session.py)) is a full `GaussianRenderer` owning its
own `_scene_buffers` dict. Today that dict holds exactly one buffer: `splat_params`
(`_SCENE_SHADER_VARS`, [gaussian_renderer.py:246](../src/renderer/gaussian_renderer.py)).
Roles synchronize by GPU→GPU copies in `copy_scene_state_to`
([gaussian_renderer.py:2120](../src/renderer/gaussian_renderer.py)), which iterates
`_SCENE_SHADER_VARS` by name. During training the viewport renders **directly from the
training renderer** ([presenter.py:1312](../src/viewer/presenter.py)), so the
main-renderer copy only matters for PLY-only viewing. Consequence: any new per-splat
channel must (1) live in the scene group so role copies pick it up, and (2) per-draw
state like object visibility must be *re-appliable member state*, because the same
renderer object serves both the training step and the viewport draw each frame (the
presenter already re-applies main-role params onto the training renderer at
[presenter.py:1315](../src/viewer/presenter.py)).

**b) `splat_age` / `splat_init` / Adam moments are trainer-owned, not renderer-owned.**
They live in `trainer._refinement_buffers` and `adam_optimizer.buffers` and exist only
when a trainer exists. A PLY loaded without training has only `splat_params` + the
highlight buffer. Consequence: object identity must be renderer-owned (like the
highlight buffer), not trainer-owned, or PLY-only scenes could not be organized into
objects.

**c) Refinement compaction does not preserve order.** `csRewriteRefinementSplats`
([gaussian_training_stage.slang:1368](../../shaders/renderer/gaussian_training_stage.slang))
allocates destination slots with `InterlockedAdd` on an append counter, so survivors land
in arbitrary order and clone children land in a separate append region. "Keep splats
grouped by object" is therefore not a viable invariant — object membership must be a
per-splat channel carried through the rewrite, exactly like `splat_age`,
`splat_init`, contribution history, and Adam state already are (each is one more
`g_Src*/g_Dst*/g_Append*` triple in `_refinement_vars`,
[gaussian_trainer.py:916-973](../src/training/gaussian_trainer.py)).

**d) The Adam kernel already has the hook point for per-splat gating.**
`csAdamStepPacked` ([optimizer.slang:59](../../shaders/utility/optimizer/optimizer.slang))
runs one thread per packed scalar with `paramId = paramIndex / paramGroupSize` and
(implicitly) `elementId = paramIndex % paramGroupSize` where `paramGroupSize` is the
splat count. Per-splat masking is one buffer read and one branch at the top of the
kernel — no restructuring. `OptimizerParamSettings` stays per-param-id;
the per-splat mask and per-object LR table compose with it multiplicatively.

Also worth noting: `GaussianScene.subset(max_splats)`
([gaussian_scene.py:43](../src/scene/gaussian_scene.py)) takes a count, not a mask —
the CPU-side concat/split helpers for objects are new code, not reuse.

---

## 2. Data model

CPU-side scene graph; GPU stays one flat concatenated buffer tagged per splat.

```
SceneGraph                                   (new: src/scene/scene_graph.py)
 ├─ splat_objects: list[SplatObject]
 │     id: int              # stable u16, never reused within a session
 │     name: str
 │     visible: bool        # viewport-only; never affects the training render
 │     transform: TRS       # identity unless being manipulated; baked on commit
 │     pivot: np.ndarray    # gizmo pivot (bbox center at creation/extract time)
 │     training: ObjectTrainingConfig
 │     source: SourceRef    # ply path / colmap init mode / "extracted", for UI + project.json
 ├─ camera_rigs: list[CameraRig]
 │     name: str            # one per import source ("colmap: hotel3", …)
 │     frames: range into trainer.frames / viewer.s.training_frames
 │     overlay_visible: bool
 ├─ active_object_id: int | None      # editing scope + gizmo target
 └─ active_camera: (rig, frame_index) | None

ObjectTrainingConfig
 ├─ trainable: {position, scale, rotation, opacity, sh0, sh_rest}   # 6 bits
 ├─ allow_refine: bool               # maps onto the existing sign-of-init-radius
 └─ lr_mul: {pos, scale, rot, color, opacity, sh} | None            # stage-4 override
```

Two object kinds only, deliberately: **SplatObject** (owns splats) and **CameraRig /
CameraObject** (a view over training frames). No generic transform nodes, no nesting —
a flat list under two group headers covers the workflows in scope (frozen background +
edited foreground, imported PLY props, multiple capture rigs) and keeps the trainer
contract simple. Nesting can be added later purely on the CPU side because the GPU only
ever sees baked world-space splats.

**Presets** are just `ObjectTrainingConfig` values:

- *Frozen background (color-only)*: `trainable = {sh0, sh_rest}`, `allow_refine=False`.
- *Locked*: all-false, `allow_refine=False` (renders, never changes).
- *Full*: all-true, `allow_refine=True` (default; exactly current behavior).

The scene graph is runtime state on `viewer.s` (a new `scene_graph` field in
`ViewerState`, [state.py:247](../src/viewer/state.py)) and serializes into
`project.json → scene_graph` (feature 2). It is **not** part of `config/defaults.json`.

### Where the splat counts live

Because refinement reorders and resizes the buffer, per-object splat counts are derived
GPU-side, not tracked CPU-side: a tiny `csCountObjectSplats` reduction (256-bin atomic
histogram over `object_id`, one dispatch, 1 KiB readback) refreshed after any
refinement/edit/ingest — same cadence as `edit_selection_count()`
([gaussian_renderer.py:868](../src/renderer/gaussian_renderer.py)) and cheap enough to
run on the hierarchy panel's refresh.

---

## 3. GPU representation: the `splat_meta` channel

One new **renderer-owned** per-splat `u32` buffer, `splat_meta`, allocated next to
`splat_params` in the scene group:

```
bits  0..15  object_id        (65k objects; realistically < 100)
bits 16..21  trainable_mask   {pos, scale, rot, opacity, sh0, sh_rest}
bits 22..31  reserved         (future: per-splat flags, soft-freeze weight, …)
```

Default value `0x003F0000` (object 0, everything trainable) reproduces current
behavior bit-for-bit.

Design decisions and why:

- **Renderer-owned, not trainer-owned.** Visibility filtering, active-object tinting,
  object picking, and editor scoping all happen in renderer passes and must work on
  PLY-only scenes with no trainer (see §1b). It follows the highlight buffer's
  lifecycle pattern, but unlike the highlight it participates in
  `_SCENE_SHADER_VARS` so `copy_scene_state_to` and `_ensure_scene_buffers`
  ([gaussian_renderer.py:1360](../src/renderer/gaussian_renderer.py)) manage it
  automatically. `copy_scene_state_to` currently assumes every scene buffer has the
  same byte size (`copy_bytes` at
  [gaussian_renderer.py:2131](../src/renderer/gaussian_renderer.py)); it needs a
  per-name element size, and the CPU fallback path (`write_scene_groups`, taken when
  packed param counts differ, i.e. SH-band changes) must copy meta too.

- **Mask is per-splat, not per-object.** A per-object table would be smaller, but the
  per-splat mask buys a feature for free that is independently valuable and ships
  *before* objects exist: **"Freeze selection"** in the splat editor (write mask bits
  where the selection mask is set). It also makes object-level freezing trivial — the
  object panel bulk-writes mask bits for all splats with a given `object_id` via one
  kernel (`csWriteObjectMask`). The object stays the source of truth in the UI; the
  buffer is the materialized state the kernels read.

- **`allow_refine` reuses the existing sign encoding.** Refinement eligibility is
  already per-splat via the sign of `splat_init[:, 3]`
  ([gaussian_trainer.py:1561-1576](../src/training/gaussian_trainer.py)) and is honored
  everywhere that matters (`refinement_is_refinable` in prune-candidate marking, clone
  sampling, and the rewrite). Object-level `allow_refine=False` is implemented as a
  bulk sign-write on the object's init radii — zero kernel changes in the refinement
  path. Unifying the flag into `splat_meta` is a later cleanup, not a prerequisite.

- **Not a new row in `splat_params`.** The packed param buffer is float, param-major,
  SH-band-sensitive (it repacks when the band cap changes), and mirrored by Adam
  moments with the same indexing. A metadata channel inside it would corrupt all three
  properties.

Memory: 4 bytes/splat ≈ 0.5 % of `splat_params` at 59 packed params — negligible
(as the roadmap already assumed).

### Buffers/kernels that must carry `splat_meta`

| Path | Change |
|---|---|
| `_ensure_scene_buffers` / `clear_scene_resources` | allocate/release `splat_meta` (u32/splat, not `packed*4` bytes) |
| `set_scene` upload | pack `object_id`/mask from the scene graph (or default) |
| `copy_scene_state_to` | per-name byte sizes; CPU fallback carries meta |
| Editor compaction `csResampleCompactSurvivors` ([splat_edit_stage.slang:244](../../shaders/renderer/splat_edit_stage.slang)) | `g_SrcMeta`/`g_DstMeta`, copied like `g_DstMask` |
| Refinement rewrite `csRewriteRefinementSplats` | `g_SrcSplatMeta`/`g_DstSplatMeta`/`g_AppendSplatMeta` triple; children inherit the parent's word (same as `g_DstSplatInit` handling today) |
| Refinement adoption in `_run_refinement` ([gaussian_trainer.py:1772-1812](../src/training/gaussian_trainer.py)) | one more `copy_buffer` back into the renderer's `splat_meta` |
| `read_live_scene` / PLY export | expose `object_id` for per-object export (§10) |

---

## 4. Trainer rework

The headline: **one trainer, one concatenated buffer, one sort, one loss — unchanged.**
Per-object training is implemented entirely as per-splat gating inside the two kernels
that write parameters, plus sidecar carry in refinement. The raster forward/backward,
loss pipeline, SSIM path, and frame scheduling do not change at all.

### 4.1 Adam step gating

`csAdamStepPacked` gains:

```slang
uniform StructuredBuffer<uint> g_SplatMeta;          // 0 = ungated (legacy dispatch)
uniform StructuredBuffer<uint> g_ParamGroupMaskBit;  // paramId -> mask bit index (u8 LUT)

uint elementId = paramIndex % paramGroupSize;
uint meta = g_SplatMeta[elementId];
if ((meta & (1u << (16u + g_ParamGroupMaskBit[paramId]))) == 0u) return;
```

Early-return (rather than `lr = 0`) matches the roadmap's intent: frozen params get
**no step and no momentum accumulation**, so unfreezing later resumes from clean
moments rather than a stale momentum built while frozen. The LUT is 59 bytes uploaded
once per SH-band change alongside `param_settings`
([optimizer.py:77-126](../src/training/optimizer.py)). Cost: one u32 load per scalar
per step; with the buffer being splat-count-sized and read coalesced (consecutive
threads → consecutive `elementId`), this is noise next to the raster backward.

`csComputePackedElementGradNorms` should skip frozen params in the same way so the
debug grad-norm view doesn't report gradients that can't apply.

### 4.2 Post-step (`csProjectGaussianParams`) gating

The fused per-gaussian post step
([gaussian_optimizer_stage.slang:189](../../shaders/utility/optimizer/gaussian_optimizer_stage.slang))
applies init-position pull, push-away-from-camera, scale/opacity regularizers, SH
projection, quaternion normalization, and the safety clamps. Gate by mask bit:

- position bit off → skip init pull, push-away, random-step noise
  (`csApplyPositionRandomStep` binds the same params buffer,
  [gaussian_trainer.py:1014](../src/training/gaussian_trainer.py) — gate there too);
- scale bit off → skip scale regs and anisotropy *reg* (keep the hard safety clamp);
- opacity bit off → skip opacity reg / binarize reg;
- sh bits off → skip SH projection steps;
- quaternion normalization and NaN/huge sanitization stay **unconditional** — they are
  invariant repairs, not optimization.

### 4.3 Per-object hyperparameter overrides (stage 4)

A 256-entry `object_lr_mul` table (`float` × 6 groups, 6 KiB) indexed by
`object_id`; the Adam kernel multiplies `settings.lr` by
`objectMul[meta & 0xFFFF][g_ParamGroupMaskBit[paramId]]`. It composes with the
schedule because `GaussianOptimizer.update_step`
([optimizer.py:142](../src/training/optimizer.py)) keeps writing the *global* resolved
LR into `param_settings`; the object table is a static multiplier on top. Anything
fancier (per-object schedules, per-object reg weights) should wait for a concrete need
— every added axis multiplies the UI/config/test surface.

### 4.4 Refinement with objects

- `splat_meta` carried through the rewrite (§3 table). Children inherit the parent's
  full meta word — clones stay in the parent's object with the parent's mask.
- `allow_refine=False` objects already fall out of prune candidates and clone sampling
  via the existing refinable flag (§3).
- Prune ratios, clone budget, and `max_gaussians` stay **global** in the first cut.
  The staged `refinement_prune_lowest_contribution_ratio` operates on the eligible set,
  which excludes frozen objects, so a frozen background does not distort foreground
  pruning. Per-object budgets are a possible stage-4+ knob, not a prerequisite.
- The editor densify/sparsify paths (`resample_selection`,
  [gaussian_trainer.py:2560-2630](../src/training/gaussian_trainer.py)) reuse the same
  rewrite and inherit the carry for free once the rewrite carries meta.

### 4.5 What explicitly does not change

Frame selection, targets, photometric compensation, loss forward/backward, cached
raster grads, sorting, the schedule resolver, and the headless runner. `viewer.py
--headless` gains object support only through the project file (feature 2) — a
`scene_graph` section listing per-object PLYs + training configs.

---

## 5. Renderer rework

### 5.1 Per-object visibility = per-draw member state, not buffer state

Because the viewport renders from the training renderer during training (§1a), hiding
an object must never mutate the scene buffers (the very next training step needs the
splat). Implementation: a 256-bit visibility bitmask as renderer member state
(`spy.uint4x2`-style uniform array, 8 × u32), consumed at the top of
`csProjectVisibleSplats` ([gaussian_project_stage.slang:167](../../shaders/renderer/gaussian_project_stage.slang)):

```slang
uint objectId = g_SplatMeta[splatId] & 0xFFFFu;
if ((g_ObjectVisibleBits[objectId >> 5u] & (1u << (objectId & 31u))) == 0u) return;
```

The splat is simply never emitted into the visible list — no sort/raster changes, no
cost when everything is visible. The trainer's step leaves the mask all-ones; the
presenter applies the UI mask before the viewport draw, in the exact slot where it
already re-applies main-role params onto the training renderer
([presenter.py:1315](../src/viewer/presenter.py) →
`_apply_renderer_role_params`). Training-camera debug draws and the metrics evaluator
also force all-ones, so hidden objects can never leak into losses or reported PSNR.

*Policy question this raises*: should "hidden" objects still train? Yes — visibility
is presentation-only. An object you don't want trained is *frozen*, not hidden. The UI
must keep these as clearly separate toggles (eye vs. lock).

### 5.2 Active-object emphasis

Reuse the preview-tint slot in the prepass
([gaussian_project_stage.slang:183-185](../../shaders/renderer/gaussian_project_stage.slang)):
a per-draw `g_ActiveObjectId` + "dim others" mix desaturates/dims splats whose
`object_id` differs from the active object (only when the hierarchy panel requests
isolate-highlight). Zero extra dispatches, same pattern as the existing
committed-selection highlight and candidate preview, and it stacks under them (a
selected splat still reads as selected inside the active object).

### 5.3 Per-object transforms: live preview in prepass, bake on commit

Split the roadmap's either/or into both, each where it is cheap:

- **While the gizmo is dragging** (main viewport only): a per-draw
  `g_ObjectTransforms` table (float3x4 per object, identity by default) applied in
  `load_gaussian3d` — position `p' = R S p + t`, rotation `q' = qR ⊗ q`, log-scale
  `+= log s` (uniform s only). This is display-only; the training step never sees it
  (trainer draws with identity table). It makes dragging a 5M-splat object free — no
  buffer rewrites per frame.
- **On commit** (mouse-up / Apply): one `csBakeObjectTransform` kernel rewrites the
  object's splats in `splat_params` in place, plus:
  - transform the object's `splat_init` anchors with the same rigid transform (they
    are world-space positions; radius scales by `s`) so the init-position pull doesn't
    drag the object back to its pre-move location — the exact failure mode found in the
    [training-loss-divergence investigation](init_regularization_plan.md);
  - zero Adam moments for the object's splats only (position/rotation moments are
    directional in world space; a rotation invalidates them — a small
    `csZeroObjectMoments` over `packed_param_count × object_splats`). Property-edit
    momentum preservation is unaffected because in-place color/opacity edits don't
    change parameter frames.

  Baking during live training is safe at the same point edits already are: between
  steps, on the render thread, like `edit_properties` / `resample_selection` today.

Non-uniform per-object scale is rejected: it turns rigid gaussians into sheared
gaussians (rotation no longer orthogonal against anisotropic scaling) and breaks the
log-scale parameterization. Uniform scale + rotation + translation covers layout work.

---

## 6. Editing with objects

The current editor model — GPU selection mask that doubles as the highlight, box +
histogram-range selection combining in replace/add/subtract/intersect, in-place
property edits, refinement-based resample
([splat_editor.py](../src/viewer/splat_editor.py), Edit Splat window in
[ui.py:2348-2473](../src/viewer/ui.py)) — generalizes cleanly; objects become a
*scope* on top of it, not a parallel mechanism.

**Selection scope.** Every selection kernel (`csSelectBox`, `csSelectRange`,
`csSelectSetAll`) gets `uniform uint g_SelectScopeObjectId` (`0xFFFFFFFF` = all). With
scope = active object, box/range/invert/clear only touch that object's splats. The
Edit Splat window grows a two-state `Scope: [All | Active object]` control that reads
the scene graph's `active_object_id`. Histograms optionally follow scope (the
`csEditScalarMinMax`/`csEditScalarHistogram` kernels take the same uniform), so
"opacity histogram of just the statue" works.

**Freeze selection (pre-objects feature).** `csWriteMaskBitsWhereSelected` writes
trainable bits on selected splats. Ships in stage 1 as the first user-visible payoff
of the mask plumbing, with an `Unfreeze all` escape hatch and an `Unrefinable` debug
view already existing to visualize frozen topology
(`DEBUG_MODE_UNREFINABLE`, [gaussian_renderer.py:141](../src/renderer/gaussian_renderer.py) —
add a sibling `frozen_params` mode).

**Extract selection → new object.** The primary segmentation gesture: box-select the
statue, `Extract as object`, name it. Implementation: allocate a new `object_id`,
`csWriteObjectIdWhereSelected`, register the `SplatObject` (pivot = GPU bounds of the
selection via the existing `csEditPositionBounds` pattern with an object/selection
filter), set it active. Inverse gesture: `Merge into…` rewrites `object_id` in bulk.

**Object picking in the viewport.** Alt-click (or a pick tool) selects the object
under the cursor. Cheapest correct-enough approach: a `csPickNearestSplat` kernel over
the prepass outputs (`screen_center_radius_depth` is already populated for visible
splats) — find the splat whose projected ellipse covers the cursor with the nearest
depth, atomically min-reduce `(depth, splatId)`, read back 8 bytes, map
`splat_meta[splatId] → object_id`. Runs on demand (on click), not per frame. This
avoids a full id-raster pass while behaving correctly for overlapping objects in
almost all cases; a proper id-raster debug mode can come later if picking accuracy on
heavy overlap becomes an issue.

**Interaction with live training.** Object edits obey the same rules the editor
already established: property/mask/object-id writes are in-place (momentum preserved,
[Viewer.md "Edit Splat"](Viewer.md)); topology changes (extract does *not* change
topology; resample/delete does) go through the refinement rewrite and reset moments +
bookkeeping exactly as today.

---

## 7. Gizmos and object transforms

Reuse the ImGuizmo integration wholesale — it is already proven for the selection box:

- `camera_view_projection_matrices` ([splat_editor.py:56](../src/viewer/splat_editor.py))
  builds the view/projection pair each frame; the presenter publishes it as the
  `_splat_editor_gizmo` payload ([presenter.py:653](../src/viewer/presenter.py)); the
  toolkit calls `_IM_GUIZMO.manipulate` over a model matrix and decomposes on change
  ([ui.py:2255-2288](../src/viewer/ui.py)); drag ownership is latched so the camera
  never steals a gizmo drag ([Viewer.md "Camera Controls"](Viewer.md)).

**Object gizmo** = the same `manipulate` call over the active object's TRS matrix
(pivot-translated), with operations `translate | rotate | scale(uniform) | universal`.
Differences from the box gizmo:

| | Selection-box gizmo | Object gizmo |
|---|---|---|
| Target matrix | box center/half-extents/euler | object TRS around stored pivot |
| During drag | updates box state only (cheap) | writes per-draw `g_ObjectTransforms[id]` → whole object moves live |
| On release | nothing to commit (box is a query) | bake kernel + init-anchor transform + object moment reset (§5.3) |
| Scale handles | anisotropic (box extents) | **uniform only** (see §5.3) |
| Exclusivity | one gizmo at a time | shared — hierarchy panel's transform mode replaces the box gizmo while active |

Only one gizmo is active at a time, keyed by UI mode: Edit Splat's box mode vs.
Hierarchy's transform mode. Both funnel through the same
`_splat_editor_gizmo_capturing` flag so input routing needs no changes. Numeric
fallback fields (position / rotation / uniform scale drag-floats) mirror the gizmo for
precision, following the existing Center/Half size/Rotation fields pattern.

A lightweight object bounds overlay (wireframe box from the object's cached GPU
bounds, drawn like the current `_splat_editor_box_segments` line overlay) gives the
gizmo visual context and doubles as the hierarchy panel's hover-highlight.

---

## 8. Camera-pose objects

Camera poses are already 90 % "objects" — they only lack a face in the hierarchy:

- Geometry/overlays: frustum outlines and equirect cube markers, cached by signature
  ([presenter_state.py:706-831](../src/viewer/presenter_state.py)), with labels and
  active-frame highlighting.
- Inspection: the Training Views window lists per-frame loss/PSNR/params; the
  training-camera viewport mode previews any frame at full resolution with COLMAP
  point-match overlays; `Move Main View Here` already exists
  ([Viewer.md "Training Camera Mode"](Viewer.md)).
- Identity: `ColmapFrame` carries image id, camera id, pose, intrinsics, distortion.

**Design: `CameraObject` is a *view*, not a data owner.** The hierarchy panel lists
one `CameraRig` node per import source, expandable to individual frames. Selecting a
frame sets it active everywhere the "selected frame" concept already exists (Training
Views row highlight, camera-overlay active highlight, training-camera-mode frame
selector — these currently share `debug_frame_idx`). Row actions map to existing
verbs: *Preview* (enter training-camera mode on that frame), *Go here* (move main
view), *Show overlay* (per-rig gate on the existing overlay toggle). Per-frame loss
badges reuse the Training Views metric snapshot.

**Pose editing (deliberately deferred).** A gizmo on a camera pose is technically easy
(same ImGuizmo path; frustum overlay gives feedback) but semantically heavy: editing
`q_wxyz/t_xyz` invalidates the refinement camera buffer (auto-rebuilds via the
signature check, [gaussian_trainer.py:1628-1666](../src/training/gaussian_trainer.py)),
photometric state, cached target association, and any COLMAP-derived init. It's a
*dataset* edit, not a scene edit. Recommendation: stage 5+, behind an explicit "Edit
pose" toggle per frame, scoped to manual pose-fixing workflows — and note that
feature 1 (in-viewer SfM) reduces the need for it.

**Not in scope as objects:** per-rig transforms ("move all cameras") — that is a
world-space realignment (the PCA auto-rotate already owns scene orientation) and would
silently decouple cameras from splats trained against them.

---

## 9. UI design

### 9.1 Scene Hierarchy window

A new dockable tool window (`View -> Scene Hierarchy`), same `ToolkitWindow` pattern as
Edit Splat: presenter publishes a `_scene_hierarchy_payload` +
`_scene_hierarchy_state`, the toolkit draws it, mutations go through
`self.callbacks.hierarchy_*` bound in [app.py:671-704](../src/viewer/app.py).

```
┌─ Scene Hierarchy ────────────────────────────────┐
│ Scene: hotel3                    5.2M splats     │
│──────────────────────────────────────────────────│
│ ▼ Splat objects                                  │
│   👁 🔒 background            4.1M   [color-only]│
│ ● 👁    statue                0.9M   [full]      │
│   👁    imported: chair.ply   0.2M   [locked]    │
│ ▼ Cameras                                        │
│   👁 colmap: hotel3                  312 poses   │
│      ├ 📷 IMG_0041   loss 0.031     ← active     │
│      ├ 📷 IMG_0042   loss 0.028                  │
│      └ …                                         │
│──────────────────────────────────────────────────│
│ Active: statue                                   │
│ ▸ Transform   [Move][Rotate][Scale][All]  Reset  │
│     Position   x 0.124  y -0.310  z 1.207        │
│     Rotation   x 0.0    y 32.5   z 0.0           │
│     Scale      1.000                             │
│ ▸ Training                                       │
│     Preset  [Full ▾]        ☑ Allow refine       │
│     ☑ Position ☑ Scale ☑ Rotation                │
│     ☑ Opacity  ☑ Color (DC) ☑ SH                 │
│   ▸ Advanced: LR multipliers…                    │
│ ▸ Info: source colmap pointcloud · id 2          │
│──────────────────────────────────────────────────│
│ [＋ From selection] [＋ Load PLY…] [Isolate] [🗑] │
└──────────────────────────────────────────────────┘
```

Row anatomy: `●` active marker (click row = set active), 👁 visibility eye
(presentation-only, §5.1), 🔒 lock badge derived from the training preset, splat
count (from the GPU histogram, §2), preset badge. Double-click renames. Context menu:
Freeze color-only / Lock / Unfreeze · Isolate · Extract selection into this object ·
Merge into… · Export PLY… · Delete (confirmation modal, like Exit).

Distinct iconography for the two orthogonal toggles (eye = visible, lock = training)
is load-bearing — §5.1's "hidden still trains" policy is only understandable if the UI
never conflates them.

### 9.2 Edit Splat window changes

Minimal: a `Scope: [All | Active object]` control at the top of the Selection group,
a `Freeze selection ▾` (freeze / color-only / unfreeze) button row next to
Invert/Clear, and `Extract as object` next to Resample. Status line gains the active
object name. Everything else (box, histograms, preview, resample, property edits)
works unchanged under scope.

### 9.3 Viewport

- Header adds an active-object indicator with an isolate toggle (matches the existing
  debug-mode/overlay toggle row).
- Alt-click picks objects (§6).
- Object bounds wireframe + gizmo when the hierarchy transform mode is active (§7).
- New debug view `Object ID` (distinct color per object, from `splat_meta` — one more
  case in `csRasterizeDebug`'s mode switch) and `Frozen Params` (tint by mask).

### 9.4 Training panel

The Training section header line gains a per-object breakdown when more than one
object trains (e.g. `optimizing 0.9M of 5.2M splats · 1 frozen object`), sourced from
the same payload as the hierarchy panel. Schedule/optimizer tabs stay global —
per-object overrides live only in the hierarchy inspector to avoid two editing
surfaces for one value.

---

## 10. Ingest, merge, export

- **Add PLY as object** (`File -> Add to Scene…`): load via the existing
  `ply_loader`, concat CPU-side into the upload path with a fresh `object_id`, default
  preset *Locked* (imported props usually shouldn't train until poses cover them),
  register in the graph. During an active training session this is a topology change →
  same reset semantics as editor resample (renumber, reset moments/ages/init,
  preserve step) — the mechanism exists.
- **COLMAP import → objects**: one SplatObject per *import* for the regular init
  sources (pointcloud/diffused/PLY/mesh are one photometric unit), **plus the
  Fibonacci shell as its own auto-object** (planned, stage 2). Users routinely want to
  freeze, hide, or delete the shell independently — it already has separate
  refinability (`fibonacci_sphere_refinable`) as precedent. Implementation is cheap
  because the initial scene is a per-source concat
  (`_concat_gaussian_scenes`, [session.py:1096](../src/viewer/session.py)): record the
  source ranges during concat and assign the shell range a fresh `object_id`
  (named e.g. `sky shell`) when the scene graph is built. Default preset stays *Full*
  so training behavior is unchanged — the win is the one-click grouping. The same
  range hook makes "every source as its own object" a trivial later checkbox if
  wanted. `Reinitialize Gaussians` rebuilds from the same sources and must recreate
  the same auto-objects (ids may change; names/presets persist by source name).
- **Export**: per-object PLY = the existing `save_gaussian_ply` over
  `read_live_scene()` filtered by `object_id` (readback already exists; the filter is
  a mask). Whole-scene PLY export unchanged (objects flatten silently — external
  tools see a normal gaussian PLY). Optional later: write `object_id` as a custom
  PLY property for round-tripping single files.
- **Project format (feature 2)**: `project.json → scene_graph` holds the object list
  (name/id/preset/transform history not needed — transforms are baked), `result`
  becomes per-object PLYs, exactly as roadmap §2 anticipated.

---

## 11. Staging plan

Each stage is independently shippable and verifiable; the risk ordering follows the
roadmap ("land mask plumbing single-object before multi-object") with the carry
plumbing pulled to the very front because it is the piece with silent-corruption risk.

**Stage 0 — `splat_meta` plumbing, invisible.**
Renderer scene-group buffer + default fill; carry through `copy_scene_state_to` (with
per-name sizes), editor compaction, refinement rewrite triple, `_run_refinement`
adoption. No UI, no behavior change.
*Verify:* existing kernel/optimizer/session tests untouched; new test that meta
survives resample + refinement + renderer recreation with correct inheritance; a
fixed-seed training run reproduces current loss curves bit-for-bit (dispatch shapes
unchanged when mask is all-default — gate the new kernel reads behind a
`g_SplatMetaEnabled` uniform if bitwise parity is required).

**Stage 1 — per-splat trainable mask, single object.**
Adam early-return + LUT; post-step gating; `Freeze selection` / unfreeze in Edit
Splat; `Frozen Params` debug view.
*Verify:* unit test freezing {position} keeps positions bit-identical across N steps
while colors move; moments of frozen params stay zero; freeze-all matches
pause; hotel3 A/B: frozen-background-color-only vs full training (quality + it/s).

**Stage 2 — multiple objects, hierarchy panel, presets.**
SceneGraph state + payloads; object-id bulk writes; extract/merge; visibility bitmask
per draw; active-object tint; object count histogram; hierarchy window (tree +
inspector minus transform); Edit Splat scope; add-PLY-as-object; **Fibonacci shell as
auto-object at import** (source-range tagging in the concat path, §10); per-object
export; freeze presets; `Object ID` debug view.
*Verify:* extract → freeze → continue training → background PSNR stable while
foreground improves; hide object in viewport during training → loss unaffected;
refinement keeps children in parents' objects (assert via histogram); importing with
the shell enabled yields a `sky shell` object whose count matches
`fibonacci_sphere_point_count` and whose freeze/hide works like any other object.

**Stage 3 — object transforms + gizmo.**
Per-draw transform table in prepass; bake kernel + init-anchor transform + object
moment zeroing; hierarchy transform mode + gizmo + bounds overlay; numeric fields.
*Verify:* move object mid-training → init-pull does not drag it back (regression test
against the failure mode from the loss-divergence investigation); moved-object PSNR
recovers within a few hundred steps; drag at 5M splats stays at frame rate.

**Stage 4 — camera objects + per-object LR.**
Camera rig/frame rows wired to existing selection/overlay/preview verbs; per-object
LR multiplier table in Adam; training-panel breakdown line.
*Verify:* frame selection round-trips with Training Views + overlays; LR override 0
equals mask freeze for that group.

**Stage 5+ (unscheduled)** — pose editing, per-object refinement budgets,
every-init-source-as-object import checkbox, id-raster picking,
`splat_meta`-unified refinable flag, headless scene-graph runs via project files.

---

## 12. Risks and open questions

- **Silent meta corruption** is the top risk: a missed carry (some future compaction
  or a copy path like `write_scene_groups`) reassigns splats between objects without
  crashing. Mitigation: a debug assertion kernel (`csValidateSplatMeta`: object ids
  < graph size, mask ≠ 0) run after every topology change in debug builds, plus the
  stage-0 tests.
- **Frozen-object contribution statistics.** Frozen splats still accumulate
  contribution/viewed-fraction EMAs; they are excluded from refinement by flag, but
  the *global* prune-ratio percentile is computed over eligible splats only — verified
  correct in `csPrepareRefinementPruneSortInputs`
  ([gaussian_training_stage.slang:1290-1305](../../shaders/renderer/gaussian_training_stage.slang)).
  No change needed; noted so nobody "fixes" it into including frozen splats.
- **SH-band repacks** (`packed_trainable_param_count` changes) rebuild `splat_params`
  via the CPU groups path — meta must ride along; covered in stage 0 but easy to
  regress. Consider asserting scene-group buffer counts match in `_require_scene`.
- **Object count ceiling**: 16-bit ids and a 256-entry visibility/LR/transform table
  disagree. Tables cap the *practical* object count at 256; ids keep headroom. Fine
  for the target workflows; document the limit in the UI (refuse to create object 257).
- **Picking on heavy overlap** (semi-transparent object in front of another): nearest
  ellipse-hit may disagree with what the user perceives. Acceptable v1; id-raster is
  the upgrade path.
- **Open**: should *Delete object* during training route through the refinement
  rewrite (like editor delete — moment-safe) or require pause? Recommend the former
  (mechanism exists; `resample_selection` with a full-object prune mask).
- **Open**: default preset for `Add PLY as object` — *Locked* proposed above; could be
  a persisted import setting (`viewer.import.added_object_preset`).
- **Open**: does the main (non-training) renderer need per-object visibility for
  PLY-only multi-object scenes? Yes, and it comes for free since the bitmask is role
  member state applied via `_apply_renderer_role_params`.

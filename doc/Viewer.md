# Viewer

## Overview

The realtime viewer is a single `spy.AppWindow` that renders the Gaussian scene into an offscreen texture and composites the UI through `imgui_bundle` into the swapchain.

`src/viewer` is split into:

- `app.py`: window lifecycle, camera state, callbacks, and event routing.
- `ui.py`: widget layout, menu handling, documentation/about windows, and docked tool windows.
- `presenter.py`: per-frame scene rendering, debug composition, plots, and UI text refresh.
- `presenter_state.py`: derived presenter/debug state such as training-view rows and camera-overlay segments.
- `session.py`: scene loading, COLMAP import, training initialization, reload/export/reinitialize actions, and defaults updates.
- `state.py`: persistent runtime state.

The main UI is a docked layout with:

- a top menu bar,
- a central viewport window for scene presentation,
- a right-side panel for control sections and optional tool windows,
- auxiliary windows such as `Documentation`, `About`, `Buffers`, `Histograms`, `Training Views`, and `COLMAP Import`.

Viewer defaults, renderer defaults, and training defaults all serialize into `config/defaults.json`. The footer `Update Defaults` action writes the current stable UI state back into that file.

## Main Menus

### File

The `File` menu exposes the primary scene/runtime actions:

- `Load PLY...`: load a gaussian scene from `.ply`.
- `Export PLY...`: export the current runtime gaussian scene when export is possible.
- `Load COLMAP...`: open the COLMAP import window.
- `Reload`: reload the current scene/import source.
- `Reinitialize Gaussians`: rebuild the training scene from the current initialization source while keeping the training dataset.
- `Exit`: opens an ImGui confirmation modal before shutdown so accidental exits do not immediately discard unsaved runtime state.

### View

The `View` menu controls presentation-only settings:

- interface scale presets from `75%` to `200%`,
- `Reset Interface Scale`,
- theme selection and theme reset.

### Debug

The `Debug` menu toggles the docked inspection windows:

- `Buffers`
- `Histograms`
- `Training Views`

### Help

The `Help` menu opens:

- `Documentation`
- `About`

## Viewport And Camera Controls

The viewport header exposes quick access to:

- debug mode selection,
- camera overlay toggles,
- camera label toggles,
- training-camera debug toggles,
- main camera reset,
- the active SH band cap.

### Training Camera Mode

When `Training Cameras` is enabled in the viewport, the viewport switches from the free-fly scene render to the currently selected training frame/debug view.

- The overlay keeps the frame selector and loss/debug view selector.
- `Full Resolution` bypasses the current training subsample path and renders the selected training camera at its source frame resolution.
- The viewport image is displayed with aspect-fit pan/zoom controls.
- mouse wheel zooms,
- left-drag pans while zoomed,
- double-click resets the zoom/pan state.
- `Move Main View Here` copies the selected training camera position and view direction into the main free-fly camera while keeping the viewer controls roll-free, then automatically exits training-camera mode.
- `Reset Camera` returns the main free-fly camera to the current scene fit and exits training-camera mode.
- The overlay also shows the selected frame/image id, source camera id, pose vectors, projection parameters, distortion coefficients, and current target resolution.

### Input Routing

Input routing is UI-first:

- keyboard events go to ImGui first,
- mouse capture outside the viewport blocks camera movement,
- mouse interaction inside the viewport passes through to the camera controls.
- `File -> Exit` opens the confirmation modal: `Do you want to exit? Any unsaved data will be lost`.
- Native title-bar close recreates the viewer window and opens the same ImGui confirmation modal without platform-specific hooks.

### Camera Controls

Camera controls:

- `LMB + drag`: look
- `RMB + drag`: pan
- `WASDQE`: move
- mouse wheel: adjust move speed

When a training scene is initialized, the viewer prefers a real training-camera position when one is available. That startup path only copies the camera position and keeps the viewer orientation controls intact; if no usable training pose is available, the viewer falls back to a scene-bounds fit.

## Frame Flow

Each frame follows this order:

1. `SplatViewer.render(...)` calls `presenter.render_frame(...)` to produce the scene or active debug view into an offscreen texture sized from the docked viewport.
2. The swapchain is cleared for UI composition.
3. The overlay begins an `imgui_bundle` frame using the current surface size and frame delta.
4. The presenter composes the currently displayed texture into a UI-display texture.
5. In the normal scene path that texture is viewport-sized. In training-camera mode it keeps the selected debug resolution so the UI can pan and zoom the raw training-camera image.
6. The docked viewport window emits the texture widget for that image, while the control windows emit ordinary Dear ImGui draw lists.
7. Slangpy marshals the external textures referenced by the draw data.
8. Slangpy renders the UI draw data into the swapchain.

This keeps the renderer resolution tied to the docked viewport instead of the outer window size.

## COLMAP Import

`File -> Load COLMAP...` opens a dedicated import window. The window collects:

- the dataset root containing the COLMAP reconstruction,
- the training image folder,
- an optional depth folder,
- camera-model selection,
- import-time image downscale settings,
- initialization mode,
- initialization parameters,
- import modifiers such as auto-rotation, BC7 dataset compression, and alpha masking.

### Dataset Discovery

After the dataset root is selected, the viewer:

1. resolves the reconstruction from `sparse/0`,
2. samples image names from the COLMAP database when present, otherwise from reconstruction image names,
3. searches the selected root and its subfolders for the first directory containing those images,
4. skips directories whose path or name contains `depth` when auto-discovering the RGB image tree.

The image folder can still be overridden manually before import.

### Camera Selection

If the reconstruction contains multiple camera models, the import window shows a camera-selection table with:

- camera id,
- model name,
- frame/pose count,
- resolution,
- focal parameters,
- principal point and radial distortion summary.

The import path can keep all models or restrict training frames to the selected subset.

### Import Modifiers

The import window exposes three key toggles:

- `Auto Rotate Scene`: run or skip the COLMAP auto-alignment pass derived from the camera layout.
- `Compress Dataset using BC7`: compress imported training images into reusable BC7 DDS cache files under the image-folder cache.
- `Use Alpha Mask`: treat transparent target pixels as masked-out training pixels.

### Image Downscale Modes

Import-time training-frame image sizes can be resolved in three ways:

- `Original`: keep source resolution.
- `Max Size`: clamp the longer side and preserve aspect ratio.
- `Scale Factor`: uniformly scale both dimensions by a chosen factor.

The importer never upscales. These resized dimensions also determine the final per-frame intrinsics used by training.

Training-frame metadata and native texture preparation both use fixed CPU thread pools for image decode/resize work while keeping GPU texture creation on the owning thread.

### Initialization Modes

The current initialization modes are:

- `COLMAP Pointcloud`
  - Seeds directly from sparse COLMAP points.
  - Respects `Min Camera Observations` filtering.
  - Uses local covariance eigenframes for gaussian rotation and anisotropy, with `NN Radius Scale Coef` setting the overall scale magnitude.

- `Diffused Pointcloud`
  - Resamples filtered sparse points with replacement.
  - Offsets each sample by a local covariance-shaped Gaussian scaled by `Diffusion Radius`.
  - Uses the same covariance-based point initializer afterward.

- `Custom PLY`
  - Keeps the COLMAP cameras and training frames.
  - Seeds the scene from a chosen `.ply` gaussian scene.

- `Custom Mesh`
  - Keeps the COLMAP cameras and training frames.
  - Uniformly samples the selected triangle mesh by triangle area.
  - Uses vectorized barycentric surface sampling.
  - Colors the samples from the mesh texture before gaussian initialization.

- `From Depth`
  - Matches RGB and depth files by relative path stem under the chosen depth root.
  - Expects scalar `16-bit PNG` depth maps.
  - Uses each pose's own positive COLMAP observations to sample raw depth.
  - Rejects strong local depth-gradient spikes before calibration.
  - Solves a robust per-pose affine map `a + b * raw_depth`.
  - Supports two reverse-projection modes:
    - `Depth Is Distance`
    - `Depth Is Z-Depth`
  - Samples a dataset-wide calibrated point budget from the usable depth maps.

### Fibonacci Shell

Point-based initializers can optionally append a Fibonacci shell around the arithmetic mean of the COLMAP camera centers.

- `Sphere Point Count` controls the number of appended shell points.
- `Sphere Radius Multiplier` scales the max aligned COLMAP point distance from the shell center; each shell point also gets a deterministic radial jitter of up to 10% to reduce ordering aliasing.

Those appended shell splats now use the same covariance-based point initializer as the other non-PLY point sources.

## Scene I/O Section

The right-side `Scene I/O` section mirrors the active import state and provides a quick `Open COLMAP Import` button. It shows the currently selected root, image folder, optional depth folder, depth interpretation, and active initialization mode.

## Training Controls

The viewer keeps training controls grouped into five main sections.

### Training

The `Training` section shows:

- current step,
- elapsed pause-aware training time,
- average iterations per second,
- rolling loss,
- SSIM,
- density metric,
- PSNR,
- instability warnings when present.

It also exposes:

- `Start`
- `Stop`
- `Reinitialize Gaussians`

### Train Setup

`Train Setup` controls the runtime training dataset and schedule entry conditions.

Important current behaviors:

- training background is independent from the viewer clear color,
- train resolution is controlled by both downscale and subsample,
- the panel reports the resolved active train resolution, current downscale state, schedule state, and refinement state.

The train-resolution system has two independent components:

- `Train Downscale`
  - `Auto` starts from `Auto Start Downscale` and walks toward `1x`.
  - manual modes force a fixed integer factor.

- `Train Subsample`
  - `Auto` chooses a factor from `1..8` that brings the effective resolution closest to a `1000 x 1000` target area after the active downscale factor is applied.
  - manual modes force a fixed subsample factor.

The effective training render factor is the product of downscale and subsample.

### Optimizer

The `Optimizer` section is tabbed:

- `Schedule`: schedule duration, stage breakpoints, and stage-specific values.
- `Adam`: learning-rate and ADAM hyperparameters.
- `Regularization`: DSSIM, density, opacity, scale, SH, visible-angle, and refinement controls.
- `Raster Grads`: cached raster-gradient mode and related ranges.

The schedule UI includes stage tabs that expose per-stage overrides for:

- SH band,
- DSSIM weight,
- visible-angle limits,
- sorting-order dithering,
- position-random-step noise,
- colorspace modulation,
- stage-specific learning-rate multipliers for position, scale, rotation, color, opacity, and non-DC SH.

The viewport SH-band dropdown writes back to the currently active schedule stage.

### Stability

The `Stability` section holds post-step clamps and stability controls such as:

- position bounds,
- opacity bounds,
- max anisotropy,
- gradient clip and norm clip,
- max update.

### Render Params

The `Render Params` section exposes the runtime renderer controls that apply to the viewport:

- `radius_scale`
- `alpha_cutoff`
- `trans_threshold`

### Defaults Footer

The footer `Update Defaults` action writes the current viewer, renderer, and training-control state into `config/defaults.json`.

## Debug Views

The renderer debug modes share the same forward replay path as normal rendering. Important viewer-facing modes include:

- `Contribution Amount`: visible-average leave-one-out RGB contribution accumulated during training backward; it estimates how much the final pixel color would change if each splat were removed while ignoring final alpha differences and averages only nonzero per-view observations.
- `Grad Variance`: per-splat raster contribution-gradient variance computed from `(sum, sumSq)` statistics accumulated since the last refinement reset.
- `Depth Local Mismatch`: contribution-weighted local deviation from the online front-to-back depth estimate.
- `Splat Density`, `Spatial Density`, and `Screen Density`.
- `Processed Count`, `Grad Norm`, `Ellipse Outlines`, and SH inspection modes.

The loss-debug controls also expose a runtime `Abs Diff Scale` slider for `Abs Diff` visualization.

## Histograms

`Debug -> Histograms` opens grouped live histograms for the current training scene.

Current behavior:

- the first open requests a refresh automatically,
- after that, recomputation happens on explicit refresh, bin/range edits, or another histogram-triggering action,
- position histogram values use the current linear min/max controls; rotation and color rows use their own linear ranges; scale, opacity, contribution, and refinement distributions use log10 bins from their own range passes; all histograms default to `256` bins,
- `Update Range` derives min/max from the position ranges before refreshing the histograms,
- `Update Y Scale` derives a shared y-axis limit from the combined histogram payload,
- groups are shown under type tabs with closable tabs for position, scale, quaternion, base color (SH0/DC), higher SH bands, opacity, contribution distribution, and refinement distribution.

## Buffers

`Debug -> Buffers` opens the live GPU resource table.

It reports:

- combined tracked GPU consumption,
- buffer and texture counts,
- largest-first entries,
- texture dimensions and format information when available,
- exact duplicate-looking allocation groups.

`Write Log` exports the current table and duplicate groups to `temp/resource_logs`.

## Training Views And Camera Overlays

`Debug -> Training Views` opens the per-frame training-view inspector. The window can show:

- per-frame rows with loss, PSNR, visited state, and camera parameters,
- optional world-camera overlays in the viewport,
- optional camera labels,
- active-frame highlighting.

Camera overlays and labels are independently gated so expensive per-frame label/metric generation is only done when needed.

## Metrics And Plots

The plot section shows both short-horizon and run-level throughput and quality summaries.

- `Iter/s` is derived from the recent plot-history window.
- `Avg it/s` is computed from total optimizer steps over accumulated active training time.
- `Time` is pause/resume-aware.
- FPS, loss, PSNR, and related viewer histories are plotted in the docked UI.

# Viewer

## Overview

The realtime viewer is a single `spy.AppWindow` that renders the control UI into the swapchain and presents the Gaussian scene inside a docked central viewport window.

The overlay uses a right-side control panel with a menu bar:

- `File`: scene load, scene export, COLMAP import, reload, and gaussian reinitialization actions
- `View`: interface scale presets from `75%` to `200%`, with a reset action
- `Debug`: resource, histogram, and training-view inspection windows
- `Help`: `Documentation` and `About` windows
- Auxiliary windows such as `Documentation`, `About`, `Buffers`, `Histograms`, `Training Views`, and `COLMAP Import` open as tabs in that right-side panel by default.
- The menu bar spans the full viewport width and the left control panel starts below it to avoid overlap.
- The viewport header row includes quick controls for debug mode, camera overlays, training-camera debug view, and the active SH band cap (`SH0` / `SH1` / `SH2` / `SH3`).

`src/viewer` is split into:
- `app.py`: window lifecycle, camera input, and UI event routing
- `ui.py`: `imgui_bundle` overlay state, widget layout, and draw-data submission
- `presenter.py`: per-frame scene rendering, debug-view composition, and status text updates
- `session.py`: scene loading, renderer recreation, and training control actions
- `state.py`: persistent viewer runtime state

Viewer and CLI defaults are stored in `config/defaults.json`. The toolkit footer includes `Update Defaults`, which writes the current stable viewer, renderer, and training-control values back into that file.

## Frame Flow

Each frame follows this order:

1. `SplatViewer.render(...)` calls `presenter.render_frame(...)` to render the scene or debug view into an offscreen texture sized from the current viewport content rect.
2. The swapchain is cleared for UI composition.
3. The overlay begins an `imgui_bundle` frame with the current surface size and frame delta.
4. The viewer presentation pass composes the currently displayed texture into a viewport-sized present texture, applying the viewport letterbox fit when needed and converting linear RGB to sRGB for UI display.
5. The docked `Viewport` window emits an `imgui.image(...)` draw call referencing that present texture, while the rest of the control panels and plots emit normal Dear ImGui draw lists.
6. Slangpy marshals any external font or image textures referenced by the draw data.
7. Slangpy renders the full draw data into the swapchain with the current command encoder.

This keeps scene rendering and UI composition in one window and one graphics context while letting the renderer resolution follow the docked viewport size instead of the outer window size.

## Input Routing

Keyboard and mouse events are forwarded to the overlay first.

- If ImGui reports keyboard capture, the viewer skips its own key handling.
- If ImGui reports mouse capture outside the viewport, the viewer skips camera drag and scroll handling and only refreshes its cursor reference state.
- Mouse events inside the viewport content area are passed back through to the camera so the docked viewport keeps the same fly/look controls as the old full-background renderer.
- If ImGui does not capture the event, the existing camera controls run unchanged.

This avoids camera movement while using sliders, combo boxes, text inputs, plot interactions, or scrollable UI regions.

## COLMAP Import

`File -> Load COLMAP...` opens a dedicated import window instead of using an image-subdirectory selector in the main panel.

The import window collects:

- the top-level dataset root
- the image folder used for training frames
- an optional `Depth Folder` used only by depth-based initialization
- image downscale settings: original resolution, fixed max size, or uniform scale factor
- the initialization mode: `COLMAP Pointcloud`, `Diffused Pointcloud`, `Custom PLY`, or `From Depth`
- the nearest-neighbor radius scale coefficient used by COLMAP-based initialization
- for `From Depth`: a dataset-wide `Depth Point Count` budget
- for `Diffused Pointcloud`: synthesized point count and a dimensionless diffusion-radius multiplier
- for point-cloud initializers: an optional Fibonacci sphere shell, specified by point count and radius, appended around the mean COLMAP camera center

After the dataset root is selected, the viewer resolves the COLMAP reconstruction from `sparse/0`. If a COLMAP database is present it samples image names from the `images` table; otherwise it falls back to the image names stored in `images.bin`. It then walks the selected root and its subfolders until it finds the first directory that contains one of those image entries. The image folder can still be overridden manually before pressing `Import`.

Automatic RGB folder discovery skips directories whose name or full path contains `depth` so a colocated depth tree is not mistaken for the main image set.

Import-time downscale is applied before training textures are uploaded and before frame intrinsics are finalized. `Max Size` preserves aspect ratio by clamping the longer image side and scaling the shorter side to match. `Scale Factor` multiplies both dimensions uniformly. Both modes clamp to the source resolution, so the importer never upscales images.

Training-frame metadata discovery opens source images in a fixed 16-thread loader pool, which keeps large datasets from stalling on serial image-size probes while preserving the original sorted frame order.

Native training-target import uses the same 16-thread CPU loader for RGBA decode and resize work. Texture creation and `copy_from_numpy(...)` stay on the main thread, but the import path pipelines those uploads against ongoing background decode/resize work instead of waiting for the entire dataset to be processed serially.

Pointcloud initialization builds gaussians directly from the COLMAP sparse points and scales them from the median nearest-neighbor spacing multiplied by the selected coefficient. Diffused pointcloud initialization resamples sparse points with replacement and offsets each sample by `nrand3() * diffusion_radius * original_nn_distance`, where `original_nn_distance` comes from the source COLMAP point cloud, then applies the same nearest-neighbor scale initialization to the synthesized positions. The optional Fibonacci sphere appends evenly distributed neutral-color points on a shell centered at the mean camera pose; their nearest-neighbor spacing flows through the same scale initializer so neighboring shell splats overlap. Custom PLY initialization keeps the COLMAP cameras and training frames, but seeds the scene from the chosen `.ply` file instead.

`From Depth` adds a CPU-only calibration stage before the initial point cloud is built:

- RGB images are matched to depth files by relative path stem under `Depth Folder`, extension-agnostically.
- Depth maps are expected to be scalar `16-bit PNG` images.
- For each matched frame, the importer gathers positive COLMAP `points2d_point3d_ids`, reprojects those observed 3D points through the resized frame camera model, and samples the raw depth map at that projected location.
- Calibration is solved per pose from that pose's own observed COLMAP correspondences, so occluded or otherwise unobserved points from other poses never enter the fit.
- Reprojected correspondences that land on strong local depth discontinuities are discarded before fitting; the importer compares the sampled pixel-footprint depth gradients against nearby gradients and culls only local `4x` gradient spikes.
- The remap is deliberately reduced to a robust affine model:
  - `target ~= a + b * raw_depth`
- `Depth Interpretation` selects the fitted target and reverse-projection mode:
  - `Depth Is Distance`: `target` is Euclidean camera-to-point distance and reconstruction uses the camera ray.
  - `Depth Is Z-Depth`: `target` is camera-space `z` depth and reconstruction uses `screen_to_world(...)`.
- Fitting uses an iteratively reweighted ridge-regularized 2-parameter least-squares solve with MAD-scaled Tukey weights, so contaminated correspondences get downweighted instead of relying on one hard inlier cutoff.
- Frames with missing or unusable depth, or with too few usable per-pose correspondences, are still imported as training views; they are only skipped when generating the depth-derived initialization cloud.
- After calibration, the importer samples a unique dataset-wide point budget across usable frames approximately proportional to each frame's valid calibrated pixel count, reverse-projects those pixels through the COLMAP camera model, and colors them from the aligned RGB image.
- The resulting positions/colors are then passed through the same nearest-neighbor scale initialization used by the point-based import modes.
- Depth maps and calibration intermediates stay on CPU and are discarded once the calibrated point cloud has been built; they are not uploaded into the runtime training textures.

## Debug Views

The loss-debug controls expose a runtime `Abs Diff Scale` slider when `View = Abs Diff`.

- The shader computes `abs(rendered - target) * scale` in linear RGB.
- `scale = 1.0` shows the raw absolute color difference.
- Higher values amplify subtle differences without changing the rendered or target views.
- Training-camera debug rendering uses the same effective training resolution, training-forward raster path, background mode, active SH band, and native-camera sampling parameters as training.
- When training subsampling is active, the debug target view is sampled from the native target with the same per-pixel subsample mapping as training; its random seed comes from the current viewer render frame so repeated viewport frames preview the live stochastic sample pattern.
- Scheduled sorting-order dithering is also applied to the training-camera debug prepass using the current viewer render-frame seed; the projection shader expands that seed into independent per-splat sort-camera offsets.
- Density debug views share the same range controls. `Splat Density` accumulates a soft per-pixel splat count using `sqrt(transmittance) * alpha / opacity`, while `Spatial Density` and `Screen Density` continue to normalize by 3D volume and projected ellipse area respectively.
- `Contribution Amount` visualizes the per-splat `g_SplatContribution` atomic buffer accumulated during training forward from the linear-RGB color change each fragment causes after blending, normalized by observed dataset pixels as `count / 256 / observed_pixels`.
- The heatmap is logarithmic and uses dedicated `Contribution Min` and `Contribution Max` controls in normalized color-change units instead of sharing density ranges.
- `Depth Local Mismatch` tracks a front-to-back online depth estimate per pixel and visualizes the contribution-weighted local absolute depth deviation from that estimate, normalized by mean depth like `Depth Std`, with separate smooth and reject sigma-multiple controls to avoid coupling distant transparent layers.
- The renderer debug colorbar is drawn as a horizontal legend near the bottom of the docked viewport window.

## Histogram Window

`Debug -> Histograms` opens a dedicated histogram window for live semantic splat parameters.

- Histogram values are computed over `log10(abs(value))`.
- The window groups the current training scene into `position`, `scale`, `quat`, `baseColor (SH0/DC)`, `SH1`, `SH2`, `SH3`, and `opacity`.
- `baseColor` is derived directly from the clamped SH0/DC term, so it is not presented as a separate parameter from DC anymore.
- The first open requests one histogram refresh automatically.
- After that, histogram data is recomputed only when `Refresh` is pressed or another histogram action explicitly requests it.

## Buffer Window

`Debug -> Buffers` opens a resource table for live viewer GPU allocations.

- All project-owned buffers and textures are created through the shared resource helpers with Slangpy labels.
- The helpers read Slangpy's native per-resource `memory_usage.device` value when available and keep a Python-side name registry because Slangpy exposes per-resource memory but not a live allocation enumerator.
- The window reports combined tracked GPU consumption, buffer count, total buffer bytes, mean buffer size, median buffer size, texture count, and texture bytes.
- The table includes buffers and textures reachable from the active viewer renderers, trainer, and viewer debug resources, de-duplicates shared references, and displays entries largest-first.
- Buffer details include element count when Slangpy reports a structured element stride; texture details include dimensions, format, array count, and mip count when present.
- `Write Log` saves the current largest-first table, summary, and exact duplicate-looking allocation groups as a tab-separated text file under `temp/resource_logs`.

## Cached Gradient Atomics

`Render Params` includes a `Cached Grad Atomics` selector:

- `Float Atomics`
- `Fixed Point`

`Fixed Point` is the default and uses the current cached-gradient quantization path. `Float Atomics` remains available as a fallback/reference path when the backend supports float atomics. Changing this setting only swaps the raster-backward shader/buffer path; it does not reset scene state, optimizer state, or training progress.

## Training Resolution

The `Train Setup` section exposes train downscale as a mode selector:

It also exposes per-stage `SH Band` dropdowns. `SH0` uses only the DC term, while `SH1`, `SH2`, and `SH3` progressively enable the higher bands in both the viewport and the training schedule.

The `Optimizer -> Regularization` tab exposes the DSSIM controls used by training:

- `Color >= 0 Reg`
- `DSSIM Weight`
- `SSIM C1`
- `SSIM C2`

The blur window remains the fixed 11-tap separable Gaussian used by the shader path.

Training background is configured separately from the viewer clear color:

- `Background Mode`: `Custom` or `Random`
- `Custom` exposes a fixed training RGB color picker and defaults to white
- `Random` re-samples the training background color per step
- `Start Densification After` now defaults to `500`.
- Refinement contribution culling is expressed as observed-pixel-normalized color-change contribution, with a base default threshold of `1e-7`.
- `Refinement Cull Decay` multiplies that threshold after each completed refinement pass and defaults to `0.995` (`0.5%` drop per pass).
- `Refinement Sample Radius` controls the local-space radius used for newly spawned refinement samples and defaults to `4.0`.
- `Refinement Clone Scale Mul` multiplies the split-family sigma after the default `family_size^(-1/3)` refinement shrink and defaults to `1.0`.
- Schedule step sliders in `Train Setup`, `Learning Rates`, and `Regularization` all clamp to the current `Schedule Steps` value so breakpoint timing can be edited directly in the viewer without touching code.

- `Auto`
- manual `1x` through `16x`
- Subsampling `Auto`, `Off`, and manual `1/2` through `1/8`

Auto mode has its own `Auto Start Downscale` parameter and schedule controls:

- `Downscale Base Iters`
- `Downscale Iter Step`
- `Downscale Max Iters`

Behavior:

- Manual modes force a fixed training downscale immediately.
- Auto mode starts from `Auto Start Downscale`, then descends toward `1x`.
- Each lower factor lasts `base_iters + level_index * iter_step`.
- Training render resolution is always `ceil(native_width / N) x ceil(native_height / N)` for the effective factor, where `N` is downscale multiplied by the active subsampling factor.
- Non-subsampled loss targets are generated from the native dataset image with an exact `NxN` box filter on the GPU; subsampled loss uses seeded native-pixel samples inside each effective block.
- Changing mode or crossing an auto schedule boundary recreates only the train-resolution renderer and target resources; scene state, ADAM moments, shuffle order, step counter, and pause/run state are preserved.

The panel shows both the resolved active train resolution and the current downscale status so the training renderer, loss target, and debug target view are easy to verify.

## Training Schedule

The `Optimizer` panel exposes the active training schedule directly:

- `Schedule Steps` defines the shared max-iteration budget for the LR, depth-ratio, SH band, and noise schedules.
- `LR Stage 1 Step` and `LR Stage 2 Step` move the two intermediate LR breakpoints.
- `Noise End Step` moves the point where random-step position noise reaches zero.
- `Depth Reg Stage 1/2/3` move the three intermediate depth-ratio regularizer breakpoints.
- The viewport SH dropdown writes back to the currently active schedule-stage `SH Band` control, so the visible band cap always targets the phase currently being trained.

These breakpoint controls are regular integer sliders with a live `0..Schedule Steps` range rather than a compound multi-value slider.

## Training Metrics

The training panel shows both short-horizon and run-level throughput data.

- `Iter/s` in the status and plot sections is computed from the recent history window used for viewer plots.
- `Avg it/s` in the training table is computed from total optimizer steps divided by total accumulated active training time for the current initialized scene.
- `Time` is pause/resume-aware and accumulates only while training is active.
